"""Refitting sensitivity of the headline contrasts by a delete-d jackknife over datasets.

The review's bootstrap intervals resample completed dataset means and so hold the fitted
priors and choices fixed. This script re-runs the whole nested procedure (priors, weight
choices, adaptive choices, nested policy choice and the train-selected reference) on S
task-stratified subsets that each delete d of the n datasets, and computes the headline
dataset-level contrasts theta_s in every subset. It reports
SE = sqrt(((n - d) / d) * Var_s(theta_s)), the delete-d jackknife formula (Shao & Wu 1989)
with one global factor applied to S sampled, proportionally stratified subsets. The value
mixes refitting variability, the smaller training sets of the subsets and the Monte Carlo
noise of their outer splits, so it is a refitting-sensitivity diagnostic, not a validated
procedure-level standard error.

Outer splits use common random numbers, so subsets differ only through the deleted
datasets. Under ``--split-design random`` every dataset carries fixed keys u[r, d] and
split r holds out the lowest-key quarter of each task's kept datasets. Under ``cyclic``
every dataset gets a fixed fold per cycle (its key rank within its task, modulo the number
of folds), and the splits of a cycle hold out one fold each, so every kept dataset is held
out exactly once per cycle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
import yaml

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.ranking import set_rank_rule  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_jackknife", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

NESTED = "nested_policy_selected"
TRAIN_SELECTED = "train_selected_fixed_reference"

_ARTIFACT: dict[str, Any] | None = None
_POOL: list[str] = []


def subset_datasets(
    tasks: pd.Series,
    *,
    n_delete: int,
    rng: np.random.Generator,
) -> list[str]:
    """Delete ``n_delete`` datasets, proportionally to task sizes, keeping >= 3 per task."""
    counts = tasks.value_counts().sort_index()
    raw = counts / counts.sum() * n_delete
    per_task = np.floor(raw).astype(int)
    remainder = n_delete - int(per_task.sum())
    for task in (raw - per_task).sort_values(ascending=False).index[:remainder]:
        per_task[task] += 1
    kept: list[str] = []
    for task, count in counts.items():
        members = sorted(tasks[tasks == task].index.astype(str))
        delete = min(int(per_task[task]), max(0, len(members) - 3))
        dropped = set(rng.choice(np.array(members, dtype=object), size=delete, replace=False)) if delete else set()
        kept.extend(name for name in members if name not in dropped)
    return sorted(kept)


def crn_split(
    kept: list[str],
    tasks: pd.Series,
    keys: pd.Series,
    *,
    heldout_fraction: float,
) -> tuple[list[str], list[str]]:
    """Task-stratified split of ``kept`` holding out the lowest-key datasets per task."""
    heldout: list[str] = []
    kept_tasks = tasks.reindex(kept)
    for task in sorted(kept_tasks.unique()):
        members = kept_tasks[kept_tasks == task].index.astype(str)
        n_heldout = max(1, int(round(len(members) * heldout_fraction)))
        n_heldout = min(n_heldout, len(members) - 1)
        heldout.extend(keys.reindex(members).sort_values(kind="stable").index[:n_heldout])
    heldout = sorted(heldout)
    train = [name for name in kept if name not in set(heldout)]
    return train, heldout


def cyclic_split(
    kept: list[str],
    tasks: pd.Series,
    keys: pd.Series,
    *,
    fold: int,
    n_folds: int,
) -> tuple[list[str], list[str]]:
    """Hold out fold ``fold`` of each task, folds dealt by key rank among all its datasets.

    The folds are assigned over every dataset of the task before deletion, so a kept
    dataset keeps its fold in every subset.
    """
    kept_set = set(kept)
    heldout: list[str] = []
    for task in sorted(tasks.unique()):
        members = tasks[tasks == task].index.astype(str)
        order = keys.reindex(members).sort_values(kind="stable").index
        heldout.extend(name for position, name in enumerate(order) if position % n_folds == fold and name in kept_set)
    heldout = sorted(heldout)
    train = [name for name in kept if name not in set(heldout)]
    kept_tasks = tasks.reindex(kept)
    if any(not set(kept_tasks[kept_tasks == task].index) - set(heldout) for task in kept_tasks.unique()):
        raise ValueError("A cyclic split would leave a task without training datasets.")
    return train, heldout


def evaluate_task(task: tuple[int, int, list[str], list[str]]) -> dict[str, Any]:
    assert _ARTIFACT is not None
    subset_id, split_id, train, heldout = task
    kept = set(train) | set(heldout)
    artifact = dict(_ARTIFACT)
    artifact["candidates"] = _ARTIFACT["candidates"][_ARTIFACT["candidates"]["dataset"].isin(kept)]
    split_result, train_scores, _, unit_scores = bench._evaluate_artifact_split(
        artifact,
        split_id=split_id,
        train_datasets=train,
        heldout_datasets=heldout,
        split_mode="task_stratified_crn",
    )
    held = unit_scores[unit_scores["phase"].eq("heldout")]
    per_dataset = held.groupby(["method", "dataset"], as_index=False)["normalized_rank_regret"].mean()
    pool_train = train_scores[train_scores["method"].isin(_POOL)]
    return {
        "subset_id": subset_id,
        "split_id": split_id,
        "selected_policy": str(split_result["selected_policy"].iloc[0]),
        "per_dataset": per_dataset,
        "train_scores": pool_train[["method", "method_order", "mean_normalized_rank_regret", "mean_total_seconds"]],
    }


def contrasts_for_subset(results: list[dict[str, Any]], *, pool: list[str], contrasts: list[tuple[str, str]]) -> dict[str, float]:
    frames = []
    order = {method: index for index, method in enumerate(pool)}
    for result in results:
        per_dataset = result["per_dataset"]
        train = result["train_scores"].copy()
        train["_order"] = train["method"].map(order)
        chosen = train.sort_values(
            ["mean_normalized_rank_regret", "mean_total_seconds", "_order"], kind="mergesort"
        ).iloc[0]["method"]
        reference = per_dataset[per_dataset["method"].eq(chosen)].assign(method=TRAIN_SELECTED)
        frames.append(pd.concat([per_dataset, reference], ignore_index=True).assign(split_id=result["split_id"]))
    stacked = pd.concat(frames, ignore_index=True)
    means = stacked.groupby(["dataset", "method"])["normalized_rank_regret"].mean().unstack("method")
    return {
        f"{left} - {right}": float((means[left] - means[right]).dropna().mean())
        for left, right in contrasts
    }


def run(args: argparse.Namespace) -> None:
    global _ARTIFACT, _POOL
    started = time.time()
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config["selector"])
    selector.pop("reference_selectors", None)
    artifact_config["selector"] = selector
    artifact = bench._load_artifact(artifact_config, seed=int(benchmark.get("seed", 20260604)))
    review = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8"))
    _POOL = [str(value) for value in review["fixed_reference_pools"][args.reference_pool]]
    _ARTIFACT = artifact
    tasks = (
        artifact["candidates"][["dataset", "task"]].drop_duplicates().set_index("dataset")["task"].astype(str)
    )
    datasets = sorted(tasks.index.astype(str))
    heldout_fraction = float(benchmark.get("heldout_fraction", 0.25))
    n_folds = int(round(1.0 / heldout_fraction))
    if args.split_design == "cyclic" and int(args.splits) % n_folds:
        raise ValueError(f"--splits must be a multiple of {n_folds} folds under the cyclic design.")
    n_key_rows = int(args.splits) if args.split_design == "random" else int(args.splits) // n_folds
    rng_keys = np.random.default_rng(int(args.seed))
    keys = pd.DataFrame(rng_keys.uniform(size=(n_key_rows, len(datasets))), columns=datasets)
    subsets = []
    job_tasks = []
    for subset_id in range(int(args.subsets)):
        kept = subset_datasets(
            tasks,
            n_delete=int(args.drop),
            rng=np.random.default_rng(int(args.seed) + 1 + subset_id),
        )
        subsets.append(kept)
        for split_id in range(int(args.splits)):
            if args.split_design == "cyclic":
                train, heldout = cyclic_split(
                    kept, tasks, keys.iloc[split_id // n_folds], fold=split_id % n_folds, n_folds=n_folds
                )
            else:
                train, heldout = crn_split(kept, tasks, keys.iloc[split_id], heldout_fraction=heldout_fraction)
            job_tasks.append((subset_id, split_id, train, heldout))
    print(f"{len(job_tasks)} split evaluations over {len(subsets)} subsets", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = [tuple(pair.split(":")) for pair in args.contrasts]
    context = multiprocessing.get_context("fork")
    grouped: dict[int, list[dict[str, Any]]] = {}
    rows = []
    with ProcessPoolExecutor(max_workers=int(args.jobs), mp_context=context) as pool:
        for completed, result in enumerate(pool.map(evaluate_task, job_tasks, chunksize=1), start=1):
            grouped.setdefault(result["subset_id"], []).append(result)
            if len(grouped[result["subset_id"]]) == int(args.splits):
                subset_id = result["subset_id"]
                values = contrasts_for_subset(grouped.pop(subset_id), pool=_POOL, contrasts=contrasts)
                rows.append({"subset_id": subset_id, "n_datasets": len(subsets[subset_id]), **values})
                pd.DataFrame(rows).sort_values("subset_id").to_csv(output_dir / "subset_contrasts.csv", index=False)
            if completed % 50 == 0:
                print(f"{completed}/{len(job_tasks)} split evaluations", flush=True)
    frame = pd.DataFrame(rows).sort_values("subset_id")
    n = len(datasets)
    d = int(args.drop)
    summary = []
    for name in [f"{left} - {right}" for left, right in contrasts]:
        values = frame[name].to_numpy(dtype=float)
        se = math.sqrt((n - d) / d * float(np.var(values, ddof=1)))
        summary.append(
            {
                "contrast": name,
                "n_subsets": int(values.size),
                "subset_mean": float(values.mean()),
                "subset_sd": float(values.std(ddof=1)),
                "jackknife_se": se,
                "share_subsets_negative": float(np.mean(values < 0)),
                "share_subsets_positive": float(np.mean(values > 0)),
            }
        )
    pd.DataFrame(summary).to_csv(output_dir / "jackknife_summary.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": str(args.benchmark_config),
                "n_datasets": n,
                "drop": d,
                "subsets": int(args.subsets),
                "splits_per_subset": int(args.splits),
                "split_design": str(args.split_design),
                "mean_train_datasets": float(np.mean([len(train) for _, _, train, _ in job_tasks])),
                "mean_heldout_datasets": float(np.mean([len(heldout) for _, _, _, heldout in job_tasks])),
                "seed": int(args.seed),
                "reference_pool": args.reference_pool,
                "git": bench._git_state(),
                "source": bench._source_digest(),
                "wall_seconds": time.time() - started,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--reference-pool", default="declared_fixed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsets", type=int, default=200)
    parser.add_argument("--drop", type=int, default=10)
    parser.add_argument("--splits", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--split-design", choices=["random", "cyclic"], default="random")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=[
            f"{NESTED}:meta_selector_single",
            f"{NESTED}:meta_selector_family_prior_budgeted",
            f"{NESTED}:{TRAIN_SELECTED}",
            "meta_selector_family_prior_budgeted:meta_selector_single",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
