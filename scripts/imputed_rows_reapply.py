"""The main run's fitted rules, reapplied to the candidate sets without the imputed rows.

Removing TabArena's imputed rows (scripts/imputed_rows.py) moves a fitted policy's held-out
regret in three steps. The yardstick changes, because units have fewer candidates to rank
against. The rule's output can change with every fitted quantity held fixed, because it is
applied to a unit with fewer candidates: validation ranks and their normalization, the
fit-time budget and its fallback, and a guard's score-table summaries of the target are
recomputed from the target's own rows. And refitting on the reduced table changes the fitted
quantities and the nested choice. scripts/imputed_rows_rescore.py measures the first step
(the main run's choices, scored on the reduced candidate sets); the rerun without the rows
(configs/score_level_tabarena_no_imputed.yaml) measures all three.

This script measures the first two. On every outer split it fits each policy on the training
datasets' rows of the full results file, as the main run did, applies it to the held-out
datasets' rows of the reduced file, and scores the outputs against the reduced candidate
sets. Nested selection keeps the main run's policy in each split, and the train-selected rule
the main review's choice. Held-out rows do not depend on how the adaptive policies' inner
estimates are fitted (they read the source rows built before the per-target rebuild), so the
script uses the shared inner fit, which is far faster.

Before anything else, with the full file on the held-out side the same code must reproduce the
main run's held-out rows exactly on the first ``--check-splits`` splits. Afterwards,
validation-best's dataset-level regrets must equal the rerun's, since it fits nothing, and the
family-prior rule's chosen weights must equal the main run's in every split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from tabular_sota_benchmark.inference import ArtifactResampler, contrast_table, paired_contrast  # noqa: E402
from tabular_sota_benchmark.ranking import set_rank_rule  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_reapply", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)

NESTED = bench.NESTED_SELECTED_METHOD
VALIDATION_BEST = "meta_selector_single"
FAMILY_PRIOR = "meta_selector_family_prior_budgeted"
TRAIN_SELECTED = "train_selected_fixed_reference__declared_fixed"
PRIMARY = [
    "meta_selector_task_adaptive_budgeted",
    "meta_selector_meta_feature_guard_005",
    FAMILY_PRIOR,
    TRAIN_SELECTED,
    VALIDATION_BEST,
]
UNIT = ["dataset", "fold", "seed"]
KEEP = ["split_id", *UNIT, "method", "selected_member", "ensemble_weights", "normalized_rank_regret"]

_WORKER: dict[str, Any] = {}


def load(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact = bench._load_artifact(dict(config["artifacts"][0]), seed=int(benchmark.get("seed", 20260604)))
    artifact["selector"] = {**artifact["selector"], "inner_source_fit": "shared"}
    return artifact


def by_datasets(frame: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    return frame[frame["dataset"].astype(str).isin(set(datasets))]


def evaluate(task: tuple[int, list[str], list[str], str, bool]) -> pd.DataFrame:
    """Held-out unit rows of one split: training rows from the full file, held-out rows from
    the reduced one (or from the full one when ``reproduce``)."""
    split_id, train, heldout, mode, reproduce = task
    full, reduced = _WORKER["full"], _WORKER["reduced"]
    target = full if reproduce else reduced
    # Static rows (validation-best and validation rank) depend on a unit's own candidates
    # only, and are stored per method for the whole file.
    static = {
        method: pd.concat(
            [by_datasets(frame, train), by_datasets(target["static_selector_rows"][method], heldout)], ignore_index=True
        )
        for method, frame in full["static_selector_rows"].items()
    }
    hybrid = {
        **full,
        "candidates": pd.concat([by_datasets(full["candidates"], train), by_datasets(target["candidates"], heldout)], ignore_index=True),
        "static_selector_rows": static,
    }
    _, _, _, units = bench._evaluate_artifact_split(
        hybrid, split_id=split_id, train_datasets=train, heldout_datasets=heldout, split_mode=mode
    )
    held = units[units["phase"].eq("heldout") & ~units["method"].eq(NESTED)]
    return held[KEEP].reset_index(drop=True)


def run_splits(tasks: list[tuple[int, list[str], list[str], str, bool]], jobs: int) -> pd.DataFrame:
    if jobs <= 1:
        return pd.concat([evaluate(task) for task in tasks], ignore_index=True)
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        return pd.concat(list(pool.map(evaluate, tasks)), ignore_index=True)


def with_references(units: pd.DataFrame, selected: pd.Series, reference_choices: pd.DataFrame) -> pd.DataFrame:
    """Add nested selection (the main run's policy per split) and the train-selected rule."""
    frames = [units]
    nested = units.merge(selected.rename("method").reset_index(), on=["split_id", "method"])
    frames.append(nested.assign(method=NESTED))
    picks = reference_choices[reference_choices["pool"].eq("declared_fixed")][["split_id", "selected_reference_method"]]
    chosen = units.merge(picks, left_on=["split_id", "method"], right_on=["split_id", "selected_reference_method"])
    frames.append(chosen.drop(columns="selected_reference_method").assign(method=TRAIN_SELECTED))
    return pd.concat(frames, ignore_index=True)


def dataset_means(units: pd.DataFrame) -> pd.DataFrame:
    per_split = units.groupby(["method", "split_id", "dataset"])["normalized_rank_regret"].mean().reset_index()
    return per_split.groupby(["dataset", "method"])["normalized_rank_regret"].mean().unstack("method")


def weights(frame: pd.DataFrame) -> pd.Series:
    """The family-prior rule's chosen weight tuple per split (a fitted quantity)."""
    rows = frame[frame["method"].eq(FAMILY_PRIOR)]
    tuples = rows["ensemble_weights"].astype(str).str.split(";").map(
        lambda parts: ";".join(part for part in parts if part.startswith("w_"))
    )
    return tuples.groupby(rows["split_id"]).agg(lambda values: "|".join(sorted(set(values))))


def run(args: argparse.Namespace) -> None:
    global _WORKER
    started = time.time()
    review_config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    full, reduced = load(Path(args.main_config)), load(Path(args.reduced_config))
    _WORKER = {"full": full, "reduced": reduced}
    seed = int(yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))["benchmark"]["seed"])
    splits = bench._split_datasets(full["candidates"], n_splits=100, heldout_fraction=0.25, seed=seed)
    if splits != bench._split_datasets(reduced["candidates"], n_splits=100, heldout_fraction=0.25, seed=seed):
        raise ValueError("The two files give different outer splits.")

    main_units = pd.read_parquet(Path(args.main_output) / "unit_scores.parquet", columns=["phase", *KEEP])
    main_units = main_units[main_units["phase"].eq("heldout") & ~main_units["method"].eq(NESTED)].drop(columns="phase")
    check_ids = [split_id for split_id, *_ in splits[: int(args.check_splits)]]
    check = run_splits([(*split, True) for split in splits[: int(args.check_splits)]], int(args.jobs))
    key = ["split_id", *UNIT, "method"]
    expected = main_units[main_units["split_id"].isin(check_ids)].set_index(key).sort_index()
    got = check.set_index(key).sort_index()
    if not got.index.equals(expected.index) or not (got["selected_member"] == expected["selected_member"]).all() or not np.allclose(
        got["normalized_rank_regret"].to_numpy(), expected["normalized_rank_regret"].to_numpy(), rtol=0, atol=1e-12
    ):
        raise ValueError("With the full file the hybrid split does not reproduce the main run's held-out rows.")
    print(f"reproduces the main run on {len(check_ids)} split(s)", flush=True)
    if args.check_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reproduction_check.json").write_text(
            json.dumps({"reproduced_splits": len(check_ids), "wall_seconds": time.time() - started, "command": sys.argv}, indent=2),
            encoding="utf-8",
        )
        return

    units = run_splits([(*split, False) for split in splits], int(args.jobs))
    selected = pd.read_csv(Path(args.main_output) / "split_results.csv").set_index("split_id")["selected_policy"]
    choices = pd.read_csv(Path(args.main_review) / "train_selected_reference_choices.csv")
    units = with_references(units, selected, choices)
    means = dataset_means(units)

    rerun = pd.read_csv(Path(args.rerun_review) / "dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    gap = float((means[VALIDATION_BEST] - rerun[VALIDATION_BEST].reindex(means.index)).abs().max())
    if gap > 1e-9:
        raise ValueError(f"Reapplied validation-best does not equal the rerun's (max gap {gap:.3g}).")
    main_weights = weights(main_units)
    weight_mismatch = int((weights(units).reindex(main_weights.index) != main_weights).sum())
    if weight_mismatch:
        raise ValueError(f"The family-prior weights differ from the main run's in {weight_mismatch} split(s).")

    bootstrap = dict(review_config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        means.index, seed=int(bootstrap.get("seed", 20260608)), n_resamples=int(bootstrap.get("resamples", 10_000))
    )
    margins = [float(value) for value in review_config.get("margins", [0.005, 0.010, 0.015, 0.020])]
    rows = [
        {"family": "reapplied_primary", "left": NESTED, "right": right,
         **paired_contrast(means[NESTED], means[right], resampler, margins=margins, rank_scale=float(review_config["rank_scale"]))}
        for right in PRIMARY
    ]
    contrasts = contrast_table(rows)

    main = pd.read_csv(Path(args.main_review) / "dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    rescored = pd.read_csv(Path(args.rescored_output) / "rescored_summary.csv").set_index("method")["rescored_regret"]
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "main_regret": float(main[method].mean()),
                "rescored_regret": float(rescored.get(method, np.nan)),
                "reapplied_regret": float(means[method].mean()),
                "rerun_regret": float(rerun[method].mean()),
            }
            for method in [NESTED, *PRIMARY]
        ]
    )
    changed = units[units["method"].eq(NESTED)].merge(
        main_units.merge(selected.rename("method").reset_index(), on=["split_id", "method"]),
        on=["split_id", *UNIT], suffixes=("", "_main"),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    units.to_parquet(output_dir / "heldout_unit_scores.parquet", index=False)
    means.to_csv(output_dir / "dataset_mean_scores.csv")
    summary.to_csv(output_dir / "reapplied_summary.csv", index=False)
    contrasts.to_csv(output_dir / "reapplied_contrasts.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "main_config": str(args.main_config),
                "reduced_config": str(args.reduced_config),
                "main_output": str(args.main_output),
                "main_review": str(args.main_review),
                "rerun_review": str(args.rerun_review),
                "rescored_output": str(args.rescored_output),
                "inner_source_fit": "shared (held-out rows do not depend on it)",
                "reproduced_splits": check_ids,
                "validation_best_rerun_max_gap": gap,
                "family_prior_weight_mismatches": weight_mismatch,
                "nested_member_changed_share": float((changed["selected_member"] != changed["selected_member_main"]).mean()),
                "wall_seconds": time.time() - started,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(4).to_string(index=False))
    print(contrasts[["right", "mean_difference", "ci_low", "ci_high", "bootstrap_holm_p", "ttest_holm_p"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-config", type=Path, default=Path("configs/score_level_tabarena.yaml"))
    parser.add_argument("--reduced-config", type=Path, default=Path("configs/score_level_tabarena_no_imputed.yaml"))
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--main-review", type=Path, required=True)
    parser.add_argument("--rerun-review", type=Path, required=True)
    parser.add_argument("--rescored-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--check-splits", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check-only", action="store_true", help="only reproduce the main run on --check-splits splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
