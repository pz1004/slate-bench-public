"""2^5 factorial decomposition of the family-prior scoring functional.

Factors: validation rank (w_v = 0.45), task-family prior (w_tf = 0.35), global-method
prior (w_gm = 0.15), runtime term (w_r = 0.05) and the 10000 s fit-time budget. A present
component keeps its full-rule weight and an absent one gets weight zero, without
renormalising the rest, so the relative weight of every remaining component is unchanged
(the v1 ablations moved removed weight into validation). Every cell is a fixed rule
evaluated on the same nested outer splits as the main audit: priors come from the
outer-training datasets, held-out test errors enter only the endpoint. A cell with no
score component picks uniformly among eligible candidates; its regret is the exact
expectation of that draw.

Outputs the regret of all 32 cells, Shapley values and main effects of each factor, and
pairwise interactions, each with a dataset-bootstrap interval (all are linear in the
per-dataset cell regrets).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import sys
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

from tabular_sota_benchmark.artifacts import (  # noqa: E402
    ARTIFACT_UNIT_COLUMNS,
    _apply_candidate_budget,
    _calibrated_selector_from_history,
    _with_rank_normalized_validation,
)
from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402
from tabular_sota_benchmark.ranking import rank_against_candidates, set_rank_rule  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_factorial", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

FACTORS = {
    "validation": ("validation_weight", 0.45),
    "task_family_prior": ("task_family_prior_weight", 0.35),
    "global_method_prior": ("global_method_prior_weight", 0.15),
    "runtime": ("runtime_weight", 0.05),
    "budget": (None, 10000.0),
}
FACTOR_NAMES = list(FACTORS)
WEIGHT_KEYS = [
    "validation_weight",
    "task_method_prior_weight",
    "task_family_prior_weight",
    "global_method_prior_weight",
    "runtime_weight",
]


def cells() -> list[dict[str, Any]]:
    result = []
    for mask in itertools.product([0, 1], repeat=len(FACTOR_NAMES)):
        present = {name for name, flag in zip(FACTOR_NAMES, mask) if flag}
        weights = {key: 0.0 for key in WEIGHT_KEYS}
        for name in present:
            key, value = FACTORS[name]
            if key is not None:
                weights[key] = float(value)
        cell_id = "+".join(name for name in FACTOR_NAMES if name in present) or "none"
        result.append(
            {
                "cell": cell_id,
                "present": present,
                "weights": weights,
                "max_budget_seconds": FACTORS["budget"][1] if "budget" in present else None,
            }
        )
    return result


_CANDIDATES: pd.DataFrame | None = None


def _ranked_heldout(heldout: pd.DataFrame, selector_rows: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([heldout, selector_rows], ignore_index=True)
    frame["total_seconds"] = frame["fit_seconds"].astype(float) + frame["predict_seconds"].astype(float)
    return rank_against_candidates(frame, unit_columns=ARTIFACT_UNIT_COLUMNS)


def evaluate_split(task: tuple[int, list[str], list[str]]) -> pd.DataFrame:
    """Per (cell, held-out dataset) mean normalized rank regret for one outer split."""
    assert _CANDIDATES is not None
    split_id, train_datasets, heldout_datasets = task
    train = _CANDIDATES[_CANDIDATES["dataset"].isin(set(train_datasets))].copy()
    heldout = _CANDIDATES[_CANDIDATES["dataset"].isin(set(heldout_datasets))].copy()
    endpoint = _ranked_heldout(heldout, heldout.iloc[0:0])
    candidate_regret = endpoint[endpoint["method_type"].eq("candidate")][
        [*ARTIFACT_UNIT_COLUMNS, "method", "normalized_rank_regret"]
    ]
    rows = []
    for cell in cells():
        if any(value > 0 for value in cell["weights"].values()):
            selector, _ = _calibrated_selector_from_history(
                train,
                heldout,
                method_name=cell["cell"],
                weight_grid=[cell["weights"]],
                weight_selection="lodo",
                max_budget_seconds=cell["max_budget_seconds"],
                budget_feature="fit_seconds",
            )
            ranked = _ranked_heldout(heldout, selector)
            unit_regret = ranked[ranked["method_type"].eq("selector")][
                [*ARTIFACT_UNIT_COLUMNS, "normalized_rank_regret"]
            ]
        else:
            eligible = _apply_candidate_budget(
                _with_rank_normalized_validation(heldout),
                max_budget_seconds=cell["max_budget_seconds"],
                budget_feature="fit_seconds",
            )[[*ARTIFACT_UNIT_COLUMNS, "method"]]
            unit_regret = (
                eligible.merge(candidate_regret, on=[*ARTIFACT_UNIT_COLUMNS, "method"], how="left")
                .groupby(ARTIFACT_UNIT_COLUMNS, as_index=False)["normalized_rank_regret"]
                .mean()
            )
        per_dataset = unit_regret.groupby("dataset", as_index=False)["normalized_rank_regret"].mean()
        per_dataset["cell"] = cell["cell"]
        per_dataset["split_id"] = int(split_id)
        rows.append(per_dataset)
    return pd.concat(rows, ignore_index=True)


def _cell_lookup() -> dict[frozenset[str], str]:
    return {frozenset(cell["present"]): cell["cell"] for cell in cells()}


def shapley_values(
    regret: pd.DataFrame,
    *,
    players: list[str],
    base: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Per-dataset Shapley value of each player (regret reduction from adding it).

    ``base`` components are present in every coalition, so with ``base = {validation}``
    the values split R(validation only) - R(full rule) among the other four components.
    """
    n = len(players)
    lookup = _cell_lookup()
    columns = {}
    for player in players:
        others = [name for name in players if name != player]
        total = pd.Series(0.0, index=regret.index)
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                without = lookup[base | frozenset(subset)]
                with_player = lookup[base | frozenset((*subset, player))]
                total = total + weight * (regret[without] - regret[with_player])
        columns[player] = total
    return pd.DataFrame(columns)


def main_effects(
    regret: pd.DataFrame,
    *,
    players: list[str],
    base: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Average regret reduction from adding a player, over every context of the others."""
    lookup = _cell_lookup()
    columns = {}
    for player in players:
        others = [name for name in players if name != player]
        contexts = [
            base | frozenset(subset)
            for size in range(len(others) + 1)
            for subset in itertools.combinations(others, size)
        ]
        columns[player] = sum(
            regret[lookup[context]] - regret[lookup[context | {player}]] for context in contexts
        ) / len(contexts)
    return pd.DataFrame(columns)


def pairwise_interactions(
    regret: pd.DataFrame,
    *,
    players: list[str],
    base: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Average two-way interaction in regret reduction over the contexts of the rest.

    Positive values mean the two components reduce regret by more together than the sum
    of their separate reductions (complements); negative values mean substitutes.
    """
    lookup = _cell_lookup()
    columns = {}
    for first, second in itertools.combinations(players, 2):
        others = [name for name in players if name not in {first, second}]
        contexts = [
            base | frozenset(subset)
            for size in range(len(others) + 1)
            for subset in itertools.combinations(others, size)
        ]
        total = 0.0
        for context in contexts:
            base_regret = regret[lookup[context]]
            both = regret[lookup[context | {first, second}]]
            only_first = regret[lookup[context | {first}]]
            only_second = regret[lookup[context | {second}]]
            total = total + (
                (base_regret - both) - (base_regret - only_first) - (base_regret - only_second)
            )
        columns[f"{first}:{second}"] = total / len(contexts)
    return pd.DataFrame(columns)


def summarise(values: pd.DataFrame, resampler: ArtifactResampler, *, label: str) -> pd.DataFrame:
    rows = []
    for column in values.columns:
        estimate, low, high = resampler.mean_ci(values[column])
        rows.append({label: column, "estimate": estimate, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    global _CANDIDATES
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config.get("selector", {}))
    selector.pop("reference_selectors", None)
    # Only candidate rows are needed: skip prediction ensembles and their vector loading.
    selector["policy_grid"] = {**dict(selector.get("policy_grid", {})), "tabrepo_ensemble_sizes": []}
    artifact_config["selector"] = selector
    artifact_config["artifact_source"] = {
        **dict(artifact_config["artifact_source"]),
        "load_predictions": False,
    }
    seed = int(benchmark.get("seed", 20260604))
    artifact = bench._load_artifact(artifact_config, seed=seed)
    _CANDIDATES = artifact["candidates"]
    splits = bench._split_datasets(
        _CANDIDATES,
        n_splits=int(args.splits or benchmark.get("splits", 100)),
        heldout_fraction=float(benchmark.get("heldout_fraction", 0.25)),
        seed=seed,
    )
    tasks = [(split_id, train, heldout) for split_id, train, heldout, _ in splits]
    if int(args.jobs) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=int(args.jobs), mp_context=context) as pool:
            frames = list(pool.map(evaluate_split, tasks))
    else:
        frames = [evaluate_split(task) for task in tasks]
    per_split = pd.concat(frames, ignore_index=True)
    regret = (
        per_split.groupby(["dataset", "cell"])["normalized_rank_regret"].mean().unstack("cell")
    )
    resampler = ArtifactResampler(regret.index, seed=int(args.bootstrap_seed))
    cell_rows = []
    for cell in cells():
        estimate, low, high = resampler.mean_ci(regret[cell["cell"]])
        cell_rows.append(
            {
                "cell": cell["cell"],
                **{f"has_{name}": int(name in cell["present"]) for name in FACTOR_NAMES},
                **{f"w_{key.replace('_weight', '')}": value for key, value in cell["weights"].items()},
                "max_budget_seconds": cell["max_budget_seconds"],
                "mean_normalized_rank_regret": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(output_dir / "factorial_split_dataset_regret.csv", index=False)
    regret.to_csv(output_dir / "factorial_dataset_regret.csv")
    pd.DataFrame(cell_rows).to_csv(output_dir / "factorial_cells.csv", index=False)
    full = "+".join(FACTOR_NAMES)
    # Primary decomposition: validation always present, the other four components are the
    # players (the 2^4 design asked for in review; it avoids crediting components with
    # the degenerate cells that have no validation and no prior, e.g. runtime alone).
    given_validation = [name for name in FACTOR_NAMES if name != "validation"]
    base = frozenset({"validation"})
    total_given = float(regret["validation"].mean() - regret[full].mean())
    shapley_given = summarise(
        shapley_values(regret, players=given_validation, base=base), resampler, label="factor"
    )
    shapley_given["share_of_total_reduction"] = (
        shapley_given["estimate"] / total_given if total_given else np.nan
    )
    shapley_given.to_csv(output_dir / "factorial_shapley_given_validation.csv", index=False)
    summarise(
        main_effects(regret, players=given_validation, base=base), resampler, label="factor"
    ).to_csv(output_dir / "factorial_main_effects_given_validation.csv", index=False)
    summarise(
        pairwise_interactions(regret, players=given_validation, base=base), resampler, label="pair"
    ).to_csv(output_dir / "factorial_interactions_given_validation.csv", index=False)
    # Full 2^5 decomposition, reported in the appendix.
    total = float(regret["none"].mean() - regret[full].mean())
    shapley = summarise(shapley_values(regret, players=FACTOR_NAMES), resampler, label="factor")
    shapley["share_of_total_reduction"] = shapley["estimate"] / total if total else np.nan
    shapley.to_csv(output_dir / "factorial_shapley_all.csv", index=False)
    # Validation's own contribution to the full rule (does validation add beyond priors?).
    no_validation = "+".join(name for name in FACTOR_NAMES if name != "validation")
    summarise(
        pd.DataFrame({"validation_given_all_other_components": regret[no_validation] - regret[full]}),
        resampler,
        label="contrast",
    ).to_csv(output_dir / "factorial_validation_contribution.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": str(args.benchmark_config),
                "n_splits": len(tasks),
                "n_datasets": resampler.n,
                "bootstrap_seed": int(args.bootstrap_seed),
                "factors": {name: {"weight_key": key, "value": value} for name, (key, value) in FACTORS.items()},
                "total_reduction_none_to_full": total,
                "total_reduction_validation_to_full": total_given,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
