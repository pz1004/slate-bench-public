"""How far runtime-perturbed validation-best is from a lexicographic tie-break.

The validation-best row of every configuration selects the candidate with the lowest
validation error plus ``runtime_weight * log(1 + fit seconds)`` (``runtime_weight`` is
10^-6 in the declared configs). An additive term can reorder two unequal validation errors
that differ by less than it, so the rule is not a pure tie-break. This script compares it,
unit by unit, with the lexicographic rule (lowest validation error, then lowest fit time,
then candidate order) and reports how many units differ, in how many the selected test
error differs, and the dataset-level mean normalized rank regret under both rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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

from tabular_sota_benchmark.artifacts import ARTIFACT_UNIT_COLUMNS  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_tie_rule", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def picks(candidates: pd.DataFrame, *, runtime_weight: float) -> pd.DataFrame:
    """The perturbed and the lexicographic validation-best candidate of every unit."""
    frame = candidates.reset_index(drop=True).copy()
    frame["_order"] = np.arange(len(frame))
    frame["_perturbed"] = frame["val_primary_metric"] + runtime_weight * np.log1p(frame["fit_seconds"])
    # np.argmin within a unit breaks exact ties by candidate order, as the selector does.
    perturbed = frame.loc[frame.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)["_perturbed"].idxmin()]
    lexicographic = frame.sort_values(
        [*ARTIFACT_UNIT_COLUMNS, "val_primary_metric", "fit_seconds", "_order"], kind="stable"
    ).groupby(ARTIFACT_UNIT_COLUMNS, sort=False).head(1)
    return perturbed.drop(columns=["_order", "_perturbed"]), lexicographic.drop(columns=["_order", "_perturbed"])


def compare(candidates: pd.DataFrame, *, runtime_weight: float) -> tuple[dict[str, float], pd.DataFrame]:
    perturbed, lexicographic = picks(candidates, runtime_weight=runtime_weight)
    selectors = []
    for name, rows in [("perturbed", perturbed), ("lexicographic", lexicographic)]:
        rows = rows.copy()
        rows["selected_member"] = rows["method"]
        rows["method"] = f"validation_best_{name}"
        rows["method_type"] = "selector"
        selectors.append(rows)
    ranked = bench._rank_against_candidates(pd.concat([candidates, *selectors], ignore_index=True))
    rules = ranked[ranked["method"].astype(str).str.startswith("validation_best_")]
    per_unit = rules.pivot_table(
        index=ARTIFACT_UNIT_COLUMNS, columns="method", values="normalized_rank_regret", aggfunc="first"
    )
    members = rules.pivot_table(index=ARTIFACT_UNIT_COLUMNS, columns="method", values="selected_member", aggfunc="first")
    tests = rules.pivot_table(index=ARTIFACT_UNIT_COLUMNS, columns="method", values="primary_metric", aggfunc="first")
    per_dataset = per_unit.groupby(level="dataset").mean()
    difference = per_dataset["validation_best_lexicographic"] - per_dataset["validation_best_perturbed"]
    summary = {
        "n_units": int(len(per_unit)),
        "units_with_other_pick": int((members["validation_best_lexicographic"] != members["validation_best_perturbed"]).sum()),
        "units_with_other_test_error": int(
            (tests["validation_best_lexicographic"] != tests["validation_best_perturbed"]).sum()
        ),
        "datasets_affected": int((difference != 0).sum()),
        "regret_perturbed": float(per_dataset["validation_best_perturbed"].mean()),
        "regret_lexicographic": float(per_dataset["validation_best_lexicographic"].mean()),
        "mean_difference": float(difference.mean()),
    }
    return summary, difference.rename("lexicographic_minus_perturbed").reset_index()


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    bench.set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config.get("selector", {}))
    runtime_weight = float(selector.get("runtime_weight", 0.0))
    selector.pop("reference_selectors", None)
    selector["policy_grid"] = {**dict(selector.get("policy_grid", {})), "tabrepo_ensemble_sizes": []}
    artifact_config["selector"] = selector
    artifact_config["artifact_source"] = {**dict(artifact_config["artifact_source"]), "load_predictions": False}
    artifact = bench._load_artifact(artifact_config, seed=int(benchmark.get("seed", 20260604)))
    summary, per_dataset = compare(artifact["candidates"], runtime_weight=runtime_weight)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"artifact": artifact["name"], "runtime_weight": runtime_weight, **summary}]).to_csv(
        output_dir / "tie_rule_summary.csv", index=False
    )
    per_dataset.to_csv(output_dir / "tie_rule_by_dataset.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"benchmark_config": str(args.benchmark_config), "command": sys.argv}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
