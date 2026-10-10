"""Rank-rule sensitivity of the TabRepo fixed-ensemble rows.

An output that is not itself a candidate, such as a prediction ensemble, can differ from a
tied candidate's error only in the last digits. The legacy tie rule counted a candidate
lying strictly below such an output, but within the tie tolerance, both as better and as
tied, which inflates the output's rank. This script re-ranks the fixed-ensemble outputs of
the nested TabRepo run under both rules against the same candidates and reports the
dataset-level mean normalized rank regret. The consistent rule must reproduce the stored
per-unit regrets exactly; the script stops otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

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

from tabular_sota_benchmark.artifacts import ARTIFACT_UNIT_COLUMNS  # noqa: E402
from tabular_sota_benchmark.ranking import candidate_relative_ranks, set_rank_rule  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_rank_rule", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

ENSEMBLES = [
    "meta_selector_tabrepo_ensemble_k3",
    "meta_selector_tabrepo_ensemble_k5",
    "meta_selector_tabrepo_ensemble_k10",
]


def load_candidates(config_path: Path) -> pd.DataFrame:
    """Candidate rows of the pinned slice; prediction vectors are not needed here."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = dict(config.get("benchmark", {}))
    set_rank_rule(str(benchmark.get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config.get("selector", {}))
    selector.pop("reference_selectors", None)
    selector["policy_grid"] = {**dict(selector.get("policy_grid", {})), "tabrepo_ensemble_sizes": []}
    artifact_config["selector"] = selector
    artifact_config["artifact_source"] = {**dict(artifact_config["artifact_source"]), "load_predictions": False}
    artifact = bench._load_artifact(artifact_config, seed=int(benchmark.get("seed", 20260604)))
    return artifact["candidates"]


def rerank(candidates: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        key: frame["primary_metric"].to_numpy(dtype=float)
        for key, frame in candidates.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)
    }
    pieces = []
    for key, frame in outputs.groupby(ARTIFACT_UNIT_COLUMNS, sort=False):
        pool = grouped[key]
        values = frame["primary_metric"].to_numpy(dtype=float)
        scale = max(len(pool) - 1, 1)
        consistent = (candidate_relative_ranks(pool, values, rule="consistent") - 1.0) / scale
        legacy = (candidate_relative_ranks(pool, values, rule="legacy") - 1.0) / scale
        pieces.append(frame.assign(consistent_regret=consistent, legacy_regret=legacy))
    return pd.concat(pieces, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    candidates = load_candidates(Path(args.benchmark_config))
    columns = [*ARTIFACT_UNIT_COLUMNS, "split_id", "phase", "method", "primary_metric", "normalized_rank_regret"]
    units = pd.read_parquet(Path(args.nested_output) / "unit_scores.parquet", columns=columns)
    outputs = units[units["phase"].eq("heldout") & units["method"].isin(ENSEMBLES)]
    ranked = rerank(candidates, outputs)
    mismatch = float((ranked["consistent_regret"] - ranked["normalized_rank_regret"]).abs().max())
    if mismatch > 1e-12:
        raise ValueError(f"Consistent re-ranking does not reproduce the stored regrets (max diff {mismatch})")
    rows = []
    for method, frame in ranked.groupby("method", sort=False):
        per_dataset = frame.groupby("dataset")[["consistent_regret", "legacy_regret"]].mean()
        changed = frame.drop_duplicates(ARTIFACT_UNIT_COLUMNS)
        rows.append(
            {
                "method": method,
                "n_datasets": int(per_dataset.shape[0]),
                "consistent_mean_normalized_rank_regret": float(per_dataset["consistent_regret"].mean()),
                "legacy_mean_normalized_rank_regret": float(per_dataset["legacy_regret"].mean()),
                "share_units_rank_changed": float(
                    np.mean(np.abs(changed["legacy_regret"] - changed["consistent_regret"]) > 1e-12)
                ),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "rank_rule_sensitivity.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"command": sys.argv, "max_consistent_mismatch": mismatch}, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-config", type=Path, default=Path("configs/score_level_tabrepo_nested.yaml"))
    parser.add_argument("--nested-output", type=Path, default=Path("analysis-output/v2/score_level_tabrepo_nested"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output/v2/tabrepo_rank_rule_sensitivity"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
