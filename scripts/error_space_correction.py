"""A deployable error-space optimism correction, evaluated on the nested run's outer splits.

Remark 1 concerns a correction in the metric itself: add to each candidate's validation
error an offset estimated on other datasets, which helps on average when the offsets'
spread exceeds their estimation error. The correction the main runs evaluate works on
validation ranks instead. This script evaluates the error-space version on held-out data.

For a unit u, let s_u be the range of its candidates' validation errors (``--scale range``)
or the median candidate's validation error minus the best (``--scale median``, the
normalization the paper uses as a sensitivity); either is a scale a rule can read on a new
dataset. On each outer split, a method's offset is the mean over
the split's training datasets of (test error - validation error) / s_u, averaged over a
dataset's units first. On a held-out unit the rule selects the candidate with the lowest
validation error + offset * s_u (+ 1e-6 log(1 + fit time), validation-best's tie-break).
With ``--family-offsets`` a candidate takes the mean offset of its learner family instead.
The selected candidate is scored by the review's normalized rank regret, and the rule is
compared with validation-best, the rank-space correction and nested selection on the
review's dataset-level scores and shared resample matrix. Before anything else, the same
code with zero offsets must reproduce the review's validation-best regrets exactly. The
metadata also records how often the median candidate's validation error lies within one
percent of the range from the best, where the median scale is nearly zero.
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

from tabular_sota_benchmark.artifacts import ARTIFACT_UNIT_COLUMNS, _method_family  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, contrast_table, paired_contrast  # noqa: E402
from tabular_sota_benchmark.ranking import grouped_within_unit_ranks  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_error_space", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

UNIT = list(ARTIFACT_UNIT_COLUMNS)
VALIDATION_BEST = "meta_selector_single"
RANK_CORRECTION = "reference_optimism_corrected"
NESTED = "nested_policy_selected"


def load_candidates(config: dict) -> pd.DataFrame:
    """The artifact's candidate rows, without prediction ensembles or their vectors."""
    bench.set_rank_rule(str(dict(config.get("benchmark", {})).get("rank_rule", "consistent")))
    artifact_config = dict(config["artifacts"][0])
    artifact_config.pop("standard_meta_features", None)
    selector = dict(artifact_config.get("selector", {}))
    selector.pop("reference_selectors", None)
    selector["policy_grid"] = {**dict(selector.get("policy_grid", {})), "tabrepo_ensemble_sizes": []}
    artifact_config["selector"] = selector
    artifact_config["artifact_source"] = {**dict(artifact_config["artifact_source"]), "load_predictions": False}
    artifact = bench._load_artifact(artifact_config, seed=int(dict(config.get("benchmark", {})).get("seed", 20260604)))
    frame = artifact["candidates"]
    frame = frame[frame["method_type"].astype(str).eq("candidate")].reset_index(drop=True)
    return frame


def prepare(frame: pd.DataFrame, *, scale: str = "range") -> pd.DataFrame:
    out = frame[[*UNIT, "method", "primary_metric", "val_primary_metric", "fit_seconds"]].copy()
    out["unit"] = pd.factorize(pd.MultiIndex.from_frame(out[UNIT].astype(str)), sort=False)[0]
    ranks = grouped_within_unit_ranks(out, unit_columns=UNIT, value_column="primary_metric")
    size = out.groupby("unit")["method"].transform("size").astype(float)
    out["regret"] = (ranks - 1.0) / (size - 1.0)
    val = out.groupby("unit")["val_primary_metric"]
    upper = val.transform("max") if scale == "range" else val.transform("median")
    out["scale"] = (upper - val.transform("min")).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z"] = np.where(out["scale"] > 0, (out["primary_metric"] - out["val_primary_metric"]) / out["scale"], np.nan)
    out["tie"] = 1e-6 * np.log1p(out["fit_seconds"].astype(float).clip(lower=0))
    out["family"] = out["method"].map(_method_family)
    return out


def dataset_offsets(frame: pd.DataFrame, *, by: str) -> pd.DataFrame:
    """Dataset-by-key matrix of mean normalized test-minus-validation error."""
    per_unit = frame.groupby(["dataset", "unit", by])["z"].mean().reset_index()
    return per_unit.groupby(["dataset", by])["z"].mean().unstack(by)


def evaluate(frame: pd.DataFrame, held_out: pd.DataFrame, offsets: pd.DataFrame | None, *, by: str) -> pd.Series:
    """Dataset-level mean held-out regret of the corrected rule (offsets None: validation-best)."""
    datasets = sorted(frame["dataset"].unique())
    rows = []
    for split_id, group in held_out.groupby("split_id", sort=True):
        heldout = set(group["dataset"])
        units = frame[frame["dataset"].isin(heldout)]
        if offsets is None:
            score = units["val_primary_metric"] + units["tie"]
        else:
            train = [dataset for dataset in datasets if dataset not in heldout]
            offset = offsets.loc[offsets.index.intersection(train)].mean(axis=0, skipna=True)
            shift = units[by].map(offset).fillna(0.0) * units["scale"].fillna(0.0)
            score = units["val_primary_metric"] + shift + units["tie"]
        chosen = units.loc[score.groupby(units["unit"]).idxmin()]
        per_dataset = chosen.groupby("dataset")["regret"].mean()
        rows.append(pd.DataFrame({"split_id": split_id, "dataset": per_dataset.index, "regret": per_dataset.values}))
    split_scores = pd.concat(rows, ignore_index=True)
    return split_scores.groupby("dataset")["regret"].mean()


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    review_config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    review_dir = Path(args.review_output)
    frame = prepare(load_candidates(config), scale=str(args.scale))
    val = frame.groupby("unit")["val_primary_metric"]
    low = val.min()
    span = val.max() - low
    median_near_best = ((val.median() - low) / span.where(span > 0)).dropna().lt(0.01)
    held_out = pd.read_parquet(review_dir / "dataset_policy_scores.parquet", columns=["split_id", "dataset"]).drop_duplicates()
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")

    check = evaluate(frame, held_out, None, by="method")
    gap = float((check - regret[VALIDATION_BEST].reindex(check.index)).abs().max())
    if gap > 1e-9:
        raise ValueError(f"Zero offsets do not reproduce validation-best (max gap {gap:.3g}).")

    rules = {"error_space_correction": evaluate(frame, held_out, dataset_offsets(frame, by="method"), by="method")}
    if args.family_offsets:
        rules["error_space_correction_family"] = evaluate(
            frame, held_out, dataset_offsets(frame, by="family"), by="family"
        )
    bootstrap = dict(review_config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        regret.index, seed=int(bootstrap.get("seed", 20260608)), n_resamples=int(bootstrap.get("resamples", 10_000))
    )
    margins = [float(value) for value in review_config.get("margins", [0.005, 0.010, 0.015, 0.020])]
    rows = []
    for name, values in rules.items():
        for right in [VALIDATION_BEST, RANK_CORRECTION, NESTED]:
            if right not in regret.columns:
                continue
            result = paired_contrast(
                values.reindex(regret.index), regret[right], resampler,
                margins=margins, rank_scale=float(review_config["rank_scale"]),
            )
            rows.append({"family": "error_space", "left": name, "right": right,
                         "left_mean": float(values.mean()), "right_mean": float(regret[right].mean()), **result})
    contrasts = contrast_table(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rules)
    scores.index.name = "dataset"
    scores.to_csv(output_dir / "dataset_scores.csv")
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": str(args.benchmark_config),
                "review_output": str(review_dir),
                "review_config": str(args.review_config),
                "validation_best_reproduction_max_gap": gap,
                "n_datasets": int(resampler.n),
                "n_splits": int(held_out["split_id"].nunique()),
                "units_without_scale": int(frame.loc[~(frame["scale"] > 0), "unit"].nunique()),
                "share_units_median_within_one_percent_of_best": float(median_near_best.mean()),
                "family_offsets": bool(args.family_offsets),
                "scale": str(args.scale),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cols = ["left", "right", "left_mean", "right_mean", "mean_difference", "ci_low", "ci_high", "bootstrap_holm_p"]
    print(contrasts[cols].to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-offsets", action="store_true")
    parser.add_argument("--scale", choices=["range", "median"], default="range")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
