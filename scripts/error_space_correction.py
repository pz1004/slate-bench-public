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

The script also computes Remark 1's spread-to-noise ratio on the offsets this correction
uses (``offset_snr_matched.csv``): the dataset-by-candidate matrix of mean normalized test
minus validation error on the correction's validation-derived scale, with the standard error
of an offset estimated from one outer split's training datasets. It reports the ratio for
every candidate observed on every dataset (the set the correction corrects), for the tenth
with the best mean validation rank, and, with ``--family-offsets``, for learner families.
``scripts/validation_provenance.py`` computes the same ratio on test-range-normalized offsets;
with ``--provenance-output`` the script first checks that it reproduces that ratio when given
the test-range scale and the same candidates.
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
from scipy import stats

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
from tabular_sota_benchmark.provenance import offset_snr, top_tenth  # noqa: E402
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


def top_validation_tenth(frame: pd.DataFrame) -> list[str]:
    """The tenth of candidates with the best mean validation rank, as the provenance script
    chooses them: within-unit average ranks, averaged over a dataset's units, then datasets."""
    ranked = frame.assign(
        val_rank=frame.groupby("unit")["val_primary_metric"].rank(method="average")
    )
    per_dataset = ranked.groupby(["method", "dataset"])["val_rank"].mean().reset_index()
    table = per_dataset.groupby("method", as_index=False)["val_rank"].mean()
    return top_tenth(table)["method"].tolist()


def matched_snr(
    frame: pd.DataFrame, *, n_train: int, resampler: ArtifactResampler, family_offsets: bool, scale: str
) -> pd.DataFrame:
    """Remark 1's ratio on the offsets the correction estimates (columns observed everywhere)."""
    rows = []
    sets = [("all", "method", dataset_offsets(frame, by="method"))]
    top = top_validation_tenth(frame)
    sets.append(("top_tenth", "method", dataset_offsets(frame[frame["method"].isin(top)], by="method")))
    if family_offsets:
        sets.append(("all", "family", dataset_offsets(frame, by="family")))
    for candidates, by, matrix in sets:
        complete = matrix.dropna(axis=1, how="any")
        result = offset_snr(complete, n_train=n_train, resampler=resampler)
        association = offset_quality(frame if candidates == "all" else frame[frame["method"].isin(top)], by=by)
        rows.append({"scale": scale, "candidates": candidates, "by": by, "n_dropped": int(matrix.shape[1] - complete.shape[1]), **result, **association})
    return pd.DataFrame(rows)


def offset_quality(frame: pd.DataFrame, *, by: str) -> dict[str, float]:
    """Spearman correlation, across candidates (or families), between the mean normalized
    test-minus-validation error and the mean test rank regret. A negative value means the
    optimistic candidates are the better ones on test, the case in which Proposition 1(iv)
    says that removing the offsets entirely is not optimal."""
    offsets = dataset_offsets(frame, by=by).mean(axis=0)
    regret = frame.groupby(["dataset", by])["regret"].mean().unstack(by).mean(axis=0)
    common = offsets.index.intersection(regret.index)
    rho, p = stats.spearmanr(offsets[common], regret[common])
    return {"offset_quality_spearman": float(rho), "offset_quality_p": float(p), "offset_quality_n": int(len(common))}


def reproduce_provenance_snr(frame: pd.DataFrame, provenance_dir: Path, *, n_train: int, resampler: ArtifactResampler) -> float:
    """With the test-range scale and the provenance script's candidates, the matched code must
    give the provenance script's error-space ratio."""
    stored = pd.read_csv(provenance_dir / "offset_snr.csv").set_index("space")
    top = pd.read_csv(provenance_dir / "offsets_top_candidates.csv")["method"].unique().tolist()
    units = frame.groupby("unit")["primary_metric"]
    span = units.transform("max") - units.transform("min")
    test_scaled = frame.assign(z=(frame["primary_metric"] - frame["val_primary_metric"]) / span.where(span > 0))
    matrix = dataset_offsets(test_scaled[test_scaled["method"].isin(top)], by="method")
    ratio = float(offset_snr(matrix, n_train=n_train, resampler=resampler)["ratio"])
    gap = abs(ratio - float(stored.loc["error", "ratio"]))
    if gap > 1e-9:
        raise ValueError(f"The matched code does not reproduce the provenance ratio (gap {gap:.3g}).")
    return gap


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
    heldout_counts = held_out.groupby("split_id")["dataset"].nunique().unique()
    if len(heldout_counts) != 1:
        raise ValueError("The outer splits hold out different numbers of datasets.")
    n_train = int(resampler.n - heldout_counts[0])
    reproduction_gap = (
        reproduce_provenance_snr(frame, Path(args.provenance_output), n_train=n_train, resampler=resampler)
        if args.provenance_output
        else None
    )
    snr = matched_snr(frame, n_train=n_train, resampler=resampler, family_offsets=bool(args.family_offsets), scale=str(args.scale))
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
    snr.to_csv(output_dir / "offset_snr_matched.csv", index=False)
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
                "snr_n_train": n_train,
                "snr_reproduces_provenance_gap": reproduction_gap,
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
    parser.add_argument(
        "--provenance-output", type=Path, default=None,
        help="scripts/validation_provenance.py output whose error-space ratio the matched code must reproduce",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
