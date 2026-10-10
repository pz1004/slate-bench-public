"""Dataset-level review of one nested score-level run.

Reads the nested benchmark's per-unit parquet, training scores and split results, and
writes every dataset-level quantity the manuscript reports for that run:

* dataset-by-split and dataset-mean scores for every policy, reference row, the oracle
  and the nested alias (normalized rank regret, max-min and median-normalized error
  regret, the rate of beating the best candidate, recorded runtime);
* train-selected fixed references for each declared reference pool;
* a clustered summary with 95% intervals from the artifact's single resample matrix,
  plus the gap-closed ratio between the single best solver and the oracle;
* the declared contrast families, each contrast reported with the full decision rule
  (``tabular_sota_benchmark.inference.paired_contrast``) and Holm-adjusted within family;
* task-stratified summaries and selection counts.

Nothing is recomputed from the artifact: the benchmark's unit rows are the single source
of every score, so the review cannot drift from the nested run it reviews.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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

from tabular_sota_benchmark.inference import (  # noqa: E402
    ArtifactResampler,
    contrast_table,
    gap_closed,
    paired_contrast,
)
from tabular_sota_benchmark.labels import policy_label  # noqa: E402


NESTED = "nested_policy_selected"
ORACLE = "oracle"
TRAIN_SELECTED_PREFIX = "train_selected_fixed_reference__"
METRICS = [
    "mean_normalized_rank_regret",
    "mean_maxmin_error_regret",
    "mean_median_error_regret",
    "beats_best_rate",
    "mean_total_seconds",
    "mean_rank",
]


def load_nested_output(nested_dir: Path) -> dict[str, Any]:
    required = ["unit_scores.parquet", "training_policy_scores.csv", "split_results.csv", "metadata.json"]
    missing = [name for name in required if not (nested_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{nested_dir} is missing {missing}; run the nested benchmark first.")
    return {
        "units": pd.read_parquet(nested_dir / "unit_scores.parquet"),
        "training": pd.read_csv(nested_dir / "training_policy_scores.csv"),
        "splits": pd.read_csv(nested_dir / "split_results.csv"),
        "metadata": json.loads((nested_dir / "metadata.json").read_text(encoding="utf-8")),
    }


def with_error_regret(units: pd.DataFrame) -> pd.DataFrame:
    """Per-unit error-scale endpoints.

    max-min: (e - best) / (worst - best), 0 when every candidate ties. Median-normalized:
    (e - best) / (median - best), undefined when the median candidate ties the best. Both
    are negative when an output (a prediction ensemble) beats every candidate.
    """
    frame = units.copy()
    error = frame["primary_metric"].astype(float)
    best = frame["candidate_min_error"].astype(float)
    span = frame["candidate_max_error"].astype(float) - best
    median_span = frame["candidate_median_error"].astype(float) - best
    frame["maxmin_error_regret"] = np.where(span > 0, (error - best) / span.where(span > 0, 1.0), 0.0)
    frame["median_error_regret"] = np.where(
        median_span > 0, (error - best) / median_span.where(median_span > 0, 1.0), np.nan
    )
    tolerance = 1e-12 + 1e-12 * best.abs()
    frame["beats_best"] = (error < best - tolerance).astype(float)
    return frame


def dataset_split_scores(units: pd.DataFrame) -> pd.DataFrame:
    heldout = with_error_regret(units[units["phase"].eq("heldout")])
    grouped = heldout.groupby(["artifact", "split_id", "dataset", "task", "method"], as_index=False)
    scores = grouped.agg(
        mean_rank=("candidate_rank", "mean"),
        mean_normalized_rank_regret=("normalized_rank_regret", "mean"),
        mean_rank_regret=("rank_regret", "mean"),
        tied_best_rate=("tied_best", "mean"),
        mean_total_seconds=("total_seconds", "mean"),
        mean_maxmin_error_regret=("maxmin_error_regret", "mean"),
        mean_median_error_regret=("median_error_regret", "mean"),
        beats_best_rate=("beats_best", "mean"),
        n_units=("candidate_rank", "size"),
    )
    original = (
        heldout[heldout["method"].eq(NESTED)]
        .groupby(["artifact", "split_id"], as_index=False)["original_method"]
        .first()
        .rename(columns={"original_method": "nested_original_method"})
    )
    scores = scores.merge(original, on=["artifact", "split_id"], how="left")
    scores["original_method"] = np.where(scores["method"].eq(NESTED), scores["nested_original_method"], "")
    return scores.drop(columns=["nested_original_method"])


def train_selected_choices(
    training: pd.DataFrame,
    *,
    pool_name: str,
    members: list[str],
) -> pd.DataFrame:
    """Per split, the declared pool member with the lowest outer-training regret.

    Ties are broken by runtime and then by stable policy order, as for the nested choice.
    """
    frame = training[training["method"].astype(str).isin(members)].copy()
    found = set(frame["method"].astype(str))
    absent = [member for member in members if member not in found]
    if absent:
        raise ValueError(f"Reference pool `{pool_name}` names methods absent from training scores: {absent}")
    per_split = frame.groupby(["artifact", "split_id"])["method"].nunique()
    if (per_split != len(members)).any():
        raise ValueError(f"Reference pool `{pool_name}` is incomplete in some splits.")
    frame["_order"] = frame["method"].map({method: index for index, method in enumerate(members)})
    chosen = (
        frame.sort_values(
            ["artifact", "split_id", "mean_normalized_rank_regret", "mean_total_seconds", "_order"],
            kind="mergesort",
        )
        .groupby(["artifact", "split_id"], as_index=False)
        .head(1)
    )
    chosen = chosen.rename(
        columns={
            "method": "selected_reference_method",
            "mean_normalized_rank_regret": "train_mean_normalized_rank_regret",
            "mean_total_seconds": "train_mean_total_seconds",
        }
    )
    chosen["pool"] = pool_name
    return chosen[
        [
            "artifact",
            "split_id",
            "pool",
            "selected_reference_method",
            "train_mean_normalized_rank_regret",
            "train_mean_total_seconds",
        ]
    ].reset_index(drop=True)


def append_train_selected(scores: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    frames = [scores]
    for pool_name, pool_choices in choices.groupby("pool", sort=False):
        selected = scores.merge(
            pool_choices[["artifact", "split_id", "selected_reference_method"]],
            left_on=["artifact", "split_id", "method"],
            right_on=["artifact", "split_id", "selected_reference_method"],
            how="inner",
        )
        n_expected = int(scores.loc[scores["method"].eq(NESTED), ["split_id", "dataset"]].drop_duplicates().shape[0])
        if selected.shape[0] != n_expected:
            raise ValueError(
                f"Pool `{pool_name}`: {selected.shape[0]} held-out rows matched, expected {n_expected}."
            )
        selected["original_method"] = selected["selected_reference_method"]
        selected["method"] = TRAIN_SELECTED_PREFIX + str(pool_name)
        frames.append(selected[scores.columns])
    return pd.concat(frames, ignore_index=True)


def dataset_means(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["artifact", "dataset", "task", "method"], as_index=False)
        .agg(
            mean_rank=("mean_rank", "mean"),
            mean_normalized_rank_regret=("mean_normalized_rank_regret", "mean"),
            mean_rank_regret=("mean_rank_regret", "mean"),
            tied_best_rate=("tied_best_rate", "mean"),
            mean_total_seconds=("mean_total_seconds", "mean"),
            mean_maxmin_error_regret=("mean_maxmin_error_regret", "mean"),
            mean_median_error_regret=("mean_median_error_regret", "mean"),
            beats_best_rate=("beats_best_rate", "mean"),
            n_split_appearances=("split_id", "nunique"),
            n_units_mean=("n_units", "mean"),
        )
        .sort_values(["artifact", "dataset", "method"])
        .reset_index(drop=True)
    )


def _pivot(means: pd.DataFrame, metric: str) -> pd.DataFrame:
    return means.pivot_table(index="dataset", columns="method", values=metric, aggfunc="first")


def clustered_summary(
    means: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    single_best: str | None,
) -> pd.DataFrame:
    rows = []
    pivots = {metric: _pivot(means, metric) for metric in METRICS}
    regret = pivots["mean_normalized_rank_regret"]
    for method in regret.columns:
        row: dict[str, Any] = {"method": method, "label": policy_label(method)}
        for metric in METRICS:
            column = pivots[metric][method].dropna()
            if column.shape[0] != resampler.n:
                # Median-normalized regret is undefined in units where the median candidate
                # ties the best; report the defined datasets with their own resampler.
                subset = resampler.subset(column.index, salt=f"{metric}:{method}")
                estimate, low, high = subset.mean_ci(column)
                row[f"{metric}_n_datasets"] = int(column.shape[0])
            else:
                estimate, low, high = resampler.mean_ci(column)
            row[metric] = estimate
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        if single_best and single_best in regret.columns and ORACLE in regret.columns:
            estimate, low, high = gap_closed(regret[method], regret[single_best], regret[ORACLE], resampler)
            row["gap_closed"] = estimate
            row["gap_closed_ci_low"] = low
            row["gap_closed_ci_high"] = high
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["mean_normalized_rank_regret", "mean_total_seconds", "method"])
        .reset_index(drop=True)
    )


def contrast_families(
    means: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    families: dict[str, dict[str, Any]],
    margins: list[float],
    rank_scale: float,
) -> pd.DataFrame:
    regret = _pivot(means, "mean_normalized_rank_regret")
    rows = []
    for family, spec in families.items():
        left = str(spec["left"])
        for right in spec["references"]:
            right = str(right)
            for method in [left, right]:
                if method not in regret.columns:
                    raise ValueError(f"Family `{family}` names `{method}`, which has no scores.")
            result = paired_contrast(
                regret[left],
                regret[right],
                resampler,
                margins=margins,
                rank_scale=rank_scale,
            )
            rows.append(
                {
                    "family": family,
                    "left": left,
                    "right": right,
                    "left_label": policy_label(left),
                    "right_label": policy_label(right),
                    "left_mean": float(regret[left].mean()),
                    "right_mean": float(regret[right].mean()),
                    **result,
                }
            )
    return contrast_table(rows)


def task_stratified_summary(
    means: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    for task, group in means.groupby("task", sort=True):
        subset = resampler.subset(group["dataset"].unique(), salt=f"task:{task}")
        for metric in ["mean_normalized_rank_regret", "mean_total_seconds"]:
            pivot = _pivot(group, metric)
            for method in methods:
                if method not in pivot.columns:
                    continue
                estimate, low, high = subset.mean_ci(pivot[method])
                rows.append(
                    {
                        "task": task,
                        "n_datasets": subset.n,
                        "method": method,
                        "label": policy_label(method),
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    nested = load_nested_output(Path(args.nested_output))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = dataset_split_scores(nested["units"])
    pools = {str(name): [str(member) for member in members] for name, members in config.get("fixed_reference_pools", {}).items()}
    choice_frames = [
        train_selected_choices(nested["training"], pool_name=name, members=members)
        for name, members in pools.items()
    ]
    choices = pd.concat(choice_frames, ignore_index=True) if choice_frames else pd.DataFrame()
    if not choices.empty:
        scores = append_train_selected(scores, choices)
    means = dataset_means(scores)

    bootstrap = dict(config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        means["dataset"].unique(),
        seed=int(bootstrap.get("seed", 20260608)),
        n_resamples=int(bootstrap.get("resamples", 10_000)),
    )
    summary = clustered_summary(means, resampler, single_best=config.get("single_best"))
    contrasts = contrast_families(
        means,
        resampler,
        families=dict(config.get("families", {})),
        margins=[float(value) for value in config.get("margins", [0.005, 0.010, 0.015, 0.020])],
        rank_scale=float(config["rank_scale"]),
    )
    strata = task_stratified_summary(
        means,
        resampler,
        methods=[str(value) for value in config.get("task_strata_methods", [])],
    )
    selection_counts = (
        nested["splits"].groupby("selected_policy").size().rename("n_splits").reset_index()
        .sort_values(["n_splits", "selected_policy"], ascending=[False, True])
    )
    choice_counts = (
        choices.groupby(["pool", "selected_reference_method"]).size().rename("n_splits").reset_index()
        if not choices.empty
        else pd.DataFrame()
    )

    scores.to_parquet(output_dir / "dataset_policy_scores.parquet", index=False)
    means.to_csv(output_dir / "dataset_mean_scores.csv", index=False)
    summary.to_csv(output_dir / "clustered_summary.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    strata.to_csv(output_dir / "task_stratified_summary.csv", index=False)
    selection_counts.to_csv(output_dir / "policy_selection_counts.csv", index=False)
    choices.to_csv(output_dir / "train_selected_reference_choices.csv", index=False)
    choice_counts.to_csv(output_dir / "train_selected_reference_counts.csv", index=False)
    metadata = {
        "config": str(args.config),
        "nested_output": str(args.nested_output),
        "nested_metadata": nested["metadata"],
        "command": sys.argv,
        "n_datasets": resampler.n,
        "bootstrap_seed": resampler.seed,
        "bootstrap_resamples": resampler.n_resamples,
        "wall_seconds": time.time() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
