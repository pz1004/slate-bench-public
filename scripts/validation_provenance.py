"""Validation provenance: is a validation score comparable across candidates?

For every candidate method this script measures, over the artifact's evaluation units,
the mean validation rank, the mean test rank and their difference (rank calibration: a
positive value means the method looks better on validation than it turns out on test,
relative to the other candidates), and the error-space optimism, test minus validation
error normalised by each unit's range of candidate test errors
(``tabular_sota_benchmark.provenance``). It summarises both by provenance class (TabArena
variants default / tuned / tuned + ensemble, crossed with task type; TabRepo base-learner
families), computes the spread-to-estimation-error ratio of the candidates' offsets
(Remark 1) in rank and in error space, and reports, from the nested run's per-unit rows,
which provenance class every policy selects on held-out units. All intervals are
dataset-level bootstrap intervals on the artifact's shared resample matrix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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

from tabular_sota_benchmark.artifacts import (  # noqa: E402
    ARTIFACT_UNIT_COLUMNS,
    _method_family,
    _method_variant,
)
from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402
from tabular_sota_benchmark.labels import plain_label, policy_label  # noqa: E402
from tabular_sota_benchmark.provenance import error_optimism, offset_snr, top_tenth  # noqa: E402

BENCH_PATH = Path(__file__).resolve().with_name("score_level_policy_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_provenance", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def candidate_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    """Within-unit average-tie validation and test ranks (1 = best) of every candidate."""
    frame = candidates.copy()
    grouped = frame.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)
    frame["val_rank"] = grouped["val_primary_metric"].rank(method="average")
    frame["test_rank"] = grouped["primary_metric"].rank(method="average")
    frame["n_candidates"] = grouped["method"].transform("size").astype(float)
    frame["optimism"] = frame["test_rank"] - frame["val_rank"]
    # Rank calibration on the normalised-regret scale, for the spread-to-error ratio.
    frame["optimism_norm"] = frame["optimism"] / (frame["n_candidates"] - 1.0)
    frame["family"] = frame["method"].map(_method_family)
    frame["variant"] = frame["method"].map(_method_variant)
    return error_optimism(frame, list(ARTIFACT_UNIT_COLUMNS))


def _clustered(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
    resampler: ArtifactResampler,
) -> pd.DataFrame:
    """Dataset-clustered means: average units within dataset, then datasets with a CI."""
    per_dataset = frame.groupby([*group_columns, "dataset"], as_index=False)[value_columns].mean()
    rows = []
    for key, group in per_dataset.groupby(group_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        datasets = group["dataset"].astype(str)
        rs = resampler if set(datasets) == set(resampler.datasets) else resampler.subset(
            datasets, salt="|".join(str(value) for value in key)
        )
        row: dict[str, Any] = dict(zip(group_columns, key))
        row["n_datasets"] = int(group.shape[0])
        for column in value_columns:
            values = group.set_index("dataset")[column].dropna()
            # A dataset with no defined value (an error scale of zero in every unit) is
            # left out of that column only.
            column_rs = rs if len(values) == len(group) else resampler.subset(
                values.index, salt="|".join([*map(str, key), column])
            )
            estimate, low, high = column_rs.mean_ci(values)
            row[column] = estimate
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_tables(ranks: pd.DataFrame, resampler: ArtifactResampler, *, has_variants: bool) -> dict[str, pd.DataFrame]:
    values = ["val_rank", "test_rank", "optimism", "err_optimism", "err_optimism_median"]
    tables = {
        "calibration_by_method": _clustered(
            ranks, group_columns=["method", "family", "variant"], value_columns=values, resampler=resampler
        ),
        "calibration_by_family": _clustered(
            ranks, group_columns=["family"], value_columns=values, resampler=resampler
        ),
    }
    if has_variants:
        tables["calibration_by_variant"] = _clustered(
            ranks, group_columns=["variant"], value_columns=values, resampler=resampler
        )
        tables["calibration_by_variant_task"] = _clustered(
            ranks, group_columns=["variant", "task"], value_columns=values, resampler=resampler
        )
    return tables


def selection_shares(
    units: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    methods: list[str],
    has_variants: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of held-out units in which each policy picks each provenance class.

    Shares are averaged within dataset (over units and split appearances), then across
    datasets. Ensemble outputs are their own class. Also returns each policy's most
    frequently selected candidates.
    """
    heldout = units[units["phase"].eq("heldout") & units["method"].isin(methods)].copy()
    is_ensemble = heldout["ensemble_members"].astype(str).str.len() > 0
    member = heldout["selected_member"].astype(str)
    heldout["class"] = np.where(
        is_ensemble,
        "prediction ensemble",
        member.map(_method_variant) if has_variants else member.map(_method_family),
    )
    classes = sorted(heldout["class"].unique())
    for name in classes:
        heldout[f"share::{name}"] = (heldout["class"] == name).astype(float)
    share_columns = [f"share::{name}" for name in classes]
    shares = _clustered(heldout, group_columns=["method"], value_columns=share_columns, resampler=resampler)
    shares.insert(1, "label", shares["method"].map(policy_label))
    per_dataset_member = (
        heldout[~is_ensemble]
        .assign(_one=1.0)
        .groupby(["method", "dataset", "selected_member"], as_index=False)["_one"]
        .sum()
    )
    per_dataset_member["share"] = per_dataset_member["_one"] / per_dataset_member.groupby(
        ["method", "dataset"]
    )["_one"].transform("sum")
    n_datasets = heldout.groupby("method")["dataset"].nunique()
    top = (
        per_dataset_member.groupby(["method", "selected_member"], as_index=False)["share"].sum()
        .assign(share=lambda frame: frame["share"] / frame["method"].map(n_datasets))
        .sort_values(["method", "share"], ascending=[True, False])
        .groupby("method")
        .head(5)
        .reset_index(drop=True)
    )
    return shares, top


def dataset_variant_optimism(ranks: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset means of rank calibration and error optimism by provenance class."""
    return ranks.groupby(["dataset", "variant"], as_index=False)[
        ["optimism", "err_optimism", "err_optimism_median"]
    ].mean()


def offset_matrices(ranks: pd.DataFrame, top: list[str]) -> pd.DataFrame:
    """Dataset-by-candidate means, for the candidates in ``top``, of the offset measures."""
    frame = ranks[ranks["method"].isin(top)]
    return frame.groupby(["dataset", "method"], as_index=False)[
        ["optimism_norm", "err_optimism", "err_optimism_median"]
    ].mean()


def snr_table(matrices: pd.DataFrame, *, n_train: int, resampler: ArtifactResampler) -> pd.DataFrame:
    """Spread-to-estimation-error ratio of the top candidates' offsets in each space."""
    rows = []
    for space, column in (("rank", "optimism_norm"), ("error", "err_optimism"), ("error_median", "err_optimism_median")):
        matrix = matrices.pivot(index="dataset", columns="method", values=column)
        rows.append({"space": space, **offset_snr(matrix, n_train=n_train, resampler=resampler)})
    return pd.DataFrame(rows)


def median_distance(ranks: pd.DataFrame) -> pd.DataFrame:
    """How close each unit's median candidate is to its best one, relative to the range.

    The median normalisation of error-space optimism divides by the median candidate's
    distance from the best (``tabular_sota_benchmark.provenance.error_optimism``), which
    is close to zero, and the values unstable, when most candidates lie near the best.
    """
    grouped = ranks.groupby(ARTIFACT_UNIT_COLUMNS, sort=False)["primary_metric"]
    low = grouped.min()
    span = grouped.max() - low
    median_span = grouped.median() - low
    share = (median_span / span.where(span > 0)).dropna()
    return pd.DataFrame(
        [
            {
                "n_units": int(len(low)),
                "quantile_10": float(share.quantile(0.10)),
                "quantile_25": float(share.quantile(0.25)),
                "quantile_50": float(share.quantile(0.50)),
                "share_units_below_one_percent": float(share.lt(0.01).mean()),
                "units_median_at_best": int(median_span.le(0.0).sum()),
            }
        ]
    )


def budget_exclusion(ranks: pd.DataFrame, *, budget_seconds: float) -> pd.DataFrame:
    frame = ranks.assign(over_budget=(ranks["fit_seconds"] > float(budget_seconds)).astype(float))
    per_dataset = frame.groupby(["method", "variant", "dataset"], as_index=False)["over_budget"].mean()
    return (
        per_dataset.groupby(["method", "variant"], as_index=False)["over_budget"].mean()
        .rename(columns={"over_budget": "share_units_over_budget"})
        .assign(unit_share=frame.groupby(["method", "variant"])["over_budget"].mean().to_numpy())
        .sort_values("share_units_over_budget", ascending=False)
        .reset_index(drop=True)
    )


def calibration_figure(table: pd.DataFrame, output: Path, *, has_variants: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # Shape and lightness both separate the variants, so the figure also reads in greyscale:
    # open circles for defaults, mid-tone triangles for tuned, dark squares for tuned + ensemble.
    styles = {
        "default": {"marker": "o", "facecolor": "white", "edgecolor": "#08519c", "s": 34},
        "tuned": {"marker": "^", "facecolor": "#fd8d3c", "edgecolor": "black", "s": 42},
        "tuned + ensemble": {"marker": "s", "facecolor": "#7f0000", "edgecolor": "black", "s": 32},
    }
    groups = table.groupby("variant" if has_variants else "family", sort=True)
    for name, group in groups:
        style = styles.get(str(name), {"marker": "o", "facecolor": None, "edgecolor": "black", "s": 34})
        ax.scatter(
            group["val_rank"],
            group["test_rank"],
            s=style["s"],
            label=str(name),
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.7,
        )
    limits = [0.5, float(max(table["val_rank"].max(), table["test_rank"].max()) + 1)]
    ax.plot(limits, limits, color="grey", linewidth=0.8, linestyle="--", label="val rank = test rank")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Mean validation rank (1 = best)")
    ax.set_ylabel("Mean test rank (1 = best)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    bench.set_rank_rule(str(dict(config.get("benchmark", {})).get("rank_rule", "consistent")))
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
    artifact = bench._load_artifact(artifact_config, seed=int(dict(config.get("benchmark", {})).get("seed", 20260604)))
    ranks = candidate_ranks(artifact["candidates"])
    has_variants = bool((ranks["variant"] != "n/a").any())
    resampler = ArtifactResampler(ranks["dataset"].unique(), seed=int(args.bootstrap_seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = calibration_tables(ranks, resampler, has_variants=has_variants)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    if has_variants:
        dataset_variant_optimism(ranks).to_csv(output_dir / "optimism_by_dataset_variant.csv", index=False)
    # Remark 1: the spread of the top candidates' offsets against the error of an offset
    # estimated from one outer split's training datasets.
    n_train = pd.read_csv(Path(args.nested_output) / "split_results.csv")["n_train_datasets"].unique()
    if len(n_train) != 1:
        raise ValueError("The nested run's splits do not all have the same number of training datasets.")
    top_candidates = top_tenth(tables["calibration_by_method"])["method"].tolist()
    matrices = offset_matrices(ranks, top_candidates)
    matrices.to_csv(output_dir / "offsets_top_candidates.csv", index=False)
    snr_table(matrices, n_train=int(n_train[0]), resampler=resampler).to_csv(output_dir / "offset_snr.csv", index=False)
    median_distance(ranks).to_csv(output_dir / "median_distance.csv", index=False)
    budget_exclusion(ranks, budget_seconds=float(args.budget_seconds)).to_csv(
        output_dir / "budget_exclusion.csv", index=False
    )
    units = pd.read_parquet(Path(args.nested_output) / "unit_scores.parquet")
    methods = sorted(units.loc[units["phase"].eq("heldout"), "method"].unique())
    shares, top = selection_shares(units, resampler, methods=methods, has_variants=has_variants)
    shares.to_csv(output_dir / "selection_shares.csv", index=False)
    top.to_csv(output_dir / "top_selected_members.csv", index=False)
    if args.figure:
        calibration_figure(tables["calibration_by_method"], Path(args.figure), has_variants=has_variants)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": str(args.benchmark_config),
                "nested_output": str(args.nested_output),
                "bootstrap_seed": int(args.bootstrap_seed),
                "n_datasets": resampler.n,
                "n_units": int(ranks[ARTIFACT_UNIT_COLUMNS].drop_duplicates().shape[0]),
                "n_candidates": int(ranks["method"].nunique()),
                "n_train_datasets": int(n_train[0]),
                "top_candidates": len(top_candidates),
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
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    parser.add_argument("--budget-seconds", type=float, default=10000.0)
    parser.add_argument("--figure", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
