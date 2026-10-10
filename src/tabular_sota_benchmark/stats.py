from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tabular_sota_benchmark.markdown import dataframe_to_markdown


def comparison_unit_columns(results: pd.DataFrame) -> list[str]:
    columns = ["dataset", "task", "primary_metric_name"]
    for optional in ["fold", "seed"]:
        if optional in results.columns:
            columns.append(optional)
    return columns


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.6g}"


def _rank_results(results: pd.DataFrame) -> pd.DataFrame:
    ranked = results.copy()
    ranked["rank"] = ranked.groupby(comparison_unit_columns(ranked))["primary_metric"].rank(
        method="average",
        ascending=True,
    )
    return ranked


def _paired_tests(ranked: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    unit_columns = comparison_unit_columns(ranked)
    reference = ranked[ranked["method"] == reference_method][
        [*unit_columns, "rank"]
    ].rename(columns={"rank": "reference_rank"})
    rows: list[dict[str, float | int | str]] = []
    for method, group in ranked.groupby("method"):
        if method == reference_method:
            continue
        paired = group.merge(reference, on=unit_columns, how="inner")
        if paired.empty:
            continue
        diff = paired["rank"] - paired["reference_rank"]
        p_value = float("nan")
        statistic = float("nan")
        if len(diff) >= 5 and not np.allclose(diff, 0.0):
            try:
                statistic, p_value = wilcoxon(diff)
            except ValueError:
                statistic, p_value = float("nan"), float("nan")
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "n_pairs": int(len(diff)),
                "comparison_value": "rank",
                "mean_method_minus_reference": float(diff.mean()),
                "median_method_minus_reference": float(diff.median()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        )
    paired = pd.DataFrame(rows)
    if not paired.empty:
        paired["holm_p_value"] = _holm_adjust(paired["wilcoxon_p_value"].to_numpy(dtype=float))
    return paired


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = np.where(np.isfinite(p_values))[0]
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    previous = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return adjusted


def summarize_results(
    results_csv: str | Path,
    analysis_dir: str | Path,
    reference_method: str = "meta_selector_ensemble",
    extra_reference_methods: list[str] | None = None,
) -> dict[str, Path]:
    results_path = Path(results_csv)
    out_dir = Path(analysis_dir)
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(results_path, low_memory=False)
    ranked = _rank_results(results)
    ranked_path = out_dir / "ranks.csv"
    ranked.to_csv(ranked_path, index=False)

    summary = (
        ranked.groupby(["method", "method_type"], as_index=False)
        .agg(
            n=("primary_metric", "size"),
            mean_primary_metric=("primary_metric", "mean"),
            std_primary_metric=("primary_metric", "std"),
            mean_rank=("rank", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_predict_seconds=("predict_seconds", "mean"),
        )
        .sort_values(["mean_rank", "mean_primary_metric"], ascending=True)
    )
    unit_columns = comparison_unit_columns(ranked)
    winners = ranked.loc[ranked.groupby(unit_columns)["primary_metric"].idxmin()]
    win_rate = (
        winners["method"]
        .value_counts(normalize=True)
        .rename_axis("method")
        .reset_index(name="win_rate")
    )
    summary = summary.merge(win_rate, on="method", how="left")
    summary["win_rate"] = summary["win_rate"].fillna(0.0)
    unit_best = ranked.groupby(unit_columns, as_index=False)["primary_metric"].min().rename(
        columns={"primary_metric": "unit_best_primary_metric"}
    )
    tied_best = ranked.merge(unit_best, on=unit_columns, how="left")
    tied_best["is_tied_best"] = np.isclose(
        tied_best["primary_metric"].to_numpy(dtype=float),
        tied_best["unit_best_primary_metric"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    tie_win_rate = (
        tied_best.groupby("method", as_index=False)["is_tied_best"]
        .mean()
        .rename(columns={"is_tied_best": "tie_win_rate"})
    )
    summary = summary.merge(tie_win_rate, on="method", how="left")
    summary["tie_win_rate"] = summary["tie_win_rate"].fillna(0.0)
    summary["mean_total_seconds"] = (
        summary["mean_fit_seconds"].astype(float)
        + summary["mean_predict_seconds"].astype(float)
    )
    summary["pareto_efficient"] = _pareto_efficient(summary)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    task_summary = (
        ranked.groupby(["task", "method", "method_type"], as_index=False)
        .agg(n=("rank", "size"), mean_rank=("rank", "mean"))
        .sort_values(["task", "mean_rank"], ascending=True)
    )
    task_summary_path = out_dir / "task_summary.csv"
    task_summary.to_csv(task_summary_path, index=False)

    selector_choices = _selector_choices(results)
    selector_choices_path = out_dir / "selector_choices.csv"
    selector_choices.to_csv(selector_choices_path, index=False)

    reference_methods = [reference_method]
    for extra_reference in extra_reference_methods or []:
        if extra_reference not in reference_methods:
            reference_methods.append(extra_reference)
    paired_frames = [
        _paired_tests(ranked, reference_method=method)
        for method in reference_methods
        if method in set(ranked["method"].astype(str))
    ]
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(
            columns=[
                "method",
                "reference_method",
                "n_pairs",
                "comparison_value",
                "mean_method_minus_reference",
                "median_method_minus_reference",
                "wilcoxon_statistic",
                "wilcoxon_p_value",
                "holm_p_value",
            ]
        )
    )
    paired_path = out_dir / "paired_tests.csv"
    paired.to_csv(paired_path, index=False)

    figure_path = figure_dir / "figure-01-average-rank.png"
    plt.figure(figsize=(8, max(4, 0.35 * len(summary))))
    plot_data = summary.sort_values("mean_rank", ascending=False)
    plt.barh(plot_data["method"], plot_data["mean_rank"], color="#4772b3")
    plt.xlabel("Average rank (lower is better)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    task_figure_path = figure_dir / "figure-02-task-stratified-rank.png"
    _plot_task_stratified_rank(task_summary, summary, task_figure_path, reference_method)

    selector_figure_path = figure_dir / "figure-03-selector-choices.png"
    _plot_selector_choices(selector_choices, selector_figure_path, reference_method)

    runtime_pareto_path = out_dir / "runtime_pareto.csv"
    runtime_pareto = summary.sort_values(["mean_total_seconds", "mean_rank"], ascending=True)
    runtime_pareto.to_csv(runtime_pareto_path, index=False)

    runtime_figure_path = figure_dir / "figure-04-runtime-rank-pareto.png"
    _plot_runtime_rank_pareto(runtime_pareto, runtime_figure_path, reference_method)

    source_values = set(str(value) for value in results.get("source", pd.Series(dtype=str)).dropna().unique())
    evidence_boundary = (
        "These findings use TabArena score artifacts. They support score-level benchmark analysis, "
        "but not local retraining claims or prediction-level ensembling claims."
        if "tabarena_benchmark_results" in source_values
        else "These findings are smoke evidence only unless the run used benchmark-scale artifacts."
    )

    report_path = out_dir / "analysis-report.md"
    _write_analysis_report(
        report_path,
        results_path,
        summary,
        paired,
        reference_method,
        evidence_boundary,
    )
    stats_path = out_dir / "stats-appendix.md"
    _write_stats_appendix(stats_path, ranked, summary, paired, evidence_boundary)
    catalog_path = out_dir / "figure-catalog.md"
    _write_figure_catalog(
        catalog_path,
        figure_path,
        task_figure_path,
        selector_figure_path,
        runtime_figure_path,
        runtime_pareto_path,
        results_path,
        evidence_boundary,
    )

    return {
        "summary": summary_path,
        "task_summary": task_summary_path,
        "selector_choices": selector_choices_path,
        "ranks": ranked_path,
        "paired_tests": paired_path,
        "analysis_report": report_path,
        "stats_appendix": stats_path,
        "figure_catalog": catalog_path,
        "average_rank_figure": figure_path,
        "task_stratified_rank_figure": task_figure_path,
        "selector_choices_figure": selector_figure_path,
        "runtime_pareto": runtime_pareto_path,
        "runtime_rank_pareto_figure": runtime_figure_path,
    }


def _pareto_efficient(summary: pd.DataFrame) -> pd.Series:
    ordered = summary.sort_values(["mean_total_seconds", "mean_rank"], ascending=True)
    best_rank = float("inf")
    efficient: dict[int, bool] = {}
    for index, row in ordered.iterrows():
        rank = float(row["mean_rank"])
        is_efficient = rank < best_rank - 1e-12
        efficient[int(index)] = is_efficient
        if is_efficient:
            best_rank = rank
    return pd.Series(efficient).reindex(summary.index).fillna(False).astype(bool)


def _selector_choices(results: pd.DataFrame) -> pd.DataFrame:
    if "selected_member" not in results.columns:
        return pd.DataFrame(columns=["method", "selected_member", "count", "fraction"])
    selectors = results[
        (results["method_type"] == "selector")
        & results["selected_member"].notna()
        & (results["selected_member"].astype(str) != "")
    ].copy()
    if selectors.empty:
        return pd.DataFrame(columns=["method", "selected_member", "count", "fraction"])
    counts = (
        selectors.groupby(["method", "selected_member"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = counts.groupby("method")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    return counts.sort_values(["method", "count"], ascending=[True, False])


def _plot_task_stratified_rank(
    task_summary: pd.DataFrame,
    summary: pd.DataFrame,
    figure_path: Path,
    reference_method: str,
    top_n: int = 12,
) -> None:
    if task_summary.empty:
        return
    top_methods = summary.head(top_n)["method"].tolist()
    if reference_method not in top_methods and reference_method in set(summary["method"]):
        top_methods = [*top_methods[:-1], reference_method]
    plot_data = task_summary[task_summary["method"].isin(top_methods)]
    pivot = plot_data.pivot(index="method", columns="task", values="mean_rank")
    pivot = pivot.loc[[method for method in top_methods if method in pivot.index]]
    plt.figure(figsize=(8, max(4, 0.32 * len(pivot))))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(image, label="Mean rank (lower is better)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def _plot_selector_choices(
    selector_choices: pd.DataFrame,
    figure_path: Path,
    reference_method: str,
    top_n: int = 12,
) -> None:
    if selector_choices.empty:
        return
    data = selector_choices[selector_choices["method"] == reference_method].head(top_n)
    if data.empty:
        data = selector_choices.head(top_n)
    plt.figure(figsize=(8, max(4, 0.35 * len(data))))
    plot_data = data.sort_values("count", ascending=True)
    plt.barh(plot_data["selected_member"], plot_data["count"], color="#51815d")
    plt.xlabel("Selection count")
    plt.ylabel("Selected member")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def _plot_runtime_rank_pareto(
    runtime_pareto: pd.DataFrame,
    figure_path: Path,
    reference_method: str,
) -> None:
    if runtime_pareto.empty:
        return
    data = runtime_pareto.copy()
    data["plot_seconds"] = np.maximum(data["mean_total_seconds"].astype(float), 1e-6)
    selectors = data["method_type"] == "selector"
    pareto = data["pareto_efficient"].astype(bool)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        data.loc[~selectors, "plot_seconds"],
        data.loc[~selectors, "mean_rank"],
        s=26,
        color="#9aa3ad",
        alpha=0.65,
        label="Candidate",
    )
    plt.scatter(
        data.loc[selectors, "plot_seconds"],
        data.loc[selectors, "mean_rank"],
        s=54,
        color="#2f7d6d",
        alpha=0.9,
        label="Selector",
    )
    plt.scatter(
        data.loc[pareto, "plot_seconds"],
        data.loc[pareto, "mean_rank"],
        s=95,
        facecolors="none",
        edgecolors="#c05a2b",
        linewidths=1.4,
        label="Pareto-efficient",
    )

    labels = {
        reference_method,
        "REALMLP (tuned + ensemble)",
        "TABPFNV2 (tuned + ensemble)",
        "GBM (tuned + ensemble)",
    }
    labels.update(
        method
        for method in data["method"].astype(str)
        if "budget" in method.lower() or method == reference_method
    )
    for _, row in data[data["method"].isin(labels)].iterrows():
        plt.annotate(
            str(row["method"]),
            (float(row["plot_seconds"]), float(row["mean_rank"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    plt.xscale("log")
    plt.xlabel("Mean train + inference seconds, log scale")
    plt.ylabel("Average rank (lower is better)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def _write_analysis_report(
    path: Path,
    results_path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    reference_method: str,
    evidence_boundary: str,
) -> None:
    best = summary.iloc[0]
    reference = summary[summary["method"] == reference_method]
    is_artifact_run = "TabArena score artifacts" in evidence_boundary
    uncertainty = (
        "The public TabArena artifact is score-level only; it does not include predictions for real ensembling or local retraining evidence."
        if is_artifact_run
        else "Smoke datasets are too small and too few for journal-grade SOTA claims."
    )
    next_check = (
        "Add prediction-level artifacts or locally reproduce selected SOTA baselines before making ensemble or retraining claims."
        if is_artifact_run
        else "Run the same analysis on TabArena-compatible artifacts."
    )
    reference_text = "not available"
    if not reference.empty:
        row = reference.iloc[0]
        reference_text = (
            f"mean rank {_format_float(row['mean_rank'])}, "
            f"mean primary metric {_format_float(row['mean_primary_metric'])}, "
            f"win rate {_format_float(row['win_rate'])}"
        )
    lines = [
        "# Analysis Report",
        "",
        f"Data source: `{results_path}`.",
        "",
        "## Key Findings",
        "",
        (
            f"- Best average-rank method in this run: `{best['method']}` "
            f"(mean rank {_format_float(best['mean_rank'])})."
        ),
        f"- Reference method `{reference_method}`: {reference_text}.",
        f"- {evidence_boundary}",
        "",
        "## Method Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Paired Comparisons Against Reference",
        "",
        "The paired tests compare within-unit rank differences, not raw metric differences, because TabArena mixes metric families across tasks. Holm adjustment is applied within each configured reference method.",
        "",
        dataframe_to_markdown(paired) if not paired.empty else "No paired comparisons available.",
        "",
        "## Claim Candidates",
        "",
        "- Claim:",
        "  - Source evidence: generated result CSV and average-rank summary.",
        "  - Allowed wording: The workflow produced finite split-level results and ranks for the configured baselines.",
        "  - Forbidden stronger wording: The workflow establishes full tabular SOTA.",
        f"  - Uncertainty: {uncertainty}",
        f"  - Next check: {next_check}",
        "  - Decision: keep as workflow validation only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_stats_appendix(
    path: Path,
    ranked: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    evidence_boundary: str,
) -> None:
    n_datasets = ranked["dataset"].nunique()
    unit_columns = comparison_unit_columns(ranked)
    n_splits = ranked[unit_columns].drop_duplicates().shape[0]
    is_artifact_run = "TabArena score artifacts" in evidence_boundary
    metric_line = (
        "Primary metric is TabArena `metric_error`; raw values mix ROC-AUC error, log loss, and RMSE, so cross-task comparisons should use ranks."
        if is_artifact_run
        else "Primary metrics: classification uses log loss; regression uses RMSE."
    )
    limit_line = (
        "The public artifact is score-level only; it cannot validate local retraining or prediction-level ensembling claims."
        if is_artifact_run
        else "The default smoke config is not powered for publication claims."
    )
    lines = [
        "# Stats Appendix",
        "",
        f"- Datasets: {n_datasets}",
        f"- Comparison units: {n_splits}",
        f"- {metric_line}",
        "- Rank direction: lower primary metric is better.",
        "- Paired tests use within-unit rank differences so heterogeneous TabArena metrics are not mixed on their raw scales.",
        "",
        "## Average-Rank Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Task-Stratified Summary",
        "",
        "See `task_summary.csv` for per-task mean ranks.",
        "",
        "## Selector Choices",
        "",
        "See `selector_choices.csv` for selected-member counts by selector method.",
        "",
        "## Paired Tests",
        "",
        dataframe_to_markdown(paired) if not paired.empty else "No paired tests available.",
        "",
        "## Limits",
        "",
        f"- {limit_line}",
        "- Wilcoxon tests are only reported when enough paired units exist and differences are not all zero.",
        "- Holm-adjusted p-values are reported for the configured paired comparisons.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure_catalog(
    path: Path,
    figure_path: Path,
    task_figure_path: Path,
    selector_figure_path: Path,
    runtime_figure_path: Path,
    runtime_pareto_path: Path,
    results_path: Path,
    evidence_boundary: str,
) -> None:
    is_artifact_run = "TabArena score artifacts" in evidence_boundary
    caveat = (
        "TabArena public artifacts are score-level; this figure is a benchmark-rank summary, not local retraining evidence."
        if is_artifact_run
        else "Default smoke datasets do not support SOTA claims."
    )
    summary_path = figure_path.parent.parent / "summary.csv"
    lines = [
        "# Figure Catalog",
        "",
        "## Figure 1: Average Rank",
        "",
        f"- Filename: `{figure_path}`",
        f"- Data source: `{results_path}`",
        "- Purpose: compare methods by average rank across configured comparison units.",
        "- Caption requirement: state that lower rank is better and identify the configured datasets.",
        f"- Key observation: read from `{summary_path}`.",
        f"- Caveat: {caveat}",
        "",
        "## Figure 2: Task-Stratified Rank",
        "",
        f"- Filename: `{task_figure_path}`",
        f"- Data source: `{results_path}` and `task_summary.csv`",
        "- Purpose: compare method ranks separately for binary, multiclass, and regression tasks.",
        "- Caption requirement: state that lower rank is better and that ranks are computed within dataset-fold units.",
        f"- Caveat: {caveat}",
        "",
        "## Figure 3: Selector Choices",
        "",
        f"- Filename: `{selector_figure_path}`",
        f"- Data source: `{results_path}` and `selector_choices.csv`",
        "- Purpose: show which candidate methods the configured reference selector chooses most often.",
        "- Caption requirement: identify the selector method and number of comparison units.",
        f"- Caveat: {caveat}",
        "",
        "## Figure 4: Runtime-Rank Pareto",
        "",
        f"- Filename: `{runtime_figure_path}`",
        f"- Data source: `{results_path}` and `{runtime_pareto_path}`",
        "- Purpose: compare average rank against mean train-plus-inference time and identify Pareto-efficient methods.",
        "- Caption requirement: state that the x-axis uses a log scale and that lower rank is better.",
        f"- Caveat: {caveat}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
