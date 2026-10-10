from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabular_sota_benchmark.markdown import dataframe_to_markdown


def write_results_section(
    analysis_dir: str | Path,
    paper_dir: str | Path = "paper",
    reference_method: str = "meta_selector_ensemble",
) -> Path:
    analysis_path = Path(analysis_dir)
    paper_path = Path(paper_dir)
    paper_path.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_path / "summary.csv"
    paired_path = analysis_path / "paired_tests.csv"
    output_path = paper_path / "results_auto.md"

    if not summary_path.exists():
        output_path.write_text(
            "# Results\n\nNo analysis summary exists yet. Run the benchmark loop first.\n",
            encoding="utf-8",
        )
        return output_path

    summary = pd.read_csv(summary_path, low_memory=False)
    paired = pd.read_csv(paired_path, low_memory=False) if paired_path.exists() else pd.DataFrame()
    best = summary.sort_values(["mean_rank", "mean_primary_metric"]).iloc[0]
    reference = summary[summary["method"] == reference_method]
    result_source = str(summary_path)
    ranks_path = analysis_path / "ranks.csv"
    evidence_boundary = (
        "These results use TabArena score artifacts and support benchmark-level ranking analysis. "
        "They do not represent local retraining, and they cannot validate a true prediction-level ensemble because the public artifact does not include predictions."
    )
    if ranks_path.exists():
        ranks = pd.read_csv(ranks_path, nrows=200, low_memory=False)
        if "source" not in ranks.columns or "tabarena_benchmark_results" not in set(
            ranks["source"].dropna().astype(str)
        ):
            evidence_boundary = (
                "These results verify that the workflow runs and produces comparable split-level metrics. "
                "They do not establish full tabular SOTA. A journal claim requires the same analysis on TabArena-compatible or similarly broad benchmark artifacts."
            )

    lines = [
        "# Results",
        "",
        f"This section is generated from `{result_source}` and is intentionally cautious.",
        "",
        "## Benchmark Summary",
        "",
        (
            f"The best average-rank method in the current run is `{best['method']}` "
            f"with mean rank {best['mean_rank']:.4f}."
        ),
    ]
    if not reference.empty:
        row = reference.iloc[0]
        lines.append(
            f"The configured reference selector `{reference_method}` has mean rank "
            f"{row['mean_rank']:.4f} and win rate {row['win_rate']:.4f}."
        )
    else:
        lines.append(f"The configured reference selector `{reference_method}` was not present.")

    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            evidence_boundary,
            "",
            "## Paired Comparisons",
            "",
            "The paired comparisons use within-unit rank differences rather than raw metric differences.",
            "",
        ]
    )
    if paired.empty:
        lines.append("No paired comparisons are available.")
    else:
        lines.append(dataframe_to_markdown(paired))
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
