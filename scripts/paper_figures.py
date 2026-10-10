"""Manuscript figures from the v2 review outputs, with registry labels.

* ``runtime_pareto.pdf``: dataset-level mean normalized rank regret against the recorded
  runtime of the selected candidate (log scale), points labelled directly;
* ``dataset_paired_forest.pdf``: nested selection minus each reference, with 95%
  intervals from the shared resample matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.labels import plain_label  # noqa: E402

PARETO_ROWS = [
    "meta_selector_family_prior_budgeted",
    "train_selected_fixed_reference__declared_fixed",
    "meta_selector_meta_feature_guard_005",
    "nested_policy_selected",
    "meta_selector_task_adaptive_budgeted",
    "meta_selector_calibrated_budget_10000s",
    "meta_selector_single",
    "reference_single_best_global",
    "reference_validation_best_default",
    "reference_optimism_corrected",
    "reference_optimism_corrected_budget",
]
# Label offsets (points) for the crowded low-runtime cluster, chosen so no two labels
# overlap at the manuscript's figure width.
LABEL_OFFSETS = {
    "meta_selector_task_adaptive_budgeted": (-80, 2),
    "meta_selector_calibrated_budget_10000s": (7, 5),
    "nested_policy_selected": (7, 3),
    "meta_selector_meta_feature_guard_005": (-78, -3),
    "train_selected_fixed_reference__declared_fixed": (7, -5),
    "meta_selector_family_prior_budgeted": (7, -11),
    "reference_optimism_corrected_budget": (-150, -12),
}
FOREST_FAMILIES = [
    ("primary", "Declared references"),
    ("revision_references", "References added later"),
]
FOREST_SKIP = {"reference_single_best_task", "reference_validation_best_tuned", "reference_optimism_corrected_task"}


def pareto(summary: pd.DataFrame, output: Path) -> None:
    rows = summary.set_index("method").loc[[m for m in PARETO_ROWS if m in set(summary["method"])]]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    frontier = []
    best = float("inf")
    for method, item in rows.sort_values("mean_total_seconds").iterrows():
        if item["mean_normalized_rank_regret"] < best:
            frontier.append(method)
            best = item["mean_normalized_rank_regret"]
    # The runtime Pareto frontier is drawn as filled black markers joined by a dashed step
    # line, so it reads without colour; the other rows are open markers.
    front = rows.loc[frontier].sort_values("mean_total_seconds")
    ax.plot(
        front["mean_total_seconds"],
        front["mean_normalized_rank_regret"],
        linestyle="--",
        color="black",
        linewidth=0.8,
        drawstyle="steps-post",
        zorder=2,
    )
    for method, item in rows.iterrows():
        on_frontier = method in frontier
        added = method.startswith("reference_")
        ax.scatter(
            item["mean_total_seconds"],
            item["mean_normalized_rank_regret"],
            s=46 if method == "nested_policy_selected" else 32,
            marker="D" if method == "nested_policy_selected" else ("s" if added else "o"),
            facecolor="black" if on_frontier else "white",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            plain_label(method),
            (item["mean_total_seconds"], item["mean_normalized_rank_regret"]),
            xytext=LABEL_OFFSETS.get(method, (5, 3)),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Recorded fit + predict time of the selected candidate (s, log scale)")
    ax.set_ylabel("Mean normalized rank regret")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def forest(contrasts: pd.DataFrame, output: Path) -> None:
    rows = []
    for family, title in FOREST_FAMILIES:
        subset = contrasts[contrasts["family"].eq(family) & ~contrasts["right"].isin(FOREST_SKIP)]
        for _, item in subset.iterrows():
            rows.append((title, plain_label(item["right"]), item["mean_difference"], item["ci_low"], item["ci_high"]))
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(rows) + 1.0))
    labels = []
    for position, (title, label, mean, low, high) in enumerate(reversed(rows)):
        colour = "#1f5aa6" if title == FOREST_FAMILIES[0][1] else "#b35806"
        ax.plot([low, high], [position, position], color=colour, linewidth=1.4)
        ax.scatter([mean], [position], marker="D", color=colour, zorder=3, s=22)
        labels.append(label)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvspan(-0.5 / 43, 0.5 / 43, color="#dddddd", alpha=0.6, zorder=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Nested selection minus reference (normalized rank regret)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    review = Path(args.review)
    pareto(pd.read_csv(review / "clustered_summary.csv"), Path(args.figures) / "runtime_pareto.pdf")
    forest(pd.read_csv(review / "contrasts.csv"), Path(args.figures) / "dataset_paired_forest.pdf")
    print(f"Wrote figures to {args.figures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", default="analysis-output/v2/score_level_tabarena_review")
    parser.add_argument("--figures", default="paper/figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
