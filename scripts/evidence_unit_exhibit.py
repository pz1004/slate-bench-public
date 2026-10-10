"""The same contrasts analysed at three evidence units.

For each headline contrast of a nested run this script reports the paired difference and
a Wilcoxon signed-rank p-value when the evidence unit is

* the outer split (each of the 100 overlapping splits treated as independent, with the
  split's held-out units averaged, as a leaderboard over splits would);
* the unit appearance (every held-out unit in every split, which also weights each
  dataset by its number of folds and seeds);
* the dataset (repeated appearances averaged within dataset: the declared unit).

Only the dataset-level row matches the estimand. The other two show how the same numbers
change their verdict when overlapping or repeated evidence is counted as independent. They
also change the estimand: both weight a dataset by its held-out appearances. To separate
the two effects, the unit-appearance estimate is also tested with whole datasets
resampled (a cluster bootstrap on the shared resample matrix), and the dataset-level row
with the mean test of the declared rule: ``p_datasets_resampled``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p, paired_contrast  # noqa: E402
from tabular_sota_benchmark.labels import policy_label  # noqa: E402

UNIT_KEY = ["split_id", "dataset", "fold", "seed", "task", "primary_metric_name"]
TRAIN_SELECTED_PREFIX = "train_selected_fixed_reference__"


def heldout_unit_rows(units: pd.DataFrame, choices: pd.DataFrame, method: str) -> pd.DataFrame:
    """Held-out unit rows of ``method``; train-selected references follow their choices."""
    heldout = units[units["phase"].eq("heldout")]
    if method.startswith(TRAIN_SELECTED_PREFIX):
        pool = method[len(TRAIN_SELECTED_PREFIX):]
        chosen = choices[choices["pool"].eq(pool)][["split_id", "selected_reference_method"]]
        rows = heldout.merge(
            chosen, left_on=["split_id", "method"], right_on=["split_id", "selected_reference_method"]
        )
    else:
        rows = heldout[heldout["method"].eq(method)]
    return rows[[*UNIT_KEY, "normalized_rank_regret"]].reset_index(drop=True)


def _wilcoxon(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-12]
    return float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue) if nonzero.size else 1.0


def unit_weighted_cluster_p(paired: pd.DataFrame, resampler: ArtifactResampler) -> float:
    """Mean test of the unit-weighted difference with whole datasets resampled."""
    by_dataset = paired.groupby("dataset")["difference"].agg(["sum", "size"])
    sums = resampler.align(by_dataset["sum"])
    counts = resampler.align(by_dataset["size"].astype(float))
    boot = sums[resampler.indices].sum(axis=1) / counts[resampler.indices].sum(axis=1)
    return bootstrap_mean_p(boot)


def exhibit(
    units: pd.DataFrame,
    choices: pd.DataFrame,
    means: pd.DataFrame,
    *,
    left: str,
    right: str,
    resampler: ArtifactResampler,
) -> list[dict[str, object]]:
    left_units = heldout_unit_rows(units, choices, left)
    right_units = heldout_unit_rows(units, choices, right)
    paired = left_units.merge(right_units, on=UNIT_KEY, suffixes=("_left", "_right"))
    if paired.shape[0] != left_units.shape[0]:
        raise ValueError(f"Unit rows of {left} and {right} do not align.")
    paired["difference"] = paired["normalized_rank_regret_left"] - paired["normalized_rank_regret_right"]
    per_split = paired.groupby("split_id")["difference"].mean().to_numpy()
    per_unit = paired["difference"].to_numpy()
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    dataset_level = paired_contrast(regret[left], regret[right], resampler)
    common = {"left": left, "right": right, "left_label": policy_label(left), "right_label": policy_label(right)}
    return [
        {
            **common,
            "evidence_unit": "outer split (overlapping, treated as independent)",
            "n": int(per_split.size),
            "mean_difference": float(per_split.mean()),
            "wilcoxon_p": _wilcoxon(per_split),
            "wins": int(np.sum(per_split < -1e-12)),
            "losses": int(np.sum(per_split > 1e-12)),
        },
        {
            **common,
            "evidence_unit": "unit appearance (all held-out units in all splits)",
            "n": int(per_unit.size),
            "mean_difference": float(per_unit.mean()),
            "wilcoxon_p": _wilcoxon(per_unit),
            "wins": int(np.sum(per_unit < -1e-12)),
            "losses": int(np.sum(per_unit > 1e-12)),
            "p_datasets_resampled": unit_weighted_cluster_p(paired, resampler),
        },
        {
            **common,
            "evidence_unit": "dataset (declared)",
            "n": int(dataset_level["n_datasets"]),
            "mean_difference": float(dataset_level["mean_difference"]),
            "wilcoxon_p": float(dataset_level["wilcoxon_p"]),
            "wins": int(dataset_level["wins"]),
            "losses": int(dataset_level["losses"]),
            "ci_low": float(dataset_level["ci_low"]),
            "ci_high": float(dataset_level["ci_high"]),
            "p_datasets_resampled": float(dataset_level["bootstrap_p"]),
        },
    ]


def run(args: argparse.Namespace) -> None:
    units = pd.read_parquet(Path(args.nested_output) / "unit_scores.parquet")
    review_dir = Path(args.review_output)
    choices = pd.read_csv(review_dir / "train_selected_reference_choices.csv")
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    resampler = ArtifactResampler(means["dataset"].unique(), seed=int(args.bootstrap_seed))
    rows = []
    for pair in args.contrasts:
        left, right = pair.split(":")
        rows.extend(exhibit(units, choices, means, left=left, right=right, resampler=resampler))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "evidence_unit_exhibit.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({"command": sys.argv}, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=[
            "nested_policy_selected:meta_selector_single",
            "nested_policy_selected:meta_selector_family_prior_budgeted",
            "nested_policy_selected:train_selected_fixed_reference__declared_fixed",
        ],
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
