"""Corrected resampled t-test (Nadeau and Bengio 2003) for every contrast of a review.

The dataset bootstrap of the review resamples dataset-level differences with every fitted
choice held fixed, so it ignores that the 100 outer splits share training datasets. The
corrected resampled t-test works on the splits instead and inflates the variance of the
mean over the J splits by the factor 1/J + n_test/n_train, a heuristic for the overlap of
the training sets of random subsampling; it is not validated for this nested design. The
statistic of a split is its held-out mean difference with each held-out dataset weighted by
J/(N n_d), where n_d is the number of splits that hold dataset d out and N the number of
datasets, so that the mean over splits equals the review's dataset-level mean difference.
The t reference has J - 1 degrees of freedom, as in Nadeau and Bengio; the script also
reports the p-value with N - 1, the number of datasets less one. For every contrast of the
review's families it reports the corrected standard error, its ratio to the review's
bootstrap standard error, the two-sided p-value (Holm-adjusted within the family, as the
review does) and the equivalence verdicts at the review's margins from the 90% interval.
Group-aware outer splits (scripts/grouped_outer_splits.py) differ in size by a dataset or two;
with ``--unequal-splits`` the factor uses the mean training and held-out counts.
"""

from __future__ import annotations

import argparse
import json
import math
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

from tabular_sota_benchmark.inference import holm  # noqa: E402


def split_means(scores: pd.DataFrame) -> pd.DataFrame:
    """Split-by-method held-out statistic whose mean over splits is the dataset-level mean.

    Each held-out dataset d of split j contributes its mean regret times J/(N n_d), with n_d
    the number of splits that hold d out and N the number of datasets."""
    held = scores[["split_id", "dataset"]].drop_duplicates()
    n_splits = held["split_id"].nunique()
    n_datasets = held["dataset"].nunique()
    times_held = held.groupby("dataset")["split_id"].nunique()
    weighted = scores.assign(
        _w=scores["mean_normalized_rank_regret"] * n_splits / (n_datasets * scores["dataset"].map(times_held))
    )
    return weighted.pivot_table(index="split_id", columns="method", values="_w", aggfunc="sum")


def corrected_test(
    differences: np.ndarray, *, n_train: float, n_test: float, margins: list[float], n_datasets: int | None = None
) -> dict[str, float]:
    j = int(differences.size)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    se = math.sqrt((1.0 / j + n_test / n_train) * variance)
    df = j - 1

    def two_sided(dof: int) -> float:
        if se > 0:
            return float(2.0 * stats.t.sf(abs(mean) / se, dof))
        return 1.0 if mean == 0 else 0.0

    p = two_sided(df)
    half = float(stats.t.ppf(0.95, df) * se)
    row: dict[str, float] = {
        "n_splits": j,
        "split_mean_difference": mean,
        "nb_se": se,
        "nb_p": p,
        "nb_p_df_datasets": two_sided(int(n_datasets) - 1) if n_datasets else p,
        "nb_ci90_low": mean - half,
        "nb_ci90_high": mean + half,
        "nb_smallest_equivalence_margin": max(abs(mean - half), abs(mean + half)),
    }
    for margin in margins:
        row[f"nb_equivalent_{margin:.3f}".replace(".", "p")] = bool(mean - half > -margin and mean + half < margin)
    return row


def run(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_output)
    nested_dir = Path(args.nested_output)
    config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    margins = [float(value) for value in config.get("margins", [0.005, 0.010, 0.015, 0.020])]
    scores = pd.read_parquet(review_dir / "dataset_policy_scores.parquet")
    contrasts = pd.read_csv(review_dir / "contrasts.csv")
    splits = pd.read_csv(nested_dir / "split_results.csv")
    sizes = splits[["n_train_datasets", "n_heldout_datasets"]].drop_duplicates()
    if sizes.shape[0] != 1 and not args.unequal_splits:
        raise ValueError("The outer splits do not all have the same training and held-out sizes.")
    n_train = float(splits["n_train_datasets"].mean())
    n_test = float(splits["n_heldout_datasets"].mean())
    means = split_means(scores)
    n_datasets = int(scores["dataset"].nunique())
    rows = []
    for _, contrast in contrasts.iterrows():
        left, right = str(contrast["left"]), str(contrast["right"])
        differences = (means[left] - means[right]).dropna().to_numpy(dtype=float)
        result = corrected_test(differences, n_train=n_train, n_test=n_test, margins=margins, n_datasets=n_datasets)
        rows.append(
            {
                "family": contrast["family"],
                "left": left,
                "right": right,
                "mean_difference": float(contrast["mean_difference"]),
                "bootstrap_se": float(contrast["se_bootstrap"]),
                "bootstrap_holm_p": float(contrast["bootstrap_holm_p"]),
                **result,
                "nb_se_ratio": result["nb_se"] / float(contrast["se_bootstrap"]),
            }
        )
    table = pd.DataFrame(rows)
    table["nb_holm_p"] = np.nan
    table["nb_holm_p_df_datasets"] = np.nan
    for _, index in table.groupby("family", sort=False).groups.items():
        table.loc[index, "nb_holm_p"] = holm(table.loc[index, "nb_p"].tolist())
        table.loc[index, "nb_holm_p_df_datasets"] = holm(table.loc[index, "nb_p_df_datasets"].tolist())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "corrected_resampled_tests.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "review_output": str(review_dir),
                "nested_output": str(nested_dir),
                "review_config": str(args.review_config),
                "n_splits": int(means.shape[0]),
                "n_train_datasets": n_train,
                "n_test_datasets": n_test,
                "unequal_splits": bool(args.unequal_splits),
                "n_datasets": n_datasets,
                "split_weights": "J / (N n_d): the mean over splits is the dataset-level mean",
                "variance_factor": 1.0 / means.shape[0] + n_test / n_train,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'corrected_resampled_tests.csv'} ({len(table)} contrasts)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unequal-splits", action="store_true", help="use the mean split sizes when they differ")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
