"""Corrected resampled t-test (Nadeau and Bengio 2003) for every contrast of a review.

The dataset bootstrap of the review resamples dataset-level differences with every fitted
choice held fixed, so it ignores that the 100 outer splits share training datasets. The
corrected resampled t-test works on the splits instead: for each outer split it takes the
held-out mean difference (datasets weighted equally) and inflates the variance of the
mean over the J splits by the factor 1/J + n_test/n_train, which accounts for the overlap
of the training sets of random subsampling. It is an approximation too, but one whose
behaviour under refitting is documented. For every contrast of the review's families the
script reports the split mean, the corrected standard error, its ratio to the review's
bootstrap standard error, the two-sided p-value (Holm-adjusted within the family, as the
review does) and the equivalence verdicts at the review's margins from the 90% interval.
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
    """Split-by-method held-out mean regret, each held-out dataset weighted equally."""
    return scores.pivot_table(
        index="split_id", columns="method", values="mean_normalized_rank_regret", aggfunc="mean"
    )


def corrected_test(
    differences: np.ndarray, *, n_train: int, n_test: int, margins: list[float]
) -> dict[str, float]:
    j = int(differences.size)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    se = math.sqrt((1.0 / j + n_test / n_train) * variance)
    df = j - 1
    if se > 0:
        p = float(2.0 * stats.t.sf(abs(mean) / se, df))
    else:
        p = 1.0 if mean == 0 else 0.0
    half = float(stats.t.ppf(0.95, df) * se)
    row: dict[str, float] = {
        "n_splits": j,
        "split_mean_difference": mean,
        "nb_se": se,
        "nb_p": p,
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
    if sizes.shape[0] != 1:
        raise ValueError("The outer splits do not all have the same training and held-out sizes.")
    n_train = int(sizes["n_train_datasets"].iloc[0])
    n_test = int(sizes["n_heldout_datasets"].iloc[0])
    means = split_means(scores)
    rows = []
    for _, contrast in contrasts.iterrows():
        left, right = str(contrast["left"]), str(contrast["right"])
        differences = (means[left] - means[right]).dropna().to_numpy(dtype=float)
        result = corrected_test(differences, n_train=n_train, n_test=n_test, margins=margins)
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
    for _, index in table.groupby("family", sort=False).groups.items():
        table.loc[index, "nb_holm_p"] = holm(table.loc[index, "nb_p"].tolist())
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
