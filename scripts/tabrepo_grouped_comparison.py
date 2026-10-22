"""Group-aware against random outer splits on TabRepo, at the same number of splits.

scripts/grouped_outer_splits.py reruns the TabRepo audit on 24 outer splits (4 folds repeated
6 times) that keep related datasets together, against the audit's 100 random splits. A
difference between the two runs can come from the grouping or from the Monte Carlo noise of
a design with fewer splits. For the central contrasts this script compares the grouped run's
dataset-level mean with the random run's in a two-sample test, from the review's split-level
scores (every split's choices are made independently of the others, so a set of splits is a
valid design). The random run's Monte Carlo variance comes from resampling its 100 splits.
The grouped run's comes either from random designs of 24 splits drawn with replacement from
the 100 (as noisy as a random design of its size) or from the spread of its six repeat-level
means, each a full partition of the datasets (its own noise, with 5 degrees of freedom).
p-values are Holm-adjusted over the contrasts of a panel. The earlier summary, where the
grouped value falls among subsets of 24 of the 100 random splits drawn without replacement,
is kept for reference. Nothing is refitted.
"""

from __future__ import annotations

import argparse
import json
import math
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

from tabular_sota_benchmark.inference import holm  # noqa: E402

NESTED = "nested_policy_selected"
REFERENCES = {
    "score_only": [
        "train_selected_fixed_reference__score_only",
        "meta_selector_single",
        "meta_selector_family_prior_budgeted",
        "meta_selector_task_adaptive_budgeted",
        "meta_selector_meta_feature_guard_005",
        "reference_optimism_corrected",
    ],
    "ensemble": [
        "train_selected_fixed_reference__ensemble_inclusive",
        "train_selected_fixed_reference__score_only",
        "meta_selector_single",
        "meta_selector_tabrepo_ensemble_k3",
    ],
}


def dataset_contrast(scores: pd.DataFrame, splits: np.ndarray, right: str) -> float:
    kept = scores[scores["split_id"].isin(splits) & scores["method"].isin([NESTED, right])]
    means = kept.groupby(["dataset", "method"])["mean_normalized_rank_regret"].mean().unstack("method")
    return float((means[NESTED] - means[right]).dropna().mean())


def split_matrix(scores: pd.DataFrame, right: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ids and the split-by-dataset matrices of held-out differences (zero where a
    dataset is not held out) and of held-out indicators."""
    kept = scores[scores["method"].isin([NESTED, right])]
    wide = kept.pivot_table(index=["split_id", "dataset"], columns="method", values="mean_normalized_rank_regret")
    diff = (wide[NESTED] - wide[right]).rename("diff").reset_index()
    values = diff.pivot(index="split_id", columns="dataset", values="diff")
    return values.index.to_numpy(), values.fillna(0.0).to_numpy(dtype=float), values.notna().to_numpy(dtype=float)


def design_means(counts: np.ndarray, values: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Dataset-level means of designs given as split counts (rows of ``counts``)."""
    sums, times = counts @ values, counts @ held
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(times > 0, sums / np.where(times > 0, times, 1.0), np.nan), axis=1)


def two_sample(grouped_scores: pd.DataFrame, random_scores: pd.DataFrame, right: str, *, folds: int, rng: np.random.Generator, draws: int) -> dict[str, float]:
    ids, values, held = split_matrix(random_scores, right)
    n = ids.size
    full = design_means(np.ones((1, n)), values, held)[0]
    boot = design_means(np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]).astype(float), values, held)
    g_ids, g_values, g_held = split_matrix(grouped_scores, right)
    if not np.allclose(g_held.reshape(-1, folds, g_held.shape[1]).sum(axis=1), 1.0):
        raise SystemExit("A repeat of the grouped splits does not hold out every dataset exactly once.")
    grouped = design_means(np.ones((1, g_ids.size)), g_values, g_held)[0]
    repeats = g_values.reshape(-1, folds, g_values.shape[1]).sum(axis=1).mean(axis=1)
    small = design_means(np.stack([np.bincount(rng.integers(0, n, g_ids.size), minlength=n) for _ in range(draws)]).astype(float), values, held)
    se_random = float(np.std(boot, ddof=1))
    se_like = float(np.std(small, ddof=1))
    se_own = float(np.std(repeats, ddof=1) / math.sqrt(repeats.size))
    shift = grouped - full
    out = {"grouped": float(grouped), "random_all": float(full), "shift": float(shift), "se_random_all": se_random,
           "se_grouped_random_like": se_like, "se_grouped_own": se_own, "repeats": int(repeats.size)}
    if abs(shift) <= 1e-12:
        out.update({"p_random_like": 1.0, "p_own": 1.0})
        return out
    out["p_random_like"] = float(2 * stats.norm.sf(abs(shift) / math.hypot(se_like, se_random)))
    own = math.hypot(se_own, se_random)
    dof = own**4 / (se_own**4 / (repeats.size - 1) + se_random**4 / (n - 1)) if se_own > 0 else float(n - 1)
    out["p_own"] = float(2 * stats.t.sf(abs(shift) / own, dof))
    return out


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(int(args.seed))
    # A separate stream, so that the subset draws stay those of the earlier summary.
    test_rng = np.random.default_rng([int(args.seed), 1])
    folds = int(json.loads((Path(args.grouped_splits)).read_text(encoding="utf-8"))["folds"])
    rows, tests = [], []
    for panel, references in REFERENCES.items():
        random_dir, grouped_dir = Path(getattr(args, f"{panel}_random")), Path(getattr(args, f"{panel}_grouped"))
        random_scores = pd.read_parquet(random_dir / "dataset_policy_scores.parquet")
        grouped_scores = pd.read_parquet(grouped_dir / "dataset_policy_scores.parquet")
        random_ids = np.sort(random_scores["split_id"].unique())
        grouped_ids = np.sort(grouped_scores["split_id"].unique())
        draws = [rng.choice(random_ids, size=grouped_ids.size, replace=False) for _ in range(int(args.subsets))]
        for right in references:
            subset_values = np.array([dataset_contrast(random_scores, draw, right) for draw in draws])
            grouped = dataset_contrast(grouped_scores, grouped_ids, right)
            low, high = np.percentile(subset_values, [2.5, 97.5])
            rows.append(
                {
                    "panel": panel,
                    "right": right,
                    "random_all": dataset_contrast(random_scores, random_ids, right),
                    "random_subset_mean": float(subset_values.mean()),
                    "random_subset_low": float(low),
                    "random_subset_high": float(high),
                    "grouped": grouped,
                    "share_subsets_at_or_below_grouped": float((subset_values <= grouped).mean()),
                    "share_subsets_at_or_above_grouped": float((subset_values >= grouped).mean()),
                }
            )
            result = two_sample(grouped_scores, random_scores, right, folds=folds, rng=test_rng, draws=int(args.draws))
            if not math.isclose(result["grouped"], grouped, abs_tol=1e-12):
                raise SystemExit("The split matrices do not reproduce the grouped run's dataset-level mean.")
            tests.append({"panel": panel, "right": right, **result})
    out = pd.DataFrame(rows)
    test_table = pd.DataFrame(tests)
    for column in ["p_random_like", "p_own"]:
        test_table[f"{column}_holm"] = np.nan
        for _, index in test_table.groupby("panel", sort=False).groups.items():
            test_table.loc[index, f"{column}_holm"] = holm(test_table.loc[index, column].tolist())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "grouped_vs_random_subsets.csv", index=False)
    test_table.to_csv(output_dir / "grouped_vs_random_tests.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"subsets": int(args.subsets), "draws": int(args.draws), "seed": int(args.seed), "folds": folds, "command": sys.argv}, indent=2),
        encoding="utf-8",
    )
    print(test_table.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-only-random", type=Path, required=True)
    parser.add_argument("--score-only-grouped", type=Path, required=True)
    parser.add_argument("--ensemble-random", type=Path, required=True)
    parser.add_argument("--ensemble-grouped", type=Path, required=True)
    parser.add_argument("--grouped-splits", type=Path, default=Path("analysis-output/v2/score_level_tabrepo_grouped/grouped_splits.json"))
    parser.add_argument("--subsets", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=4000, help="bootstrap draws of split designs for the two-sample test")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
