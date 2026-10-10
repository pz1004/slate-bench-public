"""Policy-grid degrees-of-freedom diagnostics and nested-objective sensitivity.

Re-selects the nested policy from the completed run's training scores, so no selector is
rebuilt:

* prefix-grid sweep: the nested choice restricted to the first K pool policies, with the
  number of distinct selected policies, selection entropy and the optimism gap
  (held-out minus training regret). These are descriptive quantities;
* objective and tie-break sensitivity: the declared rule (unit-weighted training regret,
  then runtime, then stable order), without the runtime tie-break, with the order
  reversed, and with a dataset-weighted training objective that matches the
  dataset-level estimand.

Held-out values come from the review's dataset-by-split scores and are summarised on
the artifact's shared resample matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import ArtifactResampler, paired_contrast  # noqa: E402


def _entropy_bits(choices: pd.Series) -> float:
    counts = choices.value_counts().to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _choose(frame: pd.DataFrame, *, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    return (
        frame.sort_values(["split_id", *columns], ascending=[True, *ascending], kind="mergesort")
        .groupby("split_id", as_index=False)
        .head(1)[["split_id", "method"]]
        .rename(columns={"method": "selected_policy"})
    )


def _heldout_dataset_means(choices: pd.DataFrame, dataset_scores: pd.DataFrame) -> pd.DataFrame:
    joined = choices.merge(
        dataset_scores,
        left_on=["split_id", "selected_policy"],
        right_on=["split_id", "method"],
        how="left",
    )
    if joined["method"].isna().any():
        raise ValueError("Held-out dataset scores are missing for some selected policies.")
    return joined.groupby("dataset", as_index=False).agg(
        mean_normalized_rank_regret=("mean_normalized_rank_regret", "mean"),
        mean_total_seconds=("mean_total_seconds", "mean"),
    )


def prefix_sweep(
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    counts: list[int],
    n_train_datasets: float,
) -> pd.DataFrame:
    pool = training[training["role"].eq("pool")]
    held = heldout[heldout["role"].eq("pool")]
    rows = []
    for count in counts:
        subset = pool[pool["method_order"].astype(int) < int(count)]
        chosen = _choose(
            subset,
            columns=["mean_normalized_rank_regret", "mean_total_seconds", "method_order"],
            ascending=[True, True, True],
        )
        train_values = chosen.merge(
            subset.rename(columns={"method": "selected_policy"}),
            on=["split_id", "selected_policy"],
        )
        held_values = chosen.merge(
            held.rename(columns={"method": "selected_policy"}),
            on=["split_id", "selected_policy"],
        )
        rows.append(
            {
                "n_policy_methods": int(count),
                "policy_to_train_dataset_ratio": float(count / n_train_datasets),
                "n_selected_policies": int(chosen["selected_policy"].nunique()),
                "selection_entropy_bits": _entropy_bits(chosen["selected_policy"]),
                "max_selection_share": float(chosen["selected_policy"].value_counts().max() / chosen.shape[0]),
                "mean_train_norm_regret": float(train_values["mean_normalized_rank_regret"].mean()),
                "mean_heldout_norm_regret": float(held_values["mean_normalized_rank_regret"].mean()),
                "mean_optimism_gap": float(
                    held_values["mean_normalized_rank_regret"].mean()
                    - train_values["mean_normalized_rank_regret"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def dataset_weighted_training_scores(units: pd.DataFrame) -> pd.DataFrame:
    train = units[units["phase"].eq("train") & units["role"].eq("pool")]
    per_dataset = train.groupby(["split_id", "method", "dataset"], as_index=False).agg(
        regret=("normalized_rank_regret", "mean"),
        seconds=("total_seconds", "mean"),
    )
    return per_dataset.groupby(["split_id", "method"], as_index=False).agg(
        mean_normalized_rank_regret=("regret", "mean"),
        mean_total_seconds=("seconds", "mean"),
    )


def objective_sensitivity(
    training: pd.DataFrame,
    units: pd.DataFrame,
    dataset_scores: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    nested_reference: pd.Series,
) -> pd.DataFrame:
    pool = training[training["role"].eq("pool")]
    orders = pool[["method", "method_order"]].drop_duplicates()
    dataset_weighted = dataset_weighted_training_scores(units).merge(orders, on="method")
    rules = {
        "regret_runtime_order": (pool, ["mean_normalized_rank_regret", "mean_total_seconds", "method_order"], [True, True, True]),
        "regret_order": (pool, ["mean_normalized_rank_regret", "method_order"], [True, True]),
        "regret_runtime_reverse_order": (pool, ["mean_normalized_rank_regret", "mean_total_seconds", "method_order"], [True, True, False]),
        "dataset_weighted_regret_runtime_order": (
            dataset_weighted,
            ["mean_normalized_rank_regret", "mean_total_seconds", "method_order"],
            [True, True, True],
        ),
    }
    rows = []
    for rule, (frame, columns, ascending) in rules.items():
        chosen = _choose(frame, columns=columns, ascending=ascending)
        means = _heldout_dataset_means(chosen, dataset_scores).set_index("dataset")
        estimate, low, high = resampler.mean_ci(means["mean_normalized_rank_regret"])
        contrast = paired_contrast(means["mean_normalized_rank_regret"], nested_reference, resampler)
        rows.append(
            {
                "rule": rule,
                "n_selected_policies": int(chosen["selected_policy"].nunique()),
                "selection_entropy_bits": _entropy_bits(chosen["selected_policy"]),
                "mean_normalized_rank_regret": estimate,
                "ci_low": low,
                "ci_high": high,
                "mean_total_seconds": float(means["mean_total_seconds"].mean()),
                "difference_vs_declared_rule": contrast["mean_difference"],
                "difference_ci_low": contrast["ci_low"],
                "difference_ci_high": contrast["ci_high"],
                "splits_with_same_choice_as_declared": np.nan,
                "_choices": chosen,
            }
        )
    declared = rows[0]["_choices"].set_index("split_id")["selected_policy"]
    for row in rows:
        choices = row.pop("_choices").set_index("split_id")["selected_policy"]
        row["splits_with_same_choice_as_declared"] = int((choices == declared.reindex(choices.index)).sum())
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    nested_dir = Path(args.nested_output)
    review_dir = Path(args.review_output)
    training = pd.read_csv(nested_dir / "training_policy_scores.csv")
    heldout = pd.read_csv(nested_dir / "heldout_policy_scores.csv")
    units = pd.read_parquet(nested_dir / "unit_scores.parquet")
    dataset_scores = pd.read_parquet(review_dir / "dataset_policy_scores.parquet")
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    split_results = pd.read_csv(nested_dir / "split_results.csv")
    resampler = ArtifactResampler(means["dataset"].unique(), seed=int(args.bootstrap_seed))
    nested = means[means["method"].eq("nested_policy_selected")].set_index("dataset")[
        "mean_normalized_rank_regret"
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = [int(value) for value in args.prefix_counts]
    prefix_sweep(
        training,
        heldout,
        counts=counts,
        n_train_datasets=float(split_results["n_train_datasets"].mean()),
    ).to_csv(output_dir / "prefix_grid_sweep.csv", index=False)
    objective_sensitivity(
        training,
        units,
        dataset_scores,
        resampler,
        nested_reference=nested,
    ).to_csv(output_dir / "objective_sensitivity.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"nested_output": str(nested_dir), "review_output": str(review_dir), "command": sys.argv}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-counts", nargs="+", default=[8, 16, 24, 32, 45])
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
