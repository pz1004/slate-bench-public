"""How much one contrast depends on single datasets (leave-one-out and leave-top-k-out).

Nested selection's gain over validation-best on TabArena comes mostly from a few datasets.
This script reruns the contrast's dataset-level statistics after removing each dataset in
turn, and after removing the k datasets with the largest gains. Nothing is refitted: the
nested choices were made on all training datasets, so this removes datasets from the
evaluation only. Intervals and tests use the review's shared resample matrix restricted
to the kept datasets, and the corrected resampled t-test uses the split means of the kept
held-out datasets (scripts/corrected_resampled_test.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from corrected_resampled_test import corrected_test  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p  # noqa: E402


def _interval(resampler: ArtifactResampler, values: pd.Series) -> tuple[float, float, float, float]:
    aligned = resampler.align(values)
    boot = aligned[resampler.indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(aligned.mean()), float(low), float(high), float(bootstrap_mean_p(boot))


def run(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_output)
    config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    bootstrap = dict(config.get("bootstrap", {}))
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    diff = (regret[args.left] - regret[args.right]).dropna()
    full = ArtifactResampler(
        diff.index, seed=int(bootstrap.get("seed", 20260608)), n_resamples=int(bootstrap.get("resamples", 10_000))
    )
    rows = []
    for dataset in diff.index:
        kept = diff.drop(dataset)
        mean, low, high, p = _interval(full.subset(kept.index, salt=f"loo|{dataset}"), kept)
        rows.append({"dropped": dataset, "mean_difference": mean, "ci_low": low, "ci_high": high, "bootstrap_p": p})
    loo = pd.DataFrame(rows).sort_values("mean_difference").reset_index(drop=True)

    top = diff.sort_values().index[: int(args.top)].tolist()
    kept = diff.drop(top)
    mean, low, high, p = _interval(full.subset(kept.index, salt=f"top{args.top}"), kept)
    scores = pd.read_parquet(review_dir / "dataset_policy_scores.parquet")
    scores = scores[~scores["dataset"].isin(top)]
    split = scores.pivot_table(index="split_id", columns="method", values="mean_normalized_rank_regret")
    splits = pd.read_csv(Path(args.nested_output) / "split_results.csv")
    n_train = int(splits["n_train_datasets"].iloc[0])
    # Held-out datasets left per split once the top datasets are removed from the evaluation.
    n_test = float(scores.groupby("split_id")["dataset"].nunique().mean())
    corrected = corrected_test(
        (split[args.left] - split[args.right]).dropna().to_numpy(dtype=float),
        n_train=n_train,
        n_test=n_test,
        margins=[],
    )
    full_mean, full_low, full_high, full_p = _interval(full, diff)
    summary = pd.DataFrame(
        [
            {
                "left": args.left,
                "right": args.right,
                "n_datasets": int(diff.size),
                "mean_difference": full_mean,
                "ci_low": full_low,
                "ci_high": full_high,
                "bootstrap_p": full_p,
                "loo_min_mean": float(loo["mean_difference"].min()),
                "loo_max_mean": float(loo["mean_difference"].max()),
                "loo_max_ci_high": float(loo["ci_high"].max()),
                "loo_all_exclude_zero": bool((loo["ci_high"] < 0).all() or (loo["ci_low"] > 0).all()),
                "loo_max_bootstrap_p": float(loo["bootstrap_p"].max()),
                "top": int(args.top),
                "top_datasets": ";".join(top),
                "top_share_of_sum": float(diff[top].sum() / diff.sum()),
                "topout_mean_difference": mean,
                "topout_ci_low": low,
                "topout_ci_high": high,
                "topout_bootstrap_p": p,
                "topout_nb_p": corrected["nb_p"],
                "topout_nb_split_mean": corrected["split_mean_difference"],
            }
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loo.to_csv(output_dir / "leave_one_out.csv", index=False)
    summary.to_csv(output_dir / "influence_summary.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"review_output": str(review_dir), "command": sys.argv}, indent=2), encoding="utf-8"
    )
    print(summary.T.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--left", default="nested_policy_selected")
    parser.add_argument("--right", default="meta_selector_single")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
