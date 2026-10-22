"""How much one contrast depends on single datasets, on trimming, and on dataset size.

Nested selection's gain over validation-best on TabArena comes mostly from a few datasets.
This script recomputes the contrast's dataset-level statistics:

- after removing each dataset in turn, with the bootstrap p-values of every contrast of the
  contrast's Holm family (``--family``) recomputed on the kept datasets and Holm-adjusted;
- after symmetric trimming: trimmed means with Yuen's one-sample test, and the mean after
  removing the k largest gains together with the k largest losses;
- after removing the k largest gains only, a deliberately adverse check;
- by dataset size: the Spearman correlation between the dataset's number of rows and the
  difference, and the mean difference by the number of evaluation units per dataset
  (TabArena evaluates small datasets with more repeats).

Nothing is refitted: the nested choices were made with every dataset available for
training, so removing a dataset changes the evaluation only. Intervals and tests on a subset
of datasets use a resample matrix drawn for that subset from the review's seed salted with
the subset's name (``ArtifactResampler.subset``); the corrected resampled t-test uses the
weighted split means of ``scripts/corrected_resampled_test.py`` on the kept datasets.
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

from corrected_resampled_test import corrected_test, split_means  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p, holm  # noqa: E402


def _interval(resampler: ArtifactResampler, values: pd.Series) -> tuple[float, float, float, float]:
    aligned = resampler.align(values)
    boot = aligned[resampler.indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(aligned.mean()), float(low), float(high), float(bootstrap_mean_p(boot))


def yuen_one_sample(values: np.ndarray, trim: float) -> tuple[float, float]:
    """Trimmed mean and the two-sided p-value of Yuen's one-sample test of a zero trimmed mean."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    g = int(math.floor(trim * n))
    trimmed = x[g : n - g]
    winsorized = np.clip(x, x[g], x[n - g - 1])
    h = n - 2 * g
    se = math.sqrt(float(np.var(winsorized, ddof=1))) / ((1.0 - 2.0 * trim) * math.sqrt(n))
    mean = float(trimmed.mean())
    p = float(2.0 * stats.t.sf(abs(mean) / se, h - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    return mean, p


def _corrected(scores: pd.DataFrame, drop: list[str], left: str, right: str, n_train: int) -> dict[str, float]:
    kept = scores[~scores["dataset"].isin(drop)]
    split = split_means(kept)
    # Held-out datasets left per split once the dropped datasets leave the evaluation.
    n_test = float(kept.groupby("split_id")["dataset"].nunique().mean())
    return corrected_test(
        (split[left] - split[right]).dropna().to_numpy(dtype=float), n_train=n_train, n_test=n_test, margins=[]
    )


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
    contrasts = pd.read_csv(review_dir / "contrasts.csv")
    family = contrasts[contrasts["family"].eq(args.family)][["left", "right"]].drop_duplicates()
    pairs = [(str(a), str(b)) for a, b in family.itertuples(index=False)]
    if (args.left, args.right) not in pairs:
        raise ValueError(f"{args.left} - {args.right} is not in family {args.family}")
    position = pairs.index((args.left, args.right))
    family_diffs = [(regret[a] - regret[b]).dropna() for a, b in pairs]

    rows = []
    for dataset in diff.index:
        kept = diff.drop(dataset)
        resampler = full.subset(kept.index, salt=f"loo|{dataset}")
        mean, low, high, p = _interval(resampler, kept)
        raw = [_interval(resampler, values.drop(dataset))[3] for values in family_diffs]
        rows.append(
            {
                "dropped": dataset,
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_p": p,
                "bootstrap_holm_p": float(holm(raw)[position]),
            }
        )
    loo = pd.DataFrame(rows).sort_values("mean_difference").reset_index(drop=True)

    scores = pd.read_parquet(review_dir / "dataset_policy_scores.parquet")
    splits = pd.read_csv(Path(args.nested_output) / "split_results.csv")
    n_train = int(splits["n_train_datasets"].iloc[0])
    k = int(args.top)
    ordered = diff.sort_values()
    top = ordered.index[:k].tolist()
    bottom = ordered.index[-k:].tolist()

    kept = diff.drop(top)
    top_mean, top_low, top_high, top_p = _interval(full.subset(kept.index, salt=f"top{k}"), kept)
    top_nb = _corrected(scores, top, args.left, args.right, n_train)
    kept = diff.drop(top + bottom)
    sym_mean, sym_low, sym_high, sym_p = _interval(full.subset(kept.index, salt=f"symmetric{k}"), kept)
    sym_nb = _corrected(scores, top + bottom, args.left, args.right, n_train)
    trim10, trim10_p = yuen_one_sample(diff.to_numpy(), 0.10)
    trim20, trim20_p = yuen_one_sample(diff.to_numpy(), 0.20)

    metadata = pd.read_csv(args.dataset_metadata).set_index("dataset_name")
    rows_per_dataset = metadata["num_instances"].reindex(diff.index).astype(float)
    rho, rho_p = stats.spearmanr(np.log(rows_per_dataset), diff.to_numpy())
    n_units = scores.groupby("dataset")["n_units"].max().reindex(diff.index)
    many = n_units >= n_units.max()
    full_mean, full_low, full_high, full_p = _interval(full, diff)
    summary = pd.DataFrame(
        [
            {
                "left": args.left,
                "right": args.right,
                "family": args.family,
                "family_size": len(pairs),
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
                "loo_max_holm_p": float(loo["bootstrap_holm_p"].max()),
                "loo_n_not_separated": int((loo["bootstrap_holm_p"] >= args.alpha).sum()),
                "loo_not_separated": ";".join(loo.loc[loo["bootstrap_holm_p"] >= args.alpha, "dropped"]),
                "top": k,
                "top_datasets": ";".join(top),
                "top_share_of_sum": float(diff[top].sum() / diff.sum()),
                "topout_mean_difference": top_mean,
                "topout_ci_low": top_low,
                "topout_ci_high": top_high,
                "topout_bootstrap_p": top_p,
                "topout_nb_p": top_nb["nb_p"],
                "symmetric_mean_difference": sym_mean,
                "symmetric_ci_low": sym_low,
                "symmetric_ci_high": sym_high,
                "symmetric_bootstrap_p": sym_p,
                "symmetric_nb_p": sym_nb["nb_p"],
                "trim10_mean": trim10,
                "trim10_yuen_p": trim10_p,
                "trim20_mean": trim20,
                "trim20_yuen_p": trim20_p,
                "size_spearman": float(rho),
                "size_spearman_p": float(rho_p),
                "many_units": int(n_units.max()),
                "many_units_datasets": int(many.sum()),
                "many_units_mean": float(diff[many].mean()),
                "many_units_max_rows": float(rows_per_dataset[many].max()),
                "few_units_min_rows": float(rows_per_dataset[~many].min()),
                "few_units": int(n_units.min()),
                "few_units_datasets": int((~many).sum()),
                "few_units_mean": float(diff[~many].mean()),
            }
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loo.to_csv(output_dir / "leave_one_out.csv", index=False)
    summary.to_csv(output_dir / "influence_summary.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "review_output": str(review_dir),
                "nested_output": str(args.nested_output),
                "review_config": str(args.review_config),
                "dataset_metadata": str(args.dataset_metadata),
                "bootstrap_seed": int(bootstrap.get("seed", 20260608)),
                "resamples": int(bootstrap.get("resamples", 10_000)),
                "subset_salts": ["loo|<dataset>", f"top{k}", f"symmetric{k}"],
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.T.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path, default=Path("data/tabarena/tabarena_dataset_metadata.csv"))
    parser.add_argument("--left", default="nested_policy_selected")
    parser.add_argument("--right", default="meta_selector_single")
    parser.add_argument("--family", default="primary")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
