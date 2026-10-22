"""Compare two policy pools with the same contrasts, tests and Holm families.

The declared 45-row pool (configs/review_tabarena.yaml) has a five-member primary family and
the 8-row reduced pool (configs/review_tabarena_shrink.yaml) a three-member one, so a verdict
that differs between their reviews can come from the family size rather than from the pool.
This script recomputes, for each pool, nested selection minus the same five rows (task-
adaptive selection, the guard with delta = 0.05, the family-prior rule, the pool's train-
selected fixed rule and validation-best) and Holm-adjusts the bootstrap test, the paired
t-test and the corrected resampled t-test (scripts/corrected_resampled_test.py) both within
the five contrasts and within the three that the reduced pool's review declares. It also
compares the two nested selections directly: their dataset-level difference on the shared
resample matrix and, because the two runs use the same outer splits, the corrected test on
their split statistics. The train-selected rule is each review's own, over its own fixed pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from corrected_resampled_test import corrected_test, split_means  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, holm, paired_contrast  # noqa: E402

NESTED = "nested_policy_selected"
FIVE = [
    "meta_selector_task_adaptive_budgeted",
    "meta_selector_meta_feature_guard_005",
    "meta_selector_family_prior_budgeted",
    "train_selected_fixed_reference__declared_fixed",
    "meta_selector_single",
]
THREE = FIVE[2:]
TESTS = {"bootstrap": "bootstrap_p", "ttest": "ttest_p", "corrected": "nb_p"}


def load(review_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    return means, split_means(pd.read_parquet(review_dir / "dataset_policy_scores.parquet"))


def split_sizes(nested_dir: Path) -> tuple[int, int]:
    splits = pd.read_csv(nested_dir / "split_results.csv")
    sizes = splits[["n_train_datasets", "n_heldout_datasets"]].drop_duplicates()
    if sizes.shape[0] != 1:
        raise ValueError(f"{nested_dir}: the outer splits differ in size.")
    return int(sizes["n_train_datasets"].iloc[0]), int(sizes["n_heldout_datasets"].iloc[0])


def run(args: argparse.Namespace) -> None:
    review_config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    bootstrap = dict(review_config.get("bootstrap", {}))
    seed = int(bootstrap.get("seed", 20260608))
    resamples = int(bootstrap.get("resamples", 10_000))
    rank_scale = float(review_config["rank_scale"])
    pools = {"declared": (Path(args.declared_review), Path(args.declared_nested)),
             "reduced": (Path(args.reduced_review), Path(args.reduced_nested))}
    loaded = {name: load(review) for name, (review, _) in pools.items()}
    datasets = sorted(loaded["declared"][0].index.astype(str))
    if sorted(loaded["reduced"][0].index.astype(str)) != datasets:
        raise ValueError("The two reviews cover different datasets.")
    resampler = ArtifactResampler(datasets, seed=seed, n_resamples=resamples)

    rows = []
    for pool, (_, nested_dir) in pools.items():
        means, split = loaded[pool]
        n_train, n_test = split_sizes(nested_dir)
        for right in FIVE:
            result = paired_contrast(means[NESTED], means[right], resampler, rank_scale=rank_scale)
            nb = corrected_test(
                (split[NESTED] - split[right]).dropna().to_numpy(dtype=float), n_train=n_train, n_test=n_test, margins=[]
            )
            rows.append({"pool": pool, "left": NESTED, "right": right, **result, "nb_p": nb["nb_p"], "nb_se": nb["nb_se"]})
    table = pd.DataFrame(rows)
    for family, members in {"five": FIVE, "three": THREE}.items():
        for test, column in TESTS.items():
            table[f"{test}_holm_{family}"] = float("nan")
            for pool in pools:
                index = table.index[table["pool"].eq(pool) & table["right"].isin(members)]
                table.loc[index, f"{test}_holm_{family}"] = holm(table.loc[index, column].tolist())

    declared_means, declared_split = loaded["declared"]
    reduced_means, reduced_split = loaded["reduced"]
    direct = paired_contrast(declared_means[NESTED], reduced_means[NESTED], resampler, rank_scale=rank_scale)
    n_train, n_test = split_sizes(pools["declared"][1])
    if split_sizes(pools["reduced"][1]) != (n_train, n_test):
        raise ValueError("The two runs' outer splits differ in size.")
    direct_nb = corrected_test(
        (declared_split[NESTED] - reduced_split[NESTED]).dropna().to_numpy(dtype=float),
        n_train=n_train,
        n_test=n_test,
        margins=[float(value) for value in review_config.get("margins", [0.005, 0.010, 0.015, 0.020])],
    )
    direct_row = pd.DataFrame([{"left": "nested (declared pool)", "right": "nested (reduced pool)", **direct, **direct_nb}])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "matched_contrasts.csv", index=False)
    direct_row.to_csv(output_dir / "direct_difference.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pools": {name: {"review": str(review), "nested": str(nested)} for name, (review, nested) in pools.items()},
                "review_config": str(args.review_config),
                "families": {"five": FIVE, "three": THREE},
                "bootstrap_seed": seed,
                "resamples": resamples,
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shown = ["pool", "right", "mean_difference"] + [f"{t}_holm_{f}" for f in ("three", "five") for t in TESTS]
    print(table[shown].round(4).to_string(index=False))
    print(direct_row[["mean_difference", "ci_low", "ci_high", "bootstrap_p", "ttest_p", "nb_p"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declared-review", type=Path, required=True)
    parser.add_argument("--declared-nested", type=Path, required=True)
    parser.add_argument("--reduced-review", type=Path, required=True)
    parser.add_argument("--reduced-nested", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True, help="the declared review's config (seed, scale)")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
