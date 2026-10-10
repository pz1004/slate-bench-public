"""Search-aware inference over the policy pool of one nested run.

Treats the policy pool as a searched family and asks:

* which rows cannot be separated from the best row: model confidence set (Hansen,
  Lunde & Nason 2011) over the pool plus the nested procedure, the train-selected
  reference and the declared references;
* whether any searched policy beats a committed benchmark: SPA test (Hansen 2005) with
  lower, consistent and upper p-values, and White's reality check;
* how likely an in-sample winner among the pool is to fall below the median out of
  sample: probability of backtest overfitting by CSCV (Bailey et al. 2017).

All statistics use the dataset-level held-out means from the review, and the bootstrap
uses the artifact's shared resample matrix. They inherit the review's conditioning on
the realised outer-training fits (manuscript Section 3.3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from _path_setup import ensure_src_path
except ModuleNotFoundError:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _path_setup import ensure_src_path

ensure_src_path()

from tabular_sota_benchmark.inference import (  # noqa: E402
    ArtifactResampler,
    model_confidence_set,
    probability_of_backtest_overfitting,
    superior_predictive_ability,
)
from tabular_sota_benchmark.labels import policy_label  # noqa: E402


def run(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_output)
    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    metadata = json.loads((Path(args.nested_output) / "metadata.json").read_text(encoding="utf-8"))
    artifact_meta = metadata["artifacts"][0]
    pool = [str(method) for method in artifact_meta["policy_methods"]]
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    tasks = means.drop_duplicates("dataset").set_index("dataset")["task"]
    resampler = ArtifactResampler(regret.index, seed=int(args.bootstrap_seed))
    extra = [method for method in args.extra_rows if method in regret.columns]
    rows_for_mcs = [*pool, *extra]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcs = model_confidence_set(regret[rows_for_mcs], resampler)
    mcs.insert(1, "label", mcs["policy"].map(policy_label))
    mcs["role"] = mcs["policy"].map(lambda method: "pool" if method in pool else "procedure_or_reference")
    mcs.to_csv(output_dir / "model_confidence_set.csv", index=False)
    mcs_pool = model_confidence_set(regret[pool], resampler)
    mcs_pool.insert(1, "label", mcs_pool["policy"].map(policy_label))
    mcs_pool.to_csv(output_dir / "model_confidence_set_pool_only.csv", index=False)

    spa_rows = []
    for benchmark in args.benchmarks:
        if benchmark not in regret.columns:
            continue
        alternatives = [method for method in pool if method != benchmark]
        result = superior_predictive_ability(regret[[benchmark, *alternatives]], benchmark, resampler)
        result["benchmark_label"] = policy_label(benchmark)
        result["best_alternative_label"] = policy_label(result["best_alternative"])
        spa_rows.append(result)
    pd.DataFrame(spa_rows).to_csv(output_dir / "superior_predictive_ability.csv", index=False)

    pbo = probability_of_backtest_overfitting(
        -regret[pool],
        n_blocks=int(args.pbo_blocks),
        strata=tasks.reindex(regret.index),
        seed=int(args.bootstrap_seed),
    )
    pd.DataFrame([pbo]).to_csv(output_dir / "probability_of_backtest_overfitting.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "review_output": str(review_dir),
                "nested_output": str(args.nested_output),
                "n_pool_policies": len(pool),
                "extra_rows": extra,
                "benchmarks": list(args.benchmarks),
                "bootstrap_seed": int(args.bootstrap_seed),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-rows",
        nargs="*",
        default=[
            "nested_policy_selected",
            "train_selected_fixed_reference__declared_fixed",
            "train_selected_fixed_reference__score_only",
            "train_selected_fixed_reference__ensemble_inclusive",
        ],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "meta_selector_family_prior_budgeted",
            "meta_selector_single",
            "meta_selector_tabrepo_ensemble_k3",
        ],
    )
    parser.add_argument("--pbo-blocks", type=int, default=10)
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
