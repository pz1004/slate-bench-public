#!/usr/bin/env python
"""Split-half replication of the TabRepo capability-dependence finding (E2).

The paper's TabRepo finding is that nested selection only looks good because the
policy pool contains prediction-ensemble rows: in the ensemble-inclusive pool
nested selection reaches ensemble-level regret, but in the score-only ablation it
falls back to raw-validation regret. To show this is not an artifact of the
particular 200-dataset slice, we split the 200 TabRepo datasets into two disjoint
seeded halves and re-cluster the *existing* nested runs on each half. If the
finding replicates, both halves show: ensemble-inclusive nested approx fixed
ensemble k=3 (low regret), while score-only nested approx raw validation
(higher regret).

Inputs are existing cached dataset_mean_scores.csv files; no nested run is
re-executed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

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

from tabular_sota_benchmark.inference import ArtifactResampler  # noqa: E402

VALUE = "mean_normalized_rank_regret"

# (display label, source review, method)
ROWS = [
    ("Ensemble-inclusive nested", "ensemble", "nested_policy_selected"),
    ("Fixed ensemble k=3", "ensemble", "meta_selector_tabrepo_ensemble_k3"),
    ("Raw validation", "ensemble", "meta_selector_single"),
    ("Score-only nested", "score_only", "nested_policy_selected"),
    ("Train-selected fixed rule (score-only)", "score_only", "train_selected_fixed_reference__score_only"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--ensemble-review",
        default="analysis-output/v2/score_level_tabrepo_nested_review/dataset_mean_scores.csv",
    )
    parser.add_argument(
        "--score-only-review",
        default="analysis-output/v2/score_level_tabrepo_score_only_review/dataset_mean_scores.csv",
    )
    parser.add_argument("--output-dir", default="analysis-output/v2/tabrepo_split_half")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260611, help="seed of the half split")
    parser.add_argument("--bootstrap-seed", type=int, default=20260608)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    ens = pd.read_csv(root / args.ensemble_review)
    sco = pd.read_csv(root / args.score_only_review)
    sources = {"ensemble": ens, "score_only": sco}

    datasets = np.array(sorted(ens["dataset"].unique()))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(datasets.size)
    half_a = set(datasets[perm[: datasets.size // 2]])
    half_b = set(datasets[perm[datasets.size // 2 :]])
    halves = {"Half A": half_a, "Half B": half_b}

    out_rows = []
    for half_name, members in halves.items():
        for label, source, method in ROWS:
            df = sources[source]
            sub = df[(df["method"] == method) & (df["dataset"].isin(members))]
            resampler = ArtifactResampler(
                sub["dataset"].unique(), seed=int(args.bootstrap_seed), n_resamples=int(args.bootstrap)
            ).subset(sub["dataset"].unique(), salt=half_name)
            mean, lo, hi = resampler.mean_ci(sub.set_index("dataset")[VALUE])
            out_rows.append(
                {
                    "half": half_name,
                    "n_datasets": int(len(members)),
                    "policy": label,
                    "mean_normalized_rank_regret": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

    summary = pd.DataFrame(out_rows)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "split_half_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== TabRepo split-half replication of capability dependence ===")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
