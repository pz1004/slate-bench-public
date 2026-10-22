"""Separate the two effects of removing TabArena's imputed rows: the yardstick and the refit.

Removing the imputed rows (scripts/imputed_rows.py) changes the candidate set of 378 units,
so it changes the rank scale on which every choice is scored, and it changes what every
fitted policy learns. This script keeps the main run's held-out choices fixed and scores
them against the candidate sets without the imputed rows, with the shared rank rule. The
difference between this rescoring and the main run is the yardstick effect; the difference
between it and the rerun without the rows (configs/score_level_tabarena_no_imputed.yaml)
is the effect of refitting. Before anything else, validation-best's rescored regrets must
equal the rerun's, because removing exact or near copies leaves its choices' test errors
and ranks unchanged up to the rows removed.

Rows are aggregated as the review does (units, then splits, then datasets). The
train-selected reference is rebuilt from the review's per-split choices. The primary
contrasts are tested on the review's shared resample matrix and Holm-adjusted within the
primary family. The script also reports how often nested selection's held-out choice
differs between the main run and the rerun.
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

from tabular_sota_benchmark.inference import ArtifactResampler, contrast_table, paired_contrast  # noqa: E402
from tabular_sota_benchmark.ranking import rank_against_candidates, set_rank_rule  # noqa: E402

UNIT = ["dataset", "fold", "seed"]
NESTED = "nested_policy_selected"
VALIDATION_BEST = "meta_selector_single"
TRAIN_SELECTED = "train_selected_fixed_reference__declared_fixed"
PRIMARY = [
    "meta_selector_task_adaptive_budgeted",
    "meta_selector_meta_feature_guard_005",
    "meta_selector_family_prior_budgeted",
    TRAIN_SELECTED,
    VALIDATION_BEST,
]


def heldout_choices(nested_dir: Path) -> pd.DataFrame:
    columns = ["split_id", "phase", *UNIT, "method", "selected_member", "primary_metric"]
    units = pd.read_parquet(nested_dir / "unit_scores.parquet", columns=columns)
    return units[units["phase"].eq("heldout")].drop(columns="phase").reset_index(drop=True)


def rescore(choices: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Normalized rank regret of every held-out choice against ``candidates`` of its unit."""
    cand = candidates[[*UNIT, "metric_error"]].rename(columns={"metric_error": "primary_metric"})
    frame = pd.concat(
        [cand.assign(_candidate=True), choices.assign(_candidate=False)], ignore_index=True, sort=False
    )
    ranked = rank_against_candidates(
        frame, unit_columns=UNIT, value_column="primary_metric", candidate_mask=frame["_candidate"].to_numpy()
    )
    out = ranked[~ranked["_candidate"].astype(bool)].copy()
    return out[["split_id", *UNIT, "method", "selected_member", "normalized_rank_regret"]]


def dataset_means(scores: pd.DataFrame) -> pd.DataFrame:
    per_split = scores.groupby(["method", "split_id", "dataset"])["normalized_rank_regret"].mean().reset_index()
    return per_split.groupby(["dataset", "method"])["normalized_rank_regret"].mean().unstack("method")


def with_train_selected(scores: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    picks = choices[choices["pool"].eq("declared_fixed")][["split_id", "selected_reference_method"]]
    rows = scores.merge(picks, left_on=["split_id", "method"], right_on=["split_id", "selected_reference_method"])
    rows = rows.drop(columns="selected_reference_method").assign(method=TRAIN_SELECTED)
    return pd.concat([scores, rows], ignore_index=True)


def run(args: argparse.Namespace) -> None:
    set_rank_rule("consistent")
    review_config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    main_dir, rerun_dir = Path(args.main_output), Path(args.rerun_output)
    main_review, rerun_review = Path(args.main_review), Path(args.rerun_review)
    candidates = pd.read_csv(args.candidates)
    choices = heldout_choices(main_dir)
    rescored = rescore(choices, candidates)
    reference_picks = pd.read_csv(main_review / "train_selected_reference_choices.csv")
    rescored = with_train_selected(rescored, reference_picks)
    means = dataset_means(rescored)

    rerun = pd.read_csv(rerun_review / "dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    main = pd.read_csv(main_review / "dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    gap = float((means[VALIDATION_BEST] - rerun[VALIDATION_BEST].reindex(means.index)).abs().max())
    if gap > 1e-9:
        raise ValueError(f"Rescored validation-best does not reproduce the rerun (max gap {gap:.3g}).")

    bootstrap = dict(review_config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        means.index, seed=int(bootstrap.get("seed", 20260608)), n_resamples=int(bootstrap.get("resamples", 10_000))
    )
    margins = [float(value) for value in review_config.get("margins", [0.005, 0.010, 0.015, 0.020])]
    rows = []
    for right in PRIMARY:
        result = paired_contrast(
            means[NESTED], means[right], resampler, margins=margins, rank_scale=float(review_config["rank_scale"])
        )
        rows.append({"family": "rescored_primary", "left": NESTED, "right": right, **result})
    contrasts = contrast_table(rows)

    rerun_choices = heldout_choices(rerun_dir)
    key = ["split_id", *UNIT]
    nested_main = choices[choices["method"].eq(NESTED)].set_index(key)["selected_member"]
    nested_rerun = rerun_choices[rerun_choices["method"].eq(NESTED)].set_index(key)["selected_member"]
    joined = pd.concat([nested_main.rename("main"), nested_rerun.rename("rerun")], axis=1, join="inner")
    changed = float((joined["main"] != joined["rerun"]).mean())

    summary = pd.DataFrame(
        [
            {
                "method": method,
                "main_regret": float(main[method].mean()) if method in main.columns else np.nan,
                "rescored_regret": float(means[method].mean()),
                "rerun_regret": float(rerun[method].mean()) if method in rerun.columns else np.nan,
            }
            for method in [NESTED, *PRIMARY]
        ]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "rescored_summary.csv", index=False)
    contrasts.to_csv(output_dir / "rescored_contrasts.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "main_output": str(main_dir),
                "rerun_output": str(rerun_dir),
                "main_review": str(main_review),
                "rerun_review": str(rerun_review),
                "candidates": str(args.candidates),
                "validation_best_reproduction_max_gap": gap,
                "nested_choices_compared": int(joined.shape[0]),
                "nested_choice_changed_share": changed,
                "bootstrap_seed": int(bootstrap.get("seed", 20260608)),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.round(4).to_string(index=False))
    print(contrasts[["right", "mean_difference", "ci_low", "ci_high", "bootstrap_p", "bootstrap_holm_p"]].round(4).to_string(index=False))
    print(f"nested choices changed by the refit: {changed:.1%}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--main-review", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    parser.add_argument("--rerun-review", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True, help="the results file without the imputed rows")
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
