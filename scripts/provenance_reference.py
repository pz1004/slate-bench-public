"""A reference that compares validation scores of one provenance class only.

Validation-best compares candidates whose validation scores absorbed different amounts of
search: defaults, tuned candidates, and candidates tuned and ensembled on the same folds.
This rule, added after the main results were known, compares like with like and fits
nothing: on each unit it takes the learner family whose default has the lowest validation
error (the validation-rank rule on the defaults, as for validation-best among defaults, so
that exact ties go to the earlier candidate) and deploys that family's
tuned-and-ensembled candidate, or its tuned candidate, or its default, whichever the unit
has first. It reads only the declared validation-side fields of the unit.

The rule needs no training data, so its held-out regret on a dataset is the same in every
outer split. The script scores it by the review's normalized rank regret, checks first that
the same code reproduces the review's validation-best and validation-best among defaults
exactly, and compares it with the review's rows on the review's dataset-level scores and
shared resample matrix (bootstrap test, paired t-test and signed-rank test, Holm-adjusted
within this script's family), and with the corrected resampled t-test of
``scripts/corrected_resampled_test.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
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

from corrected_resampled_test import corrected_test, split_means  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, contrast_table, gap_closed, holm, paired_contrast  # noqa: E402
from tabular_sota_benchmark.labels import policy_label  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "error_space_correction_for_reference", Path(__file__).resolve().with_name("error_space_correction.py")
)
assert _SPEC is not None and _SPEC.loader is not None
esc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(esc)

RULE = "reference_defaults_then_tuned"
VALIDATION_BEST = "meta_selector_single"
VALIDATION_BEST_DEFAULT = "reference_validation_best_default"
REFERENCES = [
    VALIDATION_BEST,
    VALIDATION_BEST_DEFAULT,
    "nested_policy_selected",
    "meta_selector_family_prior_budgeted",
    "train_selected_fixed_reference__declared_fixed",
    "reference_optimism_corrected",
]
DEPLOY_ORDER = ["tuned + ensemble", "tuned", "default"]
SINGLE_BEST = "reference_single_best_global"
ORACLE = "oracle"


def variant(method: str) -> str:
    """The parenthesized variant of a TabArena method, e.g. CAT (tuned + ensemble) -> tuned + ensemble."""
    return method.rsplit("(", 1)[-1].rstrip(")").strip() if "(" in method else ""


def unit_choices(frame: pd.DataFrame) -> pd.DataFrame:
    """Per unit: the rule's choice, validation-best's and validation-best among defaults."""
    frame = frame.assign(variant=frame["method"].map(variant), score=frame["val_primary_metric"] + frame["tie"])
    rows = []
    for unit, group in frame.groupby("unit", sort=False):
        best = group.loc[group["score"].idxmin()]
        defaults = group[group["variant"].eq("default")]
        if defaults.empty:
            raise ValueError(f"Unit {unit} has no default candidate.")
        # Validation-best among defaults is the validation-rank rule on the defaults: lowest
        # validation error, exact ties to the earlier candidate (no runtime tie-break).
        best_default = defaults.loc[defaults["val_primary_metric"].idxmin()]
        same_family = group[group["family"].eq(best_default["family"])]
        deployed = best_default
        for name in DEPLOY_ORDER:
            match = same_family[same_family["variant"].eq(name)]
            if not match.empty:
                deployed = match.loc[match["score"].idxmin()]
                break
        rows.append(
            {
                "unit": unit,
                "dataset": group["dataset"].iloc[0],
                RULE: float(deployed["regret"]),
                VALIDATION_BEST: float(best["regret"]),
                VALIDATION_BEST_DEFAULT: float(best_default["regret"]),
                "deployed_variant": str(deployed["variant"]),
                "deployed_method": str(deployed["method"]),
                "validation_best_method": str(best["method"]),
            }
        )
    return pd.DataFrame(rows)


def endpoints(units: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-unit error regrets and recorded runtime of the rule's deployed candidate, as the
    review computes them for its rows (max-min and median-normalized error regret, fit plus
    prediction seconds)."""
    keyed = candidates.assign(unit=pd.factorize(pd.MultiIndex.from_frame(candidates[esc.UNIT].astype(str)), sort=False)[0])
    stats_ = keyed.groupby("unit")["primary_metric"].agg(best="min", worst="max", middle="median")
    chosen = keyed.set_index(["unit", "method"])[["primary_metric", "fit_seconds", "predict_seconds"]]
    picked = chosen.reindex(pd.MultiIndex.from_arrays([units["unit"], units["deployed_method"]])).reset_index(drop=True)
    out = units[["unit", "dataset"]].copy()
    error = picked["primary_metric"].to_numpy(dtype=float)
    best = units["unit"].map(stats_["best"]).to_numpy(dtype=float)
    span = units["unit"].map(stats_["worst"]).to_numpy(dtype=float) - best
    middle = units["unit"].map(stats_["middle"]).to_numpy(dtype=float) - best
    out["maxmin_error_regret"] = np.where(span > 0, (error - best) / np.where(span > 0, span, 1.0), 0.0)
    out["median_error_regret"] = np.where(middle > 0, (error - best) / np.where(middle > 0, middle, 1.0), np.nan)
    out["total_seconds"] = picked["fit_seconds"].to_numpy(dtype=float) + picked["predict_seconds"].to_numpy(dtype=float)
    return out


def summary_row(units: pd.DataFrame, extra: pd.DataFrame, regret: pd.DataFrame, resampler: ArtifactResampler) -> pd.DataFrame:
    """The rule's row in the review's ``clustered_summary.csv`` columns."""
    by_dataset = units.groupby("dataset")[RULE].mean()
    per_dataset = extra.groupby("dataset")[["maxmin_error_regret", "median_error_regret", "total_seconds"]].mean()
    row: dict[str, object] = {"method": RULE, "label": policy_label(RULE)}
    for name, values in [
        ("mean_normalized_rank_regret", by_dataset),
        ("mean_maxmin_error_regret", per_dataset["maxmin_error_regret"]),
        ("mean_median_error_regret", per_dataset["median_error_regret"]),
        ("mean_total_seconds", per_dataset["total_seconds"]),
    ]:
        values = values.dropna()
        rs = resampler if values.shape[0] == resampler.n else resampler.subset(values.index, salt=f"{name}:{RULE}")
        estimate, low, high = rs.mean_ci(values)
        row[name], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = estimate, low, high
    row["beats_best_rate"] = row["beats_best_rate_ci_low"] = row["beats_best_rate_ci_high"] = 0.0
    if SINGLE_BEST in regret.columns and ORACLE in regret.columns:
        estimate, low, high = gap_closed(by_dataset.reindex(regret.index), regret[SINGLE_BEST], regret[ORACLE], resampler)
        row["gap_closed"], row["gap_closed_ci_low"], row["gap_closed_ci_high"] = estimate, low, high
    return pd.DataFrame([row])


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.benchmark_config).read_text(encoding="utf-8"))
    review_config = yaml.safe_load(Path(args.review_config).read_text(encoding="utf-8")) or {}
    review_dir = Path(args.review_output)
    candidates = esc.load_candidates(config)
    frame = esc.prepare(candidates)
    units = unit_choices(frame)
    by_dataset = units.groupby("dataset")[[RULE, VALIDATION_BEST, VALIDATION_BEST_DEFAULT]].mean()

    means = pd.read_csv(review_dir / "dataset_mean_scores.csv")
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    gaps = {
        name: float((by_dataset[name] - regret[name].reindex(by_dataset.index)).abs().max())
        for name in [VALIDATION_BEST, VALIDATION_BEST_DEFAULT]
    }
    if max(gaps.values()) > 1e-9:
        raise ValueError(f"The rule's code does not reproduce the review's rows: {gaps}")

    bootstrap = dict(review_config.get("bootstrap", {}))
    resampler = ArtifactResampler(
        regret.index, seed=int(bootstrap.get("seed", 20260608)), n_resamples=int(bootstrap.get("resamples", 10_000))
    )
    margins = [float(value) for value in review_config.get("margins", [0.005, 0.010, 0.015, 0.020])]
    rule = by_dataset[RULE].reindex(regret.index)

    scores = pd.read_parquet(review_dir / "dataset_policy_scores.parquet")
    held = scores[["split_id", "dataset"]].drop_duplicates()
    rule_scores = held.assign(method=RULE, mean_normalized_rank_regret=held["dataset"].map(by_dataset[RULE]))
    split = split_means(pd.concat([scores, rule_scores], ignore_index=True))
    nested = pd.read_csv(Path(args.nested_output) / "split_results.csv")
    n_train = int(nested["n_train_datasets"].iloc[0])
    n_test = int(nested["n_heldout_datasets"].iloc[0])

    rows = []
    for right in REFERENCES:
        if right not in regret.columns:
            continue
        result = paired_contrast(
            rule, regret[right], resampler, margins=margins, rank_scale=float(review_config["rank_scale"])
        )
        corrected = corrected_test(
            (split[RULE] - split[right]).dropna().to_numpy(dtype=float),
            n_train=n_train,
            n_test=n_test,
            margins=margins,
            n_datasets=int(regret.shape[0]),
        )
        rows.append(
            {
                "family": "provenance_reference",
                "left": RULE,
                "right": right,
                "left_mean": float(rule.mean()),
                "right_mean": float(regret[right].mean()),
                **result,
                **corrected,
                "nb_se_ratio": corrected["nb_se"] / float(result["se_bootstrap"]),
            }
        )
    contrasts = contrast_table(rows)
    contrasts["nb_holm_p"] = holm(contrasts["nb_p"].tolist())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_dataset.to_csv(output_dir / "dataset_scores.csv")
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    # The endpoint code must reproduce validation-best's error regret and runtime from the review.
    vb_extra = endpoints(units.assign(deployed_method=units["validation_best_method"]), candidates)
    vb_means = vb_extra.groupby("dataset")[["maxmin_error_regret", "total_seconds"]].mean()
    review_means = means[means["method"].eq(VALIDATION_BEST)].set_index("dataset")
    endpoint_gap = {
        "maxmin_error_regret": float((vb_means["maxmin_error_regret"] - review_means["mean_maxmin_error_regret"].reindex(vb_means.index)).abs().max()),
        "total_seconds_relative": float(((vb_means["total_seconds"] - review_means["mean_total_seconds"].reindex(vb_means.index)).abs() / review_means["mean_total_seconds"].reindex(vb_means.index)).max()),
    }
    if max(endpoint_gap.values()) > 1e-6:
        raise ValueError(f"The endpoint code does not reproduce the review's validation-best: {endpoint_gap}")
    summary_row(units, endpoints(units, candidates), regret, resampler).to_csv(output_dir / "summary.csv", index=False)
    deployed = units["deployed_variant"].value_counts(normalize=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": str(args.benchmark_config),
                "review_output": str(review_dir),
                "review_config": str(args.review_config),
                "nested_output": str(args.nested_output),
                "reproduction_max_gap": gaps,
                "endpoint_reproduction_gap": endpoint_gap,
                "n_units": int(units.shape[0]),
                "n_datasets": int(by_dataset.shape[0]),
                "deployed_variant_share": {str(k): float(v) for k, v in deployed.items()},
                "bootstrap_seed": int(bootstrap.get("seed", 20260608)),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cols = ["right", "left_mean", "right_mean", "mean_difference", "ci_low", "ci_high", "bootstrap_holm_p", "ttest_holm_p", "nb_holm_p"]
    print(contrasts[cols].to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--nested-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
