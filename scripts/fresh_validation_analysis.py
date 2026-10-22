"""Analyse the fresh-validation experiment as docs/fresh_validation_protocol.md prespecifies.

Reads the run's frozen candidates (``candidates.parquet``), applies the prespecified rules to
every unit (one task and repetition), ranks each choice against the unit's candidates by the
audit's own rule, averages the units of a task, and tests the two primary contrasts:

* C1, validation-best on the reused construction-validation scores minus validation-best on
  the fresh selection-validation scores;
* C2, reused validation-best minus the defaults rule.

Every analysis the protocol lists is written, whatever its result; ``decisions.json`` states
each primary contrast's test outcome and supported conclusion under the protocol's rules.
"""

from __future__ import annotations

import argparse
import json
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

from tabular_sota_benchmark import fresh_validation as fv  # noqa: E402
from tabular_sota_benchmark.inference import ArtifactResampler, contrast_table, paired_contrast  # noqa: E402
from tabular_sota_benchmark.ranking import rank_against_candidates  # noqa: E402

UNIT = ["dataset", "repetition"]
REUSED, FRESH = "validation_best_reused", "validation_best_fresh"
DEFAULTS_RULE, VB_DEFAULTS, ORACLE = "defaults_rule", "validation_best_defaults", "oracle"
RULES = [REUSED, FRESH, DEFAULTS_RULE, VB_DEFAULTS, ORACLE]
PRIMARY = [("C1", REUSED, FRESH), ("C2", REUSED, DEFAULTS_RULE)]
SECONDARY = [("S1", FRESH, DEFAULTS_RULE), ("S2", VB_DEFAULTS, REUSED)]
ENDPOINTS = {
    "normalized_rank_regret": "rank",
    "median_error_regret": "median-normalized error regret",
    "log_loss_regret": "test log loss above the unit's best candidate",
}
ALPHA = 0.05


def unit_endpoints(candidates: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    """Each rule's choice ranked against its unit's candidates by test log loss."""
    frame = pd.concat(
        [
            candidates.assign(method=candidates["candidate"], method_type="candidate"),
            choices.assign(method=choices["rule"], method_type="selector"),
        ],
        ignore_index=True,
    )
    ranked = rank_against_candidates(frame, unit_columns=UNIT, value_column="test_loss", rule="consistent")
    rows = ranked[ranked["method_type"].eq("selector")].copy()
    best = rows["candidate_min_error"].astype(float)
    middle = rows["candidate_median_error"].astype(float) - best
    rows["log_loss_regret"] = rows["test_loss"].astype(float) - best
    rows["median_error_regret"] = np.where(middle > 0, rows["log_loss_regret"] / middle.where(middle > 0, 1.0), np.nan)
    return rows[[*UNIT, "rule", "candidate", "family", "variant", "test_loss", "candidate_rank", "n_candidates",
                 "normalized_rank_regret", "median_error_regret", "log_loss_regret"]].reset_index(drop=True)


def decide(row: pd.Series, margin: float | None) -> dict[str, object]:
    """The protocol's supported conclusion for a prespecified contrast between rules that fit nothing."""
    boot, t = bool(row["bootstrap_holm_p"] < ALPHA), bool(row["ttest_holm_p"] < ALPHA)
    if boot and t:
        conclusion = "difference supported"
    elif not boot and not t:
        conclusion = "no difference detected"
    else:
        conclusion = "difference unresolved (the tests disagree)"
    out: dict[str, object] = {"bootstrap_rejects": boot, "ttest_rejects": t, "conclusion": conclusion}
    if margin is not None:
        inside = bool(abs(float(row["tost_90_ci_low"])) < margin and abs(float(row["tost_90_ci_high"])) < margin)
        out["equivalence_margin"] = margin
        out["equivalence_supported"] = bool(conclusion == "no difference detected" and inside)
    return out


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    candidates = pd.read_parquet(run_dir / "candidates.parquet")
    families = list(config["families"])
    order = fv.candidate_order(families)
    candidates["order"] = candidates["candidate"].map(order)
    if candidates["order"].isna().any():
        raise ValueError("Candidates outside the configured families")
    analysis = dict(config.get("analysis", {}))
    n_candidates = len(order)
    rank_scale = float(n_candidates - 1)
    margin = float(analysis.get("equivalence_rank_positions", 0.5)) / rank_scale

    choices = fv.rule_choices(candidates, UNIT)
    units = unit_endpoints(candidates, choices)
    complete = candidates.groupby(UNIT)["candidate"].nunique()
    units = units.merge(complete.rename("n_candidates_unit").reset_index(), on=UNIT)
    datasets = units.groupby(["dataset", "rule"])[list(ENDPOINTS)].mean().reset_index()

    resampler = ArtifactResampler(
        sorted(datasets["dataset"].unique()),
        seed=int(analysis.get("bootstrap_seed", 20260608)),
        n_resamples=int(analysis.get("bootstrap_resamples", 10_000)),
    )
    rows = []
    for endpoint in ENDPOINTS:
        pivot = datasets.pivot(index="dataset", columns="rule", values=endpoint)
        for family, contrasts in [("primary", PRIMARY), ("secondary", SECONDARY)]:
            for name, left, right in contrasts:
                result = paired_contrast(
                    pivot[left], pivot[right], resampler, margins=[margin],
                    rank_scale=rank_scale if endpoint == "normalized_rank_regret" else None,
                )
                rows.append({"family": f"{family}:{endpoint}", "contrast": name, "left": left, "right": right,
                             "endpoint": endpoint, "left_mean": float(pivot[left].mean()),
                             "right_mean": float(pivot[right].mean()), **result})
    contrasts = contrast_table(rows)

    decisions = {}
    for name, left, right in PRIMARY:
        row = contrasts[contrasts["family"].eq("primary:normalized_rank_regret") & contrasts["contrast"].eq(name)].iloc[0]
        decisions[name] = {
            "left": left,
            "right": right,
            "mean_difference": float(row["mean_difference"]),
            "ci_95": [float(row["ci_low"]), float(row["ci_high"])],
            "bootstrap_holm_p": float(row["bootstrap_holm_p"]),
            "ttest_holm_p": float(row["ttest_holm_p"]),
            **decide(row, margin if name == "C1" else None),
        }

    # Offsets by class: test minus construction (reused) and test minus selection (fresh) log loss.
    offsets = candidates.assign(
        reused_offset=candidates["test_loss"] - candidates["construction_loss"],
        fresh_offset=candidates["test_loss"] - candidates["selection_loss"],
    )
    by_dataset = offsets.groupby(["dataset", "variant"])[["reused_offset", "fresh_offset"]].mean().reset_index()
    offset_rows = []
    for column in ["reused_offset", "fresh_offset"]:
        pivot = by_dataset.pivot(index="dataset", columns="variant", values=column)
        for variant in fv.VARIANTS:
            estimate, low, high = resampler.mean_ci(pivot[variant])
            offset_rows.append({"offset": column, "variant": variant, "mean": estimate, "ci_low": low, "ci_high": high})
        for upper, lower in [("tuned + ensemble", "default"), ("tuned", "default"), ("tuned + ensemble", "tuned")]:
            difference = pivot[upper] - pivot[lower]
            above = int((difference > 0).sum())
            below = int((difference < 0).sum())
            estimate, low, high = resampler.mean_ci(difference)
            offset_rows.append({
                "offset": column, "variant": f"{upper} minus {lower}", "mean": estimate, "ci_low": low, "ci_high": high,
                "datasets_above": above, "datasets_below": below,
                "sign_test_p_unadjusted": float(stats.binomtest(above, above + below, 0.5).pvalue) if above + below else 1.0,
            })
    offset_table = pd.DataFrame(offset_rows)

    shares = (
        units.groupby(["rule", "variant"]).size().rename("units").reset_index()
        .assign(share=lambda frame: frame["units"] / frame.groupby("rule")["units"].transform("sum"))
    )
    agree = units.pivot_table(index=UNIT, columns="rule", values="candidate", aggfunc="first")
    same_choice = float((agree[REUSED] == agree[FRESH]).mean())

    # By task size (rows after the cap): the halves below and above the median, descriptive.
    sizes = candidates.groupby("dataset")["n_rows"].first()
    median_rows = float(sizes.median())
    size_rows = []
    for half, members in [("small", sizes[sizes <= median_rows].index), ("large", sizes[sizes > median_rows].index)]:
        subset = resampler.subset(members, salt=f"size:{half}")
        pivot = datasets[datasets["dataset"].isin(members)].pivot(index="dataset", columns="rule", values="normalized_rank_regret")
        for name, left, right in PRIMARY:
            estimate, low, high = subset.mean_ci(pivot[left] - pivot[right])
            size_rows.append({"half": half, "n_datasets": int(len(members)), "contrast": name, "mean_difference": estimate,
                              "ci_low": low, "ci_high": high})
    size_table = pd.DataFrame(size_rows)

    summary = datasets.groupby("rule")[list(ENDPOINTS)].mean()
    regret_ci = {rule: resampler.mean_ci(datasets[datasets["rule"].eq(rule)].set_index("dataset")["normalized_rank_regret"]) for rule in RULES}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_dir / "unit_choices.csv", index=False)
    datasets.to_csv(output_dir / "dataset_scores.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    offset_table.to_csv(output_dir / "offsets.csv", index=False)
    shares.to_csv(output_dir / "selection_shares.csv", index=False)
    size_table.to_csv(output_dir / "by_size.csv", index=False)
    summary.to_csv(output_dir / "rule_summary.csv")
    (output_dir / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": decisions,
                "n_datasets": int(resampler.n),
                "n_units": int(units[UNIT].drop_duplicates().shape[0]),
                "n_candidates": n_candidates,
                "units_with_fewer_candidates": int((complete < n_candidates).sum()),
                "rank_regret": {rule: list(map(float, regret_ci[rule])) for rule in RULES},
                "reused_and_fresh_choose_the_same_candidate": same_choice,
                "median_rows": median_rows,
                "run_metadata": json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")),
                "command": sys.argv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decisions, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/fresh_validation_experiment.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
