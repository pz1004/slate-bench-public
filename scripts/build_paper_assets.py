"""Generate every number, table and label the manuscript prints from the v2 outputs.

Writes, under ``paper/generated/`` (the manuscript folder, which is not part of the public
repository):

* ``numbers.tex``: ``\\defres{key}{value}`` for every number quoted in the prose, read
  in the manuscript as ``\\res{key}`` (an undefined key is a LaTeX error);
* ``labels.tex``: one ``\\pol...`` macro per policy from ``labels.py``;
* ``tables/*.tex``: complete ``tabular`` environments (captions and labels stay in
  ``main.tex``).

Nothing in the result sections of ``main.tex`` is typed by hand, so a rerun of the
pipeline followed by this script updates the paper consistently. With
``--allow-missing`` absent inputs produce ``??`` instead of an error (drafting only).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

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

from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p, holm  # noqa: E402
from tabular_sota_benchmark.labels import latex_macros, policy_label  # noqa: E402
from tabular_sota_benchmark.provenance import top_tenth  # noqa: E402
from tabular_sota_benchmark.meta_study import (  # noqa: E402
    COUNT_CHECK_FILE,
    COUNT_CHECK_ITEMS as META_COUNT_CHECK_ITEMS,
    EXTRA_COLUMNS as META_EXTRA_COLUMNS,
    FIELD_COLUMNS as META_FIELD_COLUMNS,
    FIELDS as META_FIELDS,
    ITEMS as META_ITEMS,
    RECHECK_CODING_FILE,
    check_count_check_state,
    check_recheck_state,
    cohen_kappa,
    item_key as meta_item_key,
    load_declaration,
    read_claims,
    recheck_final,
    resolved_codes,
)

# Short keys used in \res{...}; the manuscript never spells a policy id.
POLICY_KEYS = {
    "nested_policy_selected": "nested",
    "meta_selector_family_prior_budgeted": "fp",
    "meta_selector_family_prior_budgeted_100s": "fphundred",
    "meta_selector_family_prior_budgeted_1000s": "fpthousand",
    "meta_selector_family_prior_budgeted_5000s": "fpfivethousand",
    "train_selected_fixed_reference__declared_fixed": "tsf",
    "train_selected_fixed_reference__validation_only": "tsfv",
    "train_selected_fixed_reference__family_prior": "tsff",
    "train_selected_fixed_reference__with_calibrated": "tsfc",
    "train_selected_fixed_reference__broader": "tsfb",
    "train_selected_fixed_reference__score_only": "tsfs",
    "train_selected_fixed_reference__ensemble_inclusive": "tsfe",
    "meta_selector_single": "vb",
    "meta_selector_ablate_validation_only": "vr",
    "meta_selector_task_adaptive_budgeted": "tad",
    "meta_selector_meta_feature_guard_002": "gtwo",
    "meta_selector_meta_feature_guard_005": "gfive",
    "meta_selector_calibrated_budget_10000s": "cal",
    "oracle": "oracle",
    "reference_single_best_global": "sbs",
    "reference_single_best_task": "sbst",
    "reference_priors_only": "prior",
    "reference_validation_best_default": "vbdef",
    "reference_validation_best_tuned": "vbtun",
    "reference_validation_best_tuned_ensemble": "vbte",
    "reference_optimism_corrected": "opt",
    "reference_defaults_then_tuned": "dtt",
    "reference_optimism_corrected_budget": "optb",
    "reference_optimism_corrected_task": "optt",
    "reference_metafeature_knn": "knn",
    "reference_metafeature_knn_validation": "knnv",
    "extended_metafeature_knn": "eknn",
    "extended_metafeature_knn_validation": "eknnv",
    "extended_descriptor_guard_002": "edgtwo",
    "extended_descriptor_guard_005": "edgfive",
    "meta_selector_tabrepo_ensemble_k3": "ethree",
    "meta_selector_tabrepo_ensemble_k5": "efive",
    "meta_selector_tabrepo_ensemble_k10": "eten",
}


class Missing(Exception):
    pass


class Assets:
    def __init__(self, out: Path, *, allow_missing: bool, meta_verified_only: bool = False) -> None:
        self.out = out
        self.allow_missing = allow_missing
        self.meta_verified_only = meta_verified_only
        self.numbers: dict[str, str] = {}
        self.tables: dict[str, str] = {}
        self.missing: list[str] = []
        # Computed verdicts of every contrast, by result-key base (see ``contrast_verdicts``).
        self.verdicts: dict[str, dict[str, str]] = {}

    def csv(self, relative: str) -> pd.DataFrame:
        path = self.out / relative
        if not path.exists():
            raise Missing(str(path))
        return pd.read_csv(path)

    def define(self, key: str, value: str) -> None:
        if key in self.numbers and self.numbers[key] != value:
            raise ValueError(f"Result key {key} defined twice with different values")
        self.numbers[key] = value

    def define_p(self, key: str, value: str) -> None:
        """A p-value key plus ``key``+``eq``, the full relation for running text."""
        self.define(key, value)
        relation = "p<0.001" if value == "$<$0.001" else f"p={value}"
        self.define(f"{key}eq", f"${relation}$")

    def section(self, name: str, build: Callable[[], None]) -> None:
        try:
            build()
        except Missing as error:
            if not self.allow_missing:
                raise
            self.missing.append(f"{name}: {error}")


# ----------------------------------------------------------------------------- format


def num(value: float, digits: int = 3, *, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    if text.startswith("-") or text.startswith("+"):
        sign, body = text[0], text[1:]
        if float(body) == 0.0:
            return f"{0:.{digits}f}"
        return ("$-$" if sign == "-" else "$+$") + body
    return text


def interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{num(low, digits)}, {num(high, digits)}]"


def pvalue(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value < 0.001:
        return "$<$0.001"
    text = f"{value:.3f}"
    # A value that rounds to the 0.05 threshold gets a fourth decimal, so the printed
    # number shows on which side of it the test falls.
    if text == f"{ALPHA:.3f}" and value != ALPHA:
        return f"{value:.4f}"
    return text


def seconds(value: float) -> str:
    # Thousands are grouped with a thin space, as in the running text (10\,000).
    return f"{value:,.0f}".replace(",", "\\,")


def percent(value: float, digits: int = 0) -> str:
    text = f"{100 * value:.{digits}f}"
    if text.startswith("-"):
        body = text[1:]
        return f"{body}\\%" if float(body) == 0.0 else f"$-${body}\\%"
    return f"{text}\\%"


def latex_escape(text: str) -> str:
    for character in "&%#_":
        text = text.replace(character, "\\" + character)
    return text


def wtl(row: pd.Series) -> str:
    return f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"


def tabular(columns: str, header: list[str], rows: list[list[str]], *, groups: dict[int, str] | None = None) -> str:
    lines = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    width = len(header)
    for index, row in enumerate(rows):
        if groups and index in groups:
            if index:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{groups[index]}}}}} \\\\")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- blocks


def summary_numbers(assets: Assets, prefix: str, summary: pd.DataFrame, *, rank_scale: float) -> None:
    for _, row in summary.iterrows():
        key = POLICY_KEYS.get(str(row["method"]))
        if key is None:
            continue
        base = f"{prefix}.{key}"
        assets.define(f"{base}.reg", num(row["mean_normalized_rank_regret"]))
        assets.define(f"{base}.regfour", num(row["mean_normalized_rank_regret"], 4))
        assets.define(
            f"{base}.regci",
            interval(row["mean_normalized_rank_regret_ci_low"], row["mean_normalized_rank_regret_ci_high"]),
        )
        assets.define(f"{base}.err", num(row["mean_maxmin_error_regret"]))
        if "mean_median_error_regret" in row and math.isfinite(row["mean_median_error_regret"]):
            assets.define(f"{base}.merr", num(row["mean_median_error_regret"]))
        assets.define(f"{base}.beats", percent(row["beats_best_rate"], 1))
        assets.define(f"{base}.rt", seconds(row["mean_total_seconds"]))
        assets.define(f"{base}.rank", num(1 + row["mean_normalized_rank_regret"] * rank_scale, 1))
        if "gap_closed" in row and math.isfinite(row.get("gap_closed", float("nan"))):
            assets.define(f"{base}.gap", num(row["gap_closed"], 2))
            assets.define(f"{base}.gappct", percent(row["gap_closed"]))
            assets.define(f"{base}.gapci", interval(row["gap_closed_ci_low"], row["gap_closed_ci_high"], 2))


ALPHA = 0.05

# The verdict the manuscript states for a contrast, by result-key base and target. The build
# fails when a computed verdict differs, so that a rerun cannot leave a sentence behind its
# numbers. Targets: "interval" (the 95% percentile interval excludes or includes zero),
# "mean" (Holm-adjusted mean test, the separation verdict), "t" (Holm-adjusted paired
# t-test), "rank" (Holm-adjusted signed-rank test) and "equiv0p010" etc. (TOST at a margin).
STATED_VERDICTS: dict[str, dict[str, str]] = {
    # TabArena (Sections 6.1-6.3, Table 2, the Conclusion).
    # Only the bootstrap test separates the headline: the paired t-test and the corrected test,
    # which targets the same dataset-level mean, do not (Sections 6.1 and 3.4).
    "ta.primary.nested.vb": {"interval": "excludes zero", "mean": "better", "t": "not separated", "rank": "not separated", "corrected": "not separated"},
    # The equivalences hold under the bootstrap only; the corrected test withdraws them (Section 6.3).
    "ta.primary.nested.fp": {"mean": "not separated", "equiv0p015": "equivalent", "equiv0p010": "not equivalent", "corrected": "not separated", "corrected_equiv0p015": "not equivalent", "corrected_equiv0p010": "not equivalent"},
    "ta.primary.nested.tsf": {"mean": "not separated", "equiv0p010": "equivalent", "corrected": "not separated", "corrected_equiv0p010": "not equivalent", "corrected_equiv0p015": "not equivalent"},
    # The policy pool (Table 2, Section 6.4): the 8-row grid and the 49-row extended pool.
    "ta.shrink.primary.nested.vb": {"mean": "better", "t": "better", "rank": "not separated", "corrected": "better"},
    "ta.shrink.primary.nested.fp": {"mean": "not separated"},
    "ta.ext.primary.nested.vb": {"mean": "better", "t": "not separated", "corrected": "not separated"},
    "ta.ext.primary.nested.fp": {"mean": "not separated", "equiv0p015": "not equivalent"},
    "ta.ext.primary.nested.tsf": {"mean": "not separated", "equiv0p015": "not equivalent"},
    # Implementation conventions (Appendix J): legacy ranks with in-sample weights (vone),
    # consistent ranks with in-sample weights (rankfix), the shared inner fit (inner), and
    # the reported conventions (vtwo).
    "ta.revision_references.nested.vbdef": {"interval": "includes zero", "mean": "not separated"},
    "attr.vone.vb": {"mean": "not separated"},
    "attr.rankfix.vb": {"mean": "not separated"},
    "attr.inner.vb": {"mean": "better"},
    "attr.vtwo.vb": {"mean": "better"},
    "attr.vone.fp": {"mean": "not separated", "equiv0p015": "not equivalent"},
    "attr.rankfix.fp": {"mean": "not separated", "equiv0p015": "not equivalent"},
    "attr.inner.fp": {"mean": "not separated", "equiv0p015": "not equivalent"},
    "attr.vtwo.fp": {"mean": "not separated", "equiv0p015": "equivalent"},
    "attr.vone.tsf": {"mean": "not separated", "equiv0p010": "not equivalent"},
    "attr.rankfix.tsf": {"mean": "not separated", "equiv0p010": "not equivalent"},
    "attr.inner.tsf": {"mean": "not separated", "equiv0p010": "not equivalent"},
    "attr.vtwo.tsf": {"mean": "not separated", "equiv0p010": "equivalent"},
    "ta.revision_references.nested.sbs": {"interval": "includes zero", "mean": "not separated"},
    "ta.revision_references.nested.opt": {"mean": "not separated"},
    "ta.revision_references.nested.optt": {"mean": "not separated"},
    "ta.provenance.vb.sbs": {"interval": "includes zero", "mean": "not separated"},
    "ta.provenance.vb.vbdef": {"interval": "excludes zero", "mean": "not separated"},
    "ta.provenance.vb.opt": {"mean": "worse", "t": "worse", "rank": "worse"},
    "ta.mechanism.fp.opt": {"mean": "not separated"},
    "ta.mechanism.fp.optb": {"mean": "not separated"},
    # TabRepo (Section 7, Table 2, the Limitations): the ensemble panel (tr) and the
    # score-only panel (trs).
    "tr.primary.nested.tsfe": {"mean": "not separated", "rank": "not separated"},
    "tr.capability.tsfe.tsfs": {"interval": "excludes zero", "mean": "better", "rank": "better"},
    "tr.fixed_ensembles.nested.ethree": {"interval": "excludes zero", "mean": "worse", "t": "worse", "rank": "not separated", "corrected": "not separated"},
    "tr.mechanism.vb.opt": {"interval": "includes zero", "mean": "not separated", "t": "not separated", "rank": "better"},
    "tr.mechanism.vb.fp": {"interval": "includes zero", "mean": "not separated"},
    "tr.mechanism.vb.sbs": {"mean": "better"},
    "trs.primary.nested.vb": {"interval": "includes zero", "mean": "not separated", "rank": "not separated"},
    "trs.primary.nested.tsfs": {"interval": "includes zero", "mean": "not separated", "rank": "not separated"},
    "trs.score_level_rows.nested.tad": {"interval": "excludes zero", "mean": "worse", "rank": "not separated", "corrected": "not separated"},
    "trs.score_level_rows.nested.gfive": {"interval": "excludes zero", "mean": "worse", "rank": "not separated", "corrected": "not separated"},
    "trs.score_level_rows.nested.vr": {"mean": "not separated"},
    "tr.primary.nested.vb": {"interval": "excludes zero", "mean": "better"},
    "trs.score_level_rows.nested.fp": {"mean": "not separated"},
    # Evidence units (Section 6.3, Section 7): what the splits, the unit appearances and the
    # datasets would conclude, from nested selection's side.
    "ta.eu.fp.split": {"wilcoxon": "worse"},
    "ta.eu.tsf.split": {"wilcoxon": "worse"},
    "ta.eu.vb.unit": {"wilcoxon_any": "separated"},
    "ta.eu.fp.unit": {"wilcoxon_any": "separated", "clustered": "not separated"},
    "ta.eu.tsf.unit": {"wilcoxon_any": "separated", "clustered": "not separated"},
    "ta.eu.fp.dataset": {"wilcoxon": "not separated"},
    "ta.eu.tsf.dataset": {"wilcoxon": "not separated"},
    "trs.eu.tsfs.split": {"wilcoxon": "better"},
    "trs.eu.tsfs.dataset": {"wilcoxon": "not separated"},
    "trs.eu.vb.split": {"wilcoxon": "not separated"},
    "trs.eu.vb.dataset": {"wilcoxon": "not separated"},
    # The inner fit on TabRepo's score-only panel (Appendix J, Section 7).
    "attr.tr.inner.tsfs": {"interval": "excludes zero", "mean": "better"},
    "attr.tr.vtwo.tsfs": {"interval": "includes zero", "mean": "not separated"},
    "attr.tr.inner.vb": {"mean": "not separated"},
    "attr.tr.vtwo.vb": {"mean": "not separated"},
    # Validation-best within the tuned classes (Section 6.2).
    "ta.revision_references.nested.vbtun": {"mean": "better", "t": "not separated", "corrected": "not separated"},
    "ta.revision_references.nested.vbte": {"mean": "better", "t": "not separated", "corrected": "not separated"},
    # The main audit rerun without TabArena's copied rows (Section 6.1, Appendix G).
    "imp.primary.nested.vb": {"interval": "excludes zero", "mean": "not separated", "corrected": "not separated"},
    "imp.mechanism.fp.vb": {"mean": "better", "t": "not separated", "corrected": "not separated"},
    # The main run's own choices, scored without the imputed rows, keep the separation.
    "imp.rescored_primary.nested.vb": {"mean": "better"},
    "imp.revision_references.nested.sbs": {"mean": "not separated"},
    "imp.provenance_reference.dtt.vb": {"mean": "better", "t": "better", "corrected": "better"},
    "imp.provenance_reference.dtt.nested": {"mean": "not separated"},
    # Section 6.2: selecting on defaults and deploying tuned + ensemble (added after the main results).
    "ta.provenance_reference.dtt.vb": {"mean": "better", "t": "better", "corrected": "better"},
    "ta.provenance_reference.dtt.vbdef": {"mean": "not separated"},
    "ta.provenance_reference.dtt.nested": {"mean": "not separated", "corrected": "not separated", "equiv0p020": "not equivalent"},
    "ta.provenance_reference.dtt.fp": {"mean": "not separated"},
    "ta.provenance_reference.dtt.tsf": {"mean": "not separated"},
    "ta.provenance_reference.dtt.opt": {"mean": "not separated"},
    "imp.provenance.vb.opt": {"mean": "worse"},
    # Third review: reapplied and documented-mask stages, the defaults rule's non-equivalence,
    # the adaptive rows' contrast and TabRepo's grouped splits (Sections 6.1, 6.2, 7).
    "imp.reapplied_primary.nested.vb": {"interval": "excludes zero", "mean": "better"},
    "flag.primary.nested.vb": {"interval": "excludes zero", "mean": "not separated"},
    "flag.rescored_primary.nested.vb": {"mean": "better"},
    "flag.reapplied_primary.nested.vb": {"mean": "better"},
    # Also no corrected equivalence within the largest margin (Section 6.3): the smallest
    # corrected margin only just exceeds 0.020.
    "ta.comparator_pools.nested.tsfc": {"interval": "includes zero", "mean": "not separated", "t": "not separated", "corrected": "not separated", "corrected_equiv0p020": "not equivalent"},
    "trg.primary.nested.vb": {"interval": "excludes zero", "mean": "better"},
    "trgs.primary.nested.vb": {"interval": "includes zero", "mean": "not separated"},
    "trgs.primary.nested.tsfs": {"interval": "excludes zero", "mean": "better"},
    "trgs.score_level_rows.nested.tad": {"mean": "not separated"},
    "trgs.score_level_rows.nested.gfive": {"mean": "not separated"},
    # The deployable error-space correction (Sections 6.2 and 7, Appendix G).
    "ta.es.vb": {"mean": "not separated"},
    "ta.esm.vb": {"mean": "not separated"},
    "ta.es.opt": {"mean": "worse"},
    "ta.esm.opt": {"mean": "worse"},
    "tr.es.vb": {"mean": "worse"},
    "tr.esf.vb": {"mean": "worse"},
    "tr.esm.vb": {"mean": "worse"},
    "tr.esmf.vb": {"mean": "worse"},
}


def contrast_verdicts(row: pd.Series) -> dict[str, str]:
    """Verdicts of one contrast row of ``left - right`` (lower is better) under each target."""

    def separated(p_column: str, direction: float) -> str | None:
        if p_column not in row or pd.isna(row[p_column]):
            return None
        if float(row[p_column]) >= ALPHA:
            return "not separated"
        return "better" if direction < 0 else "worse"

    verdicts = {
        "interval": "excludes zero" if row["ci_low"] > 0 or row["ci_high"] < 0 else "includes zero",
    }
    location = row.get("hodges_lehmann_difference", row["mean_difference"])
    for target, column, direction in [
        ("mean", "bootstrap_holm_p", row["mean_difference"]),
        ("t", "ttest_holm_p", row["mean_difference"]),
        ("rank", "wilcoxon_holm_p", location),
    ]:
        verdict = separated(column, float(direction))
        if verdict is not None:
            verdicts[target] = verdict
    for margin in ["0p005", "0p010", "0p015", "0p020"]:
        column = f"tost_p_{margin}"
        if column in row and pd.notna(row[column]):
            verdicts[f"equiv{margin}"] = "equivalent" if float(row[column]) < ALPHA else "not equivalent"
    return verdicts


# Orderings and sizes the manuscript states in words, as (description, keys, test). The
# values are the numbers as printed.
STATED_COMPARISONS: list[tuple[str, tuple[str, ...], Callable[..., bool]]] = [
    # Section 6.3: the refitting diagnostic is slightly larger than the conditional standard
    # error for nested - VB and several times larger for the committed rules, whose rows are
    # far more correlated with nested selection than validation-best's.
    ("jackknife factor for nested - VB is slightly above one", ("jk.ta.nested.vb.ratio",), lambda r: 1.0 <= r < 1.5),
    ("jackknife factors for the committed rules are about three and four", ("jk.ta.nested.fp.ratio", "jk.ta.nested.tsf.ratio"), lambda a, b: round(a) == 3 and round(b) == 4),
    ("at the jackknife SE the unadjusted normal test of nested - VB rejects", ("jk.ta.nested.vb.pnorm",), lambda p: p < 0.05),
    ("at the jackknife SE the Holm-adjusted test of nested - VB does not reject", ("jk.ta.nested.vb.pholm",), lambda q: q >= 0.05),
    ("the family-prior half-width is about one rank position", ("jk.ta.nested.fp.hwrank",), lambda r: 0.75 <= r < 1.5),
    ("TabRepo: neither fit's difference from TSF would separate at the jackknife SE", ("attr.tr.inner.tsfs.diff", "attr.tr.vtwo.tsfs.diff", "jk.tr.nested.tsf.se"), lambda d1, d2, se: max(abs(d1), abs(d2)) / se < 1.96),
    # Section 7 and Appendix G: statements about intervals and relative sizes.
    ("TabRepo: the ensembles' relative lead is smaller in error regret", ("tr.ethree.err.rel", "tr.ethree.reg.rel"), lambda e, r: e < r),
    ("TabRepo: both Shapley intervals of the priors include zero", ("tr.shap.tf.ci", "tr.shap.gm.ci"), lambda a, b: a[0] < 0 < a[1] and b[0] < 0 < b[1]),
    ("TabRepo: the priors' Shapley point estimates are negative", ("tr.shap.tf", "tr.shap.gm"), lambda a, b: a < 0 and b < 0),
    ("TabRepo: the full rule has higher mean regret than validation alone", ("tr.cell.full", "tr.cell.valonly"), lambda a, b: a > b),
    ("TabRepo: the correction's mean interval admits only a small gain", ("tr.mechanism.vb.opt.ci",), lambda ci: ci[1] < 0.001),
    ("TabArena: the median-normalized interval of tuned candidates includes zero", ("ta.erroptim.tun.medci",), lambda ci: ci[0] < 0 < ci[1]),
    ("committed rules correlate more with nested selection than VB", ("ta.nested.fp.corr", "ta.nested.tsf.corr", "ta.nested.vb.corr"), lambda a, b, c: min(a, b) > c),
    # Appendix J: the shared fit's effect on nested selection follows the order of
    # task-adaptive and family-prior on held-out data, in opposite directions on the artifacts.
    ("TabRepo: task-adaptive below family-prior", ("trs.tad.regfour", "trs.fp.regfour"), lambda a, b: a < b),
    ("TabRepo: shared fit leaves nested selection lower", ("attr.tr.inner.nested.reg", "attr.tr.vtwo.nested.reg"), lambda a, b: a < b),
    ("TabArena: task-adaptive above family-prior", ("ta.tad.regfour", "ta.fp.regfour"), lambda a, b: a > b),
    ("TabArena: shared fit leaves nested selection higher", ("attr.inner.nested.reg", "ta.nested.regfour"), lambda a, b: a > b),
    ("both artifacts: the shared fit picks adaptive rows more often", ("attr.inner.adaptivechosen", "attr.vtwo.adaptivechosen", "attr.tr.inner.adaptivechosen", "attr.tr.vtwo.adaptivechosen"), lambda a, b, c, d: a > b and c > d),
    ("App. J: the guards' gap relative to FP falls with the per-target fit", ("attr.inner.gap.shared.rel", "attr.inner.gap.pertarget.rel", "attr.tr.inner.gap.shared.rel", "attr.tr.inner.gap.pertarget.rel"), lambda a, b, c, d: a > b and c > d),
    # Section 7: a few datasets carry the mean loss against fixed ensemble-3.
    ("Section 7: five datasets carry most of the loss against ensemble-3", ("tr.nested.ethree.topfiveshare",), lambda s: s >= 50.0),
    # Abstract, Section 7, Discussion and Conclusion: with ensembles admitted, nested selection
    # chooses exactly what the ensemble-inclusive train-selected rule chooses.
    ("TabRepo: nested selection ties the ensemble-inclusive train-selected rule on every dataset", ("tr.primary.nested.tsfe.wins", "tr.primary.nested.tsfe.losses"), lambda w, l: w == 0 and l == 0),
    # Sections 5 and 7, Appendix G: what the spread-to-noise ratios show, and what they do not.
    ("TabRepo: the error-space ratio's interval includes one", ("tr.optim.snr.err.ci",), lambda ci: ci[0] < 1 < ci[1]),
    ("TabRepo: the rank-space ratio's interval lies above one", ("tr.optim.snr.ci",), lambda ci: ci[0] > 1),
    ("TabArena: both ratios' intervals lie above one", ("ta.optim.snr.err.ci", "ta.optim.snr.ci"), lambda a, b: a[0] > 1 and b[0] > 1),
    # Section 7: the TabRepo refitting diagnostic is several times the difference itself.
    ("TabRepo: the jackknife SE is several times the score-only nested - TSF difference", ("jk.tr.nested.tsf.se", "trs.primary.nested.tsfs.diff"), lambda se, d: se >= 3 * abs(d)),
    # Sections 1, 6.4 and 7 and the Conclusion: what the SPA test finds after the search.
    ("SPA: no pool policy beats the family-prior rule on TabArena", ("ta.spa.fp.c", "ta.spa.fp.rc"), lambda c, rc: c >= 0.05 and rc >= 0.05),
    ("SPA: some pool policy beats validation-best on TabArena", ("ta.spa.vb.c", "ta.spa.vb.rc"), lambda c, rc: c < 0.05 and rc < 0.05),
    ("SPA: no score-only pool policy beats validation-best on TabRepo", ("trs.spa.vb.c",), lambda c: c >= 0.05),
    ("SPA: no pool row beats fixed ensemble-3 on TabRepo", ("tr.spa.ethree.c",), lambda c: c >= 0.05),
    # Section 6.4: neither tie-break change moves held-out regret.
    ("tie-break changes leave held-out regret unchanged", ("ta.obj.noruntime.diff", "ta.obj.reverse.diff"), lambda a, b: a == 0 and b == 0),
    # Section 3.4 and Appendix A: where the two tests of the mean disagree, only the bootstrap
    # test separates, the differences are skewed toward large gains, and the tests that follow
    # the skew give the smaller p-values; none disagree on TabRepo.
    ("only the bootstrap test separates where the two mean tests disagree", ("ta.mtests.onlyt", "ta.mtests.differ.main"), lambda t, d: t == 0 and d > 0),
    ("the mean tests never disagree on TabRepo", ("tr.mtests.differ",), lambda d: d == 0),
    ("every disagreement is skewed and ordered by skew-handling", ("ta.mtests.skewordered", "ta.mtests.differ.main", "ta.mtests.differ.ext"), lambda o, m, e: o == m + e),
    ("headline: skew-following tests reject after Holm, symmetric ones do not", ("ta.mtests.vb.percholm", "ta.mtests.vb.bteqholm", "ta.mtests.vb.tholm", "ta.mtests.vb.btsymholm"), lambda a, b, c, d: max(a, b) < 0.05 <= min(c, d)),
    ("headline: the differences are skewed toward large gains", ("ta.mtests.vb.skew",), lambda k: k < 0),
    ("headline verdict holds on every other resample matrix", ("ta.mc.vb.below", "ta.mc.vb.seeds"), lambda b, n: b == n),
    # Section 6.2: in the regression stratum validation-best has the lowest mean regret.
    ("regression: validation-best lowest of the rules compared", ("ta.strata.regression.vb", "ta.strata.regression.nested", "ta.strata.regression.fp", "ta.strata.regression.sbs", "ta.strata.regression.tsf", "ta.strata.regression.opt", "ta.strata.regression.vbdef"), lambda v, *others: v <= min(others)),
    # Section 6.2: the strata do not isolate provenance. The error-space gap between
    # tuned-and-ensembled candidates and defaults is largest in binary tasks, where
    # validation-best is the worst rule and the correction the best; it is about as small in
    # multiclass tasks as in regression, yet the correction helps only in the first.
    ("strata: binary has the largest error-space gap", ("ta.erroptim.tegap.binary", "ta.erroptim.tegap.multiclass", "ta.erroptim.tegap.regression"), lambda b, m, r: b > 1.5 * max(m, r)),
    ("strata: the multiclass and regression gaps are about equal", ("ta.erroptim.tegap.binary", "ta.erroptim.tegap.multiclass", "ta.erroptim.tegap.regression"), lambda b, m, r: abs(m - r) < 0.25 * b),
    ("strata: in binary tasks validation-best is worst and the correction best", ("ta.strata.binary.vb", "ta.strata.binary.opt", "ta.strata.binary.nested", "ta.strata.binary.fp", "ta.strata.binary.sbs", "ta.strata.binary.tsf", "ta.strata.binary.vbdef"), lambda v, o, *others: v > max(o, *others) and o < min(v, *others)),
    ("strata: the correction helps in multiclass and not in regression", ("ta.strata.multiclass.opt", "ta.strata.multiclass.vb", "ta.strata.regression.opt", "ta.strata.regression.vb"), lambda mo, mv, ro, rv: mo < mv and ro > rv),
    # Section 6.2: the committed rules' share of tuned-and-ensembled picks is closer to the
    # oracle's than validation-best's is.
    ("shares: committed rules move toward the oracle's tuned + ensemble share", ("ta.share.fp.te", "ta.share.nested.te", "ta.share.oracle.te", "ta.share.vb.te"), lambda f, n, o, v: max(abs(f - o), abs(n - o)) < abs(v - o)),
    # Section 6.2: the budgeted correction tracks the family-prior rule across datasets.
    ("the budgeted correction tracks the family-prior rule", ("ta.optb.fp.corr",), lambda r: r >= 0.8),
    # Section 6.3: read at face value, the jackknife half-widths exceed the equivalence margins.
    ("at face value the jackknife half-widths exceed the FP and TSF equivalence margins", ("jk.ta.nested.fp.hw", "jk.ta.nested.tsf.hw"), lambda f, t: f > 0.015 and t > 0.010),
    # Discussion: the largest TabArena jackknife factor is the TSF contrast's.
    ("Discussion: the TSF jackknife factor is the largest on TabArena", ("jk.ta.nested.tsf.ratio", "jk.ta.nested.fp.ratio", "jk.ta.nested.vb.ratio"), lambda t, f, v: t >= max(f, v)),
    # Checklist: the pooled direction check, kept in the outputs, shows no direction.
    ("pooled direction check: interval contains zero, sign test does not reject", ("pool.disjoint.ci", "pool.disjoint.sign"), lambda ci, p: ci[0] < 0 < ci[1] and p >= 0.05),
    ("pooled direction check: TabArena plus the TabRepo datasets it does not share", ("pool.disjoint.n", "ovl.excluded"), lambda n, e: n == 51 + 200 - e),
    # Section 6.1 and Appendix E: how the running claim depends on single datasets.
    ("influence: every leave-one-out interval excludes zero", ("ta.infl.loo.maxcihigh",), lambda h: h < 0),
    ("influence: without the top five the interval includes zero", ("ta.infl.topout.ci",), lambda ci: ci[0] < 0 < ci[1]),
    ("influence: without the top five neither test rejects", ("ta.infl.topout.mp", "ta.infl.topout.nbp"), lambda m, n: m >= 0.05 and n >= 0.05),
    ("influence: two single deletions lose the Holm separation", ("ta.infl.loo.maxholm", "ta.infl.loo.notsep"), lambda h, n: h >= 0.05 and n == 2),
    ("influence: trimmed means keep a smaller gain", ("ta.infl.trimten.diff", "ta.infl.trimtwenty.diff", "ta.primary.nested.vb.diff"), lambda a, b, m: m < a < 0 and m < b < 0),
    ("influence: the gain is concentrated on small datasets", ("ta.infl.small.diff", "ta.infl.large.diff", "ta.infl.size.rho"), lambda s, l, r: s < l < 0 and r > 0),
    # Introduction, Section 6.2: the provenance gap holds on every dataset, with or without the copies.
    ("tuned + ensemble more optimistic than defaults on every dataset", ("ta.erroptim.te.abovedef", "ta.erroptim.ndatasets"), lambda a, n: a == n),
    ("copy-free: tuned + ensemble more optimistic than defaults on every dataset", ("imp.erroptim.te.abovedef", "imp.erroptim.ndatasets"), lambda a, n: a == n),
    ("copy-free: the in-context defaults lie lower still", ("imp.optim.method.tabpfnv2default", "ta.optim.method.tabpfnv2default", "imp.optim.method.tabicldefault", "ta.optim.method.tabicldefault"), lambda a, b, c, d: a < b and c < d),
    ("copy-free: the single best solver is below nested selection", ("imp.sbs.reg", "imp.nested.reg"), lambda s, n: s < n),
    # Sections 6.1 and 6.3: the corrected test against the bootstrap.
    ("corrected SE about the bootstrap SE for nested - VB", ("ta.primary.nested.vb.nbratio",), lambda r: 0.8 <= r <= 1.25),
    ("corrected 90% intervals reach beyond both margins", ("ta.primary.nested.fp.nbci", "ta.primary.nested.tsf.nbci"), lambda f, t: f[1] > 0.015 and t[1] > 0.015 and t[0] < -0.015),
    # Section 6.2: validation-best restricted to the tuned classes changes little.
    ("tuned classes: validation-best changes little", ("ta.vbtun.reg", "ta.vbte.reg", "ta.vb.reg"), lambda a, b, v: abs(a - v) < 0.02 and abs(b - v) < 0.02),
    # Sections 6.2 and 7: the error-space correction.
    ("error-space (range) no better than validation-best on TabArena", ("ta.es.reg", "ta.vb.reg"), lambda e, v: e >= v),
    ("error-space worse than validation-best on TabRepo", ("tr.es.reg", "tr.esf.reg", "trs.vb.reg"), lambda a, b, v: a > v and b > v),
    # Section 6.4: many pool rows duplicate others.
    ("some pool rows duplicate others", ("ta.pool.distinct", "ta.pool.rows"), lambda d, n: d < n),
    ("imputed rows repeat RandomForest within one percent", ("imp.maxgap",), lambda g: g < 1.0),
    ("copy-free: where the mean tests disagree, only the bootstrap test rejects", ("imp.mtests.onlyboot", "imp.mtests.differ"), lambda b, d: b == d),
    ("the refit, not the yardstick, raises nested selection", ("ta.nested.reg", "imp.resc.nested.reg", "imp.nested.reg"), lambda m, r, n: m < r < n),
    ("the yardstick alone leaves validation-best as in the rerun", ("imp.resc.vb.reg", "imp.vb.reg"), lambda r, n: r == n),
    # Third review (Sections 3.4, 6.1, 6.4, 7, Appendices E, G and H).
    ("matched three-member family: every mean test rejects for both pools",
     ("pool.full.three.mholm", "pool.full.three.tholm", "pool.full.three.nbholm", "pool.reduced.three.mholm", "pool.reduced.three.tholm", "pool.reduced.three.nbholm"),
     lambda *ps: max(ps) < 0.05),
    ("matched five-member family: only the bootstrap test rejects, for both pools",
     ("ta.primary.nested.vb.mholm", "ta.primary.nested.vb.tholm", "ta.primary.nested.vb.nbholm", "pool.reduced.five.mholm", "pool.reduced.five.tholm", "pool.reduced.five.nbholm"),
     lambda m, t, n, rm, rt, rn: m < 0.05 and rm < 0.05 and min(t, n, rt, rn) >= 0.05),
    ("the two pools' nested selections are not separated", ("pool.direct.ci",), lambda ci: ci[0] < 0 < ci[1]),
    ("imputed rows: most of the rescored advantage remains after refitting", ("imp.retained", "flag.retained"), lambda a, b: 50 < a < 100 and 50 < b < 100),
    ("imputed rows: rescored and reapplied keep the separation, refitted loses it",
     ("imp.rescored_primary.nested.vb.mholm", "imp.reapplied_primary.nested.vb.mholm", "imp.primary.nested.vb.mholm",
      "flag.rescored_primary.nested.vb.mholm", "flag.reapplied_primary.nested.vb.mholm", "flag.primary.nested.vb.mholm"),
     lambda a, b, c, d, e, f: max(a, b, d, e) < 0.05 <= min(c, f)),
    ("imputed rows: refitting changes most nested choices", ("imp.choice.same",), lambda n: n < 50),
    ("imputed rows: reapplication reproduces the main run on every split", ("imp.reap.checked",), lambda n: n == 100),
    ("matched spread-to-noise: above one on TabArena, below one on TabRepo", ("ta.esnr.all", "tr.esnr.all", "tr.esnr.fam"), lambda a, b, c: a > 1 > max(b, c)),
    ("optimistic candidates are the better ones on both artifacts", ("ta.offq.all.rho", "ta.offq.all.p", "tr.offq.all.rho", "tr.offq.all.p"), lambda a, pa, b, pb: a < 0 and b < 0 and max(pa, pb) < 0.05),
    ("TabRepo source groups: no verdict changes under the cluster bootstrap", ("trgrp.flips",), lambda n: n == 0),
    ("TabRepo source groups: intervals widen modestly", ("trgrp.widthmax",), lambda w: 1.0 <= w < 1.5),
    ("TabRepo grouped splits: the shift against validation-best exceeds design noise under both estimates",
     ("trgs.two.vb.likeholm", "trgs.two.vb.ownholm"), lambda a, b: max(a, b) < 0.05),
    ("TabRepo grouped splits: the shift against the train-selected rule only under the random-design estimate",
     ("trgs.two.tsfs.likeholm", "trgs.two.tsfs.ownholm"), lambda a, b: a < 0.05 <= b),
    ("TabRepo grouped splits: the shifts against FP, task-adaptive and the guard do not under the random-design estimate",
     ("trgs.two.fp.likeholm", "trgs.two.tad.likeholm", "trgs.two.gfive.likeholm"), lambda *ps: min(ps) >= 0.05),
    ("TabRepo grouped splits: no ensemble-inclusive shift exceeds design noise", ("trg.two.minholm",), lambda p: p >= 0.05),
    ("coverage study: the bootstrap is adequate for both no-fit contrasts in the base world",
     ("cal.pt.base.c5.boot.code", "cal.pt.base.c6.boot.code", "cal.pt.base.c5.t.code", "cal.pt.base.c6.t.code"), lambda *c: max(c) == 0),
    ("coverage study: no-fit bootstrap adequate under instability and dependence, inconclusive under skew",
     ("cal.pt.instability.c5.boot.code", "cal.pt.instability.c6.boot.code", "cal.pt.dependence.c5.boot.code", "cal.pt.dependence.c6.boot.code", "cal.pt.skew.c5.boot.code", "cal.pt.skew.c6.boot.code"),
     lambda a, b, c, d, e, f: max(a, b, c, d) == 0 and e == f == 1),
    ("coverage study: the bootstrap is anti-conservative for nested selection minus the committed rules in the base world and at 100 splits",
     ("cal.pt.base.c2.boot.code", "cal.pt.base.c3.boot.code", "cal.sh100.base.c2.boot.code", "cal.sh100.base.c3.boot.code"), lambda *c: min(c) == max(c) == 2),
    ("coverage study: anti-conservative for nested - TSF in every scenario, for nested - FP in three",
     ("cal.pt.c3.boot.nanti", "cal.pt.c2.boot.nanti"), lambda c3, c2: c3 == 4 and c2 == 3),
    ("coverage study: for nested - VB the bootstrap is inconclusive, for FP - VB anti-conservative, in the base world",
     ("cal.pt.base.c1.boot.code", "cal.pt.base.c4.boot.code"), lambda a, b: a == 1 and b == 2),
    ("coverage study: the corrected test is adequate for nested minus the committed rules, at 20 and at 100 splits",
     ("cal.pt.base.c2.nb.code", "cal.pt.base.c3.nb.code", "cal.sh100.base.c2.nb.code", "cal.sh100.base.c3.nb.code"), lambda *c: max(c) == 0),
    ("coverage study: at 100 splits the corrected test is inconclusive for nested - VB, below its 20-split coverage",
     ("cal.sh100.base.c1.nb.code", "cal.sh100.base.c1.nb.cov", "cal.sh20.base.c1.nb.cov", "cal.sh100.base.c1.nb.seratio", "cal.sh20.base.c1.nb.seratio"),
     lambda code, c100, c20, r100, r20: code == 1 and c100 < c20 and r100 < 1 < r20),
    ("coverage study: the bootstrap's coverage for nested - VB does not change with the number of splits",
     ("cal.sh20.base.c1.boot.cov", "cal.sh100.base.c1.boot.cov"), lambda a, b: abs(a - b) <= 0.02),
    ("coverage study: the bootstrap's SE ratios are below one for every fitted contrast in the base world",
     ("cal.pt.base.fit.boot.ratiomax",), lambda r: r < 1),
    ("coverage study: simulated differences are less skewed than TabArena's",
     ("cal.skewness.base.c1", "cal.skewness.skew.c1", "ta.mtests.vb.skew"), lambda b, k, t: t < k < 0 and t < b < 0),
    ("coverage study: the inner fit barely moves theta", ("cal.innershift.c1",), lambda d: abs(d) < 0.002),
    ("coverage study: no interval is labelled conservative", ("cal.nconservative",), lambda n: n == 0),
    ("coverage study: the conservative label cannot be reached with 100 replicates", ("cal.rule.maxlower",), lambda b: b < 0.975),
    ("coverage study: under skew the paired t is adequate for the defaults rule, inconclusive for VB among defaults",
     ("cal.pt.skew.c5.t.code", "cal.pt.skew.c6.t.code"), lambda a, b: a == 0 and b == 1),
    ("coverage study: base-world bootstrap labels in Section 6.3 (C2-C4 anti-conservative, C1 inconclusive)",
     ("cal.pt.base.c1.boot.code", "cal.pt.base.c2.boot.code", "cal.pt.base.c3.boot.code", "cal.pt.base.c4.boot.code"), lambda a, b, c, d: a == 1 and b == c == d == 2),
    ("coverage study: the corrected test is wider than needed for nested minus the committed rules",
     ("cal.pt.base.c2.nb.seratio", "cal.pt.base.c3.nb.seratio", "cal.sh100.base.c2.nb.seratioci", "cal.sh100.base.c3.nb.seratioci"),
     lambda a, b, c, d: min(a, b) > 1 and c[0] > 1 and d[0] > 1),
    ("coverage study: the corrected test is adequate for nested - FP in every scenario and for nested - TSF in three",
     ("cal.pt.c2.nb.nadequate", "cal.pt.c3.nb.nadequate"), lambda a, b: a == 4 and b == 3),
    ("coverage study: only FP - VB under dependence is an adequate bootstrap label among C1 and C4",
     ("cal.pt.dependence.c4.boot.code", "cal.pt.base.c4.boot.code", "cal.pt.skew.c4.boot.code", "cal.pt.instability.c4.boot.code",
      "cal.pt.base.c1.boot.code", "cal.pt.skew.c1.boot.code", "cal.pt.instability.c1.boot.code", "cal.pt.dependence.c1.boot.code"),
     lambda d4, *others: d4 == 0 and min(others) > 0),
    ("Section 6.3: the bootstrap separates nested selection and the family-prior rule from validation-best, the corrected test confirms neither",
     ("ta.primary.nested.vb.mholm", "ta.mechanism.fp.vb.mholm", "ta.primary.nested.vb.nbholm", "ta.mechanism.fp.vb.nbholm"),
     lambda a, b, c, d: max(a, b) < 0.05 <= min(c, d)),
    ("coverage study: at 20 splits the audit's weights widen the corrected interval for nested - VB",
     ("cal.pt.base.c1.nb.seratio", "cal.pt.base.c1.nbt.seratio"), lambda w, t: w > 1 > t),
    ("coverage study: nested - VB is about two rank positions in the base world", ("cal.pt.base.c1.theta",), lambda t: 1.75 < -t * 43 < 2.25),
    ("coverage study: the base world is milder than TabArena, the skew world harsher on datasets with 30 units",
     ("cal.fact.base.vbregret", "cal.fact.ta.vbregret", "cal.fact.base.vbregretsmall", "cal.fact.ta.vbregretsmall", "cal.fact.skew.vbregretsmall"),
     lambda b, t, bs, ts, ss: b < t and bs < ts < ss),
    ("TabRepo grouped splits: the correction's shift under exactly one estimate", ("trgs.two.opt.likeholm", "trgs.two.opt.ownholm"), lambda a, b: (a < 0.05) != (b < 0.05)),
    ("TabRepo grouped splits: FP, task-adaptive and the guard under neither estimate", ("trgs.two.rest.minholm",), lambda p: p >= 0.05),
    ("TabRepo grouped splits: the paired t-test no longer rejects nested - ensemble-3",
     ("tr.fixed_ensembles.nested.ethree.tholm", "trg.fixed_ensembles.nested.ethree.tholm", "trg.fixed_ensembles.nested.ethree.mholm"),
     lambda a, b, m: a < 0.05 <= b and m < 0.05),
    ("every disagreement between the mean tests, in every run, is a bootstrap rejection", ("mtests.differ.all", "mtests.differ.onlyboot"), lambda a, b: a == b > 0),
    ("the shared inner fit and the reported conventions separate nested selection from validation-best only borderline",
     ("attr.inner.vb.mholm", "attr.inner.vb.tholm", "attr.vtwo.vb.mholm", "attr.vtwo.vb.tholm"), lambda a, b, c, d: max(a, c) < 0.05 <= min(b, d)),
    ("documented-rows mask: the reapplied stage is not borderline (the paired t-test also rejects)",
     ("flag.reapplied_primary.nested.vb.mholm", "flag.reapplied_primary.nested.vb.tholm"), lambda m, t: max(m, t) < 0.05),
    ("the defaults rule is separated from neither committed rule nor from validation-best among defaults",
     ("ta.provenance_reference.dtt.fp.mholm", "ta.provenance_reference.dtt.tsf.mholm", "ta.provenance_reference.dtt.vbdef.mholm", "ta.provenance_reference.dtt.vbdef.rholm"),
     lambda a, b, c, r: min(a, b, c) >= 0.05 and r < 0.05),
    ("grouped score-only splits: the bootstrap test newly rejects against the correction and kNN with validation",
     ("trs.revision_references.nested.opt.mholm", "trgs.revision_references.nested.opt.mholm", "trs.revision_references.nested.knnv.mholm", "trgs.revision_references.nested.knnv.mholm"),
     lambda a, b, c, d: a >= 0.05 > b and c >= 0.05 > d),
    ("the extended pool's gain over validation-best is borderline",
     ("ta.ext.primary.nested.vb.mholm", "ta.ext.primary.nested.vb.tholm"), lambda m, t: m < 0.05 <= t),
    ("imputed rows: refitting moves the contrast by an amount whose interval includes zero",
     ("imp.stage.refit.ci", "flag.stage.refit.ci"), lambda a, b: a[0] < 0 < a[1] and b[0] < 0 < b[1]),
    ("imputed rows: the rescored and reapplied separations are borderline (paired t-test does not reject)",
     ("imp.rescored_primary.nested.vb.tholm", "imp.reapplied_primary.nested.vb.tholm"), lambda a, b: min(a, b) >= 0.05),
    ("copy-free: validation-best minus the single best solver is not separated, interval excludes zero",
     ("imp.provenance.vb.sbs.ci", "imp.provenance.vb.sbs.mholm"), lambda ci, p: ci[0] > 0 and p >= 0.05),
    ("TabRepo score-only: task-adaptive and guard beat nested selection; the paired t-test also rejects, not the corrected or signed-rank tests",
     ("trs.score_level_rows.nested.tad.mholm", "trs.score_level_rows.nested.gfive.mholm", "trs.score_level_rows.nested.tad.tholm", "trs.score_level_rows.nested.gfive.tholm",
      "trs.score_level_rows.nested.tad.nbholm", "trs.score_level_rows.nested.gfive.nbholm", "trs.score_level_rows.nested.tad.rholm", "trs.score_level_rows.nested.gfive.rholm"),
     lambda a, b, c, d, e, f, g, h: max(a, b, c, d) < 0.05 <= min(e, f, g, h)),
    ("TabRepo score-only nested - TSF: the bootstrap test rejects with grouped splits and with the shared inner fit, not as reported",
     ("trgs.primary.nested.tsfs.mholm", "attr.tr.inner.tsfs.mholm", "trs.primary.nested.tsfs.mholm"), lambda g, i, r: max(g, i) < 0.05 <= r),
    ("TabRepo grouped splits: nested selection ties the ensemble-inclusive reference", ("trg.primary.nested.tsfe.wins", "trg.primary.nested.tsfe.losses"), lambda w, l: w == 0 and l == 0),
    ("TabRepo source overlap: no verdict or sign changes", ("ovls.flips", "ovls.signs"), lambda f, g: f == 0 and g == 0),
    # Table 3 and Appendix H: TabRepo's RandomForest and ExtraTrees rows are single forests with
    # out-of-bag validation, its KNN rows single models with leave-one-out validation.
    ("TabRepo slice: RF and XT test predictions mostly on 1/300, validation predictions never",
     ("trc.rf.test", "trc.xt.test", "trc.rf.val", "trc.xt.val"), lambda a, b, c, d: min(a, b) > 50 and max(c, d) == 0),
    ("TabRepo slice: out-of-bag denominators near 300/e", ("trc.rf.valden", "trc.xt.valden"), lambda a, b: 80 <= min(a, b) and max(a, b) <= 130),
    ("TabRepo slice: KNN validation leave-one-out in most tasks, never a bag", ("trc.knn.val", "trc.knn.bag"), lambda v, b: v > 50 and b == 0),
    ("TabRepo slice: the control families lie on no lattice", ("trc.ctrl.lattice",), lambda c: c == 0),
    # Section 7 and Appendix H: the group-aware rerun.
    ("TabRepo grouped at every level: nested selection ties the ensemble-inclusive reference", ("trgi.primary.nested.tsfe.wins", "trgi.primary.nested.tsfe.losses"), lambda w, l: w == 0 and l == 0),
    ("TabRepo grouped at every level: score-only nested - TSF, VB and the rank correction not rejected, outer-only TSF rejected",
     ("trgis.primary.nested.tsfs.mholm", "trgis.primary.nested.vb.mholm", "trgis.revision_references.nested.opt.mholm", "trgs.primary.nested.tsfs.mholm"),
     lambda a, b, c, o: min(a, b, c) >= 0.05 and o < 0.05),
    ("TabRepo grouped at every level: no score-only shift against the random splits", ("trgis.two.minholm",), lambda m: m >= 0.05),
    ("TabRepo grouped at every level: kNN with validation still rejected, task-adaptive and guard still not; outer-only rejected the correction and kNN",
     ("trgis.revision_references.nested.knnv.mholm", "trgis.score_level_rows.nested.tad.mholm", "trgis.score_level_rows.nested.gfive.mholm",
      "trgs.revision_references.nested.opt.mholm", "trgs.revision_references.nested.knnv.mholm", "trgs.score_level_rows.nested.tad.mholm", "trgs.score_level_rows.nested.gfive.mholm"),
     lambda k, t, g, o, ok, ot, og: k < 0.05 <= min(t, g) and max(o, ok) < 0.05 <= min(ot, og)),
    ("TabRepo grouped at every level: the paired changes are positive, some rejected by the dataset bootstrap, none by the cluster bootstrap",
     ("trgics.change.min", "trgics.nreject", "trgics.cluster.minholm"), lambda lo, n, c: lo > 0 and n >= 1 and c >= 0.05),
    ("TabRepo grouped at every level: the cluster bootstrap changes no test result", ("trgi.cluster.flips",), lambda f: f == 0),
    ("TabRepo: nested selection and the ensemble-inclusive reference choose the same row in every split of all three runs",
     ("tr.samerow", "tr.splits", "trg.samerow", "trg.splits", "trgi.samerow", "trgi.splits"),
     lambda a, n, b, m, c, k: a == n and b == m and c == k),
    ("TabRepo grouped runs: the same ensemble-inclusive gain over validation-best in both, unchanged on every dataset",
     ("trg.primary.nested.vb.diff", "trgi.primary.nested.vb.diff", "trg.primary.nested.vb.ci", "trgi.primary.nested.vb.ci", "trgic.vb.changed"),
     lambda a, b, c, d, n: a == b and c == d and n == 0),
    ("TabRepo outer-grouped run: the bootstrap test rejects against TSF, the rank correction and kNN with validation, the corrected test for none",
     ("trgs.primary.nested.tsfs.mholm", "trgs.revision_references.nested.opt.mholm", "trgs.revision_references.nested.knnv.mholm",
      "trgs.primary.nested.tsfs.nbholm", "trgs.revision_references.nested.opt.nbholm", "trgs.revision_references.nested.knnv.nbholm"),
     lambda a, b, c, d, e, f: max(a, b, c) < 0.05 <= min(d, e, f)),
    ("TabRepo outer-grouped run: task-adaptive and the guard rejected over the random splits, no longer here",
     ("trs.score_level_rows.nested.tad.mholm", "trs.score_level_rows.nested.gfive.mholm",
      "trgs.score_level_rows.nested.tad.mholm", "trgs.score_level_rows.nested.gfive.mholm"),
     lambda a, b, c, d: max(a, b) < 0.05 <= min(c, d)),
    ("TabRepo outer-grouped run: every interval of the six score-only contrasts is wider than over the random splits",
     tuple(f"{run}.{key}.ci" for key in ["primary.nested.vb", "primary.nested.tsfs", "revision_references.nested.opt",
                                           "revision_references.nested.knnv", "score_level_rows.nested.tad", "score_level_rows.nested.gfive"]
           for run in ("trs", "trgs")),
     lambda *cis: all(cis[i + 1][1] - cis[i + 1][0] > cis[i][1] - cis[i][0] for i in range(0, len(cis), 2))),
    ("TabRepo fully grouped run: every two-sample Holm p is 1 in both pools", ("trgi.two.allminholm",), lambda p: p == 1.0),
    ("TabRepo fully grouped run: the largest score-only change, against VB and the rank correction alike, rejected by the dataset bootstrap only; their rows unchanged",
     ("trgics.vb.change", "trgics.opt.change", "trgics.change.max", "trgics.vb.holm", "trgics.vb.clusterholm", "trgics.refs.changed"),
     lambda v, o, m, h, c, n: v == o == m and h < 0.05 <= c and n == 0),
    # Section 7: with uniform selection provenance the five lowest family rank calibrations are
    # TabPFN, linear models, KNN, ExtraTrees and RandomForest.
    ("TabRepo: the five lowest family rank calibrations",
     tuple(f"tr.optim.family.{name}" for name in ["tabpfn", "linearmodel", "kneighbors", "extratrees", "randomforest",
                                                    "xgboost", "neuralnetfastai", "neuralnettorch", "lightgbm", "fttransformer", "catboost"]),
     lambda *v: max(v[:5]) < min(v[5:])),
    ("corrected test: the degrees of freedom change few verdicts, the reduced pool's among them",
     ("nb.dfflips", "ta.shrink.primary.nested.vb.nbholm", "ta.shrink.primary.nested.vb.nbholmdf"),
     lambda f, a, b: f == 2 and a < 0.05 <= b),
]


def printed_value(text: str) -> Any:
    """The number in a printed value such as `$-$0.0031`, `72\\%` or `0.96`, or the two ends
    of a printed interval such as `[$-$0.09, 0.22]`."""
    cleaned = text.replace("$-$", "-").replace("$+$", "").replace("\\%", "").replace("$", "").replace("{,}", "").replace("\\,", "")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        low, high = cleaned[1:-1].split(",")
        return (float(low), float(high))
    if cleaned.startswith("<"):
        # A bound such as p < 0.001: any value strictly below it.
        return float(cleaned[1:]) * (1.0 - 1e-9)
    return float(cleaned)


def check_stated_comparisons(assets: Assets) -> list[str]:
    problems = []
    for description, keys, test in STATED_COMPARISONS:
        if any(key not in assets.numbers for key in keys):
            problems.append(f"{description}: missing {[key for key in keys if key not in assets.numbers]}")
            continue
        values = [printed_value(assets.numbers[key]) for key in keys]
        if not test(*values):
            problems.append(f"{description}: the text says so, the numbers give {dict(zip(keys, values))}")
    return problems


def check_stated_verdicts(assets: Assets) -> list[str]:
    """Every verdict the manuscript states, checked against the computed one."""
    problems = []
    for base, stated in STATED_VERDICTS.items():
        computed = assets.verdicts.get(base)
        if computed is None:
            problems.append(f"{base}: no computed contrast")
            continue
        for target, verdict in stated.items():
            if computed.get(target) != verdict:
                problems.append(f"{base} [{target}]: the text says {verdict!r}, the numbers give {computed.get(target)!r}")
    return problems


def contrast_numbers(assets: Assets, prefix: str, contrasts: pd.DataFrame) -> None:
    for _, row in contrasts.iterrows():
        left = POLICY_KEYS.get(str(row["left"]))
        right = POLICY_KEYS.get(str(row["right"]))
        if left is None or right is None:
            continue
        base = f"{prefix}.{row['family']}.{left}.{right}"
        assets.define(f"{base}.diff", num(row["mean_difference"], 4, signed=True))
        assets.define(f"{base}.ci", interval(row["ci_low"], row["ci_high"], 4))
        assets.define_p(f"{base}.p", pvalue(row["wilcoxon_p"]))
        assets.define_p(f"{base}.holm", pvalue(row["wilcoxon_holm_p"]))
        # Each test under its own name: m = the mean test that inverts the percentile
        # interval (the separation verdict), t = the paired t-test, r = the signed-rank test.
        for name, raw, adjusted in (
            ("m", "bootstrap_p", "bootstrap_holm_p"),
            ("t", "ttest_p", "ttest_holm_p"),
            ("r", "wilcoxon_p", "wilcoxon_holm_p"),
        ):
            if raw in row and pd.notna(row[raw]):
                assets.define_p(f"{base}.{name}p", pvalue(row[raw]))
                assets.define_p(f"{base}.{name}holm", pvalue(row[adjusted]))
        assets.define_p(f"{base}.sign", pvalue(row["sign_test_p"]))
        if "hodges_lehmann_difference" in row and pd.notna(row["hodges_lehmann_difference"]):
            # The location the signed-rank test is about, and the median difference.
            assets.define(f"{base}.hl", num(row["hodges_lehmann_difference"], 4, signed=True))
            assets.define(f"{base}.med", num(row["median_difference"], 4, signed=True))
        assets.verdicts[base] = contrast_verdicts(row)
        assets.define(f"{base}.wtl", wtl(row))
        if "se_bootstrap" in row and math.isfinite(row["se_bootstrap"]):
            assets.define(f"{base}.sebs", num(row["se_bootstrap"], 4))
        assets.define(f"{base}.wins", str(int(row["wins"])))
        assets.define(f"{base}.losses", str(int(row["losses"])))
        for margin in ["0p005", "0p010", "0p015", "0p020"]:
            column = f"tost_p_{margin}"
            if column in row and math.isfinite(row[column]):
                assets.define_p(f"{base}.tost{margin.replace('0p', '')}", pvalue(row[column]))
        if math.isfinite(row.get("smallest_equivalence_margin", float("nan"))):
            assets.define(f"{base}.margin", num(row["smallest_equivalence_margin"], 4))
            assets.define(f"{base}.tostci", interval(row["tost_90_ci_low"], row["tost_90_ci_high"], 4))
        for power in ["80", "90"]:
            column = f"mde{power}_bootstrap"
            if column in row and math.isfinite(row[column]):
                assets.define(f"{base}.mde{power}", num(row[column], 4))
                if f"mde{power}_bootstrap_rank_positions" in row:
                    assets.define(f"{base}.mde{power}rank", num(row[f"mde{power}_bootstrap_rank_positions"], 2))
                    assets.define(f"{base}.mde{power}param", num(row[f"mde{power}_parametric"], 4))
            for effect in ["0p5", "0p25"]:
                column = f"required_n{power}_{effect}rank"
                if column in row and pd.notna(row[column]):
                    assets.define(f"{base}.req{power}{effect.replace('0p', '')}", str(int(row[column])))


def build_tabarena(assets: Assets) -> None:
    summary = assets.csv("score_level_tabarena_review/clustered_summary.csv")
    contrasts = assets.csv("score_level_tabarena_review/contrasts.csv")
    # The rule that selects on defaults (scripts/provenance_reference.py) joins the table as a
    # post hoc reference, with the same endpoints computed by the same code.
    summary = pd.concat([summary, assets.csv("tabarena_provenance_reference/summary.csv")], ignore_index=True)
    summary_numbers(assets, "ta", summary, rank_scale=43.0)
    contrast_numbers(assets, "ta", contrasts)
    indexed = summary.set_index("method")

    def row(method: str) -> list[str]:
        values = indexed.loc[method]
        gap = values.get("gap_closed", float("nan"))
        return [
            policy_label(method),
            f"{num(values['mean_normalized_rank_regret'])} {interval(values['mean_normalized_rank_regret_ci_low'], values['mean_normalized_rank_regret_ci_high'])}",
            num(gap, 2) if math.isfinite(gap) else "--",
            f"{num(values['mean_maxmin_error_regret'])} {interval(values['mean_maxmin_error_regret_ci_low'], values['mean_maxmin_error_regret_ci_high'])}",
            seconds(values["mean_total_seconds"]),
        ]

    declared = [
        "oracle",
        "meta_selector_family_prior_budgeted",
        "train_selected_fixed_reference__declared_fixed",
        "train_selected_fixed_reference__with_calibrated",
        "meta_selector_meta_feature_guard_002",
        "nested_policy_selected",
        "meta_selector_meta_feature_guard_005",
        "meta_selector_task_adaptive_budgeted",
        "meta_selector_calibrated_budget_10000s",
        "meta_selector_single",
    ]
    added = [
        "reference_optimism_corrected_task",
        "reference_optimism_corrected",
        "reference_defaults_then_tuned",
        "reference_optimism_corrected_budget",
        "reference_metafeature_knn_validation",
        "reference_validation_best_default",
        "reference_single_best_global",
        "reference_metafeature_knn",
        "reference_single_best_task",
        "reference_priors_only",
    ]
    rows = [row(method) for method in declared] + [row(method) for method in added]
    assets.tables["tabarena_main"] = tabular(
        "lrrrr",
        ["Policy", "Norm.\\ rank regret [95\\% CI]", "Gap closed", "Error regret [95\\% CI]", "Runtime (s)"],
        rows,
        groups={0: "Policies and references of the nested audit", len(declared): "References added after the main results (post hoc)"},
    )

    primary = contrasts[contrasts["family"].eq("primary")]
    # The corrected resampled t-test (scripts/corrected_resampled_test.py) sits next to the
    # dataset bootstrap: it accounts for the overlap of the outer splits' training sets.
    corrected = assets.csv("tabarena_corrected_test/corrected_resampled_tests.csv")
    corrected = corrected[corrected["family"].eq("primary")].set_index("right")
    assets.tables["tabarena_paired"] = tabular(
        "lrrrrrrr",
        [
            "Reference",
            "Nested $-$ reference [95\\% CI]",
            "Holm $p$, bootstrap",
            "Holm $p$, $t$",
            "Holm $p$, corrected",
            "Holm $p$, signed-rank",
            "Sign $p$",
            "W/T/L",
        ],
        [
            [
                policy_label(item["right"]),
                f"{num(item['mean_difference'], 4, signed=True)} {interval(item['ci_low'], item['ci_high'], 4)}",
                pvalue(item["bootstrap_holm_p"]),
                pvalue(item["ttest_holm_p"]),
                pvalue(corrected.loc[item["right"], "nb_holm_p"]),
                pvalue(item["wilcoxon_holm_p"]),
                pvalue(item["sign_test_p"]),
                wtl(item),
            ]
            for _, item in primary.iterrows()
        ],
    )
    equivalence_rows = []
    for right in ["meta_selector_family_prior_budgeted", "train_selected_fixed_reference__declared_fixed"]:
        item = contrasts[contrasts["family"].eq("primary") & contrasts["right"].eq(right)].iloc[0]
        nb = corrected.loc[right]
        equivalence_rows.append(
            [
                policy_label(right),
                num(item["mean_difference"], 4, signed=True),
                interval(item["tost_90_ci_low"], item["tost_90_ci_high"], 4),
                *[pvalue(item[f"tost_p_{margin}"]) for margin in ["0p005", "0p010", "0p015", "0p020"]],
                num(item["smallest_equivalence_margin"], 4),
                interval(nb["nb_ci90_low"], nb["nb_ci90_high"], 4),
                num(nb["nb_smallest_equivalence_margin"], 4),
            ]
        )
    assets.tables["tabarena_equivalence"] = tabular(
        "lrrrrrrrrr",
        [
            "Reference", "Mean diff.", "90\\% CI", "0.005", "0.010", "0.015", "0.020", "Smallest margin",
            "Corrected 90\\% CI", "Corrected smallest margin",
        ],
        equivalence_rows,
    )
    pools = contrasts[contrasts["family"].eq("comparator_pools")]
    counts = assets.csv("score_level_tabarena_review/train_selected_reference_counts.csv")
    pool_rows = []
    for _, item in pools.iterrows():
        pool = str(item["right"]).split("__", 1)[1]
        chosen = counts[counts["pool"].eq(pool)].sort_values("n_splits", ascending=False)
        chosen_text = "; ".join(
            f"{policy_label(choice['selected_reference_method'])} {int(choice['n_splits'])}"
            for _, choice in chosen.head(3).iterrows()
        )
        reference = indexed.loc[item["right"]]
        pool_rows.append(
            [
                policy_label(item["right"]),
                chosen_text,
                f"{num(reference['mean_normalized_rank_regret'], 4)}",
                f"{num(item['mean_difference'], 4, signed=True)} {interval(item['ci_low'], item['ci_high'], 4)}",
            ]
        )
    assets.tables["tabarena_pools"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.26\\linewidth}>{\\raggedright\\arraybackslash}p{0.34\\linewidth}rr",
        ["Reference pool", "Most-selected rows (splits of 100)", "Reference regret", "Nested $-$ reference [95\\% CI]"],
        pool_rows,
    )
    # How concentrated the gain over validation-best is: share of the summed dataset-level
    # difference contributed by the five datasets with the largest gains.
    means = assets.csv("score_level_tabarena_review/dataset_mean_scores.csv")
    wide = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    gain = (wide["nested_policy_selected"] - wide["meta_selector_single"]).sort_values()
    assets.define("ta.nested.vb.topfiveshare", percent(float(gain.head(5).sum() / gain.sum())))
    assets.define("ta.nested.vb.median", num(float(gain.median()), 4, signed=True))
    # Why the refitting diagnostic exceeds the conditional standard error most for the small
    # contrasts (Section 6.3): the correlation of nested selection's dataset-level regrets with
    # each reference's; highly correlated rows leave their difference little conditional spread.
    for key, method in [
        ("vb", "meta_selector_single"),
        ("fp", "meta_selector_family_prior_budgeted"),
        ("tsf", "train_selected_fixed_reference__declared_fixed"),
    ]:
        assets.define(f"ta.nested.{key}.corr", num(float(wide["nested_policy_selected"].corr(wide[method])), 2))
    # Section 6.2: the budgeted correction and the family-prior rule agree on average; how
    # closely they track each other across datasets, and how far apart they are per dataset.
    correction, prior = wide["reference_optimism_corrected_budget"], wide["meta_selector_family_prior_budgeted"]
    assets.define("ta.optb.fp.corr", num(float(correction.corr(prior)), 2))
    assets.define("ta.optb.fp.madrank", num(float(43.0 * (correction - prior).abs().mean()), 1))
    selection = assets.csv("score_level_tabarena_review/policy_selection_counts.csv")
    assets.define("ta.nested.distinct", str(int(selection.shape[0])))
    top = selection.iloc[0]
    assets.define("ta.nested.topcount", str(int(top["n_splits"])))
    assets.define("ta.nested.top", policy_label(top["selected_policy"]))
    strata = assets.csv("score_level_tabarena_review/task_stratified_summary.csv")
    strata = strata[strata["metric"].eq("mean_normalized_rank_regret")]
    strata_rows = []
    for (task, n_datasets), group in strata.groupby(["task", "n_datasets"], sort=True):
        # Section 6.2 quotes a stratum's size and an interval to show how imprecise it is.
        assets.define(f"ta.strata.{task}.n", str(int(n_datasets)))
        for _, item in group.sort_values("estimate").iterrows():
            strata_rows.append(
                [f"{task.capitalize()} ({int(n_datasets)})", policy_label(item["method"]), f"{num(item['estimate'])} {interval(item['ci_low'], item['ci_high'])}"]
            )
            key = POLICY_KEYS.get(str(item["method"]))
            if key:
                assets.define(f"ta.strata.{task}.{key}", num(item["estimate"]))
                assets.define(f"ta.strata.{task}.{key}ci", interval(item["ci_low"], item["ci_high"]))
    assets.tables["tabarena_strata"] = tabular("llr", ["Task (datasets)", "Policy", "Norm.\\ rank regret [95\\% CI]"], strata_rows)


# TabArena method identifiers as the TabArena paper names them.
CANDIDATE_NAMES = {
    "CAT": "CatBoost", "EBM": "EBM", "FASTAI": "FastaiMLP", "GBM": "LightGBM", "KNN": "KNN",
    "LR": "Linear", "MNCA": "ModernNCA", "NN_TORCH": "TorchMLP", "REALMLP": "RealMLP",
    "RF": "RandomForest", "TABDPT": "TabDPT", "TABICL": "TabICL", "TABM": "TabM",
    "TABPFNV2": "TabPFNv2", "XGB": "XGBoost", "XT": "ExtraTrees",
}


def candidate_name(method: str) -> str:
    family, _, variant = str(method).partition(" (")
    variant = variant.rstrip(")").replace("tuned + ensemble", "tuned+ens.")
    return f"{CANDIDATE_NAMES.get(family, family)} ({variant})" if variant else CANDIDATE_NAMES.get(family, family)


def build_members(assets: Assets) -> None:
    """Most-selected candidates per row, with the median-normalised error regret."""
    top = assets.csv("tabarena_provenance/top_selected_members.csv")
    summary = assets.csv("score_level_tabarena_review/clustered_summary.csv").set_index("method")
    rows = []
    for method in [
        "meta_selector_single",
        "reference_validation_best_default",
        "reference_single_best_global",
        "meta_selector_family_prior_budgeted",
        "nested_policy_selected",
        "reference_optimism_corrected",
        "reference_optimism_corrected_task",
        "reference_priors_only",
        "oracle",
    ]:
        chosen = top[top["method"].eq(method)].sort_values("share", ascending=False).head(3)
        members = "; ".join(f"{candidate_name(item['selected_member'])} {percent(item['share'])}" for _, item in chosen.iterrows())
        rows.append([policy_label(method), members, num(summary.loc[method, "mean_median_error_regret"])])
        for rank, (_, item) in enumerate(chosen.iterrows(), start=1):
            key = POLICY_KEYS[method]
            assets.define(f"ta.top.{key}.{rank}", candidate_name(item["selected_member"]))
            assets.define(f"ta.top.{key}.{rank}.share", percent(item["share"]))
    assets.tables["ta_members"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.30\\linewidth}>{\\raggedright\\arraybackslash}p{0.48\\linewidth}r",
        ["Policy", "Most-selected candidates (share of held-out units)", "Error regret (median-normalized)"],
        rows,
    )


def build_optimism_snr(assets: Assets, prefix: str, directory: str) -> None:
    """Remark 1's spread-to-estimation-error ratio of the top candidates' offsets.

    ``offset_snr.csv`` (scripts/validation_provenance.py) gives the ratio in rank space,
    the space of the rank correction evaluated, and in error space, the space of the
    remark, with dataset-bootstrap intervals (``tabular_sota_benchmark.provenance``).
    """
    table = assets.csv(f"{directory}/offset_snr.csv").set_index("space")
    names = {"rank": "", "error": ".err", "error_median": ".errmed"}
    for space, suffix in names.items():
        row = table.loc[space]
        assets.define(f"{prefix}.optim.snr{suffix}", num(row["ratio"], 1))
        assets.define(f"{prefix}.optim.snr{suffix}.ci", interval(row["ratio_ci_low"], row["ratio_ci_high"], 1))
        assets.define(f"{prefix}.optim.snr{suffix}.spread", num(row["spread"], 3))
        assets.define(f"{prefix}.optim.snr{suffix}.se", num(row["se_train"], 3))
    assets.define(f"{prefix}.optim.snr.ntrain", str(int(table.loc["rank", "n_train"])))
    assets.define(f"{prefix}.optim.snr.ncand", str(int(table.loc["rank", "n_candidates"])))


def build_median_distance(assets: Assets, prefix: str, directory: str) -> None:
    """How close the median candidate is to the best: why the median normalisation of
    error-space optimism is used on TabArena only (Appendix G)."""
    distance = assets.csv(f"{directory}/median_distance.csv").iloc[0]
    assets.define(f"{prefix}.meddist.below", percent(distance["share_units_below_one_percent"]))
    assets.define(f"{prefix}.meddist.atbest", str(int(distance["units_median_at_best"])))


def build_provenance(assets: Assets, prefix: str, directory: str, *, variants: bool, rank_scale: float) -> None:
    by_family = assets.csv(f"{directory}/calibration_by_family.csv")
    method_table = assets.csv(f"{directory}/calibration_by_method.csv")
    shares = assets.csv(f"{directory}/selection_shares.csv")
    # Proposition 1: a common optimism shift does not tilt validation-best; differences in
    # optimism among the candidates do. Normalised by the rank scale so the two artifacts
    # are comparable; "top" is the best tenth by mean validation rank.
    normalised = method_table["optimism"] / rank_scale
    top = top_tenth(method_table)
    top_normalised = top["optimism"] / rank_scale
    assets.define(f"{prefix}.optim.sd", num(float(normalised.std()), 3))
    assets.define(f"{prefix}.optim.topn", str(int(len(top))))
    assets.define(f"{prefix}.optim.topmean", num(float(top_normalised.mean()), 3, signed=True))
    assets.define(f"{prefix}.optim.topsd", num(float(top_normalised.std()), 3))
    if variants:
        # Per-method keys only for TabArena's 44 named candidates (TabRepo has 1530 configurations).
        for _, item in method_table.iterrows():
            key = "".join(character for character in str(item["method"]).lower() if character.isalnum())
            assets.define(f"{prefix}.optim.method.{key}", num(item["optimism"], 1, signed=True))
        by_variant = assets.csv(f"{directory}/calibration_by_variant.csv")
        by_variant_task = assets.csv(f"{directory}/calibration_by_variant_task.csv")
        budget = assets.csv(f"{directory}/budget_exclusion.csv")
        names = {"default": "def", "tuned": "tun", "tuned + ensemble": "te"}
        for _, item in by_variant.iterrows():
            key = names[item["variant"]]
            assets.define(f"{prefix}.optim.{key}", num(item["optimism"], 1, signed=True))
            assets.define(f"{prefix}.optim.{key}.ci", interval(item["optimism_ci_low"], item["optimism_ci_high"], 1))
            assets.define(f"{prefix}.valrank.{key}", num(item["val_rank"], 1))
            assets.define(f"{prefix}.testrank.{key}", num(item["test_rank"], 1))
            if "err_optimism" in item:
                # Error-space optimism: test minus validation error over the unit's range of
                # candidate test errors (tabular_sota_benchmark.provenance).
                assets.define(f"{prefix}.erroptim.{key}", num(item["err_optimism"], 3, signed=True))
                assets.define(
                    f"{prefix}.erroptim.{key}.ci", interval(item["err_optimism_ci_low"], item["err_optimism_ci_high"], 3)
                )
                assets.define(f"{prefix}.erroptim.{key}.med", num(item["err_optimism_median"], 2, signed=True))
                assets.define(
                    f"{prefix}.erroptim.{key}.medci",
                    interval(item["err_optimism_median_ci_low"], item["err_optimism_median_ci_high"], 2),
                )
        if (assets.out / directory / "optimism_by_dataset_variant.csv").exists():
            per_dataset = assets.csv(f"{directory}/optimism_by_dataset_variant.csv").pivot(
                index="dataset", columns="variant", values="err_optimism"
            )
            assets.define(f"{prefix}.erroptim.ndatasets", str(int(per_dataset.shape[0])))
            for variant, short in [("tuned + ensemble", "te"), ("tuned", "tun")]:
                above = int((per_dataset[variant] > per_dataset["default"]).sum())
                assets.define(f"{prefix}.erroptim.{short}.abovedef", str(above))
        has_error = "err_optimism" in by_variant_task.columns
        rows = []
        for _, item in by_variant_task.iterrows():
            key = names[item["variant"]]
            assets.define(f"{prefix}.optim.{key}.{item['task']}", num(item["optimism"], 1, signed=True))
            row = [
                item["variant"],
                f"{item['task']} ({int(item['n_datasets'])})",
                num(item["val_rank"], 1),
                num(item["test_rank"], 1),
                f"{num(item['optimism'], 1, signed=True)} {interval(item['optimism_ci_low'], item['optimism_ci_high'], 1)}",
            ]
            if has_error:
                assets.define(f"{prefix}.erroptim.{key}.{item['task']}", num(item["err_optimism"], 3, signed=True))
                row.append(
                    f"{num(item['err_optimism'], 3, signed=True)} "
                    f"{interval(item['err_optimism_ci_low'], item['err_optimism_ci_high'], 3)}"
                )
            rows.append(row)
        if has_error:
            # Section 6.2: the gap between tuned-and-ensembled candidates and defaults in error
            # space, the heterogeneity that tilts validation-best, per task type. Two decimals, so
            # that it agrees with the difference of the rounded values in the optimism table.
            gaps = by_variant_task.pivot(index="task", columns="variant", values="err_optimism")
            for task, gap in (gaps["tuned + ensemble"] - gaps["default"]).items():
                assets.define(f"{prefix}.erroptim.tegap.{task}", num(gap, 2))
        header = ["Variant", "Task (datasets)", "Validation rank", "Test rank", "Rank calibration [95\\% CI]"]
        if has_error:
            header.append("Error optimism [95\\% CI]")
        assets.tables[f"{prefix}_optimism"] = tabular("llrrr" + ("r" if has_error else ""), header, rows)
        # Dataset-level means, matching every other share in the paper.
        indexed_budget = budget.set_index("method")["share_units_over_budget"]
        assets.define(f"{prefix}.budget.mncate", percent(indexed_budget["MNCA (tuned + ensemble)"]))
        assets.define(f"{prefix}.budget.tabmte", percent(indexed_budget["TABM (tuned + ensemble)"]))
        assets.define(f"{prefix}.budget.realmlpte", percent(indexed_budget["REALMLP (tuned + ensemble)"]))
        share_rows = []
        selected = [
            "meta_selector_single",
            "reference_validation_best_default",
            "meta_selector_family_prior_budgeted",
            "train_selected_fixed_reference__declared_fixed",
            "nested_policy_selected",
            "reference_optimism_corrected",
            "reference_optimism_corrected_task",
            "reference_single_best_global",
            "oracle",
        ]
        indexed = shares.set_index("method")
        for method in selected:
            if method not in indexed.index:
                continue
            item = indexed.loc[method]
            key = POLICY_KEYS[method]
            values = []
            for variant, short in names.items():
                column = f"share::{variant}"
                value = item[column] if column in item else 0.0
                assets.define(f"{prefix}.share.{key}.{short}", percent(value))
                values.append(percent(value))
            share_rows.append([policy_label(method), *values])
        assets.tables[f"{prefix}_shares"] = tabular(
            "lrrr", ["Policy", "Default", "Tuned", "Tuned + ensemble"], share_rows
        )
    else:
        has_error = "err_optimism" in by_family.columns
        rows = []
        for _, item in by_family.sort_values("optimism").iterrows():
            key = "".join(character for character in str(item["family"]).lower() if character.isalnum())
            assets.define(f"{prefix}.optim.family.{key}", num(item["optimism"], 1, signed=True))
            row = [
                str(item["family"]),
                num(item["val_rank"], 1),
                num(item["test_rank"], 1),
                f"{num(item['optimism'], 1, signed=True)} {interval(item['optimism_ci_low'], item['optimism_ci_high'], 1)}",
            ]
            if has_error:
                assets.define(f"{prefix}.erroptim.family.{key}", num(item["err_optimism"], 3, signed=True))
                row.append(
                    f"{num(item['err_optimism'], 3, signed=True)} "
                    f"{interval(item['err_optimism_ci_low'], item['err_optimism_ci_high'], 3)}"
                )
            rows.append(row)
        header = ["Family", "Validation rank", "Test rank", "Rank calibration [95\\% CI]"]
        if has_error:
            header.append("Error optimism [95\\% CI]")
        assets.tables[f"{prefix}_optimism"] = tabular("lrrr" + ("r" if has_error else ""), header, rows)
        spread = by_family["optimism"]
        assets.define(f"{prefix}.optim.range", f"{num(spread.min(), 1, signed=True)} to {num(spread.max(), 1, signed=True)}")
        if has_error:
            errors = by_family["err_optimism"]
            assets.define(
                f"{prefix}.erroptim.range",
                f"{num(errors.min(), 3, signed=True)} to {num(errors.max(), 3, signed=True)}",
            )
            # The family with by far the widest interval is reported apart from the range of
            # the others, so that one unstable estimate does not stand for all of them.
            width = by_family["err_optimism_ci_high"] - by_family["err_optimism_ci_low"]
            widest = by_family.loc[width.idxmax()]
            rest = by_family.drop(index=width.idxmax())
            assets.define(f"{prefix}.erroptim.widest", str(widest["family"]))
            assets.define(f"{prefix}.erroptim.widest.value", num(widest["err_optimism"], 3, signed=True))
            assets.define(
                f"{prefix}.erroptim.widest.ci",
                interval(widest["err_optimism_ci_low"], widest["err_optimism_ci_high"], 3),
            )
            assets.define(
                f"{prefix}.erroptim.range.rest",
                f"{num(rest['err_optimism'].min(), 3, signed=True)} to {num(rest['err_optimism'].max(), 3, signed=True)}",
            )
            assets.define(f"{prefix}.erroptim.range.rest.n", str(int(len(rest))))
            assets.define(f"{prefix}.erroptim.negative", str(int(errors.lt(0).sum())))
            assets.define(f"{prefix}.erroptim.families", str(int(len(by_family))))


def build_tie_rule(assets: Assets, prefix: str, directory: str) -> None:
    """Runtime-perturbed validation-best against the lexicographic tie-break (Table 5)."""
    row = assets.csv(f"{directory}/tie_rule_summary.csv").iloc[0]
    assets.define(f"sens.{prefix}.vblex.units", str(int(row["n_units"])))
    assets.define(f"sens.{prefix}.vblex.changed", str(int(row["units_with_other_pick"])))
    assets.define(f"sens.{prefix}.vblex.testchanged", str(int(row["units_with_other_test_error"])))
    assets.define(f"sens.{prefix}.vblex.datasets", str(int(row["datasets_affected"])))
    assets.define(f"sens.{prefix}.vblex.reg", num(row["regret_lexicographic"], 4))
    assets.define(f"sens.{prefix}.vb.regfour", num(row["regret_perturbed"], 4))
    assets.define(f"sens.{prefix}.vblex.diff", num(row["mean_difference"], 5, signed=True))


def build_factorial(assets: Assets, prefix: str, directory: str) -> None:
    shapley = assets.csv(f"{directory}/factorial_shapley_given_validation.csv")
    interactions = assets.csv(f"{directory}/factorial_interactions_given_validation.csv")
    cells = assets.csv(f"{directory}/factorial_cells.csv").set_index("cell")
    contribution = assets.csv(f"{directory}/factorial_validation_contribution.csv").iloc[0]
    names = {
        "task_family_prior": "tf",
        "global_method_prior": "gm",
        "runtime": "rt",
        "budget": "bud",
        "validation": "val",
    }
    labels = {
        "task_family_prior": "Task-family prior",
        "global_method_prior": "Global-method prior",
        "runtime": "Runtime term",
        "budget": "Fit-time budget",
    }
    rows = []
    for _, item in shapley.iterrows():
        key = names[item["factor"]]
        assets.define(f"{prefix}.shap.{key}", num(item["estimate"], 4, signed=True))
        assets.define(f"{prefix}.shap.{key}.ci", interval(item["ci_low"], item["ci_high"], 4))
        assets.define(f"{prefix}.shap.{key}.share", percent(item["share_of_total_reduction"]))
        rows.append([labels[item["factor"]], f"{num(item['estimate'], 4, signed=True)} {interval(item['ci_low'], item['ci_high'], 4)}", percent(item["share_of_total_reduction"])])
    for _, item in interactions.iterrows():
        first, second = str(item["pair"]).split(":")
        assets.define(f"{prefix}.int.{names[first]}.{names[second]}", num(item["estimate"], 4, signed=True))
        assets.define(f"{prefix}.int.{names[first]}.{names[second]}.ci", interval(item["ci_low"], item["ci_high"], 4))
    assets.define(f"{prefix}.valcontrib", num(contribution["estimate"], 4, signed=True))
    assets.define(f"{prefix}.valcontrib.ci", interval(contribution["ci_low"], contribution["ci_high"], 4))
    full = "validation+task_family_prior+global_method_prior+runtime+budget"
    keyed = {
        "full": full,
        "valonly": "validation",
        "nobudget": "validation+task_family_prior+global_method_prior+runtime",
        "priorsonly": "task_family_prior+global_method_prior+runtime+budget",
        "gmonly": "global_method_prior",
        "uniform": "none",
    }
    for key, cell in keyed.items():
        assets.define(f"{prefix}.cell.{key}", num(cells.loc[cell, "mean_normalized_rank_regret"], 4))
        assets.define(f"{prefix}.cell.{key}.ci", interval(cells.loc[cell, "ci_low"], cells.loc[cell, "ci_high"], 4))
    assets.tables[f"{prefix}_shapley"] = tabular(
        "lrr", ["Component", "Shapley value [95\\% CI]", "Share of total change"], rows
    )
    cell_rows = []
    for cell, item in cells.iterrows():
        present = [labels.get(name, "Validation") for name in names if item.get(f"has_{name}", 0)]
        cell_rows.append(
            [
                ", ".join(present) if present else "none (uniform draw)",
                f"({num(item['w_validation'], 2)}, {num(item['w_task_method_prior'], 2)}, {num(item['w_task_family_prior'], 2)}, {num(item['w_global_method_prior'], 2)}, {num(item['w_runtime'], 2)})",
                "10\\,000\\,s" if pd.notna(item["max_budget_seconds"]) else "none",
                f"{num(item['mean_normalized_rank_regret'], 4)} {interval(item['ci_low'], item['ci_high'], 4)}",
            ]
        )
    assets.tables[f"{prefix}_cells"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.34\\linewidth}lll",
        ["Components present", "$(w_v, w_{tm}, w_{tf}, w_{gm}, w_r)$", "Budget", "Norm.\\ rank regret [95\\% CI]"],
        cell_rows,
    )


def build_grid(assets: Assets) -> None:
    prefix = assets.csv("tabarena_grid_diagnostics/prefix_grid_sweep.csv")
    # The same sweep with the shared inner fit of the adaptive policies (Appendix J).
    shared = assets.csv("attribution/inner_shared_grid_diagnostics/prefix_grid_sweep.csv").set_index("n_policy_methods")
    rows = []
    for _, item in prefix.iterrows():
        count = int(item["n_policy_methods"])
        assets.define(f"ta.grid.k{count}.entropy", num(item["selection_entropy_bits"], 2))
        assets.define(f"ta.grid.k{count}.gap", num(item["mean_optimism_gap"], 4))
        assets.define(f"ta.grid.k{count}.distinct", str(int(item["n_selected_policies"])))
        assets.define(f"ta.grid.k{count}.rho", num(item["policy_to_train_dataset_ratio"], 2))
        shared_gap = shared.loc[count, "mean_optimism_gap"]
        assets.define(f"attr.inner.grid.k{count}.gap", num(shared_gap, 4))
        rows.append(
            [
                str(count),
                num(item["policy_to_train_dataset_ratio"], 2),
                str(int(item["n_selected_policies"])),
                num(item["selection_entropy_bits"], 2),
                num(item["mean_optimism_gap"], 4),
                num(shared_gap, 4),
            ]
        )
    assets.tables["tabarena_prefix"] = tabular(
        "rrrrrr",
        ["$K$", "$K/38$", "Distinct selected policies", "Entropy (bits)", "Optimism gap", "Gap, shared inner fit"],
        rows,
    )
    objective = assets.csv("tabarena_grid_diagnostics/objective_sensitivity.csv")
    labels = {
        "regret_runtime_order": "Regret, then runtime, then order (declared)",
        "regret_order": "Regret, then order",
        "regret_runtime_reverse_order": "Regret, then runtime, then reverse order",
        "dataset_weighted_regret_runtime_order": "Dataset-weighted regret, then runtime, then order",
    }
    rows = []
    for _, item in objective.iterrows():
        rule = str(item["rule"])
        rows.append(
            [
                labels[rule],
                f"{num(item['mean_normalized_rank_regret'], 4)} {interval(item['ci_low'], item['ci_high'], 4)}",
                f"{num(item['difference_vs_declared_rule'], 4, signed=True)} {interval(item['difference_ci_low'], item['difference_ci_high'], 4)}",
                str(int(item["n_selected_policies"])),
                str(int(item["splits_with_same_choice_as_declared"])),
            ]
        )
        # Section 6.4 states the tie-break results in the text (the table is in the outputs).
        if rule in {"regret_order", "regret_runtime_reverse_order"}:
            key = "noruntime" if rule == "regret_order" else "reverse"
            assets.define(f"ta.obj.{key}.same", str(int(item["splits_with_same_choice_as_declared"])))
            assets.define(f"ta.obj.{key}.diff", num(item["difference_vs_declared_rule"], 4, signed=True))
        if rule == "dataset_weighted_regret_runtime_order":
            assets.define("ta.obj.dw.reg", num(item["mean_normalized_rank_regret"], 4))
            assets.define("ta.obj.dw.diff", num(item["difference_vs_declared_rule"], 4, signed=True))
            assets.define("ta.obj.dw.diffci", interval(item["difference_ci_low"], item["difference_ci_high"], 4))
            assets.define("ta.obj.dw.same", str(int(item["splits_with_same_choice_as_declared"])))
    assets.tables["tabarena_objective"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.36\\linewidth}rrrr",
        ["Nested choice rule", "Norm.\\ rank regret [95\\% CI]", "Difference vs declared [95\\% CI]", "Distinct policies", "Same choice"],
        rows,
    )


def build_search(assets: Assets, prefix: str, directory: str) -> None:
    mcs = assets.csv(f"{directory}/model_confidence_set.csv")
    spa = assets.csv(f"{directory}/superior_predictive_ability.csv")
    pbo = assets.csv(f"{directory}/probability_of_backtest_overfitting.csv").iloc[0]
    assets.define(f"{prefix}.mcs.size", str(int(mcs["in_mcs_90"].sum())))
    assets.define(f"{prefix}.mcs.sizeseventyfive", str(int(mcs["in_mcs_75"].sum())))
    assets.define(f"{prefix}.mcs.rows", str(int(mcs.shape[0])))
    excluded = mcs[~mcs["in_mcs_90"]]["label"].tolist()
    assets.define(f"{prefix}.mcs.excluded", "; ".join(excluded) if excluded else "none")
    for method in ["meta_selector_single", "nested_policy_selected", "meta_selector_family_prior_budgeted"]:
        if method in set(mcs["policy"]):
            assets.define_p(f"{prefix}.mcs.p.{POLICY_KEYS[method]}", pvalue(float(mcs.set_index("policy").loc[method, "mcs_p_value"])))
    for _, item in spa.iterrows():
        key = POLICY_KEYS.get(str(item["benchmark"]))
        if key is None:
            continue
        assets.define_p(f"{prefix}.spa.{key}.c", pvalue(item["spa_p_consistent"]))
        assets.define_p(f"{prefix}.spa.{key}.l", pvalue(item["spa_p_lower"]))
        assets.define_p(f"{prefix}.spa.{key}.u", pvalue(item["spa_p_upper"]))
        assets.define_p(f"{prefix}.spa.{key}.rc", pvalue(item["reality_check_p"]))
        assets.define(f"{prefix}.spa.{key}.best", policy_label(item["best_alternative"]))
        assets.define(f"{prefix}.spa.{key}.gain", num(item["best_mean_improvement"], 4))
    assets.define(f"{prefix}.pbo", percent(pbo["pbo"]))
    assets.define(f"{prefix}.pbo.degradation", num(pbo["mean_performance_degradation"], 4, signed=True))
    assets.define(f"{prefix}.pbo.combinations", str(int(pbo["n_combinations"])))


def build_evidence(assets: Assets, prefix: str, directory: str) -> None:
    frame = assets.csv(f"{directory}/evidence_unit_exhibit.csv")
    unit_keys = {
        "outer split (overlapping, treated as independent)": "split",
        "unit appearance (all held-out units in all splits)": "unit",
        "dataset (declared)": "dataset",
    }
    rows = []
    for _, item in frame.iterrows():
        unit = unit_keys[item["evidence_unit"]]
        right = POLICY_KEYS[str(item["right"])]
        base = f"{prefix}.eu.{right}.{unit}"
        assets.define(f"{base}.diff", num(item["mean_difference"], 4, signed=True))
        assets.define_p(f"{base}.p", pvalue(item["wilcoxon_p"]))
        assets.define(f"{base}.n", f"{int(item['n']):,}".replace(",", "\\,"))
        clustered = item.get("p_datasets_resampled", float("nan"))
        if pd.notna(clustered):
            # The row's own estimate tested with whole datasets resampled (the mean test).
            assets.define_p(f"{base}.pclus", pvalue(clustered))
        # Verdicts the text states about evidence units (STATED_VERDICTS): the Wilcoxon test
        # of the row, with its direction taken from the mean difference, and the mean test
        # with whole datasets resampled.
        separated = float(item["wilcoxon_p"]) < ALPHA
        verdict = {
            "wilcoxon_any": "separated" if separated else "not separated",
            "wilcoxon": ("better" if item["mean_difference"] < 0 else "worse") if separated else "not separated",
        }
        if pd.notna(clustered):
            verdict["clustered"] = "separated" if float(clustered) < ALPHA else "not separated"
        assets.verdicts[base] = verdict
        rows.append(
            [
                policy_label(item["right"]),
                {"split": "Outer split", "unit": "Unit appearance", "dataset": "Dataset (declared)"}[unit],
                f"{int(item['n']):,}".replace(",", "\\,"),
                num(item["mean_difference"], 4, signed=True),
                f"{int(item['wins'])}/{int(item['losses'])}",
                pvalue(item["wilcoxon_p"]),
                pvalue(clustered) if pd.notna(clustered) else "--",
            ]
        )
    assets.tables[f"{prefix}_evidence"] = tabular(
        "llrrrrr",
        ["Reference", "Evidence unit", "$n$", "Nested $-$ reference", "Wins/losses", "Wilcoxon $p$", "$p$, datasets resampled"],
        rows,
    )


def build_small_runs(assets: Assets) -> None:
    extended = assets.csv("score_level_tabarena_extended_pool_review/clustered_summary.csv").set_index("method")
    extended_contrasts = assets.csv("score_level_tabarena_extended_pool_review/contrasts.csv")
    assets.define("ta.ext.nested.reg", num(extended.loc["nested_policy_selected", "mean_normalized_rank_regret"], 3))
    assets.define(
        "ta.ext.nested.regci",
        interval(
            extended.loc["nested_policy_selected", "mean_normalized_rank_regret_ci_low"],
            extended.loc["nested_policy_selected", "mean_normalized_rank_regret_ci_high"],
        ),
    )
    for method in ["extended_metafeature_knn", "extended_metafeature_knn_validation", "extended_descriptor_guard_002", "extended_descriptor_guard_005"]:
        assets.define(f"ta.ext.{POLICY_KEYS[method]}.reg", num(extended.loc[method, "mean_normalized_rank_regret"], 4))
    contrast_numbers(assets, "ta.ext", extended_contrasts)
    selections = assets.csv("score_level_tabarena_extended_pool/split_results.csv")
    extra = selections["selected_policy"].str.startswith("extended_").sum()
    assets.define("ta.ext.extrachosen", str(int(extra)))
    shrink = assets.csv("score_level_tabarena_shrink_review/clustered_summary.csv").set_index("method")
    assets.define("ta.shrink.nested.reg", num(shrink.loc["nested_policy_selected", "mean_normalized_rank_regret"], 4))
    assets.define(
        "ta.shrink.nested.regci",
        interval(
            shrink.loc["nested_policy_selected", "mean_normalized_rank_regret_ci_low"],
            shrink.loc["nested_policy_selected", "mean_normalized_rank_regret_ci_high"],
            4,
        ),
    )
    contrast_numbers(assets, "ta.shrink", assets.csv("score_level_tabarena_shrink_review/contrasts.csv"))
    shrink_grid = assets.csv("score_level_tabarena_shrink_grid_diagnostics/prefix_grid_sweep.csv").iloc[-1]
    assets.define("ta.shrink.entropy", num(shrink_grid["selection_entropy_bits"], 2))
    assets.define("ta.shrink.gap", num(shrink_grid["mean_optimism_gap"], 4))
    assets.define("ta.shrink.rho", num(shrink_grid["policy_to_train_dataset_ratio"], 2))


def build_tabrepo(assets: Assets) -> None:
    for prefix, directory in [("tr", "score_level_tabrepo_nested_review"), ("trs", "score_level_tabrepo_score_only_review")]:
        summary = assets.csv(f"{directory}/clustered_summary.csv")
        summary_numbers(assets, prefix, summary, rank_scale=1529.0)
        contrast_numbers(assets, prefix, assets.csv(f"{directory}/contrasts.csv"))
        counts = assets.csv(f"{directory}/train_selected_reference_counts.csv")
        for _, item in counts.iterrows():
            key = POLICY_KEYS.get(str(item["selected_reference_method"]))
            if key:
                assets.define(f"{prefix}.choice.{item['pool']}.{key}", str(int(item["n_splits"])))
    # Nested selection against fixed ensemble-3 (Section 7): the share of the summed
    # dataset-level loss contributed by the five datasets with the largest losses.
    means = assets.csv("score_level_tabrepo_nested_review/dataset_mean_scores.csv")
    wide = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    loss = (wide["nested_policy_selected"] - wide["meta_selector_tabrepo_ensemble_k3"]).sort_values(ascending=False)
    assets.define("tr.nested.ethree.topfiveshare", percent(float(loss.head(5).sum() / loss.sum())))
    nested_splits = assets.csv("score_level_tabrepo_nested/split_results.csv")
    assets.define("tr.splits", str(int(nested_splits.shape[0])))
    assets.define("tr.samerow", str(same_row_splits(assets, "score_level_tabrepo_nested", "score_level_tabrepo_nested_review")))
    nested_counts = nested_splits["selected_policy"].value_counts()
    for method, count in nested_counts.items():
        key = POLICY_KEYS.get(str(method))
        if key:
            assets.define(f"tr.nestedchoice.{key}", str(int(count)))
    ens = assets.csv("score_level_tabrepo_nested_review/clustered_summary.csv").set_index("method")
    score = assets.csv("score_level_tabrepo_score_only_review/clustered_summary.csv").set_index("method")
    # The ensembles' lead over validation-best relative to validation-best's own regret, in
    # error regret (which can credit outputs better than every candidate) and in rank regret.
    ensemble, validation = ens.loc["meta_selector_tabrepo_ensemble_k3"], ens.loc["meta_selector_single"]
    assets.define("tr.ethree.err.rel", percent(1.0 - float(ensemble["mean_maxmin_error_regret"]) / float(validation["mean_maxmin_error_regret"])))
    assets.define("tr.ethree.reg.rel", percent(1.0 - float(ensemble["mean_normalized_rank_regret"]) / float(validation["mean_normalized_rank_regret"])))
    # Four decimals, so that the printed pair reproduces the relative difference.
    assets.define("tr.ethree.errfour", num(float(ensemble["mean_maxmin_error_regret"]), 4))
    assets.define("tr.vb.errfour", num(float(validation["mean_maxmin_error_regret"]), 4))

    def line(frame: pd.DataFrame, method: str, label: str | None = None) -> list[str]:
        item = frame.loc[method]
        return [
            label or policy_label(method),
            f"{num(item['mean_normalized_rank_regret'], 4)} {interval(item['mean_normalized_rank_regret_ci_low'], item['mean_normalized_rank_regret_ci_high'], 4)}",
            num(item["mean_maxmin_error_regret"], 4),
            percent(item["beats_best_rate"], 1),
            seconds(item["mean_total_seconds"]),
        ]

    rows = [
        line(ens, "nested_policy_selected"),
        line(ens, "train_selected_fixed_reference__ensemble_inclusive"),
        line(ens, "meta_selector_tabrepo_ensemble_k3"),
        line(ens, "meta_selector_tabrepo_ensemble_k5"),
        line(ens, "meta_selector_tabrepo_ensemble_k10"),
        line(score, "nested_policy_selected"),
        line(score, "train_selected_fixed_reference__score_only"),
        line(score, "meta_selector_single"),
        line(score, "meta_selector_ablate_validation_only"),
        line(score, "meta_selector_family_prior_budgeted"),
        line(score, "meta_selector_task_adaptive_budgeted"),
        line(score, "reference_optimism_corrected"),
        line(score, "reference_single_best_global"),
        line(score, "oracle"),
    ]
    assets.tables["tabrepo_main"] = tabular(
        "lrrrr",
        ["Policy", "Norm.\\ rank regret [95\\% CI]", "Error regret", "Outperforms all", "Runtime (s)"],
        rows,
        groups={0: "Ensemble-inclusive pool", 5: "Score-only pool (ensemble rows removed before the nested choice)"},
    )


# The review's resampling seed (configs/review_*.yaml); build_mean_tests checks that it
# reproduces the stored bootstrap p-values.
REVIEW_SEED = 20260608
NUMERIC_IDENTITY = ["mean_difference", "ci_low", "ci_high", "bootstrap_p", "ttest_p"]


def mean_test_variants(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    """Unadjusted two-sided p-values of four tests of a zero mean on the same resamples: the
    percentile test (the declared separation test), an equal-tailed and a symmetric
    bootstrap-t test, and the paired t-test."""
    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return {"perc": 1.0, "bteq": 1.0, "btsym": 1.0, "t": 1.0}
    resampled = values[indices]
    boot = resampled.mean(axis=1)
    t_obs = mean / (sd / math.sqrt(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = (boot - mean) / (resampled.std(axis=1, ddof=1) / math.sqrt(n))
    t_star = t_star[np.isfinite(t_star)]
    return {
        "perc": bootstrap_mean_p(boot),
        "bteq": float(min(1.0, 2.0 * min(np.mean(t_star <= t_obs), np.mean(t_star >= t_obs)))),
        "btsym": float(np.mean(np.abs(t_star) >= abs(t_obs))),
        "t": float(2.0 * stats.t.sf(abs(t_obs), n - 1)),
    }


def build_mean_tests(assets: Assets) -> None:
    """Section 3.4 and Appendix A: where the declared bootstrap test and the paired t-test of
    the same mean give different Holm-adjusted verdicts, which one separates and why, and the
    Monte Carlo spread of the headline p-value over other resample matrices."""
    runs = {
        "main": "score_level_tabarena_review",
        "ext": "score_level_tabarena_extended_pool_review",
        "trs": "score_level_tabrepo_score_only_review",
        "tre": "score_level_tabrepo_nested_review",
    }
    differ: dict[str, pd.DataFrame] = {}
    for key, directory in runs.items():
        contrasts = assets.csv(f"{directory}/contrasts.csv")
        split = (contrasts["bootstrap_holm_p"] < ALPHA) != (contrasts["ttest_holm_p"] < ALPHA)
        differ[key] = contrasts[split].drop_duplicates(subset=NUMERIC_IDENTITY)
    assets.define("ta.mtests.differ.main", str(len(differ["main"])))
    # The same comparison over every contrast file the reported runs wrote, the rescored and
    # reapplied stages, the overlap checks and the convention runs included.
    # The matched-family comparison of the pools adds each pool's contrasts in a three- and a
    # five-member family. A contrast that two runs share with identical numbers, such as the
    # family-prior rule against validation-best, which no inner fit changes, counts once.
    frames = []
    for path in sorted(assets.out.rglob("*contrasts.csv")):
        frame = pd.read_csv(path)
        if {"bootstrap_holm_p", "ttest_holm_p"} <= set(frame.columns):
            frames.append(frame[["left", "right", "mean_difference", "ci_low", "ci_high", "bootstrap_holm_p", "ttest_holm_p"]].assign(source=str(path)))
        elif {"bootstrap_holm_five", "ttest_holm_five"} <= set(frame.columns):
            for size in ["three", "five"]:
                frames.append(
                    frame[["left", "right", "mean_difference", "ci_low", "ci_high", f"bootstrap_holm_{size}", f"ttest_holm_{size}"]]
                    .rename(columns={f"bootstrap_holm_{size}": "bootstrap_holm_p", f"ttest_holm_{size}": "ttest_holm_p"})
                    .assign(source=f"{path}:{size}")
                )
    everything = pd.concat(frames, ignore_index=True)
    differing = everything[(everything["bootstrap_holm_p"] < ALPHA) != (everything["ttest_holm_p"] < ALPHA)].copy()
    # Distinct by their numbers, as for the main run: a contrast tested in two families, or
    # against validation rank, which on TabArena chooses as validation-best does, counts once.
    for column in ["mean_difference", "ci_low", "ci_high"]:
        differing[f"_{column}"] = differing[column].round(10)
    differing = differing.drop_duplicates(subset=["_mean_difference", "_ci_low", "_ci_high"])
    assets.define("mtests.differ.all", str(len(differing)))
    assets.define("mtests.differ.onlyboot", str(int((differing["bootstrap_holm_p"] < ALPHA).sum())))
    assets.define("mtests.differ.runs", str(int(differing["source"].nunique())))
    assets.define("ta.mtests.differ.ext", str(len(differ["ext"])))
    assets.define("tr.mtests.differ", str(len(differ["trs"]) + len(differ["tre"])))
    tabarena = pd.concat([differ["main"].assign(run="main"), differ["ext"].assign(run="ext")])
    assets.define("ta.mtests.onlyt", str(int((tabarena["ttest_holm_p"] < ALPHA).sum())))
    # Each differing contrast: is it skewed toward large gains, and do the tests that follow
    # the skew (percentile, equal-tailed bootstrap-t) give smaller p-values than the
    # symmetric ones (paired t, symmetric bootstrap-t)?
    ordered = 0
    for _, row in tabarena.iterrows():
        means = assets.csv(f"{runs[row['run']]}/dataset_mean_scores.csv")
        wide = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
        diff = (wide[row["left"]] - wide[row["right"]]).dropna()
        resampler = ArtifactResampler(list(diff.index), seed=REVIEW_SEED)
        values = resampler.align(diff)
        p = mean_test_variants(values, resampler.indices)
        if abs(p["perc"] - float(row["bootstrap_p"])) > 1e-12:
            raise ValueError(f"The resample matrix does not reproduce the stored bootstrap p of {row['left']} - {row['right']}")
        if stats.skew(values) < 0 and max(p["perc"], p["bteq"]) < min(p["t"], p["btsym"]):
            ordered += 1
    assets.define("ta.mtests.skewordered", str(ordered))
    # The headline contrast: skew, and Holm-adjusted p-values of the four tests within the
    # five-member primary family.
    contrasts = assets.csv(f"{runs['main']}/contrasts.csv")
    primary = contrasts[contrasts["family"].eq("primary")]
    means = assets.csv(f"{runs['main']}/dataset_mean_scores.csv")
    wide = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")
    diffs = [(wide[row["left"]] - wide[row["right"]]).dropna() for _, row in primary.iterrows()]
    position = list(primary["right"]).index("meta_selector_single")
    resampler = ArtifactResampler(list(diffs[position].index), seed=REVIEW_SEED)
    variants = [mean_test_variants(resampler.align(diff), resampler.indices) for diff in diffs]
    for name in ["perc", "bteq", "btsym", "t"]:
        adjusted = holm([variant[name] for variant in variants])[position]
        assets.define_p(f"ta.mtests.vb.{name}holm", pvalue(adjusted))
    assets.define("ta.mtests.vb.skew", num(float(stats.skew(resampler.align(diffs[position]))), 2, signed=True))
    # Monte Carlo spread: the same family over other resample matrices.
    seeds = list(range(1, 51))
    adjusted = []
    for seed in seeds:
        other = ArtifactResampler(list(diffs[position].index), seed=seed)
        raw = [bootstrap_mean_p(other.align(diff)[other.indices].mean(axis=1)) for diff in diffs]
        adjusted.append(holm(raw)[position])
    assets.define("ta.mc.vb.seeds", str(len(seeds)))
    assets.define("ta.mc.vb.holmmin", pvalue(min(adjusted)))
    assets.define("ta.mc.vb.holmmax", pvalue(max(adjusted)))
    assets.define("ta.mc.vb.below", str(int(sum(value < ALPHA for value in adjusted))))


def build_misc(assets: Assets) -> None:
    pooled = assets.csv("pooled_cross_artifact/pooled_contrasts.csv").set_index("scope")
    # Section 7 pools the datasets the two artifacts do not share: without the TabRepo datasets
    # that share a name with a TabArena dataset or together make one up ("distinct" drops only
    # the names).
    for scope, key in [("Pooled, no shared datasets", "disjoint"), ("Pooled, distinct datasets", "distinct"), ("Pooled, naive stack", "naive"), ("TabArena", "ta"), ("TabRepo score-only", "trs")]:
        item = pooled.loc[scope]
        base = f"pool.{key}"
        assets.define(f"{base}.n", str(int(item["n_datasets"])))
        assets.define(f"{base}.diff", num(item["mean_difference"], 4, signed=True))
        assets.define(f"{base}.ci", interval(item["ci_low"], item["ci_high"], 4))
        assets.define(f"{base}.wins", str(int(item["wins"])))
        assets.define(f"{base}.losses", str(int(item["losses"])))
        assets.define_p(f"{base}.sign", pvalue(item["sign_test_p"]))
        assets.define(f"{base}.mde", num(item["mde80_bootstrap"], 4))
    half = assets.csv("tabrepo_split_half/split_half_summary.csv")
    keys = {
        "Ensemble-inclusive nested": "nested",
        "Fixed ensemble k=3": "ethree",
        "Raw validation": "vb",
        "Score-only nested": "snested",
        "Train-selected fixed rule (score-only)": "tsfs",
    }
    for _, item in half.iterrows():
        assets.define(f"half.{item['half'][-1].lower()}.{keys[item['policy']]}", num(item["mean_normalized_rank_regret"], 3))
    compact = assets.csv("third_artifact_validation/method_summary.csv").set_index("method")
    paired = assets.csv("third_artifact_validation/paired_contrasts.csv").set_index("endpoint")
    for method, key in [("meta_selector_local_ensemble", "ens"), ("meta_selector_local_validation", "vb")]:
        item = compact.loc[method]
        assets.define(f"compact.{key}.reg", num(item["mean_normalized_rank_regret"], 3))
        assets.define(f"compact.{key}.regci", interval(item["mean_normalized_rank_regret_ci_low"], item["mean_normalized_rank_regret_ci_high"], 3))
        assets.define(f"compact.{key}.metric", num(item["mean_primary_metric"], 5))
    for endpoint, key in [("normalized_rank_regret", "reg"), ("primary_metric", "metric")]:
        item = paired.loc[endpoint]
        digits = 4 if key == "reg" else 5
        assets.define(f"compact.diff.{key}", num(item["mean_method_minus_reference"], digits, signed=True))
        assets.define(f"compact.diff.{key}.ci", interval(item["ci95_low"], item["ci95_high"], digits))
        # Holm over the two endpoints: the mean test that inverts the interval (the
        # separation verdict) and the signed-rank test.
        assets.define_p(f"compact.diff.{key}.mholm", pvalue(item["bootstrap_holm_p_value"]))
        assets.define_p(f"compact.diff.{key}.rholm", pvalue(item["holm_p_value"]))
        assets.define_p(f"compact.diff.{key}.sign", pvalue(item["sign_test_p_value"]))
        assets.define(f"compact.diff.{key}.wtl", f"{int(item['wins_method_lower'])}/{int(item['ties'])}/{int(item['losses_method_higher'])}")


def build_overlap(assets: Assets) -> None:
    """TabRepo contrasts without the datasets that TabArena also contains."""
    contrasts = assets.csv("tabrepo_overlap_sensitivity/overlap_contrasts.csv")
    metadata_path = assets.out / "tabrepo_overlap_sensitivity" / "metadata.json"
    pooled_path = assets.out / "pooled_cross_artifact" / "metadata.json"
    for path in [metadata_path, pooled_path]:
        if not path.exists():
            raise Missing(str(path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
    if int(metadata["n_overlap_by_name"]) != int(pooled["n_overlapping_datasets"]):
        raise ValueError("The overlap sensitivity and the pooled diagnostic disagree on the shared datasets")
    if sorted(pooled.get("excluded_tabrepo_datasets", [])) != sorted(metadata["excluded"]):
        raise ValueError("The pooled diagnostic and the overlap sensitivity exclude different TabRepo datasets")
    kept = {int(panel["n_kept"]) for panel in metadata["panels"].values()}
    if len(kept) != 1:
        raise ValueError("The TabRepo panels kept different numbers of datasets")
    # Section 7 says these contrasts keep their sign, interval status and verdicts (the mean
    # test and the signed-rank test, Holm-adjusted) without the shared datasets, except that
    # the interval of the correction contrast, which it reports, may change. Refuse to build
    # otherwise.
    discussed = [
        ("ensemble", "capability", "train_selected_fixed_reference__ensemble_inclusive", "train_selected_fixed_reference__score_only"),
        ("ensemble", "fixed_ensembles", "nested_policy_selected", "meta_selector_tabrepo_ensemble_k3"),
        ("ensemble", "primary", "nested_policy_selected", "train_selected_fixed_reference__score_only"),
        ("ensemble", "mechanism", "meta_selector_single", "meta_selector_family_prior_budgeted"),
        ("ensemble", "mechanism", "meta_selector_single", "reference_single_best_global"),
        ("score_only", "primary", "nested_policy_selected", "meta_selector_single"),
        ("score_only", "primary", "nested_policy_selected", "train_selected_fixed_reference__score_only"),
        ("score_only", "score_level_rows", "nested_policy_selected", "meta_selector_task_adaptive_budgeted"),
        ("score_only", "score_level_rows", "nested_policy_selected", "meta_selector_meta_feature_guard_005"),
    ]
    correction = ("ensemble", "mechanism", "meta_selector_single", "reference_optimism_corrected")
    full_reviews = {
        "ensemble": assets.csv("score_level_tabrepo_nested_review/contrasts.csv"),
        "score_only": assets.csv("score_level_tabrepo_score_only_review/contrasts.csv"),
    }
    keyed = contrasts.set_index(["panel", "family", "left", "right"])
    for key in [*discussed, correction]:
        if key not in keyed.index:
            raise ValueError(f"Overlap sensitivity lacks the contrast {key}")
        row = keyed.loc[key]
        if key != correction:
            status = (np.sign(row["mean_difference"]), row["ci_low"] > 0 or row["ci_high"] < 0)
            full = (np.sign(row["full_mean_difference"]), row["full_ci_low"] > 0 or row["full_ci_high"] < 0)
            if status != full:
                raise ValueError(f"Without the shared datasets, {key} changes sign or interval status; revise Section 7")
        full_row = full_reviews[key[0]].set_index(["family", "left", "right"]).loc[key[1:]]
        for column in ["bootstrap_holm_p", "wilcoxon_holm_p"]:
            if (float(row[column]) < ALPHA) != (float(full_row[column]) < ALPHA):
                raise ValueError(f"Without the shared datasets, {key} changes its {column} verdict; revise Section 7")
    assets.define("ovl.byname", str(int(metadata["n_overlap_by_name"])))
    assets.define("ovl.related", str(len(metadata["related"])))
    assets.define("ovl.excluded", str(int(metadata["n_excluded"])))
    assets.define("ovl.n", str(kept.pop()))
    contrast_numbers(assets, "ovl.tr", contrasts[contrasts["panel"].eq("ensemble")])
    contrast_numbers(assets, "ovl.trs", contrasts[contrasts["panel"].eq("score_only")])


def build_attribution(assets: Assets) -> None:
    """Sensitivity to the two implementation conventions that move numbers (Appendix J)."""
    runs = {
        "vone": "attribution/legacy_v1_review",
        "rankfix": "attribution/rank_fix_only_review",
        "inner": "attribution/inner_shared_review",
        "vtwo": "score_level_tabarena_review",
    }
    frames = {key: assets.csv(f"{directory}/clustered_summary.csv").set_index("method") for key, directory in runs.items()}
    contrasts = {key: assets.csv(f"{directory}/contrasts.csv") for key, directory in runs.items()}
    methods = [
        "meta_selector_family_prior_budgeted",
        "train_selected_fixed_reference__declared_fixed",
        "nested_policy_selected",
        "meta_selector_task_adaptive_budgeted",
        "meta_selector_calibrated_budget_10000s",
        "meta_selector_meta_feature_guard_005",
        "meta_selector_single",
        "oracle",
    ]
    rows = []
    for method in methods:
        rows.append([policy_label(method), *[num(frames[key].loc[method, "mean_normalized_rank_regret"], 4) for key in runs]])
    for right in ["meta_selector_family_prior_budgeted", "train_selected_fixed_reference__declared_fixed", "meta_selector_single"]:
        values = []
        for key in runs:
            item = contrasts[key][contrasts[key]["family"].eq("primary") & contrasts[key]["right"].eq(right)].iloc[0]
            values.append(num(item["mean_difference"], 4, signed=True))
        rows.append([f"Nested $-$ {policy_label(right)}", *values])
    for column, label in [("bootstrap_holm_p", "mean"), ("wilcoxon_holm_p", "signed-rank")]:
        holm = []
        for key in runs:
            item = contrasts[key][contrasts[key]["family"].eq("primary") & contrasts[key]["right"].eq("meta_selector_single")].iloc[0]
            holm.append(pvalue(item[column]) if column in item and pd.notna(item[column]) else "--")
        rows.append([f"Holm $p$ ({label}), nested $-$ {policy_label('meta_selector_single')}", *holm])
    # The equivalence verdicts of Section 6.3 under each convention.
    for right, margin, text in [
        ("meta_selector_family_prior_budgeted", "0p015", "0.015"),
        ("train_selected_fixed_reference__declared_fixed", "0p010", "0.010"),
    ]:
        values = []
        for key in runs:
            item = contrasts[key][contrasts[key]["family"].eq("primary") & contrasts[key]["right"].eq(right)].iloc[0]
            values.append(pvalue(item[f"tost_p_{margin}"]))
        rows.append([f"TOST $p$ at $\\pm{text}$, nested $-$ {policy_label(right)}", *values])
    # Keys and verdicts of the primary contrasts in every column (Appendix J).
    for key in runs:
        primary = contrasts[key][contrasts[key]["family"].eq("primary")]
        for _, item in primary.iterrows():
            right = POLICY_KEYS.get(str(item["right"]))
            if right not in {"vb", "fp", "tsf"}:
                continue
            base = f"attr.{key}.{right}"
            assets.verdicts[base] = contrast_verdicts(item)
            if "bootstrap_holm_p" in item and pd.notna(item["bootstrap_holm_p"]):
                assets.define_p(f"{base}.mholm", pvalue(item["bootstrap_holm_p"]))
            if "ttest_holm_p" in item and pd.notna(item["ttest_holm_p"]):
                assets.define_p(f"{base}.tholm", pvalue(item["ttest_holm_p"]))
            for margin in ["0p010", "0p015"]:
                assets.define_p(f"{base}.tost{margin.replace('0p', '')}", pvalue(item[f"tost_p_{margin}"]))
    assets.tables["attribution"] = tabular(
        "lrrrr",
        [
            "Row",
            "Double-counting ties, in-sample weights",
            "Consistent ties, in-sample weights",
            "Reported, shared inner fit",
            "Reported conventions",
        ],
        rows,
    )
    policies = [method for method in methods if method != "oracle"]

    def largest_change(first: str, second: str) -> float:
        return float(
            (frames[first].loc[policies, "mean_normalized_rank_regret"] - frames[second].loc[policies, "mean_normalized_rank_regret"])
            .abs()
            .max()
        )

    assets.define("attr.rankfix.maxchange", num(largest_change("rankfix", "vone"), 4))
    assets.define("attr.lodo.maxchange", num(largest_change("vtwo", "rankfix"), 4))
    # The inner fit of the adaptive policies: shared source rows against per-target rows.
    assets.define("attr.inner.maxchange", num(largest_change("vtwo", "inner"), 4))
    for method in ["nested_policy_selected", "meta_selector_meta_feature_guard_005", "meta_selector_task_adaptive_budgeted"]:
        assets.define(
            f"attr.inner.{POLICY_KEYS[method]}.reg",
            num(frames["inner"].loc[method, "mean_normalized_rank_regret"], 4),
        )
    # Held-out minus training regret under both inner fits: under the shared fit the inner
    # targets enter the guards' training rows, which then look better in training.
    _inner_fit_gaps(assets, "attr.inner", [("shared", "attribution/inner_shared"), ("pertarget", "score_level_tabarena")])
    _check_inner_fit_rows(
        assets,
        "attribution/inner_shared_review",
        "score_level_tabarena_review",
        {"nested_policy_selected", "train_selected_fixed_reference__broader"},
    )
    _adaptive_choices(assets, "attr.inner.adaptivechosen", "attribution/inner_shared")
    _adaptive_choices(assets, "attr.vtwo.adaptivechosen", "score_level_tabarena")
    assets.define("attr.rankfix.oracle", num(frames["rankfix"].loc["oracle", "mean_normalized_rank_regret"], 4))
    # TabRepo fixed ensembles re-ranked under both tie rules (scripts/rank_rule_sensitivity.py).
    sensitivity = assets.csv("tabrepo_rank_rule_sensitivity/rank_rule_sensitivity.csv").set_index("method")
    for method in ["meta_selector_tabrepo_ensemble_k3", "meta_selector_tabrepo_ensemble_k5", "meta_selector_tabrepo_ensemble_k10"]:
        item = sensitivity.loc[method]
        base = f"sens.tr.{POLICY_KEYS[method]}"
        assets.define(f"{base}.consistent", num(item["consistent_mean_normalized_rank_regret"], 4))
        assets.define(f"{base}.legacy", num(item["legacy_mean_normalized_rank_regret"], 4))
        assets.define(f"{base}.changed", percent(item["share_units_rank_changed"]))


ADAPTIVE_PREFIXES = ("meta_selector_task_adaptive", "meta_selector_meta_feature_guard")


def _inner_fit_gaps(assets: Assets, prefix: str, runs: list[tuple[str, str]]) -> None:
    """Held-out minus training regret of the guards, the family-prior rows and validation-best,
    averaged over splits, under each inner fit; ``rel`` is the guards' gap minus the family-prior
    rows' gap, since every row's gap shares a design component."""
    for key, directory in runs:
        columns = ["split_id", "method", "mean_normalized_rank_regret"]
        training = assets.csv(f"{directory}/training_policy_scores.csv")[columns]
        heldout = assets.csv(f"{directory}/heldout_policy_scores.csv")[columns]
        merged = training.merge(heldout, on=["split_id", "method"], suffixes=("_train", "_heldout"))
        gap = merged["mean_normalized_rank_regret_heldout"] - merged["mean_normalized_rank_regret_train"]
        method = merged["method"].astype(str)
        means = {}
        for name, mask in [
            ("guard", method.str.startswith("meta_selector_meta_feature_guard")),
            ("fp", method.str.startswith("meta_selector_family_prior_budgeted")),
            ("vb", method.eq("meta_selector_single")),
        ]:
            means[name] = float(gap[mask].mean())
            assets.define(f"{prefix}.gap.{key}.{name}", num(means[name], 4, signed=True))
        assets.define(f"{prefix}.gap.{key}.rel", num(means["guard"] - means["fp"], 4, signed=True))


def _adaptive_choices(assets: Assets, key: str, directory: str) -> None:
    """How many outer splits' nested choices are adaptive policies."""
    selected = assets.csv(f"{directory}/split_results.csv")["selected_policy"].astype(str)
    assets.define(key, str(int(selected.str.startswith(ADAPTIVE_PREFIXES).sum())))


def _check_inner_fit_rows(assets: Assets, shared: str, reported: str, allowed: set[str]) -> None:
    """Appendix J says the inner fit leaves every held-out policy row unchanged and moves only
    rows chosen by training regret. Refuse to build otherwise."""
    a = assets.csv(f"{shared}/clustered_summary.csv").set_index("method")["mean_normalized_rank_regret"]
    b = assets.csv(f"{reported}/clustered_summary.csv").set_index("method")["mean_normalized_rank_regret"]
    common = a.index.intersection(b.index)
    changed = {str(method) for method in common if abs(float(a[method]) - float(b[method])) > 1e-12}
    if not changed <= allowed:
        raise ValueError(f"The inner fit also moves {sorted(changed - allowed)}; revise Appendix J")


def build_attribution_tabrepo(assets: Assets) -> None:
    """The inner fit of the adaptive policies on TabRepo (Appendix J): the shared fit against
    the reported per-target fit, in the score-only panel, where the adaptive policies compete
    without ensemble rows."""
    runs = {
        "inner": "attribution/inner_shared_tabrepo_score_only_review",
        "vtwo": "score_level_tabrepo_score_only_review",
    }
    frames = {key: assets.csv(f"{directory}/clustered_summary.csv").set_index("method") for key, directory in runs.items()}
    contrasts = {key: assets.csv(f"{directory}/contrasts.csv") for key, directory in runs.items()}
    methods = [
        "meta_selector_family_prior_budgeted",
        "train_selected_fixed_reference__score_only",
        "nested_policy_selected",
        "meta_selector_task_adaptive_budgeted",
        "meta_selector_calibrated_budget_10000s",
        "meta_selector_meta_feature_guard_005",
        "meta_selector_single",
    ]
    change = (
        frames["vtwo"].loc[methods, "mean_normalized_rank_regret"] - frames["inner"].loc[methods, "mean_normalized_rank_regret"]
    ).abs().max()
    assets.define("attr.tr.inner.maxchange", num(float(change), 4))
    for key in runs:
        assets.define(f"attr.tr.{key}.nested.reg", num(frames[key].loc["nested_policy_selected", "mean_normalized_rank_regret"], 4))
        primary = contrasts[key][contrasts[key]["family"].eq("primary")]
        for _, item in primary.iterrows():
            right = POLICY_KEYS.get(str(item["right"]))
            if right is None:
                continue
            base = f"attr.tr.{key}.{right}"
            assets.verdicts[base] = contrast_verdicts(item)
            assets.define(f"{base}.diff", num(item["mean_difference"], 4, signed=True))
            assets.define(f"{base}.ci", interval(item["ci_low"], item["ci_high"], 4))
            if "bootstrap_holm_p" in item and pd.notna(item["bootstrap_holm_p"]):
                assets.define_p(f"{base}.mholm", pvalue(item["bootstrap_holm_p"]))
    _adaptive_choices(assets, "attr.tr.inner.adaptivechosen", "attribution/inner_shared_tabrepo_score_only")
    _adaptive_choices(assets, "attr.tr.vtwo.adaptivechosen", "score_level_tabrepo_score_only")
    _inner_fit_gaps(
        assets,
        "attr.tr.inner",
        [("shared", "attribution/inner_shared_tabrepo"), ("pertarget", "score_level_tabrepo_nested")],
    )
    _check_inner_fit_rows(
        assets,
        "attribution/inner_shared_tabrepo_score_only_review",
        "score_level_tabrepo_score_only_review",
        {"nested_policy_selected"},
    )


def build_jackknife(assets: Assets) -> None:
    # Each jackknife contrast is compared with the conditional bootstrap standard error of the
    # same contrast in the artifact's review (primary family, or mechanism for fp - vb).
    reviews = {
        "jk.ta": ("score_level_tabarena_review", "train_selected_fixed_reference__declared_fixed"),
        "jk.tr": ("score_level_tabrepo_score_only_review", "train_selected_fixed_reference__score_only"),
    }
    for prefix, directory in [("jk.ta", "tabarena_jackknife"), ("jk.tr", "tabrepo_jackknife")]:
        frame = assets.csv(f"{directory}/jackknife_summary.csv")
        review_dir, train_selected = reviews[prefix]
        review = assets.csv(f"{review_dir}/contrasts.csv")
        metadata = json.loads((assets.out / directory / "metadata.json").read_text(encoding="utf-8"))
        assets.define(f"{prefix}.subsets", str(int(metadata["subsets"])))
        assets.define(f"{prefix}.drop", str(int(metadata["drop"])))
        assets.define(f"{prefix}.splits", str(int(metadata["splits_per_subset"])))
        if "mean_train_datasets" in metadata:
            assets.define(f"{prefix}.ntrain", num(metadata["mean_train_datasets"], 0))
            # Relative Monte Carlo error of a standard error estimated from S subsets.
            assets.define(f"{prefix}.mcerror", percent(1.0 / math.sqrt(2.0 * (int(metadata["subsets"]) - 1))))
        keys = {
            "nested_policy_selected - meta_selector_single": "nested.vb",
            "nested_policy_selected - meta_selector_family_prior_budgeted": "nested.fp",
            "nested_policy_selected - train_selected_fixed_reference": "nested.tsf",
            "meta_selector_family_prior_budgeted - meta_selector_single": "fp.vb",
        }
        for _, item in frame.iterrows():
            key = keys.get(str(item["contrast"]))
            if key is None:
                continue
            assets.define(f"{prefix}.{key}.se", num(item["jackknife_se"], 4))
            # Half-width of a 90% normal interval at the procedure level, the analogue of TOST's.
            assets.define(f"{prefix}.{key}.hw", num(1.6449 * float(item["jackknife_se"]), 4))
            assets.define(f"{prefix}.{key}.mean", num(item["subset_mean"], 4, signed=True))
            assets.define(f"{prefix}.{key}.neg", percent(item["share_subsets_negative"]))
            assets.define(f"{prefix}.{key}.pos", percent(item["share_subsets_positive"]))
            left, right = [part.strip() for part in str(item["contrast"]).split(" - ")]
            right = train_selected if right == "train_selected_fixed_reference" else right
            match = review[review["left"].eq(left) & review["right"].eq(right)]
            if not match.empty:
                bootstrap_se = float(match.iloc[0]["se_bootstrap"])
                assets.define(f"{prefix}.{key}.ratio", num(float(item["jackknife_se"]) / bootstrap_se, 1))
                if prefix == "jk.ta" and key == "nested.vb":
                    # Section 6.3: the declared separation at the diagnostic's standard error.
                    # A normal test of the full-data difference, Holm-adjusted as the smallest
                    # p-value of its family (the other contrasts are far smaller).
                    family = review[review["family"].eq(match.iloc[0]["family"])]
                    z = abs(float(match.iloc[0]["mean_difference"])) / float(item["jackknife_se"])
                    p_normal = math.erfc(z / math.sqrt(2.0))
                    assets.define_p(f"{prefix}.{key}.pnorm", pvalue(p_normal))
                    assets.define_p(f"{prefix}.{key}.pholm", pvalue(min(1.0, len(family) * p_normal)))
            if prefix == "jk.ta" and key == "nested.fp":
                # The 90% half-width in TabArena rank positions (one position is 1/43).
                assets.define(f"{prefix}.{key}.hwrank", num(43.0 * 1.6449 * float(item["jackknife_se"]), 1))


FAMILY_NAMES = {
    "primary": "Primary",
    "comparator_pools": "Reference pools",
    "individual_fixed": "Individual fixed rows",
    "revision_references": "References added later",
    "mechanism": "Mechanism",
    "provenance": "Provenance",
    "fixed_ensembles": "Fixed ensembles",
    "score_level_rows": "Score-level rows",
    "capability": "Capability",
}


def build_families(assets: Assets) -> None:
    """The Holm families actually run, read from each review's contrast table."""
    panels = [
        ("TabArena", "score_level_tabarena_review"),
        ("TabRepo, ensemble-inclusive", "score_level_tabrepo_nested_review"),
        ("TabRepo, score-only", "score_level_tabrepo_score_only_review"),
    ]
    rows = []
    groups = {}
    for artifact, directory in panels:
        contrasts = assets.csv(f"{directory}/contrasts.csv")
        groups[len(rows)] = artifact
        for family, group in contrasts.groupby("family", sort=False):
            lefts = group["left"].unique().tolist()
            left_text = "; ".join(policy_label(left) for left in lefts)
            rights = "; ".join(policy_label(right) for right in group["right"])
            rows.append([FAMILY_NAMES.get(family, family), str(int(group.shape[0])), f"{left_text} $-$ \\{{{rights}\\}}"])
    assets.tables["families"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.20\\linewidth}r>{\\raggedright\\arraybackslash}p{0.66\\linewidth}",
        ["Family", "Size", "Contrasts"],
        rows,
        groups=groups,
    )


COVERAGE_FIELDS = [
    "Artifact",
    "Capability set",
    "Policy pool",
    "Fixed references",
    "Evidence unit",
    "Inference rule",
    "Endpoint",
    "Cost field",
    "Validation provenance",
]
COVERAGE_CLASSES = ["Checklists", "Statistics", "Benchmarks, AC", "Algorithm selection"]
# How each class is named in the generated tables (the keys above are the record's values).
COVERAGE_CLASS_LABELS = {"Benchmarks, AC": "Benchmarks, algorithm configuration"}
COVERAGE_RANK = {"no": 0, "part": 1, "yes": 2}


def build_coverage(assets: Assets, root: Path) -> None:
    """Table 3 and its instrument-level appendix table, from the coded coverage record.

    ``docs/positioning_coverage.csv`` codes every instrument on all nine fields, with a
    locator and a paraphrase per cell. A class's cell is the strongest code of its
    instruments, so the class-level table cannot drift from the record.
    """
    path = root / "docs" / "positioning_coverage.csv"
    if not path.exists():
        raise Missing(str(path))
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    frame["field_number"] = frame["field_number"].astype(int)
    if not frame["code"].isin(list(COVERAGE_RANK)).all():
        raise ValueError(f"Unknown coverage codes: {sorted(set(frame['code']) - set(COVERAGE_RANK))}")
    if not frame["class"].isin(COVERAGE_CLASSES).all():
        raise ValueError(f"Unknown instrument classes: {sorted(set(frame['class']) - set(COVERAGE_CLASSES))}")
    cells = set(zip(frame["instrument"], frame["field_number"]))
    expected = {(instrument, field) for instrument in frame["instrument"].unique() for field in range(1, 10)}
    if len(cells) != len(frame) or cells != expected:
        raise ValueError("The coverage record must code every instrument exactly once on each of the nine fields.")
    if any(not str(value).strip() for value in frame["locator"]):
        raise ValueError("Every coverage cell needs a locator.")
    rows = []
    for number, field in enumerate(COVERAGE_FIELDS, start=1):
        row = [field]
        for name in COVERAGE_CLASSES:
            codes = frame.loc[frame["field_number"].eq(number) & frame["class"].eq(name), "code"]
            row.append(max(codes, key=COVERAGE_RANK.__getitem__))
        rows.append(row)
    assets.tables["coverage"] = tabular("lcccc", ["Field", *[COVERAGE_CLASS_LABELS.get(name, name) for name in COVERAGE_CLASSES]], rows)
    short = {"yes": "yes", "part": "part", "no": "--"}
    rows = []
    groups = {}
    for name in COVERAGE_CLASSES:
        members = frame[frame["class"].eq(name)]
        groups[len(rows)] = COVERAGE_CLASS_LABELS.get(name, name)
        for instrument in sorted(members["instrument"].unique()):
            cells = members[members["instrument"].eq(instrument)].sort_values("field_number")
            rows.append([f"\\citet{{{cells['citation_key'].iloc[0]}}}", *[short[code] for code in cells["code"]]])
    assets.tables["coverage_instruments"] = tabular(
        "l" + "c" * len(COVERAGE_FIELDS),
        ["Instrument", "Art.", "Cap.", "Pool", "Ref.", "Unit", "Inf.", "End.", "Cost", "Prov."],
        rows,
        groups=groups,
    )
    assets.define("cov.instruments", str(int(frame["instrument"].nunique())))
    assets.define("cov.cells", str(int(len(frame))))
    for code in COVERAGE_RANK:
        assets.define(f"cov.{code}", str(int(frame["code"].eq(code).sum())))
    # The most fields any one instrument asks for explicitly, and which instrument that is.
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    yes_counts = frame[frame["code"].eq("yes")].groupby("instrument").size()
    most = int(yes_counts.max()) if not yes_counts.empty else 0
    leaders = sorted(yes_counts[yes_counts.eq(most)].index) if most else []
    assets.define("cov.maxyes", words[most])
    keys = [str(frame.loc[frame["instrument"].eq(name), "citation_key"].iloc[0]) for name in leaders]
    assets.define("cov.maxyes.who", "\\citet{" + ",".join(keys) + "}")


def build_meta_study(assets: Assets, root: Path) -> None:
    directory = root / "docs" / "meta_study"
    claims_path = directory / "claims.csv"
    screening_path = directory / "screening.csv"
    if not claims_path.exists():
        raise Missing(str(claims_path))
    claims = read_claims(claims_path)
    screening = pd.read_csv(screening_path, keep_default_na=False, dtype=str)
    status = claims["verification"].astype(str)
    unknown = sorted(set(status) - {"unverified", "verified", "excluded"})
    if unknown:
        raise ValueError(f"Unknown verification status in claims.csv: {unknown}")
    if assets.meta_verified_only:
        pending = claims.loc[status.eq("unverified"), "record_id"].tolist()
        if pending:
            raise ValueError(f"Submission build needs every claim verified or excluded; still unverified: {pending}")
    # Claims the author excluded at verification are left out of every count.
    claims = claims[~status.eq("excluded")]
    if assets.meta_verified_only:
        claims = claims[claims["verification"].eq("verified")]
    assets.define("meta.screened", str(int(screening.shape[0])))
    assets.define("meta.notretrieved", str(int(screening["decision"].eq("not retrieved").sum())))
    assets.define("meta.included", str(int(claims.shape[0])))
    assets.define("meta.verified", str(int(claims["verification"].eq("verified").sum())))
    assets.define("meta.excludedatverification", str(int(status.eq("excluded").sum())))
    verified = claims[claims["verification"].eq("verified")]
    build_meta_agreement(assets, verified)
    build_meta_recheck(assets, claims, directory)
    build_meta_count_check(assets, claims, directory)
    rows = []
    for column, label in META_FIELDS:
        counts = claims[column].value_counts()
        values = [str(int(counts.get(code, 0))) for code in ["D", "P", "A", "NA"]]
        key = meta_item_key(column)
        assets.define(f"meta.{key}.declared", values[0])
        assets.define(f"meta.{key}.partial", values[1])
        assets.define(f"meta.{key}.absent", values[2])
        assets.define(f"meta.{key}.na", values[3])
        # Claims the field applies to: the not-applicable code leaves a claim out of the denominator.
        assets.define(f"meta.{key}.applicable", str(int(claims[column].ne("NA").sum())))
        assets.define(f"meta.{key}.declared.ai", str(int(resolved_codes(claims, column).eq("D").sum())))
        rows.append([label, *values])
    assets.tables["meta_rates"] = tabular("lrrrr", ["Field", "Declared", "Partial", "Absent", "NA"], rows)
    for column in META_EXTRA_COLUMNS:
        counts = claims[column].value_counts()
        key = meta_item_key(column)
        for code in ["yes", "no", "unclear"]:
            assets.define(f"meta.{key}.{code}", str(int(counts.get(code, 0))))
        assets.define(f"meta.{key}.yes.ai", str(int(resolved_codes(claims, column).eq("yes").sum())))
    # Cite the coded papers from the sheet, so dropping an unverified row updates the text.
    assets.define("meta.citations", "\\citep{" + ",".join(claims["cite_key"]) + "}")
    code_rows = []
    extra_codes = {"yes": "y", "no": "n", "unclear": "?"}
    for _, item in claims.iterrows():
        tag = str(item["paper"]).partition(" (")[2].rstrip(")")
        # A comma, not a second parenthesis, after the citation's year.
        paper = f"\\citet{{{item['cite_key']}}}" + (f", {latex_escape(tag)}" if tag else "")
        code_rows.append(
            [
                paper,
                *[str(item[column]) for column, _ in META_FIELDS],
                *[extra_codes[str(item[column])] for column in META_EXTRA_COLUMNS],
            ]
        )
    assets.tables["meta_codes"] = tabular(
        ">{\\raggedright\\arraybackslash}p{0.27\\linewidth}" + "c" * (len(META_FIELDS) + len(META_EXTRA_COLUMNS)),
        ["Paper", "Art.", "Cap.", "Pool", "Ref.", "Unit", "Inf.", "End.", "Cost", "Prov.", "E1", "E2", "E3"],
        code_rows,
    )


def _agreement_numbers(assets: Assets, prefix: str, first: list[str], second: list[str]) -> None:
    observed, _, kappa = cohen_kappa(first, second)
    assets.define(f"{prefix}.checked", str(len(first)))
    assets.define(f"{prefix}.agreed", str(sum(a == b for a, b in zip(first, second))))
    assets.define(f"{prefix}.agreepct", percent(observed))
    assets.define(f"{prefix}.kappa", num(kappa, 2))


def build_meta_agreement(assets: Assets, verified: pd.DataFrame) -> None:
    """Agreement of the two AI codings, and how the resolution against the sources went.

    Every code of every verified claim (nine fields and three extra items) is compared
    twice: the blind second coding (``_check``) against the first (``_draft``), and the
    AI-resolved code against the first.
    """
    keys = ["checked", "agreed", "changed", "kappa", "disagreed", "disagreed.check", "disagreed.draft"]
    keys += [f"nine.{key}" for key in ["checked", "agreed", "agreepct", "kappa", "disagreed"]]
    keys += [f"extras.{key}" for key in ["checked", "agreed", "agreepct", "kappa", "disagreed"]]
    keys += [f"{meta_item_key(item)}.{key}" for item in META_ITEMS for key in ["agreed", "agreepct", "kappa"]]
    if verified.empty:
        for key in keys:
            assets.define(f"meta.codes.{key}", "\\textbf{[pending]}")
        return
    missing = []
    for column in META_ITEMS:
        check = verified.get(f"{column}_check", pd.Series("", index=verified.index))
        missing += [f"{record}:{column}" for record in verified.loc[check.eq(""), "record_id"]]
    if missing:
        raise ValueError(f"Verified claims need a second-pass code for every field; missing: {missing}")

    def stack(items: list[str], kind: str) -> list[str]:
        if kind == "ai":
            return [code for item in items for code in resolved_codes(verified, item)]
        return [code for item in items for code in verified[f"{item}_{kind}"]]

    draft, check, ai = stack(META_ITEMS, "draft"), stack(META_ITEMS, "check"), stack(META_ITEMS, "ai")
    observed, _, kappa = cohen_kappa(draft, check)
    assets.define("meta.codes.checked", str(len(draft)))
    assets.define("meta.codes.agreed", str(sum(a == b for a, b in zip(draft, check))))
    assets.define("meta.codes.kappa", num(kappa, 2))
    # The AI resolution against the sources: how many first codes it changed, and which
    # coding it kept where the two disagreed.
    assets.define("meta.codes.changed", str(sum(a != r for a, r in zip(draft, ai))))
    disagreed = [(a, b, r) for a, b, r in zip(draft, check, ai) if a != b]
    assets.define("meta.codes.disagreed", str(len(disagreed)))
    assets.define("meta.codes.disagreed.check", str(sum(r == b for _, b, r in disagreed)))
    assets.define("meta.codes.disagreed.draft", str(sum(r == a for a, _, r in disagreed)))
    _agreement_numbers(assets, "meta.codes.nine", stack(META_FIELD_COLUMNS, "draft"), stack(META_FIELD_COLUMNS, "check"))
    _agreement_numbers(assets, "meta.codes.extras", stack(META_EXTRA_COLUMNS, "draft"), stack(META_EXTRA_COLUMNS, "check"))
    # The disagreements split by the nine fields and the three extra items (they add up to
    # meta.codes.disagreed).
    for name, items in [("nine", META_FIELD_COLUMNS), ("extras", META_EXTRA_COLUMNS)]:
        pairs = zip(stack(items, "draft"), stack(items, "check"))
        assets.define(f"meta.codes.{name}.disagreed", str(sum(a != b for a, b in pairs)))
    for item in META_ITEMS:
        observed, _, kappa = cohen_kappa(verified[f"{item}_draft"], verified[f"{item}_check"])
        key = meta_item_key(item)
        assets.define(f"meta.codes.{key}.agreed", str(int((verified[f"{item}_draft"] == verified[f"{item}_check"]).sum())))
        assets.define(f"meta.codes.{key}.agreepct", percent(observed))
        assets.define(f"meta.codes.{key}.kappa", num(kappa, 2))


def build_meta_recheck(assets: Assets, claims: pd.DataFrame, directory: Path) -> None:
    """The third, blind AI coding of the declared subset (protocol.md, Deviations 8 and 9).

    The agreement keys compare the third codes with the AI-resolved codes; ``changed`` counts
    the final codes that resolving the disagreements against the sources changed. They are
    pending until every declared cell is coded and every disagreement resolved; the
    submission build refuses to run before.
    """
    declaration = load_declaration(directory)
    sheet_path = directory / RECHECK_CODING_FILE
    sheet = pd.read_csv(sheet_path, keep_default_na=False, dtype=str) if sheet_path.exists() else None
    coded, uncoded, unresolved = check_recheck_state(
        claims, declaration, sheet=sheet, require_complete=assets.meta_verified_only
    )
    keys = ["agreed", "agreepct", "kappa", "disagreed", "changed"]
    keys += [f"nine.{key}" for key in ["cells", "agreed", "agreepct", "kappa", "changed"]]
    keys += ["extras.cells"]
    keys += [f"{meta_item_key(item)}.{key}" for item in META_ITEMS for key in ["cells", "agreed", "kappa", "changed"]]
    if declaration is None:
        for key in ["cells", "coded", "claims", "full", "seed", *keys]:
            assets.define(f"meta.recheck.{key}", "\\textbf{[pending]}")
        return
    assets.define("meta.recheck.cells", str(len(coded) + len(uncoded)))
    assets.define("meta.recheck.coded", str(len(coded)))
    assets.define("meta.recheck.claims", str(len(declaration["population"])))
    assets.define("meta.recheck.full", str(len(declaration["full_claims"])))
    assets.define("meta.recheck.seed", str(int(declaration["seed"])))
    if uncoded or unresolved:
        for key in keys:
            assets.define(f"meta.recheck.{key}", "\\textbf{[pending]}")
        return
    rows = claims.set_index("record_id")

    def cells_of(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        cells = [(record, item) for record, item in coded if item in items]
        return (
            [rows.at[record, f"{item}_recheck"] for record, item in cells],
            [rows.at[record, f"{item}_ai"] for record, item in cells],
            [rows.at[record, item] for record, item in cells],
        )

    def define(prefix: str, third: list[str], ai: list[str], final: list[str], *, share: bool = True) -> None:
        observed, _, kappa = cohen_kappa(third, ai)
        agreed = sum(a == b for a, b in zip(third, ai))
        assets.define(f"{prefix}.agreed", str(agreed))
        assets.define(f"{prefix}.changed", str(sum(a != f for a, f in zip(ai, final))))
        assets.define(f"{prefix}.kappa", num(kappa, 2))
        if share:
            assets.define(f"{prefix}.agreepct", percent(observed))

    third, ai, final = cells_of(META_ITEMS)
    define("meta.recheck", third, ai, final)
    assets.define("meta.recheck.disagreed", str(sum(a != b for a, b in zip(third, ai))))
    third, ai, final = cells_of(META_FIELD_COLUMNS)
    assets.define("meta.recheck.nine.cells", str(len(third)))
    define("meta.recheck.nine", third, ai, final)
    assets.define("meta.recheck.extras.cells", str(len(cells_of(META_EXTRA_COLUMNS)[0])))
    for item in META_ITEMS:
        third, ai, final = cells_of([item])
        key = f"meta.recheck.{meta_item_key(item)}"
        assets.define(f"{key}.cells", str(len(third)))
        define(key, third, ai, final, share=False)


def build_meta_count_check(assets: Assets, claims: pd.DataFrame, directory: Path) -> None:
    """The check of the codes behind the abstract's counts (protocol.md, Deviations 11 and 12).

    At the author's request the check was made by fresh Claude sessions, not by the author.
    ``cells`` and ``checked`` count the cells of the check; ``agreed`` and ``changed`` how
    many codes it confirmed and changed, overall and per item. Pending until every cell is
    checked; the submission build refuses to run before.
    """
    checked, unchecked = check_count_check_state(claims, require_complete=assets.meta_verified_only)
    assets.define("meta.countcheck.cells", str(len(checked) + len(unchecked)))
    assets.define("meta.countcheck.checked", str(len(checked)))
    items = [meta_item_key(item) for item in META_COUNT_CHECK_ITEMS]
    keys = ["agreed", "changed", *[f"{item}.{key}" for item in items for key in ["agreed", "changed"]]]
    if unchecked:
        for key in [*keys, "firstagreed", "firstkappa"]:
            assets.define(f"meta.countcheck.{key}", "\\textbf{[pending]}")
        return
    sheet_path = directory / RECHECK_CODING_FILE
    sheet = pd.read_csv(sheet_path, keep_default_na=False, dtype=str) if sheet_path.exists() else pd.DataFrame()
    entries = {(str(row["record_id"]).strip(), str(row["item"]).strip()): row for _, row in sheet.iterrows()}
    rows = claims.set_index("record_id")
    changed = {
        (record, item)
        for record, item in checked
        if rows.at[record, f"{item}_countcheck"] != recheck_final(rows.loc[record], item, entries.get((record, item)))[0]
    }
    assets.define("meta.countcheck.agreed", str(len(checked) - len(changed)))
    assets.define("meta.countcheck.changed", str(len(changed)))
    # The checkers' own codes, reported before they saw the code under check (Deviation 12).
    if not checked:
        for key in ["firstagreed", "firstkappa"]:
            assets.define(f"meta.countcheck.{key}", "\\textbf{[pending]}")
        return
    check_path = directory / COUNT_CHECK_FILE
    if not check_path.exists():
        raise Missing(str(check_path))
    check = pd.read_csv(check_path, keep_default_na=False, dtype=str)
    first = check["first_code"].str.strip().str.upper()
    under_check = check["code"].str.strip().str.upper()
    assets.define("meta.countcheck.firstagreed", str(int(first.eq(under_check).sum())))
    assets.define("meta.countcheck.firstkappa", num(cohen_kappa(list(first), list(under_check))[2], 2))
    for item in META_COUNT_CHECK_ITEMS:
        cells = [cell for cell in checked if cell[1] == item]
        n_changed = sum(cell in changed for cell in cells)
        assets.define(f"meta.countcheck.{meta_item_key(item)}.agreed", str(len(cells) - n_changed))
        assets.define(f"meta.countcheck.{meta_item_key(item)}.changed", str(n_changed))


# ----------------------------------------------------------------------------- revision checks

# The corrected resampled t-test of every review (scripts/corrected_resampled_test.py), by
# result-key prefix, output folder and the run's name in the appendix table.
CORRECTED_RUNS = [
    ("ta", "tabarena_corrected_test", "TabArena"),
    ("ta.ext", "tabarena_extended_corrected_test", "TabArena, extended pool"),
    ("ta.shrink", "tabarena_shrink_corrected_test", "TabArena, reduced pool"),
    ("tr", "tabrepo_corrected_test", "TabRepo, with ensembles"),
    ("trs", "tabrepo_score_only_corrected_test", "TabRepo, score-only"),
]

# Rows of the appendix table: (prefix, family, references of nested selection).
CORRECTED_TABLE_ROWS = [
    ("ta", "primary", [
        "meta_selector_single", "meta_selector_family_prior_budgeted",
        "train_selected_fixed_reference__declared_fixed", "meta_selector_task_adaptive_budgeted",
        "meta_selector_meta_feature_guard_005",
    ]),
    ("ta.ext", "primary", ["meta_selector_single", "meta_selector_family_prior_budgeted"]),
    ("ta.shrink", "primary", ["meta_selector_single", "meta_selector_family_prior_budgeted"]),
    ("tr", "primary", ["meta_selector_single", "train_selected_fixed_reference__score_only"]),
    ("tr", "fixed_ensembles", ["meta_selector_tabrepo_ensemble_k3"]),
    ("trs", "primary", ["meta_selector_single", "train_selected_fixed_reference__score_only"]),
    ("trs", "score_level_rows", ["meta_selector_task_adaptive_budgeted", "meta_selector_meta_feature_guard_005"]),
]


def corrected_keys(assets: Assets, prefix: str, frame: pd.DataFrame) -> None:
    """Result keys and verdicts of the corrected resampled t-test, by contrast."""
    for _, row in frame.iterrows():
        left = POLICY_KEYS.get(str(row["left"]))
        right = POLICY_KEYS.get(str(row["right"]))
        if left is None or right is None:
            continue
        base = f"{prefix}.{row['family']}.{left}.{right}"
        assets.define(f"{base}.nbse", num(row["nb_se"], 4))
        if math.isfinite(row["nb_se_ratio"]):
            assets.define(f"{base}.nbratio", num(row["nb_se_ratio"], 1))
        assets.define_p(f"{base}.nbp", pvalue(row["nb_p"]))
        assets.define_p(f"{base}.nbholm", pvalue(row["nb_holm_p"]))
        if "nb_holm_p_df_datasets" in row and pd.notna(row["nb_holm_p_df_datasets"]):
            # The same test with the number of datasets less one as degrees of freedom.
            assets.define_p(f"{base}.nbholmdf", pvalue(row["nb_holm_p_df_datasets"]))
        assets.define(f"{base}.nbci", interval(row["nb_ci90_low"], row["nb_ci90_high"], 4))
        assets.define(f"{base}.nbmargin", num(row["nb_smallest_equivalence_margin"], 4))
        # Five decimals where four would print a margin that only just fails as equal to it.
        assets.define(f"{base}.nbmargin5", num(row["nb_smallest_equivalence_margin"], 5))
        # Added to the bootstrap verdicts of the same contrast (contrast_numbers runs first).
        verdicts = assets.verdicts.setdefault(base, {})
        if float(row["nb_holm_p"]) >= ALPHA:
            verdicts["corrected"] = "not separated"
        else:
            verdicts["corrected"] = "better" if float(row["split_mean_difference"]) < 0 else "worse"
        for margin in ["0p005", "0p010", "0p015", "0p020"]:
            column = f"nb_equivalent_{margin}"
            if column in row:
                verdicts[f"corrected_equiv{margin}"] = "equivalent" if bool(row[column]) else "not equivalent"


def build_corrected(assets: Assets) -> None:
    frames = {}
    primary = assets.csv("score_level_tabarena_review/contrasts.csv")
    primary = primary[primary["family"].eq("primary") & primary["left"].eq("nested_policy_selected")].set_index("right")
    committed = ["meta_selector_family_prior_budgeted", "train_selected_fixed_reference__declared_fixed"]
    corrected_ta = assets.csv("tabarena_corrected_test/corrected_resampled_tests.csv")
    corrected_ta = corrected_ta[corrected_ta["family"].eq("primary")].set_index("right")
    committed_p = [float(primary.loc[r, c]) for r in committed for c in ("bootstrap_holm_p", "ttest_holm_p")]
    committed_p += [float(corrected_ta.loc[r, "nb_holm_p"]) for r in committed]
    if len({pvalue(v) for v in committed_p}) != 1:
        raise ValueError("Nested selection minus the committed rules: the six Holm p-values are not all equal.")
    assets.define_p("ta.committed.holm.all", pvalue(committed_p[0]))
    flips = 0
    for prefix, folder, _ in CORRECTED_RUNS:
        frame = assets.csv(f"{folder}/corrected_resampled_tests.csv")
        frames[prefix] = frame
        corrected_keys(assets, prefix, frame)
        # Degrees of freedom: J - 1 (Nadeau and Bengio) against the number of datasets less one.
        if "nb_holm_p_df_datasets" in frame.columns:
            flips += int(((frame["nb_holm_p"] < ALPHA) != (frame["nb_holm_p_df_datasets"] < ALPHA)).sum())
    assets.define("nb.dfflips", str(flips))
    # The minimum detectable effect of nested - family-prior at the corrected standard error
    # (80% power, two-sided alpha), in normalized regret and rank positions (1/43).
    fp = frames["ta"][frames["ta"]["family"].eq("primary") & frames["ta"]["right"].eq("meta_selector_family_prior_budgeted")].iloc[0]
    mde = float(stats.norm.ppf(1.0 - ALPHA / 2.0) + stats.norm.ppf(0.80)) * float(fp["nb_se"])
    assets.define("ta.primary.nested.fp.nbmde80", num(mde, 4))
    assets.define("ta.primary.nested.fp.nbmde80rank", num(mde * 43.0, 1))
    labels = {prefix: label for prefix, _, label in CORRECTED_RUNS}
    rows = []
    for prefix, family, references in CORRECTED_TABLE_ROWS:
        frame = frames[prefix]
        for reference in references:
            item = frame[frame["family"].eq(family) & frame["right"].eq(reference)].iloc[0]
            rows.append(
                [
                    labels[prefix],
                    "Train-selected fixed rule (score-only pool)" if (prefix, reference) == ("tr", "train_selected_fixed_reference__score_only") else policy_label(reference),
                    num(item["mean_difference"], 4, signed=True),
                    num(item["bootstrap_se"], 4),
                    num(item["nb_se"], 4),
                    pvalue(item["bootstrap_holm_p"]),
                    pvalue(item["nb_holm_p"]),
                    interval(item["nb_ci90_low"], item["nb_ci90_high"], 4),
                ]
            )
    assets.tables["corrected_tests"] = tabular(
        "llrrrrrr",
        [
            "Run", "Reference", "Nested $-$ reference", "SE, bootstrap", "SE, corrected",
            "Holm $p$, bootstrap", "Holm $p$, corrected", "Corrected 90\\% CI",
        ],
        rows,
    )
    # The variance factor of the correction, 1/J + n_test/n_train, for the text.
    for prefix, folder, _ in CORRECTED_RUNS:
        meta = json.loads((assets.out / folder / "metadata.json").read_text(encoding="utf-8"))
        assets.define(f"{prefix}.nbfactor", num(float(meta["variance_factor"]), 2))


def build_influence(assets: Assets) -> None:
    row = assets.csv("tabarena_influence/influence_summary.csv").iloc[0]
    assets.define("ta.infl.loo.min", num(row["loo_min_mean"], 4, signed=True))
    assets.define("ta.infl.loo.max", num(row["loo_max_mean"], 4, signed=True))
    assets.define("ta.infl.loo.maxcihigh", num(row["loo_max_ci_high"], 4, signed=True))
    assets.define("ta.infl.loo.allexclude", "yes" if bool(row["loo_all_exclude_zero"]) else "no")
    assets.define_p("ta.infl.loo.maxp", pvalue(row["loo_max_bootstrap_p"]))
    assets.define("ta.infl.top.n", str(int(row["top"])))
    assets.define("ta.infl.top.share", percent(row["top_share_of_sum"]))
    assets.define("ta.infl.topout.diff", num(row["topout_mean_difference"], 4, signed=True))
    assets.define("ta.infl.topout.ci", interval(row["topout_ci_low"], row["topout_ci_high"], 4))
    assets.define_p("ta.infl.topout.mp", pvalue(row["topout_bootstrap_p"]))
    assets.define_p("ta.infl.topout.nbp", pvalue(row["topout_nb_p"]))
    names = [latex_escape(name) for name in str(row["top_datasets"]).split(";")]
    assets.define("ta.infl.top.datasets", ", ".join(names[:-1]) + " and " + names[-1])
    # Leave-one-out with the primary family Holm-adjusted on the kept datasets.
    assets.define_p("ta.infl.loo.maxholm", pvalue(row["loo_max_holm_p"]))
    assets.define("ta.infl.loo.notsep", str(int(row["loo_n_not_separated"])))
    dropped = [latex_escape(name) for name in str(row["loo_not_separated"]).split(";") if name]
    assets.define("ta.infl.loo.notsep.datasets", " and ".join(dropped) if dropped else "none")
    # Symmetric trimming (Yuen's one-sample test; unadjusted).
    assets.define("ta.infl.trimten.diff", num(row["trim10_mean"], 4, signed=True))
    assets.define_p("ta.infl.trimten.p", pvalue(row["trim10_yuen_p"]))
    assets.define("ta.infl.trimtwenty.diff", num(row["trim20_mean"], 4, signed=True))
    assets.define_p("ta.infl.trimtwenty.p", pvalue(row["trim20_yuen_p"]))
    # Dataset size: the gain by the number of evaluation units (TabArena gives small datasets more).
    assets.define("ta.infl.size.rho", num(row["size_spearman"], 2))
    assets.define_p("ta.infl.size.p", pvalue(row["size_spearman_p"]))
    assets.define("ta.infl.small.n", str(int(row["many_units_datasets"])))
    assets.define("ta.infl.small.units", str(int(row["many_units"])))
    assets.define("ta.infl.small.diff", num(row["many_units_mean"], 3, signed=True))
    assets.define("ta.infl.small.maxrows", seconds(float(row["many_units_max_rows"])))
    assets.define("ta.infl.large.n", str(int(row["few_units_datasets"])))
    assets.define("ta.infl.large.units", str(int(row["few_units"])))
    assets.define("ta.infl.large.diff", num(row["few_units_mean"], 3, signed=True))


def build_imputed(assets: Assets) -> None:
    """TabArena's copied rows (scripts/imputed_rows.py) and the main audit rerun without them."""
    summary = assets.csv("tabarena_imputed/copied_summary.csv")
    total = summary[summary["method"].eq("all")].iloc[0]
    assets.define("imp.rows", str(int(total["rows"])))
    assets.define("imp.rowstotal", seconds(float(total["rows_total"])))
    assets.define("imp.share", percent(float(total["rows"]) / float(total["rows_total"]), 1))
    assets.define("imp.datasets", str(int(total["datasets"])))
    assets.define("imp.units", str(int(total["units"])))
    assets.define("imp.unitstotal", str(int(total["units_total"])))
    assets.define("imp.flagged", str(int(total["flagged_rows"])))
    # Rows that carry RandomForest (default)'s recorded prediction time without a metadata flag.
    assets.define("imp.timing", str(int(total["timing_rows"])))
    assets.define("imp.exact", str(int(total["exact_error_rows"])))
    assets.define("imp.maxgap", percent(float(total["max_relative_error_gap"]), 1))
    by_method = summary.set_index("method")
    assets.define("imp.tabpfn.datasets", str(int(by_method.loc["TABPFNV2 (default)", "datasets"])))
    assets.define("imp.tabicl.datasets", str(int(by_method.loc["TABICL (default)", "datasets"])))
    review = "score_level_tabarena_no_imputed_review"
    summary_numbers(assets, "imp", assets.csv(f"{review}/clustered_summary.csv"), rank_scale=43.0)
    contrast_numbers(assets, "imp", assets.csv(f"{review}/contrasts.csv"))
    corrected_keys(assets, "imp", assets.csv("tabarena_imputed/corrected_test/corrected_resampled_tests.csv"))
    # The yardstick against the refit: the main run's own held-out choices scored against the
    # candidate sets without the imputed rows (scripts/imputed_rows_rescore.py).
    rescored = assets.csv("tabarena_imputed/rescored/rescored_summary.csv").set_index("method")
    for method, key in [("nested_policy_selected", "nested"), ("meta_selector_single", "vb"), ("meta_selector_family_prior_budgeted", "fp")]:
        assets.define(f"imp.resc.{key}.reg", num(float(rescored.loc[method, "rescored_regret"])))
    contrast_numbers(assets, "imp", assets.csv("tabarena_imputed/rescored/rescored_contrasts.csv"))
    rescore_meta = json.loads((assets.out / "tabarena_imputed" / "rescored" / "metadata.json").read_text(encoding="utf-8"))
    assets.define("imp.resc.changed", percent(float(rescore_meta["nested_choice_changed_share"])))
    _imputation_stages(assets, "imp", "tabarena_imputed", review)
    imp_contrasts = assets.csv(f"{review}/contrasts.csv")
    differ = imp_contrasts[(imp_contrasts["bootstrap_holm_p"] < ALPHA) != (imp_contrasts["ttest_holm_p"] < ALPHA)]
    assets.define("imp.mtests.differ", str(int(differ.drop_duplicates(subset=NUMERIC_IDENTITY).shape[0])))
    only_bootstrap = differ[differ["bootstrap_holm_p"] < ALPHA].drop_duplicates(subset=NUMERIC_IDENTITY)
    assets.define("imp.mtests.onlyboot", str(int(only_bootstrap.shape[0])))
    build_provenance(assets, "imp", "tabarena_imputed/provenance", variants=True, rank_scale=43.0)
    # Without the copies, the single best solver is a model that does not run everywhere:
    # it falls back to the next-best method where its first choice is absent. The share is
    # dataset-level, as in the members table.
    top = assets.csv("tabarena_imputed/provenance/top_selected_members.csv")
    first = top[top["method"].eq("reference_single_best_global")].sort_values("share", ascending=False).iloc[0]
    assets.define("imp.sbs.first", latex_escape(candidate_name(str(first["selected_member"]))))
    assets.define("imp.sbs.firstshare", percent(float(first["share"])))


def build_provenance_reference(assets: Assets) -> None:
    """The rule that selects on defaults and deploys tuned + ensemble (scripts/provenance_reference.py)."""
    for prefix, folder in [("ta", "tabarena_provenance_reference"), ("imp", "tabarena_imputed/provenance_reference")]:
        contrasts = assets.csv(f"{folder}/contrasts.csv")
        contrast_numbers(assets, prefix, contrasts)
        corrected_keys(assets, prefix, contrasts)
        assets.define(f"{prefix}.dtt.reg", num(float(contrasts["left_mean"].iloc[0])))
        meta = json.loads((assets.out / folder / "metadata.json").read_text(encoding="utf-8"))
        assets.define(f"{prefix}.dtt.te.share", percent(float(meta["deployed_variant_share"].get("tuned + ensemble", 0.0))))


def build_tabrepo_candidates(assets: Assets) -> None:
    """How many configurations each TabRepo unit has (the slice has up to 1530)."""
    units = pd.read_parquet(
        assets.out / "score_level_tabrepo_score_only" / "unit_scores.parquet",
        columns=["dataset", "fold", "seed", "method", "n_candidates"],
    )
    units = units[units["method"].eq("meta_selector_single")].drop_duplicates(["dataset", "fold", "seed"])
    full = int(units["n_candidates"].max())
    assets.define("tr.cand.min", seconds(float(units["n_candidates"].min())))
    assets.define("tr.cand.max", seconds(float(full)))
    assets.define("tr.units.incomplete", str(int((units["n_candidates"] < full).sum())))
    assets.define("tr.units.total", str(int(units.shape[0])))


def build_pool_distinct(assets: Assets) -> None:
    """How many of the declared pool's rows differ in their dataset-level held-out regrets."""
    training = assets.csv("score_level_tabarena/training_policy_scores.csv")
    pool = sorted(training.loc[training["role"].eq("pool"), "method"].unique())
    means = assets.csv("score_level_tabarena_review/dataset_mean_scores.csv")
    regret = means.pivot_table(index="dataset", columns="method", values="mean_normalized_rank_regret")[pool]
    assets.define("ta.pool.rows", str(len(pool)))
    assets.define("ta.pool.distinct", str(int(regret.round(12).T.drop_duplicates().shape[0])))


# The deployable error-space correction (scripts/error_space_correction.py): result-key
# prefix, output folder, rule, and the table's artifact, offsets and scale.
ERROR_SPACE_RUNS = [
    ("ta.es", "tabarena_error_space", "error_space_correction", "TabArena", "candidate", "range"),
    ("ta.esm", "tabarena_error_space_median", "error_space_correction", "TabArena", "candidate", "median"),
    ("tr.es", "tabrepo_error_space", "error_space_correction", "TabRepo", "configuration", "range"),
    ("tr.esf", "tabrepo_error_space", "error_space_correction_family", "TabRepo", "learner family", "range"),
    ("tr.esm", "tabrepo_error_space_median", "error_space_correction", "TabRepo", "configuration", "median"),
    ("tr.esmf", "tabrepo_error_space_median", "error_space_correction_family", "TabRepo", "learner family", "median"),
]


def build_error_space(assets: Assets) -> None:
    rows = []
    for prefix, folder, rule, artifact, offsets, scale in ERROR_SPACE_RUNS:
        contrasts = assets.csv(f"{folder}/contrasts.csv")
        contrasts = contrasts[contrasts["left"].eq(rule)].set_index("right")
        assets.define(f"{prefix}.reg", num(float(contrasts["left_mean"].iloc[0])))
        for right, item in contrasts.iterrows():
            base = f"{prefix}.{POLICY_KEYS[str(right)]}"
            assets.define(f"{base}.diff", num(item["mean_difference"], 4, signed=True))
            assets.define(f"{base}.ci", interval(item["ci_low"], item["ci_high"], 4))
            assets.define_p(f"{base}.mholm", pvalue(item["bootstrap_holm_p"]))
            assets.verdicts[base] = contrast_verdicts(item)
        vb = contrasts.loc["meta_selector_single"]
        opt = contrasts.loc["reference_optimism_corrected"]
        rows.append(
            [
                artifact, offsets, scale,
                num(float(vb["left_mean"])),
                f"{num(vb['mean_difference'], 4, signed=True)} {interval(vb['ci_low'], vb['ci_high'], 4)}",
                pvalue(vb["bootstrap_holm_p"]),
                f"{num(opt['mean_difference'], 4, signed=True)} {interval(opt['ci_low'], opt['ci_high'], 4)}",
                pvalue(opt["bootstrap_holm_p"]),
            ]
        )
    for prefix, folder in [("ta.esm", "tabarena_error_space_median"), ("tr.esm", "tabrepo_error_space_median")]:
        meta = json.loads((assets.out / folder / "metadata.json").read_text(encoding="utf-8"))
        assets.define(f"{prefix}.nearbest", percent(meta["share_units_median_within_one_percent_of_best"]))
    assets.tables["error_space"] = tabular(
        "lllrrrrr",
        [
            "Artifact", "Offsets per", "Scale", "Regret",
            "Minus validation-best [95\\% CI]", "Holm $p$",
            "Minus rank correction [95\\% CI]", "Holm $p$",
        ],
        rows,
    )


# ----------------------------------------------------------------------------- third review


def _contrast_row(frame: pd.DataFrame, family: str, right: str = "meta_selector_single") -> pd.Series:
    return frame[frame["family"].eq(family) & frame["left"].eq("nested_policy_selected") & frame["right"].eq(right)].iloc[0]


def _imputation_stages(assets: Assets, prefix: str, folder: str, review: str) -> None:
    """The advantage over validation-best kept after refitting, the reapplied rules, and the
    split-level agreement of the nested choice between the main run and the refit."""
    refit = _contrast_row(assets.csv(f"{review}/contrasts.csv"), "primary")
    rescored = _contrast_row(assets.csv(f"{folder}/rescored/rescored_contrasts.csv"), "rescored_primary")
    assets.define(f"{prefix}.retained", percent(float(refit["mean_difference"]) / float(rescored["mean_difference"])))
    reapplied = assets.csv(f"{folder}/reapplied/reapplied_contrasts.csv")
    contrast_numbers(assets, prefix, reapplied)
    summary = assets.csv(f"{folder}/reapplied/reapplied_summary.csv").set_index("method")
    for method, key in [("nested_policy_selected", "nested"), ("meta_selector_single", "vb"), ("meta_selector_family_prior_budgeted", "fp")]:
        assets.define(f"{prefix}.reap.{key}.reg", num(float(summary.loc[method, "reapplied_regret"])))
    meta = json.loads((assets.out / folder / "reapplied" / "metadata.json").read_text(encoding="utf-8"))
    assets.define(f"{prefix}.reap.memberchanged", percent(float(meta["nested_member_changed_share"]), 1))
    check = assets.out / folder / "reapplied" / "reproduction_check.json"
    checked = json.loads(check.read_text(encoding="utf-8"))["reproduced_splits"] if check.exists() else len(meta["reproduced_splits"])
    assets.define(f"{prefix}.reap.checked", str(int(checked)))
    # How far refitting moves the contrast beyond reapplying the fitted rules: the paired
    # per-dataset change of nested selection minus validation-best between the two stages.
    reap_means = assets.csv(f"{folder}/reapplied/dataset_mean_scores.csv").set_index("dataset")
    refit_means = assets.csv(f"{review}/dataset_mean_scores.csv").pivot_table(
        index="dataset", columns="method", values="mean_normalized_rank_regret"
    )
    if float((reap_means["meta_selector_single"] - refit_means["meta_selector_single"]).abs().max()) > 1e-12:
        raise ValueError("Validation-best differs between the reapplied and refitted stages.")
    change = (refit_means["nested_policy_selected"] - refit_means["meta_selector_single"]) - (
        reap_means["nested_policy_selected"] - reap_means["meta_selector_single"]
    )
    resampler = ArtifactResampler(list(change.index), seed=REVIEW_SEED)
    values = resampler.align(change)
    boot = values[resampler.indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    assets.define(f"{prefix}.stage.refit.diff", num(float(values.mean()), 4, signed=True))
    assets.define(f"{prefix}.stage.refit.ci", interval(float(low), float(high), 4))
    assets.define_p(f"{prefix}.stage.refit.p", pvalue(bootstrap_mean_p(boot)))
    main = assets.csv("score_level_tabarena/split_results.csv").set_index("split_id")["selected_policy"]
    rerun_dir = review[: -len("_review")]
    rerun = assets.csv(f"{rerun_dir}/split_results.csv").set_index("split_id")["selected_policy"]
    assets.define(f"{prefix}.choice.same", str(int((main == rerun.reindex(main.index)).sum())))


def build_flagged(assets: Assets) -> None:
    """The audit without only the 726 rows that TabArena's metadata documents as imputed."""
    review = "score_level_tabarena_no_flagged_review"
    summary_numbers(assets, "flag", assets.csv(f"{review}/clustered_summary.csv"), rank_scale=43.0)
    contrast_numbers(assets, "flag", assets.csv(f"{review}/contrasts.csv"))
    corrected_keys(assets, "flag", assets.csv("tabarena_imputed_flagged/corrected_test/corrected_resampled_tests.csv"))
    contrast_numbers(assets, "flag", assets.csv("tabarena_imputed_flagged/rescored/rescored_contrasts.csv"))
    rescored = assets.csv("tabarena_imputed_flagged/rescored/rescored_summary.csv").set_index("method")
    assets.define("flag.resc.nested.reg", num(float(rescored.loc["nested_policy_selected", "rescored_regret"])))
    meta = json.loads((assets.out / "tabarena_imputed_flagged" / "metadata.json").read_text(encoding="utf-8"))
    assets.define("flag.removed", str(int(meta["removed_rows"])))
    _imputation_stages(assets, "flag", "tabarena_imputed_flagged", review)
    # The stage table: nested selection minus validation-best under both masks.
    main = _contrast_row(assets.csv("score_level_tabarena_review/contrasts.csv"), "primary")
    main_regret = assets.csv("score_level_tabarena_review/clustered_summary.csv").set_index("method").loc["nested_policy_selected", "mean_normalized_rank_regret"]
    rows = [["Full results file", "Main run", num(float(main_regret)), f"{num(main['mean_difference'], 4, signed=True)} {interval(main['ci_low'], main['ci_high'], 4)}", pvalue(main["bootstrap_holm_p"]), pvalue(main["ttest_holm_p"])]]
    for mask, prefix_dir, rerun_review, removed in [
        ("All 992 identified rows removed", "tabarena_imputed", "score_level_tabarena_no_imputed_review", "imp.rows"),
        ("Only the 726 documented rows removed", "tabarena_imputed_flagged", review, "flag.removed"),
    ]:
        stages = [
            ("Rescored", _contrast_row(assets.csv(f"{prefix_dir}/rescored/rescored_contrasts.csv"), "rescored_primary"),
             assets.csv(f"{prefix_dir}/rescored/rescored_summary.csv").set_index("method").loc["nested_policy_selected", "rescored_regret"]),
            ("Reapplied", _contrast_row(assets.csv(f"{prefix_dir}/reapplied/reapplied_contrasts.csv"), "reapplied_primary"),
             assets.csv(f"{prefix_dir}/reapplied/reapplied_summary.csv").set_index("method").loc["nested_policy_selected", "reapplied_regret"]),
            ("Refitted", _contrast_row(assets.csv(f"{rerun_review}/contrasts.csv"), "primary"),
             assets.csv(f"{rerun_review}/clustered_summary.csv").set_index("method").loc["nested_policy_selected", "mean_normalized_rank_regret"]),
        ]
        for index, (stage, row, regret) in enumerate(stages):
            label = mask.replace("992", assets.numbers.get(removed, "?")).replace("726", assets.numbers.get(removed, "?")) if index == 0 else ""
            rows.append([label, stage, num(float(regret)), f"{num(row['mean_difference'], 4, signed=True)} {interval(row['ci_low'], row['ci_high'], 4)}", pvalue(row["bootstrap_holm_p"]), pvalue(row["ttest_holm_p"])])
    assets.tables["imputed_stages"] = tabular(
        "llrrrr",
        ["Candidate sets", "Stage", "Nested selection", "Nested $-$ validation-best [95\\% CI]", "Holm $p$, bootstrap", "Holm $p$, $t$"],
        rows,
    )


def build_pool_matched(assets: Assets) -> None:
    """The 45-row and 8-row pools tested with the same contrasts and Holm families."""
    table = assets.csv("tabarena_pool_matched/matched_contrasts.csv")
    vb = table[table["right"].eq("meta_selector_single")].set_index("pool")
    for pool, key in [("declared", "full"), ("reduced", "reduced")]:
        for family in ["three", "five"]:
            for test, short in [("bootstrap", "m"), ("ttest", "t"), ("corrected", "nb")]:
                assets.define_p(f"pool.{key}.{family}.{short}holm", pvalue(float(vb.loc[pool, f"{test}_holm_{family}"])))
    direct = assets.csv("tabarena_pool_matched/direct_difference.csv").iloc[0]
    assets.define("pool.direct.diff", num(direct["mean_difference"], 4, signed=True))
    assets.define("pool.direct.ci", interval(direct["ci_low"], direct["ci_high"], 4))
    assets.define_p("pool.direct.mp", pvalue(direct["bootstrap_p"]))
    assets.define_p("pool.direct.nbp", pvalue(direct["nb_p"]))
    assets.define("pool.direct.nbci", interval(direct["nb_ci90_low"], direct["nb_ci90_high"], 4))


def build_matched_snr(assets: Assets) -> None:
    """Remark 1's ratio on the offsets the deployable error-space correction uses."""
    for prefix, folder in [("ta.esnr", "tabarena_error_space"), ("ta.esnrm", "tabarena_error_space_median"),
                           ("tr.esnr", "tabrepo_error_space"), ("tr.esnrm", "tabrepo_error_space_median")]:
        table = assets.csv(f"{folder}/offset_snr_matched.csv")
        for _, row in table.iterrows():
            key = {("all", "method"): "all", ("top_tenth", "method"): "top", ("all", "family"): "fam"}[(row["candidates"], row["by"])]
            assets.define(f"{prefix}.{key}", num(row["ratio"], 2))
            assets.define(f"{prefix}.{key}.ci", interval(row["ratio_ci_low"], row["ratio_ci_high"], 2))
            assets.define(f"{prefix}.{key}.n", str(int(row["n_candidates"])))
            if "offset_quality_spearman" in row and prefix.endswith("esnr"):
                base = prefix.replace("esnr", "offq")
                assets.define(f"{base}.{key}.rho", num(row["offset_quality_spearman"], 2, signed=True))
                assets.define_p(f"{base}.{key}.p", pvalue(row["offset_quality_p"]))


def build_source_groups(assets: Assets) -> None:
    """TabRepo's groups of related datasets and the cluster bootstrap over them."""
    groups = pd.read_csv(Path(__file__).resolve().parents[1] / "configs" / "tabrepo_source_groups.csv", comment="#", dtype=str).fillna("")
    documented = groups[groups["documented_group"].ne("")]
    assets.define("trgrp.groups", str(groups["group"].nunique()))
    assets.define("trgrp.datasets", str(len(groups)))
    assets.define("trgrp.doc.groups", str(documented["documented_group"].nunique()))
    assets.define("trgrp.doc.datasets", str(len(documented)))
    meta = json.loads((assets.out / "tabrepo_source_groups" / "metadata.json").read_text(encoding="utf-8"))
    assets.define("trgrp.clusters", str(int(meta["n_clusters"]["group"])))
    assets.define("trgrp.doc.clusters", str(int(meta["n_clusters"]["documented_group"])))
    clusters = assets.csv("tabrepo_source_groups/cluster_contrasts.csv")
    flips = clusters[(clusters["bootstrap_holm_p"] < ALPHA) != (clusters["cluster_holm_p"] < ALPHA)]
    assets.define("trgrp.flips", str(int(flips.shape[0])))
    random_panels = clusters[clusters["panel"].isin(["ensemble", "score_only"])]
    assets.define("trgrp.widthmax", num(float(random_panels["cluster_width_ratio"].max()), 1))
    shares = assets.csv("tabrepo_source_groups/sibling_share.csv")
    random = shares[shares["panel"].eq("score_only")]
    for grouping, key in [("group", "trgrp.sibling"), ("documented_group", "trgrp.doc.sibling")]:
        values = random[random["grouping"].eq(grouping)]["share"]
        assets.define(f"{key}.mean", percent(float(values.mean())))
        assets.define(f"{key}.min", percent(float(values.min())))
        assets.define(f"{key}.max", percent(float(values.max())))
    grouped = shares[shares["panel"].str.endswith("grouped") & shares["grouping"].eq("group")]
    assets.define("trgrp.grouped.sibling.max", percent(float(grouped["share"].max())))


def same_row_splits(assets: Assets, run: str, review: str) -> int:
    """Splits in which nested selection chooses the same row as the ensemble-inclusive
    train-selected reference (Section 7)."""
    splits = assets.csv(f"{run}/split_results.csv")[["split_id", "selected_policy"]]
    choices = assets.csv(f"{review}/train_selected_reference_choices.csv")
    choices = choices[choices["pool"].eq("ensemble_inclusive")][["split_id", "selected_reference_method"]]
    merged = splits.merge(choices, on="split_id", how="inner", validate="one_to_one")
    if len(merged) != len(splits):
        raise ValueError(f"{review}: the reference choices do not cover every split")
    return int(merged["selected_policy"].eq(merged["selected_reference_method"]).sum())


def build_grouped(assets: Assets) -> None:
    """The TabRepo audit on outer splits that keep related datasets together."""
    for prefix, directory, corrected in [
        ("trg", "score_level_tabrepo_grouped_review", "tabrepo_grouped_corrected_test"),
        ("trgs", "score_level_tabrepo_grouped_score_only_review", "tabrepo_grouped_score_only_corrected_test"),
    ]:
        summary_numbers(assets, prefix, assets.csv(f"{directory}/clustered_summary.csv"), rank_scale=1529.0)
        contrast_numbers(assets, prefix, assets.csv(f"{directory}/contrasts.csv"))
        corrected_keys(assets, prefix, assets.csv(f"{corrected}/corrected_resampled_tests.csv"))
    splits = assets.csv("score_level_tabrepo_grouped/split_results.csv")
    assets.define("trg.splits", str(int(splits.shape[0])))
    assets.define("trg.samerow", str(same_row_splits(assets, "score_level_tabrepo_grouped", "score_level_tabrepo_grouped_review")))
    assets.define("trg.nestedchoice.ethree", str(int(splits["selected_policy"].eq("meta_selector_tabrepo_ensemble_k3").sum())))
    meta = json.loads((assets.out / "score_level_tabrepo_grouped" / "grouped_splits.json").read_text(encoding="utf-8"))
    assets.define("trg.repeats", str(int(meta["repeats"])))
    assets.define("trg.folds", str(int(meta["folds"])))
    comparison = assets.csv("tabrepo_grouped_comparison/grouped_vs_random_subsets.csv")
    for _, row in comparison.iterrows():
        prefix = "trg" if row["panel"] == "ensemble" else "trgs"
        key = POLICY_KEYS[str(row["right"])]
        share = min(float(row["share_subsets_at_or_below_grouped"]), float(row["share_subsets_at_or_above_grouped"]))
        assets.define(f"{prefix}.subset.{key}.tail", percent(share, 1))
        assets.define(f"{prefix}.subset.{key}.range", interval(row["random_subset_low"], row["random_subset_high"], 4))
    score = comparison[comparison["panel"].eq("score_only")]
    tails = np.minimum(score["share_subsets_at_or_below_grouped"], score["share_subsets_at_or_above_grouped"])
    assets.define("trgs.subset.maxtail", percent(float(tails.max()), 1))
    ensemble = comparison[comparison["panel"].eq("ensemble") & ~comparison["right"].eq("train_selected_fixed_reference__ensemble_inclusive")]
    tails = np.minimum(ensemble["share_subsets_at_or_below_grouped"], ensemble["share_subsets_at_or_above_grouped"])
    assets.define("trg.subset.mintail", percent(float(tails.min()), 1))
    assets.define("trg.subsets", str(int(json.loads((assets.out / "tabrepo_grouped_comparison" / "metadata.json").read_text(encoding="utf-8"))["subsets"])))
    tests = assets.csv("tabrepo_grouped_comparison/grouped_vs_random_tests.csv")
    for _, row in tests.iterrows():
        prefix = "trg" if row["panel"] == "ensemble" else "trgs"
        key = POLICY_KEYS[str(row["right"])]
        assets.define(f"{prefix}.two.{key}.shift", num(float(row["shift"]), 4, signed=True))
        assets.define_p(f"{prefix}.two.{key}.likeholm", pvalue(float(row["p_random_like_holm"])))
        assets.define_p(f"{prefix}.two.{key}.ownholm", pvalue(float(row["p_own_holm"])))
    score = tests[tests["panel"].eq("score_only")]
    assets.define("trgs.two.n", str(len(score)))
    rest = score[score["right"].isin(["meta_selector_family_prior_budgeted", "meta_selector_task_adaptive_budgeted", "meta_selector_meta_feature_guard_005"])]
    assets.define_p("trgs.two.rest.minholm", pvalue(float(rest[["p_random_like_holm", "p_own_holm"]].min().min())))
    ensemble = tests[tests["panel"].eq("ensemble")]
    assets.define_p("trg.two.minholm", pvalue(float(ensemble[["p_random_like_holm", "p_own_holm"]].min().min())))


def build_grouped_inner(assets: Assets) -> None:
    """The TabRepo audit on the grouped outer splits with whole groups also left out of every
    inner estimate (scripts/grouped_inner_splits.py), against the random splits and, split by
    split, against the outer-grouped run."""
    for prefix, directory, corrected in [
        ("trgi", "score_level_tabrepo_grouped_inner_review", "tabrepo_grouped_inner_corrected_test"),
        ("trgis", "score_level_tabrepo_grouped_inner_score_only_review", "tabrepo_grouped_inner_score_only_corrected_test"),
    ]:
        summary_numbers(assets, prefix, assets.csv(f"{directory}/clustered_summary.csv"), rank_scale=1529.0)
        contrast_numbers(assets, prefix, assets.csv(f"{directory}/contrasts.csv"))
        corrected_keys(assets, prefix, assets.csv(f"{corrected}/corrected_resampled_tests.csv"))
    splits = assets.csv("score_level_tabrepo_grouped_inner/split_results.csv")
    assets.define("trgi.splits", str(int(splits.shape[0])))
    assets.define("trgi.samerow", str(same_row_splits(assets, "score_level_tabrepo_grouped_inner", "score_level_tabrepo_grouped_inner_review")))
    policies = splits["selected_policy"].astype(str)
    assets.define("trgi.nestedchoice.ens", str(int(policies.str.contains("tabrepo_ensemble").sum())))
    assets.define("trgi.nestedchoice.ethree", str(int(policies.eq("meta_selector_tabrepo_ensemble_k3").sum())))
    inner = json.loads((assets.out / "score_level_tabrepo_grouped_inner" / "grouped_splits.json").read_text(encoding="utf-8"))
    outer = json.loads((assets.out / "score_level_tabrepo_grouped" / "grouped_splits.json").read_text(encoding="utf-8"))
    if [item["heldout"] for item in inner["splits"]] != [item["heldout"] for item in outer["splits"]]:
        raise ValueError("The inner-grouped run does not share the outer-grouped run's splits")
    assets.define("trgi.groups", str(int(inner["n_inner_groups_used"])))
    assets.define("trgi.groupeddatasets", str(int(inner["n_inner_grouped_datasets"])))
    tests = assets.csv("tabrepo_grouped_inner_comparison/grouped_vs_random_tests.csv")
    for _, row in tests.iterrows():
        prefix = "trgi" if row["panel"] == "ensemble" else "trgis"
        key = POLICY_KEYS[str(row["right"])]
        assets.define(f"{prefix}.two.{key}.shift", num(float(row["shift"]), 4, signed=True))
        assets.define_p(f"{prefix}.two.{key}.likeholm", pvalue(float(row["p_random_like_holm"])))
        assets.define_p(f"{prefix}.two.{key}.ownholm", pvalue(float(row["p_own_holm"])))
    paired = assets.csv("tabrepo_grouped_inner_vs_outer/grouped_inner_vs_outer.csv")
    for _, row in paired.iterrows():
        prefix = "trgic" if row["panel"] == "ensemble" else "trgics"
        key = POLICY_KEYS[str(row["right"])]
        assets.define(f"{prefix}.{key}.change", num(float(row["change"]), 4, signed=True))
        assets.define(f"{prefix}.{key}.ci", interval(row["ci_low"], row["ci_high"], 4))
        assets.define_p(f"{prefix}.{key}.holm", pvalue(float(row["p_holm"])))
        assets.define_p(f"{prefix}.{key}.clusterholm", pvalue(float(row["cluster_p_holm"])))
        assets.define(f"{prefix}.{key}.changed", str(int(row["n_datasets_changed"])))
    score = paired[paired["panel"].eq("score_only")]
    assets.define_p("trgics.minholm", pvalue(float(score["p_holm"].min())))
    assets.define("trgics.change.min", num(float(score["change"].min()), 4, signed=True))
    assets.define("trgics.change.max", num(float(score["change"].max()), 4, signed=True))
    assets.define("trgics.nreject", str(int((score["p_holm"] < ALPHA).sum())))
    assets.define("trgics.ncontrasts", str(int(score.shape[0])))
    assets.define_p("trgics.cluster.minholm", pvalue(float(score["cluster_p_holm"].min())))
    for panel, prefix in [("score_only", "trgis"), ("ensemble", "trgi")]:
        rows = tests[tests["panel"].eq(panel)]
        assets.define_p(f"{prefix}.two.minholm", pvalue(float(rows[["p_random_like_holm", "p_own_holm"]].min().min())))
    assets.define("trgi.two.n", str(int(tests.shape[0])))
    assets.define_p("trgi.two.allminholm", pvalue(float(tests[["p_random_like_holm", "p_own_holm"]].min().min())))
    # Validation-best and the rank correction use no training-side choice that the inner
    # grouping changes, so their held-out rows are the same in both runs and a change of a
    # contrast against them is a change of nested selection's own regret (Appendix H).
    references = ["meta_selector_single", "reference_optimism_corrected"]
    keys = ["split_id", "dataset", "method"]
    rows = {}
    for run in ["score_level_tabrepo_grouped_score_only_review", "score_level_tabrepo_grouped_inner_score_only_review"]:
        frame = pd.read_parquet(assets.out / run / "dataset_policy_scores.parquet", columns=keys + ["mean_normalized_rank_regret"])
        rows[run] = frame[frame["method"].isin(references)]
    merged = rows["score_level_tabrepo_grouped_score_only_review"].merge(
        rows["score_level_tabrepo_grouped_inner_score_only_review"], on=keys, suffixes=("_outer", "_inner"), validate="one_to_one")
    if len(merged) != len(rows["score_level_tabrepo_grouped_score_only_review"]) or merged.empty:
        raise ValueError("The two grouped runs do not hold the same reference rows")
    changed = merged["mean_normalized_rank_regret_inner"].ne(merged["mean_normalized_rank_regret_outer"])
    assets.define("trgics.refs.changed", str(int(changed.sum())))
    clusters = assets.csv("tabrepo_source_groups_inner/cluster_contrasts.csv")
    flips = int(((clusters["cluster_holm_p"] < ALPHA) != (clusters["bootstrap_holm_p"] < ALPHA)).sum())
    assets.define("trgi.cluster.flips", str(flips))
    assets.define("trgi.cluster.widthmax", num(float(clusters["cluster_width_ratio"].max()), 1))


def build_construction(assets: Assets) -> None:
    """How the TabRepo slice built RandomForest, ExtraTrees and KNN predictions (App. H)."""
    summary = assets.csv("tabrepo_construction/construction_summary.csv").set_index("family")
    meta = json.loads((assets.out / "tabrepo_construction" / "metadata.json").read_text(encoding="utf-8"))
    assets.define("trc.ndatasets", str(int(meta["n_binary_datasets"])))
    assets.define("trc.ntasks", str(int(summary.loc["rf", "n_tasks"])))
    assets.define("trc.trees", str(int(meta["trees"])))
    assets.define("trc.knnk", str(int(meta["knn_neighbours"])))
    for family in ["rf", "xt"]:
        row = summary.loc[family]
        assets.define(f"trc.{family}.test", percent(float(row["tasks_test_on_1_300"])))
        assets.define(f"trc.{family}.val", percent(float(row["tasks_val_on_1_300"])))
        assets.define(f"trc.{family}.valshare", percent(float(row["median_val_share_1_300"])))
        assets.define(f"trc.{family}.valden", str(int(round(float(row["median_val_denominator"])))))
    knn = summary.loc["knn"]
    assets.define("trc.knn.val", percent(float(knn["tasks_val_on_1_k"])))
    assets.define("trc.knn.test", percent(float(knn["tasks_test_on_1_k_or_1_k1"])))
    assets.define("trc.knn.bag", percent(float(knn["tasks_test_bag_only"])))
    controls = summary.loc[["cat", "gbm"]]
    assets.define("trc.ctrl.lattice", percent(float(controls[["tasks_test_on_1_300", "tasks_val_on_1_300"]].to_numpy().max())))


def build_overlap_source(assets: Assets) -> None:
    """TabRepo without the datasets that share a name or a documented source with TabArena."""
    contrasts = assets.csv("tabrepo_overlap_sensitivity_source/overlap_contrasts.csv")
    meta = json.loads((assets.out / "tabrepo_overlap_sensitivity_source" / "metadata.json").read_text(encoding="utf-8"))
    assets.define("ovls.excluded", str(int(meta["n_excluded"])))
    assets.define("ovls.n", str(int(next(iter(meta["panels"].values()))["n_kept"])))
    full = {
        "ensemble": assets.csv("score_level_tabrepo_nested_review/contrasts.csv").set_index(["family", "left", "right"]),
        "score_only": assets.csv("score_level_tabrepo_score_only_review/contrasts.csv").set_index(["family", "left", "right"]),
    }
    # Verdict changes (Holm-adjusted bootstrap test), sign changes of differences whose full
    # interval excludes zero, and changes in whether an interval excludes zero.
    flips = signs = intervals = 0
    for _, row in contrasts.iterrows():
        reference = full[row["panel"]].loc[(row["family"], row["left"], row["right"])]
        flips += int((float(row["bootstrap_holm_p"]) < ALPHA) != (float(reference["bootstrap_holm_p"]) < ALPHA))
        excludes = bool(reference["ci_low"] > 0 or reference["ci_high"] < 0)
        signs += int(excludes and np.sign(row["mean_difference"]) != np.sign(reference["mean_difference"]))
        intervals += int(excludes != bool(row["ci_low"] > 0 or row["ci_high"] < 0))
    assets.define("ovls.flips", str(flips))
    assets.define("ovls.signs", str(signs))
    assets.define("ovls.intervals", str(intervals))
    contrast_numbers(assets, "ovls.tr", contrasts[contrasts["panel"].eq("ensemble")])


def build_calibration_design(assets: Assets) -> None:
    """The calibration study's declared design (configs/calibration_study.yaml, world.json)."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "calibration_study.yaml").read_text(encoding="utf-8"))
    world_path = assets.out / "calibration_study" / "world.json"
    if not world_path.exists():
        raise Missing(str(world_path))
    world = json.loads(world_path.read_text(encoding="utf-8"))
    assets.define("cal.shrinkage", percent(float(world["shrinkage"])))
    assets.define("cal.base.familyshare", percent(float(world["family_noise_share"]["test"])))
    assets.define("cal.bridge.tables", str(int(config["bridge"]["tables"])))
    assets.define("cal.splits", str(int(config["splits"])))
    assets.define("cal.truthruns.base", str(int(config["truth_runs"]["base"])))
    assets.define("cal.reps.base", str(int(config["replicates"]["base"])))
    others = sorted({int(v) for k, v in config["replicates"].items() if k != "base"})
    assets.define("cal.reps.other", "/".join(str(v) for v in others))
    scenarios = config["scenarios"]
    assets.define("cal.skew.factor", num(float(scenarios["skew"]["small_gap_factor"]), 1))
    assets.define("cal.instability.sd", num(float(scenarios["instability"]["tuning_sd"]), 2))
    assets.define("cal.dependence.share", percent(float(scenarios["dependence"]["family_noise_share"])))
    # The adequacy rule, declared in the config before any coverage estimate was computed.
    rule = " ".join(str(config["adequacy_rule"]).split())
    if not all(part in rule for part in ["Wilson", "below 0.925", "above 0.975", "at least 0.925", "ratio is at least 0.9"]):
        raise ValueError("The coverage study's adequacy rule no longer states 0.925, 0.975 and 0.9; update the build.")
    assets.define("cal.rule.coverage", "0.925")
    assets.define("cal.rule.upper", "0.975")
    assets.define("cal.rule.seratio", "0.9")
    assets.define("cal.rule.earlier", str(len(config.get("adequacy_rule_history", []))))
    assets.define("cal.truthshared.base", str(int(config["truth_shared_runs"]["base"])))
    # The lower Wilson bound at full coverage: the conservative label needs it above 0.975.
    assets.define("cal.rule.maxlower", num(wilson_interval(int(config["replicates"]["base"]), int(config["replicates"]["base"]))[0], 3))
    others = sorted({int(v) for k, v in config["truth_runs"].items() if k != "base"})
    assets.define("cal.truthruns.other", "/".join(str(v) for v in others))
    reps = sorted({int(v) for k, v in config["replicates"].items() if k != "base"})
    assets.define("cal.detect.base", num(detectable(int(config["replicates"]["base"])), 2))
    assets.define("cal.detect.other", num(detectable(reps[0]), 2) if len(reps) == 1 else "/".join(num(detectable(r), 2) for r in reps))
    facts = assets.csv("calibration_study/world_facts.csv")
    for column in facts.columns[1:]:
        assets.define(f"cal.fact.ta.{column.replace('_', '')}", num(float(facts.iloc[0][column]), 3))
        for source in facts["source"].iloc[1:]:
            assets.define(f"cal.fact.{source}.{column.replace('_', '')}", num(float(facts[facts["source"].eq(source)].iloc[0][column]), 3))
    names = {"vb_regret": "Validation-best regret", "vb_regret_large": "\\quad on datasets with 9 units", "vb_regret_small": "\\quad on datasets with 30 units",
             "vb_default_regret": "Validation-best among defaults", "defaults_rule_regret": "Defaults rule",
             "vb_tuned_ensemble_share": "Validation-best picks tuned + ensemble", "test_best_default": "Test-best candidate is a default",
             "test_best_tuned": "\\ldots{} tuned", "test_best_tuned_ensemble": "\\ldots{} tuned + ensemble"}
    fact_rows = [[name] + [num(float(value), 3) for value in facts[column]] for column, name in names.items()]
    assets.tables["calibration_facts"] = tabular(
        "l" + "r" * len(facts),
        ["Quantity"] + ["TabArena" if i == 0 else str(source) for i, source in enumerate(facts["source"])],
        fact_rows,
    )


def wilson_interval(covered: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The 95% Wilson interval of a proportion (as in scripts/calibration_study.py)."""
    p = covered / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def detectable(total: int) -> float:
    """The largest coverage that ``total`` replicates label anti-conservative."""
    return max(k for k in range(total + 1) if wilson_interval(k, total)[1] < 0.925) / total


CAL_DESIGNS = {"per-target, J=20": "pt", "shared, J=100": "sh100", "shared, J=20": "sh20", "per-target, J=20, bridge tables": "ptb"}
CAL_METHODS = {"boot": "Bootstrap", "t": "Paired $t$", "nb": "Corrected", "nbt": "Corrected, task weights"}
CAL_CONTRASTS = {
    "C1": "Nested $-$ validation-best", "C2": "Nested $-$ family-prior", "C3": "Nested $-$ train-selected",
    "C4": "Family-prior $-$ validation-best", "C5": "Defaults rule $-$ validation-best", "C6": "Validation-best (defaults) $-$ validation-best",
}
CAL_MARK = {"anti-conservative": "$^{-}$", "conservative": "$^{+}$", "inconclusive": "$^{?}$", "adequate": ""}


def main_design(coverage: pd.DataFrame) -> pd.DataFrame:
    return coverage[coverage["design"].eq("per-target, J=20")]


def build_calibration(assets: Assets) -> None:
    """Results of the coverage study (scripts/calibration_study.py, stage summarize)."""
    coverage = assets.csv("calibration_study/coverage.csv")
    meta = json.loads((assets.out / "calibration_study" / "metadata.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "calibration_study.yaml").read_text(encoding="utf-8"))
    for kind, done in [("truth_runs", meta["truth_runs"]), ("replicates", meta["replicates"]), ("truth_shared_runs", meta.get("truth_shared_runs", {}))]:
        if {k: int(v) for k, v in done.items()} != {k: int(v) for k, v in config[kind].items()}:
            raise ValueError(f"The coverage study is incomplete: {kind} {done} against {config[kind]}")
    if set(coverage["design"]) != set(CAL_DESIGNS):
        raise ValueError(f"Unexpected coverage designs: {sorted(set(coverage['design']))}")
    if float(meta["max_nofit_vb_gap"]) > 1e-12 or float(meta["max_unaffected_truth_gap"]) > 1e-12:
        raise ValueError("The coverage study's reproduction checks failed.")
    for _, row in coverage.iterrows():
        design = CAL_DESIGNS[row["design"]]
        base = f"cal.{design}.{row['scenario']}.{row['contrast'].lower()}.{row['method']}"
        assets.define(f"{base}.cov", num(float(row["coverage"]), 2))
        assets.define(f"{base}.covci", interval(float(row["coverage_low"]), float(row["coverage_high"]), 2))
        assets.define(f"{base}.seratio", num(float(row["se_ratio"]), 2))
        assets.define(f"{base}.seratioci", interval(float(row["se_ratio_low"]), float(row["se_ratio_high"]), 2))
        assets.define(f"{base}.reject", num(float(row["reject_rate"]), 2))
        assets.define(f"{base}.verdict", str(row["verdict"]))
        assets.define(f"cal.{design}.{row['scenario']}.{row['contrast'].lower()}.theta", num(float(row["theta"]), 4, signed=True))
    # Label codes for the guards (0 adequate, 1 inconclusive, 2 anti-conservative, 3 conservative),
    # the number of scenarios with each label at the replicates' design, and the skewness of the
    # dataset-level differences, which drives the bootstrap's tails.
    codes = {"adequate": 0, "inconclusive": 1, "anti-conservative": 2, "conservative": 3}
    for _, row in coverage.iterrows():
        assets.define(f"cal.{CAL_DESIGNS[row['design']]}.{row['scenario']}.{row['contrast'].lower()}.{row['method']}.code", str(codes[row["verdict"]]))
    for (contrast, method), group in main_design(coverage).groupby(["contrast", "method"]):
        assets.define(f"cal.pt.{contrast.lower()}.{method}.nanti", str(int(group["verdict"].eq("anti-conservative").sum())))
        assets.define(f"cal.pt.{contrast.lower()}.{method}.nadequate", str(int(group["verdict"].eq("adequate").sum())))
    for _, row in main_design(coverage)[lambda f: f["method"].eq("boot")].iterrows():
        assets.define(f"cal.skewness.{row['scenario']}.{row['contrast'].lower()}", num(float(row["mean_skew"]), 2, signed=True))
    # Ranges by kind of contrast (fitted: C1-C4; no fit: C5-C6), for the text.
    for (design, scenario, kind, method), group in coverage.groupby(["design", "scenario", "kind", "method"]):
        base = f"cal.{CAL_DESIGNS[design]}.{scenario}.{'fit' if kind == 'fitted' else 'nofit'}.{method}"
        assets.define(f"{base}.covmin", num(float(group["coverage"].min()), 2))
        assets.define(f"{base}.covmax", num(float(group["coverage"].max()), 2))
        assets.define(f"{base}.ratiomin", num(float(group["se_ratio"].min()), 2))
        assets.define(f"{base}.ratiomax", num(float(group["se_ratio"].max()), 2))
    # theta's own Monte Carlo error: its largest share of a replicate's standard deviation, and
    # the most it can lower the estimated coverage of a correct 95% interval.
    ratio = float(coverage["theta_se_over_sd"].max())
    loss = 0.95 - float(2 * stats.norm.cdf(1.959963984540054 / math.sqrt(1 + ratio**2)) - 1)
    assets.define("cal.thetase.max", num(ratio, 2))
    assets.define("cal.thetaloss.max", num(100 * loss, 1))
    assets.define("cal.dropped", str(int(meta["replicates_with_a_dataset_never_held_out"])))
    changed = coverage[(coverage["verdict"] != coverage["verdict_first"]) | (coverage["verdict"] != coverage["verdict_second"])]
    assets.define("cal.labels.changed", str(int(changed.shape[0])))
    assets.define("cal.nconservative", str(int(coverage["verdict"].eq("conservative").sum())))
    assets.define("cal.labels.total", str(int(coverage.shape[0])))
    for contrast, shift in meta["inner_fit_theta_shift"].items():
        assets.define(f"cal.innershift.{contrast.lower()}", num(float(shift), 4, signed=True))
    # The main table: every scenario and contrast at the replicates' design.
    main = coverage[coverage["design"].eq("per-target, J=20")]
    rows = []
    for scenario in config["scenarios"]:
        for contrast, label in CAL_CONTRASTS.items():
            item = main[main["scenario"].eq(scenario) & main["contrast"].eq(contrast)].set_index("method")
            cells = [f"{num(float(item.loc[m, 'coverage']), 2)}{CAL_MARK[item.loc[m, 'verdict']]} ({num(float(item.loc[m, 'se_ratio']), 2)})" for m in CAL_METHODS]
            rows.append([scenario if contrast == "C1" else "", label, num(float(item.loc["boot", "theta"]), 4, signed=True)] + cells)
    assets.tables["calibration_coverage"] = tabular("llr" + "r" * len(CAL_METHODS), ["Scenario", "Contrast", "$\\theta$"] + list(CAL_METHODS.values()), rows)
    # The bridge: base tables at the replicates' number of splits and at the audit's.
    rows = []
    for contrast, label in CAL_CONTRASTS.items():
        cells = []
        for method in ["boot", "nb", "nbt"]:
            for design in ["shared, J=20", "shared, J=100"]:
                item = coverage[coverage["design"].eq(design) & coverage["contrast"].eq(contrast) & coverage["method"].eq(method)].iloc[0]
                cells.append(f"{num(float(item['coverage']), 2)}{CAL_MARK[item['verdict']]} ({num(float(item['se_ratio']), 2)})")
        theta = coverage[coverage["design"].eq("shared, J=100") & coverage["contrast"].eq(contrast)].iloc[0]["theta"]
        rows.append([label, num(float(theta), 4, signed=True)] + cells)
    assets.tables["calibration_bridge"] = tabular(
        "lr" + "r" * 6,
        ["Contrast", "$\\theta$", "Bootstrap, $J=20$", "$J=100$", "Corrected, $J=20$", "$J=100$", "Task weights, $J=20$", "$J=100$"],
        rows,
    )


# The paper calls the rank correction "offset-corrected validation" (Section 5): what it
# removes is a candidate's offset, which mixes selection optimism with the effects of how
# predictions were constructed. The pipeline's outputs keep their stored label.
PAPER_NAMES = {"Optimism-corrected validation": "Offset-corrected validation"}


def paper_names(text: str) -> str:
    for old, new in PAPER_NAMES.items():
        text = text.replace(old, new)
    return text


def write(assets: Assets, generated: Path) -> None:
    (generated / "tables").mkdir(parents=True, exist_ok=True)
    # A complete build makes an undefined key a LaTeX error; a draft build (--allow-missing)
    # prints a bold ?? and a warning so the paper still compiles while runs are pending.
    undefined = (
        "\\textbf{??}\\PackageWarning{results}{Undefined result key #1}"
        if assets.allow_missing
        else "\\errmessage{Undefined result key #1}"
    )
    lines = [
        "% Generated by scripts/build_paper_assets.py from analysis-output/v2; do not edit by hand.",
        "\\makeatletter",
        "\\newcommand{\\defres}[2]{\\expandafter\\def\\csname res@#1\\endcsname{#2}}",
        "\\newcommand{\\res}[1]{\\ifcsname res@#1\\endcsname\\csname res@#1\\endcsname"
        f"\\else{undefined}\\fi}}",
        "\\makeatother",
    ]
    for key in sorted(assets.numbers):
        lines.append(f"\\defres{{{key}}}{{{paper_names(assets.numbers[key])}}}")
    (generated / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (generated / "labels.tex").write_text(paper_names(latex_macros()), encoding="utf-8")
    for name, body in assets.tables.items():
        (generated / "tables" / f"{name}.tex").write_text(
            "% Generated by scripts/build_paper_assets.py; do not edit by hand.\n" + paper_names(body), encoding="utf-8"
        )


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    assets = Assets(
        Path(args.out), allow_missing=bool(args.allow_missing), meta_verified_only=bool(args.meta_verified_only)
    )
    assets.section("tabarena", lambda: build_tabarena(assets))
    assets.section("mean tests", lambda: build_mean_tests(assets))
    assets.section(
        "tabarena provenance",
        lambda: build_provenance(assets, "ta", "tabarena_provenance", variants=True, rank_scale=43.0),
    )
    assets.section("tabarena members", lambda: build_members(assets))
    assets.section(
        "tabrepo provenance",
        lambda: build_provenance(assets, "tr", "tabrepo_provenance", variants=False, rank_scale=1529.0),
    )
    assets.section(
        "tabarena optimism snr",
        lambda: build_optimism_snr(assets, "ta", "tabarena_provenance"),
    )
    assets.section(
        "tabrepo optimism snr",
        lambda: build_optimism_snr(assets, "tr", "tabrepo_provenance"),
    )
    assets.section("tabarena median distance", lambda: build_median_distance(assets, "ta", "tabarena_provenance"))
    assets.section("tabrepo median distance", lambda: build_median_distance(assets, "tr", "tabrepo_provenance"))
    assets.section("tabarena tie rule", lambda: build_tie_rule(assets, "ta", "tabarena_tie_rule"))
    assets.section("tabrepo tie rule", lambda: build_tie_rule(assets, "tr", "tabrepo_tie_rule"))
    assets.section("tabarena factorial", lambda: build_factorial(assets, "ta", "tabarena_factorial"))
    assets.section("tabrepo factorial", lambda: build_factorial(assets, "tr", "tabrepo_factorial"))
    assets.section("grid", lambda: build_grid(assets))
    assets.section("tabarena search", lambda: build_search(assets, "ta", "tabarena_search_inference"))
    assets.section("tabrepo search", lambda: build_search(assets, "tr", "score_level_tabrepo_nested_search_inference"))
    assets.section("tabrepo score-only search", lambda: build_search(assets, "trs", "score_level_tabrepo_score_only_search_inference"))
    assets.section("tabarena evidence", lambda: build_evidence(assets, "ta", "tabarena_evidence_unit"))
    assets.section("tabrepo evidence", lambda: build_evidence(assets, "tr", "score_level_tabrepo_nested_evidence_unit"))
    assets.section(
        "tabrepo score-only evidence",
        lambda: build_evidence(assets, "trs", "score_level_tabrepo_score_only_evidence_unit"),
    )
    assets.section("small runs", lambda: build_small_runs(assets))
    assets.section("tabrepo", lambda: build_tabrepo(assets))
    assets.section("misc", lambda: build_misc(assets))
    assets.section("tabrepo overlap", lambda: build_overlap(assets))
    assets.section("attribution", lambda: build_attribution(assets))
    assets.section("tabrepo attribution", lambda: build_attribution_tabrepo(assets))
    assets.section("jackknife", lambda: build_jackknife(assets))
    assets.section("meta-study", lambda: build_meta_study(assets, root))
    assets.section("coverage", lambda: build_coverage(assets, root))
    assets.section("families", lambda: build_families(assets))
    assets.section("corrected tests", lambda: build_corrected(assets))
    assets.section("imputed rows", lambda: build_imputed(assets))
    assets.section("influence", lambda: build_influence(assets))
    assets.section("error-space correction", lambda: build_error_space(assets))
    assets.section("pool distinct rows", lambda: build_pool_distinct(assets))
    assets.section("tabrepo candidates", lambda: build_tabrepo_candidates(assets))
    assets.section("provenance reference", lambda: build_provenance_reference(assets))
    assets.section("documented imputation mask", lambda: build_flagged(assets))
    assets.section("matched pool families", lambda: build_pool_matched(assets))
    assets.section("matched spread-to-noise", lambda: build_matched_snr(assets))
    assets.section("tabrepo source groups", lambda: build_source_groups(assets))
    assets.section("tabrepo grouped splits", lambda: build_grouped(assets))
    assets.section("tabrepo source overlap", lambda: build_overlap_source(assets))
    assets.section("tabrepo construction", lambda: build_construction(assets))
    assets.section("tabrepo grouped inner", lambda: build_grouped_inner(assets))
    assets.section("calibration design", lambda: build_calibration_design(assets))
    assets.section("calibration", lambda: build_calibration(assets))
    if assets.allow_missing and (root / "paper" / "main.tex").exists():
        # Draft build: stand-ins for the manuscript's tables whose inputs are still being computed.
        manuscript = (root / "paper" / "main.tex").read_text(encoding="utf-8")
        for name in re.findall(r"\\input\{generated/tables/([A-Za-z0-9_]+)\.tex\}", manuscript):
            if name not in assets.tables:
                assets.tables[name] = "\\begin{tabular}{l}\\textbf{?? table pending}\\end{tabular}\n"
                assets.missing.append(f"table {name}: placeholder written")
    problems = check_stated_verdicts(assets) + check_stated_comparisons(assets)
    if problems and not assets.allow_missing:
        raise ValueError("Stated verdicts disagree with the numbers:\n  " + "\n  ".join(problems))
    assets.missing.extend(f"stated verdict: {problem}" for problem in problems)
    write(assets, Path(args.generated))
    print(f"{len(assets.numbers)} numbers, {len(assets.tables)} tables")
    for item in assets.missing:
        print(f"MISSING {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="analysis-output/v2")
    parser.add_argument("--generated", default="paper/generated")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--meta-verified-only",
        action="store_true",
        help="Use only meta-study claims whose coding the author has verified (for submission).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
