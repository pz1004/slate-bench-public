"""One display name per policy id, shared by every table, figure and LaTeX macro.

Tables, figures and prose must use the same strings. Scripts look labels up here instead
of keeping their own maps, and ``scripts/build_paper_assets.py`` turns the registry into
LaTeX macros (``\\pol<Key>``) so the manuscript never retypes a policy name.
"""

from __future__ import annotations

import re


# policy id -> (display label, LaTeX macro suffix)
POLICY_LABELS: dict[str, tuple[str, str]] = {
    "nested_policy_selected": ("Nested selection", "Nested"),
    "train_selected_fixed_reference__declared_fixed": ("Train-selected fixed rule", "TrainFixed"),
    "train_selected_fixed_reference__validation_only": (
        "Train-selected fixed rule (validation rows)",
        "TrainFixedValidation",
    ),
    "train_selected_fixed_reference__family_prior": (
        "Train-selected fixed rule (family-prior rows)",
        "TrainFixedFamily",
    ),
    "train_selected_fixed_reference__with_calibrated": (
        "Train-selected fixed rule (with calibrated rule)",
        "TrainFixedCalibrated",
    ),
    "train_selected_fixed_reference__broader": (
        "Train-selected rule (fixed and adaptive rows)",
        "TrainBroader",
    ),
    "train_selected_fixed_reference__score_only": ("Train-selected fixed rule", "TrainFixedScore"),
    "train_selected_fixed_reference__ensemble_inclusive": (
        "Train-selected fixed rule (with ensembles)",
        "TrainFixedEnsemble",
    ),
    "meta_selector_family_prior_budgeted": ("Family-prior rule", "FamilyPrior"),
    "meta_selector_family_prior_budgeted_100s": ("Family-prior (100 s)", "FamilyPriorHundred"),
    "meta_selector_family_prior_budgeted_500s": ("Family-prior (500 s)", "FamilyPriorFiveHundred"),
    "meta_selector_family_prior_budgeted_1000s": ("Family-prior (1000 s)", "FamilyPriorThousand"),
    "meta_selector_family_prior_budgeted_5000s": ("Family-prior (5000 s)", "FamilyPriorFiveThousand"),
    "meta_selector_calibrated_budget_10000s": ("Calibrated rule", "Calibrated"),
    "meta_selector_task_adaptive_budgeted": ("Task-adaptive", "TaskAdaptive"),
    "meta_selector_meta_feature_guard_002": ("Guard ($\\delta = 0.02$)", "GuardTwo"),
    "meta_selector_meta_feature_guard_005": ("Guard ($\\delta = 0.05$)", "GuardFive"),
    "meta_selector_meta_feature_guard_010": ("Guard ($\\delta = 0.10$)", "GuardTen"),
    "meta_selector_meta_feature_guard_020": ("Guard ($\\delta = 0.20$)", "GuardTwenty"),
    "meta_selector_single": ("Validation-best", "ValidationBest"),
    # The pool's validation-best row as implemented: validation error plus 1e-6 log(1 + fit seconds).
    # On TabRepo the literal rule takes the name above as the reference (scripts/literal_validation_best.py).
    "meta_selector_single_runtime": ("Validation-best with a runtime term", "ValidationBestRuntime"),
    "meta_selector_ablate_validation_only": ("Validation rank", "ValidationRank"),
    "reference_single_best_global": ("Single best (global)", "SingleBest"),
    "reference_single_best_task": ("Single best (per task)", "SingleBestTask"),
    "reference_validation_best_default": ("Validation-best (defaults)", "VBDefault"),
    "reference_defaults_then_tuned": ("Defaults, then tuned + ensemble", "DefaultsThenTuned"),
    "reference_validation_best_tuned": ("Validation-best (tuned)", "VBTuned"),
    "reference_validation_best_tuned_ensemble": (
        "Validation-best (tuned + ensemble)",
        "VBTunedEnsemble",
    ),
    "reference_optimism_corrected": ("Optimism-corrected validation", "OptimismCorrected"),
    "reference_optimism_corrected_budget": (
        "Optimism-corrected validation (10\\,000\\,s)",
        "OptimismCorrectedBudget",
    ),
    "reference_optimism_corrected_task": (
        "Optimism-corrected validation (per task)",
        "OptimismCorrectedTask",
    ),
    "reference_priors_only": ("Priors only (no validation)", "PriorsOnly"),
    "reference_metafeature_knn": ("Meta-feature kNN", "MetaKnn"),
    "reference_metafeature_knn_validation": ("Meta-feature kNN + validation", "MetaKnnValidation"),
    "extended_metafeature_knn": ("Meta-feature kNN", "ExtMetaKnn"),
    "extended_metafeature_knn_validation": ("Meta-feature kNN + validation", "ExtMetaKnnValidation"),
    "extended_descriptor_guard_002": ("Descriptor guard ($\\delta = 0.02$)", "ExtDescriptorGuardTwo"),
    "extended_descriptor_guard_005": ("Descriptor guard ($\\delta = 0.05$)", "ExtDescriptorGuardFive"),
    "meta_selector_tabrepo_ensemble_k3": ("Ensemble ($k = 3$)", "EnsembleThree"),
    "meta_selector_tabrepo_ensemble_k5": ("Ensemble ($k = 5$)", "EnsembleFive"),
    "meta_selector_tabrepo_ensemble_k10": ("Ensemble ($k = 10$)", "EnsembleTen"),
    "oracle": ("Oracle bound", "Oracle"),
}

_GUARD_PATTERN = re.compile(
    r"^meta_selector_meta_feature_guard_(?P<margin>\d{3})_k(?P<k>\d+)_rt(?P<rt>[0-9p]+)$"
)


def policy_label(policy: object) -> str:
    """Display label; unknown guard variants are spelled out, anything else is returned."""
    key = str(policy)
    if key in POLICY_LABELS:
        return POLICY_LABELS[key][0]
    match = _GUARD_PATTERN.match(key)
    if match:
        margin = int(match.group("margin")) / 100.0
        runtime = match.group("rt").replace("p", ".")
        return f"Guard ($\\delta = {margin:.2f}$, $k = {match.group('k')}$, $\\lambda = {runtime}$)"
    return key


def plain_label(policy: object) -> str:
    """The label without LaTeX math markup (for figures rendered by matplotlib)."""
    text = policy_label(policy)
    text = text.replace("$\\delta = ", "δ = ").replace("$k = ", "k = ").replace("$\\lambda = ", "λ = ")
    return text.replace("$", "").replace("\\,", "\u2009")


def latex_macro_name(policy: object) -> str:
    key = str(policy)
    if key not in POLICY_LABELS:
        raise KeyError(f"Policy {key!r} has no LaTeX macro; add it to POLICY_LABELS.")
    return f"pol{POLICY_LABELS[key][1]}"


def latex_macros() -> str:
    lines = ["% Generated from src/tabular_sota_benchmark/labels.py; do not edit by hand."]
    for key, (label, suffix) in POLICY_LABELS.items():
        lines.append(f"\\newcommand{{\\pol{suffix}}}{{{label}\\xspace}}  % {key}")
    return "\n".join(lines) + "\n"
