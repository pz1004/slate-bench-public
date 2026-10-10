from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_sota_benchmark.config import load_config
from tabular_sota_benchmark.failures import append_failure


@dataclass(frozen=True)
class VerificationResult:
    step: str
    passed: bool
    baseline: str
    verification: str
    reason: str = ""
    cause_analysis: str = ""
    corrective_action: str = ""
    restart_point: str = ""


def _record_if_failed(result: VerificationResult) -> VerificationResult:
    if not result.passed:
        append_failure(
            step=result.step,
            baseline=result.baseline,
            verification=result.verification,
            reason=result.reason,
            cause_analysis=result.cause_analysis,
            corrective_action=result.corrective_action,
            restart_point=result.restart_point,
        )
    return result


def verify_step_1() -> VerificationResult:
    baseline = "SOTA context and idea are documented before experiment claims."
    paths = [Path("docs/sota_baselines.md"), Path("docs/claim_ledger.md")]
    if not all(path.exists() for path in paths):
        return _record_if_failed(
            VerificationResult(
                "Step 1",
                False,
                baseline,
                "Required docs exist.",
                "Missing Step 1 documentation.",
                "The idea cannot be audited against current SOTA context.",
                "Create `docs/sota_baselines.md` and `docs/claim_ledger.md`.",
                "Step 1",
            )
        )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    required = ["TabArena", "TabPFN-3", "AutoGluon", "TabICL", "meta-selector"]
    missing = [item for item in required if item not in text]
    if missing:
        return _record_if_failed(
            VerificationResult(
                "Step 1",
                False,
                baseline,
                "Required SOTA terms are present.",
                f"Missing terms: {', '.join(missing)}.",
                "The idea may ignore a current high-impact baseline.",
                "Update SOTA context and claim ledger.",
                "Step 1",
            )
        )
    return VerificationResult("Step 1", True, baseline, "Required SOTA docs are present.")


def verify_step_2() -> VerificationResult:
    baseline = "Theory file contains assumptions, propositions, proof sketches, and limitations."
    path = Path("docs/theory.md")
    if not path.exists():
        return _record_if_failed(
            VerificationResult(
                "Step 2",
                False,
                baseline,
                "`docs/theory.md` exists.",
                "Theory file missing.",
                "The selector has no mathematical claim boundary.",
                "Create mathematical setup and claim limits.",
                "Step 2",
            )
        )
    text = path.read_text(encoding="utf-8")
    required = ["Assumptions", "Proposition", "Proof sketch", "Limitations"]
    missing = [item for item in required if item not in text]
    if missing:
        return _record_if_failed(
            VerificationResult(
                "Step 2",
                False,
                baseline,
                "Required theory sections are present.",
                f"Missing sections: {', '.join(missing)}.",
                "The manuscript could overstate the mathematical result.",
                "Add missing theory sections.",
                "Step 2",
            )
        )
    return VerificationResult("Step 2", True, baseline, "Theory baseline is present.")


def verify_step_3(config_path: str | Path) -> VerificationResult:
    config = load_config(config_path)
    verification = config.get("verification", {})
    output_dir = Path(config["experiment"]["output_dir"])
    result_path = output_dir / "results.csv"
    experiment_name = config["experiment"].get("name", "configured")
    baseline = f"{experiment_name} results contain finite metrics across required datasets and models."
    if not result_path.exists():
        return _record_if_failed(
            VerificationResult(
                "Step 3",
                False,
                baseline,
                "`results.csv` exists.",
                "No experiment results found.",
                "Benchmark did not run or wrote to a different output directory.",
                "Run `run-experiment` or `run-loop`.",
                "Step 3",
            )
        )

    results = pd.read_csv(result_path, low_memory=False)
    min_datasets = int(verification.get("min_datasets", 1))
    min_models = int(verification.get("min_candidate_models", 1))
    require_selector = bool(verification.get("require_selector", True))
    candidate_count = results[results["method_type"] == "candidate"]["method"].nunique()
    dataset_count = results["dataset"].nunique()
    finite_metrics = np.isfinite(results["primary_metric"].to_numpy(dtype=float)).all()
    selector_present = (results["method_type"] == "selector").any()
    unit_columns = [
        column
        for column in ["dataset", "task", "primary_metric_name", "fold", "seed"]
        if column in results.columns
    ]
    comparison_units = results[unit_columns].drop_duplicates().shape[0]

    reasons: list[str] = []
    if dataset_count < min_datasets:
        reasons.append(f"datasets {dataset_count} < {min_datasets}")
    if candidate_count < min_models:
        reasons.append(f"candidate models {candidate_count} < {min_models}")
    if not finite_metrics:
        reasons.append("non-finite primary metrics")
    if require_selector and not selector_present:
        reasons.append("selector rows missing")
    required_selectors = verification.get("selector_must_be_present")
    if required_selectors:
        if isinstance(required_selectors, str):
            required_selector_names = [required_selectors]
        else:
            required_selector_names = [str(value) for value in required_selectors]
        present_selectors = set(
            results.loc[results["method_type"].eq("selector"), "method"].astype(str)
        )
        missing_selectors = [
            name for name in required_selector_names if name not in present_selectors
        ]
        if missing_selectors:
            reasons.append(
                "required selector methods missing: " + ", ".join(missing_selectors)
            )
    min_units = verification.get("min_comparison_units")
    if min_units is not None and comparison_units < int(min_units):
        reasons.append(f"comparison units {comparison_units} < {int(min_units)}")

    if bool(verification.get("selector_must_beat_median_rank", False)) and selector_present:
        ranked = results.copy()
        ranked["rank"] = ranked.groupby(unit_columns)["primary_metric"].rank(
            ascending=True,
            method="average",
        )
        mean_ranks = ranked.groupby("method")["rank"].mean()
        reference_method = config.get("selector", {}).get(
            "reference_method_name",
            config.get("selector", {}).get(
                "ensemble_method_name",
                config.get("selector", {}).get("selector_method_name", "meta_selector_ensemble"),
            ),
        )
        selector_rank = float(mean_ranks.get(reference_method, np.inf))
        candidate_median_rank = float(
            mean_ranks[mean_ranks.index.isin(results[results["method_type"] == "candidate"]["method"].unique())].median()
        )
        if selector_rank > candidate_median_rank:
            reasons.append(
                f"{reference_method} mean rank {selector_rank:.4f} worse than candidate median rank {candidate_median_rank:.4f}"
            )

    if reasons:
        return _record_if_failed(
            VerificationResult(
                "Step 3",
                False,
                baseline,
                "Dataset/model counts, finite metrics, and selector rank gate.",
                "; ".join(reasons),
                "The experiment is incomplete or the selector is not yet competitive under the configured baseline.",
                f"Inspect `{result_path}`, revise selector/config, and rerun.",
                "Step 3",
            )
        )
    return VerificationResult("Step 3", True, baseline, "Experiment metrics passed smoke gates.")


def verify_step_4(config_path: str | Path) -> VerificationResult:
    config = load_config(config_path)
    analysis_dir = Path(config["experiment"]["analysis_dir"])
    baseline = "Paper artifacts exist and keep full SOTA claims bounded."
    required_paths = [
        Path("paper/main.md"),
        Path("paper/results_auto.md"),
        analysis_dir / "analysis-report.md",
        analysis_dir / "stats-appendix.md",
        Path("docs/claim_ledger.md"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return _record_if_failed(
            VerificationResult(
                "Step 4",
                False,
                baseline,
                "Required paper and analysis artifacts exist.",
                f"Missing files: {', '.join(missing)}.",
                "Results cannot be organized into manuscript evidence.",
                "Run analysis and paper generation.",
                "Step 4",
            )
        )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in required_paths)
    forbidden_phrases = [
        "we establish full tabular SOTA",
        "we surpass current full tabular SOTA",
        "we beat TabPFN-3",
        "our method beats TabPFN-3",
    ]
    combined_lower = combined.lower()
    found = [phrase for phrase in forbidden_phrases if phrase.lower() in combined_lower]
    if found:
        return _record_if_failed(
            VerificationResult(
                "Step 4",
                False,
                baseline,
                "No unsupported full-SOTA wording appears.",
                f"Unsupported wording found: {', '.join(found)}.",
                "The manuscript text overstates current evidence.",
                "Weaken paper wording or add compatible SOTA evidence.",
                "Step 4",
            )
        )
    return VerificationResult("Step 4", True, baseline, "Paper artifacts passed claim boundary gate.")


def verify_all(config_path: str | Path) -> list[VerificationResult]:
    return [
        verify_step_1(),
        verify_step_2(),
        verify_step_3(config_path),
        verify_step_4(config_path),
    ]
