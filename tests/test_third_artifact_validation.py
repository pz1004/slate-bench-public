from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "third_artifact_validation.py"
SPEC = importlib.util.spec_from_file_location("third_artifact_validation", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
third = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(third)


def _row(
    *,
    dataset: str,
    method: str,
    method_type: str,
    primary_metric: float,
    selected_member: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "fold": 0,
        "seed": 0,
        "task": "classification",
        "primary_metric_name": "log_loss",
        "method": method,
        "method_type": method_type,
        "primary_metric": primary_metric,
        "val_primary_metric": primary_metric,
        "fit_seconds": 1.0 if method_type == "candidate" else 0.0,
        "predict_seconds": 0.1 if method_type == "candidate" else 0.0,
        "selected_member": selected_member,
        "ensemble_members": "",
        "ensemble_weights": "",
    }


def test_candidate_relative_rank_and_metric_contrast_are_separate() -> None:
    results = pd.DataFrame(
        [
            _row(dataset="d1", method="a", method_type="candidate", primary_metric=0.20),
            _row(dataset="d1", method="b", method_type="candidate", primary_metric=0.30),
            _row(
                dataset="d1",
                method="meta_selector_local_validation",
                method_type="selector",
                primary_metric=0.20,
                selected_member="a",
            ),
            _row(
                dataset="d1",
                method="meta_selector_local_ensemble",
                method_type="selector",
                primary_metric=0.19,
            ),
            _row(dataset="d2", method="a", method_type="candidate", primary_metric=0.40),
            _row(dataset="d2", method="b", method_type="candidate", primary_metric=0.35),
            _row(
                dataset="d2",
                method="meta_selector_local_validation",
                method_type="selector",
                primary_metric=0.40,
                selected_member="a",
            ),
            _row(
                dataset="d2",
                method="meta_selector_local_ensemble",
                method_type="selector",
                primary_metric=0.34,
            ),
        ]
    )

    ranked = third.rank_against_candidates(results)
    dataset_metrics = third.dataset_method_metrics(ranked)
    paired, per_dataset = third.paired_contrasts(
        dataset_metrics,
        method="meta_selector_local_ensemble",
        reference="meta_selector_local_validation",
        n_bootstrap=100,
        seed=1,
    )

    d1_ensemble = dataset_metrics[
        dataset_metrics["dataset"].eq("d1")
        & dataset_metrics["method"].eq("meta_selector_local_ensemble")
    ].iloc[0]
    metric = paired[paired["endpoint"].eq("primary_metric")].iloc[0]
    regret = paired[paired["endpoint"].eq("normalized_rank_regret")].iloc[0]

    assert d1_ensemble["mean_rank"] == 1.0
    assert metric["mean_method_minus_reference"] < 0.0
    assert regret["mean_method_minus_reference"] < 0.0
    assert set(per_dataset["dataset"]) == {"d1", "d2"}
