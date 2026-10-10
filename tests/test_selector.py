from __future__ import annotations

import numpy as np

from tabular_sota_benchmark.selector import (
    CandidatePrediction,
    ensemble_candidates,
    select_candidate,
    weighted_prediction,
)


def _candidate(name: str, val_primary: float, fit_seconds: float) -> CandidatePrediction:
    pred = np.array([val_primary, val_primary + 1.0])
    return CandidatePrediction(
        method=name,
        val_primary=val_primary,
        fit_seconds=fit_seconds,
        val_prediction=pred,
        test_prediction=pred,
    )


def test_select_candidate_uses_validation_utility() -> None:
    candidates = [_candidate("slow_good", 0.1, 100.0), _candidate("fast_ok", 0.12, 0.01)]
    selected, _ = select_candidate(candidates, runtime_weight=0.03)
    assert selected.method == "fast_ok"


def test_weighted_prediction_shape() -> None:
    candidates = [_candidate("a", 0.1, 1.0), _candidate("b", 0.2, 1.0)]
    selected, weights = ensemble_candidates(candidates, 0.0, max_ensemble_size=2, temperature=0.2)
    pred = weighted_prediction(selected, weights, "test")
    assert pred.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)

