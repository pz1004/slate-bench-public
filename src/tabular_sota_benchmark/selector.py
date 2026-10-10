from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidatePrediction:
    method: str
    val_primary: float
    fit_seconds: float
    val_prediction: np.ndarray
    test_prediction: np.ndarray


def candidate_utility(candidate: CandidatePrediction, runtime_weight: float) -> float:
    return float(candidate.val_primary + runtime_weight * np.log1p(candidate.fit_seconds))


def select_candidate(
    candidates: list[CandidatePrediction],
    runtime_weight: float,
) -> tuple[CandidatePrediction, float]:
    if not candidates:
        raise ValueError("No candidates available for selection.")
    utilities = [candidate_utility(candidate, runtime_weight) for candidate in candidates]
    best_index = int(np.argmin(utilities))
    return candidates[best_index], float(utilities[best_index])


def ensemble_candidates(
    candidates: list[CandidatePrediction],
    runtime_weight: float,
    max_ensemble_size: int,
    temperature: float,
) -> tuple[list[CandidatePrediction], np.ndarray]:
    if not candidates:
        raise ValueError("No candidates available for ensembling.")
    utilities = np.asarray(
        [candidate_utility(candidate, runtime_weight) for candidate in candidates],
        dtype=float,
    )
    order = np.argsort(utilities)
    selected_indices = order[: max(1, min(max_ensemble_size, len(order)))]
    selected = [candidates[int(index)] for index in selected_indices]
    selected_utilities = utilities[selected_indices]
    utility_span = float(selected_utilities.max() - selected_utilities.min())
    if utility_span <= 1e-12:
        normalized = np.zeros_like(selected_utilities)
    else:
        normalized = (selected_utilities - selected_utilities.min()) / utility_span
    raw_weights = np.exp(-normalized / max(temperature, 1e-6))
    weights = raw_weights / raw_weights.sum()
    return selected, weights


def weighted_prediction(
    selected: list[CandidatePrediction],
    weights: np.ndarray,
    split: str,
) -> np.ndarray:
    if split not in {"val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    predictions = [
        candidate.val_prediction if split == "val" else candidate.test_prediction
        for candidate in selected
    ]
    stacked = np.stack(predictions, axis=0)
    weight_shape = (len(weights),) + (1,) * (stacked.ndim - 1)
    return np.sum(stacked * weights.reshape(weight_shape), axis=0)

