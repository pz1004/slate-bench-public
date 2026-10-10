from __future__ import annotations

import numpy as np

from tabular_sota_benchmark.metrics import evaluate_predictions


def test_classification_metrics_are_finite() -> None:
    y = np.array([0, 1, 1, 0])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
        ]
    )
    metrics = evaluate_predictions(y, proba, "classification", [0, 1])
    assert metrics["primary_metric"] == metrics["log_loss"]
    assert metrics["accuracy"] == 1.0
    assert np.isfinite(metrics["roc_auc"])


def test_regression_metrics_are_finite() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.1, 1.9, 3.2])
    metrics = evaluate_predictions(y, pred, "regression", None)
    assert metrics["primary_metric"] == metrics["rmse"]
    assert metrics["rmse"] > 0.0

