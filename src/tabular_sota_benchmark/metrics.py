from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def primary_metric_name(task: str) -> str:
    if task == "classification":
        return "log_loss"
    if task == "regression":
        return "rmse"
    raise ValueError(f"Unknown task: {task}")


def align_proba(
    proba: np.ndarray,
    model_classes: np.ndarray,
    expected_classes: list[int],
) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(expected_classes)), dtype=float)
    class_to_index = {int(label): idx for idx, label in enumerate(expected_classes)}
    for source_idx, label in enumerate(model_classes):
        target_idx = class_to_index[int(label)]
        aligned[:, target_idx] = proba[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


def predict_for_task(
    model: Any,
    X: Any,
    task: str,
    classes: list[int] | None,
) -> np.ndarray:
    if task == "classification":
        if classes is None:
            raise ValueError("Classification predictions require classes.")
        proba = model.predict_proba(X)
        return align_proba(np.asarray(proba), np.asarray(model.classes_), classes)
    return np.asarray(model.predict(X), dtype=float)


def evaluate_predictions(
    y_true: Any,
    prediction: np.ndarray,
    task: str,
    classes: list[int] | None,
) -> dict[str, float]:
    y_array = np.asarray(y_true)
    if task == "classification":
        if classes is None:
            raise ValueError("Classification metrics require classes.")
        y_pred = np.asarray(classes)[np.argmax(prediction, axis=1)]
        y_index = {int(label): index for index, label in enumerate(classes)}
        y_one_hot = np.zeros((len(y_array), len(classes)), dtype=float)
        for row_index, label in enumerate(y_array):
            class_index = y_index.get(int(label))
            if class_index is not None:
                y_one_hot[row_index, class_index] = 1.0
        confidences = np.max(prediction, axis=1)
        correctness = (y_pred == y_array).astype(float)
        calibration_error = 0.0
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            if bin_index == 9:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)
            if not np.any(in_bin):
                continue
            calibration_error += float(np.mean(in_bin)) * abs(
                float(np.mean(correctness[in_bin])) - float(np.mean(confidences[in_bin]))
            )
        metrics = {
            "log_loss": float(log_loss(y_array, prediction, labels=classes)),
            "accuracy": float(accuracy_score(y_array, y_pred)),
            "brier_multiclass": float(np.mean(np.sum((prediction - y_one_hot) ** 2, axis=1))),
            "ece_10": float(calibration_error),
        }
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_array, prediction[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_array,
                        prediction,
                        multi_class="ovr",
                        average="weighted",
                        labels=classes,
                    )
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")
        metrics["primary_metric"] = metrics["log_loss"]
        return metrics

    rmse = float(np.sqrt(mean_squared_error(y_array, prediction)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_array, prediction)),
        "r2": float(r2_score(y_array, prediction)),
        "primary_metric": rmse,
    }
