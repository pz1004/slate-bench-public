from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabular_sota_benchmark.config import load_config
from tabular_sota_benchmark.artifacts import run_artifact_experiment
from tabular_sota_benchmark.data import TabularDataset, load_builtin_dataset
from tabular_sota_benchmark.meta_features import compute_meta_features
from tabular_sota_benchmark.metrics import evaluate_predictions, predict_for_task
from tabular_sota_benchmark.models import is_model_available, make_model
from tabular_sota_benchmark.selector import (
    CandidatePrediction,
    ensemble_candidates,
    select_candidate,
    weighted_prediction,
)


def _split_dataset(
    dataset: TabularDataset,
    seed: int,
    test_size: float,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(dataset.X.shape[0])
    stratify = dataset.y if dataset.task == "classification" else None
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_val_y = dataset.y.iloc[train_val_idx]
    val_stratify = train_val_y if dataset.task == "classification" else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_size,
        random_state=seed + 1009,
        stratify=val_stratify,
    )
    return train_idx, val_idx, test_idx


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _result_row(
    *,
    dataset: TabularDataset,
    seed: int,
    method: str,
    method_type: str,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    fit_seconds: float,
    predict_seconds: float,
    selected_member: str = "",
    ensemble_members: str = "",
    ensemble_weights: str = "",
    fold: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset.name,
        "fold": seed if fold is None else fold,
        "task": dataset.task,
        "seed": seed,
        "method": method,
        "method_type": method_type,
        "primary_metric_name": "log_loss" if dataset.task == "classification" else "rmse",
        "primary_metric": test_metrics["primary_metric"],
        "val_primary_metric": val_metrics["primary_metric"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "selected_member": selected_member,
        "ensemble_members": ensemble_members,
        "ensemble_weights": ensemble_weights,
    }
    row.update(_prefixed(val_metrics, "val"))
    row.update(_prefixed(test_metrics, "test"))
    return row


def _fit_candidate(
    *,
    model_name: str,
    dataset: TabularDataset,
    seed: int,
    n_jobs: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, Any], CandidatePrediction]:
    X_train = dataset.X.iloc[train_idx]
    y_train = dataset.y.iloc[train_idx]
    X_val = dataset.X.iloc[val_idx]
    y_val = dataset.y.iloc[val_idx]
    X_test = dataset.X.iloc[test_idx]
    y_test = dataset.y.iloc[test_idx]
    classes = dataset.classes

    model = make_model(model_name, dataset.task, seed, n_jobs=n_jobs)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    val_prediction = predict_for_task(model, X_val, dataset.task, classes)
    test_prediction = predict_for_task(model, X_test, dataset.task, classes)
    predict_seconds = time.perf_counter() - start

    val_metrics = evaluate_predictions(y_val, val_prediction, dataset.task, classes)
    test_metrics = evaluate_predictions(y_test, test_prediction, dataset.task, classes)
    row = _result_row(
        dataset=dataset,
        seed=seed,
        method=model_name,
        method_type="candidate",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )
    candidate = CandidatePrediction(
        method=model_name,
        val_primary=val_metrics["primary_metric"],
        fit_seconds=fit_seconds,
        val_prediction=val_prediction,
        test_prediction=test_prediction,
    )
    return row, candidate


def run_experiment(config_path: str | Path) -> Path:
    config = load_config(config_path)
    if config.get("artifact_source"):
        return run_artifact_experiment(config, config_path)

    experiment = config["experiment"]
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    seeds = [int(seed) for seed in experiment["random_seeds"]]
    baselines = list(config["baselines"])
    n_jobs = int(experiment.get("n_jobs", 1))

    selector_config = config.get("selector", {})
    runtime_weight = float(selector_config.get("runtime_weight", 0.0))
    max_ensemble_size = int(selector_config.get("max_ensemble_size", 3))
    temperature = float(selector_config.get("temperature", 0.2))
    selector_method_name = selector_config.get("selector_method_name", "meta_selector_single")
    ensemble_method_name = selector_config.get("ensemble_method_name", "meta_selector_ensemble")

    for dataset_name in config["datasets"]:
        dataset = load_builtin_dataset(str(dataset_name))
        meta_rows.append(compute_meta_features(dataset))
        for seed in seeds:
            train_idx, val_idx, test_idx = _split_dataset(
                dataset,
                seed=seed,
                test_size=float(experiment["test_size"]),
                validation_size=float(experiment["validation_size"]),
            )
            candidates: list[CandidatePrediction] = []
            for model_name in baselines:
                if not is_model_available(model_name):
                    skipped[model_name] = "optional dependency is not installed"
                    continue
                try:
                    row, candidate = _fit_candidate(
                        model_name=model_name,
                        dataset=dataset,
                        seed=seed,
                        n_jobs=n_jobs,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                    )
                except Exception as exc:
                    skipped[model_name] = f"{type(exc).__name__}: {exc}"
                    continue
                rows.append(row)
                candidates.append(candidate)

            if not candidates:
                raise RuntimeError(f"No candidate models completed for {dataset.name}, seed {seed}.")

            y_val = dataset.y.iloc[val_idx]
            y_test = dataset.y.iloc[test_idx]
            classes = dataset.classes

            selected, utility = select_candidate(candidates, runtime_weight=runtime_weight)
            selected_row = next(
                row
                for row in rows
                if row["dataset"] == dataset.name
                and row["seed"] == seed
                and row["method"] == selected.method
            )
            rows.append(
                _result_row(
                    dataset=dataset,
                    seed=seed,
                    method=selector_method_name,
                    method_type="selector",
                    val_metrics={
                        key.removeprefix("val_"): value
                        for key, value in selected_row.items()
                        if key.startswith("val_")
                    },
                    test_metrics={
                        key.removeprefix("test_"): value
                        for key, value in selected_row.items()
                        if key.startswith("test_")
                    },
                    fit_seconds=0.0,
                    predict_seconds=0.0,
                    selected_member=selected.method,
                    ensemble_members="",
                    ensemble_weights=f"utility={utility:.6g}",
                )
            )

            ensemble_members, weights = ensemble_candidates(
                candidates,
                runtime_weight=runtime_weight,
                max_ensemble_size=max_ensemble_size,
                temperature=temperature,
            )
            val_ensemble_prediction = weighted_prediction(ensemble_members, weights, "val")
            test_ensemble_prediction = weighted_prediction(ensemble_members, weights, "test")
            val_ensemble_metrics = evaluate_predictions(
                y_val,
                val_ensemble_prediction,
                dataset.task,
                classes,
            )
            test_ensemble_metrics = evaluate_predictions(
                y_test,
                test_ensemble_prediction,
                dataset.task,
                classes,
            )
            rows.append(
                _result_row(
                    dataset=dataset,
                    seed=seed,
                    method=ensemble_method_name,
                    method_type="selector",
                    val_metrics=val_ensemble_metrics,
                    test_metrics=test_ensemble_metrics,
                    fit_seconds=0.0,
                    predict_seconds=0.0,
                    selected_member="",
                    ensemble_members=";".join(member.method for member in ensemble_members),
                    ensemble_weights=";".join(f"{weight:.6f}" for weight in weights),
                )
            )

    result_path = output_dir / "results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    pd.DataFrame(meta_rows).to_csv(output_dir / "dataset_meta_features.csv", index=False)
    metadata = {
        "config_path": str(config_path),
        "n_rows": len(rows),
        "skipped_baselines": skipped,
        "generated_at_epoch": time.time(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result_path
