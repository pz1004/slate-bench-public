from __future__ import annotations

import json

import pandas as pd

from tabular_sota_benchmark.artifacts import (
    ARTIFACT_OUTPUT_COLUMNS,
    build_split_score_artifact_selector_rows,
    build_tabrepo_result_rows,
    build_tabarena_result_rows,
    load_tabrepo_results,
    load_dataset_descriptors,
    tabarena_to_result_rows,
)


class FakeTabRepo:
    def __init__(self) -> None:
        self.ensemble_calls: list[dict[str, object]] = []

    def datasets(self) -> list[str]:
        return ["d1", "d2"]

    def configs(self) -> list[str]:
        return ["A", "B", "C"]

    def dataset_info(self, dataset: str) -> dict[str, str]:
        return {"problem_type": "binary", "metric": "log_loss"}

    def metrics(self, datasets=None, folds=None, configs=None) -> pd.DataFrame:
        datasets = datasets or self.datasets()
        folds = folds or [0]
        configs = configs or self.configs()
        rows = []
        for dataset in datasets:
            for fold in folds:
                for config in configs:
                    val = {"A": 0.10, "B": 0.20, "C": 0.30}[config]
                    test = {"A": 0.30, "B": 0.10, "C": 0.20}[config]
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "framework": config,
                            "metric_error": test,
                            "metric_error_val": val,
                            "time_train_s": {"A": 10.0, "B": 20.0, "C": 30.0}[config],
                            "time_infer_s": 1.0,
                            "rank": {"A": 3, "B": 1, "C": 2}[config],
                        }
                    )
        return pd.DataFrame(rows).set_index(["dataset", "fold", "framework"])

    def evaluate_ensemble(
        self,
        *,
        datasets,
        configs,
        ensemble_size,
        rank,
        folds,
        backend,
    ):
        self.ensemble_calls.append(
            {
                "datasets": list(datasets),
                "configs": list(configs),
                "ensemble_size": ensemble_size,
                "rank": rank,
                "folds": list(folds),
                "backend": backend,
            }
        )
        index = pd.MultiIndex.from_product([datasets, folds], names=["dataset", "fold"])
        errors = pd.Series([0.05 for _ in index], index=index, name="error")
        weights = pd.DataFrame(
            [[1.0 / len(configs) for _ in configs] for _ in index],
            index=index,
            columns=configs,
        )
        return errors, weights


def test_tabrepo_artifact_adapter_maps_metrics_to_canonical_rows() -> None:
    repo = FakeTabRepo()
    frame = load_tabrepo_results({"kind": "tabrepo_repository"}, repository=repo)

    assert set(frame.columns) == {
        "dataset",
        "fold",
        "method",
        "metric_error",
        "time_train_s",
        "time_infer_s",
        "metric_error_val",
        "seed",
        "problem_type",
        "metric",
    }
    assert frame.shape[0] == 6
    assert set(frame["method"]) == {"A", "B", "C"}
    assert set(frame["problem_type"]) == {"binary"}


def _canonical_candidate_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["fold"] = frame.get("fold", 0)
    frame["seed"] = frame.get("seed", 0)
    frame["task"] = frame.get("task", "binary")
    frame["method_type"] = "candidate"
    frame["primary_metric_name"] = "synthetic_error"
    frame["val_primary_metric"] = frame["metric_error_val"].astype(float)
    frame["primary_metric"] = frame["metric_error"].astype(float)
    frame["fit_seconds"] = frame["time_train_s"].astype(float)
    frame["predict_seconds"] = frame["time_infer_s"].astype(float)
    frame["selected_member"] = ""
    frame["ensemble_members"] = ""
    frame["ensemble_weights"] = ""
    frame["source"] = "synthetic"
    frame["problem_type"] = "binary"
    frame["metric"] = "synthetic_error"
    return frame[ARTIFACT_OUTPUT_COLUMNS]


def test_split_selector_uses_history_priors_not_heldout_test_labels() -> None:
    rows = []
    for dataset in ["d1", "d2"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "method": "A",
                    "metric_error": 0.90,
                    "metric_error_val": 0.50,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                },
                {
                    "dataset": dataset,
                    "method": "B",
                    "metric_error": 0.10,
                    "metric_error_val": 0.50,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                },
            ]
        )
    history = _canonical_candidate_rows(rows)
    heldout = _canonical_candidate_rows(
        [
            {
                "dataset": "d3",
                "method": "A",
                "metric_error": 0.10,
                "metric_error_val": 0.50,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
            },
            {
                "dataset": "d3",
                "method": "B",
                "metric_error": 0.90,
                "metric_error_val": 0.50,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
            },
        ]
    )

    selectors, _ = build_split_score_artifact_selector_rows(
        history,
        heldout,
        runtime_weight=0.0,
        selector_method_name="raw",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "history_prior",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 0.0,
                        "task_method_prior_weight": 1.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            }
        ],
    )

    prior = selectors[selectors["method"].eq("history_prior")].iloc[0]
    assert prior["dataset"] == "d3"
    assert prior["selected_member"] == "B"


def test_split_selector_rejects_precomputed_selector_results() -> None:
    rows = _canonical_candidate_rows(
        [
            {
                "dataset": "d1",
                "method": "A",
                "metric_error": 0.1,
                "metric_error_val": 0.1,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
            },
            {
                "dataset": "d2",
                "method": "A",
                "metric_error": 0.1,
                "metric_error_val": 0.1,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
            },
        ]
    )
    try:
        build_split_score_artifact_selector_rows(
            rows[rows["dataset"].eq("d1")],
            rows[rows["dataset"].eq("d2")],
            runtime_weight=0.0,
            selector_method_name="raw",
            derived_selectors=[
                {
                    "kind": "include_selector_results",
                    "selector_results_path": "results/unsafe.csv",
                }
            ],
        )
    except ValueError as exc:
        assert "precomputed" in str(exc)
    else:
        raise AssertionError("Expected split-aware builder to reject selector_results_path.")


def test_tabrepo_prediction_ensemble_uses_validation_top_k_only() -> None:
    repo = FakeTabRepo()
    frame = load_tabrepo_results({"kind": "tabrepo_repository"}, repository=repo)
    result, _ = build_tabrepo_result_rows(
        frame,
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "tabrepo_prediction_ensemble",
                "method_name": "meta_selector_tabrepo_ensemble_k5",
                "top_k": 2,
                "ensemble_size": 5,
                "runtime_weight": 0.0,
                "backend": "native",
            }
        ],
        ensemble_repository=repo,
    )

    ensemble = result[result["method"].eq("meta_selector_tabrepo_ensemble_k5")]
    assert ensemble.shape[0] == 2
    assert set(ensemble["primary_metric"]) == {0.05}
    assert all(call["configs"] == ["A", "B"] for call in repo.ensemble_calls)
    assert all(call["rank"] is False for call in repo.ensemble_calls)


def test_tabarena_artifact_conversion_adds_selector() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "fold": 0,
                "method": "A",
                "metric_error": 0.30,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.20,
                "seed": 10,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
            {
                "dataset": "d1",
                "fold": 0,
                "method": "B",
                "metric_error": 0.10,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.40,
                "seed": 10,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
        ]
    )
    result = tabarena_to_result_rows(
        frame,
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
    )
    selector = result[result["method_type"] == "selector"].iloc[0]
    assert result.shape[0] == 3
    assert selector["selected_member"] == "A"
    assert selector["primary_metric"] == 0.30


def test_include_selector_results_does_not_require_method_name(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "fold": 0,
                "method": "A",
                "metric_error": 0.30,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.20,
                "seed": 10,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
            {
                "dataset": "d1",
                "fold": 0,
                "method": "B",
                "metric_error": 0.10,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.40,
                "seed": 10,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
        ]
    )
    source_result, _ = build_tabarena_result_rows(
        frame,
        runtime_weight=0.0,
        selector_method_name="meta_selector_source",
    )
    source_path = tmp_path / "source_results.csv"
    source_result.to_csv(source_path, index=False)

    result, _ = build_tabarena_result_rows(
        frame,
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "include_selector_results",
                "selector_results_path": str(source_path),
                "selector_methods": ["meta_selector_source"],
            },
        ],
    )

    selectors = result[result["method_type"].eq("selector")]
    assert set(selectors["method"]) == {"meta_selector_single", "meta_selector_source"}


def test_task_conditioned_selector_uses_validation_ranks() -> None:
    rows = []
    for dataset in ["d1", "d2"]:
        for fold in [0, 1]:
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": "accurate_slow",
                        "metric_error": 0.10,
                        "time_train_s": 100.0,
                        "time_infer_s": 1.0,
                        "metric_error_val": 0.10,
                        "seed": fold,
                        "problem_type": "binary",
                        "metric": "roc_auc",
                    },
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": "fast_bad",
                        "metric_error": 0.30,
                        "time_train_s": 1.0,
                        "time_infer_s": 0.1,
                        "metric_error_val": 0.30,
                        "seed": fold,
                        "problem_type": "binary",
                        "metric": "roc_auc",
                    },
                ]
            )
    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {"kind": "validation_rank", "method_name": "meta_selector_val_rank"},
            {
                "kind": "task_conditioned",
                "method_name": "meta_selector_task_conditioned",
                "runtime_weight_grid": [0.0, 0.1],
                "tuning_runtime_weight": 0.01,
                "leave_one_dataset_out": True,
            },
        ],
    )
    selectors = result[result["method_type"] == "selector"]
    assert set(selectors["method"]) == {
        "meta_selector_single",
        "meta_selector_val_rank",
        "meta_selector_task_conditioned",
    }
    conditioned = selectors[selectors["method"] == "meta_selector_task_conditioned"]
    assert set(conditioned["selected_member"]) == {"accurate_slow"}
    assert set(tuning["selector_method"]) == {"meta_selector_task_conditioned"}


def test_calibrated_selector_uses_non_target_dataset_priors() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "validation_winner (default)",
                    "metric_error": 0.40,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.10,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "prior_winner (default)",
                    "metric_error": 0.10,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "calibrated",
                "method_name": "meta_selector_calibrated",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    },
                    {
                        "validation_weight": 0.0,
                        "task_method_prior_weight": 1.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    },
                ],
            },
        ],
    )

    calibrated = result[result["method"] == "meta_selector_calibrated"]
    assert calibrated.shape[0] == 3
    assert set(calibrated["selected_member"]) == {"prior_winner (default)"}
    calibration_rows = tuning[tuning["selector_method"] == "meta_selector_calibrated"]
    assert set(calibration_rows["n_training_datasets"]) == {2}
    assert set(calibration_rows["leave_one_dataset_out"]) == {True}
    assert set(calibration_rows["task_method_prior_weight"]) == {1.0}


def test_budgeted_calibrated_selector_excludes_over_budget_candidates() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_strong (default)",
                    "metric_error": 0.05,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_allowed (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_selector_calibrated_budget_10s",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
        ],
    )

    budgeted = result[result["method"] == "meta_selector_calibrated_budget_10s"]
    assert budgeted.shape[0] == 3
    assert set(budgeted["selected_member"]) == {"fast_allowed (default)"}
    calibration_rows = tuning[
        tuning["selector_method"] == "meta_selector_calibrated_budget_10s"
    ]
    assert set(calibration_rows["budget_feature"]) == {"fit_seconds"}
    assert set(calibration_rows["max_budget_seconds"]) == {10.0}
    assert set(calibration_rows["n_target_budget_fallback_units"]) == {0}


def test_calibrated_selector_can_filter_method_families() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "NN (default)",
                    "metric_error": 0.05,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "GBM (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "calibrated",
                "method_name": "meta_selector_tree_only",
                "leave_one_dataset_out": True,
                "method_family_allowlist": ["GBM"],
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
        ],
    )

    tree_only = result[result["method"] == "meta_selector_tree_only"]
    assert tree_only.shape[0] == 3
    assert set(tree_only["selected_member"]) == {"GBM (default)"}
    calibration_rows = tuning[tuning["selector_method"] == "meta_selector_tree_only"]
    assert set(calibration_rows["n_unfiltered_candidates"]) == {6}
    assert set(calibration_rows["n_filtered_candidates"]) == {3}
    assert calibration_rows["candidate_filter"].str.contains("method_family_allowlist=GBM").all()


def test_validation_confidence_prior_selects_near_validation_prior_winner() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "validation_best_bad (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "near_validation_good (default)",
                    "metric_error": 0.05,
                    "time_train_s": 2.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.06,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "validation_far_good (default)",
                    "metric_error": 0.01,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.50,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "validation_confidence_prior",
                "method_name": "meta_confidence_prior",
                "leave_one_dataset_out": True,
                "validation_rank_margin": 1.0,
                "weight_grid": [
                    {
                        "validation_weight": 0.0,
                        "task_method_prior_weight": 1.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
        ],
    )

    selector = result[result["method"] == "meta_confidence_prior"]
    assert selector.shape[0] == 3
    assert set(selector["selected_member"]) == {"near_validation_good (default)"}
    assert selector["ensemble_weights"].str.contains("validation_rank_margin=1").all()
    assert not selector["selected_member"].str.contains("validation_far_good").any()

    confidence_tuning = tuning[tuning["selector_method"] == "meta_confidence_prior"]
    assert set(confidence_tuning["n_training_datasets"]) == {2}
    assert set(confidence_tuning["confidence_set_mean_size"]) == {2.0}


def test_task_adaptive_selector_uses_non_target_selector_performance() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": 0.05,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_bad (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_unrestricted",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_budget_10s",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "task_adaptive_selector",
                "method_name": "meta_task_adaptive",
                "candidate_selector_methods": ["meta_budget_10s", "meta_unrestricted"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
            },
        ],
    )

    adaptive = result[result["method"] == "meta_task_adaptive"]
    assert adaptive.shape[0] == 3
    assert set(adaptive["selected_member"]) == {"slow_good (default)"}
    assert adaptive["ensemble_weights"].str.contains("adaptive_choice=meta_unrestricted").all()

    adaptive_tuning = tuning[tuning["selector_method"] == "meta_task_adaptive"]
    assert set(adaptive_tuning["selected_selector_method"]) == {"meta_unrestricted"}
    assert set(adaptive_tuning["n_training_units"]) == {2}
    assert set(adaptive_tuning["leave_one_dataset_out"]) == {True}


def test_risk_constrained_selector_uses_dual_reference_gate() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": 0.10,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_good (default)",
                    "metric_error": 0.10,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_task_ref",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_family_ref",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "risk_constrained_selector",
                "method_name": "meta_risk_guard",
                "candidate_selector_methods": ["meta_family_ref"],
                "task_reference_method": "meta_task_ref",
                "family_reference_method": "meta_family_ref",
                "fallback_selector_methods": ["meta_task_ref", "meta_family_ref"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
                "noninferiority_margin_rank": 0.5,
                "min_training_datasets": 2,
                "require_runtime_gain": True,
            },
        ],
    )

    risk = result[result["method"] == "meta_risk_guard"]
    assert risk.shape[0] == 3
    assert set(risk["selected_member"]) == {"fast_good (default)"}
    assert risk["ensemble_weights"].str.contains("risk_gate_passed=1").all()
    assert risk["ensemble_weights"].str.contains("risk_choice=meta_family_ref").all()

    risk_tuning = tuning[tuning["selector_method"] == "meta_risk_guard"]
    assert set(risk_tuning["selected_selector_method"]) == {"meta_family_ref"}
    assert set(risk_tuning["risk_gate_passed"]) == {True}
    assert set(risk_tuning["n_training_datasets"]) == {2}


def test_risk_constrained_selector_falls_back_when_gate_fails() -> None:
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": 0.05,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_bad (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_task_ref",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_family_ref",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "risk_constrained_selector",
                "method_name": "meta_risk_guard",
                "candidate_selector_methods": ["meta_family_ref"],
                "task_reference_method": "meta_task_ref",
                "family_reference_method": "meta_family_ref",
                "fallback_selector_methods": ["meta_task_ref", "meta_family_ref"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
                "noninferiority_margin_rank": 0.5,
                "min_training_datasets": 2,
                "require_runtime_gain": True,
            },
        ],
    )

    risk = result[result["method"] == "meta_risk_guard"]
    assert risk.shape[0] == 3
    assert set(risk["selected_member"]) == {"slow_good (default)"}
    assert risk["ensemble_weights"].str.contains("risk_gate_passed=0").all()
    assert risk["ensemble_weights"].str.contains(
        "risk_fallback_reason=no_candidate_passed_risk_gate"
    ).all()

    risk_tuning = tuning[tuning["selector_method"] == "meta_risk_guard"]
    assert set(risk_tuning["selected_selector_method"]) == {"meta_task_ref"}
    assert set(risk_tuning["risk_gate_passed"]) == {False}


def test_meta_feature_adaptive_selector_uses_nearest_non_target_dataset() -> None:
    rows = []
    for dataset in ["d1", "d2"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": 0.05,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_bad (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )
    rows.extend(
        [
            {
                "dataset": "d3",
                "fold": 0,
                "method": "slow_good (default)",
                "metric_error": 0.30,
                "time_train_s": 100.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.10,
                "seed": 0,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
            {
                "dataset": "d3",
                "fold": 0,
                "method": "fast_bad (default)",
                "metric_error": 0.05,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.11,
                "seed": 0,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
        ]
    )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_unrestricted",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_budget_10s",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "meta_feature_adaptive_selector",
                "method_name": "meta_feature_adaptive",
                "candidate_selector_methods": ["meta_budget_10s", "meta_unrestricted"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
                "feature_columns": ["val_spread_mean"],
                "top_k": 1,
            },
        ],
    )

    adaptive_d1 = result[
        (result["method"] == "meta_feature_adaptive") & (result["dataset"] == "d1")
    ].iloc[0]
    assert adaptive_d1["selected_member"] == "slow_good (default)"
    assert "adaptive_choice=meta_unrestricted" in adaptive_d1["ensemble_weights"]
    assert "nearest_datasets=d2" in adaptive_d1["ensemble_weights"]

    tuning_d1 = tuning[
        (tuning["selector_method"] == "meta_feature_adaptive")
        & (tuning["dataset"] == "d1")
    ].iloc[0]
    assert tuning_d1["selected_selector_method"] == "meta_unrestricted"
    assert int(tuning_d1["n_neighbor_datasets"]) == 1
    assert tuning_d1["meta_feature_columns"] == "val_spread_mean"


def test_meta_feature_adaptive_selector_can_use_dataset_descriptors() -> None:
    rows = []
    for dataset in ["d1", "d2"]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": 0.05,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_bad (default)",
                    "metric_error": 0.30,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )
    rows.extend(
        [
            {
                "dataset": "d3",
                "fold": 0,
                "method": "slow_good (default)",
                "metric_error": 0.30,
                "time_train_s": 100.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.05,
                "seed": 0,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
            {
                "dataset": "d3",
                "fold": 0,
                "method": "fast_bad (default)",
                "metric_error": 0.05,
                "time_train_s": 1.0,
                "time_infer_s": 0.1,
                "metric_error_val": 0.30,
                "seed": 0,
                "problem_type": "binary",
                "metric": "roc_auc",
            },
        ]
    )
    descriptors = pd.DataFrame(
        [
            {"dataset": "d1", "descriptor_log_num_instances": 4.6},
            {"dataset": "d2", "descriptor_log_num_instances": 9.2},
            {"dataset": "d3", "descriptor_log_num_instances": 4.7},
        ]
    )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_unrestricted",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_budget_10s",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "meta_feature_adaptive_selector",
                "method_name": "meta_feature_descriptor_adaptive",
                "candidate_selector_methods": ["meta_budget_10s", "meta_unrestricted"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
                "feature_column_prefixes": ["descriptor_"],
                "top_k": 1,
            },
        ],
        dataset_descriptors=descriptors,
    )

    adaptive_d1 = result[
        (result["method"] == "meta_feature_descriptor_adaptive")
        & (result["dataset"] == "d1")
    ].iloc[0]
    assert adaptive_d1["selected_member"] == "fast_bad (default)"
    assert "adaptive_choice=meta_budget_10s" in adaptive_d1["ensemble_weights"]
    assert "nearest_datasets=d3" in adaptive_d1["ensemble_weights"]

    tuning_d1 = tuning[
        (tuning["selector_method"] == "meta_feature_descriptor_adaptive")
        & (tuning["dataset"] == "d1")
    ].iloc[0]
    assert set(tuning_d1["meta_feature_columns"].split(",")) == {
        "descriptor_log_num_instances",
        "descriptor_missing",
    }


def test_meta_feature_adaptive_selector_scales_feature_groups() -> None:
    rows = []
    for dataset, slow_error, fast_error in [
        ("d1", 0.05, 0.30),
        ("d2", 0.05, 0.30),
        ("d3", 0.30, 0.05),
    ]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "slow_good (default)",
                    "metric_error": slow_error,
                    "time_train_s": 100.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.05,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
                {
                    "dataset": dataset,
                    "fold": 0,
                    "method": "fast_bad (default)",
                    "metric_error": fast_error,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                    "metric_error_val": 0.30,
                    "seed": 0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                },
            ]
        )
    descriptors = pd.DataFrame(
        [
            {
                "dataset": "d1",
                "descriptor_log_num_instances": 0.0,
                "descriptor_quality_a": 0.0,
                "descriptor_quality_b": 0.0,
                "descriptor_quality_c": 0.0,
                "descriptor_quality_d": 0.0,
            },
            {
                "dataset": "d2",
                "descriptor_log_num_instances": 12.0,
                "descriptor_quality_a": 0.0,
                "descriptor_quality_b": 0.0,
                "descriptor_quality_c": 0.0,
                "descriptor_quality_d": 0.0,
            },
            {
                "dataset": "d3",
                "descriptor_log_num_instances": 0.1,
                "descriptor_quality_a": 2.0,
                "descriptor_quality_b": 2.0,
                "descriptor_quality_c": 2.0,
                "descriptor_quality_d": 2.0,
            },
        ]
    )

    result, tuning = build_tabarena_result_rows(
        pd.DataFrame(rows),
        runtime_weight=0.0,
        selector_method_name="meta_selector_single",
        derived_selectors=[
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_unrestricted",
                "leave_one_dataset_out": True,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "budgeted_calibrated",
                "method_name": "meta_budget_10s",
                "leave_one_dataset_out": True,
                "budget_feature": "fit_seconds",
                "max_budget_seconds": 10.0,
                "weight_grid": [
                    {
                        "validation_weight": 1.0,
                        "task_method_prior_weight": 0.0,
                        "task_family_prior_weight": 0.0,
                        "global_method_prior_weight": 0.0,
                        "runtime_weight": 0.0,
                    }
                ],
            },
            {
                "kind": "meta_feature_adaptive_selector",
                "method_name": "meta_feature_group_scaled",
                "candidate_selector_methods": ["meta_budget_10s", "meta_unrestricted"],
                "choice_scope": "task",
                "leave_one_dataset_out": True,
                "feature_column_prefixes": ["descriptor_"],
                "feature_groups": [
                    {
                        "name": "size",
                        "columns": ["descriptor_log_num_instances"],
                    },
                    {
                        "name": "quality",
                        "column_prefixes": ["descriptor_quality_"],
                    },
                ],
                "top_k": 1,
            },
        ],
        dataset_descriptors=descriptors,
    )

    adaptive_d1 = result[
        (result["method"] == "meta_feature_group_scaled") & (result["dataset"] == "d1")
    ].iloc[0]
    assert adaptive_d1["selected_member"] == "fast_bad (default)"
    assert "adaptive_choice=meta_budget_10s" in adaptive_d1["ensemble_weights"]
    assert "nearest_datasets=d3" in adaptive_d1["ensemble_weights"]

    tuning_d1 = tuning[
        (tuning["selector_method"] == "meta_feature_group_scaled")
        & (tuning["dataset"] == "d1")
    ].iloc[0]
    assert "size:n=1:w=1/mean" in tuning_d1["meta_feature_groups"]
    assert "quality:n=4:w=1/mean" in tuning_d1["meta_feature_groups"]


def test_dataset_descriptor_loader_adds_curation_yaml_features(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    curation_dir = tmp_path / "curation"
    dataset_dir = curation_dir / "d1"
    dataset_dir.mkdir(parents=True)
    metadata_path.write_text(
        "\n".join(
            [
                "dataset_name,num_features,num_instances,num_classes,percentage_cat_features,year,can_run_tabpfnv2,can_run_tabicl,data_source,domain,problem_type",
                "d1,6,100,2,40,2024,True,False,UCI,medical,binary",
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "d1.yaml").write_text(
        "\n".join(
            [
                'problem_type: "classification"',
                'target_feature: "label"',
                "categorical_features:",
                '  - "cat_a"',
                '  - "cat_b"',
                "curation_comments: |",
                "  - Converted null strings to missing values.",
            ]
        ),
        encoding="utf-8",
    )

    descriptors = load_dataset_descriptors(
        {
            "local_path": str(metadata_path),
            "download": False,
            "include_curation_yaml": True,
            "curation_local_dir": str(curation_dir),
        }
    )
    assert descriptors is not None
    row = descriptors.iloc[0]
    assert row["descriptor_curation_feature_count"] == 5.0
    assert row["descriptor_curation_categorical_feature_count"] == 2.0
    assert row["descriptor_curation_numeric_feature_count"] == 3.0
    assert row["descriptor_target_cardinality"] == 2.0
    assert row["descriptor_curation_mentions_missing"] == 1.0


def test_dataset_descriptor_loader_adds_quality_mirror_features(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    raw_cache_dir = tmp_path / "raw"
    raw_dataset_dir = raw_cache_dir / "data" / "toy"
    raw_dataset_dir.mkdir(parents=True)
    datapackage_path = tmp_path / "datapackage.json"
    metadata_path.write_text(
        "\n".join(
            [
                "dataset_name,openml_dataset_name,num_features,num_instances,num_classes,percentage_cat_features,year,can_run_tabpfnv2,can_run_tabicl,data_source,domain,problem_type,target_feature",
                "toy,toy,3,4,2,0,2024,True,False,OpenML,test,binary,y",
            ]
        ),
        encoding="utf-8",
    )
    datapackage_path.write_text(
        json.dumps({"resources": [{"path": "data/toy/toy.csv"}]}),
        encoding="utf-8",
    )
    (raw_dataset_dir / "toy.csv").write_text(
        "\n".join(
            [
                "x1,x2,y",
                "1,,yes",
                "2,?,no",
                ",3,yes",
                "4,4,yes",
            ]
        ),
        encoding="utf-8",
    )

    descriptors = load_dataset_descriptors(
        {
            "local_path": str(metadata_path),
            "download": False,
            "quality_mirror": {
                "datapackage_local_path": str(datapackage_path),
                "raw_cache_dir": str(raw_cache_dir),
                "download": False,
            },
        }
    )

    assert descriptors is not None
    row = descriptors.iloc[0]
    assert row["descriptor_quality_raw_coverage"] == 1.0
    assert row["descriptor_quality_label_coverage"] == 1.0
    assert row["descriptor_quality_n_instances"] == 4.0
    assert row["descriptor_quality_n_features"] == 2.0
    assert row["descriptor_quality_missing_value_fraction"] == 0.375
    assert row["descriptor_quality_instances_with_missing_fraction"] == 0.75
    assert row["descriptor_quality_features_with_missing_fraction"] == 1.0
    assert row["descriptor_quality_majority_class_fraction"] == 0.75
    assert row["descriptor_quality_minority_class_fraction"] == 0.25
    assert row["descriptor_quality_label_imbalance_ratio"] == 3.0


def test_dataset_descriptor_loader_adds_openml_parquet_quality_features(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    parquet_cache_dir = tmp_path / "parquet"
    parquet_dataset_dir = parquet_cache_dir / "0000" / "1"
    parquet_dataset_dir.mkdir(parents=True)
    metadata_path.write_text(
        "\n".join(
            [
                "dataset_name,dataset_id,openml_dataset_name,num_features,num_instances,num_classes,percentage_cat_features,year,can_run_tabpfnv2,can_run_tabicl,data_source,domain,problem_type,target_feature",
                "toy,1,toy,3,4,2,0,2024,True,False,OpenML,test,binary,y",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "x1": [1.0, 2.0, None, 4.0],
            "x2": [1.0, None, 3.0, 4.0],
            "y": ["yes", "no", "yes", "yes"],
        }
    ).to_parquet(parquet_dataset_dir / "dataset_1.pq", index=False)

    descriptors = load_dataset_descriptors(
        {
            "local_path": str(metadata_path),
            "download": False,
            "quality_mirror": {
                "kind": "openml_parquet",
                "parquet_cache_dir": str(parquet_cache_dir),
                "download": False,
            },
        }
    )

    assert descriptors is not None
    row = descriptors.iloc[0]
    assert row["descriptor_quality_raw_coverage"] == 1.0
    assert row["descriptor_quality_label_coverage"] == 1.0
    assert row["descriptor_quality_missing_value_fraction"] == 0.25
    assert row["descriptor_quality_instances_with_missing_fraction"] == 0.5
    assert row["descriptor_quality_features_with_missing_fraction"] == 1.0
    assert row["descriptor_quality_majority_class_fraction"] == 0.75
    assert row["descriptor_quality_label_imbalance_ratio"] == 3.0
