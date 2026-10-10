from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_level_policy_benchmark.py"
SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
nested = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nested)


def test_score_level_tabrepo_nested_config_expands_policy_grid() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "score_level_tabrepo_nested.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    artifact_config = config["artifacts"][0]
    derived, policy_methods, _ = nested._selector_grid(
        artifact_config["selector"],
        artifact_config["artifact_source"]["kind"],
    )

    assert config["output_dir"] == "analysis-output/v2/score_level_tabrepo_nested"
    assert artifact_config["artifact_source"]["kind"] == "tabrepo_repository"
    assert "meta_selector_tabrepo_ensemble_k3" in config["references"]
    assert "meta_selector_tabrepo_ensemble_k5" in policy_methods
    assert any(selector["kind"] == "tabrepo_prediction_ensemble" for selector in derived)


def test_score_level_tabrepo_score_only_config_disables_ensembles() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "score_level_tabrepo_score_only.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    artifact_config = config["artifacts"][0]
    derived, policy_methods, _ = nested._selector_grid(
        artifact_config["selector"],
        artifact_config["artifact_source"]["kind"],
    )

    assert config["output_dir"] == "analysis-output/v2/score_level_tabrepo_score_only_direct"
    # Without ensemble rows the prediction vectors are not needed.
    assert artifact_config["artifact_source"]["load_predictions"] is False
    assert artifact_config["selector"]["policy_grid"]["tabrepo_ensemble_sizes"] == []
    assert all("tabrepo_ensemble" not in method for method in policy_methods)
    assert not any(selector["kind"] == "tabrepo_prediction_ensemble" for selector in derived)


def test_choose_policy_uses_regret_then_runtime_then_method_order() -> None:
    scores = pd.DataFrame(
        [
            {
                "method": "slow_first",
                "mean_normalized_rank_regret": 0.10,
                "mean_total_seconds": 10.0,
            },
            {
                "method": "fast_second",
                "mean_normalized_rank_regret": 0.10,
                "mean_total_seconds": 2.0,
            },
            {
                "method": "later_same",
                "mean_normalized_rank_regret": 0.10,
                "mean_total_seconds": 2.0,
            },
        ]
    )

    selected = nested._choose_policy(
        scores,
        ["slow_first", "fast_second", "later_same"],
    )

    assert selected["method"] == "fast_second"


def test_guard_method_names_match_manuscript_margin_codes() -> None:
    assert nested._guard_method_name(0.02, 8, 0.03) == "meta_selector_meta_feature_guard_002"
    assert nested._guard_method_name(0.05, 8, 0.03) == "meta_selector_meta_feature_guard_005"
    assert nested._guard_method_name(0.10, 8, 0.03) == "meta_selector_meta_feature_guard_010"
    assert nested._guard_method_name(0.20, 8, 0.03) == "meta_selector_meta_feature_guard_020"


def test_nested_rank_normalized_regret_uses_unit_candidate_count() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "small",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "a",
                "method_type": "candidate",
                "primary_metric": 0.1,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "small",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "b",
                "method_type": "candidate",
                "primary_metric": 0.2,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "small",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "selector",
                "method_type": "selector",
                "primary_metric": 0.2,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "large",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "a",
                "method_type": "candidate",
                "primary_metric": 0.1,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "large",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "b",
                "method_type": "candidate",
                "primary_metric": 0.2,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "large",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "c",
                "method_type": "candidate",
                "primary_metric": 0.3,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
            {
                "dataset": "large",
                "fold": 0,
                "seed": 0,
                "task": "binary",
                "primary_metric_name": "metric",
                "method": "selector",
                "method_type": "selector",
                "primary_metric": 0.2,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            },
        ]
    )

    ranked = nested._rank_against_candidates(frame)
    small_selector = ranked[
        ranked["dataset"].eq("small") & ranked["method"].eq("selector")
    ].iloc[0]
    large_selector = ranked[
        ranked["dataset"].eq("large") & ranked["method"].eq("selector")
    ].iloc[0]

    assert small_selector["rank_regret"] == 1.0
    assert small_selector["normalized_rank_regret"] == 1.0
    assert large_selector["rank_regret"] == 1.0
    assert large_selector["normalized_rank_regret"] == 0.5


def test_split_part_resume_bookkeeping_does_not_double_count(tmp_path: Path) -> None:
    split_result = pd.DataFrame(
        [
            {
                "artifact": "TabArena",
                "split_id": 0,
                "selected_policy": "raw",
            }
        ]
    )
    train_scores = pd.DataFrame(
        [
            {
                "artifact": "TabArena",
                "split_id": 0,
                "method": "raw",
                "mean_normalized_rank_regret": 0.0,
            }
        ]
    )
    heldout_scores = train_scores.copy()
    unit_scores = pd.DataFrame([{"artifact": "TabArena", "split_id": 0, "method": "raw"}])

    for _ in range(2):
        nested._write_split_parts(
            tmp_path,
            artifact="TabArena",
            split_id=0,
            split_result=split_result,
            train_scores=train_scores,
            heldout_scores=heldout_scores,
            unit_scores=unit_scores,
        )
    # A split whose unit part is missing is not complete.
    nested._write_split_parts(
        tmp_path,
        artifact="TabArena",
        split_id=1,
        split_result=split_result.assign(split_id=1),
        train_scores=train_scores,
        heldout_scores=heldout_scores,
        unit_scores=unit_scores,
    )
    nested._part_path(tmp_path, "unit_scores", "TabArena", 1).unlink()

    assert nested._completed_parts(tmp_path, "TabArena", [0, 1]) == {0}
    loaded = nested._load_parts(tmp_path, "split_results")
    assert loaded.shape[0] == 2
    assert nested._load_parts(tmp_path, "unit_scores").shape[0] == 1


def _synthetic_artifact(*, inner_source_fit: str, seed: int = 0) -> dict:
    """A small TabArena-like artifact: 2 tasks x 5 datasets x 2 folds, 4 candidates.

    The weight grids include prior-heavy tuples, so that one dataset's test ranks move the
    source rows of the others enough to change an adaptive choice.
    """
    import numpy as np

    from tabular_sota_benchmark.artifacts import ARTIFACT_OUTPUT_COLUMNS, build_static_selector_rows

    rng = np.random.default_rng(seed)
    methods = [f"{family} ({variant})" for family in ["A", "B"] for variant in ["default", "tuned"]]
    rows = []
    for task in ["binary", "regression"]:
        for index in range(5):
            skill = rng.normal(0.0, 0.05, len(methods))
            for fold in range(2):
                for method, quality in zip(methods, skill):
                    test = 0.3 + quality + rng.normal(0.0, 0.02)
                    optimism = 0.03 if "tuned" in method else 0.0
                    rows.append(
                        {
                            "dataset": f"{task[:3]}{index}",
                            "fold": fold,
                            "seed": 0,
                            "task": task,
                            "method": method,
                            "method_type": "candidate",
                            "primary_metric_name": "metric_error",
                            "primary_metric": test,
                            "val_primary_metric": test - optimism + rng.normal(0.0, 0.02),
                            "fit_seconds": float(rng.uniform(1.0, 100.0)),
                            "predict_seconds": 0.1,
                            "selected_member": method,
                            "ensemble_members": "",
                            "ensemble_weights": "",
                            "source": "synthetic",
                            "problem_type": task,
                            "metric": "error",
                            "metric_error": test,
                            "metric_error_val": 0.0,
                            "time_train_s": 1.0,
                            "time_infer_s": 0.1,
                        }
                    )
    candidates = pd.DataFrame(rows)
    candidates["metric_error_val"] = candidates["val_primary_metric"]
    candidates = candidates[ARTIFACT_OUTPUT_COLUMNS]
    validation_only = {"validation_weight": 1.0, "task_method_prior_weight": 0.0, "task_family_prior_weight": 0.0, "global_method_prior_weight": 0.0, "runtime_weight": 0.0}
    selector = {
        "runtime_weight": 0.000001,
        "selector_method_name": "meta_selector_single",
        "family_prior_weight_grid": [
            validation_only,
            {"validation_weight": 0.1, "task_method_prior_weight": 0.0, "task_family_prior_weight": 0.5, "global_method_prior_weight": 0.4, "runtime_weight": 0.0},
        ],
        "calibrated_weight_grid": [
            validation_only,
            {"validation_weight": 0.1, "task_method_prior_weight": 0.9, "task_family_prior_weight": 0.0, "global_method_prior_weight": 0.0, "runtime_weight": 0.0},
        ],
        "weight_selection": "lodo",
        "inner_source_fit": inner_source_fit,
        "policy_grid": {
            "family_prior_budgets": [50, 10000],
            "calibrated_budget": 10000,
            "guard_margins": [0.0],
            "guard_top_k": [2],
            "guard_runtime_penalties": [0.0],
        },
    }
    derived, policy_methods, reference_methods = nested._selector_grid(selector, "tabarena_results")
    static_rows = build_static_selector_rows(
        candidates,
        runtime_weight=0.000001,
        selector_method_name="meta_selector_single",
        derived_selectors=derived,
        ensemble_repository=None,
    )
    return {
        "name": "Synthetic",
        "kind": "tabarena_results",
        "source": {},
        "candidates": candidates,
        "repository": None,
        "selector": selector,
        "derived_selectors": derived,
        "policy_methods": policy_methods,
        "reference_methods": reference_methods,
        "dataset_descriptors": None,
        "static_selector_rows": static_rows,
    }


HELDOUT = ["bin4", "reg4"]


def _selections(artifact: dict, *, phase: str) -> pd.DataFrame:
    datasets = sorted(artifact["candidates"]["dataset"].unique())
    train = [name for name in datasets if name not in HELDOUT]
    _, _, _, units = nested._evaluate_artifact_split(
        artifact, split_id=0, train_datasets=train, heldout_datasets=HELDOUT, split_mode="task_stratified"
    )
    frame = units[units["phase"].eq(phase) & units["role"].eq("pool")]
    return frame.set_index(["dataset", "fold", "method"])["selected_member"].sort_index()


def _favour(artifact: dict, dataset: str, method: str) -> dict:
    """Make ``method`` the clear test winner on ``dataset``; validation errors are kept."""
    candidates = artifact["candidates"].copy()
    mask = candidates["dataset"].eq(dataset)
    candidates.loc[mask, "primary_metric"] = 0.5
    candidates.loc[mask & candidates["method"].eq(method), "primary_metric"] = 0.0
    candidates.loc[mask, "metric_error"] = candidates.loc[mask, "primary_metric"]
    return {**artifact, "candidates": candidates}


def _changed_on(dataset: str, before: pd.Series, after: pd.Series) -> int:
    on = before.index.get_level_values("dataset") == dataset
    return int((after[on] != before[on]).sum())


def test_no_test_error_of_a_dataset_changes_a_selection_on_it() -> None:
    # The leakage boundary at both levels. Each training dataset is the inner target of the
    # leave-one-dataset-out estimate, so no pool policy's training-side selection on it may
    # depend on its own test errors; no held-out selection may depend on held-out test
    # errors. The shared hierarchy breaks the first condition for adaptive policies.
    shared = _synthetic_artifact(inner_source_fit="shared")
    assert _changed_on("bin1", _selections(shared, phase="train"), _selections(_favour(shared, "bin1", "A (default)"), phase="train")) > 0

    artifact = _synthetic_artifact(inner_source_fit="per_target")
    base_train = _selections(artifact, phase="train")
    base_held = _selections(artifact, phase="heldout")
    for target, method in [("bin1", "A (default)"), ("bin0", "B (tuned)"), ("reg2", "A (tuned)")]:
        assert _changed_on(target, base_train, _selections(_favour(artifact, target, method), phase="train")) == 0
    assert _selections(_favour(artifact, "bin4", "B (default)"), phase="heldout").equals(base_held)


def test_only_adaptive_rows_differ_between_the_hierarchies() -> None:
    shared = _selections(_synthetic_artifact(inner_source_fit="shared"), phase="train")
    per_target = _selections(_synthetic_artifact(inner_source_fit="per_target"), phase="train")
    methods = shared.index.get_level_values("method")
    adaptive = methods.str.contains("task_adaptive") | methods.str.contains("guard")
    assert shared[~adaptive].equals(per_target[~adaptive])
    with pytest.raises(ValueError):
        nested._inner_source_fit({"inner_source_fit": "sometimes"})


def test_guard_memo_changes_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guards of one builder call share their scored selector pool and candidate frames;
    # switching that memo off must leave every output of a split unchanged.
    from tabular_sota_benchmark import artifacts

    artifact = _synthetic_artifact(inner_source_fit="per_target")
    datasets = sorted(artifact["candidates"]["dataset"].unique())
    train = [name for name in datasets if name not in HELDOUT]

    def evaluate() -> tuple[pd.DataFrame, ...]:
        return nested._evaluate_artifact_split(
            artifact, split_id=0, train_datasets=train, heldout_datasets=HELDOUT, split_mode="task_stratified"
        )

    with_memo = evaluate()
    monkeypatch.setattr(artifacts, "_USE_GUARD_MEMO", False)
    without_memo = evaluate()
    for left, right in zip(with_memo, without_memo):
        pd.testing.assert_frame_equal(left, right)
