"""Tests for the audit conventions: leave-one-dataset-out weight choice, the shared rank
rule, declared grids, reference rows, reuse of history rows, and the selector kinds."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabular_sota_benchmark import artifacts
from tabular_sota_benchmark.artifacts import (
    ARTIFACT_OUTPUT_COLUMNS,
    ARTIFACT_UNIT_COLUMNS,
    STANDARD_META_FEATURE_COLUMNS,
    build_split_score_artifact_selector_rows,
    build_static_selector_rows,
    selector_provenance,
)
from tabular_sota_benchmark.ranking import (
    candidate_relative_ranks,
    oracle_rank,
    rank_against_candidates,
    rank_rule,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_level_policy_benchmark.py"
SPEC = importlib.util.spec_from_file_location("score_level_policy_benchmark_revision", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

FAMILIES = ["GBM", "NET", "TREE"]
VARIANTS = ["default", "tuned", "tuned + ensemble"]
WEIGHT_COLUMNS = [
    "validation_weight",
    "task_method_prior_weight",
    "task_family_prior_weight",
    "global_method_prior_weight",
    "runtime_weight",
]


def _weights(*values: float) -> dict[str, float]:
    return dict(zip(WEIGHT_COLUMNS, values))


GRID = [
    _weights(1.0, 0.0, 0.0, 0.0, 0.0),
    _weights(0.6, 0.0, 0.25, 0.15, 0.0),
    _weights(0.3, 0.2, 0.3, 0.15, 0.05),
    _weights(0.0, 0.0, 0.0, 1.0, 0.0),
]


def _synthetic_candidates(
    *,
    n_datasets: int = 8,
    n_folds: int = 2,
    seed: int = 7,
    drop_method_in: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Random TabArena-like candidates: two tasks, three families times three variants."""
    rng = np.random.default_rng(seed)
    rows = []
    methods = [f"{family} ({variant})" for family in FAMILIES for variant in VARIANTS]
    for dataset_index in range(n_datasets):
        dataset = f"d{dataset_index:02d}"
        task = "binary" if dataset_index % 2 == 0 else "regression"
        for fold in range(n_folds):
            for method_index, method in enumerate(methods):
                if drop_method_in and drop_method_in.get(dataset) == method:
                    continue
                test = rng.uniform(0.1, 1.0) + 0.02 * method_index
                optimism = 0.15 if "ensemble" in method else 0.0
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": 0,
                        "task": task,
                        "method": method,
                        "method_type": "candidate",
                        "primary_metric_name": f"err:{task}",
                        "primary_metric": test,
                        "val_primary_metric": test - optimism + rng.normal(0.0, 0.05),
                        "fit_seconds": float(rng.uniform(1.0, 5000.0)),
                        "predict_seconds": float(rng.uniform(0.1, 5.0)),
                        "selected_member": "",
                        "ensemble_members": "",
                        "ensemble_weights": "",
                        "source": "synthetic",
                        "problem_type": task,
                        "metric": "err",
                        "metric_error": test,
                        "metric_error_val": test,
                        "time_train_s": 1.0,
                        "time_infer_s": 0.1,
                    }
                )
    return pd.DataFrame(rows)[ARTIFACT_OUTPUT_COLUMNS]


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    return artifacts._with_rank_normalized_validation(frame)


# ------------------------------------------------ leave-one-dataset-out weight choice


def test_loo_prior_features_match_per_dataset_priors() -> None:
    frame = _synthetic_candidates(drop_method_in={"d00": "NET (tuned)", "d03": "NET (tuned)"})
    # A family present in a single dataset exercises the global-family fallback.
    frame.loc[frame["dataset"].eq("d05") & frame["method"].eq("TREE (default)"), "method"] = "SOLO (default)"
    ranked = _ranked(frame)
    loo, _ = artifacts._loo_prior_features(ranked)
    for dataset in sorted(ranked["dataset"].unique()):
        mask = ranked["dataset"].eq(dataset)
        expected, _ = artifacts._calibrated_prior_features(ranked[mask], ranked[~mask])
        for column in [
            "task_method_prior_rank_norm",
            "task_family_prior_rank_norm",
            "global_method_prior_rank_norm",
        ]:
            assert np.allclose(
                loo.loc[mask, column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
            ), (dataset, column)


def test_loo_prior_features_single_dataset_uses_neutral_prior() -> None:
    ranked = _ranked(_synthetic_candidates(n_datasets=1))
    loo, fallback = artifacts._loo_prior_features(ranked)
    assert fallback
    assert np.allclose(loo["global_method_prior_rank_norm"], 0.5)


def test_lodo_weight_selection_matches_bruteforce() -> None:
    ranked = _ranked(_synthetic_candidates())
    weights, objective = artifacts._select_calibration_weights(
        ranked, GRID, weight_selection="lodo"
    )
    frames = []
    for dataset in sorted(ranked["dataset"].unique()):
        mask = ranked["dataset"].eq(dataset)
        features, _ = artifacts._calibrated_prior_features(ranked[mask], ranked[~mask])
        frames.append(features)
    brute = pd.concat(frames, ignore_index=True)
    brute_scores = [artifacts._calibration_objective(brute, candidate) for candidate in GRID]
    assert objective == pytest.approx(min(brute_scores))
    assert weights == GRID[int(np.argmin(brute_scores))]


def test_resubstitution_objective_differs_from_lodo() -> None:
    # Latin square: each method is best on exactly one dataset. With its own test ranks in
    # the prior every method looks equally good (objective 0.5); left out, the prior points
    # to the method that is worst on the held-out dataset (objective 1.0).
    test_ranks = {"d1": [0, 1, 2], "d2": [2, 0, 1], "d3": [1, 2, 0]}
    rows = []
    for dataset, ranks in test_ranks.items():
        for method, rank in zip(["A (default)", "B (default)", "C (default)"], ranks):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": 0,
                    "seed": 0,
                    "task": "binary",
                    "method": method,
                    "method_type": "candidate",
                    "primary_metric_name": "err",
                    "primary_metric": 0.1 + 0.1 * rank,
                    "val_primary_metric": 0.5,
                    "fit_seconds": 1.0,
                    "predict_seconds": 0.1,
                    "selected_member": "",
                    "ensemble_members": "",
                    "ensemble_weights": "",
                    "source": "synthetic",
                    "problem_type": "binary",
                    "metric": "err",
                    "metric_error": 0.1,
                    "metric_error_val": 0.5,
                    "time_train_s": 1.0,
                    "time_infer_s": 0.1,
                }
            )
    ranked = _ranked(pd.DataFrame(rows)[ARTIFACT_OUTPUT_COLUMNS])
    prior_only = [_weights(0.0, 0.0, 0.0, 1.0, 0.0)]
    _, lodo = artifacts._select_calibration_weights(
        ranked, prior_only + prior_only, weight_selection="lodo"
    )
    _, resub = artifacts._select_calibration_weights(
        ranked, prior_only + prior_only, weight_selection="resubstitution"
    )
    assert resub == pytest.approx(0.5)
    assert lodo == pytest.approx(1.0)


def test_weight_selection_mode_is_required() -> None:
    ranked = _ranked(_synthetic_candidates(n_datasets=3))
    with pytest.raises(TypeError):
        artifacts._select_calibration_weights(ranked, GRID)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        artifacts._select_calibration_weights(ranked, GRID, weight_selection="in_sample")


@pytest.mark.parametrize("weight_selection", ["lodo", "resubstitution"])
def test_fast_path_matches_slow_path(weight_selection: str) -> None:
    frame = _synthetic_candidates(drop_method_in={"d01": "GBM (tuned)"})
    common = {
        "method_name": "calibrated",
        "weight_grid": GRID,
        "leave_one_dataset_out": True,
        "weight_selection": weight_selection,
        "max_budget_seconds": 3000.0,
    }
    slow, slow_tuning = artifacts._calibrated_selector(frame, fast_path_min_rows=10**9, **common)
    fast, fast_tuning = artifacts._calibrated_selector(frame, fast_path_min_rows=0, **common)
    key = [*ARTIFACT_UNIT_COLUMNS]
    slow = slow.sort_values(key).reset_index(drop=True)
    fast = fast.sort_values(key).reset_index(drop=True)
    assert slow["selected_member"].tolist() == fast["selected_member"].tolist()
    chosen_slow = slow_tuning.sort_values("dataset")[WEIGHT_COLUMNS].to_numpy()
    chosen_fast = fast_tuning.sort_values("dataset")[WEIGHT_COLUMNS].to_numpy()
    assert np.allclose(chosen_slow, chosen_fast)


# ---------------------------------------------------------- declared calibrated grid


def test_split_builder_requires_declared_grid() -> None:
    frame = _synthetic_candidates(n_datasets=3)
    with pytest.raises(ValueError, match="weight_grid"):
        build_split_score_artifact_selector_rows(
            frame,
            frame,
            runtime_weight=0.0,
            selector_method_name="raw",
            derived_selectors=[{"kind": "budgeted_calibrated", "method_name": "no_grid"}],
        )


def test_selector_grid_requires_calibrated_grid() -> None:
    config = {"family_prior_weight_grid": GRID, "policy_grid": {}}
    with pytest.raises(ValueError, match="calibrated_weight_grid"):
        bench._selector_grid(config, "tabarena_results")
    derived, pool, references = bench._selector_grid(
        {**config, "calibrated_weight_grid": GRID}, "tabarena_results"
    )
    calibrated = next(spec for spec in derived if spec["method_name"] == bench.CALIBRATED_BUDGET_METHOD)
    assert calibrated["weight_grid"] == GRID
    assert calibrated["weight_selection"] == "lodo"
    assert references == []
    guards = [spec for spec in derived if spec["kind"] == "meta_feature_adaptive_selector"]
    assert guards and all(spec["exclude_feature_column_prefixes"] == ["descriptor_"] for spec in guards)


# -------------------------------------------------------- tolerance-consistent ranks


def test_near_tie_rank_is_not_double_counted() -> None:
    # TabRepo Titanic fold 2: the ensemble error sits 2e-17 above 667 tied candidates.
    candidates = np.array([0.1] * 283 + [0.2080536912751678] * 667 + [0.3] * 580)
    value = np.array([0.20805369127516782])
    with rank_rule("consistent"):
        assert candidate_relative_ranks(candidates, value)[0] == pytest.approx(617.0)
    with rank_rule("legacy"):
        assert candidate_relative_ranks(candidates, value)[0] == pytest.approx(1284.0)


def test_candidate_ranks_equal_average_tie_ranks() -> None:
    rng = np.random.default_rng(3)
    values = np.round(rng.uniform(0, 1, 60), 2)
    expected = pd.Series(values).rank(method="average").to_numpy()
    for rule in ["consistent", "legacy"]:
        assert np.allclose(candidate_relative_ranks(values, values, rule=rule), expected)


def test_oracle_rank_uses_average_tie_rank() -> None:
    values = np.array([0.2, 0.2, 0.2, 0.5, 0.9])
    assert oracle_rank(values, rule="consistent") == pytest.approx(2.0)
    assert oracle_rank(values, rule="legacy") == pytest.approx(1.0)


def test_rank_against_candidates_reports_candidate_error_summary() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["a"] * 4,
            "fold": 0,
            "seed": 0,
            "task": "binary",
            "primary_metric_name": "m",
            "method_type": ["candidate", "candidate", "candidate", "selector"],
            "primary_metric": [0.1, 0.3, 0.5, 0.05],
        }
    )
    ranked = rank_against_candidates(frame, unit_columns=ARTIFACT_UNIT_COLUMNS, rule="consistent")
    selector = ranked.iloc[3]
    assert selector["candidate_rank"] == pytest.approx(1.0)
    assert selector["candidate_min_error"] == pytest.approx(0.1)
    assert selector["candidate_median_error"] == pytest.approx(0.3)
    assert selector["candidate_max_error"] == pytest.approx(0.5)
    assert ranked["normalized_rank_regret"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.0])


# ----------------------------------------------------------------------------- parsers


def test_selector_provenance_records_innermost_choice_first() -> None:
    family_row = "utility=0.1;w_val=0.45"
    task_adaptive = family_row + ";adaptive_choice=family_prior_1000s;adaptive_lodo=1"
    guard = task_adaptive + ";adaptive_choice=task_adaptive;adaptive_objective=0.2"
    assert selector_provenance("family_prior_1000s", family_row) == ("", "family_prior_1000s")
    assert selector_provenance("task_adaptive", task_adaptive) == (
        "family_prior_1000s",
        "family_prior_1000s",
    )
    assert selector_provenance("guard_005", guard) == ("task_adaptive", "family_prior_1000s")


def test_method_family_and_variant_parsers() -> None:
    assert artifacts._method_family("MNCA (tuned + ensemble)") == "MNCA"
    assert artifacts._method_family("CatBoost_r12_BAG_L1") == "CatBoost"
    assert artifacts._method_family("NeuralNetTorch_c1_BAG_L1") == "NeuralNetTorch"
    assert artifacts._method_variant("MNCA (tuned + ensemble)") == "tuned + ensemble"
    assert artifacts._method_variant("TABDPT (default)") == "default"
    assert artifacts._method_variant("REALMLP (tuned)") == "tuned"
    assert artifacts._method_variant("CatBoost_r12_BAG_L1") == "n/a"


def test_tabrepo_family_parser_covers_all_configurations() -> None:
    paths = sorted(
        (Path(__file__).resolve().parents[1] / ".deps" / "tabrepo" / "data" / "results").glob(
            "*/configs.parquet"
        )
    )
    if not paths:
        pytest.skip("TabRepo data not available")
    names = pd.read_parquet(paths[0], columns=["framework"])["framework"].astype(str).unique()
    families = {artifacts._method_family(name) for name in names}
    assert all(artifacts._TABREPO_CONFIG_PATTERN.match(name) for name in names)
    assert len(families) < 20 < len(names)


# ----------------------------------------------------------------------------- new kinds


def test_optimism_offsets_exclude_the_target_dataset() -> None:
    frame = _synthetic_candidates(n_datasets=4, n_folds=1)
    rows = artifacts._optimism_corrected_selector(
        frame,
        frame,
        method_name="optimism",
        scope="global",
        max_budget_seconds=None,
        budget_feature="fit_seconds",
        target_is_history=True,
    )
    ranked = _ranked(frame).assign(gap=lambda f: f["test_rank_norm"] - f["val_rank_norm"])
    target = "d02"
    others = ranked[ranked["dataset"].ne(target)]
    offsets = others.groupby("method")["gap"].mean()
    target_rows = ranked[ranked["dataset"].eq(target)].copy()
    target_rows["utility"] = target_rows["val_rank_norm"] + target_rows["method"].map(offsets)
    expected = target_rows.loc[target_rows["utility"].idxmin(), "method"]
    chosen = rows[rows["dataset"].eq(target)]["selected_member"].iloc[0]
    assert chosen == expected


def test_metafeature_knn_uses_same_task_neighbours_without_target() -> None:
    frame = _synthetic_candidates(n_datasets=8, n_folds=1)
    datasets = sorted(frame["dataset"].unique())
    descriptors = pd.DataFrame(
        {
            "dataset": datasets,
            "descriptor_log_instances": np.arange(len(datasets), dtype=float),
            "descriptor_log_features": 1.0,
            "descriptor_n_classes": 2.0,
            "descriptor_cat_share": 0.0,
        }
    )
    rows = artifacts._metafeature_knn_selector(
        frame,
        frame,
        method_name="knn",
        descriptors=descriptors,
        feature_columns=STANDARD_META_FEATURE_COLUMNS,
        top_k=2,
        validation_weight=0.0,
        max_budget_seconds=None,
        budget_feature="fit_seconds",
        target_is_history=True,
    )
    detail = rows[rows["dataset"].eq("d04")]["ensemble_weights"].iloc[0]
    neighbours = detail.split("knn_neighbors=")[1].split(";")[0].split(",")
    # d04 is a binary task (even index); nearest same-task datasets are d02 and d06.
    assert sorted(neighbours) == ["d02", "d06"]


def test_variant_filter_restricts_validation_best() -> None:
    frame = _synthetic_candidates(n_datasets=3)
    rows = artifacts._validation_rank_rows(
        frame,
        {"kind": "validation_rank", "method_name": "vb_default", "method_variant_allowlist": ["default"]},
    )
    assert rows["selected_member"].map(artifacts._method_variant).eq("default").all()


# ----------------------------------------------------------------------------- reuse


def _small_selector_config() -> dict[str, object]:
    return {
        "runtime_weight": 1e-6,
        "selector_method_name": "meta_selector_single",
        "weight_selection": "lodo",
        "family_prior_weight_grid": GRID[:3],
        "calibrated_weight_grid": GRID,
        "policy_grid": {
            "family_prior_budgets": [1000, 10000],
            "calibrated_budget": 10000,
            "guard_margins": [0.02],
            "guard_top_k": [2],
            "guard_runtime_penalties": [0.0],
        },
    }


def _small_artifact(selector_config: dict[str, object]) -> dict[str, object]:
    candidates = _synthetic_candidates(n_datasets=8)
    derived, pool, references = bench._selector_grid(selector_config, "tabarena_results")
    return {
        "name": "Synthetic",
        "kind": "tabarena_results",
        "source": {},
        "candidates": candidates,
        "repository": None,
        "selector": selector_config,
        "derived_selectors": derived,
        "policy_methods": pool,
        "reference_methods": references,
        "dataset_descriptors": None,
        "static_selector_rows": build_static_selector_rows(
            candidates,
            runtime_weight=1e-6,
            selector_method_name="meta_selector_single",
            derived_selectors=derived,
        ),
    }


def test_precomputed_history_and_static_rows_reproduce_heldout_rows() -> None:
    artifact = _small_artifact(_small_selector_config())
    candidates = artifact["candidates"]
    train = candidates[candidates["dataset"].isin(["d00", "d01", "d02", "d03", "d04", "d05"])]
    heldout = candidates[candidates["dataset"].isin(["d06", "d07"])]
    options = {
        "runtime_weight": 1e-6,
        "selector_method_name": "meta_selector_single",
        "derived_selectors": artifact["derived_selectors"],
    }
    baseline, _ = build_split_score_artifact_selector_rows(train, heldout, **options)
    history, _ = build_split_score_artifact_selector_rows(
        train, train, static_selector_rows=artifact["static_selector_rows"], **options
    )
    reused, _ = build_split_score_artifact_selector_rows(
        train,
        heldout,
        precomputed_history_selectors=history,
        static_selector_rows=artifact["static_selector_rows"],
        **options,
    )
    key = [*ARTIFACT_UNIT_COLUMNS, "method"]
    left = baseline.sort_values(key).reset_index(drop=True)
    right = reused.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_reference_rows_never_change_the_nested_choice() -> None:
    base_config = _small_selector_config()
    with_references = {
        **base_config,
        "reference_selectors": [
            {
                "kind": "budgeted_calibrated",
                "method_name": "reference_single_best",
                "weights": _weights(0.0, 0.0, 0.0, 1.0, 0.0),
                "max_budget_seconds": None,
            },
            {
                "kind": "optimism_corrected_validation",
                "method_name": "reference_optimism",
                "scope": "global",
            },
        ],
    }
    splits = [("d00", "d01"), ("d02", "d05"), ("d06", "d07")]
    for with_refs in [False, True]:
        artifact = _small_artifact(with_references if with_refs else base_config)
        chosen = []
        for split_id, heldout in enumerate(splits):
            train = [dataset for dataset in sorted(artifact["candidates"]["dataset"].unique()) if dataset not in heldout]
            split_result, train_scores, heldout_scores, unit_scores = bench._evaluate_artifact_split(
                artifact,
                split_id=split_id,
                train_datasets=train,
                heldout_datasets=list(heldout),
                split_mode="global",
            )
            chosen.append(split_result["selected_policy"].iloc[0])
            if with_refs:
                assert set(train_scores.loc[train_scores["role"].eq("reference"), "method"]) == {
                    "reference_single_best",
                    "reference_optimism",
                }
                assert (train_scores.loc[train_scores["role"].eq("reference"), "method_order"] < 0).all()
                assert {"pool", "reference", "oracle", "alias"} <= set(unit_scores["role"])
        if with_refs:
            assert chosen == baseline_choices
        else:
            baseline_choices = chosen


def test_unit_scores_aggregate_to_heldout_policy_scores() -> None:
    artifact = _small_artifact(_small_selector_config())
    datasets = sorted(artifact["candidates"]["dataset"].unique())
    _, _, heldout_scores, unit_scores = bench._evaluate_artifact_split(
        artifact,
        split_id=0,
        train_datasets=datasets[:6],
        heldout_datasets=datasets[6:],
        split_mode="global",
    )
    held = unit_scores[unit_scores["phase"].eq("heldout")]
    means = held.groupby("method")["normalized_rank_regret"].mean()
    summary = heldout_scores.set_index("method")["mean_normalized_rank_regret"]
    for method, value in summary.items():
        assert means[method] == pytest.approx(value)


def test_history_selector_fallback_flag_matches_per_dataset_priors() -> None:
    """The per-target fallback flag must equal what _calibrated_prior_features reports for
    that target alone, including a (task, method) prior that is missing although the method
    appears under another task (a regression once made this flag both wrong and very slow)."""
    frame = _synthetic_candidates(n_datasets=10)
    train_datasets = {f"d{index:02d}" for index in range(6)}
    history = frame[frame["dataset"].isin(train_datasets)]
    history = history[~(history["task"].eq("binary") & history["method"].eq("NET (tuned)"))]
    target = frame[~frame["dataset"].isin(train_datasets)].copy()
    target.loc[target["dataset"].eq("d07") & target["method"].eq("TREE (default)"), "method"] = "SOLO (default)"

    _, tuning = artifacts._calibrated_selector_from_history(
        history, target, method_name="m", weight_grid=[GRID[1]], weight_selection="lodo"
    )

    def prepared(rows: pd.DataFrame) -> pd.DataFrame:
        return artifacts._apply_candidate_budget(_ranked(rows), max_budget_seconds=None, budget_feature="fit_seconds")

    history_ranked = prepared(history)
    target_ranked = prepared(target)
    for _, row in tuning.iterrows():
        rows = target_ranked[target_ranked["dataset"].eq(row["dataset"])]
        _, expected = artifacts._calibrated_prior_features(rows, history_ranked)
        assert bool(row["fallback_used"]) == expected, row["dataset"]
    assert tuning.set_index("dataset")["fallback_used"].to_dict() == {
        "d06": True, "d07": True, "d08": True, "d09": False
    }


class _FakeRepository:
    """Minimal TabRepo stand-in: records which datasets the loader asks metrics for."""

    def __init__(self, datasets: list[str]) -> None:
        self._datasets = datasets
        self.requested: list[str] | None = None

    def datasets(self) -> list[str]:
        return list(self._datasets)

    def metrics(self, datasets: list[str], folds: object = None, configs: object = None) -> pd.DataFrame:
        self.requested = list(datasets)
        rows = [
            {"dataset": dataset, "fold": 0, "framework": f"GBM_c{index}_BAG_L1", "metric_error": 0.1 * (index + 1),
             "metric_error_val": 0.1 * (index + 1), "time_train_s": 1.0, "time_infer_s": 0.1}
            for dataset in datasets
            for index in range(3)
        ]
        return pd.DataFrame(rows).set_index(["dataset", "fold", "framework"])

    def dataset_info(self, dataset: str) -> dict[str, str]:
        return {"problem_type": "binary", "metric": "roc_auc"}


def test_tabrepo_dataset_allowlist_file_pins_the_slice(tmp_path: Path) -> None:
    listing = tmp_path / "slice.txt"
    listing.write_text("# comment line\nalpha\n\ngamma\n", encoding="utf-8")
    repository = _FakeRepository(["alpha", "beta", "gamma", "delta"])
    frame = artifacts.load_tabrepo_results({"dataset_allowlist_file": str(listing)}, repository=repository)
    assert repository.requested == ["alpha", "gamma"]
    assert sorted(frame["dataset"].unique()) == ["alpha", "gamma"]
    with pytest.raises(ValueError, match="not both"):
        artifacts.load_tabrepo_results(
            {"dataset_allowlist": ["alpha"], "dataset_allowlist_file": str(listing)}, repository=repository
        )


def test_method_family_is_mapped_per_distinct_method() -> None:
    methods = pd.Series(["GBM (tuned)", "CatBoost_r12_BAG_L1", "GBM (tuned)", "RF (default)"] * 3, name="method")
    for series in [methods, methods.astype(object)]:
        expected = series.map(artifacts._method_family)
        result = artifacts._map_method_family(series)
        pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("weight_selection", ["lodo", "resubstitution"])
def test_fast_path_rows_for_some_targets_match_all_targets(weight_selection: str) -> None:
    # Restricting the rows to some target datasets changes none of their rows: every other
    # dataset still enters the priors and the weight objective.
    frame = _synthetic_candidates(drop_method_in={"d01": "GBM (tuned)"})
    common = {
        "method_name": "calibrated",
        "weight_grid": GRID,
        "leave_one_dataset_out": True,
        "weight_selection": weight_selection,
        "max_budget_seconds": 3000.0,
        "fast_path_min_rows": 0,
    }
    full, full_tuning = artifacts._calibrated_selector(frame, **common)
    targets = set(sorted(frame["dataset"].unique())[1:3])
    some, some_tuning = artifacts._calibrated_selector(frame, lodo_targets=targets, **common)
    expected = full[full["dataset"].isin(targets)].reset_index(drop=True)
    pd.testing.assert_frame_equal(some.reset_index(drop=True), expected)
    expected_tuning = full_tuning[full_tuning["dataset"].isin(targets)].reset_index(drop=True)
    pd.testing.assert_frame_equal(some_tuning.reset_index(drop=True), expected_tuning)
