from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


corrected = _load("corrected_resampled_test")
imputed = _load("imputed_rows")
error_space = _load("error_space_correction")
reference = _load("provenance_reference")
influence = _load("influence_check")


def test_corrected_test_inflates_the_variance_by_the_nadeau_bengio_factor() -> None:
    differences = np.array([1.0, 2.0, 3.0, 4.0])
    row = corrected.corrected_test(differences, n_train=3, n_test=1, margins=[5.0, 1.0])
    se = math.sqrt((1 / 4 + 1 / 3) * differences.var(ddof=1))
    assert math.isclose(row["nb_se"], se)
    assert math.isclose(row["nb_p"], 2 * stats.t.sf(2.5 / se, 3))
    half = stats.t.ppf(0.95, 3) * se
    assert math.isclose(row["nb_ci90_low"], 2.5 - half)
    assert row["nb_equivalent_5p000"] == bool(2.5 + half < 5.0)
    assert row["nb_equivalent_1p000"] is False


def test_copied_rows_marks_flagged_methods_and_randomforest_timing() -> None:
    unit = {"dataset": "a", "fold": 0, "seed": 0}
    rf = {"metric_error": 0.30, "metric_error_val": 0.20, "time_infer_s": 0.123456789}
    results = pd.DataFrame(
        [
            {**unit, "method": "RF (default)", **rf, "time_train_s": 2.0},
            {**unit, "method": "KNN (default)", **rf, "time_train_s": 2.0},  # imputed
            {**unit, "method": "KNN (tuned)", "metric_error": 0.301, "metric_error_val": 0.20,
             "time_infer_s": 0.123456789, "time_train_s": 102.0},  # imputed: same timing, near errors
            {**unit, "method": "XT (tuned)", "metric_error": 0.30, "metric_error_val": 0.20,
             "time_infer_s": 0.5, "time_train_s": 9.0},  # equal errors, own timing: not imputed
            {**unit, "method": "CAT (default)", "metric_error": 0.25, "metric_error_val": 0.21,
             "time_infer_s": 0.2, "time_train_s": 5.0},
            {**unit, "method": "TABPFNV2 (tuned)", **rf, "time_train_s": 402.0},  # flagged
            {"dataset": "b", "fold": 0, "seed": 0, "method": "TABPFNV2 (tuned)", "metric_error": 0.2,
             "metric_error_val": 0.1, "time_infer_s": 0.3, "time_train_s": 1.0},  # runs on b
            {"dataset": "b", "fold": 0, "seed": 0, "method": "RF (default)", "metric_error": 0.4,
             "metric_error_val": 0.3, "time_infer_s": 0.9, "time_train_s": 1.0},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"dataset_name": "a", "can_run_tabpfnv2": False, "can_run_tabicl": True},
            {"dataset_name": "b", "can_run_tabpfnv2": True, "can_run_tabicl": True},
        ]
    )
    copies = imputed.copied_rows(results, metadata)
    marked = dict(zip(copies["method"], copies["rule"]))
    assert marked == {
        "KNN (default)": "RandomForest timing",
        "KNN (tuned)": "RandomForest timing",
        "TABPFNV2 (tuned)": "metadata flag",
    }
    assert set(copies["dataset"]) == {"a"}
    gaps = dict(zip(copies["method"], copies["relative_error_gap"]))
    assert gaps["KNN (default)"] == 0.0 and 0.0 < gaps["KNN (tuned)"] < 0.01
    assert imputed.family("TABPFNV2 (tuned + ensemble)") == "TABPFNV2"


def _toy_units() -> pd.DataFrame:
    """Two datasets; candidate 'opt' looks best on validation but is worst on test."""
    rows = []
    for dataset in ["d1", "d2", "d3"]:
        for method, val, test in [("opt", 0.10, 0.40), ("fair", 0.20, 0.21), ("other", 0.30, 0.31)]:
            rows.append(
                {
                    "dataset": dataset, "fold": 0, "seed": 0, "task": "binary", "primary_metric_name": "error",
                    "method": method, "val_primary_metric": val, "primary_metric": test, "fit_seconds": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_error_space_correction_without_offsets_is_validation_best() -> None:
    frame = error_space.prepare(_toy_units())
    held_out = pd.DataFrame({"split_id": [0, 1, 2], "dataset": ["d1", "d2", "d3"]})
    plain = error_space.evaluate(frame, held_out, None, by="method")
    # Validation-best takes 'opt', the worst of three on test: normalized regret 1.
    assert np.allclose(plain.to_numpy(), 1.0)
    corrected_regret = error_space.evaluate(
        frame, held_out, error_space.dataset_offsets(frame, by="method"), by="method"
    )
    # Offsets learned on the other two datasets push 'opt' down; 'fair' is the best on test.
    assert np.allclose(corrected_regret.to_numpy(), 0.0)


def test_weighted_split_means_average_to_the_dataset_level_mean() -> None:
    rows = []
    held_out = {0: ["a", "b"], 1: ["a", "c"], 2: ["a", "b"]}  # a held out 3 times, b 2, c 1
    for split, datasets in held_out.items():
        for dataset in datasets:
            value = {"a": 0.1, "b": 0.4, "c": 0.7}[dataset] + 0.01 * split
            rows.append({"split_id": split, "dataset": dataset, "method": "m", "mean_normalized_rank_regret": value})
    scores = pd.DataFrame(rows)
    means = corrected.split_means(scores)["m"]
    dataset_level = scores.groupby("dataset")["mean_normalized_rank_regret"].mean().mean()
    assert math.isclose(float(means.mean()), float(dataset_level))


def test_defaults_then_tuned_deploys_the_family_chosen_on_defaults() -> None:
    rows = []
    # Unit 0: family A's default has the best default validation error, but B's tuned + ensemble
    # has the best validation error overall (the optimistic candidate validation-best takes).
    for method, val, regret in [
        ("A (default)", 0.20, 0.6), ("A (tuned)", 0.18, 0.4), ("A (tuned + ensemble)", 0.17, 0.1),
        ("B (default)", 0.25, 0.9), ("B (tuned + ensemble)", 0.10, 0.8),
    ]:
        rows.append({"unit": 0, "dataset": "d", "method": method, "val_primary_metric": val, "tie": 0.0,
                     "family": method.split(" (")[0], "regret": regret})
    # Unit 1: the chosen family has no tuned + ensemble row, so its tuned row is deployed.
    for method, val, regret in [("C (default)", 0.20, 0.5), ("C (tuned)", 0.19, 0.2), ("D (default)", 0.30, 0.0)]:
        rows.append({"unit": 1, "dataset": "d", "method": method, "val_primary_metric": val, "tie": 0.0,
                     "family": method.split(" (")[0], "regret": regret})
    choices = reference.unit_choices(pd.DataFrame(rows)).set_index("unit")
    assert choices.loc[0, "deployed_method"] == "A (tuned + ensemble)"
    assert choices.loc[0, reference.VALIDATION_BEST] == 0.8
    assert choices.loc[0, reference.VALIDATION_BEST_DEFAULT] == 0.6
    assert choices.loc[1, "deployed_method"] == "C (tuned)"


def test_yuen_one_sample_reduces_to_the_t_test_without_trimming() -> None:
    values = np.array([-0.3, -0.1, 0.0, 0.05, 0.2, -0.25, -0.4, 0.1, -0.05, -0.2])
    mean, p = influence.yuen_one_sample(values, 0.0)
    assert math.isclose(mean, float(values.mean()))
    assert math.isclose(p, float(stats.ttest_1samp(values, 0.0).pvalue))
    trimmed, _ = influence.yuen_one_sample(values, 0.2)
    assert math.isclose(trimmed, float(stats.trim_mean(values, 0.2)))



source_groups = _load("tabrepo_source_groups")
calibration = _load("calibration_study")


def test_cluster_bootstrap_with_singleton_groups_is_the_dataset_bootstrap() -> None:
    from tabular_sota_benchmark.inference import ArtifactResampler, bootstrap_mean_p

    rng = np.random.default_rng(0)
    diff = pd.Series(rng.normal(-0.02, 0.05, 30), index=[f"d{i:02d}" for i in range(30)])
    low, high, p, clusters = source_groups.cluster_interval(diff, {}, seed=11, resamples=2000)
    resampler = ArtifactResampler(diff.index, seed=11, n_resamples=2000)
    boot = resampler.align(diff)[resampler.indices].mean(axis=1)
    assert clusters == 30
    assert math.isclose(low, float(np.quantile(boot, 0.025)), abs_tol=1e-12)
    assert math.isclose(high, float(np.quantile(boot, 0.975)), abs_tol=1e-12)
    assert math.isclose(p, bootstrap_mean_p(boot), abs_tol=1e-12)
    # Grouping keeps the point estimate and resamples groups as units.
    groups = {name: "g" + name[-1] for name in diff.index}
    _, _, _, clusters = source_groups.cluster_interval(diff, groups, seed=11, resamples=2000)
    assert clusters == 10


def test_metadata_mask_removes_only_documented_rows(tmp_path: Path) -> None:
    rows = []
    for dataset in ["a", "b"]:
        for method, infer in [("RF (default)", 1.5), ("TABPFNV2 (default)", 1.5), ("KNN (tuned)", 1.5), ("CAT (default)", 0.7)]:
            rows.append({"dataset": dataset, "fold": 0, "seed": 0, "method": method, "metric_error": 0.2,
                         "metric_error_val": 0.2, "time_train_s": 1.0, "time_infer_s": infer,
                         "problem_type": "binary", "metric": "roc_auc"})
    results, metadata = tmp_path / "results.csv", tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(results, index=False)
    pd.DataFrame({"dataset_name": ["a", "b"], "can_run_tabpfnv2": [False, True], "can_run_tabicl": [True, True]}).to_csv(metadata, index=False)
    import argparse

    for mask, expected in [("all", {"RF (default)", "CAT (default)"}), ("metadata", {"RF (default)", "CAT (default)", "KNN (tuned)"})]:
        out = tmp_path / mask
        imputed.run(argparse.Namespace(results=str(results), metadata=str(metadata), output_dir=out, mask=mask))
        name = "df_results_no_imputed.csv" if mask == "all" else "df_results_no_flagged.csv"
        kept = pd.read_csv(out / name)
        # TabPFNv2 is documented as imputed on dataset a only; on b its row has the timing
        # signature and is removed only by the full mask.
        assert set(kept.loc[kept["dataset"].eq("a"), "method"]) == expected
        assert ("TABPFNV2 (default)" in set(kept.loc[kept["dataset"].eq("b"), "method"])) == (mask == "metadata")


def _tiny_world() -> dict:
    methods = [f"{family} ({variant})" for family in ["RF", "XT"] for variant in ["default", "tuned", "tuned + ensemble"]]
    mu = {f"{task}|{size}|{m}": 0.1 * index for task in calibration.TASKS for size in ["small", "large"] for index, m in enumerate(methods)}
    noise = {size: {"test_var": 0.1, "val_var": 0.03, "rho": -0.2} for size in ["small", "large"]}
    return {
        "methods": methods, "mu_test": mu, "mu_val": {k: v - 0.05 for k, v in mu.items()},
        "profile_root": np.eye(2 * len(methods)).tolist(), "profile_scale": {"small": 1.0, "large": 1.0},
        "shrinkage": 0.05, "tuning_sd": 0.0, "noise": noise, "family_noise_share": {"test": 0.45, "val": 0.4},
        "fit": {"family_log_default": {"RF": 1.0, "XT": 0.5}, "family_log_ratio": {"RF": 5.0, "XT": 5.0},
                "zeta": {size: {"mean": 0.0, "sd": 0.3} for size in ["small", "large"]}, "resid_sd": 0.05,
                "predict_base": {m: -2.0 for m in methods}, "predict_sd": 0.1},
        "scale_pairs": {task: [[0.3, 0.05]] for task in calibration.TASKS}, "imputation": {},
        "small_share": {task: 0.3 for task in calibration.TASKS},
    }


def test_calibration_generator_writes_tabarena_shaped_tables() -> None:
    table = calibration.generate_table(_tiny_world(), np.random.default_rng(3), {"binary": 4, "regression": 2})
    assert list(table.columns) == ["dataset", "fold", "seed", "method", "metric_error", "time_train_s", "time_infer_s", "metric_error_val", "problem_type", "metric"]
    units = table.groupby("dataset")["fold"].nunique()
    assert set(units) <= {calibration.SMALL_UNITS, calibration.LARGE_UNITS} and len(units) == 6
    assert (table.groupby(["dataset", "fold"]).size() == 6).all()
    assert (table[["metric_error", "metric_error_val"]] > 0).all().all()
    wide = table.pivot_table(index=["dataset", "fold"], columns="method", values="time_train_s")
    # A tuned-and-ensembled candidate records its tuned candidate's fit time, as in TabArena.
    assert np.allclose(wide["RF (tuned + ensemble)"], wide["RF (tuned)"])
    world = calibration.scenario_world(_tiny_world(), {"small_share_factor": 0.5, "tuning_sd": 0.35, "family_noise_share": 0.85})
    assert world["small_share"]["binary"] == 0.15 and world["tuning_sd"] == 0.35
    assert world["family_noise_share"] == {"test": 0.85, "val": 0.85}


def test_coverage_labels_rest_on_wilson_intervals() -> None:
    low, high = calibration.wilson(95, 100)
    # The Wilson interval of 95/100 (z = 1.96).
    assert math.isclose(low, 0.8882, abs_tol=1e-4) and math.isclose(high, 0.9785, abs_tol=1e-4)
    assert calibration.wilson(50, 50)[1] == 1.0
    # The largest coverage that R replicates can label anti-conservative (upper end below 0.925).
    assert calibration.detectable_coverage(100) == 0.87
    assert calibration.detectable_coverage(50) == 0.84
    assert calibration.wilson(87, 100)[1] < 0.925 <= calibration.wilson(88, 100)[1]


def test_task_weighted_split_means_use_the_artifact_task_mix() -> None:
    rows = []
    for split in range(3):
        for dataset, task_value in [("d0_bin", 1.0), ("d1_bin", 1.0), ("d2_mul", 2.0), ("d3_reg", 4.0)]:
            rows.append({"split_id": split, "dataset": dataset, "method": "L", "regret": task_value + 0.5})
            rows.append({"split_id": split, "dataset": dataset, "method": "R", "regret": 0.5})
    values = calibration.task_weighted_split_means(pd.DataFrame(rows), "L", "R")
    share = calibration.TASK_SHARE
    expected = share["bin"] * 1.0 + share["mul"] * 2.0 + share["reg"] * 4.0
    assert values.shape == (3,) and np.allclose(values, expected)
    assert math.isclose(sum(share.values()), 1.0)


grouped_comparison = _load("tabrepo_grouped_comparison")


def test_design_means_reproduce_the_dataset_level_contrast() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for split in range(8):
        for dataset in rng.choice([f"d{i}" for i in range(10)], size=4, replace=False):
            for method, shift in [("nested_policy_selected", 0.0), ("meta_selector_single", 0.01)]:
                rows.append({"split_id": split, "dataset": dataset, "method": method, "mean_normalized_rank_regret": rng.normal(0.2, 0.05) + shift})
    scores = pd.DataFrame(rows)
    ids, values, held = grouped_comparison.split_matrix(scores, "meta_selector_single")
    full = grouped_comparison.design_means(np.ones((1, ids.size)), values, held)[0]
    assert math.isclose(full, grouped_comparison.dataset_contrast(scores, ids, "meta_selector_single"), abs_tol=1e-12)
    # A design given as split counts: splits drawn twice count twice.
    counts = np.zeros((1, ids.size))
    counts[0, [0, 1, 3]] = [1, 2, 1]
    kept = scores[scores["split_id"].isin(ids[[0, 1, 3]])]
    doubled = pd.concat([kept, kept[kept["split_id"].eq(ids[1])]], ignore_index=True)
    means = doubled.groupby(["dataset", "method"])["mean_normalized_rank_regret"].mean().unstack("method")
    expected = float((means["nested_policy_selected"] - means["meta_selector_single"]).mean())
    assert math.isclose(grouped_comparison.design_means(counts, values, held)[0], expected, abs_tol=1e-12)


def _script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_construction_lattice_tells_a_single_forest_from_out_of_bag_and_bags() -> None:
    check = _script("tabrepo_construction_check")
    rng = np.random.default_rng(0)
    single = rng.integers(1, 300, 500) / 300
    bagged = rng.integers(1, 2400, 500) / 2400
    out_of_bag = rng.integers(1, 100, 500) / rng.integers(100, 120, 500)
    assert check.lattice_share(single, 300) == 1.0
    assert check.lattice_share(bagged, 300) < 0.3
    assert check.lattice_share(out_of_bag, 300) < 0.3
    # 0 and 1 lie on every lattice and are left out.
    assert np.isnan(check.lattice_share(np.array([0.0, 1.0]), 300))
    assert list(check.reduced_denominators(np.array([0.5, 0.2, 1 / 3]))) == [2, 5, 3]


def test_paired_inner_comparison_is_zero_for_a_run_and_itself() -> None:
    module = _script("tabrepo_grouped_inner_comparison")
    datasets = [f"d{i}" for i in range(30)]
    outer = pd.Series([0.01 * (i % 7) - 0.03 for i in range(30)], index=datasets)
    same = module.paired_change(outer, outer.copy(), {})
    assert same["change"] == 0.0 and same["p"] == 1.0 and same["n_datasets_changed"] == 0
    shifted = module.paired_change(outer, outer + 0.002, {"d0": "g", "d1": "g"})
    assert abs(shifted["change"] - 0.002) < 1e-12 and shifted["ci_low"] > 0
    assert shifted["n_clusters"] == 29
