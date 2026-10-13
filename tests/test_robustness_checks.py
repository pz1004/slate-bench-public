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


def test_copied_rows_marks_flagged_methods_and_exact_randomforest_copies() -> None:
    unit = {"dataset": "a", "fold": 0, "seed": 0}
    results = pd.DataFrame(
        [
            {**unit, "method": "RF (default)", "metric_error": 0.30, "metric_error_val": 0.20},
            {**unit, "method": "KNN (default)", "metric_error": 0.30, "metric_error_val": 0.20},  # copy
            {**unit, "method": "CAT (default)", "metric_error": 0.30, "metric_error_val": 0.21},  # not a copy
            {**unit, "method": "TABPFNV2 (tuned)", "metric_error": 0.25, "metric_error_val": 0.10},  # flagged
            {"dataset": "b", "fold": 0, "seed": 0, "method": "TABPFNV2 (tuned)", "metric_error": 0.2,
             "metric_error_val": 0.1},  # runs on b
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
    assert marked == {"KNN (default)": "copy of a RandomForest row", "TABPFNV2 (tuned)": "metadata flag"}
    assert set(copies["dataset"]) == {"a"}
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
