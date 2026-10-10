from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_sota_benchmark.inference import ArtifactResampler
from tabular_sota_benchmark.provenance import _snr_statistics, error_optimism, offset_snr


def test_snr_recovers_the_spread_and_the_training_error() -> None:
    # Offsets with spread s_beta, per-dataset noise sigma and a large component shared by
    # all candidates of a dataset, which changes no selection and must not count.
    rng = np.random.default_rng(0)
    n, k, s_beta, sigma, n_train = 60, 8, 0.03, 0.05, 40
    spreads, errors = [], []
    for _ in range(2000):
        beta = rng.normal(0.0, s_beta, k)
        values = beta + rng.normal(0.0, 0.5, (n, 1)) + rng.normal(0.0, sigma, (n, k))
        spread, se, _ = _snr_statistics(values, n_train)
        spreads.append(spread**2)
        errors.append(se**2)
    assert np.mean(spreads) == pytest.approx(s_beta**2, rel=0.05)
    assert np.mean(errors) == pytest.approx(sigma**2 / n_train, rel=0.02)


def test_snr_ignores_a_shared_component_and_floors_at_zero() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 0.05, (30, 6))
    shifted = values + rng.normal(0.0, 3.0, (30, 1))
    assert _snr_statistics(values, 20) == pytest.approx(_snr_statistics(shifted, 20))
    noise_only = rng.normal(0.0, 0.05, (30, 6)) * 0.0 + rng.normal(0.0, 1.0, (30, 1))
    assert _snr_statistics(noise_only + 1e-12 * rng.normal(size=(30, 6)), 20)[0] == pytest.approx(0.0, abs=1e-9)


def test_offset_snr_drops_incomplete_datasets_and_gives_an_interval() -> None:
    rng = np.random.default_rng(2)
    names = [f"d{index:02d}" for index in range(40)]
    matrix = pd.DataFrame(rng.normal(0.0, 0.03, 5) + rng.normal(0.0, 0.02, (40, 5)), index=names)
    matrix.iloc[3, :] = np.nan
    result = offset_snr(matrix, n_train=30, resampler=ArtifactResampler(names, n_resamples=300))
    assert result["n_datasets"] == 39
    assert result["ratio_ci_low"] <= result["ratio"] <= result["ratio_ci_high"]


def test_error_optimism_normalises_by_the_unit_range() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["a"] * 3 + ["b"] * 2,
            "fold": 0,
            "method": ["x", "y", "z", "x", "y"],
            "primary_metric": [0.10, 0.20, 0.30, 0.5, 0.5],
            "val_primary_metric": [0.05, 0.25, 0.30, 0.4, 0.6],
        }
    )
    result = error_optimism(frame, ["dataset", "fold"])
    assert result["err_optimism"].iloc[:3].tolist() == pytest.approx([0.25, -0.25, 0.0])
    # Unit b has no spread of test errors: its normalised optimism is undefined.
    assert result["err_optimism"].iloc[3:].isna().all()


def test_median_distance_flags_units_whose_median_sits_at_the_best() -> None:
    # The median normalisation divides by the median candidate's distance from the best;
    # the diagnostic reports how often that distance is tiny or zero.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "validation_provenance.py"
    spec = importlib.util.spec_from_file_location("validation_provenance_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unit = {"fold": 0, "seed": 0, "task": "binary", "primary_metric_name": "err"}
    frame = pd.DataFrame(
        [
            # Unit a: median halfway between best and worst.
            *[{**unit, "dataset": "a", "primary_metric": value} for value in [0.0, 0.5, 1.0]],
            # Unit b: most candidates tie at the best, so the median sits at the best.
            *[{**unit, "dataset": "b", "primary_metric": value} for value in [0.2, 0.2, 0.2, 0.9]],
            # Unit c: the median is close to the best (0.5% of the range).
            *[{**unit, "dataset": "c", "primary_metric": value} for value in [0.0, 0.005, 1.0]],
        ]
    )
    result = module.median_distance(frame).iloc[0]
    assert result["n_units"] == 3
    assert result["units_median_at_best"] == 1
    assert result["share_units_below_one_percent"] == pytest.approx(2 / 3)
    assert result["quantile_50"] == pytest.approx(0.005)
