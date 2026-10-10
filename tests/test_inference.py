from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tabular_sota_benchmark.inference import (
    ArtifactResampler,
    bootstrap_mean_p,
    contrast_table,
    hodges_lehmann,
    gap_closed,
    holm,
    model_confidence_set,
    paired_contrast,
    probability_of_backtest_overfitting,
    superior_predictive_ability,
)


DATASETS = [f"d{index:02d}" for index in range(40)]


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=DATASETS)


def test_resampler_aligns_by_name_and_is_reproducible() -> None:
    first = ArtifactResampler(DATASETS, seed=1, n_resamples=200)
    second = ArtifactResampler(list(reversed(DATASETS)), seed=1, n_resamples=200)
    assert np.array_equal(first.indices, second.indices)
    values = _series(np.arange(40, dtype=float))
    shuffled = values.sample(frac=1.0, random_state=0)
    assert first.mean_ci(values) == first.mean_ci(shuffled)
    with pytest.raises(ValueError):
        first.align(values.iloc[:-1])


def test_identical_series_give_a_null_contrast() -> None:
    rs = ArtifactResampler(DATASETS, n_resamples=500)
    values = _series(np.linspace(0.1, 0.4, 40))
    result = paired_contrast(values, values, rs)
    assert result["mean_difference"] == 0.0
    assert (result["ci_low"], result["ci_high"]) == (0.0, 0.0)
    assert result["wilcoxon_p"] == 1.0
    assert result["bootstrap_p"] == 1.0
    assert result["ttest_p"] == 1.0
    assert result["sign_test_p"] == 1.0
    assert result["ties"] == 40


def test_tost_matches_two_one_sided_t_tests() -> None:
    rng = np.random.default_rng(5)
    left = _series(rng.normal(0.20, 0.05, 40))
    right = _series(left.to_numpy() - rng.normal(0.004, 0.02, 40))
    rs = ArtifactResampler(DATASETS, n_resamples=500)
    result = paired_contrast(left, right, rs, margins=(0.015,), rank_scale=43.0)
    diff = (left - right).to_numpy()
    lower = stats.ttest_1samp(diff, popmean=-0.015, alternative="greater").pvalue
    upper = stats.ttest_1samp(diff, popmean=0.015, alternative="less").pvalue
    assert result["tost_p_0p015"] == pytest.approx(max(lower, upper))
    equivalent = result["tost_p_0p015"] < 0.05
    assert equivalent == (result["smallest_equivalence_margin"] < 0.015)
    assert result["required_n80_0p5rank"] >= 1


def test_mean_test_inverts_the_percentile_interval() -> None:
    # The separation test targets the mean: its p-value is below 0.05 exactly when the
    # reported 95% percentile interval excludes zero (up to one resample).
    rng = np.random.default_rng(11)
    rs = ArtifactResampler(DATASETS, n_resamples=4000)
    noise = rng.normal(0.0, 0.03, 40)
    for shift in np.linspace(-0.02, 0.02, 17):
        diff = _series(noise + shift)
        result = paired_contrast(diff, diff * 0.0, rs)
        boot = rs.boot_means(diff)
        tail = min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
        if abs(tail - 0.025) <= 1.0 / rs.n_resamples:
            continue
        excludes_zero = result["ci_low"] > 0.0 or result["ci_high"] < 0.0
        assert (result["bootstrap_p"] < 0.05) == excludes_zero
        reverse = paired_contrast(diff * 0.0, diff, rs)
        assert reverse["bootstrap_p"] == pytest.approx(result["bootstrap_p"])
    assert bootstrap_mean_p(np.array([0.1, 0.2, 0.3])) == 0.0
    assert bootstrap_mean_p(np.array([-0.1, 0.1])) == 1.0


def test_paired_t_test_matches_scipy() -> None:
    rng = np.random.default_rng(12)
    left = _series(rng.normal(0.20, 0.05, 40))
    right = _series(left.to_numpy() - rng.normal(0.01, 0.02, 40))
    result = paired_contrast(left, right, ArtifactResampler(DATASETS, n_resamples=200))
    expected = stats.ttest_1samp((left - right).to_numpy(), popmean=0.0).pvalue
    assert result["ttest_p"] == pytest.approx(expected)


def test_holm_matches_known_values() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([]) == []
    with pytest.raises(ValueError):
        holm([0.01, float("nan")])


def test_contrast_table_adjusts_within_families() -> None:
    table = contrast_table(
        [
            {"family": "a", "wilcoxon_p": 0.01, "bootstrap_p": 0.02, "ttest_p": 0.03},
            {"family": "a", "wilcoxon_p": 0.04, "bootstrap_p": 0.01, "ttest_p": 0.04},
            {"family": "b", "wilcoxon_p": 0.04, "bootstrap_p": 0.04, "ttest_p": 0.04},
        ]
    )
    assert table["wilcoxon_holm_p"].tolist() == pytest.approx([0.02, 0.04, 0.04])
    assert table["bootstrap_holm_p"].tolist() == pytest.approx([0.02, 0.02, 0.04])
    assert table["ttest_holm_p"].tolist() == pytest.approx([0.06, 0.06, 0.04])
    # Tables of older runs carry only the signed-rank column.
    legacy = contrast_table([{"family": "a", "wilcoxon_p": 0.01}])
    assert "bootstrap_holm_p" not in legacy.columns


def test_mcs_keeps_equal_policies_and_isolates_a_dominant_one() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 0.05, 40)
    rs = ArtifactResampler(DATASETS, n_resamples=500)
    equal = pd.DataFrame({"a": base, "b": base, "c": base}, index=DATASETS)
    assert model_confidence_set(equal, rs)["in_mcs_90"].all()
    dominant = pd.DataFrame(
        {"best": base - 0.2, "b": base + rng.normal(0, 0.01, 40), "c": base + 0.05},
        index=DATASETS,
    )
    result = model_confidence_set(dominant, rs).set_index("policy")
    assert result.loc["best", "mcs_p_value"] == 1.0
    assert result["in_mcs_90"].sum() == 1


def test_spa_detects_a_superior_alternative_only_when_present() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.3, 0.05, 40)
    rs = ArtifactResampler(DATASETS, n_resamples=1000)
    worse = pd.DataFrame(
        {"bench": base, "x": base + 0.05 + rng.normal(0, 0.01, 40), "y": base + 0.1},
        index=DATASETS,
    )
    result = superior_predictive_ability(worse, "bench", rs)
    assert result["spa_p_consistent"] == 1.0
    better = worse.assign(z=base - 0.05 + rng.normal(0, 0.01, 40))
    result = superior_predictive_ability(better, "bench", rs)
    assert result["spa_p_consistent"] < 0.01
    assert result["best_alternative"] == "z"
    assert result["spa_p_lower"] <= result["spa_p_consistent"] <= result["spa_p_upper"]


def test_pbo_separates_a_dominant_policy_from_pure_overfitting() -> None:
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(size=(40, 30)), index=DATASETS)
    noise.columns = [f"p{index}" for index in range(30)]
    dominant = noise.assign(best=noise.max(axis=1) + 1.0)
    assert probability_of_backtest_overfitting(dominant)["pbo"] == 0.0
    # Each policy is excellent on exactly one dataset and neutral elsewhere: the
    # in-sample winner is always a policy whose dataset is in-sample, so out of sample
    # it falls below the median every time.
    overfit = pd.DataFrame(10.0 * np.eye(40), index=DATASETS, columns=[f"q{i}" for i in range(40)])
    assert probability_of_backtest_overfitting(overfit)["pbo"] == 1.0


def test_gap_closed_is_one_at_the_oracle_and_zero_at_single_best() -> None:
    rng = np.random.default_rng(3)
    single_best = _series(rng.uniform(0.2, 0.4, 40))
    oracle = _series(np.zeros(40))
    rs = ArtifactResampler(DATASETS, n_resamples=300)
    assert gap_closed(oracle, single_best, oracle, rs)[0] == pytest.approx(1.0)
    assert gap_closed(single_best, single_best, oracle, rs)[0] == pytest.approx(0.0)


def test_hodges_lehmann_is_the_median_of_walsh_averages() -> None:
    assert hodges_lehmann(np.array([1.0, 2.0, 9.0])) == pytest.approx(3.5)
    rng = np.random.default_rng(13)
    skewed = rng.exponential(1.0, 41)
    # A long right tail pulls the mean above the location the signed-rank test targets.
    assert np.median(skewed) < hodges_lehmann(skewed) < skewed.mean()
