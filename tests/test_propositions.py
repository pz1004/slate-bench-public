"""Monte Carlo checks of the closed forms in Propositions 1 and 2 and Remark 1 of the paper,
including the unequal-quality case of Proposition 1(iv), and an exact check of the
two-point noise example in the proof of Proposition 1(iii).

Validation-best selects argmin_m V_m with V_m = T_m - beta_m + eps_m; the corrected rule adds
an estimated offset to each V_m. Every simulation uses a fixed seed, and each assertion allows
five Monte Carlo standard errors.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate, stats

N = 400_000
TAU, SIGMA = 1.0, 0.5
S = np.sqrt(TAU**2 + 2 * SIGMA**2)


def _two_candidates(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exchangeable test errors with D = T_a - T_b ~ N(0, tau^2), and validation noise
    # E = eps_a - eps_b ~ N(0, 2 sigma^2).
    rng = np.random.default_rng(seed)
    t_a, t_b = rng.normal(0.0, TAU / np.sqrt(2), (2, N))
    noise = rng.normal(0.0, SIGMA, N) - rng.normal(0.0, SIGMA, N)
    return t_a, t_b, noise


def _selected_error(t_a: np.ndarray, t_b: np.ndarray, noise: np.ndarray, threshold: float) -> np.ndarray:
    # Candidate a is selected exactly when D + E < threshold.
    return np.where(t_a - t_b + noise < threshold, t_a, t_b)


def _close(values: np.ndarray, expected: float) -> bool:
    return abs(values.mean() - expected) < 5 * values.std(ddof=1) / np.sqrt(values.size)


def _unequal_expected_error(quality_gap: float, offset_gap: float) -> float:
    # Proposition 1(iv): E[T_b] + mu Phi((c - mu) / s) - (tau^2 / s) phi((c - mu) / s), E[T_b] = 0.
    x = (offset_gap - quality_gap) / S
    return quality_gap * stats.norm.cdf(x) - (TAU**2 / S) * stats.norm.pdf(x)


@pytest.mark.parametrize("quality_gap", [0.5, -0.4])
def test_unequal_quality_closed_form_and_minimizer(quality_gap: float) -> None:
    # Proposition 1(iv): with D = T_a - T_b ~ N(mu, tau^2) the expected selected error has the
    # closed form above and is smallest at c* = -2 sigma^2 mu / tau^2, not at equal offsets.
    rng = np.random.default_rng(7)
    t_b = rng.normal(0.0, TAU / np.sqrt(2), N)
    t_a = rng.normal(quality_gap, TAU / np.sqrt(2), N)
    noise = rng.normal(0.0, SIGMA, N) - rng.normal(0.0, SIGMA, N)
    best = -2 * SIGMA**2 * quality_gap / TAU**2
    for gap in [-0.8, best, 0.0, 0.6]:
        assert _close(_selected_error(t_a, t_b, noise, gap), _unequal_expected_error(quality_gap, gap))
    grid = np.linspace(-1.5, 1.5, 3001)
    assert grid[np.argmin([_unequal_expected_error(quality_gap, gap) for gap in grid])] == pytest.approx(best, abs=1e-3)
    # Equal offsets are worse than the minimizer, and the error rises on both sides of it.
    paired = _selected_error(t_a, t_b, noise, 0.0) - _selected_error(t_a, t_b, noise, best)
    assert paired.mean() > 5 * paired.std(ddof=1) / np.sqrt(N)
    assert _unequal_expected_error(quality_gap, best - 0.3) > _unequal_expected_error(quality_gap, best)
    assert _unequal_expected_error(quality_gap, best + 0.3) > _unequal_expected_error(quality_gap, best)


@pytest.mark.parametrize("offset_gap", [0.0, 0.3, 0.8])
def test_two_candidate_selection_probability(offset_gap: float) -> None:
    # Proposition 1(ii): validation-best selects a with probability Phi((beta_a - beta_b) / s).
    t_a, t_b, noise = _two_candidates(seed=1)
    selected_a = (t_a - t_b + noise < offset_gap).astype(float)
    assert _close(selected_a, stats.norm.cdf(offset_gap / S))


def test_two_candidate_expected_test_error_grows_with_the_offset_gap() -> None:
    # Proposition 1(iii): E[T_a] - (tau^2 / s) phi((beta_a - beta_b) / s), with E[T_a] = 0.
    t_a, t_b, noise = _two_candidates(seed=2)
    means = []
    for gap in [0.0, 0.3, 0.8, -0.8]:
        errors = _selected_error(t_a, t_b, noise, gap)
        assert _close(errors, -(TAU**2 / S) * stats.norm.pdf(gap / S))
        means.append(errors.mean())
    assert means[0] < means[1] < means[2]
    assert means[3] == pytest.approx(means[2], abs=0.01)


def test_equal_offsets_minimize_expected_error_and_rank() -> None:
    # Proposition 1(iii) for four exchangeable Gaussian candidates: unequal offsets raise the
    # expected test error and test rank of the candidate validation-best selects.
    rng = np.random.default_rng(3)
    k = 4
    tests = rng.normal(0.0, 1.0, (N, 1)) + rng.normal(0.0, 1.0, (N, k))
    validation = tests + rng.normal(0.0, 0.7, (N, k))
    ranks = tests.argsort(axis=1).argsort(axis=1) + 1
    rows = np.arange(N)
    outcomes = {}
    for name, offsets in {"equal": np.full(k, 0.2), "unequal": np.array([0.0, 0.3, 0.6, 0.9])}.items():
        choice = (validation - offsets).argmin(axis=1)
        outcomes[name] = (tests[rows, choice], ranks[rows, choice].astype(float))
    for index in range(2):
        paired = outcomes["unequal"][index] - outcomes["equal"][index]
        assert paired.mean() > 5 * paired.std(ddof=1) / np.sqrt(N)


def _expected_max_of_normals(k: int) -> float:
    density = lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)
    return integrate.quad(density, -np.inf, np.inf)[0]


@pytest.mark.parametrize("estimation_sd, helps", [(0.3, True), (0.9, False)])
def test_random_offsets_act_as_extra_validation_noise(estimation_sd: float, helps: bool) -> None:
    # Remark 1: with offsets drawn as N(beta_bar, s_beta^2) and estimation errors N(0, s_e^2),
    # the expected test error is E[gamma] - lambda^2 e_K / sqrt(lambda^2 + v), with
    # v = sigma^2 + s_beta^2 for validation-best and sigma^2 + s_e^2 for the corrected rule.
    rng = np.random.default_rng(4)
    k, lam, sigma, beta_bar, s_beta = 5, 1.0, 0.5, 0.4, 0.6
    tests = rng.normal(0.0, 1.0, (N, 1)) + rng.normal(0.0, lam, (N, k))
    validation_noise = rng.normal(0.0, sigma, (N, k))
    offsets = rng.normal(beta_bar, s_beta, (N, k))
    estimates = offsets + rng.normal(0.0, estimation_sd, (N, k))
    validation = tests - offsets + validation_noise
    rows = np.arange(N)
    e_k = _expected_max_of_normals(k)
    expected = lambda v: -(lam**2) * e_k / np.sqrt(lam**2 + v)
    best = tests[rows, validation.argmin(axis=1)]
    corrected = tests[rows, (validation + estimates).argmin(axis=1)]
    assert _close(best, expected(sigma**2 + s_beta**2))
    assert _close(corrected, expected(sigma**2 + estimation_sd**2))
    assert (corrected.mean() < best.mean()) == helps


@pytest.mark.parametrize("estimated_gap, helps", [(0.3, True), (1.3, False)])
def test_two_candidate_correction_helps_exactly_when_its_error_is_small(estimated_gap: float, helps: bool) -> None:
    # Proposition 2(c): with true gap c = 0.6, the corrected rule selects a when
    # D + E < c - c_hat, and it beats validation-best if and only if |c_hat - c| < |c|.
    t_a, t_b, noise = _two_candidates(seed=5)
    gap = 0.6
    best = _selected_error(t_a, t_b, noise, gap)
    corrected = _selected_error(t_a, t_b, noise, gap - estimated_gap)
    assert _close(corrected, -(TAU**2 / S) * stats.norm.pdf((gap - estimated_gap) / S))
    paired = corrected - best
    assert (paired.mean() < 0) == helps
    assert abs(paired.mean()) > 5 * paired.std(ddof=1) / np.sqrt(N)


def _two_point_expected_error(offset_gap: float) -> float:
    # Test errors 0 and 0.9 in random order, noise uniform on {-1/2, 1/2}: enumerate all cases.
    total = 0.0
    for order, noise_a, noise_b in itertools.product([(0.0, 0.9), (0.9, 0.0)], [-0.5, 0.5], [-0.5, 0.5]):
        validation_a = order[0] - offset_gap + noise_a
        validation_b = order[1] + noise_b
        assert validation_a != validation_b
        total += (order[0] if validation_a < validation_b else order[1]) / 8
    return total


def test_two_point_noise_example() -> None:
    # The example in the proof of Proposition 1(iii): with this noise, an offset difference
    # strictly between 0.1 and 0.9 lowers the expected test error from 0.225 to 0.1125.
    assert _two_point_expected_error(0.0) == pytest.approx(0.225)
    for gap in [0.15, 0.5, 0.85, -0.5]:
        assert _two_point_expected_error(gap) == pytest.approx(0.1125)
