"""Shared dataset-level inference for the score-level audit.

One declared rule is applied to every contrast (manuscript Section 3.5). Every statistic
targets the mean dataset-level paired difference except the two distributional summaries:

* estimate: the mean dataset-level paired difference with a 95% percentile bootstrap
  interval. Every interval for one artifact uses the same resample index matrix, so a
  contrast has one interval wherever it is reported;
* separation: the two-sided bootstrap test that inverts that percentile interval, so a
  contrast's unadjusted test and its interval never disagree, Holm-adjusted within its
  declared family. A paired t-test of the mean is reported alongside;
* distribution: a two-sided Wilcoxon signed-rank test (symmetry of the differences about
  zero, i.e. the typical dataset), Holm-adjusted within the same family;
* breadth: an exact sign test with win/tie/loss counts (the win probability);
* equivalence: two one-sided t tests at alpha = 0.05 (the 90% t interval inside the
  margin), plus the smallest margin at which equivalence would hold; not adjusted for
  multiplicity;
* resolution: bootstrap and parametric minimum detectable effects, and the number of
  datasets needed to resolve an effect of a given number of rank positions.

The module also provides search-aware tools: the model confidence set (Hansen, Lunde &
Nason 2011), the superior predictive ability test (Hansen 2005) with White's (2000)
reality check, the probability of backtest overfitting by combinatorially symmetric
cross-validation (Bailey et al. 2017), and the gap-closed ratio between the single best
solver and the oracle.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_BOOTSTRAP_SEED = 20260608
DEFAULT_RESAMPLES = 10_000
DEFAULT_MARGINS = (0.005, 0.010, 0.015, 0.020)
TIE_TOLERANCE = 1e-12
# (raw, Holm-adjusted) p-value columns of a contrast table, each adjusted separately.
HOLM_COLUMNS = (
    ("wilcoxon_p", "wilcoxon_holm_p"),
    ("bootstrap_p", "bootstrap_holm_p"),
    ("ttest_p", "ttest_holm_p"),
)


def canonical_datasets(datasets: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in datasets})


def _derived_seed(seed: int, salt: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


@dataclass
class ArtifactResampler:
    """A fixed dataset-bootstrap index matrix for one artifact.

    ``indices[b, i]`` is the position (in canonical dataset order) of the i-th draw of
    resample b. Series passed to the methods are aligned by dataset name, so callers never
    depend on row order.
    """

    datasets: list[str]
    seed: int = DEFAULT_BOOTSTRAP_SEED
    n_resamples: int = DEFAULT_RESAMPLES

    def __post_init__(self) -> None:
        self.datasets = canonical_datasets(self.datasets)
        if not self.datasets:
            raise ValueError("A resampler needs at least one dataset.")
        rng = np.random.default_rng(int(self.seed))
        self.indices = rng.integers(0, len(self.datasets), size=(int(self.n_resamples), len(self.datasets)))
        self._position = {name: index for index, name in enumerate(self.datasets)}

    @property
    def n(self) -> int:
        return len(self.datasets)

    def align(self, values: pd.Series) -> np.ndarray:
        series = pd.Series(values)
        series.index = series.index.astype(str)
        missing = [name for name in self.datasets if name not in series.index]
        if missing:
            raise ValueError(f"Values missing for {len(missing)} dataset(s), e.g. {missing[:5]}")
        extra = sorted(set(series.index) - set(self.datasets))
        if extra:
            raise ValueError(f"Values for datasets outside the resampler: {extra[:5]}")
        return series.reindex(self.datasets).to_numpy(dtype=float)

    def subset(self, datasets: Iterable[object], *, salt: str) -> "ArtifactResampler":
        """A resampler for a declared subset (for example a task stratum)."""
        return ArtifactResampler(
            canonical_datasets(datasets),
            seed=_derived_seed(self.seed, salt),
            n_resamples=self.n_resamples,
        )

    def boot_means(self, values: pd.Series) -> np.ndarray:
        return self.align(values)[self.indices].mean(axis=1)

    def mean_ci(self, values: pd.Series, *, level: float = 0.95) -> tuple[float, float, float]:
        aligned = self.align(values)
        means = aligned[self.indices].mean(axis=1)
        low, high = _percentile_interval(means, level)
        return float(aligned.mean()), low, high

    def ratio_ci(
        self,
        numerator: pd.Series,
        denominator: pd.Series,
        *,
        level: float = 0.95,
    ) -> tuple[float, float, float]:
        num = self.align(numerator)
        den = self.align(denominator)
        estimate = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = num[self.indices].mean(axis=1) / den[self.indices].mean(axis=1)
        ratios = ratios[np.isfinite(ratios)]
        low, high = _percentile_interval(ratios, level)
        return estimate, low, high


def _percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(level)) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_mean_p(boot_means: np.ndarray) -> float:
    """Two-sided p-value of "mean difference = 0" that inverts the percentile interval.

    ``2 * min(P*(mean* <= 0), P*(mean* >= 0))`` over the resampled means: the (1 - a)
    percentile interval excludes zero exactly when this is below a, up to the
    granularity of the resample count.
    """
    boot_means = np.asarray(boot_means, dtype=float)
    if boot_means.size == 0:
        return float("nan")
    return float(min(1.0, 2.0 * min(np.mean(boot_means <= 0.0), np.mean(boot_means >= 0.0))))


def hodges_lehmann(values: np.ndarray) -> float:
    """Median of the Walsh averages: the location the signed-rank test is about."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    upper = np.triu_indices(values.size)
    return float(np.median((values[upper[0]] + values[upper[1]]) / 2.0))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order.

    A missing p-value would silently take the running maximum of the smaller ones, so it
    is refused.
    """
    values = [float(value) for value in p_values]
    if not values:
        return []
    if any(math.isnan(value) for value in values):
        raise ValueError("Holm adjustment needs every p-value; got NaN.")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_contrast(
    left: pd.Series,
    right: pd.Series,
    resampler: ArtifactResampler,
    *,
    margins: Sequence[float] = DEFAULT_MARGINS,
    rank_scale: float | None = None,
    alpha: float = 0.05,
    powers: Sequence[float] = (0.80, 0.90),
    effects_in_rank_positions: Sequence[float] = (0.5, 0.25),
    tolerance: float = TIE_TOLERANCE,
) -> dict[str, float | int]:
    """Every statistic of the declared decision rule for ``left - right``.

    Values are dataset-level means (lower is better), so a negative difference favours
    ``left``. ``rank_scale`` converts normalized regret to rank positions, i.e.
    ``|M| - 1`` for an artifact with ``|M|`` candidates.
    """
    diff = pd.Series(left).astype(float) - pd.Series(right).astype(float)
    diff = diff.dropna()
    values = resampler.align(diff)
    n = int(values.size)
    mean = float(values.mean())
    boot = values[resampler.indices].mean(axis=1)
    ci_low, ci_high = _percentile_interval(boot, 0.95)

    nonzero = values[np.abs(values) > tolerance]
    if nonzero.size:
        wilcoxon_p = float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue)
    else:
        wilcoxon_p = 1.0
    bootstrap_p = bootstrap_mean_p(boot) if nonzero.size else 1.0
    wins = int(np.sum(values < -tolerance))
    losses = int(np.sum(values > tolerance))
    ties = n - wins - losses
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0

    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    df = n - 1
    if not nonzero.size:
        ttest_p = 1.0
    elif n > 1 and se > 0:
        ttest_p = float(2.0 * stats.t.sf(abs(mean) / se, df))
    elif n > 1:
        ttest_p = 0.0  # every difference equal and non-zero
    else:
        ttest_p = float("nan")
    result: dict[str, float | int] = {
        "n_datasets": n,
        "mean_difference": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_p": bootstrap_p,
        "ttest_p": ttest_p,
        "median_difference": float(np.median(values)) if n else float("nan"),
        "hodges_lehmann_difference": hodges_lehmann(values),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sd_difference": sd,
        "se_difference": se,
    }
    if n > 1 and se > 0:
        t90 = float(stats.t.ppf(1.0 - alpha, df))
        tost_low = mean - t90 * se
        tost_high = mean + t90 * se
        result["tost_90_ci_low"] = tost_low
        result["tost_90_ci_high"] = tost_high
        result["smallest_equivalence_margin"] = float(max(abs(tost_low), abs(tost_high)))
        for margin in margins:
            p_lower = float(stats.t.sf((mean + float(margin)) / se, df))
            p_upper = float(stats.t.cdf((mean - float(margin)) / se, df))
            result[f"tost_p_{_margin_key(margin)}"] = max(p_lower, p_upper)
        se_boot = float(np.std(boot, ddof=1))
        result["se_bootstrap"] = se_boot
        for power in powers:
            z = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
            t = float(stats.t.ppf(1.0 - alpha / 2.0, df) + stats.t.ppf(power, df))
            key = f"{int(round(power * 100))}"
            mde_boot = z * se_boot
            result[f"mde{key}_bootstrap"] = mde_boot
            result[f"mde{key}_parametric"] = t * se
            if rank_scale:
                result[f"mde{key}_bootstrap_rank_positions"] = mde_boot * float(rank_scale)
                for effect in effects_in_rank_positions:
                    delta = float(effect) / float(rank_scale)
                    result[f"required_n{key}_{_effect_key(effect)}rank"] = int(
                        math.ceil(n * (mde_boot / delta) ** 2)
                    )
    return result


def _margin_key(margin: float) -> str:
    return f"{float(margin):.3f}".replace(".", "p")


def _effect_key(effect: float) -> str:
    return f"{float(effect):g}".replace(".", "p")


def contrast_table(
    rows: Sequence[dict[str, object]],
    *,
    family_column: str = "family",
) -> pd.DataFrame:
    """Stack contrast dictionaries and Holm-adjust each test's p-values within each family.

    The mean test (``bootstrap_p``), the paired t-test (``ttest_p``) and the signed-rank
    test (``wilcoxon_p``) are adjusted separately, each over the same family.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    for raw, adjusted in HOLM_COLUMNS:
        if raw not in frame.columns:
            continue
        frame[adjusted] = np.nan
        for _, group in frame.groupby(family_column, sort=False):
            frame.loc[group.index, adjusted] = holm(group[raw].tolist())
    return frame


# --------------------------------------------------------------------------- MCS


def model_confidence_set(
    losses: pd.DataFrame,
    resampler: ArtifactResampler,
    *,
    alphas: Sequence[float] = (0.10, 0.25),
) -> pd.DataFrame:
    """Model confidence set with the T_max statistic (Hansen, Lunde & Nason 2011).

    ``losses`` has one row per dataset and one column per policy (lower is better).
    Elimination continues to a single survivor so that every policy gets an MCS p-value;
    the (1 - alpha) set is the policies whose MCS p-value is at least alpha. The
    bootstrap is the iid dataset bootstrap on the artifact's shared index matrix.
    """
    matrix = np.column_stack([resampler.align(losses[column]) for column in losses.columns])
    names = [str(column) for column in losses.columns]
    means = matrix.mean(axis=0)
    boot = np.stack([matrix[row_indices].mean(axis=0) for row_indices in resampler.indices])
    alive = list(range(len(names)))
    eliminated: list[tuple[int, float]] = []
    running = 0.0
    while len(alive) > 1:
        current_mean = means[alive].mean()
        deviation = means[alive] - current_mean
        boot_deviation = boot[:, alive] - boot[:, alive].mean(axis=1, keepdims=True)
        centred = boot_deviation - deviation
        scale = np.sqrt(np.mean(centred**2, axis=0))
        scale = np.where(scale > 0, scale, np.inf)
        t_values = deviation / scale
        statistic = float(np.max(t_values))
        boot_statistic = np.max(centred / scale, axis=1)
        p_value = float(np.mean(boot_statistic >= statistic))
        running = max(running, p_value)
        worst = alive[int(np.argmax(t_values))]
        eliminated.append((worst, running))
        alive.remove(worst)
    p_values = {index: value for index, value in eliminated}
    p_values[alive[0]] = 1.0
    frame = pd.DataFrame(
        {
            "policy": names,
            "mean_loss": means,
            "mcs_p_value": [p_values[index] for index in range(len(names))],
        }
    )
    for alpha in alphas:
        frame[f"in_mcs_{int(round((1 - alpha) * 100))}"] = frame["mcs_p_value"] >= float(alpha)
    return frame.sort_values(["mcs_p_value", "mean_loss"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- SPA / RC


def superior_predictive_ability(
    losses: pd.DataFrame,
    benchmark: str,
    resampler: ArtifactResampler,
) -> dict[str, float]:
    """SPA test of "no alternative beats the benchmark" (Hansen 2005) and White's RC.

    d_k = L_benchmark - L_k, so positive values favour alternative k. The studentised
    statistic is T = max(max_k sqrt(n) mean(d_k) / omega_k, 0); the bootstrap recentres by
    g(x) = max(x, 0) (lower p-value), x * 1{x >= -omega * sqrt(2 log log n / n)}
    (consistent) or x (upper). The reality check is the non-studentised statistic
    max_k sqrt(n) mean(d_k) with full recentring.
    """
    if benchmark not in losses.columns:
        raise ValueError(f"Benchmark {benchmark!r} is not a column of the loss matrix.")
    alternatives = [column for column in losses.columns if column != benchmark]
    base = resampler.align(losses[benchmark])
    diffs = np.column_stack([base - resampler.align(losses[column]) for column in alternatives])
    n = diffs.shape[0]
    mean = diffs.mean(axis=0)
    boot = np.stack([diffs[row_indices].mean(axis=0) for row_indices in resampler.indices])
    omega = np.sqrt(n * np.mean((boot - mean) ** 2, axis=0))
    omega = np.where(omega > 0, omega, np.inf)
    statistic = float(max(np.max(np.sqrt(n) * mean / omega), 0.0))
    threshold = -omega * math.sqrt(2.0 * math.log(max(math.log(n), 1.0 + 1e-12)) / n)
    recentring = {
        "lower": np.maximum(mean, 0.0),
        "consistent": np.where(mean >= threshold, mean, 0.0),
        "upper": mean,
    }
    result = {
        "benchmark": benchmark,
        "n_alternatives": len(alternatives),
        "statistic": statistic,
        "best_alternative": str(alternatives[int(np.argmax(mean))]),
        "best_mean_improvement": float(np.max(mean)),
    }
    for name, g in recentring.items():
        boot_statistic = np.maximum(np.max(np.sqrt(n) * (boot - g) / omega, axis=1), 0.0)
        result[f"spa_p_{name}"] = float(np.mean(boot_statistic >= statistic))
    rc_statistic = float(np.max(np.sqrt(n) * mean))
    rc_boot = np.max(np.sqrt(n) * (boot - mean), axis=1)
    result["reality_check_statistic"] = rc_statistic
    result["reality_check_p"] = float(np.mean(rc_boot >= rc_statistic))
    return result


# --------------------------------------------------------------------------- PBO


def probability_of_backtest_overfitting(
    performance: pd.DataFrame,
    *,
    n_blocks: int = 10,
    strata: pd.Series | None = None,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """PBO by combinatorially symmetric cross-validation (Bailey et al. 2017).

    ``performance`` has one row per dataset and one column per policy, higher is better.
    Datasets are dealt into ``n_blocks`` blocks (stratified when ``strata`` is given) with
    a fixed seed; every half of the blocks serves once as the in-sample set. PBO is the
    share of splits in which the in-sample best policy ranks at or below the
    out-of-sample median.
    """
    if n_blocks % 2:
        raise ValueError("n_blocks must be even.")
    frame = performance.copy()
    frame.index = frame.index.astype(str)
    frame = frame.sort_index()
    rng = np.random.default_rng(seed)
    blocks = pd.Series(-1, index=frame.index, dtype=int)
    groups = (
        [frame.index.tolist()]
        if strata is None
        else [
            sorted(group.index.astype(str))
            for _, group in pd.Series(strata).astype(str).groupby(pd.Series(strata).astype(str))
        ]
    )
    offset = 0
    for members in groups:
        members = [name for name in members if name in frame.index]
        order = rng.permutation(len(members))
        for position, member_index in enumerate(order):
            blocks[members[member_index]] = (offset + position) % n_blocks
        offset += len(members)
    values = frame.to_numpy(dtype=float)
    block_values = blocks.to_numpy()
    n_policies = values.shape[1]
    logits = []
    degradations = []
    for in_blocks in itertools.combinations(range(n_blocks), n_blocks // 2):
        in_sample = np.isin(block_values, in_blocks)
        in_mean = values[in_sample].mean(axis=0)
        out_mean = values[~in_sample].mean(axis=0)
        best = int(np.argmax(in_mean))
        relative_rank = stats.rankdata(out_mean)[best] / (n_policies + 1.0)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
        degradations.append(float(out_mean[best] - in_mean[best]))
    logits_array = np.asarray(logits)
    return {
        "n_blocks": int(n_blocks),
        "n_combinations": int(logits_array.size),
        "n_policies": int(n_policies),
        "pbo": float(np.mean(logits_array <= 0.0)),
        "median_logit": float(np.median(logits_array)),
        "mean_performance_degradation": float(np.mean(degradations)),
    }


# --------------------------------------------------------------------------- gap closed


def gap_closed(
    policy: pd.Series,
    single_best: pd.Series,
    oracle: pd.Series,
    resampler: ArtifactResampler,
) -> tuple[float, float, float]:
    """Share of the single-best-to-oracle regret gap closed by ``policy`` (ASlib style)."""
    numerator = pd.Series(single_best).astype(float) - pd.Series(policy).astype(float)
    denominator = pd.Series(single_best).astype(float) - pd.Series(oracle).astype(float)
    return resampler.ratio_ci(numerator, denominator)
