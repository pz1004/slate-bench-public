"""Validation provenance measurements shared by the provenance script and the paper builder.

Three notions are kept apart (manuscript Section 5):

* documented provenance: how the artifact says each candidate's validation score was
  produced (default, tuned, tuned and ensembled on the same folds);
* error-space optimism: the mean of test error minus validation error in the metric
  itself, the beta_m of the propositions. Errors are normalised per unit by the range of
  the candidates' test errors so that units with different metrics can be averaged;
* rank calibration: mean test rank minus mean validation rank. It is relative: over a
  common candidate set with average ties it sums to zero in every unit.

``offset_snr`` is the spread-to-estimation-error diagnostic of Remark 1 for a set of
candidates, computed from a dataset-by-candidate matrix of either quantity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tabular_sota_benchmark.inference import ArtifactResampler


def top_tenth(method_table: pd.DataFrame) -> pd.DataFrame:
    """The tenth of candidates with the best mean validation rank (at least five)."""
    return method_table.nsmallest(max(5, int(round(0.1 * len(method_table)))), "val_rank")


def error_optimism(frame: pd.DataFrame, unit_columns: list[str]) -> pd.DataFrame:
    """Add per-unit normalised error-space optimism to a candidate frame.

    ``primary_metric`` is the test error and ``val_primary_metric`` the validation error
    (lower is better). ``err_optimism`` divides test minus validation error by the unit's
    range of candidate test errors (the error-regret scale of Section 3.3);
    ``err_optimism_median`` divides by the median candidate's distance from the best. A
    unit whose scale is zero gets NaN and is left out of the means.
    """
    result = frame.copy()
    grouped = result.groupby(unit_columns, sort=False)["primary_metric"]
    low = grouped.transform("min")
    span = grouped.transform("max") - low
    median_span = grouped.transform("median") - low
    gap = result["primary_metric"].astype(float) - result["val_primary_metric"].astype(float)
    result["err_optimism"] = gap / span.where(span > 0)
    result["err_optimism_median"] = gap / median_span.where(median_span > 0)
    return result


def _snr_statistics(matrix: np.ndarray, n_train: int) -> tuple[float, float, float]:
    """Spread of the candidates' offsets, error of a training estimate, and their ratio.

    ``matrix`` is datasets x candidates. Each row is centred first: a component shared by
    all candidates on a dataset changes no selection (Remark 1), so only the
    candidate-specific parts count, and centring also removes the covariance that the
    shared component induces between candidates. With N datasets and K candidates,

        beta_m   = mean over datasets of the centred values,
        sigma2   = K / (K - 1) * mean_m Var_d(centred values of m),
        spread^2 = Var_m(beta_m) - sigma2 / N        (floored at zero),
        se^2     = sigma2 / n_train.

    sigma2 estimates the per-candidate noise variance of one dataset's value (centring
    shrinks it by (K - 1) / K when the noise is independent across candidates), sigma2 / N
    is the expected inflation of Var_m(beta_m) by the noise of the means, and se is the
    standard error of an offset estimated from ``n_train`` datasets. The ratio
    spread / se exceeds one exactly when, in the Gaussian model of Remark 1, such an
    offset lowers the expected error of the selected candidate. Noise that is correlated
    within groups of candidates (for example one learner family) is not modelled.
    """
    values = np.asarray(matrix, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan"), float("nan")
    centred = values - values.mean(axis=1, keepdims=True)
    beta = centred.mean(axis=0)
    sigma2 = k / (k - 1) * float(centred.var(axis=0, ddof=1).mean())
    spread = math.sqrt(max(float(beta.var(ddof=1)) - sigma2 / n, 0.0))
    se = math.sqrt(sigma2 / float(n_train))
    return spread, se, (spread / se if se > 0 else float("nan"))


def offset_snr(
    matrix: pd.DataFrame,
    *,
    n_train: int,
    resampler: ArtifactResampler,
    level: float = 0.95,
) -> dict[str, Any]:
    """``_snr_statistics`` with a dataset-bootstrap interval for the ratio.

    ``matrix`` has one row per dataset (indexed by name) and one column per candidate. The
    candidate set is held fixed; datasets are resampled with the artifact's shared index
    matrix. A dataset with a missing value (a normalisation scale of zero in all its units)
    is left out, and the remaining datasets get their own derived resampler.
    """
    frame = matrix.copy()
    frame.index = frame.index.astype(str)
    complete = frame.dropna(axis=0, how="any")
    if len(complete) < len(frame):
        resampler = resampler.subset(complete.index, salt="offset_snr_complete_datasets")
    aligned = np.column_stack([resampler.align(complete[column]) for column in complete.columns])
    spread, se, ratio = _snr_statistics(aligned, n_train)
    ratios = np.array([_snr_statistics(aligned[row], n_train)[2] for row in resampler.indices])
    ratios = ratios[np.isfinite(ratios)]
    tail = (1.0 - float(level)) / 2.0
    low, high = (np.quantile(ratios, [tail, 1.0 - tail]) if ratios.size else (float("nan"), float("nan")))
    return {
        "n_datasets": int(aligned.shape[0]),
        "n_candidates": int(aligned.shape[1]),
        "n_train": int(n_train),
        "spread": spread,
        "se_train": se,
        "ratio": ratio,
        "ratio_ci_low": float(low),
        "ratio_ci_high": float(high),
    }
