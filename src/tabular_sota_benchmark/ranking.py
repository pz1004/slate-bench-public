"""Candidate-relative ranking shared by every score-level analysis.

A selector output is ranked against the candidate rows of its evaluation unit by test
error. Values within ``atol + rtol * |v|`` of each other are ties and share the average
tie rank; every other candidate is either strictly better or strictly worse. The
``"legacy"`` rule reproduces the earlier convention, which counted a candidate that lies
strictly below ``v`` but within tolerance both as better and as tied, and which fixed
the oracle at rank 1. It exists only for the attribution runs of the
implementation-conventions appendix.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


RANK_ATOL = 1e-12
RANK_RTOL = 1e-12
RANK_RULES = ("consistent", "legacy")

_ACTIVE_RULE = "consistent"


def _validate_rule(rule: str) -> str:
    if rule not in RANK_RULES:
        raise ValueError(f"rank rule must be one of {RANK_RULES}, got {rule!r}")
    return rule


def get_rank_rule() -> str:
    return _ACTIVE_RULE


def set_rank_rule(rule: str) -> None:
    """Set the process-wide rule used when callers do not pass one explicitly."""
    global _ACTIVE_RULE
    _ACTIVE_RULE = _validate_rule(str(rule))


@contextmanager
def rank_rule(rule: str) -> Iterator[None]:
    previous = get_rank_rule()
    set_rank_rule(rule)
    try:
        yield
    finally:
        set_rank_rule(previous)


def _tolerance(values: np.ndarray, *, atol: float, rtol: float) -> np.ndarray:
    return atol + rtol * np.abs(values)


def candidate_relative_ranks(
    candidate_values: np.ndarray,
    values: np.ndarray,
    *,
    rule: str | None = None,
    atol: float = RANK_ATOL,
    rtol: float = RANK_RTOL,
) -> np.ndarray:
    """Average-tie rank (1 = best) of each value against one unit's candidates.

    Non-finite candidates are never counted as better or tied, matching v1.
    """
    rule = _validate_rule(rule or get_rank_rule())
    values = np.asarray(values, dtype=float)
    candidates = np.asarray(candidate_values, dtype=float)
    candidates = np.sort(candidates[np.isfinite(candidates)])
    tol = _tolerance(values, atol=atol, rtol=rtol)
    lower = np.searchsorted(candidates, values - tol, side="left")
    upper = np.searchsorted(candidates, values + tol, side="right")
    equal = (upper - lower).astype(float)
    if rule == "legacy":
        less = np.searchsorted(candidates, values, side="left").astype(float)
    else:
        less = lower.astype(float)
    ranks = 1.0 + less + np.where(equal > 0.0, (equal - 1.0) / 2.0, 0.0)
    if rule == "legacy":
        # v1 ranked a non-finite value as 1 + 0 + 0; keep that for exact reproduction.
        return ranks
    return np.where(np.isfinite(values), ranks, np.nan)


def within_unit_ranks(
    values: np.ndarray,
    *,
    rule: str | None = None,
    atol: float = RANK_ATOL,
    rtol: float = RANK_RTOL,
) -> np.ndarray:
    """Rank every candidate against its own unit (used for normalised test ranks).

    Under the consistent rule this equals the endpoint's candidate-relative rank, so
    priors and training objectives share the endpoint's tie convention. The legacy rule
    keeps pandas exact average ranks.
    """
    rule = _validate_rule(rule or get_rank_rule())
    values = np.asarray(values, dtype=float)
    if rule == "legacy":
        return pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype=float)
    return candidate_relative_ranks(values, values, rule="consistent", atol=atol, rtol=rtol)


def grouped_within_unit_ranks(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    value_column: str,
    rule: str | None = None,
    atol: float = RANK_ATOL,
    rtol: float = RANK_RTOL,
) -> np.ndarray:
    """``within_unit_ranks`` for every unit of ``frame``, aligned with its rows."""
    rule = _validate_rule(rule or get_rank_rule())
    if frame.empty:
        return np.zeros(0, dtype=float)
    if rule == "legacy":
        return (
            frame.groupby(list(unit_columns), sort=False)[value_column]
            .rank(method="average", ascending=True)
            .to_numpy(dtype=float)
        )
    values = frame[value_column].to_numpy(dtype=float)
    ranks = np.full(values.shape[0], np.nan, dtype=float)
    unit_codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(frame[list(unit_columns)].astype(str)), sort=False
    )
    order = np.argsort(unit_codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(unit_codes[order])) + 1
    for positions in np.split(order, boundaries):
        ranks[positions] = within_unit_ranks(
            values[positions], rule=rule, atol=atol, rtol=rtol
        )
    return ranks


def oracle_rank(
    candidate_values: np.ndarray,
    *,
    rule: str | None = None,
    atol: float = RANK_ATOL,
    rtol: float = RANK_RTOL,
) -> float:
    """Rank of the best candidate: 1 + (k - 1) / 2 when k candidates tie at the best."""
    rule = _validate_rule(rule or get_rank_rule())
    candidates = np.asarray(candidate_values, dtype=float)
    finite = candidates[np.isfinite(candidates)]
    if finite.size == 0:
        return float("nan")
    if rule == "legacy":
        return 1.0
    best = float(finite.min())
    return float(candidate_relative_ranks(finite, np.array([best]), rule=rule, atol=atol, rtol=rtol)[0])


def rank_against_candidates(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    value_column: str = "primary_metric",
    candidate_mask: pd.Series | np.ndarray | None = None,
    rule: str | None = None,
    atol: float = RANK_ATOL,
    rtol: float = RANK_RTOL,
) -> pd.DataFrame:
    """Rank every row of ``frame`` against its unit's candidate rows.

    Adds ``candidate_rank``, ``n_candidates``, ``tied_best``, ``rank_regret``,
    ``normalized_rank_regret`` and the unit's candidate ``min``/``median``/``max``
    errors (``candidate_min_error`` etc.). Rows of units without candidates keep NaN
    ranks, as in v1.
    """
    rule = _validate_rule(rule or get_rank_rule())
    result = frame.copy()
    if candidate_mask is None:
        is_candidate = result["method_type"].astype(str).eq("candidate").to_numpy()
    else:
        is_candidate = np.asarray(candidate_mask, dtype=bool)
    values = result[value_column].to_numpy(dtype=float)
    n_rows = int(result.shape[0])
    candidate_rank = np.full(n_rows, np.nan, dtype=float)
    n_candidates = np.zeros(n_rows, dtype=np.int64)
    tied_best = np.zeros(n_rows, dtype=float)
    min_error = np.full(n_rows, np.nan, dtype=float)
    median_error = np.full(n_rows, np.nan, dtype=float)
    max_error = np.full(n_rows, np.nan, dtype=float)

    if n_rows:
        unit_codes, _ = pd.factorize(
            pd.MultiIndex.from_frame(result[list(unit_columns)].astype(str)), sort=False
        )
        order = np.argsort(unit_codes, kind="stable")
        boundaries = np.flatnonzero(np.diff(unit_codes[order])) + 1
        for positions in np.split(order, boundaries):
            unit_values = values[positions]
            unit_candidates = unit_values[is_candidate[positions]]
            if unit_candidates.size == 0:
                continue
            finite = unit_candidates[np.isfinite(unit_candidates)]
            candidate_rank[positions] = candidate_relative_ranks(
                unit_candidates, unit_values, rule=rule, atol=atol, rtol=rtol
            )
            n_candidates[positions] = int(unit_candidates.size)
            if finite.size:
                best = float(finite.min())
                tied_best[positions] = np.isclose(
                    unit_values, best, rtol=rtol, atol=atol
                ).astype(float)
                min_error[positions] = best
                median_error[positions] = float(np.median(finite))
                max_error[positions] = float(finite.max())

    result["candidate_rank"] = candidate_rank
    result["n_candidates"] = n_candidates
    result["tied_best"] = tied_best
    result["candidate_min_error"] = min_error
    result["candidate_median_error"] = median_error
    result["candidate_max_error"] = max_error
    result["rank_regret"] = result["candidate_rank"] - 1.0
    denominator = result["n_candidates"].astype(float) - 1.0
    result["normalized_rank_regret"] = np.where(
        denominator > 0.0,
        result["rank_regret"] / denominator.where(denominator > 0.0, 1.0),
        0.0,
    )
    return result
