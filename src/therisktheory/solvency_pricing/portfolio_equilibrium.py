"""Portfolio equilibrium module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def _as_finite_array(val: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate that input consists of real, finite numbers (allows negatives)."""
    try:
        arr = np.asarray(val, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError(f"'{name}' must be convertible to a numeric array.") from err

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must contain only finite numbers.")
    return arr


def portfolio_equilibrium(
    initial_capital: ArrayLike,
    accumulated_profit: ArrayLike,
    expected_claims: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    r"""Calculate the portfolio equilibrium surplus balance.

    Evaluates the net technical surplus incorporating initial capital reserves,
    cumulative underwriting performance, and outstanding/expected losses:
        Surplus = initial_capital + accumulated_profit - expected_claims

    Double Counting Precaution:
        If `accumulated_profit` already represents net underwriting profit
        (i.e., premiums minus incurred claims), set `expected_claims = 0.0`
        (the default) to prevent subtracting claim liabilities twice.

    Negative Profits:
        `accumulated_profit` explicitly accepts negative values to reflect
        technical deficits or cumulative underwriting losses.

    References:
        - Daykin, C., Pentikäinen, T., Pesonen, E. (1994). Practical Risk Theory
          for Actuaries. Section 1.4.
        - Beard, R., Pentikäinen, T., Pesonen, E. (1984). Risk Theory. Chapter 2.

    Args:
        initial_capital: Initial solvency margin or surplus reserve (u >= 0).
        accumulated_profit: Cumulative underwriting result over the horizon.
            Can be negative.
        expected_claims: Future or unallocated claims to be deducted (S >= 0).
            Defaults to 0.0.

    Returns:
        np.ndarray: Net surplus balance. A negative value indicates insolvency.

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If `initial_capital` or `expected_claims` are negative or infinite.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prof_arr = _as_finite_array(accumulated_profit, "accumulated_profit")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")

    check_broadcastable(
        initial_capital=cap_arr,
        accumulated_profit=prof_arr,
        expected_claims=claims_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = cap_arr + prof_arr - claims_arr

    return check_finite_result(result, "portfolio_equilibrium")
