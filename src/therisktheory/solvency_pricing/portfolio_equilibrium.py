"""Portfolio equilibrium module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def portfolio_equilibrium(
    initial_capital: ArrayLike,
    accumulated_profit: ArrayLike,
    expected_claims: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate the portfolio equilibrium value.

    The portfolio equilibrium evaluates the balance incorporating initial capital,
    accumulated profits, and expected claims.

    Args:
        initial_capital: Starting capital of the insurance portfolio.
        accumulated_profit: Accumulated profits generated over the period.
        expected_claims: Expected aggregate claims or losses.

    Returns:
        np.ndarray: The resulting portfolio equilibrium measure.

    Raises:
        TypeError: If inputs are not real numbers.
        ValueError: If inputs are negative, infinite, or have incompatible shapes.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prof_arr = as_nonnegative_finite_array(accumulated_profit, "accumulated_profit")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")
    
    check_broadcastable(
        initial_capital=cap_arr,
        accumulated_profit=prof_arr,
        expected_claims=claims_arr,
    )

    with np.errstate(over="ignore"):
        result = cap_arr + prof_arr - claims_arr

    return check_finite_result(result, "portfolio_equilibrium")
