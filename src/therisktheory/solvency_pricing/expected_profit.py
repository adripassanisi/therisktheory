"""Expected profit module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def expected_profit(
    premium: ArrayLike,
    expected_claims: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate the expected profit of an insurance portfolio.

    The expected profit is defined as the difference between the total 
    earned premium and the expected aggregate claims (losses).

    Args:
        premium: Total premium collected or loaded premium.
        expected_claims: Expected aggregate claims or losses.

    Returns:
        np.ndarray: The resulting expected profit.

    Raises:
        TypeError: If inputs are not real numbers.
        ValueError: If inputs are negative, infinite, or have incompatible shapes.
    """
    prem_arr = as_nonnegative_finite_array(premium, "premium")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")
    check_broadcastable(premium=prem_arr, expected_claims=claims_arr)

    with np.errstate(over="ignore"):
        result = prem_arr - claims_arr

    return check_finite_result(result, "expected_profit")
