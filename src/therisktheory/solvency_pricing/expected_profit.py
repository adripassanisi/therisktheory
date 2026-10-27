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
    expenses: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    r"""Calculate the expected underwriting profit of an insurance portfolio.

    Under classical risk theory, expected underwriting profit is defined as
    the excess of earned premiums over expected aggregate claims and
    operational expenses:
        E[Pi] = Premium - E[Claims] - Expenses

    If premiums are loaded using the expected value principle with safety loading
    theta (Premium = (1 + theta) * E[Claims]) and expenses are zero:
        E[Pi] = theta * E[Claims]

    References:
        - Bowers, N. et al. (1997). Actuarial Mathematics. Chapter 1.
        - Daykin, C., Pentikäinen, T., Pesonen, E. (1994). Practical Risk Theory
          for Actuaries. Section 1.3.

    Args:
        premium: Total premium collected or loaded premium (P >= 0).
        expected_claims: Expected aggregate claims or pure risk losses (E[S] >= 0).
        expenses: Acquisition or administrative expenses (E >= 0). Defaults to 0.0.

    Returns:
        np.ndarray: Expected underwriting profit. Can be negative, indicating an
            expected technical underwriting deficit.

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If inputs are negative, infinite, or have incompatible shapes.
    """
    prem_arr = as_nonnegative_finite_array(premium, "premium")
    claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")
    exp_arr = as_nonnegative_finite_array(expenses, "expenses")

    check_broadcastable(
        premium=prem_arr,
        expected_claims=claims_arr,
        expenses=exp_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = prem_arr - claims_arr - exp_arr

    return check_finite_result(result, "expected_profit")
