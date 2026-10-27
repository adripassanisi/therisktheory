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

    The expected underwriting profit (E[Pi]) is defined under classical risk
    theory as:
        E[Pi] = Premium - E[Claims] - Expenses

    If premiums are loaded with a relative safety loading theta such that
    Premium = (1 + theta) * E[Claims], then in the absence of expenses:
        E[Pi] = theta * E[Claims]

    Note:
        The resulting expected profit can be strictly negative when expected
        claims (and expenses) exceed premium income, indicating an expected
        technical underwriting loss (deficit).

    Args:
        premium: Total earned or loaded premium collected (P >= 0).
        expected_claims: Expected aggregate claims or losses (E[S] >= 0).
        expenses: Optional administrative/acquisition expenses (E >= 0).
            Defaults to 0.0.

    Returns:
        np.ndarray: The resulting expected underwriting profit. Can be negative.

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
