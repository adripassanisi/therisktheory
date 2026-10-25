"""Finite horizon ruin estimate module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def finite_horizon_ruin_estimate(
    initial_capital: ArrayLike,
    premium_rate: ArrayLike,
    claim_intensity: ArrayLike,
    time_horizon: ArrayLike,
) -> NDArray[np.float64]:
    """Estimate the probability of ruin over a finite time horizon.

    Calculates an approximation of the finite-horizon ruin probability based on 
    initial capital, premium income rate, claim intensity, and time horizon.

    Args:
        initial_capital: Starting surplus capital.
        premium_rate: Rate of premium collection per unit time.
        claim_intensity: Expected claim frequency parameter.
        time_horizon: Finite time horizon duration.

    Returns:
        np.ndarray: The estimated finite-horizon ruin probability.

    Raises:
        TypeError: If inputs are not real numbers.
        ValueError: If inputs are negative, infinite, or have incompatible shapes.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prem_arr = as_nonnegative_finite_array(premium_rate, "premium_rate")
    lam_arr = as_nonnegative_finite_array(claim_intensity, "claim_intensity")
    time_arr = as_nonnegative_finite_array(time_horizon, "time_horizon")

    check_broadcastable(
        initial_capital=cap_arr,
        premium_rate=prem_arr,
        claim_intensity=lam_arr,
        time_horizon=time_arr,
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = np.maximum(prem_arr * time_arr, 1e-8)
        result = np.exp(-cap_arr / denominator)
        result = np.clip(result, 0.0, 1.0)

    return check_finite_result(result, "finite_horizon_ruin_estimate")
