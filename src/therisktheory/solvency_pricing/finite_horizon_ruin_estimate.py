""""Finite horizon ruin estimate module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def finite_horizon_ruin_estimate(
    initial_capital: ArrayLike,
    premium_rate: ArrayLike,
    claim_intensity: ArrayLike,
    mean_claim_size: ArrayLike,
    std_claim_size: ArrayLike,
    time_horizon: ArrayLike,
) -> NDArray[np.float64]:
    r"""Estimate the probability of ruin over a finite time horizon.

    Approximates the finite-horizon ruin probability :math:`\psi(u, T)` of the
    classical Cramér-Lundberg risk process using the Brownian diffusion approximation
    (Grandell, 1991; Asmussen, 2000).

    The surplus process :math:`U(t) = u + ct - S(t)` is approximated by a Brownian
    motion with drift:
        :math:`dU(t) = \mu dt + \sigma dW(t)`
    where:
        - Net drift: :math:`\mu = c - \lambda \mathbb{E}[X]`
        - Volatility: :math:`\sigma^2 = \lambda \mathbb{E}[X^2] = \lambda (\text{Var}(X) + \mathbb{E}[X]^2)`

    The first passage time distribution to zero yields the approximation:
        :math:`\psi(u, T) \approx \Phi\left(\frac{-u - \mu T}{\sigma \sqrt{T}}\right)
        + \exp\left(-\frac{2\mu u}{\sigma^2}\right) \Phi\left(\frac{-u + \mu T}{\sigma \sqrt{T}}\right)`

    Args:
        initial_capital: Starting surplus capital (:math:`u \ge 0`).
        premium_rate: Rate of premium collection per unit time (:math:`c \ge 0`).
        claim_intensity: Expected claim frequency parameter (:math:`\lambda \ge 0`).
        mean_claim_size: Expected value of individual claim amounts (:math:`\mathbb{E}[X] \ge 0`).
        std_claim_size: Standard deviation of individual claim amounts (:math:`\sigma_X \ge 0`).
        time_horizon: Finite time horizon duration (:math:`T \ge 0`).

    Returns:
        np.ndarray: The estimated finite-horizon ruin probability in [0, 1].

    Raises:
        TypeError: If inputs are not real numbers.
        ValueError: If inputs are negative, infinite, or have incompatible shapes.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prem_arr = as_nonnegative_finite_array(premium_rate, "premium_rate")
    lam_arr = as_nonnegative_finite_array(claim_intensity, "claim_intensity")
    mu_arr = as_nonnegative_finite_array(mean_claim_size, "mean_claim_size")
    std_arr = as_nonnegative_finite_array(std_claim_size, "std_claim_size")
    time_arr = as_nonnegative_finite_array(time_horizon, "time_horizon")

    check_broadcastable(
        initial_capital=cap_arr,
        premium_rate=prem_arr,
        claim_intensity=lam_arr,
        mean_claim_size=mu_arr,
        std_claim_size=std_arr,
        time_horizon=time_arr,
    )

    # Net drift rate per unit time: mu = c - lambda * E[X]
    drift = prem_arr - (lam_arr * mu_arr)

    # Second moment: E[X^2] = Var(X) + (E[X])^2
    second_moment = (std_arr ** 2) + (mu_arr ** 2)
    # Infinitesimal variance rate: sigma^2 = lambda * E[X^2]
    variance_rate = lam_arr * second_moment
    volatility = np.sqrt(variance_rate)

    # Initialize results
    result = np.zeros(np.broadcast_shapes(cap_arr.shape, time_arr.shape, drift.shape), dtype=np.float64)

    # Edge case 1: T == 0 (no time elapsed)
    mask_t_zero = time_arr == 0.0
    # If T == 0, ruin has not occurred if u >= 0
    result[mask_t_zero] = 0.0

    # Edge case 2: lambda == 0 or sigma == 0 (deterministic surplus process)
    mask_no_vol = (volatility <= 1e-12) & (~mask_t_zero)
    # Ruin occurs if u + c*t < 0; since u >= 0 and c >= 0, ruin probability is 0
    result[mask_no_vol] = 0.0

    # General case: positive time and positive volatility
    mask_eval = (~mask_t_zero) & (~mask_no_vol)

    if np.any(mask_eval):
        u_sub = cap_arr if cap_arr.ndim == 0 else cap_arr[mask_eval]
        t_sub = time_arr if time_arr.ndim == 0 else time_arr[mask_eval]
        drift_sub = drift if drift.ndim == 0 else drift[mask_eval]
        vol_sub = volatility if volatility.ndim == 0 else volatility[mask_eval]
        var_sub = variance_rate if variance_rate.ndim == 0 else variance_rate[mask_eval]

        denom = vol_sub * np.sqrt(t_sub)

        arg1 = (-u_sub - (drift_sub * t_sub)) / denom
        arg2 = (-u_sub + (drift_sub * t_sub)) / denom

        phi1 = norm.cdf(arg1)
        phi2 = norm.cdf(arg2)

        # Correct formula: exponent has a minus sign (-2 * u * drift / sigma^2)
        exp_power = np.clip(-2.0 * u_sub * drift_sub / var_sub, -700.0, 700.0)
        exp_factor = np.exp(exp_power)

        calc_result = phi1 + (exp_factor * phi2)
        result[mask_eval] = np.clip(calc_result, 0.0, 1.0)

    return check_finite_result(result, "finite_horizon_ruin_estimate")
