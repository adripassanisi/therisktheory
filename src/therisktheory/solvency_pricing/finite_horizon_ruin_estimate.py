"""Finite horizon ruin estimate module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from therisktheory._validation import (
    as_nonnegative_finite_array,
    check_broadcastable,
    check_finite_result,
)


def _std_normal_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def finite_horizon_ruin_estimate(
    initial_capital: ArrayLike,
    premium_rate: ArrayLike,
    claim_intensity: ArrayLike,
    mean_claim_size: ArrayLike,
    time_horizon: ArrayLike,
    claim_size_variance: ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""Estimate finite-horizon ruin probability via diffusion approximation.

    Under the classical Cramér-Lundberg risk process:
        U(t) = u + c*t - \sum_{i=1}^{N(t)} X_i
    approximated as a Brownian motion with drift:
        drift (mu_D) = c - lambda * E[X]
        variance rate (sigma^2) = lambda * E[X^2] = lambda * (Var(X) + (E[X])^2)

    The probability of ruin before time T, psi(u, T) = P(inf_{0 <= t <= T} U(t) < 0),
    is approximated using the first hitting time distribution:
        psi(u, T) = Phi((-u - mu_D * T) / (sigma * sqrt(T)))
                  + exp(-2 * mu_D * u / sigma^2) * Phi((-u + mu_D * T) / (sigma * sqrt(T)))

    References:
        - Daykin, C., Pentikäinen, T., Pesonen, E. (1994). Practical Risk Theory
          for Actuaries. Section 3.4 & 8.2 (Diffusion Approximations).
        - Beard, R., Pentikäinen, T., Pesonen, E. (1984). Risk Theory. Section 3.6.

    Args:
        initial_capital: Starting surplus capital (u >= 0).
        premium_rate: Premium income collection rate per unit time (c >= 0).
        claim_intensity: Expected claim frequency / Poisson intensity (lambda >= 0).
        mean_claim_size: Expected individual claim severity (E[X] >= 0).
        time_horizon: Finite time window duration (T >= 0).
        claim_size_variance: Variance of claim severity Var(X). If None, assumes
            exponentially distributed claims where Var(X) = (mean_claim_size)^2.

    Returns:
        np.ndarray: Estimated ruin probability in [0.0, 1.0].

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If inputs are negative, infinite, or shapes cannot broadcast.
    """
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")
    prem_arr = as_nonnegative_finite_array(premium_rate, "premium_rate")
    lam_arr = as_nonnegative_finite_array(claim_intensity, "claim_intensity")
    mean_claim_arr = as_nonnegative_finite_array(mean_claim_size, "mean_claim_size")
    time_arr = as_nonnegative_finite_array(time_horizon, "time_horizon")

    if claim_size_variance is None:
        var_claim_arr = mean_claim_arr**2
    else:
        var_claim_arr = as_nonnegative_finite_array(claim_size_variance, "claim_size_variance")

    check_broadcastable(
        initial_capital=cap_arr,
        premium_rate=prem_arr,
        claim_intensity=lam_arr,
        mean_claim_size=mean_claim_arr,
        time_horizon=time_arr,
        claim_size_variance=var_claim_arr,
    )

    cap, c, lam, mu_x, t, var_x = np.broadcast_arrays(
        cap_arr, prem_arr, lam_arr, mean_claim_arr, time_arr, var_claim_arr
    )

    second_moment_x = var_x + mu_x**2
    drift = c - (lam * mu_x)
    sigma2 = lam * second_moment_x
    sigma = np.sqrt(np.maximum(sigma2, 1e-12))

    result = np.zeros_like(cap, dtype=np.float64)

    # Boundary conditions
    zero_time = t <= 0.0
    zero_cap = cap <= 0.0
    valid_mask = (~zero_time) & (~zero_cap)

    result[zero_cap & (~zero_time)] = 1.0
    result[zero_time] = 0.0

    if np.any(valid_mask):
        u_v = cap[valid_mask]
        t_v = t[valid_mask]
        mu_v = drift[valid_mask]
        sig_v = sigma[valid_mask]
        sig_sqrt_t = sig_v * np.sqrt(t_v)

        term1 = _std_normal_cdf((-u_v - mu_v * t_v) / sig_sqrt_t)

        exponent = -2.0 * mu_v * u_v / (sig_v**2)
        exponent = np.clip(exponent, -700.0, 700.0)

        cdf2 = _std_normal_cdf((-u_v + mu_v * t_v) / sig_sqrt_t)
        term2 = np.where(cdf2 == 0.0, 0.0, np.exp(exponent) * cdf2)

        prob = term1 + term2
        result[valid_mask] = np.clip(prob, 0.0, 1.0)

    return check_finite_result(result, "finite_horizon_ruin_estimate")
