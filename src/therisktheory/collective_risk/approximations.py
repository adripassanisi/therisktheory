"""Analytical approximations for compound aggregate loss distributions.

Topic 3: Estimation of aggregate loss distribution.
Author: Adrianna Passanisi

References:
    - Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
      Modern Actuarial Risk Theory: Using R. 2nd ed. Springer.
      Section 3.8: Approximations for compound distributions, pp. 56-60.
    - Daykin, C., Pentikäinen, T., and Pesonen, E. (1994).
      Practical Risk Theory for Actuaries. Chapman & Hall.
      Section 4.3: The NP method, pp. 110-117.
    - Beard, R., Pentikäinen, T., and Pesonen, E. (1984).
      Risk Theory. 3rd ed. Chapman & Hall. Section 3.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt

_SQRT_2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _std_normal_cdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(z / _SQRT_2))


def _std_normal_pdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate standard normal probability density function."""
    return np.exp(-0.5 * z**2) / _SQRT_2PI


def normal_approximation(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    variance: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Approximate aggregate loss CDF using the standard Normal distribution.

    Evaluates:
        F_S(x) \approx \Phi\left(\frac{x - \mu}{\sigma}\right)

    Args:
        x: Real loss level(s) at which to evaluate cumulative probability.
        mean: Expected aggregate loss E[S] >= 0.
        variance: Variance of aggregate loss Var(S) >= 0.

    Returns:
        Cumulative probability array in [0.0, 1.0].

    Raises:
        TypeError: If inputs cannot be converted to real numbers.
        ValueError: If variance is negative or shapes cannot broadcast.
    """
    x_arr = _validation.as_real_array(x, "x")
    mu_arr = _validation.as_nonnegative_finite_array(mean, "mean")
    var_arr = _validation.as_nonnegative_finite_array(variance, "variance")

    _validation.check_broadcastable(x=x_arr, mean=mu_arr, variance=var_arr)
    x_b, mu_b, var_b = np.broadcast_arrays(x_arr, mu_arr, var_arr)

    result = np.zeros_like(x_b, dtype=np.float64)
    zero_var = var_b == 0.0

    # Deterministic limit when variance is zero
    result[zero_var & (x_b >= mu_b)] = 1.0
    result[zero_var & (x_b < mu_b)] = 0.0

    valid = ~zero_var
    if np.any(valid):
        sigma = np.sqrt(var_b[valid])
        z = (x_b[valid] - mu_b[valid]) / sigma
        result[valid] = np.clip(_std_normal_cdf(z), 0.0, 1.0)

    return _validation.check_finite_result(result, "normal_approximation")


def edgeworth_approximation(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    variance: npt.ArrayLike,
    skewness: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Approximate aggregate loss CDF using first-order Edgeworth expansion.

    Evaluates:
        F_S(x) \approx \Phi(y) - \frac{\gamma_1}{6}(y^2 - 1)\phi(y)
    where y = (x - \mu) / \sigma and \gamma_1 is the skewness coefficient.

    Reference:
        Kaas et al. (2008), Section 3.8.2, eq. (3.54), p. 58.

    Args:
        x: Loss level(s) to evaluate.
        mean: Expected aggregate loss E[S] >= 0.
        variance: Variance of aggregate loss Var(S) >= 0.
        skewness: Skewness coefficient gamma_1 = E[(S - mu)^3] / sigma^3.

    Returns:
        Approximated cumulative probabilities clipped to [0.0, 1.0].
    """
    x_arr = _validation.as_real_array(x, "x")
    mu_arr = _validation.as_nonnegative_finite_array(mean, "mean")
    var_arr = _validation.as_nonnegative_finite_array(variance, "variance")
    skew_arr = _validation.as_real_array(skewness, "skewness")

    _validation.check_broadcastable(
        x=x_arr, mean=mu_arr, variance=var_arr, skewness=skew_arr
    )
    x_b, mu_b, var_b, skew_b = np.broadcast_arrays(
        x_arr, mu_arr, var_arr, skew_arr
    )

    result = np.zeros_like(x_b, dtype=np.float64)
    zero_var = var_b == 0.0

    result[zero_var & (x_b >= mu_b)] = 1.0
    result[zero_var & (x_b < mu_b)] = 0.0

    valid = ~zero_var
    if np.any(valid):
        sigma = np.sqrt(var_b[valid])
        y = (x_b[valid] - mu_b[valid]) / sigma
        phi_y = _std_normal_pdf(y)
        cdf_y = _std_normal_cdf(y)

        correction = (skew_b[valid] / 6.0) * (y**2 - 1.0) * phi_y
        prob = cdf_y - correction
        result[valid] = np.clip(prob, 0.0, 1.0)

    return _validation.check_finite_result(result, "edgeworth_approximation")


def normal_power_approximation(
    x: npt.ArrayLike,
    mean: npt.ArrayLike,
    variance: npt.ArrayLike,
    skewness: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Approximate aggregate loss CDF using the Normal Power (NP) method.

    For positive skewness gamma_1 > 0, solves:
        y = z + \frac{\gamma_1}{6}(z^2 - 1)
    yielding:
        z = \sqrt{\frac{9}{\gamma_1^2} + 1 + \frac{6}{\gamma_1} y}
            - \frac{3}{\gamma_1}
    with F_S(x) \approx \Phi(z) for values in the valid parabolic domain.

    Reference:
        Kaas et al. (2008), Section 3.8.3, eq. (3.58), p. 59.

    Args:
        x: Loss level(s) to evaluate.
        mean: Expected aggregate loss E[S] >= 0.
        variance: Variance of aggregate loss Var(S) >= 0.
        skewness: Skewness coefficient gamma_1.

    Returns:
        Approximated cumulative probabilities in [0.0, 1.0].
    """
    x_arr = _validation.as_real_array(x, "x")
    mu_arr = _validation.as_nonnegative_finite_array(mean, "mean")
    var_arr = _validation.as_nonnegative_finite_array(variance, "variance")
    skew_arr = _validation.as_real_array(skewness, "skewness")

    _validation.check_broadcastable(
        x=x_arr, mean=mu_arr, variance=var_arr, skewness=skew_arr
    )
    x_b, mu_b, var_b, g1_b = np.broadcast_arrays(
        x_arr, mu_arr, var_arr, skew_arr
    )

    result = np.zeros_like(x_b, dtype=np.float64)
    zero_var = var_b == 0.0

    result[zero_var & (x_b >= mu_b)] = 1.0
    result[zero_var & (x_b < mu_b)] = 0.0

    valid = ~zero_var
    if np.any(valid):
        sigma = np.sqrt(var_b[valid])
        y = (x_b[valid] - mu_b[valid]) / sigma
        g1 = g1_b[valid]

        prob = np.zeros_like(y, dtype=np.float64)

        # For near-symmetric or zero skewness, NP reduces to Normal
        symmetric = np.abs(g1) < 1e-8
        prob[symmetric] = _std_normal_cdf(y[symmetric])

        skewed = ~symmetric
        if np.any(skewed):
            y_s = y[skewed]
            g1_s = g1[skewed]

            # Argument under square root from Kaas eq. (3.58)
            radicand = (9.0 / g1_s**2) + 1.0 + (6.0 * y_s / g1_s)

            # Inside valid parabolic support
            in_support = radicand >= 0.0
            z = np.zeros_like(y_s)
            z[in_support] = np.sqrt(radicand[in_support]) - (3.0 / g1_s[in_support])
            z[~in_support] = -np.inf

            prob[skewed] = _std_normal_cdf(z)

        result[valid] = np.clip(prob, 0.0, 1.0)

    return _validation.check_finite_result(
        result, "normal_power_approximation"
    )
