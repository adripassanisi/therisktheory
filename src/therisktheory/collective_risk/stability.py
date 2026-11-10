"""Stability parameters, sensitivity, and safety loadings in collective risk.

Topic 5: Collective risk process, Lundberg, and reinsurance.
Author: Adrianna Passanisi

References:
    - Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer.
      Chapter 6: The Risk Process, Section 6.4: Stability criteria.
    - Beard, R., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
      3rd ed. Chapman & Hall. Section 3.2: The stability criterion.
    - Daykin, C., Pentikäinen, T., and Pesonen, E. (1994).
      Practical Risk Theory for Actuaries. Chapman & Hall.
      Section 3.2 & Section 8.3: Safety loading and reserve relationships.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from therisktheory._validation import (
    as_nonnegative_finite_array,
    as_probability_array,
    check_broadcastable,
    check_finite_result,
)

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class StabilitySensitivity:
    """Analytical sensitivities and elasticities of the stability parameter.

    Attributes:
        stability: Base stability parameter K = E[G] / sqrt(Var(G)).
        d_theta: Derivative w.r.t. safety loading dK/dtheta.
        d_lambda: Derivative w.r.t. claim frequency rate dK/dlambda.
        elasticity_theta: Constant elasticity w.r.t. safety loading (equals 1.0).
        elasticity_lambda: Constant elasticity w.r.t. frequency (equals 0.5).
    """

    stability: npt.NDArray[np.float64]
    d_theta: npt.NDArray[np.float64]
    d_lambda: npt.NDArray[np.float64]
    elasticity_theta: npt.NDArray[np.float64]
    elasticity_lambda: npt.NDArray[np.float64]


def stability_parameter(
    claim_rate: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
    safety_loading: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Compute the classical collective stability parameter.

    Defined as the ratio of expected annual result to standard deviation:
        K = \frac{E[G]}{\sigma(G)} = \frac{\theta \lambda \mu_1}{\sqrt{\lambda \mu_2}}
          = \theta \sqrt{\lambda} \frac{\mu_1}{\sqrt{\mu_2}}

    Args:
        claim_rate: Poisson claim frequency intensity lambda > 0.
        severity_mean: Expected individual claim severity mu_1 > 0.
        severity_second_moment: Second raw moment of severity mu_2 >= mu_1^2.
        safety_loading: Relative premium safety loading theta >= 0.

    Returns:
        Stability parameter K as a float64 NumPy array.

    Raises:
        TypeError: If inputs cannot be cast to real numbers.
        ValueError: If parameters are negative or mu_2 < mu_1^2.
    """
    lam = as_nonnegative_finite_array(claim_rate, "claim_rate")
    mu1 = as_nonnegative_finite_array(severity_mean, "severity_mean")
    mu2 = as_nonnegative_finite_array(
        severity_second_moment, "severity_second_moment"
    )
    theta = as_nonnegative_finite_array(safety_loading, "safety_loading")

    if np.any(lam <= 0.0) or np.any(mu1 <= 0.0):
        raise ValueError("claim_rate and severity_mean must be strictly positive.")

    check_broadcastable(
        claim_rate=lam,
        severity_mean=mu1,
        severity_second_moment=mu2,
        safety_loading=theta,
    )
    lam, mu1, mu2, theta = np.broadcast_arrays(lam, mu1, mu2, theta)

    if np.any(mu2 < mu1**2):
        raise ValueError("severity_second_moment must be at least severity_mean^2.")

    with np.errstate(over="ignore", invalid="ignore"):
        k = theta * np.sqrt(lam) * (mu1 / np.sqrt(mu2))

    return check_finite_result(k, "stability_parameter")


def stability_sensitivity(
    claim_rate: npt.ArrayLike,
    severity_mean: npt.ArrayLike,
    severity_second_moment: npt.ArrayLike,
    safety_loading: npt.ArrayLike,
) -> StabilitySensitivity:
    r"""Calculate analytical derivatives a
