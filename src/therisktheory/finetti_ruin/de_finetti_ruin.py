"""De Finetti ruin probability, theorem terms, and identity validation.

Topic 2: De Finetti, Games, and Ruin.
Author: Adrianna Passanisi

This module implements the ruin probability under De Finetti's discrete
risk model, extracts the analytical terms associated with the
Lundberg-De Finetti theorem, and validates the theoretical identity
connecting the adjustment coefficient, safety loading, and ruin bound.

References:
    - Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
      Modern Actuarial Risk Theory: Using R. 2nd ed. Springer.
      Section 4.5, pp. 98-99.
    - Moriconi, F. (1982). A Pricing Model in a Sensitive Insurance Market.
      ASTIN Bulletin, 13(2), 135-149.
    - Dubourdieu, J. (1952). Théorie Mathématique du Risque dans les
      Assurances de Répartition. Gauthier-Villars, Paris.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation
from therisktheory.finetti_ruin.de_finetti_safety import (
    de_finetti_safety_loading,
    safety_index,
)
from therisktheory.finetti_ruin.games import gambler_ruin_probability

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class DeFinettiTheoremTerms:
    """Analytical components of De Finetti's ruin theorem.

    Attributes:
        ruin_probability: Exact probability of eventual ruin psi(u).
        upper_bound: Exponential Lundberg-De Finetti bound exp(-R * u).
        safety_index: Adjustment coefficient R = ln(p / q).
        safety_loading: Expected surplus drift E[G] = 2p - 1.
        decay_factor: Exact ratio psi(u) / exp(-R * u) = exp(-R) = q / p.
    """

    ruin_probability: npt.NDArray[np.float64]
    upper_bound: npt.NDArray[np.float64]
    safety_index: npt.NDArray[np.float64]
    safety_loading: npt.NDArray[np.float64]
    decay_factor: npt.NDArray[np.float64]


def de_finetti_ruin_probability(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
    target_reserve: npt.ArrayLike = np.inf,
) -> npt.NDArray[np.float64]:
    """Calculate ruin probability under De Finetti's discrete reserve model.

    In De Finetti's unit-step model, the reserve evolves as:
        U(n) = u + sum_{i=1}^n G_i
    where G_i = +1 with probability p and G_i = -1 with probability q = 1 - p.
    Ruin occurs when U(n) < 0 (i.e. reaching -1). Starting from reserve u >= 0,
    this corresponds to a gambler's ruin process starting at z = u + 1 and
    stopping at 0.

    When target_reserve is finite (b), the process stops upon reaching b,
    corresponding to a gambler target of a = b + 1.

    Mathematical Conditions:
        - initial_reserve u is a non-negative integer (u >= 0).
        - win_probability p satisfies 0 <= p <= 1.
        - target_reserve b is either np.inf or an integer with b >= u.

    Args:
        initial_reserve: Starting reserve u >= 0 (in integer units of stake).
        win_probability: Probability p of positive unit increment.
        target_reserve: Optional upper barrier b >= u, or np.inf for
            unlimited horizon. Defaults to np.inf.

    Returns:
        The ruin probability as a float64 NumPy array in [0.0, 1.0].

    Raises:
        TypeError: If inputs cannot be converted to real numbers.
        ValueError: If u is negative, u or b are non-integers, or shapes
            cannot be broadcast together.

    Examples:
        >>> de_finetti_ruin_probability(2, 0.6)
        array(0.2962963)
        >>> de_finetti_ruin_probability(0, 0.5)
        array(1.)
    """
    u = _validation.as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    p = _validation.as_probability_array(win_probability, "win_probability")
    b = _validation.as_real_array(
        target_reserve, "target_reserve", allow_inf=True
    )

    _validation.check_broadcastable(
        initial_reserve=u,
        win_probability=p,
        target_reserve=b,
    )

    # Unit-step models require integer stake counts
    if np.any(u != np.floor(u)):
        raise ValueError("initial_reserve must be a whole number of units")

    finite_b = np.isfinite(b)
    if np.any(b[finite_b] != np.floor(b[finite_b])):
        raise ValueError("target_reserve must be a whole number or inf")

    if np.any(u > b):
        raise ValueError("initial_reserve cannot exceed target_reserve")

    # Mapping to gambler's ruin coordinates: z = u + 1, a = b + 1
    z = u + 1.0
    a = np.where(finite_b, b + 1.0, np.inf)

    return gambler_ruin_probability(
        initial_capital=z,
        target_capital=a,
        win_probability=p,
    )


def de_finetti_theorem_terms(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
) -> DeFinettiTheoremTerms:
    """Compute the analytical components of the Lundberg-De Finetti theorem.

    For an infinite horizon and favorable game (0.5 < p < 1), De Finetti's
    discrete ruin model satisfies:
        psi(u) = (q / p)^(u + 1) = exp(-R * (u + 1))
        psi(u) <= exp(-R * u)

    where:
        - R = ln(p / q) is the safety index (adjustment coefficient).
        - E[G] = 2p - 1 is the expected surplus drift (safety loading).
        - exp(-R) = q / p is the constant decay slack factor.

    Mathematical Conditions:
        - initial_reserve u is a non-negative integer (u >= 0).
        - win_probability p satisfies 0.5 < p < 1 (strictly favorable game).

    Args:
        initial_reserve: Starting reserve u >= 0 (integer units).
        win_probability: Winning probability p in (0.5, 1.0).

    Returns:
        A DeFinettiTheoremTerms dataclass containing the exact ruin
        probability, upper bound, safety index, safety loading, and ratio.

    Raises:
        TypeError: If inputs cannot be cast to real numbers.
        ValueError: If u is negative or non-integer, or if p is outside
            the strictly favorable domain (0.5, 1.0).
    """
    u = _validation.as_nonnegative_finite_array(
        initial_reserve, "initial_reserve"
    )
    p = _validation.as_probability_array(win_probability, "win_probability")

    _validation.check_broadcastable(initial_reserve=u, win_probability=p)

    if np.any(u != np.floor(u)):
        raise ValueError("initial_reserve must be a whole number of units")
    if np.any(p <= 0.5) or np.any(p >= 1.0):
        raise ValueError("win_probability must be strictly between 0.5 and 1.0")

    # Reuse safety_index and de_finetti_safety_loading from Garcia's PR
    r = safety_index(p)
    drift = de_finetti_safety_loading(p)

    # Exact ruin probability via Bracamonte's gambler ruin
    psi = de_finetti_ruin_probability(u, p, target_reserve=np.inf)

    # Lundberg-De Finetti exponential upper bound: exp(-R * u)
    bound = np.exp(-r * u)
    decay = np.exp(-r)

    return DeFinettiTheoremTerms(
        ruin_probability=psi,
        upper_bound=bound,
        safety_index=r,
        safety_loading=drift,
        decay_factor=decay,
    )


def validate_de_finetti_identity(
    initial_reserve: npt.ArrayLike,
    win_probability: npt.ArrayLike,
    *,
    atol: float = 1e-12,
) -> npt.NDArray[np.bool_]:
    """Validate mathematical consistency of De Finetti's ruin identity.

    Checks that the computed ruin probability satisfies both:
        1. Exact relation: psi(u) == exp(-R * (u + 1)) within tolerance atol.
        2. Lundberg inequality: psi(u) <= exp(-R * u) + atol.
        3. Fundamental equation: p * exp(-R) + q * exp(R) == 1 within atol.

    Args:
        initial_reserve: Starting reserve u >= 0.
        win_probability: Winning probability p in (0.5, 1.0).
        atol: Absolute numerical tolerance. Defaults to 1e-12.

    Returns:
        NumPy boolean array indicating whether the theoretical identity holds.

    Raises:
        TypeError: If inputs cannot be cast to real numbers.
        ValueError: If domain constraints are violated.
    """
    terms = de_finetti_theorem_terms(initial_reserve, win_probability)

    # 1. Exact formula equality
    expected_psi = np.exp(-terms.safety_index * (initial_reserve + 1.0))
    exact_match = np.isclose(terms.ruin_probability, expected_psi, atol=atol)

    # 2. Lundberg-De Finetti inequality
    bound_holds = terms.ruin_probability <= (terms.upper_bound + atol)

    # 3. Fundamental adjustment coefficient equation
    p_arr = _validation.as_probability_array(win_probability, "win_probability")
    q_arr = 1.0 - p_arr
    r = terms.safety_index
    equation_root = p_arr * np.exp(-r) + q_arr * np.exp(r)
    root_match = np.isclose(equation_root, 1.0, atol=atol)

    return exact_match & bound_holds & root_match
