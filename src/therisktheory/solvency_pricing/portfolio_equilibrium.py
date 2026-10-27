"""Portfolio equilibrium module for TheRiskTheory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from therisktheory._validation import (
    as_finite_array,               # Permite valores negativos (pérdidas)
    as_nonnegative_finite_array,   # Solo para magnitudes estrictamente no negativas
    check_broadcastable,
    check_finite_result,
)


def portfolio_equilibrium(
    initial_capital: ArrayLike,
    accumulated_profit: ArrayLike,
    expected_claims: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Calculate the portfolio equilibrium (expected net surplus balance).

    Mathematical Formulation:
        Let u be the initial capital (u >= 0).
        - If `accumulated_profit` represents gross underwriting margin / premium income,
          and `expected_claims` (S >= 0) are provided:
              Surplus = u + accumulated_profit - expected_claims
        - If `accumulated_profit` is already NET of incurred claims, `expected_claims`
          can be set to 0 (or omitted), avoiding double counting:
              Surplus = u + net_profit

    Note on Signs:
        `accumulated_profit` accepts negative values, reflecting technical deficits
        or underwriting losses experienced by the portfolio.

    Args:
        initial_capital: Starting capital / reserve of the portfolio (u >= 0).
        accumulated_profit: Cumulative underwriting result over the period.
            Can be negative in case of technical losses.
        expected_claims: Expected aggregate claims to be deducted.
            Default is 0.0 (if accumulated_profit is already net of claims).

    Returns:
        np.ndarray: Net portfolio equilibrium balance (surplus). Can be negative,
            indicating insolvency/ruin.

    Raises:
        TypeError: If inputs cannot be cast to floating-point numbers.
        ValueError: If `initial_capital` or `expected_claims` are negative,
            or if array shapes cannot be broadcast together.
    """
    # 1. Capital inicial: no negativo
    cap_arr = as_nonnegative_finite_array(initial_capital, "initial_capital")

    # 2. Beneficio: DEBE permitir negativos (pérdidas técnicas)
    prof_arr = as_finite_array(accumulated_profit, "accumulated_profit")

    # 3. Siniestros esperados (opcionales para evitar doble deducción)
    if expected_claims is None:
        claims_arr = np.zeros_like(prof_arr, dtype=np.float64)
    else:
        claims_arr = as_nonnegative_finite_array(expected_claims, "expected_claims")

    check_broadcastable(
        initial_capital=cap_arr,
        accumulated_profit=prof_arr,
        expected_claims=claims_arr,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        # Saldo = Capital Inicial + Resultado - Siniestros deducibles
        result = cap_arr + prof_arr - claims_arr

    return check_finite_result(result, "portfolio_equilibrium")
