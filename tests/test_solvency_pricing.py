"""Tests and boundary conditions for Passanisi's solvency and ruin functions."""

import numpy as np
import pytest

from therisktheory.solvency_pricing.expected_profit import expected_profit
from therisktheory.solvency_pricing.portfolio_equilibrium import portfolio_equilibrium
from therisktheory.solvency_pricing.finite_horizon_ruin_estimate import (
    finite_horizon_ruin_estimate,
)


# =========================================================================
# 1. Tests for expected_profit
# =========================================================================

def test_expected_profit_mathematical_identity():
    """Verify safety loading identity: E[Profit] = theta * E[Claims]."""
    e_claims = 5000.0
    theta = 0.15
    premium = (1.0 + theta) * e_claims
    profit = expected_profit(premium, e_claims)
    assert np.isclose(profit, theta * e_claims)


def test_expected_profit_allows_negative_result():
    """Boundary test: Claims exceeding premium results in technical deficit."""
    premium = 1000.0
    claims = 1800.0
    profit = expected_profit(premium, claims)
    assert profit == -800.0
    assert profit < 0.0


def test_expected_profit_with_expenses():
    """Verify operational expenses deduction."""
    profit = expected_profit(premium=2000.0, expected_claims=1200.0, expenses=300.0)
    assert profit == 500.0


# =========================================================================
# 2. Tests for portfolio_equilibrium (Addressing double counting & negatives)
# =========================================================================

def test_portfolio_equilibrium_negative_accumulated_profit():
    """Ensure accumulated technical losses (negative profit) are permitted."""
    u = 1500.0
    negative_profit = -400.0
    surplus = portfolio_equilibrium(initial_capital=u, accumulated_profit=negative_profit)
    assert surplus == 1100.0


def test_portfolio_equilibrium_insolvency_balance():
    """Deficit test: Severe loss depletes capital below zero."""
    u = 500.0
    negative_profit = -800.0
    surplus = portfolio_equilibrium(initial_capital=u, accumulated_profit=negative_profit)
    assert surplus == -300.0


def test_portfolio_equilibrium_no_double_deduction():
    """Verify that net profit does not discount claims a second time by default."""
    u = 2000.0
    net_profit = 350.0  # Already net of claims
    surplus = portfolio_equilibrium(initial_capital=u, accumulated_profit=net_profit)
    assert surplus == 2350.0


def test_portfolio_equilibrium_break_even():
    """Boundary test: Zero initial capital and zero net profit."""
    surplus = portfolio_equilibrium(initial_capital=0.0, accumulated_profit=0.0)
    assert surplus == 0.0


# =========================================================================
# 3. Tests for finite_horizon_ruin_estimate (Addressing the professor's core critique)
# =========================================================================

def test_finite_ruin_sensitivity_to_claim_intensity():
    """Mathematical property: Ruin probability MUST increase with higher frequency (lambda).
    
    This directly proves the fix to Prof. Pitta's observation that two portfolios
    with different claim intensities had identical ruin estimates.
    """
    u = 50.0
    c = 15.0
    mean_x = 1.0
    t = 10.0

    psi_low_lambda = finite_horizon_ruin_estimate(
        initial_capital=u,
        premium_rate=c,
        claim_intensity=5.0,
        mean_claim_size=mean_x,
        time_horizon=t,
    )
    psi_high_lambda = finite_horizon_ruin_estimate(
        initial_capital=u,
        premium_rate=c,
        claim_intensity=12.0,
        mean_claim_size=mean_x,
        time_horizon=t,
    )

    assert psi_high_lambda > psi_low_lambda, (
        f"Higher lambda must produce higher ruin probability: {psi_high_lambda} <= {psi_low_lambda}"
    )


def test_finite_ruin_boundary_zero_capital():
    """Boundary case: u = 0 implies immediate ruin (psi = 1.0) for positive time."""
    psi = finite_horizon_ruin_estimate(
        initial_capital=0.0,
        premium_rate=10.0,
        claim_intensity=2.0,
        mean_claim_size=3.0,
        time_horizon=5.0,
    )
    assert np.isclose(psi, 1.0)


def test_finite_ruin_boundary_zero_time():
    """Boundary case: T = 0 with u > 0 implies zero probability of ruin."""
    psi = finite_horizon_ruin_estimate(
        initial_capital=100.0,
        premium_rate=10.0,
        claim_intensity=2.0,
        mean_claim_size=3.0,
        time_horizon=0.0,
    )
    assert np.isclose(psi, 0.0)


def test_finite_ruin_vectorization():
    """Test multi-dimensional array support (NumPy vectorization standard)."""
    u_vec = np.array([10.0, 50.0, 100.0])
    psi = finite_horizon_ruin_estimate(
        initial_capital=u_vec,
        premium_rate=20.0,
        claim_intensity=2.0,
        mean_claim_size=5.0,
        time_horizon=5.0,
    )
    assert psi.shape == (3,)
    # Ruin probability must decrease as capital increases (monotonicity)
    assert psi[0] >= psi[1] >= psi[2]
