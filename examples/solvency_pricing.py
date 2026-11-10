"""Example: Solvency, pricing, and finite-horizon ruin estimation.

Author: Adrianna Passanisi
Topic 1: Solvency, Pricing, Retention, and Ruin.
"""

import numpy as np

from therisktheory.solvency_pricing import (
    expected_profit,
    finite_horizon_ruin_estimate,
    portfolio_equilibrium,
)

# 1. Expected profit with safety loading
claims = 100_000.0
loading = 0.20
premium = (1.0 + loading) * claims
expenses = 5_000.0

profit = expected_profit(premium, claims, expenses)
print(f"Expected Underwriting Profit: {profit.item():,.2f}")

# 2. Portfolio equilibrium balance
initial_capital = 50_000.0
surplus = portfolio_equilibrium(
    initial_capital=initial_capital,
    accumulated_profit=profit,
)
print(f"Portfolio Surplus Balance: {surplus.item():,.2f}")

# 3. Finite-horizon ruin probability estimation
t_horizon = 5.0
c_rate = premium / t_horizon
lam = 50.0
mean_severity = claims / (lam * t_horizon)

psi = finite_horizon_ruin_estimate(
    initial_capital=initial_capital,
    premium_rate=c_rate,
    claim_intensity=lam,
    mean_claim_size=mean_severity,
    time_horizon=t_horizon,
)
print(f"5-Year Ruin Probability psi(u={initial_capital:,.0f}): {psi.item():.6f}")

# 4. Deterministic case (Prof. Pitta review scenario: lambda=0, c=0)
psi_deterministic = finite_horizon_ruin_estimate(
    initial_capital=1e-7,
    premium_rate=0.0,
    claim_intensity=0.0,
    mean_claim_size=100.0,
    time_horizon=1.0,
)
print(f"Deterministic zero-claim ruin probability: {psi_deterministic.item():.1f}")
