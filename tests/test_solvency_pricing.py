"""Tests for solvency and pricing functions (Passanisi topic)."""

import numpy as np
import pytest

from therisktheory.solvency_pricing import (
    expected_profit,
    finite_horizon_ruin_estimate,
    portfolio_equilibrium,
)


def test_expected_profit():
    """Test expected profit calculation."""
    res = expected_profit(1200.0, 950.0)
    np.testing.assert_allclose(res, 250.0)


def test_portfolio_equilibrium():
    """Test portfolio equilibrium calculation."""
    res = portfolio_equilibrium(5000.0, 200.0, 950.0)
    np.testing.assert_allclose(res, 4250.0)


def test_finite_horizon_ruin_estimate():
    """Test finite horizon ruin estimate calculation."""
    res = finite_horizon_ruin_estimate(1000.0, 1200.0, 0.1, 5.0)
    assert 0.0 <= res <= 1.0


def test_invalid_inputs():
    """Test that invalid or negative inputs raise errors."""
    with pytest.raises(ValueError):
        expected_profit(-50.0, 100.0)
    with pytest.raises(TypeError):
        portfolio_equilibrium("invalid", 200.0, 100.0)
