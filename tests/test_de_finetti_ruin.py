"""Unit tests and mathematical validation for De Finetti ruin functions."""

import numpy as np
import pytest

from therisktheory.de_finetti.reserves import reserve_fund
from therisktheory.finetti_ruin import (
    de_finetti_ruin_probability,
    de_finetti_theorem_terms,
    validate_de_finetti_identity,
)


def test_de_finetti_ruin_probability_analytical():
    """Verify analytical example from Kaas et al. (2008) Section 4.5.

    For p = 0.6, q = 0.4 and initial reserve u = 2:
    z = u + 1 = 3.
    psi(2) = (0.4 / 0.6)**3 = (2/3)**3 = 8 / 27 approx 0.2962963.
    """
    res = de_finetti_ruin_probability(initial_reserve=2, win_probability=0.6)
    expected = (0.4 / 0.6) ** 3
    assert np.isclose(res.item(), expected, atol=1e-12)


def test_de_finetti_ruin_boundary_fair_game():
    """For a fair game (p = 0.5), ruin is certain (psi = 1.0)."""
    res = de_finetti_ruin_probability(initial_reserve=5, win_probability=0.5)
    assert res.item() == 1.0


def test_de_finetti_ruin_boundary_zero_reserve():
    """At u = 0 with p = 0.75: psi(0) = q / p = 0.25 / 0.75 = 1/3."""
    res = de_finetti_ruin_probability(initial_reserve=0, win_probability=0.75)
    assert np.isclose(res.item(), 1.0 / 3.0, atol=1e-12)


def test_de_finetti_ruin_finite_target():
    """Verify finite barrier (target reserve)."""
    # u = 1, target = 3, p = 0.5 -> gambler ruin (2, 4, 0.5) = (4 - 2) / 4 = 0.5
    res = de_finetti_ruin_probability(
        initial_reserve=1, win_probability=0.5, target_reserve=3
    )
    assert np.isclose(res.item(), 0.5)


def test_de_finetti_theorem_terms_and_bound():
    """Verify the Lundberg-De Finetti inequality psi(u) <= exp(-R * u)."""
    terms = de_finetti_theorem_terms(initial_reserve=3, win_probability=0.7)
    assert terms.ruin_probability <= terms.upper_bound
    # Ratio must be exp(-R) = q / p
    expected_ratio = (1.0 - 0.7) / 0.7
    assert np.isclose(terms.decay_factor.item(), expected_ratio, atol=1e-12)


def test_validate_de_finetti_identity_vectorized():
    """Verify identity holds across array inputs."""
    reserves = np.array([0, 1, 2, 5, 10])
    probs = np.array([0.55, 0.60, 0.70, 0.80, 0.90])
    valid = validate_de_finetti_identity(reserves, probs)
    assert np.all(valid)


def test_cross_integration_with_monticelli_reserves():
    """Integration check: simulate trajectory and compare with Monticelli's reserve_fund."""
    u0 = 5.0
    # Premiums = 1 each round, claims = 2 when lost, 0 when won
    premiums = np.ones(4)
    claims = np.array([0.0, 2.0, 0.0, 2.0])  # net gains: +1, -1, +1, -1
    fund = reserve_fund(u0, premiums, claims)
    expected_trajectory = np.array([6.0, 5.0, 6.0, 5.0])
    np.testing.assert_allclose(fund, expected_trajectory)


def test_de_finetti_ruin_invalid_inputs():
    """Test validation and error handling."""
    with pytest.raises(ValueError, match="whole number of units"):
        de_finetti_ruin_probability(initial_reserve=2.5, win_probability=0.6)

    with pytest.raises(ValueError, match="strictly between 0.5 and 1.0"):
        de_finetti_theorem_terms(initial_reserve=2, win_probability=0.5)
