"""Unit tests and mathematical properties for aggregate loss approximations."""

import numpy as np
import pytest

from therisktheory.collective_risk.approximations import (
    edgeworth_approximation,
    normal_approximation,
    normal_power_approximation,
)

# Optional integration import following repo practice (PR #14)
aggregate_loss = pytest.importorskip(
    "therisktheory.collective_risk.aggregate_loss",
    reason="Optional test: requires Monticelli aggregate loss PR #20",
)


def test_normal_approximation_symmetry() -> None:
    """Verify median evaluation at mean equals 0.5."""
    prob = normal_approximation(x=100.0, mean=100.0, variance=400.0)
    assert np.isclose(prob.item(), 0.5)


def test_skewness_zero_reduces_to_normal() -> None:
    """Mathematical property: when skewness is zero, Edgeworth and NP reduce to Normal."""
    x = 130.0
    mu = 100.0
    var = 400.0

    p_norm = normal_approximation(x, mu, var)
    p_edge = edgeworth_approximation(x, mu, var, skewness=0.0)
    p_np = normal_power_approximation(x, mu, var, skewness=0.0)

    assert np.isclose(p_edge.item(), p_norm.item(), atol=1e-12)
    assert np.isclose(p_np.item(), p_norm.item(), atol=1e-6)


def test_edgeworth_and_np_positive_skewness_shift() -> None:
    """Analytical property: positive skewness increases heavy upper tail probability."""
    x = 150.0  # +2.5 standard deviations
    mu = 100.0
    var = 400.0  # sigma = 20.0
    g1 = 1.2

    p_norm = normal_approximation(x, mu, var).item()
    p_edge = edgeworth_approximation(x, mu, var, g1).item()
    p_np = normal_power_approximation(x, mu, var, g1).item()

    # For positively skewed insurance losses, tail survival is fatter
    assert p_norm > p_edge
    assert p_norm > p_np


def test_deterministic_zero_variance() -> None:
    """Boundary test: variance zero yields step function."""
    assert normal_approximation(10.0, 10.0, 0.0).item() == 1.0
    assert normal_approximation(9.9, 10.0, 0.0).item() == 0.0


def test_vectorization_and_broadcasting() -> None:
    """Verify array inputs and broadcasting."""
    x = np.array([80.0, 100.0, 120.0])
    p = normal_power_approximation(x, 100.0, 400.0, skewness=0.5)
    assert p.shape == (3,)
    assert p[0] < p[1] < p[2]


def test_invalid_negative_variance() -> None:
    """Invalid input test: negative variance raises ValueError."""
    with pytest.raises(ValueError):
        normal_approximation(100.0, 100.0, -10.0)


def test_joint_integration_with_monticelli_model() -> None:
    """Integration test combining Monticelli moments with NP approximation."""
    e_n, v_n = 50.0, 50.0
    e_x, v_x = 200.0, 40000.0

    mean_s, var_s, _ = aggregate_loss.aggregate_loss_moments(
        e_n, v_n, e_x, v_x
    )
    # Compound Poisson skewness approximation gamma_1 ~ E[N]*E[X^3] / sigma^3
    skew_s = 0.45

    prob = normal_power_approximation(12000.0, mean_s, var_s, skew_s)
    assert 0.0 <= prob.item() <= 1.0
