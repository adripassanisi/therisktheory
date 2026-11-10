"""Unit tests and mathematical properties for collective stability and loading."""

import numpy as np
import pytest

from therisktheory.collective_risk.stability import (
    collective_safety_loading,
    loading_from_initial_reserve,
    stability_parameter,
    stability_sensitivity,
)

# Optional integration check with Bladimir's collective risk process (PR #19)
process_module = pytest.importorskip(
    "therisktheory.collective_risk.process",
    reason="Optional test: requires Bladimir collective risk process PR #19",
)


def test_stability_parameter_analytical() -> None:
    """Verify analytical example: lambda=100, mu1=10, mu2=200 (CV^2=1), theta=0.15."""
    # K = 0.15 * sqrt(100) * (10 / sqrt(200)) = 0.15 * 10 * (10 / 14.1421356) = 1.06066
    k = stability_parameter(
        claim_rate=100.0,
        severity_mean=10.0,
        severity_second_moment=200.0,
        safety_loading=0.15,
    )
    expected = 0.15 * 10.0 * (10.0 / np.sqrt(200.0))
    assert np.isclose(k.item(), expected, atol=1e-5)


def test_stability_sensitivity_elasticities() -> None:
    """Mathematical property: verify exact elasticities (1.0 for theta, 0.5 for lambda)."""
    sens = stability_sensitivity(
        claim_rate=50.0,
        severity_mean=20.0,
        severity_second_moment=800.0,
        safety_loading=0.20,
    )
    assert np.isclose(sens.elasticity_theta.item(), 1.0)
    assert np.isclose(sens.elasticity_lambda.item(), 0.5)
    # Check derivative relation dK/dtheta = K / theta
    assert np.isclose(sens.d_theta.item(), sens.stability.item() / 0.20)


def test_collective_safety_loading_roundtrip() -> None:
    """Mathematical identity: collective_safety_loading inverts stability_parameter."""
    target_k = 1.50
    lam = 80.0
    mu1 = 15.0
    mu2 = 450.0

    theta_req = collective_safety_loading(target_k, lam, mu1, mu2)
    k_achieved = stability_parameter(lam, mu1, mu2, theta_req)

    assert np.isclose(k_achieved.item(), target_k)


def test_loading_from_initial_reserve_tradeoff() -> None:
    """Actuarial property: required loading decreases as initial capital increases."""
    eps = 0.01  # 1% ruin probability
    mu1, mu2 = 100.0, 20000.0

    u_low = 1000.0
    u_high = 5000.0

    theta_low_cap = loading_from_initial_reserve(u_low, eps, mu1, mu2)
    theta_high_cap = loading_from_initial_reserve(u_high, eps, mu1, mu2)

    assert theta_low_cap > theta_high_cap
    assert np.isclose(theta_low_cap.item(), 5.0 * theta_high_cap.item())


def test_invalid_moments_violate_cauchy_schwarz() -> None:
    """Invalid input test: second moment strictly below mean^2 raises ValueError."""
    with pytest.raises(ValueError, match="severity_second_moment"):
        stability_parameter(
            claim_rate=10.0,
            severity_mean=10.0,
            severity_second_moment=50.0,  # 50 < 10^2 = 100
            safety_loading=0.10,
        )


def test_interoperability_with_collective_process() -> None:
    """Integration test connecting with Bladimir's process (PR #19)."""
    mean_claim = 200.0
    moments = [mean_claim, 2 * mean_claim**2, 6 * mean_claim**3]
    proc = process_module.collective_risk_process(
        initial_capital=1000.0,
        claim_rate=3.0,
        severity_moments=moments,
        safety_loading=0.10,
    )
    k = stability_parameter(
        proc.claim_rate,
        proc.severity_moments[0],
        proc.severity_moments[1],
        proc.safety_loading,
    )
    assert k.item() > 0.0
