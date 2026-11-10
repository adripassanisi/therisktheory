"""Tests and boundary validation for full credibility standards."""

import numpy as np
import pytest

from therisktheory.credibility.full_credibility import (
    full_credibility_required_exposure,
    full_credibility_standard,
)

# Optional integration check with Monticelli's partial credibility module
partial_module = pytest.importorskip(
    "therisktheory.credibility.partial_credibility",
    reason="Optional test: requires Monticelli partial credibility PR #21",
)


def test_classical_1082_claims_standard() -> None:
    """Verify classical actuarial 1,082 claims benchmark (p=0.90, k=0.05)."""
    # z for p=0.90 is approx 1.644853. (1.644853 / 0.05)**2 approx 1082.2
    n_0 = full_credibility_standard(target_accuracy=0.05, confidence_level=0.90)
    assert np.isclose(n_0.item(), 1082.22, atol=0.5)


def test_standard_95_confidence_5_percent_accuracy() -> None:
    """Verify standard 95% confidence and 5% accuracy (z=1.95996 -> n_0=1536.6)."""
    n_0 = full_credibility_standard(target_accuracy=0.05, confidence_level=0.95)
    expected = (1.959963984540054 / 0.05) ** 2
    assert np.isclose(n_0.item(), expected, atol=1e-2)


def test_severity_variability_increases_standard() -> None:
    """Property test: positive severity variance strictly increases required standard."""
    n_pure_freq = full_credibility_standard(target_accuracy=0.05)
    n_compound = full_credibility_standard(
        target_accuracy=0.05,
        severity_mean=100.0,
        severity_variance=10000.0,  # CV^2 = 1.0 -> standard doubles
    )
    assert np.isclose(n_compound.item(), 2.0 * n_pure_freq.item())


def test_required_exposure_scaling() -> None:
    """Verify exposure units scale inversely with claim frequency rate."""
    lam = 0.10  # 1 claim per 10 vehicle-years
    n_std = full_credibility_standard(target_accuracy=0.05, confidence_level=0.95)
    exp_req = full_credibility_required_exposure(
        claim_frequency_rate=lam,
        target_accuracy=0.05,
        confidence_level=0.95,
    )
    assert np.isclose(exp_req.item(), n_std.item() / lam)


def test_vectorization_across_accuracies() -> None:
    """Verify array broadcasting across multiple accuracy thresholds."""
    k_vec = np.array([0.01, 0.05, 0.10])
    res = full_credibility_standard(target_accuracy=k_vec)
    assert res.shape == (3,)
    # Stricter accuracy (smaller k) requires strictly more claims
    assert res[0] > res[1] > res[2]


def test_invalid_parameters_raise() -> None:
    """Boundary and domain tests for out-of-range parameters."""
    with pytest.raises(ValueError, match="confidence_level"):
        full_credibility_standard(confidence_level=1.0)
    with pytest.raises(ValueError, match="target_accuracy"):
        full_credibility_standard(target_accuracy=-0.05)
    with pytest.raises(ValueError, match="claim_frequency_rate"):
        full_credibility_required_exposure(claim_frequency_rate=0.0)


def test_interoperability_with_monticelli_standard() -> None:
    """Integration test: Verify consistency with Monticelli's standard (PR #21)."""
    k = 0.05
    p = 0.95
    passanisi_std = full_credibility_standard(target_accuracy=k, confidence_level=p)
    monticelli_std = partial_module.partial_credibility_standard(
        expected_claim_count=100.0, target_accuracy=k, confidence_level=p
    )
    assert np.isclose(passanisi_std.item(), float(monticelli_std), atol=1e-4)
