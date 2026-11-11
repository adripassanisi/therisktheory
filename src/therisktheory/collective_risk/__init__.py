"""Collective risk theory, Lundberg methods, and ruin models."""

from therisktheory.collective_risk.stability import (
    StabilitySensitivity,
    collective_safety_loading,
    loading_from_initial_reserve,
    stability_parameter,
    stability_sensitivity,
)

try:
    from therisktheory.collective_risk.process import (
        CollectiveRiskProcess,
        ModelAssumption,
        annual_result_generator,
        collective_model_assumptions,
        collective_risk_process,
        simulate_surplus,
    )
    _PROCESS_EXPORTS = [
        "CollectiveRiskProcess",
        "ModelAssumption",
        "annual_result_generator",
        "collective_model_assumptions",
        "collective_risk_process",
        "simulate_surplus",
    ]
except ImportError:
    _PROCESS_EXPORTS = []

__all__ = [
    "StabilitySensitivity",
    "collective_safety_loading",
    "loading_from_initial_reserve",
    "stability_parameter",
    "stability_sensitivity",
    *_PROCESS_EXPORTS,
]
