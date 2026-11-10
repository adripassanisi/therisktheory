"""Example demonstrating full credibility standards and required exposures."""

import numpy as np

from therisktheory.credibility import (
    full_credibility_required_exposure,
    full_credibility_standard,
)

# 1. Classical benchmarks for claim counts
n_90_5 = full_credibility_standard(target_accuracy=0.05, confidence_level=0.90)
n_95_5 = full_credibility_standard(target_accuracy=0.05, confidence_level=0.95)

print("Full Credibility Standards for Claim Frequency:")
print(f"90% Confidence, +/-5% Error: {n_90_5.item():,.1f} claims")
print(f"95% Confidence, +/-5% Error: {n_95_5.item():,.1f} claims")

# 2. Compound aggregate loss standard (accounting for severity variance)
mean_x = 2_500.0
var_x = 6_250_000.0  # CV_X = 1.0
n_compound = full_credibility_standard(
    target_accuracy=0.05,
    confidence_level=0.95,
    severity_mean=mean_x,
    severity_variance=var_x,
)
print(f"Compound Loss Standard (CV=1.0): {n_compound.item():,.1f} claims")

# 3. Required policyholders/exposures
annual_freq = 0.08  # 8 claims per 100 vehicle-years
req_vehicles = full_credibility_required_exposure(
    claim_frequency_rate=annual_freq,
    target_accuracy=0.05,
    confidence_level=0.95,
)
print(f"Required Vehicle-Years for Full Credibility: {req_vehicles.item():,.0f}")
