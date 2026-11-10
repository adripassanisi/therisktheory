"""Example: Aggregate loss distribution approximations (Topic 3)."""

import numpy as np

from therisktheory.collective_risk.approximations import (
    edgeworth_approximation,
    normal_approximation,
    normal_power_approximation,
)

# Portfolio aggregate parameters (Kaas et al. 2008, Section 3.8)
mu_s = 10_000.0
var_s = 2_500_000.0  # sigma = 1,581.14
skew_s = 0.60

levels = np.array([10_000.0, 12_000.0, 14_000.0])

p_norm = normal_approximation(levels, mu_s, var_s)
p_edge = edgeworth_approximation(levels, mu_s, var_s, skew_s)
p_np = normal_power_approximation(levels, mu_s, var_s, skew_s)

print("Aggregate Loss Distribution Approximations:")
print(f"Mean: {mu_s:,.0f}, StdDev: {np.sqrt(var_s):,.2f}, Skewness: {skew_s}")
print("-" * 55)
print(" Level      Normal     Edgeworth   Normal Power (NP)")
print("-" * 55)
for x, pn, pe, pnp in zip(levels, p_norm, p_edge, p_np, strict=True):
    print(f"{x:8.0f}   {pn:9.5f}   {pe:9.5f}   {pnp:9.5f}")
