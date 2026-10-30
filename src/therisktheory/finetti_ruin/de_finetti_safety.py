"""Restricted safety measures for unit-step ruin model.

Reference:
    Kaas, R., Goovaerts, M., Dhaene, J., and Denuit, M. (2008).
    Modern Actuarial Risk Theory: Using R. 2nd ed. Springer. Section 4.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from therisktheory import _validation

if TYPE_CHECKING:
    import numpy.typing as npt


def safety_index(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate the safety index R = log(p / q) in unit-step model."""
    p = _validation.as_probability_array(win_probability, "win_probability")
    if np.any(p < 0.5):
        raise ValueError("win_probability must be at least 0.5")
    if np.any(p == 1.0):
        raise ValueError("win_probability must be below 1.0")

    result = np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)
    return _validation.check_finite_result(result, "safety_index")


def de_finetti_safety_loading(
    win_probability: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Calculate expected surplus gain E[G] = 2p - 1 per round."""
    p = _validation.as_probability_array(win_probability, "win_probability")
    if np.any(p < 0.5):
        raise ValueError("win_probability must be at least 0.5")

    result = np.asarray(2.0 * p - 1.0, dtype=np.float64)
    return _validation.check_finite_result(result, "de_finetti_safety_loading")
