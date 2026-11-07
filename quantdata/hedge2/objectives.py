"""Hedging objectives in fixed initial-notional units, including power-exponential OCE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True, kw_only=True)
class PowerOce:
    certainty_equivalent: float
    cash_shift: float
    marginal_utility: np.ndarray
    cash_first_order_residual: float


def power_oce(wealth: np.ndarray, weights: np.ndarray, *, risk_aversion: float) -> PowerOce:
    r"""Optimize cash for u(x)=(1-exp(-lambda*x))/lambda above zero, x-lambda*x²/2 below.

    Risk aversion multiplies wealth measured in initial spot/notional units;
    there is no hidden basis-point conversion. Scenario weights are normalized
    explicitly. This is the legacy benchmark's power-exponential utility,
    not a squared-exponential loss or the separate square-root utility.
    """
    if (wealth.ndim != 1 or weights.shape != wealth.shape or not np.isfinite(wealth).all()
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0
            or not np.isfinite(risk_aversion) or risk_aversion <= 0):
        raise ValueError("Power OCE requires finite wealth, positive total weight and risk aversion")
    probabilities = weights / weights.sum()
    mean = float(probabilities @ wealth)
    centered = wealth - mean

    def marginal(shift: float) -> np.ndarray:
        scaled = risk_aversion * (centered - shift)
        return np.exp(-np.maximum(scaled, 0.)) - np.minimum(scaled, 0.)

    lower, upper = float(centered.min()), float(centered.max())
    if lower == upper:
        shift = lower
    else:
        shift = brentq(lambda value: float(probabilities @ marginal(value) - 1.), lower, upper,
                        xtol=1e-13, rtol=1e-13)
    translated = centered - shift
    positive = np.maximum(translated, 0.)
    negative = np.minimum(translated, 0.)
    utility = -np.expm1(-risk_aversion * positive) / risk_aversion + negative - .5 * risk_aversion * negative**2
    derivative = marginal(shift)
    return PowerOce(certainty_equivalent=float(mean + shift + probabilities @ utility),
                     cash_shift=mean + shift, marginal_utility=derivative,
                     cash_first_order_residual=float(probabilities @ derivative - 1.))
