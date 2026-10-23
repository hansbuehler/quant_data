"""Keep the front SANOS marginal exact under a common scale-mixture tail law."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri


def normal_score_from_scale_mixture(
    raw_spot_score: np.ndarray,
    *,
    stationary_log_variance: float,
    unconditional_scale: float,
    quadrature_nodes: int = 48,
    conditional_log_variance_mean: np.ndarray | float = 0.0,
    conditional_log_variance_variance: float | None = None,
) -> np.ndarray:
    """Gaussianize a common-scale spot shock without removing tail dependence.

    The raw shock has distribution ``sqrt(V) Z`` with independent standard
    normal ``Z`` and ``V = scale * exp(g - Var(g)/2)``.  The probability
    integral transform maps its predictive marginal to N(0,1) before passing
    it through the decoded one-day front SANOS quantile function.  Pass the
    conditional OU mean and variance to retain this property on later steps.
    """
    raw = np.asarray(raw_spot_score, dtype=np.float64)
    conditional_mean = np.asarray(conditional_log_variance_mean, dtype=np.float64)
    conditional_variance = (
        stationary_log_variance
        if conditional_log_variance_variance is None
        else conditional_log_variance_variance
    )
    if (
        np.any(~np.isfinite(raw))
        or np.any(~np.isfinite(conditional_mean))
        or stationary_log_variance < 0.0
        or conditional_variance < 0.0
        or unconditional_scale <= 0.0
        or quadrature_nodes < 8
    ):
        raise ValueError("invalid scale-mixture spot score")
    if conditional_variance == 0.0:
        conditional_scale = unconditional_scale * np.exp(
            conditional_mean - 0.5 * stationary_log_variance
        )
        return raw / np.sqrt(conditional_scale)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    log_scale = (
        np.log(unconditional_scale)
        + conditional_mean[..., None]
        + np.sqrt(2.0 * conditional_variance) * nodes
        - 0.5 * stationary_log_variance
    )
    standard_deviation = np.exp(0.5 * log_scale)
    probabilities = np.sum(
        ndtr(raw[..., None] / standard_deviation) * weights,
        axis=-1,
    ) / np.sqrt(np.pi)
    return ndtri(np.clip(probabilities, 1.0e-12, 1.0 - 1.0e-12))
