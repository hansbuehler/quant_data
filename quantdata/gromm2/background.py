"""Background-factor estimation for GROMM2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class CirEstimate:
    """Gaussian quasi-likelihood estimate of a one-factor CIR variance law."""

    mean_reversion: float
    long_run_variance: float
    volatility_of_variance: float
    negative_log_likelihood: float
    standardized_innovations: FloatArray

    @property
    def long_run_volatility(self) -> float:
        return float(np.sqrt(self.long_run_variance))

    @property
    def feller_ratio(self) -> float:
        return float(
            2.0
            * self.mean_reversion
            * self.long_run_variance
            / self.volatility_of_variance**2
        )


def kernel_annual_volatilities(
    model_expiries_y: FloatArray,
    kernel_total_variances: FloatArray,
    target_expiries_y: FloatArray,
    *,
    smoothness: float,
) -> FloatArray:
    """Interpolate the released SANOS Black-kernel volatility term structure."""
    source_expiries = np.asarray(model_expiries_y, dtype=np.float64)
    source_variances = np.asarray(kernel_total_variances, dtype=np.float64)
    targets = np.asarray(target_expiries_y, dtype=np.float64)
    if (
        source_expiries.ndim != 1
        or source_variances.shape != source_expiries.shape
        or targets.ndim != 1
        or np.any(np.diff(source_expiries) <= 0.0)
        or np.any(source_variances <= 0.0)
        or np.any(targets <= 0.0)
        or smoothness <= 0.0
    ):
        raise ValueError("invalid SANOS kernel-volatility inputs")
    clipped = np.clip(targets, source_expiries[0], source_expiries[-1])
    total_variances = np.interp(clipped, source_expiries, source_variances)
    return np.sqrt(smoothness * total_variances / clipped)


def cir_conditional_moments(
    current_variance: FloatArray,
    *,
    mean_reversion: float,
    long_run_variance: float,
    volatility_of_variance: float,
    step_years: float,
) -> tuple[FloatArray, FloatArray]:
    """Return exact CIR conditional mean and variance over one time step."""
    current = np.asarray(current_variance, dtype=np.float64)
    if (
        mean_reversion <= 0.0
        or long_run_variance <= 0.0
        or volatility_of_variance <= 0.0
        or step_years <= 0.0
        or np.any(current <= 0.0)
    ):
        raise ValueError("CIR parameters and current variances must be positive")
    decay = np.exp(-mean_reversion * step_years)
    conditional_mean = long_run_variance + (current - long_run_variance) * decay
    conditional_variance = (
        current
        * volatility_of_variance**2
        * decay
        * (1.0 - decay)
        / mean_reversion
        + long_run_variance
        * volatility_of_variance**2
        * (1.0 - decay) ** 2
        / (2.0 * mean_reversion)
    )
    return conditional_mean, conditional_variance


def fit_cir_quasi_likelihood(
    origin_variance: FloatArray,
    endpoint_variance: FloatArray,
    *,
    step_years: float = 1.0 / 252.0,
) -> CirEstimate:
    """Fit a positive CIR law using its exact first two conditional moments."""
    origins = np.asarray(origin_variance, dtype=np.float64)
    endpoints = np.asarray(endpoint_variance, dtype=np.float64)
    if (
        origins.ndim != 1
        or endpoints.shape != origins.shape
        or len(origins) < 20
        or np.any(~np.isfinite(origins))
        or np.any(~np.isfinite(endpoints))
        or np.any(origins <= 0.0)
        or np.any(endpoints <= 0.0)
        or step_years <= 0.0
    ):
        raise ValueError("invalid CIR estimation panel")

    long_run_initial = float(np.mean(np.concatenate((origins, endpoints))))
    centered = origins - np.mean(origins)
    denominator = float(centered @ centered)
    slope = (
        float(centered @ (endpoints - np.mean(endpoints))) / denominator
        if denominator > 0.0
        else 0.95
    )
    slope = float(np.clip(slope, 1.0e-4, 0.9999))
    mean_reversion_initial = -np.log(slope) / step_years
    residual = endpoints - (
        long_run_initial + (origins - long_run_initial) * slope
    )
    volatility_initial = float(
        np.sqrt(
            np.mean(residual**2)
            / max(np.mean(origins) * step_years, np.finfo(np.float64).tiny)
        )
    )
    volatility_initial = max(volatility_initial, 1.0e-5)

    def objective(raw: FloatArray) -> float:
        mean_reversion, long_run_variance, volatility_of_variance = np.exp(raw)
        conditional_mean, conditional_variance = cir_conditional_moments(
            origins,
            mean_reversion=float(mean_reversion),
            long_run_variance=float(long_run_variance),
            volatility_of_variance=float(volatility_of_variance),
            step_years=step_years,
        )
        variance_floor = np.finfo(np.float64).eps * max(
            float(np.mean(origins**2)), 1.0e-12
        )
        safe_variance = np.maximum(conditional_variance, variance_floor)
        residuals = endpoints - conditional_mean
        return float(
            0.5 * np.sum(np.log(safe_variance) + residuals**2 / safe_variance)
        )

    result = minimize(
        objective,
        np.log(
            np.asarray(
                [
                    mean_reversion_initial,
                    long_run_initial,
                    volatility_initial,
                ]
            )
        ),
        method="L-BFGS-B",
        bounds=[
            (np.log(1.0e-3), np.log(2_000.0)),
            (np.log(1.0e-8), np.log(4.0)),
            (np.log(1.0e-5), np.log(20.0)),
        ],
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        raise RuntimeError(f"CIR quasi-likelihood fit failed: {result.message}")
    mean_reversion, long_run_variance, volatility_of_variance = np.exp(result.x)
    conditional_mean, conditional_variance = cir_conditional_moments(
        origins,
        mean_reversion=float(mean_reversion),
        long_run_variance=float(long_run_variance),
        volatility_of_variance=float(volatility_of_variance),
        step_years=step_years,
    )
    standardized = (endpoints - conditional_mean) / np.sqrt(conditional_variance)
    return CirEstimate(
        mean_reversion=float(mean_reversion),
        long_run_variance=float(long_run_variance),
        volatility_of_variance=float(volatility_of_variance),
        negative_log_likelihood=float(result.fun),
        standardized_innovations=standardized,
    )
