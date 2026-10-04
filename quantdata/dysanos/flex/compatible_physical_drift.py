"""Estimate physical DLV drifts that admit a controlled change to Q."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from scipy.stats import chi2


@dataclass(frozen=True, kw_only=True)
class CompatibleConstantDriftFit:
    r"""A constant factor drift in ``mu_P = mu_Q + L lambda``.

    ``weekly_factor_mean`` is :math:`\sqrt{h}\lambda`.  Its squared norm is
    consequently the one-week variance of the Gaussian log likelihood ratio.
    """

    raw_weekly_factor_mean: np.ndarray
    supported_weekly_factor_mean: np.ndarray
    weekly_factor_mean: np.ndarray
    annualized_market_price_of_risk: np.ndarray
    hac_standard_error: np.ndarray
    hac_t_value: np.ndarray
    raw_p_value: np.ndarray
    holm_adjusted_p_value: np.ndarray
    supported_modes: np.ndarray
    radial_scale: float
    equivalent_ridge_penalty: float
    one_week_log_weight_variance: float
    horizon_log_weight_variance: float
    horizon_expected_ess_fraction: float


def newey_west_mean_covariance(values: np.ndarray, *, lag: int) -> np.ndarray:
    """Estimate the HAC covariance of a vector sample mean.

    Bartlett weights are used.  ``lag=4`` corrects the serial dependence from
    five-session returns observed at daily origins.
    """

    observations = np.asarray(values, dtype=np.float64)
    verify_inp(
        observations.ndim == 2
        and observations.shape[0] > lag + 1
        and observations.shape[1] > 0
        and np.all(np.isfinite(observations)),
        lambda: f"values must be a finite matrix with more than {lag + 1} rows",
    )
    verify_inp(
        isinstance(lag, int) and not isinstance(lag, bool) and lag >= 0,
        lambda: f"lag must be a non-negative integer, found {lag!r}",
    )
    centered = observations - np.mean(observations, axis=0, keepdims=True)
    sample_count = len(centered)
    long_run_covariance = centered.T @ centered / sample_count
    for offset in range(1, lag + 1):
        autocovariance = centered[offset:].T @ centered[:-offset] / sample_count
        weight = 1.0 - offset / (lag + 1.0)
        long_run_covariance += weight * (autocovariance + autocovariance.T)
    covariance = long_run_covariance / sample_count
    return 0.5 * (covariance + covariance.T)


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Return Holm familywise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    verify_inp(
        values.ndim == 1
        and len(values) > 0
        and np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)),
        lambda: "p_values must be a non-empty finite vector in [0, 1]",
    )
    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * ordered
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def fit_compatible_constant_drift(
    factor_innovations: np.ndarray,
    *,
    time_increment_y: float,
    horizon_steps: int,
    target_horizon_ess_fraction: float,
    familywise_alpha: float = 0.05,
    hac_lag: int = 4,
) -> CompatibleConstantDriftFit:
    r"""Fit a supported, likelihood-budgeted constant market price of risk.

    If ``e`` denotes historical innovations relative to the analytical Q
    drift, the compatible Gaussian model is

    ``E_P[e] = sqrt(h) * lambda``.

    Holm's procedure first removes factor means unsupported after multiplicity
    correction.  The remaining maximum-likelihood vector is projected onto
    the log-likelihood budget

    ``horizon_steps * ||sqrt(h) lambda||^2 <= -log(target ESS fraction)``.

    The radial projection is also the solution of an isotropic ridge-penalized
    Gaussian likelihood.  ``equivalent_ridge_penalty`` reports that Lagrange
    multiplier under the convention ``0.5*rho*||m||^2``.
    """

    observations = np.asarray(factor_innovations, dtype=np.float64)
    verify_inp(
        observations.ndim == 2
        and observations.shape[0] > hac_lag + 1
        and observations.shape[1] > 0
        and np.all(np.isfinite(observations)),
        lambda: "factor_innovations must be a finite non-empty matrix",
    )
    verify_inp(
        math.isfinite(time_increment_y) and time_increment_y > 0.0,
        lambda: f"time_increment_y must be finite and positive, found {time_increment_y!r}",
    )
    verify_inp(
        isinstance(horizon_steps, int)
        and not isinstance(horizon_steps, bool)
        and horizon_steps > 0,
        lambda: f"horizon_steps must be a positive integer, found {horizon_steps!r}",
    )
    verify_inp(
        math.isfinite(target_horizon_ess_fraction)
        and 0.0 < target_horizon_ess_fraction <= 1.0,
        lambda: "target_horizon_ess_fraction must lie in (0, 1]",
    )
    verify_inp(
        math.isfinite(familywise_alpha) and 0.0 < familywise_alpha < 1.0,
        lambda: "familywise_alpha must lie in (0, 1)",
    )

    mean = np.mean(observations, axis=0)
    covariance = newey_west_mean_covariance(observations, lag=hac_lag)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = np.divide(
        mean,
        standard_error,
        out=np.where(mean == 0.0, 0.0, np.copysign(np.inf, mean)),
        where=standard_error > 0.0,
    )
    p_value = np.asarray(
        [math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in t_value],
        dtype=np.float64,
    )
    adjusted_p_value = holm_adjusted_p_values(p_value)
    supported = adjusted_p_value <= familywise_alpha
    supported_mean = np.where(supported, mean, 0.0)

    supported_norm = float(np.linalg.vector_norm(supported_mean))
    horizon_budget = -math.log(target_horizon_ess_fraction)
    maximum_weekly_norm = math.sqrt(horizon_budget / horizon_steps)
    if supported_norm == 0.0:
        scale = 1.0
    else:
        scale = min(1.0, maximum_weekly_norm / supported_norm)
    regularized_mean = scale * supported_mean
    one_week_cost = float(regularized_mean @ regularized_mean)
    horizon_cost = horizon_steps * one_week_cost
    ridge_penalty = 0.0 if scale >= 1.0 else 1.0 / scale - 1.0

    return CompatibleConstantDriftFit(
        raw_weekly_factor_mean=mean,
        supported_weekly_factor_mean=supported_mean,
        weekly_factor_mean=regularized_mean,
        annualized_market_price_of_risk=regularized_mean / math.sqrt(time_increment_y),
        hac_standard_error=standard_error,
        hac_t_value=t_value,
        raw_p_value=p_value,
        holm_adjusted_p_value=adjusted_p_value,
        supported_modes=supported,
        radial_scale=scale,
        equivalent_ridge_penalty=ridge_penalty,
        one_week_log_weight_variance=one_week_cost,
        horizon_log_weight_variance=horizon_cost,
        horizon_expected_ess_fraction=math.exp(-horizon_cost),
    )


def factor_drift_audit(
    factor_innovations: np.ndarray,
    weekly_factor_mean: np.ndarray,
    *,
    hac_lag: int,
) -> dict[str, float | int]:
    """Audit a proposed factor mean against historical innovations."""

    observations = np.asarray(factor_innovations, dtype=np.float64)
    candidate = np.asarray(weekly_factor_mean, dtype=np.float64)
    verify_inp(
        observations.ndim == 2
        and candidate.shape == (observations.shape[1],)
        and np.all(np.isfinite(observations))
        and np.all(np.isfinite(candidate)),
        lambda: "factor observations and candidate mean have incompatible shapes",
    )
    residual = observations - candidate[None]
    residual_mean = np.mean(residual, axis=0)
    mean_covariance = newey_west_mean_covariance(residual, lag=hac_lag)
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    tolerance = (
        np.finfo(np.float64).eps
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
        * len(eigenvalues)
    )
    retained = eigenvalues > tolerance
    projected_mean = eigenvectors[:, retained].T @ residual_mean
    statistic = float(np.sum(np.square(projected_mean) / eigenvalues[retained]))
    degrees_of_freedom = int(np.sum(retained))
    standard_error = np.sqrt(np.maximum(np.diag(mean_covariance), 0.0))
    t_value = np.divide(
        residual_mean,
        standard_error,
        out=np.where(residual_mean == 0.0, 0.0, np.copysign(np.inf, residual_mean)),
        where=standard_error > 0.0,
    )
    log_score_gain = observations @ candidate - 0.5 * float(candidate @ candidate)
    score_covariance = newey_west_mean_covariance(log_score_gain[:, None], lag=hac_lag)
    return {
        "num_observations": int(len(observations)),
        "num_factors": int(observations.shape[1]),
        "mean_gaussian_log_score_gain_over_q": float(np.mean(log_score_gain)),
        "log_score_gain_hac_standard_error": float(math.sqrt(max(score_covariance[0, 0], 0.0))),
        "residual_root_mean_square_factor_mean": float(
            np.sqrt(np.mean(np.square(residual_mean)))
        ),
        "residual_maximum_absolute_hac_t": float(np.max(np.abs(t_value))),
        "residual_joint_wald_statistic": statistic,
        "residual_joint_degrees_of_freedom": degrees_of_freedom,
        "residual_joint_p_value": float(chi2.sf(statistic, degrees_of_freedom)),
    }


__all__ = [
    "CompatibleConstantDriftFit",
    "factor_drift_audit",
    "fit_compatible_constant_drift",
    "holm_adjusted_p_values",
    "newey_west_mean_covariance",
]
