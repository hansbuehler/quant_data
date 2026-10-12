"""Risk objectives and robust linear hedges shared by benchmark models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class TylerConfig:
    """Numerical configuration for the automatically regularized Tyler shape."""

    maximum_iterations: int = 500
    relative_tolerance: float = 1.0e-10
    minimum_scale: float = 1.0e-8
    existence_margin: float = 1.0e-3
    maximum_shrinkage: float = 0.99

    def __post_init__(self) -> None:
        verify_inp(
            self.maximum_iterations > 0,
            lambda: "maximum_iterations must be positive",
        )
        verify_inp(
            math.isfinite(self.relative_tolerance) and self.relative_tolerance > 0.0,
            lambda: "relative_tolerance must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.minimum_scale) and self.minimum_scale > 0.0,
            lambda: "minimum_scale must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.existence_margin) and self.existence_margin > 0.0,
            lambda: "existence_margin must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.maximum_shrinkage)
            and 0.0 < self.maximum_shrinkage < 1.0,
            lambda: "maximum_shrinkage must lie strictly between zero and one",
        )


@dataclass(frozen=True, kw_only=True)
class CovarianceEstimate:
    """Positive-definite covariance estimate and convergence diagnostics."""

    covariance: FloatArray
    center: FloatArray
    marginal_scales: FloatArray
    shrinkage: float
    iterations: int
    converged: bool


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class CostLassoConfig:
    """Coordinate-descent controls for a spread-penalized linear hedge."""

    maximum_iterations: int = 10_000
    absolute_tolerance: float = 1.0e-12
    ridge: float = 1.0e-10

    def __post_init__(self) -> None:
        verify_inp(
            self.maximum_iterations > 0,
            lambda: "maximum_iterations must be positive",
        )
        verify_inp(
            math.isfinite(self.absolute_tolerance) and self.absolute_tolerance > 0.0,
            lambda: "absolute_tolerance must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.ridge) and self.ridge >= 0.0,
            lambda: "ridge must be finite and non-negative",
        )


@dataclass(frozen=True, kw_only=True)
class LinearHedge:
    """Spread-penalized action for a long claim and solver diagnostics."""

    action: FloatArray
    covariance: FloatArray
    claim_cross_covariance: FloatArray
    claim_variance: float
    penalty: FloatArray
    objective: float
    iterations: int
    converged: bool
    covariance_shrinkage: float
    covariance_iterations: int
    covariance_converged: bool


def regularized_tyler_covariance(
    samples: ArrayLike,
    *,
    config: TylerConfig = TylerConfig(),
) -> CovarianceEstimate:
    """Estimate covariance using robust marginals and a shrinkage Tyler shape.

    The shrinkage intensity is recomputed from the rolling sample. The smooth
    ratio ``dimension / (observations + dimension)`` supplies the automatic
    target weight, while the regularized-Tyler existence bound contributes the
    floor ``1 - observations / dimension`` in high dimension. The configured
    margin makes that inequality strict and ``maximum_shrinkage < 1`` prevents
    the estimator from collapsing exactly to its diagonal target.
    """
    values = _finite_matrix(samples, name="samples")
    observations, dimension = values.shape
    verify_inp(
        observations >= 2,
        lambda: f"at least two samples are required, found {observations}",
    )
    center = np.median(values, axis=0)
    centered = values - center
    median_absolute_deviation = np.median(np.abs(centered), axis=0)
    marginal_scales = np.maximum(
        1.4826 * median_absolute_deviation,
        config.minimum_scale,
    )
    standardized = centered / marginal_scales
    nonzero = np.linalg.norm(standardized, axis=1) > config.minimum_scale
    standardized = standardized[nonzero]
    if len(standardized) < 2:
        raise ValueError("Tyler covariance requires at least two nonzero observations")

    effective_observations = len(standardized)
    automatic_shrinkage = dimension / (effective_observations + dimension)
    existence_floor = max(
        0.0,
        1.0 - effective_observations / dimension + config.existence_margin,
    )
    shrinkage = min(
        config.maximum_shrinkage,
        max(automatic_shrinkage, existence_floor),
    )
    identity = np.eye(dimension, dtype=np.float64)
    shape = identity.copy()
    converged = False
    iterations = 0
    for iterations in range(1, config.maximum_iterations + 1):
        inverse = np.linalg.inv(shape)
        radial = np.einsum(
            "ni,ij,nj->n",
            standardized,
            inverse,
            standardized,
        )
        radial = np.maximum(radial, config.minimum_scale**2)
        scatter = (dimension / len(standardized)) * np.einsum(
            "ni,nj,n->ij",
            standardized,
            standardized,
            1.0 / radial,
        )
        updated = (1.0 - shrinkage) * scatter + shrinkage * identity
        updated *= dimension / np.trace(updated)
        relative_change = np.linalg.norm(updated - shape) / np.linalg.norm(shape)
        shape = updated
        if relative_change <= config.relative_tolerance:
            converged = True
            break

    covariance = shape * np.outer(marginal_scales, marginal_scales)
    covariance = 0.5 * (covariance + covariance.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue <= 0.0:
        covariance += identity * (config.minimum_scale**2 - minimum_eigenvalue)
    return CovarianceEstimate(
        covariance=covariance,
        center=np.asarray(center, dtype=np.float64),
        marginal_scales=np.asarray(marginal_scales, dtype=np.float64),
        shrinkage=shrinkage,
        iterations=iterations,
        converged=converged,
    )


def cost_lasso_minimum_variance_hedge(
    *,
    instrument_returns: ArrayLike,
    claim_returns: ArrayLike,
    linear_transaction_costs: ArrayLike,
    cost_multiplier: float = 1.0,
    tyler_config: TylerConfig = TylerConfig(),
    solver_config: CostLassoConfig = CostLassoConfig(),
) -> LinearHedge:
    """Fit a long-claim hedge with robust covariance and entry-cost lasso.

    For action ``a`` and long-claim centered return ``Y``, this minimizes

    ``0.5 Var(Y + X a) + cost_multiplier * spread' |a|``.
    """
    instruments = _finite_matrix(instrument_returns, name="instrument_returns")
    claim = np.asarray(claim_returns, dtype=np.float64)
    if claim.shape != (len(instruments),) or not np.all(np.isfinite(claim)):
        raise ValueError("claim_returns must be a finite vector aligned with samples")
    transaction_costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if transaction_costs.shape != (instruments.shape[1],) or np.any(
        ~np.isfinite(transaction_costs) | (transaction_costs < 0.0)
    ):
        raise ValueError(
            "linear_transaction_costs must be finite, non-negative, and aligned"
        )
    verify_inp(
        math.isfinite(cost_multiplier) and cost_multiplier >= 0.0,
        lambda: "cost_multiplier must be finite and non-negative",
    )

    joint = np.column_stack((instruments, claim))
    estimate = regularized_tyler_covariance(joint, config=tyler_config)
    covariance = estimate.covariance[:-1, :-1].copy()
    covariance.flat[:: len(covariance) + 1] += solver_config.ridge
    cross_covariance = estimate.covariance[:-1, -1]
    claim_variance = float(estimate.covariance[-1, -1])
    penalty = cost_multiplier * transaction_costs
    action = np.zeros(instruments.shape[1], dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(1, solver_config.maximum_iterations + 1):
        previous = action.copy()
        for coordinate in range(len(action)):
            residual = -cross_covariance[coordinate] - (
                covariance[coordinate] @ action
                - covariance[coordinate, coordinate] * action[coordinate]
            )
            action[coordinate] = _soft_threshold(
                residual,
                penalty[coordinate],
            ) / covariance[coordinate, coordinate]
        if np.max(np.abs(action - previous)) <= solver_config.absolute_tolerance:
            converged = True
            break
    objective = float(
        0.5 * claim_variance
        + 0.5 * action @ covariance @ action
        + cross_covariance @ action
        + penalty @ np.abs(action)
    )
    return LinearHedge(
        action=action,
        covariance=covariance,
        claim_cross_covariance=cross_covariance,
        claim_variance=claim_variance,
        penalty=penalty,
        objective=objective,
        iterations=iterations,
        converged=converged,
        covariance_shrinkage=estimate.shrinkage,
        covariance_iterations=estimate.iterations,
        covariance_converged=estimate.converged,
    )


def _soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite matrix, found {matrix.shape}")
    return matrix
