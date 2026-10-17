"""Model-neutral removal of one-day conditional return means."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


def adaptive_scenario_count(
    instrument_count: int,
    *,
    minimum: int = 10_000,
    paths_per_instrument: int = 10,
    multiple: int = 2,
) -> int:
    """Return the antithetic scenario budget for one hedge universe."""
    verify_inp(instrument_count >= 1, lambda: "instrument_count must be positive")
    verify_inp(minimum >= 2, lambda: "minimum must be at least two")
    verify_inp(
        paths_per_instrument >= 1,
        lambda: "paths_per_instrument must be positive",
    )
    verify_inp(multiple >= 2 and multiple % 2 == 0, lambda: "multiple must be even")
    required = max(minimum, paths_per_instrument * instrument_count)
    return multiple * math.ceil(required / multiple)


class MeanReduction(StrEnum):
    """Supported conventions for removing one-day scenario means."""

    DEMEAN = "demean"
    MEMM = "memm"


class MeanReductionError(RuntimeError):
    """Base class for failures to construct a drift-neutral scenario law."""


class MEMMInfeasibleError(MeanReductionError):
    """The origin is outside the convex hull of instrument returns."""


class MEMMBoundaryError(MeanReductionError):
    """A zero-mean law exists only on the boundary of the scenario simplex."""


class MEMMConvergenceError(MeanReductionError):
    """The finite exponential tilt did not reach the requested tolerance."""


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class MeanReductionConfig:
    """Numerical controls for demeaning and exact discrete MEMM reduction."""

    maximum_iterations: int = 200
    moment_tolerance: float = 1.0e-11
    rank_tolerance: float = 1.0e-12
    minimum_scale: float = 1.0e-14
    line_search_minimum_step: float = 1.0e-12
    feasibility_check_maximum_samples: int = 10_000
    strict_feasibility_relative_weight: float = 1.0e-12

    def __post_init__(self) -> None:
        verify_inp(
            self.maximum_iterations > 0,
            lambda: "maximum_iterations must be positive",
        )
        for name, value in (
            ("moment_tolerance", self.moment_tolerance),
            ("rank_tolerance", self.rank_tolerance),
            ("minimum_scale", self.minimum_scale),
            ("line_search_minimum_step", self.line_search_minimum_step),
            (
                "strict_feasibility_relative_weight",
                self.strict_feasibility_relative_weight,
            ),
        ):
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name: f"{name} must be finite and positive",
            )
        verify_inp(
            self.feasibility_check_maximum_samples >= 0,
            lambda: "feasibility_check_maximum_samples must be non-negative",
        )


@dataclass(frozen=True, kw_only=True)
class MeanReductionDiagnostics:
    """Diagnostics for the fitted drift-neutral scenario law."""

    method: MeanReduction
    observations: int
    instruments: int
    targets: int
    constraint_rank: int
    raw_max_standardized_mean: float
    reduced_max_standardized_mean: float
    effective_sample_size: float
    kl_divergence: float
    minimum_weight: float
    maximum_weight: float
    moment_residual: float
    iterations: int
    converged: bool


@dataclass(frozen=True, kw_only=True)
class MeanReductionResult:
    """Drift-neutral returns and their scenario probabilities.

    Under ``DEMEAN``, both matrices are translated and probabilities remain
    uniform. Under ``MEMM``, instrument scenarios retain their original values
    and probabilities are exponentially tilted so their weighted mean is zero.
    Target returns are then centered under the same probabilities. This removes
    target alpha without adding target constraints to the measure change.
    """

    instrument_returns: FloatArray
    target_returns: FloatArray
    weights: FloatArray
    instrument_mean_removed: FloatArray
    target_mean_removed: FloatArray
    diagnostics: MeanReductionDiagnostics


def reduce_one_day_means(
    instrument_returns: ArrayLike,
    target_returns: ArrayLike,
    *,
    method: MeanReduction | str,
    config: MeanReductionConfig = MeanReductionConfig(),
) -> MeanReductionResult:
    """Remove one-day means by translation or an exact discrete MEMM.

    Parameters
    ----------
    instrument_returns:
        Matrix ``(scenarios, instruments)`` of tradable returns.
    target_returns:
        Matrix ``(scenarios, targets)`` of claim returns. MEMM constraints do
        not include these columns.
    method:
        Uniform demeaning or minimum-entropy martingale probabilities.
    config:
        Rank and nonlinear-solver tolerances.

    Returns
    -------
    MeanReductionResult
        Float64 return matrices, scenario probabilities, removed means, and
        numerical diagnostics.
    """
    instruments = _finite_matrix(instrument_returns, name="instrument_returns")
    targets = _finite_matrix(target_returns, name="target_returns")
    if len(instruments) != len(targets):
        raise ValueError(
            "instrument_returns and target_returns must have the same samples"
        )
    if len(instruments) < 2:
        raise ValueError("at least two scenarios are required")
    try:
        reduction = MeanReduction(method)
    except ValueError as error:
        raise ValueError(f"unknown mean-reduction method {method!r}") from error

    observations = len(instruments)
    uniform_weights = np.full(observations, 1.0 / observations, dtype=np.float64)
    scales = _root_mean_square_scales(instruments, minimum=config.minimum_scale)
    standardized = instruments / scales
    constraint_rank = _matrix_rank(standardized, tolerance=config.rank_tolerance)
    raw_standardized_mean = np.mean(standardized, axis=0)
    raw_max_standardized_mean = _max_abs(raw_standardized_mean)

    if reduction is MeanReduction.DEMEAN:
        instrument_mean = np.mean(instruments, axis=0)
        target_mean = np.mean(targets, axis=0)
        reduced_instruments = instruments - instrument_mean
        reduced_targets = targets - target_mean
        reduced_residual = _max_abs(
            np.mean(reduced_instruments / scales, axis=0)
        )
        diagnostics = MeanReductionDiagnostics(
            method=reduction,
            observations=observations,
            instruments=instruments.shape[1],
            targets=targets.shape[1],
            constraint_rank=constraint_rank,
            raw_max_standardized_mean=raw_max_standardized_mean,
            reduced_max_standardized_mean=reduced_residual,
            effective_sample_size=float(observations),
            kl_divergence=0.0,
            minimum_weight=float(uniform_weights[0]),
            maximum_weight=float(uniform_weights[0]),
            moment_residual=reduced_residual,
            iterations=0,
            converged=True,
        )
        return MeanReductionResult(
            instrument_returns=reduced_instruments,
            target_returns=reduced_targets,
            weights=uniform_weights,
            instrument_mean_removed=instrument_mean,
            target_mean_removed=target_mean,
            diagnostics=diagnostics,
        )

    reduced_constraints, constraint_rank = _rank_reduced_constraints(
        standardized,
        tolerance=config.rank_tolerance,
    )
    if constraint_rank == 0:
        weights = uniform_weights
        iterations = 0
    else:
        if observations <= config.feasibility_check_maximum_samples:
            _verify_strict_feasibility(reduced_constraints, config=config)
        try:
            weights, iterations = _solve_memm(reduced_constraints, config=config)
        except MEMMConvergenceError:
            if observations > config.feasibility_check_maximum_samples:
                _verify_strict_feasibility(reduced_constraints, config=config)
            raise
        if observations > config.feasibility_check_maximum_samples:
            relative_minimum_weight = observations * float(np.min(weights))
            if relative_minimum_weight <= config.strict_feasibility_relative_weight:
                _verify_strict_feasibility(reduced_constraints, config=config)

    instrument_mean = weights @ instruments
    target_mean = weights @ targets
    reduced_targets = targets - target_mean
    standardized_residual = weights @ standardized
    moment_residual = _max_abs(standardized_residual)
    if moment_residual > config.moment_tolerance:
        raise MEMMConvergenceError(
            "MEMM weighted instrument mean exceeds tolerance: "
            f"{moment_residual:.3e} > {config.moment_tolerance:.3e}"
        )
    positive = weights > 0.0
    kl_divergence = float(
        np.dot(weights[positive], np.log(observations * weights[positive]))
    )
    diagnostics = MeanReductionDiagnostics(
        method=reduction,
        observations=observations,
        instruments=instruments.shape[1],
        targets=targets.shape[1],
        constraint_rank=constraint_rank,
        raw_max_standardized_mean=raw_max_standardized_mean,
        reduced_max_standardized_mean=moment_residual,
        effective_sample_size=float(1.0 / np.dot(weights, weights)),
        kl_divergence=kl_divergence,
        minimum_weight=float(np.min(weights)),
        maximum_weight=float(np.max(weights)),
        moment_residual=moment_residual,
        iterations=iterations,
        converged=True,
    )
    return MeanReductionResult(
        instrument_returns=instruments.copy(),
        target_returns=reduced_targets,
        weights=weights,
        instrument_mean_removed=instrument_mean,
        target_mean_removed=target_mean,
        diagnostics=diagnostics,
    )


def _finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _root_mean_square_scales(values: FloatArray, *, minimum: float) -> FloatArray:
    return np.maximum(np.sqrt(np.mean(np.square(values), axis=0)), minimum)


def _matrix_rank(values: FloatArray, *, tolerance: float) -> int:
    singular_values = np.linalg.svd(values, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > tolerance * singular_values[0]))


def _rank_reduced_constraints(
    standardized: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, int]:
    left, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        return np.empty((len(standardized), 0), dtype=np.float64), 0
    keep = singular_values > tolerance * singular_values[0]
    rank = int(np.count_nonzero(keep))
    # Multiplication by sqrt(n) gives every retained constraint unit RMS scale.
    constraints = left[:, keep] * math.sqrt(len(standardized))
    return np.asarray(constraints, dtype=np.float64), rank


def _verify_strict_feasibility(
    constraints: FloatArray,
    *,
    config: MeanReductionConfig,
) -> None:
    observations = len(constraints)
    equality_weights = csr_matrix(
        np.vstack((np.ones(observations, dtype=np.float64), constraints.T))
    )
    equality = hstack(
        (equality_weights, csr_matrix((constraints.shape[1] + 1, 1))),
        format="csr",
    )
    right_hand_side = np.zeros(constraints.shape[1] + 1, dtype=np.float64)
    right_hand_side[0] = 1.0
    objective = np.zeros(observations + 1, dtype=np.float64)
    objective[-1] = -1.0
    # Maximize the smallest scenario probability. A positive optimum is
    # equivalent to the origin lying in the relative interior of the hull.
    lower_weight = hstack(
        (-eye(observations, format="csr"), np.ones((observations, 1))),
        format="csr",
    )
    feasible = linprog(
        objective,
        A_eq=equality,
        b_eq=right_hand_side,
        A_ub=lower_weight,
        b_ub=np.zeros(observations, dtype=np.float64),
        bounds=[(0.0, None)] * observations + [(0.0, 1.0 / observations)],
        method="highs",
    )
    if not feasible.success:
        raise MEMMInfeasibleError(
            "no scenario probabilities make all instrument means zero"
        )
    relative_minimum_weight = observations * float(feasible.x[-1])
    if relative_minimum_weight <= config.strict_feasibility_relative_weight:
        raise MEMMBoundaryError(
            "zero instrument means require zero probability on at least one scenario"
        )


def _solve_memm(
    constraints: FloatArray,
    *,
    config: MeanReductionConfig,
) -> tuple[FloatArray, int]:
    parameters = np.zeros(constraints.shape[1], dtype=np.float64)
    objective = math.log(len(constraints))
    for iteration in range(1, config.maximum_iterations + 1):
        logits = constraints @ parameters
        log_normalizer = float(logsumexp(logits))
        weights = np.exp(logits - log_normalizer)
        gradient = weights @ constraints
        if _max_abs(gradient) <= config.moment_tolerance:
            return np.asarray(weights, dtype=np.float64), iteration

        centered = constraints - gradient
        hessian = (centered * weights[:, None]).T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        positive_floor = max(config.minimum_scale, eigenvalues[-1] * 1.0e-14)
        inverse_eigenvalues = 1.0 / np.maximum(eigenvalues, positive_floor)
        direction = -eigenvectors @ (inverse_eigenvalues * (eigenvectors.T @ gradient))
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise MEMMConvergenceError("MEMM Newton direction is not a descent direction")

        step = 1.0
        accepted = False
        while step >= config.line_search_minimum_step:
            candidate = parameters + step * direction
            candidate_objective = float(logsumexp(constraints @ candidate))
            if candidate_objective <= objective + 1.0e-4 * step * directional_derivative:
                parameters = candidate
                objective = candidate_objective
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise MEMMConvergenceError("MEMM line search failed")

    raise MEMMConvergenceError(
        f"MEMM did not converge in {config.maximum_iterations} iterations"
    )


def _max_abs(values: FloatArray) -> float:
    return float(np.max(np.abs(values), initial=0.0))
