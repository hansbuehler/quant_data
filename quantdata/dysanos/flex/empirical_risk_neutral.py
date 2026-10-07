"""Minimum-relative-entropy measures on a fixed empirical return population.

The routines in this module fit one density on a *fixed* set of paths.  They do
not resimulate paths and they do not estimate conditional martingale
restrictions.  For path-return matrix ``R`` and symmetric cost bands ``c``, the
primal problem is

``min mean(w * log(w))`` subject to ``mean(w) = 1`` and
``abs(mean(w[:, None] * R, axis=0)) <= c``.

The equivalent signed dual is ``log(mean(exp(-R @ a))) + c @ abs(a)``.  Since
the absolute value is nonsmooth, the implementation uses an exact active-set
method.  On each fixed orthant, ``a = sign * magnitude / scale`` and the
nonnegative magnitudes form a smooth bound-constrained dual.  Outer KKT passes
add violated inactive bands and drop or flip multipliers that reach zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from scipy.optimize import OptimizeResult, minimize


class ReturnMatrix(Protocol):
    """Array-like path returns accepted by the streaming solver."""

    shape: tuple[int, ...]
    ndim: int

    def __getitem__(self, key: slice) -> np.ndarray: ...


@dataclass(frozen=True, kw_only=True)
class EmpiricalRiskNeutralFit:
    """A KL/I-projection of an empirical path population.

    ``weights`` are densities relative to the uniform empirical training
    measure, not probabilities.  Consequently their sample mean is one and
    the probability of path ``i`` under the fitted measure is
    ``weights[i] / num_paths``.
    """

    action: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray
    weighted_mean: np.ndarray
    cost_band: np.ndarray
    primal_relative_entropy: float
    dual_objective: float
    primal_dual_gap: float
    effective_sample_size: float
    effective_sample_fraction: float
    maximum_constraint_violation: float
    maximum_kkt_residual: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    objective_evaluations: int
    active_coordinates: int
    active_set_iterations: int


@dataclass(frozen=True, kw_only=True)
class _ColumnStatistics:
    mean: np.ndarray
    standard_deviation: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def _matrix_shape(returns: ReturnMatrix) -> tuple[int, int]:
    if returns.ndim not in {2, 3}:
        raise ValueError("returns must have shape (paths,coordinates) or (paths,steps,instruments)")
    if any(not isinstance(size, (int, np.integer)) or size <= 0 for size in returns.shape):
        raise ValueError(f"returns has an invalid shape {returns.shape}")
    paths = int(returns.shape[0])
    coordinates = int(math.prod(returns.shape[1:]))
    if paths < 2:
        raise ValueError("at least two paths are required")
    return paths, coordinates


def _return_chunk(
    returns: ReturnMatrix,
    first: int,
    last: int,
    coordinates: int,
) -> np.ndarray:
    values = np.asarray(returns[first:last], dtype=np.float64)
    values = values.reshape(last - first, coordinates)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"returns contain a non-finite value in paths [{first}, {last})")
    return values


def _column_statistics(
    returns: ReturnMatrix,
    *,
    chunk_size: int,
) -> _ColumnStatistics:
    paths, coordinates = _matrix_shape(returns)
    total = np.zeros(coordinates, dtype=np.float64)
    total_square = np.zeros(coordinates, dtype=np.float64)
    minimum = np.full(coordinates, np.inf, dtype=np.float64)
    maximum = np.full(coordinates, -np.inf, dtype=np.float64)
    for first in range(0, paths, chunk_size):
        last = min(first + chunk_size, paths)
        values = _return_chunk(returns, first, last, coordinates)
        total += np.sum(values, axis=0, dtype=np.float64)
        total_square += np.einsum("ij,ij->j", values, values)
        minimum = np.minimum(minimum, np.min(values, axis=0))
        maximum = np.maximum(maximum, np.max(values, axis=0))
    mean = total / paths
    variance = np.maximum(total_square / paths - np.square(mean), 0.0)
    return _ColumnStatistics(
        mean=mean,
        standard_deviation=np.sqrt(variance),
        minimum=minimum,
        maximum=maximum,
    )


class _OrthantDual:
    """Streaming smooth objective for one active signed orthant."""

    def __init__(
        self,
        returns: ReturnMatrix,
        cost_band: np.ndarray,
        scale: np.ndarray,
        active_indices: np.ndarray,
        active_signs: np.ndarray,
        *,
        chunk_size: int,
    ) -> None:
        self.returns = returns
        self.paths, self.coordinates = _matrix_shape(returns)
        self.cost_band = cost_band
        self.scale = scale
        self.scaled_cost = cost_band / scale
        self.active_indices = active_indices
        self.active_signs = active_signs
        self.chunk_size = chunk_size
        self.evaluations = 0
        self.last_weighted_mean = np.empty(self.coordinates, dtype=np.float64)
        self.last_log_normalizer = math.nan

    def __call__(self, magnitude: np.ndarray) -> tuple[float, np.ndarray]:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        expected_shape = (len(self.active_indices),)
        if magnitude.shape != expected_shape or not np.all(np.isfinite(magnitude)):
            raise ValueError(
                f"orthant magnitude must be a finite vector with shape {expected_shape}"
            )
        action = np.zeros(self.coordinates, dtype=np.float64)
        action[self.active_indices] = (
            self.active_signs * magnitude / self.scale[self.active_indices]
        )

        # Online log-sum-exp also carries the weighted return numerator.  Each
        # chunk is shifted before exponentiation, so arbitrarily dispersed
        # finite path scores do not overflow.
        score_maximum = -np.inf
        exponential_sum = 0.0
        weighted_return_sum = np.zeros(self.coordinates, dtype=np.float64)
        for first in range(0, self.paths, self.chunk_size):
            last = min(first + self.chunk_size, self.paths)
            values = _return_chunk(self.returns, first, last, self.coordinates)
            scores = -(values @ action)
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("dual path scores became non-finite")
            chunk_maximum = float(np.max(scores))
            chunk_exponential = np.exp(scores - chunk_maximum)
            chunk_sum = float(np.sum(chunk_exponential, dtype=np.float64))
            chunk_weighted_returns = values.T @ chunk_exponential
            if chunk_maximum > score_maximum:
                old_scale = 0.0 if score_maximum == -np.inf else math.exp(
                    score_maximum - chunk_maximum
                )
                exponential_sum = old_scale * exponential_sum + chunk_sum
                weighted_return_sum = (
                    old_scale * weighted_return_sum + chunk_weighted_returns
                )
                score_maximum = chunk_maximum
            else:
                new_scale = math.exp(chunk_maximum - score_maximum)
                exponential_sum += new_scale * chunk_sum
                weighted_return_sum += new_scale * chunk_weighted_returns

        log_normalizer = score_maximum + math.log(exponential_sum) - math.log(self.paths)
        weighted_mean = weighted_return_sum / exponential_sum
        value = log_normalizer + float(
            self.scaled_cost[self.active_indices] @ magnitude
        )
        gradient = (
            self.cost_band[self.active_indices]
            - self.active_signs * weighted_mean[self.active_indices]
        ) / self.scale[self.active_indices]
        self.evaluations += 1
        self.last_weighted_mean = weighted_mean
        self.last_log_normalizer = log_normalizer
        return value, gradient


class _TorchOrthantDual:
    """Full-population smooth orthant objective evaluated on one Torch device."""

    def __init__(
        self,
        returns: torch.Tensor,
        cost_band: np.ndarray,
        scale: np.ndarray,
        active_indices: np.ndarray,
        active_signs: np.ndarray,
    ) -> None:
        if returns.ndim != 2:
            raise ValueError("Torch returns must be a path-by-coordinate matrix")
        self.returns = returns
        self.paths, self.coordinates = returns.shape
        self.cost_band = cost_band
        self.scale = scale
        self.active_indices = active_indices
        self.active_signs = active_signs
        self.active_indices_tensor = torch.as_tensor(
            active_indices, dtype=torch.int64, device=returns.device
        )
        self.active_signs_tensor = torch.as_tensor(
            active_signs, dtype=returns.dtype, device=returns.device
        )
        self.active_scale_tensor = torch.as_tensor(
            scale[active_indices], dtype=returns.dtype, device=returns.device
        )
        self.scaled_active_cost = torch.as_tensor(
            cost_band[active_indices] / scale[active_indices],
            dtype=returns.dtype,
            device=returns.device,
        )
        self.evaluations = 0
        self.last_weighted_mean = np.empty(self.coordinates, dtype=np.float64)
        self.last_log_normalizer = math.nan

    def __call__(self, magnitude: np.ndarray) -> tuple[float, np.ndarray]:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        expected_shape = (len(self.active_indices),)
        if magnitude.shape != expected_shape or not np.all(np.isfinite(magnitude)):
            raise ValueError(
                f"orthant magnitude must be a finite vector with shape {expected_shape}"
            )
        magnitude_tensor = torch.as_tensor(
            magnitude, dtype=self.returns.dtype, device=self.returns.device
        )
        action = torch.zeros(
            self.coordinates,
            dtype=self.returns.dtype,
            device=self.returns.device,
        )
        action[self.active_indices_tensor] = (
            self.active_signs_tensor * magnitude_tensor / self.active_scale_tensor
        )
        scores = -(self.returns @ action)
        log_normalizer = torch.logsumexp(scores, dim=0) - math.log(self.paths)
        probability = torch.softmax(scores, dim=0)
        weighted_mean = self.returns.T @ probability
        value = log_normalizer + torch.sum(
            self.scaled_active_cost * magnitude_tensor
        )
        gradient = (
            torch.as_tensor(
                self.cost_band[self.active_indices],
                dtype=self.returns.dtype,
                device=self.returns.device,
            )
            - self.active_signs_tensor * weighted_mean[self.active_indices_tensor]
        ) / self.active_scale_tensor
        if not bool(torch.isfinite(value)) or not bool(torch.all(torch.isfinite(gradient))):
            raise FloatingPointError("Torch dual objective became non-finite")
        self.evaluations += 1
        self.last_weighted_mean = weighted_mean.detach().cpu().double().numpy()
        self.last_log_normalizer = float(log_normalizer.detach().cpu())
        return float(value.detach().cpu()), gradient.detach().cpu().double().numpy()


class _ZeroCostDual:
    """Streamed smooth entropy dual with exact covariance Hessian products."""

    def __init__(
        self,
        returns: ReturnMatrix,
        scale: np.ndarray,
        *,
        chunk_size: int,
    ) -> None:
        self.returns = returns
        self.paths, self.coordinates = _matrix_shape(returns)
        self.scale = scale
        self.chunk_size = chunk_size
        self.evaluations = 0
        self.hessian_evaluations = 0
        self._scaled_action: np.ndarray | None = None
        self._probability = np.empty(self.paths, dtype=np.float64)
        self._scaled_mean = np.empty(self.coordinates, dtype=np.float64)
        self._value = math.nan

    def __call__(self, scaled_action: np.ndarray) -> tuple[float, np.ndarray]:
        scaled_action = np.asarray(scaled_action, dtype=np.float64)
        if scaled_action.shape != (self.coordinates,) or not np.all(
            np.isfinite(scaled_action)
        ):
            raise ValueError("scaled zero-cost action is invalid")
        if self._scaled_action is not None and np.array_equal(
            scaled_action,
            self._scaled_action,
        ):
            return self._value, -self._scaled_mean

        action = scaled_action / self.scale
        scores = self._probability
        for first in range(0, self.paths, self.chunk_size):
            last = min(first + self.chunk_size, self.paths)
            values = _return_chunk(self.returns, first, last, self.coordinates)
            scores[first:last] = -(values @ action)
        maximum = float(np.max(scores))
        np.exp(scores - maximum, out=scores)
        exponential_sum = float(np.sum(scores, dtype=np.float64))
        scores /= exponential_sum

        weighted_mean = np.zeros(self.coordinates, dtype=np.float64)
        for first in range(0, self.paths, self.chunk_size):
            last = min(first + self.chunk_size, self.paths)
            values = _return_chunk(self.returns, first, last, self.coordinates)
            weighted_mean += values.T @ scores[first:last]
        self._scaled_mean = weighted_mean / self.scale
        self._value = maximum + math.log(exponential_sum) - math.log(self.paths)
        self._scaled_action = scaled_action.copy()
        self.evaluations += 1
        return self._value, -self._scaled_mean

    def hessian_product(
        self,
        scaled_action: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        self(scaled_action)
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (self.coordinates,) or not np.all(np.isfinite(direction)):
            raise ValueError("zero-cost Hessian direction is invalid")
        action_direction = direction / self.scale
        result = np.zeros(self.coordinates, dtype=np.float64)
        for first in range(0, self.paths, self.chunk_size):
            last = min(first + self.chunk_size, self.paths)
            values = _return_chunk(self.returns, first, last, self.coordinates)
            projected = values @ action_direction
            result += values.T @ (self._probability[first:last] * projected)
        self.hessian_evaluations += 1
        return (
            result / self.scale
            - self._scaled_mean * float(self._scaled_mean @ direction)
            + 1.0e-10 * direction
        )


class _TorchZeroCostDual:
    """Device-resident zero-cost dual and covariance Hessian products."""

    def __init__(self, returns: torch.Tensor, scale: np.ndarray) -> None:
        if returns.ndim != 2:
            raise ValueError("Torch returns must be a path-by-coordinate matrix")
        self.returns = returns
        self.paths, self.coordinates = returns.shape
        self.scale = torch.as_tensor(scale, dtype=returns.dtype, device=returns.device)
        self.evaluations = 0
        self.hessian_evaluations = 0
        self._scaled_action: np.ndarray | None = None
        self._probability = torch.empty(
            self.paths,
            dtype=returns.dtype,
            device=returns.device,
        )
        self._scaled_mean = torch.empty(
            self.coordinates,
            dtype=returns.dtype,
            device=returns.device,
        )
        self._value = math.nan

    def __call__(self, scaled_action: np.ndarray) -> tuple[float, np.ndarray]:
        scaled_action = np.asarray(scaled_action, dtype=np.float64)
        if scaled_action.shape != (self.coordinates,) or not np.all(
            np.isfinite(scaled_action)
        ):
            raise ValueError("scaled zero-cost action is invalid")
        if self._scaled_action is not None and np.array_equal(
            scaled_action,
            self._scaled_action,
        ):
            return self._value, -self._scaled_mean.cpu().double().numpy()

        scaled_action_tensor = torch.as_tensor(
            scaled_action,
            dtype=self.returns.dtype,
            device=self.returns.device,
        )
        scores = -(self.returns @ (scaled_action_tensor / self.scale))
        log_normalizer = torch.logsumexp(scores, dim=0) - math.log(self.paths)
        self._probability = torch.softmax(scores, dim=0)
        weighted_mean = self.returns.T @ self._probability
        self._scaled_mean = weighted_mean / self.scale
        self._value = float(log_normalizer.cpu())
        self._scaled_action = scaled_action.copy()
        self.evaluations += 1
        return self._value, -self._scaled_mean.cpu().double().numpy()

    def hessian_product(
        self,
        scaled_action: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        self(scaled_action)
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (self.coordinates,) or not np.all(np.isfinite(direction)):
            raise ValueError("zero-cost Hessian direction is invalid")
        direction_tensor = torch.as_tensor(
            direction,
            dtype=self.returns.dtype,
            device=self.returns.device,
        )
        projected = self.returns @ (direction_tensor / self.scale)
        result = self.returns.T @ (self._probability * projected) / self.scale
        result -= self._scaled_mean * torch.dot(self._scaled_mean, direction_tensor)
        result += 1.0e-10 * direction_tensor
        self.hessian_evaluations += 1
        return result.cpu().double().numpy()


def _upload_return_matrix(
    returns: ReturnMatrix,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> torch.Tensor:
    """Upload a finite return population without first copying the memmap in RAM."""

    paths, coordinates = _matrix_shape(returns)
    values = torch.empty((paths, coordinates), dtype=dtype, device=device)
    for first in range(0, paths, chunk_size):
        last = min(first + chunk_size, paths)
        chunk = np.asarray(returns[first:last]).reshape(last - first, coordinates)
        if not np.all(np.isfinite(chunk)):
            raise ValueError(
                f"returns contain a non-finite value in paths [{first}, {last})"
            )
        values[first:last].copy_(torch.as_tensor(np.array(chunk, copy=True)))
    return values


def _normalized_density(
    returns: ReturnMatrix,
    action: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    paths, coordinates = _matrix_shape(returns)
    scores = np.empty(paths, dtype=np.float64)
    for first in range(0, paths, chunk_size):
        last = min(first + chunk_size, paths)
        values = _return_chunk(returns, first, last, coordinates)
        scores[first:last] = -(values @ action)
    maximum = float(np.max(scores))
    log_normalizer = maximum + math.log(
        float(np.sum(np.exp(scores - maximum), dtype=np.float64))
    ) - math.log(paths)
    log_weights = scores - log_normalizer

    # Correct the final few ulps in the density mass before producing weights.
    log_weight_maximum = float(np.max(log_weights))
    log_mass = log_weight_maximum + math.log(
        float(np.sum(np.exp(log_weights - log_weight_maximum), dtype=np.float64))
    ) - math.log(paths)
    log_weights -= log_mass
    weights = np.exp(log_weights)
    return log_weights, weights, log_normalizer + log_mass


def _weighted_mean(
    returns: ReturnMatrix,
    weights: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    paths, coordinates = _matrix_shape(returns)
    total = np.zeros(coordinates, dtype=np.float64)
    for first in range(0, paths, chunk_size):
        last = min(first + chunk_size, paths)
        values = _return_chunk(returns, first, last, coordinates)
        total += values.T @ weights[first:last]
    return total / paths


def _fit_zero_cost_dual(
    returns: ReturnMatrix,
    scale: np.ndarray,
    dual: _ZeroCostDual | _TorchZeroCostDual,
    *,
    chunk_size: int,
    maximum_iterations: int,
    moment_tolerance: float,
    optimizer_gradient_tolerance: float,
    solver_label: str,
) -> EmpiricalRiskNeutralFit:
    """Solve the smooth equality-constrained entropy dual without orthants."""

    paths, coordinates = _matrix_shape(returns)
    optimizer: OptimizeResult = minimize(
        dual,
        np.zeros(coordinates, dtype=np.float64),
        method="trust-ncg",
        jac=True,
        hessp=dual.hessian_product,
        options={
            "maxiter": maximum_iterations,
            "gtol": optimizer_gradient_tolerance,
        },
    )
    action = np.asarray(optimizer.x, dtype=np.float64) / scale
    log_weights, weights, log_normalizer = _normalized_density(
        returns,
        action,
        chunk_size=chunk_size,
    )
    weighted_mean = _weighted_mean(
        returns,
        weights,
        chunk_size=chunk_size,
    )
    maximum_kkt_residual = float(np.max(np.abs(weighted_mean)))
    dual_objective = log_normalizer
    primal_relative_entropy = float(np.mean(weights * log_weights))
    optimizer_success = maximum_kkt_residual <= moment_tolerance
    optimizer_status = 0 if optimizer_success else int(optimizer.status)
    optimizer_message = (
        f"{solver_label} unconstrained zero-cost KKT conditions satisfied"
        if optimizer_success
        else str(optimizer.message)
    )
    effective_sample_size = float(
        paths**2 / np.sum(np.square(weights), dtype=np.float64)
    )
    return EmpiricalRiskNeutralFit(
        action=action,
        weights=weights,
        log_weights=log_weights,
        weighted_mean=weighted_mean,
        cost_band=np.zeros(coordinates, dtype=np.float64),
        primal_relative_entropy=primal_relative_entropy,
        dual_objective=dual_objective,
        primal_dual_gap=primal_relative_entropy + dual_objective,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / paths,
        maximum_constraint_violation=maximum_kkt_residual,
        maximum_kkt_residual=maximum_kkt_residual,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        optimizer_iterations=int(optimizer.nit),
        objective_evaluations=dual.evaluations,
        active_coordinates=int(np.sum(np.abs(action) > 1.0e-12)),
        active_set_iterations=0,
    )


def _orthant_kkt_residual(
    signs: np.ndarray,
    scaled_magnitude: np.ndarray,
    weighted_mean: np.ndarray,
    cost_band: np.ndarray,
    *,
    magnitude_tolerance: float = 1.0e-12,
) -> np.ndarray:
    active = (signs != 0) & (scaled_magnitude > magnitude_tolerance)
    residual = np.maximum(np.abs(weighted_mean) - cost_band, 0.0)
    residual[active] = np.abs(
        weighted_mean[active] - signs[active] * cost_band[active]
    )
    return residual


def fit_empirical_risk_neutral_density(
    returns: ReturnMatrix,
    cost_band: np.ndarray,
    *,
    chunk_size: int = 32_768,
    maximum_iterations: int = 2_000,
    moment_tolerance: float = 1.0e-8,
    optimizer_gradient_tolerance: float = 1.0e-10,
    optimizer_function_tolerance: float = 1.0e-15,
    maximum_line_search_steps: int = 50,
    maximum_active_set_iterations: int = 50,
) -> EmpiricalRiskNeutralFit:
    r"""Fit the empirical KL/I-projection into symmetric moment bands.

    Parameters
    ----------
    returns:
        Gross one-step returns with shape ``(paths, coordinates)`` or
        ``(paths, steps, instruments)``.  Three-dimensional input is flattened
        step-major in C order.  Array slices are consumed in chunks, allowing a
        ``numpy.memmap`` input without a dense copy.
    cost_band:
        Nonnegative half-width of every gross-return moment band.  This is in
        return units: one basis point is ``1e-4``.
    chunk_size:
        Number of paths converted to float64 for each streamed matrix product.

    Returns
    -------
    EmpiricalRiskNeutralFit
        The density and numerical/financial optimality diagnostics.  A fit is
        marked successful only if the raw return-moment KKT residual is no
        larger than ``moment_tolerance``.  This KKT certificate is sufficient
        for the convex problem even if the final inner optimizer reports a
        non-success termination after reaching that tolerance.
    """

    paths, coordinates = _matrix_shape(returns)
    cost_band = np.asarray(cost_band, dtype=np.float64)
    if cost_band.shape != (coordinates,):
        raise ValueError(
            f"cost_band must have shape ({coordinates},), found {cost_band.shape}"
        )
    if not np.all(np.isfinite(cost_band) & (cost_band >= 0.0)):
        raise ValueError("cost_band must be finite and nonnegative")
    integer_controls = (
        chunk_size,
        maximum_iterations,
        maximum_line_search_steps,
        maximum_active_set_iterations,
    )
    if any(not isinstance(value, int) or value <= 0 for value in integer_controls):
        raise ValueError("chunk and optimizer iteration controls must be positive integers")
    tolerances = (
        moment_tolerance,
        optimizer_gradient_tolerance,
        optimizer_function_tolerance,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("solver tolerances must be finite and positive")

    statistics = _column_statistics(returns, chunk_size=chunk_size)
    impossible_above = statistics.minimum > cost_band
    impossible_below = statistics.maximum < -cost_band
    if np.any(impossible_above | impossible_below):
        indices = np.flatnonzero(impossible_above | impossible_below).tolist()
        raise ValueError(
            "no feasible density exists because the cost band misses the empirical "
            f"range in coordinates {indices}"
        )

    # Scaling changes only the numerical parameterization, not the fitted
    # density.  The cost floor keeps nearly deterministic columns well scaled.
    scale = np.maximum.reduce(
        (
            statistics.standard_deviation,
            cost_band,
            np.full(coordinates, 1.0e-8, dtype=np.float64),
        )
    )
    if not np.any(cost_band):
        dual = _ZeroCostDual(
            returns,
            scale,
            chunk_size=chunk_size,
        )
        return _fit_zero_cost_dual(
            returns,
            scale,
            dual,
            chunk_size=chunk_size,
            maximum_iterations=maximum_iterations,
            moment_tolerance=moment_tolerance,
            optimizer_gradient_tolerance=optimizer_gradient_tolerance,
            solver_label="CPU trust-region Newton-CG",
        )
    signs = np.zeros(coordinates, dtype=np.int8)
    signs[statistics.mean > cost_band] = 1
    signs[statistics.mean < -cost_band] = -1
    scaled_magnitude = np.zeros(coordinates, dtype=np.float64)
    total_iterations = 0
    total_evaluations = 0
    active_set_iterations = 0
    optimizer_status = 0
    optimizer_message = "identity empirical measure satisfies every cost band"
    inner_success = True
    final_weighted_mean = statistics.mean

    # Every outer pass solves one smooth orthant exactly enough for its KKT
    # residual, then changes the orthant only at zero or at a violated inactive
    # coordinate.  The global signed objective remains convex throughout.
    for active_set_iteration in range(1, maximum_active_set_iterations + 1):
        active_set_iterations = active_set_iteration
        active_indices = np.flatnonzero(signs)
        if not len(active_indices):
            final_weighted_mean = statistics.mean
            break
        active_signs = signs[active_indices].astype(np.float64)
        dual = _OrthantDual(
            returns,
            cost_band,
            scale,
            active_indices,
            active_signs,
            chunk_size=chunk_size,
        )
        optimizer: OptimizeResult = minimize(
            dual,
            scaled_magnitude[active_indices],
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * len(active_indices),
            options={
                "maxiter": maximum_iterations,
                "ftol": optimizer_function_tolerance,
                "gtol": optimizer_gradient_tolerance,
                "maxls": maximum_line_search_steps,
                "maxcor": min(50, len(active_indices)),
            },
        )
        total_iterations += int(optimizer.nit)
        scaled_magnitude[active_indices] = np.asarray(optimizer.x, dtype=np.float64)
        _, _ = dual(scaled_magnitude[active_indices])
        total_evaluations += dual.evaluations
        final_weighted_mean = dual.last_weighted_mean
        optimizer_status = int(optimizer.status)
        optimizer_message = str(optimizer.message)
        inner_success = bool(optimizer.success)

        reached_zero = (signs != 0) & (scaled_magnitude <= 1.0e-12)
        signs[reached_zero] = 0
        scaled_magnitude[reached_zero] = 0.0
        inactive = signs == 0
        violated_inactive = inactive & (
            np.abs(final_weighted_mean) - cost_band > moment_tolerance
        )
        if np.any(violated_inactive):
            signs[violated_inactive] = np.where(
                final_weighted_mean[violated_inactive] > 0.0,
                1,
                -1,
            )
        orthant_changed = bool(np.any(reached_zero | violated_inactive))
        current_kkt = float(
            np.max(
                _orthant_kkt_residual(
                    signs,
                    scaled_magnitude,
                    final_weighted_mean,
                    cost_band,
                )
            )
        )
        if not orthant_changed and current_kkt <= moment_tolerance:
            break
        if not orthant_changed and not optimizer.success:
            break
    else:
        optimizer_status = 1
        optimizer_message = "active-set orthant updates did not stabilize"
        inner_success = False

    action = signs.astype(np.float64) * scaled_magnitude / scale
    log_weights, weights, log_normalizer = _normalized_density(
        returns,
        action,
        chunk_size=chunk_size,
    )
    weighted_mean = _weighted_mean(
        returns,
        weights,
        chunk_size=chunk_size,
    )
    constraint_violation = np.maximum(np.abs(weighted_mean) - cost_band, 0.0)
    kkt_residual = _orthant_kkt_residual(
        signs,
        scaled_magnitude,
        weighted_mean,
        cost_band,
    )
    maximum_constraint_violation = float(np.max(constraint_violation))
    maximum_kkt_residual = float(np.max(kkt_residual))
    dual_objective = log_normalizer + float(cost_band @ np.abs(action))
    primal_relative_entropy = float(np.mean(weights * log_weights))
    primal_dual_gap = primal_relative_entropy + dual_objective
    effective_sample_size = float(paths**2 / np.sum(np.square(weights), dtype=np.float64))
    optimizer_success = maximum_kkt_residual <= moment_tolerance
    if optimizer_success:
        optimizer_status = 0
        optimizer_message = (
            "active-set KKT conditions satisfied"
            if inner_success
            else "active-set KKT conditions satisfied after inner solver termination"
        )

    return EmpiricalRiskNeutralFit(
        action=action,
        weights=weights,
        log_weights=log_weights,
        weighted_mean=weighted_mean,
        cost_band=cost_band.copy(),
        primal_relative_entropy=primal_relative_entropy,
        dual_objective=dual_objective,
        primal_dual_gap=primal_dual_gap,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / paths,
        maximum_constraint_violation=maximum_constraint_violation,
        maximum_kkt_residual=maximum_kkt_residual,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        optimizer_iterations=total_iterations,
        objective_evaluations=total_evaluations,
        active_coordinates=int(np.sum(signs != 0)),
        active_set_iterations=active_set_iterations,
    )


def fit_empirical_risk_neutral_density_torch(
    returns: ReturnMatrix,
    cost_band: np.ndarray,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.float64,
    upload_chunk_size: int = 32_768,
    final_chunk_size: int = 32_768,
    maximum_iterations: int = 2_000,
    moment_tolerance: float = 1.0e-8,
    optimizer_gradient_tolerance: float = 1.0e-10,
    optimizer_function_tolerance: float = 1.0e-15,
    maximum_line_search_steps: int = 50,
    maximum_active_set_iterations: int = 50,
) -> EmpiricalRiskNeutralFit:
    r"""Fit the same empirical KL projection using GPU dual evaluations.

    The complete fixed return matrix is uploaded once.  SciPy controls the
    low-dimensional bound-constrained active-set iterations, while Torch
    evaluates the exact full-population log-sum-exp objective and analytic
    gradient.  Final weights and KKT diagnostics are recomputed from the
    original matrix in streamed NumPy float64 arithmetic so reports remain
    directly comparable with :func:`fit_empirical_risk_neutral_density`.
    """

    paths, coordinates = _matrix_shape(returns)
    cost_band = np.asarray(cost_band, dtype=np.float64)
    if cost_band.shape != (coordinates,):
        raise ValueError(
            f"cost_band must have shape ({coordinates},), found {cost_band.shape}"
        )
    if not np.all(np.isfinite(cost_band) & (cost_band >= 0.0)):
        raise ValueError("cost_band must be finite and nonnegative")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Torch empirical solver dtype must be float32 or float64")
    integer_controls = (
        upload_chunk_size,
        final_chunk_size,
        maximum_iterations,
        maximum_line_search_steps,
        maximum_active_set_iterations,
    )
    if any(not isinstance(value, int) or value <= 0 for value in integer_controls):
        raise ValueError("chunk and optimizer iteration controls must be positive integers")
    tolerances = (
        moment_tolerance,
        optimizer_gradient_tolerance,
        optimizer_function_tolerance,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("solver tolerances must be finite and positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {torch_device}")

    device_returns = _upload_return_matrix(
        returns,
        device=torch_device,
        dtype=dtype,
        chunk_size=upload_chunk_size,
    )
    with torch.no_grad():
        mean_tensor = torch.mean(device_returns, dim=0)
        standard_deviation_tensor = torch.std(
            device_returns, dim=0, correction=0
        )
        minimum_tensor = torch.amin(device_returns, dim=0)
        maximum_tensor = torch.amax(device_returns, dim=0)
    statistics = _ColumnStatistics(
        mean=mean_tensor.cpu().double().numpy(),
        standard_deviation=standard_deviation_tensor.cpu().double().numpy(),
        minimum=minimum_tensor.cpu().double().numpy(),
        maximum=maximum_tensor.cpu().double().numpy(),
    )
    impossible_above = statistics.minimum > cost_band
    impossible_below = statistics.maximum < -cost_band
    if np.any(impossible_above | impossible_below):
        indices = np.flatnonzero(impossible_above | impossible_below).tolist()
        raise ValueError(
            "no feasible density exists because the cost band misses the empirical "
            f"range in coordinates {indices}"
        )

    scale = np.maximum.reduce(
        (
            statistics.standard_deviation,
            cost_band,
            np.full(coordinates, 1.0e-8, dtype=np.float64),
        )
    )
    if not np.any(cost_band):
        dual = _TorchZeroCostDual(
            device_returns,
            scale,
        )
        fit = _fit_zero_cost_dual(
            returns,
            scale,
            dual,
            chunk_size=final_chunk_size,
            maximum_iterations=maximum_iterations,
            moment_tolerance=moment_tolerance,
            optimizer_gradient_tolerance=optimizer_gradient_tolerance,
            solver_label=f"{torch_device} trust-region Newton-CG",
        )
        del device_returns
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        return fit
    signs = np.zeros(coordinates, dtype=np.int8)
    signs[statistics.mean > cost_band] = 1
    signs[statistics.mean < -cost_band] = -1
    scaled_magnitude = np.zeros(coordinates, dtype=np.float64)
    total_iterations = 0
    total_evaluations = 0
    active_set_iterations = 0
    optimizer_status = 0
    optimizer_message = "identity empirical measure satisfies every cost band"
    inner_success = True
    final_weighted_mean = statistics.mean

    for active_set_iteration in range(1, maximum_active_set_iterations + 1):
        active_set_iterations = active_set_iteration
        active_indices = np.flatnonzero(signs)
        if not len(active_indices):
            final_weighted_mean = statistics.mean
            break
        active_signs = signs[active_indices].astype(np.float64)
        dual = _TorchOrthantDual(
            device_returns,
            cost_band,
            scale,
            active_indices,
            active_signs,
        )
        optimizer: OptimizeResult = minimize(
            dual,
            scaled_magnitude[active_indices],
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * len(active_indices),
            options={
                "maxiter": maximum_iterations,
                "ftol": optimizer_function_tolerance,
                "gtol": optimizer_gradient_tolerance,
                "maxls": maximum_line_search_steps,
                "maxcor": min(50, len(active_indices)),
            },
        )
        total_iterations += int(optimizer.nit)
        scaled_magnitude[active_indices] = np.asarray(optimizer.x, dtype=np.float64)
        _, _ = dual(scaled_magnitude[active_indices])
        total_evaluations += dual.evaluations
        final_weighted_mean = dual.last_weighted_mean
        optimizer_status = int(optimizer.status)
        optimizer_message = str(optimizer.message)
        inner_success = bool(optimizer.success)

        reached_zero = (signs != 0) & (scaled_magnitude <= 1.0e-12)
        signs[reached_zero] = 0
        scaled_magnitude[reached_zero] = 0.0
        inactive = signs == 0
        violated_inactive = inactive & (
            np.abs(final_weighted_mean) - cost_band > moment_tolerance
        )
        if np.any(violated_inactive):
            signs[violated_inactive] = np.where(
                final_weighted_mean[violated_inactive] > 0.0,
                1,
                -1,
            )
        orthant_changed = bool(np.any(reached_zero | violated_inactive))
        current_kkt = float(
            np.max(
                _orthant_kkt_residual(
                    signs,
                    scaled_magnitude,
                    final_weighted_mean,
                    cost_band,
                )
            )
        )
        if not orthant_changed and current_kkt <= moment_tolerance:
            break
        if not orthant_changed and not optimizer.success:
            break
    else:
        optimizer_status = 1
        optimizer_message = "active-set orthant updates did not stabilize"
        inner_success = False

    action = signs.astype(np.float64) * scaled_magnitude / scale
    del device_returns
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    log_weights, weights, log_normalizer = _normalized_density(
        returns,
        action,
        chunk_size=final_chunk_size,
    )
    weighted_mean = _weighted_mean(
        returns,
        weights,
        chunk_size=final_chunk_size,
    )
    constraint_violation = np.maximum(np.abs(weighted_mean) - cost_band, 0.0)
    kkt_residual = _orthant_kkt_residual(
        signs,
        scaled_magnitude,
        weighted_mean,
        cost_band,
    )
    maximum_constraint_violation = float(np.max(constraint_violation))
    maximum_kkt_residual = float(np.max(kkt_residual))
    dual_objective = log_normalizer + float(cost_band @ np.abs(action))
    primal_relative_entropy = float(np.mean(weights * log_weights))
    primal_dual_gap = primal_relative_entropy + dual_objective
    effective_sample_size = float(paths**2 / np.sum(np.square(weights), dtype=np.float64))
    optimizer_success = maximum_kkt_residual <= moment_tolerance
    if optimizer_success:
        optimizer_status = 0
        optimizer_message = (
            "GPU active-set KKT conditions satisfied"
            if inner_success
            else "GPU active-set KKT conditions satisfied after inner solver termination"
        )

    return EmpiricalRiskNeutralFit(
        action=action,
        weights=weights,
        log_weights=log_weights,
        weighted_mean=weighted_mean,
        cost_band=cost_band.copy(),
        primal_relative_entropy=primal_relative_entropy,
        dual_objective=dual_objective,
        primal_dual_gap=primal_dual_gap,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / paths,
        maximum_constraint_violation=maximum_constraint_violation,
        maximum_kkt_residual=maximum_kkt_residual,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        optimizer_iterations=total_iterations,
        objective_evaluations=total_evaluations,
        active_coordinates=int(np.sum(signs != 0)),
        active_set_iterations=active_set_iterations,
    )


__all__ = [
    "EmpiricalRiskNeutralFit",
    "ReturnMatrix",
    "fit_empirical_risk_neutral_density",
    "fit_empirical_risk_neutral_density_torch",
]
