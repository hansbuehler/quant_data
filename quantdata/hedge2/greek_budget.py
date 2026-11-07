"""Certified numerical recovery of the unchanged cost-constrained Greek problem."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True, kw_only=True)
class GreekBudgetProjection:
    positions: np.ndarray
    objective_gap_bound: float
    normalization_scale: float
    solver: str
    status: str
    warnings: tuple[str, ...]


def greek_projection_gap(*, matrix: np.ndarray, target: np.ndarray, costs: np.ndarray,
                         previous: np.ndarray, positions: np.ndarray, budget: float,
                         position_cap: float) -> float:
    """Bound squared-residual suboptimality by independently evaluating a tangent.

    The turnover multiplier's concave piecewise-linear lower bound attains
    its maximum at zero or |gradient_i|/cost_i. Every such value supplies a
    valid lower bound regardless of any numerical optimizer's status.
    Exactly zero budget instead fixes all positive-cost coordinates.
    Inputs must describe a feasible candidate in the original box/budget.
    Convert original inputs before arithmetic. Explicit residual/gradient
    error bounds cover the tangent's approximate gradient over the box;
    support terms retain pre-cancellation magnitudes. This is conservative
    floating-point error accounting, not directed interval arithmetic.
    """
    if (matrix.ndim != 2 or not min(matrix.shape) or target.shape != (matrix.shape[0],)
            or costs.shape != (matrix.shape[1],) or previous.shape != costs.shape or positions.shape != costs.shape
            or not all(np.isfinite(values).all() for values in (matrix, target, costs, previous, positions))
            or not np.isfinite([budget, position_cap]).all() or budget < 0. or position_cap <= 0.
            or np.any(costs < 0.) or np.any(np.abs(previous) > position_cap)):
        raise ValueError("Greek tangent inputs must be finite, aligned and valid")
    precision = np.longdouble
    matrix, target, costs, previous, positions = (
        np.asarray(value, dtype=precision) for value in (matrix, target, costs, previous, positions))
    budget, position_cap = precision(budget), precision(position_cap)
    epsilon = precision(np.finfo(precision).eps)

    def gamma(count: int) -> np.longdouble:
        product = precision(count) * epsilon
        if not np.isfinite(product) or product >= .5:
            raise RuntimeError("Non-finite Greek tangent arithmetic precision bound")
        return product / (1. - product)

    with np.errstate(over="ignore", invalid="ignore"):
        residual = matrix @ positions - target
        gradient = 2. * matrix.T @ residual
    if not np.isfinite(residual).all() or not np.isfinite(gradient).all():
        raise RuntimeError("Non-finite Greek tangent residual or gradient")
    paid = costs > 0.
    if budget == 0.:
        lower = (gradient[paid] @ previous[paid]
            - position_cap * np.sum(np.abs(gradient[~paid])))
        magnitude = (np.abs(gradient[paid] * previous[paid]).sum()
            + position_cap * np.abs(gradient[~paid]).sum())
    else:
        multipliers = np.r_[precision(0.), np.abs(gradient[paid]) / costs[paid]]
        if not np.isfinite(multipliers).all():
            raise RuntimeError("Greek tangent multipliers are non-finite")
        choices = np.array([-np.full(len(costs), position_cap), previous,
                            np.full(len(costs), position_cap)])
        terms = gradient[None, :, None] * choices[:, :, None] + multipliers[None, None, :] * (
            costs[None, :, None] * np.abs(choices[:, :, None] - previous[None, :, None]))
        minima = terms.min(axis=0)
        bounds = minima.sum(axis=0) - multipliers * budget
        if not np.isfinite(terms).all() or not np.isfinite(bounds).all():
            raise RuntimeError("Non-finite Greek tangent support arithmetic")
        best = int(np.argmax(bounds))
        lower = bounds[best]
        magnitudes = np.abs(gradient[None, :] * choices) + multipliers[best] * (
            costs[None, :] * np.abs(choices - previous[None, :]))
        magnitude = magnitudes.max(axis=0).sum() + multipliers[best] * budget
    if not np.isfinite([lower, magnitude]).all():
        raise RuntimeError("Non-finite Greek tangent bound or arithmetic magnitude")
    absolute_matrix = np.abs(matrix)
    with np.errstate(over="ignore", invalid="ignore"):
        residual_error = 2. * gamma(matrix.shape[1] + 1) * (absolute_matrix @ np.abs(positions) + np.abs(target))
        gradient_error = 4. * (absolute_matrix.T @ residual_error
            + gamma(matrix.shape[0] + 1) * (absolute_matrix.T @ np.abs(residual)))
        gradient_allowance = 2. * np.sum((np.abs(positions) + position_cap) * gradient_error)
        rounding = 2. * (32. + 8. * len(costs)) * epsilon * max(
            magnitude + np.abs(gradient * positions).sum(), precision(np.finfo(precision).tiny))
        raw_gap = gradient @ positions - lower + rounding + gradient_allowance
    if not np.isfinite([rounding, gradient_allowance, raw_gap]).all():
        raise RuntimeError("Non-finite Greek tangent objective gap")
    result = float(np.nextafter(float(max(precision(0.), raw_gap)), np.inf))
    if not np.isfinite(result):
        raise RuntimeError("Non-finite Greek tangent returned objective gap")
    return result


def solve_greek_budget_projection(*, matrix: np.ndarray, target: np.ndarray, costs: np.ndarray,
                                  previous: np.ndarray, budget: float, position_cap: float,
                                  gap_tolerance: float = 1e-8) -> GreekBudgetProjection:
    """Rescale only numerics; retain the weighted Greek norm, box and turnover.

    ``matrix`` and ``target`` already include the prescribed risk weights.
    The acceptance tolerance is in squared-residual units divided by the
    squared normalization scale, relative to max(1, candidate objective).
    A feasibility repair is only a candidate: its unchanged objective must
    satisfy the independently evaluated tangent bound before acceptance.
    """
    if (matrix.ndim != 2 or not min(matrix.shape) or target.shape != (matrix.shape[0],)
            or costs.shape != (matrix.shape[1],) or previous.shape != costs.shape
            or not all(np.isfinite(values).all() for values in (matrix, target, costs, previous))
            or not np.isfinite([budget, position_cap, gap_tolerance]).all()
            or budget < 0. or position_cap <= 0. or gap_tolerance <= 0. or np.any(costs < 0.)
            or np.any(np.abs(previous) > position_cap)):
        raise ValueError("Greek projection needs aligned finite inputs and feasible prior solver holdings")
    scale = max(float(np.linalg.norm(target)), float(np.linalg.norm(matrix @ previous - target)), 1e-10)
    if not np.isfinite(scale) or scale <= 0.:
        raise RuntimeError("Greek recovery normalization scale is non-finite")
    scaled_matrix, scaled_target = matrix / scale, target / scale
    if not np.isfinite(scaled_matrix).all() or not np.isfinite(scaled_target).all():
        raise RuntimeError("Scaled Greek recovery geometry is non-finite")
    messages, failures = [], []
    paid = costs > 0.
    if budget == 0.:
        proposal = previous.copy()
        if (~paid).any():
            fit = lsq_linear(scaled_matrix[:, ~paid],
                scaled_target - scaled_matrix[:, paid] @ previous[paid],
                bounds=(-position_cap, position_cap), tol=1e-12, lsq_solver="exact", max_iter=1000)
            proposal[~paid] = fit.x
            status = str(fit.message)
        else:
            status = "Positive-cost holdings fixed by exactly zero budget"
        candidates = [(proposal, "SciPy bounded free-asset least squares", status)]
    else:
        candidates = []
    for attempt in range(1 if budget == 0. else 2):
        if budget > 0.:
            variable = cp.Variable(len(costs))
            residual = scaled_matrix @ variable - scaled_target
            objective = cp.sum_squares(residual) if attempt == 0 else cp.norm(residual)
            problem = cp.Problem(cp.Minimize(objective), [variable <= position_cap, variable >= -position_cap,
                costs @ cp.abs(variable - previous) / budget <= 1.])
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10 if attempt == 0 else 1e-12,
                        tol_gap_rel=1e-10 if attempt == 0 else 1e-12,
                        tol_feas=1e-10 if attempt == 0 else 1e-12, max_iter=500)
                if variable.value is None:
                    failures.append(f"Rescaled CLARABEL: {problem.status}")
                    continue
                candidates = [(np.asarray(variable.value), "Rescaled CLARABEL", str(problem.status))]
            except cp.error.SolverError as error:
                failures.append(f"Rescaled CLARABEL: {error}")
                continue
            finally:
                messages.extend(str(item.message) for item in caught)
        proposal, solver, status = candidates[0]
        if not np.isfinite(proposal).all():
            failures.append(f"{solver}: non-finite candidate")
            continue
        positions = np.clip(proposal, -position_cap, position_cap)
        if budget == 0.:
            positions[paid] = previous[paid]
        else:
            turnover = float(costs @ np.abs(positions - previous))
            if turnover > budget:
                factor = np.nextafter(budget / turnover, 0.)
                positions[paid] = previous[paid] + factor * (positions[paid] - previous[paid])
        cost = float(costs @ np.abs(positions - previous))
        tolerance = 64. * np.finfo(float).eps * max(budget, float(costs @ np.abs(positions - previous)), 1e-30)
        if np.any(np.abs(positions) > position_cap) or cost > budget + tolerance:
            failures.append(f"{solver}: independent repaired feasibility failed")
            continue
        value = scaled_matrix @ positions - scaled_target
        objective_value = float(value @ value)
        if not np.isfinite(value).all() or not np.isfinite(objective_value):
            failures.append(f"{solver}: non-finite squared-risk objective")
            continue
        original_gap = greek_projection_gap(matrix=matrix, target=target, costs=costs,
            previous=previous, positions=positions, budget=budget, position_cap=position_cap)
        normalized_gap = np.longdouble(original_gap) / np.longdouble(scale) / np.longdouble(scale)
        gap = float(np.nextafter(float(normalized_gap), np.inf))
        if not np.isfinite(gap) or gap > gap_tolerance * max(1., objective_value):
            failures.append(f"{solver}: normalized tangent gap={gap:.17g}")
            continue
        adjustment = float(np.max(np.abs(positions - proposal), initial=0.))
        if adjustment:
            messages.append(f"Greek optimization candidate feasibility repair: maximum adjustment={adjustment:.17g}")
        messages.extend(failures)
        messages.append(f"Greek tangent bound arithmetic: longdouble epsilon={np.finfo(np.longdouble).eps}; "
            f"normalized gap <= {gap:.17g}; original-input gradient uncertainty included")
        return GreekBudgetProjection(positions=positions, objective_gap_bound=original_gap,
            normalization_scale=scale, solver=solver, status=status, warnings=tuple(messages))
    raise RuntimeError("Rescaled Greek projection failed independent certification: " + "; ".join(failures))
