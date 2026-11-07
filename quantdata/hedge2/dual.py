"""Independent quadratic and power-OCE objective-unit lower bounds."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.special import xlogy

from quantdata.hedge2.objectives import power_oce


@dataclass(frozen=True, kw_only=True)
class DualCertificate:
    primal_upper_bound: float
    dual_lower_bound: float
    objective_gap: float
    density: np.ndarray
    solver: str
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class QuadraticCertificate:
    primal_upper_bound: float
    dual_lower_bound: float
    objective_gap: float
    dual_variable: np.ndarray
    solver: str
    status: str
    warnings: tuple[str, ...]


def certify_mean_variance(*, instrument_gains: np.ndarray, claim_gains: np.ndarray,
                          weights: np.ndarray, costs: np.ndarray, positions: np.ndarray,
                          risk_aversion: float, target_side: int, position_cap: float = 1000.,
                          previous_positions: np.ndarray | None = None,
                          gap_tolerance: float = 1e-4) -> QuadraticCertificate:
    """Certify variance-plus-turnover with an unconstrained quadratic Fenchel dual.

    For centered claim Z and centered returns R, any y gives the lower bound
    E[y Z] - E[y**2]/(2*k) plus the exact box minimum of
    E[y R_i]*a_i + c_i*abs(a_i-previous_i). Finite risk aversion also
    subtracts expected wealth; infinite risk aversion is pure variance.
    Unlike an OCE density, y has no positivity or normalization constraint.
    """
    gains = np.asarray(instrument_gains, dtype=float)
    target = target_side * np.asarray(claim_gains, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    transaction_costs = np.asarray(costs, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if (gains.ndim != 2 or target.shape != (len(gains),) or probabilities.shape != target.shape
            or transaction_costs.shape != (gains.shape[1],) or positions.shape != transaction_costs.shape
            or not all(np.isfinite(values).all() for values in (gains, target, probabilities, transaction_costs, positions))
            or np.any(probabilities < 0) or probabilities.sum() <= 0 or np.any(transaction_costs < 0)
            or target_side not in (-1, 1) or risk_aversion <= 0 or np.isnan(risk_aversion)
            or not np.isfinite(position_cap) or position_cap <= 0 or np.any(np.abs(positions) > position_cap)
            or not np.isfinite(gap_tolerance) or gap_tolerance <= 0):
        raise ValueError("The quadratic certificate requires finite feasible positions, weights and costs")
    probabilities = probabilities / probabilities.sum()
    previous = np.zeros_like(positions) if previous_positions is None else np.asarray(previous_positions, dtype=float)
    if previous.shape != positions.shape or not np.isfinite(previous).all() or np.any(np.abs(previous) > position_cap):
        raise ValueError("Prior holdings are outside the same position box")
    finite_risk = np.isfinite(risk_aversion)
    multiplier = risk_aversion if finite_risk else 1.
    penalty = transaction_costs if finite_risk else np.zeros_like(transaction_costs)
    target_mean = float(probabilities @ target)
    return_means = probabilities @ gains
    centered_target = target - target_mean
    centered_gains = gains - return_means
    centered_wealth = centered_target + centered_gains @ positions
    mean_term = target_mean if finite_risk else 0.
    linear_mean = return_means if finite_risk else np.zeros_like(return_means)
    primal = float(.5 * multiplier * (probabilities @ centered_wealth**2)
        + penalty @ np.abs(positions - previous) - mean_term - linear_mean @ positions)
    choices = np.array([-np.full_like(positions, position_cap), previous, np.full_like(positions, position_cap)])

    def evaluate(variable: np.ndarray) -> tuple[float, np.ndarray]:
        candidate = np.asarray(variable, dtype=float)
        if candidate.shape != probabilities.shape or not np.isfinite(candidate).all():
            raise ValueError("Non-finite or misaligned quadratic dual candidate")
        candidate = candidate - probabilities @ candidate
        coefficients = centered_gains.T @ (probabilities * candidate) - linear_mean
        trading_lower = np.min(choices * coefficients + penalty * np.abs(choices - previous), axis=0).sum()
        bound = probabilities @ (candidate * centered_target - candidate**2 / (2. * multiplier)) + trading_lower - mean_term
        return float(bound), candidate

    best_value, best_variable = evaluate(multiplier * centered_wealth)
    zero_value, zero_variable = evaluate(np.zeros_like(probabilities))
    if zero_value > best_value:
        best_value, best_variable = zero_value, zero_variable
    positive_weight = probabilities > 0
    scaling = np.sqrt(multiplier * probabilities[positive_weight])
    variable = cp.Variable(int(positive_weight.sum()))
    support = cp.Variable(gains.shape[1])
    coefficients = centered_gains[positive_weight].T @ cp.multiply(scaling, variable) - linear_mean
    constraints = [support <= cp.multiply(choice, coefficients) + penalty * np.abs(choice - previous)
                   for choice in choices]
    objective = .5 * cp.sum_squares(variable) - (scaling * centered_target[positive_weight]) @ variable - cp.sum(support)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    selected_solver, selected_status = "analytical candidate", "feasible lower bound"
    collected = []
    for solver in ("CLARABEL", "OSQP"):
        if primal - best_value <= gap_tolerance:
            break
        try:
            variable.value = best_variable[positive_weight] * np.sqrt(probabilities[positive_weight] / multiplier)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if solver == "CLARABEL":
                    problem.solve(solver=solver, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=1000)
                else:
                    problem.solve(solver=solver, eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)
                collected.extend(str(item.message) for item in caught)
            if variable.value is not None:
                candidate = np.zeros_like(probabilities)
                candidate[positive_weight] = np.asarray(variable.value) * np.sqrt(multiplier / probabilities[positive_weight])
                value, candidate = evaluate(candidate)
                if value > best_value:
                    best_value, best_variable = value, candidate
                    selected_solver, selected_status = solver, str(problem.status)
        except (cp.error.SolverError, ValueError) as error:
            collected.append(f"{solver}: {error}")
    if best_value > primal + 1e-8 * max(1., abs(primal)):
        raise RuntimeError("Quadratic certificate violates weak duality beyond floating-point tolerance")
    return QuadraticCertificate(primal_upper_bound=primal, dual_lower_bound=best_value,
        objective_gap=max(primal - best_value, 0.), dual_variable=best_variable,
        solver=selected_solver, status=selected_status, warnings=tuple(collected))


def certify_power(*, instrument_gains: np.ndarray, claim_gains: np.ndarray,
                  weights: np.ndarray, costs: np.ndarray, positions: np.ndarray,
                  risk_aversion: float, target_side: int, position_cap: float = 1000.,
                  previous_positions: np.ndarray | None = None,
                  gap_tolerance: float = 1e-4) -> DualCertificate:
    """Optimize a lower bound, but certify it by feasibility rather than solver status.

    For q>=0, E[q]=1, the conjugate is (q log q-q+1)/lambda below one,
    and (q-1)^2/(2lambda) above one. Minimize the linearized trading term
    over each position box exactly, including turnover from prior holdings.
    Any repaired density gives a valid lower bound even if not dual-optimal.
    """
    gains = np.asarray(instrument_gains, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if (gains.ndim != 2 or claim_gains.shape != (len(gains),) or probabilities.shape != (len(gains),)
            or costs.shape != (gains.shape[1],) or positions.shape != costs.shape
            or not all(np.isfinite(values).all() for values in (gains, claim_gains, probabilities, costs, positions))
            or np.any(probabilities < 0) or probabilities.sum() <= 0 or np.any(costs < 0)
            or target_side not in (-1, 1) or risk_aversion <= 0 or not np.isfinite(risk_aversion)
            or not np.isfinite(position_cap) or position_cap <= 0 or np.any(np.abs(positions) > position_cap)
            or gap_tolerance <= 0):
        raise ValueError("The primal and dual certificate require feasible finite inputs")
    probabilities = probabilities / probabilities.sum()
    previous = np.zeros_like(positions) if previous_positions is None else np.asarray(previous_positions, dtype=float)
    if previous.shape != positions.shape or not np.isfinite(previous).all() or np.any(np.abs(previous) > position_cap):
        raise ValueError("Prior holdings are outside the same position box")
    target = target_side * claim_gains
    wealth = gains @ positions + target
    oce = power_oce(wealth, probabilities, risk_aversion=risk_aversion)
    primal = float(-oce.certainty_equivalent + costs @ np.abs(positions - previous))
    choices = np.array([-np.full_like(positions, position_cap), previous, np.full_like(positions, position_cap)])

    def evaluate(density: np.ndarray) -> tuple[float, np.ndarray]:
        repaired = np.maximum(np.asarray(density), 0.)
        normalization = float(probabilities @ repaired)
        if not np.isfinite(repaired).all() or normalization <= 0:
            raise ValueError("The candidate dual density cannot be normalized")
        repaired = repaired / normalization
        below = repaired <= 1.
        conjugate = np.empty_like(repaired)
        conjugate[below] = xlogy(repaired[below], repaired[below]) - repaired[below] + 1.
        conjugate[~below] = .5 * (repaired[~below] - 1.)**2
        mean_return = gains.T @ (probabilities * repaired)
        trading_lower = np.min(costs * np.abs(choices - previous) - choices * mean_return, axis=0).sum()
        value = -probabilities @ (repaired * target) - probabilities @ conjugate / risk_aversion + trading_lower
        return float(value), repaired

    best_value, best_density = evaluate(oce.marginal_utility)
    uniform_value, uniform = evaluate(np.ones(len(gains)))
    if uniform_value > best_value:
        best_value, best_density = uniform_value, uniform
    positive = cp.Variable(len(gains), nonneg=True)
    negative = cp.Variable(len(gains), nonneg=True)
    density = 1. + positive - negative
    support = cp.Variable(gains.shape[1])
    mean_return = gains.T @ cp.multiply(probabilities, density)
    constraints = [negative <= 1., probabilities @ density == 1.]
    for choice in choices:
        constraints.append(support <= costs * np.abs(choice - previous) - cp.multiply(choice, mean_return))
    conjugate = cp.kl_div(1. - negative, 1.) + .5 * cp.square(positive)
    objective = probabilities @ cp.multiply(density, target) + probabilities @ conjugate / risk_aversion - cp.sum(support)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    selected_solver, selected_status = "analytical candidate", "feasible lower bound"
    collected = []
    for solver in ("CLARABEL", "SCS"):
        if primal - best_value <= gap_tolerance:
            break
        try:
            positive.value = np.maximum(best_density - 1., 0.)
            negative.value = np.maximum(1. - best_density, 0.)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if solver == "CLARABEL":
                    problem.solve(solver=solver, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=1000)
                else:
                    problem.solve(solver=solver, eps=1e-7, max_iters=100000)
                collected.extend(str(item.message) for item in caught)
            if density.value is not None:
                value, repaired = evaluate(density.value)
                if value > best_value:
                    best_value, best_density = value, repaired
                    selected_solver, selected_status = solver, str(problem.status)
        except (cp.error.SolverError, ValueError) as error:
            collected.append(f"{solver}: {error}")
    if best_value > primal + 1e-8:
        raise RuntimeError("Power certificate violates weak duality beyond floating-point tolerance")
    return DualCertificate(primal_upper_bound=primal, dual_lower_bound=best_value,
                            objective_gap=max(primal - best_value, 0.), density=best_density,
                            solver=selected_solver, status=selected_status, warnings=tuple(collected))
