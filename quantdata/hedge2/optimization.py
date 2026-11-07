"""Reusable convex scenario hedges with turnover costs and observable certification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.polish import polish_power


@dataclass(frozen=True, kw_only=True)
class OptimizedHedge:
    positions: np.ndarray
    entry_cost: float
    training_objective: float
    training_mean: float
    training_std: float
    cash_first_order_residual: float
    convex_first_order_gap_bound: float
    cap_hits: int
    solver: str
    status: str
    warnings: tuple[str, ...]
    iterations: int


class ScenarioOptimizer:
    """Cache convex program structure across targets and risk aversions on one date.

    Asset standardization uses the larger of return std and trading cost.
    This is an invertible numerical change of variables,
    not covariance shrinkage or an additional economic risk penalty. All
    scenario gains, including spot, must already be demeaned using weights.
    Reduced-precision OPTIMAL_INACCURATE results retain their exact warning;
    independent checks reject non-finite or materially infeasible positions.
    """

    def __init__(self, *, instrument_gains: np.ndarray, weights: np.ndarray,
                 costs: np.ndarray, position_cap: float = 1000.) -> None:
        gains = np.asarray(instrument_gains, dtype=float)
        probabilities = np.asarray(weights, dtype=float)
        transaction_costs = np.asarray(costs, dtype=float)
        if (gains.ndim != 2 or probabilities.shape != (len(gains),)
                or transaction_costs.shape != (gains.shape[1],)
                or not all(np.isfinite(array).all() for array in (gains, probabilities, transaction_costs))
                or np.any(probabilities < 0) or probabilities.sum() <= 0
                or np.any(transaction_costs < 0) or not np.isfinite(position_cap) or position_cap <= 0):
            raise ValueError("Invalid scenario gains, weights, costs or position cap")
        self.gains = gains
        self.weights = probabilities / probabilities.sum()
        self.costs = transaction_costs
        self.position_cap = position_cap
        if np.max(np.abs(self.weights @ gains), initial=0.) > 1e-10:
            raise ValueError("Remove speculative asset means before constructing the optimizer")
        deviations = np.sqrt(self.weights @ gains**2)
        self.scales = np.maximum(np.maximum(deviations, self.costs), max(float(deviations.max()) * 1e-6, 1e-10))
        self.standardized = gains / self.scales
        instruments = gains.shape[1]
        self.action = cp.Variable(instruments)
        self.turnover = cp.Variable(instruments, nonneg=True)
        self.target = cp.Parameter(len(gains))
        self.previous = cp.Parameter(instruments)
        self.cost_multiplier = cp.Parameter(nonneg=True)
        self.lambda_parameter = cp.Parameter(nonneg=True)
        self.power_linear_weight = cp.Parameter(nonneg=True)
        self.power_exponential_weight = cp.Parameter(nonneg=True)
        self.power_quadratic_weight = cp.Parameter(nonneg=True)
        constraints = [self.action <= position_cap * self.scales,
                       self.action >= -position_cap * self.scales,
                       self.turnover >= self.action - self.previous,
                       self.turnover >= self.previous - self.action]
        costs_expression = (self.costs / self.scales) @ self.turnover
        wealth = self.standardized @ self.action + self.target
        mean_variance = .5 * cp.sum_squares(cp.multiply(np.sqrt(self.weights), wealth))
        self.mv_problem = cp.Problem(cp.Minimize(mean_variance + self.cost_multiplier * costs_expression), constraints)
        positive = cp.Variable(len(gains), nonneg=True)
        negative = cp.Variable(len(gains), nonneg=True)
        shift = cp.Variable()
        exponential = cp.Variable(len(gains))
        power_constraint = positive - negative == self.standardized @ self.action + self.target - shift
        power_loss = (self.power_linear_weight * (-shift + self.weights @ negative + costs_expression)
            + self.power_exponential_weight * (self.weights @ exponential)
            + .5 * self.power_quadratic_weight * cp.sum_squares(cp.multiply(np.sqrt(self.weights), negative)))
        self.power_problem = cp.Problem(cp.Minimize(power_loss), constraints + [power_constraint,
            cp.ExpCone(-self.lambda_parameter * positive, np.ones(len(gains)), exponential)])
        self.last_power_positions = np.zeros(instruments)

    def solve(self, *, claim_gains: np.ndarray, objective: str, risk_aversion: float,
              target_side: int = -1, previous_positions: np.ndarray | None = None) -> OptimizedHedge:
        if (claim_gains.shape != (len(self.gains),) or not np.isfinite(claim_gains).all()
                or objective not in ("mean_variance", "power_exponential") or target_side not in (-1, 1)
                or risk_aversion <= 0 or np.isnan(risk_aversion)
                or objective == "power_exponential" and not np.isfinite(risk_aversion)):
            raise ValueError("Invalid target, objective, risk aversion or long/short side")
        previous = np.zeros(self.gains.shape[1]) if previous_positions is None else np.asarray(previous_positions, dtype=float)
        if previous.shape != self.costs.shape or not np.isfinite(previous).all() or np.any(np.abs(previous) > self.position_cap):
            raise ValueError("Prior positions must be finite, aligned and inside the same position limit")
        target = target_side * claim_gains
        centered = target - self.weights @ target
        self.previous.value = previous * self.scales
        if objective == "mean_variance":
            self.target.value = centered
            self.cost_multiplier.value = 1. / risk_aversion if np.isfinite(risk_aversion) else 0.
            problem = self.mv_problem
            solvers = ("CLARABEL", "OSQP")
        else:
            self.target.value = centered
            self.lambda_parameter.value = risk_aversion
            normalizer = max(1., risk_aversion)
            self.power_linear_weight.value = 1. / normalizer
            self.power_exponential_weight.value = 1. / (risk_aversion * normalizer)
            self.power_quadratic_weight.value = risk_aversion / normalizer
            problem = self.power_problem
            solvers = ("CLARABEL", "SCS")
        failures, collected = [], []
        for solver in solvers:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if solver == "CLARABEL":
                            tolerance = 1e-10 if objective == "power_exponential" else 1e-9
                            problem.solve(solver=solver, tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance, max_iter=500)
                        elif solver == "OSQP":
                            problem.solve(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
                        else:
                            problem.solve(solver=solver, eps=1e-6, max_iters=100000)
                    finally:
                        collected.extend(str(item.message) for item in caught)
                if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.action.value is None:
                    failures.append(f"{solver}: {problem.status}")
                    continue
                positions = np.asarray(self.action.value) / self.scales
                if not np.isfinite(positions).all() or np.max(np.abs(positions), initial=0.) > self.position_cap * (1. + 1e-6):
                    failures.append(f"{solver}: independent finite-position/limit check failed")
                    continue
                projected = np.clip(positions, -self.position_cap, self.position_cap)
                adjustment = float(np.max(np.abs(projected - positions), initial=0.))
                if adjustment > 0:
                    collected.append(f"Projected solver position roundoff onto the configured cap; maximum adjustment={adjustment:.17g}")
                    positions = projected
                status = problem.status
                iterations = int(problem.solver_stats.num_iters or 0)
                break
            except cp.error.SolverError as error:
                failures.append(f"{solver}: {error}")
        else:
            if objective != "power_exponential":
                raise RuntimeError(f"{objective} solve failed: " + "; ".join(failures))
            starts = (previous, self.last_power_positions, np.zeros_like(previous))
            initial = min(starts, key=lambda values: -power_oce(self.gains @ values + target, self.weights,
                risk_aversion=risk_aversion).certainty_equivalent + self.costs @ np.abs(values - previous))
            refined = polish_power(instrument_gains=self.gains, claim_gains=claim_gains, weights=self.weights,
                costs=self.costs, initial_positions=initial, risk_aversion=risk_aversion, target_side=target_side,
                position_cap=self.position_cap, previous_positions=previous)
            positions = np.clip(refined.positions, -self.position_cap, self.position_cap)
            collected.extend(refined.warnings)
            solver, status, iterations = "SLSQP cash-eliminated fallback", refined.solver_message, refined.iterations
        wealth = self.gains @ positions + target
        mean = float(self.weights @ wealth)
        centered_wealth = wealth - mean
        variance = float(self.weights @ centered_wealth**2)
        cost = float(self.costs @ np.abs(positions - previous))
        cash_residual = 0.
        if objective == "mean_variance":
            multiplier = risk_aversion if np.isfinite(risk_aversion) else 1.
            cost_coefficient = 1. if np.isfinite(risk_aversion) else 0.
            value = .5 * multiplier * variance + cost_coefficient * cost - (mean if np.isfinite(risk_aversion) else 0.)
            gradient = multiplier * self.gains.T @ (self.weights * centered_wealth)
            penalty = cost_coefficient * self.costs
        else:
            oce = power_oce(wealth, self.weights, risk_aversion=risk_aversion)
            value = -oce.certainty_equivalent + cost
            gradient = -self.gains.T @ (self.weights * oce.marginal_utility)
            penalty = self.costs
            cash_residual = oce.cash_first_order_residual
            self.last_power_positions = positions.copy()
        choices = np.array([-np.full_like(positions, self.position_cap), previous,
                            np.full_like(positions, self.position_cap)])
        linearized = choices * gradient + penalty * np.abs(choices - previous)
        current_linearized = positions * gradient + penalty * np.abs(positions - previous)
        gap = max(float(np.sum(current_linearized - linearized.min(axis=0))), 0.)
        return OptimizedHedge(positions=positions, entry_cost=cost, training_objective=value,
                              training_mean=mean - cost, training_std=float(np.sqrt(variance)),
                              cash_first_order_residual=cash_residual, convex_first_order_gap_bound=gap,
                              cap_hits=int(np.sum(np.abs(positions) >= self.position_cap * (1. - 1e-6))),
                              solver=solver, status=status, warnings=tuple([*collected, *failures]),
                              iterations=iterations)
