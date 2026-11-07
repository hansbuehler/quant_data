"""Direct cash-eliminated refinement of power-OCE conic candidates."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from quantdata.hedge2.objectives import power_oce


@dataclass(frozen=True, kw_only=True)
class PolishedPowerHedge:
    positions: np.ndarray
    before_objective: float
    after_objective: float
    before_gap: float
    after_gap: float
    solver_success: bool
    solver_message: str
    iterations: int
    warnings: tuple[str, ...] = ()


def polish_power(*, instrument_gains: np.ndarray, claim_gains: np.ndarray, weights: np.ndarray,
                 costs: np.ndarray, initial_positions: np.ndarray, risk_aversion: float,
                 target_side: int, position_cap: float = 1000.,
                 previous_positions: np.ndarray | None = None) -> PolishedPowerHedge:
    gains = np.asarray(instrument_gains, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    if (gains.ndim != 2 or claim_gains.shape != (len(gains),)
            or costs.shape != (gains.shape[1],) or initial_positions.shape != costs.shape
            or not all(np.isfinite(values).all() for values in (gains, claim_gains, probabilities, costs, initial_positions))
            or np.any(probabilities < 0) or np.any(costs < 0) or target_side not in (-1, 1)):
        raise ValueError("Invalid power-OCE polishing inputs")
    previous = np.zeros_like(initial_positions) if previous_positions is None else np.asarray(previous_positions, dtype=float)
    if previous.shape != costs.shape or not np.isfinite(previous).all() or np.any(np.abs(previous) > position_cap):
        raise ValueError("Prior holdings must be feasible in the same portfolio box")
    deviations = np.sqrt(probabilities @ gains**2)
    scales = np.maximum(np.maximum(deviations, costs), max(float(deviations.max()) * 1e-6, 1e-10))
    matrix = gains / scales
    count = gains.shape[1]
    target = target_side * claim_gains
    scaled_costs = costs / scales
    identity = np.eye(count)
    constraints = LinearConstraint(np.block([[-identity, identity], [identity, identity]]),
                                   np.r_[-previous * scales, previous * scales], np.inf)
    bounds = Bounds(np.r_[-position_cap * scales, np.zeros(count)],
                     np.r_[position_cap * scales, np.full(count, np.inf)])
    positions = np.clip(initial_positions, -position_cap, position_cap)
    initial = np.r_[positions * scales, np.abs(positions - previous) * scales]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        wealth = matrix @ values[:count] + target
        oce = power_oce(wealth, probabilities, risk_aversion=risk_aversion)
        value = -oce.certainty_equivalent + scaled_costs @ values[count:]
        gradient = np.r_[-matrix.T @ (probabilities * oce.marginal_utility), scaled_costs]
        return float(value), gradient

    def diagnostics(holdings: np.ndarray) -> tuple[float, float]:
        wealth = gains @ holdings + target
        oce = power_oce(wealth, probabilities, risk_aversion=risk_aversion)
        gradient = -gains.T @ (probabilities * oce.marginal_utility)
        choices = np.array([-np.full(count, position_cap), previous, np.full(count, position_cap)])
        gap = np.sum(holdings * gradient + costs * np.abs(holdings - previous)
                     - np.min(choices * gradient + costs * np.abs(choices - previous), axis=0))
        return float(-oce.certainty_equivalent + costs @ np.abs(holdings - previous)), max(float(gap), 0.)

    before, before_gap = diagnostics(positions)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(objective, initial, jac=True, method="SLSQP", bounds=bounds, constraints=[constraints],
                           options={"maxiter": 1000, "ftol": 1e-11})
    proposal = result.x[:count] / scales
    after, after_gap = diagnostics(proposal)
    if (not np.isfinite(proposal).all() or np.max(np.abs(proposal), initial=0.) > position_cap + 1e-6
            or after > before + 1e-9):
        proposal, after, after_gap = positions, before, before_gap
    return PolishedPowerHedge(positions=proposal, before_objective=before, after_objective=after,
                              before_gap=before_gap, after_gap=after_gap, solver_success=bool(result.success),
                              solver_message=str(result.message), iterations=int(result.nit),
                              warnings=tuple(str(item.message) for item in caught))
