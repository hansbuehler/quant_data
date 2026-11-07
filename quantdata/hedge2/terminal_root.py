"""Separate closed-box oracle for the exact sampled final-step terminal objective."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np

from quantdata.hedge2.dual import certify_mean_variance, certify_power
from quantdata.hedge2.terminal import (
    TerminalEvaluation,
    TerminalPaths,
    TerminalRisk,
    evaluate_terminal_policy,
    terminal_risk,
)


@dataclass(frozen=True, kw_only=True)
class TerminalRootResult:
    contract_positions: np.ndarray
    premium_positions: np.ndarray
    evaluation: TerminalEvaluation
    risk: TerminalRisk
    primal_upper_bound: float
    dual_lower_bound: float
    objective_gap_bound: float
    certified: bool
    deterministic_mandatory_cost: float
    solver: str
    status: str
    warnings: tuple[str, ...]
    original_scenarios: int
    optimization_scenarios: int
    strategy_class: str = "unrestricted closed root premium box; not fitted neural weights"


def solve_terminal_root(*, paths: TerminalPaths, root_normalized_prices: np.ndarray,
                        initial_inventory: np.ndarray, objective: str, risk_aversion: float,
                        target_side: int, premium_cap: float = 10.,
                        gap_tolerance: float = 1e-6) -> TerminalRootResult:
    """Solve one remaining step without demeaning the retained Monte Carlo sample.

    Prices and gains use the ORIGINAL episode spot numeraire; costs already
    include the cumulative episode discount. Root prices are observable
    before trade. Identical root fees make turnover deterministic, so finite
    MV/power risk includes them by cash translation. Pure variance excludes
    that deterministic fee from its objective but reports actual net wealth.
    An old physical inventory may be outside today's premium box.
    Identical sampled gain/payoff rows are compressed only inside CVXPY,
    retaining their original summed draw weights. Independent dual bounds
    and cash replay still evaluate every original uncompressed sample.

    This is an independent empirical-objective oracle, not a neural-policy
    fit or a substitute for multi-step terminal optimization. An uncertified
    result is a retained numerical candidate, never a permitted live action.
    """
    count, steps, instruments = paths.gains.shape
    prices = np.asarray(root_normalized_prices, dtype=float)
    initial = np.asarray(initial_inventory, dtype=float)
    if (steps != 1 or prices.shape != (instruments,) or initial.shape != prices.shape
            or not np.isfinite(prices).all() or np.any(prices < 0.) or not np.isfinite(initial).all()
            or np.any(initial[~paths.live_contracts[0]] != 0.)
            or target_side not in (-1, 1) or objective not in ("mean_variance", "power_exponential")
            or risk_aversion <= 0. or np.isnan(risk_aversion)
            or objective == "power_exponential" and not np.isfinite(risk_aversion)
            or not np.isfinite([premium_cap, gap_tolerance]).all() or premium_cap <= 0. or gap_tolerance <= 0.):
        raise ValueError("Root oracle requires one remaining step, valid original-unit inventory/prices and declared risk")
    if (np.any(paths.history_ids[:, 0] != paths.history_ids[0, 0])
            or not np.array_equal(paths.costs[:, 0], np.broadcast_to(paths.costs[0, 0], (count, instruments)))):
        raise ValueError("Root fees and information must be identical before the single shared decision")
    total_weight = float(paths.weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.:
        raise ValueError("Root oracle weights must have finite positive total mass")
    weights = paths.weights / total_weight
    costs = paths.costs[0, 0]
    tradable = paths.live_contracts[0] & (prices > 0.) & paths.initially_active
    finite_mv = objective == "mean_variance" and np.isfinite(risk_aversion)
    pure_variance = objective == "mean_variance" and not finite_mv
    old = initial.copy() if paths.initially_active else np.zeros_like(initial)
    mandatory = float(costs[~tradable] @ np.abs(old[~tradable]))
    premium_previous = old[tradable] * prices[tradable]
    previous = np.clip(premium_previous, -premium_cap, premium_cap)
    transaction_costs = costs[tradable] / prices[tradable]
    gains = paths.gains[:, 0, tradable] / prices[tradable]
    mandatory += float(transaction_costs @ np.abs(premium_previous - previous))
    if (not np.isfinite(mandatory) or not np.isfinite(gains).all()
            or not np.isfinite(transaction_costs).all() or not np.isfinite(premium_previous).all()):
        raise ValueError("Root premium conversion overflowed; no price floor or hidden cap is permitted")

    def replay(premium_positions: np.ndarray) -> tuple[np.ndarray, TerminalEvaluation, TerminalRisk]:
        positions = np.zeros(instruments)
        positions[tradable] = premium_positions / prices[tradable]
        if not np.isfinite(positions).all() or np.any(np.abs(positions * prices) > premium_cap * (1. + 1e-12)):
            raise RuntimeError("Root oracle candidate failed its original-unit premium check")
        evaluation = evaluate_terminal_policy(paths=paths, initial_inventory=initial,
            policy=lambda features, inventory, step: np.broadcast_to(positions, (count, instruments)),
            target_side=target_side, position_cap=None)
        expected_cost = mandatory + float(transaction_costs @ np.abs(premium_positions - previous))
        magnitude = float(costs @ (np.abs(old) + np.abs(positions)))
        allowance = (32. + 8. * instruments) * np.finfo(float).eps * max(1., magnitude)
        if (not np.isfinite([expected_cost, magnitude, allowance]).all()
                or np.max(np.abs(evaluation.entry_cost - expected_cost)) > allowance):
            raise RuntimeError("Root oracle deterministic turnover disagrees with original-unit replay")
        risk = terminal_risk(evaluation.wealth, weights, objective=objective, risk_aversion=risk_aversion)
        if not np.isfinite([risk.value, risk.mean_wealth, risk.std_wealth]).all():
            raise RuntimeError("Root oracle original-unit risk replay is non-finite")
        return positions, evaluation, risk

    if not tradable.any():
        positions, evaluation, risk = replay(np.empty(0))
        return TerminalRootResult(contract_positions=positions, premium_positions=positions * prices,
            evaluation=evaluation, risk=risk, primal_upper_bound=risk.value,
            dual_lower_bound=risk.value, objective_gap_bound=0., certified=True,
            deterministic_mandatory_cost=mandatory, solver="no tradable root assets",
            status="single feasible holding vector", warnings=(),
            original_scenarios=count, optimization_scenarios=0)
    mean_returns = weights @ gains
    deviations = gains - mean_returns
    scales = np.maximum.reduce((np.sqrt(weights @ deviations**2), np.abs(mean_returns), transaction_costs,
                                np.full(gains.shape[1], 1e-8)))
    if not np.isfinite(scales).all():
        raise ValueError("Root oracle numerical asset scales are non-finite")
    unique, inverse = np.unique(np.c_[gains, paths.claim_gain], axis=0, return_inverse=True)
    optimization_weights = np.bincount(inverse, weights=weights, minlength=len(unique))
    scaled_gains = unique[:, :-1] / scales
    variable = cp.Variable(gains.shape[1])
    wealth = target_side * unique[:, -1] + scaled_gains @ variable
    turnover = (transaction_costs / scales) @ cp.abs(variable - previous * scales)
    constraints = [variable <= premium_cap * scales, variable >= -premium_cap * scales]
    if objective == "mean_variance":
        mean = optimization_weights @ wealth
        multiplier = risk_aversion if finite_mv else 1.
        criterion = .5 * multiplier * cp.sum(cp.multiply(optimization_weights, cp.square(wealth - mean)))
        if finite_mv:
            criterion += turnover - mean
    else:
        shifted = cp.Variable()
        positive, negative = cp.Variable(len(unique), nonneg=True), cp.Variable(len(unique), nonneg=True)
        constraints.append(positive - negative == risk_aversion * wealth - shifted)
        criterion = (optimization_weights @ (cp.exp(-positive) + negative + .5 * cp.square(negative)) - shifted - 1.) / risk_aversion + turnover
    problem = cp.Problem(cp.Minimize(criterion), constraints)
    collected, failures = [], []
    best = None
    best_lower = -np.inf
    solvers = ("CLARABEL", "OSQP") if objective == "mean_variance" else ("CLARABEL", "SCS")
    for solver in solvers:
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if solver == "CLARABEL":
                    problem.solve(solver=solver, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=1000)
                elif solver == "OSQP":
                    problem.solve(solver=solver, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000)
                else:
                    problem.solve(solver=solver, eps=1e-7, max_iters=100000)
            if variable.value is None or not np.isfinite(variable.value).all():
                failures.append(f"{solver}: no finite candidate; {problem.status}")
                continue
            candidate = np.clip(np.asarray(variable.value) / scales, -premium_cap, premium_cap)
            positions, evaluation, risk = replay(candidate)
            certifier = certify_mean_variance if objective == "mean_variance" else certify_power
            certificate = certifier(instrument_gains=gains, claim_gains=paths.claim_gain,
                weights=weights, costs=transaction_costs, positions=candidate,
                risk_aversion=risk_aversion, target_side=target_side, position_cap=premium_cap,
                previous_positions=previous, gap_tolerance=gap_tolerance)
            constant = 0. if pure_variance else mandatory
            upper, lower = certificate.primal_upper_bound + constant, certificate.dual_lower_bound + constant
            if not np.isfinite([upper, lower, certificate.objective_gap]).all():
                raise RuntimeError("Root oracle certificate is non-finite")
            tolerance = 1e-8 * max(1., abs(upper), abs(risk.value), abs(mandatory))
            if abs(upper - risk.value) > tolerance:
                raise RuntimeError("Root oracle bounds disagree with uncompressed original-unit terminal replay")
            rounding = float(256. * np.finfo(float).eps * max(1., abs(upper), abs(lower), abs(mandatory)))
            if lower > min(upper, risk.value) + rounding:
                raise RuntimeError("Root oracle certificate violates independent weak duality")
            primal = float(max(upper, risk.value) + rounding)
            best_lower = float(max(best_lower, lower - rounding))
            collected.extend(certificate.warnings)
            result = TerminalRootResult(contract_positions=positions, premium_positions=positions * prices,
                evaluation=evaluation, risk=risk, primal_upper_bound=primal, dual_lower_bound=best_lower,
                objective_gap_bound=float(max(0., primal - best_lower)), certified=False,
                deterministic_mandatory_cost=mandatory, solver=solver, status=str(problem.status), warnings=(),
                original_scenarios=count, optimization_scenarios=len(unique))
            if best is None or result.primal_upper_bound < best.primal_upper_bound:
                best = result
            if best_lower > best.primal_upper_bound + rounding:
                raise RuntimeError("Root oracle cross-candidate bounds violate weak duality")
            best = replace(best, dual_lower_bound=best_lower,
                objective_gap_bound=float(max(0., best.primal_upper_bound - best_lower)))
            if best.objective_gap_bound <= gap_tolerance:
                break
        except cp.error.SolverError as error:
            failures.append(f"{solver}: {error}")
        finally:
            collected.extend(str(item.message) for item in caught)
    if best is None:
        raise RuntimeError("Root oracle produced no independently replayed candidate: " + "; ".join(failures))
    return replace(best, certified=bool(best.objective_gap_bound <= gap_tolerance),
        warnings=tuple((*collected, *failures)))
