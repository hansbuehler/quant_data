"""Linear-program helpers for multi-expiry static-arbitrage audits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cdxcore.err import verify_inp
from scipy.optimize import linprog

from .conditional_static_arbitrage import weighted_mean_standard_error

Hedge = Literal["raw", "delta"]


@dataclass(frozen=True)
class MultiExpiryStaticResult:
    """Result of the lexicographic complete-support LP at one entry state."""

    action: np.ndarray
    lower_bound: float
    numerical_band: float
    witness_value: float
    strict_lower_bound: float
    strict_numerical_band: float
    strict_certificate: bool
    static_arbitrage_candidate: bool
    tail_slope: np.ndarray


@dataclass(frozen=True)
class LocalMultiExpiryResult:
    """Result of a return-fitted multi-expiry LP in one local state ball."""

    action: np.ndarray
    training_pnl: np.ndarray
    training_band: np.ndarray
    training_minimum_pnl: float
    training_statistics: dict[str, float | int]
    training_candidate: bool
    active_constraints: int
    constraint_generation_iterations: int


def _validate_primitives(
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    verify_inp(option_price.ndim == 2, "option prices must have shape (expiries, strikes)")
    E, K = option_price.shape
    verify_inp(
        delta.shape == cash_strike.shape == option_price.shape
        and costs.shape == (E, K + 1),
        "multi-expiry primitive arrays have incompatible shapes",
    )
    values = np.concatenate(
        (
            np.asarray([spot]),
            option_price.ravel(),
            delta.ravel(),
            cash_strike.ravel(),
            costs.ravel(),
        )
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("multi-expiry primitive input contains NaN or infinity")
    verify_inp(spot > 0.0, lambda: f"spot must be positive, got {spot}")
    verify_inp(np.all(cash_strike > 0.0), "cash strikes must be positive")
    verify_inp(np.all(costs >= 0.0), "transaction costs must be nonnegative")
    verify_inp(
        np.all(np.diff(cash_strike, axis=1) >= 0.0),
        "cash strikes must be nondecreasing within each expiry",
    )
    return option_price, delta, cash_strike, costs, E, K


def _expiry_scenarios(
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    hedge: Hedge,
    include_tail_probe: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    E, K = option_price.shape
    scenarios: list[np.ndarray] = []
    slopes = np.empty((E, K + 1), dtype=np.float64)
    for expiry in range(E):
        knots = np.concatenate(([0.0], cash_strike[expiry]))
        if include_tail_probe:
            tail = max(
                2.0 * float(cash_strike[expiry, -1]),
                float(cash_strike[expiry, -1]) + spot,
            )
            terminal_spot = np.concatenate((knots, [tail]))
        else:
            terminal_spot = knots
        option_return = (
            np.maximum(
                terminal_spot[:, None] - cash_strike[expiry, None, :],
                0.0,
            )
            - option_price[expiry, None, :]
        )
        if hedge == "delta":
            option_return -= delta[expiry, None, :] * (
                terminal_spot[:, None] - spot
            )
            option_slope = 1.0 - delta[expiry]
        else:
            option_slope = np.ones(K, dtype=np.float64)
        scenarios.append(
            np.concatenate(((terminal_spot - spot)[:, None], option_return), axis=1)
        )
        slopes[expiry] = np.concatenate(([1.0], option_slope))
    return scenarios, slopes


def multi_expiry_terminal_pnl(
    action: np.ndarray,
    terminal_spot: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
) -> np.ndarray:
    """Evaluate a frozen multi-expiry portfolio on joint terminal spots."""

    option_price, delta, cash_strike, costs, E, K = _validate_primitives(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    action = np.asarray(action, dtype=np.float64)
    terminal_spot = np.asarray(terminal_spot, dtype=np.float64)
    verify_inp(action.shape == (E, K + 1), "multi-expiry action has invalid shape")
    verify_inp(
        terminal_spot.ndim >= 1 and terminal_spot.shape[-1] == E,
        "terminal spots must have one final axis per expiry",
    )
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(terminal_spot)):
        raise ValueError("multi-expiry action or terminal spots contain NaN/infinity")

    scenario_return = multi_expiry_scenario_returns(
        terminal_spot,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
    )
    flat_return = scenario_return.reshape(-1, E, K + 1)
    pnl = np.sum(flat_return * action[None, :, :], axis=(1, 2))
    pnl -= float(np.sum(costs * np.abs(action)))
    if not np.all(np.isfinite(pnl)):
        raise FloatingPointError("multi-expiry terminal P&L is non-finite")
    return pnl.reshape(terminal_spot.shape[:-1])


def multi_expiry_scenario_returns(
    terminal_spot: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    hedge: Hedge,
) -> np.ndarray:
    """Return pre-cost instrument returns for joint expiry-spot scenarios."""

    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    terminal_spot = np.asarray(terminal_spot, dtype=np.float64)
    verify_inp(option_price.ndim == 2, "option prices must have shape (expiries, strikes)")
    E, K = option_price.shape
    verify_inp(
        delta.shape == cash_strike.shape == (E, K)
        and terminal_spot.ndim >= 1
        and terminal_spot.shape[-1] == E,
        "multi-expiry scenario-return arrays have incompatible shapes",
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (terminal_spot, option_price, delta, cash_strike)
    ):
        raise ValueError("multi-expiry scenario-return input contains NaN/infinity")
    verify_inp(spot > 0.0, lambda: f"spot must be positive, got {spot}")
    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")

    leading_shape = terminal_spot.shape[:-1]
    flat_spot = terminal_spot.reshape(-1, E)
    result = np.empty((flat_spot.shape[0], E, K + 1), dtype=np.float64)
    for expiry in range(E):
        y = flat_spot[:, expiry]
        result[:, expiry, 0] = y - spot
        option_return = (
            np.maximum(y[:, None] - cash_strike[expiry, None, :], 0.0)
            - option_price[expiry, None, :]
        )
        if hedge == "delta":
            option_return -= delta[expiry, None, :] * (y[:, None] - spot)
        result[:, expiry, 1:] = option_return
    return result.reshape(*leading_shape, E, K + 1)


def multi_expiry_lower_bound(
    action: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the exact complete-support lower bound of a frozen action."""

    option_price, delta, cash_strike, costs, E, K = _validate_primitives(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    action = np.asarray(action, dtype=np.float64)
    verify_inp(action.shape == (E, K + 1), "multi-expiry action has invalid shape")
    scenarios, slopes = _expiry_scenarios(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
        include_tail_probe=False,
    )
    per_expiry = np.empty(E, dtype=np.float64)
    tail_slope = np.sum(action * slopes, axis=1)
    slope_band = 32.0 * np.finfo(np.float32).eps * np.maximum(
        1.0,
        np.abs(action).sum(axis=1),
    )
    for expiry, scenario_return in enumerate(scenarios):
        pnl = scenario_return @ action[expiry] - costs[expiry] @ np.abs(
            action[expiry]
        )
        per_expiry[expiry] = (
            float(np.min(pnl))
            if tail_slope[expiry] >= -slope_band[expiry]
            else -math.inf
        )
    lower_bound = float(np.sum(per_expiry))
    return lower_bound, per_expiry, tail_slope


def multi_expiry_numerical_band(
    action: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
    roundoff_multiplier: float = 32.0,
) -> float:
    """Propagate float32 arithmetic scale through one multi-expiry action."""

    option_price, delta, cash_strike, costs, E, K = _validate_primitives(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    action = np.asarray(action, dtype=np.float64)
    verify_inp(action.shape == (E, K + 1), "multi-expiry action has invalid shape")
    scenarios, _ = _expiry_scenarios(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
        include_tail_probe=False,
    )
    scale = 0.0
    for expiry, scenario_return in enumerate(scenarios):
        instrument_scale = np.max(np.abs(scenario_return), axis=0) + costs[expiry]
        scale += float(np.abs(action[expiry]) @ instrument_scale)
    band = roundoff_multiplier * np.finfo(np.float32).eps * max(1.0, scale)
    if not math.isfinite(band) or band <= 0.0:
        raise FloatingPointError("multi-expiry numerical band is not finite-positive")
    return band


def _complete_support_lp_geometry(
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, float]:
    scenarios, slopes = _expiry_scenarios(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
        include_tail_probe=False,
    )
    E, J = costs.shape
    P = E * J
    rows: list[np.ndarray] = []
    bounds = np.zeros(sum(value.shape[0] for value in scenarios) + E + 1)
    row_number = 0
    for expiry, scenario_return in enumerate(scenarios):
        interval = slice(expiry * J, (expiry + 1) * J)
        for value in scenario_return:
            row = np.zeros(2 * P + E, dtype=np.float64)
            row[interval] = -(value - costs[expiry])
            row[P + interval.start : P + interval.stop] = -(-value - costs[expiry])
            row[2 * P + expiry] = 1.0
            rows.append(row)
            row_number += 1
        row = np.zeros(2 * P + E, dtype=np.float64)
        row[interval] = -slopes[expiry]
        row[P + interval.start : P + interval.stop] = slopes[expiry]
        rows.append(row)
        row_number += 1
    budget = np.zeros(2 * P + E, dtype=np.float64)
    budget[: 2 * P] = 1.0
    rows.append(budget)
    bounds[-1] = 1.0
    assert row_number + 1 == bounds.size

    maximum_instrument_scale = 1.0
    for expiry, scenario_return in enumerate(scenarios):
        maximum_instrument_scale = max(
            maximum_instrument_scale,
            float(np.max(np.max(np.abs(scenario_return), axis=0) + costs[expiry])),
        )
    safe_band = 32.0 * np.finfo(np.float32).eps * maximum_instrument_scale
    return np.stack(rows), bounds, scenarios, slopes, safe_band


def solve_multi_expiry_static_arbitrage_lp(
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Hedge,
) -> MultiExpiryStaticResult:
    r"""Find strict and zero-floor static arbitrage across all expiries.

    The first LP maximizes the complete-support lower P&L.  The second permits
    only the propagated float32 non-loss band and maximizes a fixed positive
    linear functional of the payoff.  The latter detects nonzero butterflies
    and other ordinary arbitrages whose exact lower bound is zero.
    """

    option_price, delta, cash_strike, costs, E, K = _validate_primitives(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    J = K + 1
    P = E * J
    A_ub, b_ub, scenarios, slopes, _ = _complete_support_lp_geometry(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    variable_bounds = [(0.0, None)] * (2 * P) + [(None, None)] * E

    strict_objective = np.zeros(2 * P + E, dtype=np.float64)
    strict_objective[2 * P :] = -1.0
    strict = linprog(
        strict_objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=variable_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not strict.success or strict.x is None:
        raise RuntimeError(
            "multi-expiry worst-margin LP failed with status "
            f"{strict.status}: {strict.message}"
        )
    strict_action = (
        strict.x[:P] - strict.x[P : 2 * P]
    ).reshape(E, J)
    strict_lower, _, _ = multi_expiry_lower_bound(
        strict_action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    strict_band = multi_expiry_numerical_band(
        strict_action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )

    # Uniform positive mass at every kink and at one upper-tail probe is a
    # deterministic witness that a continuous nonnegative payoff is nonzero.
    # The optimization constraint is exact zero rather than the reporting
    # tolerance: otherwise the objective can manufacture a tiny loss of the
    # permitted size and turn the numerical band itself into apparent profit.
    witness_scenarios, _ = _expiry_scenarios(
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        hedge=hedge,
        include_tail_probe=True,
    )
    mean_return = np.concatenate(
        [scenario.mean(axis=0) for scenario in witness_scenarios]
    )
    flat_cost = costs.reshape(-1)
    witness_objective = np.zeros(2 * P + E, dtype=np.float64)
    witness_objective[:P] = -(mean_return - flat_cost)
    witness_objective[P : 2 * P] = -(-mean_return - flat_cost)
    nonloss = np.zeros((1, 2 * P + E), dtype=np.float64)
    nonloss[0, 2 * P :] = -1.0
    witness = linprog(
        witness_objective,
        A_ub=np.concatenate((A_ub, nonloss), axis=0),
        b_ub=np.concatenate((b_ub, np.asarray([0.0]))),
        bounds=variable_bounds,
        method="highs",
        options={"presolve": True},
    )
    if not witness.success or witness.x is None:
        raise RuntimeError(
            "multi-expiry non-loss LP failed with status "
            f"{witness.status}: {witness.message}"
        )
    witness_action = (
        witness.x[:P] - witness.x[P : 2 * P]
    ).reshape(E, J)
    witness_lower, _, tail_slope = multi_expiry_lower_bound(
        witness_action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    witness_band = multi_expiry_numerical_band(
        witness_action,
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    witness_value = float(-witness.fun)
    strict_certificate = strict_lower > strict_band
    weak_candidate = witness_lower >= -witness_band and witness_value > witness_band
    action = strict_action if strict_certificate else witness_action
    lower_bound = strict_lower if strict_certificate else witness_lower
    numerical_band = strict_band if strict_certificate else witness_band
    if strict_certificate:
        tail_slope = np.sum(strict_action * slopes, axis=1)
        selected_witness = float(
            sum(
                scenario.mean(axis=0) @ strict_action[expiry]
                - costs[expiry] @ np.abs(strict_action[expiry])
                for expiry, scenario in enumerate(witness_scenarios)
            )
        )
    else:
        selected_witness = witness_value
    if not np.all(np.isfinite(action)) or not math.isfinite(lower_bound):
        raise FloatingPointError("multi-expiry LP returned NaN or infinity")
    verify_inp(
        float(np.abs(action).sum()) <= 1.0 + 1.0e-8,
        "multi-expiry LP violated its global L1 budget",
    )
    return MultiExpiryStaticResult(
        action=action,
        lower_bound=lower_bound,
        numerical_band=numerical_band,
        witness_value=selected_witness,
        strict_lower_bound=strict_lower,
        strict_numerical_band=strict_band,
        strict_certificate=strict_certificate,
        static_arbitrage_candidate=strict_certificate or weak_candidate,
        tail_slope=tail_slope,
    )


def observation_numerical_band(
    action: np.ndarray,
    scenario_return: np.ndarray,
    costs: np.ndarray,
    *,
    roundoff_multiplier: float = 32.0,
) -> np.ndarray:
    """Return the propagated arithmetic band for each realized scenario."""

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    scenario_return = np.asarray(scenario_return, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    verify_inp(
        scenario_return.ndim == 2
        and scenario_return.shape[1] == action.size == costs.size,
        "observation-band arrays have incompatible shapes",
    )
    scale = (np.abs(scenario_return) + costs[None, :]) @ np.abs(action)
    return roundoff_multiplier * np.finfo(np.float32).eps * np.maximum(1.0, scale)


def solve_local_multi_expiry_portfolio(
    scenario_return: np.ndarray,
    costs: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    initial_constraints: int | None = 64,
) -> LocalMultiExpiryResult:
    """Fit a zero-floor, maximum-mean portfolio on one empirical state ball."""

    scenario_return = np.asarray(scenario_return, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    verify_inp(
        scenario_return.ndim == 2
        and scenario_return.shape[0] > 1
        and scenario_return.shape[1] == costs.size,
        "local multi-expiry arrays have incompatible shapes",
    )
    if not np.all(np.isfinite(scenario_return)) or not np.all(np.isfinite(costs)):
        raise ValueError("local multi-expiry inputs contain NaN or infinity")
    verify_inp(np.all(costs >= 0.0), "local transaction costs must be nonnegative")
    N, P = scenario_return.shape
    relative_weight = (
        np.ones(N, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    verify_inp(relative_weight.shape == (N,), "local weights have invalid shape")
    if not np.all(np.isfinite(relative_weight)) or not np.all(relative_weight >= 0.0):
        raise ValueError("local weights must be finite and nonnegative")
    verify_inp(float(relative_weight.sum()) > 0.0, "local weights have zero mass")
    verify_inp(
        initial_constraints is None
        or (isinstance(initial_constraints, int) and initial_constraints > 1),
        lambda: f"invalid initial constraint count {initial_constraints}",
    )
    normalized_weight = relative_weight / relative_weight.sum()

    A_budget = np.ones((1, 2 * P), dtype=np.float64)
    bounds = [(0.0, None)] * (2 * P)
    mean_return = normalized_weight @ scenario_return
    mean_objective = np.zeros(2 * P, dtype=np.float64)
    mean_objective[:P] = -(mean_return - costs)
    mean_objective[P : 2 * P] = -(-mean_return - costs)

    if initial_constraints is None or initial_constraints >= N:
        active = np.arange(N, dtype=np.int64)
    else:
        active = np.unique(
            np.linspace(0, N - 1, initial_constraints, dtype=np.int64)
        )
    fitted = None
    action = np.zeros(P, dtype=np.float64)
    max_iterations = 32
    for iteration in range(1, max_iterations + 1):
        active_return = scenario_return[active]
        A_scenario = np.concatenate(
            (
                -(active_return - costs[None, :]),
                -(-active_return - costs[None, :]),
            ),
            axis=1,
        )
        fitted = linprog(
            mean_objective,
            A_ub=np.concatenate((A_scenario, A_budget), axis=0),
            b_ub=np.concatenate((np.zeros(active.size), np.asarray([1.0]))),
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not fitted.success or fitted.x is None:
            raise RuntimeError(
                "local maximum-mean LP failed with status "
                f"{fitted.status}: {fitted.message}"
            )
        action = fitted.x[:P] - fitted.x[P : 2 * P]
        action_l1 = float(np.abs(action).sum())
        verify_inp(
            action_l1 <= 1.0 + 1.0e-5,
            lambda: f"local LP violated its global L1 budget with norm {action_l1}",
        )
        # HiGHS uses a primal-feasibility tolerance near 1e-7.  Restore the
        # exact budget after accepting only a solver-scale deviation.
        if action_l1 > 1.0:
            action *= (1.0 - 8.0 * np.finfo(np.float64).eps) / action_l1
        pnl = scenario_return @ action - costs @ np.abs(action)
        violated = np.flatnonzero(pnl < -1.0e-8)
        if violated.size == 0:
            break
        inactive = np.setdiff1d(violated, active, assume_unique=False)
        if inactive.size == 0 or active.size == N:
            break
        worst = inactive[np.argsort(pnl[inactive])[:64]]
        active = np.unique(np.concatenate((active, worst)))
    else:
        # This should be rare. A final full solve keeps the audit exact rather
        # than accepting a constraint-generation iteration cap.
        active = np.arange(N, dtype=np.int64)
        active_return = scenario_return
        A_scenario = np.concatenate(
            (
                -(active_return - costs[None, :]),
                -(-active_return - costs[None, :]),
            ),
            axis=1,
        )
        fitted = linprog(
            mean_objective,
            A_ub=np.concatenate((A_scenario, A_budget), axis=0),
            b_ub=np.concatenate((np.zeros(N), np.asarray([1.0]))),
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not fitted.success or fitted.x is None:
            raise RuntimeError(
                "local full fallback LP failed with status "
                f"{fitted.status}: {fitted.message}"
            )
        action = fitted.x[:P] - fitted.x[P : 2 * P]
        action_l1 = float(np.abs(action).sum())
        if action_l1 > 1.0:
            action *= (1.0 - 8.0 * np.finfo(np.float64).eps) / action_l1
        pnl = scenario_return @ action - costs @ np.abs(action)
    pnl = scenario_return @ action - costs @ np.abs(action)
    band = observation_numerical_band(action, scenario_return, costs)
    statistics = weighted_mean_standard_error(pnl, relative_weight)
    loss = pnl < -band
    gain = pnl > band
    training_candidate = (
        not np.any(loss)
        and np.any(gain)
        and float(statistics["mean"]) > 3.0 * float(statistics["standard_error"])
    )
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(pnl)):
        raise FloatingPointError("local LP returned NaN or infinity")
    verify_inp(
        float(np.abs(action).sum()) <= 1.0 + np.finfo(np.float64).eps,
        "local LP budget repair failed",
    )
    return LocalMultiExpiryResult(
        action=action,
        training_pnl=pnl,
        training_band=band,
        training_minimum_pnl=float(np.min(pnl)),
        training_statistics=statistics,
        training_candidate=training_candidate,
        active_constraints=int(active.size),
        constraint_generation_iterations=iteration,
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    """Return a left-continuous weighted empirical quantile."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    verify_inp(values.size > 0 and values.shape == weights.shape, "invalid quantile arrays")
    verify_inp(0.0 <= probability <= 1.0, lambda: f"invalid probability {probability}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted quantile input contains NaN or infinity")
    verify_inp(np.all(weights >= 0.0) and float(weights.sum()) > 0.0, "invalid weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    target = probability * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)
    return float(values[order[index]])


__all__ = [
    "LocalMultiExpiryResult",
    "MultiExpiryStaticResult",
    "multi_expiry_lower_bound",
    "multi_expiry_numerical_band",
    "multi_expiry_scenario_returns",
    "multi_expiry_terminal_pnl",
    "observation_numerical_band",
    "solve_local_multi_expiry_portfolio",
    "solve_multi_expiry_static_arbitrage_lp",
    "weighted_quantile",
]
