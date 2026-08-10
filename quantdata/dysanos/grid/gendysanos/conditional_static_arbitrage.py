"""Conditional-support helpers for one-expiry static portfolios."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from cdxcore.err import verify_inp
from scipy.optimize import linprog


def terminal_portfolio_pnl(
    action: np.ndarray,
    terminal_spot: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Literal["raw", "delta"],
) -> np.ndarray:
    """Evaluate one entry-state portfolio at arbitrary terminal spot values."""

    action = np.asarray(action, dtype=np.float64)
    terminal_spot = np.asarray(terminal_spot, dtype=np.float64)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    K = option_price.size
    verify_inp(
        action.shape == costs.shape == (K + 1,)
        and delta.shape == cash_strike.shape == (K,),
        "terminal-portfolio arrays have incompatible shapes",
    )
    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")
    values = (spot, *action, *terminal_spot.ravel(), *option_price, *delta, *cash_strike, *costs)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("terminal-portfolio input contains NaN or infinity")
    verify_inp(spot > 0.0, lambda: f"spot must be positive, got {spot}")
    verify_inp(np.all(costs >= 0.0), "terminal-portfolio costs must be nonnegative")

    y = terminal_spot.reshape(-1)
    option_return = np.maximum(y[:, None] - cash_strike[None, :], 0.0) - option_price
    if hedge == "delta":
        option_return -= delta[None, :] * (y[:, None] - spot)
    scenario_return = np.concatenate(((y - spot)[:, None], option_return), axis=1)
    pnl = scenario_return @ action - costs @ np.abs(action)
    if not np.all(np.isfinite(pnl)):
        raise FloatingPointError("terminal portfolio P&L is non-finite")
    return pnl.reshape(terminal_spot.shape)


def solve_conditional_support_lp(
    terminal_spot: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Literal["raw", "delta"],
) -> tuple[np.ndarray, float]:
    """Maximize the minimum P&L over an empirical conditional support sample."""

    terminal_spot = np.asarray(terminal_spot, dtype=np.float64).reshape(-1)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    K = option_price.size
    verify_inp(terminal_spot.size > 0, "conditional support cannot be empty")
    verify_inp(
        costs.shape == (K + 1,) and delta.shape == cash_strike.shape == (K,),
        "conditional-support LP arrays have incompatible shapes",
    )
    scenario_return = np.empty((terminal_spot.size, K + 1), dtype=np.float64)
    scenario_return[:, 0] = terminal_spot - spot
    option_return = (
        np.maximum(terminal_spot[:, None] - cash_strike[None, :], 0.0)
        - option_price[None, :]
    )
    if hedge == "delta":
        option_return -= delta[None, :] * (terminal_spot[:, None] - spot)
    elif hedge != "raw":
        raise ValueError(f"unknown hedge {hedge!r}")
    scenario_return[:, 1:] = option_return

    J = K + 1
    # Variables are long[J], short[J], and the guaranteed sample margin.
    A_scenario = np.concatenate(
        (
            -(scenario_return - costs[None, :]),
            -(-scenario_return - costs[None, :]),
            np.ones((terminal_spot.size, 1)),
        ),
        axis=1,
    )
    A_budget = np.concatenate((np.ones(2 * J), np.asarray([0.0])))[None, :]
    objective = np.zeros(2 * J + 1, dtype=np.float64)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=np.concatenate((A_scenario, A_budget), axis=0),
        b_ub=np.concatenate((np.zeros(terminal_spot.size), np.asarray([1.0]))),
        bounds=[(0.0, None)] * (2 * J) + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"conditional-support LP failed with status {result.status}: {result.message}"
        )
    action = result.x[:J] - result.x[J : 2 * J]
    margin = float(result.x[-1])
    if not np.all(np.isfinite(action)) or not math.isfinite(margin):
        raise FloatingPointError("conditional-support LP solution is non-finite")
    verify_inp(
        float(np.abs(action).sum()) <= 1.0 + 1.0e-8,
        "conditional-support LP violated its L1 budget",
    )
    return action, margin


def frozen_payoff_cone_lower_bound(
    action: np.ndarray,
    *,
    spot: float,
    option_price: np.ndarray,
    delta: np.ndarray,
    cash_strike: np.ndarray,
    costs: np.ndarray,
    hedge: Literal["raw", "delta"],
) -> tuple[float, float]:
    """Return a frozen action's lower bound over every terminal spot."""

    action = np.asarray(action, dtype=np.float64)
    option_price = np.asarray(option_price, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cash_strike = np.asarray(cash_strike, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    kink_pnl = terminal_portfolio_pnl(
        action,
        np.concatenate(([0.0], cash_strike)),
        spot=spot,
        option_price=option_price,
        delta=delta,
        cash_strike=cash_strike,
        costs=costs,
        hedge=hedge,
    )
    option_slope = 1.0 - delta if hedge == "delta" else np.ones_like(delta)
    tail_slope = float(action[0] + action[1:] @ option_slope)
    slope_tolerance = (
        32.0 * np.finfo(np.float32).eps * max(1.0, float(np.abs(action).sum()))
    )
    lower_bound = (
        float(kink_pnl.min())
        if tail_slope >= -slope_tolerance
        else -math.inf
    )
    return lower_bound, tail_slope


def weighted_mean_standard_error(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return a mean and finite-sample standard error for normalized weights."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    verify_inp(values.size > 1, "at least two values are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or infinity")
    if weights is None:
        weights = np.ones(values.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    verify_inp(weights.shape == values.shape, "value and weight shapes differ")
    if not np.all(np.isfinite(weights)) or not np.all(weights >= 0.0):
        raise ValueError("weights must be finite and nonnegative")
    weight_sum = float(weights.sum())
    verify_inp(weight_sum > 0.0, "at least one weight must be positive")
    normalized = weights / weight_sum
    weight_square = float(np.square(normalized).sum())
    mean = float(normalized @ values)
    centered = values - mean
    denominator = max(np.finfo(np.float64).eps, 1.0 - weight_square)
    standard_error = math.sqrt(float(np.sum(np.square(normalized * centered))) / denominator)
    return {
        "observations": int(values.size),
        "mean": mean,
        "standard_error": standard_error,
        "mean_over_standard_error": (
            mean / standard_error if standard_error > 0.0 else math.inf
        ),
        "effective_sample_size": 1.0 / weight_square,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


__all__ = [
    "frozen_payoff_cone_lower_bound",
    "solve_conditional_support_lp",
    "terminal_portfolio_pnl",
    "weighted_mean_standard_error",
]
