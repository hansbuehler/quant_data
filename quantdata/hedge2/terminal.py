"""Nonanticipative terminal-wealth accounting and remaining-horizon objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quantdata.hedge2.objectives import power_oce


def shared_history_values(values: np.ndarray, histories: np.ndarray, *, tolerance: float) -> bool:
    """Check node-shared values with sorted reductions rather than quadratic masks."""
    order = np.argsort(histories, kind="stable")
    ordered = histories[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    maximum = np.maximum.reduceat(values[order], starts, axis=0)
    minimum = np.minimum.reduceat(values[order], starts, axis=0)
    return bool(np.max(maximum - minimum, initial=0.) <= tolerance)


@dataclass(frozen=True, kw_only=True)
class TerminalPaths:
    """Exogenous paths in one fixed initial-notional discounted numeraire.

    Gains are predictably alpha-adjusted before policy training. Costs
    are causal forecast half-spreads, not later observed market spreads.
    Live-contract masks include the following date for free settlement.
    History identifiers group paths sharing all information before trade.
    Termination flags become known only AFTER the corresponding gain.
    """

    gains: np.ndarray
    costs: np.ndarray
    features: np.ndarray
    history_ids: np.ndarray
    live_contracts: np.ndarray
    terminate_after_step: np.ndarray
    claim_gain: np.ndarray
    weights: np.ndarray
    initially_active: bool = True

    def __post_init__(self) -> None:
        if self.gains.ndim != 3:
            raise ValueError("Terminal gains must have path, time and frozen-contract axes")
        paths, steps, instruments = self.gains.shape
        if (min(paths, steps, instruments) < 1 or self.costs.shape != self.gains.shape
                or self.features.ndim != 3 or self.features.shape[:2] != (paths, steps)
                or self.history_ids.shape != (paths, steps)
                or not np.issubdtype(self.history_ids.dtype, np.integer)
                or self.live_contracts.shape != (steps + 1, instruments) or self.live_contracts.dtype != bool
                or self.terminate_after_step.shape != (paths, steps) or self.terminate_after_step.dtype != bool
                or self.claim_gain.shape != (paths,) or self.weights.shape != (paths,)
                or not all(np.isfinite(values).all() for values in (self.gains, self.costs, self.features, self.claim_gain, self.weights))
                or np.any(self.costs < 0) or np.any(self.weights < 0) or self.weights.sum() <= 0
                or np.any(self.live_contracts[1:] & ~self.live_contracts[:-1])):
            raise ValueError("Terminal path arrays, positive weights and contract lifetimes must align")
        if np.any(self.history_ids[:, 0] != self.history_ids[0, 0]):
            raise ValueError("All simulated paths start at one observed root history")
        active = np.full(paths, self.initially_active, dtype=bool)
        for step in range(steps):
            histories = self.history_ids[:, step]
            if not shared_history_values(self.features[:, step], histories, tolerance=1e-12):
                raise ValueError("Observable features differ between paths sharing the same trade-time history")
            if not shared_history_values(self.costs[:, step], histories, tolerance=1e-12):
                raise ValueError("Forecast transaction costs use information not yet observed")
            if not shared_history_values(active.astype(int), histories, tolerance=0.):
                raise ValueError("Knockout state differs between paths sharing the same history")
            if step and not shared_history_values(self.history_ids[:, step - 1], histories, tolerance=0.):
                raise ValueError("Information histories cannot merge after revealing different observations")
            active &= ~self.terminate_after_step[:, step]


@dataclass(frozen=True, kw_only=True)
class TerminalEvaluation:
    wealth: np.ndarray
    hedge_gain: np.ndarray
    entry_cost: np.ndarray
    positions: np.ndarray
    step_gains: np.ndarray
    step_costs: np.ndarray
    cap_hits: np.ndarray


@dataclass(frozen=True, kw_only=True)
class TerminalRisk:
    value: float
    mean_wealth: float
    std_wealth: float
    wealth_gradient: np.ndarray
    cash_shift: float | None = None
    cash_first_order_residual: float = 0.


def evaluate_terminal_policy(*, paths: TerminalPaths, initial_inventory: np.ndarray,
                             policy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                             target_side: int, position_cap: float | None = 1000.) -> TerminalEvaluation:
    """Trade from current inventory through expiry, with no final exit charge.

    The policy receives only current features, current holdings and time.
    Knockout liquidation occurs at mid after the observation-period gain;
    subsequent positions and cash gains are zero. Expiry also clears the
    matured inventory without another turnover fee or settlement payment.
    """
    count, steps, instruments = paths.gains.shape
    initial = np.asarray(initial_inventory, dtype=float)
    if (initial.shape != (instruments,) or not np.isfinite(initial).all() or target_side not in (-1, 1)
            or np.any(initial[~paths.live_contracts[0]] != 0.)
            or position_cap is not None and (not np.isfinite(position_cap) or position_cap <= 0
                                            or np.any(np.abs(initial) > position_cap))):
        raise ValueError("Initial inventory, side and optional position cap must be feasible")
    inventory = np.broadcast_to(initial, (count, instruments)).copy()
    active = np.full(count, paths.initially_active, dtype=bool)
    inventory[~active] = 0.
    positions = np.zeros_like(paths.gains)
    step_gains, step_costs = np.zeros((count, steps)), np.zeros((count, steps))
    cap_hits = np.zeros((count, steps), dtype=int)
    for step in range(steps):
        proposed = np.array(policy(paths.features[:, step].copy(), inventory.copy(), step), dtype=float, copy=True)
        if proposed.shape != inventory.shape or not np.isfinite(proposed).all():
            raise ValueError("The terminal policy returned misaligned or non-finite holdings")
        proposed[~active] = 0.
        proposed[:, ~paths.live_contracts[step]] = 0.
        if position_cap is not None and np.any(np.abs(proposed) > position_cap * (1. + 1e-12)):
            raise ValueError("Terminal policy holdings exceed the declared position cap")
        if not shared_history_values(proposed, paths.history_ids[:, step], tolerance=1e-10):
            raise ValueError("The policy anticipates a future branch or randomizes shared-history decisions")
        positions[:, step] = proposed
        step_costs[:, step] = np.sum(paths.costs[:, step] * np.abs(proposed - inventory), axis=1)
        step_gains[:, step] = np.sum(proposed * paths.gains[:, step], axis=1)
        cap_hits[:, step] = np.sum(np.abs(proposed) >= (1000. if position_cap is None else position_cap) * (1. - 1e-6), axis=1)
        inventory = proposed.copy()
        inventory[:, ~paths.live_contracts[step + 1]] = 0.
        active &= ~paths.terminate_after_step[:, step]
        inventory[~active] = 0.
    gain, cost = step_gains.sum(axis=1), step_costs.sum(axis=1)
    wealth = target_side * paths.claim_gain + gain - cost
    return TerminalEvaluation(wealth=wealth, hedge_gain=gain, entry_cost=cost, positions=positions,
                              step_gains=step_gains, step_costs=step_costs, cap_hits=cap_hits)


def terminal_risk(wealth: np.ndarray, weights: np.ndarray, *, objective: str,
                  risk_aversion: float) -> TerminalRisk:
    """Risk of NET terminal wealth, including path-dependent future trading fees.

    Finite MV is -mean + lambda*population_variance/2; infinite MV is
    half-variance alone. It can prefer costly loss-shaping strategies.
    Power-exponential optimizes its cash shift on the terminal distribution.
    """
    values, probabilities = np.asarray(wealth, dtype=float), np.asarray(weights, dtype=float)
    if (values.ndim != 1 or probabilities.shape != values.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or probabilities.sum() <= 0
            or objective not in ("mean_variance", "power_exponential") or risk_aversion <= 0
            or np.isnan(risk_aversion) or objective == "power_exponential" and not np.isfinite(risk_aversion)):
        raise ValueError("Terminal risk needs finite aligned wealth and the specified positive risk aversion")
    probabilities = probabilities / probabilities.sum()
    mean = float(probabilities @ values)
    centered = values - mean
    std = float(np.sqrt(probabilities @ centered**2))
    if objective == "mean_variance":
        finite = np.isfinite(risk_aversion)
        multiplier = risk_aversion if finite else 1.
        value = .5 * multiplier * std**2 - (mean if finite else 0.)
        gradient = probabilities * (multiplier * centered - (1. if finite else 0.))
        return TerminalRisk(value=value, mean_wealth=mean, std_wealth=std, wealth_gradient=gradient)
    oce = power_oce(values, probabilities, risk_aversion=risk_aversion)
    return TerminalRisk(value=-oce.certainty_equivalent, mean_wealth=mean, std_wealth=std,
        wealth_gradient=-probabilities * oce.marginal_utility, cash_shift=oce.cash_shift,
        cash_first_order_residual=oce.cash_first_order_residual)


def center_finite_tree_gains(gains: np.ndarray, weights: np.ndarray,
                           history_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact finite-tree reference only, NOT pooled centering of Monte Carlo paths.

    Each history must include its complete conditional innovation support.
    Sampled paths with unique future histories need separate conditional
    integration, not this empirical grouping operation.
    """
    if (gains.ndim != 3 or weights.shape != (len(gains),) or history_ids.shape != gains.shape[:2]
            or not np.isfinite(gains).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Finite-tree gains, weights and histories do not align")
    centered, removed = gains.copy(), np.zeros_like(gains)
    for step in range(gains.shape[1]):
        for history in np.unique(history_ids[:, step]):
            selected = history_ids[:, step] == history
            mass = float(weights[selected].sum())
            if mass <= 0 or selected.sum() < 2:
                raise ValueError("Finite-tree conditional integration needs complete nontrivial branch support")
            mean = weights[selected] @ gains[selected, step] / mass
            removed[selected, step] = mean
            centered[selected, step] -= mean
    return centered, removed
