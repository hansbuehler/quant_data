"""Current-premium position limits with unchanged financed cash and turnover."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from quantdata.hedge2.greek_hedge import GreekHedge, solve_greek_hedge
from quantdata.hedge2.optimization import OptimizedHedge
from quantdata.hedge2.parametric import ParametricHedge, ParametricOptimizer


@dataclass(frozen=True, kw_only=True)
class PremiumUnits:
    """Convert episode-normalized contract positions to current premium dollars.

    Prices are current cash prices divided by the ORIGINAL episode spot,
    without today's discount factor. Costs already carry the cumulative
    episode discount. This is a unit conversion, not a daily wealth reset.
    """

    normalized_prices: np.ndarray
    normalized_costs: np.ndarray
    position_cap: float = 10.

    def __post_init__(self) -> None:
        if (self.normalized_prices.ndim != 1 or not len(self.normalized_prices)
                or self.normalized_costs.shape != self.normalized_prices.shape
                or not np.isfinite(self.normalized_prices).all() or np.any(self.normalized_prices <= 0.)
                or not np.isfinite(self.normalized_costs).all() or np.any(self.normalized_costs < 0.)
                or not np.isfinite(self.position_cap) or self.position_cap <= 0.):
            raise ValueError("Premium units require aligned positive prices, nonnegative fees and a finite positive cap")

    @property
    def costs(self) -> np.ndarray:
        return self.normalized_costs / self.normalized_prices

    def to_premium(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape != self.normalized_prices.shape or not np.isfinite(positions).all():
            raise ValueError("Contract inventory must be finite and aligned with today's premium prices")
        return positions * self.normalized_prices

    def to_contracts(self, positions: np.ndarray) -> np.ndarray:
        if positions.shape != self.normalized_prices.shape or not np.isfinite(positions).all():
            raise ValueError("Premium positions must be finite and aligned with today's prices")
        return positions / self.normalized_prices

    def carried_inventory(self, positions: np.ndarray | None) -> tuple[np.ndarray, float]:
        """Project solver inventory only, retaining its exact mandatory turnover fee.

        For every feasible b, |b-old|=|b-clip(old)|+|old-clip(old)|.
        Old physical holdings need not satisfy today's post-trade limit.
        The second term is a deterministic cost, not a free liquidation.
        """
        old = np.zeros_like(self.normalized_prices) if positions is None else self.to_premium(positions)
        clipped = np.clip(old, -self.position_cap, self.position_cap)
        return clipped, float(self.costs @ np.abs(old - clipped))


@dataclass(frozen=True, kw_only=True)
class PremiumParametricHedge:
    result: ParametricHedge
    contract_positions: np.ndarray
    mandatory_turnover_cost: float


def _add_mandatory_cost(hedge: OptimizedHedge, *, cost: float,
                        objective: str, risk_aversion: float) -> OptimizedHedge:
    pure_variance = objective == "mean_variance" and np.isinf(risk_aversion)
    return replace(hedge, entry_cost=hedge.entry_cost + cost,
        training_mean=hedge.training_mean - cost,
        training_objective=hedge.training_objective + (0. if pure_variance else cost))


class PremiumParametricOptimizer:
    """Reuse the original Gaussian risk problem in premium-dollar coordinates."""

    def __init__(self, *, exposures: np.ndarray, covariance: np.ndarray, units: PremiumUnits) -> None:
        if exposures.ndim != 2 or exposures.shape[0] != len(units.normalized_prices):
            raise ValueError("Greek rows and premium prices must identify the same instruments")
        self.units = units
        self.optimizer = ParametricOptimizer(exposures=exposures / units.normalized_prices[:, None],
            covariance=covariance, costs=units.costs, position_cap=units.position_cap)

    def solve(self, *, target_exposures: np.ndarray, objective: str, risk_aversion: float,
              target_side: int = -1, previous_positions: np.ndarray | None = None) -> PremiumParametricHedge:
        previous, mandatory = self.units.carried_inventory(previous_positions)
        result = self.optimizer.solve(target_exposures=target_exposures, objective=objective,
            risk_aversion=risk_aversion, target_side=target_side, previous_positions=previous)
        result = replace(result, hedge=_add_mandatory_cost(result.hedge, cost=mandatory,
            objective=objective, risk_aversion=risk_aversion))
        return PremiumParametricHedge(result=result,
            contract_positions=self.units.to_contracts(result.hedge.positions), mandatory_turnover_cost=mandatory)


@dataclass(frozen=True, kw_only=True)
class PremiumGreekHedge:
    result: GreekHedge
    contract_positions: np.ndarray
    mandatory_turnover_cost: float


class GreekBudgetInfeasible(ValueError):
    """A cost budget cannot pay mandatory rebalancing into the agreed position box."""

    def __init__(self, *, full_entry_cost: float, mandatory_turnover_cost: float,
                 budget_fraction: float) -> None:
        self.full_entry_cost = full_entry_cost
        self.mandatory_turnover_cost = mandatory_turnover_cost
        self.discretionary_full_cost = full_entry_cost - mandatory_turnover_cost
        self.budget_fraction = budget_fraction
        self.turnover_budget = budget_fraction * full_entry_cost
        self.shortfall = mandatory_turnover_cost - self.turnover_budget
        self.portfolio_context: dict[str, object] | None = None
        super().__init__(f"Greek budget {self.turnover_budget:.17g} cannot pay "
            f"mandatory turnover cost {mandatory_turnover_cost:.17g} within the current-premium cap")


def solve_premium_greek(*, exposures: np.ndarray, target_exposures: np.ndarray,
                        units: PremiumUnits, previous_positions: np.ndarray | None = None,
                        risk_scales: np.ndarray | None = None, target_side: int = -1,
                        budget_fraction: float = 1.) -> PremiumGreekHedge:
    """Preserve the exact-match/reoptimized frontier and charge original turnover."""
    if not np.isfinite(budget_fraction) or not 0. <= budget_fraction <= 1.:
        raise ValueError("Greek cost-budget fraction must be in [0,1]")
    previous, mandatory = units.carried_inventory(previous_positions)
    arguments = dict(exposures=exposures / units.normalized_prices[:, None],
        target_exposures=target_exposures, costs=units.costs, previous_positions=previous,
        risk_scales=risk_scales, target_side=target_side, position_cap=units.position_cap)
    full = solve_greek_hedge(**arguments)

    def original_row_error(hedge: GreekHedge) -> float:
        physical = units.to_contracts(hedge.positions)
        residual = physical @ exposures + target_side * target_exposures
        scale = np.maximum(np.max(np.abs(exposures), axis=0), np.abs(target_exposures))
        scale = np.where(scale > 0., scale, 1.)
        error = float(np.max(np.abs(residual) / scale, initial=0.))
        if not np.isfinite(error):
            raise RuntimeError("Premium Greek exact match has a non-finite original-contract residual")
        return error

    initial_error = original_row_error(full)
    if initial_error > 1e-6:
        arguments["whiten_equalities"] = True
        recovered = solve_greek_hedge(**arguments)
        recovered_error = original_row_error(recovered)
        if recovered_error > 1e-6:
            raise RuntimeError(f"Premium Greek original-row recovery needs review: "
                f"initial residual={initial_error:.17g}; recovered residual={recovered_error:.17g}")
        full = replace(recovered, warnings=(*full.warnings,
            f"Original contract-row exact Greek residual={initial_error:.17g}; "
            f"equivalent-row numerical recovery residual={recovered_error:.17g}", *recovered.warnings))
    full_cost = full.entry_cost + mandatory
    available = budget_fraction * full_cost - mandatory
    tolerance = 64. * np.finfo(float).eps * max(full_cost, mandatory, np.finfo(float).tiny)
    if available < -tolerance:
        raise GreekBudgetInfeasible(full_entry_cost=full_cost,
            mandatory_turnover_cost=mandatory, budget_fraction=budget_fraction)
    result = full
    if budget_fraction < 1. and full.entry_cost > 0.:
        result = solve_greek_hedge(**arguments,
            budget_fraction=float(np.clip(available / full.entry_cost, 0., 1.)))
        reference_tolerance = 128. * np.finfo(float).eps * max(
            abs(full.entry_cost), abs(result.full_entry_cost), np.finfo(float).tiny)
        if abs(result.full_entry_cost - full.entry_cost) > reference_tolerance:
            raise RuntimeError("Premium Greek partial solve changed its independently recovered full-cost reference")
        result = replace(result, warnings=tuple(dict.fromkeys((*full.warnings, *result.warnings))))
    result = replace(result, entry_cost=result.entry_cost + mandatory, full_entry_cost=full_cost)
    budget = budget_fraction * full_cost
    budget_tolerance = max(1e-6 * full_cost, 32. * np.nextafter(0., 1.))
    if result.entry_cost > budget + budget_tolerance:
        raise RuntimeError(f"Premium Greek turnover {result.entry_cost:.17g} exceeds "
            f"budget {budget:.17g} with cash tolerance {budget_tolerance:.17g}")
    if result.entry_cost > budget:
        result = replace(result, warnings=(*result.warnings,
            f"Greek turnover budget roundoff: cost={result.entry_cost:.17g}, "
            f"budget={budget:.17g}, tolerance={budget_tolerance:.17g}"))
    return PremiumGreekHedge(result=result, contract_positions=units.to_contracts(result.positions),
        mandatory_turnover_cost=mandatory)
