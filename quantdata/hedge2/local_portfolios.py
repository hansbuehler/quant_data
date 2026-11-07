"""Bounded mixed-loading portfolios spanning the current option-price Greek space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from scipy.optimize import linprog

from quantdata.hedge2.factor_portfolios import FactorPortfolioConfig, box_gross_support


@version("0.2", dependencies=[FactorPortfolioConfig])
@dataclass(frozen=True, kw_only=True)
class LocalPortfolioConfig:
    limits: FactorPortfolioConfig = FactorPortfolioConfig()
    maximum_sweeps: int = 512
    _stationarity_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        verify_inp(isinstance(self.maximum_sweeps, int) and self.maximum_sweeps > 0
                   and np.isfinite(self._stationarity_tolerance) and self._stationarity_tolerance > 0,
                   lambda: "Local portfolio iterations and stationarity tolerance must be positive")


@dataclass(frozen=True, kw_only=True)
class LocalPortfolios:
    premium_positions: np.ndarray
    factor_loadings: np.ndarray
    singular_values: np.ndarray
    condition_number: float
    gross_positions: np.ndarray
    entry_costs: np.ndarray
    minimum_positions: np.ndarray
    maximum_positions: np.ndarray
    log_volumes: np.ndarray
    block_improvement_bounds: np.ndarray
    sweeps: int
    converged: bool


@dataclass(frozen=True, kw_only=True)
class CostRefinedBasis:
    premium_positions: np.ndarray
    greek_loadings: np.ndarray
    entry_costs: np.ndarray
    cost_lower_bounds: np.ndarray
    cost_gaps: np.ndarray
    scaled_exposure_residuals: np.ndarray
    solver_messages: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class LocalBasisHedge:
    basket_holdings: np.ndarray
    extra_spot_position: float
    premium_positions: np.ndarray
    greek_residual: np.ndarray
    augmented_condition_number: float


def _extreme_portfolio(scores: np.ndarray, *, instrument_cap: float, gross_cap: float) -> np.ndarray:
    """Exact linear support maximizer of a box intersected with a gross-dollar ball."""
    order = np.argsort(np.abs(scores), kind="stable")[::-1]
    amounts = np.clip(gross_cap - instrument_cap * np.arange(len(scores)), 0., instrument_cap)
    result = np.zeros(len(scores))
    result[order] = np.sign(scores[order]) * amounts
    return result


def fit_local_portfolios(*, price_factor_greeks: np.ndarray, proportional_costs: np.ndarray,
                         config: LocalPortfolioConfig = LocalPortfolioConfig()) -> LocalPortfolios:
    """Find a well-spanning local basis; do not replicate historical latent innovations.

    Greeks are asset-by-PC price responses PER DOLLAR of current premium.
    Each basket can mix every PC and retain spot exposure, handled by a
    separately traded underlying. No zero off-diagonal Greek or portfolio
    covariance constraints are imposed. Payout per factor is an output,
    not an arbitrary requirement to pay $1 for a native PC unit.

    Feasible SVD initialization is followed by exact coordinate maximizers
    of log absolute loading volume. This is a local joint-coverage heuristic,
    NOT a globally certified optimum or an OOS replication claim. Costs are
    reported, not hidden in that coverage objective. Later hedge costs and
    limits must be computed on NET look-through instrument positions.
    """
    greeks = np.asarray(price_factor_greeks, dtype=float)
    costs = np.asarray(proportional_costs, dtype=float)
    if (greeks.ndim != 2 or min(greeks.shape) < 1 or costs.shape != (len(greeks),)
            or not np.isfinite(greeks).all() or not np.isfinite(costs).all() or np.any(costs < 0)):
        raise ValueError("Local portfolios require finite asset-by-factor Greeks and nonnegative costs")
    if np.linalg.matrix_rank(greeks) != greeks.shape[1]:
        raise ValueError("The current option-price Greeks cannot span every requested factor")
    directions = np.linalg.svd(greeks, full_matrices=False)[0]
    limits = config.limits
    scales = np.minimum(limits.instrument_cap / np.max(np.abs(directions), axis=0),
                        limits.gross_cap / np.sum(np.abs(directions), axis=0))
    positions = directions * scales
    loading = positions.T @ greeks
    _, initial_volume = np.linalg.slogdet(loading)
    volumes = [float(initial_volume)]
    converged = False
    improvements = np.empty(greeks.shape[1])
    for sweep in range(config.maximum_sweeps):
        for factor in range(greeks.shape[1]):
            inverse_column = np.linalg.solve(loading, np.eye(greeks.shape[1])[:, factor])
            scores = greeks @ inverse_column
            candidate = _extreme_portfolio(scores, instrument_cap=limits.instrument_cap, gross_cap=limits.gross_cap)
            gain = float(scores @ candidate)
            if gain > 1. + config._stationarity_tolerance:
                positions[:, factor] = candidate
                loading = positions.T @ greeks
        _, volume = np.linalg.slogdet(loading)
        if not np.isfinite(volume) or volume < volumes[-1] - 1e-9:
            raise RuntimeError("The local coverage update lost rank or reduced its loading volume")
        volumes.append(float(volume))
        scores = greeks @ np.linalg.inv(loading)
        improvements = np.maximum(0., box_gross_support(scores,
            instrument_cap=limits.instrument_cap, gross_cap=limits.gross_cap) - 1.)
        if improvements.max() <= config._stationarity_tolerance:
            converged = True
            break
    singular = np.linalg.svd(loading, compute_uv=False)
    if (np.max(np.abs(positions), initial=0.) > limits.instrument_cap + 1e-10
            or np.max(np.abs(positions).sum(axis=0), initial=0.) > limits.gross_cap + 1e-10):
        raise RuntimeError("Local baskets exceed their independently replayed dollar limits")
    return LocalPortfolios(premium_positions=positions, factor_loadings=loading,
        singular_values=singular, condition_number=float(singular[0] / singular[-1]),
        gross_positions=np.abs(positions).sum(axis=0), entry_costs=costs @ np.abs(positions),
        minimum_positions=positions.min(axis=0), maximum_positions=positions.max(axis=0),
        log_volumes=np.array(volumes), block_improvement_bounds=improvements,
        sweeps=sweep+1, converged=converged)


def lookthrough_positions(*, basis_positions: np.ndarray, basket_holdings: np.ndarray,
                          extra_spot_position: float = 0.) -> np.ndarray:
    """Net overlapping baskets before computing asset-level costs or risk limits."""
    basis, holdings = np.asarray(basis_positions, dtype=float), np.asarray(basket_holdings, dtype=float)
    if (basis.ndim != 2 or holdings.shape != (basis.shape[1],) or not np.isfinite(basis).all()
            or not np.isfinite(holdings).all() or not np.isfinite(extra_spot_position)):
        raise ValueError("Basket holdings and dollar allocation matrix must align")
    positions = basis @ holdings
    positions[0] += extra_spot_position
    return positions


def hedge_in_local_basis(*, exposure_map: np.ndarray, basis_positions: np.ndarray,
                         target_exposures: np.ndarray, target_side: int) -> LocalBasisHedge:
    """Offset today's target Greeks using mixed baskets plus separately traded spot.

    Exposure rows contain spot and PC price responses per dollar invested.
    Targets retain the claim payout units. Basis-dollar constraints do NOT
    constrain the final combined hedge; its net positions require their own
    limit and turnover checks before production trading.
    """
    exposures = np.asarray(exposure_map, dtype=float)
    basis = np.asarray(basis_positions, dtype=float)
    target = np.asarray(target_exposures, dtype=float)
    if (exposures.ndim != 2 or basis.ndim != 2 or min(*exposures.shape) < 1
            or basis.shape != (exposures.shape[1], exposures.shape[0] - 1)
            or target.shape != (exposures.shape[0],) or target_side not in (-1, 1)
            or not all(np.isfinite(value).all() for value in (exposures, basis, target))):
        raise ValueError("Mixed-basis hedge Greeks, baskets, target and side must align")
    augmented = np.c_[exposures[:, 0], exposures @ basis]
    if np.linalg.matrix_rank(augmented) != len(target):
        raise ValueError("Spot and the local baskets cannot span all target Greek axes")
    coefficients = np.linalg.solve(augmented, -target_side * target)
    positions = lookthrough_positions(basis_positions=basis, basket_holdings=coefficients[1:],
                                      extra_spot_position=float(coefficients[0]))
    residual = exposures @ positions + target_side * target
    scale = max(float(np.max(np.abs(target))), np.finfo(float).tiny)
    if np.max(np.abs(residual)) > 1e-8 * scale:
        raise RuntimeError("Mixed-basis hedge failed its independently replayed exposure check")
    return LocalBasisHedge(basket_holdings=coefficients[1:], extra_spot_position=float(coefficients[0]),
        premium_positions=positions, greek_residual=residual,
        augmented_condition_number=float(np.linalg.cond(augmented)))


def minimize_basis_costs(*, exposure_map: np.ndarray, premium_positions: np.ndarray,
                         proportional_costs: np.ndarray,
                         limits: FactorPortfolioConfig = FactorPortfolioConfig()) -> CostRefinedBasis:
    """Preserve every MIXED Greek loading, including spot, while minimizing entry cost.

    This does not impose any new zero or orthogonality. Each column is a
    cost-optimal representation of its existing exposure vector within the
    same box/gross limits. Independently evaluated box-dual lower bounds
    distinguish objective accuracy from a solver's status message.
    """
    greeks, initial = np.asarray(exposure_map, dtype=float), np.asarray(premium_positions, dtype=float)
    costs = np.asarray(proportional_costs, dtype=float)
    if (greeks.ndim != 2 or initial.ndim != 2 or min(*greeks.shape, initial.shape[1]) < 1
            or initial.shape[0] != greeks.shape[1] or costs.shape != (initial.shape[0],)
            or not all(np.isfinite(values).all() for values in (greeks, initial, costs)) or np.any(costs < 0)
            or np.max(np.abs(initial), initial=0.) > limits.instrument_cap + 1e-8
            or np.max(np.abs(initial).sum(axis=0), initial=0.) > limits.gross_cap + 1e-8):
        raise ValueError("Cost refinement needs feasible dollar baskets, aligned Greeks and nonnegative fees")
    target = greeks @ initial
    asset_count, basket_count = initial.shape
    objective = np.r_[costs, costs]
    gross_row = np.ones((1, 2 * asset_count))
    positions = np.empty_like(initial)
    lower, gaps, residuals, messages = [], [], [], []
    for basket in range(basket_count):
        scale = np.maximum(np.max(np.abs(greeks), axis=1), np.abs(target[:, basket]))
        scale = np.where(scale > 0., scale, 1.)
        equality = np.c_[greeks, -greeks] / scale[:, None]
        rhs = target[:, basket] / scale
        result = linprog(c=objective, A_ub=gross_row, b_ub=[limits.gross_cap], A_eq=equality, b_eq=rhs,
            bounds=(0., limits.instrument_cap), method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if result.x is None or not np.isfinite(result.x).all():
            raise RuntimeError(f"Local mixed-basket cost refinement returned no finite candidate: {result.message}")
        candidate = np.clip(result.x[:asset_count] - result.x[asset_count:], -limits.instrument_cap, limits.instrument_cap)
        candidate *= min(1., limits.gross_cap / max(float(np.abs(candidate).sum()), 1e-30))
        residual = float(np.max(np.abs((greeks @ candidate - target[:, basket]) / scale)))
        if residual > 1e-7:
            raise RuntimeError(f"Local mixed-basket cost refinement needs review: {result.message}; exposure residual={residual}")
        dual = np.minimum(np.asarray(result.ineqlin.marginals), 0.)
        equality_dual = np.asarray(result.eqlin.marginals)
        if (dual.shape != (1,) or equality_dual.shape != rhs.shape
                or not np.isfinite(dual).all() or not np.isfinite(equality_dual).all()):
            raise RuntimeError(f"Local mixed-basket cost refinement has no usable objective bound: {result.message}")
        coefficient = objective - equality.T @ equality_dual - gross_row.T @ dual
        bound = float(rhs @ equality_dual + limits.gross_cap * dual[0]
            + limits.instrument_cap * np.minimum(coefficient, 0.).sum())
        actual_cost = float(costs @ np.abs(candidate))
        gap = actual_cost - bound
        if gap < -1e-8:
            raise RuntimeError("The local basket cost bound violates weak duality")
        if gap > limits._gap_tolerance * max(1., actual_cost):
            raise RuntimeError(f"Local mixed-basket cost refinement needs greater objective accuracy: {result.message}; gap={gap}")
        positions[:, basket] = candidate
        lower.append(bound)
        gaps.append(max(0., gap))
        residuals.append(residual)
        messages.append(str(result.message))
    return CostRefinedBasis(premium_positions=positions, greek_loadings=greeks @ positions,
        entry_costs=costs @ np.abs(positions), cost_lower_bounds=np.array(lower), cost_gaps=np.array(gaps),
        scaled_exposure_residuals=np.array(residuals), solver_messages=tuple(messages))
