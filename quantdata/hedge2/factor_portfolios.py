"""Joint approximate replication of factor shocks with bounded premium portfolios."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class FactorPortfolioConfig:
    instrument_cap: float = 10.
    gross_cap: float = 100.
    _gap_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        verify_inp(np.isfinite(self.instrument_cap) and self.instrument_cap > 0
                   and np.isfinite(self.gross_cap) and self.gross_cap > 0
                   and np.isfinite(self._gap_tolerance) and self._gap_tolerance > 0,
                   lambda: "Portfolio position, gross and numerical limits must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class FactorPortfolioFit:
    premium_positions: np.ndarray
    predicted_shocks: np.ndarray
    removed_target_mean: np.ndarray
    training_rmse: np.ndarray
    training_r_squared: np.ndarray
    target_std: np.ndarray
    portfolio_std: np.ndarray
    target_portfolio_correlation: np.ndarray
    target_correlation_matrix: np.ndarray
    portfolio_correlation_matrix: np.ndarray
    tracking_error_covariance: np.ndarray
    entry_cost: np.ndarray
    gross_positions: np.ndarray
    instrument_cap_hits: np.ndarray
    gross_cap_binding: np.ndarray
    primal_objective: np.ndarray
    dual_lower_bound: np.ndarray
    objective_gap: np.ndarray
    certified: bool
    instrument_return_rank: int
    solver: str
    status: str
    warnings: tuple[str, ...]


def box_gross_support(gradient: np.ndarray, *, instrument_cap: float, gross_cap: float) -> np.ndarray:
    """Support function of |b_i|<=cap, sum|b_i|<=gross, independently by factor."""
    ordered = np.sort(np.abs(gradient), axis=0)[::-1]
    allocations = np.clip(gross_cap - instrument_cap * np.arange(len(ordered)), 0., instrument_cap)
    return allocations @ ordered


def premium_to_cash_holdings(premium_positions: np.ndarray, current_cash_prices: np.ndarray) -> np.ndarray:
    """Units of the quoted asset per $1 tracked payout, before any exchange multiplier."""
    positions, prices = np.asarray(premium_positions, dtype=float), np.asarray(current_cash_prices, dtype=float)
    if (positions.ndim != 2 or prices.shape != (len(positions),)
            or not np.isfinite(positions).all() or not np.isfinite(prices).all() or np.any(prices <= 0.)):
        raise ValueError("Cash holdings require finite portfolios and positive unnormalized asset prices")
    return positions / prices[:, None]


def _correlation(covariance: np.ndarray) -> np.ndarray:
    deviations = np.sqrt(np.maximum(np.diag(covariance), 0.))
    denominator = deviations[:, None] * deviations[None, :]
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.)


def fit_factor_portfolios(*, instrument_returns: np.ndarray, factor_shocks: np.ndarray,
                          weights: np.ndarray, proportional_costs: np.ndarray,
                          config: FactorPortfolioConfig = FactorPortfolioConfig()) -> FactorPortfolioFit:
    """Minimize weighted joint squared tracking error without orthogonality or a ridge.

    Each factor gets its own $10-per-instrument and $100-gross premium
    portfolio. Financed spot and prepaid forwards use their notional/price
    bases alongside ITM option entry-premium returns. Target shocks retain
    their declared PCA units. Current entry costs are reported, not added
    as an undeclared tracking penalty. All statistics here are IN SAMPLE.
    """
    returns, targets = np.asarray(instrument_returns, dtype=float), np.asarray(factor_shocks, dtype=float)
    probabilities, costs = np.asarray(weights, dtype=float), np.asarray(proportional_costs, dtype=float)
    if (returns.ndim != 2 or targets.ndim != 2 or len(returns) != len(targets)
            or min(*returns.shape, targets.shape[1]) < 1 or probabilities.shape != (len(returns),)
            or costs.shape != (returns.shape[1],)
            or not all(np.isfinite(values).all() for values in (returns, targets, probabilities, costs))
            or np.any(probabilities < 0) or probabilities.sum() <= 0 or np.any(costs < 0)):
        raise ValueError("Factor matching needs finite aligned returns, shocks, weights and costs")
    probabilities = probabilities / probabilities.sum()
    if np.max(np.abs(probabilities @ returns), initial=0.) > 1e-10:
        raise ValueError("Remove instrument alpha before fitting tradable factor portfolios")
    removed_mean = probabilities @ targets
    targets = targets - removed_mean
    variable = cp.Variable((returns.shape[1], targets.shape[1]))
    expression = cp.multiply(np.sqrt(probabilities)[:, None], returns @ variable - targets)
    constraints = [variable <= config.instrument_cap, variable >= -config.instrument_cap,
                   cp.sum(cp.abs(variable), axis=0) <= config.gross_cap]
    zero_columns = np.max(np.abs(returns), axis=0) == 0.
    if zero_columns.any():
        constraints.append(variable[zero_columns] == 0.)
    problem = cp.Problem(cp.Minimize(.5 * cp.sum_squares(expression)), constraints)
    best = None
    collected = []
    for solver in ("CLARABEL", "OSQP"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if solver == "CLARABEL":
                    problem.solve(solver=solver, max_iter=1000, tol_gap_abs=1e-9,
                                  tol_gap_rel=1e-9, tol_feas=1e-9)
                else:
                    problem.solve(solver=solver, max_iter=200000, eps_abs=1e-9, eps_rel=1e-9)
                collected.extend(str(item.message) for item in caught)
            if variable.value is None or not np.isfinite(variable.value).all():
                collected.append(f"{solver}: no finite portfolio; status={problem.status}")
                continue
            proposed = np.asarray(variable.value)
            if (np.max(np.abs(proposed), initial=0.) > config.instrument_cap + 1e-5
                    or np.max(np.abs(proposed).sum(axis=0), initial=0.) > config.gross_cap + 1e-5):
                collected.append(f"{solver}: independently checked position limits exceeded")
                continue
            positions = np.clip(proposed, -config.instrument_cap, config.instrument_cap)
            positions[zero_columns] = 0.
            gross = np.abs(positions).sum(axis=0)
            positions *= np.minimum(1., config.gross_cap / np.maximum(gross, 1e-30))
            correction = float(np.max(np.abs(proposed - positions), initial=0.))
            if correction > 0:
                collected.append(f"Projected numerical cap roundoff; maximum adjustment={correction:.17g}")
            prediction = returns @ positions
            errors = prediction - targets
            objective = .5 * (probabilities @ errors**2)
            gradient = returns.T @ (probabilities[:, None] * errors)
            gap = np.sum(gradient * positions, axis=0) + box_gross_support(gradient,
                instrument_cap=config.instrument_cap, gross_cap=config.gross_cap)
            if np.min(gap, initial=0.) < -1e-8:
                raise RuntimeError("Factor tracking certificate violates weak duality")
            gap = np.maximum(gap, 0.)
            lower = objective - gap
            if best is None or objective.sum() < best[0].sum():
                best = (objective, positions, prediction, errors, lower, gap, solver, str(problem.status))
            if np.max(gap, initial=0.) <= config._gap_tolerance:
                break
        except (cp.error.SolverError, ValueError) as error:
            collected.append(f"{solver}: {error}")
    if best is None:
        raise RuntimeError("No independently feasible factor portfolios: " + "; ".join(collected))
    objective, positions, prediction, errors, lower, gap, solver, status = best
    target_variance = probabilities @ targets**2
    target_std = np.sqrt(target_variance)
    predicted_std = np.sqrt(probabilities @ prediction**2)
    correlation = np.divide(probabilities @ (targets * prediction), target_std * predicted_std,
        out=np.zeros_like(target_std), where=target_std * predicted_std > 0.)
    target_covariance = targets.T @ (probabilities[:, None] * targets)
    portfolio_covariance = prediction.T @ (probabilities[:, None] * prediction)
    r_squared = 1. - np.divide(2. * objective, target_variance,
        out=np.zeros_like(target_variance), where=target_variance > 0.)
    gross = np.abs(positions).sum(axis=0)
    return FactorPortfolioFit(premium_positions=positions, predicted_shocks=prediction,
        removed_target_mean=removed_mean, training_rmse=np.sqrt(2. * objective), training_r_squared=r_squared,
        target_std=target_std, portfolio_std=predicted_std, target_portfolio_correlation=correlation,
        target_correlation_matrix=_correlation(target_covariance),
        portfolio_correlation_matrix=_correlation(portfolio_covariance),
        tracking_error_covariance=errors.T @ (probabilities[:, None] * errors),
        entry_cost=costs @ np.abs(positions), gross_positions=gross,
        instrument_cap_hits=np.sum(np.abs(positions) >= config.instrument_cap * (1. - 1e-6), axis=0),
        gross_cap_binding=gross >= config.gross_cap * (1. - 1e-6), primal_objective=objective,
        dual_lower_bound=lower, objective_gap=gap, certified=bool(np.max(gap, initial=0.) <= config._gap_tolerance),
        instrument_return_rank=int(np.linalg.matrix_rank(np.sqrt(probabilities)[:, None] * returns)),
        solver=solver, status=status, warnings=tuple(collected))
