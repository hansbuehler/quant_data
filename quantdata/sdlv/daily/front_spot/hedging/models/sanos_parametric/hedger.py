"""Cost-aware mean-variance hedge in SANOS spot/factor Greek coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import DlvGreeks

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class ParametricAction:
    """One solved spot/option portfolio and its covariance diagnostics."""

    spot_action: float
    option_actions: FloatArray
    risk_variance: float
    transaction_cost_proxy: float
    solver_status: str
    cap_hits: int


def solve_parametric_hedge(
    *,
    target: DlvGreeks,
    options: DlvGreeks,
    shock_covariance: FloatArray,
    option_cost_weights: FloatArray,
    spot_cost_weight: float,
    risk_parameter: float,
    position_cap: float,
) -> ParametricAction:
    """Minimize factor-P&L variance plus executable spread-cost proxy."""
    if len(target.model_prices) != 1 or target.factor_rank != options.factor_rank:
        raise ValueError("target and option Greek dimensions do not align")
    rank = target.factor_rank
    covariance = np.asarray(shock_covariance, dtype=np.float64)
    costs = np.concatenate(
        (np.asarray([spot_cost_weight]), np.asarray(option_cost_weights))
    )
    if covariance.shape != (rank + 1, rank + 1) or not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError("shock covariance must be a finite spot/factor matrix")
    if (
        costs.shape != (len(options.model_prices) + 1,)
        or np.any(~np.isfinite(costs) | (costs < 0.0))
        or not np.isfinite(risk_parameter)
        or risk_parameter <= 0.0
        or not np.isfinite(position_cap)
        or position_cap <= 0.0
    ):
        raise ValueError("hedge costs must be non-negative and risk controls positive and finite")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if np.min(eigenvalues) < -1.0e-12:
        raise ValueError("shock covariance is not positive semidefinite")
    covariance_root = np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
    target_exposure = np.concatenate(
        (target.spot_deltas, target.factor_vegas[0])
    )
    hedge_exposures = np.column_stack(
        (
            np.concatenate((np.ones(1), options.spot_deltas)),
            np.vstack((np.zeros((1, rank)), options.factor_vegas)),
        )
    ).T
    actions = cp.Variable(len(costs))
    residual = target_exposure + hedge_exposures @ actions
    objective = 1.0e4 * (
        0.5 * risk_parameter * cp.sum_squares(covariance_root @ residual)
        + cp.norm1(cp.multiply(costs, actions))
    )
    problem = cp.Problem(
        cp.Minimize(objective),
        [actions <= position_cap, actions >= -position_cap],
    )
    problem.solve(solver="CLARABEL")
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or actions.value is None:
        raise FloatingPointError(f"SANOS parametric hedge failed: {problem.status}")
    solved = np.asarray(actions.value, dtype=np.float64)
    if not np.all(np.isfinite(solved)):
        raise FloatingPointError("SANOS parametric hedge contains non-finite actions")
    final_exposure = target_exposure + hedge_exposures @ solved
    return ParametricAction(
        spot_action=float(solved[0]),
        option_actions=solved[1:],
        risk_variance=float(final_exposure @ covariance @ final_exposure),
        transaction_cost_proxy=float(costs @ np.abs(solved)),
        solver_status=str(problem.status),
        cap_hits=int(np.count_nonzero(np.isclose(np.abs(solved), position_cap))),
    )
