"""Cost-aware Gaussian hedges in SANOS spot/factor Greek coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri

from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import DlvGreeks

FloatArray = NDArray[np.float64]
ParametricObjective = Literal["mean-variance", "entropy", "power-exponential", "cvar"]


@dataclass(frozen=True, kw_only=True)
class ParametricAction:
    """One solved spot/option portfolio and its covariance diagnostics."""

    spot_action: float
    option_actions: FloatArray
    risk_variance: float
    transaction_cost_proxy: float
    solver_status: str
    cap_hits: int
    objective_value: float = 0.0
    converged: bool = True


def _gaussian_power_risk(standard_deviation: float, risk_parameter: float) -> tuple[float, float]:
    sigma = float(standard_deviation)
    risk = float(risk_parameter)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("standard_deviation must be finite and non-negative")
    if not math.isfinite(risk) or risk <= 0.0:
        raise ValueError("risk_parameter must be finite and positive")
    if risk * sigma <= 1.0e-7:
        return 0.5 * risk * sigma**2, risk * sigma

    scaled_sigma = risk * sigma

    def exponential(log_value: float) -> float:
        if log_value <= math.log(np.finfo(np.float64).tiny):
            return 0.0
        return math.exp(min(log_value, math.log(np.finfo(np.float64).max)))

    def first_order_condition(residual_mean: float) -> float:
        standardized_mean = residual_mean / sigma
        log_exponential_part = (
            -risk * residual_mean + 0.5 * scaled_sigma**2
            + float(log_ndtr(standardized_mean - scaled_sigma))
        )
        exponential_part = exponential(log_exponential_part)
        negative_probability = float(ndtr(-standardized_mean))
        density = math.exp(-0.5 * standardized_mean**2) / math.sqrt(2.0 * math.pi)
        negative_first_moment = residual_mean * negative_probability - sigma * density
        return exponential_part + negative_probability - risk * negative_first_moment - 1.0

    width = max(sigma, 1.0 / risk)
    lower, upper = -width, width
    for _ in range(64):
        if first_order_condition(lower) >= 0.0 and first_order_condition(upper) <= 0.0:
            break
        if first_order_condition(lower) < 0.0:
            lower *= 2.0
        if first_order_condition(upper) > 0.0:
            upper *= 2.0
    else:
        raise FloatingPointError("failed to bracket Gaussian power-OCE translation")
    residual_mean = float(brentq(first_order_condition, lower, upper, xtol=1.0e-13))
    standardized_mean = residual_mean / sigma
    shifted_argument = standardized_mean - scaled_sigma
    exponential_part = exponential(
        -risk * residual_mean + 0.5 * scaled_sigma**2 + float(log_ndtr(shifted_argument))
    )
    positive_probability = float(ndtr(standardized_mean))
    negative_probability = float(ndtr(-standardized_mean))
    density = math.exp(-0.5 * standardized_mean**2) / math.sqrt(2.0 * math.pi)
    negative_first_moment = residual_mean * negative_probability - sigma * density
    negative_second_moment = (
        (residual_mean**2 + sigma**2) * negative_probability
        - residual_mean * sigma * density
    )
    expected_utility = (
        (positive_probability - exponential_part) / risk
        + negative_first_moment - 0.5 * risk * negative_second_moment
    )
    risk_value = residual_mean - expected_utility
    log_mills_ratio = (
        -0.5 * shifted_argument**2 - 0.5 * math.log(2.0 * math.pi)
        - float(log_ndtr(shifted_argument))
    )
    exponential_z_moment = exponential_part * (exponential(log_mills_ratio) - scaled_sigma)
    risk_slope = density + scaled_sigma * negative_probability - exponential_z_moment
    if not np.isfinite(risk_value) or not np.isfinite(risk_slope) or risk_slope < 0.0:
        raise FloatingPointError("Gaussian power-OCE risk is invalid")
    return risk_value, risk_slope


def solve_parametric_hedge(
    *,
    target: DlvGreeks,
    options: DlvGreeks,
    shock_covariance: FloatArray,
    option_cost_weights: FloatArray,
    spot_cost_weight: float,
    risk_parameter: float,
    position_cap: float,
    objective: ParametricObjective = "mean-variance",
) -> ParametricAction:
    """Minimize a centered-Gaussian factor risk plus executable spread cost."""
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
        or objective not in ("mean-variance", "entropy", "power-exponential", "cvar")
    ):
        raise ValueError("hedge costs, objective, and risk controls are invalid")
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
    def solve_weighted(variance_aversion: float | None) -> tuple[FloatArray, str]:
        actions = cp.Variable(len(costs))
        residual = target_exposure + hedge_exposures @ actions
        factor_risk = covariance_root @ residual
        if variance_aversion is None:
            alpha = 1.0 / (1.0 + risk_parameter)
            quantile = float(ndtri(alpha))
            tail_loading = math.exp(-0.5 * quantile**2) / (math.sqrt(2.0 * math.pi) * alpha)
            risk_expression = tail_loading * cp.norm(factor_risk, 2)
        else:
            risk_expression = 0.5 * variance_aversion * cp.sum_squares(factor_risk)
        loss = 1.0e4 * (risk_expression + cp.norm1(cp.multiply(costs, actions)))
        problem = cp.Problem(cp.Minimize(loss),
                             [actions <= position_cap, actions >= -position_cap])
        problem.solve(solver="CLARABEL")
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or actions.value is None:
            raise FloatingPointError(f"SANOS parametric hedge failed: {problem.status}")
        solved = np.asarray(actions.value, dtype=np.float64)
        if not np.all(np.isfinite(solved)):
            raise FloatingPointError("SANOS parametric hedge contains non-finite actions")
        return solved, str(problem.status)

    if objective == "cvar":
        solved, solver_status = solve_weighted(None)
        converged = True
    elif objective == "power-exponential":
        variance_aversion = risk_parameter
        previous = np.zeros(len(costs), dtype=np.float64)
        converged = False
        for _ in range(50):
            solved, solver_status = solve_weighted(variance_aversion)
            exposure = target_exposure + hedge_exposures @ solved
            sigma = math.sqrt(max(float(exposure @ covariance @ exposure), 0.0))
            _, slope = _gaussian_power_risk(sigma, risk_parameter)
            updated_aversion = slope / sigma if sigma > 1.0e-12 else risk_parameter
            if (np.max(np.abs(solved - previous)) <= 1.0e-7
                    and abs(updated_aversion - variance_aversion)
                    <= 1.0e-6 * max(1.0, variance_aversion)):
                converged = True
                break
            previous = solved
            variance_aversion = 0.5 * (variance_aversion + updated_aversion)
    else:
        solved, solver_status = solve_weighted(risk_parameter)
        converged = True
    final_exposure = target_exposure + hedge_exposures @ solved
    final_variance = max(float(final_exposure @ covariance @ final_exposure), 0.0)
    if objective == "power-exponential":
        risk_value, _ = _gaussian_power_risk(math.sqrt(final_variance), risk_parameter)
    elif objective == "cvar":
        alpha = 1.0 / (1.0 + risk_parameter)
        quantile = float(ndtri(alpha))
        risk_value = (math.exp(-0.5 * quantile**2)
                      / (math.sqrt(2.0 * math.pi) * alpha)) * math.sqrt(final_variance)
    else:
        risk_value = 0.5 * risk_parameter * final_variance
    transaction_cost = float(costs @ np.abs(solved))
    return ParametricAction(
        spot_action=float(solved[0]),
        option_actions=solved[1:],
        risk_variance=final_variance,
        transaction_cost_proxy=transaction_cost,
        solver_status=solver_status,
        cap_hits=int(np.count_nonzero(np.isclose(np.abs(solved), position_cap))),
        objective_value=risk_value + transaction_cost,
        converged=converged,
    )
