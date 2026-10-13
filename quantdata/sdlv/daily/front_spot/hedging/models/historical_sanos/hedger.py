"""Empirical four-objective training on historical public SANOS returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosPolicyFit:
    """One fitted static action and held-out training diagnostics."""

    action: FloatArray
    cash_translation: float | None
    training_loss: float
    validation_loss: float
    iterations: int
    converged: bool
    maximum_absolute_gradient: float
    cap_hits: int


@dataclass(frozen=True, kw_only=True)
class HistoricalSanosHedgeFit:
    """Claim, zero-claim, and differenced empirical policies."""

    action: FloatArray
    claim_policy: HistoricalSanosPolicyFit
    zero_policy: HistoricalSanosPolicyFit
    objective: float
    converged: bool
    internal_cap_hits: int
    executable_cap_hits: int
    cap_hits: int


def fit_historical_sanos_hedge(
    *,
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> HistoricalSanosHedgeFit:
    """Fit claim and zero empirical policies and return their difference."""
    instruments, claim, costs = _validated_inputs(
        instrument_returns,
        claim_returns,
        linear_transaction_costs,
    )
    claim_policy = _fit_policy(
        instruments,
        claim,
        costs,
        config=config,
    )
    zero_policy = _fit_policy(
        instruments,
        np.zeros_like(claim),
        costs,
        config=config,
    )
    raw_action = claim_policy.action - zero_policy.action
    action = np.clip(raw_action, -config.position_cap, config.position_cap)
    executable_cap_hits = int(np.count_nonzero(action != raw_action))
    internal_cap_hits = claim_policy.cap_hits + zero_policy.cap_hits
    cap_hits = internal_cap_hits + executable_cap_hits
    objective = _objective_and_gradient(
        np.concatenate(
            (
                action,
                np.asarray(
                    [0.0 if claim_policy.cash_translation is None else claim_policy.cash_translation]
                ),
            )
        ),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL,
    )[0]
    return HistoricalSanosHedgeFit(
        action=action,
        claim_policy=claim_policy,
        zero_policy=zero_policy,
        objective=objective,
        converged=claim_policy.converged and zero_policy.converged and cap_hits == 0,
        internal_cap_hits=internal_cap_hits,
        executable_cap_hits=executable_cap_hits,
        cap_hits=cap_hits,
    )


def _fit_policy(
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> HistoricalSanosPolicyFit:
    validation_count = max(1, int(round(len(instruments) * config.validation_fraction)))
    split = len(instruments) - validation_count
    if split < 2:
        raise ValueError("historical SANOS training split has fewer than two rows")
    include_translation = config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    parameter_count = instruments.shape[1] + int(include_translation)
    initial = np.zeros(parameter_count, dtype=np.float64)
    bounds = [
        (-config.solver_position_cap, config.solver_position_cap)
    ] * instruments.shape[1]
    if include_translation:
        bounds.append((None, None))
    selection = _minimize_policy(
        initial,
        instruments[:split],
        claim[:split],
        costs,
        config=config,
        include_translation=include_translation,
        bounds=bounds,
    )
    validation_loss = _objective_and_gradient(
        selection.x,
        instruments[split:],
        claim[split:],
        costs,
        config=config,
        include_translation=include_translation,
    )[0]
    final = _minimize_policy(
        selection.x,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
        bounds=bounds,
    )
    final_loss, final_gradient = _objective_and_gradient(
        final.x,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
    )
    action = np.asarray(final.x[: instruments.shape[1]], dtype=np.float64)
    cap_hits = int(
        np.count_nonzero(
            np.abs(action) >= 0.999 * config.solver_position_cap
        )
    )
    optimality = float(
        getattr(final, "optimality", np.max(np.abs(final_gradient)))
    )
    return HistoricalSanosPolicyFit(
        action=action,
        cash_translation=float(final.x[-1]) if include_translation else None,
        training_loss=float(final_loss),
        validation_loss=float(validation_loss),
        iterations=int(selection.nit) + int(final.nit),
        converged=bool(selection.success and final.success and cap_hits == 0),
        maximum_absolute_gradient=optimality,
        cap_hits=cap_hits,
    )


def _minimize_policy(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
    bounds: list[tuple[float | None, float | None]],
):
    if config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
        return _minimize_mean_variance(
            initial,
            instruments,
            claim,
            costs,
            config=config,
        )
    if config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
        return _minimize_mean_volatility(
            initial,
            instruments,
            claim,
            costs,
            config=config,
        )

    def value_and_gradient(parameters: NDArray[np.floating]) -> tuple[float, FloatArray]:
        return _objective_and_gradient(
            parameters,
            instruments,
            claim,
            costs,
            config=config,
            include_translation=include_translation,
        )

    return minimize(
        value_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": config.maximum_iterations,
            "gtol": config.gradient_tolerance,
            "ftol": 1.0e-14,
        },
    )


def _minimize_mean_volatility(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> OptimizeResult:
    """Solve the empirical mean-volatility cost-lasso as an exact SOCP."""
    observations, instrument_count = instruments.shape
    instrument_center = np.mean(instruments, axis=0)
    claim_center = float(np.mean(claim))
    centered_instruments = instruments - instrument_center
    centered_claim = claim - claim_center
    action = cp.Variable(instrument_count)
    expected_wealth = claim_center + instrument_center @ action - costs @ cp.abs(action)
    standard_deviation = cp.norm(
        centered_claim + centered_instruments @ action,
        2,
    ) / math.sqrt(observations)
    objective = cp.Minimize(
        -expected_wealth + config.risk_parameter * standard_deviation
    )
    problem = cp.Problem(objective)
    action.value = np.asarray(initial[:instrument_count], dtype=np.float64)
    try:
        problem.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            max_iter=config.maximum_iterations,
            tol_gap_abs=config.gradient_tolerance,
            tol_gap_rel=config.gradient_tolerance,
            tol_feas=config.gradient_tolerance,
        )
        if problem.status in {cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE}:
            problem = cp.Problem(
                objective,
                [
                    action >= -config.solver_position_cap,
                    action <= config.solver_position_cap,
                ],
            )
            action.value = np.asarray(initial[:instrument_count], dtype=np.float64)
            problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                max_iter=config.maximum_iterations,
                tol_gap_abs=config.gradient_tolerance,
                tol_gap_rel=config.gradient_tolerance,
                tol_feas=config.gradient_tolerance,
            )
    except cp.error.SolverError as error:
        return OptimizeResult(
            x=np.asarray(initial[:instrument_count], dtype=np.float64),
            nit=0,
            success=False,
            optimality=math.inf,
            message=f"CLARABEL failed: {error}",
        )
    if action.value is None:
        fitted_action = np.asarray(initial[:instrument_count], dtype=np.float64)
    else:
        fitted_action = np.clip(
            np.asarray(action.value, dtype=np.float64),
            -config.solver_position_cap,
            config.solver_position_cap,
        )
    residual = centered_claim + centered_instruments @ fitted_action
    residual_standard_deviation = math.sqrt(float(np.mean(residual**2)))
    zero_residual_tolerance = max(
        config.gradient_tolerance,
        np.finfo(np.float64).eps,
    ) * max(1.0, float(np.std(claim, ddof=0)))
    if residual_standard_deviation > zero_residual_tolerance:
        smooth_gradient = (
            -instrument_center
            + config.risk_parameter
            * (centered_instruments.T @ residual)
            / (observations * residual_standard_deviation)
        )
        optimality = _maximum_kkt_violation(
            fitted_action,
            smooth_gradient,
            costs,
            position_cap=config.solver_position_cap,
        )
    else:
        # At zero residual volatility the Euclidean norm is nondifferentiable;
        # CLARABEL's conic certificate is the relevant optimality test.
        optimality = 0.0 if problem.status == cp.OPTIMAL else math.inf
    return OptimizeResult(
        x=fitted_action,
        nit=int(problem.solver_stats.num_iters or 0),
        success=problem.status == cp.OPTIMAL,
        optimality=optimality,
        message=problem.status,
        fun=float(problem.value) if problem.value is not None else math.inf,
    )


def _minimize_mean_variance(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
) -> OptimizeResult:
    """Solve the quadratic spread-cost problem by coordinate descent."""
    centered_instruments = instruments - np.mean(instruments, axis=0)
    centered_claim = claim - np.mean(claim)
    risk = config.risk_parameter
    covariance = centered_instruments.T @ centered_instruments / len(instruments)
    hessian = risk * covariance
    hessian.flat[:: len(hessian) + 1] += config.quadratic_ridge
    linear = (
        risk * (centered_instruments.T @ centered_claim / len(instruments))
        - np.mean(instruments, axis=0)
    )
    action = np.asarray(initial[: instruments.shape[1]], dtype=np.float64).copy()
    gradient = hessian @ action + linear
    converged = False
    optimality = math.inf
    iterations = 0
    for iterations in range(1, config.maximum_iterations + 1):
        for coordinate in range(len(action)):
            diagonal = hessian[coordinate, coordinate]
            if diagonal <= 0.0:
                raise FloatingPointError("mean-variance Hessian diagonal is not positive")
            updated = np.clip(
                _soft_threshold(
                    diagonal * action[coordinate] - gradient[coordinate],
                    costs[coordinate],
                )
                / diagonal,
                -config.solver_position_cap,
                config.solver_position_cap,
            )
            change = updated - action[coordinate]
            if change != 0.0:
                action[coordinate] = updated
                gradient += hessian[:, coordinate] * change
        optimality = _maximum_kkt_violation(
            action,
            gradient,
            costs,
            position_cap=config.solver_position_cap,
        )
        if optimality <= config.gradient_tolerance:
            converged = True
            break
    return OptimizeResult(
        x=action,
        nit=iterations,
        success=converged,
        optimality=optimality,
        message=("converged" if converged else "maximum iterations reached"),
    )


def _soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)


def _maximum_kkt_violation(
    action: FloatArray,
    smooth_gradient: FloatArray,
    costs: FloatArray,
    *,
    position_cap: float,
) -> float:
    derivative = smooth_gradient + costs * np.sign(action)
    zero = action == 0.0
    derivative[zero] = np.maximum(np.abs(smooth_gradient[zero]) - costs[zero], 0.0)
    upper = action >= position_cap
    lower = action <= -position_cap
    derivative[upper] = np.maximum(derivative[upper], 0.0)
    derivative[lower] = np.minimum(derivative[lower], 0.0)
    return float(np.max(np.abs(derivative)))


def _objective_and_gradient(
    parameters: NDArray[np.floating],
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
) -> tuple[float, FloatArray]:
    values = np.asarray(parameters, dtype=np.float64)
    instrument_count = instruments.shape[1]
    action = values[:instrument_count]
    smoothing = config.cost_smoothing
    smooth_absolute = np.sqrt(action**2 + smoothing**2) - smoothing
    cost = float(costs @ smooth_absolute)
    cost_gradient = costs * action / np.sqrt(action**2 + smoothing**2)
    wealth = claim + instruments @ action - cost
    jacobian = instruments - cost_gradient[None, :]
    sample_count = len(wealth)
    risk = config.risk_parameter
    if config.objective is HistoricalSanosObjective.MEAN_VARIANCE:
        centered = wealth - np.mean(wealth)
        loss = -np.mean(wealth) + 0.5 * risk * np.mean(centered**2)
        path_gradient = (-1.0 + risk * centered) / sample_count
    elif config.objective is HistoricalSanosObjective.MEAN_VOLATILITY:
        centered = wealth - np.mean(wealth)
        standard_deviation = math.sqrt(
            float(np.mean(centered**2)) + np.finfo(np.float64).eps
        )
        loss = -np.mean(wealth) + risk * standard_deviation
        path_gradient = (-1.0 + risk * centered / standard_deviation) / sample_count
    elif config.objective is HistoricalSanosObjective.ENTROPY:
        exponent = -risk * wealth
        maximum = float(np.max(exponent))
        weights = np.exp(exponent - maximum)
        weights /= np.sum(weights)
        loss = (maximum + math.log(float(np.mean(np.exp(exponent - maximum))))) / risk
        path_gradient = -weights
    elif config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL:
        if not include_translation or len(values) != instrument_count + 1:
            raise ValueError("power-exponential objective requires cash translation")
        translation = float(values[-1])
        residual = wealth - translation
        utility = np.where(
            residual >= 0.0,
            -np.expm1(-risk * np.maximum(residual, 0.0)) / risk,
            residual - 0.5 * risk * residual**2,
        )
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk * np.maximum(residual, 0.0)),
            1.0 - risk * residual,
        )
        loss = -(translation + float(np.mean(utility)))
        path_gradient = -marginal / sample_count
    else:
        raise ValueError(f"unsupported historical SANOS objective {config.objective!r}")
    gradient = jacobian.T @ path_gradient
    if include_translation:
        translation_gradient = -1.0 - float(np.sum(path_gradient))
        gradient = np.concatenate((gradient, np.asarray([translation_gradient])))
    if not math.isfinite(float(loss)) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("historical SANOS objective is non-finite")
    return float(loss), np.asarray(gradient, dtype=np.float64)


def _validated_inputs(
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    instruments = np.asarray(instrument_returns, dtype=np.float64)
    claim = np.asarray(claim_returns, dtype=np.float64)
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if instruments.ndim != 2 or min(instruments.shape) == 0:
        raise ValueError("instrument_returns must be a non-empty matrix")
    if claim.shape != (len(instruments),) or costs.shape != (instruments.shape[1],):
        raise ValueError("historical SANOS arrays do not align")
    if not np.all(np.isfinite(instruments)) or not np.all(np.isfinite(claim)):
        raise ValueError("historical SANOS returns must be finite")
    if np.any(~np.isfinite(costs) | (costs < 0.0)):
        raise ValueError("historical SANOS costs must be finite and non-negative")
    return instruments, claim, costs
