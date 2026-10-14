"""Empirical four-objective training on historical public SANOS returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, brentq, minimize

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
    return fit_historical_sanos_hedges(
        instrument_returns=instruments,
        claim_returns=claim[:, None],
        linear_transaction_costs=costs,
        config=config,
    )[0]


def fit_historical_sanos_hedges(
    *,
    instrument_returns: NDArray[np.floating],
    claim_returns: NDArray[np.floating],
    linear_transaction_costs: NDArray[np.floating],
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
) -> tuple[HistoricalSanosHedgeFit, ...]:
    """Fit a claim book while reusing its common zero-claim policy."""
    instruments = np.asarray(instrument_returns, dtype=np.float64)
    claims = np.asarray(claim_returns, dtype=np.float64)
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if claims.ndim != 2 or claims.shape[1] == 0:
        raise ValueError("claim_returns must be a non-empty matrix")
    _validated_inputs(instruments, claims[:, 0], costs)
    if claims.shape[0] != len(instruments) or not np.all(np.isfinite(claims)):
        raise ValueError("historical SANOS claim book must align and be finite")
    zero_policy = _fit_policy(
        instruments,
        np.zeros(len(instruments), dtype=np.float64),
        costs,
        config=config,
    )
    return tuple(
        _compose_hedge_fit(
            instruments,
            claims[:, index],
            costs,
            claim_policy=_fit_policy(
                instruments,
                claims[:, index],
                costs,
                config=config,
            ),
            zero_policy=zero_policy,
            config=config,
        )
        for index in range(claims.shape[1])
    )


def _compose_hedge_fit(
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    claim_policy: HistoricalSanosPolicyFit,
    zero_policy: HistoricalSanosPolicyFit,
    config: HistoricalSanosConfig,
) -> HistoricalSanosHedgeFit:
    raw_action = claim_policy.action - zero_policy.action
    action = np.clip(raw_action, -config.position_cap, config.position_cap)
    executable_cap_hits = int(np.count_nonzero(action != raw_action))
    internal_cap_hits = claim_policy.cap_hits + zero_policy.cap_hits
    cap_hits = internal_cap_hits + executable_cap_hits
    include_translation = (
        config.objective is HistoricalSanosObjective.POWER_EXPONENTIAL
    )
    translation = 0.0
    if include_translation:
        wealth = claim + instruments @ action - float(costs @ np.abs(action))
        translation = _optimal_power_cash_translation(
            wealth,
            risk_parameter=config.risk_parameter,
        )
    objective = _objective_and_gradient(
        np.concatenate(
            (
                action,
                np.asarray([translation]) if include_translation else np.empty(0),
            )
        ),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
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
    selection = _minimize_policy(
        initial,
        instruments[:split],
        claim[:split],
        costs,
        config=config,
        include_translation=include_translation,
    )
    selection = _retry_exact_cost_policy(
        selection,
        instruments[:split],
        claim[:split],
        costs,
        config=config,
        include_translation=include_translation,
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
    )
    final = _retry_exact_cost_policy(
        final,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
    )
    final_loss = _objective_and_gradient(
        final.x,
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
    )[0]
    action = np.asarray(final.x[: instruments.shape[1]], dtype=np.float64)
    cap_hits = int(
        np.count_nonzero(
            np.abs(action) >= 0.999 * config.solver_position_cap
        )
    )
    optimality = float(getattr(final, "optimality", math.inf))
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
) -> OptimizeResult:
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
    if config.objective in (
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ):
        return _minimize_exact_cost_policy(
            initial,
            instruments,
            claim,
            costs,
            config=config,
            include_translation=include_translation,
        )
    raise ValueError(f"unsupported historical SANOS objective {config.objective!r}")


def _retry_exact_cost_policy(
    result: OptimizeResult,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
) -> OptimizeResult:
    """Restart a stalled split solve from its canonical signed action."""
    if result.success or config.objective not in (
        HistoricalSanosObjective.ENTROPY,
        HistoricalSanosObjective.POWER_EXPONENTIAL,
    ):
        return result
    retry = _minimize_policy(
        np.asarray(result.x, dtype=np.float64),
        instruments,
        claim,
        costs,
        config=config,
        include_translation=include_translation,
    )
    retry.nit = int(result.nit) + int(retry.nit)
    retry.message = f"canonical restart after {result.message}; {retry.message}"
    return retry


def _minimize_exact_cost_policy(
    initial: FloatArray,
    instruments: FloatArray,
    claim: FloatArray,
    costs: FloatArray,
    *,
    config: HistoricalSanosConfig,
    include_translation: bool,
) -> OptimizeResult:
    """Solve a cash-additive OCE with exact linear spread costs."""
    instrument_count = instruments.shape[1]
    initial_action = np.asarray(initial[:instrument_count], dtype=np.float64)
    cost_free_translation = np.empty(0, dtype=np.float64)
    if include_translation:
        cost_free_translation = np.asarray(
            [float(initial[-1]) + float(costs @ np.abs(initial_action))],
            dtype=np.float64,
        )
    split_initial = np.concatenate(
        (
            np.maximum(initial_action, 0.0),
            np.maximum(-initial_action, 0.0),
            cost_free_translation,
        )
    )
    split_bounds: list[tuple[float | None, float | None]] = [
        (0.0, config.solver_position_cap)
    ] * (2 * instrument_count)
    if include_translation:
        split_bounds.append((None, None))
    zero_costs = np.zeros_like(costs)

    def value_and_gradient(parameters: NDArray[np.floating]) -> tuple[float, FloatArray]:
        values = np.asarray(parameters, dtype=np.float64)
        positive = values[:instrument_count]
        negative = values[instrument_count : 2 * instrument_count]
        action = positive - negative
        risk_parameters = np.concatenate(
            (
                action,
                np.asarray([values[-1]]) if include_translation else np.empty(0),
            )
        )
        risk_value, risk_gradient = _objective_and_gradient(
            risk_parameters,
            instruments,
            claim,
            zero_costs,
            config=config,
            include_translation=include_translation,
        )
        action_gradient = risk_gradient[:instrument_count]
        gradient = np.concatenate(
            (
                action_gradient + costs,
                -action_gradient + costs,
                np.asarray([risk_gradient[-1]])
                if include_translation
                else np.empty(0),
            )
        )
        return (
            float(risk_value + costs @ (positive + negative)),
            np.asarray(gradient, dtype=np.float64),
        )

    result = minimize(
        value_and_gradient,
        split_initial,
        method="L-BFGS-B",
        jac=True,
        bounds=split_bounds,
        options={
            "maxiter": config.maximum_iterations,
            "maxfun": 4 * config.maximum_iterations,
            "gtol": config.gradient_tolerance,
            "ftol": 0.0,
            "maxls": 50,
        },
    )
    split_solution = np.asarray(result.x, dtype=np.float64)
    action = (
        split_solution[:instrument_count]
        - split_solution[instrument_count : 2 * instrument_count]
    )
    internal_parameters = np.concatenate(
        (
            action,
            np.asarray([split_solution[-1]])
            if include_translation
            else np.empty(0),
        )
    )
    risk_value, risk_gradient = _objective_and_gradient(
        internal_parameters,
        instruments,
        claim,
        zero_costs,
        config=config,
        include_translation=include_translation,
    )
    optimality = _maximum_kkt_violation(
        action,
        risk_gradient[:instrument_count],
        costs,
        position_cap=config.solver_position_cap,
    )
    if include_translation:
        optimality = max(optimality, abs(float(risk_gradient[-1])))
    translation = np.empty(0, dtype=np.float64)
    if include_translation:
        translation = np.asarray(
            [split_solution[-1] - float(costs @ np.abs(action))],
            dtype=np.float64,
        )
    result.x = np.concatenate((action, translation))
    result.fun = float(risk_value + costs @ np.abs(action))
    result.optimality = float(optimality)
    optimizer_success = bool(result.success)
    result.success = bool(optimality <= config.gradient_tolerance)
    if not result.success:
        result.message = (
            f"{result.message}; KKT residual {optimality:.6g} exceeds "
            f"{config.gradient_tolerance:.6g}"
        )
    elif not optimizer_success:
        result.message = (
            f"{result.message}; accepted by independent KKT certificate "
            f"{optimality:.6g}"
        )
    return result


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
    cost = float(costs @ np.abs(action))
    cost_gradient = costs * np.sign(action)
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


def _optimal_power_cash_translation(
    wealth: FloatArray,
    *,
    risk_parameter: float,
) -> float:
    width = 50.0 / risk_parameter
    lower = float(np.min(wealth) - width)
    upper = float(np.max(wealth) + width)

    def first_order_condition(translation: float) -> float:
        residual = wealth - translation
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk_parameter * np.maximum(residual, 0.0)),
            1.0 - risk_parameter * residual,
        )
        return 1.0 - float(np.mean(marginal))

    return float(brentq(first_order_condition, lower, upper))


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
