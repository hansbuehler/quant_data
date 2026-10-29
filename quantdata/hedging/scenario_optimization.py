"""Model-neutral static hedging under weighted one-day scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import cvxpy as cp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lstsq
from scipy.optimize import brentq, lsq_linear, minimize
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


class ScenarioObjective(StrEnum):
    """Supported convex risk criteria for static scenario hedges."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"
    CVAR_OCE = "cvar-oce"


@version("1.4")
@dataclass(frozen=True, kw_only=True)
class ScenarioOptimizationConfig:
    """Numerical controls for a weighted static scenario hedge."""

    objective: ScenarioObjective = ScenarioObjective.MEAN_VARIANCE
    risk_parameter: float = 10.0
    maximum_iterations: int = 50_000
    gradient_tolerance: float = 1.0e-8
    certificate_tolerance: float = 1.0e-6
    position_cap: float = 1_000_000.0
    scale_floor: float = 1.0e-8
    raw_action_ridge: float = 0.0

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.objective, ScenarioObjective),
            lambda: "objective must be a ScenarioObjective",
        )
        verify_inp(
            not math.isnan(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be positive or infinity",
        )
        verify_inp(
            self.maximum_iterations > 0,
            lambda: "maximum_iterations must be positive",
        )
        verify_inp(
            math.isfinite(self.gradient_tolerance)
            and self.gradient_tolerance > 0.0,
            lambda: "gradient_tolerance must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.certificate_tolerance)
            and self.certificate_tolerance >= self.gradient_tolerance,
            lambda: "certificate_tolerance must be at least gradient_tolerance",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.scale_floor) and self.scale_floor > 0.0,
            lambda: "scale_floor must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.raw_action_ridge)
            and self.raw_action_ridge >= 0.0,
            lambda: "raw_action_ridge must be finite and non-negative",
        )


@dataclass(frozen=True, kw_only=True)
class ScenarioHedgeFit:
    """Static action and an independent first-order certificate."""

    action: FloatArray
    cash_translation: float | None
    objective_value: float
    iterations: int
    maximum_kkt_residual: float
    cap_hits: int
    converged: bool
    solver_message: str


def fit_static_scenario_hedge(
    *,
    instrument_returns: ArrayLike,
    claim_returns: ArrayLike,
    scenario_weights: ArrayLike,
    linear_transaction_costs: ArrayLike,
    config: ScenarioOptimizationConfig = ScenarioOptimizationConfig(),
) -> ScenarioHedgeFit:
    """Minimize weighted risk plus exact linear entry costs.

    Returns are terminal wealth increments. Transaction costs are charged once
    as ``costs @ abs(action)`` and are not smoothed.
    """
    instruments = _finite_matrix(instrument_returns, name="instrument_returns")
    observations, instrument_count = instruments.shape
    claim = np.asarray(claim_returns, dtype=np.float64)
    if claim.shape != (observations,) or not np.all(np.isfinite(claim)):
        raise ValueError("claim_returns must be a finite aligned vector")
    weights = _weights(scenario_weights, observations)
    costs = np.asarray(linear_transaction_costs, dtype=np.float64)
    if costs.shape != (instrument_count,) or np.any(
        ~np.isfinite(costs) | (costs < 0.0)
    ):
        raise ValueError("linear_transaction_costs must be finite and aligned")

    if math.isinf(config.risk_parameter):
        if config.objective is ScenarioObjective.MEAN_VARIANCE:
            return _fit_minimum_variance(
                instruments=instruments,
                claim=claim,
                weights=weights,
                costs=costs,
                config=config,
            )
        return _fit_tail_epigraph(
            instruments=instruments,
            claim=claim,
            weights=weights,
            costs=costs,
            config=config,
        )
    if config.objective is ScenarioObjective.CVAR_OCE:
        return _fit_tail_epigraph(
            instruments=instruments,
            claim=claim,
            weights=weights,
            costs=costs,
            config=config,
        )

    gain_rms = np.sqrt(weights @ np.square(instruments))
    action_scale = np.clip(
        1.0 / np.maximum(gain_rms, config.scale_floor),
        1.0e-6,
        1.0e6,
    )
    include_translation = config.objective is ScenarioObjective.POWER_EXPONENTIAL
    translation = (
        _optimal_power_translation(
            claim,
            weights=weights,
            risk_parameter=config.risk_parameter,
        )
        if include_translation
        else None
    )
    initial = np.zeros(2 * instrument_count + int(include_translation))
    if translation is not None:
        initial[-1] = translation
    scaled_costs = action_scale * costs
    bounds: list[tuple[float | None, float | None]] = [
        *((0.0, config.position_cap / scale) for scale in action_scale),
        *((0.0, config.position_cap / scale) for scale in action_scale),
    ]
    if include_translation:
        bounds.append((None, None))

    def split_value_and_gradient(parameters: FloatArray) -> tuple[float, FloatArray]:
        positive = parameters[:instrument_count]
        negative = parameters[instrument_count : 2 * instrument_count]
        action = action_scale * (positive - negative)
        cash_translation = float(parameters[-1]) if include_translation else None
        value, action_gradient, translation_gradient = _risk_and_gradient(
            action,
            cash_translation,
            instruments=instruments,
            claim=claim,
            weights=weights,
            config=config,
        )
        scaled_gradient = action_scale * action_gradient
        gradient = np.concatenate(
            (
                scaled_gradient + scaled_costs,
                -scaled_gradient + scaled_costs,
                np.asarray([translation_gradient])
                if include_translation
                else np.empty(0),
            )
        )
        return float(value + scaled_costs @ (positive + negative)), gradient

    iterations = 0
    best_result = None
    best_kkt = math.inf
    for attempt in range(3):
        result = minimize(
            split_value_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": config.maximum_iterations,
                "maxfun": 4 * config.maximum_iterations,
                "gtol": config.gradient_tolerance,
                "ftol": 0.0,
                "maxls": 50 + 25 * attempt,
            },
        )
        iterations += int(result.nit)
        parameters = np.asarray(result.x, dtype=np.float64)
        action = action_scale * (
            parameters[:instrument_count]
            - parameters[instrument_count : 2 * instrument_count]
        )
        cash_translation = float(parameters[-1]) if include_translation else None
        _, smooth_gradient, translation_gradient = _risk_and_gradient(
            action,
            cash_translation,
            instruments=instruments,
            claim=claim,
            weights=weights,
            config=config,
        )
        kkt = _maximum_kkt_residual(
            action,
            smooth_gradient,
            costs,
            position_cap=config.position_cap,
            coordinate_scale=action_scale,
        )
        if include_translation:
            kkt = max(kkt, abs(translation_gradient))
        if kkt < best_kkt:
            best_result = result
            best_kkt = kkt
        if kkt <= config.certificate_tolerance:
            break
        if best_result is None:
            raise RuntimeError("scenario optimizer did not produce a candidate")
        parameters = np.asarray(best_result.x, dtype=np.float64)
        action = action_scale * (
            parameters[:instrument_count]
            - parameters[instrument_count : 2 * instrument_count]
        )
        canonical = action / action_scale
        initial = np.concatenate(
            (
                np.maximum(canonical, 0.0),
                np.maximum(-canonical, 0.0),
                parameters[-1:] if include_translation else np.empty(0),
            )
        )
    if best_result is None:
        raise RuntimeError("scenario optimizer did not produce a candidate")
    result = best_result
    parameters = np.asarray(result.x, dtype=np.float64)
    action = action_scale * (
        parameters[:instrument_count]
        - parameters[instrument_count : 2 * instrument_count]
    )
    cash_translation = float(parameters[-1]) if include_translation else None
    risk_value, smooth_gradient, translation_gradient = _risk_and_gradient(
        action,
        cash_translation,
        instruments=instruments,
        claim=claim,
        weights=weights,
        config=config,
    )
    kkt = _maximum_kkt_residual(
        action,
        smooth_gradient,
        costs,
        position_cap=config.position_cap,
        coordinate_scale=action_scale,
    )
    if include_translation:
        kkt = max(kkt, abs(translation_gradient))
    cap_hits = int(np.count_nonzero(np.abs(action) >= 0.999 * config.position_cap))
    converged = bool(kkt <= config.certificate_tolerance and cap_hits == 0)
    message = str(result.message)
    if converged and not result.success:
        message += f"; accepted by independent KKT certificate {kkt:.6g}"
    elif not converged:
        message += f"; KKT residual {kkt:.6g}"
    return ScenarioHedgeFit(
        action=np.asarray(action, dtype=np.float64),
        cash_translation=cash_translation,
        objective_value=float(risk_value + costs @ np.abs(action)),
        iterations=iterations,
        maximum_kkt_residual=float(kkt),
        cap_hits=cap_hits,
        converged=converged,
        solver_message=message,
    )


def _fit_minimum_variance(
    *,
    instruments: FloatArray,
    claim: FloatArray,
    weights: FloatArray,
    costs: FloatArray,
    config: ScenarioOptimizationConfig,
) -> ScenarioHedgeFit:
    """Solve pure minimum variance, using costs only to break exact ties."""
    _, instrument_count = instruments.shape
    centered_instruments = instruments - weights @ instruments
    centered_claim = claim - weights @ claim
    weighted_instruments = np.sqrt(weights)[:, None] * centered_instruments
    weighted_claim = np.sqrt(weights) * centered_claim
    action, _, rank, _ = lstsq(
        weighted_instruments,
        -weighted_claim,
        cond=1.0e-12,
        lapack_driver="gelsd",
    )
    solver_message = "minimum-norm least squares"
    if np.max(np.abs(action)) > config.position_cap:
        bounded = lsq_linear(
            weighted_instruments,
            -weighted_claim,
            bounds=(-config.position_cap, config.position_cap),
            tol=config.gradient_tolerance,
            max_iter=config.maximum_iterations,
        )
        if not bounded.success:
            raise RuntimeError(f"minimum-variance solve failed: {bounded.message}")
        action = bounded.x
        solver_message = str(bounded.message)
    elif rank < instrument_count and (
        np.any(costs > 0.0) or config.raw_action_ridge > 0.0
    ):
        _, _, right_vectors = np.linalg.svd(
            weighted_instruments,
            full_matrices=False,
        )
        candidate = cp.Variable(instrument_count)
        objective = costs @ cp.abs(candidate)
        if config.raw_action_ridge > 0.0:
            objective += 0.5 * config.raw_action_ridge * cp.sum_squares(candidate)
        tie_break = cp.Problem(
            cp.Minimize(objective),
            [
                right_vectors[:rank] @ candidate
                == right_vectors[:rank] @ action,
                candidate <= config.position_cap,
                candidate >= -config.position_cap,
            ],
        )
        tie_break.solve(solver=cp.CLARABEL)
        if tie_break.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"minimum-variance tie-break failed: {tie_break.status}")
        action = np.asarray(candidate.value, dtype=np.float64)
        solver_message += "; cost/ridge tie-break"
    residual = centered_claim + centered_instruments @ action
    variance = float(weights @ residual**2)
    gradient = 2.0 * centered_instruments.T @ (weights * residual)
    kkt = float(np.max(np.abs(gradient)))
    cap_hits = int(np.count_nonzero(np.abs(action) >= 0.999 * config.position_cap))
    return ScenarioHedgeFit(
        action=np.asarray(action, dtype=np.float64),
        cash_translation=None,
        objective_value=variance,
        iterations=1,
        maximum_kkt_residual=kkt,
        cap_hits=cap_hits,
        converged=bool(
            np.all(np.isfinite(action))
            and kkt <= config.certificate_tolerance
            and cap_hits == 0
        ),
        solver_message=solver_message,
    )


def _weighted_lower_tail_mean(
    wealth: FloatArray,
    weights: FloatArray,
    *,
    alpha: float,
) -> float:
    """Return the exact empirical lower-tail mean, including fractional mass."""
    order = np.argsort(wealth)
    ordered_wealth = wealth[order]
    ordered_weights = weights[order]
    remaining = alpha
    total = 0.0
    for value, weight in zip(ordered_wealth, ordered_weights, strict=True):
        selected = min(remaining, float(weight))
        total += selected * float(value)
        remaining -= selected
        if remaining <= 1.0e-14:
            break
    if remaining > 1.0e-12:
        raise RuntimeError("scenario weights do not cover the requested CVaR tail")
    return total / alpha


def _fit_tail_epigraph(
    *,
    instruments: FloatArray,
    claim: FloatArray,
    weights: FloatArray,
    costs: FloatArray,
    config: ScenarioOptimizationConfig,
) -> ScenarioHedgeFit:
    """Certify a tail hedge, retrying at higher conic precision if needed."""
    initial_error: cp.SolverError | RuntimeError | None = None
    try:
        initial = _solve_tail_epigraph(
            instruments=instruments,
            claim=claim,
            weights=weights,
            costs=costs,
            config=config,
            solver_tolerance=1.0e-7,
        )
    except (cp.SolverError, RuntimeError) as error:
        initial_error = error
    else:
        if initial.converged or initial.cap_hits:
            return initial
    try:
        strict = _solve_tail_epigraph(
            instruments=instruments,
            claim=claim,
            weights=weights,
            costs=costs,
            config=config,
            solver_tolerance=1.0e-9,
        )
    except (cp.SolverError, RuntimeError) as error:
        if initial_error is not None:
            raise RuntimeError(
                "tail hedge failed at both solver tolerances: "
                f"initial={type(initial_error).__name__}: {initial_error}; "
                f"strict={type(error).__name__}: {error}"
            ) from error
        raise
    if initial_error is not None:
        return replace(
            strict,
            solver_message=(
                f"{strict.solver_message}; initial_solve_error="
                f"{type(initial_error).__name__}: {initial_error}"
            ),
        )
    return strict


def _polish_tail_zero_actions(
    action: FloatArray,
    *,
    instruments: FloatArray,
    costs: FloatArray,
    dual_linear_gradient: FloatArray,
    raw_action_ridge: float,
    certificate_tolerance: float,
) -> tuple[FloatArray, int, float]:
    polished = np.asarray(action, dtype=np.float64).copy()
    scenario_impact = np.max(np.abs(instruments), axis=0) * np.abs(polished)
    objective_impact = (
        costs * np.abs(polished)
        + 0.5 * raw_action_ridge * polished**2
    )
    rounding_bound = scenario_impact + objective_impact
    rounding_budget = 0.25 * certificate_tolerance
    rounded_coordinates = 0
    consumed_budget = 0.0
    for coordinate in np.argsort(rounding_bound):
        if (
            polished[coordinate] == 0.0
            or abs(dual_linear_gradient[coordinate]) > costs[coordinate]
            or consumed_budget + rounding_bound[coordinate] > rounding_budget
        ):
            continue
        consumed_budget += rounding_bound[coordinate]
        polished[coordinate] = 0.0
        rounded_coordinates += 1
    return polished, rounded_coordinates, consumed_budget


def _solve_tail_epigraph(
    *,
    instruments: FloatArray,
    claim: FloatArray,
    weights: FloatArray,
    costs: FloatArray,
    config: ScenarioOptimizationConfig,
    solver_tolerance: float,
) -> ScenarioHedgeFit:
    """Solve finite CVaR or the finite-scenario worst-case limit exactly."""
    observations, instrument_count = instruments.shape
    if math.isinf(config.risk_parameter):
        initial_count = max(256, 2 * instrument_count)
    elif config.risk_parameter <= 1.0:
        initial_count = observations
    else:
        tail_count = int(math.ceil(observations / (1.0 + config.risk_parameter)))
        initial_count = max(256, 2 * instrument_count, int(math.ceil(1.5 * tail_count)))
    selected = np.sort(np.argsort(claim)[: min(observations, initial_count)])
    total_iterations = 0
    solver_retries: list[str] = []
    for exchange in range(21):
        action = cp.Variable(instrument_count)
        threshold = cp.Variable()
        active_wealth = claim[selected] + instruments[selected] @ action
        objective = -threshold + costs @ cp.abs(action)
        if config.raw_action_ridge > 0.0:
            objective += 0.5 * config.raw_action_ridge * cp.sum_squares(action)
        if math.isinf(config.risk_parameter):
            tail_constraint = threshold <= active_wealth
            constraints = [tail_constraint]
            shortfall = None
        else:
            shortfall = cp.Variable(len(selected), nonneg=True)
            tail_constraint = shortfall >= threshold - active_wealth
            constraints = [tail_constraint]
            objective += (1.0 + config.risk_parameter) * (
                weights[selected] @ shortfall
            )
        constraints.extend((action <= config.position_cap, action >= -config.position_cap))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solver_name = "CLARABEL"
        active_tolerance = solver_tolerance
        reported_tolerance: float | None = active_tolerance
        try:
            problem.solve(
                solver=cp.CLARABEL,
                max_iter=config.maximum_iterations,
                tol_gap_abs=active_tolerance,
                tol_gap_rel=active_tolerance,
                tol_feas=active_tolerance,
            )
        except cp.SolverError as error:
            solver_retries.append(f"CLARABEL at {active_tolerance:.0e}: {error}")
            active_tolerance = max(solver_tolerance, 1.0e-6)
            reported_tolerance = active_tolerance
            try:
                problem.solve(
                    solver=cp.CLARABEL,
                    max_iter=config.maximum_iterations,
                    tol_gap_abs=active_tolerance,
                    tol_gap_rel=active_tolerance,
                    tol_feas=active_tolerance,
                )
            except cp.SolverError as error:
                solver_retries.append(f"CLARABEL at {active_tolerance:.0e}: {error}")
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            solver_retries.append(f"CLARABEL at {active_tolerance:.0e}: {problem.status}")
            if config.raw_action_ridge > 0.0:
                solver_name = "OSQP"
                osqp_tolerance = min(active_tolerance, 1.0e-8)
                reported_tolerance = osqp_tolerance
                try:
                    problem.solve(
                        solver=cp.OSQP,
                        eps_abs=osqp_tolerance,
                        eps_rel=osqp_tolerance,
                        max_iter=config.maximum_iterations,
                    )
                except cp.SolverError as error:
                    solver_retries.append(f"OSQP at {osqp_tolerance:.0e}: {error}")
                if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    solver_retries.append(f"OSQP at {osqp_tolerance:.0e}: {problem.status}")
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solver_name = "HIGHS"
                reported_tolerance = None
                try:
                    problem.solve(solver=cp.HIGHS, time_limit=600.0)
                except cp.SolverError as error:
                    solver_retries.append(f"HIGHS: {error}")
                    raise cp.SolverError(
                        f"tail solvers failed: {solver_retries!r}"
                    ) from error
        total_iterations += int(problem.solver_stats.num_iters or 0)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"tail-risk solve failed: {problem.status}")
        fitted_action = np.asarray(action.value, dtype=np.float64)
        fitted_wealth = claim + instruments @ fitted_action
        omitted = np.ones(observations, dtype=bool)
        omitted[selected] = False
        violations = np.flatnonzero(
            omitted & (float(threshold.value) - fitted_wealth > 1.0e-8)
        )
        if len(violations) == 0:
            break
        if exchange == 20 or len(selected) == observations:
            raise RuntimeError("tail-risk active-set exchange did not converge")
        selected = (
            np.arange(observations)
            if exchange == 19 or len(selected) + len(violations) > 0.9 * observations
            else np.union1d(selected, violations)
        )
    else:
        raise RuntimeError("tail-risk active-set exchange did not terminate")
    dual_weights = np.zeros(observations, dtype=np.float64)
    dual_weights[selected] = np.asarray(tail_constraint.dual_value, dtype=np.float64)
    dual_linear_gradient = -instruments.T @ dual_weights
    fitted_action, rounded_coordinates, consumed_budget = _polish_tail_zero_actions(
        fitted_action,
        instruments=instruments,
        costs=costs,
        dual_linear_gradient=dual_linear_gradient,
        raw_action_ridge=config.raw_action_ridge,
        certificate_tolerance=config.certificate_tolerance,
    )
    if rounded_coordinates:
        fitted_wealth = claim + instruments @ fitted_action
    if math.isinf(config.risk_parameter):
        certainty_equivalent = float(np.min(fitted_wealth))
    else:
        certainty_equivalent = _weighted_lower_tail_mean(
            fitted_wealth,
            weights,
            alpha=1.0 / (1.0 + config.risk_parameter),
        )
    exact_objective = (
        -certainty_equivalent
        + float(costs @ np.abs(fitted_action))
        + 0.5 * config.raw_action_ridge * float(fitted_action @ fitted_action)
    )
    smooth_gradient = (
        dual_linear_gradient
        + config.raw_action_ridge * fitted_action
    )
    signed_gradient = smooth_gradient + costs * np.sign(fitted_action)
    at_zero = np.abs(fitted_action) <= 5.0 * config.certificate_tolerance
    signed_gradient[at_zero] = np.sign(smooth_gradient[at_zero]) * np.maximum(
        np.abs(smooth_gradient[at_zero]) - costs[at_zero],
        0.0,
    )
    at_upper = fitted_action >= config.position_cap - 1.0e-8
    at_lower = fitted_action <= -config.position_cap + 1.0e-8
    signed_gradient[at_upper] = np.maximum(signed_gradient[at_upper], 0.0)
    signed_gradient[at_lower] = np.minimum(signed_gradient[at_lower], 0.0)
    tail_dual_error = 0.0
    if not math.isinf(config.risk_parameter):
        tail_dual_error = float(
            np.max(np.maximum(dual_weights - (1.0 + config.risk_parameter) * weights, 0.0))
        )
        primal_violation = float(
            np.max(
                np.maximum(
                    float(threshold.value)
                    - fitted_wealth[selected]
                    - np.asarray(shortfall.value),
                    0.0,
                )
            )
        )
    else:
        primal_violation = float(
            np.max(np.maximum(float(threshold.value) - fitted_wealth[selected], 0.0))
        )
    if np.any(omitted):
        primal_violation = max(
            primal_violation,
            float(np.max(np.maximum(float(threshold.value) - fitted_wealth[omitted], 0.0))),
        )
    certificate_components = {
        "stationarity": float(np.max(np.abs(signed_gradient))),
        "dual_mass": abs(float(np.sum(dual_weights)) - 1.0),
        "dual_tail_bound": tail_dual_error,
        "primal_feasibility": primal_violation,
        "objective_gap": abs(exact_objective - float(problem.value)),
        "position_cap": float(
            np.max(np.maximum(np.abs(fitted_action) - config.position_cap, 0.0))
        ),
    }
    certificate = max(certificate_components.values())
    stationarity_coordinate = int(np.argmax(np.abs(signed_gradient)))
    cap_hits = int(np.count_nonzero(np.abs(fitted_action) >= 0.999 * config.position_cap))
    return ScenarioHedgeFit(
        action=fitted_action,
        cash_translation=float(threshold.value),
        objective_value=exact_objective,
        iterations=total_iterations,
        maximum_kkt_residual=certificate,
        cap_hits=cap_hits,
        converged=bool(certificate <= config.certificate_tolerance and cap_hits == 0),
        solver_message=(
            f"{solver_name} {problem.status}; certificate "
            + ", ".join(
                f"{name}={value:.6g}"
                for name, value in certificate_components.items()
            )
            + "; worst_coordinate="
            + f"{stationarity_coordinate}"
            + f" action={fitted_action[stationarity_coordinate]:.6g}"
            + f" cost={costs[stationarity_coordinate]:.6g}"
            + f" smooth_gradient={smooth_gradient[stationarity_coordinate]:.6g}"
            + f" active_scenarios={len(selected)}/{observations}"
            + f" exchanges={exchange}"
            + f" rounded_coordinates={rounded_coordinates}"
            + f" rounding_bound={consumed_budget:.6g}"
            + (
                f" solver_tolerance={reported_tolerance:.0e}"
                if reported_tolerance is not None
                else " solver_tolerance=HIGHS default"
            )
            + (f" retries={solver_retries!r}" if solver_retries else "")
        ),
    )


def _risk_and_gradient(
    action: FloatArray,
    cash_translation: float | None,
    *,
    instruments: FloatArray,
    claim: FloatArray,
    weights: FloatArray,
    config: ScenarioOptimizationConfig,
) -> tuple[float, FloatArray, float]:
    wealth = claim + instruments @ action
    risk = config.risk_parameter
    if config.objective is ScenarioObjective.MEAN_VARIANCE:
        mean = float(weights @ wealth)
        centered = wealth - mean
        value = -mean + 0.5 * risk * float(weights @ centered**2)
        path_gradient = weights * (-1.0 + risk * centered)
        translation_gradient = 0.0
    elif config.objective is ScenarioObjective.MEAN_VOLATILITY:
        mean = float(weights @ wealth)
        centered = wealth - mean
        volatility = math.sqrt(float(weights @ centered**2))
        value = -mean + risk * volatility
        zero_tolerance = max(
            config.gradient_tolerance,
            np.finfo(np.float64).eps,
        ) * max(1.0, math.sqrt(float(weights @ np.square(claim))))
        path_gradient = (
            weights * (-1.0 + risk * centered / volatility)
            if volatility > zero_tolerance
            else -weights
        )
        translation_gradient = 0.0
    elif config.objective is ScenarioObjective.ENTROPY:
        exponent = -risk * wealth
        log_weights = np.log(weights, where=weights > 0.0, out=np.full_like(weights, -np.inf))
        normalizer = float(logsumexp(log_weights + exponent))
        tilted_weights = np.exp(log_weights + exponent - normalizer)
        value = normalizer / risk
        path_gradient = -tilted_weights
        translation_gradient = 0.0
    elif config.objective is ScenarioObjective.POWER_EXPONENTIAL:
        if cash_translation is None:
            raise ValueError("power-exponential objective requires cash translation")
        residual = wealth - cash_translation
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
        value = -(cash_translation + float(weights @ utility))
        path_gradient = -weights * marginal
        translation_gradient = -1.0 - float(np.sum(path_gradient))
    else:  # pragma: no cover - enum validation makes this unreachable.
        raise ValueError(f"unsupported scenario objective {config.objective!r}")
    value += 0.5 * config.raw_action_ridge * float(action @ action)
    gradient = instruments.T @ path_gradient
    gradient += config.raw_action_ridge * action
    if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("scenario objective is non-finite")
    return float(value), np.asarray(gradient, dtype=np.float64), translation_gradient


def _optimal_power_translation(
    wealth: FloatArray,
    *,
    weights: FloatArray,
    risk_parameter: float,
) -> float:
    width = 50.0 / risk_parameter

    def first_order_condition(value: float) -> float:
        residual = wealth - value
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk_parameter * np.maximum(residual, 0.0)),
            1.0 - risk_parameter * residual,
        )
        return 1.0 - float(weights @ marginal)

    return float(
        brentq(
            first_order_condition,
            float(np.min(wealth) - width),
            float(np.max(wealth) + width),
        )
    )


def _maximum_kkt_residual(
    action: FloatArray,
    smooth_gradient: FloatArray,
    costs: FloatArray,
    *,
    position_cap: float,
    coordinate_scale: FloatArray,
) -> float:
    residual = np.empty_like(action)
    positive = action > 0.0
    negative = action < 0.0
    zero = ~(positive | negative)
    residual[positive] = smooth_gradient[positive] + costs[positive]
    residual[negative] = smooth_gradient[negative] - costs[negative]
    residual[zero] = np.sign(smooth_gradient[zero]) * np.maximum(
        np.abs(smooth_gradient[zero]) - costs[zero],
        0.0,
    )
    upper = action >= position_cap
    lower = action <= -position_cap
    residual[upper] = np.minimum(residual[upper], 0.0)
    residual[lower] = np.maximum(residual[lower], 0.0)
    return float(np.max(np.abs(residual * coordinate_scale)))


def _finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return matrix


def _weights(values: ArrayLike, observations: int) -> FloatArray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (observations,) or np.any(~np.isfinite(weights) | (weights < 0.0)):
        raise ValueError("scenario_weights must be finite, non-negative, and aligned")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("scenario_weights must have finite-positive mass")
    return np.asarray(weights / total, dtype=np.float64)
