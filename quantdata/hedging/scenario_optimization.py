"""Model-neutral static hedging under weighted one-day scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]


class ScenarioObjective(StrEnum):
    """Supported convex risk criteria for static scenario hedges."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"


@version("1.0")
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

    def __post_init__(self) -> None:
        verify_inp(
            isinstance(self.objective, ScenarioObjective),
            lambda: "objective must be a ScenarioObjective",
        )
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite-positive",
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
    gradient = instruments.T @ path_gradient
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
