"""Causal Q-volatility and attainable P-drift models for GROMM2 factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from cdxcore.err import verify_inp
from cdxcore.version import version
from numpy.typing import NDArray

from quantdata.gromm2.estimation import load_estimation_panel

FloatArray = NDArray[np.float64]
Architecture = Literal[
    "fixed",
    "ar1",
    "lstm",
    "mamba",
    "gru",
    "lstm_corr",
    "mamba_corr",
    "gru_corr",
]


@version("0.3")
@dataclass(frozen=True, kw_only=True)
class FactorLawConfig:
    """Configuration shared by every conditional factor-law estimate."""

    validation_fraction: float = 0.20
    hidden_size: int = 32
    max_epochs: int = 2_000
    patience: int = 200
    learning_rate: float = 2.0e-3
    log_scale_bound: float = math.log(3.0)
    spot_correlation_bound: float = 0.95
    spot_correlation_ridge: float = 1.0e-2
    drift_loss_weight: float = 1.0
    ridge: float = 1.0e-3
    squared_score_floor: float = 0.05**2
    recurrent_step_year_fraction: float = 1.0 / 252.0
    gru_initial_annual_decay: float = 25.0
    seed: int = 20261013
    progress_every: int = 100

    def __post_init__(self) -> None:
        verify_inp(
            0.05 <= self.validation_fraction <= 0.40,
            lambda: "validation_fraction must lie in [0.05, 0.40]",
        )
        verify_inp(
            self.hidden_size > 0,
            lambda: "hidden_size must be positive",
        )
        verify_inp(
            self.max_epochs > 0 and self.patience > 0,
            lambda: "max_epochs and patience must be positive",
        )
        verify_inp(
            self.learning_rate > 0.0,
            lambda: "learning_rate must be positive",
        )
        verify_inp(
            self.log_scale_bound > 0.0,
            lambda: "log_scale_bound must be positive",
        )
        verify_inp(
            0.0 < self.spot_correlation_bound < 1.0,
            lambda: "spot_correlation_bound must lie in (0, 1)",
        )
        verify_inp(
            self.spot_correlation_ridge >= 0.0,
            lambda: "spot_correlation_ridge must be non-negative",
        )
        verify_inp(
            self.drift_loss_weight >= 0.0 and self.ridge >= 0.0,
            lambda: "loss weights must be non-negative",
        )
        verify_inp(
            self.squared_score_floor > 0.0,
            lambda: "squared_score_floor must be positive",
        )
        verify_inp(
            self.recurrent_step_year_fraction > 0.0,
            lambda: "recurrent_step_year_fraction must be positive",
        )
        verify_inp(
            self.gru_initial_annual_decay > 0.0,
            lambda: "gru_initial_annual_decay must be positive",
        )
        verify_inp(
            self.progress_every > 0,
            lambda: "progress_every must be positive",
        )


@dataclass(frozen=True, kw_only=True)
class FactorLawDataset:
    """One chronological factor panel with training-only normalization."""

    dates: NDArray[np.str_]
    origin_indices: NDArray[np.int64]
    inputs: FloatArray
    reset: NDArray[np.bool_]
    factor_scores: FloatArray
    physical_risk_prices: FloatArray
    fixed_factor_scales: FloatArray
    risk_price_mean: FloatArray
    risk_price_scale: FloatArray
    input_mean: FloatArray
    input_scale: FloatArray
    train_count: int

    @property
    def factor_count(self) -> int:
        return self.factor_scores.shape[1]


@dataclass(frozen=True, kw_only=True)
class FactorLawFit:
    """One fitted factor law and its chronological holdout diagnostics."""

    architecture: Architecture
    parameters: dict[str, FloatArray]
    terminal_state: tuple[FloatArray, ...]
    epochs: int
    train_q_negative_log_likelihood: float
    validation_q_negative_log_likelihood: float
    train_p_drift_standardized_rmse: float
    validation_p_drift_standardized_rmse: float
    validation_p_drift_r2: float
    accepted_q: bool
    accepted_p: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_factor_law_dataset(
    *,
    base_estimate: Path,
    history_path: Path,
    dataset: Path,
    config: FactorLawConfig,
    backbone_annual_volatility: float = 0.05,
) -> FactorLawDataset:
    """Build causal inputs from a fitted diffusion and its historical states."""
    with np.load(base_estimate, allow_pickle=False) as values:
        factor_scores = np.asarray(values["factor_scores"], dtype=np.float64)
        physical_risk_prices = np.asarray(
            values["physical_market_prices_of_risk"], dtype=np.float64
        )
        level_mean = np.asarray(values["level_mean"], dtype=np.float64).reshape(-1)
        level_basis = np.asarray(values["level_basis"], dtype=np.float64)
        level_scale = np.asarray(values["level_scale"], dtype=np.float64)
        origin_indices = np.asarray(values["pair_origin_indices"], dtype=np.int64)
        direct_level_scores = (
            np.asarray(values["level_scores"], dtype=np.float64)
            if "level_scores" in values.files
            else None
        )
        direct_dates = (
            np.asarray(values["dates"]).astype("U10")
            if "dates" in values.files
            else None
        )
    if (
        factor_scores.ndim != 2
        or physical_risk_prices.shape != factor_scores.shape
        or len(factor_scores) != len(origin_indices)
    ):
        raise ValueError("factor scores and physical drift coefficients are misaligned")
    observations, factor_count = factor_scores.shape
    train_count = int(math.floor((1.0 - config.validation_fraction) * observations))
    if train_count <= max(32, factor_count + 2) or observations - train_count < 16:
        raise ValueError("factor panel is too short for the requested split")

    if direct_level_scores is None:
        panel = load_estimation_panel(
            history_path=history_path,
            dataset=dataset,
            backbone_annual_volatility=backbone_annual_volatility,
        )
        panel_indices = np.asarray(panel["origin_indices"], dtype=np.int64)
        if not np.array_equal(origin_indices, panel_indices):
            raise ValueError("base estimate and reconstructed panel origins differ")
        levels = np.asarray(panel["log_dlvsq_origins"], dtype=np.float64)
        flat_levels = levels.reshape(observations, -1)
        if flat_levels.shape[1] != len(level_mean):
            raise ValueError("base level representation does not match the panel")
        level_scores = (flat_levels - level_mean) @ level_basis
        level_scores /= np.maximum(level_scale, 1.0e-8)
        dates = np.asarray(panel["dates"])[origin_indices].astype("U10")
    else:
        level_scores = direct_level_scores
        if level_scores.shape[0] != observations:
            raise ValueError("direct level scores do not align with factor scores")
        if direct_dates is None or direct_dates.shape != (observations,):
            raise ValueError("direct factor-law dates are missing or misaligned")
        dates = direct_dates
    contiguous = np.concatenate(([False], np.diff(origin_indices) == 1))
    lagged_scores = np.vstack((np.zeros((1, factor_count)), factor_scores[:-1]))
    lagged_scores[~contiguous] = 0.0
    raw_inputs = np.concatenate((level_scores, lagged_scores), axis=1)
    input_mean = np.mean(raw_inputs[:train_count], axis=0)
    input_scale = np.std(raw_inputs[:train_count], axis=0, ddof=1)
    input_scale = np.maximum(input_scale, 1.0e-6)
    inputs = (raw_inputs - input_mean) / input_scale

    fixed_factor_scales = np.sqrt(
        np.mean(factor_scores[:train_count] ** 2, axis=0)
    )
    fixed_factor_scales = np.maximum(fixed_factor_scales, 1.0e-6)
    risk_price_mean = np.mean(physical_risk_prices[:train_count], axis=0)
    risk_price_scale = np.std(physical_risk_prices[:train_count], axis=0, ddof=1)
    risk_price_scale = np.maximum(risk_price_scale, 1.0e-6)
    return FactorLawDataset(
        dates=dates,
        origin_indices=origin_indices,
        inputs=inputs,
        reset=~contiguous,
        factor_scores=factor_scores,
        physical_risk_prices=physical_risk_prices,
        fixed_factor_scales=fixed_factor_scales,
        risk_price_mean=risk_price_mean,
        risk_price_scale=risk_price_scale,
        input_mean=input_mean,
        input_scale=input_scale,
        train_count=train_count,
    )


def _bounded_spot_correlations(
    raw_correlations: FloatArray,
    *,
    bound: float,
) -> FloatArray:
    """Map unconstrained outputs inside the positive-definite correlation ball."""
    squared_norm = np.sum(raw_correlations**2, axis=-1, keepdims=True)
    return bound * raw_correlations / np.sqrt(1.0 + squared_norm)


def _q_point_negative_log_likelihood(
    factor_scores: FloatArray,
    factor_scales: FloatArray,
) -> FloatArray:
    correlations = np.zeros(
        factor_scores.shape[:-1] + (factor_scores.shape[-1] - 1,),
        dtype=factor_scores.dtype,
    )
    return _q_point_spot_correlated_negative_log_likelihood(
        factor_scores,
        factor_scales,
        correlations,
    )


def _q_point_spot_correlated_negative_log_likelihood(
    factor_scores: FloatArray,
    factor_scales: FloatArray,
    spot_correlations: FloatArray,
) -> FloatArray:
    """Gaussian score with correlations from spot to orthogonal residual factors."""
    standardized = factor_scores / factor_scales
    residual = standardized[..., 1:]
    squared_norm = np.sum(spot_correlations**2, axis=-1)
    schur = np.maximum(1.0 - squared_norm, np.finfo(np.float64).eps)
    conditional_spot = standardized[..., 0] - np.sum(
        spot_correlations * residual,
        axis=-1,
    )
    quadratic = np.sum(residual**2, axis=-1) + conditional_spot**2 / schur
    log_determinant = np.log(schur) + 2.0 * np.sum(
        np.log(factor_scales),
        axis=-1,
    )
    return 0.5 * (quadratic + log_determinant) / factor_scores.shape[-1]


def _q_negative_log_likelihood(
    factor_scores: FloatArray,
    factor_scales: FloatArray,
    spot_correlations: FloatArray | None = None,
) -> float:
    point = (
        _q_point_negative_log_likelihood(factor_scores, factor_scales)
        if spot_correlations is None
        else _q_point_spot_correlated_negative_log_likelihood(
            factor_scores,
            factor_scales,
            spot_correlations,
        )
    )
    return float(np.mean(point))


def _drift_metrics(
    targets: FloatArray,
    predictions: FloatArray,
    *,
    mean: FloatArray,
    scale: FloatArray,
) -> tuple[float, float]:
    standardized_error = (predictions - targets) / scale
    rmse = float(np.sqrt(np.mean(standardized_error**2)))
    centered = targets - mean
    total = float(np.sum(centered**2))
    residual = float(np.sum((targets - predictions) ** 2))
    r2 = 1.0 - residual / total if total > np.finfo(np.float64).tiny else 1.0
    return rmse, r2


def fit_fixed_factor_law(data: FactorLawDataset) -> FactorLawFit:
    """Fit constant Q factor scales and constant attainable P drift coefficients."""
    split = data.train_count
    scales = np.broadcast_to(data.fixed_factor_scales, data.factor_scores.shape)
    drift = np.broadcast_to(data.risk_price_mean, data.physical_risk_prices.shape)
    train_rmse, _ = _drift_metrics(
        data.physical_risk_prices[:split],
        drift[:split],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    validation_rmse, validation_r2 = _drift_metrics(
        data.physical_risk_prices[split:],
        drift[split:],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    return FactorLawFit(
        architecture="fixed",
        parameters={
            "factor_scale": np.sqrt(np.mean(data.factor_scores**2, axis=0)),
            "risk_price": np.mean(data.physical_risk_prices, axis=0),
        },
        terminal_state=(),
        epochs=0,
        train_q_negative_log_likelihood=_q_negative_log_likelihood(
            data.factor_scores[:split], scales[:split]
        ),
        validation_q_negative_log_likelihood=_q_negative_log_likelihood(
            data.factor_scores[split:], scales[split:]
        ),
        train_p_drift_standardized_rmse=train_rmse,
        validation_p_drift_standardized_rmse=validation_rmse,
        validation_p_drift_r2=validation_r2,
        accepted_q=True,
        accepted_p=True,
    )


def _diagonal_ar1(
    values: FloatArray,
    *,
    reset: NDArray[np.bool_],
    train_count: int,
    ridge: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    previous = np.vstack((np.zeros((1, values.shape[1])), values[:-1]))
    previous[reset] = 0.0
    train = np.arange(len(values)) < train_count
    train &= ~reset
    x = previous[train]
    y = values[train]
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = np.sum(centered_x**2, axis=0) + ridge
    slope = np.sum(centered_x * centered_y, axis=0) / denominator
    slope = np.clip(slope, -0.995, 0.995)
    intercept = y_mean - slope * x_mean
    prediction = intercept + slope * previous
    prediction[reset] = intercept
    return intercept, slope, prediction


def fit_ar1_factor_law(
    data: FactorLawDataset,
    *,
    config: FactorLawConfig,
) -> FactorLawFit:
    """Fit diagonal AR(1) factor variance and attainable P-drift laws."""
    standardized_risk_prices = (
        data.physical_risk_prices - data.risk_price_mean
    ) / data.risk_price_scale
    drift_intercept, drift_slope, standardized_drift = _diagonal_ar1(
        standardized_risk_prices,
        reset=data.reset,
        train_count=data.train_count,
        ridge=config.ridge,
    )
    drift = data.risk_price_mean + data.risk_price_scale * standardized_drift
    log_squared_scores = np.log(
        (data.factor_scores / data.fixed_factor_scales) ** 2
        + config.squared_score_floor
    )
    variance_intercept, variance_slope, predicted_log_squared = _diagonal_ar1(
        log_squared_scores,
        reset=data.reset,
        train_count=data.train_count,
        ridge=config.ridge,
    )
    factor_scales = data.fixed_factor_scales * np.exp(
        np.clip(
            0.5 * predicted_log_squared,
            -config.log_scale_bound,
            config.log_scale_bound,
        )
    )
    calibration = np.sqrt(
        np.mean(
            (data.factor_scores[: data.train_count] / factor_scales[: data.train_count])
            ** 2,
            axis=0,
        )
    )
    factor_scales *= np.maximum(calibration, 1.0e-6)
    split = data.train_count
    train_rmse, _ = _drift_metrics(
        data.physical_risk_prices[:split],
        drift[:split],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    validation_rmse, validation_r2 = _drift_metrics(
        data.physical_risk_prices[split:],
        drift[split:],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    fixed = fit_fixed_factor_law(data)
    validation_q_nll = _q_negative_log_likelihood(
        data.factor_scores[split:], factor_scales[split:]
    )
    production_drift_intercept, production_drift_slope, _ = _diagonal_ar1(
        standardized_risk_prices,
        reset=data.reset,
        train_count=len(standardized_risk_prices),
        ridge=config.ridge,
    )
    (
        production_variance_intercept,
        production_variance_slope,
        production_log_squared,
    ) = _diagonal_ar1(
        log_squared_scores,
        reset=data.reset,
        train_count=len(log_squared_scores),
        ridge=config.ridge,
    )
    production_scales = data.fixed_factor_scales * np.exp(
        np.clip(
            0.5 * production_log_squared,
            -config.log_scale_bound,
            config.log_scale_bound,
        )
    )
    production_calibration = np.sqrt(
        np.mean((data.factor_scores / production_scales) ** 2, axis=0)
    )
    return FactorLawFit(
        architecture="ar1",
        parameters={
            "drift_intercept": production_drift_intercept,
            "drift_slope": production_drift_slope,
            "variance_intercept": production_variance_intercept,
            "variance_slope": production_variance_slope,
            "scale_calibration": production_calibration,
        },
        terminal_state=(
            standardized_risk_prices[-1],
            log_squared_scores[-1],
        ),
        epochs=0,
        train_q_negative_log_likelihood=_q_negative_log_likelihood(
            data.factor_scores[:split], factor_scales[:split]
        ),
        validation_q_negative_log_likelihood=validation_q_nll,
        train_p_drift_standardized_rmse=train_rmse,
        validation_p_drift_standardized_rmse=validation_rmse,
        validation_p_drift_r2=validation_r2,
        accepted_q=validation_q_nll < fixed.validation_q_negative_log_likelihood,
        accepted_p=validation_rmse < fixed.validation_p_drift_standardized_rmse,
    )


def _mamba_forward(
    parameters: dict[str, jax.Array],
    inputs: jax.Array,
    reset: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    hidden_size = parameters["candidate_bias"].shape[0]

    def step(hidden: jax.Array, current: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        values, current_reset = current
        hidden = jnp.where(current_reset, jnp.zeros_like(hidden), hidden)
        decay = jax.nn.sigmoid(
            values @ parameters["decay_weight"] + parameters["decay_bias"]
        )
        candidate = jnp.tanh(
            values @ parameters["candidate_weight"]
            + parameters["candidate_bias"]
        )
        hidden = decay * hidden + (1.0 - decay) * candidate
        output = hidden @ parameters["output_weight"] + parameters["output_bias"]
        return hidden, output

    terminal, outputs = jax.lax.scan(
        step,
        jnp.zeros((hidden_size,), dtype=inputs.dtype),
        (inputs, reset),
    )
    return outputs, (terminal,)


def _lstm_forward(
    parameters: dict[str, jax.Array],
    inputs: jax.Array,
    reset: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    hidden_size = parameters["gate_bias"].shape[0] // 4

    def step(
        carry: tuple[jax.Array, jax.Array],
        current: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        hidden, cell = carry
        values, current_reset = current
        hidden = jnp.where(current_reset, jnp.zeros_like(hidden), hidden)
        cell = jnp.where(current_reset, jnp.zeros_like(cell), cell)
        gates = (
            values @ parameters["input_weight"]
            + hidden @ parameters["hidden_weight"]
            + parameters["gate_bias"]
        )
        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4)
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * jnp.tanh(cell)
        output = hidden @ parameters["output_weight"] + parameters["output_bias"]
        return (hidden, cell), output

    zeros = jnp.zeros((hidden_size,), dtype=inputs.dtype)
    terminal, outputs = jax.lax.scan(step, (zeros, zeros), (inputs, reset))
    return outputs, terminal


def _gru_forward(
    parameters: dict[str, jax.Array],
    inputs: jax.Array,
    reset: jax.Array,
    *,
    step_year_fraction: float,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Continuous-time GRU with exponential retention over one time step."""
    hidden_size = parameters["decay_bias"].shape[0]

    def step(
        hidden: jax.Array,
        current: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        values, current_reset = current
        hidden = jnp.where(current_reset, jnp.zeros_like(hidden), hidden)
        annual_decay = jax.nn.softplus(
            values @ parameters["decay_input_weight"]
            + hidden @ parameters["decay_hidden_weight"]
            + parameters["decay_bias"]
        )
        retention = jnp.exp(-annual_decay * step_year_fraction)
        reset_gate = jax.nn.sigmoid(
            values @ parameters["reset_input_weight"]
            + hidden @ parameters["reset_hidden_weight"]
            + parameters["reset_bias"]
        )
        candidate = jnp.tanh(
            values @ parameters["candidate_input_weight"]
            + (reset_gate * hidden) @ parameters["candidate_hidden_weight"]
            + parameters["candidate_bias"]
        )
        hidden = retention * hidden + (1.0 - retention) * candidate
        output = hidden @ parameters["output_weight"] + parameters["output_bias"]
        return hidden, output

    terminal, outputs = jax.lax.scan(
        step,
        jnp.zeros((hidden_size,), dtype=inputs.dtype),
        (inputs, reset),
    )
    return outputs, (terminal,)


def _initial_recurrent_parameters(
    *,
    architecture: Literal["lstm", "mamba", "gru"],
    input_size: int,
    hidden_size: int,
    factor_count: int,
    learn_spot_correlation: bool,
    seed: int,
    gru_initial_annual_decay: float,
) -> dict[str, FloatArray]:
    generator = np.random.default_rng(seed)
    correlation_count = factor_count - 1 if learn_spot_correlation else 0
    output_size = 2 * factor_count + correlation_count
    output_weight = np.zeros((hidden_size, output_size), dtype=np.float64)
    output_bias = np.zeros((output_size,), dtype=np.float64)
    if architecture == "mamba":
        return {
            "decay_weight": generator.normal(
                scale=0.03 / math.sqrt(input_size),
                size=(input_size, hidden_size),
            ),
            "decay_bias": np.full((hidden_size,), 1.5),
            "candidate_weight": generator.normal(
                scale=0.10 / math.sqrt(input_size),
                size=(input_size, hidden_size),
            ),
            "candidate_bias": np.zeros((hidden_size,)),
            "output_weight": output_weight,
            "output_bias": output_bias,
        }
    if architecture == "gru":
        return {
            "decay_input_weight": generator.normal(
                scale=0.03 / math.sqrt(input_size),
                size=(input_size, hidden_size),
            ),
            "decay_hidden_weight": generator.normal(
                scale=0.03 / math.sqrt(hidden_size),
                size=(hidden_size, hidden_size),
            ),
            "decay_bias": np.full((hidden_size,), gru_initial_annual_decay),
            "reset_input_weight": generator.normal(
                scale=0.10 / math.sqrt(input_size),
                size=(input_size, hidden_size),
            ),
            "reset_hidden_weight": generator.normal(
                scale=0.10 / math.sqrt(hidden_size),
                size=(hidden_size, hidden_size),
            ),
            "reset_bias": np.zeros((hidden_size,)),
            "candidate_input_weight": generator.normal(
                scale=0.10 / math.sqrt(input_size),
                size=(input_size, hidden_size),
            ),
            "candidate_hidden_weight": generator.normal(
                scale=0.10 / math.sqrt(hidden_size),
                size=(hidden_size, hidden_size),
            ),
            "candidate_bias": np.zeros((hidden_size,)),
            "output_weight": output_weight,
            "output_bias": output_bias,
        }
    gate_size = 4 * hidden_size
    gate_bias = np.zeros((gate_size,), dtype=np.float64)
    gate_bias[hidden_size : 2 * hidden_size] = 1.0
    return {
        "input_weight": generator.normal(
            scale=0.10 / math.sqrt(input_size),
            size=(input_size, gate_size),
        ),
        "hidden_weight": generator.normal(
            scale=0.10 / math.sqrt(hidden_size),
            size=(hidden_size, gate_size),
        ),
        "gate_bias": gate_bias,
        "output_weight": output_weight,
        "output_bias": output_bias,
    }


def _jax_bounded_spot_correlations(
    raw_correlations: jax.Array,
    *,
    bound: float,
) -> jax.Array:
    squared_norm = jnp.sum(raw_correlations**2, axis=-1, keepdims=True)
    return bound * raw_correlations / jnp.sqrt(1.0 + squared_norm)


def _jax_q_point_negative_log_likelihood(
    factor_scores: jax.Array,
    factor_scales: jax.Array,
    spot_correlations: jax.Array,
) -> jax.Array:
    standardized = factor_scores / factor_scales
    residual = standardized[:, 1:]
    squared_norm = jnp.sum(spot_correlations**2, axis=1)
    schur = jnp.maximum(1.0 - squared_norm, jnp.finfo(factor_scores.dtype).eps)
    conditional_spot = standardized[:, 0] - jnp.sum(
        spot_correlations * residual,
        axis=1,
    )
    quadratic = jnp.sum(residual**2, axis=1) + conditional_spot**2 / schur
    log_determinant = jnp.log(schur) + 2.0 * jnp.sum(
        jnp.log(factor_scales),
        axis=1,
    )
    return 0.5 * (quadratic + log_determinant) / factor_scores.shape[1]


def _split_recurrent_outputs(
    outputs: jax.Array,
    *,
    factor_count: int,
    learn_spot_correlation: bool,
    config: FactorLawConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    standardized_drift = outputs[:, :factor_count]
    relative_log_scale = jnp.clip(
        outputs[:, factor_count : 2 * factor_count],
        -config.log_scale_bound,
        config.log_scale_bound,
    )
    if learn_spot_correlation:
        spot_correlations = _jax_bounded_spot_correlations(
            outputs[:, 2 * factor_count :],
            bound=config.spot_correlation_bound,
        )
    else:
        spot_correlations = jnp.zeros(
            (outputs.shape[0], factor_count - 1),
            dtype=outputs.dtype,
        )
    return standardized_drift, relative_log_scale, spot_correlations


def fit_recurrent_factor_law(
    data: FactorLawDataset,
    *,
    architecture: Literal["lstm", "mamba", "gru"],
    config: FactorLawConfig,
    learn_spot_correlation: bool = False,
) -> FactorLawFit:
    """Fit a recurrent Q covariance and attainable P-drift model chronologically."""
    factor_count = data.factor_count
    if learn_spot_correlation and factor_count < 2:
        raise ValueError("spot correlation requires at least two factors")
    fit_architecture: Architecture = (
        f"{architecture}_corr" if learn_spot_correlation else architecture
    )
    parameters = _initial_recurrent_parameters(
        architecture=architecture,
        input_size=data.inputs.shape[1],
        hidden_size=config.hidden_size,
        factor_count=factor_count,
        learn_spot_correlation=learn_spot_correlation,
        seed=(
            config.seed
            + {"lstm": 0, "mamba": 1, "gru": 4}[architecture]
            + (2 if learn_spot_correlation else 0)
        ),
        gru_initial_annual_decay=config.gru_initial_annual_decay,
    )
    parameters_jax = jax.tree.map(jnp.asarray, parameters)
    inputs = jnp.asarray(data.inputs)
    reset = jnp.asarray(data.reset)
    factor_scores = jnp.asarray(data.factor_scores)
    fixed_scales = jnp.asarray(data.fixed_factor_scales)
    risk_targets = jnp.asarray(
        (data.physical_risk_prices - data.risk_price_mean)
        / data.risk_price_scale
    )
    split = data.train_count
    if architecture == "lstm":
        forward = _lstm_forward
    elif architecture == "mamba":
        forward = _mamba_forward
    else:
        forward = lambda parameters, values, current_reset: _gru_forward(
            parameters,
            values,
            current_reset,
            step_year_fraction=config.recurrent_step_year_fraction,
        )

    def losses(current: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        outputs, _ = forward(current, inputs, reset)
        standardized_drift, relative_log_scale, spot_correlations = (
            _split_recurrent_outputs(
                outputs,
                factor_count=factor_count,
                learn_spot_correlation=learn_spot_correlation,
                config=config,
            )
        )
        scales = fixed_scales * jnp.exp(relative_log_scale)
        q_point = _jax_q_point_negative_log_likelihood(
            factor_scores,
            scales,
            spot_correlations,
        )
        p_point = jnp.mean((standardized_drift - risk_targets) ** 2, axis=1)
        train_q = jnp.mean(q_point[:split])
        validation_q = jnp.mean(q_point[split:])
        train_p = jnp.mean(p_point[:split])
        validation_p = jnp.mean(p_point[split:])
        correlation_penalty = config.spot_correlation_ridge * jnp.mean(
            spot_correlations[:split] ** 2
        )
        train = train_q + config.drift_loss_weight * train_p + correlation_penalty
        return train, validation_q, train_p, validation_p

    value_and_gradient = jax.jit(
        jax.value_and_grad(lambda current: losses(current)[0])
    )
    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(parameters_jax)
    best_parameters = parameters_jax
    best_validation = math.inf
    best_epoch = 0
    started = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_loss, gradients = value_and_gradient(parameters_jax)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, parameters_jax
        )
        parameters_jax = optax.apply_updates(parameters_jax, updates)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.max_epochs:
            _, validation_q, _, validation_p = losses(parameters_jax)
            validation = float(
                validation_q + config.drift_loss_weight * validation_p
            )
            if validation < best_validation - 1.0e-7:
                best_validation = validation
                best_parameters = jax.tree.map(lambda value: value.copy(), parameters_jax)
                best_epoch = epoch
            if epoch == 1 or epoch % config.progress_every == 0:
                elapsed = perf_counter() - started
                rate = epoch / max(elapsed, 1.0e-9)
                eta = (config.max_epochs - epoch) / max(rate, 1.0e-9)
                print(
                    f"factor-law architecture={fit_architecture} "
                    f"epoch={epoch}/{config.max_epochs} "
                    f"train={float(train_loss):.6f} "
                    f"val_q_nll={float(validation_q):.6f} "
                    f"val_p_mse={float(validation_p):.6f} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            if epoch - best_epoch >= config.patience:
                break

    outputs, terminal = forward(best_parameters, inputs, reset)
    outputs_np = np.asarray(outputs)
    drift = data.risk_price_mean + data.risk_price_scale * outputs_np[:, :factor_count]
    relative_log_scale = outputs_np[:, factor_count : 2 * factor_count]
    scales = data.fixed_factor_scales * np.exp(
        np.clip(
            relative_log_scale,
            -config.log_scale_bound,
            config.log_scale_bound,
        )
    )
    spot_correlations = (
        _bounded_spot_correlations(
            outputs_np[:, 2 * factor_count :],
            bound=config.spot_correlation_bound,
        )
        if learn_spot_correlation
        else np.zeros((len(outputs_np), factor_count - 1), dtype=np.float64)
    )
    train_rmse, _ = _drift_metrics(
        data.physical_risk_prices[:split],
        drift[:split],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    validation_rmse, validation_r2 = _drift_metrics(
        data.physical_risk_prices[split:],
        drift[split:],
        mean=data.risk_price_mean,
        scale=data.risk_price_scale,
    )
    fixed = fit_fixed_factor_law(data)
    validation_q_nll = _q_negative_log_likelihood(
        data.factor_scores[split:],
        scales[split:],
        spot_correlations[split:],
    )

    def full_loss(current: dict[str, jax.Array]) -> jax.Array:
        full_outputs, _ = forward(current, inputs, reset)
        standardized_drift, relative_log_scale, full_correlations = (
            _split_recurrent_outputs(
                full_outputs,
                factor_count=factor_count,
                learn_spot_correlation=learn_spot_correlation,
                config=config,
            )
        )
        full_scales = fixed_scales * jnp.exp(relative_log_scale)
        q_loss = jnp.mean(
            _jax_q_point_negative_log_likelihood(
                factor_scores,
                full_scales,
                full_correlations,
            )
        )
        p_loss = jnp.mean((standardized_drift - risk_targets) ** 2)
        correlation_penalty = config.spot_correlation_ridge * jnp.mean(
            full_correlations**2
        )
        return q_loss + config.drift_loss_weight * p_loss + correlation_penalty

    production_parameters = jax.tree.map(
        jnp.asarray,
        _initial_recurrent_parameters(
            architecture=architecture,
            input_size=data.inputs.shape[1],
            hidden_size=config.hidden_size,
            factor_count=factor_count,
            learn_spot_correlation=learn_spot_correlation,
            seed=(
                config.seed
                + {"lstm": 0, "mamba": 1, "gru": 4}[architecture]
                + (2 if learn_spot_correlation else 0)
            ),
            gru_initial_annual_decay=config.gru_initial_annual_decay,
        ),
    )
    production_optimizer_state = optimizer.init(production_parameters)
    full_value_and_gradient = jax.jit(jax.value_and_grad(full_loss))
    for epoch in range(1, best_epoch + 1):
        _, gradients = full_value_and_gradient(production_parameters)
        updates, production_optimizer_state = optimizer.update(
            gradients,
            production_optimizer_state,
            production_parameters,
        )
        production_parameters = optax.apply_updates(production_parameters, updates)
        if epoch == best_epoch or epoch % config.progress_every == 0:
            print(
                f"factor-law-refit architecture={fit_architecture} "
                f"epoch={epoch}/{best_epoch}",
                flush=True,
            )
    _, production_terminal = forward(production_parameters, inputs, reset)
    return FactorLawFit(
        architecture=fit_architecture,
        parameters=jax.tree.map(
            lambda value: np.asarray(value), production_parameters
        ),
        terminal_state=tuple(np.asarray(value) for value in production_terminal),
        epochs=best_epoch,
        train_q_negative_log_likelihood=_q_negative_log_likelihood(
            data.factor_scores[:split],
            scales[:split],
            spot_correlations[:split],
        ),
        validation_q_negative_log_likelihood=validation_q_nll,
        train_p_drift_standardized_rmse=train_rmse,
        validation_p_drift_standardized_rmse=validation_rmse,
        validation_p_drift_r2=validation_r2,
        accepted_q=validation_q_nll < fixed.validation_q_negative_log_likelihood,
        accepted_p=validation_rmse < fixed.validation_p_drift_standardized_rmse,
    )


def _fit_report(fit: FactorLawFit) -> dict[str, Any]:
    return {
        "architecture": fit.architecture,
        "epochs": fit.epochs,
        "train_q_negative_log_likelihood": fit.train_q_negative_log_likelihood,
        "validation_q_negative_log_likelihood": (
            fit.validation_q_negative_log_likelihood
        ),
        "train_p_drift_standardized_rmse": (
            fit.train_p_drift_standardized_rmse
        ),
        "validation_p_drift_standardized_rmse": (
            fit.validation_p_drift_standardized_rmse
        ),
        "validation_p_drift_r2": fit.validation_p_drift_r2,
        "accepted_q": fit.accepted_q,
        "accepted_p": fit.accepted_p,
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def estimate_factor_laws(
    *,
    base_estimate: Path,
    history_path: Path,
    dataset: Path,
    output: Path,
    config: FactorLawConfig = FactorLawConfig(),
) -> dict[str, Any]:
    """Fit and serialize fixed, AR(1), LSTM, Mamba and GRU factor laws."""
    data = build_factor_law_dataset(
        base_estimate=base_estimate,
        history_path=history_path,
        dataset=dataset,
        config=config,
    )
    fits = (
        fit_fixed_factor_law(data),
        fit_ar1_factor_law(data, config=config),
        fit_recurrent_factor_law(data, architecture="lstm", config=config),
        fit_recurrent_factor_law(data, architecture="mamba", config=config),
        fit_recurrent_factor_law(data, architecture="gru", config=config),
    )
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("gromm2-factor-laws-v1"),
        "config": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "dates": data.dates.astype("S10"),
        "origin_indices": data.origin_indices,
        "input_mean": data.input_mean.astype(np.float32),
        "input_scale": data.input_scale.astype(np.float32),
        "fixed_factor_scales": data.fixed_factor_scales.astype(np.float32),
        "risk_price_mean": data.risk_price_mean.astype(np.float32),
        "risk_price_scale": data.risk_price_scale.astype(np.float32),
        "train_count": np.asarray(data.train_count, dtype=np.int32),
    }
    for fit in fits:
        for name, value in fit.parameters.items():
            arrays[f"{fit.architecture}_parameter_{name}"] = np.asarray(
                value, dtype=np.float32
            )
        for index, value in enumerate(fit.terminal_state):
            arrays[f"{fit.architecture}_terminal_{index}"] = np.asarray(
                value, dtype=np.float32
            )
    _atomic_save_npz(output, arrays)
    report = {
        "schema": "gromm2-factor-laws-v1",
        "base_estimate": str(base_estimate.resolve()),
        "base_estimate_sha256": _file_sha256(base_estimate),
        "history_path": str(history_path.resolve()),
        "history_sha256": _file_sha256(history_path),
        "dataset": str(dataset.resolve()),
        "config": asdict(config),
        "pair_count": len(data.factor_scores),
        "train_count": data.train_count,
        "validation_count": len(data.factor_scores) - data.train_count,
        "factor_count": data.factor_count,
        "train_end_date": str(data.dates[data.train_count - 1]),
        "validation_start_date": str(data.dates[data.train_count]),
        "validation_end_date": str(data.dates[-1]),
        "interpretation": {
            "q": "zero factor means with fitted conditional scales",
            "p": (
                "Q drift plus the base loading field times fitted attainable "
                "factor-drift coefficients; with conditional scale s, the "
                "equivalent market price of risk is coefficient/s"
            ),
            "mamba": "bounded selective state-space recurrence",
            "gru": "reset-gated candidate with exponential continuous-time retention",
        },
        "models": {fit.architecture: _fit_report(fit) for fit in fits},
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-estimate", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=2_000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261013)
    arguments = parser.parse_args()
    report = estimate_factor_laws(
        base_estimate=arguments.base_estimate,
        history_path=arguments.history,
        dataset=arguments.dataset,
        output=arguments.output,
        config=FactorLawConfig(
            max_epochs=arguments.max_epochs,
            patience=arguments.patience,
            hidden_size=arguments.hidden_size,
            seed=arguments.seed,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
