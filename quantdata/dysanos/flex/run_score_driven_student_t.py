"""Fit and audit a no-replay heavy-tailed Direct-200 innovation law."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.config import (
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    JointDirectDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.score_driven_student_t import (
    ScoreDrivenStudentTConfig,
    filter_score_driven_student_t,
    fit_score_driven_student_t,
)
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--step-sessions",
        type=int,
        choices=(1, 5),
        default=5,
        help=(
            "Innovation/scale update frequency. Five sessions matches the "
            "horizon on which the affine state law is fitted."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42_071)
    return parser.parse_args()


def _checkpoint_config(payload: Mapping[str, Any]) -> JointDirectDynamicsConfig:
    values = payload.get("model_config")
    if not isinstance(values, Mapping):
        raise ValueError("Direct-200 checkpoint has no model configuration")
    geometry_values = values.get("geometry")
    network_values = values.get("decoder_network")
    if not isinstance(geometry_values, Mapping) or not isinstance(network_values, Mapping):
        raise ValueError("Direct-200 checkpoint geometry/network is invalid")
    geometry_payload = dict(geometry_values)
    for key in ("nms_range", "explicit_model_business_dtes"):
        if geometry_payload.get(key) is not None:
            geometry_payload[key] = tuple(geometry_payload[key])
    geometry_type = (
        FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig
        if geometry_payload.get("front_term_eta") is True
        else SharedStrikeCappedUnboundedDlvFlexGeometryConfig
    )
    geometry = geometry_type(**geometry_payload)
    return JointDirectDynamicsConfig(
        label=str(values["label"]),
        state_size=int(values["state_size"]),
        group=str(values["group"]),
        user_version=str(values["user_version"]),
        geometry=geometry,
        decoder_network=FeedForwardConfig(**dict(network_values)),
        minimum_mean_reversion_per_week=float(values["minimum_mean_reversion_per_week"]),
        innovation_std_floor=float(values["innovation_std_floor"]),
    )


def _load_model(
    checkpoint: Path,
    *,
    device: str,
) -> tuple[JointDirectDynamicsModel, Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "flex-joint-direct-dynamics-v1"
        or payload.get("phase") != "complete"
    ):
        raise ValueError(f"not a completed Direct-200 checkpoint: {checkpoint}")
    config = _checkpoint_config(payload)
    if payload.get("model_config_version") != config.version.unique_id48:
        raise ValueError("Direct-200 checkpoint code version is stale")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Direct-200 checkpoint has no model state")
    daily_states = state_dict.get("daily_state_embedding.weight")
    if not isinstance(daily_states, torch.Tensor) or daily_states.ndim != 2:
        raise ValueError("Direct-200 checkpoint has no daily state history")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = JointDirectDynamicsModel(
        env,
        num_days=daily_states.shape[0],
        config=config,
    )
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    model.eval()
    return model, payload


def _load_dates(path: Path) -> tuple[date, ...]:
    payload = json.loads(path.read_text())
    values = payload.get("market_dates")
    if not isinstance(values, list):
        raise ValueError(f"native completion has no market dates: {path}")
    dates = tuple(date.fromisoformat(str(value)) for value in values)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("native completion dates are not unique and increasing")
    return dates


def _exact_session_pairs(
    dates: tuple[date, ...],
    *,
    sessions: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    index = {value: position for position, value in enumerate(dates)}
    calendar = get_business_days(dates[0], dates[-1])
    calendar_index = {value: position for position, value in enumerate(calendar)}
    pairs = [
        (origin, index[calendar[position + sessions]])
        for origin, value in enumerate(dates)
        if (position := calendar_index[value]) + sessions < len(calendar)
        and calendar[position + sessions] in index
    ]
    return (
        np.asarray([item[0] for item in pairs], dtype=np.int64),
        np.asarray([item[1] for item in pairs], dtype=np.int64),
    )


def _daily_law(
    model: JointDirectDynamicsModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with torch.no_grad():
        weekly_transition = model.weekly_transition().cpu().double().numpy()
        weekly_factor = model.innovation_cholesky().cpu().double().numpy()
        generator = model.mean_reversion_generator().cpu().double()
        daily_transition = torch.matrix_exp(-generator / 5.0).numpy()
    weekly_covariance = weekly_factor @ weekly_factor.T
    stationary = solve_discrete_lyapunov(
        weekly_transition.T,
        weekly_covariance,
    )
    stationary = 0.5 * (stationary + stationary.T)
    daily_covariance = stationary - daily_transition.T @ stationary @ daily_transition
    daily_covariance = 0.5 * (daily_covariance + daily_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(daily_covariance)
    floor = 64.0 * np.finfo(np.float64).eps * max(float(np.max(eigenvalues)), 1.0)
    clipped = np.maximum(eigenvalues, floor)
    daily_factor = eigenvectors @ np.diag(np.sqrt(clipped))
    composed_covariance = np.zeros_like(daily_covariance)
    power = np.eye(len(daily_transition))
    for _ in range(5):
        composed_covariance += power.T @ daily_covariance @ power
        power = power @ daily_transition
    diagnostics = {
        "transition_maximum_absolute_error": float(np.max(np.abs(power - weekly_transition))),
        "covariance_maximum_absolute_error": float(
            np.max(np.abs(composed_covariance - weekly_covariance))
        ),
        "daily_covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
        "daily_covariance_clipped_eigenvalues": int(np.sum(eigenvalues < floor)),
    }
    return daily_transition, daily_factor, stationary, diagnostics


def _horizon_covariance(
    transition: np.ndarray,
    covariance: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    output = np.zeros_like(covariance)
    power = np.eye(len(transition))
    for _ in range(steps):
        output += power.T @ covariance @ power
        power = power @ transition
    return 0.5 * (output + output.T)


def _squared_radii(
    residuals: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = 64.0 * np.finfo(np.float64).eps * max(float(np.max(eigenvalues)), 1.0)
    rotated = residuals @ eigenvectors
    return np.sum(np.square(rotated) / np.maximum(eigenvalues, floor), axis=1)


def _historical_residuals(
    states: np.ndarray,
    dates: tuple[date, ...],
    *,
    mean: np.ndarray,
    session_transition: np.ndarray,
    session_covariance: np.ndarray,
    sessions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact-horizon residuals from a one-session transition law."""
    origin, endpoint = _exact_session_pairs(dates, sessions=sessions)
    horizon_transition = np.linalg.matrix_power(session_transition, sessions)
    prediction = mean + (states[origin] - mean) @ horizon_transition
    residual = states[endpoint] - prediction
    horizon_covariance = _horizon_covariance(
        session_transition,
        session_covariance,
        steps=sessions,
    )
    radii = _squared_radii(residual, horizon_covariance)
    return origin, endpoint, residual, radii


def _segment_starts(origin: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    starts = np.ones(len(origin), dtype=np.bool_)
    if len(origin) > 1:
        starts[1:] = origin[1:] != endpoint[:-1]
    return starts


def _filter_order(
    origin: np.ndarray,
    dates: tuple[date, ...],
    *,
    sessions: int,
) -> np.ndarray:
    """Order overlapping transitions into independent session-offset chains."""
    calendar = get_business_days(dates[0], dates[-1])
    positions = {value: index for index, value in enumerate(calendar)}
    return np.asarray(
        sorted(
            range(len(origin)),
            key=lambda index: (
                positions[dates[int(origin[index])]] % sessions,
                positions[dates[int(origin[index])]],
            ),
        ),
        dtype=np.int64,
    )


def _nll_summary(
    radii: np.ndarray,
    starts: np.ndarray,
    *,
    dimension: int,
    dynamic_config: ScoreDrivenStudentTConfig,
    iid_degrees_of_freedom: float,
    iid_unconditional_scale: float,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if evaluation_mask is None:
        selected = np.ones(len(radii), dtype=np.bool_)
    else:
        selected = np.asarray(evaluation_mask, dtype=np.bool_)
        if selected.shape != radii.shape or not np.any(selected):
            raise ValueError("evaluation_mask must select aligned likelihood terms")
    dynamic = filter_score_driven_student_t(
        radii,
        starts,
        dimension=dimension,
        config=dynamic_config,
    )
    iid = filter_score_driven_student_t(
        radii,
        starts,
        dimension=dimension,
        config=ScoreDrivenStudentTConfig(
            degrees_of_freedom=iid_degrees_of_freedom,
            score_loading=0.0,
            persistence=0.0,
            unconditional_scale=iid_unconditional_scale,
        ),
    )
    gaussian = 0.5 * (dimension * math.log(2.0 * math.pi) + radii)
    return {
        "observations": int(np.sum(selected)),
        "gaussian_nll_per_dimension": float(np.mean(gaussian[selected]) / dimension),
        "iid_student_t_nll_per_dimension": float(
            -np.mean(iid.log_likelihood[selected]) / dimension
        ),
        "dynamic_student_t_nll_per_dimension": float(
            -np.mean(dynamic.log_likelihood[selected]) / dimension
        ),
    }


def _market_tail_diagnostics(
    states: np.ndarray,
    dates: tuple[date, ...],
    *,
    mean: np.ndarray,
    daily_transition: np.ndarray,
    daily_covariance: np.ndarray,
) -> tuple[dict[int, float], dict[str, object]]:
    dimension = states.shape[1]
    thresholds: dict[int, float] = {}
    output: dict[str, object] = {}
    one_week_by_origin: dict[int, float] = {}
    for weeks in (1, 2, 3):
        origin, endpoint, _, radii = _historical_residuals(
            states,
            dates,
            mean=mean,
            session_transition=daily_transition,
            session_covariance=daily_covariance,
            sessions=5 * weeks,
        )
        normalized = radii / dimension
        threshold = float(np.quantile(normalized, 0.95))
        training = np.asarray(
            [dates[index].year < 2025 for index in endpoint],
            dtype=np.bool_,
        )
        training_threshold = float(np.quantile(normalized[training], 0.95))
        thresholds[weeks] = training_threshold
        validation = ~training
        validation_pairs = int(np.sum(validation))
        if validation_pairs == 0:
            raise ValueError("market tail audit has no 2025 validation pairs")
        validation_indicators = normalized[validation] > training_threshold
        validation_exceedance = float(np.mean(validation_indicators))
        block_uncertainty = _circular_block_mean_uncertainty(
            validation_indicators,
            block_length=15,
            seed=83_911 + weeks,
        )
        output[f"{weeks}w"] = {
            "pairs": len(normalized),
            "normalized_radius_mean": float(np.mean(normalized)),
            "normalized_radius_q95": threshold,
            "full_sample_exceedance": float(np.mean(normalized > threshold)),
            "pre_2025_q95": training_threshold,
            "2025_pairs": validation_pairs,
            "2025_exceedance_of_pre_2025_q95": validation_exceedance,
            "2025_exceedance_binomial_standard_error": math.sqrt(
                validation_exceedance
                * (1.0 - validation_exceedance)
                / validation_pairs
            ),
            "2025_exceedance_block_bootstrap": block_uncertainty,
            "gaussian_exceedance_of_full_q95": float(
                chi2.sf(dimension * threshold, dimension)
            ),
            "gaussian_exceedance_of_pre_2025_q95": float(
                chi2.sf(dimension * training_threshold, dimension)
            ),
        }
        if weeks == 1:
            one_week_by_origin = {
                int(index): float(value) for index, value in zip(origin, normalized, strict=True)
            }

    first: list[float] = []
    second: list[float] = []
    _, one_week_endpoint = _exact_session_pairs(dates, sessions=5)
    for origin_index, endpoint_index in zip(
        sorted(one_week_by_origin), one_week_endpoint, strict=True
    ):
        if int(endpoint_index) in one_week_by_origin:
            first.append(one_week_by_origin[origin_index])
            second.append(one_week_by_origin[int(endpoint_index)])
    first_values = np.asarray(first)
    second_values = np.asarray(second)
    threshold = thresholds[1]
    tail = first_values > threshold
    output["one_week_serial_tail"] = {
        "pairs": len(first_values),
        "squared_radius_correlation": float(np.corrcoef(first_values, second_values)[0, 1]),
        "tail_origins": int(np.sum(tail)),
        "next_week_exceedance_given_tail_week": float(np.mean(second_values[tail] > threshold)),
        "unconditional_next_week_exceedance": float(np.mean(second_values > threshold)),
    }
    return thresholds, output


def _circular_block_mean_uncertainty(
    values: np.ndarray,
    *,
    block_length: int,
    replications: int = 9_999,
    seed: int,
) -> dict[str, object]:
    """Estimate uncertainty of a serially dependent sample mean."""
    sample = np.asarray(values, dtype=np.float64)
    if (
        sample.ndim != 1
        or len(sample) < 2
        or np.any(~np.isfinite(sample))
        or block_length <= 0
        or replications <= 0
    ):
        raise ValueError("block-bootstrap inputs must define a finite vector")
    block = min(block_length, len(sample))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        len(sample),
        size=(replications, math.ceil(len(sample) / block)),
    )
    offsets = np.arange(block)
    indices = (starts[..., None] + offsets) % len(sample)
    indices = indices.reshape(replications, -1)[:, : len(sample)]
    means = np.mean(sample[indices], axis=1)
    return {
        "block_length": block,
        "replications": replications,
        "standard_error": float(np.std(means, ddof=1)),
        "percentile_95_interval": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def _torch_standardized_student_t(
    paths: int,
    dimension: int,
    degrees_of_freedom: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = torch.randn(
        (paths, dimension),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    gamma_shape = torch.full(
        (paths,),
        0.5 * degrees_of_freedom,
        dtype=dtype,
        device=device,
    )
    chi_square = 2.0 * torch._standard_gamma(
        gamma_shape,
        generator=generator,
    )
    shocks = normal * torch.sqrt((degrees_of_freedom - 2.0) / chi_square)[:, None]
    return shocks, torch.sum(shocks.square(), dim=1)


def _stationary_log_scale_state(
    paths: int,
    *,
    dimension: int,
    config: ScoreDrivenStudentTConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    tolerance: float = 1.0e-10,
) -> torch.Tensor:
    """Draw the exact stationary weighted sum of iid Student-t scores."""
    state = torch.zeros(paths, dtype=dtype, device=device)
    if config.score_loading == 0.0:
        return state
    first_shape = torch.full((paths,), 0.5 * dimension, dtype=dtype, device=device)
    second_shape = torch.full((paths,), 0.5 * config.degrees_of_freedom, dtype=dtype, device=device)
    coefficient = config.score_loading
    for _ in range(100_000):
        first = torch._standard_gamma(first_shape, generator=generator)
        second = torch._standard_gamma(second_shape, generator=generator)
        beta_draw = first / (first + second)
        score = 0.5 * ((dimension + config.degrees_of_freedom) * beta_draw - dimension)
        state += coefficient * score
        coefficient *= config.persistence
        if coefficient <= tolerance:
            return state
    raise RuntimeError("stationary log-scale sampler did not converge")


def _summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, (0.01, 0.05, 0.50, 0.95, 0.99))
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "maximum": float(np.max(values)),
    }


def _simulate(
    *,
    config: ScoreDrivenStudentTConfig,
    dimension: int,
    step_transition: np.ndarray,
    step_factor: np.ndarray,
    step_sessions: int,
    thresholds: dict[int, float],
    paths: int,
    batch_size: int,
    device: str,
    seed: int,
    log_mean_normalizer: float,
) -> dict[str, object]:
    selected_device = torch.device(device)
    dtype = torch.float64
    transition = torch.as_tensor(
        step_transition,
        dtype=dtype,
        device=selected_device,
    )
    factor = torch.as_tensor(step_factor, dtype=dtype, device=selected_device)
    step_covariance = step_factor @ step_factor.T
    steps_per_week = 5 // step_sessions
    horizon_covariances = {
        week: _horizon_covariance(
            step_transition,
            step_covariance,
            steps=steps_per_week * week,
        )
        for week in (1, 2, 3)
    }
    inverse_factors = {
        week: torch.linalg.cholesky(
            torch.as_tensor(
                covariance,
                dtype=dtype,
                device=selected_device,
            )
        )
        for week, covariance in horizon_covariances.items()
    }
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(seed)
    radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    weekly_radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    scale_samples: list[np.ndarray] = []
    sums = {week: torch.zeros(dimension, dtype=dtype, device=selected_device) for week in (1, 2, 3)}
    outer_sums = {
        week: torch.zeros((dimension, dimension), dtype=dtype, device=selected_device)
        for week in (1, 2, 3)
    }
    completed = 0
    while completed < paths:
        current = min(batch_size, paths - completed)
        log_scale_state = _stationary_log_scale_state(
            current,
            dimension=dimension,
            config=config,
            dtype=dtype,
            device=selected_device,
            generator=generator,
        )
        residual = torch.zeros((current, dimension), dtype=dtype, device=selected_device)
        weekly_residual = torch.zeros_like(residual)
        for step_index in range(1, 3 * steps_per_week + 1):
            shocks, squared_radius = _torch_standardized_student_t(
                current,
                dimension,
                config.degrees_of_freedom,
                dtype=dtype,
                device=selected_device,
                generator=generator,
            )
            scale = config.unconditional_scale * torch.exp(log_scale_state - log_mean_normalizer)
            innovation = torch.sqrt(scale)[:, None] * (shocks @ factor.T)
            residual = residual @ transition + innovation
            weekly_residual = weekly_residual @ transition + innovation
            weight = squared_radius / (config.degrees_of_freedom - 2.0 + squared_radius)
            score = 0.5 * ((config.degrees_of_freedom + dimension) * weight - dimension)
            log_scale_state = config.persistence * log_scale_state + config.score_loading * score
            if step_index == 1 or step_index % steps_per_week == 0:
                scale_samples.append(scale.detach().cpu().numpy())
            if step_index % steps_per_week != 0:
                continue
            week = step_index // steps_per_week
            factor_horizon = inverse_factors[week]
            white = torch.linalg.solve_triangular(
                factor_horizon,
                residual.T,
                upper=False,
            ).T
            normalized_radius = torch.sum(white.square(), dim=1) / dimension
            radii[week].append(normalized_radius.detach().cpu().numpy())
            white_week = torch.linalg.solve_triangular(
                inverse_factors[1],
                weekly_residual.T,
                upper=False,
            ).T
            weekly_radii[week].append(
                (torch.sum(white_week.square(), dim=1) / dimension).detach().cpu().numpy()
            )
            sums[week] += torch.sum(residual, dim=0)
            outer_sums[week] += residual.T @ residual
            weekly_residual.zero_()
        completed += current
        print(f"tail simulation paths={completed}/{paths}", flush=True)

    concatenated = {week: np.concatenate(values) for week, values in radii.items()}
    consecutive = {week: np.concatenate(values) for week, values in weekly_radii.items()}
    output: dict[str, object] = {}
    for week in (1, 2, 3):
        threshold = thresholds[week]
        values = concatenated[week]
        probability = float(np.mean(values > threshold))
        mean = sums[week].cpu().numpy() / paths
        covariance = outer_sums[week].cpu().numpy() / paths - np.outer(mean, mean)
        target = horizon_covariances[week]
        output[f"{week}w"] = {
            "normalized_radius": _summary(values),
            "pre_2025_q95_threshold": threshold,
            "exceedance_probability": probability,
            "binomial_standard_error": math.sqrt(probability * (1.0 - probability) / paths),
            "covariance_trace_ratio": float(np.trace(covariance) / np.trace(target)),
            "covariance_relative_frobenius_error": float(
                np.linalg.norm(covariance - target) / np.linalg.norm(target)
            ),
            "maximum_absolute_mean_in_target_standard_deviations": float(
                np.max(np.abs(mean) / np.sqrt(np.diag(target)))
            ),
        }
    first = consecutive[1]
    second = consecutive[2]
    threshold = thresholds[1]
    tail = first > threshold
    output["one_week_serial_tail"] = {
        "squared_radius_correlation": float(np.corrcoef(first, second)[0, 1]),
        "tail_origins": int(np.sum(tail)),
        "next_week_exceedance_given_tail_week": float(np.mean(second[tail] > threshold)),
        "unconditional_next_week_exceedance": float(np.mean(second > threshold)),
    }
    output["conditional_scale"] = _summary(np.concatenate(scale_samples))
    return output


def _simulate_historical_replay(
    *,
    residual_pool: np.ndarray,
    dimension: int,
    step_transition: np.ndarray,
    step_factor: np.ndarray,
    step_sessions: int,
    thresholds: dict[int, float],
    paths: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    """Simulate independent centered historical innovations without blocks."""
    pool = np.asarray(residual_pool, dtype=np.float64)
    if (
        pool.ndim != 2
        or pool.shape[1] != dimension
        or len(pool) < 2
        or np.any(~np.isfinite(pool))
    ):
        raise ValueError("historical replay pool must be a finite residual matrix")
    pool = pool - np.mean(pool, axis=0, keepdims=True)
    selected_device = torch.device(device)
    dtype = torch.float64
    replay_pool = torch.as_tensor(pool, dtype=dtype, device=selected_device)
    transition = torch.as_tensor(
        step_transition, dtype=dtype, device=selected_device
    )
    step_covariance = step_factor @ step_factor.T
    steps_per_week = 5 // step_sessions
    horizon_covariances = {
        week: _horizon_covariance(
            step_transition,
            step_covariance,
            steps=steps_per_week * week,
        )
        for week in (1, 2, 3)
    }
    horizon_factors = {
        week: torch.linalg.cholesky(
            torch.as_tensor(covariance, dtype=dtype, device=selected_device)
        )
        for week, covariance in horizon_covariances.items()
    }
    generator = torch.Generator(device=selected_device).manual_seed(seed)
    radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    weekly_radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    sums = {
        week: torch.zeros(dimension, dtype=dtype, device=selected_device)
        for week in (1, 2, 3)
    }
    outer_sums = {
        week: torch.zeros((dimension, dimension), dtype=dtype, device=selected_device)
        for week in (1, 2, 3)
    }
    completed = 0
    while completed < paths:
        current = min(batch_size, paths - completed)
        residual = torch.zeros(
            (current, dimension), dtype=dtype, device=selected_device
        )
        weekly_residual = torch.zeros_like(residual)
        for step_index in range(1, 3 * steps_per_week + 1):
            choices = torch.randint(
                len(replay_pool),
                (current,),
                device=selected_device,
                generator=generator,
            )
            innovation = replay_pool[choices]
            residual = residual @ transition + innovation
            weekly_residual = weekly_residual @ transition + innovation
            if step_index % steps_per_week != 0:
                continue
            week = step_index // steps_per_week
            white = torch.linalg.solve_triangular(
                horizon_factors[week], residual.T, upper=False
            ).T
            radii[week].append(
                (torch.sum(white.square(), dim=1) / dimension).cpu().numpy()
            )
            white_week = torch.linalg.solve_triangular(
                horizon_factors[1], weekly_residual.T, upper=False
            ).T
            weekly_radii[week].append(
                (torch.sum(white_week.square(), dim=1) / dimension).cpu().numpy()
            )
            sums[week] += torch.sum(residual, dim=0)
            outer_sums[week] += residual.T @ residual
            weekly_residual.zero_()
        completed += current
        print(f"historical replay paths={completed}/{paths}", flush=True)

    concatenated = {week: np.concatenate(values) for week, values in radii.items()}
    consecutive = {
        week: np.concatenate(values) for week, values in weekly_radii.items()
    }
    output: dict[str, object] = {
        "source_residuals": len(pool),
        "source_residuals_centered": True,
    }
    for week in (1, 2, 3):
        values = concatenated[week]
        probability = float(np.mean(values > thresholds[week]))
        mean = sums[week].cpu().numpy() / paths
        covariance = outer_sums[week].cpu().numpy() / paths - np.outer(mean, mean)
        target = horizon_covariances[week]
        output[f"{week}w"] = {
            "normalized_radius": _summary(values),
            "pre_2025_q95_threshold": thresholds[week],
            "exceedance_probability": probability,
            "binomial_standard_error": math.sqrt(
                probability * (1.0 - probability) / paths
            ),
            "covariance_trace_ratio": float(
                np.trace(covariance) / np.trace(target)
            ),
            "covariance_relative_frobenius_error": float(
                np.linalg.norm(covariance - target) / np.linalg.norm(target)
            ),
            "maximum_absolute_mean_in_target_standard_deviations": float(
                np.max(np.abs(mean) / np.sqrt(np.diag(target)))
            ),
        }
    first, second = consecutive[1], consecutive[2]
    threshold = thresholds[1]
    tail = first > threshold
    output["one_week_serial_tail"] = {
        "squared_radius_correlation": float(np.corrcoef(first, second)[0, 1]),
        "tail_origins": int(np.sum(tail)),
        "next_week_exceedance_given_tail_week": float(
            np.mean(second[tail] > threshold)
        ),
        "unconditional_next_week_exceedance": float(np.mean(second > threshold)),
    }
    return output


def main() -> None:
    arguments = _parse_arguments()
    if arguments.paths <= 0 or arguments.batch_size <= 0:
        raise ValueError("paths and batch-size must be positive")
    model, checkpoint_payload = _load_model(
        arguments.checkpoint,
        device=arguments.device,
    )
    dates = _load_dates(arguments.native_completion)
    with torch.no_grad():
        states = model.daily_states.cpu().double().numpy()
        mean = model.equilibrium_mean.cpu().double().numpy()
    if len(states) != len(dates):
        raise ValueError("checkpoint state history and completion dates disagree")
    daily_transition, daily_factor, stationary, transition_audit = _daily_law(model)
    daily_covariance = daily_factor @ daily_factor.T
    step_transition = np.linalg.matrix_power(
        daily_transition,
        arguments.step_sessions,
    )
    step_covariance = _horizon_covariance(
        daily_transition,
        daily_covariance,
        steps=arguments.step_sessions,
    )
    step_eigenvalues, step_eigenvectors = np.linalg.eigh(step_covariance)
    step_floor = 64.0 * np.finfo(np.float64).eps * max(float(np.max(step_eigenvalues)), 1.0)
    step_factor = step_eigenvectors @ np.diag(np.sqrt(np.maximum(step_eigenvalues, step_floor)))
    origin, endpoint, step_residuals, daily_radii = _historical_residuals(
        states,
        dates,
        mean=mean,
        session_transition=daily_transition,
        session_covariance=daily_covariance,
        sessions=arguments.step_sessions,
    )
    pre_2025 = np.asarray([dates[index].year < 2025 for index in endpoint], dtype=np.bool_)
    fit_origin = origin[pre_2025]
    fit_endpoint = endpoint[pre_2025]
    fit_order = _filter_order(
        fit_origin,
        dates,
        sessions=arguments.step_sessions,
    )
    full_order = _filter_order(
        origin,
        dates,
        sessions=arguments.step_sessions,
    )
    fit = fit_score_driven_student_t(
        daily_radii[pre_2025][fit_order],
        _segment_starts(fit_origin[fit_order], fit_endpoint[fit_order]),
        dimension=model.config.state_size,
    )
    validation_likelihood = _nll_summary(
        daily_radii[full_order],
        _segment_starts(origin[full_order], endpoint[full_order]),
        dimension=model.config.state_size,
        dynamic_config=fit.config,
        iid_degrees_of_freedom=fit.iid_student_t_degrees_of_freedom,
        iid_unconditional_scale=fit.iid_student_t_unconditional_scale,
        evaluation_mask=(~pre_2025)[full_order],
    )
    thresholds, market_tails = _market_tail_diagnostics(
        states,
        dates,
        mean=mean,
        daily_transition=daily_transition,
        daily_covariance=daily_covariance,
    )
    simulation = _simulate(
        config=fit.config,
        dimension=model.config.state_size,
        step_transition=step_transition,
        step_factor=step_factor,
        step_sessions=arguments.step_sessions,
        thresholds=thresholds,
        paths=arguments.paths,
        batch_size=arguments.batch_size,
        device=arguments.device,
        seed=arguments.seed,
        log_mean_normalizer=fit.filter.log_mean_normalizer,
    )
    iid_config = ScoreDrivenStudentTConfig(
        degrees_of_freedom=fit.iid_student_t_degrees_of_freedom,
        score_loading=0.0,
        persistence=0.0,
        unconditional_scale=fit.iid_student_t_unconditional_scale,
    )
    iid_simulation = _simulate(
        config=iid_config,
        dimension=model.config.state_size,
        step_transition=step_transition,
        step_factor=step_factor,
        step_sessions=arguments.step_sessions,
        thresholds=thresholds,
        paths=arguments.paths,
        batch_size=arguments.batch_size,
        device=arguments.device,
        seed=arguments.seed + 1,
        log_mean_normalizer=0.0,
    )
    replay_simulation = _simulate_historical_replay(
        residual_pool=step_residuals[pre_2025],
        dimension=model.config.state_size,
        step_transition=step_transition,
        step_factor=step_factor,
        step_sessions=arguments.step_sessions,
        thresholds=thresholds,
        paths=arguments.paths,
        batch_size=arguments.batch_size,
        device=arguments.device,
        seed=arguments.seed + 2,
    )
    report: dict[str, object] = {
        "schema": "flex-score-driven-student-t-v1",
        "status": "complete",
        "model": (
            "multivariate Student-t innovations with a fitted global covariance "
            "scale and a unit-mean persistent score-driven common variance; no "
            "historical residual is replayed"
        ),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "native_completion": str(arguments.native_completion),
        "model_config": asdict(model.config),
        "source_checkpoint_train_config": checkpoint_payload.get("train_config"),
        "dates": {
            "first": dates[0].isoformat(),
            "last": dates[-1].isoformat(),
            "count": len(dates),
            "fit_innovations_pre_2025": int(np.sum(pre_2025)),
            "validation_innovations_2025": int(np.sum(~pre_2025)),
            "innovation_step_sessions": arguments.step_sessions,
        },
        "daily_transition_audit": transition_audit,
        "invariant_covariance": {
            "condition_number": float(np.linalg.cond(stationary)),
            "effective_rank": float(
                np.exp(
                    -np.sum(
                        (eigenvalues := np.linalg.eigvalsh(stationary))
                        / np.sum(eigenvalues)
                        * np.log(
                            np.maximum(
                                eigenvalues / np.sum(eigenvalues),
                                np.finfo(np.float64).tiny,
                            )
                        )
                    )
                )
            ),
        },
        "fit": fit.to_dict(),
        "validation_2025_likelihood": validation_likelihood,
        "market_tail_diagnostics": market_tails,
        "simulation": {
            "paths": arguments.paths,
            "batch_size": arguments.batch_size,
            "tail_threshold_definition": (
                "Pre-2025 empirical 95th percentile of the normalized "
                "exact-horizon state-residual radius."
            ),
            "dynamic_student_t": {
                "seed": arguments.seed,
                "results": simulation,
            },
            "iid_student_t": {
                "seed": arguments.seed + 1,
                "config": asdict(iid_config),
                "results": iid_simulation,
            },
            "historical_replay": {
                "seed": arguments.seed + 2,
                "sampling": (
                    "Independent draws from centered pre-2025 fitted residuals; "
                    "no multi-week block is replayed."
                ),
                "results": replay_simulation,
            },
        },
    }
    write_json_atomic(report, arguments.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
