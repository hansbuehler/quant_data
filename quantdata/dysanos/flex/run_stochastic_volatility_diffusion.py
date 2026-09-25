"""Fit and audit a no-replay stochastic-volatility DYSANOS diffusion."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.run_score_driven_student_t import (
    _daily_law,
    _filter_order,
    _historical_residuals,
    _horizon_covariance,
    _load_dates,
    _load_model,
    _market_tail_diagnostics,
    _segment_starts,
    _summary,
)
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    filter_log_variance_diffusion,
    fit_log_variance_diffusion,
    log_chi_square_moments,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--step-sessions", type=int, choices=(1, 5), default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=72_401)
    return parser.parse_args()


def _likelihood_summary(
    normalized_radii: np.ndarray,
    starts: np.ndarray,
    validation: np.ndarray,
    *,
    dimension: int,
    config: LogVarianceDiffusionConfig,
) -> dict[str, float | int]:
    if normalized_radii.shape != starts.shape or validation.shape != starts.shape:
        raise ValueError("likelihood inputs must be aligned")
    if not np.any(validation):
        raise ValueError("validation mask is empty")
    dynamic = filter_log_variance_diffusion(
        normalized_radii,
        starts,
        dimension=dimension,
        config=config,
    )
    gaussian = filter_log_variance_diffusion(
        normalized_radii,
        starts,
        dimension=dimension,
        config=LogVarianceDiffusionConfig(
            persistence=0.0,
            stationary_log_variance=0.0,
            unconditional_scale=1.0,
        ),
    )
    measurement_log_mean = log_chi_square_moments(dimension)[0]
    scaled_gaussian_scale = math.exp(
        float(np.mean(np.log(normalized_radii[~validation])))
        - measurement_log_mean
    )
    scaled_gaussian = filter_log_variance_diffusion(
        normalized_radii,
        starts,
        dimension=dimension,
        config=LogVarianceDiffusionConfig(
            persistence=0.0,
            stationary_log_variance=0.0,
            unconditional_scale=scaled_gaussian_scale,
        ),
    )
    return {
        "observations": int(np.sum(validation)),
        "gaussian_log_radius_nll": float(
            -np.mean(gaussian.log_likelihood[validation])
        ),
        "scaled_gaussian_log_radius_nll": float(
            -np.mean(scaled_gaussian.log_likelihood[validation])
        ),
        "stochastic_volatility_log_radius_nll": float(
            -np.mean(dynamic.log_likelihood[validation])
        ),
        "scaled_gaussian_training_scale": scaled_gaussian_scale,
    }


def _simulate(
    *,
    config: LogVarianceDiffusionConfig,
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
    selected_device = torch.device(device)
    dtype = torch.float64
    transition = torch.as_tensor(step_transition, dtype=dtype, device=selected_device)
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
    horizon_factors = {
        week: torch.linalg.cholesky(
            torch.as_tensor(covariance, dtype=dtype, device=selected_device)
        )
        for week, covariance in horizon_covariances.items()
    }
    generator = torch.Generator(device=selected_device).manual_seed(seed)
    radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    weekly_radii: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    scale_samples: list[np.ndarray] = []
    sums = {
        week: torch.zeros(dimension, dtype=dtype, device=selected_device)
        for week in (1, 2, 3)
    }
    outer_sums = {
        week: torch.zeros((dimension, dimension), dtype=dtype, device=selected_device)
        for week in (1, 2, 3)
    }
    stationary_standard_deviation = math.sqrt(config.stationary_log_variance)
    innovation_standard_deviation = (
        config.log_variance_innovation_standard_deviation
    )
    completed = 0
    while completed < paths:
        current = min(batch_size, paths - completed)
        latent = stationary_standard_deviation * torch.randn(
            current,
            dtype=dtype,
            device=selected_device,
            generator=generator,
        )
        residual = torch.zeros((current, dimension), dtype=dtype, device=selected_device)
        weekly_residual = torch.zeros_like(residual)
        for step_index in range(1, 3 * steps_per_week + 1):
            scale = config.unconditional_scale * torch.exp(
                latent - 0.5 * config.stationary_log_variance
            )
            shock = torch.randn(
                (current, dimension),
                dtype=dtype,
                device=selected_device,
                generator=generator,
            )
            innovation = torch.sqrt(scale)[:, None] * (shock @ factor.T)
            residual = residual @ transition + innovation
            weekly_residual = weekly_residual @ transition + innovation
            latent = (
                config.persistence * latent
                + innovation_standard_deviation
                * torch.randn(
                    current,
                    dtype=dtype,
                    device=selected_device,
                    generator=generator,
                )
            )
            if step_index == 1 or step_index % steps_per_week == 0:
                scale_samples.append(scale.cpu().numpy())
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
        print(f"diffusion tail simulation paths={completed}/{paths}", flush=True)

    concatenated = {week: np.concatenate(values) for week, values in radii.items()}
    consecutive = {
        week: np.concatenate(values) for week, values in weekly_radii.items()
    }
    output: dict[str, object] = {}
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
    output["variance_multiplier"] = _summary(np.concatenate(scale_samples))
    return output


def main() -> None:
    arguments = _arguments()
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
    step_transition = np.linalg.matrix_power(daily_transition, arguments.step_sessions)
    step_covariance = _horizon_covariance(
        daily_transition,
        daily_covariance,
        steps=arguments.step_sessions,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(step_covariance)
    floor = 64.0 * np.finfo(np.float64).eps * max(float(np.max(eigenvalues)), 1.0)
    step_factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, floor)))
    origin, endpoint, _, squared_radii = _historical_residuals(
        states,
        dates,
        mean=mean,
        session_transition=daily_transition,
        session_covariance=daily_covariance,
        sessions=arguments.step_sessions,
    )
    normalized_radii = squared_radii / model.config.state_size
    pre_2025 = np.asarray([dates[index].year < 2025 for index in endpoint])
    fit_origin, fit_endpoint = origin[pre_2025], endpoint[pre_2025]
    fit_order = _filter_order(
        fit_origin,
        dates,
        sessions=arguments.step_sessions,
    )
    full_order = _filter_order(origin, dates, sessions=arguments.step_sessions)
    fit = fit_log_variance_diffusion(
        normalized_radii[pre_2025][fit_order],
        _segment_starts(fit_origin[fit_order], fit_endpoint[fit_order]),
        dimension=model.config.state_size,
    )
    validation_likelihood = _likelihood_summary(
        normalized_radii[full_order],
        _segment_starts(origin[full_order], endpoint[full_order]),
        (~pre_2025)[full_order],
        dimension=model.config.state_size,
        config=fit.config,
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
    )
    report: dict[str, object] = {
        "schema": "flex-log-variance-diffusion-v1",
        "status": "complete",
        "model": (
            "Gaussian affine-state innovations multiplied by a unit-mean "
            "latent log variance whose exact discrete transition is that of "
            "a stationary Ornstein-Uhlenbeck diffusion; no residual replay"
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
        },
        "fit": fit.to_dict(step_sessions=arguments.step_sessions),
        "validation_2025_likelihood": validation_likelihood,
        "market_tail_diagnostics": market_tails,
        "simulation": {
            "paths": arguments.paths,
            "batch_size": arguments.batch_size,
            "seed": arguments.seed,
            "tail_threshold_definition": (
                "Pre-2025 empirical 95th percentile of the normalized "
                "exact-horizon state-residual radius."
            ),
            "results": simulation,
        },
    }
    write_json_atomic(report, arguments.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
