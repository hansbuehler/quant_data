"""Compare held-out observed and generated DYSANOS surface paths."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.path_realism import paired_block_distribution_test
from quantdata.dysanos.flex.presentation_hedging import (
    _decode_surface,
    load_presentation_model,
)
from quantdata.dysanos.flex.score_driven_student_t import (
    ScoreDrivenStudentTConfig,
    filter_score_driven_student_t,
)
from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    filter_log_variance_diffusion,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--student-t-report", type=Path, required=True)
    parser.add_argument("--diffusion-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=91_337)
    parser.add_argument("--permutations", type=int, default=1_999)
    return parser.parse_args()


def _weekly_chains(dates: list[str], *, weeks: int) -> np.ndarray:
    parsed = [date.fromisoformat(value) for value in dates]
    index = {value: position for position, value in enumerate(parsed)}
    calendar = get_business_days(parsed[0], parsed[-1], exclude_half_days=True)
    calendar_position = {value: position for position, value in enumerate(calendar)}
    chains: list[list[int]] = []
    for origin, value in enumerate(parsed):
        position = calendar_position[value]
        future = [position + 5 * step for step in range(1, weeks + 1)]
        if future[-1] >= len(calendar):
            continue
        endpoint_dates = [calendar[item] for item in future]
        if all(item in index for item in endpoint_dates):
            chains.append([origin, *(index[item] for item in endpoint_dates)])
    return np.asarray(chains, dtype=np.int64)


def _student_t_shock(
    generator: np.random.Generator,
    shape: tuple[int, int],
    degrees_of_freedom: float,
) -> tuple[np.ndarray, np.ndarray]:
    gaussian = generator.normal(size=shape)
    radial_scale = generator.chisquare(degrees_of_freedom, size=(shape[0], 1))
    shock = gaussian * np.sqrt((degrees_of_freedom - 2.0) / radial_scale)
    return shock, np.sum(np.square(shock), axis=1)


def _surface_features(
    model: Any,
    state_paths: np.ndarray,
    *,
    expiry_days: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    path_count, path_points, dimension = state_paths.shape
    flattened = torch.as_tensor(
        state_paths.reshape(-1, dimension),
        dtype=model.fdtype,
        device=model.device,
    )
    expiry_grid, strike_grid = np.meshgrid(expiry_days, strikes, indexing="ij")
    expiries = torch.as_tensor(
        expiry_grid.reshape(-1) / DAYS_PER_YEAR,
        dtype=model.fdtype,
        device=model.device,
    )
    strike_values = torch.as_tensor(
        strike_grid.reshape(-1), dtype=model.fdtype, device=model.device
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(flattened), 512):
            states = flattened[first : first + 512]
            surface = _decode_surface(model, states)
            calls = model.sanos_decoder.price_calls_batch(
                surface,
                expiries_y=expiries.unsqueeze(0).expand(len(states), -1),
                strikes=strike_values.unsqueeze(0).expand(len(states), -1),
            )
            chunks.append(calls.cpu().double().numpy())
    panels = np.concatenate(chunks).reshape(path_count, path_points, -1)
    return np.diff(panels, axis=1).reshape(path_count, -1)


def main() -> None:
    arguments = _arguments()
    _, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_tensor = history.get("states")
    if not isinstance(dates, list) or not isinstance(states_tensor, torch.Tensor):
        raise ValueError("invalid target-history payload")
    states = states_tensor.double().numpy()
    report = json.loads(arguments.student_t_report.read_text())
    fit_payload = report.get("fit")
    if not isinstance(fit_payload, dict) or not isinstance(fit_payload.get("config"), dict):
        raise ValueError("Student-t report has no fitted configuration")
    dynamic_config = ScoreDrivenStudentTConfig(**fit_payload["config"])
    iid_nu = float(fit_payload["iid_student_t_degrees_of_freedom"])
    iid_scale = float(fit_payload["iid_student_t_unconditional_scale"])
    log_normalizer = float(fit_payload["log_mean_normalizer"])
    diffusion_report = json.loads(arguments.diffusion_report.read_text())
    if diffusion_report.get("schema") != "flex-log-variance-diffusion-v1":
        raise ValueError("diffusion report has an unexpected schema")
    diffusion_fit = diffusion_report.get("fit")
    if not isinstance(diffusion_fit, dict) or not isinstance(
        diffusion_fit.get("config"), dict
    ):
        raise ValueError("diffusion report has no fitted configuration")
    diffusion_config = LogVarianceDiffusionConfig(**diffusion_fit["config"])
    with torch.no_grad():
        transition = model.weekly_transition().cpu().double().numpy()
        factor = model.innovation_cholesky().cpu().double().numpy()
        mean = model.equilibrium_mean.cpu().double().numpy()
    covariance = factor @ factor.T
    inverse_factor = np.linalg.inv(factor)
    one_week = _weekly_chains(dates, weeks=1)
    origin, endpoint = one_week[:, 0], one_week[:, 1]
    residuals = states[endpoint] - (mean + (states[origin] - mean) @ transition)
    whitened = residuals @ inverse_factor.T
    radii = np.sum(np.square(whitened), axis=1)

    parsed = [date.fromisoformat(value) for value in dates]
    calendar = get_business_days(parsed[0], parsed[-1], exclude_half_days=True)
    calendar_position = {value: position for position, value in enumerate(calendar)}
    order = np.asarray(
        sorted(
            range(len(origin)),
            key=lambda item: (
                calendar_position[parsed[int(origin[item])]] % 5,
                calendar_position[parsed[int(origin[item])]],
            ),
        )
    )
    ordered_origin, ordered_endpoint = origin[order], endpoint[order]
    starts = np.ones(len(order), dtype=np.bool_)
    starts[1:] = ordered_origin[1:] != ordered_endpoint[:-1]
    filtered = filter_score_driven_student_t(
        radii[order], starts, dimension=model.config.state_size, config=dynamic_config
    )
    diffusion_filtered = filter_log_variance_diffusion(
        radii[order] / model.config.state_size,
        starts,
        dimension=model.config.state_size,
        config=diffusion_config,
    )
    next_log_scale: dict[int, float] = {}
    next_diffusion_mean: dict[int, float] = {}
    next_diffusion_variance: dict[int, float] = {}
    for position, item in enumerate(order):
        current_log_state = (
            math.log(filtered.scales[position] / dynamic_config.unconditional_scale)
            + log_normalizer
        )
        next_log_scale[int(endpoint[item])] = (
            dynamic_config.persistence * current_log_state
            + dynamic_config.score_loading * filtered.scores[position]
        )
        next_diffusion_mean[int(endpoint[item])] = float(
            diffusion_filtered.next_predicted_means[position]
        )
        next_diffusion_variance[int(endpoint[item])] = float(
            diffusion_filtered.next_predicted_variances[position]
        )

    chains = _weekly_chains(dates, weeks=3)
    validation = np.asarray([parsed[int(item[0])].year >= 2025 for item in chains])
    validation_chains = chains[validation]
    training_chains = chains[np.asarray([parsed[int(item[-1])].year < 2025 for item in chains])]
    if len(validation_chains) < 10 or len(training_chains) < len(validation_chains):
        raise ValueError("insufficient train/validation three-week surface paths")
    observed = states[validation_chains]
    generator = np.random.default_rng(arguments.seed)
    path_count = len(validation_chains)
    dimension = model.config.state_size

    generated: dict[str, np.ndarray] = {}
    for law in (
        "gaussian",
        "iid_student_t",
        "dynamic_student_t",
        "stochastic_volatility_diffusion",
    ):
        paths = np.empty_like(observed)
        paths[:, 0] = observed[:, 0]
        log_scale = np.asarray(
            [next_log_scale.get(int(item[0]), 0.0) for item in validation_chains]
        )
        diffusion_mean = np.asarray(
            [next_diffusion_mean.get(int(item[0]), 0.0) for item in validation_chains]
        )
        diffusion_variance = np.asarray(
            [
                next_diffusion_variance.get(
                    int(item[0]), diffusion_config.stationary_log_variance
                )
                for item in validation_chains
            ]
        )
        latent_log_variance = diffusion_mean + np.sqrt(diffusion_variance) * (
            generator.normal(size=path_count)
        )
        for step in range(1, 4):
            if law == "gaussian":
                shock = generator.normal(size=(path_count, dimension))
                scale = np.ones(path_count)
                radius = np.sum(np.square(shock), axis=1)
            elif law in {"iid_student_t", "dynamic_student_t"}:
                nu = iid_nu if law == "iid_student_t" else dynamic_config.degrees_of_freedom
                shock, radius = _student_t_shock(generator, (path_count, dimension), nu)
                scale = (
                    np.full(path_count, iid_scale)
                    if law == "iid_student_t"
                    else dynamic_config.unconditional_scale * np.exp(log_scale - log_normalizer)
                )
            else:
                shock = generator.normal(size=(path_count, dimension))
                scale = diffusion_config.unconditional_scale * np.exp(
                    latent_log_variance
                    - 0.5 * diffusion_config.stationary_log_variance
                )
                radius = np.sum(np.square(shock), axis=1)
            innovation = np.sqrt(scale)[:, None] * (shock @ factor.T)
            paths[:, step] = mean + (paths[:, step - 1] - mean) @ transition + innovation
            if law == "dynamic_student_t":
                standardized_radius = radius
                weight = standardized_radius / (
                    dynamic_config.degrees_of_freedom - 2.0 + standardized_radius
                )
                score = 0.5 * ((dynamic_config.degrees_of_freedom + dimension) * weight - dimension)
                log_scale = (
                    dynamic_config.persistence * log_scale + dynamic_config.score_loading * score
                )
            elif law == "stochastic_volatility_diffusion":
                latent_log_variance = (
                    diffusion_config.persistence * latent_log_variance
                    + diffusion_config.log_variance_innovation_standard_deviation
                    * generator.normal(size=path_count)
                )
        generated[law] = paths

    replay_choice = generator.choice(len(training_chains), size=path_count, replace=False)
    replay_paths = np.empty_like(observed)
    replay_paths[:, 0] = observed[:, 0]
    residual_map = {int(left): residuals[index] for index, left in enumerate(origin)}
    for row, training_index in enumerate(replay_choice):
        source = training_chains[training_index]
        for step in range(1, 4):
            innovation = residual_map[int(source[step - 1])]
            replay_paths[row, step] = (
                mean + (replay_paths[row, step - 1] - mean) @ transition + innovation
            )
    generated["historical_replay"] = replay_paths

    model_days = np.asarray(model.config.geometry.model_expiries_y) * DAYS_PER_YEAR
    targets = np.asarray((1.0, 5.0, 10.0, 20.0, 40.0, 70.0))
    expiry_days = np.unique(
        model_days[np.argmin(np.abs(model_days[:, None] - targets[None, :]), axis=0)]
    )
    strikes = np.asarray((0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15))
    training_features = _surface_features(
        model,
        states[training_chains],
        expiry_days=expiry_days,
        strikes=strikes,
    )
    center = np.mean(training_features, axis=0)
    scale = np.std(training_features, axis=0, ddof=1)
    keep = scale > 64.0 * np.finfo(np.float64).eps
    if not np.any(keep):
        raise ValueError("training surface paths have no non-constant feature")
    observed_features = _surface_features(
        model,
        observed,
        expiry_days=expiry_days,
        strikes=strikes,
    )
    observed_features = (observed_features[:, keep] - center[keep]) / scale[keep]
    results: dict[str, object] = {}
    for name, paths in generated.items():
        features = _surface_features(model, paths, expiry_days=expiry_days, strikes=strikes)
        features = (features[:, keep] - center[keep]) / scale[keep]
        results[name] = paired_block_distribution_test(
            observed_features,
            features,
            # Each feature vector spans three weekly transitions.  Fifteen
            # adjacent origins retain the corresponding overlap dependence
            # much better than a one-week block.
            block_length=15,
            permutations=arguments.permutations,
            seed=arguments.seed,
            prestandardized=True,
        )
    output = {
        "schema": "dysanos-surface-path-realism-v1",
        "status": "complete",
        "checkpoint": str(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "student_t_report": str(arguments.student_t_report),
        "diffusion_report": str(arguments.diffusion_report),
        "evaluation_period": "2025",
        "holdout_scope": (
            "The Student-t and stochastic-volatility tail laws, historical-replay "
            "pool, and feature standardization use pre-2025 observations only. "
            "The shared affine decoder and Gaussian state transition come from "
            "the all-history checkpoint, so this is a chronological holdout of "
            "the tail-law extensions rather than a fully frozen pre-2025 model "
            "forecast."
        ),
        "path_horizon": (
            "Three consecutive one-week transitions per path; conditioning "
            "origins may overlap across paths."
        ),
        "surface_panel": {
            "expiry_business_days": expiry_days.tolist(),
            "spot_relative_strikes": strikes.tolist(),
            "feature": "three consecutive changes in normalized call prices",
        },
        "test": (
            "Energy distance and unbiased RBF-MMD after one common coordinate "
            "standardization estimated exclusively from pre-2025 observed paths; "
            "paired date blocks are label-swapped for the null distribution."
        ),
        "results": results,
    }
    write_json_atomic(output, arguments.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
