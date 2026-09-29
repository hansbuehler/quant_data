"""Sample a fitted expiry-return OCE measure with sequential resampling."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.front_spot_copula import (
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.network_oce import l1_capped_action
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _Book,
    _book,
    _decode,
    _expiry_option_returns,
    _features,
    _price_calls,
)
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-returns-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-populations", type=int, default=16)
    parser.add_argument("--particles-per-population", type=int, default=8_192)
    parser.add_argument("--resample-ess-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2_026_100_4)
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.num_populations < 2:
        raise ValueError("at least two independent SMC populations are required")
    if arguments.particles_per_population <= 1:
        raise ValueError("each SMC population must contain at least two particles")
    if not 0.0 < arguments.resample_ess_fraction <= 1.0:
        raise ValueError("resampling ESS fraction must be in (0,1]")
    if arguments.sample_returns_output.suffix.lower() != ".npy":
        raise ValueError("sample returns output must use the .npy suffix")


def _load_network(
    path: Path,
    *,
    env: Any,
) -> tuple[FeedForward, Mapping[str, Any], torch.Tensor, torch.Tensor, _Book]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("schema") != (
        "dysanos-single-state-monthly-network-oce-v1"
    ):
        raise ValueError("OCE network checkpoint has an invalid schema")
    saved_arguments = saved.get("arguments")
    feature_blocks = saved.get("feature_blocks")
    if not isinstance(saved_arguments, Mapping) or not isinstance(feature_blocks, Mapping):
        raise ValueError("OCE network checkpoint is missing arguments or feature geometry")
    if saved_arguments.get("option_return_horizon") != "to-expiry":
        raise ValueError("SMC sampling currently requires a to-expiry OCE checkpoint")
    expiry_days = tuple(int(value) for value in saved_arguments["book_expiry_days"])
    book = _book(expiry_days)
    config = FeedForwardConfig(
        activation="SiLU",
        num_depth=int(saved_arguments["hidden_layers"]),
        num_width=int(saved_arguments["hidden_width"]),
    )
    network = FeedForward(
        env,
        int(feature_blocks["total"]),
        len(book.labels),
        config,
    )
    network.load_state_dict(saved["network_state_dict"])
    network.eval()
    feature_mean = saved["feature_mean"].to(dtype=env.fdtype, device=env.device)
    feature_scale = saved["feature_scale"].to(dtype=env.fdtype, device=env.device)
    return network, saved_arguments, feature_mean, feature_scale, book


def _systematic_resample(
    weights: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    particles = len(weights)
    offset = torch.rand((), dtype=weights.dtype, device=weights.device, generator=generator)
    positions = (torch.arange(particles, device=weights.device) + offset) / particles
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False)


def _resample_list(values: list[torch.Tensor], indices: torch.Tensor) -> list[torch.Tensor]:
    return [value[indices] for value in values]


def _sample_population(
    *,
    model: Any,
    network: FeedForward,
    book: _Book,
    initial_state: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    transaction_cost: torch.Tensor,
    action_l1_cap: float,
    action_ridge: float,
    num_steps: int,
    num_particles: int,
    resample_ess_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Draw one equal-weight particle population from the OCE path tilt."""

    dtype = initial_state.dtype
    device = initial_state.device
    dimension = initial_state.shape[0]
    states = initial_state[None].expand(num_particles, -1).clone()
    parameters, surface = _decode(model, states)
    path_spot = torch.ones(num_particles, dtype=dtype, device=device)
    log_weights = torch.zeros(num_particles, dtype=torch.float64, device=device)
    ancestry = torch.arange(num_particles, device=device)
    returns = torch.full(
        (num_particles, num_steps, len(book.labels)),
        torch.nan,
        dtype=dtype,
        device=device,
    )
    strikes = torch.tensor(book.relative_strikes, dtype=dtype, device=device)
    initial_days = torch.tensor(book.initial_days, dtype=dtype, device=device)
    is_put = torch.tensor(book.is_put, dtype=torch.bool, device=device)
    horizons = torch.repeat_interleave(
        torch.tensor(book.expiry_days, device=device), 3
    )
    equilibrium = model.equilibrium_mean.detach()
    entry_spots: list[torch.Tensor] = []
    entry_prices: list[torch.Tensor] = []
    entry_option_actions: list[torch.Tensor] = []
    resample_days: list[int] = []
    pre_resample_ess: list[float] = []
    log_normalizer = 0.0
    total_transitions = num_steps + max(book.expiry_days) - 1

    with torch.inference_mode():
        for day in range(total_transitions):
            action: torch.Tensor | None = None
            incremental_pnl = torch.zeros(
                num_particles, dtype=dtype, device=device
            )
            if day < num_steps:
                state_features = _features(model, states, parameters, surface)
                raw_action = network((state_features - feature_mean) / feature_scale)
                action = l1_capped_action(raw_action, action_l1_cap)
                issue_calls = _price_calls(
                    model,
                    surface,
                    days=initial_days,
                    strikes=strikes[None].expand(num_particles, -1),
                )
                issue_values = torch.where(
                    is_put[None], issue_calls - 1.0 + strikes[None], issue_calls
                )
                entry_spots.append(path_spot)
                entry_prices.append(issue_values)
                entry_option_actions.append(action[:, 1:])
                incremental_pnl -= torch.sum(
                    transaction_cost[None] * torch.abs(action), dim=1
                )
                incremental_pnl -= 0.5 * action_ridge * torch.sum(
                    action.square(), dim=1
                )

            state_normals = torch.randn(
                (num_particles, dimension),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            independent_spot = torch.randn(
                num_particles, dtype=dtype, device=device, generator=generator
            )
            expected = equilibrium + (states - equilibrium) @ daily_transition
            next_states = expected + state_normals @ daily_factor.T
            gaussian_score = (
                state_normals @ copula_loading
                + copula_orthogonal_std * independent_spot
            ) / copula_total_std
            gross_spot = front_week_sanos_inverse_cdf(surface, gaussian_score)
            next_path_spot = path_spot * gross_spot
            next_parameters, next_surface = _decode(model, next_states)

            if action is not None:
                spot_return = gross_spot - 1.0
                returns[:, day, 0] = spot_return
                incremental_pnl += action[:, 0] * spot_return

            maturity_day = day + 1
            for horizon in book.expiry_days:
                entry_day = maturity_day - horizon
                if not 0 <= entry_day < num_steps:
                    continue
                option_indices = torch.nonzero(horizons == horizon).squeeze(1)
                option_returns = _expiry_option_returns(
                    gross_horizon=next_path_spot / entry_spots[entry_day],
                    issue_values=entry_prices[entry_day][:, option_indices],
                    strikes=strikes[option_indices],
                    is_put=is_put[option_indices],
                )
                returns[:, entry_day, option_indices + 1] = option_returns
                incremental_pnl += torch.sum(
                    entry_option_actions[entry_day][:, option_indices] * option_returns,
                    dim=1,
                )

            log_weights -= incremental_pnl.double()
            normalized = torch.softmax(log_weights, dim=0)
            ess = 1.0 / torch.sum(normalized.square())
            final_day = day + 1 == total_transitions
            if float(ess) <= resample_ess_fraction * num_particles or final_day:
                log_normalizer += float(
                    torch.logsumexp(log_weights, dim=0) - math.log(num_particles)
                )
                indices = _systematic_resample(normalized, generator=generator)
                next_states = next_states[indices]
                next_path_spot = next_path_spot[indices]
                returns = returns[indices]
                ancestry = ancestry[indices]
                entry_spots = _resample_list(entry_spots, indices)
                entry_prices = _resample_list(entry_prices, indices)
                entry_option_actions = _resample_list(entry_option_actions, indices)
                next_parameters, next_surface = _decode(model, next_states)
                log_weights.zero_()
                resample_days.append(day + 1)
                pre_resample_ess.append(float(ess))

            states = next_states
            parameters = next_parameters
            surface = next_surface
            path_spot = next_path_spot

    if not bool(torch.all(torch.isfinite(returns))):
        raise FloatingPointError("SMC OCE sample contains an invalid return")
    unique_ancestors = int(torch.unique(ancestry).numel())
    descendant_count = torch.bincount(ancestry, minlength=num_particles)
    return returns, {
        "log_normalizer": log_normalizer,
        "num_resamples": len(resample_days),
        "resample_days": resample_days,
        "minimum_pre_resample_ess_fraction": min(pre_resample_ess) / num_particles,
        "unique_initial_ancestors": unique_ancestors,
        "unique_initial_ancestor_fraction": unique_ancestors / num_particles,
        "maximum_descendants_from_one_ancestor": int(torch.amax(descendant_count)),
    }


def _population_audit(
    population_means: np.ndarray,
    *,
    cost_band: np.ndarray,
) -> dict[str, Any]:
    mean = np.mean(population_means, axis=0)
    standard_error = np.std(population_means, axis=0, ddof=1) / math.sqrt(
        len(population_means)
    )
    violation = np.maximum(np.abs(mean) - cost_band, 0.0)
    violation_z = violation / np.maximum(standard_error, np.finfo(np.float64).tiny)
    return {
        "mean": mean.tolist(),
        "between_population_standard_error": standard_error.tolist(),
        "maximum_absolute_mean_bps": float(np.max(np.abs(mean)) * 1.0e4),
        "maximum_cost_band_violation_bps": float(np.max(violation) * 1.0e4),
        "root_mean_square_cost_band_violation_bps": float(
            np.sqrt(np.mean(np.square(violation))) * 1.0e4
        ),
        "fraction_outside_cost_band": float(np.mean(violation > 0.0)),
        "maximum_cost_band_violation_z": float(np.max(violation_z)),
    }


def run(arguments: argparse.Namespace) -> None:
    """Sample independent SMC populations and publish their equal-weight returns."""

    _validate_arguments(arguments)
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    network, saved_arguments, feature_mean, feature_scale, book = _load_network(
        arguments.network_checkpoint,
        env=env,
    )
    payload = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = payload.get("dates")
    states_value = payload.get("states")
    spots_value = payload.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("target history is missing dates, states, or spot references")
    state_index_value = int(saved_arguments["state_index"])
    state_index = len(dates) // 2 if state_index_value < 0 else state_index_value
    historical_states = states_value.to(dtype=env.fdtype, device=env.device)
    initial_state = historical_states[state_index]
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit_before = date.fromisoformat(str(saved_arguments["copula_fit_before"]))
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states_value.double().numpy(),
        spots=spots_value.double().numpy(),
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=copula_fit_before,
    )
    copula_loading = torch.as_tensor(
        copula_fit.state_loading, dtype=env.fdtype, device=env.device
    )
    transaction_cost = torch.tensor(
        (
            float(saved_arguments["spot_cost_bps"]),
            *(
                float(saved_arguments["option_cost_bps"])
                for _ in book.relative_strikes
            ),
        ),
        dtype=env.fdtype,
        device=env.device,
    ) / 1.0e4
    num_steps = int(saved_arguments["num_steps"])
    total_paths = arguments.num_populations * arguments.particles_per_population
    coordinates = num_steps * len(book.labels)
    arguments.sample_returns_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.sample_returns_output.with_name(
        f".{arguments.sample_returns_output.name}.tmp"
    )
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(total_paths, num_steps, len(book.labels)),
    )
    population_means = np.empty(
        (arguments.num_populations, coordinates), dtype=np.float64
    )
    population_diagnostics: list[dict[str, Any]] = []
    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
    for population in range(arguments.num_populations):
        returns, diagnostics = _sample_population(
            model=model,
            network=network,
            book=book,
            initial_state=initial_state,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=copula_loading,
            copula_orthogonal_std=copula_fit.orthogonal_std,
            copula_total_std=copula_fit.total_std,
            transaction_cost=transaction_cost,
            action_l1_cap=float(saved_arguments["action_l1_cap"]),
            action_ridge=float(saved_arguments["action_ridge"]),
            num_steps=num_steps,
            num_particles=arguments.particles_per_population,
            resample_ess_fraction=arguments.resample_ess_fraction,
            generator=generator,
        )
        values = returns.cpu().numpy()
        first = population * arguments.particles_per_population
        output[first : first + arguments.particles_per_population] = values
        population_means[population] = np.mean(
            values.reshape(arguments.particles_per_population, -1),
            axis=0,
            dtype=np.float64,
        )
        diagnostics["population"] = population
        population_diagnostics.append(diagnostics)
        print(json.dumps(diagnostics), flush=True)
    output.flush()
    del output
    os.replace(temporary, arguments.sample_returns_output)

    one_step_cost = np.asarray(
        (
            float(saved_arguments["spot_cost_bps"]),
            *(
                float(saved_arguments["option_cost_bps"])
                for _ in book.relative_strikes
            ),
        )
    ) / 1.0e4
    report = {
        "schema": "dysanos-oce-sequential-monte-carlo-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "target": "terminal OCE density exp(-net path PnL)",
            "algorithm": (
                "bootstrap Feynman-Kac particle sampler with incremental realized-"
                "PnL potentials and adaptive systematic resampling"
            ),
            "weights_after_final_resampling": "equal",
            "uncertainty": (
                "standard errors use independent particle-population means; ancestry "
                "diagnostics expose path degeneracy hidden by equal final weights"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "state_date": str(dates[state_index]),
        "book_labels": list(book.labels),
        "num_steps": num_steps,
        "total_direct_samples": total_paths,
        "transition_diagnostics": transition_diagnostics,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "population_diagnostics": population_diagnostics,
        "log_normalizer_mean": float(
            np.mean([value["log_normalizer"] for value in population_diagnostics])
        ),
        "mean_unique_initial_ancestor_fraction": float(
            np.mean(
                [
                    value["unique_initial_ancestor_fraction"]
                    for value in population_diagnostics
                ]
            )
        ),
        "return_audit": _population_audit(
            population_means,
            cost_band=np.tile(one_step_cost, num_steps),
        ),
    }
    write_json_atomic(report, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
