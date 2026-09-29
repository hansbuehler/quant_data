"""Learn a directly sampleable Gaussian-noise law for expiry-return moments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import chi2

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.driving_noise_measure import (
    DrivingNoiseTransport,
    DrivingNoiseTransportConfig,
)
from quantdata.dysanos.flex.front_spot_copula import (
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _Book,
    _book,
    _expiry_option_returns,
)
from quantdata.dysanos.flex.weekly_simulation import front_week_sanos_inverse_cdf
from quantdata.torch.ffw import FeedForwardConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--book-expiry-days", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--audit-paths", type=int, default=100_000)
    parser.add_argument("--audit-batch-size", type=int, default=2_048)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=25)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--kl-weight", type=float, default=1.0e-4)
    parser.add_argument("--minimum-return-rms", type=float, default=1.0e-3)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2_026_100_5)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    _book(tuple(arguments.book_expiry_days))
    positive_integers = (
        arguments.num_steps,
        arguments.epochs,
        arguments.batch_size,
        arguments.audit_paths,
        arguments.audit_batch_size,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.checkpoint_every,
    )
    if min(positive_integers) <= 0:
        raise ValueError("direct-measure integer controls must be positive")
    positive_values = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.gradient_clip,
        arguments.minimum_return_rms,
        arguments.spot_cost_bps,
        arguments.option_cost_bps,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("direct-measure scalar controls must be finite-positive")
    if not math.isfinite(arguments.kl_weight) or arguments.kl_weight < 0.0:
        raise ValueError("direct-measure KL weight must be finite-nonnegative")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds initial learning rate")
    if not 0 <= arguments.warmup_epochs < arguments.epochs:
        raise ValueError("direct-measure warmup is outside the training schedule")


def _decode_differentiable(model: Any, states: torch.Tensor) -> tuple[Any, Any]:
    raw = model.raw_states_from_states(states)
    parameters = bound_native_raw(raw, geometry=model.config.geometry)
    surface = model.sanos_decoder.forward_batch(
        dlvsqs=parameters.dlvsqs,
        forward_vols=parameters.forward_vols,
        eta=parameters.eta,
    )
    return parameters, surface


def _price_calls(
    model: Any,
    surface: Any,
    *,
    days: torch.Tensor,
    strikes: torch.Tensor,
) -> torch.Tensor:
    return model.sanos_decoder.price_calls_batch(
        surface,
        expiries_y=days[None].expand(len(surface.densities), -1) / DAYS_PER_YEAR,
        strikes=strikes,
    )


def _simulate_returns(
    *,
    model: Any,
    transport: DrivingNoiseTransport,
    book: _Book,
    initial_state: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    num_steps: int,
    num_paths: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Draw expiry returns differentiably under the transported noise law."""

    dtype = initial_state.dtype
    device = initial_state.device
    state_size = initial_state.shape[0]
    states = initial_state[None].expand(num_paths, -1).clone()
    _, surface = _decode_differentiable(model, states)
    path_spot = torch.ones(num_paths, dtype=dtype, device=device)
    returns = torch.full(
        (num_paths, num_steps, len(book.labels)),
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
    total_transitions = num_steps + max(book.expiry_days) - 1
    path_kl = torch.zeros(num_paths, dtype=dtype, device=device)
    shift_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    log_scale_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    noise_count = 0

    for day in range(total_transitions):
        if day < num_steps:
            issue_calls = _price_calls(
                model,
                surface,
                days=initial_days,
                strikes=strikes[None].expand(num_paths, -1),
            )
            issue_values = torch.where(
                is_put[None], issue_calls - 1.0 + strikes[None], issue_calls
            )
            entry_spots.append(path_spot)
            entry_prices.append(issue_values)

        base_normals = torch.randn(
            (num_paths, state_size + 1),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        normalized_time = torch.full(
            (num_paths,),
            day / max(total_transitions - 1, 1),
            dtype=dtype,
            device=device,
        )
        transformed, conditional_kl, shift, log_scale = transport(
            states.detach(),
            normalized_time,
            base_normals,
        )
        state_normals = transformed[:, :state_size]
        independent_spot = transformed[:, state_size]
        expected = equilibrium + (states - equilibrium) @ daily_transition
        next_states = expected + state_normals @ daily_factor.T
        gaussian_score = (
            state_normals @ copula_loading
            + copula_orthogonal_std * independent_spot
        ) / copula_total_std
        gross_spot = front_week_sanos_inverse_cdf(surface, gaussian_score)
        next_path_spot = path_spot * gross_spot
        _, next_surface = _decode_differentiable(model, next_states)
        path_kl = path_kl + conditional_kl
        shift_square_sum += torch.sum(shift.detach().double().square())
        log_scale_square_sum += torch.sum(log_scale.detach().double().square())
        noise_count += shift.numel()

        if day < num_steps:
            returns[:, day, 0] = gross_spot - 1.0
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

        states = next_states
        surface = next_surface
        path_spot = next_path_spot

    if not bool(torch.all(torch.isfinite(returns))):
        raise FloatingPointError("direct driving-noise measure produced invalid returns")
    normalized_kl = path_kl / (total_transitions * (state_size + 1))
    return returns, normalized_kl, {
        "shift_rms": torch.sqrt(shift_square_sum / noise_count),
        "log_scale_rms": torch.sqrt(log_scale_square_sum / noise_count),
    }


def _learning_rate(arguments: argparse.Namespace, epoch: int) -> float:
    if epoch < arguments.warmup_epochs:
        return arguments.learning_rate * (epoch + 1) / max(arguments.warmup_epochs, 1)
    progress = (epoch - arguments.warmup_epochs) / max(
        arguments.epochs - arguments.warmup_epochs - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return arguments.minimum_learning_rate + (
        arguments.learning_rate - arguments.minimum_learning_rate
    ) * cosine


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    transport: DrivingNoiseTransport,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    completed_epochs: int,
    history: list[dict[str, float | int]],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": "dysanos-direct-driving-noise-measure-v1",
        "phase": "training" if completed_epochs < arguments.epochs else "complete",
        "completed_epochs": completed_epochs,
        "transport_config": transport.config,
        "transport_state_dict": copy.deepcopy(transport.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "generator_state": generator.get_state().cpu(),
        "history": history,
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
    }


def _cost_band(arguments: argparse.Namespace, book: _Book) -> torch.Tensor:
    one_step = torch.tensor(
        (
            arguments.spot_cost_bps,
            *(arguments.option_cost_bps for _ in book.relative_strikes),
        ),
        dtype=torch.float64,
    ) / 1.0e4
    return one_step.repeat(arguments.num_steps)


def run(arguments: argparse.Namespace) -> None:
    """Train and independently audit a direct equal-weight path sampler."""

    _validate_arguments(arguments)
    book = _book(tuple(arguments.book_expiry_days))
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    state_index = len(dates) // 2 if arguments.state_index < 0 else arguments.state_index
    historical_states = states_value.to(dtype=env.fdtype, device=env.device)
    initial_state = historical_states[state_index]
    state_mean = torch.mean(historical_states, dim=0)
    state_std = torch.std(historical_states, dim=0, correction=0)
    state_scale = torch.where(state_std > 1.0e-5, state_std, torch.ones_like(state_std))
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states_value.double().numpy(),
        spots=spots_value.double().numpy(),
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    copula_loading = torch.as_tensor(
        copula_fit.state_loading, dtype=env.fdtype, device=env.device
    )
    config = DrivingNoiseTransportConfig(
        state_size=len(initial_state),
        network=FeedForwardConfig(
            activation="SiLU",
            num_depth=arguments.hidden_layers,
            num_width=arguments.hidden_width,
        ),
    )
    transport = DrivingNoiseTransport(
        env,
        config=config,
        state_mean=state_mean,
        state_scale=state_scale,
    )
    optimizer = torch.optim.Adam(transport.parameters(), lr=arguments.learning_rate)
    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    if arguments.resume:
        saved = torch.load(arguments.network_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping) or saved.get("schema") != (
            "dysanos-direct-driving-noise-measure-v1"
        ):
            raise ValueError("direct-measure resume checkpoint has an invalid schema")
        transport.load_state_dict(saved["transport_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"].cpu())
        start_epoch = int(saved["completed_epochs"])
        history = list(saved.get("history", []))
    cost = _cost_band(arguments, book).to(device=env.device)
    parameter_count = sum(value.numel() for value in transport.parameters())

    for epoch in range(start_epoch, arguments.epochs):
        learning_rate = _learning_rate(arguments, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        returns, normalized_kl, diagnostics = _simulate_returns(
            model=model,
            transport=transport,
            book=book,
            initial_state=initial_state,
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=copula_loading,
            copula_orthogonal_std=copula_fit.orthogonal_std,
            copula_total_std=copula_fit.total_std,
            num_steps=arguments.num_steps,
            num_paths=arguments.batch_size,
            generator=generator,
        )
        flat = returns.reshape(arguments.batch_size, -1)
        mean = torch.mean(flat, dim=0)
        return_rms = torch.sqrt(torch.mean(flat.detach().square(), dim=0)).clamp_min(
            arguments.minimum_return_rms
        )
        violation = torch.relu(torch.abs(mean) - cost.to(dtype=mean.dtype))
        moment_loss = torch.mean((violation / return_rms).square())
        kl_loss = torch.mean(normalized_kl)
        loss = moment_loss + arguments.kl_weight * kl_loss
        loss.backward()
        gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
        for parameter in transport.parameters():
            if parameter.grad is not None:
                gradient_square += torch.sum(parameter.grad.double().square())
        gradient_rms = torch.sqrt(gradient_square / parameter_count)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            transport.parameters(), arguments.gradient_clip
        )
        optimizer.step()
        if epoch % 10 == 0 or epoch + 1 == arguments.epochs:
            record = {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "moment_loss": float(moment_loss.detach()),
                "normalized_kl": float(kl_loss.detach()),
                "learning_rate": learning_rate,
                "gradient_rms": float(gradient_rms),
                "gradient_norm_before_clip": float(gradient_norm),
                "maximum_batch_violation_bps": float(torch.amax(violation.detach()))
                * 1.0e4,
                "fraction_batch_coordinates_outside_band": float(
                    torch.mean((violation.detach() > 0.0).float())
                ),
                "shift_rms": float(diagnostics["shift_rms"]),
                "log_scale_rms": float(diagnostics["log_scale_rms"]),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if (epoch + 1) % arguments.checkpoint_every == 0 or epoch + 1 == arguments.epochs:
            _atomic_torch_save(
                _checkpoint_payload(
                    transport=transport,
                    optimizer=optimizer,
                    generator=generator,
                    completed_epochs=epoch + 1,
                    history=history,
                    arguments=arguments,
                ),
                arguments.network_checkpoint,
            )

    transport.eval()
    coordinates = arguments.num_steps * len(book.labels)
    audit_sum = torch.zeros(coordinates, dtype=torch.float64, device=env.device)
    audit_cross = torch.zeros(
        (coordinates, coordinates), dtype=torch.float64, device=env.device
    )
    audit_kl_sum = torch.zeros((), dtype=torch.float64, device=env.device)
    audit_generator = torch.Generator(device=env.device).manual_seed(arguments.seed + 1)
    with torch.inference_mode():
        for first in range(0, arguments.audit_paths, arguments.audit_batch_size):
            batch_paths = min(arguments.audit_batch_size, arguments.audit_paths - first)
            returns, normalized_kl, _ = _simulate_returns(
                model=model,
                transport=transport,
                book=book,
                initial_state=initial_state,
                daily_transition=daily_transition,
                daily_factor=daily_factor,
                copula_loading=copula_loading,
                copula_orthogonal_std=copula_fit.orthogonal_std,
                copula_total_std=copula_fit.total_std,
                num_steps=arguments.num_steps,
                num_paths=batch_paths,
                generator=audit_generator,
            )
            flat = returns.reshape(batch_paths, coordinates).double()
            audit_sum += torch.sum(flat, dim=0)
            audit_cross += flat.T @ flat
            audit_kl_sum += torch.sum(normalized_kl.double())
            if first % (25 * arguments.audit_batch_size) == 0:
                print(f"audit {first + batch_paths}/{arguments.audit_paths}", flush=True)
    mean = audit_sum / arguments.audit_paths
    sample_covariance = (
        audit_cross - arguments.audit_paths * torch.outer(mean, mean)
    ) / (arguments.audit_paths - 1)
    covariance_of_mean = sample_covariance / arguments.audit_paths
    standard_error = torch.sqrt(torch.clamp_min(torch.diag(covariance_of_mean), 0.0))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_of_mean)
    tolerance = (
        torch.finfo(torch.float64).eps
        * torch.amax(torch.abs(eigenvalues)).clamp_min(torch.finfo(torch.float64).tiny)
        * coordinates
    )
    retained = eigenvalues > tolerance
    projected = eigenvectors[:, retained].T @ mean
    statistic = torch.sum(projected.square() / eigenvalues[retained])
    degrees = int(torch.sum(retained))
    violation = torch.relu(torch.abs(mean) - cost)
    violation_z = violation / standard_error.clamp_min(torch.finfo(torch.float64).tiny)
    report = {
        "schema": "dysanos-direct-driving-noise-measure-report-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "measure": (
                "equal-weight direct samples from a state/time-conditional diagonal-"
                "Gaussian transport of all 200 state shocks and the orthogonal spot shock"
            ),
            "objective": (
                "mean squared cost-band excess standardized by detached return RMS, "
                "plus normalized path KL to P"
            ),
            "scope": (
                "all unconditional time/instrument moments; a fresh state-dependent OCE "
                "adversary is required for conditional no-arbitrage"
            ),
            "book_labels": list(book.labels),
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "state_date": str(dates[state_index]),
        "transition_diagnostics": transition_diagnostics,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "training_history": history,
        "audit": {
            "num_paths": arguments.audit_paths,
            "normalized_path_kl": float(audit_kl_sum / arguments.audit_paths),
            "mean": mean.cpu().tolist(),
            "standard_error": standard_error.cpu().tolist(),
            "maximum_absolute_mean_bps": float(torch.amax(torch.abs(mean))) * 1.0e4,
            "maximum_cost_band_violation_bps": float(torch.amax(violation)) * 1.0e4,
            "root_mean_square_cost_band_violation_bps": float(
                torch.sqrt(torch.mean(violation.square()))
            )
            * 1.0e4,
            "fraction_outside_cost_band": float(torch.mean((violation > 0.0).double())),
            "maximum_cost_band_violation_z": float(torch.amax(violation_z)),
            "joint_zero_statistic": float(statistic),
            "joint_zero_degrees_of_freedom": degrees,
            "joint_zero_p_value": float(chi2.sf(float(statistic), degrees)),
        },
    }
    write_json_atomic(report, arguments.output)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
