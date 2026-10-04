"""Fit a capped state-dependent OCE action for a seven-instrument daily book."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch

from quantdata.dysanos.flex.front_spot_copula import (
    _front_quantile_spline,
    fit_front_spot_copula_from_history,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.network_oce import (
    conditional_entropic_oce,
    l1_capped_action,
)
from quantdata.dysanos.flex.presentation_hedging import (
    _call_prices,
    _decode_surface,
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_few_instrument_oce import (
    _aggregate_audits,
    _paired_audit,
)
from quantdata.torch.ffw import FeedForward, FeedForwardConfig

_RELATIVE_STRIKES = (0.99, 1.0, 1.01, 0.99, 1.0, 1.01)
_INITIAL_DAYS = (5.0, 5.0, 5.0, 10.0, 10.0, 10.0)
_FINAL_DAYS = (4.0, 4.0, 4.0, 9.0, 9.0, 9.0)
_IS_PUT = (False, False, True, False, False, True)
_LABELS = (
    "spot",
    "5D 99% ITM call",
    "5D ATM call",
    "5D 101% ITM put",
    "10D 99% ITM call",
    "10D ATM call",
    "10D 101% ITM put",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-paths-per-origin", type=int, default=4_096)
    parser.add_argument("--generation-origin-batch-size", type=int, default=8)
    parser.add_argument("--validation-origins", type=int, default=21)
    parser.add_argument("--validation-paths-per-origin", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--action-l1-cap", type=float, default=16.0)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-ridge", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=2_026_092_8)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    integers = (
        arguments.training_paths_per_origin,
        arguments.generation_origin_batch_size,
        arguments.validation_origins,
        arguments.validation_paths_per_origin,
        arguments.epochs,
        arguments.hidden_width,
        arguments.hidden_layers,
    )
    if min(integers) <= 0:
        raise ValueError("network OCE integer controls must be positive")
    if arguments.training_paths_per_origin % 2 or arguments.validation_paths_per_origin % 2:
        raise ValueError("antithetic path counts must be even")
    positive = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.action_l1_cap,
        arguments.spot_cost_bps,
        arguments.option_cost_bps,
        arguments.gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("network OCE scalar controls must be finite-positive")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds the initial rate")
    if not math.isfinite(arguments.action_ridge) or arguments.action_ridge < 0.0:
        raise ValueError("action ridge must be finite and nonnegative")


def _state_features(model: Any, states: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """Return ``(h, sqrt(q), log(DLVsq-floor))`` for every origin."""

    with torch.inference_mode():
        raw = model.raw_states_from_states(states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        density = torch.sqrt(torch.clamp_min(surface.densities, 0.0)).reshape(len(states), -1)
        excess_dlv = torch.clamp_min(
            parameters.dlvsqs - model.config.geometry.min_dlvsq,
            torch.finfo(states.dtype).tiny,
        )
        log_dlv = torch.log(excess_dlv).reshape(len(states), -1)
        features = torch.cat((states, density, log_dlv), dim=1)
    if not bool(torch.all(torch.isfinite(features))):
        raise FloatingPointError("OCE state features are non-finite")
    return features, {
        "h": states.shape[1],
        "sqrt_density": density.shape[1],
        "log_dlvsq_excess": log_dlv.shape[1],
        "total": features.shape[1],
    }


def _draw_book_returns(
    *,
    model: Any,
    current_states: torch.Tensor,
    daily_transition: torch.Tensor,
    daily_factor: torch.Tensor,
    copula_loading: torch.Tensor,
    copula_orthogonal_std: float,
    copula_total_std: float,
    num_paths: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate price returns for the fixed-cash ITM/ATM trading book."""

    origins, dimension = current_states.shape
    dtype = current_states.dtype
    device = current_states.device
    half_state = torch.randn(
        (origins, num_paths // 2, dimension),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    half_spot = torch.randn(
        (origins, num_paths // 2), dtype=dtype, device=device, generator=generator
    )
    state_normals = torch.cat((half_state, -half_state), dim=1)
    independent_spot = torch.cat((half_spot, -half_spot), dim=1)
    del half_state, half_spot

    equilibrium = model.equilibrium_mean.detach()
    expected = equilibrium + (current_states - equilibrium) @ daily_transition
    next_states = expected[:, None, :] + state_normals @ daily_factor.T
    gaussian_score = (
        state_normals @ copula_loading + copula_orthogonal_std * independent_spot
    ) / copula_total_std

    with torch.inference_mode():
        current_surface = _decode_surface(model, current_states)
    gross_spot = np.empty((origins, num_paths), dtype=np.float64)
    score_values = gaussian_score.cpu().double().numpy()
    for origin in range(origins):
        gross_spot[origin] = _front_quantile_spline(
            support=current_surface.support_strikes[origin, 0].cpu().double().numpy(),
            density=current_surface.densities[origin, 0].cpu().double().numpy(),
            sqrt_variance=float(current_surface.backbone_total_vols[origin, 0].cpu()),
            gaussian_score=score_values[origin],
        )
    gross = torch.as_tensor(gross_spot, dtype=dtype, device=device)

    strikes = torch.tensor(_RELATIVE_STRIKES, dtype=dtype, device=device)
    initial_days = torch.tensor(_INITIAL_DAYS, dtype=dtype, device=device)
    final_days = torch.tensor(_FINAL_DAYS, dtype=dtype, device=device)
    is_put = torch.tensor(_IS_PUT, dtype=torch.bool, device=device)
    with torch.inference_mode():
        issue_calls = _call_prices(
            model,
            current_states,
            torch.ones(origins, dtype=dtype, device=device),
            remaining_days=initial_days,
            cash_strikes=strikes,
        )
        issue_values = torch.where(is_put[None], issue_calls - 1.0 + strikes[None], issue_calls)
        flat_spot = gross.reshape(-1)
        final_calls = _call_prices(
            model,
            next_states.reshape(-1, dimension),
            flat_spot,
            remaining_days=final_days,
            cash_strikes=strikes,
        ).reshape(origins, num_paths, -1)
        final_values = torch.where(
            is_put[None, None],
            final_calls - gross[..., None] + strikes[None, None],
            final_calls,
        )
    if not bool(torch.all(torch.isfinite(issue_values) & (issue_values > 0.0))):
        raise FloatingPointError("OCE book has a nonpositive issue price")
    option_returns = final_values / issue_values[:, None, :] - 1.0
    returns = torch.cat(((gross - 1.0)[..., None], option_returns), dim=2)
    if not bool(torch.all(torch.isfinite(returns))):
        raise FloatingPointError("OCE book generated non-finite returns")
    return returns


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _cost_band_audit(
    audit: Mapping[str, Any], transaction_cost: torch.Tensor
) -> dict[str, Any]:
    mean = np.asarray(audit["weighted_mean"], dtype=np.float64)
    standard_error = np.asarray(audit["standard_error"], dtype=np.float64)
    cost = transaction_cost.cpu().double().numpy()
    violation = np.maximum(np.abs(mean) - cost, 0.0)
    z = violation / np.maximum(standard_error, np.finfo(np.float64).tiny)
    return {
        "maximum_violation_bps": float(np.max(violation) * 1.0e4),
        "maximum_violation_z": float(np.max(z)),
        "coordinates_outside_band": int(np.sum(violation > 0.0)),
        "coordinates_outside_band_above_1p96_se": int(np.sum(z > 1.96)),
        "violation_bps": (violation * 1.0e4).tolist(),
        "violation_z": z.tolist(),
    }


def _aggregate_cost_bands(values: list[dict[str, Any]]) -> dict[str, Any]:
    violation = np.concatenate(
        [np.asarray(value["violation_bps"], dtype=np.float64) for value in values]
    )
    z = np.concatenate(
        [np.asarray(value["violation_z"], dtype=np.float64) for value in values]
    )
    return {
        "coordinates": len(violation),
        "maximum_violation_bps": float(np.max(violation)),
        "root_mean_square_violation_bps": float(np.sqrt(np.mean(violation**2))),
        "fraction_outside_band": float(np.mean(violation > 0.0)),
        "maximum_violation_z": float(np.max(z)),
        "fraction_outside_band_above_1p96_se": float(np.mean(z > 1.96)),
    }


def run(arguments: argparse.Namespace) -> None:
    """Generate fixed paths, fit the shared action network, and audit it."""

    _validate_arguments(arguments)
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_value = history.get("states")
    spots_value = history.get("spot_refs")
    if (
        not isinstance(dates, list)
        or not isinstance(states_value, torch.Tensor)
        or not isinstance(spots_value, torch.Tensor)
    ):
        raise ValueError("target history is missing dates, states, or spot references")
    states_numpy = states_value.double().numpy()
    spots = spots_value.double().numpy()
    states = states_value.to(dtype=env.fdtype, device=env.device)
    if arguments.validation_origins > len(states):
        raise ValueError("validation origins exceed the historical state population")

    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states_numpy,
        spots=spots,
        dates=dates,
        daily_transition=daily_transition.cpu().double().numpy(),
        daily_factor=daily_factor.cpu().double().numpy(),
        fit_before=arguments.copula_fit_before,
    )
    copula_loading = torch.as_tensor(
        copula_fit.state_loading, dtype=env.fdtype, device=env.device
    )
    features, feature_blocks = _state_features(model, states)
    feature_mean = torch.mean(features, dim=0)
    feature_std = torch.std(features, dim=0, correction=0)
    feature_scale = torch.where(feature_std > 1.0e-5, feature_std, torch.ones_like(feature_std))
    standardized_features = (features - feature_mean) / feature_scale

    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
    training_returns = torch.empty(
        (len(states), arguments.training_paths_per_origin, len(_LABELS)),
        dtype=torch.float32,
        device="cpu",
    )
    for first in range(0, len(states), arguments.generation_origin_batch_size):
        last = min(first + arguments.generation_origin_batch_size, len(states))
        values = _draw_book_returns(
            model=model,
            current_states=states[first:last],
            daily_transition=daily_transition,
            daily_factor=daily_factor,
            copula_loading=copula_loading,
            copula_orthogonal_std=copula_fit.orthogonal_std,
            copula_total_std=copula_fit.total_std,
            num_paths=arguments.training_paths_per_origin,
            generator=generator,
        )
        training_returns[first:last].copy_(values.cpu())
        print(f"generated training origins {last}/{len(states)}", flush=True)
        del values
    training_returns_gpu = training_returns.to(env.device)

    network_config = FeedForwardConfig(
        activation="SiLU",
        num_depth=arguments.hidden_layers,
        num_width=arguments.hidden_width,
    )
    network = FeedForward(
        env,
        feature_blocks["total"],
        len(_LABELS),
        network_config,
    )
    final_layer = network.I0 if len(network.In) == 0 else network.In[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is not None:
            final_layer.bias.zero_()
    optimizer = torch.optim.Adam(network.parameters(), lr=arguments.learning_rate)
    transaction_cost = torch.tensor(
        (arguments.spot_cost_bps, *(arguments.option_cost_bps for _ in range(6))),
        dtype=env.fdtype,
        device=env.device,
    ) / 1.0e4
    history_values: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    for epoch in range(arguments.epochs):
        progress = epoch / max(arguments.epochs - 1, 1)
        learning_rate = arguments.minimum_learning_rate + 0.5 * (
            arguments.learning_rate - arguments.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        raw_action = network(standardized_features)
        action = l1_capped_action(raw_action, arguments.action_l1_cap)
        values = conditional_entropic_oce(
            training_returns_gpu,
            action,
            transaction_cost,
            action_ridge=arguments.action_ridge,
        )
        values.loss.backward()
        gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
        for parameter in network.parameters():
            if parameter.grad is not None:
                gradient_square += torch.sum(parameter.grad.double().square())
        gradient_rms = torch.sqrt(gradient_square / parameter_count)
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(), arguments.gradient_clip_norm
        )
        optimizer.step()
        loss = float(values.loss.detach())
        if loss < best_loss:
            best_loss = loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
        if epoch % 10 == 0 or epoch + 1 == arguments.epochs:
            with torch.no_grad():
                raw_l1 = torch.sum(torch.abs(raw_action), dim=1)
                record = {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "gradient_rms": float(gradient_rms),
                    "gradient_norm_before_clip": float(unclipped_norm),
                    "minimum_training_ess_fraction": float(
                        torch.min(values.effective_sample_size)
                        / arguments.training_paths_per_origin
                    ),
                    "median_training_ess_fraction": float(
                        torch.median(values.effective_sample_size)
                        / arguments.training_paths_per_origin
                    ),
                    "maximum_action_l1": float(torch.max(torch.sum(torch.abs(action), dim=1))),
                    "cap_active_fraction": float(torch.mean((raw_l1 >= arguments.action_l1_cap).float())),
                }
                history_values.append(record)
            if epoch % 50 == 0 or epoch + 1 == arguments.epochs:
                print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("OCE network training produced no finite checkpoint")
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        raw_action_all = network(standardized_features)
        action_all = l1_capped_action(raw_action_all, arguments.action_l1_cap)
        final_training = conditional_entropic_oce(
            training_returns_gpu,
            action_all,
            transaction_cost,
            action_ridge=arguments.action_ridge,
        )

    origin_indices = np.linspace(
        0, len(states) - 1, arguments.validation_origins, dtype=np.int64
    )
    p_audits: list[dict[str, Any]] = []
    q_audits: list[dict[str, Any]] = []
    band_audits: list[dict[str, Any]] = []
    origin_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for position, origin_index in enumerate(origin_indices.tolist(), start=1):
            returns = _draw_book_returns(
                model=model,
                current_states=states[origin_index : origin_index + 1],
                daily_transition=daily_transition,
                daily_factor=daily_factor,
                copula_loading=copula_loading,
                copula_orthogonal_std=copula_fit.orthogonal_std,
                copula_total_std=copula_fit.total_std,
                num_paths=arguments.validation_paths_per_origin,
                generator=generator,
            )[0].double()
            action = action_all[origin_index : origin_index + 1].double()
            validation_values = conditional_entropic_oce(
                returns.unsqueeze(0),
                action,
                transaction_cost.double(),
                action_ridge=arguments.action_ridge,
            )
            p_audit = _paired_audit(
                returns,
                torch.ones(
                    len(returns), dtype=torch.float64, device=env.device
                ),
            )
            q_audit = _paired_audit(returns, validation_values.weights[0])
            band = _cost_band_audit(q_audit, transaction_cost)
            p_audits.append(p_audit)
            q_audits.append(q_audit)
            band_audits.append(band)
            origin_records.append(
                {
                    "origin_index": origin_index,
                    "date": str(dates[origin_index]),
                    "action": action[0].cpu().tolist(),
                    "action_l1": float(torch.sum(torch.abs(action))),
                    "validation_under_p": p_audit,
                    "validation_under_q": q_audit,
                    "cost_band_audit": band,
                }
            )
            print(
                f"validation {position}/{len(origin_indices)} {dates[origin_index]} "
                f"ESS={q_audit['effective_sample_size']:.1f}/{len(returns)} "
                f"band|maxz|={band['maximum_violation_z']:.3f}",
                flush=True,
            )

    checkpoint = {
        "schema": "dysanos-small-book-network-oce-v1",
        "phase": "complete",
        "network_config": {
            "activation": network_config.activation,
            "num_depth": network_config.num_depth,
            "num_width": network_config.num_width,
        },
        "network_state_dict": copy.deepcopy(network.state_dict()),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "feature_blocks": feature_blocks,
        "instrument_labels": _LABELS,
        "transaction_cost": transaction_cost.detach().cpu(),
        "action_l1_cap": arguments.action_l1_cap,
        "action_ridge": arguments.action_ridge,
        "source_checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "source_target_history_sha256": checkpoint_digest(arguments.target_history),
    }
    _atomic_torch_save(checkpoint, arguments.network_checkpoint)
    result = {
        "schema": "dysanos-small-book-network-oce-report-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "policy_input": "(h_t, vec(sqrt(q_t)), vec(log(DLVsq_t-DLVsq_min)))",
            "feature_blocks": feature_blocks,
            "policy_output": list(_LABELS),
            "option_contracts": (
                "fixed cash strikes established at the current normalized spot; "
                "5D/10D become 4D/9D after one business day"
            ),
            "return": "one-session price return V_{t+1}/V_t-1",
            "objective": (
                "mean_h log E_P[exp(-a_theta(h)'R)|h] + c'|a_theta(h)| "
                "+ action_ridge*||a_theta(h)||^2/2"
            ),
            "density": "dQ/dP proportional to exp(-a_theta(h)'R), normalized per state",
            "first_order_target": (
                "weighted return means lie inside +/-1 bp for spot and +/-10 bp "
                "for options; active interior actions reach the relevant boundary"
            ),
            "action_cap": "exact radial L1 cap applied separately at every state",
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "target_history_sha256": checkpoint_digest(arguments.target_history),
        "network_checkpoint": str(arguments.network_checkpoint),
        "network_checkpoint_sha256": checkpoint_digest(arguments.network_checkpoint),
        "network_parameter_count": parameter_count,
        "transition_diagnostics": transition_diagnostics,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "training": {
            "best_loss": best_loss,
            "final_selected_loss": float(final_training.loss),
            "minimum_ess_fraction": float(
                torch.min(final_training.effective_sample_size)
                / arguments.training_paths_per_origin
            ),
            "median_ess_fraction": float(
                torch.median(final_training.effective_sample_size)
                / arguments.training_paths_per_origin
            ),
            "maximum_action_l1": float(torch.max(torch.sum(torch.abs(action_all), dim=1))),
            "cap_active_fraction": float(
                torch.mean(
                    (
                        torch.sum(torch.abs(raw_action_all), dim=1)
                        >= arguments.action_l1_cap
                    ).float()
                )
            ),
            "history": history_values,
        },
        "independent_validation_under_p": _aggregate_audits(p_audits),
        "independent_validation_under_q": _aggregate_audits(q_audits),
        "independent_validation_cost_band": _aggregate_cost_bands(band_audits),
        "validation_origin_indices": origin_indices.tolist(),
        "origins": origin_records,
    }
    write_json_atomic(result, arguments.output)
    print(
        json.dumps(
            {
                "training": result["training"],
                "independent_validation_under_p": result["independent_validation_under_p"],
                "independent_validation_under_q": result["independent_validation_under_q"],
                "independent_validation_cost_band": result[
                    "independent_validation_cost_band"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
