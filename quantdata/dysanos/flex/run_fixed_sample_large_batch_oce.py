"""Fit a large-effective-batch OCE and audit its fixed training sample."""

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

from quantdata.dysanos.flex.front_spot_copula import fit_front_spot_copula_from_history
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.network_oce import entropic_density_from_pnl
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.run_single_state_monthly_network_oce import (
    _atomic_numpy_save,
    _audit_flat_returns,
    _book,
    _feature_statistics,
    _simulate_expiry_path_batch,
)
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--network-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-path-returns-output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--book-expiry-days", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--training-paths", type=int, default=1_000_000)
    parser.add_argument("--training-epochs", type=int, default=4)
    parser.add_argument("--simulation-batch-size", type=int, default=8_192)
    parser.add_argument("--optimizer-batch-size", type=int, default=65_536)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--action-l1-cap", type=float, default=16.0)
    parser.add_argument("--spot-cost-bps", type=float, default=1.0)
    parser.add_argument("--option-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-ridge", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2_026_100_7)
    parser.add_argument("--copula-fit-before", type=date.fromisoformat, default=date(2025, 1, 1))
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    _book(tuple(arguments.book_expiry_days))
    positive_integers = (
        arguments.num_steps,
        arguments.training_paths,
        arguments.training_epochs,
        arguments.simulation_batch_size,
        arguments.optimizer_batch_size,
        arguments.hidden_width,
        arguments.hidden_layers,
        arguments.checkpoint_every_updates,
    )
    if min(positive_integers) <= 0:
        raise ValueError("large-batch OCE integer controls must be positive")
    if arguments.training_paths % 2 or arguments.simulation_batch_size % 2:
        raise ValueError("training paths and simulation batches must be antithetic-even")
    if arguments.optimizer_batch_size < arguments.simulation_batch_size:
        raise ValueError("optimizer batch cannot be smaller than a simulation batch")
    if arguments.optimizer_batch_size % arguments.simulation_batch_size:
        raise ValueError("optimizer batch must contain complete simulation batches")
    positive_values = (
        arguments.learning_rate,
        arguments.minimum_learning_rate,
        arguments.action_l1_cap,
        arguments.spot_cost_bps,
        arguments.option_cost_bps,
        arguments.gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("large-batch OCE scalar controls must be finite-positive")
    if arguments.minimum_learning_rate > arguments.learning_rate:
        raise ValueError("minimum learning rate exceeds the initial rate")
    if arguments.warmup_updates < 0:
        raise ValueError("warmup updates cannot be negative")
    if not math.isfinite(arguments.action_ridge) or arguments.action_ridge < 0.0:
        raise ValueError("action ridge must be finite and nonnegative")


def _learning_rate(
    arguments: argparse.Namespace,
    update: int,
    total_updates: int,
) -> float:
    if update < arguments.warmup_updates:
        return arguments.learning_rate * (update + 1) / max(arguments.warmup_updates, 1)
    progress = (update - arguments.warmup_updates) / max(
        total_updates - arguments.warmup_updates - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return arguments.minimum_learning_rate + (
        arguments.learning_rate - arguments.minimum_learning_rate
    ) * cosine


def _microbatch_sizes(total_paths: int, simulation_batch_size: int) -> list[int]:
    sizes = [
        min(simulation_batch_size, total_paths - first)
        for first in range(0, total_paths, simulation_batch_size)
    ]
    if any(size % 2 for size in sizes):
        raise RuntimeError("an OCE simulation microbatch is not antithetic-even")
    return sizes


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    network: FeedForward,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    completed_updates: int,
    history: list[dict[str, float | int]],
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    feature_blocks: Mapping[str, int],
    arguments: argparse.Namespace,
    complete: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "dysanos-fixed-sample-large-batch-oce-v1",
        "phase": "complete" if complete else "training",
        "completed_updates": completed_updates,
        "network_state_dict": copy.deepcopy(network.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "generator_state": generator.get_state().cpu(),
        "history": history,
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "feature_blocks": dict(feature_blocks),
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
    }


def _cost_band(arguments: argparse.Namespace, num_options: int) -> np.ndarray:
    one_date = np.asarray(
        (
            arguments.spot_cost_bps,
            *(arguments.option_cost_bps for _ in range(num_options)),
        ),
        dtype=np.float64,
    ) / 1.0e4
    return np.tile(one_date, arguments.num_steps)


def run(arguments: argparse.Namespace) -> None:
    """Optimize and audit one fixed empirical to-expiry training population."""

    _validate_arguments(arguments)
    book = _book(tuple(arguments.book_expiry_days))
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
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
    if not 0 <= state_index < len(dates):
        raise ValueError("state index is outside the historical population")
    historical_states = states_value.to(dtype=env.fdtype, device=env.device)
    initial_state = historical_states[state_index]
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
    feature_mean, feature_scale, feature_blocks = _feature_statistics(
        model, historical_states
    )
    network = FeedForward(
        env,
        feature_blocks["total"],
        len(book.labels),
        FeedForwardConfig(
            activation="SiLU",
            num_depth=arguments.hidden_layers,
            num_width=arguments.hidden_width,
        ),
    )
    final_layer = network.I0 if len(network.In) == 0 else network.In[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is not None:
            final_layer.bias.zero_()
    optimizer = torch.optim.Adam(network.parameters(), lr=arguments.learning_rate)
    transaction_cost = torch.tensor(
        (
            arguments.spot_cost_bps,
            *(arguments.option_cost_bps for _ in book.relative_strikes),
        ),
        dtype=env.fdtype,
        device=env.device,
    ) / 1.0e4
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    optimizer_batches_per_epoch = math.ceil(
        arguments.training_paths / arguments.optimizer_batch_size
    )
    total_updates = arguments.training_epochs * optimizer_batches_per_epoch
    generator = torch.Generator(device=env.device).manual_seed(arguments.seed)
    completed_updates = 0
    history: list[dict[str, float | int]] = []

    if arguments.resume:
        saved = torch.load(arguments.network_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping) or saved.get("schema") != (
            "dysanos-fixed-sample-large-batch-oce-v1"
        ):
            raise ValueError("large-batch OCE resume checkpoint is invalid")
        saved_arguments = saved.get("arguments")
        if not isinstance(saved_arguments, Mapping):
            raise ValueError("large-batch OCE checkpoint has no arguments")
        current_arguments = {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        }
        invariant_arguments = tuple(
            key
            for key in current_arguments
            if key not in {"resume", "checkpoint_every_updates"}
        )
        if any(
            saved_arguments.get(key) != current_arguments[key]
            for key in invariant_arguments
        ):
            raise ValueError("large-batch OCE resume arguments differ from checkpoint")
        network.load_state_dict(saved["network_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"].cpu())
        completed_updates = int(saved["completed_updates"])
        history = list(saved.get("history", []))
        if not 0 <= completed_updates <= total_updates:
            raise ValueError("large-batch OCE completed-update count is invalid")
        print(f"resumed at optimizer update {completed_updates}/{total_updates}", flush=True)

    simulation_arguments = {
        "model": model,
        "network": network,
        "book": book,
        "initial_state": initial_state,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "daily_transition": daily_transition,
        "daily_factor": daily_factor,
        "copula_loading": copula_loading,
        "copula_orthogonal_std": copula_fit.orthogonal_std,
        "copula_total_std": copula_fit.total_std,
        "transaction_cost": transaction_cost,
        "action_l1_cap": arguments.action_l1_cap,
        "action_ridge": arguments.action_ridge,
        "num_steps": arguments.num_steps,
    }

    start_epoch = completed_updates // optimizer_batches_per_epoch
    start_batch = completed_updates % optimizer_batches_per_epoch
    for epoch in range(start_epoch, arguments.training_epochs):
        first_optimizer_batch = start_batch if epoch == start_epoch else 0
        if first_optimizer_batch == 0:
            generator.manual_seed(arguments.seed)
        for optimizer_batch in range(first_optimizer_batch, optimizer_batches_per_epoch):
            batch_first = optimizer_batch * arguments.optimizer_batch_size
            optimizer_paths = min(
                arguments.optimizer_batch_size,
                arguments.training_paths - batch_first,
            )
            microbatch_sizes = _microbatch_sizes(
                optimizer_paths, arguments.simulation_batch_size
            )
            start_generator_state = generator.get_state()
            pnl_parts: list[torch.Tensor] = []
            action_l1_sum = 0.0
            cap_active_sum = 0.0
            with torch.inference_mode():
                for microbatch_paths in microbatch_sizes:
                    pnl, _, diagnostics = _simulate_expiry_path_batch(
                        **simulation_arguments,
                        num_paths=microbatch_paths,
                        generator=generator,
                        retain_returns=False,
                    )
                    pnl_parts.append(pnl.cpu().double())
                    action_l1_sum += float(diagnostics["mean_action_l1"]) * microbatch_paths
                    cap_active_sum += (
                        float(diagnostics["cap_active_fraction"]) * microbatch_paths
                    )
            ending_generator_state = generator.get_state()
            batch_pnl = torch.cat(pnl_parts)
            density = entropic_density_from_pnl(batch_pnl)

            generator.set_state(start_generator_state)
            learning_rate = _learning_rate(arguments, completed_updates, total_updates)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for microbatch_paths in microbatch_sizes:
                pnl, _, _ = _simulate_expiry_path_batch(
                    **simulation_arguments,
                    num_paths=microbatch_paths,
                    generator=generator,
                    retain_returns=False,
                )
                weights = density.weights[
                    offset : offset + microbatch_paths
                ].to(device=env.device, dtype=pnl.dtype)
                (-torch.sum(weights.detach() * pnl) / optimizer_paths).backward()
                offset += microbatch_paths
            if not torch.equal(generator.get_state(), ending_generator_state):
                raise RuntimeError("large-batch OCE replay did not reproduce its paths")

            gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
            for parameter in network.parameters():
                if parameter.grad is not None:
                    gradient_square += torch.sum(parameter.grad.double().square())
            gradient_rms = torch.sqrt(gradient_square / parameter_count)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                network.parameters(), arguments.gradient_clip_norm
            )
            optimizer.step()
            completed_updates += 1
            record = {
                "epoch": epoch + 1,
                "optimizer_batch": optimizer_batch + 1,
                "update": completed_updates,
                "effective_batch_paths": optimizer_paths,
                "log_oce": float(density.log_oce),
                "effective_sample_fraction": float(
                    density.effective_sample_size / optimizer_paths
                ),
                "learning_rate": learning_rate,
                "gradient_rms": float(gradient_rms),
                "gradient_norm_before_clip": float(gradient_norm),
                "mean_action_l1": action_l1_sum / optimizer_paths,
                "cap_active_fraction": cap_active_sum / optimizer_paths,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if (
                completed_updates % arguments.checkpoint_every_updates == 0
                or completed_updates == total_updates
            ):
                _atomic_torch_save(
                    _checkpoint_payload(
                        network=network,
                        optimizer=optimizer,
                        generator=generator,
                        completed_updates=completed_updates,
                        history=history,
                        feature_mean=feature_mean,
                        feature_scale=feature_scale,
                        feature_blocks=feature_blocks,
                        arguments=arguments,
                    ),
                    arguments.network_checkpoint,
                )

    # Audit the exact fixed population used in every training epoch.  PnL is
    # evaluated first to obtain one global density normalization; the same RNG
    # stream is then replayed for returns and the full-sample OCE gradient.
    network.eval()
    training_pnl = torch.empty(arguments.training_paths, dtype=torch.float64)
    generator.manual_seed(arguments.seed)
    pair_offset = 0
    with torch.inference_mode():
        for first in range(0, arguments.training_paths, arguments.simulation_batch_size):
            microbatch_paths = min(
                arguments.simulation_batch_size, arguments.training_paths - first
            )
            pnl, _, _ = _simulate_expiry_path_batch(
                **simulation_arguments,
                num_paths=microbatch_paths,
                generator=generator,
                retain_returns=False,
            )
            half = microbatch_paths // 2
            training_pnl[pair_offset : pair_offset + half] = pnl[:half].cpu().double()
            training_pnl[
                arguments.training_paths // 2
                + pair_offset : arguments.training_paths // 2
                + pair_offset
                + half
            ] = pnl[half:].cpu().double()
            pair_offset += half
            if first % (25 * arguments.simulation_batch_size) == 0:
                print(
                    f"training audit pass 1: {first + microbatch_paths}/"
                    f"{arguments.training_paths}",
                    flush=True,
                )
    training_density = entropic_density_from_pnl(training_pnl)

    coordinates = arguments.num_steps * len(book.labels)
    training_returns = torch.empty(
        (arguments.training_paths, coordinates), dtype=torch.float32
    )
    generator.manual_seed(arguments.seed)
    network.zero_grad(set_to_none=True)
    pair_offset = 0
    action_l1_weighted = 0.0
    cap_active_weighted = 0.0
    for first in range(0, arguments.training_paths, arguments.simulation_batch_size):
        microbatch_paths = min(
            arguments.simulation_batch_size, arguments.training_paths - first
        )
        pnl, returns, diagnostics = _simulate_expiry_path_batch(
            **simulation_arguments,
            num_paths=microbatch_paths,
            generator=generator,
            retain_returns=True,
        )
        assert returns is not None
        half = microbatch_paths // 2
        first_weights = training_density.weights[pair_offset : pair_offset + half]
        second_weights = training_density.weights[
            arguments.training_paths // 2
            + pair_offset : arguments.training_paths // 2
            + pair_offset
            + half
        ]
        sequential_weights = torch.cat((first_weights, second_weights)).to(
            device=env.device, dtype=pnl.dtype
        )
        (-torch.sum(sequential_weights.detach() * pnl) / arguments.training_paths).backward()
        flat_returns = returns.detach().reshape(microbatch_paths, -1).cpu()
        training_returns[pair_offset : pair_offset + half] = flat_returns[:half]
        training_returns[
            arguments.training_paths // 2
            + pair_offset : arguments.training_paths // 2
            + pair_offset
            + half
        ] = flat_returns[half:]
        pair_offset += half
        action_l1_weighted += float(diagnostics["mean_action_l1"]) * microbatch_paths
        cap_active_weighted += float(diagnostics["cap_active_fraction"]) * microbatch_paths
        if first % (25 * arguments.simulation_batch_size) == 0:
            print(
                f"training audit pass 2: {first + microbatch_paths}/"
                f"{arguments.training_paths}",
                flush=True,
            )
    if arguments.fixed_path_returns_output is not None:
        _atomic_numpy_save(training_returns, arguments.fixed_path_returns_output)
        print(
            f"wrote fixed training returns to {arguments.fixed_path_returns_output}",
            flush=True,
        )

    training_gradient_square = torch.zeros((), dtype=torch.float64, device=env.device)
    for parameter in network.parameters():
        if parameter.grad is not None:
            training_gradient_square += torch.sum(parameter.grad.double().square())
    training_gradient_rms = torch.sqrt(training_gradient_square / parameter_count)
    p_audit = _audit_flat_returns(
        training_returns,
        torch.ones(arguments.training_paths, dtype=torch.float64),
        device=env.device,
    )
    q_audit = _audit_flat_returns(
        training_returns,
        training_density.weights,
        device=env.device,
    )
    cost = _cost_band(arguments, len(book.relative_strikes))
    q_mean = np.asarray(q_audit["weighted_mean"])
    q_standard_error = np.asarray(q_audit["standard_error"])
    cost_violation = np.maximum(np.abs(q_mean) - cost, 0.0)
    cost_violation_z = cost_violation / np.maximum(
        q_standard_error, np.finfo(np.float64).tiny
    )
    training_summary = {
        "log_oce": float(training_density.log_oce),
        "effective_sample_size": float(training_density.effective_sample_size),
        "effective_sample_fraction": float(
            training_density.effective_sample_size / arguments.training_paths
        ),
        "minimum_weight": float(torch.amin(training_density.weights)),
        "maximum_weight": float(torch.amax(training_density.weights)),
        "zero_weight_count": int(torch.sum(training_density.weights == 0.0)),
        "gradient_rms": float(training_gradient_rms),
        "mean_action_l1": action_l1_weighted / arguments.training_paths,
        "cap_active_fraction": cap_active_weighted / arguments.training_paths,
        "under_p": p_audit,
        "under_q": q_audit,
        "unconditional_cost_band": {
            "maximum_violation_bps": float(np.max(cost_violation) * 1.0e4),
            "root_mean_square_violation_bps": float(
                np.sqrt(np.mean(cost_violation**2)) * 1.0e4
            ),
            "maximum_violation_z": float(np.max(cost_violation_z)),
            "fraction_outside_band": float(np.mean(cost_violation > 0.0)),
            "fraction_outside_band_above_1p96_se": float(
                np.mean(cost_violation_z > 1.96)
            ),
        },
    }
    _atomic_torch_save(
        _checkpoint_payload(
            network=network,
            optimizer=optimizer,
            generator=generator,
            completed_updates=completed_updates,
            history=history,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            feature_blocks=feature_blocks,
            arguments=arguments,
            complete=True,
        )
        | {"training": training_summary},
        arguments.network_checkpoint,
    )
    result = {
        "schema": "dysanos-fixed-sample-large-batch-oce-report-v1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "definition": {
            "sample": (
                "one fixed empirical training population, deterministically replayed "
                "in every epoch and in the terminal audit"
            ),
            "path_horizon": (
                f"{arguments.num_steps} daily entry dates plus a "
                f"{max(book.expiry_days)}-day settlement tail"
            ),
            "rolling_contracts": (
                f"fresh fixed-cash {list(book.expiry_days)}D options opened at every "
                "entry date and held to terminal payoff"
            ),
            "policy_input": "(h_t, vec(sqrt(q_t)), vec(log(DLVsq_t-DLVsq_min)))",
            "policy_output": list(book.labels),
            "feature_blocks": feature_blocks,
            "objective": (
                "log mean(exp(-net path PnL)); each optimizer normalization spans "
                "optimizer_batch_size paths replayed through simulation microbatches"
            ),
            "density": "dQ/dP=exp(-net path PnL)/mean(exp(-net path PnL))",
            "scope": (
                "in-sample empirical OCE fit; no generalization or out-of-sample "
                "martingale claim"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in vars(arguments).items()
        },
        "state_index": state_index,
        "state_date": str(dates[state_index]),
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
        "training_history": history,
        "training": training_summary,
    }
    write_json_atomic(result, arguments.output)
    print(json.dumps({"training": training_summary}, indent=2), flush=True)
    print(f"wrote {arguments.output}", flush=True)


def main() -> None:
    run(_arguments())


if __name__ == "__main__":
    main()
