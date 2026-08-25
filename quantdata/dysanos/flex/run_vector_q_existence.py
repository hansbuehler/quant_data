"""Find a P-free conditional front-state law for eleven W2 option rolls."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw, dlvsq_cap_barrier
from quantdata.dysanos.flex.run_atm_q_existence import (
    AtmQExistenceLaw,
    AtmQExistenceLawConfig,
    FrontWeekExactMixtureSampler,
    QExistenceBatch,
    _atomic_torch_save,
    _load_inputs,
    _make_batches,
    _price_calls,
    spot_control_variate,
)
from quantdata.torch import BaseEnvironment

OBJECTIVE_ID = "pooled-fixed-stream-eleven-option-and-atm-branch-moments-v1"
_SHARED_STRIKES_TRAINING_ROOT = (
    Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
)
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-fixed-state-exact-pooled-v1"
)


def spot_control_variate_vector(
    returns: torch.Tensor,
    gross_spot_return: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the exact zero-mean spot-return control to each option column."""
    if returns.ndim != 2 or gross_spot_return.shape != returns.shape[:1]:
        raise ValueError("vector spot control variate requires aligned paths")
    spot_return = gross_spot_return - 1.0
    centered_spot = spot_return - torch.mean(spot_return)
    centered_returns = returns - torch.mean(returns, dim=0)
    denominator = torch.sum(centered_spot.square()).clamp_min(
        torch.finfo(returns.dtype).tiny
    )
    beta = (
        torch.sum(centered_spot[:, None] * centered_returns, dim=0) / denominator
    ).detach()
    adjusted = returns - spot_return[:, None] * beta
    return adjusted, torch.mean(adjusted, dim=0), beta


def vector_roll_returns(
    *,
    model: Any,
    law: AtmQExistenceLaw,
    current_state: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    batch: QExistenceBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eleven fixed-cash-strike ITM option mark changes."""
    next_front_state = law(
        spot_rank=batch.spot_rank,
        branch=batch.branch,
        epsilon=batch.epsilon,
    )
    next_state = current_state.unsqueeze(0).expand(batch.num_paths, -1).clone()
    next_state[:, : law.config.front_state_size] = next_front_state
    next_normalized_strikes = (
        normalized_strikes[None] / batch.gross_spot_return[:, None]
    )
    next_w1_calls, raw = _price_calls(
        model,
        next_state,
        expiry_index=0,
        normalized_strikes=next_normalized_strikes,
    )
    call_returns = (
        batch.gross_spot_return[:, None] * next_w1_calls - current_w2_calls[None]
    )
    # Put-call parity at the same fixed cash strike changes a call mark return
    # into the corresponding put mark return by subtracting the spot return.
    returns = torch.where(
        option_is_call[None],
        call_returns,
        call_returns - (batch.gross_spot_return - 1.0)[:, None],
    )
    return returns, raw


def _replicated_statistics(
    returns_by_replication: list[torch.Tensor],
    batches: list[QExistenceBatch],
    *,
    atm_index: int,
    branch_target_bps: float,
) -> dict[str, object]:
    if len(returns_by_replication) != len(batches) or not batches:
        raise ValueError("replicated statistics require aligned non-empty inputs")
    adjusted_means: list[torch.Tensor] = []
    raw_means: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    negative_atm_means: list[torch.Tensor] = []
    positive_atm_means: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    for returns, batch in zip(returns_by_replication, batches, strict=True):
        _, adjusted_mean, beta = spot_control_variate_vector(
            returns,
            batch.gross_spot_return,
        )
        negative = batch.branch < 0.0
        positive = batch.branch > 0.0
        _, negative_atm, _ = spot_control_variate(
            returns[negative, atm_index],
            batch.gross_spot_return[negative],
        )
        _, positive_atm, _ = spot_control_variate(
            returns[positive, atm_index],
            batch.gross_spot_return[positive],
        )
        adjusted_means.append(adjusted_mean)
        raw_means.append(torch.mean(returns, dim=0))
        betas.append(beta)
        negative_atm_means.append(negative_atm)
        positive_atm_means.append(positive_atm)
        all_returns.append(returns)
    adjusted = torch.stack(adjusted_means)
    raw = torch.stack(raw_means)
    combined = torch.cat(all_returns)
    adjusted_mean = torch.mean(adjusted, dim=0)
    adjusted_se = torch.std(adjusted, dim=0, correction=1) / math.sqrt(len(adjusted))
    raw_mean = torch.mean(raw, dim=0)
    raw_se = torch.std(raw, dim=0, correction=1) / math.sqrt(len(raw))
    raw_z = raw_mean / raw_se.clamp_min(1.0e-12)
    std = torch.std(combined, dim=0, correction=1)
    negative_atm = torch.stack(negative_atm_means)
    positive_atm = torch.stack(positive_atm_means)
    scale = 1.0e4
    return {
        "replications": len(batches),
        "paths_per_replication": batches[0].num_paths,
        "total_paths": sum(batch.num_paths for batch in batches),
        "adjusted_mean_bps_spot": (scale * adjusted_mean).tolist(),
        "adjusted_mean_mc_standard_error_bps_spot": (scale * adjusted_se).tolist(),
        "adjusted_mean_z": (adjusted_mean / adjusted_se.clamp_min(1.0e-12)).tolist(),
        "maximum_abs_adjusted_mean_bps_spot": float(
            scale * torch.max(torch.abs(adjusted_mean))
        ),
        "raw_mean_bps_spot": (scale * raw_mean).tolist(),
        "raw_mean_mc_standard_error_bps_spot": (scale * raw_se).tolist(),
        "raw_mean_z": raw_z.tolist(),
        "maximum_abs_raw_mean_bps_spot": float(
            scale * torch.max(torch.abs(raw_mean))
        ),
        "maximum_abs_raw_mean_z": float(torch.max(torch.abs(raw_z))),
        "return_sample_std_bps_spot": (scale * std).tolist(),
        "return_sample_variance_bps2_spot": ((scale * std).square()).tolist(),
        "negative_return_percent": (
            100.0 * torch.mean((combined < 0.0).float(), dim=0)
        ).tolist(),
        "positive_return_percent": (
            100.0 * torch.mean((combined > 0.0).float(), dim=0)
        ).tolist(),
        "negative_branch_atm_adjusted_mean_bps_spot": float(
            scale * torch.mean(negative_atm)
        ),
        "positive_branch_atm_adjusted_mean_bps_spot": float(
            scale * torch.mean(positive_atm)
        ),
        "branch_target_bps_spot": branch_target_bps,
        "spot_control_beta_mean": torch.mean(torch.stack(betas), dim=0).tolist(),
        "gross_spot_return_mean": float(
            torch.mean(torch.cat([batch.gross_spot_return for batch in batches]))
        ),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports" / "vector-q-existence.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints" / "vector-q-existence.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selected-validation-index", type=int, default=62)
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--fit-seed", type=int, default=9_384_101)
    parser.add_argument("--validation-seed", type=int, default=9_385_101)
    parser.add_argument("--initialization-seed", type=int, default=9_383_101)
    parser.add_argument("--stable-epochs", type=int, default=20)
    parser.add_argument("--plateau-epochs", type=int, default=100)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-5)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("vector Q-existence training is restricted to GPU0")
    counts = (
        arguments.epochs,
        arguments.fit_replications,
        arguments.validation_replications,
        arguments.spot_contexts,
        arguments.surface_paths,
        arguments.stable_epochs,
        arguments.plateau_epochs,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("all vector Q-existence counts must be positive")
    if arguments.fit_replications < 12 or arguments.validation_replications < 12:
        raise ValueError("Monte Carlo inference requires at least twelve replications")
    if arguments.surface_paths < 4 or arguments.surface_paths % 4:
        raise ValueError("surface paths must be at least four and divisible by four")
    if arguments.fit_seed == arguments.validation_seed:
        raise ValueError("fit and validation streams must be disjoint")
    if not 0.0 < arguments.minimum_relative_improvement < 1.0:
        raise ValueError("minimum relative improvement must lie strictly in (0, 1)")


def run(arguments: argparse.Namespace) -> None:
    """Train and independently validate the eleven-option conditional law."""
    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        _,
        _,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    if arguments.selected_validation_index >= len(validation_states):
        raise ValueError("selected validation index is outside the 2025 history")
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    current_state = validation_states[arguments.selected_validation_index].detach()
    model_date = validation_dates[arguments.selected_validation_index].isoformat()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = AtmQExistenceLawConfig(
        label="vector-q-existence-exact-v1",
        front_state_size=front_state_size,
    )
    law = AtmQExistenceLaw(
        env,
        current_front_state=current_state[:front_state_size],
        config=config,
        initialization_seed=arguments.initialization_seed,
    )
    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    atm_index = 5
    with torch.no_grad():
        raw = model.raw_states_from_states(current_state.unsqueeze(0))
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_calls, _ = _price_calls(
            model,
            current_state.unsqueeze(0),
            expiry_index=1,
            normalized_strikes=normalized_strikes.unsqueeze(0),
        )
        current_w2_calls = current_w2_calls[0].detach()
    front_sampler = FrontWeekExactMixtureSampler.from_sanos_components(
        support=surface.support_strikes[0, 0],
        density=surface.densities[0, 0],
        sqrt_variance=surface.backbone_total_vols[0, 0],
    )
    fit_batches = _make_batches(
        front_sampler=front_sampler,
        config=config,
        replications=arguments.fit_replications,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        seed=arguments.fit_seed,
    )
    validation_batches = _make_batches(
        front_sampler=front_sampler,
        config=config,
        replications=arguments.validation_replications,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        seed=arguments.validation_seed,
    )
    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=5.0e-5
    )
    history: list[dict[str, float]] = []
    stable = 0
    start_epoch = 0
    epochs_without_material_improvement = 0
    best_loss = math.inf
    checkpoint: dict[str, Any] | None = None
    if arguments.checkpoint.exists():
        checkpoint_payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(checkpoint_payload, Mapping)
            or checkpoint_payload.get("schema")
            != "flex-vector-q-existence-training-v1"
            or checkpoint_payload.get("source_completion_sha256")
            != sources["completion_sha256"]
            or checkpoint_payload.get("market_date") != model_date
            or checkpoint_payload.get("law_config") != asdict(config)
        ):
            raise ValueError("vector Q-existence checkpoint identity is incompatible")
        law.load_state_dict(
            cast(Mapping[str, torch.Tensor], checkpoint_payload["law_state_dict"])
        )
        optimizer.load_state_dict(
            cast(dict[str, Any], checkpoint_payload["optimizer_state_dict"])
        )
        scheduler.load_state_dict(
            cast(dict[str, Any], checkpoint_payload["scheduler_state_dict"])
        )
        history = list(cast(list[dict[str, float]], checkpoint_payload["history"]))
        stable = int(checkpoint_payload["stable_epochs"])
        epochs_without_material_improvement = int(
            checkpoint_payload.get("epochs_without_material_improvement", 0)
        )
        start_epoch = len(history)
        if history:
            best_loss = float(
                checkpoint_payload.get(
                    "best_loss", min(record["loss"] for record in history)
                )
            )
        checkpoint = dict(checkpoint_payload)
        print(
            f"resuming vector Q-existence at epoch {start_epoch} "
            f"with best loss {best_loss:.6g}",
            flush=True,
        )
    target = config.branch_atm_target_bps

    def returns_for(batch: QExistenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return vector_roll_returns(
            model=model,
            law=law,
            current_state=current_state,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            batch=batch,
        )

    stopped_for_plateau = False
    for epoch in range(start_epoch, arguments.epochs):
        epoch_returns: list[torch.Tensor] = []
        cap_values: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in fit_batches:
                returns, raw_next = returns_for(batch)
                epoch_returns.append(returns)
                cap_values.append(
                    1.0e-4
                    * torch.mean(
                        dlvsq_cap_barrier(raw_next, geometry=model.config.geometry).square()
                    )
                )
        statistics = _replicated_statistics(
            epoch_returns,
            fit_batches,
            atm_index=atm_index,
            branch_target_bps=target,
        )
        pooled = torch.tensor(
            cast(list[float], statistics["adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        negative_atm = torch.tensor(
            float(statistics["negative_branch_atm_adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        positive_atm = torch.tensor(
            float(statistics["positive_branch_atm_adjusted_mean_bps_spot"]),
            dtype=env.fdtype,
            device=env.device,
        )
        loss = (
            torch.mean(pooled.square())
            + 0.01 * ((negative_atm + target).square() + (positive_atm - target).square())
            + torch.mean(torch.stack(cap_values))
        )
        loss_value = float(loss)
        optimizer.zero_grad(set_to_none=True)
        for batch in fit_batches:
            returns, raw_next = returns_for(batch)
            _, adjusted_mean, _ = spot_control_variate_vector(
                returns, batch.gross_spot_return
            )
            negative = batch.branch < 0.0
            positive = batch.branch > 0.0
            _, negative_mean, _ = spot_control_variate(
                returns[negative, atm_index], batch.gross_spot_return[negative]
            )
            _, positive_mean, _ = spot_control_variate(
                returns[positive, atm_index], batch.gross_spot_return[positive]
            )
            replication_count = len(fit_batches)
            surrogate = (
                2.0
                * torch.sum(pooled * (1.0e4 * adjusted_mean))
                / (len(normalized_strikes) * replication_count)
                + 0.02
                * (negative_atm + target)
                * (1.0e4 * negative_mean)
                / replication_count
                + 0.02
                * (positive_atm - target)
                * (1.0e4 * positive_mean)
                / replication_count
                + 1.0e-4
                * torch.mean(
                    dlvsq_cap_barrier(raw_next, geometry=model.config.geometry).square()
                )
                / replication_count
            )
            surrogate.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        scheduler.step(loss_value)
        maximum_mean = float(statistics["maximum_abs_adjusted_mean_bps_spot"])
        maximum_raw_mean = float(statistics["maximum_abs_raw_mean_bps_spot"])
        maximum_raw_z = float(statistics["maximum_abs_raw_mean_z"])
        record = {
            "epoch": float(epoch + 1),
            "loss": loss_value,
            "maximum_abs_adjusted_mean_bps_spot": maximum_mean,
            "maximum_abs_raw_mean_bps_spot": maximum_raw_mean,
            "maximum_abs_raw_mean_z": maximum_raw_z,
            "negative_branch_atm_adjusted_mean_bps_spot": float(negative_atm),
            "positive_branch_atm_adjusted_mean_bps_spot": float(positive_atm),
            "state_noise_std": float(law.noise_std.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise FloatingPointError("non-finite vector Q-existence training record")
        history.append(record)
        if loss_value < best_loss * (1.0 - arguments.minimum_relative_improvement):
            best_loss = loss_value
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1
        passed = (
            maximum_mean <= 0.50
            and abs(float(negative_atm) + target) <= 1.0
            and abs(float(positive_atm) - target) <= 1.0
        )
        stable = stable + 1 if passed else 0
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch + 1} loss={loss_value:.6g} "
                f"max_mean={maximum_mean:.4f}bp "
                f"raw_max_mean={maximum_raw_mean:.4f}bp "
                f"raw_max_z={maximum_raw_z:.2f} "
                f"atm_branches=({float(negative_atm):.3f},{float(positive_atm):.3f})bp "
                f"noise={record['state_noise_std']:.5f} "
                f"lr={record['learning_rate']:.6g}",
                flush=True,
            )
        checkpoint = {
            "schema": "flex-vector-q-existence-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "market_date": model_date,
            "law_config": asdict(config),
            "law_state_dict": law.state_dict(),
            "model_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "stable_epochs": stable,
            "best_loss": best_loss,
            "epochs_without_material_improvement": (
                epochs_without_material_improvement
            ),
        }
        if (epoch + 1) % 5 == 0 or stable >= arguments.stable_epochs:
            _atomic_torch_save(checkpoint, arguments.checkpoint)
        if epochs_without_material_improvement >= arguments.plateau_epochs:
            stopped_for_plateau = True
            print(
                "stopping after "
                f"{epochs_without_material_improvement} epochs without material "
                f"relative improvement of {arguments.minimum_relative_improvement:g}",
                flush=True,
            )
            break

    with torch.no_grad():
        fit_audit = _replicated_statistics(
            [returns_for(batch)[0] for batch in fit_batches],
            fit_batches,
            atm_index=atm_index,
            branch_target_bps=target,
        )
        validation_audit = _replicated_statistics(
            [returns_for(batch)[0] for batch in validation_batches],
            validation_batches,
            atm_index=atm_index,
            branch_target_bps=target,
        )
    if checkpoint is None:
        raise RuntimeError("vector Q-existence run produced no checkpoint state")
    validation_mean = torch.tensor(
        cast(list[float], validation_audit["adjusted_mean_bps_spot"])
    ).abs()
    validation_se = torch.tensor(
        cast(
            list[float],
            validation_audit["adjusted_mean_mc_standard_error_bps_spot"],
        )
    )
    validation_std = torch.tensor(
        cast(list[float], validation_audit["return_sample_std_bps_spot"])
    )
    validation_positive = torch.tensor(
        cast(list[float], validation_audit["positive_return_percent"])
    )
    validation_negative = torch.tensor(
        cast(list[float], validation_audit["negative_return_percent"])
    )
    semantic_success = bool(
        torch.all(validation_mean <= torch.maximum(torch.ones(11), 3.0 * validation_se))
        and torch.all(validation_std >= 1.0)
        and torch.all(validation_positive >= 5.0)
        and torch.all(validation_negative >= 5.0)
        and validation_std[atm_index] >= 10.0
    )
    instruments = [
        {
            "index": index,
            "type": "call" if bool(option_is_call[index]) else "put",
            "log_strike_over_spot": float(torch.log(strike)),
            "strike_over_spot": float(strike),
            "current_w2_price_per_spot": float(
                current_w2_calls[index]
                if bool(option_is_call[index])
                else current_w2_calls[index] - 1.0 + strike
            ),
        }
        for index, strike in enumerate(normalized_strikes)
    ]
    report = {
        "schema": "flex-vector-q-existence-exact-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "experiment": "eleven-option-Q-existence",
        "objective": OBJECTIVE_ID,
        "market_date": model_date,
        "validation_index": arguments.selected_validation_index,
        "measure": {
            "P_used": False,
            "front_spot_marginal": "exact current front-week SANOS/lognormal mixture",
            "conditional_front_state_law": (
                "h1_next = h1_now + a + B*z_spot + b*d + sigma*epsilon"
            ),
            "full_real_support": True,
        },
        "instruments": instruments,
        "configuration": {
            "law": asdict(config),
            "fit_replications": arguments.fit_replications,
            "validation_replications": arguments.validation_replications,
            "paths_per_replication": arguments.spot_contexts * arguments.surface_paths,
            "fit_seed": arguments.fit_seed,
            "validation_seed": arguments.validation_seed,
            "initialization_seed": arguments.initialization_seed,
            "compiled_sanos_solver": not arguments.no_compile,
        },
        "training": {
            "epochs_completed": len(history),
            "stable_epochs": stable,
            "converged_fit_gate": stable >= arguments.stable_epochs,
            "stopped_for_plateau": stopped_for_plateau,
            "plateau_epochs": arguments.plateau_epochs,
            "minimum_relative_improvement": arguments.minimum_relative_improvement,
            "best_loss": best_loss,
            "history": history,
            "audit": fit_audit,
        },
        "independent_validation": validation_audit,
        "parameters": {
            "location_shift": law.location_shift.detach().cpu().tolist(),
            "spot_loading": law.spot_loading.detach().cpu().tolist(),
            "branch_direction": law.branch_direction.detach().cpu().tolist(),
            "state_noise_std": float(law.noise_std.detach()),
        },
        "sources": sources,
    }
    write_json_atomic(report, arguments.output)
    final_checkpoint = {
        **checkpoint,
        "status": "complete" if semantic_success else "failed_terminal",
        "report": str(arguments.output),
        "report_sha256": checkpoint_digest(arguments.output),
    }
    _atomic_torch_save(final_checkpoint, arguments.checkpoint)
    completion = arguments.output.with_name(f"{arguments.output.stem}-complete.json")
    write_json_atomic(
        {
            "schema": "flex-vector-q-existence-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
            "source_completion": str(arguments.joint_completion),
            "source_completion_sha256": sources["completion_sha256"],
        },
        completion,
    )
    print(json.dumps(validation_audit, indent=2), flush=True)
    if not semantic_success:
        raise RuntimeError("eleven-option Q-existence law failed independent validation")


def main() -> None:
    arguments = _parse_arguments()
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(main, program="run_vector_q_existence", clean_sigterm=True)
