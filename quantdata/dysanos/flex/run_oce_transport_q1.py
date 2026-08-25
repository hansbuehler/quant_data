"""Fit a conditional W2 power-exp OCE target and distil it into a sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch

from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.oce_transport import (
    ConditionalOCEConfig,
    SpotMarginalOCEAction,
    estimate_return_inverse_sqrt,
    finite_training_record,
    population_oce_transport_reverse_kl_loss,
    sample_conditional_paths,
    spot_marginal_oce_objective,
    w2_roll_returns,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
    WeeklyMeasureConfig,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    audit_front_week_consistency,
    audit_weekly_roll_line,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument(
        "--oce-action-checkpoint",
        type=Path,
        default=None,
        help=(
            "Validated completed OCE checkpoint used to start a fresh transport run "
            "without refitting the population action"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oce-epochs", type=int, default=2_500)
    parser.add_argument("--transport-epochs", type=int, default=1_500)
    parser.add_argument("--origins-per-batch", type=int, default=8)
    parser.add_argument("--spot-contexts-per-origin", type=int, default=4)
    parser.add_argument(
        "--audit-spot-contexts-per-origin",
        type=int,
        default=None,
        help="Optional larger stratified spot grid used only by fixed-seed audits",
    )
    parser.add_argument("--surface-paths-per-context", type=int, default=128)
    parser.add_argument("--oce-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--transport-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--minimum-learning-rate-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=25)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument("--minimum-return-rms", type=float, default=1.0e-4)
    parser.add_argument(
        "--maximum-preconditioned-action-norm",
        type=float,
        default=None,
        help=(
            "Optional smooth radial exposure budget in whitened return coordinates; "
            "use several fixed values to measure the W2-error/ESS frontier"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--audit-surface-paths", type=int, default=2_048)
    parser.add_argument("--full-audit-paths", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=7_310_421)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-transport",
        action="store_true",
        help="Compile the full SANOS transport graph (expensive experimental path)",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("The OCE transport experiment is restricted to GPU 0")
    for label in (
        "oce_epochs",
        "transport_epochs",
        "origins_per_batch",
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "checkpoint_every",
        "report_every",
        "audit_surface_paths",
        "full_audit_paths",
    ):
        if int(getattr(arguments, label)) <= 0:
            raise ValueError(f"{label} must be positive")
    for label in (
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "audit_surface_paths",
        "full_audit_paths",
    ):
        if int(getattr(arguments, label)) % 2:
            raise ValueError(f"{label} must be even")
    if (
        arguments.audit_spot_contexts_per_origin is not None
        and (
            arguments.audit_spot_contexts_per_origin <= 0
            or arguments.audit_spot_contexts_per_origin % 2
        )
    ):
        raise ValueError("audit_spot_contexts_per_origin must be positive and even")
    if (
        arguments.oce_learning_rate <= 0.0
        or arguments.transport_learning_rate <= 0.0
        or not 0.0 < arguments.minimum_learning_rate_fraction <= 1.0
        or arguments.gradient_clip <= 0.0
        or arguments.minimum_return_rms <= 0.0
        or (
            arguments.maximum_preconditioned_action_norm is not None
            and (
                not math.isfinite(arguments.maximum_preconditioned_action_norm)
                or arguments.maximum_preconditioned_action_norm <= 0.0
            )
        )
        or arguments.warmup_epochs < 0
        or arguments.warmup_epochs >= min(arguments.oce_epochs, arguments.transport_epochs)
    ):
        raise ValueError("invalid optimizer controls")


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _save_torch_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _json_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    """Return CLI arguments with filesystem paths made JSON-safe."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }


def _learning_rate_multiplier(
    epoch: int,
    epochs: int,
    minimum_fraction: float,
    warmup_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * (epoch + 1) / max(warmup_epochs, 1)
    progress = min(
        max((epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


def _active_transport_parameters(
    kernel: GenericWeeklyMeasureKernel,
) -> list[torch.nn.Parameter]:
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    result = list(kernel.context_networks[0].parameters())
    result.append(kernel.strict_lower_raw[0])
    for parameter in result:
        parameter.requires_grad_(True)
    return result


def _build_kernels(
    env: BaseEnvironment,
    dynamics: JointHierarchicalWeeklyDynamicsAdapter,
) -> tuple[WeeklyMeasureConfig, GenericWeeklyMeasureKernel, GenericWeeklyMeasureKernel]:
    config = WeeklyMeasureConfig(
        label="joint-hierarchical-200-powerexp-oce-transport-q1-v10",
        state_sizes_by_expiry=dynamics.reduced_sizes_by_expiry,
        hierarchical=True,
        context_network=FeedForwardConfig(
            activation="SiLU",
            num_depth=3,
            num_width=256,
        ),
    )
    reference = GenericWeeklyMeasureKernel(env, dynamics=dynamics, config=config)
    transport = GenericWeeklyMeasureKernel(env, dynamics=dynamics, config=config)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return config, reference, transport


def _draw_reference_returns(
    reference: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: Any,
    normalized_strikes: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        draw, log_abs_det = reference.step_first_line_with_log_abs_det(
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
        )
        if not bool(torch.all(log_abs_det == 0.0)):
            raise RuntimeError("reference P transport is not the identity")
        returns = w2_roll_returns(
            simulator,
            data,
            batch,
            draw.reduced_state,
            normalized_strikes,
        )
    return returns.reshape(batch.num_contexts, batch.surface_paths_per_context, -1)


def _estimate_preconditioner(
    reference: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    *,
    generator: torch.Generator,
    minimum_return_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    origins = min(64, len(data.reduced_states))
    indices = (
        torch.linspace(
            0,
            len(data.reduced_states) - 1,
            origins,
            device=data.reduced_states.device,
        )
        .round()
        .long()
    )
    batch = sample_conditional_paths(
        data,
        indices,
        spot_contexts_per_origin=4,
        surface_paths_per_context=128,
        generator=generator,
    )
    returns = _draw_reference_returns(reference, simulator, data, batch, normalized_strikes)
    return_rms = torch.sqrt(torch.mean(returns.square(), dim=(0, 1))).clamp_min(
        minimum_return_rms
    )
    inverse_sqrt = estimate_return_inverse_sqrt(
        returns,
        minimum_standard_deviation=minimum_return_rms,
    )
    return return_rms.detach(), inverse_sqrt.detach()


def _compile_oce_objective(
    model: SpotMarginalOCEAction,
    *,
    compile_objective: bool,
) -> Any:
    def objective(
        returns: torch.Tensor,
        origin_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, whitened_action = model(origin_state)
        oce, details = spot_marginal_oce_objective(returns, action)
        return (
            -oce,
            oce.detach(),
            torch.sqrt(torch.mean(details["mass_residual"].square())).detach(),
            torch.sqrt(torch.mean(details["action_residual"].square())).detach(),
            torch.mean(torch.abs(whitened_action)).detach(),
        )

    return (
        torch.compile(objective, dynamic=False, fullgraph=False) if compile_objective else objective
    )


def _compile_transport_objective(
    kernel: GenericWeeklyMeasureKernel,
    reference: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    oce_action: SpotMarginalOCEAction,
    *,
    compile_objective: bool,
) -> Any:
    def objective(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, details = population_oce_transport_reverse_kl_loss(
            kernel=kernel,
            reference_kernel=reference,
            simulator=simulator,
            data=data,
            batch=batch,
            normalized_strikes=normalized_strikes,
            oce_action=oce_action,
        )
        return (
            loss,
            torch.mean(details["log_abs_det"]).detach(),
            torch.std(details["log_abs_det"]).detach(),
        )

    return (
        torch.compile(objective, dynamic=False, fullgraph=False) if compile_objective else objective
    )


@torch.no_grad()
def _audit_oce_target(
    model: SpotMarginalOCEAction,
    reference: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    *,
    surface_paths_per_context: int,
    spot_contexts_per_origin: int,
    origins_per_batch: int,
    seed: int,
    label: str,
    maximum_origins: int | None = None,
) -> dict[str, Any]:
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    p_origin_moments: list[torch.Tensor] = []
    conditional_p_moments: list[torch.Tensor] = []
    target_origin_moments: list[torch.Tensor] = []
    conditional_target_moments: list[torch.Tensor] = []
    mass_residuals: list[torch.Tensor] = []
    effective_sample_sizes: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    translations: list[torch.Tensor] = []
    oce_values: list[torch.Tensor] = []
    spot_contexts = spot_contexts_per_origin
    num_origins = min(len(data.reduced_states), maximum_origins or len(data.reduced_states))
    for start in range(0, num_origins, origins_per_batch):
        stop = min(start + origins_per_batch, num_origins)
        indices = torch.arange(start, stop, device=data.reduced_states.device)
        batch = sample_conditional_paths(
            data,
            indices,
            spot_contexts_per_origin=spot_contexts,
            surface_paths_per_context=surface_paths_per_context,
            generator=generator,
        )
        returns = _draw_reference_returns(reference, simulator, data, batch, normalized_strikes)
        batch_origins = len(indices)
        grouped_returns = returns.reshape(
            batch_origins,
            spot_contexts,
            surface_paths_per_context,
            -1,
        )
        origin_state = batch.context_state[::spot_contexts]
        action, whitened_action = model(origin_state)
        _, details = spot_marginal_oce_objective(grouped_returns, action)
        translation = details["translation"]
        log_marginal = details["log_marginal"]
        weight = torch.softmax(log_marginal, dim=2)
        target_context = torch.sum(weight[..., None] * grouped_returns, dim=2)
        p_context = torch.mean(grouped_returns, dim=2)
        p_origin_moments.append(p_context.mean(dim=1).cpu())
        target_origin_moments.append(target_context.mean(dim=1).cpu())
        conditional_target_moments.append(target_context.reshape(-1, returns.shape[-1]).cpu())
        conditional_p_moments.append(p_context.reshape(-1, returns.shape[-1]).cpu())
        mass_residuals.append(details["mass_residual"].reshape(-1).cpu())
        effective_sample_sizes.append(
            (1.0 / torch.sum(weight.square(), dim=2)).reshape(-1).cpu()
        )
        actions.append(whitened_action.cpu())
        translations.append(translation.reshape(-1).cpu())
        oce_values.append(details["objective_by_origin"].cpu())
        if stop == num_origins or stop % 128 == 0:
            print(
                f"audit {label} origins={stop}/{num_origins}",
                flush=True,
            )
    p_moment = torch.cat(p_origin_moments)
    target_moment = torch.cat(target_origin_moments)
    conditional_target = torch.cat(conditional_target_moments)
    conditional_p = torch.cat(conditional_p_moments)
    mass = torch.cat(mass_residuals)
    ess = torch.cat(effective_sample_sizes)
    scaled_action = torch.cat(actions)
    action_norm = torch.linalg.vector_norm(scaled_action, dim=-1)
    translation = torch.cat(translations)
    return {
        "num_origins": num_origins,
        "spot_contexts_per_origin": spot_contexts,
        "surface_paths_per_context": surface_paths_per_context,
        "p_origin_moment_rmse_bps_spot": float(torch.sqrt(torch.mean(p_moment.square()))) * 1.0e4,
        "target_origin_moment_rmse_bps_spot": float(torch.sqrt(torch.mean(target_moment.square())))
        * 1.0e4,
        "target_origin_moment_maximum_abs_bps_spot": float(torch.max(torch.abs(target_moment)))
        * 1.0e4,
        "target_conditional_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(conditional_target.square()))
        )
        * 1.0e4,
        "p_conditional_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(conditional_p.square()))
        )
        * 1.0e4,
        "density_mass_rmse": float(torch.sqrt(torch.mean(mass.square()))),
        "density_mass_maximum_abs": float(torch.max(torch.abs(mass))),
        "minimum_marginal_effective_sample_size": float(torch.min(ess)),
        "q01_marginal_effective_sample_size": float(torch.quantile(ess, 0.01)),
        "q05_marginal_effective_sample_size": float(torch.quantile(ess, 0.05)),
        "median_marginal_effective_sample_size": float(torch.median(ess)),
        "fraction_marginal_effective_sample_size_below_5pct": float(
            torch.mean((ess < 0.05 * surface_paths_per_context).to(ess.dtype))
        ),
        "mean_oce": float(torch.mean(torch.cat(oce_values))),
        "mean_abs_preconditioned_action": float(torch.mean(torch.abs(scaled_action))),
        "maximum_abs_preconditioned_action": float(torch.max(torch.abs(scaled_action))),
        "mean_preconditioned_action_norm": float(torch.mean(action_norm)),
        "maximum_preconditioned_action_norm": float(torch.max(action_norm)),
        "maximum_abs_translation": float(torch.max(torch.abs(translation))),
    }


@torch.no_grad()
def _audit_transport_target_match(
    transport: GenericWeeklyMeasureKernel,
    oce_action: SpotMarginalOCEAction,
    reference: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    *,
    surface_paths_per_context: int,
    spot_contexts_per_origin: int,
    origins_per_batch: int,
    seed: int,
    label: str,
    maximum_origins: int | None = None,
) -> dict[str, Any]:
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    target_origin_moments: list[torch.Tensor] = []
    transport_origin_moments: list[torch.Tensor] = []
    log_determinants: list[torch.Tensor] = []
    minimum_derivative = math.inf
    spot_contexts = spot_contexts_per_origin
    num_origins = min(len(data.reduced_states), maximum_origins or len(data.reduced_states))
    for start in range(0, num_origins, origins_per_batch):
        stop = min(start + origins_per_batch, num_origins)
        indices = torch.arange(start, stop, device=data.reduced_states.device)
        batch = sample_conditional_paths(
            data,
            indices,
            spot_contexts_per_origin=spot_contexts,
            surface_paths_per_context=surface_paths_per_context,
            generator=generator,
        )
        reference_returns = _draw_reference_returns(
            reference, simulator, data, batch, normalized_strikes
        )
        batch_origins = len(indices)
        grouped_reference_returns = reference_returns.reshape(
            batch_origins,
            spot_contexts,
            surface_paths_per_context,
            -1,
        )
        origin_state = batch.context_state[::spot_contexts]
        action, _ = oce_action(origin_state)
        _, details = spot_marginal_oce_objective(grouped_reference_returns, action)
        weight = torch.softmax(details["log_marginal"], dim=2)
        target_context = torch.sum(
            weight[..., None] * grouped_reference_returns,
            dim=2,
        )
        draw, log_abs_det = transport.step_first_line_with_log_abs_det(
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
        )
        transported_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            draw.reduced_state,
            normalized_strikes,
        ).reshape(batch_origins, spot_contexts, surface_paths_per_context, -1)
        transport_context = torch.mean(transported_returns, dim=2)
        target_origin_moments.append(target_context.mean(dim=1).cpu())
        transport_origin_moments.append(transport_context.mean(dim=1).cpu())
        log_determinants.append(log_abs_det.cpu())
        minimum_derivative = min(
            minimum_derivative,
            float(
                transport.minimum_first_line_coordinate_derivative(
                    batch.current_state[:64],
                    batch.gross_spot_return[:64],
                ).cpu()
            ),
        )
        if not bool(
            torch.equal(
                draw.transformed_standard_normals[:, 0],
                batch.standard_normals[:, 0],
            )
        ):
            raise RuntimeError("OCE transport changed the implied-spot rank")
        if stop == num_origins or stop % 128 == 0:
            print(
                f"audit {label} origins={stop}/{num_origins}",
                flush=True,
            )
    target = torch.cat(target_origin_moments)
    transported = torch.cat(transport_origin_moments)
    difference = transported - target
    log_abs_det = torch.cat(log_determinants)
    return {
        "num_origins": num_origins,
        "spot_contexts_per_origin": spot_contexts,
        "surface_paths_per_context": surface_paths_per_context,
        "target_origin_moment_rmse_bps_spot": float(torch.sqrt(torch.mean(target.square())))
        * 1.0e4,
        "transport_origin_moment_rmse_bps_spot": float(torch.sqrt(torch.mean(transported.square())))
        * 1.0e4,
        "transport_to_target_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(difference.square()))
        )
        * 1.0e4,
        "transport_origin_moment_maximum_abs_bps_spot": float(torch.max(torch.abs(transported)))
        * 1.0e4,
        "mean_log_abs_det": float(torch.mean(log_abs_det)),
        "std_log_abs_det": float(torch.std(log_abs_det)),
        "minimum_coordinate_derivative": minimum_derivative,
        "spot_rank_preserved_exactly": True,
    }


def _checkpoint_payload(
    *,
    phase: str,
    arguments: argparse.Namespace,
    measure_config: WeeklyMeasureConfig,
    oce_config: ConditionalOCEConfig,
    return_rms: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    train_state_sha256: str,
    joint_completion_sha256: str,
    oce_action: SpotMarginalOCEAction,
    transport: GenericWeeklyMeasureKernel,
    epoch: int,
    histories: Mapping[str, list[dict[str, float]]],
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    return {
        "schema": "flex-oce-transport-q1-checkpoint-v10",
        "phase": phase,
        "arguments": _json_arguments(arguments),
        "measure_config": asdict(measure_config),
        "oce_config": asdict(oce_config),
        "return_rms": return_rms.detach().cpu(),
        "inverse_sqrt_second_moment": inverse_sqrt_second_moment.detach().cpu(),
        "train_state_sha256": train_state_sha256,
        "joint_completion_sha256": joint_completion_sha256,
        "oce_action_state_dict": oce_action.state_dict(),
        "transport_state_dict": transport.state_dict(),
        "epoch": epoch,
        "histories": dict(histories),
        "generator_state": generator.get_state(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
    }


def _oce_config_matches(
    serialized: object,
    expected: ConditionalOCEConfig,
) -> bool:
    """Accept pre-log-spot OCE checkpoints with the exact legacy defaults."""

    if not isinstance(serialized, Mapping):
        return False
    normalized = dict(serialized)
    normalized.setdefault("include_current_log_spot", False)
    normalized.setdefault("current_log_spot_mean", 0.0)
    normalized.setdefault("current_log_spot_std", 1.0)
    normalized.setdefault("preconditioned_action_bound_mode", "smooth_radial")
    normalized.setdefault("action_cap_excess_penalty_weight", 1.0)
    return normalized == asdict(expected)


def _restore_checkpoint(
    path: Path,
    *,
    arguments: argparse.Namespace,
    measure_config: WeeklyMeasureConfig,
    oce_config: ConditionalOCEConfig,
    return_rms: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    train_state_sha256: str,
    joint_completion_sha256: str,
    oce_action: SpotMarginalOCEAction,
    transport: GenericWeeklyMeasureKernel,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_arguments = _json_arguments(arguments)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "flex-oce-transport-q1-checkpoint-v10"
        or payload.get("arguments") != expected_arguments
        or payload.get("measure_config") != asdict(measure_config)
        or not _oce_config_matches(payload.get("oce_config"), oce_config)
        or payload.get("train_state_sha256") != train_state_sha256
        or payload.get("joint_completion_sha256") != joint_completion_sha256
        or not torch.allclose(
            cast(torch.Tensor, payload.get("return_rms")),
            return_rms.cpu(),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        or not torch.allclose(
            cast(torch.Tensor, payload.get("inverse_sqrt_second_moment")),
            inverse_sqrt_second_moment.cpu(),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    ):
        raise ValueError("Existing OCE transport checkpoint identity is invalid")
    oce_action.load_state_dict(cast(Mapping[str, torch.Tensor], payload["oce_action_state_dict"]))
    transport.load_state_dict(cast(Mapping[str, torch.Tensor], payload["transport_state_dict"]))
    return payload


def _load_oce_action_warm_start(
    path: Path,
    *,
    measure_config: WeeklyMeasureConfig,
    oce_config: ConditionalOCEConfig,
    return_rms: torch.Tensor,
    inverse_sqrt_second_moment: torch.Tensor,
    train_state_sha256: str,
    joint_completion_sha256: str,
    oce_action: SpotMarginalOCEAction,
) -> Mapping[str, Any]:
    """Restore an OCE action only after its durable convergence gate passed."""

    if not path.exists():
        raise FileNotFoundError(f"OCE action warm-start checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("OCE action warm-start checkpoint is not a mapping")
    source_arguments = payload.get("arguments")
    if (
        payload.get("schema") != "flex-oce-transport-q1-checkpoint-v10"
        or payload.get("phase") not in {"audit", "complete"}
        or not isinstance(source_arguments, Mapping)
        or payload.get("measure_config") != asdict(measure_config)
        or not _oce_config_matches(payload.get("oce_config"), oce_config)
        or payload.get("train_state_sha256") != train_state_sha256
        or payload.get("joint_completion_sha256") != joint_completion_sha256
        or not torch.allclose(
            cast(torch.Tensor, payload.get("return_rms")),
            return_rms.cpu(),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        or not torch.allclose(
            cast(torch.Tensor, payload.get("inverse_sqrt_second_moment")),
            inverse_sqrt_second_moment.cpu(),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    ):
        raise ValueError("OCE action warm-start checkpoint identity is invalid")
    gate_path = Path(str(source_arguments["report_directory"])) / "oce-target-gate.json"
    if not gate_path.exists():
        raise ValueError("OCE action warm start has no durable convergence gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    train_gate = gate.get("train")
    if not isinstance(train_gate, Mapping):
        raise ValueError("OCE action warm-start gate has no training audit")
    surface_paths = int(train_gate["surface_paths_per_context"])
    improvement = 1.0 - float(train_gate["target_origin_moment_rmse_bps_spot"]) / max(
        float(train_gate["p_origin_moment_rmse_bps_spot"]),
        np.finfo(float).tiny,
    )
    if (
        gate.get("schema") != "flex-oce-target-gate-v10"
        or gate.get("status") != "complete"
        or improvement < 0.01
        or float(train_gate["density_mass_rmse"]) > 0.25
        or float(train_gate["minimum_marginal_effective_sample_size"])
        < 0.05 * surface_paths
        or float(train_gate["maximum_abs_preconditioned_action"]) > 1_000.0
    ):
        raise ValueError("OCE action warm-start convergence gate is invalid or did not pass")
    oce_action.load_state_dict(
        cast(Mapping[str, torch.Tensor], payload["oce_action_state_dict"])
    )
    return payload


def run(arguments: argparse.Namespace) -> None:
    """Fit the conditional OCE target, distil it, and audit fresh Q paths."""

    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        train_log_spot,
        validation_dates,
        validation_states,
        _,
        joint_sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    dynamics = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(spot_fit.front_beta, dtype=env.fdtype, device=env.device),
        spot_orthogonal_std=spot_fit.orthogonal_std,
    )
    simulator = WeeklySanosSimulator(env, dynamics=dynamics, calibrated_model=model)
    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    train_data = precompute_measure_training_data(
        simulator, train_states, normalized_strikes, num_expiries=2
    )
    validation_data = precompute_measure_training_data(
        simulator, validation_states, normalized_strikes, num_expiries=2
    )
    measure_config, reference, transport = _build_kernels(env, dynamics)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(arguments.seed)
    return_rms, inverse_sqrt_second_moment = _estimate_preconditioner(
        reference,
        simulator,
        train_data,
        normalized_strikes,
        generator=generator,
        minimum_return_rms=arguments.minimum_return_rms,
    )
    oce_config = ConditionalOCEConfig(
        state_sizes_by_expiry=measure_config.state_sizes_by_expiry,
        num_instruments=len(normalized_strikes),
        maximum_preconditioned_action_norm=(
            arguments.maximum_preconditioned_action_norm
        ),
    )
    oce_action = SpotMarginalOCEAction(
        env,
        config=oce_config,
        inverse_sqrt_second_moment=inverse_sqrt_second_moment,
    )
    train_state_sha256 = _tensor_sha256(train_states)
    joint_completion_sha256 = checkpoint_digest(arguments.joint_completion)
    restored = _restore_checkpoint(
        arguments.checkpoint,
        arguments=arguments,
        measure_config=measure_config,
        oce_config=oce_config,
        return_rms=return_rms,
        inverse_sqrt_second_moment=inverse_sqrt_second_moment,
        train_state_sha256=train_state_sha256,
        joint_completion_sha256=joint_completion_sha256,
        oce_action=oce_action,
        transport=transport,
    )
    if restored is not None:
        # Use the canonical pilot reductions serialized by the first process.
        return_rms = cast(torch.Tensor, restored["return_rms"]).to(env.device)
        inverse_sqrt_second_moment = cast(
            torch.Tensor, restored["inverse_sqrt_second_moment"]
        ).to(env.device)
    histories: dict[str, list[dict[str, float]]] = {
        "oce": [],
        "transport": [],
    }
    phase = "oce_training"
    start_epoch = 0
    if restored is not None:
        phase = str(restored["phase"])
        start_epoch = int(restored["epoch"])
        histories = {
            key: list(value)
            for key, value in cast(
                Mapping[str, list[dict[str, float]]], restored["histories"]
            ).items()
        }
        generator.set_state(cast(torch.Tensor, restored["generator_state"]))
    elif arguments.oce_action_checkpoint is not None:
        source = _load_oce_action_warm_start(
            arguments.oce_action_checkpoint,
            measure_config=measure_config,
            oce_config=oce_config,
            return_rms=return_rms,
            inverse_sqrt_second_moment=inverse_sqrt_second_moment,
            train_state_sha256=train_state_sha256,
            joint_completion_sha256=joint_completion_sha256,
            oce_action=oce_action,
        )
        histories["oce"] = list(
            cast(Mapping[str, list[dict[str, float]]], source["histories"])["oce"]
        )
        phase = "oce_complete"
        _save_torch_atomic(
            _checkpoint_payload(
                phase=phase,
                arguments=arguments,
                measure_config=measure_config,
                oce_config=oce_config,
                return_rms=return_rms,
                inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                train_state_sha256=train_state_sha256,
                joint_completion_sha256=joint_completion_sha256,
                oce_action=oce_action,
                transport=transport,
                epoch=0,
                histories=histories,
                generator=generator,
                optimizer=None,
            ),
            arguments.checkpoint,
        )
    if phase == "complete":
        print("OCE transport checkpoint is already complete", flush=True)
    elif phase not in {
        "oce_training",
        "oce_complete",
        "transport_training",
        "audit",
    }:
        raise RuntimeError(f"OCE transport checkpoint has terminal phase {phase!r}")

    compile_objective = not arguments.no_compile and not arguments.smoke
    if phase == "oce_training":
        optimizer = torch.optim.Adam(oce_action.parameters(), lr=arguments.oce_learning_rate)
        if restored is not None and restored.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(cast(dict[str, Any], restored["optimizer_state_dict"]))
        objective = _compile_oce_objective(oce_action, compile_objective=compile_objective)
        oce_action.train()
        for epoch in range(start_epoch, arguments.oce_epochs):
            multiplier = _learning_rate_multiplier(
                epoch,
                arguments.oce_epochs,
                arguments.minimum_learning_rate_fraction,
                arguments.warmup_epochs,
            )
            optimizer.param_groups[0]["lr"] = arguments.oce_learning_rate * multiplier
            indices = torch.randint(
                len(train_data.reduced_states),
                (arguments.origins_per_batch,),
                device=env.device,
                generator=generator,
            )
            batch = sample_conditional_paths(
                train_data,
                indices,
                spot_contexts_per_origin=arguments.spot_contexts_per_origin,
                surface_paths_per_context=arguments.surface_paths_per_context,
                generator=generator,
            )
            returns = _draw_reference_returns(
                reference, simulator, train_data, batch, normalized_strikes
            ).reshape(
                arguments.origins_per_batch,
                arguments.spot_contexts_per_origin,
                arguments.surface_paths_per_context,
                -1,
            )
            origin_state = batch.context_state[:: arguments.spot_contexts_per_origin]
            optimizer.zero_grad(set_to_none=True)
            loss, oce, mass_rmse, moment_rmse, mean_action = objective(
                returns,
                origin_state,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                oce_action.parameters(), arguments.gradient_clip
            )
            optimizer.step()
            record = finite_training_record(
                epoch=float(epoch + 1),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                oce=float(oce.cpu()),
                density_mass_rmse=float(mass_rmse.cpu()),
                action_moment_rmse_bps_spot=float(moment_rmse.cpu()) * 1.0e4,
                mean_abs_preconditioned_action=float(mean_action.cpu()),
                gradient_norm=float(gradient_norm.detach().cpu()),
            )
            histories["oce"].append(record)
            if (epoch + 1) % arguments.report_every == 0 or epoch == start_epoch:
                print(
                    f"OCE epoch={epoch + 1}/{arguments.oce_epochs} "
                    f"oce={record['oce']:.7g} "
                    f"mass={record['density_mass_rmse']:.5g} "
                    f"moment_bps={record['action_moment_rmse_bps_spot']:.5g}",
                    flush=True,
                )
            if (epoch + 1) % arguments.checkpoint_every == 0:
                _save_torch_atomic(
                    _checkpoint_payload(
                        phase="oce_training",
                        arguments=arguments,
                        measure_config=measure_config,
                        oce_config=oce_config,
                        return_rms=return_rms,
                        inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                        train_state_sha256=train_state_sha256,
                        joint_completion_sha256=joint_completion_sha256,
                        oce_action=oce_action,
                        transport=transport,
                        epoch=epoch + 1,
                        histories=histories,
                        generator=generator,
                        optimizer=optimizer,
                    ),
                    arguments.checkpoint,
                )
        phase = "oce_complete"
        start_epoch = 0
        _save_torch_atomic(
            _checkpoint_payload(
                phase=phase,
                arguments=arguments,
                measure_config=measure_config,
                oce_config=oce_config,
                return_rms=return_rms,
                inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                train_state_sha256=train_state_sha256,
                joint_completion_sha256=joint_completion_sha256,
                oce_action=oce_action,
                transport=transport,
                epoch=0,
                histories=histories,
                generator=generator,
                optimizer=None,
            ),
            arguments.checkpoint,
        )

    for parameter in oce_action.parameters():
        parameter.requires_grad_(False)
    oce_action.eval()
    oce_gate_path = arguments.report_directory / "oce-target-gate.json"
    audit_spot_contexts = (
        arguments.audit_spot_contexts_per_origin
        or arguments.spot_contexts_per_origin
    )
    if phase == "oce_complete":
        gate_paths = min(arguments.audit_surface_paths, 512)
        oce_gate = {
            "schema": "flex-oce-target-gate-v10",
            "status": "complete",
            "train": _audit_oce_target(
                oce_action,
                reference,
                simulator,
                train_data,
                normalized_strikes,
                surface_paths_per_context=gate_paths,
                spot_contexts_per_origin=audit_spot_contexts,
                origins_per_batch=1,
                seed=arguments.seed + 70_000,
                label="train-oce-gate",
                maximum_origins=16 if arguments.smoke else None,
            ),
            "validation_2025": _audit_oce_target(
                oce_action,
                reference,
                simulator,
                validation_data,
                normalized_strikes,
                surface_paths_per_context=gate_paths,
                spot_contexts_per_origin=audit_spot_contexts,
                origins_per_batch=1,
                seed=arguments.seed + 80_000,
                label="validation-oce-gate",
                maximum_origins=16 if arguments.smoke else None,
            ),
        }
        write_json_atomic(oce_gate, oce_gate_path)
        train_gate = cast(Mapping[str, Any], oce_gate["train"])
        improvement = 1.0 - float(train_gate["target_origin_moment_rmse_bps_spot"]) / max(
            float(train_gate["p_origin_moment_rmse_bps_spot"]),
            np.finfo(float).tiny,
        )
        if not arguments.smoke and (
            improvement < 0.01
            or float(train_gate["density_mass_rmse"]) > 0.25
            or float(train_gate["minimum_marginal_effective_sample_size"])
            < 0.05 * gate_paths
            or float(train_gate["maximum_abs_preconditioned_action"]) > 1_000.0
        ):
            _save_torch_atomic(
                _checkpoint_payload(
                    phase="failed_oce_convergence",
                    arguments=arguments,
                    measure_config=measure_config,
                    oce_config=oce_config,
                    return_rms=return_rms,
                    inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                    train_state_sha256=train_state_sha256,
                    joint_completion_sha256=joint_completion_sha256,
                    oce_action=oce_action,
                    transport=transport,
                    epoch=0,
                    histories=histories,
                    generator=generator,
                    optimizer=None,
                ),
                arguments.checkpoint,
            )
            raise RuntimeError(
                "OCE target failed convergence before transport: "
                f"origin moment improvement={improvement:.3%}, "
                f"density mass RMSE={float(train_gate['density_mass_rmse']):.6g}, "
                "minimum ESS="
                f"{float(train_gate['minimum_marginal_effective_sample_size']):.6g}, "
                "maximum action="
                f"{float(train_gate['maximum_abs_preconditioned_action']):.6g}"
            )
    elif phase == "transport_training" and not oce_gate_path.exists():
        raise ValueError("Transport checkpoint has no durable OCE convergence gate")
    if phase in {"oce_complete", "transport_training"}:
        active = _active_transport_parameters(transport)
        optimizer = torch.optim.Adam(active, lr=arguments.transport_learning_rate)
        if phase == "transport_training" and restored is not None:
            optimizer.load_state_dict(cast(dict[str, Any], restored["optimizer_state_dict"]))
            start_epoch = int(restored["epoch"])
        else:
            start_epoch = 0
        objective = _compile_transport_objective(
            transport,
            reference,
            simulator,
            train_data,
            normalized_strikes,
            oce_action,
            compile_objective=compile_objective and arguments.compile_transport,
        )
        transport.train()
        for epoch in range(start_epoch, arguments.transport_epochs):
            multiplier = _learning_rate_multiplier(
                epoch,
                arguments.transport_epochs,
                arguments.minimum_learning_rate_fraction,
                arguments.warmup_epochs,
            )
            optimizer.param_groups[0]["lr"] = arguments.transport_learning_rate * multiplier
            indices = torch.randint(
                len(train_data.reduced_states),
                (arguments.origins_per_batch,),
                device=env.device,
                generator=generator,
            )
            batch = sample_conditional_paths(
                train_data,
                indices,
                spot_contexts_per_origin=arguments.spot_contexts_per_origin,
                surface_paths_per_context=arguments.surface_paths_per_context,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, mean_logdet, std_logdet = objective(batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(active, arguments.gradient_clip)
            optimizer.step()
            record = finite_training_record(
                epoch=float(epoch + 1),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                reverse_kl_up_to_constant=float(loss.detach().cpu()),
                mean_log_abs_det=float(mean_logdet.cpu()),
                std_log_abs_det=float(std_logdet.cpu()),
                gradient_norm=float(gradient_norm.detach().cpu()),
            )
            histories["transport"].append(record)
            if (epoch + 1) % arguments.report_every == 0 or epoch == start_epoch:
                print(
                    f"transport epoch={epoch + 1}/{arguments.transport_epochs} "
                    f"reverse_kl={record['reverse_kl_up_to_constant']:.7g} "
                    f"logdet={record['mean_log_abs_det']:.5g}",
                    flush=True,
                )
            if (epoch + 1) % arguments.checkpoint_every == 0:
                _save_torch_atomic(
                    _checkpoint_payload(
                        phase="transport_training",
                        arguments=arguments,
                        measure_config=measure_config,
                        oce_config=oce_config,
                        return_rms=return_rms,
                        inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                        train_state_sha256=train_state_sha256,
                        joint_completion_sha256=joint_completion_sha256,
                        oce_action=oce_action,
                        transport=transport,
                        epoch=epoch + 1,
                        histories=histories,
                        generator=generator,
                        optimizer=optimizer,
                    ),
                    arguments.checkpoint,
                )
        _save_torch_atomic(
            _checkpoint_payload(
                phase="audit",
                arguments=arguments,
                measure_config=measure_config,
                oce_config=oce_config,
                return_rms=return_rms,
                inverse_sqrt_second_moment=inverse_sqrt_second_moment,
                train_state_sha256=train_state_sha256,
                joint_completion_sha256=joint_completion_sha256,
                oce_action=oce_action,
                transport=transport,
                epoch=0,
                histories=histories,
                generator=generator,
                optimizer=None,
            ),
            arguments.checkpoint,
        )
        phase = "audit"

    transport.eval()
    audit_paths = (
        min(arguments.audit_surface_paths, 512)
        if arguments.smoke
        else arguments.audit_surface_paths
    )
    train_oce_audit = _audit_oce_target(
        oce_action,
        reference,
        simulator,
        train_data,
        normalized_strikes,
        surface_paths_per_context=audit_paths,
        spot_contexts_per_origin=audit_spot_contexts,
        origins_per_batch=1,
        seed=arguments.seed + 100_000,
        label="train-oce-target",
        maximum_origins=16 if arguments.smoke else None,
    )
    validation_oce_audit = _audit_oce_target(
        oce_action,
        reference,
        simulator,
        validation_data,
        normalized_strikes,
        surface_paths_per_context=audit_paths,
        spot_contexts_per_origin=audit_spot_contexts,
        origins_per_batch=1,
        seed=arguments.seed + 200_000,
        label="validation-oce-target",
        maximum_origins=16 if arguments.smoke else None,
    )
    train_transport_audit = _audit_transport_target_match(
        transport,
        oce_action,
        reference,
        simulator,
        train_data,
        normalized_strikes,
        surface_paths_per_context=audit_paths,
        spot_contexts_per_origin=audit_spot_contexts,
        origins_per_batch=1,
        seed=arguments.seed + 300_000,
        label="train-transport-target",
        maximum_origins=16 if arguments.smoke else None,
    )
    validation_transport_audit = _audit_transport_target_match(
        transport,
        oce_action,
        reference,
        simulator,
        validation_data,
        normalized_strikes,
        surface_paths_per_context=audit_paths,
        spot_contexts_per_origin=audit_spot_contexts,
        origins_per_batch=1,
        seed=arguments.seed + 400_000,
        label="validation-transport-target",
        maximum_origins=16 if arguments.smoke else None,
    )
    front_week_audit = audit_front_week_consistency(
        validation_data,
        normalized_strikes,
        paths_per_origin=(128 if arguments.smoke else arguments.full_audit_paths),
        origins_per_batch=4,
        seed=arguments.seed + 500_000,
    )
    p_w2_audit = audit_weekly_roll_line(
        reference,
        simulator,
        validation_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=(128 if arguments.smoke else arguments.full_audit_paths),
        origins_per_batch=2,
        seed=arguments.seed + 600_000,
    )
    fresh_w2_audit = audit_weekly_roll_line(
        transport,
        simulator,
        validation_data,
        normalized_strikes,
        current_expiry_index=1,
        paths_per_origin=(128 if arguments.smoke else arguments.full_audit_paths),
        origins_per_batch=2,
        seed=arguments.seed + 600_000,
    )
    if (
        not arguments.smoke
        and train_oce_audit["target_origin_moment_rmse_bps_spot"]
        >= train_oce_audit["p_origin_moment_rmse_bps_spot"]
    ):
        raise RuntimeError("OCE target did not improve training W2 moments")
    if validation_transport_audit["minimum_coordinate_derivative"] <= 0.0:
        raise RuntimeError("OCE transport lost full-support monotonicity")

    arguments.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_directory / "oce-transport-q1.json"
    completion_path = arguments.report_directory / "oce-transport-q1-complete.json"
    report = {
        "schema": "flex-oce-transport-q1-report-v10",
        "status": "complete",
        "mode": "smoke" if arguments.smoke else "production",
        "scope": (
            "conditional full-action power-exp OCE over eleven W2 calls, "
            "followed by a first-line full-support latent transport"
        ),
        "oce_translation": (
            "one population OCE action is learned from the current state and "
            "shared across spot ranks; cash translation is solved exactly per spot rank"
        ),
        "oce_exposure_policy": (
            "the optional fixed radial budget is applied in globally whitened return "
            "coordinates; it defines an explicit W2-error versus density-ESS frontier"
        ),
        "spot_policy": (
            "the implied front-week spot rank is unchanged; the redundant spot "
            "action is omitted from the conditional OCE"
        ),
        "train_dates": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "validation_dates": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "date_weighting": "equal dates with fixed contexts and paths per sampled date",
        "spot_integration": (
            "fresh randomized one-dimensional stratification; surface normals "
            "remain freshly antithetic within every spot context"
        ),
        "arguments": _json_arguments(arguments),
        "measure_config": asdict(measure_config),
        "oce_config": asdict(oce_config),
        "oce_action_warm_start": (
            None
            if arguments.oce_action_checkpoint is None
            else {
                "checkpoint": str(arguments.oce_action_checkpoint),
                "checkpoint_sha256": checkpoint_digest(arguments.oce_action_checkpoint),
            }
        ),
        "normalized_strikes": normalized_strikes.detach().cpu().tolist(),
        "return_rms": return_rms.detach().cpu().tolist(),
        "oce_target_gate": (
            json.loads(oce_gate_path.read_text(encoding="utf-8"))
            if oce_gate_path.exists()
            else None
        ),
        "ad_hoc_choices": {
            "minimum_return_rms": arguments.minimum_return_rms,
            "maximum_preconditioned_action_norm": (
                arguments.maximum_preconditioned_action_norm
            ),
            "reason": (
                "prevents extreme optimizer scaling for nearly deterministic "
                "conditional far-OTM call returns; it does not cap the action"
            ),
            "selective_compilation": (
                "the population OCE objective is compiled in production unless "
                "--no-compile; the full transport-plus-SANOS graph remains eager "
                "after a bounded compile probe exceeded four minutes"
            ),
        },
        "histories": histories,
        "audits": {
            "train_oce_target": train_oce_audit,
            "validation_2025_oce_target": validation_oce_audit,
            "train_transport_target_match": train_transport_audit,
            "validation_2025_transport_target_match": validation_transport_audit,
            "validation_2025_front_week": front_week_audit,
            "validation_2025_p_w2": p_w2_audit,
            "validation_2025_fresh_w2": fresh_w2_audit,
        },
        "sources": joint_sources,
    }
    write_json_atomic(report, report_path)
    _save_torch_atomic(
        _checkpoint_payload(
            phase="complete",
            arguments=arguments,
            measure_config=measure_config,
            oce_config=oce_config,
            return_rms=return_rms,
            inverse_sqrt_second_moment=inverse_sqrt_second_moment,
            train_state_sha256=train_state_sha256,
            joint_completion_sha256=joint_completion_sha256,
            oce_action=oce_action,
            transport=transport,
            epoch=0,
            histories=histories,
            generator=generator,
            optimizer=None,
        ),
        arguments.checkpoint,
    )
    write_json_atomic(
        {
            "schema": "flex-oce-transport-q1-completion-v10",
            "status": "complete",
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        },
        completion_path,
    )


def main() -> None:
    """CLI entry point with durable exception reporting."""

    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_oce_transport_q1")
