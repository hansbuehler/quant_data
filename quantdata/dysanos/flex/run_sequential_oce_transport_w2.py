"""Apply three sequential W2 OCE corrections through sampled transports."""

from __future__ import annotations

import argparse
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
    ConditionalPathBatch,
    SpotConditionedFirstLineTransport,
    SpotMarginalOCEAction,
    apply_first_line_transport_sequence,
    estimate_return_inverse_sqrt,
    finite_training_record,
    sample_conditional_paths,
    sequential_population_oce_transport_reverse_kl_loss,
    spot_marginal_oce_objective,
    w2_roll_returns,
)
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import _load_inputs
from quantdata.dysanos.flex.run_oce_transport_q1 import (
    _active_transport_parameters,
    _build_kernels,
    _estimate_preconditioner,
    _json_arguments,
    _learning_rate_multiplier,
    _load_oce_action_warm_start,
    _save_torch_atomic,
    _tensor_sha256,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--source-oce-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-rounds", type=int, default=4)
    parser.add_argument("--oce-epochs-per-round", type=int, default=1_500)
    parser.add_argument("--transport-epochs-per-round", type=int, default=1_500)
    parser.add_argument("--origins-per-batch", type=int, default=3)
    parser.add_argument("--spot-contexts-per-origin", type=int, default=16)
    parser.add_argument("--surface-paths-per-context", type=int, default=128)
    parser.add_argument("--audit-spot-contexts-per-origin", type=int, default=64)
    parser.add_argument("--audit-surface-paths-per-context", type=int, default=128)
    parser.add_argument("--oce-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--transport-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--minimum-learning-rate-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=25)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--maximum-preconditioned-action-norm", type=float, default=1.0)
    parser.add_argument("--action-cap-excess-penalty-weight", type=float, default=1.0)
    parser.add_argument("--minimum-return-rms", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--minimum-train-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-validation-degradation", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=8_240_731)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("sequential OCE transport is restricted to GPU0")
    positive_integer_fields = (
        "total_rounds",
        "oce_epochs_per_round",
        "transport_epochs_per_round",
        "origins_per_batch",
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "audit_spot_contexts_per_origin",
        "audit_surface_paths_per_context",
        "checkpoint_every",
        "report_every",
    )
    if any(int(getattr(arguments, field)) <= 0 for field in positive_integer_fields):
        raise ValueError("sequential OCE integer controls must be positive")
    if arguments.total_rounds < 2:
        raise ValueError("sequential OCE needs at least one incremental round")
    for field in (
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "audit_spot_contexts_per_origin",
        "audit_surface_paths_per_context",
    ):
        if int(getattr(arguments, field)) % 2:
            raise ValueError(f"{field} must be even for antithetic sampling")
    finite_positive = (
        arguments.oce_learning_rate,
        arguments.transport_learning_rate,
        arguments.gradient_clip,
        arguments.maximum_preconditioned_action_norm,
        arguments.action_cap_excess_penalty_weight,
        arguments.minimum_return_rms,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
        raise ValueError("sequential OCE optimizer controls must be finite-positive")
    if not 0.0 < arguments.minimum_learning_rate_fraction <= 1.0:
        raise ValueError("minimum learning-rate fraction must lie in (0,1]")
    if (
        arguments.warmup_epochs < 0
        or arguments.warmup_epochs
        >= min(arguments.oce_epochs_per_round, arguments.transport_epochs_per_round)
        or arguments.minimum_train_improvement < 0.0
        or arguments.maximum_validation_degradation < 0.0
    ):
        raise ValueError("invalid sequential OCE schedule or gate")


def _fixed_origin_order(rows: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(rows, generator=generator).to(device=device)


def _fixed_batch(
    data: WeeklyMeasureTrainingData,
    origin_order: torch.Tensor,
    batch_index: int,
    *,
    origins_per_batch: int,
    spot_contexts_per_origin: int,
    surface_paths_per_context: int,
    seed: int,
) -> tuple[torch.Tensor, ConditionalPathBatch]:
    if len(origin_order) % origins_per_batch:
        raise ValueError("fixed training order must divide exactly into origin batches")
    num_batches = len(origin_order) // origins_per_batch
    selected = batch_index % num_batches
    start = selected * origins_per_batch
    indices = origin_order[start : start + origins_per_batch]
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed + selected)
    return indices, sample_conditional_paths(
        data,
        indices,
        spot_contexts_per_origin=spot_contexts_per_origin,
        surface_paths_per_context=surface_paths_per_context,
        generator=generator,
    )


def _path_log_spot(
    origin_log_spot: torch.Tensor,
    batch: ConditionalPathBatch,
) -> torch.Tensor:
    context = torch.repeat_interleave(
        origin_log_spot,
        batch.spot_contexts_per_origin,
    )
    return torch.repeat_interleave(context, batch.surface_paths_per_context)


@torch.no_grad()
def _draw_sequence_returns(
    transports: list[GenericWeeklyMeasureKernel],
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    batch: ConditionalPathBatch,
    normalized_strikes: torch.Tensor,
    origin_log_spot: torch.Tensor,
) -> torch.Tensor:
    draw, _ = apply_first_line_transport_sequence(
        transports,
        batch.current_state,
        batch.gross_spot_return,
        batch.standard_normals,
        _path_log_spot(origin_log_spot, batch),
    )
    return w2_roll_returns(
        simulator,
        data,
        batch,
        draw.reduced_state,
        normalized_strikes,
    ).reshape(
        batch.num_origins,
        batch.spot_contexts_per_origin,
        batch.surface_paths_per_context,
        -1,
    )


@torch.no_grad()
def _estimate_current_preconditioner(
    transports: list[GenericWeeklyMeasureKernel],
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    current_log_spot: torch.Tensor,
    normalized_strikes: torch.Tensor,
    *,
    seed: int,
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
    generator = torch.Generator(device=data.reduced_states.device)
    generator.manual_seed(seed)
    batch = sample_conditional_paths(
        data,
        indices,
        spot_contexts_per_origin=4,
        surface_paths_per_context=128,
        generator=generator,
    )
    returns = _draw_sequence_returns(
        transports,
        simulator,
        data,
        batch,
        normalized_strikes,
        current_log_spot[indices],
    )
    return_rms = torch.sqrt(torch.mean(returns.square(), dim=(0, 1, 2))).clamp_min(
        minimum_return_rms
    )
    inverse_sqrt = estimate_return_inverse_sqrt(
        returns.reshape(origins * 4, 128, -1),
        minimum_standard_deviation=minimum_return_rms,
    )
    return return_rms.detach(), inverse_sqrt.detach()


def _compile_oce_objective(
    action_model: SpotMarginalOCEAction,
    *,
    compile_objective: bool,
) -> Any:
    def objective(
        returns: torch.Tensor,
        origin_state: torch.Tensor,
        origin_log_spot: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action, whitened, cap_penalty, maximum_cap_excess = (
            action_model.forward_with_cap_penalty(origin_state, origin_log_spot)
        )
        oce, details = spot_marginal_oce_objective(returns, action)
        return (
            -oce + cap_penalty,
            oce.detach(),
            torch.sqrt(torch.mean(details["action_residual"].square())).detach(),
            torch.mean(torch.linalg.vector_norm(whitened, dim=-1)).detach(),
            cap_penalty.detach(),
            maximum_cap_excess.detach(),
        )

    return (
        torch.compile(objective, dynamic=False, fullgraph=False)
        if compile_objective
        else objective
    )


@torch.no_grad()
def _audit_increment(
    *,
    current_transports: list[GenericWeeklyMeasureKernel],
    incremental_transport: SpotConditionedFirstLineTransport,
    action_model: SpotMarginalOCEAction,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    current_log_spot: torch.Tensor,
    normalized_strikes: torch.Tensor,
    spot_contexts_per_origin: int,
    surface_paths_per_context: int,
    seed: int,
    label: str,
    maximum_origins: int | None = None,
) -> dict[str, Any]:
    num_origins = min(len(data.reduced_states), maximum_origins or len(data.reduced_states))
    current_moments: list[torch.Tensor] = []
    target_moments: list[torch.Tensor] = []
    transported_moments: list[torch.Tensor] = []
    objectives: list[torch.Tensor] = []
    minimum_derivative = float("inf")
    for origin in range(num_origins):
        indices = torch.tensor((origin,), dtype=torch.long, device=data.reduced_states.device)
        generator = torch.Generator(device=data.reduced_states.device)
        generator.manual_seed(seed + origin)
        batch = sample_conditional_paths(
            data,
            indices,
            spot_contexts_per_origin=spot_contexts_per_origin,
            surface_paths_per_context=surface_paths_per_context,
            generator=generator,
        )
        origin_spot = current_log_spot[indices]
        current_returns = _draw_sequence_returns(
            current_transports,
            simulator,
            data,
            batch,
            normalized_strikes,
            origin_spot,
        )
        action, _ = action_model(batch.context_state[:1], origin_spot)
        objective, target = spot_marginal_oce_objective(current_returns, action)
        probability = torch.softmax(target["log_marginal"], dim=2)
        target_moment = torch.mean(
            torch.sum(probability[..., None] * current_returns, dim=2),
            dim=1,
        )
        transported_returns = _draw_sequence_returns(
            [*current_transports, incremental_transport],
            simulator,
            data,
            batch,
            normalized_strikes,
            origin_spot,
        )
        current_moments.append(torch.mean(current_returns, dim=(1, 2)).cpu())
        target_moments.append(target_moment.cpu())
        transported_moments.append(torch.mean(transported_returns, dim=(1, 2)).cpu())
        objectives.append(objective.reshape(1).cpu())
        derivative = incremental_transport.minimum_first_line_coordinate_derivative(
            batch.current_state,
            batch.gross_spot_return,
            _path_log_spot(origin_spot, batch),
        )
        minimum_derivative = min(minimum_derivative, float(derivative.cpu()))
        if origin + 1 == num_origins or (origin + 1) % 128 == 0:
            print(f"audit {label} origins={origin + 1}/{num_origins}", flush=True)

    def summarize(values: list[torch.Tensor]) -> tuple[float, float]:
        tensor = torch.cat(values)
        return (
            float(torch.sqrt(torch.mean(tensor.square()))) * 1.0e4,
            float(torch.max(torch.abs(tensor))) * 1.0e4,
        )

    current_rmse, current_maximum = summarize(current_moments)
    target_rmse, target_maximum = summarize(target_moments)
    transported_rmse, transported_maximum = summarize(transported_moments)
    return {
        "num_origins": num_origins,
        "spot_contexts_per_origin": spot_contexts_per_origin,
        "surface_paths_per_context": surface_paths_per_context,
        "current_origin_moment_rmse_bps_spot": current_rmse,
        "current_origin_moment_maximum_abs_bps_spot": current_maximum,
        "oce_target_origin_moment_rmse_bps_spot": target_rmse,
        "oce_target_origin_moment_maximum_abs_bps_spot": target_maximum,
        "transported_origin_moment_rmse_bps_spot": transported_rmse,
        "transported_origin_moment_maximum_abs_bps_spot": transported_maximum,
        "mean_incremental_oce_objective": float(torch.mean(torch.cat(objectives))),
        "minimum_incremental_coordinate_derivative": minimum_derivative,
        "path_measure": "equal-weight direct draws from each transported law",
    }


def _round_gate(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    arguments: argparse.Namespace,
) -> tuple[bool, dict[str, float | bool]]:
    train_before = float(train["current_origin_moment_rmse_bps_spot"])
    train_after = float(train["transported_origin_moment_rmse_bps_spot"])
    validation_before = float(validation["current_origin_moment_rmse_bps_spot"])
    validation_after = float(validation["transported_origin_moment_rmse_bps_spot"])
    train_improvement = 1.0 - train_after / max(train_before, np.finfo(float).tiny)
    validation_change = validation_after / max(
        validation_before, np.finfo(float).tiny
    ) - 1.0
    derivative_positive = (
        float(train["minimum_incremental_coordinate_derivative"]) > 0.0
        and float(validation["minimum_incremental_coordinate_derivative"]) > 0.0
    )
    passed = (
        train_improvement > arguments.minimum_train_improvement
        and validation_change <= arguments.maximum_validation_degradation
        and derivative_positive
    )
    return passed, {
        "train_improvement": train_improvement,
        "validation_change": validation_change,
        "derivative_positive": derivative_positive,
    }


def _checkpoint_payload(
    *,
    arguments: argparse.Namespace,
    source_sha256: str,
    train_state_sha256: str,
    joint_completion_sha256: str,
    transports: list[GenericWeeklyMeasureKernel],
    phase: str,
    active_round_index: int,
    accepted_rounds: int,
    epoch: int,
    action_model: SpotMarginalOCEAction | None,
    return_rms: torch.Tensor | None,
    inverse_sqrt: torch.Tensor | None,
    histories: list[dict[str, list[dict[str, float]]]],
    audits: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema": "flex-sequential-oce-transport-w2-checkpoint-v1",
        "arguments": _json_arguments(arguments),
        "source_oce_checkpoint_sha256": source_sha256,
        "train_state_sha256": train_state_sha256,
        "joint_completion_sha256": joint_completion_sha256,
        "phase": phase,
        "active_round_index": active_round_index,
        "accepted_rounds": accepted_rounds,
        "epoch": epoch,
        "transport_state_dicts": [transport.state_dict() for transport in transports],
        "action_config": None if action_model is None else asdict(action_model.config),
        "action_state_dict": None if action_model is None else action_model.state_dict(),
        "return_rms": None if return_rms is None else return_rms.detach().cpu(),
        "inverse_sqrt_second_moment": (
            None if inverse_sqrt is None else inverse_sqrt.detach().cpu()
        ),
        "histories": histories,
        "audits": audits,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "stop_reason": stop_reason,
    }


def _load_checkpoint(
    path: Path,
    *,
    arguments: argparse.Namespace,
    source_sha256: str,
    train_state_sha256: str,
    joint_completion_sha256: str,
    transports: list[GenericWeeklyMeasureKernel],
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "flex-sequential-oce-transport-w2-checkpoint-v1"
        or payload.get("arguments") != _json_arguments(arguments)
        or payload.get("source_oce_checkpoint_sha256") != source_sha256
        or payload.get("train_state_sha256") != train_state_sha256
        or payload.get("joint_completion_sha256") != joint_completion_sha256
    ):
        raise ValueError("sequential OCE transport checkpoint identity is invalid")
    states = cast(list[Mapping[str, torch.Tensor]], payload["transport_state_dicts"])
    if len(states) != len(transports):
        raise ValueError("sequential checkpoint has the wrong transport count")
    for transport, state in zip(transports, states, strict=True):
        transport.load_state_dict(state)
    return payload


def _source_action_config(payload: Mapping[str, Any]) -> ConditionalOCEConfig:
    serialized = cast(Mapping[str, Any], payload["oce_config"])
    network = FeedForwardConfig(**cast(dict[str, Any], serialized["network"]))
    return ConditionalOCEConfig(
        state_sizes_by_expiry=tuple(serialized["state_sizes_by_expiry"]),
        num_instruments=int(serialized["num_instruments"]),
        maximum_preconditioned_action_norm=cast(
            float | None, serialized["maximum_preconditioned_action_norm"]
        ),
        network=network,
    )


def run(arguments: argparse.Namespace) -> None:
    """Train sequential sampled-law OCE transports with common random numbers."""

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
        validation_log_spot,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    if len(train_states) % arguments.origins_per_batch:
        raise ValueError(
            "origins-per-batch must divide the number of training dates exactly "
            "to preserve equal-date weighting with a fixed scenario bank"
        )
    train_log_spot_tensor = torch.as_tensor(
        train_log_spot, dtype=env.fdtype, device=env.device
    )
    validation_log_spot_tensor = torch.as_tensor(
        validation_log_spot, dtype=env.fdtype, device=env.device
    )
    log_spot_mean = float(np.mean(train_log_spot))
    log_spot_std = float(np.std(train_log_spot, ddof=0))
    if not math.isfinite(log_spot_std) or log_spot_std <= 0.0:
        raise ValueError("training log-spot normalization is invalid")
    spot_fit = fit_joint_spot_correlation_under_p(
        model,
        train_states.detach().cpu().double().numpy(),
        train_log_spot,
        train_dates,
    )
    dynamics = JointHierarchicalWeeklyDynamicsAdapter(
        env,
        model=model,
        spot_front_beta=torch.as_tensor(
            spot_fit.front_beta, dtype=env.fdtype, device=env.device
        ),
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
    measure_config, p_reference, accepted_transport = _build_kernels(env, dynamics)
    source_payload = torch.load(
        arguments.source_oce_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(source_payload, Mapping):
        raise ValueError("source OCE transport checkpoint is not a mapping")
    source_config = _source_action_config(source_payload)
    source_generator = torch.Generator(device=env.device)
    source_seed = int(cast(Mapping[str, Any], source_payload["arguments"])["seed"])
    source_generator.manual_seed(source_seed)
    source_return_rms, source_inverse_sqrt = _estimate_preconditioner(
        p_reference,
        simulator,
        train_data,
        normalized_strikes,
        generator=source_generator,
        minimum_return_rms=arguments.minimum_return_rms,
    )
    source_action = SpotMarginalOCEAction(
        env,
        config=source_config,
        inverse_sqrt_second_moment=source_inverse_sqrt,
    )
    train_state_sha256 = _tensor_sha256(train_states)
    joint_completion_sha256 = checkpoint_digest(arguments.joint_completion)
    validated_source = _load_oce_action_warm_start(
        arguments.source_oce_checkpoint,
        measure_config=measure_config,
        oce_config=source_config,
        return_rms=source_return_rms,
        inverse_sqrt_second_moment=source_inverse_sqrt,
        train_state_sha256=train_state_sha256,
        joint_completion_sha256=joint_completion_sha256,
        oce_action=source_action,
    )
    accepted_transport.load_state_dict(
        cast(Mapping[str, torch.Tensor], validated_source["transport_state_dict"])
    )
    transports: list[GenericWeeklyMeasureKernel] = [accepted_transport]
    transports.extend(
        SpotConditionedFirstLineTransport(
            env,
            dynamics=dynamics,
            config=measure_config,
            current_log_spot_mean=log_spot_mean,
            current_log_spot_std=log_spot_std,
        )
        for _ in range(arguments.total_rounds - 1)
    )
    for transport in transports:
        transport.eval()
        for parameter in transport.parameters():
            parameter.requires_grad_(False)
    source_sha256 = checkpoint_digest(arguments.source_oce_checkpoint)
    restored = _load_checkpoint(
        arguments.checkpoint,
        arguments=arguments,
        source_sha256=source_sha256,
        train_state_sha256=train_state_sha256,
        joint_completion_sha256=joint_completion_sha256,
        transports=transports,
    )
    phase = "oce_training"
    active_round_index = 1
    accepted_rounds = 1
    start_epoch = 0
    stop_reason: str | None = None
    histories: list[dict[str, list[dict[str, float]]]] = [
        {"oce": [], "transport": []} for _ in range(arguments.total_rounds)
    ]
    audits: list[dict[str, Any]] = []
    if restored is not None:
        phase = str(restored["phase"])
        active_round_index = int(restored["active_round_index"])
        accepted_rounds = int(restored["accepted_rounds"])
        start_epoch = int(restored["epoch"])
        histories = cast(list[dict[str, list[dict[str, float]]]], restored["histories"])
        audits = cast(list[dict[str, Any]], restored["audits"])
        stop_reason = cast(str | None, restored["stop_reason"])
    if phase == "complete":
        print("Sequential OCE transport checkpoint is already complete", flush=True)
    elif phase not in {"oce_training", "transport_training", "audit"}:
        raise RuntimeError(f"invalid sequential OCE phase {phase!r}")

    origin_order = _fixed_origin_order(
        len(train_states),
        seed=arguments.seed,
        device=env.device,
    )
    num_fixed_batches = len(origin_order) // arguments.origins_per_batch
    fixed_path_seed = arguments.seed + 10_000
    compile_oce = not arguments.no_compile and not arguments.smoke

    while phase != "complete":
        current_transports = transports[:active_round_index]
        return_rms, inverse_sqrt = _estimate_current_preconditioner(
            current_transports,
            simulator,
            train_data,
            train_log_spot_tensor,
            normalized_strikes,
            seed=arguments.seed + 20_000 + active_round_index,
            minimum_return_rms=arguments.minimum_return_rms,
        )
        action_config = ConditionalOCEConfig(
            state_sizes_by_expiry=measure_config.state_sizes_by_expiry,
            num_instruments=len(normalized_strikes),
            maximum_preconditioned_action_norm=(
                arguments.maximum_preconditioned_action_norm
            ),
            preconditioned_action_bound_mode="hard_cap_penalty",
            action_cap_excess_penalty_weight=(
                arguments.action_cap_excess_penalty_weight
            ),
            include_current_log_spot=True,
            current_log_spot_mean=log_spot_mean,
            current_log_spot_std=log_spot_std,
        )
        action_model = SpotMarginalOCEAction(
            env,
            config=action_config,
            inverse_sqrt_second_moment=inverse_sqrt,
        )
        if restored is not None and restored.get("action_state_dict") is not None:
            if (
                restored.get("action_config") != asdict(action_config)
                or not torch.allclose(
                    cast(torch.Tensor, restored["return_rms"]),
                    return_rms.cpu(),
                    rtol=1.0e-6,
                    atol=1.0e-10,
                )
                or not torch.allclose(
                    cast(torch.Tensor, restored["inverse_sqrt_second_moment"]),
                    inverse_sqrt.cpu(),
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            ):
                raise ValueError("sequential active OCE identity changed on restore")
            action_model.load_state_dict(
                cast(Mapping[str, torch.Tensor], restored["action_state_dict"])
            )
        elif restored is not None and (phase != "oce_training" or start_epoch != 0):
            raise ValueError("active sequential stage has no restorable OCE action")

        if phase == "oce_training":
            action_model.train()
            optimizer = torch.optim.Adam(
                action_model.parameters(), lr=arguments.oce_learning_rate
            )
            if restored is not None and restored.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(
                    cast(dict[str, Any], restored["optimizer_state_dict"])
                )
            objective = _compile_oce_objective(
                action_model,
                compile_objective=compile_oce,
            )
            for epoch in range(start_epoch, arguments.oce_epochs_per_round):
                multiplier = _learning_rate_multiplier(
                    epoch,
                    arguments.oce_epochs_per_round,
                    arguments.minimum_learning_rate_fraction,
                    arguments.warmup_epochs,
                )
                optimizer.param_groups[0]["lr"] = arguments.oce_learning_rate * multiplier
                indices, batch = _fixed_batch(
                    train_data,
                    origin_order,
                    epoch,
                    origins_per_batch=arguments.origins_per_batch,
                    spot_contexts_per_origin=arguments.spot_contexts_per_origin,
                    surface_paths_per_context=arguments.surface_paths_per_context,
                    seed=fixed_path_seed,
                )
                returns = _draw_sequence_returns(
                    current_transports,
                    simulator,
                    train_data,
                    batch,
                    normalized_strikes,
                    train_log_spot_tensor[indices],
                )
                origin_state = batch.context_state[:: batch.spot_contexts_per_origin]
                optimizer.zero_grad(set_to_none=True)
                (
                    loss,
                    oce,
                    moment_rmse,
                    mean_action_norm,
                    cap_penalty,
                    maximum_cap_excess,
                ) = objective(
                    returns,
                    origin_state,
                    train_log_spot_tensor[indices],
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    action_model.parameters(), arguments.gradient_clip
                )
                optimizer.step()
                record = finite_training_record(
                    epoch=float(epoch + 1),
                    fixed_batch=float(epoch % num_fixed_batches),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    oce=float(oce.cpu()),
                    action_moment_rmse_bps_spot=float(moment_rmse.cpu()) * 1.0e4,
                    mean_action_norm=float(mean_action_norm.cpu()),
                    action_cap_excess_penalty=float(cap_penalty.cpu()),
                    maximum_preconditioned_action_cap_excess=float(
                        maximum_cap_excess.cpu()
                    ),
                    gradient_norm=float(gradient_norm.detach().cpu()),
                )
                histories[active_round_index]["oce"].append(record)
                if (epoch + 1) % arguments.report_every == 0 or epoch == start_epoch:
                    print(
                        f"sequential round={active_round_index + 1} "
                        f"OCE epoch={epoch + 1}/{arguments.oce_epochs_per_round} "
                        f"oce={record['oce']:.7g} "
                        f"moment_bps={record['action_moment_rmse_bps_spot']:.5g}",
                        flush=True,
                    )
                if (epoch + 1) % arguments.checkpoint_every == 0:
                    _save_torch_atomic(
                        _checkpoint_payload(
                            arguments=arguments,
                            source_sha256=source_sha256,
                            train_state_sha256=train_state_sha256,
                            joint_completion_sha256=joint_completion_sha256,
                            transports=transports,
                            phase=phase,
                            active_round_index=active_round_index,
                            accepted_rounds=accepted_rounds,
                            epoch=epoch + 1,
                            action_model=action_model,
                            return_rms=return_rms,
                            inverse_sqrt=inverse_sqrt,
                            histories=histories,
                            audits=audits,
                            optimizer=optimizer,
                            stop_reason=stop_reason,
                        ),
                        arguments.checkpoint,
                    )
            phase = "transport_training"
            start_epoch = 0
            restored = None
            _save_torch_atomic(
                _checkpoint_payload(
                    arguments=arguments,
                    source_sha256=source_sha256,
                    train_state_sha256=train_state_sha256,
                    joint_completion_sha256=joint_completion_sha256,
                    transports=transports,
                    phase=phase,
                    active_round_index=active_round_index,
                    accepted_rounds=accepted_rounds,
                    epoch=0,
                    action_model=action_model,
                    return_rms=return_rms,
                    inverse_sqrt=inverse_sqrt,
                    histories=histories,
                    audits=audits,
                    optimizer=None,
                    stop_reason=stop_reason,
                ),
                arguments.checkpoint,
            )

        for parameter in action_model.parameters():
            parameter.requires_grad_(False)
        action_model.eval()
        incremental = cast(SpotConditionedFirstLineTransport, transports[active_round_index])
        if phase == "transport_training":
            active_parameters = _active_transport_parameters(incremental)
            optimizer = torch.optim.Adam(
                active_parameters, lr=arguments.transport_learning_rate
            )
            if restored is not None and restored.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(
                    cast(dict[str, Any], restored["optimizer_state_dict"])
                )
            incremental.train()
            for epoch in range(start_epoch, arguments.transport_epochs_per_round):
                multiplier = _learning_rate_multiplier(
                    epoch,
                    arguments.transport_epochs_per_round,
                    arguments.minimum_learning_rate_fraction,
                    arguments.warmup_epochs,
                )
                optimizer.param_groups[0]["lr"] = (
                    arguments.transport_learning_rate * multiplier
                )
                indices, batch = _fixed_batch(
                    train_data,
                    origin_order,
                    epoch,
                    origins_per_batch=arguments.origins_per_batch,
                    spot_contexts_per_origin=arguments.spot_contexts_per_origin,
                    surface_paths_per_context=arguments.surface_paths_per_context,
                    seed=fixed_path_seed,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, details = sequential_population_oce_transport_reverse_kl_loss(
                    incremental_kernel=incremental,
                    current_kernels=current_transports,
                    simulator=simulator,
                    data=train_data,
                    batch=batch,
                    normalized_strikes=normalized_strikes,
                    oce_action=action_model,
                    current_log_spot=train_log_spot_tensor[indices],
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    active_parameters, arguments.gradient_clip
                )
                optimizer.step()
                record = finite_training_record(
                    epoch=float(epoch + 1),
                    fixed_batch=float(epoch % num_fixed_batches),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    reverse_kl_up_to_constant=float(loss.detach().cpu()),
                    mean_incremental_log_abs_det=float(
                        torch.mean(details["incremental_log_abs_det"]).detach().cpu()
                    ),
                    gradient_norm=float(gradient_norm.detach().cpu()),
                )
                histories[active_round_index]["transport"].append(record)
                if (epoch + 1) % arguments.report_every == 0 or epoch == start_epoch:
                    print(
                        f"sequential round={active_round_index + 1} "
                        f"transport epoch={epoch + 1}/{arguments.transport_epochs_per_round} "
                        f"reverse_kl={record['reverse_kl_up_to_constant']:.7g}",
                        flush=True,
                    )
                if (epoch + 1) % arguments.checkpoint_every == 0:
                    _save_torch_atomic(
                        _checkpoint_payload(
                            arguments=arguments,
                            source_sha256=source_sha256,
                            train_state_sha256=train_state_sha256,
                            joint_completion_sha256=joint_completion_sha256,
                            transports=transports,
                            phase=phase,
                            active_round_index=active_round_index,
                            accepted_rounds=accepted_rounds,
                            epoch=epoch + 1,
                            action_model=action_model,
                            return_rms=return_rms,
                            inverse_sqrt=inverse_sqrt,
                            histories=histories,
                            audits=audits,
                            optimizer=optimizer,
                            stop_reason=stop_reason,
                        ),
                        arguments.checkpoint,
                    )
            phase = "audit"
            start_epoch = 0
            restored = None

        incremental.eval()
        for parameter in incremental.parameters():
            parameter.requires_grad_(False)
        if phase == "audit":
            maximum_origins = 16 if arguments.smoke else None
            audit = {
                "schema": "flex-sequential-oce-transport-w2-round-audit-v1",
                "round": active_round_index + 1,
                "train": _audit_increment(
                    current_transports=current_transports,
                    incremental_transport=incremental,
                    action_model=action_model,
                    simulator=simulator,
                    data=train_data,
                    current_log_spot=train_log_spot_tensor,
                    normalized_strikes=normalized_strikes,
                    spot_contexts_per_origin=arguments.audit_spot_contexts_per_origin,
                    surface_paths_per_context=arguments.audit_surface_paths_per_context,
                    seed=arguments.seed + 100_000 + 10_000 * active_round_index,
                    label=f"train-round-{active_round_index + 1}",
                    maximum_origins=maximum_origins,
                ),
                "validation_2025": _audit_increment(
                    current_transports=current_transports,
                    incremental_transport=incremental,
                    action_model=action_model,
                    simulator=simulator,
                    data=validation_data,
                    current_log_spot=validation_log_spot_tensor,
                    normalized_strikes=normalized_strikes,
                    spot_contexts_per_origin=arguments.audit_spot_contexts_per_origin,
                    surface_paths_per_context=arguments.audit_surface_paths_per_context,
                    seed=arguments.seed + 200_000 + 10_000 * active_round_index,
                    label=f"validation-round-{active_round_index + 1}",
                    maximum_origins=maximum_origins,
                ),
            }
            passed, gate = _round_gate(
                cast(Mapping[str, Any], audit["train"]),
                cast(Mapping[str, Any], audit["validation_2025"]),
                arguments=arguments,
            )
            audit["gate"] = {"passed": passed, **gate}
            audits.append(audit)
            arguments.report_directory.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                audit,
                arguments.report_directory
                / f"sequential-oce-transport-round-{active_round_index + 1}.json",
            )
            if not passed:
                stop_reason = (
                    f"round {active_round_index + 1} direct-sampling gate failed: "
                    f"train improvement={float(gate['train_improvement']):.6%}, "
                    f"validation change={float(gate['validation_change']):.6%}"
                )
                phase = "complete"
            else:
                accepted_rounds = active_round_index + 1
                if accepted_rounds >= arguments.total_rounds:
                    stop_reason = "requested sequential OCE transports completed"
                    phase = "complete"
                else:
                    active_round_index += 1
                    phase = "oce_training"
                    start_epoch = 0
                    restored = None
            _save_torch_atomic(
                _checkpoint_payload(
                    arguments=arguments,
                    source_sha256=source_sha256,
                    train_state_sha256=train_state_sha256,
                    joint_completion_sha256=joint_completion_sha256,
                    transports=transports,
                    phase=phase,
                    active_round_index=active_round_index,
                    accepted_rounds=accepted_rounds,
                    epoch=0,
                    action_model=(None if phase == "oce_training" else action_model),
                    return_rms=(None if phase == "oce_training" else return_rms),
                    inverse_sqrt=(None if phase == "oce_training" else inverse_sqrt),
                    histories=histories,
                    audits=audits,
                    optimizer=None,
                    stop_reason=stop_reason,
                ),
                arguments.checkpoint,
            )

    first_indices, first_batch = _fixed_batch(
        train_data,
        origin_order,
        0,
        origins_per_batch=arguments.origins_per_batch,
        spot_contexts_per_origin=arguments.spot_contexts_per_origin,
        surface_paths_per_context=arguments.surface_paths_per_context,
        seed=fixed_path_seed,
    )
    report_path = arguments.report_directory / "sequential-oce-transport-w2.json"
    report = {
        "schema": "flex-sequential-oce-transport-w2-report-v1",
        "status": "complete",
        "semantic_success": accepted_rounds == arguments.total_rounds,
        "stop_reason": stop_reason,
        "accepted_rounds": accepted_rounds,
        "arguments": _json_arguments(arguments),
        "train_dates": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "validation_dates": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "date_weighting": "each training date appears once per fixed scenario cycle",
        "sampling": (
            "each accepted law is sampled directly with equal path weights; each "
            "incremental transport acts on a deterministic common-random-number bank"
        ),
        "fixed_random_numbers": {
            "origin_order_seed": arguments.seed,
            "path_seed_base": fixed_path_seed,
            "num_origin_batches": num_fixed_batches,
            "first_batch_origin_indices": first_indices.detach().cpu().tolist(),
            "first_batch_standard_normals_sha256": _tensor_sha256(
                first_batch.standard_normals
            ),
            "regeneration": (
                "batch b always resets the device generator to path_seed_base+b; "
                "no training step consumes an evolving random stream"
            ),
        },
        "current_log_spot_state": {
            "included_in_actions_and_incremental_transports": True,
            "train_mean": log_spot_mean,
            "train_population_std": log_spot_std,
        },
        "normalized_strikes": normalized_strikes.detach().cpu().tolist(),
        "histories": histories,
        "round_audits": audits,
        "sources": sources,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-sequential-oce-transport-w2-completion-v1",
            "status": "complete",
            "semantic_success": accepted_rounds == arguments.total_rounds,
            "accepted_rounds": accepted_rounds,
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
            "stop_reason": stop_reason,
        },
        arguments.report_directory / "sequential-oce-transport-w2-complete.json",
    )


def main() -> None:
    """CLI entry point with durable failure reporting."""

    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_sequential_oce_transport_w2")
