"""Fit one sampleable W2 law directly to the conditional ATM-call moment."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from filelock import FileLock

from quantdata.dysanos.flex.conditional_moment import (
    direct_w2_conditional_moment_transport_loss,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.joint_spot_correlation import (
    fit_joint_spot_correlation_under_p,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.oce_transport import (
    ConditionalPathBatch,
    SpotConditionedFirstLineTransport,
    SpotMarginalOCEAction,
    apply_first_line_transport_sequence,
    sample_conditional_paths,
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
from quantdata.dysanos.flex.run_sequential_oce_transport_w2 import (
    _estimate_current_preconditioner,
    _fixed_batch,
    _fixed_origin_order,
    _path_log_spot,
    _source_action_config,
)
from quantdata.dysanos.flex.weekly_measure import (
    GenericWeeklyMeasureKernel,
    JointHierarchicalWeeklyDynamicsAdapter,
)
from quantdata.dysanos.flex.weekly_measure_training import (
    WeeklyMeasureTrainingData,
    audit_front_week_consistency,
    precompute_measure_training_data,
)
from quantdata.dysanos.flex.weekly_simulation import WeeklySanosSimulator
from quantdata.torch import BaseEnvironment


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, required=True)
    parser.add_argument("--source-oce-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--worker-heartbeat", type=Path, default=None)
    parser.add_argument("--generation-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2_500)
    parser.add_argument("--origins-per-batch", type=int, default=3)
    parser.add_argument("--spot-contexts-per-origin", type=int, default=16)
    parser.add_argument("--surface-paths-per-context", type=int, default=128)
    parser.add_argument("--audit-spot-contexts-per-origin", type=int, default=32)
    parser.add_argument("--audit-surface-paths-per-context", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--minimum-learning-rate-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=25)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--kl-weight", type=float, default=1.0e-3)
    parser.add_argument("--minimum-return-rms", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--minimum-train-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-validation-degradation", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=8_240_913)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.device != "cuda:0":
        raise ValueError("direct W2 moment transport is restricted to GPU0")
    positive_integers = (
        "epochs",
        "origins_per_batch",
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "audit_spot_contexts_per_origin",
        "audit_surface_paths_per_context",
        "checkpoint_every",
        "report_every",
    )
    if any(int(getattr(arguments, name)) <= 0 for name in positive_integers):
        raise ValueError("direct W2 moment integer controls must be positive")
    for name in (
        "spot_contexts_per_origin",
        "surface_paths_per_context",
        "audit_spot_contexts_per_origin",
        "audit_surface_paths_per_context",
    ):
        if int(getattr(arguments, name)) % 2:
            raise ValueError(f"{name} must be even for antithetic sampling")
    finite_positive = (
        arguments.learning_rate,
        arguments.gradient_clip,
        arguments.minimum_return_rms,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
        raise ValueError("direct W2 moment optimizer controls must be finite-positive")
    if not math.isfinite(arguments.kl_weight) or arguments.kl_weight < 0.0:
        raise ValueError("direct W2 moment KL weight must be finite-nonnegative")
    if not 0.0 < arguments.minimum_learning_rate_fraction <= 1.0:
        raise ValueError("minimum learning-rate fraction must lie in (0,1]")
    if (
        arguments.warmup_epochs < 0
        or arguments.warmup_epochs >= arguments.epochs
        or arguments.minimum_train_improvement < 0.0
        or arguments.maximum_validation_degradation < 0.0
    ):
        raise ValueError("invalid direct W2 moment schedule or gate")
    if (arguments.worker_heartbeat is None) != (arguments.generation_token is None):
        raise ValueError("worker heartbeat and generation token must be specified together")
    if arguments.generation_token is not None and not arguments.generation_token:
        raise ValueError("generation token must be non-empty")


def _write_worker_heartbeat(
    arguments: argparse.Namespace,
    *,
    phase: str,
    epoch: int,
    status: str = "RUNNING",
) -> None:
    if arguments.worker_heartbeat is None:
        return
    write_json_atomic(
        {
            "schema": "flex-direct-moment-w2-worker-heartbeat-v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "generation": arguments.generation_token,
            "worker_pid": os.getpid(),
            "status": status,
            "last_progress_value": {
                "phase": phase,
                "epoch": epoch,
                "total_epochs": arguments.epochs,
            },
        },
        arguments.worker_heartbeat,
    )


def _checkpoint_payload(
    *,
    arguments: argparse.Namespace,
    source_sha256: str,
    train_state_sha256: str,
    joint_completion_sha256: str,
    measure_config: Mapping[str, Any],
    incremental: SpotConditionedFirstLineTransport,
    inverse_sqrt: torch.Tensor,
    return_rms: torch.Tensor,
    phase: str,
    completed_epochs: int,
    history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer | None,
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": "flex-direct-moment-transport-w2-checkpoint-v1",
        "generation": arguments.generation_token,
        "arguments": _json_arguments(arguments),
        "source_oce_checkpoint_sha256": source_sha256,
        "train_state_sha256": train_state_sha256,
        "joint_completion_sha256": joint_completion_sha256,
        "measure_config": dict(measure_config),
        "phase": phase,
        "completed_epochs": completed_epochs,
        "history": history,
        "model_state_dict": incremental.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "inverse_sqrt_second_moment": inverse_sqrt.detach().cpu(),
        "return_rms": return_rms.detach().cpu(),
        "audit": None if audit is None else dict(audit),
    }


def _load_checkpoint(
    path: Path,
    *,
    arguments: argparse.Namespace,
    source_sha256: str,
    train_state_sha256: str,
    joint_completion_sha256: str,
    measure_config: Mapping[str, Any],
    incremental: SpotConditionedFirstLineTransport,
    inverse_sqrt: torch.Tensor,
    return_rms: torch.Tensor,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "flex-direct-moment-transport-w2-checkpoint-v1"
        or payload.get("arguments") != _json_arguments(arguments)
        or payload.get("source_oce_checkpoint_sha256") != source_sha256
        or payload.get("train_state_sha256") != train_state_sha256
        or payload.get("joint_completion_sha256") != joint_completion_sha256
        or payload.get("measure_config") != dict(measure_config)
        or not torch.allclose(
            cast(torch.Tensor, payload.get("inverse_sqrt_second_moment")),
            inverse_sqrt.cpu(),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        or not torch.allclose(
            cast(torch.Tensor, payload.get("return_rms")),
            return_rms.cpu(),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
    ):
        raise ValueError("direct W2 moment checkpoint identity is invalid")
    incremental.load_state_dict(
        cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
    )
    return payload


def _compile_objective(
    *,
    incremental: SpotConditionedFirstLineTransport,
    source_transport: GenericWeeklyMeasureKernel,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    normalized_strikes: torch.Tensor,
    inverse_sqrt: torch.Tensor,
    spot_contexts_per_origin: int,
    surface_paths_per_context: int,
    kl_weight: float,
    compile_objective: bool,
) -> Any:
    def objective(
        context_state: torch.Tensor,
        context_gross_spot_return: torch.Tensor,
        current_state: torch.Tensor,
        gross_spot_return: torch.Tensor,
        standard_normals: torch.Tensor,
        repeated_origin: torch.Tensor,
        origin_log_spot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = ConditionalPathBatch(
            context_state=context_state,
            context_gross_spot_return=context_gross_spot_return,
            current_state=current_state,
            gross_spot_return=gross_spot_return,
            standard_normals=standard_normals,
            repeated_origin=repeated_origin,
            spot_contexts_per_origin=spot_contexts_per_origin,
            surface_paths_per_context=surface_paths_per_context,
        )
        loss, details = direct_w2_conditional_moment_transport_loss(
            incremental_kernel=incremental,
            current_kernels=(source_transport,),
            simulator=simulator,
            data=data,
            batch=batch,
            normalized_strikes=normalized_strikes,
            inverse_sqrt_second_moment=inverse_sqrt,
            current_log_spot=origin_log_spot,
            kl_weight=kl_weight,
        )
        moments = details["conditional_moments"]
        return (
            loss,
            details["moment_loss"].detach(),
            details["incremental_kl"].detach(),
            torch.sqrt(torch.mean(moments.square())).detach(),
            torch.max(torch.abs(moments)).detach(),
        )

    return (
        torch.compile(objective, dynamic=False, fullgraph=False)
        if compile_objective
        else objective
    )


@torch.no_grad()
def _audit_law(
    *,
    source_transport: GenericWeeklyMeasureKernel,
    incremental: SpotConditionedFirstLineTransport,
    simulator: WeeklySanosSimulator,
    data: WeeklyMeasureTrainingData,
    current_log_spot: torch.Tensor,
    normalized_strikes: torch.Tensor,
    inverse_sqrt: torch.Tensor,
    spot_contexts_per_origin: int,
    surface_paths_per_context: int,
    seed: int,
    label: str,
    maximum_origins: int | None = None,
) -> dict[str, Any]:
    num_origins = min(len(data.reduced_states), maximum_origins or len(data.reduced_states))
    source_moments: list[torch.Tensor] = []
    candidate_moments: list[torch.Tensor] = []
    whitened_candidate_moments: list[torch.Tensor] = []
    incremental_kls: list[torch.Tensor] = []
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
        source_draw, _ = apply_first_line_transport_sequence(
            (source_transport,),
            batch.current_state,
            batch.gross_spot_return,
            batch.standard_normals,
            _path_log_spot(current_log_spot[indices], batch),
        )
        source_returns = w2_roll_returns(
            simulator,
            data,
            batch,
            source_draw.reduced_state,
            normalized_strikes,
        ).reshape(1, spot_contexts_per_origin, surface_paths_per_context, -1)
        _, details = direct_w2_conditional_moment_transport_loss(
            incremental_kernel=incremental,
            current_kernels=(source_transport,),
            simulator=simulator,
            data=data,
            batch=batch,
            normalized_strikes=normalized_strikes,
            inverse_sqrt_second_moment=inverse_sqrt,
            current_log_spot=current_log_spot[indices],
            kl_weight=0.0,
        )
        source_moments.append(torch.mean(source_returns, dim=(1, 2)).cpu())
        candidate_moments.append(details["conditional_moments"].cpu())
        whitened_candidate_moments.append(
            details["whitened_conditional_moments"].cpu()
        )
        incremental_kls.append(details["incremental_kl"].reshape(1).cpu())
        derivative = incremental.minimum_first_line_coordinate_derivative(
            batch.current_state,
            batch.gross_spot_return,
            _path_log_spot(current_log_spot[indices], batch),
        )
        minimum_derivative = min(minimum_derivative, float(derivative.cpu()))
        if origin + 1 == num_origins or (origin + 1) % 128 == 0:
            print(f"audit {label} origins={origin + 1}/{num_origins}", flush=True)

    source = torch.cat(source_moments)
    candidate = torch.cat(candidate_moments)
    whitened = torch.cat(whitened_candidate_moments)
    return {
        "num_origins": num_origins,
        "num_options": candidate.shape[-1],
        "spot_contexts_per_origin": spot_contexts_per_origin,
        "surface_paths_per_context": surface_paths_per_context,
        "source_origin_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(source.square()))
        )
        * 1.0e4,
        "source_origin_moment_maximum_abs_bps_spot": float(torch.max(torch.abs(source)))
        * 1.0e4,
        "candidate_origin_moment_rmse_bps_spot": float(
            torch.sqrt(torch.mean(candidate.square()))
        )
        * 1.0e4,
        "candidate_origin_moment_maximum_abs_bps_spot": float(
            torch.max(torch.abs(candidate))
        )
        * 1.0e4,
        "candidate_whitened_moment_rmse": float(
            torch.sqrt(torch.mean(whitened.square()))
        ),
        "mean_incremental_kl_to_source": float(torch.mean(torch.cat(incremental_kls))),
        "minimum_incremental_coordinate_derivative": minimum_derivative,
        "path_measure": "equal-weight direct draws from the conditional transport",
    }


def _gate(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    arguments: argparse.Namespace,
) -> dict[str, float | bool]:
    train_before = float(train["source_origin_moment_rmse_bps_spot"])
    train_after = float(train["candidate_origin_moment_rmse_bps_spot"])
    validation_before = float(validation["source_origin_moment_rmse_bps_spot"])
    validation_after = float(validation["candidate_origin_moment_rmse_bps_spot"])
    train_improvement = 1.0 - train_after / max(train_before, np.finfo(float).tiny)
    validation_change = validation_after / max(
        validation_before, np.finfo(float).tiny
    ) - 1.0
    derivative_positive = (
        float(train["minimum_incremental_coordinate_derivative"]) > 0.0
        and float(validation["minimum_incremental_coordinate_derivative"]) > 0.0
    )
    return {
        "candidate_accepted": (
            train_improvement > arguments.minimum_train_improvement
            and validation_change <= arguments.maximum_validation_degradation
            and derivative_positive
        ),
        "train_improvement": train_improvement,
        "validation_change": validation_change,
        "derivative_positive": derivative_positive,
    }


def run(arguments: argparse.Namespace) -> None:
    """Train and audit one ATM W2 conditional-moment transport."""

    _validate_arguments(arguments)
    _write_worker_heartbeat(arguments, phase="initializing", epoch=0)
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
            "origins-per-batch must divide training dates for equal-date weighting"
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
    measure_config, p_reference, source_transport = _build_kernels(env, dynamics)
    source_payload = torch.load(
        arguments.source_oce_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(source_payload, Mapping):
        raise ValueError("source OCE checkpoint is not a mapping")
    source_config = _source_action_config(source_payload)
    source_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    source_train_data = precompute_measure_training_data(
        simulator, train_states, source_strikes, num_expiries=2
    )
    source_generator = torch.Generator(device=env.device)
    source_generator.manual_seed(
        int(cast(Mapping[str, Any], source_payload["arguments"])["seed"])
    )
    source_return_rms, source_inverse_sqrt = _estimate_preconditioner(
        p_reference,
        simulator,
        source_train_data,
        source_strikes,
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
    source_transport.load_state_dict(
        cast(Mapping[str, torch.Tensor], validated_source["transport_state_dict"])
    )
    source_transport.eval()
    for parameter in source_transport.parameters():
        parameter.requires_grad_(False)

    # The first feasibility experiment imposes only the ATM two-week call.
    normalized_strikes = torch.ones(1, dtype=env.fdtype, device=env.device)
    train_data = precompute_measure_training_data(
        simulator, train_states, normalized_strikes, num_expiries=2
    )
    validation_data = precompute_measure_training_data(
        simulator, validation_states, normalized_strikes, num_expiries=2
    )
    return_rms, inverse_sqrt = _estimate_current_preconditioner(
        [source_transport],
        simulator,
        train_data,
        train_log_spot_tensor,
        normalized_strikes,
        seed=arguments.seed + 20_000,
        minimum_return_rms=arguments.minimum_return_rms,
    )
    incremental = SpotConditionedFirstLineTransport(
        env,
        dynamics=dynamics,
        config=measure_config,
        current_log_spot_mean=log_spot_mean,
        current_log_spot_std=log_spot_std,
    )
    active_parameters = _active_transport_parameters(incremental)
    source_sha256 = checkpoint_digest(arguments.source_oce_checkpoint)
    restored = _load_checkpoint(
        arguments.checkpoint,
        arguments=arguments,
        source_sha256=source_sha256,
        train_state_sha256=train_state_sha256,
        joint_completion_sha256=joint_completion_sha256,
        measure_config=asdict(measure_config),
        incremental=incremental,
        inverse_sqrt=inverse_sqrt,
        return_rms=return_rms,
    )
    phase = "training"
    start_epoch = 0
    history: list[dict[str, float]] = []
    audit: Mapping[str, Any] | None = None
    if restored is not None:
        phase = str(restored["phase"])
        start_epoch = int(restored["completed_epochs"])
        history = cast(list[dict[str, float]], restored["history"])
        audit = cast(Mapping[str, Any] | None, restored["audit"])
    if phase not in {"training", "audit", "complete"}:
        raise RuntimeError(f"invalid direct W2 moment phase {phase!r}")
    if phase == "complete":
        print("Direct W2 moment checkpoint is already complete", flush=True)

    optimizer: torch.optim.Optimizer | None = None
    origin_order = _fixed_origin_order(
        len(train_states), seed=arguments.seed, device=env.device
    )
    num_fixed_batches = len(origin_order) // arguments.origins_per_batch
    fixed_path_seed = arguments.seed + 10_000
    first_indices, first_batch = _fixed_batch(
        train_data,
        origin_order,
        0,
        origins_per_batch=arguments.origins_per_batch,
        spot_contexts_per_origin=arguments.spot_contexts_per_origin,
        surface_paths_per_context=arguments.surface_paths_per_context,
        seed=fixed_path_seed,
    )
    if phase == "training":
        optimizer = torch.optim.Adam(active_parameters, lr=arguments.learning_rate)
        if restored is not None and restored.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(
                cast(dict[str, Any], restored["optimizer_state_dict"])
            )
        objective = _compile_objective(
            incremental=incremental,
            source_transport=source_transport,
            simulator=simulator,
            data=train_data,
            normalized_strikes=normalized_strikes,
            inverse_sqrt=inverse_sqrt,
            spot_contexts_per_origin=arguments.spot_contexts_per_origin,
            surface_paths_per_context=arguments.surface_paths_per_context,
            kl_weight=arguments.kl_weight,
            compile_objective=not arguments.no_compile and not arguments.smoke,
        )
        incremental.train()
        for epoch in range(start_epoch, arguments.epochs):
            multiplier = _learning_rate_multiplier(
                epoch,
                arguments.epochs,
                arguments.minimum_learning_rate_fraction,
                arguments.warmup_epochs,
            )
            optimizer.param_groups[0]["lr"] = arguments.learning_rate * multiplier
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
            loss, moment_loss, incremental_kl, moment_rmse, moment_maximum = objective(
                batch.context_state,
                batch.context_gross_spot_return,
                batch.current_state,
                batch.gross_spot_return,
                batch.standard_normals,
                batch.repeated_origin,
                train_log_spot_tensor[indices],
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                active_parameters, arguments.gradient_clip
            )
            optimizer.step()
            record = {
                "epoch": float(epoch + 1),
                "fixed_batch": float(epoch % num_fixed_batches),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "moment_loss": float(moment_loss.cpu()),
                "conditional_moment_rmse_bps_spot": float(moment_rmse.cpu()) * 1.0e4,
                "conditional_moment_maximum_abs_bps_spot": float(moment_maximum.cpu())
                * 1.0e4,
                "incremental_kl_to_source": float(incremental_kl.cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise FloatingPointError("direct W2 moment training record is non-finite")
            history.append(record)
            if (epoch + 1) % arguments.report_every == 0 or epoch == start_epoch:
                print(
                    f"direct W2 ATM epoch={epoch + 1}/{arguments.epochs} "
                    f"moment_bps={record['conditional_moment_rmse_bps_spot']:.6g} "
                    f"kl={record['incremental_kl_to_source']:.6g}",
                    flush=True,
                )
            if (epoch + 1) % arguments.checkpoint_every == 0:
                _save_torch_atomic(
                    _checkpoint_payload(
                        arguments=arguments,
                        source_sha256=source_sha256,
                        train_state_sha256=train_state_sha256,
                        joint_completion_sha256=joint_completion_sha256,
                        measure_config=asdict(measure_config),
                        incremental=incremental,
                        inverse_sqrt=inverse_sqrt,
                        return_rms=return_rms,
                        phase="training",
                        completed_epochs=epoch + 1,
                        history=history,
                        optimizer=optimizer,
                        audit=None,
                    ),
                    arguments.checkpoint,
                )
                _write_worker_heartbeat(
                    arguments, phase="training", epoch=epoch + 1
                )
        phase = "audit"
        _save_torch_atomic(
            _checkpoint_payload(
                arguments=arguments,
                source_sha256=source_sha256,
                train_state_sha256=train_state_sha256,
                joint_completion_sha256=joint_completion_sha256,
                measure_config=asdict(measure_config),
                incremental=incremental,
                inverse_sqrt=inverse_sqrt,
                return_rms=return_rms,
                phase=phase,
                completed_epochs=arguments.epochs,
                history=history,
                optimizer=None,
                audit=None,
            ),
            arguments.checkpoint,
        )
        _write_worker_heartbeat(arguments, phase=phase, epoch=arguments.epochs)

    if phase == "audit":
        incremental.eval()
        maximum_origins = 16 if arguments.smoke else None
        train_audit = _audit_law(
            source_transport=source_transport,
            incremental=incremental,
            simulator=simulator,
            data=train_data,
            current_log_spot=train_log_spot_tensor,
            normalized_strikes=normalized_strikes,
            inverse_sqrt=inverse_sqrt,
            spot_contexts_per_origin=arguments.audit_spot_contexts_per_origin,
            surface_paths_per_context=arguments.audit_surface_paths_per_context,
            seed=arguments.seed + 100_000,
            label="train-direct-W2-ATM",
            maximum_origins=maximum_origins,
        )
        validation_audit = _audit_law(
            source_transport=source_transport,
            incremental=incremental,
            simulator=simulator,
            data=validation_data,
            current_log_spot=validation_log_spot_tensor,
            normalized_strikes=normalized_strikes,
            inverse_sqrt=inverse_sqrt,
            spot_contexts_per_origin=arguments.audit_spot_contexts_per_origin,
            surface_paths_per_context=arguments.audit_surface_paths_per_context,
            seed=arguments.seed + 200_000,
            label="validation-direct-W2-ATM",
            maximum_origins=maximum_origins,
        )
        front_week = audit_front_week_consistency(
            validation_data,
            normalized_strikes,
            paths_per_origin=(128 if arguments.smoke else 4_096),
            origins_per_batch=4,
            seed=arguments.seed + 300_000,
        )
        gate = _gate(train_audit, validation_audit, arguments=arguments)
        audit = {
            "schema": "flex-direct-moment-transport-w2-atm-audit-v1",
            "status": "complete",
            "candidate_accepted": gate["candidate_accepted"],
            "gate": gate,
            "train": train_audit,
            "validation_2025": validation_audit,
            "validation_2025_front_week": front_week,
        }
        arguments.report_directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            audit,
            arguments.report_directory / "direct-moment-transport-w2-atm-audit.json",
        )
        phase = "complete"
        _save_torch_atomic(
            _checkpoint_payload(
                arguments=arguments,
                source_sha256=source_sha256,
                train_state_sha256=train_state_sha256,
                joint_completion_sha256=joint_completion_sha256,
                measure_config=asdict(measure_config),
                incremental=incremental,
                inverse_sqrt=inverse_sqrt,
                return_rms=return_rms,
                phase=phase,
                completed_epochs=arguments.epochs,
                history=history,
                optimizer=None,
                audit=audit,
            ),
            arguments.checkpoint,
        )
        _write_worker_heartbeat(
            arguments, phase=phase, epoch=arguments.epochs, status="SUCCEEDED"
        )

    if audit is None:
        raise RuntimeError("direct W2 moment experiment has no audit")
    report_path = arguments.report_directory / "direct-moment-transport-w2-atm.json"
    report = {
        "schema": "flex-direct-moment-transport-w2-atm-report-v1",
        "status": "complete",
        "candidate_accepted": bool(audit["candidate_accepted"]),
        "arguments": _json_arguments(arguments),
        "train_dates": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "validation_dates": [
            validation_dates[0].isoformat(),
            validation_dates[-1].isoformat(),
        ],
        "option_universe": {
            "count": 1,
            "type": "call",
            "current_expiry": "W2",
            "rolled_expiry": "W1",
            "normalized_strike": 1.0,
        },
        "objective": (
            "equal-date mean squared conditional ATM-call moment, whitened by "
            "the source-law return second moment, plus incremental KL to Q1"
        ),
        "date_weighting": "each training date appears once per fixed scenario cycle",
        "fixed_random_numbers": {
            "origin_order_seed": arguments.seed,
            "path_seed_base": fixed_path_seed,
            "num_origin_batches": num_fixed_batches,
            "first_batch_origin_indices": first_indices.detach().cpu().tolist(),
            "first_batch_standard_normals_sha256": _tensor_sha256(
                first_batch.standard_normals
            ),
        },
        "current_log_spot_state": {
            "included_in_incremental_transport": True,
            "train_mean": log_spot_mean,
            "train_population_std": log_spot_std,
        },
        "history": history,
        "audit": audit,
        "sources": sources,
        "source_oce_checkpoint": str(arguments.source_oce_checkpoint),
        "source_oce_checkpoint_sha256": source_sha256,
    }
    write_json_atomic(report, report_path)
    write_json_atomic(
        {
            "schema": "flex-direct-moment-transport-w2-atm-completion-v1",
            "status": "complete",
            "semantic_success": True,
            "candidate_accepted": bool(audit["candidate_accepted"]),
            "generation": arguments.generation_token,
            "report": str(report_path),
            "report_sha256": checkpoint_digest(report_path),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        },
        arguments.report_directory / "direct-moment-transport-w2-atm-complete.json",
    )


def main() -> None:
    """CLI entry point with a logical-checkpoint training lock."""

    arguments = _parse_arguments()
    lock_path = arguments.checkpoint.with_suffix(arguments.checkpoint.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_direct_moment_transport_w2",
        clean_sigterm=True,
    )
