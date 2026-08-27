"""Fit one generic conditional front-state transport across all FLEX history."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from filelock import FileLock

from quantdata import env as quantdata_env
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_all_state_vector_q_existence import (
    CommonQDraw,
    normalize_front_components,
)
from quantdata.dysanos.flex.run_atm_q_existence import (
    _atomic_torch_save,
    _load_inputs,
)
from quantdata.dysanos.flex.run_six_state_joint_generic_q_existence import (
    JointGenericFrontTransport,
    JointGenericTransportConfig,
    _draws,
    _evaluate_joint,
    joint_roll_returns,
)
from quantdata.dysanos.flex.run_state_generic_q_existence import (
    _summary as generic_state_summary,
)
from quantdata.torch import BaseEnvironment

OBJECTIVE_ID = "all-state-joint-generic-continuous-eleven-option-moments-v1"
_SHARED_STRIKES_TRAINING_ROOT = Path(quantdata_env.locations.training.path) / "flex-shared-strikes"
_DEFAULT_JOINT_COMPLETION = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "joint-hierarchical"
    / "v1"
    / "reports"
    / "joint-hierarchical-200-complete.json"
)
_DEFAULT_TEACHER = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "generic-joint-six-v1"
    / "checkpoints"
    / "joint-generic-q.pt"
)
_DEFAULT_AFFINE_TEACHER = (
    _SHARED_STRIKES_TRAINING_ROOT
    / "q-existence"
    / "vector-all-history-exact-common-random-statewise-cv-v2"
    / "snapshots"
    / "all-state-vector-q-epoch351.pt"
)
_DEFAULT_OUTPUT_ROOT = (
    _SHARED_STRIKES_TRAINING_ROOT / "q-existence" / "generic-joint-all-history-affine-init-v1"
)


def _state_batches(num_states: int, state_batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + state_batch_size, num_states))
        for start in range(0, num_states, state_batch_size)
    ]


def _aggregate_state_summaries(
    per_state: list[dict[str, Any]],
    *,
    z_target: float,
    paths_per_state: int,
) -> dict[str, Any]:
    """Aggregate state-local summaries without pooling their control variates."""

    if not per_state:
        raise ValueError("an all-state audit requires at least one state")
    raw: list[tuple[float, int, int]] = []
    adjusted: list[tuple[float, int, int]] = []
    for state in per_state:
        state_index = int(state["state_index"])
        raw.extend(
            (abs(float(value)), state_index, instrument)
            for instrument, value in enumerate(state["raw_mean_z"])
        )
        adjusted.extend(
            (abs(float(value)), state_index, instrument)
            for instrument, value in enumerate(state["noise_scaled_adjusted_mean_z"])
        )
    raw.sort(reverse=True)
    adjusted.sort(reverse=True)
    worst_count = max(1, math.ceil(0.10 * len(adjusted)))
    return {
        "num_states": len(per_state),
        "num_moments": len(adjusted),
        "paths_per_state": paths_per_state,
        "total_state_paths": len(per_state) * paths_per_state,
        "common_random_numbers_across_states": True,
        "z_target": z_target,
        "maximum_abs_raw_mean_z": raw[0][0],
        "worst_10_percent_mean_abs_raw_mean_z": sum(value for value, _, _ in raw[:worst_count])
        / worst_count,
        "maximum_abs_noise_scaled_adjusted_mean_z": adjusted[0][0],
        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": sum(
            value for value, _, _ in adjusted[:worst_count]
        )
        / worst_count,
        "worst_raw_state_index": raw[0][1],
        "worst_raw_instrument_index": raw[0][2],
        "worst_state_index": adjusted[0][1],
        "worst_instrument_index": adjusted[0][2],
        "passes_target": adjusted[0][0] <= z_target,
        "per_state": per_state,
    }


def _audit_all_states(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    history_indices: tuple[int, ...],
    dates: tuple[str, ...],
    state_kinds: tuple[str, ...],
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    state_batch_size: int,
    z_target: float,
) -> dict[str, Any]:
    """Audit every state in bounded-memory batches on one fixed dataset."""

    per_state: list[dict[str, Any]] = []
    for state_slice in _state_batches(len(current_states), state_batch_size):
        with torch.no_grad():
            values = _evaluate_joint(
                model=model,
                law=law,
                current_states=current_states[state_slice],
                current_w2_calls=current_w2_calls[state_slice],
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support[state_slice],
                density=density[state_slice],
                sqrt_variance=sqrt_variance[state_slice],
                draws=draws,
            )
        for local_index in range(state_slice.stop - state_slice.start):
            summary = generic_state_summary(
                {
                    "raw_replication_means": values["raw_replication_means"][:, local_index],
                    "spot_replication_means": values["spot_replication_means"][:, local_index],
                    "adjusted_replication_means": values["adjusted_replication_means"][
                        :, local_index
                    ],
                    "beta": values["beta"][local_index],
                    "path_sum": values["path_sum"][local_index],
                    "path_square_sum": values["path_square_sum"][local_index],
                    "cap_mean_square": values["cap_mean_square"][local_index],
                    "num_paths": values["num_paths"],
                },
                z_target=z_target,
            )
            absolute_index = state_slice.start + local_index
            summary.update(
                {
                    "state_index": history_indices[absolute_index],
                    "date": dates[absolute_index],
                    "history_partition": state_kinds[absolute_index],
                }
            )
            per_state.append(summary)
    return _aggregate_state_summaries(
        per_state,
        z_target=z_target,
        paths_per_state=sum(draw.num_paths for draw in draws),
    )


def _batch_objective(
    values: Mapping[str, torch.Tensor],
    *,
    temperature: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct a detached studentized smooth-maximum batch objective."""

    adjusted_rep = values["adjusted_replication_means"].double()
    raw_rep = values["raw_replication_means"].double()
    spot_rep = values["spot_replication_means"].double()
    adjusted_mean = torch.mean(adjusted_rep, dim=0)
    raw_mean = torch.mean(raw_rep, dim=0)
    raw_se = torch.std(raw_rep, dim=0, correction=1) / math.sqrt(len(raw_rep))
    spot_se = torch.std(spot_rep, dim=0, correction=1) / math.sqrt(len(spot_rep))
    beta = values["beta"].double()
    noise_scale = torch.maximum(torch.abs(beta) * spot_se[..., None], raw_se)
    scaled = adjusted_mean / noise_scale.clamp_min(1.0e-12)
    squared = scaled.square()
    loss = temperature * (
        torch.logsumexp(squared.reshape(-1) / temperature, dim=0) - math.log(squared.numel())
    )
    weights = torch.softmax(squared.reshape(-1) / temperature, dim=0).reshape_as(squared)
    mean_gradient = (2.0 * scaled * weights / noise_scale.clamp_min(1.0e-12)).to(dtype=dtype)
    absolute_raw = torch.abs(raw_mean / raw_se.clamp_min(1.0e-12))
    absolute_adjusted = torch.abs(scaled)
    worst_count = max(1, math.ceil(0.10 * absolute_adjusted.numel()))
    metrics = {
        "loss": float(loss),
        "raw_worst": float(torch.max(absolute_raw)),
        "raw_worst10": float(torch.mean(torch.topk(absolute_raw.reshape(-1), worst_count).values)),
        "adjusted_worst": float(torch.max(absolute_adjusted)),
        "adjusted_worst10": float(
            torch.mean(torch.topk(absolute_adjusted.reshape(-1), worst_count).values)
        ),
    }
    return loss, mean_gradient, metrics


def _train_state_batch(
    *,
    model: Any,
    law: JointGenericFrontTransport,
    optimizer: torch.optim.Optimizer,
    current_states: torch.Tensor,
    current_w2_calls: torch.Tensor,
    normalized_strikes: torch.Tensor,
    option_is_call: torch.Tensor,
    support: torch.Tensor,
    density: torch.Tensor,
    sqrt_variance: torch.Tensor,
    draws: list[CommonQDraw],
    temperature: float,
) -> dict[str, float]:
    """Take one optimizer step using all fixed paths for a state batch."""

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        values = _evaluate_joint(
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draws=draws,
        )
        _, mean_gradient, metrics = _batch_objective(
            values,
            temperature=temperature,
            dtype=current_states.dtype,
        )
        beta = values["beta"].to(dtype=current_states.dtype)
    for draw in draws:
        returns, _, gross = joint_roll_returns(
            model=model,
            law=law,
            current_states=current_states,
            current_w2_calls=current_w2_calls,
            normalized_strikes=normalized_strikes,
            option_is_call=option_is_call,
            support=support,
            density=density,
            sqrt_variance=sqrt_variance,
            draw=draw,
        )
        adjusted_mean = (
            torch.mean(returns, dim=1) - beta * torch.mean(gross - 1.0, dim=1)[..., None]
        )
        surrogate = torch.sum(mean_gradient * adjusted_mean) / len(draws)
        surrogate.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
    optimizer.step()
    metrics["gradient_norm"] = float(gradient_norm.detach())
    return metrics


def _distill_affine_history(
    *,
    law: JointGenericFrontTransport,
    current_front_states: torch.Tensor,
    affine_payload: Mapping[str, Any],
    steps: int,
    states_per_step: int,
    paths_per_state: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Warm-start the shared map from the preserved statewise affine law."""

    teacher_state = cast(Mapping[str, torch.Tensor], affine_payload["law_state_dict"])
    device = current_front_states.device
    dtype = current_front_states.dtype
    front_state_size = current_front_states.shape[1]
    teacher_front = teacher_state["current_front_states"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    if not torch.equal(teacher_front, current_front_states):
        raise ValueError("affine teacher and requested historical states differ")
    location = teacher_state["location_shift"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    spot_loading = teacher_state["spot_loading"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    raw_noise = teacher_state["raw_noise_std"].to(device=device, dtype=dtype)[
        : len(current_front_states)
    ]
    teacher_config = cast(Mapping[str, Any], affine_payload["law_config"])
    noise = float(teacher_config["minimum_state_noise_std"]) + torch.nn.functional.softplus(
        raw_noise
    )
    innovation_scale = torch.sqrt(
        torch.mean(
            location.square() + spot_loading.square() + noise[:, None].square(),
            dim=0,
        )
    ).clamp_min(1.0e-2)

    validation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    validation_states = (
        torch.linspace(
            0,
            len(current_front_states) - 1,
            min(128, len(current_front_states)),
            dtype=torch.float64,
            device=device,
        )
        .round()
        .long()
    )
    validation_rank = torch.randn(
        (len(validation_states), paths_per_state),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_epsilon = torch.randn(
        (paths_per_state, front_state_size),
        dtype=dtype,
        device=device,
        generator=validation_generator,
    )
    validation_target = (
        current_front_states[validation_states, None]
        + location[validation_states, None]
        + validation_rank[..., None] * spot_loading[validation_states, None]
        + noise[validation_states, None, None] * validation_epsilon[None]
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(law.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(steps):
        indices = torch.randint(
            len(current_front_states),
            (min(states_per_step, len(current_front_states)),),
            device=device,
            generator=generator,
        )
        spot_rank = torch.randn(
            (len(indices), paths_per_state),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        epsilon = torch.randn(
            (paths_per_state, front_state_size),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        target = (
            current_front_states[indices, None]
            + location[indices, None]
            + spot_rank[..., None] * spot_loading[indices, None]
            + noise[indices, None, None] * epsilon[None]
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = law(
            current_front_states=current_front_states[indices],
            spot_rank=spot_rank,
            epsilon=epsilon,
        )
        scaled_error = (prediction - target) / innovation_scale
        loss = torch.mean(scaled_error.square())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(law.parameters(), 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            with torch.no_grad():
                validation_prediction = law(
                    current_front_states=current_front_states[validation_states],
                    spot_rank=validation_rank,
                    epsilon=validation_epsilon,
                )
                validation_rmse = torch.sqrt(
                    torch.mean(
                        ((validation_prediction - validation_target) / innovation_scale).square()
                    )
                )
            record = {
                "step": float(step + 1),
                "fit_normalized_rmse": float(torch.sqrt(loss.detach())),
                "validation_normalized_rmse": float(validation_rmse),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            print(
                f"affine_distill step={step + 1} "
                f"fit_rmse={record['fit_normalized_rmse']:.5f} "
                f"validation_rmse={record['validation_normalized_rmse']:.5f}",
                flush=True,
            )
    return {
        "steps": steps,
        "states_per_step": states_per_step,
        "paths_per_state_per_step": paths_per_state,
        "learning_rate": learning_rate,
        "seed": seed,
        "branch_component_included": False,
        "innovation_scale": innovation_scale.detach().cpu().tolist(),
        "history": history,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-completion", type=Path, default=_DEFAULT_JOINT_COMPLETION)
    parser.add_argument("--teacher-checkpoint", type=Path, default=_DEFAULT_TEACHER)
    parser.add_argument(
        "--affine-teacher-checkpoint",
        type=Path,
        default=_DEFAULT_AFFINE_TEACHER,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "reports/all-state-joint-generic-q.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "checkpoints/all-state-joint-generic-q.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--maximum-shift", type=float, default=16.0)
    parser.add_argument("--maximum-log-scale", type=float, default=5.5)
    parser.add_argument("--maximum-lower-loading", type=float, default=6.0)
    parser.add_argument("--affine-distillation-steps", type=int, default=1_000)
    parser.add_argument("--affine-distillation-states", type=int, default=128)
    parser.add_argument("--affine-distillation-paths", type=int, default=128)
    parser.add_argument("--affine-distillation-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--affine-distillation-seed", type=int, default=12_482_101)
    parser.add_argument("--state-batch-size", type=int, default=64)
    parser.add_argument("--fit-replications", type=int, default=24)
    parser.add_argument("--validation-replications", type=int, default=24)
    parser.add_argument("--confirmation-replications", type=int, default=24)
    parser.add_argument("--spot-contexts", type=int, default=125)
    parser.add_argument("--surface-paths", type=int, default=8)
    parser.add_argument("--fit-seed", type=int, default=12_484_101)
    parser.add_argument("--validation-seed", type=int, default=12_485_101)
    parser.add_argument("--confirmation-seed", type=int, default=12_487_101)
    parser.add_argument("--order-seed", type=int, default=12_483_101)
    parser.add_argument("--validation-z-target", type=float, default=2.0)
    parser.add_argument("--confirmation-z-target", type=float, default=2.0)
    parser.add_argument("--soft-maximum-temperature", type=float, default=1.0)
    parser.add_argument("--plateau-sweeps", type=int, default=12)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1.0e-4)
    parser.add_argument("--maximum-states", type=int)
    parser.add_argument("--audit-initial-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    counts = (
        arguments.sweeps,
        arguments.state_batch_size,
        arguments.affine_distillation_steps,
        arguments.affine_distillation_states,
        arguments.affine_distillation_paths,
        arguments.fit_replications,
        arguments.validation_replications,
        arguments.confirmation_replications,
        arguments.spot_contexts,
        arguments.surface_paths,
        arguments.plateau_sweeps,
    )
    if arguments.device != "cuda:0" or any(value <= 0 for value in counts):
        raise ValueError("all-state joint generic Q requires GPU0 and positive counts")
    if (
        min(
            arguments.fit_replications,
            arguments.validation_replications,
            arguments.confirmation_replications,
        )
        < 2
    ):
        raise ValueError("all-state joint generic MC errors need two replications")
    if arguments.surface_paths < 4 or arguments.surface_paths % 2:
        raise ValueError("all-state surface paths must be at least four and even")
    if (
        len(
            {
                arguments.fit_seed,
                arguments.validation_seed,
                arguments.confirmation_seed,
            }
        )
        != 3
    ):
        raise ValueError("all-state fit and audit streams must be disjoint")
    if (
        arguments.learning_rate <= 0.0
        or arguments.maximum_shift <= 0.0
        or arguments.maximum_log_scale <= 0.0
        or arguments.maximum_lower_loading <= 0.0
        or arguments.affine_distillation_learning_rate <= 0.0
        or arguments.validation_z_target <= 0.0
        or arguments.confirmation_z_target <= 0.0
        or arguments.soft_maximum_temperature <= 0.0
        or not 0.0 < arguments.minimum_relative_improvement < 1.0
    ):
        raise ValueError("all-state joint generic rates and targets are invalid")
    if arguments.maximum_states is not None and arguments.maximum_states <= 0:
        raise ValueError("maximum states must be positive")


def _audit_log(label: str, audit: Mapping[str, Any], *, seconds: float) -> None:
    print(
        f"{label} raw_worst={audit['maximum_abs_raw_mean_z']:.3f} "
        f"raw_worst10={audit['worst_10_percent_mean_abs_raw_mean_z']:.3f} "
        "adjusted_worst="
        f"{audit['maximum_abs_noise_scaled_adjusted_mean_z']:.3f} "
        "adjusted_worst10="
        f"{audit['worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z']:.3f} "
        f"worst_state={audit['worst_state_index']} "
        f"worst_instrument={audit['worst_instrument_index']} "
        f"seconds={seconds:.1f}",
        flush=True,
    )


def run(arguments: argparse.Namespace) -> None:
    """Fit and independently audit one state-conditioned history-wide map."""

    _validate_arguments(arguments)
    torch.set_float32_matmul_precision("high")
    env = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    (
        model,
        train_dates,
        train_states,
        _,
        validation_dates,
        validation_states,
        _,
        sources,
    ) = _load_inputs(arguments.joint_completion, env=env)
    date_values = train_dates + validation_dates
    state_kinds = ("model_fit",) * len(train_dates) + ("held_out_2025",) * len(validation_dates)
    current_states = torch.cat((train_states, validation_states), dim=0)
    if arguments.maximum_states is not None:
        current_states = current_states[: arguments.maximum_states]
        date_values = date_values[: arguments.maximum_states]
        state_kinds = state_kinds[: arguments.maximum_states]
    num_states = len(current_states)
    history_indices = tuple(range(num_states))
    dates = tuple(value.isoformat() for value in date_values)
    if not arguments.no_compile:
        model.sanos_decoder.enable_solver_compile()
    front_state_size = model.config.state_sizes_by_expiry[0]
    config = JointGenericTransportConfig(
        front_state_size=front_state_size,
        maximum_shift=arguments.maximum_shift,
        maximum_log_scale=arguments.maximum_log_scale,
        maximum_lower_loading=arguments.maximum_lower_loading,
    )
    law = JointGenericFrontTransport(env, config=config)

    teacher = torch.load(
        arguments.teacher_checkpoint,
        map_location=env.device,
        weights_only=False,
    )
    if (
        not isinstance(teacher, Mapping)
        or teacher.get("schema") != "flex-six-state-joint-generic-q-training-v1"
        or teacher.get("source_completion_sha256") != sources["completion_sha256"]
    ):
        raise ValueError("six-state generic teacher is incompatible")
    teacher_config = cast(Mapping[str, Any], teacher["law_config"])
    if int(teacher_config["front_state_size"]) != front_state_size:
        raise ValueError("six-state teacher front-state width is incompatible")
    law.load_state_dict(cast(Mapping[str, torch.Tensor], teacher["law_state_dict"]))
    teacher_provenance = {
        "checkpoint": str(arguments.teacher_checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.teacher_checkpoint),
        "source_epochs": len(cast(list[Any], teacher["history"])),
        "source_law_config": dict(teacher_config),
        "weights_loaded_into_wider_bounds": True,
    }
    affine_teacher = torch.load(
        arguments.affine_teacher_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(affine_teacher, Mapping)
        or affine_teacher.get("schema") != "flex-all-state-vector-q-training-v1"
        or affine_teacher.get("source_completion_sha256") != sources["completion_sha256"]
    ):
        raise ValueError("all-state affine teacher is incompatible")
    affine_provenance = {
        "checkpoint": str(arguments.affine_teacher_checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.affine_teacher_checkpoint),
        "source_epochs": len(cast(list[Any], affine_teacher["history"])),
    }
    affine_distillation: dict[str, Any] | None = None
    if not arguments.checkpoint.exists():
        affine_distillation = _distill_affine_history(
            law=law,
            current_front_states=current_states[:, :front_state_size],
            affine_payload=affine_teacher,
            steps=arguments.affine_distillation_steps,
            states_per_step=arguments.affine_distillation_states,
            paths_per_state=arguments.affine_distillation_paths,
            learning_rate=arguments.affine_distillation_learning_rate,
            seed=arguments.affine_distillation_seed,
        )
    del affine_teacher

    normalized_strikes = torch.exp(
        torch.linspace(-0.30, 0.30, 11, dtype=env.fdtype, device=env.device)
    )
    option_is_call = normalized_strikes <= 1.0
    with torch.no_grad():
        raw = model.raw_states_from_states(current_states)
        parameters = bound_native_raw(raw, geometry=model.config.geometry)
        current_surface = model.sanos_decoder.forward_batch(
            dlvsqs=parameters.dlvsqs,
            forward_vols=parameters.forward_vols,
            eta=parameters.eta,
        )
        current_w2_calls = model.sanos_decoder.price_calls_batch(
            current_surface,
            expiries_y=model.sanos_decoder.model_expiries_y[1].expand(
                num_states, len(normalized_strikes)
            ),
            strikes=normalized_strikes[None].expand(num_states, -1),
        ).detach()
        support, density, sqrt_variance = normalize_front_components(
            current_surface.support_strikes[:, 0],
            current_surface.densities[:, 0],
            current_surface.backbone_total_vols[:, 0],
        )

    fit_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.fit_replications,
        seed=arguments.fit_seed,
    )
    validation_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.validation_replications,
        seed=arguments.validation_seed,
    )
    confirmation_draws = _draws(
        env=env,
        front_state_size=front_state_size,
        spot_contexts=arguments.spot_contexts,
        surface_paths=arguments.surface_paths,
        replications=arguments.confirmation_replications,
        seed=arguments.confirmation_seed,
    )
    simulation_config = {
        "num_states": num_states,
        "date_range": [dates[0], dates[-1]],
        "fit_replications": arguments.fit_replications,
        "validation_replications": arguments.validation_replications,
        "confirmation_replications": arguments.confirmation_replications,
        "spot_contexts": arguments.spot_contexts,
        "surface_paths": arguments.surface_paths,
        "fit_seed": arguments.fit_seed,
        "validation_seed": arguments.validation_seed,
        "confirmation_seed": arguments.confirmation_seed,
        "affine_distillation_steps": arguments.affine_distillation_steps,
        "affine_distillation_states": arguments.affine_distillation_states,
        "affine_distillation_paths": arguments.affine_distillation_paths,
        "affine_distillation_learning_rate": (arguments.affine_distillation_learning_rate),
        "affine_distillation_seed": arguments.affine_distillation_seed,
        "affine_teacher_sha256": affine_provenance["checkpoint_sha256"],
        "common_random_numbers_across_states": True,
        "actual_expected_gross_spot_return": 1.0,
    }

    optimizer = torch.optim.Adam(law.parameters(), lr=arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=2.5e-6,
    )
    history: list[dict[str, float]] = []
    start_sweep = 0
    best_fit_loss = math.inf
    stale_sweeps = 0
    checkpoint: dict[str, Any] | None = None
    if arguments.checkpoint.exists():
        payload = torch.load(
            arguments.checkpoint,
            map_location=env.device,
            weights_only=False,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "flex-all-state-joint-generic-q-training-v1"
            or payload.get("source_completion_sha256") != sources["completion_sha256"]
            or payload.get("law_config") != asdict(config)
            or payload.get("simulation_config") != simulation_config
        ):
            raise ValueError("all-state joint generic checkpoint is incompatible")
        law.load_state_dict(cast(Mapping[str, torch.Tensor], payload["law_state_dict"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state_dict"]))
        scheduler.load_state_dict(cast(dict[str, Any], payload["scheduler_state_dict"]))
        history = list(cast(list[dict[str, float]], payload["history"]))
        start_sweep = len(history)
        best_fit_loss = float(payload["best_fit_loss"])
        stale_sweeps = int(payload["stale_sweeps"])
        affine_distillation = dict(cast(Mapping[str, Any], payload["affine_distillation"]))
        checkpoint = dict(payload)
        print(f"resuming all-state joint generic Q at sweep {start_sweep}", flush=True)

    started = time.perf_counter()
    validation_audit = _audit_all_states(
        model=model,
        law=law,
        history_indices=history_indices,
        dates=dates,
        state_kinds=state_kinds,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=validation_draws,
        state_batch_size=arguments.state_batch_size,
        z_target=arguments.validation_z_target,
    )
    _audit_log(
        "initial_validation",
        validation_audit,
        seconds=time.perf_counter() - started,
    )

    stopped_for_validation = bool(validation_audit["passes_target"])
    stopped_for_plateau = False
    if not arguments.audit_initial_only and not stopped_for_validation:
        order_generator = torch.Generator(device="cpu")
        for sweep in range(start_sweep, arguments.sweeps):
            sweep_started = time.perf_counter()
            order_generator.manual_seed(arguments.order_seed + sweep)
            order = torch.randperm(num_states, generator=order_generator)
            batch_metrics: list[dict[str, float]] = []
            for start in range(0, num_states, arguments.state_batch_size):
                indices = order[start : start + arguments.state_batch_size].to(device=env.device)
                batch_metrics.append(
                    _train_state_batch(
                        model=model,
                        law=law,
                        optimizer=optimizer,
                        current_states=current_states[indices],
                        current_w2_calls=current_w2_calls[indices],
                        normalized_strikes=normalized_strikes,
                        option_is_call=option_is_call,
                        support=support[indices],
                        density=density[indices],
                        sqrt_variance=sqrt_variance[indices],
                        draws=fit_draws,
                        temperature=arguments.soft_maximum_temperature,
                    )
                )
            fit_loss = sum(value["loss"] for value in batch_metrics) / len(batch_metrics)
            scheduler.step(fit_loss)
            if fit_loss < best_fit_loss * (1.0 - arguments.minimum_relative_improvement):
                best_fit_loss = fit_loss
                stale_sweeps = 0
            else:
                stale_sweeps += 1
            audit_started = time.perf_counter()
            validation_audit = _audit_all_states(
                model=model,
                law=law,
                history_indices=history_indices,
                dates=dates,
                state_kinds=state_kinds,
                current_states=current_states,
                current_w2_calls=current_w2_calls,
                normalized_strikes=normalized_strikes,
                option_is_call=option_is_call,
                support=support,
                density=density,
                sqrt_variance=sqrt_variance,
                draws=validation_draws,
                state_batch_size=arguments.state_batch_size,
                z_target=arguments.validation_z_target,
            )
            audit_seconds = time.perf_counter() - audit_started
            record = {
                "sweep": float(sweep + 1),
                "fit_batch_mean_loss": fit_loss,
                "fit_batch_maximum_abs_raw_mean_z": max(
                    value["raw_worst"] for value in batch_metrics
                ),
                "fit_batch_maximum_abs_noise_scaled_adjusted_mean_z": max(
                    value["adjusted_worst"] for value in batch_metrics
                ),
                "validation_maximum_abs_raw_mean_z": float(
                    validation_audit["maximum_abs_raw_mean_z"]
                ),
                "validation_worst_10_percent_mean_abs_raw_mean_z": float(
                    validation_audit["worst_10_percent_mean_abs_raw_mean_z"]
                ),
                "validation_maximum_abs_noise_scaled_adjusted_mean_z": float(
                    validation_audit["maximum_abs_noise_scaled_adjusted_mean_z"]
                ),
                "validation_worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z": float(
                    validation_audit["worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z"]
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "sweep_seconds": time.perf_counter() - sweep_started,
                "validation_audit_seconds": audit_seconds,
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise FloatingPointError("non-finite all-state joint training record")
            history.append(record)
            checkpoint = {
                "schema": "flex-all-state-joint-generic-q-training-v1",
                "status": "training",
                "objective": OBJECTIVE_ID,
                "source_completion_sha256": sources["completion_sha256"],
                "law_config": asdict(config),
                "simulation_config": simulation_config,
                "teacher_provenance": teacher_provenance,
                "affine_teacher_provenance": affine_provenance,
                "affine_distillation": affine_distillation,
                "law_state_dict": law.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_fit_loss": best_fit_loss,
                "stale_sweeps": stale_sweeps,
            }
            _atomic_torch_save(checkpoint, arguments.checkpoint)
            _audit_log(
                f"sweep={sweep + 1} validation",
                validation_audit,
                seconds=record["sweep_seconds"],
            )
            if bool(validation_audit["passes_target"]):
                stopped_for_validation = True
                break
            if stale_sweeps >= arguments.plateau_sweeps:
                stopped_for_plateau = True
                break

    if arguments.audit_initial_only:
        write_json_atomic(
            {
                "schema": "flex-all-state-joint-generic-q-initial-audit-v1",
                "status": "complete",
                "objective": OBJECTIVE_ID,
                "teacher_provenance": teacher_provenance,
                "affine_teacher_provenance": affine_provenance,
                "affine_distillation": affine_distillation,
                "simulation_config": simulation_config,
                "independent_validation": validation_audit,
                "sources": sources,
            },
            arguments.output,
        )
        return

    if checkpoint is None:
        checkpoint = {
            "schema": "flex-all-state-joint-generic-q-training-v1",
            "status": "training",
            "objective": OBJECTIVE_ID,
            "source_completion_sha256": sources["completion_sha256"],
            "law_config": asdict(config),
            "simulation_config": simulation_config,
            "teacher_provenance": teacher_provenance,
            "affine_teacher_provenance": affine_provenance,
            "affine_distillation": affine_distillation,
            "law_state_dict": law.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_fit_loss": best_fit_loss,
            "stale_sweeps": stale_sweeps,
        }
        _atomic_torch_save(checkpoint, arguments.checkpoint)

    confirmation_started = time.perf_counter()
    confirmation_audit = _audit_all_states(
        model=model,
        law=law,
        history_indices=history_indices,
        dates=dates,
        state_kinds=state_kinds,
        current_states=current_states,
        current_w2_calls=current_w2_calls,
        normalized_strikes=normalized_strikes,
        option_is_call=option_is_call,
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        draws=confirmation_draws,
        state_batch_size=arguments.state_batch_size,
        z_target=arguments.confirmation_z_target,
    )
    _audit_log(
        "independent_confirmation",
        confirmation_audit,
        seconds=time.perf_counter() - confirmation_started,
    )
    semantic_success = bool(
        validation_audit["passes_target"] and confirmation_audit["passes_target"]
    )
    report = {
        "schema": "flex-all-state-joint-generic-q-report-v1",
        "status": "complete" if semantic_success else "failed_semantic_validation",
        "semantic_success": semantic_success,
        "objective": OBJECTIVE_ID,
        "num_states": num_states,
        "num_model_fit_states": sum(value == "model_fit" for value in state_kinds),
        "num_held_out_2025_states": sum(value == "held_out_2025" for value in state_kinds),
        "date_range": [dates[0], dates[-1]],
        "law": {
            "form": "one triangular T_theta(h1_0,spot_rank,epsilon)",
            "full_support": True,
            "discrete_branch": False,
            "num_parameters": sum(value.numel() for value in law.parameters()),
        },
        "teacher_provenance": teacher_provenance,
        "affine_teacher_provenance": affine_provenance,
        "affine_distillation": affine_distillation,
        "simulation_config": simulation_config,
        "training": {
            "sweeps_completed": len(history),
            "best_fit_loss": best_fit_loss,
            "stopped_for_validation_target": stopped_for_validation,
            "stopped_for_plateau": stopped_for_plateau,
            "history": history,
        },
        "independent_validation": validation_audit,
        "independent_confirmation": confirmation_audit,
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
            "schema": "flex-all-state-joint-generic-q-completion-v1",
            "status": "complete" if semantic_success else "FAILED_TERMINAL",
            "semantic_success": semantic_success,
            "report": str(arguments.output),
            "report_sha256": checkpoint_digest(arguments.output),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        },
        completion,
    )
    print(
        json.dumps(
            {
                "validation": {
                    key: validation_audit[key]
                    for key in (
                        "maximum_abs_raw_mean_z",
                        "worst_10_percent_mean_abs_raw_mean_z",
                        "maximum_abs_noise_scaled_adjusted_mean_z",
                        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z",
                        "worst_state_index",
                        "worst_instrument_index",
                    )
                },
                "confirmation": {
                    key: confirmation_audit[key]
                    for key in (
                        "maximum_abs_raw_mean_z",
                        "worst_10_percent_mean_abs_raw_mean_z",
                        "maximum_abs_noise_scaled_adjusted_mean_z",
                        "worst_10_percent_mean_abs_noise_scaled_adjusted_mean_z",
                        "worst_state_index",
                        "worst_instrument_index",
                    )
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if not semantic_success:
        raise RuntimeError("all-state joint generic Q failed an audit target")


def main() -> None:
    arguments = _parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(arguments.checkpoint.with_suffix(".lock")):
        run(arguments)


if __name__ == "__main__":
    run_guarded_cli(
        main,
        program="run_all_state_joint_generic_q_existence",
        clean_sigterm=True,
    )
