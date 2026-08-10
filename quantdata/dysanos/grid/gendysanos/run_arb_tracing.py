"""Run orthogonal-state iterative direction/notional conditional-arbitrage traces."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.torch import BaseEnvironment

from .arb_tracing import (
    ArbTracingConfig,
    ArbTracingTrainConfig,
    ArbTracingUtility,
    OrthogonalArbDirection,
    fit_physical_state_whitener,
    traced_scale_statistics,
    tracing_numerical_band,
    tracing_pnl,
    tracing_pnl_right_derivative,
    tracing_raw_norm_penalty,
    tracing_utility_objective,
    weighted_quantile,
)
from .required_runs import RequiredCase
from .run_expiry_arbitrage_experiments import (
    _DEFAULT_CACHE_ROOT,
    DeviceExpiryCase,
    _load_case,
    _to_device,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/arb-tracing"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/arb-tracing/checkpoints"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_loss(
    model: OrthogonalArbDirection,
    data: DeviceExpiryCase,
    base_action: torch.Tensor,
    indices: torch.Tensor,
    *,
    active_time: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    direction, raw_l1 = model(data.state[indices])
    total_action = base_action[indices] + model.config.probe_notional * direction
    pnl = tracing_pnl(total_action, data.returns[indices], data.costs[indices])
    objective = tracing_utility_objective(
        pnl,
        data.path_weight[indices],
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    raw_penalty = tracing_raw_norm_penalty(
        raw_l1,
        soft_limit=model.config.raw_norm_soft_limit,
    )
    active_count = active_time.sum().clamp_min(1.0)
    loss = -torch.sum(active_time * objective) / active_count
    loss = loss + model.config.raw_norm_penalty * torch.sum(
        active_time * raw_penalty
    ) / active_count
    if not torch.isfinite(loss).item():
        raise FloatingPointError("arb-tracing objective became NaN or infinity")
    return loss, {
        "objective": objective,
        "raw_norm_penalty": raw_penalty,
        "direction_l1_mean": direction.abs().sum(dim=(-1, -2)).mean(dim=0),
        "direction_l1_maximum": direction.abs().sum(dim=(-1, -2)).amax(dim=0),
        "raw_l1_maximum": raw_l1.amax(dim=0),
    }


def _checkpoint(
    path: Path,
    *,
    model: OrthogonalArbDirection,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    tracker: list[dict[str, Any]],
) -> None:
    model_device = torch.device(model.device)
    value = {
        "model_code_version": ArbTracingConfig.version.unique_id48,
        "user_version": model.config.user_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tracker": tracker,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(model_device)
            if model_device.type == "cuda"
            else None
        ),
        "batch_generator_state": generator.get_state(),
    }
    _atomic_torch_save(path, value)


def _train_direction(
    data: DeviceExpiryCase,
    base_action: torch.Tensor,
    *,
    active_time: torch.Tensor,
    model_config: ArbTracingConfig,
    train_config: ArbTracingTrainConfig,
    checkpoint_path: Path,
    env: BaseEnvironment,
    state_mean: torch.Tensor,
    state_whitener: torch.Tensor,
) -> tuple[OrthogonalArbDirection, dict[str, Any]]:
    device = torch.device(env.device)
    model = OrthogonalArbDirection(
        env,
        config=model_config,
        state_mean=state_mean,
        state_whitener=state_whitener,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    generator = torch.Generator(device=env.device)
    generator.manual_seed(train_config.seed)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        if (
            saved.get("model_code_version") != ArbTracingConfig.version.unique_id48
            or saved.get("user_version") != model_config.user_version
        ):
            print("Arb-tracing checkpoint version changed; cold-starting.", flush=True)
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            tracker = list(saved.get("tracker", []))
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
            generator.set_state(saved["batch_generator_state"].cpu())
            print(f"Warm-started arb direction from step {start_step:,}.", flush=True)

    callable_model = torch.compile(model, dynamic=False) if train_config.compile else model
    fixed_generator = torch.Generator(device=env.device)
    fixed_generator.manual_seed(train_config.seed + 1)
    fixed_draw = torch.randperm(
        data.reference.numel(), generator=fixed_generator, device=env.device
    )[: min(train_config.valuation_batch_size, data.reference.numel())]
    fixed = data.reference[fixed_draw]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    converged = False
    for step in range(start_step + 1, train_config.epochs + 1):
        draw = torch.randint(
            data.reference.numel(),
            (train_config.batch_size,),
            generator=generator,
            device=env.device,
        )
        indices = data.reference[draw]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(
            callable_model,
            data,
            base_action,
            indices,
            active_time=active_time,
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("arb-tracing gradient became NaN or infinity")
        optimizer.step()
        if step % train_config.value_update == 0 or step == train_config.epochs:
            with torch.no_grad():
                fixed_loss, values = _batch_loss(
                    model,
                    data,
                    base_action,
                    fixed,
                    active_time=active_time,
                )
            item = {
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "training_loss": float(fixed_loss.cpu()),
                "objective_by_time": values["objective"].cpu().tolist(),
                "raw_norm_penalty_by_time": values["raw_norm_penalty"].cpu().tolist(),
                "direction_l1_mean_by_time": values["direction_l1_mean"].cpu().tolist(),
                "direction_l1_maximum_by_time": values[
                    "direction_l1_maximum"
                ].cpu().tolist(),
                "raw_l1_maximum_by_time": values["raw_l1_maximum"].cpu().tolist(),
            }
            tracker.append(item)
            recent = tracker[-train_config.convergence_points :]
            if step >= train_config.minimum_epochs and len(recent) == train_config.convergence_points:
                loss_values = np.asarray(
                    [point["training_loss"] for point in recent], dtype=np.float64
                )
                scale = max(1.0e-8, float(np.max(np.abs(loss_values))))
                converged = bool(
                    np.ptp(loss_values)
                    <= train_config.convergence_relative_tolerance * scale
                )
            print(
                f"[{model_config.utility}/stage={model_config.label.rsplit('-', 1)[-1]}] "
                f"step {step:,}: loss={item['training_loss']:.8g}, "
                f"direction L1 max={max(item['direction_l1_maximum_by_time']):.4g}, "
                f"raw L1 max={max(item['raw_l1_maximum_by_time']):.4g}, "
                f"converged={converged}",
                flush=True,
            )
            if step % train_config.checkpoint_update == 0 or converged:
                _checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    step=step,
                    tracker=tracker,
                )
            if converged:
                break

    final_step = tracker[-1]["step"] if tracker else start_step
    if not checkpoint_path.exists() or final_step % train_config.checkpoint_update != 0:
        _checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            generator=generator,
            step=final_step,
            tracker=tracker,
        )
    return model, {
        "steps": final_step,
        "converged": converged,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "constant_learning_rate": train_config.lr,
        "gradient_clipping": False,
        "tracker": tracker,
    }


def _gradient_diagnostics(
    model: OrthogonalArbDirection,
    data: DeviceExpiryCase,
    base_action: torch.Tensor,
    indices: torch.Tensor,
    *,
    active_time: torch.Tensor,
    sample_paths: int,
) -> dict[str, Any]:
    """Measure per-time parameter gradients without clipping or rescaling."""

    sample = indices[: min(sample_paths, indices.numel())]
    model.zero_grad(set_to_none=True)
    loss, values = _batch_loss(
        model,
        data,
        base_action,
        sample,
        active_time=active_time,
    )
    loss.backward()
    active_count = float(active_time.sum().clamp_min(1.0).cpu())
    norms: list[float] = []
    maxima: list[float] = []
    finite = True
    for time, network in enumerate(model.networks):
        gradients = [
            parameter.grad.detach().reshape(-1)
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        flat = (
            torch.cat(gradients)
            if gradients
            else torch.zeros(1, dtype=model.fdtype, device=model.device)
        )
        # The training loss averages the independent time objectives.  Undo
        # that averaging so gradient sizes are comparable across active sets.
        flat = active_count * flat
        finite = finite and bool(torch.isfinite(flat).all().item())
        norms.append(float(torch.linalg.vector_norm(flat).cpu()))
        maxima.append(float(flat.abs().max().cpu()))
    model.zero_grad(set_to_none=True)
    if not finite:
        raise FloatingPointError("arb-tracing diagnostic gradient is non-finite")
    return {
        "sample_paths": int(sample.numel()),
        "loss": float(loss.detach().cpu()),
        "objective_by_time": values["objective"].detach().cpu().tolist(),
        "parameter_gradient_l2_by_time": norms,
        "parameter_gradient_maximum_absolute_by_time": maxima,
        "active_parameter_gradient_l2_minimum": float(
            min(norm for norm, active in zip(norms, active_time.tolist(), strict=True) if active)
        ),
        "active_parameter_gradient_l2_median": float(
            np.median(
                [
                    norm
                    for norm, active in zip(norms, active_time.tolist(), strict=True)
                    if active
                ]
            )
        ),
        "active_parameter_gradient_l2_maximum": float(
            max(norm for norm, active in zip(norms, active_time.tolist(), strict=True) if active)
        ),
    }


@torch.no_grad()
def _trace_split(
    model: OrthogonalArbDirection,
    data: DeviceExpiryCase,
    base_action: torch.Tensor,
    indices: torch.Tensor,
    *,
    notionals: list[float],
    batch_size: int,
    loss_probability_epsilon: float,
    confidence_z: float,
    loss_mc_multiplier: float,
) -> dict[str, Any]:
    paths = int(indices.numel())
    times = model.config.trading_times
    scales = len(notionals)
    pnl = np.empty((scales, paths, times), dtype=np.float64)
    band = np.empty_like(pnl)
    slope = np.empty_like(pnl)
    action_l1 = np.empty((scales, paths, times), dtype=np.float32)
    direction_l1 = np.empty((paths, times), dtype=np.float32)
    raw_l1 = np.empty((paths, times), dtype=np.float32)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        direction, batch_raw_l1 = model(data.state[batch])
        base = base_action[batch]
        direction_l1[start:stop] = direction.abs().sum(dim=(-1, -2)).cpu().numpy()
        raw_l1[start:stop] = batch_raw_l1.cpu().numpy()
        for scale_index, notional in enumerate(notionals):
            action = base + notional * direction
            pnl[scale_index, start:stop] = tracing_pnl(
                action, data.returns[batch], data.costs[batch]
            ).cpu().double().numpy()
            band[scale_index, start:stop] = tracing_numerical_band(
                action,
                data.returns[batch],
                data.costs[batch],
                data.numerical_band[batch],
            ).cpu().double().numpy()
            slope[scale_index, start:stop] = tracing_pnl_right_derivative(
                base,
                direction,
                notional,
                data.returns[batch],
                data.costs[batch],
            ).cpu().double().numpy()
            action_l1[scale_index, start:stop] = action.abs().sum(
                dim=(-1, -2)
            ).cpu().numpy()

    weight = data.path_weight[indices].cpu().double().numpy()
    component = data.component[indices].cpu().numpy()
    traces: list[dict[str, Any]] = []
    for scale_index, notional in enumerate(notionals):
        reports = traced_scale_statistics(
            pnl[scale_index],
            band[scale_index],
            slope[scale_index],
            weight,
            component,
            utility=model.config.utility,
            risk_aversion=model.config.risk_aversion,
            loss_probability_epsilon=loss_probability_epsilon,
            confidence_z=confidence_z,
            loss_mc_multiplier=loss_mc_multiplier,
        )
        for time, report in enumerate(reports):
            report["action_l1_P_median"] = weighted_quantile(
                action_l1[scale_index, :, time], weight, 0.50
            )
            report["action_l1_P_99_percentile"] = weighted_quantile(
                action_l1[scale_index, :, time], weight, 0.99
            )
            report["action_l1_proposal_maximum"] = float(
                np.max(action_l1[scale_index, :, time])
            )
        traces.append({"notional": notional, "times": reports})
    return {
        "paths": paths,
        "direct_P_paths": int(np.sum(component == 0)),
        "physical_weight_mean_before_split_normalization": float(weight.mean()),
        "physical_weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "direction_l1_P_median_by_time": [
            weighted_quantile(direction_l1[:, time], weight, 0.50)
            for time in range(times)
        ],
        "direction_l1_P_99_percentile_by_time": [
            weighted_quantile(direction_l1[:, time], weight, 0.99)
            for time in range(times)
        ],
        "direction_l1_proposal_maximum_by_time": direction_l1.max(axis=0).tolist(),
        "raw_l1_P_99_percentile_by_time": [
            weighted_quantile(raw_l1[:, time], weight, 0.99)
            for time in range(times)
        ],
        "raw_l1_proposal_maximum_by_time": raw_l1.max(axis=0).tolist(),
        "trace": traces,
    }


def _select_stage(
    training: dict[str, Any],
    validation: dict[str, Any],
    *,
    notionals: list[float],
    active_time: list[bool],
    confidence_z: float,
    minimum_marginal_ess: float,
    probe_notional: float,
) -> tuple[list[dict[str, Any]], list[float], list[bool]]:
    """Choose scales on training only, then classify on validation."""

    results: list[dict[str, Any]] = []
    base_update_scale = [0.0] * len(active_time)
    next_active = list(active_time)
    for time, active in enumerate(active_time):
        if not active:
            results.append(
                {
                    "trading_time_index": time,
                    "classification": "inactive-from-earlier-stage",
                    "selected_notional": 0.0,
                }
            )
            next_active[time] = False
            continue
        training_line = [point["times"][time] for point in training["trace"]]
        validation_line = [point["times"][time] for point in validation["trace"]]
        reliable = [
            index
            for index, item in enumerate(training_line)
            if item["marginal_density_effective_sample_size"] >= minimum_marginal_ess
        ]
        if not reliable:
            results.append(
                {
                    "trading_time_index": time,
                    "classification": "unresolved-no-reliable-scale",
                    "selected_notional": 0.0,
                }
            )
            next_active[time] = False
            continue
        objective = np.asarray(
            [training_line[index]["objective"] for index in reliable],
            dtype=np.float64,
        )
        best_index = reliable[int(np.argmax(objective))]
        last_reliable = reliable[-1]
        later_negative = any(
            item["normalized_marginal_derivative"]
            + confidence_z * item["normalized_marginal_derivative_standard_error"]
            < 0.0
            for item in training_line[best_index + 1 :]
        )
        later_objective_decrease = any(
            item["objective"] < training_line[best_index]["objective"]
            for item in training_line[best_index + 1 :]
        )
        training_candidate_indices = [
            index
            for index in reliable
            if index > 0
            and training_line[index]["near_nonnegative_within_mc_noise"]
            and training_line[index]["statistically_positive_mean"]
        ]
        training_candidate = bool(training_candidate_indices)
        selected_index = (
            training_candidate_indices[-1] if training_candidate else best_index
        )
        selected_training = training_line[selected_index]
        selected_validation = validation_line[selected_index]
        validation_candidate = bool(
            selected_validation["near_nonnegative_within_mc_noise"]
            and selected_validation["statistically_positive_mean"]
        )
        finite_optimum = bool(
            best_index > 0
            and best_index + 1 < len(training_line)
            and later_negative
            and later_objective_decrease
        )
        validation_objective_improvement = (
            selected_validation["objective"] - validation_line[0]["objective"]
        )
        material_validation_losses = bool(
            not selected_validation["near_nonnegative_within_mc_noise"]
        )
        if training_candidate and validation_candidate:
            classification = "near-arbitrage-candidate"
            # Keep the learned direction available without moving the base into
            # a numerically saturated notional region.
            base_update_scale[time] = min(notionals[selected_index], probe_notional)
            next_active[time] = False
        elif finite_optimum:
            base_update_scale[time] = notionals[best_index]
            if (
                validation_objective_improvement > 0.0
                and selected_validation["statistically_positive_mean"]
                and material_validation_losses
            ):
                classification = "high-quality-finite-risk-trade"
            else:
                classification = "training-finite-risk-trade-not-validated"
            # Continuation is a training decision.  Validation classifies the
            # frozen trade but must not select the next residual direction.
            next_active[time] = True
        elif best_index == 0 or validation_objective_improvement <= 0.0:
            classification = "no-validated-utility-improvement"
            base_update_scale[time] = 0.0
            next_active[time] = False
        elif best_index == last_reliable:
            classification = "unresolved-increasing-at-trace-boundary"
            base_update_scale[time] = min(notionals[best_index], probe_notional)
            next_active[time] = False
        else:
            classification = "finite-risk-trade"
            base_update_scale[time] = notionals[best_index]
            next_active[time] = True
        results.append(
            {
                "trading_time_index": time,
                "classification": classification,
                "selected_notional": notionals[selected_index],
                "training_best_utility_notional": notionals[best_index],
                "last_reliable_notional": notionals[last_reliable],
                "finite_training_optimum": finite_optimum,
                "training_near_arbitrage_candidate": training_candidate,
                "validation_near_arbitrage_candidate": validation_candidate,
                "base_update_notional": base_update_scale[time],
                "validation_objective_improvement": validation_objective_improvement,
                "training": selected_training,
                "validation": selected_validation,
            }
        )
    return results, base_update_scale, next_active


@torch.no_grad()
def _add_direction_to_base(
    base_action: torch.Tensor,
    model: OrthogonalArbDirection,
    state: torch.Tensor,
    scale_by_time: list[float],
    *,
    batch_size: int,
) -> None:
    scale = torch.as_tensor(
        scale_by_time,
        dtype=base_action.dtype,
        device=base_action.device,
    )[None, :, None, None]
    for start in range(0, state.shape[0], batch_size):
        stop = min(state.shape[0], start + batch_size)
        direction, _ = model(state[start:stop])
        base_action[start:stop].add_(scale * direction)
    if not torch.isfinite(base_action).all().item():
        raise FloatingPointError("updated arb-tracing base action is non-finite")


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=args.paths,
    )
    utility: ArbTracingUtility = args.utility
    loaded = _load_case(case, cache_root=Path(args.cache_root), need_primitives=False)
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=args.precision, device=args.device, debug=args.debug)
    device = torch.device(env.device)
    data, split_diagnostics = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    state_mean, state_whitener, whitening = fit_physical_state_whitener(
        data.state,
        data.path_weight,
        data.reference,
        relative_eigenvalue_floor=args.covariance_relative_floor,
    )
    base_action = torch.zeros_like(data.returns)
    active_time = [True] * data.state.shape[1]
    notionals = sorted(set(float(value) for value in args.notionals))
    verify_inp(notionals[0] == 0.0, "notional trace must begin at zero")
    verify_inp(all(value >= 0.0 and math.isfinite(value) for value in notionals), "invalid notionals")
    loss_probability_epsilon = (
        args.loss_probability_epsilon
        if args.loss_probability_epsilon is not None
        else 1.0 / args.paths
    )
    stages: list[dict[str, Any]] = []
    started = time.monotonic()
    label_root = (
        f"arb-tracing-{utility}-g{args.risk_aversion:.0e}-"
        f"{case.generator}-{case.path_count}"
    )
    for stage_index in range(args.stages):
        if not any(active_time):
            break
        label = f"{label_root}-stage-{stage_index + 1}"
        model_config = ArbTracingConfig(
            label=label,
            source_cache_uid=source_cache_uid,
            group=(
                f"gendysanos/arb-tracing/{case.generator}/{utility}/"
                f"gamma-{args.risk_aversion:.0e}/{case.path_count}/stage-{stage_index + 1}"
            ),
            user_version=args.user_version,
            state_features=data.state.shape[-1],
            risk_aversion=args.risk_aversion,
            utility=utility,
            probe_notional=args.probe_notional,
            raw_norm_penalty=args.raw_norm_penalty,
            raw_norm_soft_limit=args.raw_norm_soft_limit,
            covariance_relative_floor=args.covariance_relative_floor,
        )
        train_config = ArbTracingTrainConfig(
            batch_size=args.batch_size,
            valuation_batch_size=args.valuation_batch_size,
            epochs=args.epochs,
            minimum_epochs=args.minimum_epochs,
            value_update=args.value_update,
            checkpoint_update=args.checkpoint_update,
            lr=args.lr,
            compile=not args.no_compile,
            convergence_points=args.convergence_points,
            convergence_relative_tolerance=args.convergence_relative_tolerance,
            seed=args.seed + stage_index,
        )
        checkpoint_path = Path(args.checkpoint_root) / model_config.group / (
            f"{model_config.filename}.pt"
        )
        active_tensor = torch.as_tensor(
            active_time, dtype=env.fdtype, device=env.device
        )
        with FileLock(str(checkpoint_path) + ".lock"):
            model, training = _train_direction(
                data,
                base_action,
                active_time=active_tensor,
                model_config=model_config,
                train_config=train_config,
                checkpoint_path=checkpoint_path,
                env=env,
                state_mean=state_mean,
                state_whitener=state_whitener,
            )
            gradient_diagnostics = {
                "training": _gradient_diagnostics(
                    model,
                    data,
                    base_action,
                    data.reference,
                    active_time=active_tensor,
                    sample_paths=args.valuation_batch_size,
                ),
                "validation": _gradient_diagnostics(
                    model,
                    data,
                    base_action,
                    data.validation,
                    active_time=active_tensor,
                    sample_paths=args.valuation_batch_size,
                ),
            }
            training_trace = _trace_split(
                model,
                data,
                base_action,
                data.reference,
                notionals=notionals,
                batch_size=args.valuation_batch_size,
                loss_probability_epsilon=loss_probability_epsilon,
                confidence_z=args.confidence_z,
                loss_mc_multiplier=args.loss_mc_multiplier,
            )
            validation_trace = _trace_split(
                model,
                data,
                base_action,
                data.validation,
                notionals=notionals,
                batch_size=args.valuation_batch_size,
                loss_probability_epsilon=loss_probability_epsilon,
                confidence_z=args.confidence_z,
                loss_mc_multiplier=args.loss_mc_multiplier,
            )
            selection, base_scale, next_active = _select_stage(
                training_trace,
                validation_trace,
                notionals=notionals,
                active_time=active_time,
                confidence_z=args.confidence_z,
                minimum_marginal_ess=args.minimum_marginal_ess,
                probe_notional=args.probe_notional,
            )
            _add_direction_to_base(
                base_action,
                model,
                data.state,
                base_scale,
                batch_size=args.valuation_batch_size,
            )
        classification_counts: dict[str, int] = {}
        for result in selection:
            name = result["classification"]
            classification_counts[name] = classification_counts.get(name, 0) + 1
        stage = {
            "stage_index": stage_index + 1,
            "active_times_entering_stage": active_time,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "training": training,
            "gradient_diagnostics": gradient_diagnostics,
            "training_trace": training_trace,
            "validation_trace": validation_trace,
            "selection": selection,
            "classification_counts": classification_counts,
            "base_update_notional_by_time": base_scale,
            "checkpoint": str(checkpoint_path),
        }
        stages.append(stage)
        active_time = next_active
        interim = {
            "status": "running",
            "schema_version": "1.0",
            "experiment": "arb tracing",
            "configuration": asdict(case),
            "utility": utility,
            "risk_aversion": args.risk_aversion,
            "whitening": whitening,
            "stages": stages,
        }
        _atomic_json(Path(args.output_root) / f"{label_root}.json", interim)
        print(
            f"Completed {case.generator} {utility} arb-tracing stage {stage_index + 1}: "
            f"{classification_counts}",
            flush=True,
        )

    all_selection = [item for stage in stages for item in stage["selection"]]
    report = {
        "status": "complete",
        "schema_version": "1.0",
        "experiment": "arb tracing",
        "definition": {
            "state_basis": (
                "time-specific physical-measure whitening of x=(s,h), fitted "
                "on training paths only"
            ),
            "direction": (
                "ten independent a_t(x_t) networks; each jointly trades all "
                "seven expiries and 21 instruments per expiry; d=v/(1+||v||_1)"
            ),
            "net_pnl": "X_t(a)=a_t'dH_t-|a_t|'c_t with exact combined-position cost",
            "iteration": (
                "learn at moderate probe notional; trace a scalar notional; "
                "freeze finite-risk optima and learn a residual direction"
            ),
            "near_arbitrage": (
                "training and validation physical loss probabilities are no "
                "larger than loss_mc_multiplier times their path-level MC "
                "standard errors plus epsilon, and both physical means are "
                "statistically positive"
            ),
            "high_quality_trade": (
                "utility has a finite training optimum, validation utility and "
                "mean improve, but validation downside exceeds the MC-noise band"
            ),
        },
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "split": split_diagnostics,
        "utility": utility,
        "risk_aversion": args.risk_aversion,
        "probe_notional": args.probe_notional,
        "notional_grid": notionals,
        "confidence_z": args.confidence_z,
        "loss_mc_multiplier": args.loss_mc_multiplier,
        "loss_probability_epsilon": loss_probability_epsilon,
        "minimum_marginal_effective_sample_size": args.minimum_marginal_ess,
        "whitening": whitening,
        "stages": stages,
        "summary": {
            "near_arbitrage_candidate_count": sum(
                item["classification"] == "near-arbitrage-candidate"
                for item in all_selection
            ),
            "high_quality_finite_risk_trade_count": sum(
                item["classification"] == "high-quality-finite-risk-trade"
                for item in all_selection
            ),
            "unresolved_increasing_count": sum(
                item["classification"] == "unresolved-increasing-at-trace-boundary"
                for item in all_selection
            ),
            "final_active_times": active_time,
        },
        "runtime_seconds": time.monotonic() - started,
        "device": str(env.device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    report_path = Path(args.output_root) / f"{label_root}.json"
    _atomic_json(report_path, report)
    print(
        f"Completed {label_root}: {report['summary']}; report={report_path}",
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--utility", choices=("powerexp", "entropy"), required=True)
    parser.add_argument("--risk-aversion", type=float, default=100.0)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--probe-notional", type=float, default=2.0)
    parser.add_argument(
        "--notionals",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0],
    )
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--minimum-epochs", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=8_192)
    parser.add_argument("--value-update", type=int, default=250)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--convergence-points", type=int, default=12)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--raw-norm-penalty", type=float, default=1.0e-5)
    parser.add_argument("--raw-norm-soft-limit", type=float, default=8.0)
    parser.add_argument("--covariance-relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--confidence-z", type=float, default=3.0)
    parser.add_argument("--loss-mc-multiplier", type=float, default=1.0)
    parser.add_argument("--loss-probability-epsilon", type=float)
    parser.add_argument("--minimum-marginal-ess", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=91_301)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
