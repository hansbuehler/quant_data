"""Run fixed-L1-radius utility continuation for conditional arbitrage."""

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
from scipy.special import logsumexp

from quantdata.torch import BaseEnvironment

from .arb_tracing import (
    fit_physical_state_whitener,
    tracing_numerical_band,
    tracing_utility_objective,
    weighted_quantile,
)
from .norm_continuation import (
    NormContinuationAction,
    NormContinuationConfig,
    NormContinuationTrainConfig,
    NormContinuationUtility,
    clipped_radius_pnl_derivative,
    continuation_loss,
    continuation_pnl,
    exterior_l1_penalty,
    radial_l1_clip,
)
from .required_runs import RequiredCase
from .run_expiry_arbitrage_experiments import (
    _DEFAULT_CACHE_ROOT,
    DeviceExpiryCase,
    _load_case,
    _to_device,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/norm-continuation"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/norm-continuation/checkpoints"
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


def _weighted_mean_error(
    values: np.ndarray, normalized_weight: np.ndarray
) -> tuple[float, float]:
    mean = float(np.dot(normalized_weight, values))
    error = float(
        np.sqrt(np.sum(np.square(normalized_weight * (values - mean))))
    )
    return mean, error


def _batch_loss(
    model: NormContinuationAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    radius: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return continuation_loss(
        model,
        data.state[indices],
        data.returns[indices],
        data.costs[indices],
        data.path_weight[indices],
        radius=radius,
    )


def _checkpoint(
    path: Path,
    *,
    model: NormContinuationAction,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    radius_index: int,
    radius: float,
    step: int,
    tracker: list[dict[str, Any]],
    completed: bool,
) -> None:
    device = torch.device(model.device)
    _atomic_torch_save(
        path,
        {
            "model_code_version": NormContinuationConfig.version.unique_id48,
            "user_version": model.config.user_version,
            "radius_index": radius_index,
            "radius": radius,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tracker": tracker,
            "completed": completed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "batch_generator_state": generator.get_state(),
        },
    )


def _parameter_gradient_norms(
    model: NormContinuationAction, *, objective_count: int
) -> list[float]:
    result: list[float] = []
    for network in model.networks:
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
        flat = objective_count * flat
        if not torch.isfinite(flat).all().item():
            raise FloatingPointError("continuation parameter gradient is non-finite")
        result.append(float(torch.linalg.vector_norm(flat).cpu()))
    return result


def _gradient_diagnostics(
    model: NormContinuationAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    radius: float,
    sample_paths: int,
) -> dict[str, Any]:
    """Separate economic, projection, penalty, and parameter gradients."""

    sample = indices[: min(sample_paths, indices.numel())]
    state = data.state[sample]
    returns = data.returns[sample]
    costs = data.costs[sample]
    weight = data.path_weight[sample]
    times = model.config.trading_times

    model.zero_grad(set_to_none=True)
    total_loss, values = continuation_loss(
        model, state, returns, costs, weight, radius=radius
    )
    total_loss.backward()
    total_parameter_norm = _parameter_gradient_norms(model, objective_count=times)

    model.zero_grad(set_to_none=True)
    action, raw_l1, clipped = model(state, radius)
    pnl = continuation_pnl(action, returns, costs)
    objective = tracing_utility_objective(
        pnl,
        weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    utility_loss = -objective.mean()
    utility_loss.backward()
    utility_parameter_norm = _parameter_gradient_norms(model, objective_count=times)

    model.zero_grad(set_to_none=True)
    _, penalty_raw_l1, _ = model(state, radius)
    penalty_loss = (
        model.config.exterior_penalty
        * exterior_l1_penalty(penalty_raw_l1, radius).mean()
    )
    penalty_loss.backward()
    penalty_parameter_norm = _parameter_gradient_norms(model, objective_count=times)
    model.zero_grad(set_to_none=True)

    raw = model.raw_action(state).detach().requires_grad_(True)
    projected, diagnostic_raw_l1, diagnostic_clipped = radial_l1_clip(raw, radius)
    diagnostic_pnl = continuation_pnl(projected, returns, costs)
    diagnostic_objective = tracing_utility_objective(
        diagnostic_pnl,
        weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    raw_utility_gradient = torch.autograd.grad(
        -diagnostic_objective.mean(), raw, retain_graph=True
    )[0]
    raw_penalty_gradient = torch.autograd.grad(
        model.config.exterior_penalty
        * exterior_l1_penalty(diagnostic_raw_l1, radius).mean(),
        raw,
        retain_graph=True,
    )[0]

    action_leaf = projected.detach().requires_grad_(True)
    action_pnl = continuation_pnl(action_leaf, returns, costs)
    action_objective = tracing_utility_objective(
        action_pnl,
        weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    action_utility_gradient = torch.autograd.grad(-action_objective.mean(), action_leaf)[0]

    radial_scale = torch.ones(
        times, dtype=projected.dtype, device=projected.device, requires_grad=True
    )
    scaled_action = projected.detach() * radial_scale[None, :, None, None]
    scaled_pnl = continuation_pnl(scaled_action, returns, costs)
    scaled_objective = tracing_utility_objective(
        scaled_pnl,
        weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    radial_objective_derivative = torch.autograd.grad(
        scaled_objective.sum(), radial_scale
    )[0]
    radius_pnl_derivative = clipped_radius_pnl_derivative(
        diagnostic_pnl, diagnostic_clipped, radius
    )
    radius_scale = torch.ones(
        times,
        dtype=diagnostic_pnl.dtype,
        device=diagnostic_pnl.device,
        requires_grad=True,
    )
    radius_linearization = diagnostic_pnl.detach() + (
        radius_scale[None, :] - 1.0
    ) * radius_pnl_derivative.detach()
    radius_objective = tracing_utility_objective(
        radius_linearization,
        weight,
        utility=model.config.utility,
        risk_aversion=model.config.risk_aversion,
    )
    frozen_network_radius_derivative = torch.autograd.grad(
        radius_objective.sum(), radius_scale
    )[0]

    def tensor_norm_by_time(value: torch.Tensor) -> list[float]:
        flattened = value.transpose(0, 1).reshape(times, -1)
        return torch.linalg.vector_norm(flattened, dim=1).cpu().tolist()

    def tensor_max_by_time(value: torch.Tensor) -> list[float]:
        flattened = value.transpose(0, 1).reshape(times, -1)
        return flattened.abs().amax(dim=1).cpu().tolist()

    raw_norm = tensor_norm_by_time(raw_utility_gradient)
    action_norm = tensor_norm_by_time(action_utility_gradient)
    transmission = [
        raw_value / action_value if action_value > 0.0 else 0.0
        for raw_value, action_value in zip(raw_norm, action_norm, strict=True)
    ]
    finfo = torch.finfo(pnl.dtype)
    positive_exponential_underflow = (
        (pnl > 0.0)
        & (model.config.risk_aversion * pnl > -math.log(finfo.tiny))
    )
    return {
        "sample_paths": int(sample.numel()),
        "radius": radius,
        "loss": float(total_loss.detach().cpu()),
        "objective_by_time": values["objective"].detach().cpu().tolist(),
        "exterior_penalty_by_time": values["penalty"].detach().cpu().tolist(),
        "clipped_fraction_by_time": clipped.float().mean(dim=0).detach().cpu().tolist(),
        "raw_l1_maximum_by_time": raw_l1.amax(dim=0).detach().cpu().tolist(),
        "total_parameter_gradient_l2_by_time": total_parameter_norm,
        "utility_parameter_gradient_l2_by_time": utility_parameter_norm,
        "penalty_parameter_gradient_l2_by_time": penalty_parameter_norm,
        "raw_utility_gradient_l2_by_time": raw_norm,
        "raw_utility_gradient_maximum_absolute_by_time": tensor_max_by_time(
            raw_utility_gradient
        ),
        "raw_penalty_gradient_l2_by_time": tensor_norm_by_time(
            raw_penalty_gradient
        ),
        "action_space_utility_gradient_l2_by_time": action_norm,
        "projection_gradient_transmission_ratio_by_time": transmission,
        "multiplicative_action_scale_objective_derivative_by_time": (
            radial_objective_derivative.detach().cpu().tolist()
        ),
        "frozen_network_radius_objective_derivative_by_time": (
            frozen_network_radius_derivative.detach().cpu().tolist()
        ),
        "raw_utility_gradient_exact_zero_fraction": float(
            (raw_utility_gradient == 0.0).float().mean().cpu()
        ),
        "action_space_gradient_exact_zero_fraction": float(
            (action_utility_gradient == 0.0).float().mean().cpu()
        ),
        "positive_marginal_exponential_underflow_fraction": float(
            positive_exponential_underflow.float().mean().cpu()
        ),
        "maximum_risk_adjusted_pnl": float(
            (model.config.risk_aversion * pnl).amax().detach().cpu()
        ),
        "minimum_risk_adjusted_pnl": float(
            (model.config.risk_aversion * pnl).amin().detach().cpu()
        ),
        "diagnostic_clipped_fraction": float(
            diagnostic_clipped.float().mean().detach().cpu()
        ),
    }


def _train_radius(
    model: NormContinuationAction,
    callable_model: NormContinuationAction,
    data: DeviceExpiryCase,
    *,
    radius_index: int,
    radius: float,
    train_config: NormContinuationTrainConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    device = torch.device(model.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    generator = torch.Generator(device=model.device)
    generator.manual_seed(train_config.seed + radius_index)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        compatible = (
            saved.get("model_code_version")
            == NormContinuationConfig.version.unique_id48
            and saved.get("user_version") == model.config.user_version
            and saved.get("radius_index") == radius_index
            and saved.get("radius") == radius
        )
        if compatible:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            tracker = list(saved.get("tracker", []))
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
            generator.set_state(saved["batch_generator_state"].cpu())
            print(
                f"Warm-started radius n={radius:g} from step {start_step:,}.",
                flush=True,
            )
            if bool(saved.get("completed")):
                return {
                    "steps": start_step,
                    "converged": bool(saved.get("converged", True)),
                    "resumed_complete_checkpoint": True,
                    "elapsed_seconds_this_invocation": 0.0,
                    "tracker": tracker,
                }
        else:
            print("Continuation checkpoint version changed; cold-starting stage.", flush=True)

    fixed_generator = torch.Generator(device=model.device)
    fixed_generator.manual_seed(train_config.seed + 10_000 + radius_index)
    fixed_draw = torch.randperm(
        data.reference.numel(), generator=fixed_generator, device=model.device
    )[: min(train_config.valuation_batch_size, data.reference.numel())]
    fixed = data.reference[fixed_draw]
    compiled_radius = torch.as_tensor(
        radius, dtype=model.fdtype, device=model.device
    )
    started = time.monotonic()
    converged = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, train_config.epochs_per_radius + 1):
        draw = torch.randint(
            data.reference.numel(),
            (train_config.batch_size,),
            generator=generator,
            device=model.device,
        )
        indices = data.reference[draw]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(
            callable_model, data, indices, radius=compiled_radius
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("continuation gradient became NaN or infinity")
        optimizer.step()

        if step % train_config.value_update == 0 or step == train_config.epochs_per_radius:
            with torch.no_grad():
                fixed_loss, values = _batch_loss(model, data, fixed, radius=radius)
            item = {
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "loss": float(fixed_loss.cpu()),
                "objective_by_time": values["objective"].cpu().tolist(),
                "exterior_penalty_by_time": values["penalty"].cpu().tolist(),
                "action_l1_mean_by_time": values["action_l1_mean"].cpu().tolist(),
                "action_l1_maximum_by_time": values["action_l1_maximum"].cpu().tolist(),
                "raw_l1_mean_by_time": values["raw_l1_mean"].cpu().tolist(),
                "raw_l1_maximum_by_time": values["raw_l1_maximum"].cpu().tolist(),
                "clipped_fraction_by_time": values["clipped_fraction"].cpu().tolist(),
            }
            tracker.append(item)
            recent = tracker[-train_config.convergence_points :]
            if (
                step >= train_config.minimum_epochs_per_radius
                and len(recent) == train_config.convergence_points
            ):
                recent_loss = np.asarray(
                    [point["loss"] for point in recent], dtype=np.float64
                )
                scale = max(1.0e-8, float(np.max(np.abs(recent_loss))))
                converged = bool(
                    np.ptp(recent_loss)
                    <= train_config.convergence_relative_tolerance * scale
                )
            print(
                f"[n={radius:g}] step {step:,}: loss={item['loss']:.8g}, "
                f"action L1 max={max(item['action_l1_maximum_by_time']):.5g}, "
                f"raw L1 max={max(item['raw_l1_maximum_by_time']):.5g}, "
                f"clipped={max(item['clipped_fraction_by_time']):.1%}, "
                f"converged={converged}",
                flush=True,
            )
            if step % train_config.checkpoint_update == 0:
                _checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    radius_index=radius_index,
                    radius=radius,
                    step=step,
                    tracker=tracker,
                    completed=False,
                )
            if converged:
                break

    final_step = tracker[-1]["step"] if tracker else start_step
    _checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        radius_index=radius_index,
        radius=radius,
        step=final_step,
        tracker=tracker,
        completed=True,
    )
    # Persist convergence explicitly without changing the checkpoint schema
    # required for warm-starting older partial stages.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved["converged"] = converged
    _atomic_torch_save(checkpoint_path, saved)
    return {
        "steps": final_step,
        "converged": converged,
        "resumed_complete_checkpoint": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "constant_learning_rate": train_config.lr,
        "gradient_clipping": False,
        "optimizer_reset_at_radius_start": True,
        "tracker": tracker,
    }


@torch.no_grad()
def _evaluate_split(
    model: NormContinuationAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    radius: float,
    batch_size: int,
) -> dict[str, Any]:
    paths = int(indices.numel())
    times = model.config.trading_times
    pnl = np.empty((paths, times), dtype=np.float64)
    band = np.empty_like(pnl)
    action_l1 = np.empty((paths, times), dtype=np.float32)
    raw_l1 = np.empty_like(action_l1)
    clipped = np.empty((paths, times), dtype=bool)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action, batch_raw_l1, batch_clipped = model(data.state[batch], radius)
        batch_pnl = continuation_pnl(action, data.returns[batch], data.costs[batch])
        pnl[start:stop] = batch_pnl.cpu().double().numpy()
        band[start:stop] = tracing_numerical_band(
            action,
            data.returns[batch],
            data.costs[batch],
            data.numerical_band[batch],
        ).cpu().double().numpy()
        action_l1[start:stop] = action.abs().sum(dim=(-1, -2)).cpu().numpy()
        raw_l1[start:stop] = batch_raw_l1.cpu().numpy()
        clipped[start:stop] = batch_clipped.cpu().numpy()

    weight = data.path_weight[indices].cpu().double().numpy()
    component = data.component[indices].cpu().numpy()
    normalized_weight = weight / weight.sum()
    direct = component == 0
    reports: list[dict[str, Any]] = []
    for time_index in range(times):
        value = pnl[:, time_index]
        tolerance = band[:, time_index]
        loss = value < -tolerance
        gain = value > tolerance
        mean, mean_error = _weighted_mean_error(value, normalized_weight)
        loss_probability, loss_error = _weighted_mean_error(
            loss.astype(np.float64), normalized_weight
        )
        gain_probability, gain_error = _weighted_mean_error(
            gain.astype(np.float64), normalized_weight
        )
        if model.config.utility == "entropy":
            log_weight = np.log(normalized_weight)
            log_tilt = log_weight - model.config.risk_aversion * value
            objective = -logsumexp(log_tilt) / model.config.risk_aversion
            log_tilt -= logsumexp(log_tilt)
            marginal_weight = np.exp(log_tilt)
            marginal = marginal_weight / normalized_weight
        else:
            positive = np.maximum(value, 0.0)
            negative = np.minimum(value, 0.0)
            utility_value = np.where(
                value > 0.0,
                -np.expm1(-model.config.risk_aversion * positive)
                / model.config.risk_aversion,
                negative
                - 0.5 * model.config.risk_aversion * np.square(negative),
            )
            marginal = np.where(
                value > 0.0,
                np.exp(-model.config.risk_aversion * positive),
                1.0 - model.config.risk_aversion * negative,
            )
            objective = float(np.dot(normalized_weight, utility_value))
            marginal_weight = normalized_weight * marginal
            marginal_weight /= marginal_weight.sum()
        radial_derivative, radial_error = _weighted_mean_error(
            value, marginal_weight
        )
        radius_derivative, radius_derivative_error = _weighted_mean_error(
            clipped[:, time_index] * value / radius,
            marginal_weight,
        )
        direct_radius_derivative, direct_radius_derivative_error = (
            _weighted_mean_error(
                marginal * clipped[:, time_index] * value / radius,
                normalized_weight,
            )
        )
        reports.append(
            {
                "trading_time_index": time_index,
                "objective": float(objective),
                "P_mean_pnl": mean,
                "P_mean_pnl_standard_error": mean_error,
                "P_loss_probability": loss_probability,
                "P_loss_probability_standard_error": loss_error,
                "P_gain_probability": gain_probability,
                "P_gain_probability_standard_error": gain_error,
                "P_pnl_01_quantile": weighted_quantile(value, weight, 0.01),
                "P_pnl_05_quantile": weighted_quantile(value, weight, 0.05),
                "P_pnl_median": weighted_quantile(value, weight, 0.50),
                "proposal_minimum_pnl_plus_band": float(np.min(value + tolerance)),
                "direct_P_paths": int(direct.sum()),
                "direct_P_loss_count": int(np.sum(loss & direct)),
                "direct_P_gain_count": int(np.sum(gain & direct)),
                "action_l1_P_median": weighted_quantile(
                    action_l1[:, time_index], weight, 0.50
                ),
                "action_l1_P_99_percentile": weighted_quantile(
                    action_l1[:, time_index], weight, 0.99
                ),
                "action_l1_proposal_maximum": float(
                    action_l1[:, time_index].max()
                ),
                "raw_l1_P_99_percentile": weighted_quantile(
                    raw_l1[:, time_index], weight, 0.99
                ),
                "raw_l1_proposal_maximum": float(raw_l1[:, time_index].max()),
                "P_clipped_fraction": float(
                    np.dot(normalized_weight, clipped[:, time_index])
                ),
                "proposal_clipped_fraction": float(clipped[:, time_index].mean()),
                "normalized_radial_objective_derivative": radial_derivative,
                "normalized_radial_objective_derivative_standard_error": radial_error,
                "normalized_frozen_network_radius_objective_derivative": (
                    radius_derivative
                ),
                "normalized_frozen_network_radius_objective_derivative_standard_error": (
                    radius_derivative_error
                ),
                "frozen_network_radius_objective_derivative": (
                    direct_radius_derivative
                ),
                "frozen_network_radius_objective_derivative_standard_error": (
                    direct_radius_derivative_error
                ),
                "marginal_density_effective_sample_size": float(
                    1.0 / np.square(marginal_weight).sum()
                ),
                "positive_marginal_exponential_underflow_fraction": float(
                    np.mean(
                        (value > 0.0)
                        & (
                            model.config.risk_aversion * value
                            > -math.log(np.finfo(np.float32).tiny)
                        )
                    )
                ),
            }
        )
    return {
        "paths": paths,
        "direct_P_paths": int(direct.sum()),
        "physical_weight_mean_before_split_normalization": float(weight.mean()),
        "physical_weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "times": reports,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    radii = tuple(float(radius) for radius in args.radii)
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=args.paths,
    )
    utility: NormContinuationUtility = args.utility
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
    label = (
        f"norm-continuation-{utility}-g{args.risk_aversion:.0e}-"
        f"{case.generator}-{case.path_count}"
    )
    model_config = NormContinuationConfig(
        label=label,
        source_cache_uid=source_cache_uid,
        group=(
            f"gendysanos/norm-continuation/{case.generator}/{utility}/"
            f"gamma-{args.risk_aversion:.0e}/{case.path_count}"
        ),
        user_version=args.user_version,
        state_features=data.state.shape[-1],
        risk_aversion=args.risk_aversion,
        utility=utility,
        radii=radii,
        exterior_penalty=args.exterior_penalty,
        covariance_relative_floor=args.covariance_relative_floor,
    )
    train_config = NormContinuationTrainConfig(
        batch_size=args.batch_size,
        valuation_batch_size=args.valuation_batch_size,
        epochs_per_radius=args.epochs_per_radius,
        minimum_epochs_per_radius=args.minimum_epochs_per_radius,
        value_update=args.value_update,
        checkpoint_update=args.checkpoint_update,
        lr=args.lr,
        compile=not args.no_compile,
        convergence_points=args.convergence_points,
        convergence_relative_tolerance=args.convergence_relative_tolerance,
        seed=args.seed,
    )
    model = NormContinuationAction(
        env,
        config=model_config,
        state_mean=state_mean,
        state_whitener=state_whitener,
    )
    callable_model = torch.compile(model, dynamic=False) if train_config.compile else model
    report_path = Path(args.output_root) / f"{label}.json"
    checkpoint_directory = Path(args.checkpoint_root) / model_config.group
    stages: list[dict[str, Any]] = []
    started = time.monotonic()
    for radius_index, radius in enumerate(radii):
        checkpoint_path = checkpoint_directory / (
            f"{model_config.filename}-radius-{radius_index + 1}.pt"
        )
        with FileLock(str(checkpoint_path) + ".lock"):
            pre_training_gradient = _gradient_diagnostics(
                model,
                data,
                data.reference,
                radius=radius,
                sample_paths=args.gradient_paths,
            )
            training = _train_radius(
                model,
                callable_model,
                data,
                radius_index=radius_index,
                radius=radius,
                train_config=train_config,
                checkpoint_path=checkpoint_path,
            )
            training_set = _evaluate_split(
                model,
                data,
                data.reference,
                radius=radius,
                batch_size=args.valuation_batch_size,
            )
            validation_set = _evaluate_split(
                model,
                data,
                data.validation,
                radius=radius,
                batch_size=args.valuation_batch_size,
            )
            post_training_gradient = {
                "training": _gradient_diagnostics(
                    model,
                    data,
                    data.reference,
                    radius=radius,
                    sample_paths=args.gradient_paths,
                ),
                "validation": _gradient_diagnostics(
                    model,
                    data,
                    data.validation,
                    radius=radius,
                    sample_paths=args.gradient_paths,
                ),
            }
        stage = {
            "radius_index": radius_index + 1,
            "radius": radius,
            "pre_training_gradient": pre_training_gradient,
            "training": training,
            "training_set": training_set,
            "validation_set": validation_set,
            "post_training_gradient": post_training_gradient,
            "checkpoint": str(checkpoint_path),
        }
        stages.append(stage)
        _atomic_json(
            report_path,
            {
                "status": "running",
                "schema_version": "1.0",
                "experiment": "fixed L1 norm continuation",
                "configuration": asdict(case),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "stages": stages,
            },
        )
        train_gradient = post_training_gradient["training"]
        print(
            f"Completed n={radius:g}: clipped="
            f"{max(item['P_clipped_fraction'] for item in training_set['times']):.1%}, "
            f"min raw utility |grad|="
            f"{min(train_gradient['raw_utility_gradient_l2_by_time']):.3g}, "
            f"marginal underflow="
            f"{train_gradient['positive_marginal_exponential_underflow_fraction']:.1%}",
            flush=True,
        )

    final_gradients = [
        stage["post_training_gradient"]["training"] for stage in stages
    ]
    report = {
        "status": "complete",
        "schema_version": "1.0",
        "experiment": "fixed L1 norm continuation",
        "definition": {
            "objective": (
                "maximize E_P[u(a_t(x_t)'dH_t-|a_t(x_t)|'c_t)] separately "
                "for each trading time, jointly over all expiries"
            ),
            "projection": "a_n(v)=v min(1,n/||v||_1)",
            "exterior_penalty": (
                "rho E[(||v||_1-n)_+^2]; excluded from the economic objective"
            ),
            "continuation": (
                "warm-start the same action network at each increasing radius; "
                "reset AdamW moments at each radius"
            ),
            "gradient_diagnostics": (
                "separate action-space utility, projected raw-output utility, "
                "exterior-penalty, total parameter, and radial-scale derivatives"
            ),
        },
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "split": split_diagnostics,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "whitening": whitening,
        "stages": stages,
        "summary": {
            "minimum_final_raw_utility_gradient_l2_by_radius": [
                min(value["raw_utility_gradient_l2_by_time"])
                for value in final_gradients
            ],
            "maximum_final_raw_utility_gradient_l2_by_radius": [
                max(value["raw_utility_gradient_l2_by_time"])
                for value in final_gradients
            ],
            "raw_utility_gradient_exact_zero_fraction_by_radius": [
                value["raw_utility_gradient_exact_zero_fraction"]
                for value in final_gradients
            ],
            "positive_marginal_exponential_underflow_fraction_by_radius": [
                value["positive_marginal_exponential_underflow_fraction"]
                for value in final_gradients
            ],
        },
        "runtime_seconds": time.monotonic() - started,
        "device": str(env.device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    _atomic_json(report_path, report)
    print(f"Completed {label}; report={report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument("--utility", choices=("powerexp", "entropy"), required=True)
    parser.add_argument("--risk-aversion", type=float, default=100.0)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--exterior-penalty", type=float, default=1.0e-3)
    parser.add_argument("--epochs-per-radius", type=int, default=10_000)
    parser.add_argument("--minimum-epochs-per-radius", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=16_384)
    parser.add_argument("--gradient-paths", type=int, default=16_384)
    parser.add_argument("--value-update", type=int, default=250)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--convergence-points", type=int, default=10)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--covariance-relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=94_103)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
