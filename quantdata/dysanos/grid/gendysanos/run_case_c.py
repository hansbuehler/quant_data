"""Run fixed-lambda Case C on selected PCA-IV period-return trading times."""

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
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans

from quantdata.torch import BaseEnvironment

from .arb_tracing import fit_physical_state_whitener, weighted_quantile
from .case_c import (
    CaseCAction,
    CaseCConfig,
    CaseCTrainConfig,
    case_c_numerical_band,
    case_c_objective_and_surrogate,
    case_c_pnl,
    powerexp_one,
)
from .case_c_kernel import KernelCaseCAction, KernelCaseCConfig
from .required_runs import RequiredCase
from .run_expiry_arbitrage_experiments import (
    _DEFAULT_CACHE_ROOT,
    DeviceExpiryCase,
    _load_case,
    _to_device,
)

_DEFAULT_OUTPUT_ROOT = Path("quantdata/dysanos/grid/gendysanos/reports/case-c")
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/case-c/checkpoints"
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


def _selected_data(data: DeviceExpiryCase, config: CaseCConfig) -> DeviceExpiryCase:
    """Retain only t0 and t6 while preserving the original path split."""

    times = torch.as_tensor(
        (config.deterministic_time, config.conditional_time),
        dtype=torch.int64,
        device=data.state.device,
    )
    return DeviceExpiryCase(
        state=data.state.index_select(1, times),
        returns=data.returns.index_select(1, times),
        costs=data.costs.index_select(1, times),
        numerical_band=data.numerical_band.index_select(1, times),
        path_weight=data.path_weight,
        component=data.component,
        reference=data.reference,
        validation=data.validation,
        option_price=None,
        delta=None,
        cash_strike=None,
        spot=None,
    )


@torch.no_grad()
def _fit_kernel_basis(
    data: DeviceExpiryCase,
    *,
    state_mean: torch.Tensor,
    state_whitener: torch.Tensor,
    kernels: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit fixed physical-weighted centers and local scalar bandwidths."""

    state = data.state[data.reference, 1]
    factor = (state - state_mean[None]) @ state_whitener
    sample = factor.double().cpu().numpy()
    weight = data.path_weight[data.reference].double().cpu().numpy()
    estimator = MiniBatchKMeans(
        n_clusters=kernels,
        init="k-means++",
        n_init=3,
        max_iter=200,
        batch_size=4_096,
        random_state=seed,
        reassignment_ratio=0.0,
    )
    estimator.fit(sample, sample_weight=weight)
    distance = estimator.transform(sample)
    assignment = np.argmin(distance, axis=1)
    radius_square = np.empty(kernels, dtype=np.float64)
    cluster_weight = np.empty(kernels, dtype=np.float64)
    for kernel in range(kernels):
        selected = assignment == kernel
        selected_weight = weight[selected]
        cluster_weight[kernel] = selected_weight.sum()
        if not np.any(selected) or not cluster_weight[kernel] > 0.0:
            raise RuntimeError(f"kernel center {kernel} has no physical training mass")
        radius_square[kernel] = np.average(
            np.square(distance[selected, kernel]), weights=selected_weight
        )
    radius = np.sqrt(radius_square)
    positive = radius[radius > 0.0]
    if positive.size != kernels or not np.all(np.isfinite(radius)):
        raise FloatingPointError("kernel radii are not finite-positive")
    radius_floor = 0.25 * float(np.median(positive))
    radius = np.maximum(radius, radius_floor)
    centers = torch.as_tensor(
        estimator.cluster_centers_, dtype=data.state.dtype, device=data.state.device
    )
    inverse_bandwidth = torch.as_tensor(
        np.reciprocal(radius), dtype=data.state.dtype, device=data.state.device
    )
    normalized_mass = cluster_weight / cluster_weight.sum()
    diagnostics: dict[str, Any] = {
        "definition": (
            "physical-weighted MiniBatchKMeans centers in orthogonal t6 state; "
            "bandwidth is weighted within-center RMS distance"
        ),
        "kernels": kernels,
        "radius_floor": radius_floor,
        "minimum_radius": float(radius.min()),
        "median_radius": float(np.median(radius)),
        "maximum_radius": float(radius.max()),
        "minimum_physical_cluster_mass": float(normalized_mass.min()),
        "maximum_physical_cluster_mass": float(normalized_mass.max()),
    }
    return centers, inverse_bandwidth, diagnostics


def _batch_loss(
    model: CaseCAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    notional: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action, gross, boundary, raw = model(data.state[indices, 1])
    pnl = case_c_pnl(action, data.returns[indices], data.costs[indices])
    objective, surrogate, marginal_weight = case_c_objective_and_surrogate(
        pnl,
        data.path_weight[indices],
        notional=notional,
    )
    loss = -surrogate.mean()
    values = {
        "objective": objective,
        "surrogate": surrogate,
        "gross": gross,
        "boundary": boundary,
        "raw_output_maximum_absolute": raw.abs().amax(dim=-1),
        "marginal_weight": marginal_weight,
        "pnl": pnl,
    }
    return loss, values


@torch.no_grad()
def _fixed_actions(
    model: CaseCAction, data: DeviceExpiryCase, indices: torch.Tensor
) -> torch.Tensor:
    return model(data.state[indices, 1])[0].detach().clone()


def _checkpoint(
    path: Path,
    *,
    model: CaseCAction,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    notional_index: int,
    notional: float,
    step: int,
    tracker: list[dict[str, Any]],
    completed: bool,
    converged: bool,
) -> None:
    _atomic_torch_save(
        path,
        {
            "model_code_version": type(model.config).version.unique_id48,
            "user_version": model.config.user_version,
            "notional_index": notional_index,
            "notional": notional,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tracker": tracker,
            "completed": completed,
            "converged": converged,
            "batch_generator_state": generator.get_state(),
        },
    )


def _train_notional(
    model: CaseCAction,
    callable_model: CaseCAction,
    data: DeviceExpiryCase,
    *,
    notional_index: int,
    notional: float,
    train_config: CaseCTrainConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Optimize only on the training paths using the normalized exact gradient."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=0.0)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(train_config.seed + notional_index)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        compatible = (
            saved.get("model_code_version")
            == type(model.config).version.unique_id48
            and saved.get("user_version") == model.config.user_version
            and saved.get("notional_index") == notional_index
            and saved.get("notional") == notional
        )
        if compatible:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            tracker = list(saved.get("tracker", []))
            generator.set_state(saved["batch_generator_state"].cpu())
            if bool(saved.get("completed")):
                return {
                    "steps": start_step,
                    "converged": bool(saved.get("converged")),
                    "resumed_complete_checkpoint": True,
                    "elapsed_seconds_this_invocation": 0.0,
                    "tracker": tracker,
                }

    fixed_generator = torch.Generator(device=model.device)
    fixed_generator.manual_seed(train_config.seed + 10_000 + notional_index)
    draw = torch.randperm(
        data.reference.numel(), generator=fixed_generator, device=model.device
    )[: min(train_config.valuation_batch_size, data.reference.numel())]
    fixed = data.reference[draw]
    previous_action = _fixed_actions(model, data, fixed)
    stable_points = 0
    converged = False
    compiled_notional = torch.as_tensor(
        notional, dtype=model.fdtype, device=model.device
    )
    started = time.monotonic()
    for step in range(start_step + 1, train_config.epochs_per_notional + 1):
        draw = torch.randint(
            data.reference.numel(),
            (train_config.batch_size,),
            generator=generator,
            device=model.device,
        )
        indices = data.reference[draw]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(
            callable_model, data, indices, notional=compiled_notional
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Case C analytic gradient became NaN or infinity")
        optimizer.step()
        model.project_parameters_()

        if step % train_config.value_update == 0 or step == train_config.epochs_per_notional:
            with torch.no_grad():
                fixed_loss, values = _batch_loss(
                    model, data, fixed, notional=notional
                )
                action = _fixed_actions(model, data, fixed)
                action_change = float((action - previous_action).abs().amax().cpu())
                previous_action = action
            stable_points = (
                stable_points + 1
                if action_change <= train_config.action_change_tolerance
                else 0
            )
            converged = bool(
                step >= train_config.minimum_epochs_per_notional
                and stable_points >= train_config.convergence_points
            )
            item = {
                "step": step,
                "loss": float(fixed_loss.cpu()),
                "objective_by_time": values["objective"].cpu().tolist(),
                "gross_mean_by_time": values["gross"].mean(dim=0).cpu().tolist(),
                "gross_maximum_by_time": values["gross"].amax(dim=0).cpu().tolist(),
                "raw_output_maximum_absolute_by_time": (
                    values["raw_output_maximum_absolute"].amax(dim=0).cpu().tolist()
                ),
                "boundary_fraction_by_time": values["boundary"].float().mean(dim=0).cpu().tolist(),
                "maximum_fixed_action_change": action_change,
                "stable_points": stable_points,
                "elapsed_seconds": time.monotonic() - started,
            }
            tracker.append(item)
            print(
                f"[n={notional:g}] step {step:,}: "
                f"objective={item['objective_by_time']}, "
                f"gross={item['gross_mean_by_time']}, "
                f"action change={action_change:.3g}, converged={converged}",
                flush=True,
            )
            if step % train_config.checkpoint_update == 0:
                _checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    notional_index=notional_index,
                    notional=notional,
                    step=step,
                    tracker=tracker,
                    completed=False,
                    converged=converged,
                )
            if converged:
                break

    final_step = int(tracker[-1]["step"]) if tracker else start_step
    _checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        notional_index=notional_index,
        notional=notional,
        step=final_step,
        tracker=tracker,
        completed=True,
        converged=converged,
    )
    return {
        "steps": final_step,
        "converged": converged,
        "resumed_complete_checkpoint": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "constant_learning_rate": train_config.lr,
        "gradient_clipping": False,
        "optimizer_reset_at_notional_start": True,
        "tracker": tracker,
    }


def _marginal_log_weight(
    pnl: np.ndarray, path_weight: np.ndarray, notional: float
) -> np.ndarray:
    scaled = notional * pnl
    log_marginal = np.empty_like(scaled)
    positive = scaled > 0.0
    log_marginal[positive] = -scaled[positive]
    log_marginal[~positive] = np.log1p(-scaled[~positive])
    result = np.log(path_weight) + log_marginal
    result -= logsumexp(result)
    return result


@torch.no_grad()
def _evaluate_split(
    model: CaseCAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    notional: float,
    batch_size: int,
) -> dict[str, Any]:
    paths = int(indices.numel())
    pnl = np.empty((paths, 2), dtype=np.float64)
    band = np.empty_like(pnl)
    gross = np.empty((paths, 2), dtype=np.float32)
    raw_output_maximum = np.empty_like(gross)
    boundary = np.empty((paths, 2), dtype=bool)
    overlap = np.empty((paths, 2), dtype=np.float32)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action, batch_gross, batch_boundary, raw = model(data.state[batch, 1])
        pnl[start:stop] = case_c_pnl(
            action, data.returns[batch], data.costs[batch]
        ).cpu().double().numpy()
        band[start:stop] = case_c_numerical_band(
            action,
            data.returns[batch],
            data.costs[batch],
            data.numerical_band[batch],
        ).cpu().double().numpy()
        gross[start:stop] = batch_gross.cpu().numpy()
        raw_output_maximum[start:stop] = raw.abs().amax(dim=-1).cpu().numpy()
        boundary[start:stop] = batch_boundary.cpu().numpy()
        overlap[start:stop] = torch.minimum(
            action[..., 0, :, :], action[..., 1, :, :]
        ).sum(dim=(-1, -2)).cpu().numpy()

    weight = data.path_weight[indices].cpu().double().numpy()
    component = data.component[indices].cpu().numpy()
    normalized_weight = weight / weight.sum()
    direct = component == 0
    reports: list[dict[str, Any]] = []
    names = ("t0_deterministic", "t6_conditional")
    for time_index, name in enumerate(names):
        value = pnl[:, time_index]
        tolerance = band[:, time_index]
        loss = value < -tolerance
        gain = value > tolerance
        scaled = notional * value
        positive = np.maximum(scaled, 0.0)
        negative = np.minimum(scaled, 0.0)
        utility = np.where(
            scaled > 0.0,
            -np.expm1(-positive),
            negative - 0.5 * np.square(negative),
        )
        objective, objective_error = _weighted_mean_error(
            utility, normalized_weight
        )
        mean, mean_error = _weighted_mean_error(value, normalized_weight)
        loss_probability, loss_error = _weighted_mean_error(
            loss.astype(np.float64), normalized_weight
        )
        gain_probability, gain_error = _weighted_mean_error(
            gain.astype(np.float64), normalized_weight
        )
        marginal_log_weight = _marginal_log_weight(value, weight, notional)
        marginal_weight = np.exp(marginal_log_weight)
        scale_derivative, scale_derivative_error = _weighted_mean_error(
            value, marginal_weight
        )
        reports.append(
            {
                "name": name,
                "objective": objective,
                "objective_standard_error": objective_error,
                "P_mean_unit_pnl": mean,
                "P_mean_unit_pnl_standard_error": mean_error,
                "P_loss_probability": loss_probability,
                "P_loss_probability_standard_error": loss_error,
                "P_gain_probability": gain_probability,
                "P_gain_probability_standard_error": gain_error,
                "P_unit_pnl_01_quantile": weighted_quantile(value, weight, 0.01),
                "P_unit_pnl_05_quantile": weighted_quantile(value, weight, 0.05),
                "P_unit_pnl_median": weighted_quantile(value, weight, 0.50),
                "proposal_minimum_unit_pnl_plus_band": float(
                    np.min(value + tolerance)
                ),
                "proposal_numerically_nonnegative": bool(np.all(~loss)),
                "direct_P_paths": int(direct.sum()),
                "direct_P_loss_count": int(np.sum(loss & direct)),
                "direct_P_gain_count": int(np.sum(gain & direct)),
                "base_action_gross_P_median": weighted_quantile(
                    gross[:, time_index], weight, 0.50
                ),
                "base_action_gross_P_99_percentile": weighted_quantile(
                    gross[:, time_index], weight, 0.99
                ),
                "base_action_gross_proposal_maximum": float(
                    gross[:, time_index].max()
                ),
                "economic_gross_P_median": notional
                * weighted_quantile(gross[:, time_index], weight, 0.50),
                "raw_output_maximum_absolute_proposal": float(
                    raw_output_maximum[:, time_index].max()
                ),
                "P_boundary_fraction": float(
                    np.dot(normalized_weight, boundary[:, time_index])
                ),
                "P_long_short_overlap": float(
                    np.dot(normalized_weight, overlap[:, time_index])
                ),
                "normalized_notional_derivative": scale_derivative,
                "normalized_notional_derivative_standard_error": (
                    scale_derivative_error
                ),
                "marginal_density_effective_sample_size": float(
                    1.0 / np.square(marginal_weight).sum()
                ),
            }
        )
    return {
        "paths": paths,
        "direct_P_paths": int(direct.sum()),
        "physical_weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "times": reports,
    }


def _gradient_diagnostics(
    model: CaseCAction,
    data: DeviceExpiryCase,
    indices: torch.Tensor,
    *,
    notional: float,
    sample_paths: int,
) -> dict[str, Any]:
    sample = indices[: min(sample_paths, indices.numel())]
    model.zero_grad(set_to_none=True)
    loss, values = _batch_loss(model, data, sample, notional=notional)
    loss.backward()
    parameter_norm = float(
        torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.detach().reshape(-1)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
        ).cpu()
    )
    gradient_zero_fractions = [
        float((parameter.grad == 0.0).float().mean().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    marginal_weight = values["marginal_weight"].detach()
    derivative = (marginal_weight * values["pnl"].detach()).sum(dim=0)
    model.zero_grad(set_to_none=True)
    return {
        "sample_paths": int(sample.numel()),
        "normalized_parameter_gradient_l2": parameter_norm,
        "normalized_notional_derivative_by_time": derivative.cpu().tolist(),
        "marginal_density_effective_sample_size_by_time": (
            1.0 / marginal_weight.square().sum(dim=0)
        ).cpu().tolist(),
        "gradient_exact_zero_fraction": (
            float(np.mean(gradient_zero_fractions))
            if gradient_zero_fractions
            else 1.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    notionals = tuple(float(value) for value in args.notionals)
    case = RequiredCase(
        generator="pca-iv",
        trading_frequency="weekly",
        return_horizon="incremental",
        hedge="raw",
        cost_rate=args.cost_bps * 1.0e-4,
        path_count=100_000,
    )
    loaded = _load_case(case, cache_root=Path(args.cache_root), need_primitives=False)
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=args.precision, device=args.device, debug=args.debug)
    full_data, split_diagnostics = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()

    label = f"case-c-pca-iv-weekly-period-raw-{args.cost_bps:g}bps-100000"
    if args.architecture == "kernel":
        label = f"{label}-kernel-{args.kernels}"
        config = KernelCaseCConfig(
            label=label,
            source_cache_uid=source_cache_uid,
            state_features=full_data.state.shape[-1],
            kernels=args.kernels,
            notionals=notionals,
            group=f"gendysanos/case-c-kernel/{args.cost_bps:g}bps",
            user_version=args.user_version,
        )
    else:
        config = CaseCConfig(
            label=label,
            source_cache_uid=source_cache_uid,
            state_features=full_data.state.shape[-1],
            notionals=notionals,
            group=f"gendysanos/case-c/{args.cost_bps:g}bps",
            user_version=args.user_version,
        )
    data = _selected_data(full_data, config)
    del full_data
    state_mean, state_whitener, whitening = fit_physical_state_whitener(
        data.state,
        data.path_weight,
        data.reference,
        relative_eigenvalue_floor=args.covariance_relative_floor,
    )
    kernel_basis = None
    if args.architecture == "kernel":
        centers, inverse_bandwidth, kernel_basis = _fit_kernel_basis(
            data,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            kernels=args.kernels,
            seed=args.seed,
        )
        model = KernelCaseCAction(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            centers=centers,
            inverse_bandwidth=inverse_bandwidth,
        )
    else:
        model = CaseCAction(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
        )
    train_config = CaseCTrainConfig(
        batch_size=args.batch_size,
        valuation_batch_size=args.valuation_batch_size,
        epochs_per_notional=args.epochs_per_notional,
        minimum_epochs_per_notional=args.minimum_epochs_per_notional,
        value_update=args.value_update,
        checkpoint_update=args.checkpoint_update,
        convergence_points=args.convergence_points,
        action_change_tolerance=args.action_change_tolerance,
        lr=args.lr,
        compile=not args.no_compile,
        seed=args.seed,
    )
    callable_model = torch.compile(model, dynamic=False) if train_config.compile else model
    report_path = Path(args.output_root) / f"{config.label}.json"
    checkpoint_directory = Path(args.checkpoint_root) / config.group
    stages: list[dict[str, Any]] = []
    started = time.monotonic()
    for notional_index, notional in enumerate(notionals):
        checkpoint_path = checkpoint_directory / (
            f"{config.filename}-notional-{notional_index + 1}.pt"
        )
        pre_gradient = _gradient_diagnostics(
            model,
            data,
            data.reference,
            notional=notional,
            sample_paths=args.gradient_paths,
        )
        training = _train_notional(
            model,
            callable_model,
            data,
            notional_index=notional_index,
            notional=notional,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
        )
        training_set = _evaluate_split(
            model,
            data,
            data.reference,
            notional=notional,
            batch_size=args.valuation_batch_size,
        )
        validation_set = _evaluate_split(
            model,
            data,
            data.validation,
            notional=notional,
            batch_size=args.valuation_batch_size,
        )
        post_gradient = {
            "training": _gradient_diagnostics(
                model,
                data,
                data.reference,
                notional=notional,
                sample_paths=args.gradient_paths,
            ),
            "validation": _gradient_diagnostics(
                model,
                data,
                data.validation,
                notional=notional,
                sample_paths=args.gradient_paths,
            ),
        }
        stage = {
            "notional": notional,
            "pre_training_gradient": pre_gradient,
            "training": training,
            "training_set": training_set,
            "validation_set": validation_set,
            "post_training_gradient": post_gradient,
            "checkpoint": str(checkpoint_path),
        }
        stages.append(stage)
        _atomic_json(
            report_path,
            {
                "status": "running",
                "schema_version": "1.0",
                "configuration": asdict(case),
                "model_config": asdict(config),
                "train_config": asdict(train_config),
                "stages": stages,
            },
        )
        print(
            f"Completed n={notional:g}; "
            f"training derivative={[x['normalized_notional_derivative'] for x in training_set['times']]}; "
            f"validation derivative={[x['normalized_notional_derivative'] for x in validation_set['times']]}",
            flush=True,
        )

    report = {
        "status": "complete",
        "schema_version": "1.0",
        "experiment": "Case C fixed-lambda notional continuation",
        "definition": {
            "objective": "sup over unit-gross separate long/short actions of E_P[u_1(n X)]",
            "pnl": "X=(a_plus-a_minus)'R-(a_plus+a_minus)'c",
            "gradient": "sum_s D_s grad(X_s), D proportional to (p/q) u_1'(n X_s)",
            "action_constraint": "a_plus,a_minus nonnegative and sum(a_plus+a_minus)<=1",
            "validation_policy": "held-out paths never enter fitting, convergence, or stopping",
            "architecture": args.architecture,
        },
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "split": split_diagnostics,
        "model_config": asdict(config),
        "train_config": asdict(train_config),
        "whitening": whitening,
        "kernel_basis": kernel_basis,
        "stages": stages,
        "runtime_seconds": time.monotonic() - started,
        "device": str(env.device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(env.device))
            if torch.device(env.device).type == "cuda"
            else 0
        ),
    }
    _atomic_json(report_path, report)
    print(f"Completed Case C report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-bps", type=float, choices=(1.0, 10.0), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--architecture", choices=("feedforward", "kernel"), default="feedforward"
    )
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument(
        "--notionals",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    )
    parser.add_argument("--epochs-per-notional", type=int, default=20_000)
    parser.add_argument("--minimum-epochs-per-notional", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=16_384)
    parser.add_argument("--gradient-paths", type=int, default=16_384)
    parser.add_argument("--value-update", type=int, default=250)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--convergence-points", type=int, default=8)
    parser.add_argument("--action-change-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--covariance-relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=182_071)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
