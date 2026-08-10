"""Fit independent utility policies for every trading-time/expiry cell."""

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

from .cell_measure_change import (
    CellMeasureChangeConfig,
    CellMeasureChangeTrainConfig,
    CellUtility,
    IndependentCellAction,
    cell_pnl,
    cell_utility_and_marginal,
)
from .required_runs import RequiredCase
from .run_expiry_arbitrage_experiments import (
    _DEFAULT_CACHE_ROOT,
    _load_case,
    _to_device,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/cell-measure-change"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/cell-measure-change/checkpoints"
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


def _normalization(data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    reference = data.reference
    state = data.state[reference]
    state_mean = state.mean(dim=0)
    state_scale = state.std(dim=0, correction=0).clamp_min(1.0e-6)
    weight = data.path_weight[reference]
    weight_mean = float(weight.double().mean().cpu())
    normalized_weight = weight / weight_mean
    return_rms = torch.sqrt(
        (
            normalized_weight[:, None, None, None]
            * data.returns[reference].square()
        ).mean(dim=0)
    ).clamp_min(1.0e-5)
    for name, value in (
        ("state mean", state_mean),
        ("state scale", state_scale),
        ("return RMS", return_rms),
    ):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"{name} is non-finite")
    return state_mean, state_scale, return_rms, weight_mean


def _loss(
    model: IndependentCellAction,
    data,
    indices: torch.Tensor,
    *,
    utility: CellUtility,
    risk_aversion: float,
    weight_mean: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    action = model(data.state[indices])
    pnl = cell_pnl(action, data.returns[indices], data.costs[indices])
    value, _ = cell_utility_and_marginal(
        pnl,
        utility=utility,
        risk_aversion=risk_aversion,
        weight=data.path_weight[indices] / weight_mean,
    )
    loss = -value.mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"{utility} objective became NaN or infinity")
    return loss, value


def _checkpoint(
    path: Path,
    *,
    model: IndependentCellAction,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    tracker: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = {
        "model_code_version": CellMeasureChangeConfig.version.unique_id48,
        "user_version": model.config.user_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tracker": tracker,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(model.device),
        "batch_generator_state": generator.get_state(),
    }
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _train(
    data,
    *,
    model_config: CellMeasureChangeConfig,
    train_config: CellMeasureChangeTrainConfig,
    checkpoint_path: Path,
    env: BaseEnvironment,
) -> tuple[IndependentCellAction, dict[str, Any]]:
    state_mean, state_scale, return_rms, weight_mean = _normalization(data)
    model = IndependentCellAction(
        env,
        config=model_config,
        state_mean=state_mean,
        state_scale=state_scale,
        return_rms=return_rms,
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
            saved.get("model_code_version")
            != CellMeasureChangeConfig.version.unique_id48
            or saved.get("user_version") != model_config.user_version
        ):
            print("Checkpoint version changed; cold-starting cell policies.", flush=True)
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            tracker = list(saved.get("tracker", []))
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), env.device)
            generator.set_state(saved["batch_generator_state"].cpu())
            print(f"Warm-started cell policies from step {start_step:,}.", flush=True)

    callable_model = torch.compile(model, dynamic=False) if train_config.compile else model
    torch.cuda.reset_peak_memory_stats(env.device)
    started = time.monotonic()
    converged = False
    fixed = data.reference[: min(train_config.valuation_batch_size, data.reference.numel())]
    for step in range(start_step + 1, train_config.epochs + 1):
        draw = torch.randint(
            data.reference.numel(),
            (train_config.batch_size,),
            generator=generator,
            device=env.device,
        )
        indices = data.reference[draw]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss(
            callable_model,
            data,
            indices,
            utility=model_config.utility,
            risk_aversion=model_config.risk_aversion,
            weight_mean=weight_mean,
        )
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("cell-policy gradient became NaN or infinity")
        optimizer.step()
        if step % train_config.value_update == 0 or step == train_config.epochs:
            with torch.no_grad():
                fixed_loss, fixed_value = _loss(
                    model,
                    data,
                    fixed,
                    utility=model_config.utility,
                    risk_aversion=model_config.risk_aversion,
                    weight_mean=weight_mean,
                )
            item = {
                "step": step,
                "elapsed_seconds": time.monotonic() - started,
                "training_objective_mean": float(-fixed_loss.cpu()),
                "training_objective_minimum_cell": float(fixed_value.min().cpu()),
                "training_objective_maximum_cell": float(fixed_value.max().cpu()),
            }
            tracker.append(item)
            recent = tracker[-train_config.convergence_points :]
            if (
                step >= train_config.minimum_epochs
                and len(recent) == train_config.convergence_points
            ):
                values = np.asarray(
                    [point["training_objective_mean"] for point in recent]
                )
                scale = max(1.0e-8, float(np.max(np.abs(values))))
                converged = bool(
                    np.ptp(values)
                    <= train_config.convergence_relative_tolerance * scale
                )
            print(
                f"[{model_config.utility}/gamma={model_config.risk_aversion:g}] "
                f"step {step:,}: objective={item['training_objective_mean']:.8g}, "
                f"cells=[{item['training_objective_minimum_cell']:.4g},"
                f"{item['training_objective_maximum_cell']:.4g}], "
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
    elapsed_across_sessions = 0.0
    previous_elapsed = 0.0
    for point in tracker:
        point_elapsed = float(point["elapsed_seconds"])
        if point_elapsed < previous_elapsed:
            elapsed_across_sessions += previous_elapsed
        previous_elapsed = point_elapsed
    elapsed_across_sessions += previous_elapsed
    return model, {
        "steps": final_step,
        "converged": converged,
        "optimization_elapsed_seconds_across_sessions": elapsed_across_sessions,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "training_weight_mean": weight_mean,
        "state_mean": state_mean.cpu().tolist(),
        "state_scale": state_scale.cpu().tolist(),
        "return_rms": return_rms.cpu().tolist(),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(env.device)),
        "tracker": tracker,
    }


def _marginal_numpy(
    pnl: np.ndarray,
    *,
    utility: CellUtility,
    risk_aversion: float,
    normalized_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if utility == "mean-variance":
        mean = np.mean(normalized_weight[:, None, None] * pnl, axis=0)
        centered = pnl - mean[None, :, :]
        marginal = 1.0 - risk_aversion * centered
        variance = np.mean(
            normalized_weight[:, None, None] * np.square(centered), axis=0
        )
        objective = mean - 0.5 * risk_aversion * variance
    elif utility == "exponential":
        # Match the float32 forward safeguard used by ``torch_utility``.
        z = np.minimum(-risk_aversion * pnl, 30.0)
        marginal = np.exp(z)
        objective = np.mean(
            normalized_weight[:, None, None]
            * (-np.expm1(z) / risk_aversion),
            axis=0,
        )
    else:
        positive = np.maximum(pnl, 0.0)
        negative = np.maximum(-pnl, 0.0)
        marginal = np.where(
            pnl > 0.0,
            np.exp(-risk_aversion * positive),
            1.0 + risk_aversion * negative,
        )
        utility_value = np.where(
            pnl > 0.0,
            -np.expm1(-risk_aversion * positive) / risk_aversion,
            pnl - 0.5 * risk_aversion * np.square(negative),
        )
        objective = np.mean(
            normalized_weight[:, None, None] * utility_value,
            axis=0,
        )
    if not np.all(np.isfinite(marginal)) or not np.all(np.isfinite(objective)):
        raise FloatingPointError(f"non-finite {utility} valuation")
    return marginal, objective


@torch.no_grad()
def _evaluate(
    model: IndependentCellAction,
    data,
    indices: torch.Tensor,
    *,
    utility: CellUtility,
    risk_aversion: float,
    batch_size: int,
) -> dict[str, Any]:
    paths = int(indices.numel())
    T = model.config.trading_times
    E = model.config.expiries
    J = model.config.instruments_per_expiry
    pnl = np.empty((paths, T, E), dtype=np.float64)
    action_l1 = np.empty((paths, T, E), dtype=np.float32)
    action_maximum = 0.0
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(data.state[batch])
        pnl[start:stop] = cell_pnl(
            action,
            data.returns[batch],
            data.costs[batch],
        ).cpu().double().numpy()
        action_l1[start:stop] = action.abs().sum(dim=-1).cpu().numpy()
        action_maximum = max(action_maximum, float(action.abs().max().cpu()))

    index_numpy = indices.cpu().numpy()
    weight = data.path_weight[indices].cpu().double().numpy()
    normalized_weight = weight / float(weight.mean())
    component = data.component[indices].cpu().numpy()
    direct = component == 0
    marginal, objective = _marginal_numpy(
        pnl,
        utility=utility,
        risk_aversion=risk_aversion,
        normalized_weight=normalized_weight,
    )
    density_normalizer = np.mean(
        normalized_weight[:, None, None] * marginal,
        axis=0,
    )
    if not np.all(np.isfinite(density_normalizer)) or np.any(
        np.abs(density_normalizer) <= np.finfo(np.float64).eps
    ):
        raise FloatingPointError("density normalization is zero or non-finite")
    density = marginal / density_normalizer[None, :, :]
    weighted_density = normalized_weight[:, None, None] * density
    density_mean = weighted_density.mean(axis=0)
    density_weight_square_sum = np.square(weighted_density).sum(axis=0)
    density_absolute_weight_sum = np.abs(weighted_density).sum(axis=0)
    density_effective_sample_size = np.divide(
        np.square(weighted_density.sum(axis=0)),
        density_weight_square_sum,
        out=np.zeros((T, E), dtype=np.float64),
        where=density_weight_square_sum > 0.0,
    )
    density_maximum_absolute_weight_share = np.divide(
        np.max(np.abs(weighted_density), axis=0),
        density_absolute_weight_sum,
        out=np.ones((T, E), dtype=np.float64),
        where=density_absolute_weight_sum > 0.0,
    )
    negative_density_fraction_by_cell = np.mean(density < 0.0, axis=0)

    moment_sum = np.zeros((T, E, J), dtype=np.float64)
    foc_sum = np.zeros((T, E, J), dtype=np.float64)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(data.state[batch])
        returns = data.returns[batch]
        costs = data.costs[batch]
        wd = torch.as_tensor(
            weighted_density[start:stop],
            dtype=returns.dtype,
            device=returns.device,
        )
        moment_sum += (
            wd[..., None] * returns
        ).sum(dim=0).cpu().double().numpy()
        foc_return = returns - torch.sign(action) * costs
        foc_sum += (wd[..., None] * foc_return).sum(dim=0).cpu().double().numpy()
    moment = moment_sum / paths
    foc = foc_sum / paths

    moment_influence_square = np.zeros_like(moment)
    foc_influence_square = np.zeros_like(foc)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(data.state[batch])
        returns = data.returns[batch]
        costs = data.costs[batch]
        wd = torch.as_tensor(
            weighted_density[start:stop],
            dtype=returns.dtype,
            device=returns.device,
        )
        moment_centered = returns - torch.as_tensor(
            moment,
            dtype=returns.dtype,
            device=returns.device,
        )
        foc_return = returns - torch.sign(action) * costs
        foc_centered = foc_return - torch.as_tensor(
            foc,
            dtype=returns.dtype,
            device=returns.device,
        )
        moment_influence_square += (
            wd[..., None] * moment_centered
        ).square().sum(dim=0).cpu().double().numpy()
        foc_influence_square += (
            wd[..., None] * foc_centered
        ).square().sum(dim=0).cpu().double().numpy()
    moment_se = np.sqrt(moment_influence_square / (paths * max(paths - 1, 1)))
    foc_se = np.sqrt(foc_influence_square / (paths * max(paths - 1, 1)))
    moment_z = np.divide(
        moment,
        moment_se,
        out=np.full_like(moment, np.nan),
        where=moment_se > 0.0,
    )
    foc_z = np.divide(
        foc,
        foc_se,
        out=np.full_like(foc, np.nan),
        where=foc_se > 0.0,
    )

    weighted_pnl = normalized_weight[:, None, None] * pnl
    pnl_mean = weighted_pnl.mean(axis=0)
    pnl_se = weighted_pnl.std(axis=0, ddof=1) / math.sqrt(paths)
    direct_mean = pnl[direct].mean(axis=0) if np.any(direct) else None
    direct_se = (
        pnl[direct].std(axis=0, ddof=1) / math.sqrt(int(direct.sum()))
        if np.sum(direct) > 1
        else None
    )
    cells: list[dict[str, Any]] = []
    for trading_time in range(T):
        for expiry in range(E):
            cells.append(
                {
                    "trading_time_index": trading_time,
                    "expiry_index": expiry,
                    "objective": float(objective[trading_time, expiry]),
                    "P_mean_pnl": float(pnl_mean[trading_time, expiry]),
                    "P_pnl_standard_error": float(pnl_se[trading_time, expiry]),
                    "direct_P_mean_pnl": (
                        float(direct_mean[trading_time, expiry])
                        if direct_mean is not None
                        else None
                    ),
                    "direct_P_pnl_standard_error": (
                        float(direct_se[trading_time, expiry])
                        if direct_se is not None
                        else None
                    ),
                    "density_normalizer": float(
                        density_normalizer[trading_time, expiry]
                    ),
                    "E_P_density": float(density_mean[trading_time, expiry]),
                    "negative_density_fraction": float(
                        negative_density_fraction_by_cell[trading_time, expiry]
                    ),
                    "density_effective_sample_size": float(
                        density_effective_sample_size[trading_time, expiry]
                    ),
                    "density_maximum_absolute_weight_share": float(
                        density_maximum_absolute_weight_share[trading_time, expiry]
                    ),
                    "maximum_absolute_return_moment_z": float(
                        np.nanmax(np.abs(moment_z[trading_time, expiry]))
                    ),
                    "maximum_absolute_cost_adjusted_foc_z": float(
                        np.nanmax(np.abs(foc_z[trading_time, expiry]))
                    ),
                    "action_l1_median": float(
                        np.median(action_l1[:, trading_time, expiry])
                    ),
                    "action_l1_99_percentile": float(
                        np.quantile(action_l1[:, trading_time, expiry], 0.99)
                    ),
                }
            )
    reliable_density = (
        (density_effective_sample_size >= 20.0)
        & (negative_density_fraction_by_cell == 0.0)
    )
    reliable_moment_z = np.where(reliable_density[..., None], moment_z, np.nan)
    reliable_foc_z = np.where(reliable_density[..., None], foc_z, np.nan)
    return {
        "paths": paths,
        "direct_P_paths": int(direct.sum()),
        "importance_weight_mean_before_set_normalization": float(weight.mean()),
        "importance_weight_effective_sample_size": float(
            np.square(weight.sum()) / np.square(weight).sum()
        ),
        "density_normalized_on_this_set": True,
        "maximum_E_P_density_error": float(np.max(np.abs(density_mean - 1.0))),
        "negative_density_fraction": float(np.mean(density < 0.0)),
        "minimum_density_effective_sample_size": float(
            np.min(density_effective_sample_size)
        ),
        "maximum_density_absolute_weight_share": float(
            np.max(density_maximum_absolute_weight_share)
        ),
        "reliable_density_cell_count": int(np.sum(reliable_density)),
        "maximum_absolute_return_moment_z": float(np.nanmax(np.abs(moment_z))),
        "maximum_absolute_return_moment_z_reliable_density_cells": (
            float(np.nanmax(np.abs(reliable_moment_z)))
            if np.any(np.isfinite(reliable_moment_z))
            else None
        ),
        "rms_return_moment_z": float(np.sqrt(np.nanmean(np.square(moment_z)))),
        "maximum_absolute_cost_adjusted_foc_z": float(np.nanmax(np.abs(foc_z))),
        "maximum_absolute_cost_adjusted_foc_z_reliable_density_cells": (
            float(np.nanmax(np.abs(reliable_foc_z)))
            if np.any(np.isfinite(reliable_foc_z))
            else None
        ),
        "rms_cost_adjusted_foc_z": float(np.sqrt(np.nanmean(np.square(foc_z)))),
        "maximum_absolute_action": action_maximum,
        "action_l1_median": float(np.median(action_l1)),
        "action_l1_99_percentile": float(np.quantile(action_l1, 0.99)),
        "cells": cells,
        "path_indices_checksum": int(np.sum(index_numpy, dtype=np.int64)),
    }


def _gradient_diagnostics(
    model: IndependentCellAction,
    data,
    indices: torch.Tensor,
    *,
    utility: CellUtility,
    risk_aversion: float,
) -> dict[str, Any]:
    sample = indices[: min(16_384, indices.numel())]
    weight_mean = float(data.path_weight[indices].double().mean().cpu())
    model.zero_grad(set_to_none=True)
    loss, value = _loss(
        model,
        data,
        sample,
        utility=utility,
        risk_aversion=risk_aversion,
        weight_mean=weight_mean,
    )
    loss.backward()
    cell_norms: list[float] = []
    cell_count = model.config.trading_times * model.config.expiries
    for cell in range(cell_count):
        gradients = [
            parameter.grad[cell].detach().reshape(-1)
            for parameter in (*model.network.weights, *model.network.biases)
            if parameter.grad is not None
        ]
        flat = torch.cat(gradients)
        # The reported norm is for that cell's objective, undoing the 1/70
        # averaging used to train all disjoint parameter blocks together.
        cell_norms.append(
            float(cell_count * torch.linalg.vector_norm(flat).cpu())
        )
    model.zero_grad(set_to_none=True)
    return {
        "sample_paths": int(sample.numel()),
        "loss": float(loss.detach().cpu()),
        "objective_minimum_cell": float(value.min().detach().cpu()),
        "objective_maximum_cell": float(value.max().detach().cpu()),
        "cell_parameter_gradient_l2_minimum": float(np.min(cell_norms)),
        "cell_parameter_gradient_l2_median": float(np.median(cell_norms)),
        "cell_parameter_gradient_l2_maximum": float(np.max(cell_norms)),
        "cell_parameter_gradient_l2": np.asarray(cell_norms).reshape(10, 7).tolist(),
    }


def _risk_aversion_sweep(
    *,
    fitted_risk_aversion: float,
    requested: list[float],
    training_set: dict[str, Any],
    validation_set: dict[str, Any],
) -> dict[str, Any]:
    """Report the exact risk-aversion scaling of the homogeneous policy."""

    values: list[dict[str, Any]] = []
    for risk_aversion in requested:
        verify_inp(
            math.isfinite(risk_aversion) and risk_aversion > 0.0,
            lambda: f"invalid risk-aversion sweep value {risk_aversion}",
        )
        scale = fitted_risk_aversion / risk_aversion

        def scaled(summary: dict[str, Any]) -> dict[str, Any]:
            objectives = np.asarray(
                [cell["objective"] for cell in summary["cells"]],
                dtype=np.float64,
            )
            pnl_means = np.asarray(
                [cell["P_mean_pnl"] for cell in summary["cells"]],
                dtype=np.float64,
            )
            pnl_errors = np.asarray(
                [cell["P_pnl_standard_error"] for cell in summary["cells"]],
                dtype=np.float64,
            )
            return {
                "objective_mean": float(scale * objectives.mean()),
                "P_mean_pnl_mean_over_cells": float(scale * pnl_means.mean()),
                "P_pnl_standard_error_mean_over_cells": float(
                    scale * pnl_errors.mean()
                ),
                "maximum_absolute_action": float(
                    scale * summary["maximum_absolute_action"]
                ),
                "action_l1_median": float(scale * summary["action_l1_median"]),
                "action_l1_99_percentile": float(
                    scale * summary["action_l1_99_percentile"]
                ),
                "maximum_absolute_return_moment_z": summary[
                    "maximum_absolute_return_moment_z"
                ],
                "maximum_absolute_cost_adjusted_foc_z": summary[
                    "maximum_absolute_cost_adjusted_foc_z"
                ],
                "maximum_absolute_cost_adjusted_foc_z_reliable_density_cells": summary[
                    "maximum_absolute_cost_adjusted_foc_z_reliable_density_cells"
                ],
                "negative_density_fraction": summary["negative_density_fraction"],
                "minimum_density_effective_sample_size": summary[
                    "minimum_density_effective_sample_size"
                ],
            }

        values.append(
            {
                "risk_aversion": risk_aversion,
                "action_and_pnl_scale_relative_to_fit": scale,
                "training_set": scaled(training_set),
                "validation_set": scaled(validation_set),
            }
        )
    return {
        "values": values,
        "identity": (
            "with a_gamma=v/(gamma*return_RMS) and homogeneous transaction "
            "costs, each objective is 1/gamma times the same function of v; "
            "actions, P&L, objectives, and their MC errors scale as 1/gamma, "
            "while marginal utility, normalized density, and z statistics are "
            "unchanged"
        ),
        "separate_refits_are_mathematically_redundant": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=args.paths,
    )
    utility: CellUtility = args.utility
    loaded = _load_case(case, cache_root=Path(args.cache_root), need_primitives=False)
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=32, device=args.device, debug=args.debug)
    data, split_diagnostics = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    torch.cuda.synchronize(env.device)

    label = f"cell-{utility}-g{args.risk_aversion:.0e}-{case.generator}-{case.path_count}"
    model_config = CellMeasureChangeConfig(
        label=label,
        source_cache_uid=source_cache_uid,
        group=(
            f"gendysanos/cell-measure-change/{case.generator}/{utility}/"
            f"gamma-{args.risk_aversion:.0e}/{case.path_count}"
        ),
        user_version=args.user_version,
        state_features=data.state.shape[-1],
        risk_aversion=args.risk_aversion,
        utility=utility,
    )
    train_config = CellMeasureChangeTrainConfig(
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
        seed=args.seed,
    )
    checkpoint_path = Path(args.checkpoint_root) / model_config.group / (
        f"{model_config.filename}.pt"
    )
    report_path = Path(args.output_root) / f"{label}.json"
    started = time.monotonic()
    with FileLock(str(checkpoint_path) + ".lock"):
        model, training = _train(
            data,
            model_config=model_config,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            env=env,
        )
        training_set = _evaluate(
            model,
            data,
            data.reference,
            utility=utility,
            risk_aversion=args.risk_aversion,
            batch_size=args.valuation_batch_size,
        )
        validation_set = _evaluate(
            model,
            data,
            data.validation,
            utility=utility,
            risk_aversion=args.risk_aversion,
            batch_size=args.valuation_batch_size,
        )
        gradient = {
            "training": _gradient_diagnostics(
                model,
                data,
                data.reference,
                utility=utility,
                risk_aversion=args.risk_aversion,
            ),
            "validation": _gradient_diagnostics(
                model,
                data,
                data.validation,
                utility=utility,
                risk_aversion=args.risk_aversion,
            ),
        }
        risk_aversion_sweep = _risk_aversion_sweep(
            fitted_risk_aversion=args.risk_aversion,
            requested=args.risk_sweep,
            training_set=training_set,
            validation_set=validation_set,
        )
        report = {
            "status": "complete",
            "schema_version": "1.0",
            "experiment": "independent trading-time/expiry measure change",
            "configuration": asdict(case),
            "source_cache": source_cache,
            "source_cache_uid": source_cache_uid,
            "model_definition": {
                "independence": (
                    "70 separate a_{t,e}(x_t) networks; parameters and gradients "
                    "are not shared across trading times or expiries"
                ),
                "action": (
                    "unbounded raw network output divided instrumentwise by "
                    "gamma times training return RMS; no tanh, cap, radial map, "
                    "or gradient clipping"
                ),
                "cell_pnl": "X_{t,e}=a_{t,e}'dH_{t,e}-|a_{t,e}|'c_{t,e}",
                "objective": (
                    "maximize E_P[u(X_{t,e})] independently in every cell; "
                    "joint training only averages the 70 disjoint objectives"
                ),
                "density": (
                    "D=u'(X)/E_P[u'(X)], with its denominator recomputed "
                    "separately on training and validation"
                ),
            },
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "split": split_diagnostics,
            "training": training,
            "training_set": training_set,
            "validation_set": validation_set,
            "gradient_diagnostics": gradient,
            "risk_aversion_sweep": risk_aversion_sweep,
            "runtime_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path),
            "device": str(env.device),
        }
        _atomic_json(report_path, report)
    print(
        f"Completed {label}: train/validation max |FOC z|="
        f"{training_set['maximum_absolute_cost_adjusted_foc_z']:.3g}/"
        f"{validation_set['maximum_absolute_cost_adjusted_foc_z']:.3g}; "
        f"report={report_path}",
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("dysanos", "pca-iv"), required=True)
    parser.add_argument(
        "--utility",
        choices=("powerexp", "exponential", "mean-variance"),
        required=True,
    )
    parser.add_argument("--risk-aversion", type=float, required=True)
    parser.add_argument(
        "--risk-sweep",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0],
    )
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--minimum-epochs", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--valuation-batch-size", type=int, default=16_384)
    parser.add_argument("--value-update", type=int, default=500)
    parser.add_argument("--checkpoint-update", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--convergence-points", type=int, default=8)
    parser.add_argument("--convergence-relative-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=82_061)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
