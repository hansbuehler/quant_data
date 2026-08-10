"""Find empirical CVaR levels which eliminate profitable portfolio directions."""

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
from torch import Tensor

from quantdata.torch import BaseEnvironment, BaseModel

from .arb_tracing import fit_physical_state_whitener, weighted_quantile
from .case_c_partition import (
    ParallelDirectionConfig,
    ParallelKernelDirections,
    ParallelRadialFeedForwardDirections,
)
from .required_runs import RequiredCase
from .run_case_c import _fit_kernel_basis
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case, _to_device
from .run_parallel_direction_comparison import (
    _atomic_json,
    _atomic_torch_save,
    _parallel_pnl,
    _select_times_and_fold,
    _to_itm_dollar_portfolio,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/implied-cvar"
)
_DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/implied-cvar/checkpoints"
)


def weighted_lower_tail_mean(
    value: Tensor,
    path_weight: Tensor,
    density_caps: Tensor,
) -> Tensor:
    """Return exact empirical lower-tail CVaR for every portfolio head.

    ``density_caps`` contains ``1+lambda``. Its reciprocal is the retained
    lower-tail probability. The final observation at the quantile receives
    only the probability mass needed to make the retained mass exact.
    """

    assert value.ndim == 3 and path_weight.shape == value.shape[:1]
    assert density_caps.ndim == 1 and density_caps.shape[0] == value.shape[1]
    assert torch.all(path_weight > 0.0) and torch.all(density_caps >= 1.0)
    probability = path_weight / path_weight.sum()
    expanded_probability = probability[:, None, None].expand_as(value)
    order = torch.argsort(value.detach(), dim=0)
    sorted_value = torch.gather(value, 0, order)
    sorted_probability = torch.gather(expanded_probability, 0, order)
    cumulative_before = torch.cumsum(sorted_probability, dim=0) - sorted_probability
    retained_probability = density_caps.reciprocal()[None, :, None]
    included_probability = torch.minimum(
        sorted_probability,
        torch.clamp_min(retained_probability - cumulative_before, 0.0),
    )
    result = (
        included_probability * sorted_value
    ).sum(dim=0) / retained_probability.squeeze(0)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError("weighted lower-tail CVaR contains NaN or infinity")
    return result


def _weighted_lower_tail_numpy(
    value: np.ndarray,
    normalized_weight: np.ndarray,
    density_cap: float,
) -> tuple[float, float]:
    """Return lower-tail mean and its upper VaR endpoint."""

    order = np.argsort(value, kind="stable")
    sorted_value = value[order]
    sorted_weight = normalized_weight[order]
    retained = 1.0 / density_cap
    before = np.cumsum(sorted_weight) - sorted_weight
    included = np.minimum(sorted_weight, np.maximum(retained - before, 0.0))
    positive = included > 0.0
    return (
        float(np.dot(included, sorted_value) / retained),
        float(sorted_value[np.flatnonzero(positive)[-1]]),
    )


def _weighted_lower_tail_dual_numpy(
    value: np.ndarray,
    normalized_weight: np.ndarray,
    density_cap: float,
) -> np.ndarray:
    """Return one exact minimizing CVaR density as probability masses."""

    order = np.argsort(value, kind="stable")
    sorted_weight = normalized_weight[order]
    retained = 1.0 / density_cap
    before = np.cumsum(sorted_weight) - sorted_weight
    included = np.minimum(sorted_weight, np.maximum(retained - before, 0.0))
    result = np.empty_like(included)
    result[order] = included / retained
    return result


def _checkpoint(
    path: Path,
    *,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    tracker: list[dict[str, Any]],
    completed: bool,
    best_conditional: list[dict[str, Tensor]] | None = None,
    best_objective: Tensor | None = None,
) -> None:
    _atomic_torch_save(
        path,
        {
            "model_code_version": type(model).version.unique_id48,
            "config_code_version": type(model.config).version.unique_id48,
            "user_version": model.config.user_version,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
            "tracker": tracker,
            "completed": completed,
            "best_conditional": best_conditional,
            "best_objective": best_objective,
        },
    )


def _train(
    model: BaseModel,
    callable_model: BaseModel,
    *,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    density_caps: Tensor,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    value_update: int,
    checkpoint_update: int,
    lr: float,
    seed: int,
    initial_output_noise: float,
) -> dict[str, Any]:
    """Maximize all empirical CVaR objectives with independent action rows."""

    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if model.config.architecture == "kernel":
        # SGD preserves relative coordinatewise CVaR supergradient magnitudes
        # for the concave fixed-basis optimization.
        optimizer = torch.optim.SGD(trainable, lr=lr, momentum=0.0)
        optimizer_name = "SGD"
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        optimizer_name = "AdamW"
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    tracker: list[dict[str, Any]] = []
    start_step = 0
    best_conditional: list[dict[str, Tensor]] | None = None
    best_objective: Tensor | None = None
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        compatible = (
            saved.get("model_code_version") == type(model).version.unique_id48
            and saved.get("config_code_version") == type(model.config).version.unique_id48
            and saved.get("user_version") == model.config.user_version
        )
        if compatible:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator_state"].cpu())
            tracker = list(saved.get("tracker", []))
            start_step = int(saved["step"])
            best_conditional = saved.get("best_conditional")
            best_objective = saved.get("best_objective")
            if best_objective is not None:
                best_objective = best_objective.cpu().double()
            if bool(saved.get("completed")) and start_step >= epochs:
                return {
                    "steps": start_step,
                    "resumed_complete_checkpoint": True,
                    "tracker": tracker,
                }
        else:
            print(f"Ignoring incompatible checkpoint {checkpoint_path}", flush=True)

    if model.config.architecture == "radial-feedforward":
        if best_conditional is None:
            best_conditional = [
                {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                for network in model.conditional
            ]
            best_objective = torch.zeros(
                len(model.config.notionals), dtype=torch.float64
            )
            if start_step == 0 and initial_output_noise > 0.0:
                with torch.no_grad():
                    for network in model.conditional:
                        output = network.In[-1] if network.In else network.I0
                        output.weight.normal_(
                            mean=0.0,
                            std=initial_output_noise,
                            generator=generator,
                        )
                        if output.bias is not None:
                            output.bias.normal_(
                                mean=0.0,
                                std=initial_output_noise,
                                generator=generator,
                            )
        assert best_objective is not None

    fixed_generator = torch.Generator(device=model.device)
    fixed_generator.manual_seed(seed + 1)
    fixed = torch.randperm(
        state.shape[0], generator=fixed_generator, device=model.device
    )[: min(16_384, state.shape[0])]
    started = time.monotonic()
    for step in range(start_step + 1, epochs + 1):
        indices = torch.randint(
            state.shape[0],
            (batch_size,),
            generator=generator,
            device=model.device,
        )
        optimizer.zero_grad(set_to_none=True)
        action = callable_model(state[indices])[0]
        pnl = _parallel_pnl(action, returns[indices], costs[indices])
        objective = weighted_lower_tail_mean(
            pnl,
            path_weight[indices],
            density_caps,
        )
        # Heads and trading times have disjoint action parameters. Summing
        # retains each objective's natural gradient scale.
        loss = (
            -objective[:, 1].sum()
            if model.config.architecture == "radial-feedforward"
            else -objective.sum()
        )
        loss.backward()
        for parameter in trainable:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("implied-CVaR gradient contains NaN or infinity")
        optimizer.step()
        model.project_parameters_()

        if step % value_update == 0 or step == epochs:
            with torch.no_grad():
                fixed_action = model(state[fixed])[0]
                fixed_pnl = _parallel_pnl(
                    fixed_action,
                    returns[fixed],
                    costs[fixed],
                )
                fixed_objective = weighted_lower_tail_mean(
                    fixed_pnl,
                    path_weight[fixed],
                    density_caps,
                )
            item = {
                "step": step,
                "objective_by_density_cap_and_time": fixed_objective.cpu().tolist(),
                "elapsed_seconds": time.monotonic() - started,
            }
            tracker.append(item)
            if best_conditional is not None:
                assert best_objective is not None
                conditional_objective = fixed_objective[:, 1].cpu().double()
                for head, value in enumerate(conditional_objective):
                    if value > best_objective[head]:
                        best_objective[head] = value
                        best_conditional[head] = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.conditional[head].state_dict().items()
                        }
            print(
                f"step {step:,}: CVaR={item['objective_by_density_cap_and_time']}",
                flush=True,
            )
        if step % checkpoint_update == 0:
            _checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                step=step,
                tracker=tracker,
                completed=False,
                best_conditional=best_conditional,
                best_objective=best_objective,
            )

    if best_conditional is not None:
        for network, state_dict in zip(model.conditional, best_conditional):
            network.load_state_dict(state_dict)
    _checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        step=epochs,
        tracker=tracker,
        completed=True,
        best_conditional=best_conditional,
        best_objective=best_objective,
    )
    return {
        "steps": epochs,
        "resumed_complete_checkpoint": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "constant_learning_rate": lr,
        "optimizer": optimizer_name,
        "gradient_clipping": False,
        "initial_output_noise": initial_output_noise,
        "best_fixed_sample_conditional_objective": (
            None if best_objective is None else best_objective.tolist()
        ),
        "tracker": tracker,
    }


@torch.no_grad()
def _evaluate(
    model: BaseModel,
    *,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    component: Tensor,
    density_caps: tuple[float, ...],
    indices: Tensor,
    batch_size: int,
) -> dict[str, Any]:
    paths = int(indices.numel())
    pnl = np.empty((paths, len(density_caps), 2), dtype=np.float64)
    net_l1 = np.empty_like(pnl)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(state[batch])[0]
        pnl[start:stop] = _parallel_pnl(
            action,
            returns[batch],
            costs[batch],
        ).cpu().double().numpy()
        net_l1[start:stop] = (
            action[..., 0, :, :] - action[..., 1, :, :]
        ).abs().sum(dim=(-1, -2)).cpu().double().numpy()
    weight = path_weight[indices].cpu().double().numpy()
    weight /= weight.sum()
    direct = component[indices].cpu().numpy() == 0
    result: list[dict[str, Any]] = []
    for head, cap in enumerate(density_caps):
        times: list[dict[str, Any]] = []
        for time_index, name in enumerate(("t0_deterministic", "t6_state_dependent")):
            value = pnl[:, head, time_index]
            cvar, var_endpoint = _weighted_lower_tail_numpy(value, weight, cap)
            times.append(
                {
                    "name": name,
                    "fitted_direction_empirical_OCE_CVaR": cvar,
                    "retained_lower_tail_probability": 1.0 / cap,
                    "upper_VaR_endpoint_of_retained_tail": var_endpoint,
                    "P_mean_pnl": float(np.dot(weight, value)),
                    "P_loss_probability": float(np.dot(weight, value < 0.0)),
                    "P_net_L1_mean": float(np.dot(weight, net_l1[:, head, time_index])),
                    "P_net_L1_99_quantile": weighted_quantile(
                        net_l1[:, head, time_index], weight, 0.99
                    ),
                    "sample_minimum_pnl": float(value.min()),
                    "direct_P_loss_count": int(np.sum((value < 0.0) & direct)),
                }
            )
        result.append(
            {
                "lambda": cap - 1.0,
                "density_cap": cap,
                "times": times,
            }
        )
    return {"paths": paths, "by_lambda": result}


def _frank_wolfe_gap(parameter: Tensor, gradient: Tensor) -> Tensor:
    """Return the concave maximization gap on nonnegative unit-L1 rows."""

    assert parameter.shape == gradient.shape and parameter.ndim >= 2
    current = (parameter * gradient).sum(dim=-1)
    best = torch.clamp_min(gradient.amax(dim=-1), 0.0)
    return torch.clamp_min(best - current, 0.0)


def _optimality_audit(
    model: ParallelKernelDirections,
    *,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    density_caps: tuple[float, ...],
    indices: Tensor,
    batch_size: int,
    zero_tolerance: float,
) -> dict[str, Any]:
    """Certify the fixed-kernel concave maxima with supergradient gaps."""

    paths = int(indices.numel())
    pnl = np.empty((paths, len(density_caps), 2), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, paths, batch_size):
            stop = min(paths, start + batch_size)
            batch = indices[start:stop]
            action = model(state[batch])[0]
            pnl[start:stop] = _parallel_pnl(
                action,
                returns[batch],
                costs[batch],
            ).cpu().double().numpy()

    weight = path_weight[indices].cpu().double().numpy()
    weight /= weight.sum()
    dual_mass = np.empty_like(pnl)
    objective = np.empty((len(density_caps), 2), dtype=np.float64)
    for head, cap in enumerate(density_caps):
        for time_index in range(2):
            dual_mass[:, head, time_index] = _weighted_lower_tail_dual_numpy(
                pnl[:, head, time_index], weight, cap
            )
            objective[head, time_index] = np.dot(
                dual_mass[:, head, time_index], pnl[:, head, time_index]
            )

    coefficient = torch.as_tensor(
        dual_mass,
        dtype=returns.dtype,
        device=returns.device,
    )
    model.zero_grad(set_to_none=True)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(state[batch])[0]
        batch_pnl = _parallel_pnl(action, returns[batch], costs[batch])
        (batch_pnl * coefficient[start:stop]).sum().backward()

    if model.deterministic_action.grad is None or model.template_action.grad is None:
        raise RuntimeError("CVaR optimality audit did not produce action gradients")
    deterministic_gap = _frank_wolfe_gap(
        model.deterministic_action,
        model.deterministic_action.grad,
    )
    conditional_gap = _frank_wolfe_gap(
        model.template_action,
        model.template_action.grad,
    ).sum(dim=-1)
    gap = torch.stack((deterministic_gap, conditional_gap), dim=-1)
    gap_numpy = gap.detach().cpu().double().numpy()
    model.zero_grad(set_to_none=True)

    by_lambda: list[dict[str, Any]] = []
    names = ("t0_deterministic", "t6_state_dependent")
    for head, cap in enumerate(density_caps):
        times: list[dict[str, Any]] = []
        for time_index, name in enumerate(names):
            lower = max(0.0, float(objective[head, time_index]))
            upper = max(
                lower,
                0.0,
                float(objective[head, time_index] + gap_numpy[head, time_index]),
            )
            times.append(
                {
                    "name": name,
                    "fitted_CVaR": float(objective[head, time_index]),
                    "Frank_Wolfe_supergradient_gap": float(
                        gap_numpy[head, time_index]
                    ),
                    "certified_maximum_lower_bound": lower,
                    "certified_maximum_upper_bound": upper,
                    "zero_within_tolerance": upper <= zero_tolerance,
                }
            )
        by_lambda.append(
            {"lambda": cap - 1.0, "density_cap": cap, "times": times}
        )

    minimum: dict[str, float | None] = {}
    for time_index, name in enumerate(names):
        minimum[name] = next(
            (
                cap - 1.0
                for head, cap in enumerate(density_caps)
                if by_lambda[head]["times"][time_index]["zero_within_tolerance"]
            ),
            None,
        )
    return {
        "definition": (
            "For the fixed Gaussian basis the objective is concave in the long-short "
            "portfolio rows. A CVaR supergradient gives a global upper bound via "
            "the Frank-Wolfe gap on each nonnegative unit-L1 row."
        ),
        "zero_tolerance": zero_tolerance,
        "by_lambda": by_lambda,
        "smallest_tested_lambda_certified_zero": minimum,
    }


def _feedforward_optimality_audit(
    model: ParallelRadialFeedForwardDirections,
    *,
    state: Tensor,
    returns: Tensor,
    costs: Tensor,
    path_weight: Tensor,
    density_caps: tuple[float, ...],
    indices: Tensor,
    batch_size: int,
) -> dict[str, Any]:
    """Report exact full-sample CVaR densities and network first-order residuals."""

    paths = int(indices.numel())
    heads = len(density_caps)
    pnl = np.empty((paths, heads), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, paths, batch_size):
            stop = min(paths, start + batch_size)
            batch = indices[start:stop]
            action = model(state[batch])[0]
            pnl[start:stop] = _parallel_pnl(
                action,
                returns[batch],
                costs[batch],
            )[:, :, 1].cpu().double().numpy()

    probability = path_weight[indices].cpu().double().numpy()
    probability /= probability.sum()
    target_mass = np.empty_like(pnl)
    objective = np.empty(heads, dtype=np.float64)
    var_endpoint = np.empty(heads, dtype=np.float64)
    for head, cap in enumerate(density_caps):
        target_mass[:, head] = _weighted_lower_tail_dual_numpy(
            pnl[:, head], probability, cap
        )
        objective[head], var_endpoint[head] = _weighted_lower_tail_numpy(
            pnl[:, head], probability, cap
        )

    coefficient = torch.as_tensor(
        target_mass, dtype=returns.dtype, device=returns.device
    )
    model.zero_grad(set_to_none=True)
    for start in range(0, paths, batch_size):
        stop = min(paths, start + batch_size)
        batch = indices[start:stop]
        action = model(state[batch])[0]
        batch_pnl = _parallel_pnl(
            action,
            returns[batch],
            costs[batch],
        )[:, :, 1]
        (batch_pnl * coefficient[start:stop]).sum().backward()

    result: list[dict[str, Any]] = []
    flattened_return = returns[indices, 1].reshape(paths, -1).cpu().double().numpy()
    return_rms = np.sqrt(probability @ np.square(flattened_return))
    for head, cap in enumerate(density_caps):
        gradients = [
            parameter.grad.detach().double().reshape(-1)
            for parameter in model.conditional[head].parameters()
            if parameter.grad is not None
        ]
        gradient = torch.cat(gradients) if gradients else torch.zeros(1)
        q = target_mass[:, head]
        density = q / probability
        moment = q @ flattened_return
        scaled_moment = np.divide(
            moment,
            return_rms,
            out=np.zeros_like(moment),
            where=return_rms > 0.0,
        )
        positive = density > 0.0
        tied = np.isclose(
            pnl[:, head],
            var_endpoint[head],
            rtol=0.0,
            atol=32.0 * np.finfo(np.float64).eps * max(1.0, abs(var_endpoint[head])),
        )
        result.append(
            {
                "lambda": cap - 1.0,
                "density_cap": cap,
                "fitted_full_sample_CVaR": float(objective[head]),
                "upper_VaR_endpoint_of_retained_tail": float(var_endpoint[head]),
                "paths_with_positive_density": int(positive.sum()),
                "positive_density_path_fraction": float(positive.mean()),
                "physical_probability_of_positive_density_paths": float(
                    probability[positive].sum()
                ),
                "paths_tied_at_VaR_endpoint": int(tied.sum()),
                "density_minimum": float(density.min()),
                "density_maximum": float(density.max()),
                "density_weighted_mean": float(np.dot(probability, density)),
                "target_measure_effective_sample_size": float(
                    1.0 / np.square(q).sum()
                ),
                "maximum_absolute_unconditional_return_moment": float(
                    np.abs(moment).max()
                ),
                "maximum_absolute_RMS_scaled_return_moment": float(
                    np.abs(scaled_moment).max()
                ),
                "network_parameter_gradient_L2": float(torch.linalg.vector_norm(gradient)),
                "network_parameter_gradient_maximum_absolute": float(gradient.abs().max()),
            }
        )
    model.zero_grad(set_to_none=True)
    return {
        "definition": (
            "For each fitted action, the exact lower-tail CVaR subgradient gives "
            "one density with 0<=D<=1+lambda. Network-parameter gradients test "
            "stationarity of that fitted action. Because the feed-forward problem "
            "is non-convex, these are diagnostics rather than a global certificate; "
            "when the action is identically zero, the CVaR subgradient is non-unique."
        ),
        "global_certificate_available": False,
        "by_lambda": result,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lambdas = tuple(float(value) for value in args.lambdas)
    verify_inp(
        bool(lambdas)
        and all(math.isfinite(value) and value > 0.0 for value in lambdas)
        and all(left < right for left, right in zip(lambdas, lambdas[1:])),
        lambda: f"lambdas must be finite, positive, and increasing: {lambdas}",
    )
    verify_inp(
        math.isfinite(args.zero_tolerance) and args.zero_tolerance >= 0.0,
        lambda: (
            "zero tolerance must be finite and nonnegative, got "
            f"{args.zero_tolerance!r}"
        ),
    )
    verify_inp(
        math.isfinite(args.initial_output_noise)
        and args.initial_output_noise >= 0.0,
        lambda: (
            "initial output noise must be finite and nonnegative, got "
            f"{args.initial_output_noise!r}"
        ),
    )
    density_caps = tuple(1.0 + value for value in lambdas)
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="raw",
        cost_rate=0.0,
        path_count=100_000,
    )
    loaded = _load_case(
        case,
        cache_root=Path(args.cache_root),
        need_primitives=True,
    )
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=args.precision, device=args.device, debug=args.debug)
    full_data, source_split = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    data, fold = _select_times_and_fold(
        full_data,
        deterministic_time=0,
        conditional_time=6,
        fold=0,
        seed=args.split_seed,
    )
    del full_data
    data, normalization = _to_itm_dollar_portfolio(
        data,
        cost_rate=0.0,
        vector_barrier=False,
    )
    all_indices = torch.arange(data.state.shape[0], device=env.device)
    data.reference = all_indices
    data.validation = all_indices

    state_mean, state_whitener, whitening = fit_physical_state_whitener(
        data.state,
        data.path_weight,
        all_indices,
        relative_eigenvalue_floor=args.covariance_relative_floor,
    )
    centers = inverse_bandwidth = None
    kernel_basis = None
    if args.architecture == "kernel":
        centers, inverse_bandwidth, kernel_basis = _fit_kernel_basis(
            data,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            kernels=args.kernels,
            seed=args.seed,
        )
    lambda_label = "-".join(f"{value:g}" for value in lambdas)
    label = (
        f"implied-cvar-{args.generator}-weekly-expiry-raw-0bps-100000-"
        f"{args.architecture}-"
        f"{args.kernels if args.architecture == 'kernel' else 'ffw'}-"
        f"lambda-{lambda_label}"
    )
    config = ParallelDirectionConfig(
        label=label,
        source_cache_uid=source_cache_uid,
        architecture=args.architecture,
        state_features=data.state.shape[-1],
        notionals=density_caps,
        kernels=args.kernels,
        group=f"gendysanos/implied-cvar/{args.generator}",
        user_version=args.user_version,
    )
    torch.manual_seed(args.seed)
    if torch.device(env.device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(env.device)
    if args.architecture == "kernel":
        assert centers is not None and inverse_bandwidth is not None
        model = ParallelKernelDirections(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
            centers=centers,
            inverse_bandwidth=inverse_bandwidth,
        )
        model.centers.requires_grad_(False)
        model.log_inverse_bandwidth.requires_grad_(False)
    else:
        model = ParallelRadialFeedForwardDirections(
            env,
            config=config,
            state_mean=state_mean[1],
            state_whitener=state_whitener[1],
        )
    callable_model = torch.compile(model, dynamic=False) if not args.no_compile else model
    cap_tensor = torch.as_tensor(density_caps, dtype=env.fdtype, device=env.device)
    checkpoint_path = Path(args.checkpoint_root) / config.group / f"{config.filename}.pt"
    with FileLock(f"{checkpoint_path}.lock"):
        training = _train(
            model,
            callable_model,
            state=data.state[:, 1],
            returns=data.returns,
            costs=data.costs,
            path_weight=data.path_weight,
            density_caps=cap_tensor,
            checkpoint_path=checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            value_update=args.value_update,
            checkpoint_update=args.checkpoint_update,
            lr=args.lr,
            seed=args.seed,
            initial_output_noise=args.initial_output_noise,
        )

    full = _evaluate(
        model,
        state=data.state[:, 1],
        returns=data.returns,
        costs=data.costs,
        path_weight=data.path_weight,
        component=data.component,
        density_caps=density_caps,
        indices=all_indices,
        batch_size=args.valuation_batch_size,
    )
    optimality = (
        _optimality_audit(
            model,
            state=data.state[:, 1],
            returns=data.returns,
            costs=data.costs,
            path_weight=data.path_weight,
            density_caps=density_caps,
            indices=all_indices,
            batch_size=args.valuation_batch_size,
            zero_tolerance=args.zero_tolerance,
        )
        if args.architecture == "kernel"
        else _feedforward_optimality_audit(
            model,
            state=data.state[:, 1],
            returns=data.returns,
            costs=data.costs,
            path_weight=data.path_weight,
            density_caps=density_caps,
            indices=all_indices,
            batch_size=args.valuation_batch_size,
        )
    )
    report = {
        "status": "complete",
        "schema_version": "1.0",
        "experiment": "empirical implied-CVaR martingale-level search",
        "definition": {
            "OCE": (
                "sup_{a,y} E[(1+lambda) min(y+a(x)^T R,0)]-y; "
                "implemented as the exact weighted lower-tail mean"
            ),
            "density_dual": (
                "E[D]=1, 0<=D<=1+lambda; zero maximum is equivalent within "
                "the fixed kernel strategy class to a martingale density"
            ),
            "strategy": (
                "t0 is deterministic; t6 uses independent portfolio parameters "
                f"for every lambda under the {args.architecture} architecture"
            ),
            "sample": "all 100,000 fixed paths are used for optimization and valuation",
        },
        "configuration": asdict(case),
        "lambdas": list(lambdas),
        "source_cache": source_cache,
        "source_split_unused_for_fitting": source_split,
        "reciprocal_fold_unused_for_fitting": fold,
        "portfolio_normalization": normalization,
        "model_config": asdict(config),
        "whitening": whitening,
        "kernel_basis": kernel_basis,
        "training": training,
        "full_sample": full,
        "global_optimality_certificate": optimality,
        "checkpoint": str(checkpoint_path),
        "device": str(env.device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(env.device))
            if torch.device(env.device).type == "cuda"
            else 0
        ),
    }
    report_path = Path(args.output_root) / f"{config.label}.json"
    _atomic_json(report_path, report)
    print(f"Completed implied-CVaR report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument(
        "--architecture",
        choices=("kernel", "radial-feedforward"),
        default="kernel",
    )
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--valuation-batch-size", type=int, default=8_192)
    parser.add_argument("--value-update", type=int, default=1_000)
    parser.add_argument("--checkpoint-update", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--initial-output-noise", type=float, default=1.0e-4)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--covariance-relative-floor", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=283_091)
    parser.add_argument("--split-seed", type=int, default=100_097)
    parser.add_argument("--user-version", default="0.1")
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-root", default=str(_DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
