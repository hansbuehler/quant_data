"""Construct capped empirical densities for the conditional time-six market."""

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
from torch import Tensor

from quantdata.torch import BaseEnvironment

from .arb_tracing import fit_physical_state_whitener
from .required_runs import RequiredCase
from .run_case_c import _fit_kernel_basis
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case, _to_device
from .run_parallel_direction_comparison import (
    _select_times_and_fold,
    _to_itm_dollar_portfolio,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/implied-cvar"
)
_DEFAULT_DENSITY_ROOT = Path(
    "/media/storage/tmp/quantdata/dysanos/implied-cvar/densities"
)


@torch.no_grad()
def project_weighted_capped_density_(
    density: Tensor,
    probability: Tensor,
    density_cap: float,
    *,
    iterations: int = 48,
) -> None:
    """Project in weighted L2 onto ``E_p[D]=1`` and ``0<=D<=cap``."""

    assert density.ndim == probability.ndim == 1 and density.shape == probability.shape
    assert density_cap >= 1.0 and torch.all(probability > 0.0)
    lower = torch.min(density - density_cap)
    upper = torch.max(density)
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        mean = torch.dot(probability, torch.clamp(density - midpoint, 0.0, density_cap))
        lower = torch.where(mean > 1.0, midpoint, lower)
        upper = torch.where(mean > 1.0, upper, midpoint)
    density.copy_(torch.clamp(density - 0.5 * (lower + upper), 0.0, density_cap))
    # Remove the final float32 normalization error without violating the cap.
    error = 1.0 - torch.dot(probability, density)
    if torch.abs(error) > 16.0 * torch.finfo(density.dtype).eps:
        if error > 0.0:
            room = density < density_cap
        else:
            room = density > 0.0
        correction_mass = probability[room].sum()
        density[room] += error / correction_mass
        density.clamp_(0.0, density_cap)


def conditional_moments(
    density: Tensor,
    probability: Tensor,
    kernel_weight: Tensor,
    returns: Tensor,
) -> Tensor:
    """Return ``E_p[D phi_k R_j]`` for every kernel and instrument."""

    assert density.shape == probability.shape == returns.shape[:1]
    assert kernel_weight.shape[0] == returns.shape[0]
    weighted_return = probability[:, None] * density[:, None] * returns
    return kernel_weight.mT @ weighted_return


def fit_entropy_density(
    probability: Tensor,
    kernel_weight: Tensor,
    returns: Tensor,
    *,
    max_iter: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Find an uncapped exponential-family density with zero kernel moments."""

    theta = torch.nn.Parameter(
        torch.zeros(
            (kernel_weight.shape[1], returns.shape[1]),
            dtype=returns.dtype,
            device=returns.device,
        )
    )
    optimizer = torch.optim.LBFGS(
        (theta,),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    log_probability = torch.log(probability)
    evaluations = 0

    def closure() -> Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        score = (returns @ theta.mT * kernel_weight).sum(dim=-1)
        log_normalizer = torch.logsumexp(log_probability + score, dim=0)
        log_normalizer.backward()
        evaluations += 1
        return log_normalizer

    started = time.monotonic()
    optimizer.step(closure)
    with torch.no_grad():
        score = (returns @ theta.mT * kernel_weight).sum(dim=-1)
        log_normalizer = torch.logsumexp(log_probability + score, dim=0)
        density = torch.exp(score - log_normalizer)
        moment = conditional_moments(
            density, probability, kernel_weight, returns
        )
        support = moment.abs().amax(dim=-1).sum()
    if not torch.isfinite(density).all().item():
        raise FloatingPointError("entropy initializer produced a non-finite density")
    return density, {
        "definition": "D is proportional to exp(sum_k phi_k(x) theta_k^T R)",
        "evaluations": evaluations,
        "elapsed_seconds": time.monotonic() - started,
        "weighted_mean_density": float(torch.dot(probability, density).item()),
        "minimum_density": float(density.min().item()),
        "maximum_density": float(density.max().item()),
        "conditional_OCE_maximum_upper_bound": float(support.item()),
    }


@torch.no_grad()
def _kernel_weights(
    state: Tensor,
    *,
    state_mean: Tensor,
    state_whitener: Tensor,
    centers: Tensor,
    inverse_bandwidth: Tensor,
    batch_size: int,
) -> Tensor:
    result = torch.empty(
        (state.shape[0], centers.shape[0]), dtype=state.dtype, device=state.device
    )
    for start in range(0, state.shape[0], batch_size):
        stop = min(state.shape[0], start + batch_size)
        factor = (state[start:stop] - state_mean[None]) @ state_whitener
        distance_square = (
            factor[:, None, :] - centers[None, :, :]
        ).square().sum(dim=-1)
        result[start:stop] = torch.softmax(
            -0.5 * distance_square * inverse_bandwidth.square()[None], dim=-1
        )
    return result


def _fit_one_cap(
    initial_density: Tensor,
    *,
    density_cap: float,
    probability: Tensor,
    kernel_weight: Tensor,
    returns: Tensor,
    moment_scale: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    value_update: int,
    tolerance: float,
) -> tuple[Tensor, dict[str, Any]]:
    density = torch.nn.Parameter(initial_density.detach().clone())
    project_weighted_capped_density_(density, probability, density_cap)
    verify_inp(
        batch_size == density.shape[0],
        lambda: (
            "the monotone weighted projected-gradient solve requires the full "
            f"sample ({density.shape[0]}), got batch size {batch_size}"
        ),
    )
    tracker: list[dict[str, Any]] = []
    consecutive = 0
    started = time.monotonic()
    step_size = learning_rate
    for step in range(1, steps + 1):
        density.grad = None
        moment = conditional_moments(
            density, probability, kernel_weight, returns
        )
        loss = 0.5 * torch.square(moment / moment_scale).sum()
        loss.backward()
        if density.grad is None or not torch.isfinite(density.grad).all().item():
            raise FloatingPointError("conditional density gradient is non-finite")
        # The projection uses the physical weighted L2 metric. Its matching
        # gradient is the Euclidean autograd gradient divided by p.
        direction = density.grad / probability
        previous = density.detach().clone()
        previous_loss = float(loss.item())
        accepted = False
        for _ in range(20):
            with torch.no_grad():
                density.copy_(previous - step_size * direction)
                project_weighted_capped_density_(
                    density, probability, density_cap
                )
                candidate_moment = conditional_moments(
                    density, probability, kernel_weight, returns
                )
                candidate_loss = float(
                    (0.5 * torch.square(candidate_moment / moment_scale).sum()).item()
                )
            if candidate_loss <= previous_loss:
                accepted = True
                step_size *= 1.05
                break
            step_size *= 0.5
        if not accepted:
            with torch.no_grad():
                density.copy_(previous)
            step_size = max(step_size, torch.finfo(density.dtype).eps)

        if step % value_update == 0 or step == steps:
            with torch.no_grad():
                raw_moment = conditional_moments(
                    density, probability, kernel_weight, returns
                )
                support = raw_moment.abs().amax(dim=-1).sum()
                standardized_rms = torch.sqrt(
                    torch.square(raw_moment / moment_scale).mean()
                )
                mean_density = torch.dot(probability, density)
            item = {
                "step": step,
                "conditional_OCE_maximum_upper_bound": float(support.item()),
                "standardized_moment_RMS": float(standardized_rms.item()),
                "weighted_mean_density": float(mean_density.item()),
                "minimum_density": float(density.min().item()),
                "maximum_density": float(density.max().item()),
                "projected_gradient_step_size": step_size,
                "elapsed_seconds": time.monotonic() - started,
            }
            tracker.append(item)
            print(
                f"cap={density_cap:g} step={step:,} "
                f"upper={item['conditional_OCE_maximum_upper_bound']:.6g} "
                f"std_rms={item['standardized_moment_RMS']:.6g}",
                flush=True,
            )
            consecutive = consecutive + 1 if support <= tolerance else 0
            if consecutive >= 3:
                break
    return density.detach(), {
        "density_cap": density_cap,
        "lambda": density_cap - 1.0,
        "steps": step,
        "converged": consecutive >= 3,
        "tolerance": tolerance,
        "tracker": tracker,
        "final": tracker[-1],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lambdas = tuple(float(value) for value in args.lambdas)
    verify_inp(
        bool(lambdas)
        and all(math.isfinite(value) and value > 0.0 for value in lambdas)
        and all(left < right for left, right in zip(lambdas, lambdas[1:])),
        lambda: f"lambdas must be finite, positive, and increasing: {lambdas}",
    )
    torch.set_float32_matmul_precision("high")
    case = RequiredCase(
        generator=args.generator,
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="raw",
        cost_rate=0.0,
        path_count=100_000,
    )
    loaded = _load_case(case, cache_root=Path(args.cache_root), need_primitives=True)
    source_cache = str(loaded.source_cache)
    source_cache_uid = loaded.source_cache_uid
    env = BaseEnvironment(precision=32, device=args.device, debug=False)
    full_data, source_split = _to_device(loaded, case=case, env=env)
    del loaded
    gc.collect()
    data, _ = _select_times_and_fold(
        full_data,
        deterministic_time=0,
        conditional_time=6,
        fold=0,
        seed=100_097,
    )
    del full_data
    data, normalization = _to_itm_dollar_portfolio(
        data, cost_rate=0.0, vector_barrier=False
    )
    all_indices = torch.arange(data.state.shape[0], device=env.device)
    data.reference = all_indices
    data.validation = all_indices
    state_mean, state_whitener, whitening = fit_physical_state_whitener(
        data.state,
        data.path_weight,
        all_indices,
        relative_eigenvalue_floor=1.0e-10,
    )
    centers, inverse_bandwidth, kernel_basis = _fit_kernel_basis(
        data,
        state_mean=state_mean[1],
        state_whitener=state_whitener[1],
        kernels=args.kernels,
        seed=args.seed,
    )
    kernel_weight = _kernel_weights(
        data.state[:, 1],
        state_mean=state_mean[1],
        state_whitener=state_whitener[1],
        centers=centers,
        inverse_bandwidth=inverse_bandwidth,
        batch_size=args.valuation_batch_size,
    )
    native_return = data.returns[:, 1]
    returns = native_return.reshape(native_return.shape[0], -1)
    probability = data.path_weight / data.path_weight.sum()
    # The smallest proposal weights can underflow only when normalized in
    # float32. Giving them the smallest normal mass is immaterial in total but
    # preserves the strictly positive reference measure required by D=q/p.
    probability = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    probability /= probability.sum()
    if args.dual_precision == 64:
        probability = probability.double()
        kernel_weight = kernel_weight.double()
        returns = returns.double()
    moment_scale = torch.sqrt(
        kernel_weight.square().mT
        @ (probability[:, None] * returns.square())
    ).clamp_min(1.0e-8)

    entropy_density, entropy = fit_entropy_density(
        probability,
        kernel_weight,
        returns,
        max_iter=args.entropy_steps,
    )
    print(f"Entropy density: {entropy}", flush=True)

    density = torch.ones_like(probability)
    fits: list[dict[str, Any]] = []
    saved_density: Path | None = None
    for head, lmbda in enumerate(lambdas):
        entropy_start = entropy_density.clone()
        project_weighted_capped_density_(
            entropy_start, probability, 1.0 + lmbda
        )
        with torch.no_grad():
            warm_loss = torch.square(
                conditional_moments(
                    density, probability, kernel_weight, returns
                )
                / moment_scale
            ).sum()
            entropy_loss = torch.square(
                conditional_moments(
                    entropy_start, probability, kernel_weight, returns
                )
                / moment_scale
            ).sum()
        if entropy_loss < warm_loss:
            density = entropy_start
        density, fit = _fit_one_cap(
            density,
            density_cap=1.0 + lmbda,
            probability=probability,
            kernel_weight=kernel_weight,
            returns=returns,
            moment_scale=moment_scale,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            value_update=args.value_update,
            tolerance=args.tolerance,
        )
        fits.append(fit)
        if fit["converged"] and saved_density is None:
            saved_density = (
                Path(args.density_root)
                / f"{args.generator}-weekly-expiry-itm-dollar-t6-density.npz"
            )
            saved_density.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                saved_density,
                density=density.cpu().double().numpy(),
                probability=probability.cpu().double().numpy(),
            )

    minimum = next((fit["lambda"] for fit in fits if fit["converged"]), None)
    report = {
        "status": "complete",
        "experiment": "conditional empirical CVaR density-cap upper bounds",
        "definition": (
            "For fixed normalized Gaussian state kernels phi_k, D is projected "
            "onto E_p[D]=1 and 0<=D<=1+lambda. The upper bound on the maximum "
            "conditional CVaR OCE is sum_k max_j |E_p[D phi_k R_j]|."
        ),
        "configuration": asdict(case),
        "lambdas": list(lambdas),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "source_split_unused_for_fitting": source_split,
        "portfolio_normalization": normalization,
        "whitening": whitening,
        "kernel_basis": kernel_basis,
        "entropy_density_initializer": entropy,
        "independent_return_coordinates": int(returns.shape[1]),
        "fits": fits,
        "smallest_tested_lambda_certified_zero": minimum,
        "density_path": None if saved_density is None else str(saved_density),
    }
    report_path = Path(args.output_root) / (
        f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-t6-dual.json"
    )
    _atomic = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(_atomic, report_path)
    print(f"Completed conditional dual report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2"), required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=(5, 6, 7, 8, 9, 10))
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--valuation-batch-size", type=int, default=8_192)
    parser.add_argument("--value-update", type=int, default=500)
    parser.add_argument("--entropy-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--dual-precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--seed", type=int, default=283_091)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--density-root", default=str(_DEFAULT_DENSITY_ROOT))
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
