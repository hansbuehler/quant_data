"""Solve the empirical time-six CVaR density cap on a hard state partition."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.err import verify, verify_inp

from quantdata.torch import BaseEnvironment

from .arb_tracing import fit_physical_state_whitener
from .required_runs import RequiredCase
from .run_case_c import _fit_kernel_basis
from .run_expiry_arbitrage_experiments import _DEFAULT_CACHE_ROOT, _load_case, _to_device
from .run_implied_cvar_t0_lp import (
    solve_minimum_density_cap,
    solve_strict_sample_arbitrage_margin,
)
from .run_parallel_direction_comparison import (
    _select_times_and_fold,
    _to_itm_dollar_portfolio,
)

_DEFAULT_OUTPUT_ROOT = Path(
    "quantdata/dysanos/grid/gendysanos/reports/implied-cvar"
)
_DEFAULT_DENSITY_ROOT = Path("/tmp/quantdata/dysanos/implied-cvar/densities")


def combine_partition_densities(
    probability: np.ndarray,
    assignment: np.ndarray,
    local_solutions: dict[int, tuple[float, np.ndarray]],
) -> tuple[float, np.ndarray, dict[int, float]]:
    """Combine minimum-cap conditional densities into one global density.

    If cell ``k`` has physical mass ``P_k`` and minimum conditional density
    cap ``c_k``, a global cap ``c`` assigns target mass
    ``m_k=c P_k/c_k`` to that cell. Normalization therefore gives
    ``c=1/sum_k(P_k/c_k)``. Cells without a feasible local density receive
    zero target mass.
    """

    probability = np.asarray(probability, dtype=np.float64)
    assignment = np.asarray(assignment)
    verify_inp(
        probability.ndim == assignment.ndim == 1
        and probability.shape == assignment.shape,
        lambda: (
            "probability and assignment must be aligned one-dimensional arrays; "
            f"found {probability.shape} and {assignment.shape}"
        ),
    )
    verify_inp(
        np.all(np.isfinite(probability))
        and np.all(probability > 0.0)
        and np.isclose(probability.sum(), 1.0),
        lambda: "probability must be finite, positive, and normalized",
    )
    denominator = 0.0
    cell_mass: dict[int, float] = {}
    for cell, (local_cap, local_density) in local_solutions.items():
        indices = np.flatnonzero(assignment == cell)
        verify_inp(
            indices.size == np.asarray(local_density).size,
            lambda: (
                f"cell {cell} has {indices.size} paths but "
                f"{np.asarray(local_density).size} density values"
            ),
        )
        verify_inp(
            np.isfinite(local_cap) and local_cap >= 1.0,
            lambda: f"cell {cell} has invalid density cap {local_cap}",
        )
        physical_mass = float(probability[indices].sum())
        cell_mass[cell] = physical_mass
        denominator += physical_mass / local_cap
    verify_inp(
        denominator > 0.0,
        lambda: "no state cell admits a sampled-path martingale density",
    )

    global_cap = 1.0 / denominator
    density = np.zeros_like(probability)
    target_mass: dict[int, float] = {}
    for cell, (local_cap, local_density) in local_solutions.items():
        indices = np.flatnonzero(assignment == cell)
        density[indices] = (
            global_cap / local_cap * np.asarray(local_density, dtype=np.float64)
        )
        target_mass[cell] = global_cap * cell_mass[cell] / local_cap
    return global_cap, density, target_mass


def _hard_kernel_assignment(
    state: torch.Tensor,
    *,
    state_mean: torch.Tensor,
    state_whitener: torch.Tensor,
    centers: torch.Tensor,
    inverse_bandwidth: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Assign every state to its largest fixed Gaussian kernel."""

    result = np.empty(state.shape[0], dtype=np.int64)
    with torch.no_grad():
        for start in range(0, state.shape[0], batch_size):
            stop = min(state.shape[0], start + batch_size)
            factor = (state[start:stop] - state_mean[None]) @ state_whitener
            adjusted_distance = (
                (factor[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
                * inverse_bandwidth.square()[None]
            )
            result[start:stop] = adjusted_distance.argmin(dim=-1).cpu().numpy()
    return result


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_inp(args.kernels > 1, lambda: f"kernels must exceed one, got {args.kernels}")
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
    env = BaseEnvironment(precision=64, device="cpu", debug=False)
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
    assignment = _hard_kernel_assignment(
        data.state[:, 1],
        state_mean=state_mean[1],
        state_whitener=state_whitener[1],
        centers=centers,
        inverse_bandwidth=inverse_bandwidth,
        batch_size=args.valuation_batch_size,
    )
    native_return = data.returns[:, 1].double().cpu().numpy()
    returns = native_return.reshape(native_return.shape[0], -1)
    probability = data.path_weight.double().cpu().numpy()
    probability /= probability.sum()

    local_solutions: dict[int, tuple[float, np.ndarray]] = {}
    cells: list[dict[str, Any]] = []
    for cell in range(args.kernels):
        indices = np.flatnonzero(assignment == cell)
        physical_mass = float(probability[indices].sum())
        conditional_probability = probability[indices] / physical_mass
        cell_return = returns[indices]
        scale = np.sqrt(conditional_probability @ np.square(cell_return))
        active = np.isfinite(scale) & (scale > 0.0)
        verify(
            np.all(np.isfinite(cell_return)),
            lambda: f"cell {cell} contains a non-finite return",
        )
        normalized_return = cell_return[:, active] / scale[active][None]
        attempts: list[dict[str, Any]] = []
        for method in ("highs", "highs-ipm", "highs-ds"):
            optimized, local_density = solve_minimum_density_cap(
                normalized_return,
                conditional_probability,
                method=method,
            )
            attempts.append(
                {
                    "method": method,
                    "success": bool(optimized.success),
                    "status": int(optimized.status),
                    "message": str(optimized.message),
                    "iterations": int(optimized.nit),
                    "elapsed_seconds": float(optimized.elapsed_seconds),
                }
            )
            if optimized.success or optimized.status == 2:
                break
        item: dict[str, Any] = {
            "cell": cell,
            "paths": int(indices.size),
            "physical_probability": physical_mass,
            "active_return_coordinates": int(active.sum()),
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "iterations": int(optimized.nit),
            "elapsed_seconds": float(optimized.elapsed_seconds),
            "attempts": attempts,
        }
        if local_density is not None:
            local_cap = float(optimized.fun)
            local_q = conditional_probability * local_density
            local_moment = local_q @ cell_return[:, active]
            item.update(
                {
                    "minimum_conditional_density_cap": local_cap,
                    "minimum_conditional_lambda": local_cap - 1.0,
                    "maximum_absolute_martingale_error": float(
                        np.max(np.abs(local_moment))
                    ),
                    "positive_density_path_fraction": float(
                        np.mean(local_density > 0.0)
                    ),
                }
            )
            local_solutions[cell] = (local_cap, local_density)
        else:
            separating = solve_strict_sample_arbitrage_margin(normalized_return)
            strict_margin = (
                float(-separating.fun) if separating.success else None
            )
            item["strict_separation"] = {
                "definition": (
                    "maximize delta subject to R a>=delta on every cell path "
                    "and ||a||_1<=1"
                ),
                "success": bool(separating.success),
                "status": int(separating.status),
                "message": str(separating.message),
                "iterations": int(separating.nit),
                "elapsed_seconds": float(separating.elapsed_seconds),
                "margin": strict_margin,
                "certifies_no_cell_martingale_density": bool(
                    strict_margin is not None and strict_margin > 1.0e-8
                ),
            }
        cells.append(item)
        print(
            f"cell={cell:02d} paths={indices.size:,} "
            f"cap={item.get('minimum_conditional_density_cap')}",
            flush=True,
        )

    unresolved = [
        item
        for item in cells
        if not item["success"]
        and not item.get("strict_separation", {}).get(
            "certifies_no_cell_martingale_density", False
        )
    ]
    global_cap, density, target_mass = combine_partition_densities(
        probability, assignment, local_solutions
    )
    verify_inp(
        global_cap > 1.0,
        lambda: (
            "the exact partition result has lambda=0; this experiment requires "
            f"strictly positive lambda, found cap {global_cap}"
        ),
    )
    q = probability * density
    unresolved_physical_probability = sum(
        float(item["physical_probability"]) for item in unresolved
    )
    minimum_cap_lower_bound = 1.0 / (
        1.0 / global_cap + unresolved_physical_probability
    )
    hard_moment_maximum = 0.0
    for cell in local_solutions:
        indices = assignment == cell
        hard_moment_maximum = max(
            hard_moment_maximum,
            float(np.max(np.abs(q[indices] @ returns[indices]))),
        )
    density_path = Path(args.density_root) / (
        f"{args.generator}-weekly-expiry-itm-dollar-t6-hard-"
        f"{args.kernels}-density.npz"
    )
    density_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        density_path,
        density=density,
        probability=probability,
        assignment=assignment,
    )
    for item in cells:
        item["target_measure_probability"] = float(target_mass.get(item["cell"], 0.0))

    report = {
        "status": "complete",
        "experiment": (
            "empirical time-six hard-partition CVaR density-cap construction"
        ),
        "definition": (
            "the action is an independent deterministic portfolio in each learned "
            "hard state cell; solved cells use their exact conditional minimum-cap "
            "martingale LP, strictly separated cells receive zero target mass, and "
            "numerically unresolved cells also receive zero mass for the reported "
            "constructive upper bound"
        ),
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "source_split_unused_for_fitting": source_split,
        "portfolio_normalization": normalization,
        "whitening": whitening,
        "kernel_basis": kernel_basis,
        "return_coordinates": int(returns.shape[1]),
        "hard_partition": {
            "cells": args.kernels,
            "feasible_cells": len(local_solutions),
            "strictly_separated_cells": sum(
                item.get("strict_separation", {}).get(
                    "certifies_no_cell_martingale_density", False
                )
                for item in cells
            ),
            "numerically_unresolved_cells": len(unresolved),
            "numerically_unresolved_cell_indices": [
                int(item["cell"]) for item in unresolved
            ],
            "numerically_unresolved_physical_probability": (
                unresolved_physical_probability
            ),
            "minimum_paths_per_cell": int(np.bincount(assignment).min()),
            "median_paths_per_cell": float(np.median(np.bincount(assignment))),
            "maximum_paths_per_cell": int(np.bincount(assignment).max()),
        },
        "minimum_density_cap_lower_bound": minimum_cap_lower_bound,
        "minimum_density_cap_upper_bound": global_cap,
        "minimum_lambda_lower_bound": minimum_cap_lower_bound - 1.0,
        "minimum_lambda_upper_bound": global_cap - 1.0,
        "feasible_retained_lower_tail_probability": 1.0 / global_cap,
        "density_diagnostics": {
            "minimum": float(density.min()),
            "maximum": float(density.max()),
            "weighted_mean": float(np.dot(probability, density)),
            "normalization_error": abs(float(q.sum()) - 1.0),
            "maximum_cap_excess": max(0.0, float(density.max() - global_cap)),
            "maximum_negative_target_mass": max(0.0, float(np.max(-q))),
            "maximum_absolute_hard_cell_martingale_error": hard_moment_maximum,
            "positive_density_path_fraction": float(np.mean(density > 0.0)),
            "target_measure_effective_sample_size": float(
                np.square(q.sum()) / np.square(q).sum()
            ),
        },
        "cells": cells,
        "density_path": str(density_path),
    }
    report_path = Path(args.output_root) / (
        f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-t6-"
        f"hard-{args.kernels}-exact-lp.json"
    )
    _atomic_json(report_path, report)
    print(
        f"lambda upper bound={report['minimum_lambda_upper_bound']:.12g}; "
        f"martingale error={hard_moment_maximum:.3g}",
        flush=True,
    )
    print(f"Completed t6 partition report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), required=True)
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--valuation-batch-size", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=283_091)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--density-root", default=str(_DEFAULT_DENSITY_ROOT))
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
