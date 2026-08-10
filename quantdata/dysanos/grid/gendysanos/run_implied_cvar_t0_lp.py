"""Solve the empirical time-zero minimum-density-cap problem exactly."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.err import verify_inp
from scipy import sparse
from scipy.optimize import linprog

from quantdata.torch import BaseEnvironment

from .required_runs import RequiredCase
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


def _independent_t0_returns(data) -> np.ndarray:
    """Keep every expiry-specific stock and ITM option return at time zero."""

    returns = data.returns[:, 0].double().cpu().numpy()
    result = returns.reshape(returns.shape[0], -1)
    scale = np.sqrt(np.mean(np.square(result), axis=0))
    if not np.all(np.isfinite(result)) or not np.all(scale > 0.0):
        raise FloatingPointError("time-zero normalized returns are degenerate")
    return result / scale[None]


def solve_minimum_density_cap(
    returns: np.ndarray,
    physical_probability: np.ndarray,
    *,
    method: str = "highs",
) -> tuple[Any, np.ndarray | None]:
    """Minimize ``max_s q_s/p_s`` subject to all sample moments being zero."""

    paths, instruments = returns.shape
    probability = np.asarray(physical_probability, dtype=np.float64)
    probability /= probability.sum()
    objective = np.zeros(paths + 1, dtype=np.float64)
    objective[-1] = 1.0

    equality = np.empty((instruments + 1, paths + 1), dtype=np.float64)
    equality[0, :-1] = 1.0
    equality[0, -1] = 0.0
    equality[1:, :-1] = returns.T
    equality[1:, -1] = 0.0
    equality_value = np.zeros(instruments + 1, dtype=np.float64)
    equality_value[0] = 1.0

    cap_column = sparse.csr_matrix(
        (-probability, (np.arange(paths), np.zeros(paths, dtype=np.int64))),
        shape=(paths, 1),
    )
    inequality = sparse.hstack(
        (sparse.eye(paths, format="csr"), cap_column), format="csr"
    )
    started = time.monotonic()
    options: dict[str, Any] = {"presolve": True}
    if method == "highs-ipm":
        options["run_crossover"] = "off"
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(paths, dtype=np.float64),
        A_eq=equality,
        b_eq=equality_value,
        bounds=[(0.0, None)] * paths + [(1.0, None)],
        method=method,
        options=options,
    )
    result.elapsed_seconds = time.monotonic() - started
    if not result.success:
        return result, None
    q = np.asarray(result.x[:-1], dtype=np.float64)
    density = q / probability
    return result, density


def solve_strict_sample_arbitrage_margin(returns: np.ndarray) -> Any:
    """Maximize the worst sampled return over signed unit-L1 portfolios."""

    paths, instruments = returns.shape
    objective = np.zeros(2 * instruments + 1, dtype=np.float64)
    objective[-1] = -1.0
    path_constraint = np.concatenate(
        (-returns, returns, np.ones((paths, 1), dtype=np.float64)), axis=1
    )
    gross_constraint = np.concatenate(
        (np.ones(2 * instruments, dtype=np.float64), np.zeros(1, dtype=np.float64))
    )[None]
    inequality = np.concatenate((path_constraint, gross_constraint), axis=0)
    right_hand_side = np.zeros(paths + 1, dtype=np.float64)
    right_hand_side[-1] = 1.0
    started = time.monotonic()
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=right_hand_side,
        bounds=[(0.0, None)] * (2 * instruments) + [(None, None)],
        method="highs",
        options={"presolve": True},
    )
    result.elapsed_seconds = time.monotonic() - started
    return result


def solve_density_cap_feasibility(
    returns: np.ndarray,
    physical_probability: np.ndarray,
    density_cap: float,
    *,
    method: str = "highs",
) -> Any:
    """Test exact moment feasibility at one fixed positive density cap."""

    paths, instruments = returns.shape
    probability = np.asarray(physical_probability, dtype=np.float64)
    probability /= probability.sum()
    equality = np.concatenate(
        (np.ones((1, paths), dtype=np.float64), returns.T), axis=0
    )
    value = np.zeros(instruments + 1, dtype=np.float64)
    value[0] = 1.0
    options: dict[str, Any] = {"presolve": True}
    if method == "highs-ipm":
        # A basic solution is unnecessary for a feasibility certificate. The
        # interior-point iterate is audited below, so skip the costly simplex
        # crossover on this highly degenerate zero-objective problem.
        options["run_crossover"] = "off"
    started = time.monotonic()
    result = linprog(
        np.zeros(paths, dtype=np.float64),
        A_eq=equality,
        b_eq=value,
        bounds=list(zip(np.zeros(paths), density_cap * probability, strict=True)),
        method=method,
        options=options,
    )
    result.elapsed_seconds = time.monotonic() - started
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
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
        data,
        cost_rate=0.0,
        vector_barrier=False,
    )
    returns = _independent_t0_returns(data)
    probability = data.path_weight.double().cpu().numpy()
    probability /= probability.sum()
    if args.margin_only:
        optimized = solve_strict_sample_arbitrage_margin(returns)
        report = {
            "status": "complete",
            "experiment": "strict empirical time-zero arbitrage margin",
            "definition": (
                "maximize delta subject to R a>=delta on every sampled path "
                "and ||a||_1<=1"
            ),
            "configuration": asdict(case),
            "source_cache": source_cache,
            "source_cache_uid": source_cache_uid,
            "source_split_unused_for_fitting": source_split,
            "portfolio_normalization": normalization,
            "paths": int(returns.shape[0]),
            "independent_return_coordinates": int(returns.shape[1]),
            "solver": {
                "success": bool(optimized.success),
                "status": int(optimized.status),
                "message": str(optimized.message),
                "iterations": int(optimized.nit),
                "elapsed_seconds": float(optimized.elapsed_seconds),
                "strict_sample_arbitrage_margin": (
                    float(-optimized.fun) if optimized.success else None
                ),
            },
        }
        report_path = Path(args.output_root) / (
            f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-"
            "t0-strict-margin.json"
        )
        _atomic_json(report_path, report)
        print(json.dumps(report["solver"], indent=2), flush=True)
        print(f"Completed strict t0 margin report {report_path}", flush=True)
        return report
    if args.feasibility_cap is not None:
        verify_inp(
            args.feasibility_cap > 1.0,
            lambda: (
                "--feasibility-cap must exceed one so that the reported "
                f"lambda is strictly positive; found {args.feasibility_cap}"
            ),
        )
        optimized = solve_density_cap_feasibility(
            returns,
            probability,
            args.feasibility_cap,
            method=args.solver,
        )
        density_path: Path | None = None
        density_diagnostics: dict[str, float] = {}
        if optimized.success:
            q = np.asarray(optimized.x, dtype=np.float64)
            density = q / probability
            moment = q @ returns
            normalization_error = abs(float(q.sum()) - 1.0)
            maximum_cap_excess = max(
                0.0, float(np.max(density - args.feasibility_cap))
            )
            maximum_negative_mass = max(0.0, float(np.max(-q)))
            density_diagnostics = {
                "density_minimum": float(density.min()),
                "density_maximum": float(density.max()),
                "density_weighted_mean": float(np.dot(probability, density)),
                "maximum_absolute_scaled_martingale_error": float(
                    np.max(np.abs(moment))
                ),
                "normalization_error": normalization_error,
                "maximum_density_cap_excess": maximum_cap_excess,
                "maximum_negative_target_mass": maximum_negative_mass,
                "positive_density_path_fraction": float(np.mean(density > 0.0)),
                "target_measure_effective_sample_size": float(
                    np.square(q.sum()) / np.square(q).sum()
                ),
            }
            density_path = Path(args.density_root) / (
                f"{args.generator}-weekly-expiry-itm-dollar-t0-"
                f"cap-{args.feasibility_cap:g}-density.npz"
            )
            density_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                density_path,
                density=density,
                probability=probability,
            )
        report = {
            "status": "complete",
            "experiment": "exact empirical time-zero fixed-cap feasibility",
            "configuration": asdict(case),
            "source_cache": source_cache,
            "source_cache_uid": source_cache_uid,
            "portfolio_normalization": normalization,
            "paths": int(returns.shape[0]),
            "independent_return_coordinates": int(returns.shape[1]),
            "density_cap": args.feasibility_cap,
            "lambda": args.feasibility_cap - 1.0,
            "solver": {
                "method": args.solver,
                "success": bool(optimized.success),
                "status": int(optimized.status),
                "message": str(optimized.message),
                "iterations": int(optimized.nit),
                "crossover_iterations": int(optimized.crossover_nit or 0),
                "elapsed_seconds": float(optimized.elapsed_seconds),
                **density_diagnostics,
            },
            "density_path": None if density_path is None else str(density_path),
        }
        report_path = Path(args.output_root) / (
            f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-t0-"
            f"cap-{args.feasibility_cap:g}-feasibility.json"
        )
        _atomic_json(report_path, report)
        print(json.dumps(report["solver"], indent=2), flush=True)
        print(f"Completed fixed-cap t0 report {report_path}", flush=True)
        return report
    optimized, density = solve_minimum_density_cap(returns, probability)

    diagnostics: dict[str, Any] = {
        "success": bool(optimized.success),
        "status": int(optimized.status),
        "message": str(optimized.message),
        "iterations": int(optimized.nit),
        "elapsed_seconds": float(optimized.elapsed_seconds),
    }
    density_path: Path | None = None
    if density is not None:
        q = probability * density
        moment = q @ returns
        cap = float(optimized.fun)
        diagnostics.update(
            {
                "minimum_density_cap": cap,
                "minimum_lambda": cap - 1.0,
                "density_minimum": float(density.min()),
                "density_maximum": float(density.max()),
                "density_weighted_mean": float(np.dot(probability, density)),
                "maximum_absolute_scaled_martingale_error": float(
                    np.max(np.abs(moment))
                ),
                "positive_density_path_fraction": float(np.mean(density > 0.0)),
                "target_measure_effective_sample_size": float(
                    np.square(q.sum()) / np.square(q).sum()
                ),
            }
        )
        density_path = (
            Path(args.density_root)
            / f"{args.generator}-weekly-expiry-itm-dollar-t0-density.npz"
        )
        density_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(density_path, density=density, probability=probability)

    report = {
        "status": "complete",
        "experiment": "exact empirical time-zero minimum CVaR density cap",
        "definition": (
            "minimize c subject to q>=0, sum(q)=1, q^T R=0, and "
            "q_s<=c p_s; lambda=c-1"
        ),
        "configuration": asdict(case),
        "source_cache": source_cache,
        "source_cache_uid": source_cache_uid,
        "source_split_unused_for_fitting": source_split,
        "portfolio_normalization": normalization,
        "paths": int(returns.shape[0]),
        "independent_return_coordinates": int(returns.shape[1]),
        "solver": diagnostics,
        "density_path": None if density_path is None else str(density_path),
    }
    report_path = Path(args.output_root) / (
        f"implied-cvar-{args.generator}-weekly-expiry-itm-dollar-t0-exact-lp.json"
    )
    _atomic_json(report_path, report)
    print(json.dumps(report["solver"], indent=2), flush=True)
    print(f"Completed exact t0 report {report_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("pca-iv", "dysanos"), required=True)
    parser.add_argument("--cache-root", default=str(_DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--density-root", default=str(_DEFAULT_DENSITY_ROOT))
    parser.add_argument("--margin-only", action="store_true")
    parser.add_argument("--feasibility-cap", type=float)
    parser.add_argument(
        "--solver", choices=("highs", "highs-ds", "highs-ipm"), default="highs"
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
