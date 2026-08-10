"""Empirical arbitrage diagnostics for finite simulated path sets.

Array convention
----------------
dC[omega, t, ell] = dC_t^ell(omega), with shape
(num_paths, num_steps, num_instruments).

The basic LP uses deterministic positions a[t, ell], common to every path.
It therefore does not use pathwise foresight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import csv
import json
import math
import time
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeWarning, linprog

FloatArray = NDArray[np.float64]


def _progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[empirical_arbitrage] {message}", flush=True)


@dataclass(frozen=True)
class ArbitrageConfig:
    delta: float = 0.0
    prop_cost: float = 0.0
    max_abs_position: float = 1.0
    payoff_tolerance: float = 1e-10
    active_tolerance: float = 1e-8
    method: str = "highs"
    presolve: bool = True
    threads: int = 0

    def __post_init__(self) -> None:
        for name in (
            "delta",
            "prop_cost",
            "max_abs_position",
            "payoff_tolerance",
            "active_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name!r} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.delta < 0 or self.prop_cost < 0:
            raise ValueError("delta and prop_cost must be non-negative")
        if self.max_abs_position <= 0:
            raise ValueError("max_abs_position must be positive")
        if self.payoff_tolerance < 0 or self.active_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if not isinstance(self.threads, (int, np.integer)) or self.threads < 0:
            raise ValueError("threads must be a non-negative integer (0 uses all cores)")


@dataclass
class ArbitrageResult:
    solver_success: bool
    found: bool
    message: str
    config: ArbitrageConfig
    positions: FloatArray | None = None
    payoff: FloatArray | None = None
    gross_payoff: FloatArray | None = None
    total_cost: FloatArray | None = None
    minimum_payoff: float | None = None
    mean_payoff: float | None = None
    maximum_payoff: float | None = None
    l1_position: float | None = None
    max_abs_position: float | None = None
    num_active: int | None = None
    fraction_negative: float | None = None
    mc_error: FloatArray | None = None
    solver_status: int | None = None
    solver_nit: int | None = None

    def summary(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("positions", "payoff", "gross_payoff", "total_cost", "mc_error"):
            out.pop(key, None)
        return out


@dataclass
class SparseResult:
    solver_success: bool
    message: str
    required_mean_payoff: float
    positions: FloatArray | None = None
    payoff: FloatArray | None = None
    gross_payoff: FloatArray | None = None
    total_cost: FloatArray | None = None
    minimum_payoff: float | None = None
    mean_payoff: float | None = None
    maximum_payoff: float | None = None
    l1_position: float | None = None
    num_active: int | None = None
    fraction_negative: float | None = None
    solver_status: int | None = None
    solver_nit: int | None = None

    def summary(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("positions", "payoff", "gross_payoff", "total_cost"):
            out.pop(key, None)
        return out


def _as_dC(dC: ArrayLike) -> FloatArray:
    arr = np.asarray(dC, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("dC must have shape (num_paths, num_steps, num_instruments)")
    if arr.shape[0] < 2:
        raise ValueError("at least two paths are required")
    if not np.isfinite(arr).all():
        raise ValueError("dC contains non-finite values")
    return np.ascontiguousarray(arr)


def _broadcast_X(X: ArrayLike | None, shape: tuple[int, int, int]) -> FloatArray:
    if X is None:
        return np.zeros(shape, dtype=np.float64)
    arr = np.asarray(X, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("X contains non-finite values")
    try:
        out = np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(
            f"X shape {arr.shape} is not broadcastable to dC shape {shape}; "
            "use scalar, (T,L), (1,T,L), or (Omega,T,L)"
        ) from exc
    return np.ascontiguousarray(np.abs(out), dtype=np.float64)


def mc_error_by_instrument(dC: ArrayLike) -> FloatArray:
    """Standard error of the sample mean for each (time, instrument)."""
    arr = _as_dC(dC)
    return np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])


def build_cost_tensor(
    dC: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return pathwise absolute-position cost coefficients and MC errors."""
    arr = _as_dC(dC)
    if delta < 0 or prop_cost < 0:
        raise ValueError("delta and prop_cost must be non-negative")
    # Avoid re-validating/copying dC through mc_error_by_instrument here. This
    # function is on the hot path and is called for every LP.
    stderr = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    Xb = _broadcast_X(X, arr.shape)
    cost = prop_cost * Xb + delta * stderr[None, :, :]
    return np.ascontiguousarray(cost), stderr


def evaluate_positions(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
) -> dict[str, FloatArray]:
    arr = _as_dC(dC)
    a = np.asarray(positions, dtype=np.float64)
    if a.shape != arr.shape[1:]:
        raise ValueError(f"positions must have shape {arr.shape[1:]}, got {a.shape}")
    cost, stderr = build_cost_tensor(arr, X, delta=delta, prop_cost=prop_cost)
    gross = np.sum(arr * a[None, :, :], axis=(1, 2))
    total_cost = np.sum(cost * np.abs(a)[None, :, :], axis=(1, 2))
    return {
        "gross_payoff": gross,
        "total_cost": total_cost,
        "payoff": gross - total_cost,
        "mc_error": stderr,
    }


def _net_matrix(dC: FloatArray, cost: FloatArray) -> tuple[FloatArray, int]:
    n_paths = dC.shape[0]
    M = dC.reshape(n_paths, -1)
    K = cost.reshape(n_paths, -1)
    # a = a+ - a-, |a| = a+ + a-
    return np.concatenate([M - K, -M - K], axis=1), M.shape[1]


def _highs_options(
    presolve: bool,
    threads: int,
    solver_output: bool,
) -> dict[str, Any]:
    """Options passed verbatim by SciPy to HiGHS.

    HiGHS interprets ``threads=0`` as using all available processors. Explicit
    parallel mode avoids leaving this at the solver's ``choose`` default.
    """
    if not isinstance(threads, (int, np.integer)) or threads < 0:
        raise ValueError("threads must be a non-negative integer (0 uses all cores)")
    return {
        "presolve": presolve,
        "threads": int(threads),
        "parallel": True,
        "disp": solver_output,
    }


def _solve_lp(**kwargs: Any) -> Any:
    # SciPy warns about options absent from its public signature, even though
    # its HiGHS wrapper deliberately forwards them to the solver.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*",
            category=OptimizeWarning,
        )
        return linprog(**kwargs)


def _evaluate_from_matrices(
    dC: FloatArray,
    cost: FloatArray,
    a_flat: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate a solution without rebuilding costs or 3-D temporaries."""
    n_paths = dC.shape[0]
    gross = dC.reshape(n_paths, -1) @ a_flat
    total_cost = cost.reshape(n_paths, -1) @ np.abs(a_flat)
    return gross, total_cost, gross - total_cost


def find_empirical_arbitrage(
    dC: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    max_abs_position: float = 1.0,
    payoff_tolerance: float = 1e-10,
    active_tolerance: float = 1e-8,
    method: str = "highs",
    presolve: bool = True,
    threads: int = 0,
    verbose: bool = False,
    solver_output: bool = False,
) -> ArbitrageResult:
    """Maximise mean net payoff subject to non-negative payoff on every path."""
    arr = _as_dC(dC)
    cfg = ArbitrageConfig(
        delta=delta,
        prop_cost=prop_cost,
        max_abs_position=max_abs_position,
        payoff_tolerance=payoff_tolerance,
        active_tolerance=active_tolerance,
        method=method,
        presolve=presolve,
        threads=threads,
    )
    started = time.perf_counter()
    _progress(
        f"preparing main LP: {arr.shape[0]:,} paths, "
        f"{arr.shape[1] * arr.shape[2]:,} positions",
        enabled=verbose,
    )
    cost, stderr = build_cost_tensor(arr, X, delta=delta, prop_cost=prop_cost)
    net, n_pos = _net_matrix(arr, cost)
    _progress(
        f"starting HiGHS main solve with "
        f"{'all available' if threads == 0 else threads} threads",
        enabled=verbose,
    )
    result = _solve_lp(
        c=-net.mean(axis=0),
        A_ub=-net,
        b_ub=np.full(arr.shape[0], payoff_tolerance),
        bounds=[(0.0, max_abs_position)] * (2 * n_pos),
        method=method,
        options=_highs_options(presolve, threads, solver_output),
    )
    _progress(
        f"main solve finished in {time.perf_counter() - started:.1f}s: "
        f"status={result.status}, iterations={getattr(result, 'nit', 'n/a')}",
        enabled=verbose,
    )
    if not result.success:
        return ArbitrageResult(
            False, False, result.message, cfg,
            mc_error=stderr,
            solver_status=int(result.status),
            solver_nit=getattr(result, "nit", None),
        )
    a_flat = result.x[:n_pos] - result.x[n_pos:]
    a = a_flat.reshape(arr.shape[1:])
    gross, total_cost, p = _evaluate_from_matrices(arr, cost, a_flat)
    tol = max(10.0 * payoff_tolerance, 1e-12 * max(1.0, float(np.mean(np.abs(p)))))
    found = bool(
        p.min() >= -tol
        and p.mean() > tol
        and np.any(np.abs(a_flat) > active_tolerance)
    )
    return ArbitrageResult(
        True, found, result.message, cfg,
        positions=a,
        payoff=p,
        gross_payoff=gross,
        total_cost=total_cost,
        minimum_payoff=float(p.min()),
        mean_payoff=float(p.mean()),
        maximum_payoff=float(p.max()),
        l1_position=float(np.abs(a_flat).sum()),
        max_abs_position=float(np.abs(a_flat).max()),
        num_active=int(np.count_nonzero(np.abs(a_flat) > active_tolerance)),
        fraction_negative=float(np.mean(p < -tol)),
        mc_error=stderr,
        solver_status=int(result.status),
        solver_nit=getattr(result, "nit", None),
    )


def sparsify_empirical_arbitrage(
    dC: ArrayLike,
    X: ArrayLike | None = None,
    *,
    required_mean_payoff: float,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    payoff_tolerance: float = 1e-10,
    active_tolerance: float = 1e-8,
    weights: ArrayLike | None = None,
    method: str = "highs",
    presolve: bool = True,
    threads: int = 0,
    verbose: bool = False,
    solver_output: bool = False,
) -> SparseResult:
    """Minimise weighted L1 exposure subject to an arbitrage certificate."""
    arr = _as_dC(dC)
    required_mean_payoff = float(required_mean_payoff)
    if not math.isfinite(required_mean_payoff) or required_mean_payoff <= 0:
        raise ValueError("required_mean_payoff must be finite and positive")
    started = time.perf_counter()
    _progress(
        f"preparing sparse LP for required mean payoff "
        f"{required_mean_payoff:.6g}",
        enabled=verbose,
    )
    cost, _ = build_cost_tensor(arr, X, delta=delta, prop_cost=prop_cost)
    net, n_pos = _net_matrix(arr, cost)
    if weights is None:
        w = np.ones(n_pos)
    else:
        w0 = np.asarray(weights, dtype=np.float64)
        try:
            w = np.broadcast_to(w0, arr.shape[1:]).reshape(-1).copy()
        except ValueError as exc:
            raise ValueError("weights are not broadcastable to (T,L)") from exc
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must be finite and positive")
    _progress("starting HiGHS sparse solve", enabled=verbose)
    result = _solve_lp(
        c=np.concatenate([w, w]),
        A_ub=np.concatenate([-net, -net.mean(axis=0, keepdims=True)], axis=0),
        b_ub=np.concatenate([
            np.full(arr.shape[0], payoff_tolerance),
            np.array([-required_mean_payoff]),
        ]),
        bounds=(0.0, None),
        method=method,
        options=_highs_options(presolve, threads, solver_output),
    )
    _progress(
        f"sparse solve finished in {time.perf_counter() - started:.1f}s: "
        f"status={result.status}, iterations={getattr(result, 'nit', 'n/a')}",
        enabled=verbose,
    )
    if not result.success:
        return SparseResult(
            False, result.message, required_mean_payoff,
            solver_status=int(result.status),
            solver_nit=getattr(result, "nit", None),
        )
    a_flat = result.x[:n_pos] - result.x[n_pos:]
    a = a_flat.reshape(arr.shape[1:])
    gross, total_cost, p = _evaluate_from_matrices(arr, cost, a_flat)
    return SparseResult(
        True, result.message, required_mean_payoff,
        positions=a,
        payoff=p,
        gross_payoff=gross,
        total_cost=total_cost,
        minimum_payoff=float(p.min()),
        mean_payoff=float(p.mean()),
        maximum_payoff=float(p.max()),
        l1_position=float(np.abs(a_flat).sum()),
        num_active=int(np.count_nonzero(np.abs(a_flat) > active_tolerance)),
        fraction_negative=float(np.mean(p < -payoff_tolerance)),
        solver_status=int(result.status),
        solver_nit=getattr(result, "nit", None),
    )


def iterative_reweighted_l1(
    dC: ArrayLike,
    X: ArrayLike | None = None,
    *,
    required_mean_payoff: float,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    num_iterations: int = 5,
    reweight_epsilon: float = 1e-6,
    payoff_tolerance: float = 1e-10,
    active_tolerance: float = 1e-8,
    threads: int = 0,
    verbose: bool = False,
    solver_output: bool = False,
) -> list[SparseResult]:
    """Iteratively reweighted L1 heuristic for fewer active positions."""
    arr = _as_dC(dC)
    weights = np.ones(arr.shape[1:])
    results: list[SparseResult] = []
    for _ in range(num_iterations):
        r = sparsify_empirical_arbitrage(
            arr, X,
            required_mean_payoff=required_mean_payoff,
            delta=delta,
            prop_cost=prop_cost,
            payoff_tolerance=payoff_tolerance,
            active_tolerance=active_tolerance,
            weights=weights,
            threads=threads,
            verbose=verbose,
            solver_output=solver_output,
        )
        results.append(r)
        if not r.solver_success or r.positions is None:
            break
        weights = 1.0 / (np.abs(r.positions) + reweight_epsilon)
        weights /= weights.mean()
    return results


def position_contributions(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
) -> dict[str, FloatArray]:
    arr = _as_dC(dC)
    a = np.asarray(positions, dtype=np.float64)
    cost, _ = build_cost_tensor(arr, X, delta=delta, prop_cost=prop_cost)
    gross = arr * a[None, :, :]
    cost_c = cost * np.abs(a)[None, :, :]
    net = gross - cost_c
    total = net.sum(axis=(1, 2))
    mean_net = net.mean(axis=0)
    return {
        "gross": gross,
        "cost": cost_c,
        "net": net,
        "total": total,
        "mean_net": mean_net,
        "std_net": net.std(axis=0, ddof=1),
        "cov_with_total": np.mean(
            (net - mean_net[None, :, :])
            * (total[:, None, None] - total.mean()),
            axis=0,
        ),
    }


def rank_positions(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    top_n: int = 100,
    active_tolerance: float = 1e-8,
) -> list[dict[str, float | int]]:
    c = position_contributions(dC, positions, X, delta=delta, prop_cost=prop_cost)
    a = np.asarray(positions, dtype=np.float64)
    order = np.argsort(np.abs(c["mean_net"]).ravel())[::-1]
    rows: list[dict[str, float | int]] = []
    for idx in order:
        t, ell = np.unravel_index(idx, a.shape)
        if abs(a[t, ell]) <= active_tolerance:
            continue
        rows.append({
            "time": int(t),
            "instrument": int(ell),
            "position": float(a[t, ell]),
            "mean_net_contribution": float(c["mean_net"][t, ell]),
            "std_net_contribution": float(c["std_net"][t, ell]),
            "covariance_with_total": float(c["cov_with_total"][t, ell]),
        })
        if len(rows) >= top_n:
            break
    return rows


def deletion_importance(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    active_tolerance: float = 1e-8,
) -> list[dict[str, float | int]]:
    c = position_contributions(dC, positions, X, delta=delta, prop_cost=prop_cost)
    a = np.asarray(positions, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for t, ell in np.argwhere(np.abs(a) > active_tolerance):
        without = c["total"] - c["net"][:, t, ell]
        rows.append({
            "time": int(t),
            "instrument": int(ell),
            "position": float(a[t, ell]),
            "minimum_without": float(without.min()),
            "mean_without": float(without.mean()),
            "damage_to_minimum": float(c["total"].min() - without.min()),
            "damage_to_mean": float(c["total"].mean() - without.mean()),
        })
    rows.sort(key=lambda r: float(r["minimum_without"]))
    return rows


def greedy_prune(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    minimum_mean_payoff: float,
    delta: float = 0.0,
    prop_cost: float = 0.0,
    payoff_tolerance: float = 1e-10,
    active_tolerance: float = 1e-8,
) -> FloatArray:
    a = np.asarray(positions, dtype=np.float64).copy()
    c = position_contributions(dC, a, X, delta=delta, prop_cost=prop_cost)
    payoff = c["total"].copy()
    active = [tuple(x) for x in np.argwhere(np.abs(a) > active_tolerance)]
    active.sort(key=lambda ij: abs(float(c["net"][:, ij[0], ij[1]].mean())))
    for t, ell in active:
        candidate = payoff - c["net"][:, t, ell]
        if candidate.min() >= -payoff_tolerance and candidate.mean() >= minimum_mean_payoff:
            payoff = candidate
            a[t, ell] = 0.0
    return a


def step_payoff_summary(
    dC: ArrayLike,
    positions: ArrayLike,
    X: ArrayLike | None = None,
    *,
    delta: float = 0.0,
    prop_cost: float = 0.0,
) -> list[dict[str, float | int]]:
    arr = _as_dC(dC)
    a = np.asarray(positions, dtype=np.float64)
    cost, _ = build_cost_tensor(arr, X, delta=delta, prop_cost=prop_cost)
    step = np.sum(arr * a[None, :, :] - cost * np.abs(a)[None, :, :], axis=2)
    rows: list[dict[str, float | int]] = []
    for t in range(step.shape[1]):
        x = step[:, t]
        rows.append({
            "time": int(t),
            "minimum": float(x.min()),
            "q001": float(np.quantile(x, 0.001)),
            "q01": float(np.quantile(x, 0.01)),
            "mean": float(x.mean()),
            "maximum": float(x.max()),
            "fraction_negative": float(np.mean(x < 0)),
        })
    return rows


def binding_paths(payoff: ArrayLike, count: int = 100) -> NDArray[np.int64]:
    x = np.asarray(payoff, dtype=np.float64)
    return np.argsort(x)[: min(count, len(x))]


def compare_rays(results: Sequence[SparseResult]) -> list[dict[str, float]]:
    valid = [r for r in results if r.solver_success and r.positions is not None]
    if len(valid) < 2:
        return []
    ref = valid[0].positions
    assert ref is not None
    ref = ref / np.sum(np.abs(ref))
    rows: list[dict[str, float]] = []
    for i, r in enumerate(valid[1:], 1):
        assert r.positions is not None
        a = r.positions / np.sum(np.abs(r.positions))
        rows.append({
            "reference_index": 0.0,
            "comparison_index": float(i),
            "max_abs_normalised_difference": float(np.max(np.abs(ref - a))),
        })
    return rows


def _save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_experiment(
    output_dir: str | Path,
    name: str,
    dC: ArrayLike,
    X: ArrayLike | None,
    result: ArbitrageResult,
    sparse: Sequence[SparseResult],
    *,
    delta: float,
    prop_cost: float,
) -> None:
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "arbitrage": result.summary(),
        "sparse": [r.summary() for r in sparse],
        "ray_comparison": compare_rays(sparse),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if result.positions is None or result.payoff is None:
        return
    np.save(out / "positions.npy", result.positions)
    np.save(out / "payoff.npy", result.payoff)
    if result.mc_error is not None:
        np.save(out / "mc_error.npy", result.mc_error)
    np.save(out / "binding_path_indices.npy", binding_paths(result.payoff, 200))
    _save_csv(out / "ranked_positions.csv", rank_positions(
        dC, result.positions, X, delta=delta, prop_cost=prop_cost, top_n=200
    ))
    _save_csv(out / "deletion_importance.csv", deletion_importance(
        dC, result.positions, X, delta=delta, prop_cost=prop_cost
    ))
    _save_csv(out / "step_payoff_summary.csv", step_payoff_summary(
        dC, result.positions, X, delta=delta, prop_cost=prop_cost
    ))
    for i, r in enumerate(sparse):
        if r.positions is not None:
            np.save(out / f"sparse_{i}_positions.npy", r.positions)
        if r.payoff is not None:
            np.save(out / f"sparse_{i}_payoff.npy", r.payoff)


def run_standard_experiments(
    dC: ArrayLike,
    X: ArrayLike | None,
    *,
    output_dir: str | Path = "empirical_arbitrage_results",
    sparse_fractions: Sequence[float] = (0.5, 0.25, 0.1),
    payoff_tolerance: float = 1e-10,
    active_tolerance: float = 1e-8,
    max_abs_position: float = 1.0,
    threads: int = 0,
    verbose: bool = True,
    solver_output: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run the three requested experiments and save their diagnostics."""
    experiments = [
        ("delta0_prop0", 0.0, 0.0),
        ("delta1_prop0", 1.0, 0.0),
        ("delta0_prop1bp", 0.0, 0.0001),
    ]
    all_results: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()
    _progress(
        f"starting {len(experiments)} experiments; output directory: {output_dir}",
        enabled=verbose,
    )
    for experiment_number, (name, delta, prop_cost) in enumerate(experiments, 1):
        _progress(
            f"experiment {experiment_number}/{len(experiments)} {name}: "
            f"delta={delta:g}, prop_cost={prop_cost:g}",
            enabled=verbose,
        )
        result = find_empirical_arbitrage(
            dC, X,
            delta=delta,
            prop_cost=prop_cost,
            max_abs_position=max_abs_position,
            payoff_tolerance=payoff_tolerance,
            active_tolerance=active_tolerance,
            threads=threads,
            verbose=verbose,
            solver_output=solver_output,
        )
        sparse: list[SparseResult] = []
        if result.solver_success and result.found and result.mean_payoff is not None:
            for fraction in sparse_fractions:
                _progress(
                    f"{name}: sparse certificate at {fraction:.0%} of main payoff",
                    enabled=verbose,
                )
                sparse.append(sparsify_empirical_arbitrage(
                    dC, X,
                    required_mean_payoff=fraction * result.mean_payoff,
                    delta=delta,
                    prop_cost=prop_cost,
                    payoff_tolerance=payoff_tolerance,
                    active_tolerance=active_tolerance,
                    threads=threads,
                    verbose=verbose,
                    solver_output=solver_output,
                ))
        else:
            _progress(
                f"{name}: no certificate found; skipping sparse solves",
                enabled=verbose,
            )
        _progress(f"{name}: computing and saving diagnostics", enabled=verbose)
        save_experiment(
            output_dir, name, dC, X, result, sparse,
            delta=delta, prop_cost=prop_cost,
        )
        all_results[name] = {
            "arbitrage": result.summary(),
            "sparse": [r.summary() for r in sparse],
        }
        _progress(f"experiment {experiment_number}/{len(experiments)} complete", enabled=verbose)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "all_experiments.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    _progress(
        f"all experiments complete in {time.perf_counter() - total_started:.1f}s",
        enabled=verbose,
    )
    return all_results


def _load_npz(path: str | Path) -> tuple[FloatArray, FloatArray | None]:
    data = np.load(path)
    if "dC" not in data:
        raise KeyError("NPZ file must contain 'dC'")
    return data["dC"], data["X"] if "X" in data else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Empirical arbitrage diagnostics")
    parser.add_argument("input_npz", help="NPZ containing dC and optionally X")
    parser.add_argument("--output-dir", default="empirical_arbitrage_results")
    parser.add_argument("--payoff-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-abs-position", type=float, default=1.0)
    parser.add_argument(
        "--threads", type=int, default=0,
        help="HiGHS worker threads (0 uses all available cores)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress and HiGHS output"
    )
    args = parser.parse_args(argv)
    dC, X = _load_npz(args.input_npz)
    result = run_standard_experiments(
        dC, X,
        output_dir=args.output_dir,
        payoff_tolerance=args.payoff_tolerance,
        max_abs_position=args.max_abs_position,
        threads=args.threads,
        verbose=not args.quiet,
        solver_output=not args.quiet,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
