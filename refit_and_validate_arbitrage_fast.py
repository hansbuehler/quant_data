#!/usr/bin/env python3
"""Memory-efficient LP refit on 40k paths and frozen validation on 400k."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linprog


EXPERIMENTS = {
    "delta0_prop0": (0.0, 0.0),
    "delta1_prop0": (1.0, 0.0),
    "delta0_prop1bp": (0.0, 1e-4),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def net_payoff(dx: np.ndarray, xx: np.ndarray, beta: np.ndarray,
               mc: np.ndarray, delta: float, prop: float) -> np.ndarray:
    out = dx @ beta
    ab = np.abs(beta)
    if delta:
        out -= delta * float(mc @ ab)
    if prop:
        out -= prop * (xx @ ab)
    return out


def constraint_rows(dx: np.ndarray, xx: np.ndarray, indices: np.ndarray,
                    mc: np.ndarray, delta: float, prop: float) -> np.ndarray:
    d = dx[indices]
    cost = np.zeros_like(d)
    if delta:
        cost += delta * mc
    if prop:
        cost += prop * xx[indices]
    # payoff = (d-cost) beta+ + (-d-cost) beta-, and linprog wants A_ub z <= b.
    return np.concatenate((cost - d, cost + d), axis=1)


def solve_cutting_plane(dx: np.ndarray, xx: np.ndarray, mc: np.ndarray,
                        delta: float, prop: float, tolerance: float,
                        batch: int, max_iterations: int) -> tuple[np.ndarray, dict[str, object]]:
    n, p = dx.shape
    mean_dx = dx.mean(axis=0)
    mean_cost = delta * mc + prop * xx.mean(axis=0)
    objective = np.concatenate((mean_cost - mean_dx, mean_cost + mean_dx))
    # A deterministic spread of paths gives the first relaxation useful coverage.
    active = np.unique(np.linspace(0, n - 1, min(batch, n), dtype=np.int64))
    started = time.perf_counter()
    result = None
    payoff = np.empty(n)
    # HiGHS' documented minimum primal feasibility tolerance is 1e-10.
    verification_slack = 2e-10
    for iteration in range(1, max_iterations + 1):
        rows = constraint_rows(dx, xx, active, mc, delta, prop)
        result = linprog(objective, A_ub=rows, b_ub=np.full(active.size, tolerance),
                         bounds=(0.0, 1.0), method="highs",
                         options={"presolve": True, "dual_feasibility_tolerance": 1e-10,
                                  "primal_feasibility_tolerance": 1e-10})
        if not result.success:
            raise RuntimeError(f"LP failed at iteration {iteration}: {result.message}")
        beta = result.x[:p] - result.x[p:]
        payoff = net_payoff(dx, xx, beta, mc, delta, prop)
        violated = np.flatnonzero(payoff < -tolerance - verification_slack)
        log(f"iteration {iteration}: active={active.size:,}, mean={payoff.mean():.9g}, "
            f"min={payoff.min():.9g}, violations={violated.size:,}")
        if violated.size == 0:
            break
        inactive_violated = np.setdiff1d(violated, active, assume_unique=False)
        if inactive_violated.size == 0:
            raise RuntimeError("Solver returned violations only among active constraints")
        take = inactive_violated[np.argsort(payoff[inactive_violated])[:batch]]
        active = np.unique(np.concatenate((active, take)))
    else:
        raise RuntimeError(f"Cutting-plane solve incomplete after {max_iterations} iterations")
    assert result is not None
    meta: dict[str, object] = {
        "solver_success": True, "solver_message": result.message,
        "solver_status": int(result.status), "solver_nit_last": int(result.nit),
        "iterations": iteration, "active_constraints": int(active.size),
        "verification_slack": verification_slack,
        "worst_remaining_violation": float(max(0.0, -tolerance - payoff.min())),
        "violated_paths": int(np.sum(payoff < -tolerance - verification_slack)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return beta, meta


def stats(x: np.ndarray, tolerance: float) -> dict[str, object]:
    sd = float(x.std(ddof=1)); se = sd / math.sqrt(x.size); mean = float(x.mean())
    return {
        "n": int(x.size), "min": float(x.min()), "mean": mean, "max": float(x.max()),
        "std": sd, "sem": se, "negative_count": int(np.sum(x < -tolerance)),
        "negative_fraction": float(np.mean(x < -tolerance)),
        "q0.001": float(np.quantile(x, .001)), "q0.01": float(np.quantile(x, .01)),
        "median": float(np.median(x)), "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "pathwise_certificate": bool(x.min() >= -tolerance and mean > tolerance),
    }


def validate_chunked(path: Path, beta: np.ndarray, mc: np.ndarray,
                     delta: float, prop: float, chunk: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        dd, xx = h5["root/dX"], h5["root/X"]
        if dd.shape[1] * dd.shape[2] != beta.size or xx.shape != dd.shape:
            raise ValueError(f"Validation shape mismatch: dX={dd.shape}, X={xx.shape}, beta={beta.shape}")
        out = np.empty(dd.shape[0], dtype=np.float64)
        for start in range(0, dd.shape[0], chunk):
            stop = min(start + chunk, dd.shape[0])
            d = np.asarray(dd[start:stop], dtype=np.float64).reshape(stop-start, -1)
            x = np.abs(np.asarray(xx[start:stop], dtype=np.float64)).reshape(stop-start, -1)
            out[start:stop] = net_payoff(d, x, beta, mc, delta, prop)
            if start == 0 or stop == dd.shape[0] or stop % 100000 == 0:
                log(f"validation progress: {stop:,}/{dd.shape[0]:,} paths")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-h5", type=Path, required=True)
    ap.add_argument("--validation-h5", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("refit_arbitrage_validation"))
    ap.add_argument("--tolerance", type=float, default=1e-10)
    ap.add_argument("--constraint-batch", type=int, default=1000)
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--validation-chunk", type=int, default=10000)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log("loading 40,000-path training arrays")
    with h5py.File(args.training_h5, "r") as h5:
        shape = h5["root/dX"].shape
        dx = np.asarray(h5["root/dX"], dtype=np.float64).reshape(shape[0], -1)
        xx = np.abs(np.asarray(h5["root/X"], dtype=np.float32)).reshape(shape[0], -1)
    if not np.isfinite(dx).all() or not np.isfinite(xx).all():
        raise ValueError("Training arrays contain non-finite values")
    mc = dx.std(axis=0, ddof=1) / math.sqrt(dx.shape[0])
    np.save(args.output_dir / "training_mc_error.npy", mc.reshape(shape[1:]))
    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"training_shape": shape, "tolerance": args.tolerance, "experiments": {}}
    for name, (delta, prop) in EXPERIMENTS.items():
        log(f"starting {name}: delta={delta:g}, prop_cost={prop:g}")
        position_path = args.output_dir / f"{name}_positions.npy"
        if name in summary.get("experiments", {}) and position_path.is_file():
            log(f"{name} already completed; loading its frozen beta")
            beta = np.load(position_path).reshape(-1)
            train = net_payoff(dx, xx, beta, mc, delta, prop)
            solver = summary["experiments"][name]["solver"]
        else:
            beta, solver = solve_cutting_plane(dx, xx, mc, delta, prop, args.tolerance,
                                                args.constraint_batch, args.max_iterations)
            train = net_payoff(dx, xx, beta, mc, delta, prop)
            np.save(position_path, beta.reshape(shape[1:]))
            np.save(args.output_dir / f"{name}_training_payoff.npy", train)
        log(f"{name} converged; freezing beta and evaluating 400,000 paths")
        validation = validate_chunked(args.validation_h5, beta, mc, delta, prop,
                                      args.validation_chunk)
        np.save(args.output_dir / f"{name}_validation_payoff.npy", validation)
        entry = {"cost": {"delta": delta, "prop_cost": prop}, "solver": solver,
                 "training": stats(train, args.tolerance),
                 "validation_frozen": stats(validation, args.tolerance),
                 "l1_beta": float(np.abs(beta).sum()), "max_abs_beta": float(np.abs(beta).max())}
        summary["experiments"][name] = entry  # type: ignore[index]
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"finished {name}: validation min={validation.min():.9g}, mean={validation.mean():.9g}, "
            f"negative={np.sum(validation < -args.tolerance):,}")
    log(f"all experiments complete; results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
