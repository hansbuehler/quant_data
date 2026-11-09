"""Full-panel JAX L-BFGS calibration of the affine DYSANOS surface."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.v11.jax_affine200 import load_quotes, make_functions
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("completion", "panel", "archive", "initialization", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--resume-directory", type=Path)
    parser.add_argument("--initial-backbone-smoothing", type=float, default=0.1)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--mode", choices=("variance", "volatility", "both"), default="both")
    parser.add_argument("--model-business-dtes", type=int, nargs="+")
    parser.add_argument("--direct", action="store_true", help="Fit each date's full state without affine compression")
    parser.add_argument("--allow-zero-backbone", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("A managed GPU allocation is required")
    if args.precision == 64 and not jax.config.jax_enable_x64:
        raise RuntimeError("Set JAX_ENABLE_X64=true for 64-bit calibration")
    dtype = jnp.float64 if args.precision == 64 else jnp.float32
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=args.allow_zero_backbone)
    with np.load(args.initialization / "states.npz") as saved:
        if "model_business_dtes" in saved and not np.array_equal(saved["model_business_dtes"], geometry.model_business_dtes):
            raise ValueError("Initialization and model expiry grids differ")
        dates = tuple(date.fromisoformat(str(item)) for item in saved["dates"])
        raw = saved["raw_states"].astype(np.float64 if args.precision == 64 else np.float32)
    if len(dates) != 250 or len(dates) % args.batch:
        raise ValueError("Expected 250 dates divisible by the batch size")
    recovered_raw = raw.copy() if args.direct else None
    initial_eta = float(np.sqrt(args.initial_backbone_smoothing))
    eta_probability = (initial_eta-geometry.min_eta)/(geometry.max_eta-geometry.min_eta)
    if not 0. < eta_probability < 1.:
        raise ValueError("Initial backbone smoothing lies outside the DYSANOS eta bounds")
    raw[:, -2:] = np.log(eta_probability/(1.-eta_probability))
    print(f"initial_backbone_smoothing={args.initial_backbone_smoothing} eta={initial_eta:.8g}", flush=True)
    quotes = load_quotes(dates, args.panel, args.archive, geometry)
    batches = []
    for offset in range(0, len(dates), args.batch):
        indices = np.arange(offset, offset + args.batch)
        batches.append((jnp.asarray(indices), {key: jnp.asarray(value[indices]) for key, value in quotes.items()}))
    if args.direct:
        initial = {"states": jnp.asarray(raw)}
    else:
        b = raw.mean(axis=0)
        left, singular, right = np.linalg.svd(raw-b, full_matrices=False)
        initial = {"b": jnp.asarray(b), "B": jnp.asarray(right[:200]), "h": jnp.asarray(left[:, :200]*singular[:200])}
    flattened, unravel = ravel_pytree(initial)
    for interpolation in (("variance", "volatility") if args.mode == "both" else (args.mode,)):
        started = time.monotonic()
        done = threading.Event()
        progress = {"iteration": 0, "evaluations": 0}
        def heartbeat():
            while not done.wait(60):
                elapsed = time.monotonic()-started
                completed = progress["iteration"]
                remaining = (args.max_iterations-completed)*elapsed/completed if completed else None
                print(f"heartbeat interpolation={interpolation} iteration={completed}/{args.max_iterations} evaluations={progress['evaluations']} elapsed={elapsed:.1f}s remaining_est={remaining}", flush=True)
        threading.Thread(target=heartbeat, daemon=True).start()
        grad_fn, price_fn = make_functions(geometry, interpolation, direct=args.direct)
        history = []
        last = {}
        def value_gradient(vector):
            parameters = unravel(jnp.asarray(vector, dtype=dtype))
            gradient = jax.tree.map(jnp.zeros_like, parameters)
            batch_values = []
            for indices, block in batches:
                (value, _), part = grad_fn(parameters, block, indices)
                batch_values.append(value)
                gradient = jax.tree.map(lambda total, change: total + change/len(batches), gradient, part)
            gradient_flat, _ = ravel_pytree(gradient)
            value = float(np.mean(np.asarray(jax.device_get(jnp.stack(batch_values)), dtype=np.float64)))
            gradient_host = np.asarray(gradient_flat, dtype=np.float64)
            if not np.isfinite(value) or not np.isfinite(gradient_host).all():
                raise RuntimeError("Non-finite full-panel objective or gradient")
            progress["evaluations"] += 1
            last.update(value=value, gradient_rms=float(np.sqrt(np.mean(gradient_host**2))))
            return value, gradient_host
        def assess(parameters, predict=price_fn):
            inside = total = 0
            for indices, block in batches:
                raw_batch = parameters["states"][indices] if args.direct else parameters["b"][None, :] + parameters["h"][indices] @ parameters["B"]
                price, _ = predict(raw_batch, block)
                match = block["mask"] & (price >= block["bids"]) & (price <= block["asks"])
                inside += int(jnp.sum(match))
                total += int(jnp.sum(block["mask"]))
            return 100.*inside/total, total
        def accepted(vector):
            progress["iteration"] += 1
            iteration = progress["iteration"]
            parameters = unravel(jnp.asarray(vector, dtype=dtype))
            record = {"iteration": iteration, "objective": last["value"], "gradient_rms": last["gradient_rms"], "elapsed_seconds": time.monotonic()-started}
            if iteration == 1 or iteration % 10 == 0:
                record["inside_percent"], _ = assess(parameters)
                temporary = args.output / f"{interpolation}-checkpoint.tmp.npz"
                np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in parameters.items()}, iteration=iteration)
                temporary.replace(args.output / f"{interpolation}-checkpoint.npz")
                print(json.dumps({"interpolation": interpolation, **record}), flush=True)
            history.append(record)
        start_vector = np.asarray(flattened, dtype=np.float64)
        if args.resume_directory is not None:
            with np.load(args.resume_directory / f"{interpolation}-final.npz") as saved:
                saved_parameters = {key: jnp.asarray(saved[key], dtype=dtype) for key in initial}
            start_vector = np.asarray(ravel_pytree(saved_parameters)[0], dtype=np.float64)
        start_params = unravel(jnp.asarray(start_vector, dtype=dtype))
        if recovered_raw is not None and args.resume_directory is None:
            recovered_params = {"states": jnp.asarray(recovered_raw, dtype=dtype)}
            _, zero_backbone_price = make_functions(geometry, interpolation, direct=True, zero_backbone=True)
            recovered_inside, recovered_count = assess(recovered_params, zero_backbone_price)
            print(json.dumps({"interpolation": interpolation, "stage": "recovered_dlv", "inside_percent": recovered_inside,
                              "quotes": recovered_count, "backbone": 0.}), flush=True)
        start_inside, start_count = assess(start_params)
        start_objective, _ = value_gradient(start_vector)
        print(json.dumps({"interpolation": interpolation, "stage": "initial", "inside_percent": start_inside,
                          "quotes": start_count, "objective": start_objective}), flush=True)
        result = minimize(value_gradient, start_vector, method="L-BFGS-B", jac=True,
                          callback=accepted, options={"maxiter": args.max_iterations, "maxcor": 10, "maxls": 30, "ftol": 1e-10, "gtol": 1e-6})
        if result.fun > start_objective:
            result.x, result.fun = start_vector, start_objective
            result.message = f"{result.message}; restored initial objective"
        parameters = unravel(jnp.asarray(result.x, dtype=dtype))
        inside, count = assess(parameters)
        np.savez_compressed(args.output / f"{interpolation}-final.npz", **{k: np.asarray(v) for k, v in parameters.items()}, iteration=result.nit)
        report = {"interpolation": interpolation, "precision": args.precision, "iterations": int(result.nit), "evaluations": int(result.nfev), "objective": float(result.fun),
                  "inside_percent": inside, "quotes": count, "success": bool(result.success), "message": str(result.message),
                  "elapsed_seconds": time.monotonic()-started, "history": history}
        (args.output / f"{interpolation}-report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != "history"}), flush=True)
        done.set()


if __name__ == "__main__":
    main()
