"""Historical weekly AR(1) pilot; new daily fits use ``rolling_daily``.

Retained to reproduce the earlier joint affine-200 calibration, not as the
current DYSANOS dynamics convention.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.linalg import expm
from scipy.optimize import minimize

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.weekly_validation import exact_weekly_pairs
from quantdata.dysanos.v11.jax_affine200 import load_quotes, make_functions
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry


def initialize_joint(affine: dict[str, np.ndarray], origins: np.ndarray, endpoints: np.ndarray) -> dict[str, np.ndarray]:
    """Whiten factors without changing surfaces, and initialize a stable AR."""
    states = affine["h"]
    mean = states.mean(axis=0)
    centered = states - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    scales = singular / np.sqrt(len(states))
    if np.min(scales) < 1e-8:
        raise ValueError("Affine factors are rank deficient; cannot whiten")
    whitened = centered @ right.T / scales
    loading = (scales[:, None] * right) @ affine["B"]
    bias = affine["b"] + mean @ affine["B"]
    if not np.allclose(bias + whitened @ loading, affine["b"] + states @ affine["B"], atol=1e-8, rtol=1e-10):
        raise ValueError("Whitening changed the decoded states")
    # Stable diagonal initialization; the subsequent joint fit learns a full generator.
    origin, endpoint = whitened[origins], whitened[endpoints]
    slope = np.sum(origin * endpoint, axis=0) / (np.sum(origin * origin, axis=0) + 1e-8)
    slope = np.clip(slope, .05, .995)
    residual = endpoint - origin * slope
    covariance = residual.T @ residual / len(residual)
    covariance = .8 * covariance + .2 * np.diag(np.diag(covariance)) + 1e-6 * np.eye(200)
    factor = np.linalg.cholesky(covariance)
    diag = np.maximum(np.diag(factor) - 1e-4, 1e-8)
    q_raw = np.tril(factor, -1) + np.diag(diag + np.log(-np.expm1(-diag)))
    damping = np.diag(np.sqrt(np.maximum(-np.log(slope) - .001, 1e-6)))
    return {"b": bias, "B": loading, "h": whitened, "damping": damping,
            "rotation": np.zeros((200, 200)), "mean": np.zeros(200), "q_raw": q_raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("completion", "panel", "archive", "initialization", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-business-dtes", nargs="+", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--previous-initialization", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "gpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("Requires an allocated GPU and JAX_ENABLE_X64=true")
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=True)
    with np.load(args.initialization / "states.npz") as saved:
        dates = tuple(date.fromisoformat(str(day)) for day in saved["dates"])
        if not np.array_equal(saved["model_business_dtes"], geometry.model_business_dtes):
            raise ValueError("Initialization expiry grid mismatch")
    if len(dates) != 250 or len(dates) % args.batch:
        raise ValueError("Expected 250 dates divisible by batch size")
    origins, endpoints = exact_weekly_pairs(dates, weeks=1)
    with np.load(args.source) as saved:
        if args.previous_initialization is None:
            initial = initialize_joint({key: saved[key] for key in ("b", "B", "h")}, origins, endpoints)
        else:
            initial = {key: saved[key].copy() for key in ("b", "B", "h", "damping", "rotation", "mean", "q_raw")}
            with np.load(args.previous_initialization / "states.npz") as previous:
                previous_dates = tuple(date.fromisoformat(str(day)) for day in previous["dates"])
                if not np.array_equal(previous["model_business_dtes"], geometry.model_business_dtes):
                    raise ValueError("Previous rolling geometry differs")
            if dates[-1] <= previous_dates[-1] or any(day not in previous_dates for day in dates if day <= previous_dates[-1]):
                raise ValueError("Rolling dates must overlap and advance causally")
            prev_states = dict(zip(previous_dates, initial["h"], strict=True))
            lower = np.tril(initial["damping"])
            rotation = .5 * (initial["rotation"] - initial["rotation"].T)
            daily = expm(-(lower @ lower.T + rotation + .001 * np.eye(200)) / 5.)
            state = prev_states[previous_dates[-1]]
            rows = []
            for day in dates:
                if day in prev_states:
                    rows.append(prev_states[day])
                else:
                    state = initial["mean"] + (state - initial["mean"]) @ daily
                    rows.append(state.copy())
            initial["h"] = np.asarray(rows)
            print(f"rolling_initialization_date={previous_dates[-1]} new_dates={sum(day > previous_dates[-1] for day in dates)}", flush=True)
    print(f"weekly_pairs={len(origins)} dates={len(dates)} interpolation=volatility", flush=True)
    quotes = load_quotes(dates, args.panel, args.archive, geometry)
    if args.previous_initialization is not None:
        # Daily filtering is causal: freeze the previous decoder and transition,
        # fit today's factor to today's quotes, then forecast from that factor.
        _, day_price = make_functions(geometry, "volatility")
        fixed_b, fixed_B = jnp.asarray(initial["b"]), jnp.asarray(initial["B"])
        def state_loss(h, block):
            price, cap = day_price((fixed_b + h @ fixed_B)[None, :], block)
            half = jnp.maximum(.5 * (block["asks"] - block["bids"]), 1e-12)
            excess = (jax.nn.relu(price - block["asks"]) + jax.nn.relu(block["bids"] - price)
                      + .05 * jnp.abs(price - block["mids"])) / half
            return jnp.sum(excess * block["mask"]) / jnp.sum(block["mask"]) + cap
        day_gradient = jax.jit(jax.value_and_grad(state_loss))
        day_results = []
        state = prev_states[previous_dates[-1]].copy()
        for index, day in enumerate(dates):
            if day <= previous_dates[-1]:
                continue
            start = initial["mean"] + (state - initial["mean"]) @ daily
            block = {name: jnp.asarray(value[index:index + 1]) for name, value in quotes.items()}
            def objective(h):
                value, gradient = day_gradient(jnp.asarray(h), block)
                return float(value), np.asarray(gradient, dtype=np.float64)
            result = minimize(objective, start, jac=True, method="L-BFGS-B",
                              options={"maxiter": 1500, "maxcor": 10, "ftol": 1e-10, "gtol": 1e-6})
            state = result.x
            initial["h"][index] = state
            fitted, _ = day_price((fixed_b + jnp.asarray(state) @ fixed_B)[None, :], block)
            valid = block["mask"]
            inside = float(100 * jnp.sum(valid & (fitted >= block["bids"]) & (fitted <= block["asks"])) / jnp.sum(valid))
            record = {"date": str(day), "inside_percent": inside, "objective": float(result.fun),
                      "iterations": int(result.nit), "success": bool(result.success), "message": str(result.message)}
            day_results.append(record)
            print(json.dumps({"stage": "daily_state_fit", **record}), flush=True)
        np.savez_compressed(args.output / "daily-state-initialization.npz", **initial,
                            dates=np.asarray([str(day) for day in dates]), iteration=0)
        (args.output / "daily-state-fits.json").write_text(json.dumps(day_results, indent=2))
    batches = [(jnp.arange(i, i + args.batch), {key: jnp.asarray(value[i:i + args.batch]) for key, value in quotes.items()})
               for i in range(0, len(dates), args.batch)]
    initial = jax.tree.map(jnp.asarray, initial)
    flat, unravel = ravel_pytree(initial)
    grad_fn, price_fn = make_functions(geometry, "volatility")
    origin_j, endpoint_j = jnp.asarray(origins), jnp.asarray(endpoints)

    def dynamics(params):
        damping = jnp.tril(params["damping"])
        rotation = .5 * (params["rotation"] - params["rotation"].T)
        generator = damping @ damping.T + rotation + .001 * jnp.eye(200)
        transition = jsp.linalg.expm(-generator)
        states = params["h"]
        residual = states[endpoint_j] - (params["mean"] + (states[origin_j] - params["mean"]) @ transition)
        q_raw = params["q_raw"]
        factor = jnp.tril(q_raw, -1) + jnp.diag(1e-4 + jax.nn.softplus(jnp.diag(q_raw)))
        white = jsp.linalg.solve_triangular(factor, residual.T, lower=True)
        nll = .5 * (jnp.mean(jnp.sum(white**2, axis=0)) + 2 * jnp.sum(jnp.log(jnp.diag(factor))) + 200 * np.log(2 * np.pi)) / 200
        centered = states - jnp.mean(states, axis=0)
        cov = centered.T @ centered / states.shape[0]
        gauge = jnp.mean(jnp.mean(states, axis=0)**2) + jnp.sum((cov - jnp.eye(200))**2) / 200
        return .1 * nll + .1 * gauge, (nll, gauge)

    dynamics_grad = jax.jit(jax.value_and_grad(dynamics, has_aux=True))
    progress = {"iteration": 0, "evaluations": 0}
    started = time.monotonic()
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(60):
            elapsed = time.monotonic() - started
            i = progress["iteration"]
            remaining = (args.iterations - i) * elapsed / i if i else None
            print(f"heartbeat iterations={i}/{args.iterations} elapsed={elapsed:.1f}s remaining_est={remaining}", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    last, history = {}, []
    def value_gradient(vector):
        params = unravel(jnp.asarray(vector))
        gradient = jax.tree.map(jnp.zeros_like, params)
        quote_values = []
        for indices, block in batches:
            (value, _), partial = grad_fn(params, block, indices)
            quote_values.append(value)
            gradient = jax.tree.map(lambda a, b: a + b / len(batches), gradient, partial)
        (dyn_value, (nll, gauge)), dyn_gradient = dynamics_grad(params)
        gradient = jax.tree.map(lambda a, b: a + b, gradient, dyn_gradient)
        full = float(jnp.mean(jnp.stack(quote_values)) + dyn_value)
        grad = np.asarray(ravel_pytree(gradient)[0], dtype=np.float64)
        if not np.isfinite(full) or not np.isfinite(grad).all():
            raise RuntimeError("Non-finite joint objective or gradient")
        progress["evaluations"] += 1
        last.update(objective=full, quote_and_cap=float(jnp.mean(jnp.stack(quote_values))), nll=float(nll), gauge=float(gauge), gradient_rms=float(np.sqrt(np.mean(grad**2))))
        return full, grad
    def assess(params, *, new_only=False):
        inside = total = 0
        for indices, block in batches:
            raw = params["b"][None, :] + params["h"][indices] @ params["B"]
            price, _ = price_fn(raw, block)
            mask = block["mask"]
            if new_only:
                mask = mask & (indices[:, None] >= len(dates) - new_count)
            inside += int(jnp.sum(mask & (price >= block["bids"]) & (price <= block["asks"])))
            total += int(jnp.sum(mask))
        return 100 * inside / total
    def accepted(vector):
        progress["iteration"] += 1
        i = progress["iteration"]
        if i == 1 or i % 10 == 0:
            params = unravel(jnp.asarray(vector))
            record = {"iteration": i, **last, "inside_percent": assess(params), "elapsed_seconds": time.monotonic()-started}
            temp = args.output / "checkpoint.tmp.npz"
            np.savez_compressed(temp, **{key: np.asarray(value) for key, value in params.items()}, iteration=i)
            temp.replace(args.output / "checkpoint.npz")
            history.append(record)
            print(json.dumps(record), flush=True)
    new_count = sum(day > previous_dates[-1] for day in dates) if args.previous_initialization is not None else 0
    initial_report = {"stage": "initial", "inside_percent": assess(initial), "weekly_pairs": len(origins)}
    if new_count:
        initial_report["new_dates_daily_fitted_inside_percent"] = assess(initial, new_only=True)
    print(json.dumps(initial_report), flush=True)
    result = minimize(value_gradient, np.asarray(flat), method="L-BFGS-B", jac=True, callback=accepted,
                      options={"maxiter": args.iterations, "maxcor": 10, "maxls": 30, "ftol": 1e-10, "gtol": 1e-6})
    params = unravel(jnp.asarray(result.x))
    np.savez_compressed(args.output / "final.npz", **{key: np.asarray(value) for key, value in params.items()}, iteration=result.nit)
    report = {"iterations": int(result.nit), "objective": float(result.fun), "inside_percent": assess(params),
              "success": bool(result.success), "message": str(result.message), "weekly_pairs": len(origins), "history": history}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    finished.set()


if __name__ == "__main__":
    main()
