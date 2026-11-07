"""Persistent GPU daily rolling affine DYSANOS and one-session AR(1) sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd
from cdxcore.version import version
from filelock import FileLock
from jax.flatten_util import ravel_pytree
from scipy.linalg import expm
from scipy.optimize import minimize

from quantdata.biz_calendar import get_business_days
from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.v11.audit_model_grid import grid_prices
from quantdata.dysanos.v11.jax_affine200 import load_quotes, make_functions
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.licensed_market import load_certified_pricer
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry

SCHEMA = "dysanos-daily-ar1-rolling-v1"
MODEL_DTES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25)


@version("1.0")
@dataclass(frozen=True, kw_only=True)
class RollingConfig:
    """Statistical settings; all transition rates are per business session."""

    label: str = "daily-ar1-affine200-25d"
    user_version: str = "1.0"
    window: int = 250
    state_size: int = 200
    oldest_weight: float = .25
    dynamics_weight: float = .1
    gauge_weight: float = .1
    minimum_daily_rate: float = .001
    innovation_std_floor: float = 1e-4
    batch_size: int = 10
    max_iterations: int = 1000
    state_iterations: int = 1500

    def __post_init__(self) -> None:
        if self.window <= self.state_size or self.window % self.batch_size:
            raise ValueError("Window must exceed state dimension and divide into complete batches")
        if not 0 < self.oldest_weight <= 1:
            raise ValueError("Oldest date weight must be in (0,1]")


def date_weights(count: int, oldest_weight: float = .25) -> np.ndarray:
    """Chronological weights, oldest first and newest exactly one."""
    if count < 2 or not 0 < oldest_weight <= 1:
        raise ValueError("Need at least two dates and a positive weight no larger than one")
    return oldest_weight ** (np.arange(count - 1, -1, -1, dtype=np.float64)/(count - 1))


class DeviceWindow:
    """Keep quote days in stable device slots; replace one slot at each roll."""

    def __init__(self, dates: tuple[date, ...], loader: Callable, batch_size: int):
        self.loader, self.batch_size = loader, batch_size
        loaded = [loader(day) for day in dates]
        self.capacity = int(np.ceil(max(x["mask"].shape[1] for x in loaded)/1024)*1024)
        self.slot_dates = list(dates)
        self.order = np.arange(len(dates), dtype=np.int32)
        self.days = [self._put(block) for block in loaded]
        self.batches = [self._batch(i) for i in range(0, len(dates), batch_size)]
        self.transferred_days = len(dates)

    def _put(self, block):
        count = block["mask"].shape[1]
        result = {}
        for key, value in block.items():
            fill = 1 if key in ("strikes", "asks") else 0
            padded = np.pad(value[0], (0, self.capacity-count), constant_values=fill)
            result[key] = jax.device_put(padded)
        return result

    def _batch(self, offset):
        return {key: jnp.stack([self.days[i][key] for i in range(offset, offset+self.batch_size)])
                for key in self.days[0]}

    def advance(self, day: date) -> int:
        if day <= self.slot_dates[int(self.order[-1])]:
            raise ValueError("Rolling dates must advance strictly")
        block = self.loader(day)
        count = block["mask"].shape[1]
        if count > self.capacity:
            capacity = int(np.ceil(count/1024)*1024)
            extra = capacity-self.capacity
            # Grow the retained panel on-device; do not retransmit old dates.
            self.days = [{key: jnp.pad(value, (0, extra), constant_values=1 if key in ("strikes", "asks") else 0)
                          for key, value in item.items()} for item in self.days]
            self.capacity = capacity
            self.batches = [self._batch(i) for i in range(0, len(self.days), self.batch_size)]
            print(json.dumps({"event": "device_capacity_growth", "capacity": capacity}), flush=True)
        slot = int(self.order[0])
        self.days[slot] = self._put(block)
        self.slot_dates[slot] = day
        self.order = np.r_[self.order[1:], slot].astype(np.int32)
        batch_index = slot//self.batch_size
        self.batches[batch_index] = self._batch(batch_index*self.batch_size)
        self.transferred_days += 1
        return slot

    def weights(self, config: RollingConfig):
        physical = np.empty(config.window, dtype=np.float64)
        physical[self.order] = date_weights(config.window, config.oldest_weight)
        return jnp.asarray(physical)


def initialize_daily(affine: dict, config: RollingConfig, pair_valid=None) -> dict:
    """Re-gauge prices exactly; initialize the AR exclusively from daily pairs."""
    h = np.asarray(affine["h"], dtype=np.float64)
    w = date_weights(len(h), config.oldest_weight)
    mean = np.average(h, axis=0, weights=w)
    centered = h-mean
    covariance = (centered.T*w)@centered/w.sum()
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < 1e-10:
        raise ValueError("Cannot whiten rank-deficient factors")
    scale = np.sqrt(eigenvalues)
    states = (centered@vectors)/scale
    loading = (scale[:, None]*vectors.T)@np.asarray(affine["B"])
    bias = np.asarray(affine["b"])+mean@np.asarray(affine["B"])
    if not np.allclose(bias+states@loading, affine["b"]+h@affine["B"], atol=1e-8, rtol=1e-10):
        raise RuntimeError("Factor re-gauging changed prices")
    pair_weights = w[1:]
    if pair_valid is not None:
        pair_weights = pair_weights*np.asarray(pair_valid)
    slope = np.sum(pair_weights[:, None]*states[:-1]*states[1:], axis=0)/np.sum(pair_weights[:, None]*states[:-1]**2, axis=0)
    slope = np.clip(slope, .05, .995)
    residual = states[1:]-states[:-1]*slope
    cov = (residual.T*pair_weights)@residual/pair_weights.sum()
    cov = .8*cov+.2*np.diag(np.diag(cov))+1e-6*np.eye(config.state_size)
    factor = np.linalg.cholesky(cov)
    diagonal = np.maximum(np.diag(factor)-config.innovation_std_floor, 1e-8)
    raw_factor = np.tril(factor, -1)+np.diag(diagonal+np.log(-np.expm1(-diagonal)))
    return {"b": bias, "B": loading, "h": states,
            "damping": np.diag(np.sqrt(np.maximum(-np.log(slope)-config.minimum_daily_rate, 1e-6))),
            "rotation": np.zeros_like(cov), "mean": np.zeros(config.state_size), "q_raw": raw_factor}


def daily_dynamics(params, order, weights, config: RollingConfig, pair_valid=None):
    """Daily Gaussian NLL weighted by endpoint date, and weighted affine gauge."""
    n = config.state_size
    lower = jnp.tril(params["damping"])
    rotation = .5*(params["rotation"]-params["rotation"].T)
    transition = jsp.linalg.expm(-(lower@lower.T+rotation+config.minimum_daily_rate*jnp.eye(n)))
    states = params["h"][order]
    w = weights[order]
    residual = states[1:]-(params["mean"]+(states[:-1]-params["mean"])@transition)
    q = params["q_raw"]
    factor = jnp.tril(q, -1)+jnp.diag(config.innovation_std_floor+jax.nn.softplus(jnp.diag(q)))
    white = jsp.linalg.solve_triangular(factor, residual.T, lower=True)
    pair_weights = w[1:] if pair_valid is None else w[1:]*pair_valid
    nll = .5*(jnp.sum(pair_weights*jnp.sum(white**2, axis=0))/jnp.sum(pair_weights)
              +2*jnp.sum(jnp.log(jnp.diag(factor)))+n*np.log(2*np.pi))/n
    mean = jnp.sum(w[:, None]*states, axis=0)/w.sum()
    center = states-mean
    covariance = (center.T*w)@center/w.sum()
    gauge = jnp.mean(mean**2)+jnp.sum((covariance-jnp.eye(n))**2)/n
    return config.dynamics_weight*nll+config.gauge_weight*gauge, (nll, gauge)


class Saturated(Exception):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def plot_progress(output: Path) -> None:
    """Every plotted daily point corresponds to a completed full recalibration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (output/"summary.jsonl").read_text().splitlines() if line]
    frame = pd.DataFrame(rows)
    dates = pd.to_datetime(frame.date)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.fill_between(dates, frame.daily_fit_p10, frame.daily_fit_p90, color="#477eae", alpha=.14, label="250-date training 10th–90th percentiles")
    ax.plot(dates, frame.today_inside_percent, color="#713da3", marker=".", label="Today's fit after full daily recalibration")
    future = [row for row in rows if row["forward_state_fit"] is not None]
    if future:
        ax.plot(pd.to_datetime([row["date"] for row in future]), [row["forward_state_fit"]["inside_percent"] for row in future],
                color="#dd7b28", marker=".", label="Today's state fitted under previous day's model")
    ax.axhline(90, color="gray", ls=":", lw=1)
    ax.set(xlabel="Daily full-model recalibration date", ylabel="Market quotes inside bid/ask (%)",
           title="DYSANOS 25D · daily AR(1) · rolling 250-date window")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(output/"daily-rolling-fit.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("completion", "panel", "archive", "initialization", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--through", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu" or not jax.config.jax_enable_x64:
        raise RuntimeError("Use a managed GPU allocation with JAX_ENABLE_X64=true")
    config = RollingConfig(batch_size=args.batch_size)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with FileLock(str(args.output/"sweep.lock"), timeout=0):
        run_sweep(args, config)


def run_sweep(args, config: RollingConfig) -> None:
    completion = json.loads(args.completion.read_text())
    geometry = _geometry(_geometry_from_completion(completion, requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, list(MODEL_DTES), allow_zero_backbone=True)
    candidates = sorted(date.fromisoformat(p.stem) for p in (args.panel/"days").glob("*.parquet")
                        if date(2022, 1, 1) <= date.fromisoformat(p.stem) <= args.through)
    with np.load(args.initialization/"states.npz") as source:
        initial_dates = tuple(date.fromisoformat(str(day)) for day in source["dates"])
        if not np.array_equal(source["model_business_dtes"], MODEL_DTES):
            raise ValueError("Initial grid mismatch")
    if tuple(candidates[:config.window]) != initial_dates:
        raise ValueError("Source is not the first causal 250-date window")
    sessions = get_business_days(candidates[0], candidates[-1])
    session_index = {day: i for i, day in enumerate(sessions)}
    gaps = sorted(set(sessions)-set(candidates))
    def valid_pairs(days):
        return np.asarray([session_index[b]-session_index[a]==1 for a, b in zip(days, days[1:])])
    if gaps:
        print(json.dumps({"event": "source_missing_sessions", "dates": [str(day) for day in gaps],
                          "handling": "exclude transitions crossing missing sessions from one-session AR likelihood"}), flush=True)
    # A missing fitted historical surface is a data error, never an excuse to refit it.
    for day in candidates:
        if not any((args.archive/location/str(day)/"models/all-0.1/model.h5").is_file() for location in ("days", "pilot")):
            raise ValueError(f"Missing certified historical surface at {day}")
    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    manifest = {"schema": SCHEMA, "config": asdict(config), "source_sha256": source_digest,
                "first_origin": str(initial_dates[-1]), "last_origin": str(candidates[-1]),
                "total_origins": len(candidates)-config.window+1, "model_business_dtes": MODEL_DTES,
                "ar_horizon_sessions": 1, "interpolation": "volatility", "quote_source": str(args.panel),
                "sanos_source": str(args.archive), "source_checkpoint": str(args.source), "dates": [str(x) for x in candidates],
                "missing_sessions": [str(day) for day in gaps]}
    if args.resume:
        previous = json.loads((args.output/"manifest.json").read_text())
        current = json.loads(json.dumps(manifest))
        for item in (previous, current):
            for key in ("batch_size", "max_iterations", "state_iterations"):
                item["config"].pop(key, None)
        if previous != current:
            raise ValueError("Resume manifest mismatch")
    else:
        atomic_json(args.output/"manifest.json", manifest)
    stop = threading.Event()
    resource_stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    progress = {"completed": 0, "total": manifest["total_origins"], "date": str(initial_dates[-1]), "iteration": 0, "phase": "load"}
    start_wall = time.monotonic()
    def heartbeat():
        while not stop.wait(60):
            elapsed = time.monotonic()-start_wall
            done = progress["completed"]
            remaining = (progress["total"]-done)*elapsed/done if done else None
            print(json.dumps({"event": "heartbeat", **progress, "elapsed_seconds": elapsed, "remaining_seconds": remaining}), flush=True)
            try:
                result = subprocess.run(["/home/hans/.local/bin/gpuctl", "resources", "--json"], capture_output=True, text=True, timeout=15)
                checks = json.loads(result.stdout)["checks"]
                bad = [{"name": x["name"], "health": x["health"]} for x in checks if x["health"] != "OK"]
                if bad:
                    print(json.dumps({"event": "resource_health", "checks": bad}), flush=True)
                if any(x["health"] in ("STOP", "UNAVAILABLE", "ERROR") for x in checks):
                    resource_stop.set()
                    stop.set()
            except Exception as error:
                print(json.dumps({"event": "resource_warning", "message": str(error)}), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    completed_index = config.window-2
    prior_band = None
    if args.resume and (args.output/"progress.json").exists():
        saved_progress = json.loads((args.output/"progress.json").read_text())
        completed_index = saved_progress["last_completed_index"]
        window_dates = tuple(candidates[completed_index-config.window+1:completed_index+1])
        with np.load(args.output/saved_progress["checkpoint"]) as saved:
            params_np = {key: saved[key] for key in ("b", "B", "h", "damping", "rotation", "mean", "q_raw")}
            slot_dates = [date.fromisoformat(str(x)) for x in saved["slot_dates"]]
        prior_band = saved_progress["fit_band"]
        progress["completed"] = completed_index-config.window+2
    else:
        window_dates = initial_dates
        with np.load(args.source) as saved:
            params_np = initialize_daily({key: saved[key] for key in ("b", "B", "h")}, config, valid_pairs(initial_dates))
        slot_dates = list(window_dates)
    loader = lambda day: load_quotes((day,), args.panel, args.archive, geometry)
    window = DeviceWindow(tuple(slot_dates), loader, config.batch_size)
    if args.resume:
        window.order = np.asarray([slot_dates.index(day) for day in window_dates], dtype=np.int32)
    params = jax.tree.map(jnp.asarray, params_np)
    print(json.dumps({"event": "sweep_started", "total_origins": progress["total"], "last_origin": manifest["last_origin"],
                      "daily_pairs": config.window-1, "device_days": len(window.days), "capacity": window.capacity}), flush=True)
    _, price_fn = make_functions(geometry, "volatility")

    def raw_loss(raw, quote, weights):
        block = {**quote, "date_weights": weights}
        price, cap = price_fn(raw, block)
        half = jnp.maximum(.5*(quote["asks"]-quote["bids"]), 1e-12)
        excess = (jax.nn.relu(price-quote["asks"])+jax.nn.relu(quote["bids"]-price)+.05*jnp.abs(price-quote["mids"]))/half
        daily = jnp.sum(excess*quote["mask"], axis=1)/jnp.sum(quote["mask"], axis=1)
        return jnp.sum(weights*daily)/weights.sum()+cap
    raw_gradient = jax.jit(jax.value_and_grad(raw_loss))
    dynamics_gradient = jax.jit(jax.value_and_grad(lambda p, o, w, v: daily_dynamics(p, o, w, config, v), has_aux=True))
    def fit_state_objective(h, b, B, quote):
        return raw_loss((b+h@B)[None, :], quote, jnp.ones(1))
    state_gradient = jax.jit(jax.value_and_grad(fit_state_objective))
    _, unravel = ravel_pytree(params)
    indices = [jnp.arange(i, i+config.batch_size) for i in range(0, config.window, config.batch_size)]

    def save_checkpoint(path, p, iteration):
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in p.items()}, iteration=iteration,
                            slot_dates=np.asarray([str(day) for day in window.slot_dates]), ordered_slots=window.order,
                            schema=SCHEMA, ar_horizon_sessions=1)
        temporary.replace(path)

    def assess(p):
        rows = []
        for ids, block in zip(indices, window.batches, strict=True):
            prediction, _ = price_fn(p["b"][None, :]+p["h"][ids]@p["B"], block)
            prediction = np.asarray(prediction)
            bid, ask, valid = (np.asarray(block[key]) for key in ("bids", "asks", "mask"))
            excess = (np.maximum(bid-prediction, 0)+np.maximum(prediction-ask, 0))/np.maximum(.5*(ask-bid), 1e-12)
            for j, slot in enumerate(np.asarray(ids)):
                e = excess[j][valid[j]]
                outside = e[e>0]
                rows.append({"date": str(window.slot_dates[int(slot)]), "quotes": int(len(e)),
                             "inside_percent": float(100*np.mean(e==0)),
                             "conditional_mean_excess": float(outside.mean()) if len(outside) else 0.,
                             "conditional_median_excess": float(np.median(outside)) if len(outside) else 0.})
        rows.sort(key=lambda row: row["date"])
        return rows

    first_index = max(config.window-1, completed_index+1)
    for origin_index in range(first_index, len(candidates)):
        if stop.is_set():
            break
        current_day = candidates[origin_index]
        progress.update(date=str(current_day), phase="prepare", iteration=0)
        started = time.monotonic()
        forward_fit = None
        if origin_index > config.window-1:
            newest_slot = int(window.order[-1])
            L = np.tril(np.asarray(params["damping"]))
            R = np.asarray(params["rotation"])
            horizon = session_index[current_day]-session_index[window.slot_dates[newest_slot]]
            A = expm(-horizon*(L@L.T+.5*(R-R.T)+config.minimum_daily_rate*np.eye(config.state_size)))
            mean = np.asarray(params["mean"])
            guess = mean+(np.asarray(params["h"][newest_slot])-mean)@A
            slot = window.advance(current_day)
            quote = {key: value[None, :] for key, value in window.days[slot].items()}
            def state_value(h):
                if stop.is_set():
                    raise InterruptedError("Stop requested")
                value, gradient = state_gradient(jnp.asarray(h), params["b"], params["B"], quote)
                return float(value), np.asarray(gradient)
            progress["phase"] = "daily_state"
            state_result = minimize(state_value, guess, jac=True, method="L-BFGS-B",
                                    options={"maxiter": config.state_iterations, "maxls": 30, "ftol": 1e-10, "gtol": 1e-6})
            params["h"] = params["h"].at[slot].set(jnp.asarray(state_result.x))
            prediction, _ = price_fn((params["b"]+params["h"][slot]@params["B"])[None, :], quote)
            valid = np.asarray(quote["mask"])
            fit = float(100*np.mean(((np.asarray(prediction)>=np.asarray(quote["bids"])) & (np.asarray(prediction)<=np.asarray(quote["asks"])))[valid]))
            forward_fit = {"inside_percent": fit, "iterations": int(state_result.nit), "success": bool(state_result.success),
                           "below_previous_p10": fit<prior_band[0], "above_previous_p90": fit>prior_band[1]}
            print(json.dumps({"event": "daily_state_fit", "date": str(current_day), **forward_fit}), flush=True)
        chronological = tuple(candidates[origin_index-config.window+1:origin_index+1])
        if tuple(window.slot_dates[int(i)] for i in window.order) != chronological:
            raise RuntimeError("Ring-buffer dates do not match the causal training window")
        weights = window.weights(config)
        order = jnp.asarray(window.order)
        pair_valid = jnp.asarray(valid_pairs(chronological))
        history, last = [], {}
        fit_directory = args.output/"origins"/str(current_day)
        fit_directory.mkdir(parents=True, exist_ok=args.resume)
        if args.resume and (fit_directory/"working.npz").exists():
            with np.load(fit_directory/"working.npz") as saved:
                if list(saved["slot_dates"]) != [str(day) for day in window.slot_dates]:
                    raise ValueError("Working checkpoint physical-date mapping differs")
                params = {key: jnp.asarray(saved[key]) for key in params}
            print(json.dumps({"event": "working_checkpoint_restored", "date": str(current_day), "optimizer": "L-BFGS curvature reset"}), flush=True)
        progress["phase"] = "joint_fit"

        def value_gradient(vector):
            if stop.is_set():
                raise InterruptedError("Stop requested; latest periodic checkpoint retained")
            p = unravel(jnp.asarray(vector))
            gradient = jax.tree.map(jnp.zeros_like, p)
            quote_value = jnp.asarray(0., dtype=jnp.float64)
            for ids, block in zip(indices, window.batches, strict=True):
                batch_weights = weights[ids]
                multiplier = batch_weights.sum()/weights.sum()
                raw = p["b"][None, :]+p["h"][ids]@p["B"]
                value, raw_grad = raw_gradient(raw, block, batch_weights)
                g = multiplier*raw_grad
                quote_value = quote_value+multiplier*value
                gradient["b"] = gradient["b"]+jnp.sum(g, axis=0)
                gradient["B"] = gradient["B"]+p["h"][ids].T@g
                gradient["h"] = gradient["h"].at[ids].set(g@p["B"].T)
            (dyn_value, (nll, gauge)), dyn_grad = dynamics_gradient(p, order, weights, pair_valid)
            gradient = jax.tree.map(lambda a, b: a+b, gradient, dyn_grad)
            flat_grad = np.asarray(ravel_pytree(gradient)[0], dtype=np.float64)
            total = float(quote_value+dyn_value)
            if not np.isfinite(total) or not np.isfinite(flat_grad).all():
                raise RuntimeError("Non-finite daily joint objective or gradient")
            last.update(objective=total, quote_and_cap=float(quote_value), nll=float(nll), gauge=float(gauge),
                        gradient_rms=float(np.sqrt(np.mean(flat_grad**2))))
            return total, flat_grad

        accepted_params = [params]
        accepted_iteration = [0]
        def accepted(vector):
            accepted_iteration[0] += 1
            i = accepted_iteration[0]
            progress["iteration"] = i
            p = unravel(jnp.asarray(vector))
            accepted_params[0] = p
            if i==1 or i%10==0:
                rows = assess(p)
                fit = float(np.average([row["inside_percent"] for row in rows], weights=[row["quotes"] for row in rows]))
                record = {"iteration": i, **last, "inside_percent": fit, "elapsed_seconds": time.monotonic()-started}
                history.append(record)
                save_checkpoint(fit_directory/"working.npz", p, i)
                print(json.dumps({"event": "fit_progress", "date": str(current_day), **record}), flush=True)
                previous = next((r for r in reversed(history) if r["iteration"]<=i-100), None)
                if previous and fit>=90 and fit-previous["inside_percent"]<.1 and (previous["quote_and_cap"]-record["quote_and_cap"])/abs(previous["quote_and_cap"])<.01:
                    raise Saturated("market fit saturated over 100 iterations")
        reason = ""
        try:
            result = minimize(value_gradient, np.asarray(ravel_pytree(params)[0]), jac=True, method="L-BFGS-B", callback=accepted,
                              options={"maxiter": config.max_iterations, "maxcor": 10, "maxls": 30, "ftol": 1e-10, "gtol": 1e-6})
            params = unravel(jnp.asarray(result.x))
            reason = str(result.message)
        except Saturated as error:
            params = accepted_params[0]
            reason = str(error)
        except InterruptedError:
            atomic_json(args.output/"STOPPED.json", {"date": str(current_day), "resource_stop": resource_stop.is_set(),
                                                     "checkpoint": str(fit_directory/"working.npz")})
            print(json.dumps({"event": "sweep_stopped", "date": str(current_day)}), flush=True)
            return
        rows = assess(params)
        pd.DataFrame(rows).to_csv(fit_directory/"daily-fit.csv", index=False)
        daily_fit = np.asarray([row["inside_percent"] for row in rows])
        band = [float(np.quantile(daily_fit, .1)), float(np.quantile(daily_fit, .9))]
        source, _ = load_certified_pricer(current_day, archive=args.archive)
        slot = int(window.order[-1])
        raw = np.asarray(params["b"]+params["h"][slot]@params["B"])
        strikes, model_prices = grid_prices(raw, geometry)
        source_prices = source.price(geometry.model_expiries_y[:, None], strikes[None, :], mode="surface")
        grid_mae = float(np.mean(np.abs(model_prices-source_prices)))
        model_path = fit_directory/"model.npz"
        save_checkpoint(model_path, params, accepted_iteration[0])
        summary = {"date": str(current_day), "initialization_date": str(candidates[origin_index-1]) if origin_index>config.window-1 else None,
                   "window_first": str(chronological[0]), "window_last": str(current_day), "window_dates": len(rows), "daily_pairs": int(np.sum(np.asarray(pair_valid))),
                   "market_inside_percent": float(np.average(daily_fit, weights=[r["quotes"] for r in rows])),
                   "today_inside_percent": rows[-1]["inside_percent"], "daily_fit_p10": band[0], "daily_fit_p90": band[1],
                   "grid_price_mae_today": grid_mae, "forward_state_fit": forward_fit, "iterations": accepted_iteration[0],
                   "stop_reason": reason, "elapsed_seconds": time.monotonic()-started, "transferred_days_total": window.transferred_days,
                   "source_initialization": str(args.source) if origin_index==config.window-1 else str(args.output/"origins"/str(candidates[origin_index-1])/"model.npz")}
        atomic_json(fit_directory/"report.json", {**summary, "history": history})
        atomic_json(args.output/"progress.json", {"last_completed_index": origin_index, "checkpoint": str(model_path.relative_to(args.output)),
                                               "fit_band": band, "last_completed_date": str(current_day), "completed_origins": origin_index-config.window+2,
                                               "total_origins": manifest["total_origins"]})
        with (args.output/"summary.jsonl").open("a") as stream:
            stream.write(json.dumps(summary)+"\n")
        # The final checkpoint supersedes its disposable same-origin working copy.
        (fit_directory/"working.npz").unlink(missing_ok=True)
        prior_band = band
        progress["completed"] = origin_index-config.window+2
        print(json.dumps({"event": "origin_complete", **summary, "completed_origins": progress["completed"], "total_origins": progress["total"]}), flush=True)
        if progress["completed"]<=3 or progress["completed"]%10==0:
            plot_progress(args.output)
        if args.max_origins and progress["completed"]>=args.max_origins:
            break
    if progress["completed"]==manifest["total_origins"]:
        plot_progress(args.output)
        atomic_json(args.output/"COMPLETE.json", {"last_date": manifest["last_origin"], "origins": progress["completed"],
                                                "elapsed_seconds": time.monotonic()-start_wall})
        print(json.dumps({"event": "SWEEP_COMPLETE", "last_date": manifest["last_origin"], "origins": progress["completed"]}), flush=True)
    stop.set()


if __name__ == "__main__":
    main()
