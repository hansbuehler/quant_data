"""JAX affine DYSANOS calibration with learned DLV, grid and backbone states."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.sanos import make_shared_nms_node_sets
from quantdata.dysanos.v11.licensed_market import licensed_market_day
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry
from quantdata.jax.tridiagonal import solve_tridiagonal_pcr


def make_functions(geometry, interpolation, *, direct=False, zero_backbone=False):
    """Build a differentiable shared-support DYSANOS pricer and quote loss."""
    if interpolation not in ("variance", "volatility"):
        raise ValueError(interpolation)
    time_values = np.asarray(geometry.model_expiries_y, dtype=np.float64)
    front_nodes, terminal_nodes = make_shared_nms_node_sets(num_inner_strikes=geometry.num_inner_strikes, nms_range=geometry.nms_range)
    atm_index = int(np.sum(front_nodes < 0) + np.sum(terminal_nodes < 0))
    n, t = geometry.num_inner_strikes, geometry.num_model_expiries
    cap_log = float(geometry.maximum_log_dlvsq)

    def prices_and_penalty(raw, quote):
        times = jnp.asarray(time_values, dtype=raw.dtype)
        dts = jnp.diff(times, prepend=jnp.zeros(1, dtype=raw.dtype))
        front = jnp.asarray(front_nodes, dtype=raw.dtype)
        terminal = jnp.asarray(terminal_nodes, dtype=raw.dtype)
        blocks = raw[:, :t*(n+1)].reshape(-1, t, n+1)
        log_dlv = blocks[..., :n]
        dlv = geometry.min_dlvsq + jnp.exp(jnp.minimum(log_dlv, cap_log))
        forward = geometry.min_forward_vol + (geometry.max_forward_vol-geometry.min_forward_vol)*jax.nn.sigmoid(blocks[..., n])
        eta_raw = jax.nn.sigmoid(raw[:, -2:])
        eta_pair = geometry.min_eta + (geometry.max_eta-geometry.min_eta)*eta_raw
        eta = jnp.concatenate((eta_pair[:, :1], jnp.broadcast_to(eta_pair[:, 1:2], (raw.shape[0], t-1))), axis=1)
        total = jnp.cumsum(forward*jnp.sqrt(dts), axis=1)
        backbone = jnp.cumsum(forward*eta*jnp.sqrt(dts), axis=1)
        if zero_backbone:
            backbone = jnp.zeros_like(backbone)
        scales = jnp.sqrt(((total[:, :1]**2/times[0])*geometry.nms_epsilon_y + total**2)/(1.+geometry.nms_epsilon_y))
        ordered = jnp.sort(jnp.concatenate((scales[:, :1]*front, scales[:, -1:]*terminal), axis=1), axis=1)
        min_gap = geometry.minimum_log_spacing_fraction*(ordered[:, -1:]-ordered[:, :1])/(n-1)
        lgaps = jnp.maximum(jnp.diff(ordered[:, :atm_index+1], axis=1), min_gap)
        rgaps = jnp.maximum(jnp.diff(ordered[:, atm_index:], axis=1), min_gap)
        left = -jnp.flip(jnp.cumsum(jnp.flip(lgaps, axis=1), axis=1), axis=1)
        right = jnp.cumsum(rgaps, axis=1)
        inner = jnp.exp(jnp.concatenate((left, jnp.zeros_like(ordered[:, :1]), right), axis=1))
        support = jnp.concatenate((geometry.lower_boundary_scale*inner[:, :1], inner, geometry.upper_boundary_scale*inner[:, -1:]), axis=1)
        initial = jax.nn.one_hot(jnp.full((raw.shape[0],), atm_index+1), n+2, dtype=raw.dtype)
        def step(previous, args):
            variance, dt = args
            box = inner**2*dt/(support[:, 2:]-support[:, :-2])
            wm = variance*box/(inner-support[:, :-2])
            wp = variance*box/(support[:, 2:]-inner)
            zeros = jnp.zeros_like(wm[:, :2])
            lower = jnp.concatenate((zeros, -wp), axis=1)
            diagonal = jnp.concatenate((jnp.ones_like(wm[:, :1]), 1.+wm+wp, jnp.ones_like(wm[:, :1])), axis=1)
            upper = jnp.concatenate((-wm, zeros), axis=1)
            density = solve_tridiagonal_pcr(lower, diagonal, upper, previous[..., None])[..., 0]
            return density, density
        _, values = jax.lax.scan(step, initial, (jnp.swapaxes(dlv, 0, 1), dts))
        densities = jnp.swapaxes(values, 0, 1)
        b = jnp.arange(raw.shape[0])[:, None]
        upper, lower, alpha, before = quote["upper"], quote["lower"], quote["alpha"], quote["before"]
        hi_density, lo_density = densities[b, upper], densities[b, lower]
        lo_density = jnp.where(before[..., None], initial[:, None, :], lo_density)
        high_w = backbone[b, upper]
        low_w = jnp.where(before, 0., backbone[b, lower])
        if interpolation == "variance":
            sigma = jnp.sqrt(jnp.maximum((1.-alpha)*low_w**2 + alpha*high_w**2, 1e-20))
        else:
            sigma = jnp.maximum((1.-alpha)*low_w + alpha*high_w, 1e-10)
        strike = quote["strikes"]
        z = (jnp.log(support[:, None, :]/strike[..., None])+0.5*sigma[..., None]**2)/sigma[..., None]
        calls = support[:, None, :]*jsp.special.ndtr(z)-strike[..., None]*jsp.special.ndtr(z-sigma[..., None])
        call = jnp.sum(calls*((1.-alpha[..., None])*lo_density+alpha[..., None]*hi_density), axis=-1)
        price = jnp.where(quote["is_call"], call, call-1.+strike)
        barrier = geometry.dlv_cap_barrier_log_width*jax.nn.softplus((log_dlv-cap_log)/geometry.dlv_cap_barrier_log_width)/geometry.dlv_cap_barrier_log_std
        daily_penalty = .5*jnp.sum(barrier**2, axis=(1,2))
        if "date_weights" in quote:
            penalty = jnp.sum(quote["date_weights"]*daily_penalty)/jnp.sum(quote["date_weights"])
        else:
            penalty = jnp.mean(daily_penalty)
        return price, penalty

    def loss(params, quote, indices):
        raw = params["states"][indices] if direct else params["b"][None, :] + params["h"][indices] @ params["B"]
        price, cap = prices_and_penalty(raw, quote)
        half = jnp.maximum(.5*(quote["asks"]-quote["bids"]), 1e-12)
        excess = (jax.nn.relu(price-quote["asks"])+jax.nn.relu(quote["bids"]-price)+.05*jnp.abs(price-quote["mids"])) / half
        per_day = jnp.sum(excess*quote["mask"], axis=1)/jnp.maximum(jnp.sum(quote["mask"], axis=1), 1.)
        return jnp.mean(per_day)+cap, (jnp.mean(per_day),cap)
    return jax.jit(jax.value_and_grad(loss, has_aux=True)), jax.jit(prices_and_penalty)


def load_quotes(dates, panel, archive, geometry):
    """Load filtered licensed prices and pad by date for batched JAX fitting."""
    days = []
    start = time.monotonic()
    for i, day in enumerate(dates, 1):
        days.append(licensed_market_day(day, panel=panel, archive=archive, geometry=geometry))
        if i % 25 == 0:
            print(f"market loaded={i}/{len(dates)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    q = max(day.num_quotes for day in days)
    shape = (len(days), q)
    out = {"strikes": np.ones(shape, np.float32), "bids": np.zeros(shape, np.float32), "asks": np.ones(shape, np.float32),
           "mids": np.zeros(shape, np.float32), "is_call": np.ones(shape, bool), "mask": np.zeros(shape, bool),
           "upper": np.zeros(shape, np.int32), "lower": np.zeros(shape, np.int32), "alpha": np.zeros(shape, np.float32), "before": np.zeros(shape, bool)}
    times = geometry.model_expiries_y
    for i, day in enumerate(days):
        count = day.num_quotes
        qtimes = day.expiries_y[day.expiry_index]
        raw_upper = np.searchsorted(times, qtimes, side="left")
        upper = np.minimum(raw_upper, len(times)-1)
        lower = np.maximum(raw_upper-1, 0)
        before = raw_upper == 0
        earlier = np.where(before, 0., times[lower])
        alpha = np.clip((qtimes-earlier)/(times[upper]-earlier), 0., 1.)
        for name, value in (("strikes",day.strikes),("bids",day.bids),("asks",day.asks),("mids",day.fit_prices),("is_call",day.is_call),("upper",upper),("lower",lower),("alpha",alpha),("before",before)):
            out[name][i, :count] = value
        out["mask"][i, :count] = True
    return out


def main():
    parser = argparse.ArgumentParser()
    for name in ("completion", "panel", "archive", "initialization", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--resume-directory", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("A managed GPU allocation is required")
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    with np.load(args.initialization / "states.npz") as saved:
        from datetime import date
        dates = tuple(date.fromisoformat(str(day)) for day in saved["dates"])
        raw = saved["raw_states"].astype(np.float32)
    if len(dates) != 250:
        raise ValueError("Expected exactly 250 historical dates")
    quote = load_quotes(dates, args.panel, args.archive, geometry)
    mean = raw.mean(axis=0)
    left, singular, right = np.linalg.svd(raw-mean, full_matrices=False)
    initial = {"b": mean, "B": right[:200], "h": left[:, :200]*singular[:200]}
    for interpolation in ("variance", "volatility"):
        start = time.monotonic()
        generator = np.random.default_rng(20261106)
        heartbeat_done = threading.Event()
        progress = {"epoch": 0}
        def heartbeat():
            while not heartbeat_done.wait(60):
                elapsed = time.monotonic()-start
                epoch = progress["epoch"]
                remaining = (args.epochs-epoch)*elapsed/epoch if epoch else None
                print(f"heartbeat interpolation={interpolation} epoch={epoch}/{args.epochs} elapsed={elapsed:.1f}s remaining_est={remaining}", flush=True)
        threading.Thread(target=heartbeat, daemon=True).start()
        if args.resume_directory is None:
            params = jax.tree.map(jnp.asarray, initial)
        else:
            with np.load(args.resume_directory / f"{interpolation}-checkpoint.npz") as checkpoint:
                params = {name: jnp.asarray(checkpoint[name]) for name in ("b", "B", "h")}
        optimizer = optax.adam(1e-4)
        state = optimizer.init(params)
        grad_fn, price_fn = make_functions(geometry, interpolation)
        history = []
        full_history = []
        def evaluate():
            count = inside = 0
            objective_sum = 0.
            for offset in range(0, len(dates), args.batch):
                idx = np.arange(offset, min(offset+args.batch,len(dates)))
                block = {name: jnp.asarray(value[idx]) for name,value in quote.items()}
                raw_batch = params["b"][None,:]+params["h"][idx]@params["B"]
                prediction, cap = price_fn(raw_batch, block)
                prediction, cap = np.asarray(prediction), float(cap)
                mask = quote["mask"][idx]
                bid, ask, mid = quote["bids"][idx], quote["asks"][idx], quote["mids"][idx]
                half = np.maximum(.5*(ask-bid), 1e-12)
                excess = (np.maximum(prediction-ask,0)+np.maximum(bid-prediction,0)+.05*np.abs(prediction-mid))/half
                objective_sum += float(np.sum((excess*mask).sum(axis=1)/mask.sum(axis=1))) + cap*len(idx)
                inside += int(np.sum(mask & (prediction >= bid) & (prediction <= ask)))
                count += int(mask.sum())
            return objective_sum/len(dates), 100*inside/count, count
        for epoch in range(1, args.epochs+1):
            order = generator.permutation(len(dates))
            losses = []
            for offset in range(0, len(dates), args.batch):
                idx = order[offset:offset+args.batch]
                block = {name: jnp.asarray(value[idx]) for name, value in quote.items()}
                (value, _), grad = grad_fn(params, block, jnp.asarray(idx))
                updates, state = optimizer.update(grad, state, params)
                params = optax.apply_updates(params, updates)
                losses.append(float(value)*len(idx))
            value = sum(losses)/len(dates)
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite objective for {interpolation} at epoch {epoch}")
            history.append(value)
            progress["epoch"] = epoch
            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                elapsed = time.monotonic()-start
                full_objective, inside_percent, _ = evaluate()
                full_history.append((epoch, full_objective, inside_percent))
                temporary = args.output / f"{interpolation}-checkpoint.tmp.npz"
                np.savez_compressed(temporary, **{k: np.asarray(v) for k,v in params.items()}, history=np.asarray(history), full_history=np.asarray(full_history), epoch=epoch)
                temporary.replace(args.output / f"{interpolation}-checkpoint.npz")
                print(f"interpolation={interpolation} epoch={epoch}/{args.epochs} minibatch_objective={value:.7g} full_objective={full_objective:.7g} inside={inside_percent:.4f}% elapsed={elapsed:.1f}s remaining_est={(args.epochs-epoch)*elapsed/epoch:.1f}s", flush=True)
            if len(full_history) >= 5 and epoch >= 200 and max(row[1] for row in full_history[-5:])-min(row[1] for row in full_history[-5:]) <= .001*max(abs(full_history[-5][1]),1e-12):
                break
        full_objective, inside_percent, count = evaluate()
        result = {"interpolation": interpolation, "epochs": epoch, "minibatch_objective": history[-1], "full_objective": full_objective, "quotes": count, "inside_percent": inside_percent, "elapsed_seconds": time.monotonic()-start}
        (args.output / f"{interpolation}-report.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result), flush=True)
        heartbeat_done.set()


if __name__ == "__main__":
    main()
