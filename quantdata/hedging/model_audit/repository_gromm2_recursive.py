"""Recursive observable-IV diagnostics for the repository's fixed GROMM tower."""

from __future__ import annotations

import argparse
import os
import time
from datetime import date
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cdxcore.bs import bs
from openqdata.models.gromm import estimate, pipeline, simulate
from openqdata.models.gromm.config import DAY

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.implied import (
    implied_volatilities_with_validity,
    summarize_inversions,
)
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.persistence import path_acf
from quantdata.hedging.model_audit.repository_gromm2_static import tower
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import (
    COMMON_INDICES,
    COMMON_NMS,
    GRID,
    _hash,
    _json,
    build_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel


def leading_normals(key, *, total_paths: int, paths: int, factors: int):
    """Reproduce the first antithetic pairs from a full-size JAX draw."""
    normals = jax.random.normal(key, (total_paths // 2, factors + 1))[:paths // 2]
    return jnp.stack([normals, -normals], axis=1).reshape(paths, factors + 1)


@partial(jax.jit, static_argnums=(0, 4, 5))
def leading_day(simulator, state, day, key, substeps, total_paths):
    """Advance a prefix of the original antithetic 1,000-path random stream."""
    logsig, boundaries, brownian, log_y = state
    dt = DAY / substeps
    backbone_variance = (simulator.yvar[day + 1] - simulator.yvar[day]) / substeps
    paths = logsig.shape[0]

    def substep(index, current):
        current_logsig, current_brownian, current_log_y = current
        tau = DAY - index * dt
        front, loading = jax.vmap(simulator.front, in_axes=(0, 0, None))(boundaries, current_brownian, tau)
        _, drift = jax.vmap(simulator.chain, in_axes=(0, 0, 0, None))(current_logsig, front, loading, day)
        normals = leading_normals(jax.random.fold_in(key, index), total_paths=total_paths, paths=paths, factors=simulator.F)
        increment = normals[:, :simulator.F] * jnp.sqrt(dt)
        shock = jnp.einsum("mfn,pf->pmn", simulator.xi, increment + simulator.lam * dt)
        current_logsig = current_logsig + (drift - 0.5 * simulator.xi2) * dt + shock
        current_brownian = current_brownian + increment[:, 0] + simulator.lam[0] * dt
        current_log_y = current_log_y - 0.5 * backbone_variance + jnp.sqrt(backbone_variance) * normals[:, simulator.F]
        return current_logsig, current_brownian, current_log_y

    logsig, brownian, log_y = jax.lax.fori_loop(0, substeps, substep, (logsig, brownian, log_y))
    node = jax.vmap(lambda boundary, value: jnp.clip(jnp.searchsorted(boundary, value) - 1, 0, simulator.N - 1))(boundaries, brownian)
    next_row = jnp.minimum(day + 1, simulator.M - 1)
    next_boundaries = jax.vmap(simulator.start_front, in_axes=(0, 0, None))(logsig, node, next_row)
    return logsig, next_boundaries, jnp.zeros_like(brownian), log_y


def observable(simulator, state, day: int) -> tuple[np.ndarray, list[dict]]:
    densities, levels, spots = tower(simulator, state, day)
    support = np.asarray(simulator.strikes)
    variance = np.asarray(simulator.yvar)
    logs = []
    inversions: list[dict] = []

    def price(tenor: int, strikes: np.ndarray) -> np.ndarray:
        nodes = support[day + tenor - 1][None] * np.exp(levels[:, None]) / spots[:, None]
        deviation = np.sqrt(variance[day + tenor] - variance[day])
        return np.sum(bs.price(k=strikes[:, :, None] / nodes[:, None], vol=deviation, sqrtT=1.)
            * nodes[:, None] * densities[:, day + tenor - 1, None], axis=-1)

    def implied(tenor: int, strikes: np.ndarray, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        finite = np.isfinite(values) & np.isfinite(strikes) & (values >= np.maximum(1. - strikes, 0.) - 1e-10) & (values <= 1. + 1e-10)
        output = np.full(shape, np.nan)
        if finite.any():
            implied_values, _ = implied_volatilities_with_validity(expiries_y=np.full(int(finite.sum()), tenor / 252.),
                normalized_strikes=strikes[finite], normalized_prices=values[finite],
                is_call=np.ones(int(finite.sum()), dtype=bool), diagnostics=inversions)
            output[finite] = np.where(np.isfinite(implied_values) & (implied_values > 0.), implied_values, np.nan)
        return output

    for expiry in GRID.expiries_y:
        tenor = int(round(expiry * 252.))
        if day + tenor > simulator.M:
            raise ValueError("Requested observable tenor exceeds GROMM fixed-expiry tower")
        atm_strikes = np.ones((len(spots), 1))
        atm = implied(tenor, atm_strikes, price(tenor, atm_strikes))
        strikes = NMS.from_nms(COMMON_NMS[None], expiry, vol=atm)
        valid = np.isfinite(strikes).all(axis=1)
        values = np.full(strikes.shape, np.nan)
        if valid.any():
            all_strikes = np.where(np.isfinite(strikes), strikes, 1.)
            values[valid] = price(tenor, all_strikes)[valid]
        logs.append(np.log(implied(tenor, strikes, values)))
    return np.stack(logs, axis=1).reshape(len(spots), -1), inversions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--measure", choices=("P", "Q"), required=True)
    parser.add_argument("--paths", type=int, default=32)
    parser.add_argument("--total-paths", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=115)
    args = parser.parse_args()
    if args.origin not in origins(args.reference) or args.paths < 2 or args.paths % 2 or args.total_paths % 2 or args.paths > args.total_paths:
        raise ValueError("Require an approved origin and an even prefix of the total antithetic paths")
    folder = args.run / f"gromm2/{args.origin}"
    if not (folder / "ESTIMATION_COMPLETE.json").exists():
        raise ValueError("GROMM2 origin has not completed fitting")
    dynamics = estimate.load_dynamics(folder)
    initial = simulate.Model(dynamics, np.array([1]), np.array([1.]))
    risk_price, physical = pipeline.market_price_of_risk(initial, dynamics)
    simulator = simulate.Model(dynamics, np.array([1]), np.array([1.]), lam=risk_price if args.measure == "P" else None)
    maximum = simulator.M - int(round(max(GRID.expiries_y) * 252.))
    if args.steps > maximum:
        raise ValueError(f"Only {maximum} days supported without extending the fixed-expiry tower")
    args.output.mkdir(parents=True, exist_ok=True)
    panel = PublicHedgingPanel(Path(os.environ["QUANTDATA_ROOT"]) / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, args.output)
    index = history.dates.index(date.fromisoformat(args.origin))
    training = np.flatnonzero((history.sessions_from_previous_observation == 1) & (np.arange(len(history.dates)) <= index))[-252:]
    if len(training) != 252:
        raise ValueError("Missing 252 causal one-session reference observations")
    with np.load(args.reference / "frozen_observable_pca.npz", allow_pickle=False) as saved:
        basis = saved["basis"].copy()
    reference = history.log_implied_volatilities[training][:, :, COMMON_INDICES].reshape(len(training), -1)
    state = simulator.initial(args.paths)
    key = jax.random.PRNGKey(20_261_112)
    values, inversion_records = [], []
    started = time.monotonic()
    for step in range(args.steps):
        health_check()
        state = leading_day(simulator, state, jnp.int32(step), jax.random.fold_in(key, step), 8, args.total_paths)
        surface, records = observable(simulator, state, step + 1)
        values.append(surface)
        inversion_records.extend(records)
        elapsed = time.monotonic() - started
        print(f"GROMM2 recursive {args.measure} {args.origin} completed={step+1}/{args.steps} "
            f"elapsed={elapsed:.1f}s eta={elapsed*(args.steps-step-1)/(step+1):.1f}s", flush=True)
    surfaces = np.stack(values, axis=1)
    valid = np.isfinite(surfaces).all(axis=(1, 2))
    result = {"model": f"repository GROMM2-{args.measure}", "origin": args.origin, "paths": args.paths,
        "steps": args.steps, "total_static_paths": args.total_paths, "valid_paths": int(valid.sum()),
        "invalid_path_rate": float(np.mean(~valid)), "extreme_surface_rate": float(np.mean(np.any(surfaces[valid] > np.log(3.), axis=(1, 2)))) if valid.any() else None,
        "observable_acf": path_acf(surfaces[valid] @ basis) if valid.any() else None,
        "historical_observable_acf": path_acf((reference @ basis)[None]), "iv_inversion": summarize_inversions(inversion_records),
        "physical": physical, "dynamics_sha256": _hash(folder / "dynamics.npz"), "seed": 20_261_112,
        "sampling": "leading antithetic paths of the 1000-path static simulation", "status": "complete" if valid.any() else "no_valid_paths"}
    _json(args.output / f"recursive/{args.measure}/{args.origin}.json", result)


if __name__ == "__main__":
    main()
