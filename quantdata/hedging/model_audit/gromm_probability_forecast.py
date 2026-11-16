"""Matched one-day decoded-IV variance for GROMM1 before/after p-weighting."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.run import COMMON_NMS, GRID, _hash, _json


def observable(dynamics, densities: np.ndarray, log_background: np.ndarray, spots: np.ndarray, *, elapsed: int) -> np.ndarray:
    """Same floating-NMS observable, including the unchanged independent Black Y."""
    results = []
    geometry = dynamics.geometry
    for tenor in np.asarray(GRID.expiries_y) * 252.:
        indices = np.flatnonzero(np.isclose(geometry.expiries * 252., tenor + elapsed, rtol=0., atol=1e-10))
        if len(indices) != 1:
            raise ValueError("No exact surviving model expiry for the common observable")
        row = int(indices[0])
        variance = float(geometry.variance_at(geometry.expiries[row], backbone=True)
                         - geometry.variance_at(elapsed / 252., backbone=True))
        if variance < 0:
            raise ValueError("Negative remaining Black variance")
        latent = np.exp(log_background[:, None]) * geometry.supports[row]

        def prices(strikes):
            calls = strikes >= 1.
            cash = spots[:, None] * strikes
            kernels = bs.price(k=cash[:, :, None] / latent[:, None], vol=np.sqrt(variance),
                sqrtT=1., is_call=calls[:, :, None], eps=1e-14)
            return np.sum(kernels * latent[:, None] * densities[:, row, None], axis=-1) / spots[:, None]

        atm_prices = prices(np.ones((len(spots), 1)))
        atm, valid = implied_volatilities_with_validity(expiries_y=np.full(atm_prices.shape, tenor / 252.),
            normalized_strikes=np.ones_like(atm_prices), normalized_prices=atm_prices, is_call=np.ones_like(atm_prices, bool))
        if not valid.all():
            raise ValueError("Invalid ATM IV; no path is dropped or price clipped")
        strikes = NMS.from_nms(COMMON_NMS[None], tenor / 252., vol=atm)
        values = prices(strikes)
        vols, valid = implied_volatilities_with_validity(expiries_y=np.full(values.shape, tenor / 252.),
            normalized_strikes=strikes, normalized_prices=values, is_call=strikes >= 1.)
        if not valid.all():
            raise ValueError("Invalid observable IV; no path is dropped or price clipped")
        results.append(np.log(vols))
    return np.stack(results, axis=1).reshape(len(spots), -1)


def run(baseline: Path, weighted: Path, output: Path, *, paths: int) -> None:
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm.dynamics import load_dynamics
    from openqdata.models.gromm.log_numerics import future_log_densities
    from openqdata.models.gromm.progress import Heartbeat
    from openqdata.models.gromm.simulation import Simulator, reduced_book

    if paths < 4 or paths % 2:
        raise ValueError("An even number of at least four paths is required")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "gromm1-pweight-decoded-variance-v1", "paths": paths, "substeps": 8, "seed": 20_261_112,
        "baseline_sha256": _hash(baseline), "weighted_sha256": _hash(weighted), "source_sha256": _hash(Path(__file__)),
        "observable": "49 floating log-IV coordinates; same native exact surviving expiries; independent Black Y retained",
        "scope": "single 2025-08-29 origin, not the older 44-day GROMM2 cohort; no future observation is imputed"}
    _json(output / "PROTOCOL.json", protocol)
    started = time.monotonic()
    records, observations = {}, {}

    def emit(event):
        print(dict(event, elapsed_seconds=time.monotonic() - started), flush=True)

    with Heartbeat(emit, interval=60.) as progress:
        for label, path in (("baseline", baseline), ("p_weighted", weighted)):
            dynamics = load_dynamics(path)
            for measure in ("Q", "P"):
                progress.update({"phase": f"{label}-{measure}", "completed": len(records), "total": 4})
                simulator = Simulator(dynamics, reduced_book(dynamics.geometry), substeps=8,
                    days=1, measure=measure, density_representation="log")
                pairs = paths // 2
                keys = jax.vmap(lambda index: jax.random.fold_in(jax.random.PRNGKey(protocol["seed"]), index))(
                    jnp.arange(pairs, dtype=jnp.uint32))
                front = jnp.broadcast_to(simulator.initial_front, (paths, len(simulator.initial_front)))
                future = jnp.broadcast_to(jnp.asarray(simulator.states[1:]), (paths, *simulator.states[1:].shape))
                marks = jnp.broadcast_to(simulator.initial_marks, (paths, len(simulator.initial_marks)))
                initial = (front, future, jnp.zeros(paths), marks, jnp.full(paths, -1, dtype=jnp.int64))
                _, _, operation = simulator.operations[0]
                carry, _ = operation(initial, keys)
                front, future, background, final_marks, failures = carry
                if np.any(np.asarray(failures) >= 0):
                    raise ValueError(f"{label}-{measure} had {np.sum(np.asarray(failures)>=0)} invalid paths; none dropped")
                geometry = dynamics.geometry
                logs = jax.vmap(lambda probabilities, state: future_log_densities(probabilities, state,
                    jnp.asarray(geometry.supports), jnp.asarray(geometry.transports), jnp.diff(jnp.asarray(geometry.expiries))))(front, future)
                densities = np.exp(np.asarray(logs))
                spots = np.asarray(final_marks[:, -1])
                current = observable(dynamics, np.asarray(simulator.initial_densities)[None], np.zeros(1), np.ones(1), elapsed=0)
                surface = observable(dynamics, densities, np.asarray(background), spots, elapsed=1)
                changes = surface - current
                key = f"{label}_{measure}"
                records[key] = {"origin": dynamics.metadata["origin"], "model_identity": dynamics.identity,
                    "paths": paths, "invalid_paths": 0, "conditional_log_iv_variance_trace": float(np.var(changes, axis=0, ddof=1).sum()),
                    "spot_log_return_sd": float(np.std(np.log(spots), ddof=1)),
                    "surface_mean_squared_norm": float(np.sum(changes.mean(axis=0)**2)),
                    "forecast_mse": None, "forecast_mse_note": "No matched next-session observation in this saved single-origin control"}
                observations[f"{key}_surface_changes"] = changes
                observations[f"{key}_spot_log_returns"] = np.log(spots)
                _json(output / "summary.json", records)
                progress.update({"phase": f"{label}-{measure}-complete", "completed": len(records), "total": 4, **records[key]})
    np.savez_compressed(output / "observations.npz", **observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--weighted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=512)
    args = parser.parse_args()
    run(args.baseline, args.weighted, args.output, paths=args.paths)


if __name__ == "__main__":
    main()
