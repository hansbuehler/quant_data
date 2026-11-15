"""Static-arbitrage checks on repository GROMM2's finite fixed-expiry tower."""

from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.bs import bs

from quantdata.biz_calendar import NMS
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import GRID, _hash, _json, build_history
from quantdata.hedging.model_audit.static_arb import NODES, TENORS, call_violations
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    interpolate_generated_implied_volatility,
)


def tower(simulator, state, day: int):
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm.config import DAY

    logsig, boundaries, brownian, log_y = state
    front, loading = jax.vmap(simulator.front, in_axes=(0, 0, None))(boundaries, brownian, jnp.asarray(DAY))
    densities, _ = jax.vmap(simulator.chain, in_axes=(0, 0, 0, None))(logsig, front, loading, day)
    densities, levels = np.asarray(densities), np.asarray(log_y)
    supports = np.asarray(simulator.strikes)
    spots = np.exp(levels) * np.sum(densities[:, day] * supports[day], axis=1)
    return densities, levels, spots


def calls(simulator, densities: np.ndarray, levels: np.ndarray, spots: np.ndarray,
          *, day: int, tenor: int, strikes: np.ndarray) -> np.ndarray:
    row = day + tenor - 1
    if row >= simulator.M:
        raise ValueError("Requested tenor extends beyond the fixed GROMM expiry tower")
    nodes = np.asarray(simulator.strikes)[row][None] * np.exp(levels[:, None]) / spots[:, None]
    variance = np.asarray(simulator.yvar)
    deviation = np.sqrt(variance[day + tenor] - variance[day])
    kernels = bs.price(k=strikes[None, :, None] / nodes[:, None], vol=deviation, sqrtT=1.) * nodes[:, None]
    return np.sum(kernels * densities[:, row, None], axis=-1)


def checks(simulator, state, day: int, *, surface: np.ndarray, history: np.ndarray) -> list[dict]:
    densities, levels, spots = tower(simulator, state, day)
    result = []
    for tenor_float in TENORS:
        tenor = int(tenor_float)
        if day + tenor > simulator.M:
            result.append({"simulation_day": day, "business_dte": tenor, "status": "unsupported"})
            continue
        reference_row = GRID.expiries_y.index(tenor / 252.)
        atm = np.exp(surface[reference_row, GRID.nms_nodes.index(0.)])
        strikes = NMS.from_nms(NODES, tenor / 252., vol=atm)
        values = calls(simulator, densities, levels, spots, day=day, tenor=tenor, strikes=strikes)
        later = calls(simulator, densities, levels, spots, day=day, tenor=tenor + 1, strikes=strikes) if day + tenor + 1 <= simulator.M else None
        zero = calls(simulator, densities, levels, spots, day=day, tenor=tenor, strikes=np.array([0.]))[:, 0]
        finite = np.isfinite(values)
        query = np.broadcast_to(strikes, values.shape)
        inversions: list[dict] = []
        vols = np.full(values.shape, np.nan)
        if finite.any():
            estimated, _ = implied_volatilities_with_validity(expiries_y=np.full(finite.sum(), tenor / 252.),
                normalized_strikes=query[finite], normalized_prices=values[finite],
                is_call=np.ones(finite.sum(), dtype=bool), diagnostics=inversions)
            within_bounds = (values[finite] >= np.maximum(1. - query[finite], 0.) - 1e-10) & (values[finite] <= 1. + 1e-10)
            vols[finite] = np.where(within_bounds & np.isfinite(estimated) & (estimated > 0.), estimated, np.nan)
        historical = np.stack([interpolate_generated_implied_volatility(history,
            query_expiry_y=tenor / 252., query_strikes_over_forward=np.full(len(history), strike), grid=GRID)
            for strike in strikes], axis=1)
        valid = np.isfinite(vols)
        result.append({"simulation_day": day, "business_dte": tenor, "status": "scored",
            **call_violations(strikes, values, longer_prices=later, zero_strike=zero),
            "iv_count": int(valid.sum()), "invalid_iv_count": int((~valid).sum()),
            "outside_historical_range_count": int((valid & ((vols < historical.min(axis=0)) | (vols > historical.max(axis=0)))).sum()),
            "above_300pct_iv_count": int((vols > 3.).sum()), "inversion": inversions})
    return result


def main() -> None:
    import jax
    import jax.numpy as jnp
    from openqdata.models.gromm import estimate, pipeline, simulate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--measure", choices=("P", "Q"), required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.paths < 2 or args.paths % 2 or args.steps not in (1, 5, 10):
        raise ValueError("Require an even number of paths and a supported 1, 5 or 10 day horizon")
    if args.origin not in origins(args.reference):
        raise ValueError("GROMM2 origin is outside the approved comparison dates")
    folder = args.run / f"gromm2/{args.origin}"
    if not (folder / "ESTIMATION_COMPLETE.json").exists():
        raise ValueError("GROMM2 origin has not completed fitting")
    dynamics = estimate.load_dynamics(folder)
    initial = simulate.Model(dynamics, np.array([1]), np.array([1.]))
    risk_price, physical = pipeline.market_price_of_risk(initial, dynamics)
    simulator = simulate.Model(dynamics, np.array([1]), np.array([1.]), lam=risk_price if args.measure == "P" else None)
    args.output.mkdir(parents=True, exist_ok=True)
    panel = PublicHedgingPanel(Path(os.environ["QUANTDATA_ROOT"]) / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, args.output)
    index = history.dates.index(date.fromisoformat(args.origin))
    training = np.flatnonzero((history.sessions_from_previous_observation == 1) & (np.arange(len(history.dates)) <= index))[-252:]
    if len(training) != 252:
        raise ValueError("Missing 252 causal one-session reference observations")
    state = simulator.initial(args.paths)
    key = jax.random.PRNGKey(20_261_112)
    started = time.monotonic()
    results = []
    for step in range(args.steps):
        health_check()
        state, _, _, _ = simulator.day(state, jnp.int32(step), jax.random.fold_in(key, step), 8)
        state = jax.block_until_ready(state)
        if step + 1 in (1, 5, 10):
            results.extend(checks(simulator, state, step + 1, surface=history.log_implied_volatilities[index],
                history=history.log_implied_volatilities[training]))
        elapsed = time.monotonic() - started
        print(f"GROMM2 static {args.measure} {args.origin} completed={step+1}/{args.steps} elapsed={elapsed:.1f}s eta={elapsed*(args.steps-step-1)/(step+1):.1f}s", flush=True)
    result = {"model": f"repository GROMM2-{args.measure}", "origin": args.origin, "paths": args.paths,
        "physical": physical, "dynamics_sha256": _hash(folder / "dynamics.npz"),
        "reference_sha256": _hash(args.reference / "observable_reference.json"), "results": results,
        "unsupported_horizons": [125, 259], "reason": "fixed tower has no expiry replenishment"}
    _json(args.output / f"static/{args.measure}/{args.origin}.json", result)


if __name__ == "__main__":
    main()
