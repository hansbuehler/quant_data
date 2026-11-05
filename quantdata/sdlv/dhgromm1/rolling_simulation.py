"""Generate expiry-or-day-20 returns for real GROMM option trade cohorts."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock

from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_prices,
    one_day_physical_substepped,
    one_day_risk_neutral_substepped,
)
from quantdata.sdlv.daily.wide_audit import AuditMonitor, resource_health, sha256
from quantdata.sdlv.dhgromm1.contracts import RETURN_CONVENTION, SCENARIO_SCHEMA

jax.config.update("jax_enable_x64", True)
ITM_LEVELS = (-2., -1.5, -1., -.5, -.25)
CENTERED_LEVELS = (-1., -.5, 0., .5, 1.)


def zero_strike_call_prices(support: jax.Array, densities: jax.Array,
                            background_level: jax.Array) -> jax.Array:
    """Zero-strike calls equal the model's undiscounted terminal spot mean."""
    return background_level[..., None] * jnp.einsum("...ti,i->...t", densities, support)


def generate_chunk(*, calibration: Path, paths: int, batch_size: int, substeps: int,
                   seed: np.random.SeedSequence, strike_levels: tuple[float, ...] = ITM_LEVELS,
                   include_forwards: bool = True,
                   market_price_of_risk: np.ndarray | None = None,
                   state_sink: Callable[[int, int, int, RiskNeutralState], None] | None = None,
                   market_sink: Callable[..., None] | None = None,
                   opening_sink: Callable[..., None] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired paths with returns for each separately opened trade.

    Five standardized log-moneyness levels are translated to cash strikes at
    each day's spot, at each of 1/5/10 days. Defaults are strictly ITM calls.
    Every option is a call. The separate terminal call expires at t=20 and
    resets its ATM strike to the current spot at each trading day. Each trade
    pays (S_expiry/S_open-K)^+ minus its normalized opening price, with contracts expiring after day 20 instead
    liquidated at their day-20 model marks. Spot trades are valued at day 20.
    """
    if paths < 2 or paths % 2 or batch_size < 2 or batch_size % 2 or substeps < 1:
        raise ValueError("paths and batches must be even; substeps must be positive")
    if len(strike_levels) != 5 or not np.isfinite(strike_levels).all():
        raise ValueError("exactly five finite strike levels are required")
    with np.load(calibration / "diffusion.npz", allow_pickle=False) as data:
        support = jnp.asarray(data["support"], dtype=jnp.float64)
        initial_tower = jnp.asarray(data["density_tower"], dtype=jnp.float64)
        initial_dlv = jnp.asarray(data["dlvsq"][1:], dtype=jnp.float64)
        loadings = jnp.asarray(data["annualized_loadings"], dtype=jnp.float64)
    if loadings.shape[-1] != 8:
        raise ValueError("expected eight PCA drivers")
    if market_price_of_risk is not None:
        market_price_of_risk = np.asarray(market_price_of_risk, dtype=np.float64)
        if market_price_of_risk.shape != (8,) or not np.isfinite(market_price_of_risk).all():
            raise ValueError("P simulation requires eight finite fitted Brownian risk prices")
    step_function = one_day_risk_neutral_substepped if market_price_of_risk is None else one_day_physical_substepped
    step = jax.jit(step_function, static_argnames=("expiry_dt", "time_power"))
    pricer = jax.jit(black_mixture_call_prices)
    instrument_count = 21 if include_forwards else 17
    prices = np.zeros((paths, 21, instrument_count), dtype=np.float64)
    returns = np.zeros((paths, 20, instrument_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    levels = jnp.asarray(strike_levels)
    for start in range(0, paths, batch_size):
        valid = min(batch_size, paths - start)
        shocks = rng.standard_normal((20, substeps, batch_size // 2, 8))
        shocks = np.stack((shocks, -shocks), axis=3).reshape(20, substeps, batch_size, 8)
        black = rng.standard_normal((20, batch_size // 2))
        black = np.stack((black, -black), axis=2).reshape(20, batch_size)
        state = RiskNeutralState(jnp.broadcast_to(initial_tower[0], (batch_size, len(support))),
                                 jnp.broadcast_to(initial_dlv, (batch_size, *initial_dlv.shape)),
                                 jnp.ones(batch_size, dtype=jnp.float64))
        tower = jnp.broadcast_to(initial_tower, (batch_size, *initial_tower.shape))
        spot = jnp.ones(batch_size, dtype=jnp.float64)
        spot_path = np.empty((batch_size, 21), dtype=np.float64)
        spot_path[:, 0] = 1.0
        cohort_strikes = np.empty((batch_size, 20, 3, 5), dtype=np.float64)
        cohort_days = np.empty((20, 3), dtype=np.int64)
        for day in range(20):
            if market_sink is not None:
                market_sink(day, start, valid, support, tower, state.background_level, spot)
            # Observe the opening state, before drawing/using this day's shocks.
            # The optional observer must not mutate the simulation state.
            if state_sink is not None:
                state_sink(day, start, valid, state)
            longest = min(10, 25 - day)
            maturities = jnp.asarray([1, 5, longest])
            scale = 0.20 * jnp.sqrt(maturities[:, None] / 252.0)
            terminal = 20 - day
            strikes = spot[:, None, None] * jnp.exp(levels[None, None, :] * scale[None, :, :])
            openings = pricer(support, tower[:, jnp.array([0, 4, longest - 1]), :], strikes,
                              state.background_level, maturities)
            terminal_open = pricer(support, tower[:, terminal - 1:terminal, :], spot[:, None, None],
                                   state.background_level, jnp.asarray([terminal]))[:, 0, 0]
            opening = np.column_stack((np.asarray(spot), np.asarray(terminal_open),
                                       np.asarray(openings).reshape(batch_size, 15)))
            if include_forwards:
                rows = jnp.asarray([terminal - 1, 0, 4, longest - 1])
                forwards = zero_strike_call_prices(support, tower[:, rows, :], state.background_level)
                opening = np.column_stack((opening, np.asarray(forwards)))
            cohort_strikes[:, day] = np.asarray(strikes)
            cohort_days[day] = np.asarray(maturities) + day
            if opening_sink is not None:
                opening_sink(day, start, valid, opening)
            measure_args = () if market_price_of_risk is None else (jnp.asarray(market_price_of_risk),)
            result = step(support, state, loadings[:state.future_dlvsq.shape[-2]], *measure_args, jnp.asarray(shocks[day]),
                          jnp.asarray(black[day]), expiry_dt=1.0 / 252.0, time_power=1.0)
            next_spot = result.spot
            next_tower = result.density_tower
            if not np.isfinite(opening).all() or np.any(opening < 0) or np.any(opening[:, 0] <= 0):
                raise FloatingPointError(f"invalid rolling-option marks at day {day}")
            prices[start:start + valid, day] = opening[:valid]
            spot_path[:, day + 1] = np.asarray(next_spot)
            spot, tower, state = next_spot, next_tower, result.state
        prices[start:start + valid, 20, 0] = np.asarray(spot)[:valid]
        for day in range(20):
            option_closes = np.empty((batch_size, 3, 5), dtype=np.float64)
            for tenor in range(3):
                expiry = int(cohort_days[day, tenor])
                strike = cohort_strikes[:, day, tenor]
                if expiry <= 20:
                    option_closes[:, tenor] = np.maximum(spot_path[:, expiry, None] - strike, 0.0)
                else:
                    remaining = expiry - 20
                    option_closes[:, tenor] = np.asarray(pricer(
                        support, tower[:, remaining - 1:remaining, :], jnp.asarray(strike[:, None, :]),
                        state.background_level, jnp.asarray([remaining])))[:, 0]
            close = np.column_stack((spot_path[:, 20], np.maximum(spot_path[:, 20] - spot_path[:, day], 0.0),
                                     option_closes.reshape(batch_size, 15)))
            if include_forwards:
                forward_closes = np.empty((batch_size, 4), dtype=np.float64)
                forward_closes[:, 0] = spot_path[:, 20]
                for tenor in range(3):
                    expiry = int(cohort_days[day, tenor])
                    if expiry <= 20:
                        forward_closes[:, tenor + 1] = spot_path[:, expiry]
                    else:
                        remaining = expiry - 20
                        forward_closes[:, tenor + 1] = np.asarray(zero_strike_call_prices(
                            support, tower[:, remaining - 1:remaining, :], state.background_level))[:, 0]
                close = np.column_stack((close, forward_closes))
            close = close[:valid]
            opening = prices[start:start + valid, day]
            if not np.isfinite(close).all() or np.any(close < 0):
                raise FloatingPointError(f"invalid expiry/day-20 closing marks for day-{day} trades")
            # Quantity of a floating contract, NOT cash invested in a premium.
            # This remains defined for an option with zero opening premium.
            returns[start:start + valid, day] = (close - opening) / opening[:, :1]
    payoff = np.abs(prices[:, 20, 0] / prices[:, 10, 0] - 1)
    return prices, returns, payoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths-per-sample", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--strike-menu", choices=["itm", "centered"], default="itm")
    parser.add_argument("--without-forwards", action="store_true")
    args = parser.parse_args()
    if any(x < 2 or x % 2 for x in (args.paths_per_sample, args.chunk_size, args.batch_size)) or args.substeps < 1:
        parser.error("paths and batches must be even and substeps positive")
    output = args.output.resolve()
    root = Path(os.environ["QUANTDATA_ROOT"]).resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be below the external project data root")
    checks = resource_health()
    if any(item["health"] in {"STOP", "UNAVAILABLE", "ERROR"} for item in checks):
        raise RuntimeError("resource manager blocks launching scenario generation")
    if any(item["health"] != "OK" for item in checks):
        print(json.dumps(dict(event="resource_warning", checks=checks)), flush=True)
    calibration = args.calibration.resolve()
    strike_levels = ITM_LEVELS if args.strike_menu == "itm" else CENTERED_LEVELS
    include_forwards = not args.without_forwards
    instrument_count = 21 if include_forwards else 17
    identity = dict(schema=SCENARIO_SCHEMA, return_convention=RETURN_CONVENTION, calibration=str(calibration),
        calibration_sha256=sha256(calibration / "diffusion.npz"), source_sha256=sha256(Path(__file__)),
        paths_per_sample=args.paths_per_sample, chunk_size=args.chunk_size, batch_size=args.batch_size,
        substeps=args.substeps, seed=args.seed, strikes="exp(z*0.20*sqrt(dte/252))*spot",
        z=list(strike_levels), strike_menu=args.strike_menu, instruments=instrument_count,
        dte="[1,5,min(10,25-day)]", option_type="call",
        terminal_call="daily ATM strike reset; fixed day-20 expiration", samples=["train", "evaluation"])
    identity["zero_strike_calls"] = ["terminal_day20", "dte1", "dte5", "longest_dte"] if include_forwards else []
    identity["instrument_order"] = (["spot", "terminal_atm_call"]
        + [f"{tenor}_call_z{z:g}" for tenor in ("dte1", "dte5", "longest_dte") for z in strike_levels]
        + (["terminal_forward", "dte1_forward", "dte5_forward", "longest_dte_forward"] if include_forwards else []))
    identity["return_definition"] = "(closing_mark-opening_mark)/S_open; actual expiry, or day-20 mark if later; no premium division"
    identity["gain_definition"] = "sum over opening dates and instruments: floating_contract_quantity * floating_contract_gain"
    output.mkdir(parents=True, exist_ok=True)
    monitor = AuditMonitor(output)
    monitor.thread.start()
    started = time.monotonic()
    try:
        with FileLock(output / "scenarios.lock", timeout=0):
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError("scenario identity differs; choose a new directory")
            identity_path.write_text(json.dumps(identity, indent=2) + "\n")
            completed = 0
            total = args.paths_per_sample * 2
            for sample_index, sample in enumerate(identity["samples"]):
                folder = output / sample
                folder.mkdir(exist_ok=True)
                for start in range(0, args.paths_per_sample, args.chunk_size):
                    count = min(args.chunk_size, args.paths_per_sample - start)
                    path = folder / f"{start:06d}.npz"
                    monitor.stage(f"generate-{sample}", completed, total)
                    if not path.exists():
                        prices, returns, payoff = generate_chunk(calibration=calibration, paths=count,
                            batch_size=args.batch_size, substeps=args.substeps,
                            seed=np.random.SeedSequence([args.seed, sample_index, start]),
                            strike_levels=strike_levels, include_forwards=include_forwards)
                        temporary = path.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, prices=prices, returns=returns, payoff=payoff)
                        temporary.replace(path)
                    with np.load(path, allow_pickle=False) as saved:
                        if (saved["returns"].shape != (count, 20, instrument_count)
                                or saved["prices"].shape != (count, 21, instrument_count)):
                            raise ValueError(f"invalid saved scenario geometry: {path}")
                        minimum_call_premium = float(saved["prices"][:, :20, 1:17].min())
                        maximum_call_return = float(saved["returns"][:, :, 1:17].max())
                    completed += count
                    elapsed = time.monotonic() - started
                    print(json.dumps(dict(stage="scenario_progress", completed=completed, total=total,
                          elapsed_seconds=elapsed, eta_seconds=elapsed * (total - completed) / completed,
                          minimum_call_premium=minimum_call_premium, maximum_call_return=maximum_call_return)), flush=True)
            monitor.stage("complete", total, total)
            (output / "complete.json").write_text(json.dumps(dict(status="complete", **identity), indent=2) + "\n")
    finally:
        monitor.done.set()


if __name__ == "__main__":
    main()
