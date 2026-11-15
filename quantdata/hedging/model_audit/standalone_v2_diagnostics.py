"""Complete the standalone V2 fixed-contract and recursive-surface audit."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import NMS
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedging.model_audit.implied import (
    implied_volatilities_with_validity,
    summarize_inversions,
)
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.metrics import quote_scores
from quantdata.hedging.model_audit.persistence import path_acf
from quantdata.hedging.model_audit.rolling_versions import origins
from quantdata.hedging.model_audit.run import COMMON_INDICES, GRID, _hash, _json, build_history
from quantdata.hedging.model_audit.standalone_v2_forecast import common_strikes, make_pricer
from quantdata.hedging.model_audit.static_arb import (
    HORIZONS,
    NODES,
    TENORS,
    call_violations,
    iv_stability,
    next_tenor,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    interpolate_generated_implied_volatility,
)


def recursive_states(current: np.ndarray, fit, *, paths: int, horizons: tuple[int, ...],
                     long_paths: int, long_steps: int, seed: int) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Iterate the saved one-day increment law while carrying its VAR factor scores."""
    if not 0 < long_paths <= paths or not horizons or tuple(sorted(set(horizons))) != horizons or long_steps < 1:
        raise ValueError("Invalid recursive V2 population or horizons")
    random = np.random.default_rng(seed)
    draws = random.multivariate_normal(np.zeros(fit.cov.shape[0]), fit.cov,
        size=(max(max(horizons), long_steps), paths), check_valid="raise", tol=1e-9)
    scores = np.broadcast_to(fit.scores[-1], (paths, len(fit.scores[-1]))).copy()
    sum_scores = np.zeros_like(scores)
    sum_spot = np.zeros(paths)
    long_states = np.empty((long_steps, long_paths, len(current)), dtype=float)
    snapshots = {}
    for step, innovations in enumerate(draws, start=1):
        returns = fit.r_mean + innovations[:, -1]
        scores = scores @ fit.A.T + innovations[:, :-1]
        sum_scores += scores
        sum_spot += returns
        if step <= long_steps:
            long_states[step - 1] = current + step * fit.alpha + sum_spot[:long_paths, None] * fit.beta + sum_scores[:long_paths] @ fit.directions.T
        if step in horizons:
            snapshots[step] = current + step * fit.alpha + sum_spot[:, None] * fit.beta + sum_scores @ fit.directions.T
    return snapshots, long_states, fit.r_mean + draws[0, :, -1]


def make_cash_pricer(configuration):
    import jax
    import jax.numpy as jnp
    from openqdata.models._dysanos import dlv
    from openqdata.models.dysanos2 import surface

    @jax.jit
    def prices(states, expiry, strikes):
        nodes, density, scale = surface.decode(states, configuration.surface)
        return dlv.price_at(nodes, density, scale, jnp.asarray(configuration.surface.expiries),
                            expiry, strikes[..., None], configuration.surface.eta)[..., 0]

    return prices


def invert_capped(prices: np.ndarray, strikes: np.ndarray, times: np.ndarray,
                  inversions: list[dict]) -> np.ndarray:
    """Keep finite library boundary estimates, while recording repricing failures."""
    prices, strikes, times = np.broadcast_arrays(prices, strikes, times)
    finite = np.isfinite(prices) & np.isfinite(strikes) & np.isfinite(times) & (strikes > 0.) & (times > 0.)
    intrinsic = np.maximum(1. - strikes, 0.)
    eligible = finite & (prices >= intrinsic - 1e-10) & (prices <= 1. + 1e-10)
    result = np.full(prices.shape, np.nan)
    if eligible.any():
        vols, _ = implied_volatilities_with_validity(expiries_y=times[eligible], normalized_strikes=strikes[eligible],
            normalized_prices=prices[eligible], is_call=np.ones(int(eligible.sum()), dtype=bool), diagnostics=inversions)
        result[eligible] = np.where(np.isfinite(vols) & (vols > 0.), vols, np.nan)
    return result


def observable_capped(natives: np.ndarray, *, decode, prices, times: np.ndarray, nms: np.ndarray,
                      batch_size: int) -> tuple[np.ndarray, dict]:
    """Keep finite boundary estimates and retain invalid price/IV observations as NaN."""
    import jax.numpy as jnp

    output = np.full((len(natives), len(times) * len(nms)), np.nan)
    inversions = []
    for begin in range(0, len(natives), batch_size):
        states = natives[begin:begin + batch_size]
        finite_states = np.isfinite(states).all(axis=1)
        if not finite_states.any():
            continue
        nodes, density, scale = decode(jnp.asarray(states[finite_states]))
        shape = (int(finite_states.sum()), len(times), 1)
        atm_prices = np.asarray(prices(nodes, density, scale, jnp.ones(shape)))[..., 0]
        atm = invert_capped(atm_prices, np.ones_like(atm_prices), np.broadcast_to(times, atm_prices.shape), inversions)
        valid_atm = np.isfinite(atm).all(axis=1)
        if not valid_atm.any():
            continue
        nodes, density, scale = (entry[valid_atm] for entry in (nodes, density, scale))
        strikes = common_strikes(atm[valid_atm], times, nms)
        values = np.asarray(prices(nodes, density, scale, jnp.asarray(strikes)))
        vols = invert_capped(values, strikes, np.broadcast_to(times[None, :, None], strikes.shape), inversions)
        destinations = np.flatnonzero(finite_states)[valid_atm] + begin
        output[destinations] = np.log(vols.reshape(len(destinations), -1))
    diagnostics = summarize_inversions(inversions) if inversions else {"count": 0}
    diagnostics["unavailable_surface_values"] = int(np.sum(~np.isfinite(output)))
    return output, diagnostics


def quote_forecast(origin: str, endpoint: str, current: np.ndarray, following: np.ndarray,
                   spot_returns: np.ndarray, *, dividend: float, licensed: Path, market: Path,
                   cash_prices, batch_size: int) -> dict:
    from datetime import date

    import jax.numpy as jnp

    quotes = pd.read_parquet(market / f"{origin}.parquet")
    quotes = quotes.loc[(quotes.bdte > 1. + 1e-6) & (quotes.bdte <= 5. + 1e-6) & (quotes.nms.abs() <= 1.5)].copy()
    quotes = quotes.loc[np.where(quotes.strike >= quotes.fwd, quotes.option_type == "call", quotes.option_type == "put")].reset_index(drop=True)
    curve = licensed_curve(licensed, date.fromisoformat(origin))
    spots = curve.spot * np.exp(spot_returns - dividend)
    if not np.isfinite(spots).all() or np.any(spots <= 0):
        raise ValueError("V2 generated a nonfinite or nonpositive spot")
    rolled = curve.roll(1., new_spot=curve.spot)
    strikes = quotes.strike.to_numpy(dtype=float)
    calls = quotes.option_type.to_numpy() == "call"

    def prices(states, spot, expiry_days, carry):
        fwd, _, discount = carry.at(expiry_days)
        forwards = spot[:, None] / carry.spot * fwd[None, :]
        normalized = strikes[None, :] / forwards
        call = np.asarray(cash_prices(jnp.asarray(states), jnp.asarray(expiry_days / 252.), jnp.asarray(normalized)))
        return discount[None, :] * forwards * np.where(calls[None, :], call, call + normalized - 1.)

    current_prices = prices(current[None], np.array([curve.spot]), quotes.bdte.to_numpy(dtype=float), curve)[0]
    forecast = np.zeros(len(quotes))
    for start in range(0, len(following), batch_size):
        end = min(start + batch_size, len(following))
        forecast += prices(following[start:end], spots[start:end], quotes.bdte.to_numpy(dtype=float) - 1., rolled).sum(axis=0) / len(following)
    keys = ["expiry", "strike", "option_type", "is_am"]
    quotes["forecast"] = forecast
    next_quotes = pd.read_parquet(market / f"{endpoint}.parquet")
    joined = quotes.merge(next_quotes[keys + ["observed_bid", "observed_ask"]], on=keys,
        how="left", suffixes=("", "_next"), validate="one_to_one")
    return {"paths": len(following), "endpoint": endpoint, "origin_spot": curve.spot,
        "scope": "OTM fixed cash contracts, 1<BDTE<=5, absolute NMS<=1.5; origin-only deterministic carry",
        "reconstruction": quote_scores(current_prices, quotes.observed_bid, quotes.observed_ask),
        "next_day": quote_scores(joined.forecast, joined.observed_bid_next, joined.observed_ask_next)}


def static_checks(snapshots: dict[int, np.ndarray], *, history: np.ndarray, origin_surface: np.ndarray,
                  configuration, batch_size: int) -> list[dict]:
    import jax.numpy as jnp

    model_days = np.asarray(configuration.surface.expiry_days)
    indices = sorted({int(np.flatnonzero(model_days == day)[0]) for tenor in TENORS
                      for day in (tenor, next_tenor(tenor, model_days)) if day is not None})
    decode, prices = make_pricer(configuration, np.asarray(indices))
    positions = {model_days[index]: position for position, index in enumerate(indices)}
    results = []
    for horizon, states in snapshots.items():
        print(f"V2 static day={horizon} paths={len(states)}", flush=True)
        for tenor in TENORS:
            row = GRID.expiries_y.index(tenor / 252.)
            atm = np.exp(origin_surface[row, GRID.nms_nodes.index(0.)])
            strikes = NMS.from_nms(NODES, tenor / 252., vol=atm)
            following = next_tenor(tenor, model_days)
            calls, later, zero = [], [], []
            for start in range(0, len(states), batch_size):
                selected = states[start:start + batch_size]
                nodes, density, scale = decode(jnp.asarray(selected))
                tiled = jnp.broadcast_to(jnp.asarray(strikes), (len(selected), len(indices), len(strikes)))
                values = np.asarray(prices(nodes, density, scale, tiled))
                calls.append(values[:, positions[tenor]])
                if following is not None:
                    later.append(values[:, positions[following]])
                zero_values = np.asarray(prices(nodes, density, scale, jnp.zeros((len(selected), len(indices), 1))))
                zero.append(zero_values[:, positions[tenor], 0])
            calls = np.concatenate(calls)
            historical = np.stack([interpolate_generated_implied_volatility(history,
                query_expiry_y=tenor / 252., query_strikes_over_forward=np.full(len(history), strike), grid=GRID)
                for strike in strikes], axis=1)
            results.append({"simulation_day": horizon, "business_dte": tenor, "next_business_dte": following,
                **call_violations(strikes, calls, longer_prices=np.concatenate(later) if later else None, zero_strike=np.concatenate(zero)),
                **iv_stability(calls, strikes, tenor, historical)})
    return results


def main() -> None:
    from openqdata.models._dysanos import dlv, precision
    from openqdata.models.dysanos2 import config, evaluate, factors, surface
    from openqdata.models.dysanos2.config import Config
    from openqdata.models.dysanos2.evaluate import driftless_returns
    from openqdata.models.dysanos2.factors import coordinate_weights, fit_factors

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("states", "observations", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-paths", type=int, default=32)
    parser.add_argument("--long-steps", type=int, default=115)
    parser.add_argument("--origins", nargs="+")
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu" or not 0 < args.long_paths <= args.paths or args.batch_size < 1:
        raise ValueError("Use CPU and valid path/batch sizes")
    selected = origins(args.reference)
    if args.origins:
        if not set(args.origins) <= set(selected):
            raise ValueError("Origins differ from the approved reference")
        selected = tuple(args.origins)
    cfg = Config()
    if cfg.factors.mode != "density":
        raise ValueError("Only the standalone density-mode V2 is supported")
    forecast_protocol_path = args.reference.parent / "standalone-v2-forecast-nms-20261114/PROTOCOL.json"
    forecast_protocol = json.loads(forecast_protocol_path.read_text())
    if (_hash(args.states) != forecast_protocol["states_sha256"]
            or _hash(args.observations) != forecast_protocol["observations_sha256"]
            or json.loads(cfg.to_json()) != forecast_protocol["configuration"]
            or ("float64" if precision.X64 else "float32") != forecast_protocol["jax_precision"]):
        raise ValueError("Standalone V2 inputs, configuration or precision differ from the scored one-day law")
    for module in (config, evaluate, factors, surface, dlv, precision):
        if _hash(Path(module.__file__)) != forecast_protocol["standalone_source_sha256"][Path(module.__file__).name]:
            raise ValueError(f"Standalone V2 source changed: {module.__file__}")
    with np.load(args.states, allow_pickle=False) as data:
        states = {name: data[name].copy() for name in ("U", "mass", "dates", "spot", "div_1d")}
    state_dates = states["dates"].astype(str)
    returns = driftless_returns(states["spot"], states["div_1d"], state_dates)
    observations = pd.read_parquet(args.observations)
    licensed = Path(observations.source_cache_file.iloc[0]).parent.parent
    market = Path(os.environ["QUANTDATA_ROOT"]) / "experiments/hedge2/market-full-20261106T0001/market/days"
    args.output.mkdir(parents=True, exist_ok=True)
    panel = PublicHedgingPanel(Path(os.environ["QUANTDATA_ROOT"]) / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, args.output)
    with np.load(args.reference / "frozen_observable_pca.npz", allow_pickle=False) as data:
        basis, nms, times = (data[name].copy() for name in ("basis", "nms_nodes", "expiries_y"))
    model_days = np.asarray(cfg.surface.expiry_days)
    indices = np.array([np.flatnonzero(model_days == round(day * 252.))[0] for day in times])
    decode, prices = make_pricer(cfg, indices)
    cash_prices = make_cash_pricer(cfg)
    identity = {"schema": "standalone-v2-complete-diagnostics-v1", "origins": list(selected),
        "paths": args.paths, "batch_size": args.batch_size, "long_paths": args.long_paths,
        "long_steps": args.long_steps, "horizons": HORIZONS, "seed": 20_261_112,
        "states_sha256": _hash(args.states), "reference_sha256": _hash(args.reference / "observable_reference.json"),
        "source_sha256": _hash(Path(__file__)), "one_day_protocol_sha256": _hash(forecast_protocol_path)}
    identity_path = args.output / "PROTOCOL.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != json.loads(json.dumps(identity)):
        raise ValueError("V2 diagnostics run identity changed")
    _json(identity_path, identity)
    started = time.monotonic()
    for position, origin in enumerate(selected):
        health_check()
        destination = args.output / f"daily/{origin}.json"
        if destination.exists():
            continue
        index = int(np.flatnonzero(state_dates == origin)[0])
        hindex = history.dates.index(pd.Timestamp(origin).date())
        training = np.flatnonzero((history.sessions_from_previous_observation == 1) & (np.arange(len(history.dates)) <= hindex))[-252:]
        if len(training) != 252:
            raise ValueError("Missing exact-session historical reference")
        window = cfg.factors.window
        fit = fit_factors(states["U"][index-window:index+1], returns[index-window:index],
            coordinate_weights(states["mass"][index], cfg.surface, cfg.factors), cfg.factors)
        snapshots, long_states, spot_shocks = recursive_states(states["U"][index], fit, paths=args.paths,
            horizons=HORIZONS, long_paths=args.long_paths, long_steps=args.long_steps, seed=20_261_112)
        baseline = json.loads((args.reference / f"daily/FuNVol/{origin}.json").read_text())
        result = {"model": "standalone DYSANOS v2 density-P", "origin": origin, "paths": args.paths,
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(fit.A))))}
        try:
            result["quotes"] = quote_forecast(origin, baseline["endpoint"], states["U"][index], snapshots[1], spot_shocks,
                dividend=float(states["div_1d"][index]), licensed=licensed, market=market,
                cash_prices=cash_prices, batch_size=args.batch_size)
            result["static"] = static_checks(snapshots, history=history.log_implied_volatilities[training],
                origin_surface=history.log_implied_volatilities[hindex], configuration=cfg, batch_size=args.batch_size)
            _json(args.output / f"partial/{origin}.json", result)
            values, inversion = observable_capped(long_states.reshape(-1, long_states.shape[-1]), decode=decode,
                prices=prices, times=times, nms=nms, batch_size=args.batch_size)
            values = values.reshape(args.long_steps, args.long_paths, -1).transpose(1, 0, 2)
            history_common = history.log_implied_volatilities[training][:, :, COMMON_INDICES].reshape(len(training), -1)
            valid_paths = np.isfinite(values).all(axis=(1, 2))
            result["long"] = {"paths": args.long_paths, "steps": args.long_steps,
                "valid_paths": int(valid_paths.sum()), "invalid_path_rate": float(np.mean(~valid_paths)),
                "extreme_surface_rate": float(np.mean(np.any(values[valid_paths] > np.log(3.), axis=(1, 2)))) if valid_paths.any() else None,
                "observable_acf": path_acf(values[valid_paths] @ basis) if valid_paths.any() else None,
                "historical_observable_acf": path_acf((history_common @ basis)[None]), "iv_inversion": inversion}
            if valid_paths.any():
                result["status"] = "complete"
            else:
                result.update(status="diagnostic_failure", exact_error="No recursive paths have finite observable IVs")
        except (ValueError, FloatingPointError, RuntimeError) as error:
            result.update(status="diagnostic_failure", exact_error=f"{type(error).__name__}: {error}")
            print(json.dumps(result), flush=True)
        _json(destination, result)
        processed = [json.loads(path.read_text()) for path in sorted((args.output / "daily").glob("*.json"))]
        _json(args.output / "SUMMARY.json", {"processed": len(processed),
            "complete": sum(item["status"] == "complete" for item in processed),
            "failures": [{"origin": item["origin"], "error": item["exact_error"]} for item in processed if item["status"] != "complete"]})
        elapsed = time.monotonic() - started
        print(f"standalone V2 full diagnostics={position+1}/{len(selected)} elapsed={elapsed:.1f}s eta={elapsed*(len(selected)-position-1)/(position+1):.1f}s", flush=True)


if __name__ == "__main__":
    main()
