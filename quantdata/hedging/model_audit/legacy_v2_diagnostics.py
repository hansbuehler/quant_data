"""Read-only diagnostics of completed independent STRT-16/tail-diffusion fits."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import NMS
from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import FittedDynamics
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_market import cash_europeans_batch
from quantdata.hedging.model_audit.legacy_queue import health_check
from quantdata.hedging.model_audit.legacy_v2 import checked_native
from quantdata.hedging.model_audit.metrics import linear_stability, quote_scores
from quantdata.hedging.model_audit.projected_law import ProjectedLaw
from quantdata.hedging.model_audit.rolling_versions import INITIALIZATION_POLICY, origins
from quantdata.hedging.model_audit.run import GRID, _hash, _json
from quantdata.hedging.model_audit.static_arb import (
    HORIZONS,
    NODES,
    TENORS,
    call_violations,
    iv_stability,
    next_tenor,
)
from quantdata.hedging.model_audit.tail import TailSampler
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    interpolate_generated_implied_volatility,
)


def load_origin(root: Path, origin: str) -> tuple[DailyState, ProjectedLaw, dict, dict]:
    directory = root / f"fits/dysanos2/{origin}"
    complete = json.loads((directory / "COMPLETE.json").read_text())
    if complete["dynamics_sha256"] != _hash(directory / "dynamics.npz"):
        raise ValueError(f"Completed v2 dynamics changed: {origin}")
    protocol = json.loads((directory / "PROTOCOL.json").read_text())
    if protocol["factor_rank"] != 16 or protocol["initialization"] != INITIALIZATION_POLICY:
        raise ValueError("V2 diagnostic requires the completed independent STRT-16 fit")
    native_path = next(Path(name) for name in protocol["input_hashes"] if Path(name).stem == origin)
    if _hash(native_path) != protocol["input_hashes"][str(native_path)]:
        raise ValueError(f"Origin native state changed: {native_path}")
    native, _, support = checked_native(native_path, date.fromisoformat(origin))
    config = DlvConfig(**{field.name: protocol["geometry"][field.name]
        for field in fields(DlvConfig) if field.name in protocol["geometry"]})
    state = DailyState(day=date.fromisoformat(origin), native=native, reference_support=support, config=config)
    fit = load_dynamics(directory / "dynamics.npz")
    if fit.initialisation_date is not None or str(fit.training_end) != origin:
        raise ValueError("Noncausal or warm-started dynamics cannot enter this comparison")
    with np.load(directory / "projection.npz", allow_pickle=False) as saved:
        law = ProjectedLaw(dynamics=fit, encoder=saved["encoder"], decoder=saved["decoder"],
            spot_loading=saved["native_spot_loading"])
    return state, law, json.loads((directory / "tail.json").read_text()), protocol


def advance_scores(scores: np.ndarray, fit: FittedDynamics, factor_spot_loading: np.ndarray,
                   shocks: np.ndarray) -> np.ndarray:
    return scores @ fit.ar_matrix.T + fit.ar_intercept + shocks[:, 1:] + shocks[:, :1] * factor_spot_loading


def native_snapshots(state: DailyState, law: ProjectedLaw, tail: dict, *, paths: int,
                     horizons: tuple[int, ...], seed: int,
                     retain_paths: int | None = None) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if paths < 2 or not horizons or tuple(sorted(set(horizons))) != horizons or horizons[0] < 1:
        raise ValueError("Use positive ordered horizons and at least two paths")
    retained = paths if retain_paths is None else retain_paths
    if not 0 < retained <= paths:
        raise ValueError("Retained paths must be a positive subset of the simulated population")
    fit = law.dynamics
    scores0 = (state.native @ law.encoder - fit.state_mean) @ fit.directions
    scores = np.broadcast_to(scores0, (paths, len(scores0))).copy()
    anchor = state.native - law.factor_decoder @ scores0
    accumulated_spot = np.zeros(paths)
    sampler = TailSampler(fit, tail, paths=paths, seed=seed)
    snapshots = {}
    first_shocks = None
    for step in range(1, max(horizons) + 1):
        shocks = sampler.draw()
        if step == 1:
            first_shocks = shocks.copy()
        scores = advance_scores(scores, fit, law.factor_spot_loading, shocks)
        accumulated_spot += shocks[:, 0]
        if step in horizons:
            snapshots[step] = anchor + scores[:retained] @ law.factor_decoder.T + accumulated_spot[:retained, None] * law.perpendicular_spot_loading
    return snapshots, first_shocks


def backbone(fit: FittedDynamics, tail: dict) -> dict:
    conditional = np.asarray(tail["conditional_factor"])
    loading = np.asarray(tail["conditional_spot_loading"]) + fit.spot_state_loading @ fit.directions
    variance = float(fit.joint_innovation_covariance[0, 0])
    innovations = variance * np.outer(loading, loading)
    innovations += float(tail["fit"]["config"]["unconditional_scale"]) * conditional @ conditional.T
    result = linear_stability(fit.ar_matrix, innovations)
    result.update(scope="frozen floating-volatility PCA/VAR backbone; spot level is not tested for stationarity",
        unconstrained_spectral_radius=fit.unconstrained_spectral_radius,
        initialisation_date=None, tail_optimizer_message=tail["fit"]["message"])
    return result


def static_origin(state: DailyState, snapshots: dict[int, np.ndarray], *, history: np.ndarray,
                   origin_surface: np.ndarray, paths: int) -> list[dict]:
    rows = []
    tenors = np.asarray(state.config.business_dtes)
    for horizon, native in snapshots.items():
        health_check()
        decoded = decode_batch(state, native)
        packed = native.reshape(paths, len(tenors), state.config.num_inner_strikes + 1)
        forward_clips = (packed[:, :, -1] < np.log(state.config.min_local_vol)) | (packed[:, :, -1] > np.log(state.config.max_local_vol))
        for tenor in TENORS:
            row = GRID.expiries_y.index(tenor / 252.)
            atm = float(np.exp(origin_surface[row, GRID.nms_nodes.index(0.)]))
            strikes = NMS.from_nms(NODES, tenor / 252., vol=atm)
            following = next_tenor(tenor, tenors)
            calls = decoded.europeans(business_dtes=np.full(len(strikes), tenor), strikes=strikes)
            longer = None if following is None else decoded.europeans(business_dtes=np.full(len(strikes), following), strikes=strikes)
            zero = decoded.europeans(business_dtes=np.array([tenor]), strikes=np.array([0.]))[:, 0]
            historical = np.stack([interpolate_generated_implied_volatility(history,
                query_expiry_y=tenor / 252., query_strikes_over_forward=np.full(len(history), strike), grid=GRID)
                for strike in strikes], axis=1)
            rows.append({"simulation_day": horizon, "business_dte": tenor, "next_business_dte": following,
                **call_violations(strikes, calls, longer_prices=longer, zero_strike=zero),
                **iv_stability(calls, strikes, tenor, historical),
                "paths_with_coordinate_guard": int(np.sum(decoded.lower_clips + decoded.upper_clips > 0)),
                "forward_scale_guard_paths": int(np.sum(forward_clips.any(axis=1)))})
        print(f"v2 static origin={state.day} simulation_day={horizon} paths={paths} complete", flush=True)
    return rows


def quoted_origin(root: Path, state: DailyState, following: np.ndarray, shocks: np.ndarray, endpoint: str) -> dict:
    data_root = Path(os.environ["QUANTDATA_ROOT"])
    market = data_root / "experiments/hedge2/market-full-20261106T0001/market/days"
    quotes = pd.read_parquet(market / f"{state.day}.parquet")
    quotes = quotes.loc[(quotes.bdte > 1. + 1e-6) & (quotes.bdte <= 5. + 1e-6) & (quotes.nms.abs() <= 1.5)].copy()
    quotes = quotes.loc[np.where(quotes.strike >= quotes.fwd, quotes.option_type == "call", quotes.option_type == "put")].reset_index(drop=True)
    observations = pd.read_parquet(root / "observations.parquet")
    licensed = Path(observations.source_cache_file.iloc[0]).parent.parent
    curve = licensed_curve(licensed, state.day)
    current = cash_europeans_batch(decoded=decode_batch(state, state.native[None]), template=curve,
        spots=np.array([curve.spot]), business_dtes=quotes.bdte.to_numpy(), cash_strikes=quotes.strike.to_numpy(),
        is_call=(quotes.option_type == "call").to_numpy())[0]
    _, reference, _ = curve.at(1.)
    spots = reference * (1. + shocks[:, 0])
    if np.any(spots <= 0.):
        raise ValueError("The retained normal-return law generated nonpositive spot; no paths removed")
    forecast = np.zeros(len(quotes))
    for start in range(0, len(following), 128):
        end = min(start + 128, len(following))
        prices = cash_europeans_batch(decoded=decode_batch(state, following[start:end]),
            template=curve.roll(1., new_spot=curve.spot), spots=spots[start:end], business_dtes=quotes.bdte.to_numpy() - 1.,
            cash_strikes=quotes.strike.to_numpy(), is_call=(quotes.option_type == "call").to_numpy())
        forecast += prices.sum(axis=0) / len(following)
    keys = ["expiry", "strike", "option_type", "is_am"]
    quotes["forecast"] = forecast
    next_quotes = pd.read_parquet(market / f"{endpoint}.parquet")
    joined = quotes.merge(next_quotes[keys + ["observed_bid", "observed_ask"]], on=keys,
        how="left", suffixes=("", "_next"), validate="one_to_one")
    return {"scope": "same matched diagnostic book as existing models: OTM, 1<BDTE<=5, |NMS|<=1.5",
        "paths": len(following), "endpoint": endpoint,
        "reconstruction": quote_scores(current, quotes.observed_bid, quotes.observed_ask),
        "next_day": quote_scores(joined.forecast, joined.observed_bid_next, joined.observed_ask_next),
        "price_units": "cash; divide price_mae by origin_spot for normalized comparison", "origin_spot": curve.spot}


def run(root: Path, reference: Path, output: Path, *, paths: int, horizons: tuple[int, ...],
        selected: tuple[str, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"model": "DYSANOS v2-P", "origins": list(selected), "paths": paths, "horizons": list(horizons),
        "seed": 20_261_112, "law": "completed independent STRT-16 + original TailSampler; normal spot shock, no stock-price path for static checks",
        "reference_sha256": _hash(reference / "reference_history.npz"), "source_sha256": _hash(Path(__file__)),
        "sample_policy": "one common recursive population, nested horizon snapshots; failures preserved, no repair or path removal"}
    protocol_path = output / "PROTOCOL.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Diagnostic protocol changed; use a new unique output directory")
    _json(protocol_path, protocol)
    with np.load(reference / "reference_history.npz", allow_pickle=False) as saved:
        surfaces, dates = saved["surfaces"], saved["dates"].astype(str)
    started = time.monotonic()
    for index, origin in enumerate(selected):
        health_check()
        destination = output / f"daily/{origin}.json"
        if destination.exists():
            continue
        state, law, tail, training = load_origin(root, origin)
        current = int(np.flatnonzero(dates == origin)[0])
        historical = surfaces[(dates >= training["dates"][0]) & (dates <= origin)]
        record = {"model": "DYSANOS v2-P", "origin": origin, "paths": paths,
            "dynamics_sha256": _hash(root / f"fits/dysanos2/{origin}/dynamics.npz"),
            "historical_reference_days": len(historical), "backbone": backbone(law.dynamics, tail)}
        try:
            snapshots, first_shocks = native_snapshots(state, law, tail, paths=paths, horizons=horizons, seed=20_261_112)
            baseline = json.loads((root / f"daily/DYSANOS_v2-P/{origin}.json").read_text())
            record["quotes"] = quoted_origin(root, state, snapshots[1], first_shocks, baseline["endpoint"])
            record["static"] = static_origin(state, snapshots, history=historical, origin_surface=surfaces[current], paths=paths)
            record["status"] = "complete"
        except (ValueError, FloatingPointError, RuntimeError) as error:
            record.update(status="diagnostic_failure", exact_error=f"{type(error).__name__}: {error}")
            print(f"v2 diagnostic {origin}: {record['exact_error']}", flush=True)
        _json(destination, record)
        records = [json.loads(path.read_text()) for path in sorted((output / "daily").glob("*.json"))]
        summary = {"processed": len(records), "complete": sum(item["status"] == "complete" for item in records),
            "failures": [{"origin": item["origin"], "error": item["exact_error"]} for item in records if item["status"] != "complete"]}
        _json(output / "SUMMARY.json", summary)
        rows = [{"origin": item["origin"], **row} for item in records for row in item.get("static", [])]
        if rows:
            pd.DataFrame(rows).to_csv(output / "static_daily.csv", index=False)
        elapsed = time.monotonic() - started
        print(f"v2 diagnostics completed={index+1}/{len(selected)} elapsed={elapsed:.1f}s eta={elapsed*(len(selected)-index-1)/(index+1):.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--origins", nargs="+")
    args = parser.parse_args()
    selected = tuple(args.origins) if args.origins else origins(args.reference)
    if not set(selected) <= set(origins(args.reference)) or 1 not in args.horizons:
        raise ValueError("Use the same approved origins and include the one-day diagnostic")
    run(args.root, args.reference, args.output, paths=args.paths, horizons=tuple(args.horizons), selected=selected)


if __name__ == "__main__":
    main()
