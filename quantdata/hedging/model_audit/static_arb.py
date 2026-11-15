"""Recursive normalized-call static-arbitrage and IV-stability diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from cdxcore.bs import bs
from filelock import FileLock

from quantdata.biz_calendar import NMS
from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.implied import implied_volatilities_with_validity
from quantdata.hedging.model_audit.persistence import _neural
from quantdata.hedging.model_audit.projected_law import load_law
from quantdata.hedging.model_audit.run import GRID, _hash, _json, build_history
from quantdata.hedging.model_audit.versioned import MODELS
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    interpolate_generated_implied_volatility,
)

HORIZONS = (1, 5, 10, 125, 259)
TENORS = (1., 2., 5., 10.)
NODES = np.linspace(-3., 2., 21)
PRICE_TOLERANCE = 1e-8


def next_tenor(tenor: float, grid: np.ndarray) -> float | None:
    later = np.asarray(grid)[np.asarray(grid) > tenor + 1e-8]
    return float(later.min()) if len(later) else None


def call_violations(strikes: np.ndarray, prices: np.ndarray, *, longer_prices: np.ndarray | None,
                    zero_strike: np.ndarray, tolerance: float = PRICE_TOLERANCE) -> dict:
    """Count defects in price units, including convexity on nonuniform strikes; never repair prices."""
    strikes, prices = np.asarray(strikes), np.asarray(prices)
    if (prices.ndim != 2 or strikes.shape != (prices.shape[1],) or len(strikes) < 3
            or not np.isfinite(strikes).all() or np.any(strikes <= 0.) or np.any(np.diff(strikes) <= 0.)
            or zero_strike.shape != (len(prices),) or tolerance < 0.):
        raise ValueError("Call tests require increasing positive strikes and path-by-strike prices")
    intrinsic = np.maximum(1. - strikes, 0.)
    calls = np.c_[zero_strike, prices]
    cash = np.r_[0., strikes]
    differences = np.diff(calls, axis=1)
    spacings = np.diff(cash)
    weight = spacings[1:] / (spacings[:-1] + spacings[1:])
    chords = calls[:, :-2] * weight + calls[:, 2:] * (1. - weight)
    defects = {"decreasing": differences,
        "convexity": calls[:, 1:-1] - chords,
        "vertical_spread": -differences - spacings,
        "intrinsic_lower_bound": intrinsic - prices,
        "upper_bound": prices - 1.,
        "zero_strike_identity": np.abs(zero_strike[:, None] - 1.)}
    if longer_prices is not None:
        if longer_prices.shape != prices.shape:
            raise ValueError("Calendar comparisons must use the same path and identical cash strikes")
        defects["calendar"] = prices - longer_prices
    result = {"paths": len(prices), "strikes": len(strikes), "tolerance": tolerance,
        "invalid_price_nodes": int(np.sum(~np.isfinite(prices))),
        "invalid_price_paths": int(np.sum(~np.isfinite(prices).all(axis=1))),
        "calendar_supported": longer_prices is not None}
    for name, values in defects.items():
        eligible = np.isfinite(values)
        violations = eligible & (values > tolerance)
        result.update({f"{name}_violations": int(violations.sum()),
            f"{name}_tests": int(eligible.sum()), f"{name}_violating_paths": int(violations.any(axis=1).sum()),
            f"{name}_max_defect": float(np.maximum(values[eligible], 0.).max()) if eligible.any() else None})
    return result


def iv_stability(prices: np.ndarray, strikes: np.ndarray, tenor: float,
                 historical_vols: np.ndarray) -> dict:
    """Invert unmodified prices with vector cdxcore.bs; invalid IVs remain explicit."""
    eligible = np.isfinite(prices)
    vols = np.full(prices.shape, np.nan)
    inversion = []
    if eligible.any():
        query = np.broadcast_to(strikes, prices.shape)
        estimates, valid = implied_volatilities_with_validity(expiries_y=np.full(eligible.sum(), tenor / 252.),
            normalized_strikes=query[eligible], normalized_prices=prices[eligible],
            is_call=np.ones(eligible.sum(), dtype=bool), diagnostics=inversion)
        vols[eligible] = np.where(valid, estimates, np.nan)
    finite = np.isfinite(vols)
    lower, upper = np.min(historical_vols, axis=0), np.max(historical_vols, axis=0)
    outside = finite & ((vols < lower) | (vols > upper))
    quantiles = np.quantile(vols[finite], [0., .01, .5, .99, 1.]) if finite.any() else [None] * 5
    return {"iv_count": int(finite.sum()), "invalid_iv_count": int((~finite).sum()),
        "outside_historical_range_count": int(outside.sum()), "above_300pct_iv_count": int(np.sum(vols > 3.)),
        **dict(zip(("iv_min", "iv_q01", "iv_median", "iv_q99", "iv_max"), quantiles, strict=True)),
        "inversion": inversion, "historical_dates": len(historical_vols)}


def generated_calls(surfaces: np.ndarray, strikes: np.ndarray, tenor: float) -> np.ndarray:
    """Original generator repricer: log-IV wings and total-variance time interpolation."""
    result = np.full((len(surfaces), len(strikes)), np.nan)
    valid = np.isfinite(surfaces).all(axis=(1, 2)) & (np.max(surfaces, axis=(1, 2)) < np.log(np.finfo(float).max) / 2.)
    if valid.any():
        selected = surfaces[valid]
        repeated = np.repeat(selected, len(strikes), axis=0)
        cash = np.tile(strikes, len(selected))
        volatility = interpolate_generated_implied_volatility(repeated, query_expiry_y=tenor / 252.,
            query_strikes_over_forward=cash, grid=GRID)
        result[valid] = bs.price(k=cash, vol=volatility, sqrtT=np.sqrt(tenor / 252.)).reshape(len(selected), -1)
    return result


def summarize(output: Path) -> None:
    rows = []
    for path in sorted((output / "static/daily").glob("*/*.json")):
        record = json.loads(path.read_text())
        for row in record.get("results", []):
            rows.append({"model": record["model"], "origin_date": record["origin_date"],
                         **{key: value for key, value in row.items() if not isinstance(value, list | dict)}})
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "static_arb_daily.csv", index=False)
        counts = [name for name in frame if name.endswith(("_violations", "_tests", "_count", "_nodes", "_paths"))]
        pooled = frame.groupby(["model", "simulation_day", "business_dte"], dropna=False)[counts].sum(min_count=1).reset_index()
        pooled.to_csv(output / "static_arb_summary.csv", index=False)


def run(root: Path, output: Path, *, models: list[str], paths: int, horizons: tuple[int, ...], origins: list[str] | None) -> None:
    """No standalone spot paths; model-internal return conditions keep their original law."""
    panel = PublicHedgingPanel(root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, output)
    requested = json.loads((output / "observable_reference.json").read_text())["dates"]
    selected_origins = requested if origins is None else origins
    if not set(selected_origins) <= set(requested) or paths < 2 or not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Static diagnostics require matched origins, positive ordered horizons and at least two paths")
    protocol = {"schema": "static-arbitrage-stability-v1", "models": MODELS, "origins": selected_origins,
        "paths": paths, "horizons": horizons, "nms": NODES.tolist(), "business_dtes": TENORS,
        "calendar": "next strictly later tenor in each model grid, at identical cash strikes; absent next tenor is unsupported",
        "grid": "NMS strike scale frozen at each origin's original reference ATM IV, by expiry; normalized forward=discount=1",
        "spot": "no separate spot-level path; internal return channels and correlated innovations retained where required by the model law",
        "tolerance_price_units": PRICE_TOLERANCE, "paths_shared_across_horizons": True,
        "history_range": "252 previous exact one-session endpoints, using original saved SANOS observable reference and same cash strikes",
        "reference_sha256": _hash(output / "reference_history.npz"), "source_sha256": _hash(Path(__file__)),
        "gromm_horizon_policy": "unsupported horizons are reported, never expiry-replenished or extrapolated"}
    destination = output / "STATIC_PROTOCOL.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous != json.loads(json.dumps(protocol)):
            raise ValueError("Static run identity changed; use another unique run directory")
    _json(destination, protocol)
    native_root = root / "quant_data/hedge2/runs/dlv-full-v4-20261106T1130"
    values = json.loads((native_root / "config.json").read_text())
    native_config = DlvConfig(**{field.name: values[field.name] for field in fields(DlvConfig) if field.name in values})
    started = time.monotonic()
    for model in models:
        if model not in ("DYSANOS v1", "DYSANOS v2", "FuNVol", "VolGAN PCA8"):
            raise ValueError(f"Use the managed-GPU runner for {model}")
        for position, text in enumerate(selected_origins):
            if position == 0 or time.monotonic() - last_resource_check > 240.:
                resource_check = subprocess.run(["gpuctl", "resources", "--json"], capture_output=True, text=True)
                if resource_check.returncode:
                    print("WARN: gpuctl resources could not be read", flush=True)
                else:
                    health = json.loads(resource_check.stdout)
                    if any(check["health"] in ("STOP", "UNAVAILABLE", "ERROR") for check in health["checks"]):
                        raise RuntimeError("Managed resource stop; completed origin checkpoints remain intact")
                    if any(check["health"] == "WARN" for check in health["checks"]):
                        print("WARN: managed resource headroom", flush=True)
                last_resource_check = time.monotonic()
            path = output / f"static/daily/{model.replace(' ', '_')}/{text}.json"
            if path.exists():
                continue
            day = date.fromisoformat(text)
            index = history.dates.index(day)
            training = np.flatnonzero((history.sessions_from_previous_observation == 1)
                                     & (np.arange(len(history.dates)) <= index))[-252:]
            if len(training) != 252:
                raise ValueError("The unchanged historical reference does not provide 252 exact-session endpoints")
            native_model = model.startswith("DYSANOS")
            if native_model:
                with np.load(native_root / f"days/{text}.npz", allow_pickle=False) as saved:
                    state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=native_config)
                law = load_law(output, model, day)
                snapshots = law.recursive_paths(state.native, paths=paths, steps=max(horizons),
                    seed=20_261_111, snapshot_steps=horizons)
                grid_days = np.asarray(native_config.business_dtes)
                metadata = {"native_source": str(native_root / f"days/{text}.npz"),
                    "law": str(output / f"fits/{model.replace(' ', '_')}/{text}"), "guards": []}
            else:
                snapshots, metadata = _neural(root, model, day, history, index, paths=paths, steps=max(horizons),
                    full_grid=True, snapshot_steps=horizons)
                snapshots = snapshots.reshape(paths, len(horizons), len(GRID.expiries_y), GRID.strike_count)
                grid_days = np.asarray(GRID.expiries_y) * 252.
            results, snapshot_arrays = [], {}
            for horizon_index, horizon in enumerate(horizons):
                if native_model:
                    pricer = decode_batch(state, snapshots[:, horizon_index])
                    metadata["guards"].append({"simulation_day": horizon,
                        "lower_clips": int(pricer.lower_clips.sum()), "upper_clips": int(pricer.upper_clips.sum())})
                for tenor in TENORS:
                    row = GRID.expiries_y.index(tenor / 252.)
                    atm = np.exp(history.log_implied_volatilities[index, row, GRID.nms_nodes.index(0.)])
                    strikes = NMS.from_nms(NODES, tenor / 252., vol=atm)
                    following = next_tenor(tenor, grid_days)
                    if native_model:
                        calls = pricer.europeans(business_dtes=np.full(len(strikes), tenor), strikes=strikes)
                        longer = None if following is None else pricer.europeans(business_dtes=np.full(len(strikes), following), strikes=strikes)
                        zero = pricer.europeans(business_dtes=np.array([tenor]), strikes=np.array([0.]))[:, 0]
                    else:
                        calls = generated_calls(snapshots[:, horizon_index], strikes, tenor)
                        longer = None if following is None else generated_calls(snapshots[:, horizon_index], strikes, following)
                        zero = np.ones(paths)
                    historical = np.stack([interpolate_generated_implied_volatility(
                        history.log_implied_volatilities[training], query_expiry_y=tenor / 252.,
                        query_strikes_over_forward=np.full(len(training), strike), grid=GRID) for strike in strikes], axis=1)
                    checks = call_violations(strikes, calls, longer_prices=longer, zero_strike=zero)
                    stability = iv_stability(calls, strikes, tenor, historical)
                    results.append({"simulation_day": horizon, "business_dte": tenor, "next_business_dte": following,
                                    **checks, **stability})
                    key = f"day{horizon}_tenor{int(tenor)}"
                    snapshot_arrays[f"{key}_strikes"] = strikes
                    snapshot_arrays[f"{key}_calls"] = calls
                    if longer is not None:
                        snapshot_arrays[f"{key}_next_calls"] = longer
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **snapshot_arrays)
            temporary.replace(path.with_suffix(".npz"))
            _json(path, {"model": model, "origin_date": text, "paths": paths,
                "horizons": horizons, "seed": 20_261_111, "status": "scored", "results": results,
                "metadata": metadata, "arrays_sha256": _hash(path.with_suffix(".npz"))})
            with FileLock(output / ".static-summary.lock"):
                summarize(output)
            elapsed = time.monotonic() - started
            print(f"static arb {model} origin={position+1}/{len(selected_origins)} {text} paths={paths} "
                  f"steps={max(horizons)} elapsed={elapsed:.1f}s eta={elapsed*(len(selected_origins)-position-1)/(position+1):.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["DYSANOS v1", "DYSANOS v2", "FuNVol", "VolGAN PCA8"])
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--origins", nargs="+")
    args = parser.parse_args()
    run(args.root, args.output, models=args.models, paths=args.paths, horizons=tuple(args.horizons), origins=args.origins)


if __name__ == "__main__":
    main()
