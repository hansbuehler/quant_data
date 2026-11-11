"""Quoted-contract next-day price forecasts from retained raw scenario books."""

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

from quantdata.hedge2.batch_pricing import decode_batch
from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_market import cash_europeans_batch
from quantdata.hedging.model_audit.metrics import quote_scores
from quantdata.hedging.model_audit.run import (
    _json,
    _saved,
    retained_dynamics_path,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    build_generated_daily_scenarios,
)


def contract_strike_keys(identifiers: pd.DataFrame, quoted_origin: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Recover unique original strike identities within float32 round-trip bounds."""
    keys = np.full(len(identifiers), np.nan)
    errors = np.full(len(identifiers), np.nan)
    grouping = ["expiry", "option_type", "is_am"]
    groups = {key: np.sort(group.strike.astype(float).unique())
              for key, group in quoted_origin.groupby(grouping, sort=False)}
    for position, row in enumerate(identifiers.itertuples(index=False)):
        candidates = groups.get((pd.Timestamp(row.expiry), row.option_type, row.is_am))
        if candidates is None:
            continue
        tolerance = 4. * np.finfo(np.float32).eps * max(1., abs(float(row.strike)))
        lower = np.searchsorted(candidates, row.strike - tolerance, side="left")
        upper = np.searchsorted(candidates, row.strike + tolerance, side="right")
        if upper - lower == 1:
            keys[position] = candidates[lower]
            errors[position] = abs(candidates[lower] - row.strike)
    return keys, errors


def _dysanos(root: Path, model: str, day: date, endpoint: date, *, licensed: Path,
             output: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, float]:
    base = root / "quant_data/hedge2/runs"
    directory = base / "dlv-full-v4-20261106T1130"
    dynamics = load_dynamics(retained_dynamics_path(root, model, day, output=output))
    if "(16 factors)" in model and dynamics.directions.shape[1] != 16:
        raise ValueError("The requested sixteen-factor quote forecast has a different rank")
    config_values = json.loads((directory / "config.json").read_text())
    config = DlvConfig(**{field.name: config_values[field.name] for field in fields(DlvConfig) if field.name in config_values})
    with np.load(directory / f"days/{day}.npz", allow_pickle=False) as saved:
        state = DailyState(day=day, native=saved["raw_u"], reference_support=saved["support_strikes"], config=config)
    quotes = pd.read_parquet(root / f"experiments/hedge2/market-full-20261106T0001/market/days/{day}.parquet")
    quotes = quotes.loc[(quotes.bdte > 1. + 1e-6) & (quotes.bdte <= 5. + 1e-6) & (quotes.nms.abs() <= 1.5)].copy()
    quotes = quotes.loc[np.where(quotes.strike >= quotes.fwd, quotes.option_type == "call", quotes.option_type == "put")]
    curve = licensed_curve(licensed, day)
    _, reference, _ = curve.at(1.)
    residuals = dynamics.residuals
    if model.endswith("(tail)"):
        from quantdata.hedging.model_audit.tail import TailSampler, fit_tail

        if output is None:
            raise ValueError("Tail Diffusion quotes require a dedicated output directory")
        residuals = TailSampler(dynamics, fit_tail(dynamics, root=root, output=output, model=model),
            paths=10_000, seed=20_261_111).draw()
    natives = (state.native + dynamics.forecast_state_change(state.native)
        + residuals[:, :1] * dynamics.spot_state_loading + residuals[:, 1:] @ dynamics.directions.T)
    decoded = decode_batch(state, natives, max_business_dte=5.)
    future = cash_europeans_batch(decoded=decoded, template=curve.roll(1., new_spot=curve.spot),
        spots=reference * (1. + residuals[:, 0]), business_dtes=quotes.bdte.to_numpy() - 1.,
        cash_strikes=quotes.strike.to_numpy(), is_call=(quotes.option_type == "call").to_numpy())
    return quotes, future.mean(axis=0), float(curve.spot)


def _book(root: Path, panel: PublicHedgingPanel, model: str, day: date, *, spot: float,
          audit_directory: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    origin = panel.market_day(day)
    quotes = origin.hedge_snapshot().option_quotes
    internal_name = {"VolGAN PCA8": "volgan", "AD-Seq-Vol": "diffusion-base", "AD-Seq-Vol-FT": "diffusion-ft",
                     "GROMM-P": "gromm-P", "GROMM-Q": "gromm-Q"}.get(model, model.lower())
    path = root / f"experiments/hedging/generated-objective-grid-20261026-v1/scenario-books/{internal_name}/{day}.npz"
    if model == "FuNVol":
        path = root / f"training/hedging/funvol-causal-chain-20261025-warning-v1/aprmay/scenarios/{day}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            returns = saved["instrument_returns"]
    elif model in ("AD-Seq-Vol", "AD-Seq-Vol-FT"):
        from quantdata.hedging.model_audit.run import GRID

        log_returns, surfaces, _ = _saved(root, model, day, [], audit_directory=audit_directory)
        sample = build_daily_digital_sample(panel, origin_date=day, target=daily_digital_targets(maturity_business_days=1)[0])
        returns = build_generated_daily_scenarios(sample, forward_log_returns=log_returns,
            next_log_implied_volatilities=surfaces, grid=GRID).instrument_returns
    else:
        raise FileNotFoundError(f"No retained raw option-price scenario book: {path}")
    if returns.shape[1] != 1 + len(quotes):
        raise ValueError(f"Retained quote order/dimension mismatch: {returns.shape[1]} versus {len(quotes) + 1}")
    discount = (panel.one_day_transition(day).discount_factor if model.startswith("GROMM") else origin.discount_at_business_dte(1))
    predictions = (np.array([quote.mark for quote in quotes]) + returns[:, 1:].mean(axis=0)) * spot / discount
    identifiers = pd.DataFrame({"expiry": [pd.Timestamp(quote.contract.expiry_date) for quote in quotes],
        "strike": [quote.contract.fixed_strike * spot for quote in quotes],
        "option_type": [quote.contract.right.value for quote in quotes],
        "is_am": [quote.is_am_settlement for quote in quotes], "bdte": [quote.business_dte for quote in quotes],
        "nms": [quote.standardized_log_moneyness for quote in quotes]})
    eligible = (identifiers.bdte > 1. + 1e-6) & (identifiers.bdte <= 5. + 1e-6) & (identifiers.nms.abs() <= 1.5)
    return identifiers.loc[eligible].copy(), predictions[eligible]


def _add_persistence(root: Path, panel: PublicHedgingPanel, model: str, day: date,
                     scores: dict, *, licensed: Path) -> None:
    """Evaluate origin-mid carry-only persistence on the identical quoted population."""
    market = root / "experiments/hedge2/market-full-20261106T0001/market/days"
    current = pd.read_parquet(market / f"{day}.parquet")
    following = pd.read_parquet(market / f"{scores['endpoint']}.parquet")
    curve = licensed_curve(licensed, day)
    if model.startswith("DYSANOS"):
        identifiers = current.loc[(current.bdte > 1. + 1e-6) & (current.bdte <= 5. + 1e-6) & (current.nms.abs() <= 1.5)]
        identifiers = identifiers.loc[np.where(identifiers.strike >= identifiers.fwd,
            identifiers.option_type == "call", identifiers.option_type == "put")].copy()
    else:
        quotes = panel.market_day(day).hedge_snapshot().option_quotes
        identifiers = pd.DataFrame({"expiry": [pd.Timestamp(quote.contract.expiry_date) for quote in quotes],
            "strike": [quote.contract.fixed_strike * curve.spot for quote in quotes],
            "option_type": [quote.contract.right.value for quote in quotes], "is_am": [quote.is_am_settlement for quote in quotes],
            "bdte": [quote.business_dte for quote in quotes], "nms": [quote.standardized_log_moneyness for quote in quotes]})
        identifiers = identifiers.loc[(identifiers.bdte > 1. + 1e-6) & (identifiers.bdte <= 5. + 1e-6) & (identifiers.nms.abs() <= 1.5)]
    identifiers["strike_key"], errors = contract_strike_keys(identifiers, current)
    for frame in (current, following):
        frame["strike_key"] = frame.strike.astype(float)
    keys = ["expiry", "strike_key", "option_type", "is_am"]
    joined = identifiers[keys].merge(current[keys + ["observed_mid"]], on=keys, how="left", validate="one_to_one")
    joined = joined.merge(following[keys + ["observed_bid", "observed_ask"]], on=keys, how="left", validate="one_to_one")
    discount = curve.at(1.)[2]
    baseline = quote_scores(joined.observed_mid / discount, joined.observed_bid, joined.observed_ask)
    if baseline["count"] != scores["count"]:
        scores["persistence_reference_status"] = "unavailable: identical quoted population not recovered"
        return
    scores["persistence_price_mae_origin_spot_units"] = baseline["price_mae"] / curve.spot
    scores["persistence_reference_status"] = "origin observed midpoint, carried by origin-only one-day discount; identical contracts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--models", nargs="+")
    arguments = parser.parse_args()
    root, output = arguments.root, arguments.output
    panel = PublicHedgingPanel(root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    configuration = json.loads((root / "quant_data/hedge2/runs/parametric-itm-premium-pilot-20261106T1758/config.json").read_text())
    licensed = Path(configuration["licensed_fit_directory"])
    models = ("DYSANOS ordinary", "DYSANOS spot-orthogonal", "FuNVol", "VolGAN PCA8", "AD-Seq-Vol", "AD-Seq-Vol-FT", "GROMM-P", "GROMM-Q")
    models = arguments.models or models
    origins = [day for day in panel.dates if date(2024, 4, 1) <= day <= date(2024, 5, 31)]
    started = time.monotonic()
    for model in models:
        for position, day in enumerate(origins):
            path = output / "quote_forecasts" / model.replace(" ", "_") / f"{day}.json"
            if path.exists() and not arguments.refresh:
                saved = json.loads(path.read_text())
                if saved.get("status") == "scored" and "persistence_price_mae_origin_spot_units" not in saved:
                    _add_persistence(root, panel, model, day, saved, licensed=licensed)
                    _json(path, saved)
                if saved.get("status") == "scored" or not (output / f"regenerated/{model}/{day}.npz").exists():
                    continue
            try:
                endpoint = panel.next_one_session_date(day)
                spot = licensed_curve(licensed, day).spot
                if model.startswith("DYSANOS"):
                    identifiers, predictions, _ = _dysanos(root, model, day, endpoint, licensed=licensed, output=output)
                else:
                    identifiers, predictions = _book(root, panel, model, day, spot=spot, audit_directory=output)
                current = pd.read_parquet(root / f"experiments/hedge2/market-full-20261106T0001/market/days/{day}.parquet")
                strike_keys, strike_errors = contract_strike_keys(identifiers, current)
                identifiers = identifiers.assign(prediction=predictions, strike_key=strike_keys)
                following = pd.read_parquet(root / f"experiments/hedge2/market-full-20261106T0001/market/days/{endpoint}.parquet")
                following = following.assign(strike_key=following.strike.astype(float))
                keys = ["expiry", "strike_key", "option_type", "is_am"]
                joined = identifiers[keys + ["prediction", "nms", "bdte"]].merge(
                    following[keys + ["observed_bid", "observed_ask", "observed_mid"]], on=keys, how="left", validate="one_to_one")
                matched = joined.observed_bid.notna() & joined.observed_ask.notna()
                scores = quote_scores(joined.prediction, joined.observed_bid, joined.observed_ask)
                if scores["count"] == 0:
                    raise ValueError("No quoted contracts matched their next-session cash strike/expiry")
                scores.update({"origin_date": str(day), "endpoint": str(endpoint), "model": model, "status": "scored",
                    "eligible_contracts": len(identifiers), "matched_contracts": int(matched.sum()),
                    "price_mae_origin_spot_units": scores["price_mae"] / spot,
                    "mode": "raw scenario mean option value, fixed cash strike/expiry; observed next-day bid/ask; origin-only deterministic carry",
                    "population": "surviving OTM hedge-book contracts, origin 2-5D and |NMS|<=1.5",
                    "contract_identity_policy": "unique origin-quoted strike within 4 float32 eps times cash strike; no ambiguous/remote substitutions",
                    "strike_roundtrip_max_cash_error": float(np.nanmax(strike_errors)) if np.isfinite(strike_errors).any() else None,
                    "strike_identity_unresolved": int(np.sum(~np.isfinite(strike_keys))),
                    "buckets": {}})
                for bucket, subset in joined.groupby(pd.cut(joined.nms, [-1.5001, -.5, .5, 1.5001], include_lowest=True), observed=True):
                    scores["buckets"][str(bucket)] = quote_scores(subset.prediction, subset.observed_bid, subset.observed_ask)
                _add_persistence(root, panel, model, day, scores, licensed=licensed)
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                scores = {"model": model, "origin_date": str(day), "status": "unavailable", "reason": str(error)}
            _json(path, scores)
            print(f"quote forecasts {model} {position+1}/{len(origins)} {scores['status']} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
