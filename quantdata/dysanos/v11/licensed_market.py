"""Read licensed quote books with the saved smoothness-0.1 SANOS surfaces."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from cdxcore.subdir import SubDir

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.flex.config import FlexGeometryConfig
from quantdata.dysanos.flex.market import FlexMarketDay, MarketEligibilityAudit
from quantdata.sanos.arbfree import ArbFreePricer


def load_certified_pricer(day: date, *, archive: Path) -> tuple[ArbFreePricer, str]:
    """Load the certified, saved 0.1 SANOS interpolator for one date."""
    candidates = [archive / location / day.isoformat() / "models/all-0.1/model.h5" for location in ("days", "pilot")]
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one saved 0.1 surface for {day}, found {paths}")
    path = paths[0]
    certificate = json.loads(path.with_name("certificate.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if certificate.get("status") != "passed" or certificate.get("model_sha256") != digest:
        raise ValueError(f"Invalid surface certificate for {day}")
    pricer = ArbFreePricer.from_dict(SubDir(str(path.parent), fmt=SubDir.PYTREE_HDF5, create_directory=False).read("model"))
    if pricer.smoothness != 0.1:
        raise ValueError(f"Incorrect SANOS smoothness for {day}")
    return pricer, digest


def licensed_market_day(day: date, *, panel: Path, archive: Path, geometry: FlexGeometryConfig) -> FlexMarketDay:
    """Use licensed bid/ask and midpoint; 0.1 SANOS determines ATM/NMS geometry."""
    pricer, digest = load_certified_pricer(day, archive=archive)
    quotes = pd.read_parquet(panel / "days" / f"{day.isoformat()}.parquet")
    quotes = quotes.loc[(~quotes.is_am) & (quotes.bdte > 0) & (quotes.bdte <= geometry.max_business_dte)].copy()
    if quotes.empty or not (quotes.date == pd.Timestamp(day)).all():
        raise ValueError(f"Invalid licensed panel for {day}")
    expiries = np.sort(quotes.bdte.unique().astype(np.float64)) / DAYS_PER_YEAR
    atm = np.asarray(pricer.interpolate_atm_vols(expiries), dtype=np.float64)
    model_expiries = geometry.model_expiries_y
    model_atm = np.asarray(pricer.interpolate_atm_vols(model_expiries), dtype=np.float64)
    bounds = NMS.from_nms(np.asarray(geometry.nms_range)[None, :], t=expiries[:, None], vol=atm[:, None], eps_vol=float(model_atm[0]))
    index = np.searchsorted(expiries, quotes.bdte.to_numpy(dtype=np.float64) / DAYS_PER_YEAR)
    strikes = quotes.strike.to_numpy(dtype=np.float64) / quotes.fwd.to_numpy(dtype=np.float64)
    keep = (strikes >= bounds[index, 0]) & (strikes <= bounds[index, 1])
    quotes = quotes.loc[keep].reset_index(drop=True)
    strikes, index = strikes[keep], index[keep]
    selected, index = np.unique(index, return_inverse=True)
    discount_forward = quotes.df.to_numpy(dtype=np.float64) * quotes.fwd.to_numpy(dtype=np.float64)
    bids = quotes.observed_bid.to_numpy(dtype=np.float64) / discount_forward
    asks = quotes.observed_ask.to_numpy(dtype=np.float64) / discount_forward
    expiry_dates = []
    for original_index in selected:
        dates = quotes.loc[quotes.bdte == expiries[original_index] * DAYS_PER_YEAR, "expiry"].unique()
        if len(dates) != 1:
            raise ValueError(f"Ambiguous expiry for {day}: {dates}")
        expiry_dates.append(str(pd.Timestamp(dates[0]).date()))
    is_call = quotes.option_type.to_numpy() == "call"
    return FlexMarketDay(
        ticker="SPX", trading_date=day, expiries_y=expiries[selected], expiry_dates=tuple(expiry_dates),
        expiry_ids=np.arange(len(selected), dtype=np.int64), expiry_is_am=np.zeros(len(selected), dtype=bool),
        expiry_index=index, strikes=strikes, is_call=is_call, bids=bids, asks=asks, fit_prices=(bids + asks) / 2,
        log_volumes=np.zeros(len(quotes)), log_open_interests=np.zeros(len(quotes)),
        quote_ids=tuple(f"{day}:{expiry}:{strike:.8g}:{side}" for expiry, strike, side in zip(quotes.expiry, quotes.strike, quotes.option_type, strict=True)),
        atm_annual_vols=atm[selected], model_atm_total_vols=model_atm * np.sqrt(model_expiries),
        strike_lower_bounds=bounds[selected, 0], strike_upper_bounds=bounds[selected, 1],
        market_uid=f"licensed-sanos-0.1:{digest}",
        audit=MarketEligibilityAudit(active_european_books=len(keep), horizon_eligible_books=len(keep), nms_eligible_books=len(quotes), input_expiries=len(expiries), eligible_expiries=len(selected)),
    )
