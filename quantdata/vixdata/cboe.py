"""Download, validate, and cache official Cboe end-of-day VIX observations."""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import requests
from cdxcore.err import verify_inp

from quantdata import SubDir, env

CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
_CACHE_VERSION = "1.0"
_locations = env.locations("vixdata", fmt_raw=SubDir.POLARS_PARQUET)
_raw_cache = _locations.raw
_DATA_FILE = "cboe/vix_eod"


def _as_date(value: date | str | None, *, name: str) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date, found {value!r}") from error


def parse_cboe_vix_csv(content: bytes) -> pl.DataFrame:
    """Parse Cboe's daily OHLC CSV into ascending, validated observations."""
    frame = pl.read_csv(content, try_parse_dates=False)
    expected = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    if frame.columns != expected:
        raise ValueError(f"unexpected Cboe VIX schema: {frame.columns}")
    values = (
        frame.with_columns(
            pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y", strict=True).alias("date"),
            *[pl.col(name).cast(pl.Float64, strict=True).alias(name.lower()) for name in expected[1:]],
        )
        .select("date", "open", "high", "low", "close")
        .sort("date")
    )
    if values.is_empty() or values.null_count().sum_horizontal().item() != 0:
        raise ValueError("Cboe VIX history is empty or has missing values")
    if values.select(pl.any_horizontal(pl.all().exclude("date") <= 0.0).any()).item():
        raise ValueError("Cboe VIX history has non-positive OHLC values")
    # Cboe's early historical file contains a small number of OHLC rows whose
    # intraday high/low do not bracket open or close.  VIX *closes* are the
    # data used by the models here, and remain well defined.  Preserve the
    # source OHLC columns for provenance, but do not reject valid EOD levels
    # because of those vendor-history anomalies.
    if values.select(pl.col("date").n_unique() != pl.len()).item():
        raise ValueError("Cboe VIX history has duplicate dates")
    return values


def refresh_cboe_vix(*, timeout_seconds: float = 30.0) -> pl.DataFrame:
    """Fetch official Cboe VIX closes and replace the native raw-data cache.

    The shared raw location is determined solely by ``quantdata.env``.  On the
    production host this resolves to the canonical AAE-NAS data root rather
    than a user-specific filesystem path.
    """
    verify_inp(timeout_seconds > 0.0, lambda: "timeout_seconds must be positive")
    response = requests.get(
        CBOE_VIX_HISTORY_URL,
        headers={"User-Agent": "quantdata-vixdata/1.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    values = parse_cboe_vix_csv(response.content)
    _raw_cache.write(_DATA_FILE, values, version=_CACHE_VERSION)
    _raw_cache.write(
        "cboe/vix_eod_metadata",
        {
            "source": CBOE_VIX_HISTORY_URL,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "observations": len(values),
            "first_date": values["date"].min().isoformat(),
            "last_date": values["date"].max().isoformat(),
            "units": "annualized implied-volatility percentage points",
        },
        version=_CACHE_VERSION,
        fmt=SubDir.JSON_PLAIN,
    )
    return values


def load_cboe_vix(
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    refresh: bool = False,
) -> pl.DataFrame:
    """Load cached official daily VIX OHLC observations, optionally refreshing.

    ``close`` is the official Cboe end-of-day VIX level, in annualized implied
    volatility percentage points.  The interval is inclusive.
    """
    start_date = _as_date(start, name="start")
    end_date = _as_date(end, name="end")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start must not exceed end")
    values = None if refresh else _raw_cache.read(_DATA_FILE, version=_CACHE_VERSION)
    if values is None:
        values = refresh_cboe_vix()
    if not isinstance(values, pl.DataFrame):
        raise RuntimeError("native Cboe VIX cache has an invalid type")
    filters: list[pl.Expr] = []
    if start_date is not None:
        filters.append(pl.col("date") >= start_date)
    if end_date is not None:
        filters.append(pl.col("date") <= end_date)
    return values.filter(*filters).sort("date") if filters else values
