"""
quantdata.fred.rates

Download SOFR, EFFR and Treasury rates from FRED.
Requires the ``FRED_API_KEY`` environment variable.
See https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import os
from datetime import date

import polars as pl
import requests
from cdxcore.err import verify

from quantdata import env

_fred_cache = env.locations.raw("fred")


def _fetch_fred_series(api_key: str, series_id: str, start: date, end: date) -> pl.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.strftime("%Y-%m-%d"),
        "observation_end": end.strftime("%Y-%m-%d"),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])

    if not obs:
        return pl.DataFrame(
            {"date": pl.Series([], dtype=pl.Date), series_id: pl.Series([], dtype=pl.Float64)}
        )

    df = pl.DataFrame(obs).select(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
        pl.col("value").alias(series_id),
    )
    df = df.with_columns(
        pl.when(pl.col(series_id) == ".").then(None).otherwise(pl.col(series_id)).alias(series_id)
    ).with_columns(pl.col(series_id).cast(pl.Float64))
    return df


@_fred_cache.cache("0.2", uid=lambda start, end: f"fred/{start}_{end}")
def load_fred_rates(start: str | date, end: str | date) -> pl.DataFrame:
    """
    Download SOFR, EFFR, DGS1MO, DGS3MO, DGS6MO from FRED and return as unscaled floats.

    Adds ``on`` column defined as SOFR when available, otherwise EFFR, otherwise DGS1MO.
    Only returns dates that have at least one non-zero rate. All rate columns are
    forward-filled so there are no trailing nulls.

    Requires ``FRED_API_KEY`` environment variable.

    Parameters
    ----------
    start : str | date
    end : str | date

    Returns
    -------
    pl.DataFrame
        Columns: date, on, sofr, effr, dgs1mo, dgs3mo, dgs6mo (all Float32, unscaled e.g. 0.05 for 5%)
    """
    start = start if not isinstance(start, str) else date.fromisoformat(start)
    end = end if not isinstance(end, str) else date.fromisoformat(end)
    api_key = os.environ.get("FRED_API_KEY")
    verify(
        api_key is not None,
        "Please set FRED_API_KEY in your environment. See https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    series = ["SOFR", "EFFR", "DGS1MO", "DGS3MO", "DGS6MO"]
    dfs = [_fetch_fred_series(api_key=api_key, series_id=s, start=start, end=end) for s in series]

    df = dfs[0]
    for other in dfs[1:]:
        df = df.join(other, on="date", how="full", coalesce=True)

    df = df.sort("date")

    rate_cols = ["sofr", "effr", "dgs1mo", "dgs3mo", "dgs6mo"]
    df = df.with_columns(
        (pl.col("SOFR") / 100.0).cast(pl.Float32).alias("sofr"),
        (pl.col("EFFR") / 100.0).cast(pl.Float32).alias("effr"),
        (pl.col("DGS1MO") / 100.0).cast(pl.Float32).alias("dgs1mo"),
        (pl.col("DGS3MO") / 100.0).cast(pl.Float32).alias("dgs3mo"),
        (pl.col("DGS6MO") / 100.0).cast(pl.Float32).alias("dgs6mo"),
    ).select(["date"] + rate_cols)

    df = df.filter(pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in rate_cols]) != 0.0)

    df = df.with_columns(
        pl.when(pl.col("sofr").is_not_null())
        .then(pl.col("sofr"))
        .when(pl.col("effr").is_not_null())
        .then(pl.col("effr"))
        .otherwise(pl.col("dgs1mo"))
        .alias("on")
    )

    df = df.with_columns([pl.col(c).forward_fill() for c in ["on"] + rate_cols]).sort("date")

    return df.select(["date", "on"] + rate_cols)
