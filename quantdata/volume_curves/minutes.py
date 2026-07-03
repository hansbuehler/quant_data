"""
Fetch minute bars from Massive (Polygon) API for a given ticker and date.

Key behaviors:
- Uses adjusted minute aggregates (1/minute) from Polygon/Massive.
- Computes overnight returns based on two-day data.
- Converts timestamps to America/New_York and normalizes datetime to microseconds.
- Filters strictly to regular-session minutes:
    * Full day: 09:30 <= t < 16:00  => minute_idx in [0, 389] (390 minutes)
    * Half day: 09:30 <= t < 13:00  => minute_idx in [0, 209] (210 minutes)
- Includes sanity checks (warnings) for missing minutes, duplicates, or unexpected counts.
"""

from __future__ import annotations

from datetime import date, time
from typing import Optional

import polars as pl
from cdxcore.err import verify_inp
from cdxcore.subdir import version
from cdxcore.util import fmt_date

from quantdata import Context, env
from quantdata.biz_calendar import get_half_days, prev_business_day
from quantdata.massive import MassiveClient

_a1m_cache = env.locations.raw("volume_curves")("a1m")

# Trading hours constants (ET)
FULL_DAY_OPEN = time(9, 30)
FULL_DAY_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)

# Expected regular-session minute counts
FULL_DAY_MINUTES = 390
HALF_DAY_MINUTES = 210


def _empty_minute_df() -> pl.DataFrame:
    """Return empty DF with stable schema (Datetime[ms, America/New_York])."""
    return pl.DataFrame(
        {
            "datetime": pl.Series(
                [], dtype=pl.Datetime(time_unit="ms", time_zone="America/New_York")
            ),
            "date": pl.Series([], dtype=pl.Date),
            "minute_idx": pl.Series([], dtype=pl.Int32),
            "open": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "n": pl.Series([], dtype=pl.Int64),
            "volume": pl.Series([], dtype=pl.Float64),
            "vwap": pl.Series([], dtype=pl.Float64),
        }
    )


def _warn(verbose: Context, msg: str) -> None:
    try:
        verbose.write(f"WARNING: {msg}", head=False)
    except Exception:
        print(f"WARNING: {msg}")


def _sanity_check_minutes(
    df: pl.DataFrame,
    ticker: str,
    dt: date,
    is_half_day: bool,
    verbose: Context,
) -> None:
    """
    Relaxed sanity checks:
    - warn on duplicated minute_idx (structural bug / bad API behavior)
    - warn if datetime decreases after sorting (structural bug)
    - DO NOT warn on missing minutes or count != expected (illiquid tickers)
    """
    if df.height == 0:
        return

    dup_count = int(df.select(pl.col("minute_idx")).is_duplicated().sum())
    if dup_count > 0:
        _warn(verbose, f"{ticker} {dt:%Y-%m-%d}: {dup_count} duplicated minute_idx values.")

    not_mono = df.select((pl.col("datetime") < pl.col("datetime").shift(1)).any()).item()
    if not_mono:
        _warn(
            verbose,
            f"{ticker} {dt:%Y-%m-%d}: datetime decreased after minute_idx sort (unexpected).",
        )


@_a1m_cache.cache(
    "0.1",
    uid=lambda ticker, dt: f"{ticker}/{fmt_date(dt)}",
    dependencies=[MassiveClient],
    exclude_args=["client", "verbose"],
)
def fetch_minute_bars(
    ticker: str,
    dt: date,
    is_half_day: bool | None = None,
    *,
    prev_dt: date | None = None,
    prev_is_half_day: bool | None = None,
    client: Optional[MassiveClient] = None,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Fetch adjusted minute bars for a given ticker and date from Massive (Polygon) API.
    Note this function returns the adjusted bars; for this reason it also returns
    the last minute bar of the previous day (for computing adjusted overnight returns).

    Returns a Polars DataFrame with columns:
      - datetime: pl.Datetime[ms, America/New_York]
      - minute_idx: pl.Int16   (dt minutes: 0..389/209; prev close row: -1)
      - open: pl.Float32
      - close: pl.Float32
      - n: pl.Int32
      - volume: pl.Int64
      - vwap: pl.Float32
    """
    client = MassiveClient() if client is None else client

    prev_dt = prev_dt if prev_dt is not None else prev_business_day(dt)
    verify_inp(
        prev_dt < dt, lambda: f"'prev_dt' {fmt_date(prev_dt)} must be before 'dt' {fmt_date(dt)}"
    )
    if is_half_day is None or prev_is_half_day is None:
        half_days = set(get_half_days(prev_dt, dt))
        is_half_day = (dt in half_days) if is_half_day is None else is_half_day
        prev_is_half_day = (prev_dt in half_days) if prev_is_half_day is None else prev_is_half_day

    url = f"{MassiveClient.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{prev_dt:%Y-%m-%d}/{dt:%Y-%m-%d}"

    with verbose.write_t(
        lambda: f"Fetching minute bars for {ticker} on {fmt_date(dt)}... ", end=""
    ) as tme:
        rows = client.paginate(url, params={"adjusted": "true", "limit": 50000, "sort": "asc"})
        if not rows:
            verbose.write("no data found.", head=False)
            return _empty_minute_df()

        df = pl.DataFrame(rows, infer_schema_length=None)

        if "vw" not in df.columns:
            df = df.with_columns(pl.lit(None).alias("vw"))

        df = df.with_columns(
            [
                pl.col("t").cast(pl.Int64, strict=False).alias("ts_ms"),
                pl.col("o").cast(pl.Float64, strict=False).alias("open"),
                pl.col("c").cast(pl.Float64, strict=False).alias("close"),
                pl.col("n").cast(pl.Int64, strict=False).alias("n"),
                pl.col("v").cast(pl.Float64, strict=False).alias("volume"),
                pl.col("vw").cast(pl.Float64, strict=False).alias("vwap"),
            ]
        )

        df = df.with_columns(
            pl.from_epoch("ts_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .alias("datetime")
        ).with_columns(pl.col("datetime").dt.date().alias("date"))

        dt_open = pl.datetime(
            dt.year, dt.month, dt.day, 9, 30, 0, time_unit="ms", time_zone="America/New_York"
        )
        prev_open = pl.datetime(
            prev_dt.year,
            prev_dt.month,
            prev_dt.day,
            9,
            30,
            0,
            time_unit="ms",
            time_zone="America/New_York",
        )

        dt_minutes = HALF_DAY_MINUTES if is_half_day else FULL_DAY_MINUTES
        prev_minutes = HALF_DAY_MINUTES if prev_is_half_day else FULL_DAY_MINUTES

        df = df.with_columns(
            pl.when(pl.col("date") == pl.lit(dt))
            .then((pl.col("datetime") - dt_open).dt.total_minutes().floor())
            .when(pl.col("date") == pl.lit(prev_dt))
            .then((pl.col("datetime") - prev_open).dt.total_minutes().floor())
            .otherwise(None)
            .cast(pl.Int32)
            .alias("minute_idx")
        )

        df_in_session = df.filter(
            ((pl.col("date") == pl.lit(dt)) & (pl.col("minute_idx").is_between(0, dt_minutes - 1)))
            | (
                (pl.col("date") == pl.lit(prev_dt))
                & (pl.col("minute_idx").is_between(0, prev_minutes - 1))
            )
        )

        prev_last = (
            df_in_session.filter(pl.col("date") == pl.lit(prev_dt)).sort("minute_idx").tail(1)
        )

        df_dt = (
            df_in_session.filter(pl.col("date") == pl.lit(dt))
            .sort("datetime")
            .unique(subset=["minute_idx"], keep="last")
            .sort("minute_idx")
        )

        df_dt = df_dt.with_columns(
            [
                pl.col("minute_idx").cast(pl.Int16),
                pl.col("open").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("vwap").cast(pl.Float32),
                pl.col("n").cast(pl.Int32),
                pl.col("volume").cast(pl.Int64),
            ]
        )

        df_dt = df_dt.select(["datetime", "minute_idx", "open", "close", "n", "volume", "vwap"])

        if prev_last.height == 1:
            prev_row = (
                prev_last.select(["datetime", "open", "close", "n", "volume", "vwap"])
                .with_columns(
                    [
                        pl.lit(-1).cast(pl.Int16).alias("minute_idx"),
                        pl.col("open").cast(pl.Float32),
                        pl.col("close").cast(pl.Float32),
                        pl.col("vwap").cast(pl.Float32),
                        pl.col("n").cast(pl.Int32),
                        pl.col("volume").cast(pl.Int64),
                    ]
                )
                .select(["datetime", "minute_idx", "open", "close", "n", "volume", "vwap"])
            )
            df_out = pl.concat([prev_row, df_dt], how="vertical")
        else:
            df_out = df_dt

        _sanity_check_minutes(
            df_out.filter(pl.col("minute_idx") >= 0),
            ticker=ticker,
            dt=dt,
            is_half_day=is_half_day,
            verbose=verbose,
        )

        verbose.write(f"found {df_out.height} rows. This took {tme}.", head=False)

    return df_out
