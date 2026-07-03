"""
Aggregate minute bars into time buckets and merge with calendar data.

Change requested:
- Remove the need for prev_dt in this file.
- fetch_minute_bars() now returns dt minutes PLUS (optionally) a single extra row
  with minute_idx = -1 representing the previous day's last regular-session minute.
- We use that -1 row ONLY to seed the first return for dt.
- Everything else remains based on dt only.
"""

from __future__ import annotations

import math as math
from datetime import date
from typing import Optional

import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.jcpool import JCPool
from cdxcore.subdir import SubDir, version
from cdxcore.util import fmt_date, fmt_digits, fmt_list

from quantdata import Context, env
from quantdata.biz_calendar import (
    DAYS_PER_YEAR,
    get_business_days,
    get_half_days,
    prev_business_day,
)
from quantdata.massive import MassiveClient
from quantdata.massive.tickers.basics import get_listing_date
from quantdata.tickers.ticker_calendar import EXPIRY, DateRange, build_ticker_calendar
from quantdata.volume_curves.minutes import fetch_minute_bars

_curves_cache = env.locations.processing("volume_curves")

CAL_COLS = [
    "date",
    "is_ex_div",
    "is_split",
    "has_ticker_change",
    "is_earnings",
    "is_half_day",
    "has_expiry",
]

_TOP_TRADED_ETFS = [
    "SPY",
    "QQQ",
    "IWM",
    "VOO",
    "IVV",
    "DIA",
    "XLF",
    "XLE",
    "XLK",
    "XLI",
    "XLV",
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "HYG",
    "LQD",
    "TLT",
    "VTI",
    "GLD",
    "SLV",
    "EEM",
    "FXI",
    "ARKK",
]


@version("0.2")
def _aggregate_minutes_to_buckets(
    ticker: str,
    minutes: pl.DataFrame,
    dt: date,
    bucket_minutes: int,
    is_half_day: bool,
) -> dict:
    minutes_per_day = 210 if is_half_day else 390
    num_buckets = (minutes_per_day + bucket_minutes - 1) // bucket_minutes
    days_per_year = float(DAYS_PER_YEAR)

    required = {"minute_idx", "open", "close", "vwap", "volume", "n", "datetime"}
    missing = required - set(minutes.columns)
    verify(
        not missing,
        lambda: f"Missing required minute columns for {ticker} {fmt_date(dt)}: {sorted(missing)}",
    )
    verify(
        not minutes.is_empty(),
        lambda: f"No minute bar data at all found for {ticker} {fmt_date(dt)}",
    )

    # ============================================================
    # 0) Overnight return
    # ============================================================
    anchor = minutes.filter(pl.col("minute_idx") == -1).select(pl.col("close"))
    verify(
        anchor.height == 1,
        lambda: (
            f"Expected exactly 1 anchor row minute_idx=-1 for {ticker} {fmt_date(dt)}, got {anchor.height}"
        ),
    )
    prev_close_anchor = float(anchor.item())
    verify(
        prev_close_anchor > 0.0,
        lambda: f"Bad prev_close_anchor={prev_close_anchor} for {ticker} {fmt_date(dt)}",
    )

    first_idx = minutes.filter(pl.col("minute_idx") >= 0).select(pl.col("minute_idx").min()).item()
    verify(
        first_idx is not None,
        lambda: f"No intraday minutes (minute_idx>=0) for {ticker} {fmt_date(dt)}",
    )

    open_px = (
        minutes.filter(pl.col("minute_idx") == pl.lit(int(first_idx))).select(pl.col("open")).item()
    )
    open_px = float(open_px)
    verify(open_px > 0.0, lambda: f"Bad open_px={open_px} for {ticker} {fmt_date(dt)}")

    ratio = open_px / prev_close_anchor
    if ratio <= 0.0:
        on_lr = 0.0
    elif ratio < 0.2 or ratio > 5.0:
        # likely split / stale anchor; don't blow up overnight features
        on_lr = 0.0
    else:
        on_lr = math.log(ratio)

    vol_overnight = abs(float(on_lr)) * math.sqrt(days_per_year)

    # ============================================================
    # 1) Intraday-only minutes (minute_idx >= 0)
    # ============================================================
    minutes_intra = minutes.filter(pl.col("minute_idx") >= 0).sort("minute_idx")
    verify(
        not minutes_intra.is_empty(),
        lambda: f"No minutes found after filtering minute_idx>=0 for {ticker} {fmt_date(dt)}",
    )

    # Seed prev_close: for first_idx use open; for later minutes use previous close
    minutes_intra = minutes_intra.with_columns(
        pl.when(pl.col("minute_idx") == pl.lit(int(first_idx)))
        .then(pl.col("open"))
        .otherwise(pl.col("close").shift(1))
        .alias("prev_close")
    )

    minutes_intra = minutes_intra.with_columns(
        pl.when(
            pl.col("prev_close").is_null()
            | (pl.col("prev_close") <= 0)
            | pl.col("close").is_null()
            | (pl.col("close") <= 0)
        )
        .then(pl.lit(0.0))
        .otherwise((pl.col("close") / pl.col("prev_close")).log())
        .alias("log_return")
    )

    minutes_intra = minutes_intra.with_columns(
        (pl.col("minute_idx") // bucket_minutes).cast(pl.Int16).alias("bucket_idx")
    )

    minutes_intra = minutes_intra.with_columns(
        (pl.col("vwap").fill_null(0.0) * pl.col("volume").fill_null(0.0)).alias("minute_cash_vol")
    )

    # Scheduled bucket length (NOT observed minutes)
    minutes_intra = minutes_intra.with_columns(
        pl.when(
            pl.lit(bucket_minutes)
            < (pl.lit(minutes_per_day) - pl.col("bucket_idx").cast(pl.Int32) * bucket_minutes)
        )
        .then(pl.lit(bucket_minutes))
        .otherwise(pl.lit(minutes_per_day) - pl.col("bucket_idx").cast(pl.Int32) * bucket_minutes)
        .cast(pl.Int16)
        .alias("bucket_len")
    )

    # ============================================================
    # 2) Daily metrics
    #    vol_daily includes overnight variance
    # ============================================================
    last_close = float(minutes_intra.select(pl.col("close").last()).item())
    daily_return = (
        math.log(last_close / prev_close_anchor)
        if (last_close > 0 and prev_close_anchor > 0)
        else 0.0
    )

    daily_df = minutes_intra.select(
        [
            (
                (
                    (pl.col("log_return").pow(2).sum() + pl.lit(float(on_lr * on_lr)))
                    * pl.lit(days_per_year)
                ).sqrt()
            )
            .cast(pl.Float32)
            .alias("vol_daily"),
            (pl.col("minute_cash_vol").sum() + 1.0)
            .log10()
            .cast(pl.Float32)
            .alias("cash_vol_daily"),
            pl.col("n").fill_null(0).sum().cast(pl.Int32).alias("n_trades_daily"),
        ]
    )
    verify(daily_df.height == 1, lambda: f"Expected 1 row daily_df, got {daily_df.height}")

    vol_daily = float(daily_df["vol_daily"][0])
    cash_vol_daily = float(daily_df["cash_vol_daily"][0])
    n_trades_daily = int(daily_df["n_trades_daily"][0])
    return_daily = float(daily_return)

    # ============================================================
    # 3) Bucket metrics
    # ============================================================
    buckets = (
        minutes_intra.group_by("bucket_idx")
        .agg(
            [
                (
                    (
                        pl.col("log_return").pow(2).sum()
                        * pl.lit(days_per_year * float(minutes_per_day))
                        / pl.col("bucket_len").first().cast(pl.Float64)
                    ).sqrt()
                )
                .cast(pl.Float32)
                .alias("vol"),
                (
                    pl.col("minute_cash_vol").sum()
                    * (
                        pl.lit(float(minutes_per_day))
                        / pl.col("bucket_len").first().cast(pl.Float64)
                    )
                    + 1.0
                )
                .log10()
                .cast(pl.Float32)
                .alias("cash_vol"),
                pl.col("n").fill_null(0).sum().cast(pl.Int32).alias("n_trades"),
                pl.col("log_return").sum().cast(pl.Float64).alias("return_bucket"),
            ]
        )
        .sort("bucket_idx")
    )

    date_from_minutes = minutes_intra.select(pl.col("datetime").dt.date().first()).item()

    record: dict = {
        "date": date_from_minutes,
        "cash_vol_daily": cash_vol_daily,
        "vol_daily": vol_daily,
        "n_trades_daily": n_trades_daily,
        "return_daily": return_daily,
        "vol_overnight": float(vol_overnight),
    }

    bdict = {
        int(r["bucket_idx"]): (
            float(r["vol"]),
            float(r["cash_vol"]),
            int(r["n_trades"]),
            float(r["return_bucket"]),
        )
        for r in buckets.to_dicts()
    }

    for i in range(num_buckets):
        if i in bdict:
            v, cv, nt, rb = bdict[i]
            record[f"vol_bucket_{i:02d}"] = v
            record[f"cash_vol_bucket_{i:02d}"] = cv
            record[f"n_trades_bucket_{i:02d}"] = nt
            record[f"return_bucket_{i:02d}"] = rb
        else:
            record[f"vol_bucket_{i:02d}"] = 0.0
            record[f"cash_vol_bucket_{i:02d}"] = 0.0
            record[f"n_trades_bucket_{i:02d}"] = 0
            record[f"return_bucket_{i:02d}"] = 0.0

    return record


@_curves_cache(fmt=SubDir.PICKLE).cache(
    "0.1",
    dependencies=[MassiveClient],
    uid=lambda start, end, bucket_minutes: (
        f"{fmt_date(start)} {fmt_date(end)}@{bucket_minutes}/curve_def"
    ),
)
def curve_def(
    start: date,
    end: date,
    bucket_minutes: int,
) -> dict:
    """The only point of this function is to generate the 'curve_def' file."""
    return dict(start=start, end=end, bucket_minutes=bucket_minutes)


@_curves_cache.cache(
    "0.1",
    dependencies=[
        build_ticker_calendar,
        fetch_minute_bars,
        _aggregate_minutes_to_buckets,
        get_listing_date,
        MassiveClient,
        curve_def,
    ],
    uid=lambda ticker, start, end, bucket_minutes: (
        f"{fmt_date(start)} {fmt_date(end)}@{bucket_minutes}/{ticker}"
    ),
    exclude_args=["client", "verbose", "num_workers", "threading"],
)
def ticker_curve_bins(
    ticker: str,
    start: date,
    end: date,
    bucket_minutes: int = 15,
    client: Optional[MassiveClient] = None,
    num_workers: int = -2,
    threading: bool = False,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Generates bucketed features for each day in (start..end] (i.e. uses previous day for seeding returns).

    Note: The first trading day in the range cannot include an overnight seed unless you fetched the prior day.
    We therefore start from the second trading day (len(trading_days)-1 rows), consistent with your previous logic.
    """
    verify_inp(
        390 % bucket_minutes == 0,
        lambda: f"Total minutes of a day, 390, not divisible by bucket_minutes {bucket_minutes}",
    )
    verify_inp(
        210 % bucket_minutes == 0,
        lambda: (
            f"Total minutes of a half day, 210, not divisible by bucket_minutes {bucket_minutes}"
        ),
    )

    curve_def(start, end, bucket_minutes)

    client = MassiveClient() if client is None else client
    list_date = get_listing_date(ticker, client=client)
    verify_inp(
        end >= list_date,
        lambda: (
            f"End date {fmt_date(end)} is before listing date {fmt_date(list_date)} for {ticker}"
        ),
    )
    if list_date > start:
        verbose.write(
            lambda: (
                f"Listing date {fmt_date(list_date)} is after start date {fmt_date(start)} for {ticker}; using listing date as start date"
            )
        )
        start = max(start, list_date)
    trading_days = get_business_days(max(list_date, prev_business_day(start)), end)
    verify_inp(
        len(trading_days) > 0,
        lambda: (
            f"No trading days between {fmt_date(start)} and {fmt_date(end)} (based on listing date {fmt_date(list_date)})"
        ),
    )
    half_days = set(get_half_days(start, end))

    calendar = build_ticker_calendar(
        ticker,
        DateRange(start=trading_days[0], end=trading_days[-1], exclude_half_days=False),
        verbose=verbose,
    ).rename({"expiry_code": "has_expiry"})

    day_records: list[dict] = []
    jcpool = JCPool(num_workers, threading=threading)
    tmx = verbose.timer()

    def one_day(
        ticker: str, dt: date, is_half_day: bool, prev_dt: date, prev_is_half_day: bool
    ) -> tuple[date, dict]:
        client_local = MassiveClient()
        minutes = fetch_minute_bars(
            ticker,
            dt,
            is_half_day=is_half_day,
            prev_dt=prev_dt,
            prev_is_half_day=prev_is_half_day,
            client=client_local,
            verbose=Context.quiet,
        )
        if minutes.is_empty():
            raise RuntimeError(f"No minute bar data found for {ticker} {fmt_date(dt)}")
        day_record = _aggregate_minutes_to_buckets(
            ticker=ticker,
            minutes=minutes,
            dt=dt,
            bucket_minutes=bucket_minutes,
            is_half_day=is_half_day,
        )
        return dt, day_record

    work_days = trading_days[1:]
    tmx = verbose.timer()

    for dt, day_record in jcpool.parallel(
        jcpool.delayed(one_day)(ticker, dt, dt in half_days, prev_dt, prev_dt in half_days)
        for dt, prev_dt in zip(trading_days[1:], trading_days[:-1])
    ):
        day_records.append(day_record)
        if tmx.interval_test(1):
            verbose.write(
                lambda: (
                    f"\rBucketing data for {ticker} {fmt_date(dt)}: {int(len(day_records) * 100 // len(work_days))}% "
                ),
                end="",
            )

    df = pl.DataFrame(day_records, infer_schema_length=None)

    vol_cols = [c for c in df.columns if c.startswith("vol_")]
    cash_cols = [c for c in df.columns if c.startswith("cash_vol_")]
    ntrade_cols = [c for c in df.columns if c.startswith("n_trades_")]

    if vol_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in vol_cols])
    if cash_cols:
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in cash_cols])
    if ntrade_cols:
        df = df.with_columns([pl.col(c).cast(pl.Int32) for c in ntrade_cols])

    cal_cols_present = [c for c in CAL_COLS if c in calendar.columns]
    if "date" in cal_cols_present and len(cal_cols_present) > 1:
        df = df.join(calendar.select(cal_cols_present), on="date", how="left")

    bool_cols = ["is_ex_div", "is_split", "has_ticker_change", "is_earnings", "is_half_day"]
    for col in bool_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(False).cast(pl.Boolean).alias(col))
        else:
            df = df.with_columns(pl.col(col).fill_null(False).cast(pl.Boolean))

    if "has_expiry" not in df.columns:
        df = df.with_columns(pl.lit(EXPIRY.NONE, dtype=pl.Int8).alias("has_expiry"))
    else:
        df = df.with_columns(pl.col("has_expiry").cast(pl.Int8))

    verbose.write(
        lambda: f"\rBucketed {len(work_days)} trading days for {ticker}: done; this took {tmx}."
    )
    return df.sort("date")


# ================================================================
# Command-line interface
# python -m curve_bins -tickers TSLA AAPL GS JPM NVDA -etfs
# ================================================================

if __name__ == "__main__":
    import argparse as argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", type=int, default=16)
    parser.add_argument("-etfs", action="store_true")
    parser.add_argument("-tickers", nargs="+", type=str, help="List of tickers", default=[])
    parser.add_argument("-skip", nargs="+", type=str, help="List of tickers to skip", default=[])
    parser.add_argument("-start", type=str, default="2015-01-01")
    parser.add_argument("-end", type=str, default="2026-01-01")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise ValueError("'start' must be before 'end'")
    trading_days = get_business_days(start, end)

    tickers = []
    if len(args.tickers) > 0:
        print(f"Using user defined tickers '{fmt_list(args.tickers)}'")
        tickers += args.tickers

    if args.etfs:
        print(f"Using ETFs '{fmt_list(_TOP_TRADED_ETFS)}'")
        tickers += _TOP_TRADED_ETFS

    if len(args.skip) > 0:
        print(f"Skipping tickers '{fmt_list(args.skip)}'")
        skip = set(args.skip)
        tickers = [t for t in tickers if t not in skip]

    verbose = Context.all
    for it, ticker in enumerate(tickers):
        verbose.write(
            lambda: (
                f"\rProcessing {fmt_digits(len(tickers))} tickers: {int(it * 100 // len(tickers))}% done. Now processing {ticker} from {fmt_date(start)} to {fmt_date(end)}."
            )
        )
        df = ticker_curve_bins(
            ticker,
            start,
            end,
            bucket_minutes=15,
            num_workers=args.mp,
            threading=False,
            verbose=verbose(1),
        )
    verbose.write("All done")
