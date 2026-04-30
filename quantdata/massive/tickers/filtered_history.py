"""
quantdata.massive.tickers.filtered_history

Adjusted daily bars for a ticker over a date range, handling ticker-change events
and filtering by minimum liquidity thresholds.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import polars as pl
from cdxcore.err import verify, verify_inp
from cdxcore.jcpool import JCPool
from cdxcore.util import fmt_date, fmt_digits
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata import env
from quantdata.biz_calendar import TZ_NY, prev_business_day
from quantdata.massive import MassiveClient
from quantdata.massive.tickers.basics import get_listing_date
from quantdata.tickers import SubDir, tickers_cache

MIN_CASH_VOLUME_M = 10
MIN_NUM_TRADES = 1000

_history_cache = tickers_cache


@_history_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    uid=lambda ticker, start, end, min_cash_volume_m, min_num_trades: (
        f"filtered_history/m{min_cash_volume_m}_{min_num_trades}/{fmt_date(start)}_{fmt_date(end)}/{ticker}"
    ),
    exclude_args=["verbose", "num_workers", "client"],
    dependencies=[get_listing_date, MassiveClient],
)
def ticker_filtered_history(
    ticker: str,
    start: date,
    end: date,
    *,
    min_cash_volume_m: float = MIN_CASH_VOLUME_M,
    min_num_trades: int = MIN_NUM_TRADES,
    num_workers: int = -2,
    client: Optional[MassiveClient] = None,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Adjusted daily bars for ``ticker`` between ``start`` and ``end``, handling ticker
    changes and filtering by minimum liquidity.

    Steps:

    1. Resolve the listing date and verify the ticker is active.
    2. Fetch ticker-change events to determine the correct symbol for each sub-period.
    3. Download adjusted daily bars per sub-period via :meth:`MassiveClient.fetch_bars`.
    4. Filter by ``min_cash_volume_m`` (USD millions) and ``min_num_trades``.
    5. Compute log returns.

    Parameters
    ----------
    ticker : str
        Current ticker symbol (used for events lookup).
    start, end : date
        Inclusive date range.
    min_cash_volume_m : float
        Minimum cash volume in millions of USD per day.
    min_num_trades : int
        Minimum number of trades per day.
    num_workers : int
        Parallel workers for bar downloads.
    client : MassiveClient | None
        Optional pre-constructed client.
    verbose : Context

    Returns
    -------
    pl.DataFrame
        Columns: date, ticker, cash_vol, n_trades, return_open_prev_close, return_close_open
    """
    client = MassiveClient() if client is None else client
    verify_inp(start <= end, lambda: f"start {fmt_date(start)} must be <= end {fmt_date(end)}")

    with verbose.write_t(lambda: f"Fetching listing date for {ticker}... ", end="") as tme:
        list_date = get_listing_date(ticker, client=client)
        verbose.write(f"done (list_date: {fmt_date(list_date)}); took {tme}.", head=False)

    with verbose.write_t(lambda: f"Fetching ticker events for {ticker}... ", end="") as tme:
        url = f"{MassiveClient.BASE_URL}/vX/reference/tickers/{ticker}/events"
        response = client(url)
        events_data = response.get("results", {}).get("events", [])

        ticker_events = []
        for event in events_data:
            if event.get("type") == "ticker_change":
                tc = event.get("ticker_change", {})
                ev_ticker = tc.get("ticker")
                ev_date_str = event.get("date")
                verify(
                    ev_ticker is not None, lambda: f"Missing ticker in change event for {ticker}"
                )
                verify(
                    ev_date_str is not None, lambda: f"Missing date in change event for {ticker}"
                )
                ticker_events.append({"date": date.fromisoformat(ev_date_str), "ticker": ev_ticker})

        if len(ticker_events) > 1:
            ticker_events.sort(key=lambda x: x["date"])
            min_test = max(start, list_date)
            verify(
                ticker_events[0]["date"] <= min_test,
                lambda: (
                    f"Cannot identify historic ticker for {ticker} from {fmt_date(min_test)}: "
                    f"first change event is {ticker_events[0]['date']} → {ticker_events[0]['ticker']}"
                ),
            )
        else:
            verify(
                len(ticker_events) == 0 or ticker_events[0]["ticker"] == ticker,
                lambda: f"Unexpected single change event for {ticker}: {ticker_events}",
            )
            ticker_events = [{"date": list_date, "ticker": ticker}]

        verbose.write(f"done ({len(ticker_events)} events); took {tme}.", head=False)

    effective_start = max(start, list_date)
    verify_inp(
        effective_start <= end,
        lambda: (
            f"Ticker {ticker}: effective start {fmt_date(effective_start)} is after end {fmt_date(end)}"
        ),
    )

    ticker_events.append({"date": end + timedelta(days=1), "ticker": ticker})
    fetch_start = max(list_date, prev_business_day(start, count_start=True))
    assert fetch_start >= ticker_events[0]["date"], (
        f"Internal: fetch_start {fetch_start} < first event {ticker_events[0]['date']}"
    )

    ticker_ranges = []
    current_start = fetch_start
    while len(ticker_events) > 1:
        while current_start >= ticker_events[0]["date"]:
            current_ticker = ticker_events.pop(0)["ticker"]
        ticker_ranges.append(
            (current_ticker, current_start, ticker_events[0]["date"] - timedelta(days=1))
        )
        current_start = ticker_events[0]["date"]

    all_bars = []

    def _period_bars(ticker_for_range: str, range_start: date, range_end: date):
        df = client.fetch_bars(
            ticker_for_range, range_start, range_end, adjusted=True, timespan="day"
        )
        verify(
            not df.is_empty(),
            lambda: (
                f"No bars for {ticker_for_range} from {fmt_date(range_start)} to {fmt_date(range_end)}"
            ),
        )
        df = df.with_columns(
            pl.col("ts_ms").dt.date().alias("date"),
            pl.col("open").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("num_trades").alias("n"),
            pl.col("volume").cast(pl.Int64),
            pl.col("vwap").cast(pl.Float32),
            pl.lit(ticker_for_range).alias("ticker"),
        ).drop_nulls()
        return ticker_for_range, range_start, range_end, df

    with verbose.write_t(lambda: f"Downloading daily bars for {ticker}... ", end="") as tme:
        jcpool = JCPool(num_workers, threading=True)
        for t_, rs, re, df in jcpool.parallel(
            jcpool.delayed(_period_bars)(t_, rs, re) for t_, rs, re in ticker_ranges
        ):
            verbose.write(lambda: f"\r  {t_} {fmt_date(rs)}–{fmt_date(re)}", end="")
            all_bars.append(df)

    with verbose.write_t(lambda: f"\rComputing metrics... ", end="") as tme:
        df = pl.concat(all_bars).sort("date")

        df = df.with_columns(
            pl.when(pl.col("vwap").is_not_null() & pl.col("volume").is_not_null())
            .then(pl.col("vwap") * pl.col("volume"))
            .otherwise(pl.lit(0.0))
            .alias("cash_volume")
        )
        df = df.filter(
            pl.col("open").is_not_null()
            & pl.col("close").is_not_null()
            & (pl.col("cash_volume") >= min_cash_volume_m * 1_000_000)
            & (pl.col("n").fill_null(0) >= min_num_trades)
        )
        verify(
            not df.is_empty(),
            lambda: (
                f"No valid bars for {ticker} after filtering at "
                f"{fmt_digits(min_cash_volume_m)}m$ cash vol / {fmt_digits(min_num_trades)} trades"
            ),
        )

        df = df.sort("date").with_columns(pl.col("close").shift(1).alias("prev_close"))

        df = df.with_columns(
            pl.when(
                pl.col("open").is_not_null()
                & pl.col("prev_close").is_not_null()
                & (pl.col("open") > 0)
                & (pl.col("prev_close") > 0)
            )
            .then((pl.col("open") / pl.col("prev_close")).log())
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("return_open_prev_close"),
            pl.when(
                pl.col("close").is_not_null()
                & pl.col("open").is_not_null()
                & (pl.col("close") > 0)
                & (pl.col("open") > 0)
            )
            .then((pl.col("close") / pl.col("open")).log())
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("return_close_open"),
        )

        df = df.with_columns(
            (pl.col("cash_volume") + 1.0).log10().cast(pl.Float32).alias("cash_vol")
        )

        df = df.filter(pl.col("date") >= start)

        df = df.select(
            "date",
            "ticker",
            "cash_vol",
            pl.col("n").cast(pl.Int32).alias("n_trades"),
            "return_open_prev_close",
            "return_close_open",
        )
        verify(
            df.height > 0,
            lambda: (
                f"No valid rows remain for {ticker} from {fmt_date(effective_start)} to {fmt_date(end)}"
            ),
        )
        verbose.write(f"done ({df.height} rows); took {tme}.", head=False)

    return df
