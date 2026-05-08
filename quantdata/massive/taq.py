"""
Trade-and-quote (TAQ) nearest-quote matching for Massive data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
from cdxcore.err import verify_inp
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import fmt_date, fmt_filename
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.biz_calendar import TZ_NY, DateRange
from quantdata.massive import MassiveClient, SubDir

_cache = MassiveClient.locations.processing("taq", fmt=SubDir.POLARS_PARQUET)

# -----------------------------
# Output schema
# -----------------------------

SCHEMA_TAQ = {
    "ts_ns": pl.Datetime("ns", TZ_NY),  # trade timestamp
    "price": pl.Float64,
    "size": pl.Float64,
    "trade_code": pl.UInt32,
    "quote_ts_ns": pl.Datetime("ns", TZ_NY),  # matched quote timestamp (null if no quote in window)
    "bid_price": pl.Float64,
    "bid_size": pl.Float64,
    "ask_price": pl.Float64,
    "ask_size": pl.Float64,
    "quote_dt_ns": pl.Int64,  # trade_ts_ns - quote_ts_ns: positive = quote leads trade
}

# -----------------------------
# Config
# -----------------------------


@version("0.1")
@dataclass(frozen=True)
class TAQMatchConfig:
    backward_window_ns: int = 1_000_000_000  #: Max look-back from trade time (ns). Default 1 s.
    forward_window_ns: int = 1_000_000_000  #: Max look-forward from trade time (ns). Default 1 s.

    def __post_init__(self):
        verify_inp(
            self.backward_window_ns >= 0,
            lambda: f"'backward_window_ns' must be non-negative, found {self.backward_window_ns}",
        )
        verify_inp(
            self.forward_window_ns >= 0,
            lambda: f"'forward_window_ns' must be non-negative, found {self.forward_window_ns}",
        )

    def __str__(self):
        return f"bk={self.backward_window_ns // 1_000_000}ms,fw={self.forward_window_ns // 1_000_000}ms"


# -----------------------------
# Core matching logic
# -----------------------------

_QUOTE_COLS = ["quote_ts_ns", "bid_price", "bid_size", "ask_price", "ask_size"]


def _match(trades: pl.DataFrame, quotes: pl.DataFrame, config: TAQMatchConfig) -> pl.DataFrame:
    """Nearest-quote matching on pre-loaded DataFrames (no caching)."""
    if trades.is_empty():
        return pl.DataFrame(schema=SCHEMA_TAQ)

    # cast ts to int64 so join_asof tolerance is in nanoseconds
    trades_i = (
        trades.with_row_index("_row")
        .with_columns(pl.col("ts_ns").cast(pl.Int64).alias("_ts_i"))
        .sort("_ts_i")
    )
    quotes_i = (
        quotes.rename({"ts_ns": "quote_ts_ns"})
        .with_columns(pl.col("quote_ts_ns").cast(pl.Int64).alias("_qts_i"))
        .sort("_qts_i")
    )

    def _asof(strategy: str, tolerance: int) -> pl.DataFrame:
        return (
            trades_i.join_asof(
                quotes_i, left_on="_ts_i", right_on="_qts_i", strategy=strategy, tolerance=tolerance
            )
            .select(["_row"] + _QUOTE_COLS)
            .rename({c: f"_{strategy}_{c}" for c in _QUOTE_COLS})
        )

    bk = _asof("backward", config.backward_window_ns)
    fw = _asof("forward", config.forward_window_ns)

    df = trades_i.join(bk, on="_row").join(fw, on="_row")

    # pick the closer quote; prefer backward on tie (pre-trade quote wins when |dt| is equal)
    bk_ts = pl.col("_backward_quote_ts_ns").cast(pl.Int64)
    fw_ts = pl.col("_forward_quote_ts_ns").cast(pl.Int64)
    trd_ts = pl.col("_ts_i")
    use_bk = pl.col("_forward_quote_ts_ns").is_null() | (
        pl.col("_backward_quote_ts_ns").is_not_null() & ((trd_ts - bk_ts) <= (fw_ts - trd_ts))
    )

    out = (
        df.with_columns(
            [
                pl.when(use_bk)
                .then(pl.col(f"_backward_{c}"))
                .otherwise(pl.col(f"_forward_{c}"))
                .alias(c)
                for c in _QUOTE_COLS
            ]
        )
        .with_columns(
            (pl.col("ts_ns").cast(pl.Int64) - pl.col("quote_ts_ns").cast(pl.Int64)).alias(
                "quote_dt_ns"
            )
        )
        .select(list(SCHEMA_TAQ.keys()))
        .with_columns([pl.col(c).cast(dtype) for c, dtype in SCHEMA_TAQ.items()])
    )
    return out


@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    dependencies=[MassiveClient.fetch_trades, MassiveClient.fetch_quotes, TAQMatchConfig],
    uid=lambda ticker, d, config: (
        f"{named_unique_filename48_8(str(config), config)}/{fmt_filename(ticker)}/taq/{fmt_date(d)}"
    ),
    exclude_args=["verbose"],
)
def match_taq(
    ticker: str,
    d: date,
    config: TAQMatchConfig = TAQMatchConfig(),
    *,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    For each trade in ``ticker`` on date ``d``, finds the closest quote within
    ``[ts_trade - config.backward_window_ns, ts_trade + config.forward_window_ns]``.

    When two quotes are equidistant, the backward (pre-trade) quote is preferred.
    Trades with no quote in the window have null quote columns.
    You can disregard trades with no quote in the window by filtering for non-null ``quote_ts_ns``.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. ``"AAPL230616C00150000"``.
    d : date
        Date for which to match trades to quotes.
    config : :class:`TAQMatchConfig`
        Matching window configuration.
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging.

    Returns
    -------
    :class:`polars.DataFrame`
        DataFrame with schema :attr:`SCHEMA_TAQ`. One row per trade.
        Quote columns are null for trades with no quote in the window.
    """
    client = MassiveClient()
    trades = client.fetch_trades(ticker, d)
    quotes = client.fetch_quotes(ticker, d)

    verbose.write(
        lambda: (
            f"Matching {trades.height} trades to {quotes.height} quotes for {ticker} on {fmt_date(d)}."
        )
    )

    return _match(trades, quotes, config)


@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    dependencies=[match_taq],
    uid=lambda ticker, date_range, config: (
        f"{named_unique_filename48_8(str(config), config, date_range.exclude_half_days)}/{fmt_filename(ticker)}/taq/{date_range}"
    ),
    exclude_args=["verbose"],
)
def match_taq_between(
    ticker: str,
    date_range: DateRange,
    config: TAQMatchConfig = TAQMatchConfig(),
    *,
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Match trades to quotes for ``ticker`` across all business days in ``date_range``.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. ``"AAPL230616C00150000"``.
    date_range : :class:`quantdata.biz_calendar.DateRange`
        Date range over which to match.
    config : :class:`TAQMatchConfig`
        Matching window configuration.
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging.

    Returns
    -------
    :class:`polars.DataFrame`
        Concatenated DataFrame for all dates with schema :attr:`SCHEMA_TAQ`.

    Raises
    ------
    :class:`RuntimeError`
        If no data was found for any date in the range.
    """
    dfs = []
    dates = date_range.get_business_days()
    tme = verbose.timer()
    for id, d in enumerate(dates):
        if tme.interval_test(1):
            verbose.write(
                lambda: (
                    f"\rMatching TAQ for {ticker} on {fmt_date(d)}... {int(id * 100 // len(dates))}% done."
                )
            )
        df = match_taq(ticker, d, config=config, verbose=verbose(1))
        if df is None or df.is_empty():
            continue
        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError(
            f"No matched TAQ data found for {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}"
        )

    with verbose.write_t(
        lambda: (
            f"Concatenating {len(dfs)} days for {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}... "
        ),
        end="",
    ) as tme:
        result = pl.concat(dfs)
        verbose.write(lambda: f"done. This took {tme}.")

    return result
