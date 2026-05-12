"""
quantdata.massive.processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some simple tools to make life easier when working with Massive data.
These are not meant to be comprehensive utilities, but rather simple examples of how to filter and align trades and quotes.

Function in this file:

* :func:`filter_valid_trades` removes trades with invalid timestamps, prices, sizes, or trade codes.
* :func:`filter_valid_quotes` removes quotes with invalid timestamps, prices, sizes, or crossed/locked markets.
* :func:`align_quotes_to_trades` matches each trade to a nearby valid quote based on price and timestamp criteria.
* :func:`align_trades_to_quotes` matches each quote to a nearby valid trade based on price and timestamp criteria.
"""

from quantdata import version
import quantdata
from quantdata.biz_calendar import session_open_close
from quantdata.massive.client import MassiveClient
from quantdata.massive.trade_conds import NORMAL_TRADE_EXCLUDE_FLAGS
import numpy as np
import polars as pl
from datetime import date
from cdxcore.err import verify_inp, verify
from dataclasses import dataclass, field

@version("0.1", dependencies=[quantdata.biz_calendar.session_open_close, quantdata.massive.client.MassiveClient.fetch_trades])
def filter_valid_trades(
    ticker: str,
    d: date,
    within_session: bool = True,
) -> pl.DataFrame:
    """
    Returns trades for a given underlying and date within session hours (optionally).
    Removes trades with NORMAL_TRADE_EXCLUDE_FLAGS and price or size zero or negative.

    Returns a polars data frame with :attr:`MassiveClient.SCHEMA_TRADES` schema with the following columns:
      ts_ns       Datetime[ns,tz]
      price       Float64
      size        Float64
      trade_code  UInt32 (semantic mask)
    """
    client = MassiveClient()
    trades = client.fetch_trades(ticker, d)
    assert set(trades.columns) >= {"ts_ns", "price", "size", "trade_code"}, (
        "trades must have ts_ns, price, size, trade_code",
        trades.columns,
    )

    if not within_session:
        trades = trades.filter(
            pl.col("ts_ns").is_not_null()
            & pl.col("price").is_not_null()
            & pl.col("size").is_not_null()
            & pl.col("price").is_finite()
            & pl.col("size").is_finite()
            & ((pl.col("trade_code") & int(NORMAL_TRADE_EXCLUDE_FLAGS)) == 0)
            & (pl.col("price") > 0)
            & (pl.col("size") > 0)
        )
    else:
        open_dt, close_dt, _ = session_open_close(d)
        trades = trades.filter(
            (pl.col("ts_ns") >= pl.lit(open_dt))
            & (pl.col("ts_ns") <= pl.lit(close_dt))
            & pl.col("price").is_not_null()
            & pl.col("size").is_not_null()
            & pl.col("price").is_finite()
            & pl.col("size").is_finite()
            & ((pl.col("trade_code") & int(NORMAL_TRADE_EXCLUDE_FLAGS)) == 0)
            & (pl.col("price") > 0)
            & (pl.col("size") > 0)
        )
    return trades

@version("0.1", dependencies=[quantdata.biz_calendar.session_open_close, quantdata.massive.client.MassiveClient.fetch_quotes])
def filter_valid_quotes(
    ticker: str,
    d: date,
    within_session: bool = True,
) -> pl.DataFrame:
    """
    Returns quotes for a given underlying and date within session hours (optionally).
    Removes quotes with invalid timestamps, prices, sizes, or crossed/locked markets.

    Returns a polars data frame with :attr:`MassiveClient.SCHEMA_QUOTES` schema with the following columns:
      ts_ns       Datetime[ns,tz]
      bid_price   Float64
      bid_size    Float64
      ask_price   Float64
      ask_size    Float64
    """
    client = MassiveClient()
    quotes = client.fetch_quotes(ticker, d)
    assert set(quotes.columns) >= {"ts_ns", "bid_price", "bid_size", "ask_price", "ask_size"}, (
        "quotes must have ts_ns, bid_price, bid_size, ask_price, ask_size",
        quotes.columns,
    )

    valid_quotes = (
        pl.col("ts_ns").is_not_null()
        & pl.col("bid_price").is_not_null()
        & pl.col("ask_price").is_not_null()
        & pl.col("bid_price").is_finite()
        & pl.col("ask_price").is_finite()
        & (pl.col("bid_price") > 0.0)
        & (pl.col("ask_price") > 0.0)
        & (pl.col("ask_price") > pl.col("bid_price"))
        & pl.col("bid_size").is_not_null()
        & pl.col("ask_size").is_not_null()
        & pl.col("bid_size").is_finite()
        & pl.col("ask_size").is_finite()
        & (pl.col("bid_size") >= 0.0)
        & (pl.col("ask_size") >= 0.0)
        & ((pl.col("bid_size") + pl.col("ask_size")) > 0.0)
    )

    if within_session:
        open_dt, close_dt, _ = session_open_close(d)
        valid_quotes = (
            (pl.col("ts_ns") >= pl.lit(open_dt))
            & (pl.col("ts_ns") <= pl.lit(close_dt))
            & valid_quotes
        )

    return quotes.filter(valid_quotes).sort("ts_ns")



@version("0.1")
@dataclass(frozen=True)
class TradeQuoteAlignmentConfig:
    """
    Configuration for matching trades to nearby valid quotes.
    Times are in milliseconds.
    """
    lookback_ms: int = 250
    lookfwd_ms: int = 250
    drop_unmatched: bool = True
    within_session: bool = True

    def __post_init__(self):
        verify_inp(self.lookback_ms >= 0, f"lookback_ms must be non-negative, found {self.lookback_ms}")
        verify_inp(self.lookfwd_ms >= 0, f"lookfwd_ms must be non-negative, found {self.lookfwd_ms}")


def _ts_ns(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.col("ts_ns").cast(pl.Int64)).to_series().to_numpy()

def _find_quote(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    price: float,
    left: int,
    right: int,
    prefer_last: bool,
) -> int:
    if left == right:
        return -1
    valid = np.flatnonzero((bid_price[left:right] <= price) & (price <= ask_price[left:right]))
    if len(valid) == 0:
        return -1
    return left + (valid[-1] if prefer_last else valid[0])


def _find_trade(
    trade_price: np.ndarray,
    bid: float,
    ask: float,
    left: int,
    right: int,
    prefer_last: bool,
) -> int:
    if left == right:
        return -1
    valid = np.flatnonzero((bid <= trade_price[left:right]) & (trade_price[left:right] <= ask))
    if len(valid) == 0:
        return -1
    return left + (valid[-1] if prefer_last else valid[0])


SCHEMA_ALIGN_QUOTES_TO_TRADES = {
    "ts_ns": np.dtype("datetime64[ns]"),
    "price": np.float64,
    "size": np.float64,
    "trade_code": np.uint32,
    "quote_ts_ns": np.dtype("datetime64[ns]"),
    "bid_price": np.float64,
    "bid_size": np.float64,
    "ask_price": np.float64,
    "ask_size": np.float64,
}

@version("0.1", dependencies=[filter_valid_quotes, filter_valid_trades, TradeQuoteAlignmentConfig])
def align_quotes_to_trades(
    ticker: str,
    d: date,
    config: TradeQuoteAlignmentConfig = TradeQuoteAlignmentConfig(),
) -> pl.DataFrame:
    """
    Fetch valid trades and quotes for ``ticker`` on ``d`` and align them on a common timeline.

    For each trade, this first finds the latest quote between ``config.lookback_ms``
    and the trade timestamp where ``bid_price <= price <= ask_price``. If no such
    quote exists, it finds the first matching quote up to ``config.lookfwd_ms`` after
    the trade.
    The output keeps all trade columns and adds the matched quote columns. If
    ``config.drop_unmatched`` is False, unmatched trades have null quote fields.
    """
    trades = filter_valid_trades(ticker, d, within_session=config.within_session).sort("ts_ns")
    quotes = filter_valid_quotes(ticker, d, within_session=config.within_session).sort("ts_ns")

    trade_ts = _ts_ns(trades)
    quote_ts = _ts_ns(quotes)
    trade_price = trades["price"].to_numpy() if "price" in trades.columns else np.array([], dtype=float)
    bid_price = quotes["bid_price"].to_numpy() if "bid_price" in quotes.columns else np.array([], dtype=float)
    ask_price = quotes["ask_price"].to_numpy() if "ask_price" in quotes.columns else np.array([], dtype=float)

    matched_rows = np.full(len(trades), -1, dtype=np.int64)
    lookback_ns = int(config.lookback_ms) * 1_000_000
    lookfwd_ns = int(config.lookfwd_ms) * 1_000_000

    for i, (ts, price) in enumerate(zip(trade_ts, trade_price)):
        quote_row = _find_quote(
            bid_price=bid_price,
            ask_price=ask_price,
            price=price,
            left=np.searchsorted(quote_ts, ts - lookback_ns, side="left"),
            right=np.searchsorted(quote_ts, ts, side="right"),
            prefer_last=True,
        )
        if quote_row < 0:
            quote_row = _find_quote(
                bid_price=bid_price,
                ask_price=ask_price,
                price=price,
                left=np.searchsorted(quote_ts, ts, side="left"),
                right=np.searchsorted(quote_ts, ts + lookfwd_ns, side="right"),
                prefer_last=False,
            )
        if quote_row < 0:
            continue
        matched_rows[i] = quote_row

    out = (
        trades.with_row_index("_trade_row")
        .with_columns(
            pl.Series("_quote_row", matched_rows),
        )
        .join(
            quotes.with_row_index("_quote_row")
            .with_columns(pl.col("_quote_row").cast(pl.Int64))
            .rename({"ts_ns": "quote_ts_ns"}),
            on="_quote_row",
            how="left",
        )
        .sort("_trade_row")
        .drop("_trade_row", "_quote_row")
    )

    if config.drop_unmatched:
        out = out.filter(pl.col("quote_ts_ns").is_not_null())

    verify(set(out.columns) == set(SCHEMA_ALIGN_QUOTES_TO_TRADES), f"Output is missing expected columns: {sorted(set(SCHEMA_ALIGN_QUOTES_TO_TRADES) - set(out.columns))} and has superfluous columns: {sorted(set(out.columns) - set(SCHEMA_ALIGN_QUOTES_TO_TRADES))}")
    return out


SCHEMA_ALIGN_TRADES_TO_QUOTES = {
    "ts_ns": np.dtype("datetime64[ns]"),
    "bid_price": np.float64,
    "bid_size": np.float64,
    "ask_price": np.float64,
    "ask_size": np.float64,
    "trade_ts_ns": np.dtype("datetime64[ns]"),
    "price": np.float64,
    "size": np.float64,
    "trade_code": np.uint32,
}

@version("0.1", dependencies=[filter_valid_quotes, filter_valid_trades])
def align_trades_to_quotes(
    ticker: str,
    d: date,
    config: TradeQuoteAlignmentConfig = TradeQuoteAlignmentConfig(),
) -> pl.DataFrame:
    """
    Fetch valid trades and quotes for ``ticker`` on ``d`` and align trades onto quotes.

    For each quote, this first finds the latest trade between ``config.lookback_ms``
    and the quote timestamp where ``bid_price <= price <= ask_price``. If no such
    trade exists, it finds the first matching trade up to ``config.lookfwd_ms`` after
    the quote. The output keeps all quote columns and adds matched trade columns. If
    ``config.drop_unmatched`` is False, unmatched quotes have null trade fields.
    """
    trades = filter_valid_trades(ticker, d, within_session=config.within_session).sort("ts_ns")
    quotes = filter_valid_quotes(ticker, d, within_session=config.within_session).sort("ts_ns")

    trade_ts = _ts_ns(trades)
    quote_ts = _ts_ns(quotes)
    trade_price = trades["price"].to_numpy() if "price" in trades.columns else np.array([], dtype=float)
    bid_price = quotes["bid_price"].to_numpy() if "bid_price" in quotes.columns else np.array([], dtype=float)
    ask_price = quotes["ask_price"].to_numpy() if "ask_price" in quotes.columns else np.array([], dtype=float)

    matched_rows = np.full(len(quotes), -1, dtype=np.int64)
    lookback_ns = int(config.lookback_ms) * 1_000_000
    lookfwd_ns = int(config.lookfwd_ms) * 1_000_000

    for i, (ts, bid, ask) in enumerate(zip(quote_ts, bid_price, ask_price)):
        trade_row = _find_trade(
            trade_price=trade_price,
            bid=bid,
            ask=ask,
            left=np.searchsorted(trade_ts, ts - lookback_ns, side="left"),
            right=np.searchsorted(trade_ts, ts, side="right"),
            prefer_last=True,
        )
        if trade_row < 0:
            trade_row = _find_trade(
                trade_price=trade_price,
                bid=bid,
                ask=ask,
                left=np.searchsorted(trade_ts, ts, side="left"),
                right=np.searchsorted(trade_ts, ts + lookfwd_ns, side="right"),
                prefer_last=False,
            )
        if trade_row < 0:
            continue
        matched_rows[i] = trade_row

    out = (
        quotes.with_row_index("_quote_row")
        .with_columns(
            pl.Series("_trade_row", matched_rows),
        )
        .join(
            trades.with_row_index("_trade_row")
            .with_columns(pl.col("_trade_row").cast(pl.Int64))
            .rename({"ts_ns": "trade_ts_ns"}),
            on="_trade_row",
            how="left",
        )
        .sort("_quote_row")
        .drop("_quote_row", "_trade_row")
    )

    if config.drop_unmatched:
        out = out.filter(pl.col("trade_ts_ns").is_not_null())
    verify(set(out.columns) == set(SCHEMA_ALIGN_TRADES_TO_QUOTES), f"Output is missing expected columns: {sorted(set(SCHEMA_ALIGN_TRADES_TO_QUOTES) - set(out.columns))} and has superfluous columns: {sorted(set(out.columns) - set(SCHEMA_ALIGN_TRADES_TO_QUOTES))}")
    return out
