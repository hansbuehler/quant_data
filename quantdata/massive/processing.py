

from quantdata import version
import quantdata
from quantdata.biz_calendar import session_open_close
from quantdata.massive.client import MassiveClient, NORMAL_TRADE_EXCLUDE_FLAGS
import polars as pl
from datetime import date

@version("0.1", dependencies=[quantdata.biz_calendar.session_open_close, quantdata.massive.client.MassiveClient.fetch_trades])
def extract_trades(
    ticker: str,
    d: date,
    within_session: bool = True,
) -> pl.DataFrame:
    """
    Returns trades for a given underlying and date within session hours (optionally).
    Removes trades with NORMAL_TRADE_EXCLUDE_FLAGS and price or size zero or negative.

    Returns :attr:`MassiveClient.SCHEMA_TRADES` schema with the following columns:
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
            ((pl.col("trade_code") & int(NORMAL_TRADE_EXCLUDE_FLAGS)) == 0)
            & (pl.col("price") > 0)
            & (pl.col("size") > 0)
        )
    else:
        open_dt, close_dt, _ = session_open_close(d) if within_session else (None, None, None)
        trades = trades.filter(
            (pl.col("ts_ns") >= pl.lit(open_dt))
            & (pl.col("ts_ns") <= pl.lit(close_dt))
            & ((pl.col("trade_code") & int(NORMAL_TRADE_EXCLUDE_FLAGS)) == 0)
            & (pl.col("price") > 0)
            & (pl.col("size") > 0)
        )
    return trades
