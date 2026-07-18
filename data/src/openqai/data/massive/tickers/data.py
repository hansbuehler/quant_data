"""
quantdata.massive.tickers.data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build underlying tick-level data from Massive: unadjusted trades, weighted mid-price
tape, and 1-second trade bars.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import polars as pl
from cdxcore.err import verify
from cdxcore.util import fmt_date
from cdxcore.version import version

from quantdata.biz_calendar import TZ_NY, full_second_grid_ns, session_open_close
from quantdata.massive import MassiveClient
from quantdata.tickers import SubDir, _locations_tickers

_data_cache = _locations_tickers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@version("0.1")
def _second_bin_ix_expr(dt_ns_expr: pl.Expr, open_dt: datetime) -> pl.Expr:
    base = pl.lit(open_dt).cast(pl.Datetime("ns", TZ_NY))
    return ((dt_ns_expr - base).dt.total_seconds()).cast(pl.Int32).alias("second_bin_ix")


@version("0.1")
def _bin_type_series(n: int) -> pl.Series:
    # bin_type: 1 = opening bin, 0 = middle bin, 2 = closing bin
    if n <= 0:
        return pl.Series("bin_type", [], dtype=pl.Int8)
    if n == 1:
        return pl.Series("bin_type", [1], dtype=pl.Int8)
    return pl.Series("bin_type", [1] + [0] * (n - 2) + [2], dtype=pl.Int8)


@version("0.1")
def _map_close_inclusive_to_last_bin(ts_col: str, close_dt: datetime) -> pl.Expr:
    """
    Map any timestamp exactly at ``close_dt`` to the last bin key ``close_dt - 1s``,
    implementing the semantics:

    * all bins except last: ``[t, t+1s)``
    * last bin: ``[close_dt-1s, close_dt]``
    """
    last_bin_dt = close_dt - timedelta(seconds=1)
    return (
        pl.when(pl.col(ts_col) == pl.lit(close_dt))
        .then(pl.lit(last_bin_dt))
        .otherwise(pl.col(ts_col))
    )


# ---------------------------------------------------------------------------
# Unadjusted trades
# ---------------------------------------------------------------------------


@_data_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    uid=lambda underlying, d: f"unadjusted_trades/{underlying}/{fmt_date(d)}",
    exclude_args=["client"],
    dependencies=[MassiveClient.fetch_trades, session_open_close],
)
def build_unadjusted_trades(
    underlying: str,
    d: date,
    *,
    client: MassiveClient | None = None,
) -> pl.DataFrame:
    """
    Unadjusted trades for ``underlying`` on ``d``, filtered to RTH session ``[open, close]``.

    Returns
    -------
    pl.DataFrame
        Columns: ts_ns (Datetime[ns, NY]), price (Float32), size (Float32), trade_code (UInt32)
    """
    client = MassiveClient() if client is None else client
    open_dt, close_dt, _sec, _hd = session_open_close(d)
    trades = client.fetch_trades(underlying, d)

    assert set(trades.columns) >= {"ts_ns", "price", "size"}, trades.columns
    verify(not trades.is_empty(), f"No trades for {underlying} on {d}")

    trades = trades.filter(
        (pl.col("ts_ns") >= pl.lit(open_dt)) & (pl.col("ts_ns") <= pl.lit(close_dt))
    )
    return trades


# ---------------------------------------------------------------------------
# Underlying spot (weighted mid-price)
# ---------------------------------------------------------------------------


@_data_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    uid=lambda underlying, d: f"spot_wmid/{underlying}/{fmt_date(d)}",
    exclude_args=["trades", "client"],
    dependencies=[
        MassiveClient.fetch_quotes,
        MassiveClient.fetch_day_summary,
        build_unadjusted_trades,
    ],
)
def build_unadjusted_spot_wmid(
    underlying: str,
    d: date,
    *,
    trades: pl.DataFrame | None = None,
    client: MassiveClient | None = None,
) -> pl.DataFrame:
    """
    Weighted mid-price tape for ``underlying`` on ``d``, suitable for asof-joins.

    The tape is initialised at session open and close with the unadjusted open/close
    from ``/v1/open-close`` so a spot price is always available.  Quote wmid and
    trade prices are interleaved; trades win on identical timestamps.

    Returns
    -------
    pl.DataFrame
        Columns: u_ts_ns (Datetime[ns, NY]), spot (Float32), is_trade (Boolean)
    """
    client = MassiveClient() if client is None else client
    open_dt, close_dt, _sec, _hd = session_open_close(d)

    oc = client.fetch_day_summary(underlying, d, adjusted=False)
    open_px = float(oc["open"])
    close_px = float(oc["close"])
    assert math.isfinite(open_px) and open_px > 0.0, f"Bad open price {underlying} {d}: {open_px}"
    assert math.isfinite(close_px) and close_px > 0.0, (
        f"Bad close price {underlying} {d}: {close_px}"
    )

    schema = {
        "u_ts_ns": pl.Datetime("ns", TZ_NY),
        "spot": pl.Float32,
        "is_trade": pl.Boolean,
    }

    init = pl.DataFrame(
        {"u_ts_ns": [open_dt, close_dt], "spot": [open_px, close_px], "is_trade": [False, False]},
        schema=schema,
    )

    q = client.fetch_quotes(underlying, d)
    if not q.is_empty():
        q = (
            q.filter((pl.col("ts_ns") >= pl.lit(open_dt)) & (pl.col("ts_ns") <= pl.lit(close_dt)))
            .with_columns((pl.col("bid_size") + pl.col("ask_size")).alias("_sz_sum"))
            .with_columns(
                pl.when(pl.col("_sz_sum") > 0)
                .then(
                    (
                        pl.col("bid_price") * pl.col("ask_size")
                        + pl.col("ask_price") * pl.col("bid_size")
                    )
                    / pl.col("_sz_sum")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias("spot")
            )
            .select(pl.col("ts_ns").alias("u_ts_ns"), "spot")
            .drop_nulls()
            .with_columns(pl.lit(False).cast(pl.Boolean).alias("is_trade"))
            .cast(schema)
            .sort("u_ts_ns")
        )
    else:
        q = pl.DataFrame(schema=schema)

    trades = build_unadjusted_trades(underlying, d, client=client) if trades is None else trades
    t = (
        trades.filter((pl.col("ts_ns") >= pl.lit(open_dt)) & (pl.col("ts_ns") <= pl.lit(close_dt)))
        .select(pl.col("ts_ns").alias("u_ts_ns"), pl.col("price").cast(pl.Float32).alias("spot"))
        .with_columns(pl.lit(True).cast(pl.Boolean).alias("is_trade"))
        .cast(schema)
        .sort("u_ts_ns")
    )

    out = (
        pl.concat([init, q, t], how="vertical")
        .sort(["u_ts_ns", "is_trade"], descending=[False, True])
        .unique(subset=["u_ts_ns"], keep="first")
        .sort("u_ts_ns")
    )
    verify(out.height >= 1, lambda: f"Spot wmid tape empty for {underlying} on {d}")
    return out


# ---------------------------------------------------------------------------
# 1-second trade bars
# ---------------------------------------------------------------------------


@_data_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.1",
    uid=lambda underlying, d: f"u1s_bars/{underlying}/{fmt_date(d)}",
    exclude_args=["trades", "client"],
    dependencies=[
        MassiveClient.fetch_trades,
        session_open_close,
        full_second_grid_ns,
        build_unadjusted_trades,
    ],
)
def build_1s_trade_bars(
    underlying: str,
    d: date,
    *,
    trades: pl.DataFrame | None = None,
    client: MassiveClient | None = None,
) -> pl.DataFrame:
    """
    Per-second OHLCV trade bars for ``underlying`` on ``d``.

    Bin semantics:

    * all bins except last: ``[t, t+1s)``
    * last bin: ``[close_dt-1s, close_dt]`` (inclusive close)

    Returns
    -------
    pl.DataFrame
        Columns: dt_ns, second_bin_ix, bin_type, open, close, high, low, last_trade,
        num_trades, vwap, log_cash_volume, log_volume, date, underlying, is_half_day
    """
    client = MassiveClient() if client is None else client
    open_dt, close_dt, _sec, is_half_day = session_open_close(d)

    grid = full_second_grid_ns(open_dt, close_dt)
    verify(
        grid.to_list()[-1] == close_dt - timedelta(seconds=1),
        lambda: f"Grid does not end at close_dt-1s: {grid.to_list()[-1]} vs {close_dt}",
    )

    u = pl.DataFrame({"dt_ns": grid}).with_columns(
        _second_bin_ix_expr(pl.col("dt_ns"), open_dt),
        _bin_type_series(len(grid)),
    )

    trades = build_unadjusted_trades(underlying, d, client=client) if trades is None else trades

    trades = (
        trades.with_columns(
            pl.col("ts_ns").dt.truncate("1s").alias("sec_ns_raw"),
            (pl.col("price") * pl.col("size")).alias("cash"),
        )
        .with_columns(_map_close_inclusive_to_last_bin("sec_ns_raw", close_dt).alias("sec_ns"))
        .drop("sec_ns_raw")
        .sort("ts_ns")
        .with_columns(
            pl.col("ts_ns").cast(pl.Datetime("ns", TZ_NY)),
            pl.col("price").cast(pl.Float32),
            pl.col("size").cast(pl.Float32),
            pl.col("cash").cast(pl.Float32),
            pl.col("sec_ns").cast(pl.Datetime("ns", TZ_NY)),
        )
    )

    agg = (
        trades.group_by("sec_ns")
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").last().alias("close"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("size").sum().fill_null(0.0).alias("volume"),
            pl.col("cash").sum().fill_null(0.0).alias("cash_volume"),
            pl.len().cast(pl.Int64).alias("num_trades"),
        )
        .sort("sec_ns")
        .with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("cash_volume") / pl.col("volume"))
            .otherwise(None)
            .alias("vwap"),
            (pl.col("volume") + 1.0).log10().alias("log_volume"),
            (pl.col("cash_volume") + 1.0).log10().alias("log_cash_volume"),
        )
        .drop(["volume", "cash_volume"])
    )

    u = (
        u.join(agg.rename({"sec_ns": "dt_ns"}), on="dt_ns", how="left")
        .with_columns(
            pl.col("close").forward_fill().cast(pl.Float32).alias("last_trade"),
            pl.col("open").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32),
            pl.col("num_trades").fill_null(0).cast(pl.Int64),
            pl.col("log_cash_volume").fill_null(0).cast(pl.Float32),
            pl.col("log_volume").fill_null(0).cast(pl.Float32),
        )
        .with_columns(
            pl.lit(underlying).alias("underlying"),
            pl.lit(d).alias("date").cast(pl.Date),
            pl.lit(is_half_day).alias("is_half_day").cast(pl.Boolean),
        )
    )
    return u
