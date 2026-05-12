from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from quantdata.ncde.voluvol.generate import _add_ms_since_open, _build_target


def test_add_ms_since_open_sorts_and_keeps_integer_milliseconds() -> None:
    tz_ny = ZoneInfo("America/New_York")
    open_dt = datetime(2025, 1, 2, 9, 30, tzinfo=tz_ny)
    quotes = pl.DataFrame(
        {
            "ts_ns": [
                datetime(2025, 1, 2, 9, 45, tzinfo=tz_ny),
                datetime(2025, 1, 2, 9, 30, 30, tzinfo=tz_ny),
            ],
            "bid_price": [100.5, 99.5],
            "bid_size": [12.0, 10.0],
            "ask_price": [101.5, 100.5],
            "ask_size": [13.0, 11.0],
        },
        schema={
            "ts_ns": pl.Datetime("ns", "America/New_York"),
            "bid_price": pl.Float64,
            "bid_size": pl.Float64,
            "ask_price": pl.Float64,
            "ask_size": pl.Float64,
        },
    )

    result = _add_ms_since_open(quotes, open_dt)

    assert result["time_of_day_ms"].dtype == pl.Int64
    assert result["time_of_day_ms"].to_list() == [30_000, 900_000]


def test_build_target_excludes_trades_without_full_forward_horizon() -> None:
    trades = pl.DataFrame(
        {
            "time_of_day_ms": [22_400_000, 22_600_000, 23_000_000],
            "price": [100.0, 101.0, 102.0],
            "size": [10.0, 20.0, 30.0],
        },
        schema={
            "time_of_day_ms": pl.Int64,
            "price": pl.Float64,
            "size": pl.Float64,
        },
    )

    targets = _build_target(trades, forecast_horizon_min=15.0, close_delta_ms=23_400_000)

    assert targets["time_of_day_ms"].to_list() == [22_400_000]
    assert np.allclose(targets["log_volume"].to_numpy(), [np.log1p(50.0)])
    assert np.allclose(targets["log_cash_volume"].to_numpy(), [np.log1p(50.8)])
    assert np.all(targets["volatility"].to_numpy() > 0.0)