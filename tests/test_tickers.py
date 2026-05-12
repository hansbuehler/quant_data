"""
Tests for ticker-level data loading utilities added in quantdata.massive.tickers
and the session-calendar helpers in quantdata.biz_calendar.

Mirrors the checks done manually in:
  poly/tickers/test_adj.ipynb
  poly/tickers/test_calendar.ipynb
  poly/tickers/test_filters.ipynb
  poly/expiries/test_expiries.ipynb
  poly/poly/test_fetch.ipynb
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

_test_mode = os.environ.get("QUANTDATA_TEST_MODE", "0") == "1"
_stored_massive = Path(__file__).parent / "stored_data" / "raw" / "massive"
_stored_tickers = Path(__file__).parent / "stored_data" / "processing" / "tickers"
_has_massive_data = (_stored_massive / "reference").exists() and (_stored_tickers).exists()

requires_massive = pytest.mark.skipif(
    "MASSIVE_TOKEN" not in os.environ or (_test_mode and not _has_massive_data),
    reason="MASSIVE_TOKEN not set"
    if "MASSIVE_TOKEN" not in os.environ
    else "no stored test data in tests/stored_data/",
)


# ---------------------------------------------------------------------------
# Session calendar helpers (no API key needed)
# ---------------------------------------------------------------------------


def test_session_open_close_normal_day() -> None:
    from quantdata.biz_calendar import session_open_close

    open_dt, close_dt, is_half_day = session_open_close(date(2024, 6, 17))

    assert open_dt.hour == 9 and open_dt.minute == 30
    assert close_dt.hour == 16 and close_dt.minute == 0
    assert not is_half_day


def test_session_open_close_half_day() -> None:
    from quantdata.biz_calendar import session_open_close

    # Black Friday 2024 (confirmed half day in test_biz_calendar.py)
    open_dt, close_dt, is_half_day = session_open_close(date(2024, 11, 29))

    assert open_dt.hour == 9 and open_dt.minute == 30
    assert close_dt.hour == 13 and close_dt.minute == 0
    assert is_half_day


def test_full_second_grid_ns_normal_day() -> None:
    from quantdata.biz_calendar import full_second_grid_ns, session_open_close

    open_dt, close_dt, _ = session_open_close(date(2024, 6, 17))
    grid = full_second_grid_ns(open_dt, close_dt)

    assert isinstance(grid, pl.Series)

    # last bin key is close_dt - 1s
    last = grid[-1].replace(tzinfo=None)
    expected = (close_dt - timedelta(seconds=1)).replace(tzinfo=None)
    assert last == expected


# ---------------------------------------------------------------------------
# classify_expiry (no API key needed)
# ---------------------------------------------------------------------------


def test_classify_expiry_normal_day() -> None:
    from quantdata.massive.expiries import classify_expiry

    # June 24 is after the third Friday of June 2024 (Jun 21), so it can never be
    # selected as the monthly expiry date and is always classified NONE.
    df = classify_expiry(pl.DataFrame({"date": [date(2024, 6, 24)]}))
    assert df["has_expiry"][0] == 0  # EXPIRY.NONE


def test_classify_expiry_monthly() -> None:
    from quantdata.massive.expiries import classify_expiry

    # Third Friday of January 2024 = Jan 19
    df = classify_expiry(pl.DataFrame({"date": [date(2024, 1, 19)]}))
    assert df["has_expiry"][0] == 3  # EXPIRY.MONTHLY


def test_classify_expiry_quad_witching() -> None:
    from quantdata.massive.expiries import classify_expiry

    # Third Friday of March 2024 = Mar 15 (quad witching)
    df = classify_expiry(pl.DataFrame({"date": [date(2024, 3, 15)]}))
    assert df["has_expiry"][0] == 2  # EXPIRY.QUAD_WITCHING


def test_classify_expiry_shifted_when_friday_is_holiday() -> None:
    from quantdata.massive.expiries import classify_expiry

    # Good Friday 2022 (Apr 15) is a market holiday — the third Friday of April falls on it.
    # When only trading days are passed (Apr 15 absent), expiry shifts to Thu Apr 14.
    df = classify_expiry(
        pl.DataFrame({"date": [date(2022, 4, 13), date(2022, 4, 14), date(2022, 4, 19)]})
    )
    codes = dict(zip(df["date"].to_list(), df["has_expiry"].to_list()))
    assert codes[date(2022, 4, 14)] == 3  # shifted monthly expiry (EXPIRY.MONTHLY)
    assert codes[date(2022, 4, 13)] == 0  # normal day (EXPIRY.NONE)


# ---------------------------------------------------------------------------
# Reference data and adjustment factors
# ---------------------------------------------------------------------------


@requires_massive
def test_get_listing_date() -> None:
    from quantdata.massive.tickers.basics import get_listing_date

    d = get_listing_date("SPY")

    assert isinstance(d, date)
    assert d < date(2000, 1, 1)  # SPY listed Jan 1993


@requires_massive
def test_adjustment_factors_aapl() -> None:
    from quantdata.massive import MassiveClient

    df = MassiveClient().adjustment_factors("AAPL", date(2020, 1, 1), date(2021, 1, 1))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert set(df.columns) == {"date", "adjfactor"}


# ---------------------------------------------------------------------------
# Filtered daily history (mirrors test_filters.ipynb)
# ---------------------------------------------------------------------------


@requires_massive
def test_ticker_filtered_history_spy() -> None:
    from quantdata.massive.tickers.filtered_history import ticker_filtered_history

    df = ticker_filtered_history("SPY", date(2023, 1, 1), date(2024, 1, 1))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert set(df.columns) == {
        "date",
        "ticker",
        "cash_vol",
        "n_trades",
        "return_open_prev_close",
        "return_close_open",
    }


@requires_massive
def test_ticker_filtered_history_meta_handles_ticker_change() -> None:
    """META was called FB before Oct 2021 — exercises the ticker-change code path."""
    from quantdata.massive.tickers.filtered_history import ticker_filtered_history

    df = ticker_filtered_history("META", date(2021, 1, 1), date(2023, 1, 1))

    assert df.height > 0
    tickers_seen = set(df["ticker"].to_list())
    # At least one of the historic symbols should appear
    assert tickers_seen & {"FB", "META"}


@requires_massive
def test_ticker_filtered_history_vxx() -> None:
    """VXX exercises the MANUAL_LISTING_DATES override."""
    from quantdata.massive.tickers.filtered_history import ticker_filtered_history

    df = ticker_filtered_history("VXX", date(2023, 1, 1), date(2024, 1, 1))

    assert df.height > 0


@requires_massive
def test_ticker_filtered_history_arkk() -> None:
    from quantdata.massive.tickers.filtered_history import ticker_filtered_history

    df = ticker_filtered_history("ARKK", date(2023, 1, 1), date(2024, 1, 1))

    assert df.height > 0


# ---------------------------------------------------------------------------
# Ticker calendar (mirrors test_calendar.ipynb)
# ---------------------------------------------------------------------------


@requires_massive
def test_build_ticker_calendar_spy() -> None:
    from quantdata.biz_calendar import DateRange
    from quantdata.tickers.ticker_calendar import build_ticker_calendar

    dr = DateRange(date(2024, 1, 2), date(2024, 3, 31))
    df = build_ticker_calendar("SPY", dr)

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "date" in df.columns
    assert "expiry_code" in df.columns


# ---------------------------------------------------------------------------
# Option expiry flags (mirrors test_expiries.ipynb — short window to stay fast)
# ---------------------------------------------------------------------------


@requires_massive
def test_ticker_expiries_short_window() -> None:
    from quantdata.biz_calendar import get_business_days
    from quantdata.massive.expiries import ticker_expiries

    days = get_business_days(date(2024, 1, 2), date(2024, 1, 19))  # ~13 trading days
    df = ticker_expiries("SPY", days, num_workers=4, threading=True)

    assert isinstance(df, pl.DataFrame)
    assert "date" in df.columns
    assert "has_expiry" in df.columns


# ---------------------------------------------------------------------------
# Tick-level trade data (mirrors test_fetch.ipynb)
# ---------------------------------------------------------------------------


@requires_massive
def test_build_unadjusted_trades() -> None:
    from quantdata.massive.tickers.data import build_unadjusted_trades

    df = build_unadjusted_trades("SPY", date(2024, 1, 2))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert set(df.columns) >= {"ts_ns", "price", "size"}


@requires_massive
def test_build_1s_trade_bars() -> None:
    from quantdata.massive.tickers.data import build_1s_trade_bars

    df = build_1s_trade_bars("SPY", date(2024, 1, 2))

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert "dt_ns" in df.columns
    assert "num_trades" in df.columns
    assert "log_cash_volume" in df.columns
