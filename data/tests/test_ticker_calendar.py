from __future__ import annotations

from datetime import date

import polars as pl

from core.src.openqai.core.biz_calendar import DateRange
from quantdata.tickers.ticker_calendar import EXPIRY, SCHEMA_TICKER_CALENDAR, get_standard_expiries


def _expiry_codes_by_date(date_range: DateRange) -> dict[date, int]:
    expiries = get_standard_expiries(date_range)
    return dict(zip(expiries["date"].to_list(), expiries["expiry_code"].to_list(), strict=True))


def test_get_standard_expiries_marks_quarter_end_month_end_as_quarterly() -> None:
    codes_by_date = _expiry_codes_by_date(DateRange(date(2024, 3, 1), date(2024, 3, 28)))

    assert codes_by_date[date(2024, 3, 15)] == int(EXPIRY.QUAD_WITCHING)
    assert codes_by_date[date(2024, 3, 28)] == int(EXPIRY.QUARTERLY)


def test_get_standard_expiries_keeps_non_quarter_month_end_as_eom() -> None:
    codes_by_date = _expiry_codes_by_date(DateRange(date(2024, 1, 2), date(2024, 1, 31)))

    assert codes_by_date[date(2024, 1, 19)] == int(EXPIRY.MONTHLY)
    assert codes_by_date[date(2024, 1, 31)] == int(EXPIRY.EOM)


def test_build_ticker_calendar_schema_constant_matches_output_contract() -> None:
    assert SCHEMA_TICKER_CALENDAR == {
        "date": pl.Date,
        "expiry_code": pl.Int8,
        "is_ex_div": pl.Boolean,
        "is_split": pl.Boolean,
        "has_ticker_change": pl.Boolean,
        "is_earnings": pl.Boolean,
        "is_half_day": pl.Boolean,
    }