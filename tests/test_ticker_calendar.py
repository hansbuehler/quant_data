from __future__ import annotations

from datetime import date

from quantdata.biz_calendar import DateRange
from quantdata.tickers.ticker_calendar import EXPIRY, get_standard_expiries


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