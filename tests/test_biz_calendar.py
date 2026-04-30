from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantdata.biz_calendar import (
    DateRange,
    add_business_days,
    count_business_days_between,
    get_business_days,
    get_half_days,
    next_business_day,
    prev_business_day,
    to_date,
)


def test_get_business_days_includes_christmas_eve_and_new_year_boundary() -> None:
    business_days = get_business_days(date(2024, 12, 20), date(2025, 1, 3))

    assert business_days == [
        date(2024, 12, 20),
        date(2024, 12, 23),
        date(2024, 12, 24),
        date(2024, 12, 26),
        date(2024, 12, 27),
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]


def test_get_business_days_can_exclude_half_days() -> None:
    business_days = get_business_days(date(2024, 12, 20), date(2025, 1, 3), exclude_half_days=True)

    assert business_days == [
        date(2024, 12, 20),
        date(2024, 12, 23),
        date(2024, 12, 26),
        date(2024, 12, 27),
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2024, 11, 25), date(2024, 11, 30), [date(2024, 11, 29)]),
        (date(2024, 7, 1), date(2024, 7, 5), [date(2024, 7, 3)]),
        (date(2024, 12, 20), date(2025, 1, 3), [date(2024, 12, 24)]),
    ],
)
def test_get_half_days_reports_known_early_closes(start: date, end: date, expected: list[date]) -> None:
    assert get_half_days(start, end) == expected


def test_date_range_respects_half_day_toggle() -> None:
    include_half_days = DateRange(date(2024, 12, 24), date(2024, 12, 27), exclude_half_days=False)
    exclude_half_days = DateRange(date(2024, 12, 24), date(2024, 12, 27), exclude_half_days=True)

    assert include_half_days.get_business_days() == [
        date(2024, 12, 24),
        date(2024, 12, 26),
        date(2024, 12, 27),
    ]
    assert include_half_days.get_half_days() == [date(2024, 12, 24)]

    assert exclude_half_days.get_business_days() == [
        date(2024, 12, 26),
        date(2024, 12, 27),
    ]
    assert exclude_half_days.get_half_days() == []


def test_business_day_navigation_and_counting_across_holiday_edges() -> None:
    assert add_business_days(date(2024, 12, 24), 1) == date(2024, 12, 26)
    assert prev_business_day(date(2024, 12, 25)) == date(2024, 12, 23)
    assert next_business_day(date(2024, 12, 24)) == date(2024, 12, 26)

    assert count_business_days_between(date(2024, 12, 23), date(2024, 12, 24)) == 1
    assert count_business_days_between(date(2024, 12, 23), date(2024, 12, 24), exclude_half_days=True) == 0
    assert count_business_days_between(date(2024, 12, 24), date(2024, 12, 27)) == 2


def test_date_range_requires_at_least_one_business_day() -> None:
    with pytest.raises(ValueError):
        DateRange(date(2024, 12, 25), date(2024, 12, 25))

    assert get_business_days(date(2024, 12, 25), date(2024, 12, 25)) == []


def test_to_date_handles_common_inputs() -> None:
    assert to_date("2024-12-24") == date(2024, 12, 24)
    assert to_date(date(2024, 12, 24)) == date(2024, 12, 24)
    assert to_date(np.datetime64("2024-12-24")) == date(2024, 12, 24)
    assert to_date(None) is None

    with pytest.raises(ValueError, match="Unsupported type for date conversion"):
        to_date(123)
