"""Rolling DYSANOS windows must stop at the actual forecast origin."""

from datetime import date, timedelta

import pytest

from quantdata.hedging.model_audit.rolling_dysanos import window_dates


def test_window_never_contains_a_later_observation() -> None:
    start = date(2022, 1, 3)
    calendar = [start + timedelta(days=offset) for offset in range(1000)
                if (start + timedelta(days=offset)).weekday() < 5]
    origin = calendar[400]
    result = window_dates(calendar, origin)
    assert len(result) == 250
    assert result[-1] == origin
    assert result[0] == calendar[151]
    assert max(result) <= origin


def test_window_is_not_extended_or_truncated_to_hide_missing_history() -> None:
    with pytest.raises(ValueError, match="complete"):
        window_dates([date(2024, 4, 1), date(2024, 4, 2)], date(2024, 4, 2))
