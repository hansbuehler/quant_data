"""The requested 2022--2025 period must not silently become a shorter experiment."""

from datetime import date

from quantdata.hedge2.period import ExperimentPeriod


def test_missing_late_2025_is_explicit_and_old_history_is_excluded() -> None:
    period = ExperimentPeriod()
    report = period.coverage((date(2021, 12, 31), date(2022, 1, 3), date(2023, 1, 3),
                              date(2024, 1, 2), date(2025, 8, 29), date(2026, 1, 2)),
                             as_of_date=date(2026, 11, 6))
    assert report["available_dates"] == 4
    assert report["available_start"] == "2022-01-03"
    assert report["requested_end"] == "2025-12-31"
    assert report["missing_years"] == []
    assert "2026" not in report["dates_by_year"]
    assert "2025-12-31" in report["missing_session_dates"]
    assert not report["coverage_complete_through_requested_date"]


def test_interior_session_gaps_are_not_reported_as_complete() -> None:
    period = ExperimentPeriod(history_start=date(2025, 12, 29),
                              evaluation_start=date(2025, 12, 29))
    report = period.coverage((date(2025, 12, 29), date(2025, 12, 31)),
                             as_of_date=date(2026, 11, 6))
    assert report["expected_sessions"] == 3
    assert report["missing_session_dates"] == ["2025-12-30"]
    assert not report["coverage_complete_through_requested_date"]


def test_weekend_end_does_not_create_a_missing_session() -> None:
    period = ExperimentPeriod(history_start=date(2025, 8, 28),
                              evaluation_start=date(2025, 8, 28),
                              requested_end=date(2025, 8, 31))
    report = period.coverage((date(2025, 8, 28), date(2025, 8, 29)),
                             as_of_date=date(2026, 11, 6))
    assert report["missing_sessions"] == 0
    assert report["coverage_complete_through_requested_date"]
