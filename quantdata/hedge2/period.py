"""The agreed hedge2 data period and explicit reporting of missing coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from cdxcore.err import verify_inp

from quantdata.biz_calendar import get_business_days


@dataclass(frozen=True, kw_only=True)
class ExperimentPeriod:
    history_start: date = date(2022, 1, 1)
    evaluation_start: date = date(2023, 1, 1)
    requested_end: date = date(2025, 12, 31)

    def __post_init__(self) -> None:
        verify_inp(self.history_start <= self.evaluation_start <= self.requested_end,
                   lambda: "History, evaluation and end dates must be ordered")

    def coverage(self, dates: tuple[date, ...], *, as_of_date: date) -> dict[str, object]:
        selected = sorted({day for day in dates if self.history_start <= day <= self.requested_end})
        required_through = min(self.requested_end, as_of_date)
        expected = (get_business_days(self.history_start, required_through)
                    if required_through >= self.history_start else [])
        missing = sorted(set(expected).difference(selected))
        years = range(self.history_start.year, self.requested_end.year + 1)
        counts = {str(year): sum(day.year == year for day in selected) for year in years}
        return {
            "history_start": self.history_start.isoformat(),
            "evaluation_start": self.evaluation_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "available_start": selected[0].isoformat() if selected else None,
            "available_end": selected[-1].isoformat() if selected else None,
            "available_dates": len(selected), "dates_by_year": counts,
            "missing_years": [year for year, count in counts.items()
                              if count == 0 and int(year) <= required_through.year],
            "required_through": required_through.isoformat(),
            "expected_sessions": len(expected),
            "missing_sessions": len(missing),
            "missing_session_dates": [day.isoformat() for day in missing],
            "coverage_complete_through_requested_date": bool(expected and not missing),
        }
