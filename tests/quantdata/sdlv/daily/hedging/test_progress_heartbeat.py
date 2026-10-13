from __future__ import annotations

import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners.progress import (
    EvaluationProgressHeartbeat,
)


def test_progress_heartbeat_reports_initial_and_final_state() -> None:
    now = [100.0]
    lines: list[str] = []
    heartbeat = EvaluationProgressHeartbeat(
        stage="weekly-volgan",
        total_work_units=3,
        interval_seconds=240.0,
        emit=lines.append,
        clock=lambda: now[0],
    )

    with heartbeat:
        now[0] = 104.0
        heartbeat.update(2, last_unit="2023-01-18")

    assert lines[0] == (
        "evaluation heartbeat stage=weekly-volgan elapsed_s=0.0 "
        "completed_work_units=0 total_work_units=3 eta_s=unknown "
        "last_unit=none next_check_s=240"
    )
    assert lines[-1] == (
        "evaluation heartbeat stage=weekly-volgan elapsed_s=4.0 "
        "completed_work_units=2 total_work_units=3 eta_s=2.0 "
        "last_unit=2023-01-18 next_check_s=240"
    )


@pytest.mark.parametrize(
    ("completed", "last_unit"),
    ((-1, "date"), (2, "date"), (0, "")),
)
def test_progress_heartbeat_rejects_invalid_updates(
    completed: int,
    last_unit: str,
) -> None:
    heartbeat = EvaluationProgressHeartbeat(
        stage="daily-gromm",
        total_work_units=1,
        interval_seconds=240.0,
        emit=lambda _: None,
    )

    with pytest.raises(ValueError):
        heartbeat.update(completed, last_unit=last_unit)


def test_progress_heartbeat_rejects_decreasing_completion() -> None:
    heartbeat = EvaluationProgressHeartbeat(
        stage="daily-gromm",
        total_work_units=2,
        interval_seconds=240.0,
        emit=lambda _: None,
    )
    heartbeat.update(1, last_unit="2023-01-17")

    with pytest.raises(ValueError, match="cannot decrease"):
        heartbeat.update(0, last_unit="2023-01-16")
