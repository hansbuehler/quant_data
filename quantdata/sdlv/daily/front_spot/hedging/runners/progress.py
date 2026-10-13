"""Lightweight progress heartbeats for long benchmark evaluations."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable


class EvaluationProgressHeartbeat:
    """Report completed evaluation units while one unit is still running."""

    def __init__(
        self,
        *,
        stage: str,
        total_work_units: int,
        interval_seconds: float,
        emit: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not stage.strip():
            raise ValueError("progress heartbeat stage must be non-empty")
        if total_work_units < 0:
            raise ValueError("progress heartbeat total must be non-negative")
        if not math.isfinite(interval_seconds) or not 1.0 <= interval_seconds <= 300.0:
            raise ValueError("progress heartbeat interval must lie in [1,300]")
        self._stage = stage
        self._total = total_work_units
        self._interval = interval_seconds
        self._emit_line = _print_progress_line if emit is None else emit
        self._clock = clock
        self._completed = 0
        self._last_unit = "none"
        self._started = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> EvaluationProgressHeartbeat:
        self._started = self._clock()
        self.emit()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self._stage}-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.emit()

    def update(self, completed_work_units: int, *, last_unit: str) -> None:
        """Record completed work without synchronizing the numerical worker."""
        if not 0 <= completed_work_units <= self._total:
            raise ValueError("completed progress units must lie within the total")
        if not last_unit.strip():
            raise ValueError("last completed progress unit must be non-empty")
        with self._lock:
            if completed_work_units < self._completed:
                raise ValueError("completed progress units cannot decrease")
            self._completed = completed_work_units
            self._last_unit = last_unit

    def emit(self) -> None:
        """Emit one flushed progress snapshot."""
        with self._lock:
            completed = self._completed
            last_unit = self._last_unit
        elapsed = max(self._clock() - self._started, 0.0)
        if completed >= self._total:
            eta = "0.0"
        elif completed == 0 or elapsed <= 0.0:
            eta = "unknown"
        else:
            eta = f"{elapsed / completed * (self._total - completed):.1f}"
        self._emit_line(
            f"evaluation heartbeat stage={self._stage} elapsed_s={elapsed:.1f} "
            f"completed_work_units={completed} total_work_units={self._total} "
            f"eta_s={eta} last_unit={last_unit} "
            f"next_check_s={self._interval:g}"
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.emit()


def _print_progress_line(line: str) -> None:
    print(line, flush=True)
