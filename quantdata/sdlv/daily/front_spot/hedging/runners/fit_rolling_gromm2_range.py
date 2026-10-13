"""Build or fit a resumable date range of rolling GROMM2 market artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterator

from cdxcore.err import verify_inp
from filelock import FileLock

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
    fit_rolling_gromm2_market,
    load_prepared_rolling_gromm2,
    load_rolling_gromm2,
    prepare_rolling_gromm2,
)

ROLLING_GROMM2_RANGE_SCHEMA = "hedging-rolling-gromm2-range-v1"


class RollingGromm2RangeStage(StrEnum):
    """Material stage executed across the requested origin range."""

    PREPARE = "prepare"
    FIT = "fit"


@dataclass(frozen=True, kw_only=True)
class RollingGromm2RangeRecord:
    """Authenticated outcome for one dated rolling artifact."""

    as_of_date: date
    reused: bool
    manifest_sha256: str
    physical_accepted: bool | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation."""
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "reused": self.reused,
            "manifest_sha256": self.manifest_sha256,
            "physical_accepted": self.physical_accepted,
        }


@dataclass(frozen=True, kw_only=True)
class RollingGromm2RangeSummary:
    """Restartable range-stage evidence bound to the public dataset."""

    stage: RollingGromm2RangeStage
    dataset_manifest_sha256: str
    artifact_root: Path
    evaluation_start: date
    evaluation_end: date
    eligible_origin_count: int
    selected_origin_count: int
    records: tuple[RollingGromm2RangeRecord, ...]
    config: RollingGromm2Config

    @property
    def status(self) -> str:
        """Return complete only when every eligible origin was processed."""
        if (
            self.selected_origin_count == self.eligible_origin_count
            and len(self.records) == self.eligible_origin_count
        ):
            return "complete"
        return "partial"

    @property
    def physical_accepted_count(self) -> int:
        """Count fitted origins accepted for physical-measure generation."""
        return sum(record.physical_accepted is True for record in self.records)

    def to_dict(self) -> dict[str, object]:
        """Return the stable, auditable JSON representation."""
        return {
            "schema": ROLLING_GROMM2_RANGE_SCHEMA,
            "status": self.status,
            "stage": self.stage.value,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "artifact_root": str(self.artifact_root.resolve()),
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
            "eligible_origin_count": self.eligible_origin_count,
            "selected_origin_count": self.selected_origin_count,
            "completed_count": len(self.records),
            "reused_count": sum(record.reused for record in self.records),
            "physical_accepted_count": self.physical_accepted_count,
            "config": json.loads(
                json.dumps(asdict(self.config), sort_keys=True, allow_nan=False)
            ),
            "records": [record.to_dict() for record in self.records],
        }


def eligible_rolling_gromm2_origins(
    panel: PublicHedgingPanel,
    *,
    evaluation_start: date,
    evaluation_end: date,
    window_transitions: int,
) -> tuple[date, ...]:
    """Return daily-evaluation origins with a complete trailing fit window."""
    verify_inp(
        evaluation_start <= evaluation_end,
        lambda: "rolling GROMM2 evaluation start must not exceed end",
    )
    origins: list[date] = []
    for origin_date in panel.dates:
        if not evaluation_start <= origin_date <= evaluation_end:
            continue
        try:
            panel.prior_one_day_origins(origin_date, count=window_transitions)
            panel.one_day_transition(origin_date)
        except ValueError:
            continue
        origins.append(origin_date)
    return tuple(origins)


def run_rolling_gromm2_range(
    *,
    dataset: Path,
    artifact_root: Path,
    evaluation_start: date,
    evaluation_end: date,
    stage: RollingGromm2RangeStage,
    state_file: Path,
    config: RollingGromm2Config = RollingGromm2Config(),
    stop_after: int | None = None,
    progress_seconds: float = 240.0,
    clock: Callable[[], float] = time.monotonic,
) -> RollingGromm2RangeSummary:
    """Run one authenticated range stage, reusing only valid dated artifacts."""
    verify_inp(
        stop_after is None or stop_after >= 1,
        lambda: "stop_after must be positive when supplied",
    )
    verify_inp(
        math.isfinite(progress_seconds) and 0.0 < progress_seconds <= 300.0,
        lambda: "progress_seconds must lie in (0,300]",
    )
    dataset = Path(dataset)
    artifact_root = Path(artifact_root)
    state_file = Path(state_file)
    panel = PublicHedgingPanel(dataset)
    dataset_manifest_sha256 = file_sha256(dataset / "manifest.json")
    eligible_origins = eligible_rolling_gromm2_origins(
        panel,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        window_transitions=config.window_transitions,
    )
    verify_inp(
        bool(eligible_origins),
        lambda: "rolling GROMM2 range contains no eligible origins",
    )
    selected_origins = (
        eligible_origins
        if stop_after is None
        else eligible_origins[:stop_after]
    )
    records: list[RollingGromm2RangeRecord] = []
    started = clock()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(state_file) + ".lock", timeout=0)
    with lock:
        for index, as_of_date in enumerate(selected_origins, start=1):
            try:
                with _progress_heartbeat(
                    stage=stage,
                    as_of_date=as_of_date,
                    completed=len(records),
                    total=len(selected_origins),
                    started=started,
                    progress_seconds=progress_seconds,
                    clock=clock,
                ):
                    record = _run_origin(
                        dataset=dataset,
                        artifact_root=artifact_root,
                        as_of_date=as_of_date,
                        stage=stage,
                        config=config,
                        dataset_manifest_sha256=dataset_manifest_sha256,
                    )
            except Exception as error:
                _write_json_atomic(
                    state_file,
                    {
                        "schema": ROLLING_GROMM2_RANGE_SCHEMA,
                        "status": "failed",
                        "stage": stage.value,
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "artifact_root": str(artifact_root.resolve()),
                        "evaluation_start": evaluation_start.isoformat(),
                        "evaluation_end": evaluation_end.isoformat(),
                        "eligible_origin_count": len(eligible_origins),
                        "selected_origin_count": len(selected_origins),
                        "completed_count": len(records),
                        "current_origin": as_of_date.isoformat(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "config": json.loads(
                            json.dumps(
                                asdict(config), sort_keys=True, allow_nan=False
                            )
                        ),
                        "records": [value.to_dict() for value in records],
                    },
                )
                raise
            records.append(record)
            summary = RollingGromm2RangeSummary(
                stage=stage,
                dataset_manifest_sha256=dataset_manifest_sha256,
                artifact_root=artifact_root,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                eligible_origin_count=len(eligible_origins),
                selected_origin_count=len(selected_origins),
                records=tuple(records),
                config=config,
            )
            _write_json_atomic(state_file, summary.to_dict())
            elapsed = clock() - started
            rate = index / elapsed if elapsed > 0.0 else 0.0
            eta = (len(selected_origins) - index) / rate if rate > 0.0 else None
            print(
                json.dumps(
                    {
                        "stage": "rolling-gromm2-range-progress",
                        "range_stage": stage.value,
                        "as_of_date": as_of_date.isoformat(),
                        "completed": index,
                        "total": len(selected_origins),
                        "reused": record.reused,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = RollingGromm2RangeSummary(
        stage=stage,
        dataset_manifest_sha256=dataset_manifest_sha256,
        artifact_root=artifact_root,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        eligible_origin_count=len(eligible_origins),
        selected_origin_count=len(selected_origins),
        records=tuple(records),
        config=config,
    )
    _write_json_atomic(state_file, summary.to_dict())
    return summary


def load_completed_rolling_gromm2_range(
    path: Path,
    *,
    dataset_manifest_sha256: str,
    artifact_root: Path,
) -> RollingGromm2RangeSummary:
    """Authenticate a completed fit-range summary and every dated manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rolling GROMM2 range summary must be a JSON object")
    if (
        payload.get("schema") != ROLLING_GROMM2_RANGE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("stage") != RollingGromm2RangeStage.FIT.value
        or payload.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or payload.get("artifact_root") != str(Path(artifact_root).resolve())
    ):
        raise ValueError("rolling GROMM2 range summary is stale or incomplete")
    records_payload = payload.get("records")
    eligible_count = payload.get("eligible_origin_count")
    selected_count = payload.get("selected_origin_count")
    completed_count = payload.get("completed_count")
    if (
        not isinstance(records_payload, list)
        or not isinstance(eligible_count, int)
        or not isinstance(selected_count, int)
        or not isinstance(completed_count, int)
        or eligible_count <= 0
        or selected_count != eligible_count
        or completed_count != eligible_count
        or len(records_payload) != eligible_count
    ):
        raise ValueError("rolling GROMM2 range summary has inconsistent counts")
    records: list[RollingGromm2RangeRecord] = []
    for value in records_payload:
        if not isinstance(value, dict):
            raise ValueError("rolling GROMM2 range record must be a JSON object")
        as_of_date = date.fromisoformat(str(value.get("as_of_date")))
        manifest = Path(artifact_root) / as_of_date.isoformat() / "manifest.json"
        manifest_sha256 = value.get("manifest_sha256")
        physical_accepted = value.get("physical_accepted")
        reused = value.get("reused")
        if (
            not isinstance(manifest_sha256, str)
            or manifest_sha256 != file_sha256(manifest)
            or not isinstance(physical_accepted, bool)
            or not isinstance(reused, bool)
        ):
            raise ValueError(
                f"rolling GROMM2 range record is invalid for {as_of_date}"
            )
        records.append(
            RollingGromm2RangeRecord(
                as_of_date=as_of_date,
                reused=reused,
                manifest_sha256=manifest_sha256,
                physical_accepted=physical_accepted,
            )
        )
    if tuple(record.as_of_date for record in records) != tuple(
        sorted({record.as_of_date for record in records})
    ):
        raise ValueError("rolling GROMM2 range dates are not unique and chronological")
    config_payload = payload.get("config")
    config = RollingGromm2Config()
    expected_config = json.loads(
        json.dumps(asdict(config), sort_keys=True, allow_nan=False)
    )
    if config_payload != expected_config:
        raise ValueError("rolling GROMM2 range configuration is incompatible")
    summary = RollingGromm2RangeSummary(
        stage=RollingGromm2RangeStage.FIT,
        dataset_manifest_sha256=dataset_manifest_sha256,
        artifact_root=Path(artifact_root),
        evaluation_start=date.fromisoformat(str(payload.get("evaluation_start"))),
        evaluation_end=date.fromisoformat(str(payload.get("evaluation_end"))),
        eligible_origin_count=eligible_count,
        selected_origin_count=selected_count,
        records=tuple(records),
        config=config,
    )
    if (
        summary.evaluation_start > summary.evaluation_end
        or any(
            not summary.evaluation_start
            <= record.as_of_date
            <= summary.evaluation_end
            for record in summary.records
        )
    ):
        raise ValueError("rolling GROMM2 range dates violate the evaluation range")
    if summary.to_dict()["physical_accepted_count"] != payload.get(
        "physical_accepted_count"
    ):
        raise ValueError("rolling GROMM2 physical-acceptance count is inconsistent")
    return summary


def _run_origin(
    *,
    dataset: Path,
    artifact_root: Path,
    as_of_date: date,
    stage: RollingGromm2RangeStage,
    config: RollingGromm2Config,
    dataset_manifest_sha256: str,
) -> RollingGromm2RangeRecord:
    if stage is RollingGromm2RangeStage.PREPARE:
        reused = True
        try:
            paths = load_prepared_rolling_gromm2(
                artifact_root,
                as_of_date=as_of_date,
                config=config,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
        except (FileNotFoundError, OSError, ValueError):
            reused = False
            paths = prepare_rolling_gromm2(
                dataset=dataset,
                artifact_root=artifact_root,
                as_of_date=as_of_date,
                config=config,
            )
            load_prepared_rolling_gromm2(
                artifact_root,
                as_of_date=as_of_date,
                config=config,
                dataset_manifest_sha256=dataset_manifest_sha256,
            )
        return RollingGromm2RangeRecord(
            as_of_date=as_of_date,
            reused=reused,
            manifest_sha256=file_sha256(paths.manifest),
            physical_accepted=None,
        )
    reused = True
    try:
        artifacts = load_rolling_gromm2(
            artifact_root,
            as_of_date=as_of_date,
            config=config,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
    except (FileNotFoundError, OSError, ValueError):
        reused = False
        artifacts = fit_rolling_gromm2_market(
            dataset=dataset,
            artifact_root=artifact_root,
            as_of_date=as_of_date,
            config=config,
        )
    return RollingGromm2RangeRecord(
        as_of_date=as_of_date,
        reused=reused,
        manifest_sha256=file_sha256(artifacts.paths.manifest),
        physical_accepted=artifacts.physical_accepted,
    )


@contextmanager
def _progress_heartbeat(
    *,
    stage: RollingGromm2RangeStage,
    as_of_date: date,
    completed: int,
    total: int,
    started: float,
    progress_seconds: float,
    clock: Callable[[], float],
) -> Iterator[None]:
    stopped = threading.Event()

    def report() -> None:
        while not stopped.wait(progress_seconds):
            elapsed = clock() - started
            rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
            eta = (total - completed) / rate if rate > 0.0 else None
            print(
                json.dumps(
                    {
                        "stage": "rolling-gromm2-range-heartbeat",
                        "range_stage": stage.value,
                        "as_of_date": as_of_date.isoformat(),
                        "completed": completed,
                        "total": total,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Parse and execute one resumable rolling-artifact range stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument(
        "--stage",
        type=RollingGromm2RangeStage,
        choices=tuple(RollingGromm2RangeStage),
        required=True,
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--progress-seconds", type=float, default=240.0)
    arguments = parser.parse_args()
    state_file = arguments.state_file or (
        arguments.artifact_root
        / (
            f"range-{arguments.stage.value}-{arguments.start.isoformat()}-"
            f"{arguments.end.isoformat()}.json"
        )
    )
    summary = run_rolling_gromm2_range(
        dataset=arguments.dataset,
        artifact_root=arguments.artifact_root,
        evaluation_start=arguments.start,
        evaluation_end=arguments.end,
        stage=arguments.stage,
        state_file=state_file,
        stop_after=arguments.stop_after,
        progress_seconds=arguments.progress_seconds,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
