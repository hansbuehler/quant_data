"""Execute selected ready cases from an immutable daily production matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
        PublicHedgingPanel,
    )

DAILY_MATRIX_SCHEMA = "hedging-daily-production-matrix-v1"
WEEKLY_MATRIX_SCHEMA = "hedging-weekly-production-matrix-v1"
DAILY_RESULTS_SCHEMA = "hedging-daily-results-v2"
WEEKLY_EVALUATION_SCHEMA = "hedging-weekly-evaluation-v2"
WEEKLY_RESULTS_SCHEMA = "hedging-weekly-results-v2"
QUEUE_SCHEMA = "hedging-production-queue-v1"
LEGACY_DAILY_QUEUE_SCHEMA = "hedging-daily-production-queue-v1"
DEFAULT_DAILY_RESULT_COUNT = 12_220
DEFAULT_WEEKLY_RESULT_COUNT = 6_864
GPUCTL_PATH = Path("/home/hans/.local/bin/gpuctl")
GENERATED_MODELS = frozenset(
    {"volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"}
)


def execute_production_queue(
    *,
    matrix_path: Path,
    state_path: Path,
    interpreter: Path,
    case_ids: tuple[str, ...],
    case_prefix: str | None,
    resource: str,
    expected_result_count: int | None = None,
    heartbeat_seconds: float = 240.0,
    adopt_unpinned_state: bool = False,
    continue_after_failure: bool = False,
) -> dict[str, object]:
    """Run selected matrix cases under one nonblocking queue-state lock."""
    lock = FileLock(str(state_path) + ".lock")
    try:
        with lock.acquire(timeout=0.0):
            return _execute_production_queue_locked(
                matrix_path=matrix_path,
                state_path=state_path,
                interpreter=interpreter,
                case_ids=case_ids,
                case_prefix=case_prefix,
                resource=resource,
                expected_result_count=expected_result_count,
                heartbeat_seconds=heartbeat_seconds,
                adopt_unpinned_state=adopt_unpinned_state,
                continue_after_failure=continue_after_failure,
            )
    except Timeout as error:
        raise RuntimeError(
            f"production queue state is already owned: {state_path}"
        ) from error


def _execute_production_queue_locked(
    *,
    matrix_path: Path,
    state_path: Path,
    interpreter: Path,
    case_ids: tuple[str, ...],
    case_prefix: str | None,
    resource: str,
    expected_result_count: int | None = None,
    heartbeat_seconds: float = 240.0,
    adopt_unpinned_state: bool = False,
    continue_after_failure: bool = False,
) -> dict[str, object]:
    """Run selected cases sequentially while the caller owns the state lock."""
    matrix_bytes = matrix_path.read_bytes()
    matrix_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
    matrix = _read_json_object_bytes(matrix_bytes, path=matrix_path)
    matrix_schema = matrix.get("schema")
    if matrix_schema not in {DAILY_MATRIX_SCHEMA, WEEKLY_MATRIX_SCHEMA}:
        raise ValueError(f"unexpected production-matrix schema: {matrix_path}")
    if expected_result_count is None:
        expected_result_count = (
            DEFAULT_DAILY_RESULT_COUNT
            if matrix_schema == DAILY_MATRIX_SCHEMA
            else DEFAULT_WEEKLY_RESULT_COUNT
        )
    cases_value = matrix.get("cases")
    if not isinstance(cases_value, list):
        raise ValueError("production matrix cases must be a list")
    selected = _select_cases(
        cases=cases_value,
        case_ids=case_ids,
        case_prefix=case_prefix,
        resource=resource,
    )
    state = _load_or_initialize_state(
        state_path=state_path,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        case_ids=tuple(_require_string(case, "case_id") for case in selected),
        resource=resource,
        adopt_unpinned_state=adopt_unpinned_state,
    )
    completed = {
        _require_string(record, "case_id")
        for record in _records(state)
        if record.get("status") == "complete"
    }
    failed = {
        _require_string(record, "case_id")
        for record in _records(state)
        if record.get("status") == "failed"
    }
    ran_case = False
    for case in selected:
        case_id = _require_string(case, "case_id")
        command = _production_command(
            case=case,
            interpreter=interpreter,
            expected_result_count=expected_result_count,
        )
        output = _command_option_path(command, "--output")
        dataset = (
            _command_option_path(command, "--dataset")
            if matrix_schema == DAILY_MATRIX_SCHEMA
            else None
        )
        results_path = _results_path(output, matrix_schema=str(matrix_schema))
        if case_id in completed:
            result_sha256 = _validate_results(
                results_path,
                output=output,
                matrix_schema=str(matrix_schema),
                expected_model=_require_string(case, "model"),
                expected_result_count=expected_result_count,
                dataset=dataset,
            )
            record = next(
                record
                for record in _records(state)
                if record.get("case_id") == case_id
            )
            pinned_sha256 = record.get("validated_results_sha256")
            if pinned_sha256 is not None and pinned_sha256 != result_sha256:
                raise ValueError(
                    f"completed production result hash changed: {case_id}"
                )
            record["validated_results_sha256"] = result_sha256
            record["revalidated_at"] = _utc_now()
            _write_json_atomic(state_path, state)
            print(
                f"queue case={case_id} reused authenticated completion",
                flush=True,
            )
            continue
        if continue_after_failure and case_id in failed:
            print(f"queue case={case_id} retained queue-state failure", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "run.log"
        _check_resources()
        started_at = _utc_now()
        _upsert_record(
            state,
            {
                "case_id": case_id,
                "status": "running",
                "started_at": started_at,
                "completed_at": None,
                "command": command,
                "log_path": str(log_path),
                "results_path": str(results_path),
                "return_code": None,
            },
        )
        state["status"] = "running"
        state["current_case_id"] = case_id
        _write_json_atomic(state_path, state)
        print(f"queue case={case_id} started_at={started_at}", flush=True)
        ran_case = True
        return_code = _run_logged_command(
            command=command,
            log_path=log_path,
            heartbeat_seconds=heartbeat_seconds,
            case_id=case_id,
            resource=resource,
        )
        if return_code != 0:
            _finish_record(
                state=state,
                state_path=state_path,
                case_id=case_id,
                status="failed",
                return_code=return_code,
                error=f"production command returned {return_code}",
            )
            failed.add(case_id)
            if continue_after_failure:
                print(
                    f"queue case={case_id} failed return_code={return_code}; "
                    "continuing",
                    flush=True,
                )
                continue
            raise RuntimeError(f"production case failed: {case_id} ({return_code})")
        try:
            result_sha256 = _validate_results(
                results_path,
                output=output,
                matrix_schema=str(matrix_schema),
                expected_model=_require_string(case, "model"),
                expected_result_count=expected_result_count,
                dataset=dataset,
            )
        except Exception as error:
            _finish_record(
                state=state,
                state_path=state_path,
                case_id=case_id,
                status="failed",
                return_code=return_code,
                error=f"{type(error).__name__}: {error}",
            )
            failed.add(case_id)
            if continue_after_failure:
                print(
                    f"queue case={case_id} failed validation; continuing",
                    flush=True,
                )
                continue
            raise
        _finish_record(
            state=state,
            state_path=state_path,
            case_id=case_id,
            status="complete",
            return_code=return_code,
            validated_results_sha256=result_sha256,
        )
        completed.add(case_id)
        failed.discard(case_id)
        print(f"queue case={case_id} complete", flush=True)
    terminal_status = "complete-with-failures" if failed else "complete"
    if not ran_case and state.get("status") == terminal_status:
        print(f"queue reused completion cases={len(selected)}", flush=True)
        return state
    state["status"] = terminal_status
    state["current_case_id"] = None
    state["completed_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    print(
        f"queue {terminal_status} cases={len(selected)} failures={len(failed)}",
        flush=True,
    )
    return state


def _select_cases(
    *,
    cases: list[object],
    case_ids: tuple[str, ...],
    case_prefix: str | None,
    resource: str,
) -> list[dict[str, object]]:
    selected_ids = set(case_ids)
    selected: list[dict[str, object]] = []
    known_ids: set[str] = set()
    for value in cases:
        if not isinstance(value, dict):
            raise ValueError("production matrix case must be an object")
        case = value
        case_id = _require_string(case, "case_id")
        known_ids.add(case_id)
        if selected_ids and case_id not in selected_ids:
            continue
        if case_prefix is not None and not case_id.startswith(case_prefix):
            continue
        if case.get("resource") != resource:
            continue
        if case.get("readiness") != "ready":
            raise ValueError(f"selected production case is not ready: {case_id}")
        selected.append(case)
    missing = selected_ids - known_ids
    if missing:
        raise ValueError(f"unknown production case ids: {sorted(missing)}")
    if selected_ids and len(selected) != len(selected_ids):
        unselected = selected_ids - {
            _require_string(case, "case_id") for case in selected
        }
        raise ValueError(
            f"selected cases do not match resource/readiness filters: {sorted(unselected)}"
        )
    if not selected:
        raise ValueError("production queue selection is empty")
    return selected


def _production_command(
    *,
    case: dict[str, object],
    interpreter: Path,
    expected_result_count: int,
) -> list[str]:
    model = _require_string(case, "model")
    if model in GENERATED_MODELS:
        reviewed_commit = _require_string(case, "reviewed_commit")
        gate_path = Path(_require_string(case, "production_gate_path"))
        gate_sha256 = _require_sha256(case, "production_gate_sha256")
        if hashlib.sha256(gate_path.read_bytes()).hexdigest() != gate_sha256:
            raise ValueError(
                f"production case {case.get('case_id')} gate bytes changed"
            )
        from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
            reviewable_repository_commit,
        )

        if reviewable_repository_commit() != reviewed_commit:
            raise ValueError(
                f"production case {case.get('case_id')} reviews another commit"
            )
    command_value = case.get("command")
    if not isinstance(command_value, list) or not command_value:
        raise ValueError("ready production case command must be a non-empty list")
    if command_value[0] != "python" or not all(
        isinstance(value, str) for value in command_value
    ):
        raise ValueError("production command must use the matrix Python placeholder")
    if expected_result_count <= 0:
        raise ValueError("expected_result_count must be positive")
    command = [str(interpreter), *command_value[1:]]
    if "--progress-every" not in command:
        command.extend(("--progress-every", "2"))
    return command


def _run_logged_command(
    *,
    command: list[str],
    log_path: Path,
    heartbeat_seconds: float,
    case_id: str,
    resource: str,
) -> int:
    if heartbeat_seconds <= 0.0:
        raise ValueError("heartbeat_seconds must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        _write_line(log, f"queue-command {json.dumps(command)}")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[6],
            env=_child_environment(resource=resource),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError("production child stdout pipe was not created")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        last_heartbeat = started
        while process.poll() is None:
            events = selector.select(timeout=1.0)
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    _write_line(log, line.rstrip("\n"), echo=True)
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_seconds:
                elapsed = now - started
                message = f"queue heartbeat case={case_id} elapsed_s={elapsed:.1f}"
                _write_line(log, message, echo=True)
                last_heartbeat = now
        for line in process.stdout:
            _write_line(log, line.rstrip("\n"), echo=True)
        selector.close()
        return int(process.returncode)


def _child_environment(*, resource: str) -> dict[str, str]:
    if resource not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported production resource: {resource}")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": "/home/hans/python/quant_data",
            "QUANTDATA_ROOT": "/home/hans/quantdata_data",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if resource == "cpu":
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
            }
        )
    else:
        environment.pop("JAX_PLATFORMS", None)
        if not environment.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError(
                "GPU production queue must run inside a managed gpuctl allocation"
            )
    return environment


def _check_resources() -> None:
    try:
        result = subprocess.run(
            (str(GPUCTL_PATH), "resources", "--json"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        print(
            "queue resource warning: gpuctl resources could not be launched: "
            f"{error}",
            flush=True,
        )
        return
    if result.returncode != 0:
        print(
            "queue resource warning: gpuctl resources could not be read: "
            f"{result.stderr.strip()}",
            flush=True,
        )
        return
    payload = json.loads(result.stdout)
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise RuntimeError("gpuctl resources returned no checks")
    health = {
        str(check.get("health"))
        for check in checks
        if isinstance(check, dict)
    }
    stopped = health & {"STOP", "UNAVAILABLE", "ERROR"}
    if stopped:
        raise RuntimeError(f"resource gate prevents new work: {sorted(stopped)}")
    if "WARN" in health:
        print("queue resource warning: gpuctl reported WARN", flush=True)


def _validate_results(
    path: Path,
    *,
    output: Path,
    matrix_schema: str,
    expected_model: str,
    expected_result_count: int,
    dataset: Path | None = None,
) -> str:
    payload = _read_json_object(path)
    if matrix_schema == DAILY_MATRIX_SCHEMA:
        from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
            authenticate_completed_daily_results,
        )

        if payload.get("schema") != DAILY_RESULTS_SCHEMA:
            raise ValueError(f"unexpected daily-results schema: {path}")
        if dataset is None:
            raise ValueError("daily production validation requires its public dataset")
        report = authenticate_completed_daily_results(
            _load_daily_panel(dataset),
            results_path=path,
        )
        if report.model != expected_model:
            raise ValueError(
                "daily-results model mismatch: "
                f"{report.model} != {expected_model}"
            )
    elif matrix_schema == WEEKLY_MATRIX_SCHEMA:
        run = _read_json_object(output / "run.json")
        summary = _read_json_object(output / "summary.json")
        from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
            WeeklyEvaluationSpec,
            WeeklyEvaluationSummary,
        )
        from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
            WeeklyRunResults,
        )

        spec = WeeklyEvaluationSpec.from_dict(run)
        completion = WeeklyEvaluationSummary.from_dict(summary)
        aggregate = WeeklyRunResults.from_dict(payload)
        if spec.model != expected_model:
            raise ValueError(
                f"weekly-results model mismatch: {spec.model} != {expected_model}"
            )
        if aggregate.model != expected_model:
            raise ValueError(
                f"weekly aggregate model mismatch: {aggregate.model} != "
                f"{expected_model}"
            )
        if completion.completed_origins != completion.eligible_origins:
            raise ValueError("weekly-results origins are incomplete")
        if completion.result_count != aggregate.result_count:
            raise ValueError("weekly summary and aggregate result counts differ")
        if spec.require_no_cap_hits and completion.cap_hits != 0:
            raise ValueError("weekly-results contain position-cap hits")
        if spec.require_no_cap_hits and completion.nonconverged_fits != 0:
            raise ValueError("weekly-results contain nonconverged policy fits")
        if (
            aggregate.fit_cap_hits != completion.fit_cap_hits
            or aggregate.executed_cap_hits != completion.executed_cap_hits
            or aggregate.nonconverged_fits != completion.nonconverged_fits
            or aggregate.cap_hits != completion.cap_hits
        ):
            raise ValueError("weekly numerical diagnostics do not reconcile")
        identities = {spec.run_identity, completion.run_identity, aggregate.run_identity}
        if len(identities) != 1:
            raise ValueError("weekly run, summary, and aggregate identities differ")
        if (
            aggregate.measure != spec.measure
            or aggregate.schedule != spec.schedule.value
            or aggregate.dataset_manifest_sha256
            != spec.dataset_manifest_sha256
        ):
            raise ValueError("weekly aggregate scope does not match its run")
        run_path = output / "run.json"
        summary_path = output / "summary.json"
        if aggregate.source_run_sha256 != _file_sha256(run_path):
            raise ValueError("weekly aggregate run hash changed")
        if aggregate.source_summary_sha256 != _file_sha256(summary_path):
            raise ValueError("weekly aggregate summary hash changed")
        if tuple(aggregate.record_sha256_by_origin) != completion.completed_origins:
            raise ValueError("weekly aggregate source-record dates are incomplete")
        for origin, expected_sha256 in aggregate.record_sha256_by_origin.items():
            record_path = output / "records" / f"{origin}.json"
            if expected_sha256 != _file_sha256(record_path):
                raise ValueError(f"weekly aggregate record hash changed for {origin}")
    else:
        raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")
    if payload.get("result_count") != expected_result_count:
        raise ValueError(
            "daily-results count mismatch: "
            f"{payload.get('result_count')} != {expected_result_count}"
        )
    if not isinstance(payload.get("run_identity"), str):
        raise ValueError("production-results run identity is missing")
    return _file_sha256(path)


def _load_daily_panel(dataset: Path) -> PublicHedgingPanel:
    """Load and fully validate the public panel used by a daily production run."""
    from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
        PublicHedgingPanel,
    )

    return PublicHedgingPanel(dataset, validate=True)


def _results_path(output: Path, *, matrix_schema: str) -> Path:
    if matrix_schema == DAILY_MATRIX_SCHEMA:
        return output / "results.json"
    if matrix_schema == WEEKLY_MATRIX_SCHEMA:
        return output / "results.json"
    raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")


def _load_or_initialize_state(
    *,
    state_path: Path,
    matrix_path: Path,
    matrix_sha256: str,
    case_ids: tuple[str, ...],
    resource: str,
    adopt_unpinned_state: bool,
) -> dict[str, object]:
    if state_path.exists():
        state = _read_json_object(state_path)
        if (
            state.get("schema") not in {QUEUE_SCHEMA, LEGACY_DAILY_QUEUE_SCHEMA}
            or state.get("matrix_path") != str(matrix_path.resolve())
            or tuple(state.get("case_ids", ())) != case_ids
            or state.get("resource") != resource
        ):
            raise ValueError("existing production queue state does not match request")
        pinned_sha256 = state.get("matrix_sha256")
        if pinned_sha256 is None:
            if not adopt_unpinned_state:
                raise ValueError(
                    "legacy production queue state is not matrix-pinned; "
                    "use explicit adoption after verifying its matrix"
                )
            state["legacy_state_adopted_at"] = _utc_now()
            state["matrix_sha256"] = matrix_sha256
            _write_json_atomic(state_path, state)
        elif pinned_sha256 != matrix_sha256:
            raise ValueError("production matrix bytes changed after queue creation")
        state["matrix_sha256"] = matrix_sha256
        return state
    return {
        "schema": QUEUE_SCHEMA,
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": matrix_sha256,
        "resource": resource,
        "case_ids": list(case_ids),
        "status": "pending",
        "current_case_id": None,
        "started_at": _utc_now(),
        "completed_at": None,
        "records": [],
    }


def _finish_record(
    *,
    state: dict[str, object],
    state_path: Path,
    case_id: str,
    status: str,
    return_code: int,
    error: str | None = None,
    validated_results_sha256: str | None = None,
) -> None:
    record = next(
        record for record in _records(state) if record.get("case_id") == case_id
    )
    record["status"] = status
    record["completed_at"] = _utc_now()
    record["return_code"] = return_code
    if error is None:
        record.pop("error", None)
    else:
        record["error"] = error
    if validated_results_sha256 is None:
        record.pop("validated_results_sha256", None)
    else:
        record["validated_results_sha256"] = validated_results_sha256
    state["status"] = status if status == "failed" else "running"
    state["current_case_id"] = None
    _write_json_atomic(state_path, state)


def _upsert_record(state: dict[str, object], record: dict[str, object]) -> None:
    records = _records(state)
    for index, existing in enumerate(records):
        if existing.get("case_id") == record.get("case_id"):
            records[index] = record
            return
    records.append(record)


def _records(state: dict[str, object]) -> list[dict[str, object]]:
    value = state.get("records")
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise ValueError("production queue records must be a list of objects")
    return value


def _command_option_path(command: list[str], option: str) -> Path:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"production command is missing {option}") from error
    return Path(value)


def _read_json_object(path: Path) -> dict[str, object]:
    return _read_json_object_bytes(path.read_bytes(), path=path)


def _read_json_object_bytes(value: bytes, *, path: Path) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_line(log: TextIO, line: str, *, echo: bool = False) -> None:
    log.write(line + "\n")
    log.flush()
    if echo:
        print(line, flush=True)


def _require_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(payload: dict[str, object], name: str) -> str:
    value = _require_string(payload, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256")
    return value.lower()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    """Parse a matrix queue request and run selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-prefix")
    parser.add_argument("--resource", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--expected-result-count",
        type=int,
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=240.0)
    parser.add_argument("--adopt-unpinned-state", action="store_true")
    parser.add_argument("--continue-after-failure", action="store_true")
    arguments = parser.parse_args()
    execute_production_queue(
        matrix_path=arguments.matrix,
        state_path=arguments.state,
        interpreter=arguments.interpreter,
        case_ids=tuple(arguments.case),
        case_prefix=arguments.case_prefix,
        resource=arguments.resource,
        expected_result_count=arguments.expected_result_count,
        heartbeat_seconds=arguments.heartbeat_seconds,
        adopt_unpinned_state=arguments.adopt_unpinned_state,
        continue_after_failure=arguments.continue_after_failure,
    )


if __name__ == "__main__":
    main()
