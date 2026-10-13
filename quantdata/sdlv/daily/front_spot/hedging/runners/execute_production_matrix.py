"""Execute selected ready cases from an immutable daily production matrix."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

DAILY_MATRIX_SCHEMA = "hedging-daily-production-matrix-v1"
WEEKLY_MATRIX_SCHEMA = "hedging-weekly-production-matrix-v1"
DAILY_RESULTS_SCHEMA = "hedging-daily-results-v2"
WEEKLY_RESULTS_SCHEMA = "hedging-weekly-evaluation-v1"
QUEUE_SCHEMA = "hedging-production-queue-v1"
LEGACY_DAILY_QUEUE_SCHEMA = "hedging-daily-production-queue-v1"
DEFAULT_DAILY_RESULT_COUNT = 12_220
DEFAULT_WEEKLY_RESULT_COUNT = 6_864
GPUCTL_PATH = Path("/home/hans/.local/bin/gpuctl")


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
) -> dict[str, object]:
    """Run selected matrix cases sequentially and persist atomic queue evidence."""
    matrix = _read_json_object(matrix_path)
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
        case_ids=tuple(_require_string(case, "case_id") for case in selected),
        resource=resource,
    )
    completed = {
        _require_string(record, "case_id")
        for record in _records(state)
        if record.get("status") == "complete"
    }
    ran_case = False
    for case in selected:
        case_id = _require_string(case, "case_id")
        if case_id in completed:
            print(f"queue case={case_id} reused queue-state completion", flush=True)
            continue
        command = _production_command(
            case=case,
            interpreter=interpreter,
            expected_result_count=expected_result_count,
        )
        output = _command_option_path(command, "--output")
        results_path = _results_path(output, matrix_schema=str(matrix_schema))
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
            )
            raise RuntimeError(f"production case failed: {case_id} ({return_code})")
        _validate_results(
            results_path,
            output=output,
            matrix_schema=str(matrix_schema),
            expected_model=_require_string(case, "model"),
            expected_result_count=expected_result_count,
        )
        _finish_record(
            state=state,
            state_path=state_path,
            case_id=case_id,
            status="complete",
            return_code=return_code,
        )
        completed.add(case_id)
        print(f"queue case={case_id} complete", flush=True)
    if not ran_case and state.get("status") == "complete":
        print(f"queue reused completion cases={len(selected)}", flush=True)
        return state
    state["status"] = "complete"
    state["current_case_id"] = None
    state["completed_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    print(f"queue complete cases={len(selected)}", flush=True)
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
) -> None:
    payload = _read_json_object(path)
    if matrix_schema == DAILY_MATRIX_SCHEMA:
        if payload.get("schema") != DAILY_RESULTS_SCHEMA:
            raise ValueError(f"unexpected daily-results schema: {path}")
        if payload.get("model") != expected_model:
            raise ValueError(
                "daily-results model mismatch: "
                f"{payload.get('model')} != {expected_model}"
            )
    elif matrix_schema == WEEKLY_MATRIX_SCHEMA:
        if payload.get("schema") != WEEKLY_RESULTS_SCHEMA:
            raise ValueError(f"unexpected weekly-results schema: {path}")
        run = _read_json_object(output / "run.json")
        if run.get("schema") != WEEKLY_RESULTS_SCHEMA:
            raise ValueError("weekly run metadata has an unexpected schema")
        if run.get("model") != expected_model:
            raise ValueError(
                f"weekly-results model mismatch: {run.get('model')} != {expected_model}"
            )
        completed = payload.get("completed_origins")
        eligible = payload.get("eligible_origins")
        if completed != eligible:
            raise ValueError("weekly-results origins are incomplete")
        if payload.get("cap_hits") != 0:
            raise ValueError("weekly-results contain position-cap hits")
        if run.get("run_identity") != payload.get("run_identity"):
            raise ValueError("weekly run and summary identities do not match")
    else:
        raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")
    if payload.get("result_count") != expected_result_count:
        raise ValueError(
            "daily-results count mismatch: "
            f"{payload.get('result_count')} != {expected_result_count}"
        )
    if not isinstance(payload.get("run_identity"), str):
        raise ValueError("production-results run identity is missing")


def _results_path(output: Path, *, matrix_schema: str) -> Path:
    if matrix_schema == DAILY_MATRIX_SCHEMA:
        return output / "results" / "results.json"
    if matrix_schema == WEEKLY_MATRIX_SCHEMA:
        return output / "summary.json"
    raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")


def _load_or_initialize_state(
    *,
    state_path: Path,
    matrix_path: Path,
    case_ids: tuple[str, ...],
    resource: str,
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
        return state
    return {
        "schema": QUEUE_SCHEMA,
        "matrix_path": str(matrix_path.resolve()),
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
) -> None:
    record = next(
        record for record in _records(state) if record.get("case_id") == case_id
    )
    record["status"] = status
    record["completed_at"] = _utc_now()
    record["return_code"] = return_code
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


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
    )


if __name__ == "__main__":
    main()
