"""Authenticate strict production failures without weakening numerical gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TypeAlias

from quantdata.sdlv.daily.front_spot.hedging.runners.execute_production_matrix import (
    DAILY_MATRIX_SCHEMA,
    LEGACY_DAILY_QUEUE_SCHEMA,
    QUEUE_SCHEMA,
    WEEKLY_MATRIX_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    NUMERICAL_GATE_FAILURE_FILENAME,
    load_numerical_gate_failure,
)

JsonObject: TypeAlias = dict[str, object]

FAILURE_LEDGER_SCHEMA = "hedging-production-failure-ledger-v2"
FAILURE_DECLARATIONS_SCHEMA = "hedging-production-failure-declarations-v2"
FAILURE_CLASSIFICATIONS = frozenset(
    {
        "position-cap-hit",
        "policy-fit-cap-hit",
        "nonconverged-fit",
    }
)
_PATH_OPTIONS = frozenset({"--checkpoint-directory", "--output"})


@dataclass(frozen=True, kw_only=True)
class ProductionFailure:
    """One precommitted matrix case rejected by its strict production gate."""

    case_id: str
    model: str
    measure: str
    variant: str
    objective: str | None
    risk_parameter: float | None
    schedule: str | None
    classification: str
    triggered_gates: tuple[str, ...]
    summary: str
    first_failed_origin: date | None
    source_matrix_path: Path
    source_matrix_sha256: str
    queue_state_path: Path
    queue_state_sha256: str
    queue_log_path: Path
    queue_log_sha256: str
    run_metadata_path: Path
    run_metadata_sha256: str
    gate_failure_path: Path
    gate_failure_sha256: str


@dataclass(frozen=True, kw_only=True)
class ValidatedProductionFailureLedger:
    """One validated ledger together with its bound source matrix."""

    ledger_path: Path
    ledger_sha256: str
    matrix_path: Path
    matrix_sha256: str
    failures: tuple[ProductionFailure, ...]


def load_bound_production_failure_ledger(
    ledger_path: Path,
    *,
    expected_ledger_sha256: str,
) -> ValidatedProductionFailureLedger:
    """Load a ledger using the matrix path and hash stored in its header."""
    ledger_path = Path(ledger_path)
    _require_file_sha256(
        ledger_path,
        expected_ledger_sha256,
        label="production failure ledger",
    )
    payload = _read_object(ledger_path)
    matrix_path = Path(_require_string(payload, "matrix_path"))
    matrix_sha256 = _require_sha256(payload, "matrix_sha256")
    failures = load_production_failure_ledger(
        ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        matrix_path=matrix_path,
        expected_matrix_sha256=matrix_sha256,
    )
    return ValidatedProductionFailureLedger(
        ledger_path=ledger_path,
        ledger_sha256=expected_ledger_sha256.lower(),
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha256,
        failures=failures,
    )


def load_production_failure_ledger(
    ledger_path: Path,
    *,
    expected_ledger_sha256: str,
    matrix_path: Path,
    expected_matrix_sha256: str,
) -> tuple[ProductionFailure, ...]:
    """Validate a failure ledger and every queue/log artifact it cites."""
    ledger_path = Path(ledger_path)
    matrix_path = Path(matrix_path)
    _require_file_sha256(
        ledger_path,
        expected_ledger_sha256,
        label="production failure ledger",
    )
    _require_file_sha256(
        matrix_path,
        expected_matrix_sha256,
        label="production matrix",
    )
    ledger = _read_object(ledger_path)
    matrix = _read_object(matrix_path)
    matrix_schema = _require_matrix_schema(matrix)
    if ledger.get("schema") != FAILURE_LEDGER_SCHEMA:
        raise ValueError(f"unexpected production failure ledger: {ledger_path}")
    if ledger.get("matrix_schema") != matrix_schema:
        raise ValueError("failure ledger matrix schema does not match its matrix")
    if ledger.get("matrix_sha256") != expected_matrix_sha256.lower():
        raise ValueError("failure ledger does not bind the requested matrix bytes")
    failures_value = ledger.get("failures")
    if not isinstance(failures_value, list):
        raise ValueError("failure ledger failures must be a list")
    if ledger.get("failure_count") != len(failures_value):
        raise ValueError("failure ledger count does not match its entries")
    matrix_cases = _case_map(matrix, label="production matrix")
    failures: list[ProductionFailure] = []
    seen: set[str] = set()
    for raw_failure in failures_value:
        failure = _as_object(raw_failure, "production failure")
        case_id = _require_string(failure, "case_id")
        if case_id in seen:
            raise ValueError(f"duplicate production failure: {case_id}")
        seen.add(case_id)
        try:
            matrix_case = matrix_cases[case_id]
        except KeyError as error:
            raise ValueError(
                f"failure ledger case is absent from production matrix: {case_id}"
            ) from error
        failures.append(
            _validate_failure(
                failure,
                matrix_case=matrix_case,
                matrix_schema=matrix_schema,
            )
        )
    return tuple(failures)


def build_production_failure_ledger(
    *,
    matrix_path: Path,
    declarations: tuple[JsonObject, ...],
) -> JsonObject:
    """Build a hash-bound ledger from failed production-queue records."""
    matrix_path = Path(matrix_path)
    matrix = _read_object(matrix_path)
    matrix_schema = _require_matrix_schema(matrix)
    matrix_cases = _case_map(matrix, label="production matrix")
    failures: list[JsonObject] = []
    seen: set[str] = set()
    for declaration in declarations:
        case_id = _require_string(declaration, "case_id")
        if case_id in seen:
            raise ValueError(f"duplicate production failure declaration: {case_id}")
        seen.add(case_id)
        try:
            matrix_case = matrix_cases[case_id]
        except KeyError as error:
            raise ValueError(
                f"failure declaration case is absent from production matrix: {case_id}"
            ) from error
        queue_state_path = Path(_require_string(declaration, "queue_state_path"))
        queue_state = _read_object(queue_state_path)
        source_matrix_path = Path(_require_string(queue_state, "matrix_path"))
        source_matrix = _read_object(source_matrix_path)
        source_cases = _case_map(source_matrix, label="source production matrix")
        try:
            source_case = source_cases[case_id]
        except KeyError as error:
            raise ValueError(
                f"failure source matrix has no case {case_id}"
            ) from error
        _require_same_case_identity(matrix_case, source_case, case_id=case_id)
        matrix_command = _require_command(matrix_case, label=f"matrix case {case_id}")
        source_command = _require_command(
            source_case,
            label=f"source matrix case {case_id}",
        )
        if normalized_command_identity(
            matrix_command
        ) != normalized_command_identity(source_command):
            raise ValueError(
                f"failure source command differs from final matrix case {case_id}"
            )
        record = _failed_queue_record(
            queue_state,
            case_id=case_id,
            expected_resource=source_case.get("resource"),
        )
        queue_log_path = Path(_require_string(record, "log_path"))
        if not queue_log_path.is_file():
            raise FileNotFoundError(f"failure queue log is missing: {queue_log_path}")
        recorded_gate_path, recorded_gate_sha256 = _recorded_gate_evidence(
            record,
            case_id=case_id,
        )
        (
            gate_failure_path,
            gate_failure,
            run_metadata_path,
        ) = _validated_gate_evidence(
            source_case=source_case,
            source_command=source_command,
        )
        if gate_failure_path.resolve() != recorded_gate_path.resolve():
            raise ValueError(f"failure queue gate artifact path mismatch: {case_id}")
        if _file_sha256(gate_failure_path) != recorded_gate_sha256:
            raise ValueError(f"failure queue gate artifact hash mismatch: {case_id}")
        failures.append(
            {
                "case_id": case_id,
                "case_command_sha256": _canonical_sha256(
                    normalized_command_identity(matrix_command)
                ),
                "classification": _require_classification(gate_failure),
                "triggered_gates": gate_failure["triggered_gates"],
                "summary": _require_string(gate_failure, "summary"),
                "first_failed_origin": _require_string(
                    gate_failure,
                    "origin_date",
                ),
                "source_matrix_path": str(source_matrix_path.resolve()),
                "source_matrix_sha256": _file_sha256(source_matrix_path),
                "queue_state_path": str(queue_state_path.resolve()),
                "queue_state_sha256": _file_sha256(queue_state_path),
                "queue_log_path": str(queue_log_path.resolve()),
                "queue_log_sha256": _file_sha256(queue_log_path),
                "run_metadata_path": str(run_metadata_path.resolve()),
                "run_metadata_sha256": _file_sha256(run_metadata_path),
                "gate_failure_path": str(gate_failure_path.resolve()),
                "gate_failure_sha256": _file_sha256(gate_failure_path),
            }
        )
    return {
        "schema": FAILURE_LEDGER_SCHEMA,
        "matrix_schema": matrix_schema,
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": _file_sha256(matrix_path),
        "failure_count": len(failures),
        "failures": failures,
    }


def reject_completed_failure_cases(
    *,
    matrix_path: Path,
    failures: tuple[ProductionFailure, ...],
    dataset: Path,
) -> None:
    """Reject stale failure declarations when a valid completion now exists."""
    if not failures:
        return
    matrix = _read_object(matrix_path)
    matrix_schema = _require_matrix_schema(matrix)
    expected_result_count = (
        12_220 if matrix_schema == DAILY_MATRIX_SCHEMA else 6_864
    )
    cases = _case_map(matrix, label="production matrix")
    from quantdata.sdlv.daily.front_spot.hedging.runners import (
        execute_production_matrix,
    )

    for failure in failures:
        case = cases[failure.case_id]
        command = _require_command(case, label=f"matrix case {failure.case_id}")
        output_value = _command_option(command, "--output")
        if output_value is None:
            raise ValueError(
                f"production failure case has no output: {failure.case_id}"
            )
        output = Path(output_value)
        results_path = output / "results.json"
        if not results_path.is_file():
            continue
        try:
            execute_production_matrix._validate_results(
                results_path,
                output=output,
                matrix_schema=matrix_schema,
                expected_model=failure.model,
                expected_result_count=expected_result_count,
                dataset=dataset,
            )
        except Exception:
            continue
        raise ValueError(
            "production failure ledger suppresses a valid completed case: "
            f"{failure.case_id}"
        )


def write_production_failure_ledger(
    payload: JsonObject,
    *,
    output: Path,
) -> None:
    """Write one deterministic failure ledger atomically."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> None:
    """Build an authenticated failure ledger from explicit declarations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--declarations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    declarations_payload = _read_object(arguments.declarations)
    if declarations_payload.get("schema") != FAILURE_DECLARATIONS_SCHEMA:
        raise ValueError("unexpected production failure declarations schema")
    declarations_value = declarations_payload.get("failures")
    if not isinstance(declarations_value, list):
        raise ValueError("production failure declarations must be a list")
    declarations = tuple(
        _as_object(value, "production failure declaration")
        for value in declarations_value
    )
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=arguments.matrix,
            declarations=declarations,
        ),
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "schema": FAILURE_LEDGER_SCHEMA,
                "failure_count": len(declarations),
                "output": str(arguments.output),
                "sha256": _file_sha256(arguments.output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _validate_failure(
    failure: JsonObject,
    *,
    matrix_case: JsonObject,
    matrix_schema: str,
) -> ProductionFailure:
    case_id = _require_string(failure, "case_id")
    if matrix_case.get("readiness") != "ready":
        raise ValueError(f"failure case was not precommitted ready: {case_id}")
    matrix_command = _require_command(matrix_case, label=f"matrix case {case_id}")
    expected_command_sha256 = _canonical_sha256(
        normalized_command_identity(matrix_command)
    )
    if failure.get("case_command_sha256") != expected_command_sha256:
        raise ValueError(f"failure command identity mismatch: {case_id}")
    source_matrix_path = Path(_require_string(failure, "source_matrix_path"))
    source_matrix_sha256 = _require_sha256(failure, "source_matrix_sha256")
    _require_file_sha256(
        source_matrix_path,
        source_matrix_sha256,
        label=f"failure source matrix for {case_id}",
    )
    source_matrix = _read_object(source_matrix_path)
    if _require_matrix_schema(source_matrix) != matrix_schema:
        raise ValueError(f"failure source matrix scope mismatch: {case_id}")
    source_cases = _case_map(source_matrix, label="failure source matrix")
    try:
        source_case = source_cases[case_id]
    except KeyError as error:
        raise ValueError(f"failure source matrix has no case {case_id}") from error
    _require_same_case_identity(matrix_case, source_case, case_id=case_id)
    source_command = _require_command(
        source_case,
        label=f"source matrix case {case_id}",
    )
    if normalized_command_identity(matrix_command) != normalized_command_identity(source_command):
        raise ValueError(f"failure source command mismatch: {case_id}")

    queue_state_path = Path(_require_string(failure, "queue_state_path"))
    queue_state_sha256 = _require_sha256(failure, "queue_state_sha256")
    _require_file_sha256(
        queue_state_path,
        queue_state_sha256,
        label=f"failure queue state for {case_id}",
    )
    queue_state = _read_object(queue_state_path)
    if queue_state.get("schema") not in {QUEUE_SCHEMA, LEGACY_DAILY_QUEUE_SCHEMA}:
        raise ValueError(f"failure queue schema is invalid: {case_id}")
    if queue_state.get("matrix_sha256") != source_matrix_sha256:
        raise ValueError(f"failure queue does not bind its source matrix: {case_id}")
    queue_matrix_path = Path(_require_string(queue_state, "matrix_path"))
    if queue_matrix_path.resolve() != source_matrix_path.resolve():
        raise ValueError(f"failure queue source path mismatch: {case_id}")
    record = _failed_queue_record(
        queue_state,
        case_id=case_id,
        expected_resource=source_case.get("resource"),
    )
    executed_command = _require_command(record, label=f"queue record {case_id}")
    if normalized_command_identity(source_command) != normalized_command_identity(
        executed_command
    ):
        raise ValueError(f"failure queue command mismatch: {case_id}")
    recorded_gate_path, recorded_gate_sha256 = _recorded_gate_evidence(
        record,
        case_id=case_id,
    )
    queue_log_path = Path(_require_string(failure, "queue_log_path"))
    if Path(_require_string(record, "log_path")).resolve() != queue_log_path.resolve():
        raise ValueError(f"failure queue log path mismatch: {case_id}")
    queue_log_sha256 = _require_sha256(failure, "queue_log_sha256")
    _require_file_sha256(
        queue_log_path,
        queue_log_sha256,
        label=f"failure queue log for {case_id}",
    )
    (
        expected_gate_failure_path,
        gate_failure,
        expected_run_metadata_path,
    ) = _validated_gate_evidence(
        source_case=source_case,
        source_command=source_command,
    )
    gate_failure_path = Path(_require_string(failure, "gate_failure_path"))
    if gate_failure_path.resolve() != expected_gate_failure_path.resolve():
        raise ValueError(f"failure gate artifact path mismatch: {case_id}")
    if gate_failure_path.resolve() != recorded_gate_path.resolve():
        raise ValueError(f"failure queue gate artifact path mismatch: {case_id}")
    gate_failure_sha256 = _require_sha256(failure, "gate_failure_sha256")
    if gate_failure_sha256 != recorded_gate_sha256:
        raise ValueError(f"failure queue gate artifact hash mismatch: {case_id}")
    _require_file_sha256(
        gate_failure_path,
        gate_failure_sha256,
        label=f"failure gate artifact for {case_id}",
    )
    run_metadata_path = Path(_require_string(failure, "run_metadata_path"))
    if run_metadata_path.resolve() != expected_run_metadata_path.resolve():
        raise ValueError(f"failure run metadata path mismatch: {case_id}")
    run_metadata_sha256 = _require_sha256(failure, "run_metadata_sha256")
    _require_file_sha256(
        run_metadata_path,
        run_metadata_sha256,
        label=f"failure run metadata for {case_id}",
    )
    triggered_gates = _require_string_tuple(failure, "triggered_gates")
    if list(triggered_gates) != gate_failure.get("triggered_gates"):
        raise ValueError(f"failure triggered gates mismatch: {case_id}")
    if failure.get("classification") != gate_failure.get("classification"):
        raise ValueError(f"failure classification mismatch: {case_id}")
    if failure.get("summary") != gate_failure.get("summary"):
        raise ValueError(f"failure summary mismatch: {case_id}")
    if failure.get("first_failed_origin") != gate_failure.get("origin_date"):
        raise ValueError(f"failure origin mismatch: {case_id}")
    first_failed_origin_value = failure.get("first_failed_origin")
    first_failed_origin = (
        None
        if first_failed_origin_value is None
        else date.fromisoformat(_require_string(failure, "first_failed_origin"))
    )
    parameter_value = matrix_case.get("risk_parameter")
    risk_parameter = (
        None
        if parameter_value is None
        else _require_number(matrix_case, "risk_parameter")
    )
    objective_value = matrix_case.get("objective")
    objective = (
        None
        if objective_value is None
        else _require_string(matrix_case, "objective")
    )
    return ProductionFailure(
        case_id=case_id,
        model=_require_string(matrix_case, "model"),
        measure=_require_string(matrix_case, "measure"),
        variant=_require_string(matrix_case, "variant"),
        objective=objective,
        risk_parameter=risk_parameter,
        schedule=_command_option(matrix_command, "--schedule"),
        classification=_require_classification(failure),
        triggered_gates=triggered_gates,
        summary=_require_string(failure, "summary"),
        first_failed_origin=first_failed_origin,
        source_matrix_path=source_matrix_path,
        source_matrix_sha256=source_matrix_sha256,
        queue_state_path=queue_state_path,
        queue_state_sha256=queue_state_sha256,
        queue_log_path=queue_log_path,
        queue_log_sha256=queue_log_sha256,
        run_metadata_path=run_metadata_path,
        run_metadata_sha256=run_metadata_sha256,
        gate_failure_path=gate_failure_path,
        gate_failure_sha256=gate_failure_sha256,
    )


def _validated_gate_evidence(
    *,
    source_case: JsonObject,
    source_command: tuple[str, ...],
) -> tuple[Path, JsonObject, Path]:
    output_value = _command_option(source_command, "--output")
    if output_value is None:
        raise ValueError("failure source command has no output directory")
    if "--allow-cap-hits" in source_command or "--allow-nonconvergence" in source_command:
        raise ValueError("failure source command disabled its precommitted numerical gate")
    output = Path(output_value)
    gate_failure_path = output / NUMERICAL_GATE_FAILURE_FILENAME
    gate_failure = load_numerical_gate_failure(gate_failure_path)
    run_metadata_path = output / "run.json"
    run_metadata = _read_object(run_metadata_path)
    evaluation_schema = _require_string(run_metadata, "schema")
    if evaluation_schema != gate_failure.get("evaluation_schema"):
        raise ValueError("failure gate artifact evaluation schema mismatch")
    if evaluation_schema.startswith("hedging-daily-evaluation-"):
        from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
            DailyEvaluationSpec,
        )

        spec = DailyEvaluationSpec.from_dict(run_metadata)
        strict_gate = spec.require_convergence
        expected_schedule = None
    elif evaluation_schema.startswith("hedging-weekly-evaluation-"):
        from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
            WeeklyEvaluationSpec,
        )

        spec = WeeklyEvaluationSpec.from_dict(run_metadata)
        strict_gate = spec.require_no_cap_hits
        expected_schedule = spec.schedule.value
    else:
        raise ValueError("failure run metadata has an unsupported evaluation schema")
    if not strict_gate:
        raise ValueError("failure run did not require its numerical gate")
    if gate_failure.get("run_identity") != spec.run_identity:
        raise ValueError("failure gate artifact run identity mismatch")
    if gate_failure.get("model") != spec.model or source_case.get("model") != spec.model:
        raise ValueError("failure gate artifact model mismatch")
    if gate_failure.get("measure") != spec.measure or source_case.get("measure") != spec.measure:
        raise ValueError("failure gate artifact measure mismatch")
    if gate_failure.get("schedule") != expected_schedule:
        raise ValueError("failure gate artifact schedule mismatch")
    if gate_failure.get("dataset_manifest_sha256") != spec.dataset_manifest_sha256:
        raise ValueError("failure gate artifact dataset mismatch")
    origin = date.fromisoformat(_require_string(gate_failure, "origin_date"))
    if not spec.evaluation_start <= origin <= spec.evaluation_end:
        raise ValueError("failure gate origin is outside the evaluation range")
    command_manifest = _command_option(source_command, "--dataset-manifest-sha256")
    if command_manifest is not None and command_manifest != spec.dataset_manifest_sha256:
        raise ValueError("failure source command dataset identity mismatch")
    return gate_failure_path, gate_failure, run_metadata_path


def _failed_queue_record(
    queue_state: JsonObject,
    *,
    case_id: str,
    expected_resource: object,
) -> JsonObject:
    if queue_state.get("status") != "complete-with-failures":
        raise ValueError(f"failure queue is not terminal: {case_id}")
    if queue_state.get("current_case_id") is not None:
        raise ValueError(f"failure queue still has a current case: {case_id}")
    queue_started_at = _require_timestamp(queue_state, "started_at")
    queue_completed_at = _require_timestamp(queue_state, "completed_at")
    if queue_completed_at < queue_started_at:
        raise ValueError(f"failure queue timestamps are reversed: {case_id}")
    case_ids = _require_string_tuple(queue_state, "case_ids")
    if case_id not in case_ids:
        raise ValueError(f"failure queue did not select case: {case_id}")
    if not isinstance(expected_resource, str) or not expected_resource:
        raise ValueError(f"failure source case resource is missing: {case_id}")
    if queue_state.get("resource") != expected_resource:
        raise ValueError(f"failure queue resource mismatch: {case_id}")
    records = queue_state.get("records")
    if not isinstance(records, list):
        raise ValueError(f"failure queue records must be a list: {case_id}")
    record_objects = tuple(
        _as_object(record, "failure queue record") for record in records
    )
    record_ids = tuple(_require_string(record, "case_id") for record in record_objects)
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(case_ids):
        raise ValueError(f"failure queue records do not match selected cases: {case_id}")
    for queue_record in record_objects:
        if queue_record.get("status") not in {"complete", "failed"}:
            raise ValueError(f"failure queue contains a nonterminal record: {case_id}")
        record_started_at = _require_timestamp(queue_record, "started_at")
        record_completed_at = _require_timestamp(queue_record, "completed_at")
        if record_completed_at < record_started_at:
            raise ValueError(f"failure queue record timestamps are reversed: {case_id}")
    matches = tuple(
        record for record in record_objects if record.get("case_id") == case_id
    )
    if len(matches) != 1:
        raise ValueError(f"failure queue must contain one record for {case_id}")
    record = matches[0]
    if record.get("status") != "failed":
        raise ValueError(f"failure queue record is not failed: {case_id}")
    return_code = record.get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise ValueError(f"failure queue return code is invalid: {case_id}")
    if return_code == 0:
        raise ValueError(f"failure queue command exited successfully: {case_id}")
    error = record.get("error")
    if not isinstance(error, str) or not error.strip():
        raise ValueError(f"failure queue record has no recorded error: {case_id}")
    return record


def _recorded_gate_evidence(
    record: JsonObject,
    *,
    case_id: str,
) -> tuple[Path, str]:
    path = Path(_require_string(record, "numerical_gate_failure_path"))
    sha256 = _require_sha256(record, "numerical_gate_failure_sha256")
    _require_file_sha256(
        path,
        sha256,
        label=f"queue-pinned numerical-gate failure for {case_id}",
    )
    return path, sha256


def _require_same_case_identity(
    left: JsonObject,
    right: JsonObject,
    *,
    case_id: str,
) -> None:
    fields = ("case_id", "model", "measure", "variant", "objective", "risk_parameter")
    if any(left.get(field) != right.get(field) for field in fields):
        raise ValueError(f"failure source case identity mismatch: {case_id}")


def normalized_command_identity(command: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize only interpreter, output paths, checkpoints, and progress cadence."""
    if not command:
        raise ValueError("production command must not be empty")
    normalized = ["python"]
    index = 1
    while index < len(command):
        option = command[index]
        if option == "--progress-every":
            if index + 1 >= len(command):
                raise ValueError("--progress-every has no value")
            index += 2
            continue
        normalized.append(option)
        if option in _PATH_OPTIONS:
            if index + 1 >= len(command):
                raise ValueError(f"{option} has no value")
            normalized.append(f"<{option[2:]}>")
            index += 2
            continue
        index += 1
    return tuple(normalized)


def _command_option(command: tuple[str, ...], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 >= len(command):
        raise ValueError(f"production command option has no value: {name}")
    return command[index + 1]


def _require_command(payload: JsonObject, *, label: str) -> tuple[str, ...]:
    value = payload.get("command")
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} command must be a non-empty string list")
    return tuple(value)


def _require_matrix_schema(matrix: JsonObject) -> str:
    schema = matrix.get("schema")
    if schema not in {DAILY_MATRIX_SCHEMA, WEEKLY_MATRIX_SCHEMA}:
        raise ValueError("unsupported production matrix schema")
    return str(schema)


def _case_map(matrix: JsonObject, *, label: str) -> dict[str, JsonObject]:
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label} cases must be a list")
    result: dict[str, JsonObject] = {}
    for value in cases:
        case = _as_object(value, f"{label} case")
        case_id = _require_string(case, "case_id")
        if case_id in result:
            raise ValueError(f"duplicate case {case_id} in {label}")
        result[case_id] = case
    return result


def _require_classification(payload: JsonObject) -> str:
    value = _require_string(payload, "classification")
    if value not in FAILURE_CLASSIFICATIONS:
        raise ValueError(f"unsupported production failure classification: {value}")
    return value


def _require_string_tuple(payload: JsonObject, name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    return tuple(value)


def _require_timestamp(payload: JsonObject, name: str) -> datetime:
    value = _require_string(payload, name)
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if timestamp.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return timestamp


def _require_number(payload: JsonObject, name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _require_sha256(payload: JsonObject, name: str) -> str:
    value = _require_string(payload, name).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a SHA-256")
    return value


def _require_file_sha256(path: Path, expected: str, *, label: str) -> None:
    expected = expected.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _as_object(payload, f"JSON artifact {path}")


def _as_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


if __name__ == "__main__":
    main()
