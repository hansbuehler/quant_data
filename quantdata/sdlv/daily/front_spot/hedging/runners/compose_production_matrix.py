"""Compose a final production matrix from authenticated completed cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypeAlias

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import (
    execute_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_failures import (
    ProductionFailure,
    build_production_failure_ledger,
    load_bound_production_failure_ledger,
    normalized_command_identity,
    write_production_failure_ledger,
)

JsonObject: TypeAlias = dict[str, object]


def compose_production_matrix(
    *,
    base_matrix: Path,
    source_matrices: tuple[Path, ...],
    replace_models: frozenset[str],
    expected_result_count: int,
    expected_results_schema: str,
    dataset: Path,
    output: Path,
    source_failure_ledgers: tuple[Path, ...] = (),
    source_failure_ledger_sha256: tuple[str, ...] = (),
    failure_ledger_output: Path | None = None,
) -> JsonObject:
    """Replace selected base cases, preferring any authenticated completion."""
    if not source_matrices:
        raise ValueError("at least one source matrix is required")
    if not replace_models:
        raise ValueError("at least one replacement model is required")
    if expected_result_count <= 0:
        raise ValueError("expected result count must be positive")
    if len(source_failure_ledgers) != len(source_failure_ledger_sha256):
        raise ValueError("source failure ledger paths and hashes must align")
    if (
        failure_ledger_output is not None
        and Path(failure_ledger_output).resolve() == Path(output).resolve()
    ):
        raise ValueError("failure ledger output must differ from matrix output")
    base = _read_object(base_matrix)
    matrix_schema = _required_string(base, "schema")
    expected_schema = _expected_results_schema(matrix_schema)
    if expected_results_schema != expected_schema:
        raise ValueError(
            "results schema does not match the production-matrix scope: "
            f"{expected_results_schema} != {expected_schema}"
        )
    panel = execute_production_matrix._load_daily_panel(dataset)
    base_cases = _case_map(base, label="base matrix")
    source_candidates: dict[str, list[tuple[JsonObject, Path]]] = {}
    source_hashes: list[dict[str, str]] = []
    for matrix_path in source_matrices:
        matrix = _read_object(matrix_path)
        if matrix.get("schema") != base.get("schema"):
            raise ValueError(f"source matrix schema mismatch: {matrix_path}")
        source_hashes.append(
            {"path": str(matrix_path), "sha256": _sha256(matrix_path)}
        )
        for case_id, case in _case_map(matrix, label=str(matrix_path)).items():
            source_candidates.setdefault(case_id, []).append((case, matrix_path))

    source_paths = {Path(path).resolve() for path in source_matrices}
    failure_candidates: dict[tuple[Path, str], ProductionFailure] = {}
    source_failure_evidence: list[dict[str, str]] = []
    for ledger_path, ledger_sha256 in zip(
        source_failure_ledgers,
        source_failure_ledger_sha256,
        strict=True,
    ):
        ledger = load_bound_production_failure_ledger(
            ledger_path,
            expected_ledger_sha256=ledger_sha256,
        )
        source_matrix_resolved = ledger.matrix_path.resolve()
        if source_matrix_resolved not in source_paths:
            raise ValueError(
                "failure ledger is not bound to a supplied source matrix: "
                f"{ledger_path}"
            )
        source_failure_evidence.append(
            {"path": str(ledger_path), "sha256": ledger.ledger_sha256}
        )
        for failure in ledger.failures:
            key = (source_matrix_resolved, failure.case_id)
            if key in failure_candidates:
                raise ValueError(
                    "duplicate source failure evidence for "
                    f"{failure.case_id} in {ledger.matrix_path}"
                )
            failure_candidates[key] = failure

    replacements: dict[str, JsonObject] = {}
    replacement_sources: dict[str, str] = {}
    dataset_identities: set[str] = set()
    target_notional_identities: set[str] = set()
    target_cohort_identities: set[str] = set()
    source_numerical_diagnostics: dict[str, JsonObject] = {}
    selected_failures: dict[str, ProductionFailure] = {}
    failure_replacement_sources: dict[str, str] = {}
    for case_id, base_case in base_cases.items():
        model = _required_string(base_case, "model")
        if model not in replace_models:
            continue
        completed_candidates: list[
            tuple[JsonObject, Path, tuple[str, str, str, JsonObject]]
        ] = []
        failed_candidates: list[tuple[JsonObject, Path, ProductionFailure]] = []
        candidate_errors: list[ValueError] = []
        for candidate, source_path in source_candidates.get(case_id, ()):  # priority order
            _require_same_identity(case_id, base_case, candidate)
            failure = failure_candidates.get((source_path.resolve(), case_id))
            try:
                completed_identity = _completed_output_identity(
                    candidate,
                    expected_result_count=expected_result_count,
                    expected_results_schema=expected_results_schema,
                    matrix_schema=matrix_schema,
                    dataset=dataset,
                    panel=panel,
                )
            except ValueError as error:
                candidate_errors.append(error)
                completed_identity = None
            if completed_identity is not None:
                completed_candidates.append(
                    (candidate, source_path, completed_identity)
                )
            elif failure is not None:
                failed_candidates.append((candidate, source_path, failure))
        if completed_candidates:
            candidate, source_path, completed_identity = completed_candidates[0]
            replacements[case_id] = candidate
            replacement_sources[case_id] = str(source_path)
            (
                dataset_identity,
                target_notional_identity,
                target_cohort_identity,
                numerical_diagnostics,
            ) = completed_identity
            dataset_identities.add(dataset_identity)
            target_notional_identities.add(target_notional_identity)
            target_cohort_identities.add(target_cohort_identity)
            source_numerical_diagnostics[case_id] = numerical_diagnostics
        elif failed_candidates:
            candidate, source_path, failure = failed_candidates[0]
            replacements[case_id] = candidate
            replacement_sources[case_id] = str(source_path)
            failure_replacement_sources[case_id] = str(failure.queue_state_path)
            selected_failures[case_id] = failure
            source_numerical_diagnostics[case_id] = {
                "gate_required": True,
                "classification": failure.classification,
                "triggered_gates": list(failure.triggered_gates),
                "summary": failure.summary,
                "first_failed_origin": (
                    None
                    if failure.first_failed_origin is None
                    else failure.first_failed_origin.isoformat()
                ),
            }
        elif candidate_errors:
            raise candidate_errors[0]
        else:
            raise ValueError(
                f"no complete authenticated source case for {case_id}"
            )
    if len(dataset_identities) != 1:
        raise ValueError("completed source cases use different public datasets")
    if len(target_notional_identities) != 1:
        raise ValueError("completed source cases use different target notionals")
    if len(target_cohort_identities) != 1:
        raise ValueError("completed source cases use different target cohorts")

    cases = base.get("cases")
    if not isinstance(cases, list):
        raise ValueError("base matrix cases must be a list")
    composed_cases = [
        replacements.get(_required_string(_as_object(case), "case_id"), _as_object(case))
        for case in cases
    ]
    payload = dict(base)
    payload["cases"] = composed_cases
    payload["composition"] = {
        "schema": "hedging-production-matrix-composition-v1",
        "base_matrix": str(base_matrix),
        "base_matrix_sha256": _sha256(base_matrix),
        "source_matrices": source_hashes,
        "source_failure_ledgers": source_failure_evidence,
        "replace_models": sorted(replace_models),
        "replacement_sources": replacement_sources,
        "failure_replacement_sources": failure_replacement_sources,
        "dataset_manifest_sha256": dataset_identities.pop(),
        "target_notional_identity": target_notional_identities.pop(),
        "target_cohort_sha256": target_cohort_identities.pop(),
        "source_numerical_diagnostics": source_numerical_diagnostics,
    }
    if selected_failures and failure_ledger_output is None:
        raise ValueError(
            "composed production failures require a failure-ledger output"
        )
    _write_object(output, payload)
    if failure_ledger_output is not None:
        declarations = tuple(
            {
                "case_id": failure.case_id,
                "queue_state_path": str(failure.queue_state_path),
            }
            for failure in selected_failures.values()
        )
        write_production_failure_ledger(
            build_production_failure_ledger(
                matrix_path=output,
                declarations=declarations,
            ),
            output=failure_ledger_output,
        )
    return payload


def _completed_output_identity(
    case: JsonObject,
    *,
    expected_result_count: int,
    expected_results_schema: str,
    matrix_schema: str,
    dataset: Path,
    panel: object,
) -> tuple[str, str, str, JsonObject] | None:
    if case.get("readiness") != "ready" or case.get("blocked_reason") is not None:
        return None
    command = case.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) and value for value in command
    ):
        return None
    try:
        output = Path(command[command.index("--output") + 1])
    except (ValueError, IndexError):
        return None
    run_path = output / "run.json"
    summary_path = output / "summary.json"
    results_path = output / "results.json"
    if (
        not run_path.is_file()
        or not summary_path.is_file()
        or not results_path.is_file()
    ):
        return None
    execute_production_matrix._validate_results(
        results_path,
        output=output,
        matrix_schema=matrix_schema,
        expected_model=_required_string(case, "model"),
        expected_result_count=expected_result_count,
        dataset=dataset,
        panel=panel,
    )
    run = _read_object(run_path)
    summary = _read_object(summary_path)
    results = _read_object(results_path)
    if results.get("schema") != expected_results_schema:
        return None
    if results.get("result_count") != expected_result_count:
        return None
    if results.get("run_identity") != run.get("run_identity"):
        return None
    if results.get("model") != case.get("model"):
        return None
    if results.get("measure") != case.get("measure"):
        raise ValueError("completed source measure does not match its matrix case")
    dataset_identity = results.get("dataset_manifest_sha256")
    if (
        not isinstance(dataset_identity, str)
        or not dataset_identity
        or dataset_identity != run.get("dataset_manifest_sha256")
    ):
        raise ValueError("completed source dataset identity is inconsistent")
    target_notional_identity = run.get("target_notional_identity")
    if not isinstance(target_notional_identity, str) or not target_notional_identity:
        raise ValueError("completed source target-notional identity is missing")
    target_labels = _required_string_list(run, "target_labels")
    target_quantities = _required_number_list(run, "target_quantities")
    if len(target_labels) != len(target_quantities):
        raise ValueError("completed source target cohort is not aligned")
    if matrix_schema == execute_production_matrix.DAILY_MATRIX_SCHEMA:
        expected_labels = tuple(target.label for target in daily_digital_targets())
        if tuple(target_labels) != expected_labels:
            raise ValueError("completed daily source uses the wrong target cohort")
        nonconverged = _required_integer(summary, "nonconverged_results")
        cap_hits = _required_integer(summary, "cap_hits")
        numerical_gate_required = _required_boolean(run, "require_convergence")
    elif matrix_schema == execute_production_matrix.WEEKLY_MATRIX_SCHEMA:
        nonconverged = _required_integer(summary, "nonconverged_fits")
        cap_hits = _required_integer(summary, "cap_hits")
        numerical_gate_required = _required_boolean(run, "require_no_cap_hits")
    else:
        raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")
    if numerical_gate_required and (nonconverged or cap_hits):
        raise ValueError("completed source failed its numerical gates")
    target_cohort_identity = _canonical_sha256(
        {
            "target_labels": target_labels,
            "target_quantities": target_quantities,
            "target_notional_identity": target_notional_identity,
        }
    )
    numerical_diagnostics: JsonObject = {
        "gate_required": numerical_gate_required,
        "nonconverged": nonconverged,
        "cap_hits": cap_hits,
    }
    return (
        dataset_identity,
        target_notional_identity,
        target_cohort_identity,
        numerical_diagnostics,
    )


def _expected_results_schema(matrix_schema: str) -> str:
    if matrix_schema == execute_production_matrix.DAILY_MATRIX_SCHEMA:
        return execute_production_matrix.DAILY_RESULTS_SCHEMA
    if matrix_schema == execute_production_matrix.WEEKLY_MATRIX_SCHEMA:
        return execute_production_matrix.WEEKLY_RESULTS_SCHEMA
    raise ValueError(f"unsupported production-matrix schema: {matrix_schema}")


def _case_map(matrix: JsonObject, *, label: str) -> dict[str, JsonObject]:
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{label} cases must be a list")
    result: dict[str, JsonObject] = {}
    for value in cases:
        case = _as_object(value)
        case_id = _required_string(case, "case_id")
        if case_id in result:
            raise ValueError(f"duplicate case {case_id} in {label}")
        result[case_id] = case
    return result


def _require_same_identity(
    case_id: str,
    base: JsonObject,
    candidate: JsonObject,
) -> None:
    fields = ("case_id", "model", "measure", "variant", "objective", "risk_parameter")
    if any(base.get(field) != candidate.get(field) for field in fields):
        raise ValueError(f"source case identity mismatch for {case_id}")
    base_command = tuple(_required_string_list(base, "command"))
    candidate_command = tuple(_required_string_list(candidate, "command"))
    if normalized_command_identity(base_command) != normalized_command_identity(
        candidate_command
    ):
        raise ValueError(f"source case command mismatch for {case_id}")


def _read_object(path: Path) -> JsonObject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _as_object(payload)


def _write_object(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("JSON payload must be an object")
    return value


def _required_string(payload: JsonObject, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_string_list(payload: JsonObject, field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    return value


def _required_number_list(payload: JsonObject, field: str) -> list[float | int]:
    value = payload.get(field)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, (float, int)) and not isinstance(item, bool)
        for item in value
    ):
        raise ValueError(f"{field} must be a non-empty number list")
    return value


def _required_integer(payload: JsonObject, field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _required_boolean(payload: JsonObject, field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _canonical_sha256(payload: JsonObject) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    """Compose a complete matrix from a fresh base and prior completed cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument(
        "--source-matrix",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--replace-model",
        action="append",
        required=True,
    )
    parser.add_argument("--source-failure-ledger", type=Path, action="append", default=[])
    parser.add_argument(
        "--source-failure-ledger-sha256",
        action="append",
        default=[],
    )
    parser.add_argument("--failure-ledger-output", type=Path)
    parser.add_argument("--expected-result-count", type=int, required=True)
    parser.add_argument("--expected-results-schema", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = compose_production_matrix(
        base_matrix=arguments.base_matrix,
        source_matrices=tuple(arguments.source_matrix),
        replace_models=frozenset(arguments.replace_model),
        expected_result_count=arguments.expected_result_count,
        expected_results_schema=arguments.expected_results_schema,
        dataset=arguments.dataset,
        output=arguments.output,
        source_failure_ledgers=tuple(arguments.source_failure_ledger),
        source_failure_ledger_sha256=tuple(
            arguments.source_failure_ledger_sha256
        ),
        failure_ledger_output=arguments.failure_ledger_output,
    )
    print(
        json.dumps(
            {
                "schema": payload.get("schema"),
                "case_count": payload.get("case_count"),
                "output": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "failure_ledger_output": (
                    None
                    if arguments.failure_ledger_output is None
                    else str(arguments.failure_ledger_output)
                ),
                "failure_ledger_sha256": (
                    None
                    if arguments.failure_ledger_output is None
                    else _sha256(arguments.failure_ledger_output)
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
