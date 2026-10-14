from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import (
    execute_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.compose_production_matrix import (
    compose_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import DailyEvaluationSpec
from quantdata.sdlv.daily.front_spot.hedging.runners.execute_production_matrix import (
    QUEUE_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    write_numerical_gate_failure,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_failures import (
    build_production_failure_ledger,
    load_production_failure_ledger,
    write_production_failure_ledger,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _case(
    case_id: str,
    *,
    model: str,
    output: Path,
    readiness: str = "ready",
) -> dict[str, object]:
    return {
        "blocked_reason": None if readiness == "ready" else "waiting",
        "case_id": case_id,
        "command": [
            "python",
            "runner",
            "--output",
            str(output),
            "--dataset-manifest-sha256",
            "a" * 64,
        ],
        "measure": "historical-P",
        "model": model,
        "objective": "mean-variance",
        "readiness": readiness,
        "resource": "cpu",
        "risk_parameter": 1.0,
        "variant": "test-variant",
    }


def _matrix(path: Path, cases: list[dict[str, object]]) -> None:
    _write_json(
        path,
        {
            "schema": "hedging-daily-production-matrix-v1",
            "case_count": len(cases),
            "ready_count": len(cases),
            "blocked_count": 0,
            "waiting_gate_count": 0,
            "cases": cases,
        },
    )


def _complete(
    output: Path,
    *,
    model: str,
    count: int = 7,
    require_convergence: bool = True,
    nonconverged_results: int = 0,
    cap_hits: int = 0,
) -> None:
    labels = [target.label for target in daily_digital_targets()]
    run = {
        "run_identity": "run-identity",
        "dataset_manifest_sha256": "dataset-identity",
        "target_notional_identity": "target-notional-identity",
        "target_labels": labels,
        "target_quantities": [1.0] * len(labels),
        "require_convergence": require_convergence,
    }
    results = {
        "schema": "hedging-daily-results-v2",
        "run_identity": "run-identity",
        "dataset_manifest_sha256": "dataset-identity",
        "model": model,
        "measure": "historical-P",
        "result_count": count,
    }
    _write_json(output / "run.json", run)
    _write_json(
        output / "summary.json",
        {
            "nonconverged_results": nonconverged_results,
            "cap_hits": cap_hits,
        },
    )
    _write_json(output / "results.json", results)


@pytest.fixture(autouse=True)
def _stub_deep_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execute_production_matrix,
        "_load_daily_panel",
        lambda _dataset: object(),
    )
    monkeypatch.setattr(
        execute_production_matrix,
        "_validate_results",
        lambda *_args, **_kwargs: "validated-results-sha256",
    )


def test_compose_replaces_only_complete_selected_models(tmp_path: Path) -> None:
    base_native = _case(
        "native",
        model="historical-sanos",
        output=tmp_path / "base-native",
    )
    base_generated = _case(
        "generated",
        model="gromm",
        output=tmp_path / "base-generated",
    )
    source_output = tmp_path / "source-native"
    source_native = _case(
        "native",
        model="historical-sanos",
        output=source_output,
    )
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    destination = tmp_path / "composed.json"
    _matrix(base_path, [base_native, base_generated])
    _matrix(source_path, [source_native])
    _complete(source_output, model="historical-sanos")

    payload = compose_production_matrix(
        base_matrix=base_path,
        source_matrices=(source_path,),
        replace_models=frozenset({"historical-sanos"}),
        expected_result_count=7,
        expected_results_schema="hedging-daily-results-v2",
        dataset=tmp_path / "dataset",
        output=destination,
    )

    cases = {case["case_id"]: case for case in payload["cases"]}
    assert cases["native"]["command"] == source_native["command"]
    assert cases["generated"]["command"] == base_generated["command"]
    composition = payload["composition"]
    assert composition["replacement_sources"] == {"native": str(source_path)}
    assert json.loads(destination.read_text(encoding="utf-8")) == payload


def test_compose_uses_first_complete_source(tmp_path: Path) -> None:
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    incomplete = _case(
        "native", model="historical-sanos", output=tmp_path / "incomplete"
    )
    complete_output = tmp_path / "complete"
    complete = _case("native", model="historical-sanos", output=complete_output)
    base_path = tmp_path / "base.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _matrix(base_path, [base])
    _matrix(first_path, [incomplete])
    _matrix(second_path, [complete])
    _complete(complete_output, model="historical-sanos")

    payload = compose_production_matrix(
        base_matrix=base_path,
        source_matrices=(first_path, second_path),
        replace_models=frozenset({"historical-sanos"}),
        expected_result_count=7,
        expected_results_schema="hedging-daily-results-v2",
        dataset=tmp_path / "dataset",
        output=tmp_path / "composed.json",
    )

    assert payload["composition"]["replacement_sources"] == {
        "native": str(second_path)
    }


def test_compose_prefers_later_completion_over_earlier_failure(
    tmp_path: Path,
) -> None:
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    failed = _case(
        "native",
        model="historical-sanos",
        output=tmp_path / "failed-output",
    )
    completed_output = tmp_path / "completed-output"
    completed = _case(
        "native",
        model="historical-sanos",
        output=completed_output,
    )
    base_path = tmp_path / "base.json"
    failed_path = tmp_path / "failed.json"
    completed_path = tmp_path / "completed.json"
    _matrix(base_path, [base])
    _matrix(failed_path, [failed])
    _matrix(completed_path, [completed])
    failure_ledger = _write_daily_failure_evidence(
        tmp_path=tmp_path,
        source_matrix=failed_path,
        source_case=failed,
        case_id="native",
    )
    _complete(completed_output, model="historical-sanos")

    payload = compose_production_matrix(
        base_matrix=base_path,
        source_matrices=(failed_path, completed_path),
        replace_models=frozenset({"historical-sanos"}),
        expected_result_count=7,
        expected_results_schema="hedging-daily-results-v2",
        dataset=tmp_path / "dataset",
        output=tmp_path / "composed.json",
        source_failure_ledgers=(failure_ledger,),
        source_failure_ledger_sha256=(_sha256(failure_ledger),),
    )

    assert payload["composition"]["replacement_sources"] == {
        "native": str(completed_path)
    }
    assert payload["composition"]["failure_replacement_sources"] == {}


def test_compose_rejects_missing_complete_source(tmp_path: Path) -> None:
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    source = _case("native", model="historical-sanos", output=tmp_path / "source")
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    _matrix(base_path, [base])
    _matrix(source_path, [source])

    with pytest.raises(ValueError, match="no complete authenticated source"):
        compose_production_matrix(
            base_matrix=base_path,
            source_matrices=(source_path,),
            replace_models=frozenset({"historical-sanos"}),
            expected_result_count=7,
            expected_results_schema="hedging-daily-results-v2",
            dataset=tmp_path / "dataset",
            output=tmp_path / "composed.json",
        )


def test_compose_preserves_authenticated_strict_failure(tmp_path: Path) -> None:
    base_cases = [
        _case("complete", model="historical-sanos", output=tmp_path / "base-complete"),
        _case("failed", model="historical-sanos", output=tmp_path / "base-failed"),
    ]
    source_output = tmp_path / "source-complete"
    failed_output = tmp_path / "source-failed"
    source_cases = [
        _case("complete", model="historical-sanos", output=source_output),
        _case("failed", model="historical-sanos", output=failed_output),
    ]
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    composed_path = tmp_path / "composed.json"
    source_ledger_path = tmp_path / "source-failures.json"
    composed_ledger_path = tmp_path / "composed-failures.json"
    _matrix(base_path, base_cases)
    _matrix(source_path, source_cases)
    _complete(source_output, model="historical-sanos")
    failed_output.mkdir(parents=True)
    labels = tuple(target.label for target in daily_digital_targets())
    failed_spec = DailyEvaluationSpec(
        label="failed-daily",
        model="historical-sanos",
        measure="historical-P",
        model_config_identity="model-config",
        model_config_snapshot_json="{}",
        dataset_manifest_sha256="a" * 64,
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2025, 12, 31),
        rolling_observations=252,
        target_labels=labels,
        target_quantities=tuple(1.0 for _ in labels),
        target_notional_identity="target-notional",
        require_convergence=True,
    )
    _write_json(
        failed_output / "run.json",
        {**failed_spec.to_dict(), "run_identity": failed_spec.run_identity},
    )
    gate_failure_path = write_numerical_gate_failure(
        output_directory=failed_output,
        run_identity=failed_spec.run_identity,
        evaluation_schema=failed_spec.to_dict()["schema"],
        model=failed_spec.model,
        measure=failed_spec.measure,
        dataset_manifest_sha256=failed_spec.dataset_manifest_sha256,
        origin_date=date(2023, 7, 20),
        diagnostics={"cap_hits": 13, "nonconverged_results": 0},
    )
    run_log = failed_output / "run.log"
    run_log.write_text("strict numerical gate failed\n", encoding="utf-8")
    queue_path = tmp_path / "failed-queue.json"
    _write_json(
        queue_path,
        {
            "schema": QUEUE_SCHEMA,
            "matrix_path": str(source_path.resolve()),
            "matrix_sha256": _sha256(source_path),
            "resource": "cpu",
            "case_ids": ["failed"],
            "status": "complete-with-failures",
            "current_case_id": None,
            "started_at": "2026-10-14T07:09:10+00:00",
            "completed_at": "2026-10-14T07:52:57+00:00",
            "records": [
                {
                    "case_id": "failed",
                    "status": "failed",
                    "return_code": 1,
                    "error": "production command returned 1",
                    "command": [
                        "/venv/bin/python",
                        *source_cases[1]["command"][1:],
                        "--progress-every",
                        "2",
                    ],
                    "log_path": str(run_log.resolve()),
                    "numerical_gate_failure_path": str(
                        gate_failure_path.resolve()
                    ),
                    "numerical_gate_failure_sha256": _sha256(gate_failure_path),
                    "started_at": "2026-10-14T07:09:10+00:00",
                    "completed_at": "2026-10-14T07:52:57+00:00",
                }
            ],
        },
    )
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=source_path,
            declarations=(
                {
                    "case_id": "failed",
                    "classification": "policy-fit-cap-hit",
                    "summary": "virtual policy reached its internal cap",
                    "first_failed_origin": "2023-07-20",
                    "queue_state_path": str(queue_path),
                },
            ),
        ),
        output=source_ledger_path,
    )

    payload = compose_production_matrix(
        base_matrix=base_path,
        source_matrices=(source_path,),
        replace_models=frozenset({"historical-sanos"}),
        expected_result_count=7,
        expected_results_schema="hedging-daily-results-v2",
        dataset=tmp_path / "dataset",
        output=composed_path,
        source_failure_ledgers=(source_ledger_path,),
        source_failure_ledger_sha256=(_sha256(source_ledger_path),),
        failure_ledger_output=composed_ledger_path,
    )

    assert payload["composition"]["failure_replacement_sources"] == {
        "failed": str(queue_path)
    }
    failures = load_production_failure_ledger(
        composed_ledger_path,
        expected_ledger_sha256=_sha256(composed_ledger_path),
        matrix_path=composed_path,
        expected_matrix_sha256=_sha256(composed_path),
    )
    assert tuple(failure.case_id for failure in failures) == ("failed",)


def test_compose_rejects_source_identity_change(tmp_path: Path) -> None:
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    changed = _case("native", model="historical-sanos", output=tmp_path / "source")
    changed["risk_parameter"] = 10.0
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    _matrix(base_path, [base])
    _matrix(source_path, [changed])

    with pytest.raises(ValueError, match="source case identity mismatch"):
        compose_production_matrix(
            base_matrix=base_path,
            source_matrices=(source_path,),
            replace_models=frozenset({"historical-sanos"}),
            expected_result_count=7,
            expected_results_schema="hedging-daily-results-v2",
            dataset=tmp_path / "dataset",
            output=tmp_path / "composed.json",
        )


@pytest.mark.parametrize(
    "source_suffix",
    (
        ("--position-cap", "2000"),
        ("--allow-nonconvergence",),
    ),
)
def test_compose_rejects_source_command_change(
    tmp_path: Path,
    source_suffix: tuple[str, ...],
) -> None:
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    source = _case("native", model="historical-sanos", output=tmp_path / "source")
    base["command"].extend(("--position-cap", "1000"))
    source["command"].extend(source_suffix)
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    _matrix(base_path, [base])
    _matrix(source_path, [source])

    with pytest.raises(ValueError, match="source case command mismatch"):
        compose_production_matrix(
            base_matrix=base_path,
            source_matrices=(source_path,),
            replace_models=frozenset({"historical-sanos"}),
            expected_result_count=7,
            expected_results_schema="hedging-daily-results-v2",
            dataset=tmp_path / "dataset",
            output=tmp_path / "composed.json",
        )


def test_compose_rejects_mixed_target_notionals(tmp_path: Path) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    base_cases = [
        _case("first", model="historical-sanos", output=tmp_path / "base-first"),
        _case("second", model="historical-sanos", output=tmp_path / "base-second"),
    ]
    source_cases = [
        _case("first", model="historical-sanos", output=first_output),
        _case("second", model="historical-sanos", output=second_output),
    ]
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    _matrix(base_path, base_cases)
    _matrix(source_path, source_cases)
    _complete(first_output, model="historical-sanos")
    _complete(second_output, model="historical-sanos")
    second_run_path = second_output / "run.json"
    second_run = json.loads(second_run_path.read_text(encoding="utf-8"))
    second_run["target_notional_identity"] = "different-notional-identity"
    _write_json(second_run_path, second_run)

    with pytest.raises(ValueError, match="different target notionals"):
        compose_production_matrix(
            base_matrix=base_path,
            source_matrices=(source_path,),
            replace_models=frozenset({"historical-sanos"}),
            expected_result_count=7,
            expected_results_schema="hedging-daily-results-v2",
            dataset=tmp_path / "dataset",
            output=tmp_path / "composed.json",
        )


def test_compose_enforces_precommitted_numerical_gate(tmp_path: Path) -> None:
    output = tmp_path / "source"
    base = _case("native", model="historical-sanos", output=tmp_path / "base")
    source = _case("native", model="historical-sanos", output=output)
    base_path = tmp_path / "base.json"
    source_path = tmp_path / "source.json"
    _matrix(base_path, [base])
    _matrix(source_path, [source])
    _complete(
        output,
        model="historical-sanos",
        nonconverged_results=1,
        cap_hits=1,
    )

    with pytest.raises(ValueError, match="failed its numerical gates"):
        compose_production_matrix(
            base_matrix=base_path,
            source_matrices=(source_path,),
            replace_models=frozenset({"historical-sanos"}),
            expected_result_count=7,
            expected_results_schema="hedging-daily-results-v2",
            dataset=tmp_path / "dataset",
            output=tmp_path / "composed.json",
        )


def _write_daily_failure_evidence(
    *,
    tmp_path: Path,
    source_matrix: Path,
    source_case: dict[str, object],
    case_id: str,
) -> Path:
    command = source_case["command"]
    assert isinstance(command, list)
    output = Path(command[command.index("--output") + 1])
    labels = tuple(target.label for target in daily_digital_targets())
    spec = DailyEvaluationSpec(
        label=f"{case_id}-daily",
        model=str(source_case["model"]),
        measure=str(source_case["measure"]),
        model_config_identity="model-config",
        model_config_snapshot_json="{}",
        dataset_manifest_sha256="a" * 64,
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2025, 12, 31),
        rolling_observations=252,
        target_labels=labels,
        target_quantities=tuple(1.0 for _ in labels),
        target_notional_identity="target-notional",
        require_convergence=True,
    )
    _write_json(
        output / "run.json",
        {**spec.to_dict(), "run_identity": spec.run_identity},
    )
    gate_failure_path = write_numerical_gate_failure(
        output_directory=output,
        run_identity=spec.run_identity,
        evaluation_schema=str(spec.to_dict()["schema"]),
        model=spec.model,
        measure=spec.measure,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        origin_date=date(2023, 7, 20),
        diagnostics={"cap_hits": 1, "nonconverged_results": 0},
    )
    log_path = output / "run.log"
    log_path.write_text("strict numerical gate failed\n", encoding="utf-8")
    queue_path = tmp_path / f"{case_id}-queue.json"
    _write_json(
        queue_path,
        {
            "schema": QUEUE_SCHEMA,
            "matrix_path": str(source_matrix.resolve()),
            "matrix_sha256": _sha256(source_matrix),
            "resource": "cpu",
            "case_ids": [case_id],
            "status": "complete-with-failures",
            "current_case_id": None,
            "started_at": "2026-10-14T07:09:10+00:00",
            "completed_at": "2026-10-14T07:52:57+00:00",
            "records": [
                {
                    "case_id": case_id,
                    "status": "failed",
                    "return_code": 1,
                    "error": "production command returned 1",
                    "command": [
                        "/venv/bin/python",
                        *command[1:],
                        "--progress-every",
                        "2",
                    ],
                    "log_path": str(log_path.resolve()),
                    "numerical_gate_failure_path": str(
                        gate_failure_path.resolve()
                    ),
                    "numerical_gate_failure_sha256": _sha256(gate_failure_path),
                    "started_at": "2026-10-14T07:09:10+00:00",
                    "completed_at": "2026-10-14T07:52:57+00:00",
                }
            ],
        },
    )
    ledger_path = tmp_path / f"{case_id}-failure-ledger.json"
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=source_matrix,
            declarations=(
                {
                    "case_id": case_id,
                    "queue_state_path": str(queue_path),
                },
            ),
        ),
        output=ledger_path,
    )
    return ledger_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
