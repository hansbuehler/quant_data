from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.runners import (
    execute_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.execute_production_matrix import (
    QUEUE_SCHEMA,
    WEEKLY_MATRIX_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    write_numerical_gate_failure,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_failures import (
    FAILURE_LEDGER_SCHEMA,
    build_production_failure_ledger,
    load_production_failure_ledger,
    reject_completed_failure_cases,
    write_production_failure_ledger,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
)


def test_failure_ledger_authenticates_matrix_queue_and_log(tmp_path: Path) -> None:
    source_matrix, final_matrix, queue_state, queue_log = _failure_artifacts(tmp_path)
    ledger_path = tmp_path / "failure-ledger.json"
    payload = build_production_failure_ledger(
        matrix_path=final_matrix,
        declarations=(
            {
                "case_id": "frozen-weekly-sanos-lv-reference",
                "classification": "position-cap-hit",
                "summary": "26 executed position-cap hits",
                "first_failed_origin": "2023-02-08",
                "queue_state_path": str(queue_state),
            },
        ),
    )
    write_production_failure_ledger(payload, output=ledger_path)

    failures = load_production_failure_ledger(
        ledger_path,
        expected_ledger_sha256=_sha256(ledger_path),
        matrix_path=final_matrix,
        expected_matrix_sha256=_sha256(final_matrix),
    )

    assert payload["schema"] == FAILURE_LEDGER_SCHEMA
    assert payload["failure_count"] == 1
    assert len(failures) == 1
    assert failures[0].case_id == "frozen-weekly-sanos-lv-reference"
    assert failures[0].classification == "position-cap-hit"
    assert failures[0].first_failed_origin is not None
    assert failures[0].first_failed_origin.isoformat() == "2023-02-08"
    assert failures[0].source_matrix_path == source_matrix
    assert failures[0].queue_log_path == queue_log


def test_failure_ledger_rejects_changed_log(tmp_path: Path) -> None:
    _, final_matrix, queue_state, queue_log = _failure_artifacts(tmp_path)
    ledger_path = tmp_path / "failure-ledger.json"
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "classification": "position-cap-hit",
                    "summary": "26 executed position-cap hits",
                    "first_failed_origin": "2023-02-08",
                    "queue_state_path": str(queue_state),
                },
            ),
        ),
        output=ledger_path,
    )
    ledger_sha256 = _sha256(ledger_path)
    queue_log.write_text("changed after authentication\n", encoding="utf-8")

    with pytest.raises(ValueError, match="queue log.*SHA-256 mismatch"):
        load_production_failure_ledger(
            ledger_path,
            expected_ledger_sha256=ledger_sha256,
            matrix_path=final_matrix,
            expected_matrix_sha256=_sha256(final_matrix),
        )


def test_failure_ledger_rejects_semantically_different_source_command(
    tmp_path: Path,
) -> None:
    source_matrix, final_matrix, queue_state, _ = _failure_artifacts(tmp_path)
    source = json.loads(source_matrix.read_text(encoding="utf-8"))
    command = source["cases"][0]["command"]
    command[command.index("--position-cap") + 1] = "2000"
    source_matrix.write_text(json.dumps(source), encoding="utf-8")
    queue = json.loads(queue_state.read_text(encoding="utf-8"))
    queue["matrix_sha256"] = _sha256(source_matrix)
    queue_state.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(ValueError, match="source command differs"):
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "classification": "position-cap-hit",
                    "summary": "strict failure",
                    "queue_state_path": str(queue_state),
                },
            ),
        )


def test_failure_ledger_rejects_disabled_source_gate(tmp_path: Path) -> None:
    source_matrix, final_matrix, queue_state, _ = _failure_artifacts(tmp_path)
    source = json.loads(source_matrix.read_text(encoding="utf-8"))
    source["cases"][0]["command"].append("--allow-cap-hits")
    source_matrix.write_text(json.dumps(source), encoding="utf-8")
    queue = json.loads(queue_state.read_text(encoding="utf-8"))
    queue["matrix_sha256"] = _sha256(source_matrix)
    queue_state.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(ValueError, match="source command differs"):
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "queue_state_path": str(queue_state),
                },
            ),
        )


def test_failure_ledger_requires_terminal_queue(tmp_path: Path) -> None:
    _, final_matrix, queue_state, _ = _failure_artifacts(tmp_path)
    queue = json.loads(queue_state.read_text(encoding="utf-8"))
    queue["status"] = "failed"
    queue["completed_at"] = None
    queue_state.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(ValueError, match="queue is not terminal"):
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "queue_state_path": str(queue_state),
                },
            ),
        )


def test_failure_ledger_rejects_forged_gate_artifact(tmp_path: Path) -> None:
    _, final_matrix, queue_state, queue_log = _failure_artifacts(tmp_path)
    gate_path = queue_log.parent / "numerical-gate-failure.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["summary"] = "manually declared failure"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="queue-pinned numerical-gate failure.*SHA-256"):
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "queue_state_path": str(queue_state),
                },
            ),
        )


def test_failure_ledger_cannot_suppress_later_valid_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, final_matrix, queue_state, _ = _failure_artifacts(tmp_path)
    ledger_path = tmp_path / "failure-ledger.json"
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "classification": "position-cap-hit",
                    "summary": "strict numerical failure",
                    "queue_state_path": str(queue_state),
                },
            ),
        ),
        output=ledger_path,
    )
    failures = load_production_failure_ledger(
        ledger_path,
        expected_ledger_sha256=_sha256(ledger_path),
        matrix_path=final_matrix,
        expected_matrix_sha256=_sha256(final_matrix),
    )
    final_output = tmp_path / "final-output"
    final_output.mkdir(parents=True)
    (final_output / "results.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        execute_production_matrix,
        "_validate_results",
        lambda *_args, **_kwargs: "valid-results-sha256",
    )

    with pytest.raises(ValueError, match="suppresses a valid completed case"):
        reject_completed_failure_cases(
            matrix_path=final_matrix,
            failures=failures,
            dataset=tmp_path / "dataset",
        )


def test_failure_ledger_accepts_present_but_rejected_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, final_matrix, queue_state, _ = _failure_artifacts(tmp_path)
    ledger_path = tmp_path / "failure-ledger.json"
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=final_matrix,
            declarations=(
                {
                    "case_id": "frozen-weekly-sanos-lv-reference",
                    "classification": "position-cap-hit",
                    "summary": "strict numerical failure",
                    "queue_state_path": str(queue_state),
                },
            ),
        ),
        output=ledger_path,
    )
    failures = load_production_failure_ledger(
        ledger_path,
        expected_ledger_sha256=_sha256(ledger_path),
        matrix_path=final_matrix,
        expected_matrix_sha256=_sha256(final_matrix),
    )
    final_output = tmp_path / "final-output"
    final_output.mkdir(parents=True)
    (final_output / "results.json").write_text("{}\n", encoding="utf-8")

    def reject(*_args: object, **_kwargs: object) -> str:
        raise ValueError("position-cap hits")

    monkeypatch.setattr(execute_production_matrix, "_validate_results", reject)

    reject_completed_failure_cases(
        matrix_path=final_matrix,
        failures=failures,
        dataset=tmp_path / "dataset",
    )


def _failure_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    case_id = "frozen-weekly-sanos-lv-reference"
    source_output = tmp_path / "source-output"
    final_output = tmp_path / "final-output"
    source_case = _case(case_id, output=source_output, checkpoint=tmp_path / "source-ckpt")
    final_case = _case(case_id, output=final_output, checkpoint=tmp_path / "final-ckpt")
    source_matrix = tmp_path / "source-matrix.json"
    final_matrix = tmp_path / "final-matrix.json"
    _write_matrix(source_matrix, source_case)
    _write_matrix(final_matrix, final_case)
    queue_log = source_output / "run.log"
    queue_log.parent.mkdir(parents=True)
    queue_log.write_text(
        "RuntimeError: weekly numerical gate found 26 position-cap hits\n",
        encoding="utf-8",
    )
    spec = WeeklyEvaluationSpec(
        label="sanos-lv-frozen",
        model="sanos-lv",
        measure="Q-price/historical-transition",
        model_config_identity="model-config",
        policy_config_identity="policy-config",
        configuration_snapshot_json="{}",
        dataset_manifest_sha256="a" * 64,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2025, 12, 31),
        rolling_observations=252,
        target_labels=tuple(target.label for target in weekly_targets()),
        target_quantities=tuple(1.0 for _ in weekly_targets()),
        target_notional_identity="target-notional",
        position_cap=1_000.0,
        spot_cost_bps=1.0,
        require_no_cap_hits=True,
    )
    (source_output / "run.json").write_text(
        json.dumps({**spec.to_dict(), "run_identity": spec.run_identity}),
        encoding="utf-8",
    )
    gate_failure_path = write_numerical_gate_failure(
        output_directory=source_output,
        run_identity=spec.run_identity,
        evaluation_schema=spec.to_dict()["schema"],
        model=spec.model,
        measure=spec.measure,
        schedule=spec.schedule.value,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        origin_date=date(2023, 2, 8),
        diagnostics={
            "fit_cap_hits": 0,
            "executed_cap_hits": 26,
            "nonconverged_fits": 0,
            "cap_hits": 26,
        },
    )
    executed_command = ["/venv/bin/python", *source_case["command"][1:]]
    executed_command.extend(("--progress-every", "2"))
    queue_state = tmp_path / "queue.json"
    queue_state.write_text(
        json.dumps(
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
                        "command": executed_command,
                        "log_path": str(queue_log.resolve()),
                        "numerical_gate_failure_path": str(
                            gate_failure_path.resolve()
                        ),
                        "numerical_gate_failure_sha256": _sha256(
                            gate_failure_path
                        ),
                        "started_at": "2026-10-14T07:09:10+00:00",
                        "completed_at": "2026-10-14T07:52:57+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source_matrix, final_matrix, queue_state, queue_log


def _case(
    case_id: str,
    *,
    output: Path,
    checkpoint: Path,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "model": "sanos-lv",
        "measure": "Q-price/historical-transition",
        "variant": "frozen-weekly-reference-delta-eight-factor-vega",
        "objective": None,
        "risk_parameter": None,
        "resource": "cpu",
        "readiness": "ready",
        "command": [
            "python",
            "-m",
            "runner",
            "--schedule",
            "frozen-weekly",
            "--output",
            str(output),
            "--checkpoint-directory",
            str(checkpoint),
            "--position-cap",
            "1000",
            "--dataset-manifest-sha256",
            "a" * 64,
        ],
    }


def _write_matrix(path: Path, case: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"schema": WEEKLY_MATRIX_SCHEMA, "cases": [case]}),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
