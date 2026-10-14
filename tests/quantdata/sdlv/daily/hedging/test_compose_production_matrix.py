from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.runners.compose_production_matrix import (
    compose_production_matrix,
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
        "command": ["python", "runner", "--output", str(output)],
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


def _complete(output: Path, *, model: str, count: int = 7) -> None:
    run = {
        "run_identity": "run-identity",
        "dataset_manifest_sha256": "dataset-identity",
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
    _write_json(output / "results.json", results)


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
        output=tmp_path / "composed.json",
    )

    assert payload["composition"]["replacement_sources"] == {
        "native": str(second_path)
    }


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
            output=tmp_path / "composed.json",
        )


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
            output=tmp_path / "composed.json",
        )
