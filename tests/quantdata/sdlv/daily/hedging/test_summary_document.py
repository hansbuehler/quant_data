from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

import quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_document as summary_document_module
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_document import (
    _write_rendered_summary_document,
    build_summary_document,
    write_summary_document,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    SUMMARY_SCHEMA as DAILY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    SUMMARY_SCHEMA as WEEKLY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.execute_production_matrix import (
    QUEUE_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    write_numerical_gate_failure,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.production_failures import (
    ProductionFailure,
    build_production_failure_ledger,
    write_production_failure_ledger,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
)


def test_summary_document_reports_full_grid_without_ex_post_selection() -> None:
    document = build_summary_document(
        daily=_daily_metrics(),
        weekly=_weekly_metrics(),
        status_date=date(2026, 10, 14),
        daily_objective_tables="daily-objective-tables.tex",
        weekly_objective_tables="weekly-objective-tables.tex",
        daily_target_appendix="daily-target-appendix.tex",
        weekly_target_appendix="weekly-target-appendix.tex",
    )

    assert "14 October 2026" in document
    assert "no policy is selected" in document
    assert "best-within-model" not in document
    assert "Historical SANOS" in document
    assert "frozen-weekly" in document
    assert "HH 95\\%" in document
    assert "MBB 95\\%" in document
    assert "[-15.00, -5.00]" in document
    assert "pending" not in document.lower()
    assert "daily-objective-tables.tex" in document
    assert "weekly-target-appendix.tex" in document
    assert "genuine risk-neutral expectation" in document
    catalogue = document.split("\\section{Model Catalogue}", maxsplit=1)[1]
    assert catalogue.index("\\subsection{SANOS--LV}") < catalogue.index(
        "\\subsection{Historical SANOS}"
    )


def test_summary_document_rejects_mixed_dataset_manifests() -> None:
    daily = _daily_metrics()
    weekly = _weekly_metrics()
    weekly["dataset_manifest_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="different public datasets"):
        build_summary_document(
            daily=daily,
            weekly=weekly,
            status_date=date(2026, 10, 14),
            daily_objective_tables="daily.tex",
            weekly_objective_tables="weekly.tex",
            daily_target_appendix="daily-target.tex",
            weekly_target_appendix="weekly-target.tex",
        )


def test_summary_document_rejects_unknown_model() -> None:
    daily = _daily_metrics()
    daily_models = daily["models"]
    assert isinstance(daily_models, dict)
    unknown = dict(daily_models["historical-sanos-mean-variance-1"])
    unknown["model"] = "invented-model"
    daily_models["invented"] = unknown

    with pytest.raises(ValueError, match="unknown benchmark model"):
        build_summary_document(
            daily=daily,
            weekly=_weekly_metrics(),
            status_date=date(2026, 10, 14),
            daily_objective_tables="daily.tex",
            weekly_objective_tables="weekly.tex",
            daily_target_appendix="daily-target.tex",
            weekly_target_appendix="weekly-target.tex",
        )


def test_summary_document_writer_rebuilds_from_authenticated_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = dataset / "manifest.json"
    manifest.write_text('{"schema":"test"}\n', encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    daily = _daily_metrics(manifest_sha256=manifest_sha256)
    weekly = _weekly_metrics(manifest_sha256=manifest_sha256)
    daily_matrix = _write_matrix(
        tmp_path / "daily-matrix.json",
        schema=summary_document_module.DAILY_MATRIX_SCHEMA,
        dataset=dataset,
        weekly=False,
    )
    weekly_matrix = _write_matrix(
        tmp_path / "weekly-matrix.json",
        schema=summary_document_module.WEEKLY_MATRIX_SCHEMA,
        dataset=dataset,
        weekly=True,
    )
    weekly_failure_ledger, weekly_failed_case = _write_first_case_failure_ledger(
        weekly_matrix,
        output=tmp_path / "weekly-failures.json",
    )
    bibliography = tmp_path / "source" / "hedging_benchmark_summary_refs.bib"
    bibliography.parent.mkdir()
    bibliography.write_text("% test\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def daily_inputs(
        paths: tuple[Path, ...],
        *,
        require_ready_complete: bool,
        failed_case_ids: frozenset[str],
    ) -> tuple[object, ...]:
        calls["daily_inputs"] = (paths, require_ready_complete, failed_case_ids)
        return (object(),)

    def weekly_inputs(
        paths: tuple[Path, ...],
        *,
        require_ready_complete: bool,
        failed_case_ids: frozenset[str],
    ) -> tuple[object, ...]:
        calls["weekly_inputs"] = (paths, require_ready_complete, failed_case_ids)
        return (object(),)

    monkeypatch.setattr(summary_document_module, "daily_policy_inputs_from_matrices", daily_inputs)
    monkeypatch.setattr(summary_document_module, "weekly_policy_inputs_from_matrices", weekly_inputs)
    monkeypatch.setattr(
        summary_document_module,
        "build_daily_policy_summary_metrics",
        lambda *, policies, dataset: daily,
    )
    monkeypatch.setattr(
        summary_document_module,
        "build_weekly_policy_summary_metrics",
        lambda *, policies, dataset: weekly,
    )
    monkeypatch.setattr(summary_document_module, "write_summary_metrics", _write_daily)
    monkeypatch.setattr(
        summary_document_module,
        "write_weekly_summary_metrics",
        _write_weekly,
    )
    output = tmp_path / "report" / "summary.tex"

    pdf = write_summary_document(
        dataset=dataset,
        daily_matrix=daily_matrix,
        daily_matrix_sha256=_sha256(daily_matrix),
        weekly_matrix=weekly_matrix,
        weekly_matrix_sha256=_sha256(weekly_matrix),
        weekly_failure_ledger=weekly_failure_ledger,
        weekly_failure_ledger_sha256=_sha256(weekly_failure_ledger),
        output=output,
        bibliography_source=bibliography,
        status_date=date(2026, 10, 14),
    )

    assert pdf is None
    assert calls["daily_inputs"] == ((daily_matrix,), True, frozenset())
    assert calls["weekly_inputs"] == (
        (weekly_matrix,),
        True,
        frozenset({weekly_failed_case}),
    )
    assert output.is_file()
    assert (output.parent / "hedging_benchmark_summary_refs.bib").is_file()
    bundle = json.loads(
        (output.parent / "hedging_benchmark_summary_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle["daily_matrix_sha256"] == _sha256(daily_matrix)
    assert bundle["weekly_matrix_sha256"] == _sha256(weekly_matrix)
    assert bundle["weekly_failure_ledger_sha256"] == _sha256(
        weekly_failure_ledger
    )
    assert bundle["weekly_failed_cases"] == [weekly_failed_case]
    assert bundle["daily_source_identities"]["sanos-lv-reference"][
        "aggregation_identity"
    ] == "daily-aggregation"


def test_final_report_rejects_blocked_case_selection(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    matrix = _write_matrix(
        tmp_path / "daily-matrix.json",
        schema=summary_document_module.DAILY_MATRIX_SCHEMA,
        dataset=dataset,
        weekly=False,
    )
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    payload["cases"][0]["readiness"] = "blocked"
    payload["cases"][0]["blocked_reason"] = "post-outcome suppression"
    payload["cases"][0]["command"] = None
    payload["ready_count"] -= 1
    payload["blocked_count"] += 1
    matrix.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not fully ready"):
        summary_document_module._validate_production_matrix(
            matrix,
            expected_sha256=_sha256(matrix),
            expected_schema=summary_document_module.DAILY_MATRIX_SCHEMA,
            expected_cases=summary_document_module.EXPECTED_DAILY_MATRIX_CASES,
            dataset=dataset,
            weekly=False,
        )


def test_final_report_rejects_output_aliasing_authenticated_input(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    daily_matrix = tmp_path / "daily.json"
    weekly_matrix = tmp_path / "weekly.json"
    bibliography = tmp_path / "refs.bib"
    for path in (daily_matrix, weekly_matrix, bibliography):
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="aliases an authenticated input"):
        summary_document_module._validate_output_boundaries(
            dataset=dataset,
            daily_matrix=daily_matrix,
            weekly_matrix=weekly_matrix,
            bibliography_source=bibliography,
            output=daily_matrix,
        )


def test_final_report_rejects_hardlink_to_authenticated_input(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    daily_matrix = tmp_path / "daily.json"
    weekly_matrix = tmp_path / "weekly.json"
    bibliography = tmp_path / "refs.bib"
    for path in (daily_matrix, weekly_matrix, bibliography):
        path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "report" / "summary.tex"
    output.parent.mkdir()
    os.link(daily_matrix, output)

    with pytest.raises(ValueError, match="hardlinks an authenticated input"):
        summary_document_module._validate_output_boundaries(
            dataset=dataset,
            daily_matrix=daily_matrix,
            weekly_matrix=weekly_matrix,
            bibliography_source=bibliography,
            output=output,
        )


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk unavailable")
def test_summary_document_compiles(tmp_path: Path) -> None:
    source_directory = Path(summary_document_module.__file__).resolve().parent
    fragments = tuple(
        tmp_path / name
        for name in (
            "daily-objective-tables.tex",
            "weekly-objective-tables.tex",
            "daily-target-appendix.tex",
            "weekly-target-appendix.tex",
        )
    )
    for fragment in fragments:
        fragment.write_text("% synthetic authenticated fragment\n", encoding="utf-8")
    output = tmp_path / "summary.tex"

    pdf = _write_rendered_summary_document(
        daily=_daily_metrics(),
        weekly=_weekly_metrics(),
        output=output,
        bibliography_source=source_directory / "hedging_benchmark_summary_refs.bib",
        daily_objective_tables=fragments[0],
        weekly_objective_tables=fragments[1],
        daily_target_appendix=fragments[2],
        weekly_target_appendix=fragments[3],
        weekly_failures=(
            ProductionFailure(
                case_id="frozen-weekly-sanos-lv-reference",
                model="sanos-lv",
                measure="Q-price/historical-transition",
                variant="frozen-weekly-reference-delta-eight-factor-vega",
                objective=None,
                risk_parameter=None,
                    schedule="frozen-weekly",
                    classification="position-cap-hit",
                    triggered_gates=("position-cap-hit",),
                    summary="26 executed position-cap hits",
                first_failed_origin=date(2023, 2, 8),
                source_matrix_path=tmp_path / "source-matrix.json",
                source_matrix_sha256="a" * 64,
                queue_state_path=tmp_path / "queue.json",
                queue_state_sha256="b" * 64,
                    queue_log_path=tmp_path / "run.log",
                    queue_log_sha256="c" * 64,
                    run_metadata_path=tmp_path / "run.json",
                    run_metadata_sha256="d" * 64,
                    gate_failure_path=tmp_path / "numerical-gate-failure.json",
                    gate_failure_sha256="e" * 64,
                ),
        ),
        status_date=date(2026, 10, 14),
        compile_pdf=True,
    )

    assert pdf == output.with_suffix(".pdf")
    assert pdf.is_file()


def _write_daily(
    payload: dict[str, object],
    *,
    json_output: Path,
    tex_output: Path,
    objective_tables_output: Path,
    objective_appendix_output: Path,
) -> None:
    json_output.write_text(json.dumps(payload), encoding="utf-8")
    tex_output.write_text("% macros\n", encoding="utf-8")
    objective_tables_output.write_text("% daily tables\n", encoding="utf-8")
    objective_appendix_output.write_text("% daily appendix\n", encoding="utf-8")


def _write_weekly(
    payload: dict[str, object],
    *,
    json_output: Path,
    objective_tables_output: Path,
    target_appendix_output: Path,
) -> None:
    json_output.write_text(json.dumps(payload), encoding="utf-8")
    objective_tables_output.write_text("% weekly tables\n", encoding="utf-8")
    target_appendix_output.write_text("% weekly appendix\n", encoding="utf-8")


def _write_matrix(
    path: Path,
    *,
    schema: str,
    dataset: Path,
    weekly: bool,
) -> Path:
    identities = summary_document_module._canonical_case_identities(
        dataset=dataset,
        weekly=weekly,
    )
    config = (
        summary_document_module.WeeklyProductionMatrixConfig()
        if weekly
        else summary_document_module.DailyProductionMatrixConfig()
    )
    cases = []
    for case_id, identity in identities.items():
        model, measure, variant, objective, parameter = identity
        command = [
            "python",
            "runner",
            "--output",
            str(path.parent / "results" / case_id),
            "--dataset-manifest-sha256",
            _sha256(dataset / "manifest.json"),
        ]
        if weekly:
            schedule = (
                "frozen-weekly"
                if case_id.startswith("frozen-weekly-")
                else "restarted-weekly"
            )
            command.extend(("--schedule", schedule))
        cases.append(
            {
                "case_id": case_id,
                "model": model,
                "measure": measure,
                "variant": variant,
                "objective": objective,
                "risk_parameter": parameter,
                "resource": "cpu",
                "readiness": "ready",
                "blocked_reason": None,
                "command": command,
                "reviewed_commit": None,
                "production_gate_sha256": None,
                "production_gate_path": None,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "code_version": str(type(config).version),
                "config": config.to_dict(),
                "case_count": len(cases),
                "ready_count": len(cases),
                "blocked_count": 0,
                "waiting_gate_count": 0,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_first_case_failure_ledger(
    matrix: Path,
    *,
    output: Path,
) -> tuple[Path, str]:
    matrix_payload = json.loads(matrix.read_text(encoding="utf-8"))
    case = matrix_payload["cases"][0]
    case_id = case["case_id"]
    command = case["command"]
    run_output = Path(command[command.index("--output") + 1])
    manifest_sha256 = command[command.index("--dataset-manifest-sha256") + 1]
    schedule = HedgeSchedule(command[command.index("--schedule") + 1])
    targets = weekly_targets()
    spec = WeeklyEvaluationSpec(
        label=case_id,
        model=case["model"],
        measure=(
            "Q"
            if case["measure"] == "Q-price/historical-transition"
            else case["measure"]
        ),
        model_config_identity="model-config",
        policy_config_identity="policy-config",
        configuration_snapshot_json="{}",
        dataset_manifest_sha256=manifest_sha256,
        schedule=schedule,
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2025, 12, 31),
        rolling_observations=252,
        target_labels=tuple(target.label for target in targets),
        target_quantities=tuple(1.0 for _ in targets),
        target_notional_identity="target-notional",
        position_cap=1_000.0,
        spot_cost_bps=1.0,
        require_no_cap_hits=True,
    )
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "run.json").write_text(
        json.dumps({**spec.to_dict(), "run_identity": spec.run_identity}),
        encoding="utf-8",
    )
    gate_failure_path = write_numerical_gate_failure(
        output_directory=run_output,
        run_identity=spec.run_identity,
        evaluation_schema=str(spec.to_dict()["schema"]),
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
    run_log = output.parent / f"{case_id}.log"
    run_log.write_text("strict numerical gate failed\n", encoding="utf-8")
    queue_state = output.parent / f"{case_id}-queue.json"
    queue_state.write_text(
        json.dumps(
            {
                "schema": QUEUE_SCHEMA,
                "matrix_path": str(matrix.resolve()),
                "matrix_sha256": _sha256(matrix),
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
                            *case["command"][1:],
                            "--progress-every",
                            "2",
                        ],
                        "log_path": str(run_log.resolve()),
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
    write_production_failure_ledger(
        build_production_failure_ledger(
            matrix_path=matrix,
            declarations=(
                {
                    "case_id": case_id,
                    "classification": "position-cap-hit",
                    "summary": "strict numerical gate failed",
                    "queue_state_path": str(queue_state),
                },
            ),
        ),
        output=output,
    )
    return output, case_id


def _daily_metrics(*, manifest_sha256: str = "a" * 64) -> dict[str, object]:
    return {
        "schema": DAILY_SUMMARY_SCHEMA,
        "dataset_manifest_sha256": manifest_sha256,
        "target_notional_identity": "daily-targets",
        "models": {
            "sanos-lv-reference": _daily_model(
                key="sanos-lv-reference",
                model="sanos-lv",
                display_name="SANOS--LV",
                pure_rmse=0.01,
                objective="reference",
                parameter=None,
                specification="Q price / historical path; delta + 8 DLV vegas",
            ),
            "historical-sanos-mean-variance-1": _daily_model(
                key="historical-sanos-mean-variance-1",
                model="historical-sanos",
                display_name="Historical SANOS",
                pure_rmse=0.012,
                objective="mean-variance",
                parameter=1.0,
                specification="historical P; SANOS scenarios",
            ),
        },
    }


def _daily_model(
    *,
    key: str,
    model: str,
    display_name: str,
    pure_rmse: float,
    objective: str,
    parameter: float | None,
    specification: str,
) -> dict[str, object]:
    return {
        "key": key,
        "model": model,
        "display_name": display_name,
        "measure": "test",
        "variant": "test-variant",
        "specification": specification,
        "fitted_objective": objective,
        "fitted_parameter": parameter,
        "run_identity": f"run-{key}",
        "aggregation_identity": "daily-aggregation",
        "result_count": 12_220,
        "metrics": {
            "all": {
                "unhedged_rmse": 0.02,
                "pure_rmse": pure_rmse,
                "pure_rmse_reduction": 1.0 - pure_rmse / 0.02,
                "expected_shortfall_95": 2.0 * pure_rmse,
                "mean_transaction_cost": 0.001,
                "mean_gross_exposure": 1.5,
                "economic_book_maximum_drawdown": 0.03,
            }
        },
    }


def _weekly_metrics(*, manifest_sha256: str = "a" * 64) -> dict[str, object]:
    models = {}
    comparisons = {}
    for schedule in ("frozen-weekly", "restarted-weekly"):
        reference_key = f"{schedule}-sanos-lv-reference"
        candidate_key = f"{schedule}-historical-sanos-mean-variance-1"
        models[reference_key] = _weekly_model(
            key=reference_key,
            model="sanos-lv",
            display_name="SANOS--LV",
            schedule=schedule,
            pure_rmse=0.02,
            objective="reference",
            parameter=None,
            specification="Q price / historical path; delta + 8 DLV vegas",
        )
        models[candidate_key] = _weekly_model(
            key=candidate_key,
            model="historical-sanos",
            display_name="Historical SANOS",
            schedule=schedule,
            pure_rmse=0.018,
            objective="mean-variance",
            parameter=1.0,
            specification="historical P; SANOS scenarios",
        )
        comparisons[schedule] = {
            "reference_case": reference_key,
            "paired_book_loss_differences": {
                reference_key: _paired(0.0),
                candidate_key: _paired(-0.001),
            },
        }
    return {
        "schema": WEEKLY_SUMMARY_SCHEMA,
        "dataset_manifest_sha256": manifest_sha256,
        "target_notional_identity": "weekly-targets",
        "models": models,
        "comparisons": comparisons,
    }


def _weekly_model(
    *,
    key: str,
    model: str,
    display_name: str,
    schedule: str,
    pure_rmse: float,
    objective: str,
    parameter: float | None,
    specification: str,
) -> dict[str, object]:
    return {
        "key": key,
        "model": model,
        "display_name": display_name,
        "measure": "test",
        "variant": "test-variant",
        "specification": specification,
        "schedule": schedule,
        "fitted_objective": objective,
        "fitted_parameter": parameter,
        "run_identity": f"run-{key}",
        "aggregation_identity": f"aggregation-{key}",
        "result_count": 6_864,
        "metrics": {
            "all": {
                "pure_rmse_reduction": 1.0 - pure_rmse / 0.03,
                "mean_transaction_cost": 0.002,
                "pure_hedge_loss": {"root_mean_square": pure_rmse},
            }
        },
    }


def _paired(mean: float) -> dict[str, object]:
    return {
        name: {
            "mean": mean,
            "hac_confidence_interval_95": [mean - 0.0005, mean + 0.0005],
            "bootstrap_confidence_interval_95": [mean - 0.0006, mean + 0.0006],
            "observations": 528,
            "hac_lags": 4,
            "bootstrap_block_length": 5,
        }
        for name in ("pure_hedge_loss", "economic_loss")
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
