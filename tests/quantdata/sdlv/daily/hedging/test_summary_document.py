from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_document import (
    build_summary_document,
    write_summary_document,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    SUMMARY_SCHEMA as DAILY_SUMMARY_SCHEMA,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    SUMMARY_SCHEMA as WEEKLY_SUMMARY_SCHEMA,
)


def test_summary_document_uses_authenticated_metrics_without_interim_status() -> None:
    daily = _daily_metrics()
    weekly = _weekly_metrics()

    document = build_summary_document(
        daily=daily,
        weekly=weekly,
        status_date=date(2026, 10, 14),
        daily_objective_tables="daily-objective-tables.tex",
        weekly_objective_tables="weekly-objective-tables.tex",
        daily_target_appendix="daily-target-appendix.tex",
        weekly_target_appendix="weekly-target-appendix.tex",
    )

    assert "14 October 2026" in document
    assert "largest descriptive pure-RMSE reduction" in document
    assert "50.00\\%" in document
    assert "Historical SANOS" in document
    assert "frozen-weekly" in document
    assert "HH 95\\% CI" in document
    assert "pending" not in document.lower()
    assert "daily-objective-tables.tex" in document
    assert "weekly-target-appendix.tex" in document


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


def test_summary_document_writer_copies_local_inputs(tmp_path: Path) -> None:
    daily_path = tmp_path / "source" / "daily.json"
    weekly_path = tmp_path / "source" / "weekly.json"
    daily_path.parent.mkdir()
    daily_path.write_text(json.dumps(_daily_metrics()), encoding="utf-8")
    weekly_path.write_text(json.dumps(_weekly_metrics()), encoding="utf-8")
    sources = {
        name: tmp_path / "source" / name
        for name in (
            "daily-objective-tables.tex",
            "weekly-objective-tables.tex",
            "daily-target-appendix.tex",
            "weekly-target-appendix.tex",
            "hedging_benchmark_summary_refs.bib",
        )
    }
    for path in sources.values():
        path.write_text("% test\n", encoding="utf-8")
    output = tmp_path / "report" / "summary.tex"

    pdf = write_summary_document(
        daily_metrics=daily_path,
        weekly_metrics=weekly_path,
        output=output,
        bibliography_source=sources["hedging_benchmark_summary_refs.bib"],
        daily_objective_tables=sources["daily-objective-tables.tex"],
        weekly_objective_tables=sources["weekly-objective-tables.tex"],
        daily_target_appendix=sources["daily-target-appendix.tex"],
        weekly_target_appendix=sources["weekly-target-appendix.tex"],
        status_date=date(2026, 10, 14),
    )

    assert pdf is None
    assert output.is_file()
    assert (output.parent / "hedging_benchmark_summary_refs.bib").is_file()
    assert (output.parent / "weekly-target-appendix.tex").is_file()


def _daily_metrics() -> dict[str, object]:
    return {
        "schema": DAILY_SUMMARY_SCHEMA,
        "dataset_manifest_sha256": "a" * 64,
        "target_notional_identity": "daily-targets",
        "models": {
            "sanos-lv-reference": _daily_model(
                key="sanos-lv-reference",
                model="sanos-lv",
                display_name="SANOS--LV",
                pure_rmse=0.01,
                objective="reference",
                parameter=None,
            ),
            "historical-sanos-mean-variance-1": _daily_model(
                key="historical-sanos-mean-variance-1",
                model="historical-sanos",
                display_name="Historical SANOS",
                pure_rmse=0.012,
                objective="mean-variance",
                parameter=1.0,
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
) -> dict[str, object]:
    return {
        "key": key,
        "model": model,
        "display_name": display_name,
        "fitted_objective": objective,
        "fitted_parameter": parameter,
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


def _weekly_metrics() -> dict[str, object]:
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
        )
        models[candidate_key] = _weekly_model(
            key=candidate_key,
            model="historical-sanos",
            display_name="Historical SANOS",
            schedule=schedule,
            pure_rmse=0.018,
            objective="mean-variance",
            parameter=1.0,
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
        "dataset_manifest_sha256": "a" * 64,
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
) -> dict[str, object]:
    return {
        "key": key,
        "model": model,
        "display_name": display_name,
        "schedule": schedule,
        "fitted_objective": objective,
        "fitted_parameter": parameter,
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
        "economic_loss": {
            "mean": mean,
            "hac_confidence_interval_95": [mean - 0.0005, mean + 0.0005],
        }
    }
