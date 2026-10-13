from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.reports.build_summary_metrics import (
    SUMMARY_SCHEMA,
    build_summary_metrics,
    write_summary_metrics,
)


def test_summary_metrics_pool_complete_claim_population(tmp_path: Path) -> None:
    sanos = tmp_path / "sanos" / "results.json"
    historical = tmp_path / "historical" / "results.json"
    _write_results(
        sanos,
        model="sanos-lv",
        manifest="a" * 64,
        one_day_pure_rmse=1.0,
        three_day_pure_rmse=2.0,
    )
    _write_results(
        historical,
        model="historical-sanos",
        manifest="a" * 64,
        one_day_pure_rmse=1.5,
        three_day_pure_rmse=3.0,
    )

    payload = build_summary_metrics(
        sanos_results=sanos,
        historical_results=historical,
    )

    assert payload["schema"] == SUMMARY_SCHEMA
    assert payload["target_notional_identity"] == "target-notional-test"
    models = payload["models"]
    assert isinstance(models, dict)
    sanos_all = models["sanos-lv"]["metrics"]["all"]
    assert sanos_all["pure_rmse"] == pytest.approx(math.sqrt(2.5))
    assert sanos_all["unhedged_rmse"] == pytest.approx(math.sqrt(10.0))
    assert sanos_all["pure_rmse_reduction"] == pytest.approx(0.5)
    assert sanos_all["economic_book_maximum_drawdown"] == pytest.approx(
        611 * 1.6
    )
    comparison = payload["comparisons"]
    assert isinstance(comparison, dict)
    assert comparison["sanos_pure_rmse_reduction_vs_historical"] == pytest.approx(
        1.0 / 3.0
    )

    json_output = tmp_path / "metrics.json"
    tex_output = tmp_path / "metrics.tex"
    objective_tables_output = tmp_path / "objective_tables.tex"
    objective_appendix_output = tmp_path / "objective_appendix.tex"
    write_summary_metrics(
        payload,
        json_output=json_output,
        tex_output=tex_output,
        objective_tables_output=objective_tables_output,
        objective_appendix_output=objective_appendix_output,
    )
    assert json.loads(json_output.read_text(encoding="utf-8"))["schema"] == (
        SUMMARY_SCHEMA
    )
    tex = tex_output.read_text(encoding="utf-8")
    assert r"\newcommand{\SanosVsUnhedgedReduction}{50.0\%}" in tex
    assert r"\newcommand{\SanosVsHistoricalReduction}{33.3\%}" in tex
    objective_tables = objective_tables_output.read_text(encoding="utf-8")
    assert "Mean--variance: full out-of-sample economic-risk score" in objective_tables
    assert "Historical SANOS & MVol, $\\kappa=1$" in objective_tables
    objective_appendix = objective_appendix_output.read_text(encoding="utf-8")
    assert "Target-level mean--volatility scores and drawdowns" in objective_appendix
    assert "Economic MDD" in objective_appendix


def test_summary_metrics_reject_different_dataset_manifests(tmp_path: Path) -> None:
    sanos = tmp_path / "sanos" / "results.json"
    historical = tmp_path / "historical" / "results.json"
    _write_results(
        sanos,
        model="sanos-lv",
        manifest="a" * 64,
        one_day_pure_rmse=1.0,
        three_day_pure_rmse=2.0,
    )
    _write_results(
        historical,
        model="historical-sanos",
        manifest="b" * 64,
        one_day_pure_rmse=1.5,
        three_day_pure_rmse=3.0,
    )

    with pytest.raises(ValueError, match="different dataset manifests"):
        build_summary_metrics(
            sanos_results=sanos,
            historical_results=historical,
        )


def test_summary_metrics_reject_different_target_notionals(tmp_path: Path) -> None:
    sanos = tmp_path / "sanos" / "results.json"
    historical = tmp_path / "historical" / "results.json"
    _write_results(
        sanos,
        model="sanos-lv",
        manifest="a" * 64,
        one_day_pure_rmse=1.0,
        three_day_pure_rmse=2.0,
        target_notional_identity="target-notional-a",
    )
    _write_results(
        historical,
        model="historical-sanos",
        manifest="a" * 64,
        one_day_pure_rmse=1.5,
        three_day_pure_rmse=3.0,
        target_notional_identity="target-notional-b",
    )

    with pytest.raises(ValueError, match="different target notionals"):
        build_summary_metrics(
            sanos_results=sanos,
            historical_results=historical,
        )


def _write_results(
    path: Path,
    *,
    model: str,
    manifest: str,
    one_day_pure_rmse: float,
    three_day_pure_rmse: float,
    target_notional_identity: str = "target-notional-test",
) -> None:
    path.parent.mkdir(parents=True)
    slices = []
    claims = []
    for maturity, unhedged, pure in (
        ("1d", 2.0, one_day_pure_rmse),
        ("3d", 4.0, three_day_pure_rmse),
    ):
        for index in range(10):
            claim = f"digital-put-{maturity}-nms-{index / 10:.2f}"
            claims.append((claim, unhedged, pure))
            slices.append(
                {
                    "claim": claim,
                    "slice_label": "overall",
                    "year": None,
                    "volatility_regime": None,
                    "unhedged_loss": _loss(unhedged),
                    "pure_hedge_loss": _loss(pure),
                    "economic_loss": _loss(pure + 0.1),
                    "mean_transaction_cost": 0.01,
                    "mean_entry_turnover_notional": 2.0,
                    "mean_gross_exposure": 2.0,
                    "mean_nonzero_actions": 3.0,
                    "cap_hits": 0,
                    "nonconverged_results": 0,
                }
            )
    run_identity = ("1" if model == "sanos-lv" else "2") * 64
    path.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-results-v2",
                "dataset_manifest_sha256": manifest,
                "model": model,
                "measure": "test",
                "run_identity": run_identity,
                "result_count": 12_220,
                "slices": slices,
            }
        ),
        encoding="utf-8",
    )
    run = {
        "run_identity": run_identity,
        "target_notional_identity": target_notional_identity,
        "model_config_snapshot": {
            "fields": {
                "objective": "mean-volatility",
                "risk_parameter": 1.0,
            }
        },
    }
    (path.parent / "run.json").write_text(json.dumps(run), encoding="utf-8")
    records = path.parent / "records"
    records.mkdir()
    for observation in range(611):
        origin = f"origin-{observation:04d}"
        results = [
            {
                "origin_date": origin,
                "claim": claim,
                "unhedged_pnl": -unhedged,
                "hedged_pnl": -(pure + 0.1),
                "pure_hedge_error": -pure,
                "pricing_difference": -0.1,
            }
            for claim, unhedged, pure in claims
        ]
        (records / f"{origin}.json").write_text(
            json.dumps({"origin_date": origin, "results": results}),
            encoding="utf-8",
        )


def _loss(root_mean_square: float) -> dict[str, float | int]:
    return {
        "observations": 611,
        "root_mean_square": root_mean_square,
        "expected_shortfall_95": 2.0 * root_mean_square,
        "expected_shortfall_99": 2.5 * root_mean_square,
        "mean_absolute_loss": 0.5 * root_mean_square,
        "mean_variance": {"0.1": root_mean_square, "1": root_mean_square, "10": root_mean_square},
        "mean_volatility": {
            "0.1": root_mean_square,
            "1": root_mean_square,
            "2": root_mean_square,
            "3": root_mean_square,
        },
        "entropy": {"0.1": root_mean_square, "1": root_mean_square, "10": root_mean_square},
        "power_exponential": {
            "0.1": root_mean_square,
            "1": root_mean_square,
            "10": root_mean_square,
        },
    }
