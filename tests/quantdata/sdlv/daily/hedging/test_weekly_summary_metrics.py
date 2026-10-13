from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics as weekly_summary_module
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.reports.build_weekly_summary_metrics import (
    SUMMARY_SCHEMA,
    build_weekly_policy_summary_metrics,
    weekly_policy_inputs_from_matrices,
    write_weekly_summary_metrics,
)
from quantdata.sdlv.daily.front_spot.hedging.reports.policy_labels import (
    canonical_policy_variant,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    WeeklyLossStatistics,
    WeeklyResultSlice,
    WeeklyRunResults,
)


@pytest.fixture(autouse=True)
def _synthetic_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weekly_summary_module,
        "default_public_candidate_path",
        lambda: Path("/synthetic-public-dataset"),
    )
    monkeypatch.setattr(
        weekly_summary_module,
        "PublicHedgingPanel",
        lambda _path, validate: object(),
    )


def test_weekly_summary_discovers_schedules_and_aggregates_complete_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definitions = (
        (
            "frozen-weekly-sanos-lv-reference",
            "sanos-lv",
            HedgeSchedule.FROZEN_WEEKLY,
            None,
            None,
            1.0,
        ),
        (
            "frozen-weekly-historical-sanos-mean-volatility-1",
            "historical-sanos",
            HedgeSchedule.FROZEN_WEEKLY,
            "mean-volatility",
            1.0,
            1.5,
        ),
        (
            "restarted-weekly-sanos-lv-reference",
            "sanos-lv",
            HedgeSchedule.RESTARTED_WEEKLY,
            None,
            None,
            1.2,
        ),
    )
    matrix_cases = []
    authenticated: dict[Path, WeeklyRunResults] = {}
    for case_id, model, schedule, objective, parameter, pure_rmse in definitions:
        output = tmp_path / case_id
        results_path = output / "results.json"
        spec = _write_run(
            output,
            model=model,
            schedule=schedule,
            objective=objective,
            parameter=parameter,
        )
        results_path.write_text("{}", encoding="utf-8")
        authenticated[results_path] = _weekly_result(
            spec=spec,
            pure_rmse=pure_rmse,
        )
        matrix_cases.append(
            {
                "case_id": case_id,
                "model": model,
                "measure": "test",
                "variant": canonical_policy_variant(
                    case_id=case_id,
                    model=model,
                    schedule=schedule.value,
                ),
                "objective": objective,
                "risk_parameter": parameter,
                "readiness": "ready",
                "command": [
                    "python",
                    "runner",
                    "--schedule",
                    schedule.value,
                    "--output",
                    str(output),
                ],
            }
        )
    monkeypatch.setattr(
        weekly_summary_module,
        "authenticate_completed_weekly_results",
        lambda _panel, *, results_path: authenticated[results_path],
    )
    monkeypatch.setattr(
        weekly_summary_module,
        "_read_record_losses",
        lambda _records, *, expected_result: _record_losses(expected_result),
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": matrix_cases,
            }
        ),
        encoding="utf-8",
    )

    policies = weekly_policy_inputs_from_matrices(
        (matrix,),
        require_ready_complete=True,
    )
    payload = build_weekly_policy_summary_metrics(policies=policies)

    assert payload["schema"] == SUMMARY_SCHEMA
    assert tuple(payload["models"]) == tuple(item[0] for item in definitions)
    frozen_historical = payload["models"][definitions[1][0]]
    assert frozen_historical["metrics"]["all"]["pure_hedge_loss"][
        "root_mean_square"
    ] == pytest.approx(1.5)
    assert frozen_historical["metrics"]["all"]["pure_rmse_reduction"] == (
        pytest.approx(0.25)
    )
    comparison = payload["comparisons"][HedgeSchedule.FROZEN_WEEKLY.value]
    assert comparison["pure_rmse_reduction_vs_reference"][definitions[1][0]] == (
        pytest.approx(-0.5)
    )
    paired = comparison["paired_book_loss_differences"][definitions[1][0]]
    assert paired["economic_loss"]["mean"] == pytest.approx(0.5)
    assert paired["economic_loss"]["observations"] == 528
    assert paired["economic_loss"]["bootstrap_block_length"] == 50
    assert frozen_historical["specification"] == "test; SANOS scenarios"
    assert len(payload["objective_tables"]["mean_variance"]["rows"]) == 3

    json_output = tmp_path / "weekly.json"
    tables_output = tmp_path / "weekly_tables.tex"
    appendix_output = tmp_path / "weekly_appendix.tex"
    write_weekly_summary_metrics(
        payload,
        json_output=json_output,
        objective_tables_output=tables_output,
        target_appendix_output=appendix_output,
    )
    assert json.loads(json_output.read_text(encoding="utf-8"))["schema"] == (
        SUMMARY_SCHEMA
    )
    assert "frozen-weekly" in tables_output.read_text(encoding="utf-8")
    assert "digital" in appendix_output.read_text(encoding="utf-8")


def test_weekly_summary_rejects_matrix_objective_that_disagrees_with_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "historical"
    results_path = output / "results.json"
    spec = _write_run(
        output,
        model="historical-sanos",
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        objective="mean-volatility",
        parameter=1.0,
    )
    results_path.write_text("{}", encoding="utf-8")
    result = _weekly_result(spec=spec, pure_rmse=1.0)
    monkeypatch.setattr(
        weekly_summary_module,
        "authenticate_completed_weekly_results",
        lambda _panel, *, results_path: result,
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "forged-objective",
                        "model": "historical-sanos",
                        "measure": "test",
                        "variant": (
                            "frozen-weekly-rolling-fixed-contract-scenarios"
                        ),
                        "objective": "entropy",
                        "risk_parameter": 10.0,
                        "readiness": "ready",
                        "command": [
                            "python",
                            "runner",
                            "--schedule",
                            HedgeSchedule.FROZEN_WEEKLY.value,
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    policies = weekly_policy_inputs_from_matrices(
        (matrix,),
        require_ready_complete=True,
    )
    with pytest.raises(ValueError, match="matrix objective does not match"):
        build_weekly_policy_summary_metrics(policies=policies)


def test_weekly_summary_rejects_matrix_measure_that_disagrees_with_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "historical"
    results_path = output / "results.json"
    spec = _write_run(
        output,
        model="historical-sanos",
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        objective="mean-volatility",
        parameter=1.0,
    )
    results_path.write_text("{}", encoding="utf-8")
    result = _weekly_result(spec=spec, pure_rmse=1.0)
    monkeypatch.setattr(
        weekly_summary_module,
        "authenticate_completed_weekly_results",
        lambda _panel, *, results_path: result,
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "forged-measure",
                        "model": "historical-sanos",
                        "measure": "Q",
                        "variant": (
                            "frozen-weekly-rolling-fixed-contract-scenarios"
                        ),
                        "objective": "mean-volatility",
                        "risk_parameter": 1.0,
                        "readiness": "ready",
                        "command": [
                            "python",
                            "runner",
                            "--schedule",
                            HedgeSchedule.FROZEN_WEEKLY.value,
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    policies = weekly_policy_inputs_from_matrices(
        (matrix,),
        require_ready_complete=True,
    )
    with pytest.raises(ValueError, match="measure does not match"):
        build_weekly_policy_summary_metrics(policies=policies)


def test_weekly_summary_rejects_forged_matrix_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "historical"
    results_path = output / "results.json"
    spec = _write_run(
        output,
        model="historical-sanos",
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        objective="mean-volatility",
        parameter=1.0,
    )
    results_path.write_text("{}", encoding="utf-8")
    result = _weekly_result(spec=spec, pure_rmse=1.0)
    monkeypatch.setattr(
        weekly_summary_module,
        "authenticate_completed_weekly_results",
        lambda _panel, *, results_path: result,
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "forged-variant",
                        "model": "historical-sanos",
                        "measure": "test",
                        "variant": (
                            "frozen-weekly-published-settings-full-rolling-fit"
                        ),
                        "objective": "mean-volatility",
                        "risk_parameter": 1.0,
                        "readiness": "ready",
                        "command": [
                            "python",
                            "runner",
                            "--schedule",
                            HedgeSchedule.FROZEN_WEEKLY.value,
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    policies = weekly_policy_inputs_from_matrices(
        (matrix,),
        require_ready_complete=True,
    )
    with pytest.raises(ValueError, match="variant does not match"):
        build_weekly_policy_summary_metrics(policies=policies)


def test_weekly_summary_rejects_incomplete_ready_case(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "missing",
                        "model": "sanos-lv",
                        "measure": "test",
                        "variant": (
                            "frozen-weekly-reference-delta-eight-factor-vega"
                        ),
                        "objective": None,
                        "risk_parameter": None,
                        "readiness": "ready",
                        "command": [
                            "python",
                            "runner",
                            "--schedule",
                            HedgeSchedule.FROZEN_WEEKLY.value,
                            "--output",
                            str(tmp_path / "missing"),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ready weekly production case is incomplete"):
        weekly_policy_inputs_from_matrices(
            (matrix,),
            require_ready_complete=True,
        )


def test_weekly_record_loss_reader_rechecks_authenticated_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weekly_summary_module, "EXPECTED_ORIGINS", 2)
    records = tmp_path / "records"
    records.mkdir()
    run_identity = "a" * 64
    hashes: dict[str, str] = {}
    for origin in ("2023-01-03", "2023-01-04"):
        payload = {
            "run_identity": run_identity,
            "origin_date": origin,
            "runs": [
                {
                    "result": {
                        "origin_date": origin,
                        "target_label": target.label,
                        "breakdown": {
                            "claim_quantity": 2.0,
                            "market_price": 0.4,
                            "payoff": 0.1,
                            "economic_pnl": -0.2,
                            "pure_hedge_error": -0.15,
                            "pricing_contribution": -0.05,
                        },
                    }
                }
                for target in weekly_targets()
            ],
        }
        path = records / f"{origin}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        hashes[origin] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = SimpleNamespace(
        record_sha256_by_origin=hashes,
        run_identity=run_identity,
        result_count=2 * len(weekly_targets()),
    )

    losses = weekly_summary_module._read_record_losses(
        records,
        expected_result=expected,
    )
    claim = weekly_targets()[0].label
    assert losses[claim][0]["unhedged_loss"] == pytest.approx(0.6)
    assert losses[claim][0]["economic_loss"] == pytest.approx(0.2)

    first_path = records / "2023-01-03.json"
    first_path.write_text(first_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after authentication"):
        weekly_summary_module._read_record_losses(
            records,
            expected_result=expected,
        )


def _write_run(
    output: Path,
    *,
    model: str,
    schedule: HedgeSchedule,
    objective: str | None,
    parameter: float | None,
) -> WeeklyEvaluationSpec:
    output.mkdir(parents=True)
    configuration = {"model": {"fields": {}}}
    if objective is not None:
        configuration["policy"] = {
            "fields": {
                "objective": objective,
                "risk_parameter": parameter,
            }
        }
    spec = WeeklyEvaluationSpec(
        label=f"test-{model}-{schedule.value}",
        model=model,
        measure="test",
        model_config_identity=f"model-{model}",
        policy_config_identity=f"policy-{model}",
        configuration_snapshot_json=json.dumps(configuration),
        dataset_manifest_sha256="a" * 64,
        schedule=schedule,
        evaluation_start=date(2023, 1, 1),
        evaluation_end=date(2025, 12, 31),
        rolling_observations=252,
        target_labels=tuple(target.label for target in weekly_targets()),
        target_quantities=tuple(1.0 for _target in weekly_targets()),
        target_notional_identity="weekly-target-notional-test",
        position_cap=1_000.0,
    )
    (output / "run.json").write_text(
        json.dumps({**spec.to_dict(), "run_identity": spec.run_identity}),
        encoding="utf-8",
    )
    return spec


def _weekly_result(
    *,
    spec: WeeklyEvaluationSpec,
    pure_rmse: float,
) -> WeeklyRunResults:
    origins = {
        (date(2023, 1, 1) + timedelta(days=index)).isoformat(): "b" * 64
        for index in range(528)
    }
    slices = tuple(
        WeeklyResultSlice(
            claim=target.label,
            slice_label="overall",
            year=None,
            volatility_regime=None,
            unhedged_loss=_loss(2.0),
            economic_loss=_loss(pure_rmse + 0.1),
            pure_hedge_loss=_loss(pure_rmse),
            pricing_loss=_loss(0.1),
            mean_transaction_cost=0.01,
            mean_spot_transaction_cost=0.002,
            mean_option_transaction_cost=0.008,
            mean_total_turnover_notional=2.0,
            mean_entry_turnover_notional=0.5,
            mean_gross_exposure=1.5,
            mean_net_exposure=-0.1,
            mean_absolute_net_exposure=0.1,
            mean_nonzero_actions_per_decision=3.0,
            cap_hits=0,
            numerical_cap_hit_result_rate=0.0,
        )
        for target in weekly_targets()
    )
    return WeeklyRunResults(
        run_identity=spec.run_identity,
        model=spec.model,
        measure=spec.measure,
        schedule=spec.schedule.value,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        result_count=len(origins) * len(slices),
        fit_cap_hits=0,
        executed_cap_hits=0,
        nonconverged_fits=0,
        cap_hits=0,
        aggregation_config={},
        volatility_thresholds=(0.15, 0.25),
        exposure_convention="test",
        source_run_sha256="d" * 64,
        source_summary_sha256="e" * 64,
        record_sha256_by_origin=origins,
        slices=slices,
    )


def _loss(root_mean_square: float) -> WeeklyLossStatistics:
    objectives = {
        "0.1": root_mean_square,
        "1": root_mean_square,
        "10": root_mean_square,
    }
    return WeeklyLossStatistics(
        observations=528,
        mean=0.25 * root_mean_square,
        median=0.2 * root_mean_square,
        sample_standard_deviation=0.5 * root_mean_square,
        root_mean_square=root_mean_square,
        mean_absolute_loss=0.5 * root_mean_square,
        quantile_95=1.5 * root_mean_square,
        quantile_99=1.8 * root_mean_square,
        expected_shortfall_95=2.0 * root_mean_square,
        expected_shortfall_99=2.5 * root_mean_square,
        hac_standard_error=0.1 * root_mean_square,
        raw_hac_long_run_variance=0.01 * root_mean_square**2,
        hac_long_run_variance=0.01 * root_mean_square**2,
        hac_variance_was_floored=False,
        hac_kernel="bartlett",
        hac_lags=1,
        hac_confidence_interval_95=(-root_mean_square, root_mean_square),
        hac_confidence_interval_99=(-2.0 * root_mean_square, 2.0 * root_mean_square),
        bootstrap_confidence_interval_95=(-root_mean_square, root_mean_square),
        bootstrap_confidence_interval_99=(-2.0 * root_mean_square, 2.0 * root_mean_square),
        mean_variance=dict(objectives),
        mean_volatility={
            "0.1": root_mean_square,
            "1": root_mean_square,
            "2": root_mean_square,
            "3": root_mean_square,
        },
        entropy=dict(objectives),
        power_exponential=dict(objectives),
    )


def _record_losses(
    result: WeeklyRunResults,
) -> dict[str, tuple[dict[str, object], ...]]:
    origins = tuple(result.record_sha256_by_origin)
    return {
        value.claim: tuple(
            {
                "origin_date": origin,
                "unhedged_loss": value.unhedged_loss.root_mean_square,
                "economic_loss": value.economic_loss.root_mean_square,
                "pure_hedge_loss": value.pure_hedge_loss.root_mean_square,
                "pricing_loss": value.pricing_loss.root_mean_square,
            }
            for origin in origins
        )
        for value in result.slices
        if value.slice_label == "overall"
    }
