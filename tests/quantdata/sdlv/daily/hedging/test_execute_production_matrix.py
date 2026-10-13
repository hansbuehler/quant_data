from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from filelock import FileLock

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import weekly_targets
from quantdata.sdlv.daily.front_spot.hedging.data.public_view import (
    PublicHedgingPanelView,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import (
    execute_production_matrix,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DAILY_RECORD_SCHEMA,
    DailyEvaluationSpec,
    DailyEvaluationSummary,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    summarize_completed_daily_run,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
    WeeklyEvaluationSummary,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    WeeklyLossStatistics,
    WeeklyResultsConfig,
    WeeklyResultSlice,
    WeeklyRunResults,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _DailyDataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _DailySnapshot:
    normalized_spot = 1.0

    @staticmethod
    def quote_by_id() -> dict[str, object]:
        return {}


class _DailyMarketDay:
    @staticmethod
    def atm_vol_at_business_dte(_business_dte: int) -> float:
        return 0.2

    @staticmethod
    def hedge_snapshot() -> _DailySnapshot:
        return _DailySnapshot()


class _DailyPanel(PublicHedgingPanelView):
    def __init__(self, directory: Path, *, manifest_sha256: str) -> None:
        self.geometry = {"schema": HEDGING_PUBLIC_SCHEMA}
        self.dataset = _DailyDataset(directory)
        self.dataset_manifest_sha256 = manifest_sha256
        self.dates = (
            date(2022, 12, 28),
            date(2022, 12, 29),
            date(2023, 1, 17),
            date(2023, 1, 18),
            date(2023, 1, 19),
        )

    def prior_one_day_origins(self, origin_date: date, *, count: int):
        index = self.dates.index(origin_date)
        if index < count:
            raise ValueError("insufficient history")
        return self.dates[index - count : index]

    def one_day_transition(self, origin_date: date) -> object:
        index = self.dates.index(origin_date)
        if index + 1 >= len(self.dates):
            raise ValueError("no endpoint")
        return object()

    def market_day(self, _trading_date: date) -> _DailyMarketDay:
        return _DailyMarketDay()


def _write_authenticated_daily_output(
    output: Path,
    *,
    model: str = "historical-sanos",
) -> tuple[str, Path, _DailyPanel]:
    origins = ("2023-01-17", "2023-01-18")
    target = "digital-put-1d-atm"
    dataset = output.parent / f"{output.name}-dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = _sha256(manifest_path)
    panel = _DailyPanel(dataset, manifest_sha256=manifest_sha256)
    spec = DailyEvaluationSpec(
        label="daily-test",
        model=model,
        measure="P",
        model_config_identity="model-test",
        model_config_snapshot_json='{"test":true}',
        dataset_manifest_sha256=manifest_sha256,
        evaluation_start=date.fromisoformat(origins[0]),
        evaluation_end=date.fromisoformat(origins[-1]),
        rolling_observations=2,
        target_labels=(target,),
        target_quantities=(1.0,),
        target_notional_identity="notional-test",
    )
    _write_json(output / "run.json", {**spec.to_dict(), "run_identity": spec.run_identity})
    for origin in origins:
        result = HedgeSmokeResult(
            model=model,
            measure=spec.measure,
            origin_date=origin,
            terminal_date=origin,
            claim=target,
            market_price=0.5,
            model_price=0.5,
            payoff=0.0,
            unhedged_pnl=-0.5,
            hedged_pnl=-0.5,
            transaction_cost=0.0,
            actions={},
            pricing_difference=0.0,
            pure_hedge_error=-0.5,
            objective_value=0.25,
            converged=True,
            cap_hits=0,
        )
        _write_json(
            output / "records" / f"{origin}.json",
            {
                "schema": DAILY_RECORD_SCHEMA,
                "run_identity": spec.run_identity,
                "origin_date": origin,
                "results": [result.to_dict()],
            },
        )
    summary = DailyEvaluationSummary(
        run_identity=spec.run_identity,
        eligible_origins=origins,
        completed_origins=origins,
        reused_origins=(),
        result_count=len(origins),
        nonconverged_results=0,
        cap_hits=0,
    )
    _write_json(output / "summary.json", summary.to_dict())
    summarize_completed_daily_run(panel, run_directory=output).write_json(
        output / "results.json"
    )
    return spec.run_identity, dataset, panel


def _weekly_loss_statistics(
    *,
    observations: int,
    config: WeeklyResultsConfig,
) -> WeeklyLossStatistics:
    return WeeklyLossStatistics(
        observations=observations,
        mean=0.0,
        median=0.0,
        sample_standard_deviation=0.0,
        root_mean_square=0.0,
        mean_absolute_loss=0.0,
        quantile_95=0.0,
        quantile_99=0.0,
        expected_shortfall_95=0.0,
        expected_shortfall_99=0.0,
        hac_standard_error=0.0,
        raw_hac_long_run_variance=0.0,
        hac_long_run_variance=0.0,
        hac_variance_was_floored=False,
        hac_kernel=config.inference.hac_kernel.value,
        hac_lags=1,
        hac_confidence_interval_95=(0.0, 0.0),
        hac_confidence_interval_99=(0.0, 0.0),
        bootstrap_confidence_interval_95=(0.0, 0.0),
        bootstrap_confidence_interval_99=(0.0, 0.0),
        mean_variance={
            f"{parameter:g}": 0.0 for parameter in config.mean_variance_parameters
        },
        mean_volatility={
            f"{parameter:g}": 0.0
            for parameter in config.mean_volatility_parameters
        },
        entropy={f"{parameter:g}": 0.0 for parameter in config.entropy_parameters},
        power_exponential={
            f"{parameter:g}": 0.0
            for parameter in config.power_exponential_parameters
        },
    )


def _write_authenticated_weekly_output(
    output: Path,
    *,
    cap_hits: int = 0,
    require_no_cap_hits: bool = True,
) -> WeeklyRunResults:
    origins = ("2023-01-17", "2023-01-18")
    targets = weekly_targets()
    spec = WeeklyEvaluationSpec(
        label="weekly-test",
        model="sanos-lv",
        measure="Q",
        model_config_identity="model-test",
        policy_config_identity="policy-test",
        configuration_snapshot_json='{"test":true}',
        dataset_manifest_sha256="d" * 64,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        evaluation_start=date.fromisoformat(origins[0]),
        evaluation_end=date.fromisoformat(origins[-1]),
        rolling_observations=252,
        target_labels=tuple(target.label for target in targets),
        target_quantities=tuple(1.0 for _ in targets),
        target_notional_identity="notional-test",
        position_cap=1_000.0,
        require_no_cap_hits=require_no_cap_hits,
    )
    run_path = output / "run.json"
    _write_json(run_path, {**spec.to_dict(), "run_identity": spec.run_identity})
    summary = WeeklyEvaluationSummary(
        run_identity=spec.run_identity,
        eligible_origins=origins,
        completed_origins=origins,
        reused_origins=(),
        result_count=len(origins) * len(targets),
        fit_cap_hits=cap_hits,
        executed_cap_hits=0,
        nonconverged_fits=int(cap_hits > 0),
        cap_hits=cap_hits,
    )
    summary_path = output / "summary.json"
    _write_json(summary_path, summary.to_dict())
    records = output / "records"
    record_sha256_by_origin: dict[str, str] = {}
    for origin in origins:
        record_path = records / f"{origin}.json"
        _write_json(record_path, {"origin": origin})
        record_sha256_by_origin[origin] = _sha256(record_path)
    config = WeeklyResultsConfig()
    losses = _weekly_loss_statistics(observations=len(origins), config=config)
    slices = tuple(
        WeeklyResultSlice(
            claim=target.label,
            slice_label="overall",
            year=None,
            volatility_regime=None,
            unhedged_loss=losses,
            economic_loss=losses,
            pure_hedge_loss=losses,
            pricing_loss=losses,
            mean_transaction_cost=0.0,
            mean_spot_transaction_cost=0.0,
            mean_option_transaction_cost=0.0,
            mean_total_turnover_notional=0.0,
            mean_entry_turnover_notional=0.0,
            mean_gross_exposure=0.0,
            mean_net_exposure=0.0,
            mean_absolute_net_exposure=0.0,
            mean_nonzero_actions_per_decision=0.0,
            cap_hits=cap_hits,
            numerical_cap_hit_result_rate=float(cap_hits > 0),
        )
        for target in targets
    )
    results = WeeklyRunResults(
        run_identity=spec.run_identity,
        model=spec.model,
        measure=spec.measure,
        schedule=spec.schedule.value,
        dataset_manifest_sha256=spec.dataset_manifest_sha256,
        result_count=summary.result_count,
        fit_cap_hits=summary.fit_cap_hits,
        executed_cap_hits=summary.executed_cap_hits,
        nonconverged_fits=summary.nonconverged_fits,
        cap_hits=summary.cap_hits,
        aggregation_config=config.to_dict(),
        volatility_thresholds=(0.2, 0.3),
        exposure_convention="test convention",
        source_run_sha256=_sha256(run_path),
        source_summary_sha256=_sha256(summary_path),
        record_sha256_by_origin=record_sha256_by_origin,
        slices=slices,
    )
    results.write_json(output / "results.json")
    return results


def test_daily_production_validation_reconciles_dated_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "daily"
    _, dataset, panel = _write_authenticated_daily_output(output)
    monkeypatch.setattr(
        execute_production_matrix,
        "_load_daily_panel",
        lambda _dataset: panel,
    )
    results_path = output / "results.json"

    validated = execute_production_matrix._validate_results(
        results_path,
        output=output,
        matrix_schema="hedging-daily-production-matrix-v1",
        expected_model="historical-sanos",
        expected_result_count=2,
        dataset=dataset,
    )

    assert validated == _sha256(results_path)
    record_path = output / "records" / "2023-01-17.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["results"][0]["claim"] = "tampered-claim"
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="precommitted target book"):
        execute_production_matrix._validate_results(
            results_path,
            output=output,
            matrix_schema="hedging-daily-production-matrix-v1",
            expected_model="historical-sanos",
            expected_result_count=2,
            dataset=dataset,
        )


def test_weekly_production_validation_retains_precommitted_cap_hits(
    tmp_path: Path,
) -> None:
    output = tmp_path / "weekly"
    results = _write_authenticated_weekly_output(
        output,
        cap_hits=3,
        require_no_cap_hits=False,
    )

    validated = execute_production_matrix._validate_results(
        output / "results.json",
        output=output,
        matrix_schema="hedging-weekly-production-matrix-v1",
        expected_model="sanos-lv",
        expected_result_count=results.result_count,
    )

    assert validated == _sha256(output / "results.json")


def test_execute_production_queue_runs_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    _, dataset, panel = _write_authenticated_daily_output(output)
    script = tmp_path / "write_result.py"
    script.write_text(
        'print("synthetic production complete", flush=True)\n',
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "historical-sanos-test",
                        "model": "historical-sanos",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(script),
                            str(output),
                            "--dataset",
                            str(dataset),
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execute_production_matrix, "_check_resources", lambda: None)
    monkeypatch.setattr(
        execute_production_matrix,
        "_load_daily_panel",
        lambda _dataset: panel,
    )
    state_path = tmp_path / "queue.json"

    first = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("historical-sanos-test",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
    )
    second = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("historical-sanos-test",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
    )

    assert first["status"] == "complete"
    assert second["status"] == first["status"]
    assert second["completed_at"] == first["completed_at"]
    assert len(first["matrix_sha256"]) == 64
    records = first["records"]
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["status"] == "complete"
    assert records[0]["validated_results_sha256"] == _sha256(
        output / "results.json"
    )
    assert "revalidated_at" in second["records"][0]
    assert "synthetic production complete" in (output / "run.log").read_text()

    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    payload["tampered"] = True
    _write_json(output / "results.json", payload)
    with pytest.raises(ValueError, match="daily aggregate does not match"):
        execute_production_matrix.execute_production_queue(
            matrix_path=matrix,
            state_path=state_path,
            interpreter=Path(sys.executable),
            case_ids=("historical-sanos-test",),
            case_prefix=None,
            resource="cpu",
            expected_result_count=2,
            heartbeat_seconds=1.0,
        )

    matrix.write_text(matrix.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matrix bytes changed"):
        execute_production_matrix.execute_production_queue(
            matrix_path=matrix,
            state_path=state_path,
            interpreter=Path(sys.executable),
            case_ids=("historical-sanos-test",),
            case_prefix=None,
            resource="cpu",
            expected_result_count=2,
            heartbeat_seconds=1.0,
        )


def test_execute_production_queue_rejects_unready_case(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "blocked",
                        "model": "gromm",
                        "resource": "gpu",
                        "readiness": "blocked",
                        "command": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ready"):
        execute_production_matrix.execute_production_queue(
            matrix_path=matrix,
            state_path=tmp_path / "queue.json",
            interpreter=Path(sys.executable),
            case_ids=("blocked",),
            case_prefix=None,
            resource="gpu",
            expected_result_count=2,
        )


def test_execute_production_queue_rejects_concurrent_state_owner(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "queue.json"
    lock = FileLock(str(state_path) + ".lock")

    with lock:
        with pytest.raises(RuntimeError, match="queue state is already owned"):
            execute_production_matrix.execute_production_queue(
                matrix_path=tmp_path / "unused-matrix.json",
                state_path=state_path,
                interpreter=Path(sys.executable),
                case_ids=("unused",),
                case_prefix=None,
                resource="cpu",
                expected_result_count=2,
            )


def test_execute_production_queue_can_retain_failure_and_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_output = tmp_path / "failed"
    complete_output = tmp_path / "complete"
    _, dataset, panel = _write_authenticated_daily_output(complete_output)
    failed_script = tmp_path / "fail.py"
    failed_script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    complete_script = tmp_path / "complete.py"
    complete_script.write_text(
        'print("synthetic production complete", flush=True)\n',
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "failed-case",
                        "model": "historical-sanos",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(failed_script),
                            "--dataset",
                            str(dataset),
                            "--output",
                            str(failed_output),
                        ],
                    },
                    {
                        "case_id": "complete-case",
                        "model": "historical-sanos",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(complete_script),
                            str(complete_output),
                            "--dataset",
                            str(dataset),
                            "--output",
                            str(complete_output),
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execute_production_matrix, "_check_resources", lambda: None)
    monkeypatch.setattr(
        execute_production_matrix,
        "_load_daily_panel",
        lambda _dataset: panel,
    )
    state_path = tmp_path / "queue.json"

    first = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("failed-case", "complete-case"),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
        continue_after_failure=True,
    )
    second = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("failed-case", "complete-case"),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        heartbeat_seconds=1.0,
        continue_after_failure=True,
    )

    assert first["status"] == "complete-with-failures"
    assert second["status"] == first["status"]
    assert second["completed_at"] == first["completed_at"]
    assert [record["status"] for record in first["records"]] == [
        "failed",
        "complete",
    ]
    assert first["records"][0]["return_code"] == 7
    assert "production command returned 7" in first["records"][0]["error"]


def test_execute_production_queue_requires_explicit_legacy_state_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    _, dataset, panel = _write_authenticated_daily_output(output)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-daily-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "ready",
                        "model": "historical-sanos",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            "unused",
                            "--dataset",
                            str(dataset),
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        execute_production_matrix,
        "_load_daily_panel",
        lambda _dataset: panel,
    )
    state_path = tmp_path / "queue.json"
    state_path.write_text(
        json.dumps(
            {
                "schema": execute_production_matrix.QUEUE_SCHEMA,
                "matrix_path": str(matrix.resolve()),
                "resource": "cpu",
                "case_ids": ["ready"],
                "status": "complete",
                "current_case_id": None,
                "records": [{"case_id": "ready", "status": "complete"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit adoption"):
        execute_production_matrix.execute_production_queue(
            matrix_path=matrix,
            state_path=state_path,
            interpreter=Path(sys.executable),
            case_ids=("ready",),
            case_prefix=None,
            resource="cpu",
            expected_result_count=2,
        )

    adopted = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=state_path,
        interpreter=Path(sys.executable),
        case_ids=("ready",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=2,
        adopt_unpinned_state=True,
    )
    assert len(adopted["matrix_sha256"]) == 64
    assert "legacy_state_adopted_at" in adopted
    assert adopted["records"][0]["validated_results_sha256"] == _sha256(
        output / "results.json"
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["matrix_sha256"] == adopted["matrix_sha256"]
    assert persisted["legacy_state_adopted_at"] == adopted["legacy_state_adopted_at"]


def test_generated_production_command_rechecks_reviewed_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantdata.sdlv.daily.front_spot.hedging.runners import generated_evidence

    gate = tmp_path / "gate.json"
    gate.write_text('{"status":"pass"}\n', encoding="utf-8")
    case = {
        "case_id": "volgan-test",
        "model": "volgan",
        "reviewed_commit": "a" * 40,
        "production_gate_path": str(gate),
        "production_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "command": ["python", "module.py"],
    }
    monkeypatch.setattr(
        generated_evidence,
        "reviewable_repository_commit",
        lambda: "a" * 40,
    )
    command = execute_production_matrix._production_command(
        case=case,
        interpreter=Path(sys.executable),
        expected_result_count=2,
    )
    assert command[0] == sys.executable

    gate.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gate bytes changed"):
        execute_production_matrix._production_command(
            case=case,
            interpreter=Path(sys.executable),
            expected_result_count=2,
        )
    gate.write_text('{"status":"pass"}\n', encoding="utf-8")

    monkeypatch.setattr(
        generated_evidence,
        "reviewable_repository_commit",
        lambda: "b" * 40,
    )
    with pytest.raises(ValueError, match="reviews another commit"):
        execute_production_matrix._production_command(
            case=case,
            interpreter=Path(sys.executable),
            expected_result_count=2,
        )


def test_resource_probe_warns_when_gpuctl_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        execute_production_matrix,
        "GPUCTL_PATH",
        tmp_path / "missing-gpuctl",
    )

    execute_production_matrix._check_resources()

    assert "could not be launched" in capsys.readouterr().out


def test_production_child_environment_preserves_only_managed_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    gpu = execute_production_matrix._child_environment(resource="gpu")
    cpu = execute_production_matrix._child_environment(resource="cpu")

    assert gpu["CUDA_VISIBLE_DEVICES"] == "2"
    assert "JAX_PLATFORMS" not in gpu or gpu["JAX_PLATFORMS"] != "cpu"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["JAX_PLATFORMS"] == "cpu"


def test_gpu_child_environment_requires_gpuctl_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="managed gpuctl allocation"):
        execute_production_matrix._child_environment(resource="gpu")


def test_execute_weekly_production_queue_validates_complete_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "weekly-output"
    results = _write_authenticated_weekly_output(output)
    script = tmp_path / "write_weekly_result.py"
    script.write_text(
        'print("synthetic weekly production complete", flush=True)\n',
        encoding="utf-8",
    )
    matrix = tmp_path / "weekly-matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema": "hedging-weekly-production-matrix-v1",
                "cases": [
                    {
                        "case_id": "frozen-weekly-sanos-lv-reference",
                        "model": "sanos-lv",
                        "resource": "cpu",
                        "readiness": "ready",
                        "command": [
                            "python",
                            str(script),
                            "--output",
                            str(output),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(execute_production_matrix, "_check_resources", lambda: None)

    state = execute_production_matrix.execute_production_queue(
        matrix_path=matrix,
        state_path=tmp_path / "weekly-queue.json",
        interpreter=Path(sys.executable),
        case_ids=("frozen-weekly-sanos-lv-reference",),
        case_prefix=None,
        resource="cpu",
        expected_result_count=26,
        heartbeat_seconds=1.0,
    )

    assert state["status"] == "complete"
    results_path = output / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["measure"] = "P"
    _write_json(results_path, payload)
    with pytest.raises(ValueError, match="identity is not authentic"):
        execute_production_matrix._validate_results(
            results_path,
            output=output,
            matrix_schema="hedging-weekly-production-matrix-v1",
            expected_model="sanos-lv",
            expected_result_count=26,
        )
    results.write_json(results_path)
    record_path = output / "records" / "2023-01-17.json"
    record_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="record hash changed"):
        execute_production_matrix._validate_results(
            results_path,
            output=output,
            matrix_schema="hedging-weekly-production-matrix-v1",
            expected_model="sanos-lv",
            expected_result_count=26,
        )
