from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pytest

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    LEGACY_DAILY_EVALUATION_SCHEMA,
    DailyEvaluationSpec,
    DailyEvaluationSummary,
    run_daily_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.merge_daily_shards import (
    merge_daily_shards,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA,
    NUMERICAL_GATE_FAILURE_FILENAME,
    load_numerical_gate_failure,
)


class _Dataset:
    def __init__(self, directory) -> None:
        self.directory = directory


class _RunnerPanel(PublicHedgingPanel):
    def __init__(self, directory) -> None:
        self.geometry = {"schema": HEDGING_PUBLIC_SCHEMA}
        self.dataset = _Dataset(directory)
        self.dates = (
            date(2022, 12, 29),
            date(2022, 12, 30),
            date(2023, 1, 3),
            date(2023, 1, 4),
        )

    def prior_one_day_origins(self, origin_date: date, *, count: int):
        index = self.dates.index(origin_date)
        if index < count:
            raise ValueError("insufficient history")
        return self.dates[index - count : index]

    def one_day_transition(self, origin_date: date):
        index = self.dates.index(origin_date)
        if index + 1 >= len(self.dates):
            raise ValueError("no endpoint")
        return object()


def _result(
    origin_date: date,
    *,
    claim: str,
    claim_quantity: float,
    converged: bool = True,
    cap_hits: int = 0,
    diagnostics: dict[str, object] | None = None,
):
    return HedgeSmokeResult(
        model="test-model",
        measure="Q",
        origin_date=origin_date.isoformat(),
        terminal_date="2023-01-04",
        claim=claim,
        market_price=0.1,
        model_price=0.1,
        payoff=0.0,
        unhedged_pnl=-0.1,
        hedged_pnl=-0.05,
        transaction_cost=0.001,
        actions={"spot": 0.5},
        pricing_difference=0.0,
        pure_hedge_error=-0.05,
        objective_value=0.2,
        converged=converged,
        cap_hits=cap_hits,
        claim_quantity=claim_quantity,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _spec(manifest_sha256: str) -> DailyEvaluationSpec:
    return DailyEvaluationSpec(
        label="test-daily",
        model="test-model",
        measure="Q",
        model_config_identity="test-v1",
        model_config_snapshot_json="{}",
        dataset_manifest_sha256=manifest_sha256,
        evaluation_start=date(2023, 1, 3),
        evaluation_end=date(2023, 1, 4),
        rolling_observations=2,
        target_labels=("claim-a", "claim-b"),
        target_quantities=(0.25, 0.5),
        target_notional_identity="target-notional-v1:2022-12-30",
    )


def test_daily_runner_writes_atomic_records_and_resumes(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    panel = _RunnerPanel(tmp_path)
    calls: list[date] = []

    def run_book(_panel, origin_date):
        calls.append(origin_date)
        return (
            _result(origin_date, claim="claim-a", claim_quantity=0.25),
            _result(origin_date, claim="claim-b", claim_quantity=0.5),
        )

    output = tmp_path / "run"
    first = run_daily_evaluation(
        panel,
        spec=_spec(manifest_sha256),
        run_book=run_book,
        output_directory=output,
    )
    assert calls == [date(2023, 1, 3)]
    assert first.result_count == 2
    assert first.reused_origins == ()
    assert first.unregularized_recession_direction_fits == 0
    second = run_daily_evaluation(
        panel,
        spec=_spec(manifest_sha256),
        run_book=run_book,
        output_directory=output,
    )
    assert calls == [date(2023, 1, 3)]
    assert second.reused_origins == ("2023-01-03",)
    assert not tuple(output.rglob("*.tmp"))

    run_payload = {**_spec(manifest_sha256).to_dict(), "run_identity": first.run_identity}
    assert DailyEvaluationSpec.from_dict(run_payload) == _spec(manifest_sha256)
    assert DailyEvaluationSummary.from_dict(second.to_dict()) == second


def test_daily_run_spec_rejects_forged_identity(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    payload = {
        **_spec(manifest_sha256).to_dict(),
        "run_identity": "0" * 64,
    }
    with pytest.raises(ValueError, match="identity is not authentic"):
        DailyEvaluationSpec.from_dict(payload)


def test_daily_run_spec_authenticates_legacy_v3_identity(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    legacy_spec = replace(
        _spec(manifest_sha256),
        require_no_executed_cap_hits=True,
        metadata_schema=LEGACY_DAILY_EVALUATION_SCHEMA,
        metadata_code_version="1.1",
    )
    payload = {
        **legacy_spec.to_dict(),
        "run_identity": legacy_spec.run_identity,
    }

    assert "require_no_fit_cap_hits" not in payload
    assert "require_no_executed_cap_hits" not in payload
    restored = DailyEvaluationSpec.from_dict(payload)
    assert restored.run_identity == legacy_spec.run_identity
    assert restored.require_no_fit_cap_hits
    assert restored.require_no_executed_cap_hits


def test_daily_runner_rejects_incomplete_or_nonconverged_books(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    panel = _RunnerPanel(tmp_path)
    spec = _spec(manifest_sha256)
    with pytest.raises(ValueError, match="precommitted target book"):
        run_daily_evaluation(
            panel,
            spec=spec,
            run_book=lambda _panel, origin: (
                _result(origin, claim="claim-a", claim_quantity=0.25),
            ),
            output_directory=tmp_path / "incomplete",
        )
    nonconverged_output = tmp_path / "nonconverged"
    with pytest.raises(RuntimeError, match="numerical gate failed"):
        run_daily_evaluation(
            panel,
            spec=spec,
            run_book=lambda _panel, origin: (
                _result(
                    origin,
                    claim="claim-a",
                    claim_quantity=0.25,
                    converged=False,
                ),
                _result(origin, claim="claim-b", claim_quantity=0.5),
            ),
            output_directory=nonconverged_output,
        )
    failure = load_numerical_gate_failure(
        nonconverged_output / NUMERICAL_GATE_FAILURE_FILENAME
    )
    assert failure["origin_date"] == "2023-01-03"
    assert failure["triggered_gates"] == ["nonconverged-fit"]
    assert failure["diagnostics"] == {
        "cap_hits": 0,
        "executed_cap_hits": 0,
        "fit_cap_hits": 0,
        "nonconverged_results": 1,
        "recession_direction_fits": 0,
        "unregularized_recession_direction_fits": 0,
    }


def test_daily_runner_records_executable_cap_hits_without_failing(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    output = tmp_path / "executable-cap"
    summary = run_daily_evaluation(
        _RunnerPanel(tmp_path),
        spec=_spec(manifest_sha256),
        run_book=lambda _panel, origin: (
            _result(
                origin,
                claim="claim-a",
                claim_quantity=0.25,
                cap_hits=3,
                diagnostics={"executable_cap_hits": 3},
            ),
            _result(origin, claim="claim-b", claim_quantity=0.5),
        ),
        output_directory=output,
    )

    assert summary.nonconverged_results == 0
    assert summary.cap_hits == 3
    assert not (output / NUMERICAL_GATE_FAILURE_FILENAME).exists()


def test_legacy_numerical_failure_loads_without_explicit_gate_policy(
    tmp_path,
) -> None:
    path = tmp_path / NUMERICAL_GATE_FAILURE_FILENAME
    payload = {
        "schema": LEGACY_NUMERICAL_GATE_FAILURE_SCHEMA,
        "run_identity": "legacy-run",
        "evaluation_schema": LEGACY_DAILY_EVALUATION_SCHEMA,
        "model": "historical-sanos",
        "measure": "historical-P",
        "schedule": None,
        "dataset_manifest_sha256": "a" * 64,
        "origin_date": "2023-05-30",
        "classification": "nonconverged-fit",
        "triggered_gates": ["nonconverged-fit"],
        "summary": (
            "strict numerical gate rejected nonconverged-fit "
            "(cap_hits=0, nonconverged_results=1)"
        ),
        "diagnostics": {
            "cap_hits": 0,
            "nonconverged_results": 1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_numerical_gate_failure(path) == payload


def test_daily_runner_classifies_mean_volatility_recession(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    output = tmp_path / "recession"
    with pytest.raises(RuntimeError, match="numerical gate failed"):
        run_daily_evaluation(
            _RunnerPanel(tmp_path),
            spec=_spec(manifest_sha256),
            run_book=lambda _panel, origin: (
                _result(
                    origin,
                    claim="claim-a",
                    claim_quantity=0.25,
                    diagnostics={"zero_policy_recession_direction": True},
                ),
                _result(
                    origin,
                    claim="claim-b",
                    claim_quantity=0.5,
                    diagnostics={"zero_policy_recession_direction": True},
                ),
            ),
            output_directory=output,
        )
    failure = load_numerical_gate_failure(
        output / NUMERICAL_GATE_FAILURE_FILENAME
    )

    assert failure["classification"] == "mean-volatility-recession-direction"
    assert failure["triggered_gates"] == ["mean-volatility-recession-direction"]
    assert failure["diagnostics"] == {
        "cap_hits": 0,
        "executed_cap_hits": 0,
        "fit_cap_hits": 0,
        "nonconverged_results": 0,
        "recession_direction_fits": 1,
        "unregularized_recession_direction_fits": 0,
    }


def test_daily_runner_records_nonfatal_unregularized_recession_warning(
    tmp_path,
) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    summary = run_daily_evaluation(
        _RunnerPanel(tmp_path),
        spec=_spec(manifest_sha256),
        run_book=lambda _panel, origin: (
            _result(
                origin,
                claim="claim-a",
                claim_quantity=0.25,
                diagnostics={
                    "claim_policy_unregularized_recession_direction": True,
                    "zero_policy_unregularized_recession_direction": True,
                },
            ),
            _result(
                origin,
                claim="claim-b",
                claim_quantity=0.5,
                diagnostics={
                    "zero_policy_unregularized_recession_direction": True,
                },
            ),
        ),
        output_directory=tmp_path / "warning",
    )

    assert summary.nonconverged_results == 0
    assert summary.unregularized_recession_direction_fits == 2
    assert not (tmp_path / "warning" / NUMERICAL_GATE_FAILURE_FILENAME).exists()


def test_daily_shard_merge_materializes_complete_destination(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    panel = _RunnerPanel(tmp_path)
    panel.dates = (
        date(2022, 12, 27),
        date(2022, 12, 28),
        date(2022, 12, 29),
        date(2022, 12, 30),
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
        date(2023, 1, 6),
    )
    full_spec = replace(_spec(manifest_sha256), evaluation_end=date(2023, 1, 6))

    def run_book(_panel, origin_date):
        return (
            _result(origin_date, claim="claim-a", claim_quantity=0.25),
            _result(origin_date, claim="claim-b", claim_quantity=0.5),
        )

    destination = tmp_path / "destination"
    run_daily_evaluation(
        panel,
        spec=full_spec,
        run_book=run_book,
        output_directory=destination,
        stop_after=1,
    )
    shard_one = tmp_path / "shard-one"
    run_daily_evaluation(
        panel,
        spec=replace(
            full_spec,
            evaluation_start=date(2023, 1, 4),
            evaluation_end=date(2023, 1, 4),
        ),
        run_book=run_book,
        output_directory=shard_one,
    )
    shard_two = tmp_path / "shard-two"
    run_daily_evaluation(
        panel,
        spec=replace(full_spec, evaluation_start=date(2023, 1, 5)),
        run_book=run_book,
        output_directory=shard_two,
    )

    summary = merge_daily_shards(
        panel,
        destination=destination,
        shards=(shard_one, shard_two),
        require_aggregates=False,
    )
    repeated = merge_daily_shards(
        panel,
        destination=destination,
        shards=(shard_one, shard_two),
        require_aggregates=False,
    )

    assert summary == repeated
    assert summary.completed_origins == (
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
    )
    assert summary.result_count == 6
    assert summary.nonconverged_results == 0
    assert summary.cap_hits == 0
    assert tuple(sorted(path.stem for path in (destination / "records").glob("*.json"))) == (
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
    )
    merge_payload = json.loads((destination / "merge.json").read_text(encoding="utf-8"))
    assert merge_payload["native_origin_count"] == 1
    assert [item["origin_count"] for item in merge_payload["shards"]] == [1, 1]


def test_daily_shard_merge_rejects_overlap_and_gap(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    panel = _RunnerPanel(tmp_path)
    panel.dates = (
        date(2022, 12, 27),
        date(2022, 12, 28),
        date(2022, 12, 29),
        date(2022, 12, 30),
        date(2023, 1, 3),
        date(2023, 1, 4),
        date(2023, 1, 5),
        date(2023, 1, 6),
    )
    full_spec = replace(_spec(manifest_sha256), evaluation_end=date(2023, 1, 6))

    def run_book(_panel, origin_date):
        return (
            _result(origin_date, claim="claim-a", claim_quantity=0.25),
            _result(origin_date, claim="claim-b", claim_quantity=0.5),
        )

    destination = tmp_path / "destination"
    run_daily_evaluation(
        panel,
        spec=full_spec,
        run_book=run_book,
        output_directory=destination,
        stop_after=1,
    )
    shard = tmp_path / "shard"
    run_daily_evaluation(
        panel,
        spec=replace(
            full_spec,
            evaluation_start=date(2023, 1, 4),
            evaluation_end=date(2023, 1, 5),
        ),
        run_book=run_book,
        output_directory=shard,
    )

    with pytest.raises(ValueError, match="overlap"):
        merge_daily_shards(
            panel,
            destination=destination,
            shards=(shard, shard),
            require_aggregates=False,
        )
    gap_shard = tmp_path / "gap-shard"
    run_daily_evaluation(
        panel,
        spec=replace(
            full_spec,
            evaluation_start=date(2023, 1, 4),
            evaluation_end=date(2023, 1, 4),
        ),
        run_book=run_book,
        output_directory=gap_shard,
    )
    with pytest.raises(ValueError, match="uncovered"):
        merge_daily_shards(
            panel,
            destination=destination,
            shards=(gap_shard,),
            require_aggregates=False,
        )


def test_daily_runner_rejects_wrong_target_quantity(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    panel = _RunnerPanel(tmp_path)
    with pytest.raises(ValueError, match="claim quantity"):
        run_daily_evaluation(
            panel,
            spec=_spec(manifest_sha256),
            run_book=lambda _panel, origin: (
                _result(origin, claim="claim-a", claim_quantity=0.25),
                _result(origin, claim="claim-b", claim_quantity=1.0),
            ),
            output_directory=tmp_path / "wrong-quantity",
        )
