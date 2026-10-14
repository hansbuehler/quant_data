from __future__ import annotations

from datetime import date

import pytest

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyEvaluationSpec,
    DailyEvaluationSummary,
    run_daily_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
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
        cap_hits=0,
        claim_quantity=claim_quantity,
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
        "nonconverged_results": 1,
    }


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
