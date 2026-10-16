from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quantdata.hedging.accounting import claim_pnl_breakdown
from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyHedgeResult,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyModelRun,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners import weekly as weekly_runner
from quantdata.sdlv.daily.front_spot.hedging.runners.numerical_gate_failure import (
    NUMERICAL_GATE_FAILURE_FILENAME,
    load_numerical_gate_failure,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly import (
    WeeklyEvaluationSpec,
    WeeklyEvaluationSummary,
    read_weekly_record,
    run_weekly_evaluation,
)


class _Dataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _RunnerPanel(PublicHedgingPanel):
    def __init__(self, directory: Path) -> None:
        self.geometry = {"schema": HEDGING_PUBLIC_SCHEMA}
        self.dataset = _Dataset(directory)
        self.dates = (
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 5),
            date(2023, 1, 6),
            date(2023, 1, 9),
            date(2023, 1, 10),
        )


def _spec(manifest_sha256: str) -> WeeklyEvaluationSpec:
    return WeeklyEvaluationSpec(
        label="test-weekly",
        model="test-model",
        measure="Q",
        model_config_identity="market-v1",
        policy_config_identity="policy-v1",
        configuration_snapshot_json="{}",
        dataset_manifest_sha256=manifest_sha256,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        evaluation_start=date(2023, 1, 3),
        evaluation_end=date(2023, 1, 4),
        rolling_observations=252,
        target_labels=tuple(target.label for target in weekly_targets()),
        target_quantities=tuple(0.5 for _ in weekly_targets()),
        target_notional_identity="weekly-target-notional-v1:2022-12-30",
        position_cap=1_000.0,
    )


def _run(
    origin_date: date,
    *,
    target_label: str,
    claim_quantity: float,
    spot_position: float = 0.0,
    fit_converged: bool = True,
    fit_cap_hits: int = 0,
    fit_unregularized_recession_direction: bool = False,
    fit_recession_direction: bool = False,
) -> WeeklyModelRun:
    terminal_date = origin_date + timedelta(days=7)
    market_price = 0.10
    model_price = 0.11
    payoff = 0.20
    hedge_gain = 0.01
    transaction_cost = 0.001
    breakdown = claim_pnl_breakdown(
        claim_quantity=claim_quantity,
        market_price=market_price,
        model_price=model_price,
        payoff=payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
    )
    result = WeeklyHedgeResult(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        target_label=target_label,
        origin_date=origin_date,
        terminal_date=terminal_date,
        terminal_value=payoff,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
        spot_transaction_cost=0.0004,
        option_transaction_cost=0.0006,
        spot_gain=0.004,
        option_gain=0.006,
        step_transaction_costs=np.asarray(
            [0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001]
        ),
        step_hedge_gains=np.full(5, hedge_gain / 5.0),
        breakdown=breakdown,
    )
    prices = tuple(
        WeeklyPriceRecord(
            common_market_price_per_fit_spot=(market_price if index == 0 else None),
            q_expectation_per_fit_spot=model_price,
            buyer_indifference_per_fit_spot=None,
            seller_indifference_per_fit_spot=None,
            continuation_price_per_fit_spot=model_price,
        )
        for index in range(5)
    )
    return WeeklyModelRun(
        result=result,
        market_fit_identities=tuple("market-fit" for _ in range(5)),
        policy_fit_identities=tuple(
            f"policy-fit-{target_label}" for _ in range(5)
        ),
        market_artifact_fingerprints=tuple(
            ("market:sha256",) for _ in range(5)
        ),
        policy_artifact_fingerprints=tuple(
            ("policy:sha256",) for _ in range(5)
        ),
        fit_as_of_dates=tuple(origin_date.isoformat() for _ in range(5)),
        fit_price_records=prices,
        policy_fit_converged=tuple(fit_converged for _ in range(5)),
        policy_fit_cap_hits=tuple(fit_cap_hits for _ in range(5)),
        policy_fit_unregularized_recession_directions=tuple(
            fit_unregularized_recession_direction for _ in range(5)
        ),
        policy_fit_recession_directions=tuple(
            fit_recession_direction for _ in range(5)
        ),
        fit_long_prices_origin_pv=np.full(5, model_price),
        spot_positions=np.full(5, spot_position),
        option_positions=np.zeros((5, 2)),
    )


def _book(origin_date: date, *, cap_first: bool = False) -> tuple[WeeklyModelRun, ...]:
    return tuple(
        _run(
            origin_date,
            target_label=target.label,
            claim_quantity=0.5,
            spot_position=(1_000.0 if cap_first and index == 0 else 0.0),
        )
        for index, target in enumerate(weekly_targets())
    )


def _panel_and_manifest(tmp_path: Path) -> tuple[_RunnerPanel, str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return _RunnerPanel(tmp_path), hashlib.sha256(b"{}\n").hexdigest()


def test_weekly_price_record_allows_signed_indifference_prices() -> None:
    prices = WeeklyPriceRecord(
        common_market_price_per_fit_spot=None,
        q_expectation_per_fit_spot=None,
        buyer_indifference_per_fit_spot=-0.01,
        seller_indifference_per_fit_spot=0.02,
        continuation_price_per_fit_spot=0.005,
    )

    assert prices.long_inception_price_per_fit_spot == pytest.approx(-0.01)


@pytest.mark.parametrize(
    ("market_price", "q_expectation"),
    [(-0.01, None), (None, -0.01)],
)
def test_weekly_price_record_rejects_negative_market_or_q_price(
    market_price: float | None,
    q_expectation: float | None,
) -> None:
    with pytest.raises(ValueError, match="market and risk-neutral"):
        WeeklyPriceRecord(
            common_market_price_per_fit_spot=market_price,
            q_expectation_per_fit_spot=q_expectation,
            buyer_indifference_per_fit_spot=(0.01 if q_expectation is None else None),
            seller_indifference_per_fit_spot=None,
            continuation_price_per_fit_spot=0.01,
        )


def test_weekly_runner_writes_complete_atomic_books_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, manifest_sha256 = _panel_and_manifest(tmp_path)
    origin = date(2023, 1, 3)
    monkeypatch.setattr(
        weekly_runner,
        "overlapping_weekly_origins",
        lambda *_args, **_kwargs: (origin,),
    )
    calls: list[date] = []

    def run_book(_panel: PublicHedgingPanel, origin_date: date):
        calls.append(origin_date)
        return _book(origin_date)

    output = tmp_path / "run"
    first = run_weekly_evaluation(
        panel,
        spec=_spec(manifest_sha256),
        run_book=run_book,
        output_directory=output,
    )
    second = run_weekly_evaluation(
        panel,
        spec=_spec(manifest_sha256),
        run_book=run_book,
        output_directory=output,
    )

    assert calls == [origin]
    assert first.result_count == 13
    assert first.fit_cap_hits == 0
    assert first.executed_cap_hits == 0
    assert first.nonconverged_fits == 0
    assert second.reused_origins == (origin.isoformat(),)
    assert not tuple(output.rglob("*.tmp"))
    record = read_weekly_record(
        output / "records" / f"{origin.isoformat()}.json",
        panel=panel,
        spec=_spec(manifest_sha256),
        origin_date=origin,
    )
    assert len(record) == 13
    assert record[0].result.breakdown == _book(origin)[0].result.breakdown
    assert record[0].market_artifact_fingerprints == (
        ("market:sha256",),
    ) * 5
    run_payload = {
        **_spec(manifest_sha256).to_dict(),
        "run_identity": first.run_identity,
    }
    assert WeeklyEvaluationSpec.from_dict(run_payload) == _spec(manifest_sha256)
    assert WeeklyEvaluationSummary.from_dict(second.to_dict()) == second


def test_weekly_runner_rejects_incomplete_book_and_cap_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, manifest_sha256 = _panel_and_manifest(tmp_path)
    origin = date(2023, 1, 3)
    monkeypatch.setattr(
        weekly_runner,
        "overlapping_weekly_origins",
        lambda *_args, **_kwargs: (origin,),
    )
    with pytest.raises(ValueError, match="precommitted target book"):
        run_weekly_evaluation(
            panel,
            spec=_spec(manifest_sha256),
            run_book=lambda _panel, origin_date: _book(origin_date)[:-1],
            output_directory=tmp_path / "incomplete",
        )
    cap_output = tmp_path / "cap-hit"
    with pytest.raises(RuntimeError, match="position-cap hits"):
        run_weekly_evaluation(
            panel,
            spec=_spec(manifest_sha256),
            run_book=lambda _panel, origin_date: _book(
                origin_date,
                cap_first=True,
            ),
            output_directory=cap_output,
        )
    failure = load_numerical_gate_failure(
        cap_output / NUMERICAL_GATE_FAILURE_FILENAME
    )
    assert failure["origin_date"] == origin.isoformat()
    assert failure["triggered_gates"] == ["position-cap-hit"]
    assert failure["diagnostics"] == {
        "cap_hits": 5,
        "executed_cap_hits": 5,
        "fit_cap_hits": 0,
        "nonconverged_fits": 0,
        "unregularized_recession_direction_fits": 0,
        "recession_direction_fits": 0,
    }


def test_weekly_runner_authenticates_internal_fit_diagnostics_without_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, manifest_sha256 = _panel_and_manifest(tmp_path)
    origin = date(2023, 1, 3)
    monkeypatch.setattr(
        weekly_runner,
        "overlapping_weekly_origins",
        lambda *_args, **_kwargs: (origin,),
    )
    strict = _spec(manifest_sha256)

    def book(_panel: PublicHedgingPanel, origin_date: date):
        runs = list(_book(origin_date))
        runs[0] = _run(
            origin_date,
            target_label=weekly_targets()[0].label,
            claim_quantity=0.5,
            fit_converged=True,
            fit_cap_hits=3,
        )
        return tuple(runs)

    with pytest.raises(RuntimeError, match="3 position-cap hits"):
        run_weekly_evaluation(
            panel,
            spec=strict,
            run_book=book,
            output_directory=tmp_path / "strict-fit-cap",
        )

    def nonconverged_book(_panel: PublicHedgingPanel, origin_date: date):
        runs = list(_book(origin_date))
        runs[0] = _run(
            origin_date,
            target_label=weekly_targets()[0].label,
            claim_quantity=0.5,
            fit_converged=False,
        )
        return tuple(runs)

    with pytest.raises(RuntimeError, match="1 nonconverged policy fits"):
        run_weekly_evaluation(
            panel,
            spec=strict,
            run_book=nonconverged_book,
            output_directory=tmp_path / "strict-nonconverged",
        )

    recession_output = tmp_path / "strict-recession"

    def recession_book(_panel: PublicHedgingPanel, origin_date: date):
        runs = list(_book(origin_date))
        runs[0] = _run(
            origin_date,
            target_label=weekly_targets()[0].label,
            claim_quantity=0.5,
            fit_converged=False,
            fit_cap_hits=3,
            fit_recession_direction=True,
        )
        return tuple(runs)

    with pytest.raises(RuntimeError, match="1 mean-volatility recession-direction"):
        run_weekly_evaluation(
            panel,
            spec=strict,
            run_book=recession_book,
            output_directory=recession_output,
        )
    recession_failure = load_numerical_gate_failure(
        recession_output / NUMERICAL_GATE_FAILURE_FILENAME
    )
    assert recession_failure["classification"] == (
        "mean-volatility-recession-direction"
    )
    assert recession_failure["triggered_gates"] == [
        "mean-volatility-recession-direction",
        "policy-fit-cap-hit",
        "nonconverged-fit",
    ]

    permitted = replace(strict, require_no_cap_hits=False)
    output = tmp_path / "permitted-fit-cap"
    summary = run_weekly_evaluation(
        panel,
        spec=permitted,
        run_book=book,
        output_directory=output,
    )
    assert summary.fit_cap_hits == 3
    assert summary.executed_cap_hits == 0
    assert summary.nonconverged_fits == 0
    assert summary.recession_direction_fits == 0
    assert summary.cap_hits == 3
    record = json.loads(
        (output / "records" / f"{origin.isoformat()}.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["numerical_diagnostics"] == {
        "fit_cap_hits": 3,
        "executed_cap_hits": 0,
        "nonconverged_fits": 0,
        "unregularized_recession_direction_fits": 0,
        "recession_direction_fits": 0,
        "cap_hits": 3,
    }

    def permitted_nonconverged_book(
        _panel: PublicHedgingPanel,
        origin_date: date,
    ):
        runs = list(_book(origin_date))
        runs[0] = _run(
            origin_date,
            target_label=weekly_targets()[0].label,
            claim_quantity=0.5,
            fit_converged=False,
            fit_cap_hits=3,
        )
        return tuple(runs)

    with pytest.raises(RuntimeError, match="1 nonconverged policy fits"):
        run_weekly_evaluation(
            panel,
            spec=permitted,
            run_book=permitted_nonconverged_book,
            output_directory=tmp_path / "permitted-nonconverged",
        )
    nonconverged_failure = load_numerical_gate_failure(
        tmp_path
        / "permitted-nonconverged"
        / NUMERICAL_GATE_FAILURE_FILENAME
    )
    assert nonconverged_failure["gate_policy"] == {
        "require_no_fit_cap_hits": False,
        "require_no_executed_cap_hits": False,
    }
    assert nonconverged_failure["classification"] == "nonconverged-fit"
    assert nonconverged_failure["triggered_gates"] == ["nonconverged-fit"]
    assert nonconverged_failure["diagnostics"]["fit_cap_hits"] == 3
    with pytest.raises(RuntimeError, match="1 mean-volatility recession-direction"):
        run_weekly_evaluation(
            panel,
            spec=permitted,
            run_book=recession_book,
            output_directory=tmp_path / "permitted-recession",
        )
    permitted_recession_failure = load_numerical_gate_failure(
        tmp_path / "permitted-recession" / NUMERICAL_GATE_FAILURE_FILENAME
    )
    assert permitted_recession_failure["gate_policy"] == {
        "require_no_fit_cap_hits": False,
        "require_no_executed_cap_hits": False,
    }
    assert permitted_recession_failure["classification"] == (
        "mean-volatility-recession-direction"
    )
    assert permitted_recession_failure["triggered_gates"] == [
        "mean-volatility-recession-direction",
        "nonconverged-fit",
    ]
    assert permitted_recession_failure["diagnostics"]["fit_cap_hits"] == 3


def test_weekly_runner_rejects_changed_record_and_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, manifest_sha256 = _panel_and_manifest(tmp_path)
    origin = date(2023, 1, 3)
    spec = _spec(manifest_sha256)
    monkeypatch.setattr(
        weekly_runner,
        "overlapping_weekly_origins",
        lambda *_args, **_kwargs: (origin,),
    )
    output = tmp_path / "run"
    run_weekly_evaluation(
        panel,
        spec=spec,
        run_book=lambda _panel, origin_date: _book(origin_date),
        output_directory=output,
    )
    record_path = output / "records" / f"{origin.isoformat()}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["runs"][0]["spot_positions"][0] = 1_000.0
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="position-cap hits"):
        read_weekly_record(
            record_path,
            panel=panel,
            spec=spec,
            origin_date=origin,
        )

    forged = {**spec.to_dict(), "run_identity": "0" * 64}
    with pytest.raises(ValueError, match="identity is not authentic"):
        WeeklyEvaluationSpec.from_dict(forged)


def test_weekly_runner_rejects_truncated_five_decision_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel, manifest_sha256 = _panel_and_manifest(tmp_path)
    origin = date(2023, 1, 3)
    spec = _spec(manifest_sha256)
    monkeypatch.setattr(
        weekly_runner,
        "overlapping_weekly_origins",
        lambda *_args, **_kwargs: (origin,),
    )
    output = tmp_path / "run"
    run_weekly_evaluation(
        panel,
        spec=spec,
        run_book=lambda _panel, origin_date: _book(origin_date),
        output_directory=output,
    )
    record_path = output / "records" / f"{origin.isoformat()}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    for name in (
        "market_fit_identities",
        "policy_fit_identities",
        "market_artifact_fingerprints",
        "policy_artifact_fingerprints",
        "fit_as_of_dates",
        "fit_price_records",
        "fit_long_prices_origin_pv",
        "spot_positions",
        "option_positions",
    ):
        payload["runs"][0][name] = payload["runs"][0][name][:1]
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly five decisions"):
        read_weekly_record(
            record_path,
            panel=panel,
            spec=spec,
            origin_date=origin,
        )
