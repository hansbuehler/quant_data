from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import pytest

from quantdata.hedging.data.public_dataset import HEDGING_PUBLIC_SCHEMA
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    daily_digital_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    DailyEvaluationSpec,
    run_daily_evaluation,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    DailyResultsConfig,
    summarize_completed_daily_run,
    summarize_daily_results,
)


class _Dataset:
    def __init__(self, directory) -> None:
        self.directory = directory


class _Quote:
    mark = 0.25


class _Snapshot:
    normalized_spot = 1.0

    @staticmethod
    def quote_by_id():
        return {"option": _Quote()}


class _MarketDay:
    @staticmethod
    def atm_vol_at_business_dte(_business_dte: int) -> float:
        return 0.2

    @staticmethod
    def hedge_snapshot() -> _Snapshot:
        return _Snapshot()


class _ResultsPanel(PublicHedgingPanel):
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

    def market_day(self, _trading_date: date) -> _MarketDay:
        return _MarketDay()


def _result(
    *,
    origin: date,
    claim: str,
    value: float,
    model: str = "test",
    measure: str = "Q",
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    return HedgeSmokeResult(
        model=model,
        measure=measure,
        origin_date=origin.isoformat(),
        terminal_date=(origin + timedelta(days=1)).isoformat(),
        claim=claim,
        market_price=0.5,
        model_price=0.49,
        payoff=0.5 + value,
        unhedged_pnl=value,
        hedged_pnl=0.5 * value,
        transaction_cost=0.001,
        actions={"spot": value, "option": -2.0 * value},
        pricing_difference=-0.01,
        pure_hedge_error=0.5 * value + 0.01,
        objective_value=0.0,
        converged=True,
        cap_hits=0,
        claim_quantity=claim_quantity,
    )


def test_daily_results_include_overall_year_and_volatility_slices() -> None:
    claim = daily_digital_targets()[0].label
    maturity = daily_digital_targets()[0].maturity_business_days
    dates = (
        date(2023, 1, 3),
        date(2023, 2, 3),
        date(2023, 3, 3),
        date(2024, 1, 3),
        date(2024, 2, 3),
        date(2024, 3, 3),
    )
    values = (-0.03, -0.01, 0.02, -0.02, 0.01, 0.04)
    volatilities = (0.1, 0.2, 0.3, 0.1, 0.2, 0.3)
    results = tuple(
        _result(origin=origin, claim=claim, value=value)
        for origin, value in zip(dates, values, strict=True)
    )
    slices = summarize_daily_results(
        results,
        volatility_by_origin={
            (origin.isoformat(), maturity): volatility
            for origin, volatility in zip(dates, volatilities, strict=True)
        },
        volatility_thresholds={maturity: (0.15, 0.25)},
        config=DailyResultsConfig(hac_lags=1),
    )
    by_label = {value.slice_label: value for value in slices}
    assert set(by_label) == {
        "overall",
        "year:2023",
        "year:2024",
        "regime:low",
        "regime:medium",
        "regime:high",
        "year:2023/regime:low",
        "year:2023/regime:medium",
        "year:2023/regime:high",
        "year:2024/regime:low",
        "year:2024/regime:medium",
        "year:2024/regime:high",
    }
    overall = by_label["overall"]
    assert overall.economic_loss.observations == 6
    assert overall.economic_loss.mean == pytest.approx(-0.5 * sum(values) / 6)
    assert overall.mean_transaction_cost == pytest.approx(0.001)
    assert overall.mean_nonzero_actions == 2.0
    assert overall.mean_entry_turnover_notional == pytest.approx(
        overall.mean_action_l1
    )
    assert overall.mean_gross_exposure == pytest.approx(overall.mean_action_l1)
    assert overall.mean_net_exposure == pytest.approx(-sum(values) / 6)
    assert overall.numerical_cap_hit_result_rate == 0.0
    assert overall.cap_hits == 0
    assert overall.economic_loss.hac_kernel == "bartlett"
    assert overall.economic_loss.hac_lags == 1
    assert "selected-observation" in overall.economic_loss.hac_lag_unit
    assert set(overall.economic_loss.mean_volatility) == {"0.1", "1", "2", "3"}
    assert set(overall.economic_loss.entropy) == {"0.1", "1", "10"}


def test_daily_results_reject_missing_volatility_metadata() -> None:
    target = daily_digital_targets()[0]
    result = _result(
        origin=date(2023, 1, 3),
        claim=target.label,
        value=0.01,
    )
    with pytest.raises(ValueError, match="volatility metadata"):
        summarize_daily_results(
            (result,),
            volatility_by_origin={},
            volatility_thresholds={target.maturity_business_days: (0.15, 0.25)},
        )


def test_completed_daily_results_revalidate_each_record(tmp_path) -> None:
    manifest = b"{}\n"
    (tmp_path / "manifest.json").write_bytes(manifest)
    panel = _ResultsPanel(tmp_path)
    target = daily_digital_targets()[0]
    spec = DailyEvaluationSpec(
        label="forgery-regression",
        model="test-model",
        measure="Q",
        model_config_identity="test-v1",
        model_config_snapshot_json="{}",
        dataset_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        evaluation_start=date(2023, 1, 3),
        evaluation_end=date(2023, 1, 3),
        rolling_observations=2,
        target_labels=(target.label,),
        target_quantities=(1.0,),
        target_notional_identity="target-notional-v1:2022-12-30",
    )
    output = tmp_path / "run"
    run_daily_evaluation(
        panel,
        spec=spec,
        run_book=lambda _panel, origin: (
            _result(
                origin=origin,
                claim=target.label,
                value=0.01,
                model="test-model",
            ),
        ),
        output_directory=output,
    )
    report = summarize_completed_daily_run(panel, run_directory=output)
    assert report.run_identity == spec.run_identity
    assert report.result_count == 1
    assert report.slices[0].mean_entry_turnover_notional == pytest.approx(0.015)
    assert report.aggregation_config["hac_kernel"] == "bartlett"

    record_path = output / "records" / "2023-01-03.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["results"][0].update(
        {
            "model": "forged-model",
            "measure": "P",
            "claim_quantity": -2.0,
            "converged": False,
            "cap_hits": 7,
        }
    )
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity does not match"):
        summarize_completed_daily_run(panel, run_directory=output)
