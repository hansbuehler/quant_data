"""A solver failure must recover on the same market or stop before training."""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from cvxpy.error import SolverError

from quantdata.dysanos.flex import market
from quantdata.dysanos.flex import run_production_gate as runner
from quantdata.dysanos.flex.config import (
    FlexGeometryConfig,
    FlexMarketConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
)
from quantdata.pool import PoolConfig
from quantdata.sanos.arbfree import MkArbFree


@pytest.mark.parametrize("error", [SolverError("CLARABEL failed"), RuntimeError("reached timeout 300.0 and/or iteration limit"),
                                 RuntimeError("Cannot find no-arbitrage surface mids: problem infeasible??")])
@pytest.mark.parametrize("recovered", [True, False])
def test_solver_recovery_preserves_market(monkeypatch, error, recovered):
    geometry, config = FlexGeometryConfig(), FlexMarketConfig()
    day, calls = object(), []

    def load(ticker, trading_date, **kwargs):
        calls.append((ticker, trading_date, kwargs))
        if len(calls) == 1 or not recovered:
            raise error
        return day

    monkeypatch.setattr(runner, "load_market_day", load)
    with pytest.warns(RuntimeWarning, match="3000-second"):
        outcome = runner._load_market_job("SPX", date(2022, 3, 22), geometry=geometry, market_config=config)
    assert len(calls) == 2
    assert all(ticker == "SPX" and trading_date == date(2022, 3, 22) for ticker, trading_date, _ in calls)
    assert all(kwargs["geometry"] is geometry and kwargs["config"] is config for _, _, kwargs in calls)
    assert "mk_arb_free" not in calls[0][2]
    assert calls[1][2]["mk_arb_free"].timeout_seconds == 3000.0
    assert outcome.retry_error == f"{type(error).__name__}: {error}"
    assert (outcome.day is day and outcome.error is None) if recovered else (outcome.day is None and outcome.error == outcome.retry_error)


def test_recovery_budget_reaches_smooth_market(monkeypatch):
    expected, captured = object(), {}
    fitter = MkArbFree(timeout_seconds=3000.0)

    def smooth(*args, **kwargs):
        captured.update(kwargs)
        return {"market": "cached"}

    monkeypatch.setattr(market, "smooth_interpolated_market", smooth)
    monkeypatch.setattr(market, "market_day_from_payload", lambda *args, **kwargs: expected)
    assert market.load_market_day("SPX", date(2022, 3, 22), mk_arb_free=fitter) is expected
    assert captured["mk_arb_free"] is fitter


@pytest.mark.parametrize("failure", ["solver", "missing", "recovered"])
def test_runner_requires_all_dates_and_records_recovery(monkeypatch, tmp_path, failure):
    dates = (date(2022, 3, 22), date(2022, 3, 23))
    primary = "SolverError: CLARABEL failed"
    trained = []

    def load(ticker, day, **kwargs):
        if day == dates[0] and failure == "solver":
            return runner._MarketLoadResult(None, primary, retry_error=primary)
        if day == dates[0] and failure == "missing":
            return runner._MarketLoadResult(None, None)
        return runner._MarketLoadResult(SimpleNamespace(trading_date=day, num_quotes=3), None,
                                        retry_error=primary if day == dates[0] else None)

    class ReachedTraining(Exception):
        pass

    def train(days, **kwargs):
        trained.extend(days)
        raise ReachedTraining

    monkeypatch.setattr(runner, "PoolConfig", lambda **kwargs: PoolConfig.off())
    monkeypatch.setattr(runner, "_load_market_job", load)
    monkeypatch.setattr(runner, "fit_native_states", train)
    expected = ReachedTraining if failure == "recovered" else RuntimeError
    with pytest.raises(expected):
        runner.run_production_gates(
            ticker="SPX", dates=dates, geometry=FlexGeometryConfig(), market_config=FlexMarketConfig(),
            objective_config=FlexObjectiveConfig(), train_config=FlexTrainConfig(), supervised_train_config=FlexTrainConfig(),
            workers=1, output_directory=tmp_path / "reports", checkpoint_directory=tmp_path / "checkpoints", label="recovery",
        )
    report = json.loads((tmp_path / "reports/recovery.json").read_text())
    assert report["requested_dates"] == [str(day) for day in dates]
    if failure == "recovered":
        assert [day.trading_date for day in trained] == list(dates)
        assert report["status"] == "native_running"
        assert report["recovered_market_dates"] == [dict(date=str(dates[0]), primary_error=primary, timeout_seconds=3000.0)]
    else:
        assert not trained and report["status"] == "market_loading_failed"
        assert str(dates[0]) in ([item["date"] for item in report["failed_market_dates"]] + report["missing_market_dates"])
        assert not (tmp_path / "checkpoints/native.pt").exists()
