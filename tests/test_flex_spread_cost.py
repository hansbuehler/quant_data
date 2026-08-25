import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from cdxcore.bs import bs
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.market import FlexMarketDay, MarketEligibilityAudit
from quantdata.dysanos.flex.run_spread_benchmark import _dates_from_report, regression_metrics
from quantdata.dysanos.flex.spread_cost import (
    FittedSpreadCostModel,
    SpreadFeedForwardConfig,
    TorchSpreadCostModel,
    equal_date_sample_weights,
    load_torch_spread_cost_model,
    observations_from_market_day,
    save_torch_spread_cost_model,
)
from quantdata.torch import BaseEnvironment


def _market_day() -> FlexMarketDay:
    expiry = 5.0 / DAYS_PER_YEAR
    strikes = np.array([0.98, 0.98, 1.02, 1.02], dtype=np.float64)
    is_call = np.array([True, False, True, False])
    call_prices = bs.price(
        k=strikes,
        vol=np.full(4, 0.2),
        sqrtT=np.sqrt(expiry),
    )
    fit_prices = call_prices - np.where(is_call, 0.0, 1.0 - strikes)
    return FlexMarketDay(
        ticker="TEST",
        trading_date=date(2024, 1, 2),
        expiries_y=np.array([expiry]),
        expiry_dates=("2024-01-09",),
        expiry_ids=np.array([1]),
        expiry_is_am=np.array([False]),
        expiry_index=np.zeros(4, dtype=np.int64),
        strikes=strikes,
        is_call=is_call,
        bids=fit_prices - 1.0e-4,
        asks=fit_prices + 1.0e-4,
        fit_prices=fit_prices,
        log_volumes=np.zeros(4),
        log_open_interests=np.zeros(4),
        quote_ids=("a", "b", "c", "d"),
        atm_annual_vols=np.array([0.2]),
        model_atm_total_vols=np.full(14, 0.2 * np.sqrt(4.8 / DAYS_PER_YEAR)),
        strike_lower_bounds=np.array([0.8]),
        strike_upper_bounds=np.array([1.2]),
        market_uid="test",
        audit=MarketEligibilityAudit(4, 4, 4, 1, 1),
        spot_ref=5_000.0,
    )


def test_observations_use_side_delta_and_smooth_fit_greeks() -> None:
    observations = observations_from_market_day(_market_day())

    assert observations.num_quotes == 4
    assert observations.implied_vol == pytest.approx(np.full(4, 0.2), abs=2.0e-5)
    assert observations.delta[0] - observations.delta[1] == pytest.approx(1.0)
    assert observations.delta[2] - observations.delta[3] == pytest.approx(1.0)
    assert np.all(observations.delta[observations.is_call] >= 0.0)
    assert np.all(observations.delta[~observations.is_call] <= 0.0)
    assert observations.half_spread == pytest.approx(np.full(4, 1.0e-4))
    assert np.all(np.isfinite(observations.nms))


def test_equal_date_weights_do_not_favor_quote_rich_dates() -> None:
    date_index = np.array([0, 1, 1, 1])
    weights = equal_date_sample_weights(date_index)

    assert np.sum(weights[date_index == 0]) == pytest.approx(2.0)
    assert np.sum(weights[date_index == 1]) == pytest.approx(2.0)
    assert np.mean(weights) == pytest.approx(1.0)


def test_spread_cost_api_predicts_price_proportional_cost() -> None:
    estimator = DummyRegressor(strategy="constant", constant=0.02)
    estimator.fit(np.zeros((2, 3)), np.array([0.02, 0.02]))
    model = FittedSpreadCostModel(
        name="constant",
        estimator=estimator,
        feature_names=("vega_over_sqrt_tte", "delta", "nms"),
        proportional_price_floor=0.01,
    )
    common = {
        "vega": np.array([0.1, 0.2]),
        "sqrt_tte": np.array([0.2, 0.3]),
        "nms": np.array([0.0, 1.0]),
        "log_strike": np.array([0.0, 0.1]),
        "delta": np.array([0.5, -0.2]),
    }

    assert model.predict_half_spread(**common) == pytest.approx([0.02, 0.02])
    assert model.predict_proportional_cost(
        np.array([0.1, 0.005]), **common
    ) == pytest.approx([0.2, 2.0])


def test_fitted_linear_cost_runs_in_torch_with_gradients() -> None:
    estimator = DummyRegressor(strategy="constant", constant=0.02)
    estimator.fit(np.zeros((2, 3)), np.array([0.02, 0.02]))
    # DummyRegressor is rejected because only an actual fitted linear formula
    # has meaningful feature derivatives in the Torch simulator.
    rejected = FittedSpreadCostModel(
        name="constant",
        estimator=estimator,
        feature_names=("vega_over_sqrt_tte", "delta", "nms"),
    )
    inputs = {
        "vega": torch.tensor([0.1], requires_grad=True),
        "sqrt_tte": torch.tensor([0.2]),
        "nms": torch.tensor([0.0]),
        "log_strike": torch.tensor([0.0]),
        "delta": torch.tensor([0.5]),
    }
    with pytest.raises(ValueError, match="only by the linear model"):
        rejected.predict_half_spread_torch(**inputs)

    linear = LinearRegression().fit(
        np.array([[0.5, 0.5, 0.0], [1.0, -0.5, 1.0]]),
        np.array([0.02, 0.04]),
    )
    model = FittedSpreadCostModel(
        name="linear",
        estimator=linear,
        feature_names=("vega_over_sqrt_tte", "delta", "nms"),
    )
    result = model.predict_half_spread_torch(**inputs)
    result.sum().backward()

    assert result.shape == (1,)
    assert inputs["vega"].grad is not None
    assert torch.all(torch.isfinite(inputs["vega"].grad))


def test_feed_forward_cost_is_tick_floored_and_differentiable() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = SpreadFeedForwardConfig(
        label="test",
        network=SpreadFeedForwardConfig().network,
    )
    model = TorchSpreadCostModel(
        env,
        config=config,
        feature_mean=torch.zeros(6, dtype=env.fdtype),
        feature_scale=torch.ones(6, dtype=env.fdtype),
        log_target_mean=torch.tensor(-20.0, dtype=env.fdtype),
        log_target_scale=torch.tensor(1.0, dtype=env.fdtype),
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    option_price = torch.tensor([2.0e-4, 2.0e-3], dtype=env.fdtype)
    spot = torch.tensor([5_000.0, 5_000.0], dtype=env.fdtype)
    vega = torch.tensor([0.1, 0.2], dtype=env.fdtype, requires_grad=True)
    common = {
        "vega": vega,
        "sqrt_tte": torch.tensor([0.2, 0.3], dtype=env.fdtype),
        "nms": torch.tensor([0.0, 1.0], dtype=env.fdtype),
        "log_strike": torch.tensor([0.0, 0.1], dtype=env.fdtype),
        "delta": torch.tensor([0.5, -0.2], dtype=env.fdtype),
    }
    half_spread = model.predict_half_spread_torch(
        option_price,
        spot,
        **common,
    )
    expected = torch.tensor([0.025, 0.05], dtype=env.fdtype) / spot
    torch.testing.assert_close(half_spread, expected)

    # Move the raw prediction above the tick floor to prove gradients flow
    # through the feed-forward model and its option-state features.
    with torch.no_grad():
        model.log_target_mean.fill_(-7.0)
    result = model.predict_proportional_cost_torch(
        option_price,
        spot=spot,
        **common,
    )
    result.sum().backward()
    assert vega.grad is not None
    assert torch.all(torch.isfinite(vega.grad))


def test_feed_forward_cost_checkpoint_round_trip(tmp_path: Path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = SpreadFeedForwardConfig(label="checkpoint-test")
    model = TorchSpreadCostModel(
        env,
        config=config,
        feature_mean=torch.arange(6, dtype=env.fdtype),
        feature_scale=torch.arange(1, 7, dtype=env.fdtype),
        log_target_mean=torch.tensor(-8.0, dtype=env.fdtype),
        log_target_scale=torch.tensor(0.5, dtype=env.fdtype),
    )
    path = tmp_path / "spread.pt"
    save_torch_spread_cost_model(model, path, metadata={"split": "test"})
    restored, metadata = load_torch_spread_cost_model(path, env=env)

    assert restored.config == config
    assert metadata == {"split": "test"}
    for expected, actual in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)


def test_regression_metrics_separate_raw_and_equal_date_measures() -> None:
    target = np.zeros(4)
    prediction = np.array([1.0, 0.0, 0.0, 0.0])
    result = regression_metrics(target, prediction, np.array([0, 1, 1, 1]))

    assert result["mae"] == pytest.approx(0.25)
    assert result["date_balanced_mae"] == pytest.approx(0.5)
    assert result["date_balanced_rmse"] == pytest.approx(np.sqrt(0.5))


def test_phase3_spread_date_source_rejects_2026(tmp_path: Path) -> None:
    path = tmp_path / "dates.json"
    path.write_text(
        json.dumps({"market_dates": ["2025-08-29", "2026-01-02"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only 2022"):
        _dates_from_report(path)
