from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.adapter import (
    _daily_pricing_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    dlv_digital_greeks,
    solve_delta_factor_hedge,
)


def _factor_fit() -> RollingDlvFactorFit:
    training_origins = (date(2022, 12, 27), date(2022, 12, 28))
    training_endpoints = (date(2022, 12, 28), date(2022, 12, 29))
    loadings = np.asarray([[[0.02, -0.01]]])
    return RollingDlvFactorFit(
        as_of_date=training_endpoints[-1],
        training_origin_dates=training_origins,
        training_endpoint_dates=training_endpoints,
        conditional_intercept=np.zeros((1, 1)),
        conditional_slope=np.zeros((1, 1)),
        centered_residuals=np.zeros((2, 1, 1)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        daily_loadings=loadings,
        daily_factor_variances=np.sum(loadings**2, axis=(0, 1)),
        factor_scores=np.zeros((2, 2)),
        explained_variance_fraction=1.0,
        interpolation_clip_fraction=0.0,
        forward_tail_extrapolation_fraction=0.0,
    )


def test_front_digital_has_zero_future_dlv_factor_vega() -> None:
    day = synthetic_one_day_transition().origin
    greeks = dlv_digital_greeks(
        day,
        business_dtes=np.asarray([1.0]),
        strikes_over_forward=np.asarray([1.0]),
        is_call=np.asarray([True]),
        discount_factors=np.asarray([1.0]),
        factors=_factor_fit(),
    )
    assert greeks.model_prices.shape == (1,)
    assert greeks.spot_deltas[0] > 0.0
    assert greeks.factor_vegas == pytest.approx(np.zeros((1, 2)))


def test_future_spot_bump_is_one_percent_with_legacy_override() -> None:
    assert SanosLvConfig().relative_spot_step == pytest.approx(0.01)
    assert SanosLvConfig(relative_spot_step=1.0e-4).relative_spot_step == pytest.approx(
        1.0e-4
    )
    day = synthetic_one_day_transition().origin
    query = {
        "business_dtes": np.asarray([1.0]),
        "strikes_over_forward": np.asarray([1.0]),
        "is_call": np.asarray([True]),
        "discount_factors": np.asarray([1.0]),
        "factors": _factor_fit(),
    }
    default_greeks = dlv_digital_greeks(day, **query)
    explicit_greeks = dlv_digital_greeks(day, **query, relative_spot_step=0.01)
    assert default_greeks.spot_deltas == pytest.approx(explicit_greeks.spot_deltas)


def test_zero_backbone_ablation_only_changes_daily_pricing_view() -> None:
    day = synthetic_one_day_transition().origin
    assert day.backbone_annual_volatility > 0.0
    assert _daily_pricing_day(day, config=SanosLvConfig()) is day
    pricing_day = _daily_pricing_day(
        day,
        config=SanosLvConfig(daily_backbone_annual_volatility=0.0),
    )
    assert pricing_day.backbone_annual_volatility == 0.0
    assert day.backbone_annual_volatility > 0.0
    with pytest.raises(ValueError):
        SanosLvConfig(daily_backbone_annual_volatility=-0.01)


def test_cost_weighted_hedge_neutralizes_delta_and_factor_exposure() -> None:
    target = DlvGreeks(
        model_prices=np.asarray([0.1]),
        spot_deltas=np.asarray([0.7]),
        factor_vegas=np.asarray([[0.2, -0.1]]),
    )
    options = DlvGreeks(
        model_prices=np.asarray([0.03, 0.04, 0.02]),
        spot_deltas=np.asarray([0.4, -0.2, 0.1]),
        factor_vegas=np.asarray([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]]),
    )
    hedge = solve_delta_factor_hedge(
        target=target,
        hedge_options=options,
        option_cost_weights=np.asarray([0.002, 0.004, 0.006]),
        residual_tolerance=1.0e-10,
        position_cap=1_000.0,
    )
    assert hedge.factor_residual == pytest.approx(np.zeros(2), abs=1.0e-12)
    total_delta = (
        target.spot_deltas[0]
        + hedge.spot_action
        + hedge.option_actions @ options.spot_deltas
    )
    assert total_delta == pytest.approx(0.0, abs=1.0e-12)
    assert hedge.delta_residual == pytest.approx(0.0, abs=1.0e-12)
    assert hedge.cap_hits == 0


def test_factor_hedge_scales_small_equality_rows_before_optimization() -> None:
    target = DlvGreeks(
        model_prices=np.asarray([0.1]),
        spot_deltas=np.asarray([0.0]),
        factor_vegas=np.asarray([[5.0e-8, -3.0e-8]]),
    )
    options = DlvGreeks(
        model_prices=np.asarray([0.03, 0.04]),
        spot_deltas=np.asarray([0.0, 0.0]),
        factor_vegas=np.asarray([[1.0e-8, 0.0], [0.0, 1.0e-8]]),
    )
    hedge = solve_delta_factor_hedge(
        target=target,
        hedge_options=options,
        option_cost_weights=np.asarray([0.002, 0.004]),
        residual_tolerance=1.0e-12,
        position_cap=1_000.0,
    )
    assert hedge.option_actions == pytest.approx(np.asarray([-5.0, 3.0]))
    assert hedge.factor_residual == pytest.approx(np.zeros(2), abs=1.0e-14)


def test_factor_hedge_minimizes_turnover_from_carried_positions() -> None:
    target = DlvGreeks(
        model_prices=np.asarray([0.1]),
        spot_deltas=np.asarray([0.0]),
        factor_vegas=np.asarray([[1.0]]),
    )
    options = DlvGreeks(
        model_prices=np.asarray([0.03, 0.04]),
        spot_deltas=np.asarray([0.0, 0.0]),
        factor_vegas=np.asarray([[1.0], [1.0]]),
    )
    hedge = solve_delta_factor_hedge(
        target=target,
        hedge_options=options,
        option_cost_weights=np.asarray([0.001, 0.01]),
        residual_tolerance=1.0e-10,
        position_cap=100.0,
        previous_option_actions=np.asarray([0.0, -1.0]),
    )
    assert hedge.option_actions == pytest.approx(np.asarray([0.0, -1.0]))
    assert hedge.factor_residual == pytest.approx(np.zeros(1), abs=1.0e-12)
