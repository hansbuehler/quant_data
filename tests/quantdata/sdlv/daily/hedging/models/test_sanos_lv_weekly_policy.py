from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyFitInstruction,
    WeeklyTarget,
    WeeklyTargetKind,
    build_weekly_claim,
    weekly_targets,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyDecisionState,
    WeeklyPriceRecord,
)
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_weekly_market_day,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_adapter import (
    SanosLvWeeklyAdapter,
    SanosLvWeeklyPolicy,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_conditional import (
    conditional_weekly_claim_greeks,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    price_weekly_sanos_lv_targets,
)


def _factor_fit() -> RollingDlvFactorFit:
    day = synthetic_weekly_market_day()
    state_shape = day.dlvsq[1:].shape
    loadings = np.zeros((*state_shape, 2), dtype=np.float64)
    loadings[..., 0] = np.linspace(0.002, 0.008, np.prod(state_shape)).reshape(
        state_shape
    )
    loadings[..., 1] = np.tile(
        np.asarray([-0.006, 0.0, 0.006]),
        (state_shape[0], 1),
    )
    training_origins = (date(2025, 1, 2), date(2025, 1, 3))
    training_endpoints = (date(2025, 1, 3), date(2025, 1, 6))
    return RollingDlvFactorFit(
        as_of_date=day.trading_date,
        training_origin_dates=training_origins,
        training_endpoint_dates=training_endpoints,
        conditional_intercept=np.zeros(state_shape),
        conditional_slope=np.zeros(state_shape),
        centered_residuals=np.zeros((2, *state_shape)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        daily_loadings=loadings,
        daily_factor_variances=np.sum(loadings**2, axis=(0, 1)),
        factor_scores=np.zeros((2, 2)),
        explained_variance_fraction=1.0,
        interpolation_clip_fraction=0.0,
        forward_tail_extrapolation_fraction=0.0,
    )


def test_conditional_weekly_prices_match_inception_pricer() -> None:
    day = synthetic_weekly_market_day()
    factors = _factor_fit()
    expected = price_weekly_sanos_lv_targets(day).by_label()
    for target in weekly_targets():
        claim = build_weekly_claim(day, target=target)
        greeks = conditional_weekly_claim_greeks(
            day,
            claim=claim,
            decision_index=0,
            realized_inception_forward_path=np.ones(1),
            inception_forward_over_episode_spot=np.ones(6),
            spot_over_episode_origin=1.0,
            factors=factors,
        )
        assert greeks.model_prices[0] == pytest.approx(
            expected[target.label],
            rel=0.0,
            abs=2.0e-12,
        )
        assert np.all(np.isfinite(greeks.spot_deltas))
        assert np.all(np.isfinite(greeks.factor_vegas))


def test_conditional_barrier_remains_knocked_out() -> None:
    day = synthetic_weekly_market_day()
    target = WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT)
    greeks = conditional_weekly_claim_greeks(
        day,
        claim=build_weekly_claim(day, target=target),
        decision_index=2,
        realized_inception_forward_path=np.asarray([1.0, 0.94, 1.01]),
        inception_forward_over_episode_spot=np.ones(6),
        spot_over_episode_origin=1.01,
        factors=_factor_fit(),
    )
    assert greeks.model_prices == pytest.approx(np.zeros(1))
    assert greeks.spot_deltas == pytest.approx(np.zeros(1))
    assert greeks.factor_vegas == pytest.approx(np.zeros((1, 2)))


def test_conditional_barrier_latest_observed_hit_has_zero_greeks() -> None:
    day = synthetic_weekly_market_day()
    target = WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT)
    latest_close = 0.94995
    greeks = conditional_weekly_claim_greeks(
        day,
        claim=build_weekly_claim(day, target=target),
        decision_index=2,
        realized_inception_forward_path=np.asarray([1.0, 1.0, latest_close]),
        inception_forward_over_episode_spot=np.ones(6),
        spot_over_episode_origin=latest_close,
        factors=_factor_fit(),
    )
    assert greeks.model_prices == pytest.approx(np.zeros(1))
    assert greeks.spot_deltas == pytest.approx(np.zeros(1))
    assert greeks.factor_vegas == pytest.approx(np.zeros((1, 2)))


def test_conditional_variance_delta_does_not_revalue_accrued_returns() -> None:
    day = synthetic_weekly_market_day()
    claim = build_weekly_claim(
        day,
        target=WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
    )
    common = dict(
        day=day,
        claim=claim,
        decision_index=2,
        inception_forward_over_episode_spot=np.ones(6),
        spot_over_episode_origin=1.0,
        factors=_factor_fit(),
    )
    flat_prefix = conditional_weekly_claim_greeks(
        realized_inception_forward_path=np.asarray([1.0, 1.0, 1.0]),
        **common,
    )
    accrued_prefix = conditional_weekly_claim_greeks(
        realized_inception_forward_path=np.asarray([1.0, 0.9, 1.0]),
        **common,
    )
    assert accrued_prefix.model_prices[0] > flat_prefix.model_prices[0]
    assert accrued_prefix.spot_deltas == pytest.approx(
        flat_prefix.spot_deltas,
        abs=1.0e-10,
    )
    assert accrued_prefix.factor_vegas == pytest.approx(
        flat_prefix.factor_vegas,
        abs=1.0e-12,
    )


def test_weekly_adapter_returns_zero_for_expired_contracts() -> None:
    day = synthetic_weekly_market_day()
    contracts = tuple(
        quote.contract
        for quote in day.hedge_snapshot(
            maximum_business_dte=5.0,
            episode_origin_date=day.trading_date,
        ).option_quotes
    )
    instruction = WeeklyFitInstruction(
        schedule=HedgeSchedule.FROZEN_WEEKLY,
        model="sanos-lv",
        measure="Q",
        target_label="digital-call-5d-atm",
        claim_quantity=1.0,
        model_config_identity="model",
        policy_config_identity="policy",
        episode_origin_date=day.trading_date,
        decision_date=day.trading_date,
        fit_as_of_date=day.trading_date,
        training_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )
    snapshot = day.reprice_contracts(contracts, spot_over_episode_origin=1.0)
    quote_map = snapshot.quote_by_id()
    marks = np.asarray([quote_map[contract.contract_id].mark for contract in contracts])
    spreads = np.asarray(
        [quote_map[contract.contract_id].half_spread for contract in contracts]
    )
    alive = np.ones(len(contracts), dtype=bool)
    alive[:2] = False
    state = WeeklyDecisionState(
        instruction=instruction,
        decision_index=0,
        next_session_date=day.expiry_dates[0],
        market_day=day,
        contracts=contracts,
        spot_over_episode_origin=1.0,
        inception_forward_over_episode_spot=np.ones(6),
        inception_forward_path=np.ones(1),
        rolling_forward_path=np.ones(1),
        discount_to_origin=1.0,
        option_marks_pv=np.where(alive, marks, 0.0),
        option_half_spreads_pv=np.where(alive, spreads, 0.0),
        option_alive=alive,
        previous_spot_position=0.0,
        previous_option_positions=np.zeros(len(contracts)),
    )
    claim = build_weekly_claim(
        day,
        target=WeeklyTarget(
            kind=WeeklyTargetKind.DIGITAL,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    factors = _factor_fit()
    fit = SanosLvWeeklyPolicy(
        market_fit_identity=instruction.market_fit_identity,
        policy_fit_identity=instruction.policy_fit_identity,
        prices=WeeklyPriceRecord(
            common_market_price_per_fit_spot=claim.market_price,
            q_expectation_per_fit_spot=claim.market_price,
            buyer_indifference_per_fit_spot=None,
            seller_indifference_per_fit_spot=None,
            continuation_price_per_fit_spot=claim.market_price,
        ),
        factors=factors,
        claim=claim,
    )
    action = SanosLvWeeklyAdapter().action(fit, state=state)
    assert action.option_positions[:2] == pytest.approx(np.zeros(2))
    assert np.all(np.isfinite(action.option_positions))
    assert np.isfinite(action.spot_position)
