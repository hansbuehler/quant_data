from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionRight
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    build_daily_digital_sample,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)


def test_daily_digital_catalog_contains_all_agreed_targets() -> None:
    targets = daily_digital_targets()
    assert len(targets) == 20
    assert len({target.label for target in targets}) == 20
    for maturity in (1, 3):
        selected = tuple(
            target for target in targets if target.maturity_business_days == maturity
        )
        puts = tuple(target.nms for target in selected if target.right is OptionRight.PUT)
        calls = tuple(
            target.nms for target in selected if target.right is OptionRight.CALL
        )
        assert puts == (0.0, -1.0, -0.75, -0.5, -0.25)
        assert calls == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_daily_claim_quantities_require_exact_catalog() -> None:
    targets = daily_digital_targets()
    quantities = {target.label: 0.1 + index for index, target in enumerate(targets)}
    aligned = daily_claim_quantity_vector(targets, quantities)
    np.testing.assert_allclose(
        aligned,
        tuple(quantities[target.label] for target in targets),
    )
    with pytest.raises(ValueError, match="exactly match"):
        daily_claim_quantity_vector(targets, {targets[0].label: 1.0})


@dataclass(frozen=True)
class _SyntheticPanel:
    transition: OneDayTransition
    endpoint: PublicMarketDay

    def one_day_transition(self, origin_date: date) -> OneDayTransition:
        assert origin_date == self.transition.origin.trading_date
        return self.transition

    def market_day(self, trading_date: date) -> PublicMarketDay:
        assert trading_date == self.endpoint.trading_date
        return self.endpoint


def _market_day(
    *,
    trading_date: date,
    expiry_dates: tuple[date, ...],
) -> PublicMarketDay:
    expiry_count = len(expiry_dates)
    dtes = np.arange(1, expiry_count + 1, dtype=np.float64)
    expiry_indices = np.repeat(np.arange(expiry_count), 2)
    is_call = np.tile(np.asarray([False, True]), expiry_count)
    strikes = np.tile(np.asarray([0.95, 1.05]), expiry_count)
    support = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4])
    density = np.asarray([0.0, 0.25, 0.5, 0.25, 0.0])
    minimum = 0.01**2
    model_dtes = np.arange(1, 6, dtype=np.float64)
    return PublicMarketDay(
        trading_date=trading_date,
        expiry_dates=expiry_dates,
        expiries_y=dtes / 252.0,
        business_dtes=dtes,
        is_am_settlement=np.zeros(expiry_count, dtype=bool),
        forward_over_spot=1.0 + 0.001 * dtes,
        discount_factors=1.0 - 0.001 * dtes,
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.full(expiry_count, 0.2),
        expiry_indices=expiry_indices,
        strikes_over_forward=strikes,
        standardized_log_moneyness=np.tile(
            np.asarray([-0.5, 0.5]), expiry_count
        ),
        in_recommended_hedge_nms_population=np.ones(2 * expiry_count, dtype=bool),
        is_call=is_call,
        half_spreads_over_spot=np.full(2 * expiry_count, 0.001),
        sanos_prices_over_discounted_forward=np.full(2 * expiry_count, 0.03),
        dlv_prices_over_discounted_forward=np.full(2 * expiry_count, 0.03),
        sanos_model_expiries_y=dtes / 252.0,
        sanos_support_strikes=tuple(support.copy() for _ in range(expiry_count)),
        sanos_kernel_total_variances=0.2**2 * dtes / 252.0,
        sanos_densities=tuple(density.copy() for _ in range(expiry_count)),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=model_dtes / 252.0,
        dlv_support_strikes=support,
        dlv_raw_log_excess_dlvsq=np.log(
            np.full((5, 3), 0.2**2 - minimum)
        ),
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )


def test_daily_sample_settles_one_day_and_marks_three_day_claim() -> None:
    origin_date = date(2022, 12, 27)
    endpoint_date = date(2022, 12, 28)
    origin = _market_day(
        trading_date=origin_date,
        expiry_dates=(
            endpoint_date,
            date(2022, 12, 29),
            date(2022, 12, 30),
            date(2023, 1, 3),
            date(2023, 1, 4),
        ),
    )
    endpoint = _market_day(
        trading_date=endpoint_date,
        expiry_dates=(
            date(2022, 12, 29),
            date(2022, 12, 30),
            date(2023, 1, 3),
            date(2023, 1, 4),
        ),
    )
    transition = OneDayTransition(
        origin=origin,
        next_date=endpoint_date,
        forward_adjusted_spot_return=0.01,
        discount_factor=0.999,
        forward_over_spot=1.001,
    )
    panel = _SyntheticPanel(transition=transition, endpoint=endpoint)

    one_day = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=DailyDigitalTarget(
            maturity_business_days=1,
            right=OptionRight.CALL,
            nms=0.0,
        ),
    )
    assert one_day.claim.strike == pytest.approx(1.001)
    assert one_day.terminal_value == pytest.approx(0.999)
    assert one_day.claim_return == pytest.approx(
        one_day.terminal_value - one_day.market_price
    )
    assert len(one_day.hedge_quotes) == 10
    assert np.count_nonzero(one_day.closing_half_spreads_pv == 0.0) == 2
    surviving_index = int(np.flatnonzero(one_day.closing_half_spreads_pv > 0.0)[0])
    option_actions = np.zeros(len(one_day.hedge_quotes))
    option_actions[surviving_index] = 1.0
    entry_only = evaluate_daily_digital_hedge(
        one_day,
        model="test",
        measure="Q",
        spot_action=1.0,
        option_actions=option_actions,
        model_price=one_day.market_price,
        objective_value=0.0,
        converged=True,
        cap_hits=0,
    )
    assert entry_only.transaction_cost == pytest.approx(
        one_day.hedge_quotes[surviving_index].half_spread
    )
    assert entry_only.diagnostics["formal_option_exit_cost"] == pytest.approx(
        one_day.closing_half_spreads_pv[surviving_index]
    )
    assert entry_only.diagnostics["formal_spot_exit_cost"] == 0.0

    three_day = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=DailyDigitalTarget(
            maturity_business_days=3,
            right=OptionRight.PUT,
            nms=-0.5,
        ),
    )
    assert three_day.claim.strike == pytest.approx(
        three_day.target_strike_over_forward * 1.003
    )
    assert 0.0 < three_day.terminal_value < 1.0
    assert three_day.claim_return == pytest.approx(
        three_day.terminal_value - three_day.market_price
    )

    result = evaluate_daily_digital_hedge(
        one_day,
        model="test",
        measure="Q",
        spot_action=0.0,
        option_actions=np.zeros(len(one_day.hedge_quotes)),
        model_price=one_day.market_price,
        objective_value=0.0,
        converged=True,
        cap_hits=0,
        claim_quantity=2.5,
    )
    assert result.claim_quantity == 2.5
    assert result.unhedged_pnl == pytest.approx(2.5 * one_day.claim_return)
    assert result.hedged_pnl == pytest.approx(2.5 * one_day.claim_return)
    assert result.pure_hedge_error == pytest.approx(2.5 * one_day.claim_return)
