from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.data.pricing import price_dlv_surface, price_sanos_surface
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalTarget,
    build_daily_digital_sample,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    RelativeOptionCoordinate,
    relative_return_observation,
)


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


def _market_day(*, trading_date: date, expiry_dates: tuple[date, ...]) -> PublicMarketDay:
    expiry_count = len(expiry_dates)
    dtes = np.arange(1, expiry_count + 1, dtype=np.float64)
    expiry_indices = np.repeat(np.arange(expiry_count), 2)
    nms = np.tile(np.asarray([-0.5, 0.5]), expiry_count)
    strikes = np.asarray(
        NMS.from_nms(
            nms,
            np.repeat(dtes, 2) / DAYS_PER_YEAR,
            vol=0.2,
            eps_vol=0.2,
        )
    )
    is_call = np.tile(np.asarray([False, True]), expiry_count)
    support = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4])
    density = np.asarray([0.0, 0.25, 0.5, 0.25, 0.0])
    minimum = 0.01**2
    total_variances = 0.2**2 * dtes / 252.0
    model_supports = tuple(support.copy() for _ in range(expiry_count))
    model_densities = tuple(density.copy() for _ in range(expiry_count))
    sanos_prices = price_sanos_surface(
        query_expiries_y=np.repeat(dtes, 2) / DAYS_PER_YEAR,
        query_strikes=strikes,
        is_call=is_call,
        model_expiries_y=dtes / DAYS_PER_YEAR,
        atm_variances=total_variances,
        support_strikes=model_supports,
        densities=model_densities,
        smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
    )
    dlv_raw = np.log(np.full((expiry_count, 3), 0.2**2 - minimum))
    dlv_prices = price_dlv_surface(
        query_expiries_y=np.repeat(dtes, 2) / DAYS_PER_YEAR,
        query_strikes=strikes,
        is_call=is_call,
        model_expiries_y=dtes / DAYS_PER_YEAR,
        support_strikes=support,
        raw_log_excess_dlvsq=dlv_raw,
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
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
        standardized_log_moneyness=nms,
        in_recommended_hedge_nms_population=np.ones(2 * expiry_count, dtype=bool),
        is_call=is_call,
        half_spreads_over_spot=np.full(2 * expiry_count, 0.001),
        sanos_prices_over_discounted_forward=sanos_prices,
        dlv_prices_over_discounted_forward=dlv_prices,
        sanos_model_expiries_y=dtes / 252.0,
        sanos_support_strikes=model_supports,
        sanos_kernel_total_variances=total_variances,
        sanos_densities=model_densities,
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=dtes / 252.0,
        dlv_support_strikes=support,
        dlv_raw_log_excess_dlvsq=dlv_raw,
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )


def test_relative_replay_uses_dlv_and_not_sanos_history() -> None:
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
    target = DailyDigitalTarget(
        maturity_business_days=3,
        right=OptionRight.PUT,
        nms=-0.5,
    )
    current = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=target,
    )
    changed_dlv_origin = replace(
        origin,
        dlv_raw_log_excess_dlvsq=np.log(
            np.full_like(
                origin.dlv_raw_log_excess_dlvsq,
                0.35**2 - origin.minimum_dlvsq,
            )
        ),
    )
    changed_dlv_endpoint = replace(
        endpoint,
        dlv_raw_log_excess_dlvsq=np.log(
            np.full_like(
                endpoint.dlv_raw_log_excess_dlvsq,
                0.35**2 - endpoint.minimum_dlvsq,
            )
        ),
    )
    current_dlv_changed = build_daily_digital_sample(
        _SyntheticPanel(
            transition=replace(transition, origin=changed_dlv_origin),
            endpoint=changed_dlv_endpoint,
        ),
        origin_date=origin_date,
        target=target,
    )
    assert current_dlv_changed.market_price == current.market_price
    assert current_dlv_changed.terminal_value == current.terminal_value
    np.testing.assert_allclose(
        [quote.mark for quote in current_dlv_changed.hedge_quotes],
        [quote.mark for quote in current.hedge_quotes],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        current_dlv_changed.hedge_returns,
        current.hedge_returns,
        rtol=0.0,
        atol=0.0,
    )
    coordinates = tuple(
        RelativeOptionCoordinate.from_quote(quote) for quote in current.hedge_quotes
    )
    observation = relative_return_observation(
        transition,
        endpoint=endpoint,
        coordinates=coordinates,
        targets=(target,),
    )
    changed_sanos_origin = replace(
        origin,
        sanos_prices_over_discounted_forward=(
            2.0 * origin.sanos_prices_over_discounted_forward
        ),
        sanos_kernel_total_variances=(
            1.5 * origin.sanos_kernel_total_variances
        ),
    )
    changed_sanos_endpoint = replace(
        endpoint,
        sanos_prices_over_discounted_forward=(
            2.0 * endpoint.sanos_prices_over_discounted_forward
        ),
        sanos_kernel_total_variances=(
            1.5 * endpoint.sanos_kernel_total_variances
        ),
    )
    sanos_changed = relative_return_observation(
        replace(transition, origin=changed_sanos_origin),
        endpoint=changed_sanos_endpoint,
        coordinates=coordinates,
        targets=(target,),
    )
    np.testing.assert_allclose(
        sanos_changed.instrument_returns,
        observation.instrument_returns,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        sanos_changed.target_returns,
        observation.target_returns,
        rtol=0.0,
        atol=0.0,
    )
    dlv_changed = relative_return_observation(
        transition,
        endpoint=replace(
            endpoint,
            dlv_raw_log_excess_dlvsq=np.log(
                np.full_like(
                    endpoint.dlv_raw_log_excess_dlvsq,
                    0.35**2 - endpoint.minimum_dlvsq,
                )
            ),
        ),
        coordinates=coordinates,
        targets=(target,),
    )
    assert not np.allclose(
        dlv_changed.instrument_returns,
        observation.instrument_returns,
        rtol=0.0,
        atol=1.0e-10,
    )
    assert not np.allclose(
        dlv_changed.target_returns,
        observation.target_returns,
        rtol=0.0,
        atol=1.0e-10,
    )
