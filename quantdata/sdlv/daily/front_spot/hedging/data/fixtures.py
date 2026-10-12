"""Small deterministic public-state fixtures for CPU model tests."""

from __future__ import annotations

from datetime import date

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)


def synthetic_one_day_transition() -> OneDayTransition:
    """Return a minimal martingale SANOS state and one ATM call book."""
    origin = PublicMarketDay(
        trading_date=date(2025, 1, 2),
        expiry_dates=(date(2025, 1, 3),),
        expiries_y=np.asarray([1.0 / 252.0]),
        business_dtes=np.asarray([1.0]),
        is_am_settlement=np.asarray([False]),
        forward_over_spot=np.asarray([1.0]),
        discount_factors=np.asarray([1.0]),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.asarray([0.2]),
        expiry_indices=np.asarray([0]),
        strikes_over_forward=np.asarray([1.0]),
        standardized_log_moneyness=np.asarray([0.0]),
        in_recommended_hedge_nms_population=np.asarray([True]),
        is_call=np.asarray([True]),
        half_spreads_over_spot=np.asarray([0.001]),
        sanos_prices_over_discounted_forward=np.asarray([0.05]),
        dlv_prices_over_discounted_forward=np.asarray([0.05]),
        sanos_model_expiries_y=np.asarray([1.0 / 252.0]),
        sanos_support_strikes=(np.asarray([0.8, 1.0, 1.2]),),
        sanos_kernel_total_variances=np.asarray([0.2**2 / 252.0]),
        sanos_densities=(np.asarray([0.25, 0.5, 0.25]),),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=np.asarray([1.0, 2.0]) / 252.0,
        dlv_support_strikes=np.asarray([0.8, 1.0, 1.2]),
        dlv_raw_log_excess_dlvsq=np.log(
            np.asarray([[0.2**2 - 0.01**2], [0.2**2 - 0.01**2]])
        ),
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
    return OneDayTransition(
        origin=origin,
        next_date=date(2025, 1, 3),
        forward_adjusted_spot_return=0.01,
        discount_factor=1.0,
        forward_over_spot=1.0,
    )


def synthetic_weekly_market_day() -> PublicMarketDay:
    """Return a minimal five-expiry SANOS-DLV state for weekly pricing tests."""
    model_dtes = np.arange(1, 6, dtype=np.float64)
    expiry_dates = (
        date(2025, 1, 7),
        date(2025, 1, 8),
        date(2025, 1, 9),
        date(2025, 1, 10),
        date(2025, 1, 13),
    )
    sanos_support = np.asarray([0.8, 1.0, 1.2])
    sanos_density = np.asarray([0.1, 0.8, 0.1])
    dlv_support = np.asarray([0.6, 0.85, 1.0, 1.2, 1.6])
    minimum = 0.01**2
    expiry_indices = np.repeat(np.arange(5, dtype=np.int64), 2)
    strikes = np.tile(np.asarray([0.9, 1.1]), 5)
    is_call = np.tile(np.asarray([False, True]), 5)
    return PublicMarketDay(
        trading_date=date(2025, 1, 6),
        expiry_dates=expiry_dates,
        expiries_y=model_dtes / 252.0,
        business_dtes=model_dtes,
        is_am_settlement=np.zeros(5, dtype=bool),
        forward_over_spot=np.ones(5),
        discount_factors=np.full(5, 0.999),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.full(5, 0.2),
        expiry_indices=expiry_indices,
        strikes_over_forward=strikes,
        standardized_log_moneyness=np.tile(np.asarray([-0.5, 0.5]), 5),
        in_recommended_hedge_nms_population=np.ones(10, dtype=bool),
        is_call=is_call,
        half_spreads_over_spot=np.full(10, 0.001),
        sanos_prices_over_discounted_forward=np.full(10, 0.03),
        dlv_prices_over_discounted_forward=np.full(10, 0.03),
        sanos_model_expiries_y=model_dtes / 252.0,
        sanos_support_strikes=tuple(sanos_support.copy() for _ in model_dtes),
        sanos_kernel_total_variances=0.2**2 * model_dtes / 252.0,
        sanos_densities=tuple(sanos_density.copy() for _ in model_dtes),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=model_dtes / 252.0,
        dlv_support_strikes=dlv_support,
        dlv_raw_log_excess_dlvsq=np.log(
            np.full((5, 3), 0.2**2 - minimum)
        ),
        minimum_dlvsq=minimum,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
