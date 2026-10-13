from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantdata.hedging.contracts import OptionContract, OptionRight
from quantdata.sdlv.daily.front_spot.hedging.data import market_panel
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicMarketDay


def test_public_market_day_converts_spread_to_forward_units_and_excludes_am() -> None:
    day = PublicMarketDay(
        trading_date=date(2025, 1, 2),
        expiry_dates=(date(2025, 1, 3), date(2025, 1, 6)),
        expiries_y=np.asarray([1.0, 2.0]) / 252.0,
        business_dtes=np.asarray([1.0, 2.0]),
        is_am_settlement=np.asarray([False, True]),
        forward_over_spot=np.asarray([1.01, 1.02]),
        discount_factors=np.asarray([0.99, 0.98]),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.asarray([0.2, 0.2]),
        expiry_indices=np.asarray([0, 1]),
        strikes_over_forward=np.asarray([1.01, 1.01]),
        standardized_log_moneyness=np.asarray([0.5, 0.5]),
        in_recommended_hedge_nms_population=np.asarray([True, True]),
        is_call=np.asarray([True, True]),
        half_spreads_over_spot=np.asarray([0.001, 0.001]),
        sanos_prices_over_discounted_forward=np.asarray([0.02, 0.03]),
        dlv_prices_over_discounted_forward=np.asarray([0.02, 0.03]),
        sanos_model_expiries_y=np.asarray([1.0, 2.0]) / 252.0,
        sanos_support_strikes=(
            np.asarray([0.8, 1.0, 1.2]),
            np.asarray([0.8, 1.0, 1.2]),
        ),
        sanos_kernel_total_variances=np.asarray([0.04, 0.08]) / 252.0,
        sanos_densities=(
            np.asarray([0.25, 0.5, 0.25]),
            np.asarray([0.25, 0.5, 0.25]),
        ),
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
    snapshot = day.hedge_snapshot()
    assert len(snapshot.option_quotes) == 1
    quote = snapshot.option_quotes[0]
    assert quote.mark == pytest.approx(0.99 * 1.01 * 0.02)
    assert quote.half_spread == pytest.approx(0.001)
    assert quote.contract.fixed_strike == pytest.approx(1.01 * 1.01)
    assert day.dlvsq == pytest.approx(np.full((2, 1), 0.2**2))


def test_fixed_contract_repricing_preserves_identity_and_recovers_current_kf(
    monkeypatch,
) -> None:
    day = PublicMarketDay(
        trading_date=date(2025, 1, 2),
        expiry_dates=(date(2025, 1, 6),),
        expiries_y=np.asarray([2.0 / 252.0]),
        business_dtes=np.asarray([2.0]),
        is_am_settlement=np.asarray([False]),
        forward_over_spot=np.asarray([1.01]),
        discount_factors=np.asarray([0.99]),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.asarray([0.2]),
        expiry_indices=np.asarray([0, 0]),
        strikes_over_forward=np.asarray([1.0, 1.02]),
        standardized_log_moneyness=np.asarray([0.0, 0.7]),
        in_recommended_hedge_nms_population=np.asarray([True, True]),
        is_call=np.asarray([True, True]),
        half_spreads_over_spot=np.asarray([0.001, 0.0012]),
        sanos_prices_over_discounted_forward=np.asarray([0.02, 0.01]),
        dlv_prices_over_discounted_forward=np.asarray([0.02, 0.01]),
        sanos_model_expiries_y=np.asarray([2.0 / 252.0]),
        sanos_support_strikes=(np.asarray([0.8, 1.0, 1.2]),),
        sanos_kernel_total_variances=np.asarray([0.2**2 * 2.0 / 252.0]),
        sanos_densities=(np.asarray([0.25, 0.5, 0.25]),),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=np.asarray([2.0 / 252.0]),
        dlv_support_strikes=np.asarray([0.8, 1.0, 1.2]),
        dlv_raw_log_excess_dlvsq=np.log(np.asarray([[0.2**2 - 0.01**2]])),
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
    inception = day.hedge_snapshot()
    contract = inception.option_quotes[0].contract
    repriced = day.reprice_contracts(
        (contract,),
        spot_over_episode_origin=1.02,
    )
    assert repriced.option_quotes[0].contract is contract
    assert repriced.option_quotes[0].contract.contract_id == contract.contract_id
    assert repriced.option_quotes[0].strike_over_forward == pytest.approx(
        contract.fixed_strike / (1.02 * 1.01)
    )
    assert repriced.option_quotes[0].half_spread == pytest.approx(1.02 * 0.001)

    monkeypatch.setattr(
        market_panel,
        "price_sanos_surface",
        lambda **kwargs: np.asarray([-1.0e-10]),
    )
    numerical_floor = day.reprice_contracts(
        (contract,),
        spot_over_episode_origin=1.02,
    ).option_quotes[0]
    assert numerical_floor.mark > 0.0
    assert numerical_floor.mark < 2.0e-12
    assert numerical_floor.half_spread == numerical_floor.mark
    assert numerical_floor.synthetic_bid == 0.0

    monkeypatch.setattr(
        market_panel,
        "price_sanos_surface",
        lambda **kwargs: np.asarray([-1.0e-6]),
    )
    with pytest.raises(FloatingPointError, match="materially negative SANOS"):
        day.reprice_contracts((contract,), spot_over_episode_origin=1.02)


def test_synthetic_contract_repricing_interpolates_missing_expiry() -> None:
    day = PublicMarketDay(
        trading_date=date(2025, 1, 2),
        expiry_dates=(date(2025, 1, 3), date(2025, 1, 7)),
        expiries_y=np.asarray([1.0, 3.0]) / 252.0,
        business_dtes=np.asarray([1.0, 3.0]),
        is_am_settlement=np.asarray([False, False]),
        forward_over_spot=np.asarray([1.001, 1.003]),
        discount_factors=np.asarray([0.999, 0.997]),
        book_selection_epsilon_atm_vol=0.2,
        book_selection_atm_vols=np.asarray([0.2, 0.24]),
        expiry_indices=np.asarray([0, 1]),
        strikes_over_forward=np.asarray([1.0, 1.0]),
        standardized_log_moneyness=np.asarray([0.0, 0.0]),
        in_recommended_hedge_nms_population=np.asarray([True, True]),
        is_call=np.asarray([True, True]),
        half_spreads_over_spot=np.asarray([0.001, 0.003]),
        sanos_prices_over_discounted_forward=np.asarray([0.02, 0.04]),
        dlv_prices_over_discounted_forward=np.asarray([0.02, 0.04]),
        sanos_model_expiries_y=np.asarray([1.0, 3.0]) / 252.0,
        sanos_support_strikes=(
            np.asarray([0.8, 1.0, 1.2]),
            np.asarray([0.8, 1.0, 1.2]),
        ),
        sanos_kernel_total_variances=np.asarray([0.2**2, 0.24**2 * 3.0]) / 252.0,
        sanos_densities=(
            np.asarray([0.25, 0.5, 0.25]),
            np.asarray([0.25, 0.5, 0.25]),
        ),
        sanos_smoothness=0.25,
        one_minute_y=1.0 / (252.0 * 6.5 * 60.0),
        dlv_model_expiries_y=np.asarray([1.0, 2.0, 3.0]) / 252.0,
        dlv_support_strikes=np.asarray([0.8, 1.0, 1.2]),
        dlv_raw_log_excess_dlvsq=np.log(
            np.full((3, 1), 0.2**2 - 0.01**2)
        ),
        minimum_dlvsq=0.01**2,
        maximum_dlvsq=3.0**2,
        backbone_annual_volatility=0.05,
    )
    contract = OptionContract(
        contract_id="synthetic-missing-expiry",
        expiry_date=date(2025, 1, 6),
        fixed_strike=1.002,
        right=OptionRight.CALL,
    )
    quote = day.reprice_contracts(
        (contract,),
        spot_over_episode_origin=1.0,
    ).option_quotes[0]
    assert quote.business_dte == 2.0
    assert quote.strike_over_forward == pytest.approx(1.002 / 1.002)
    assert quote.half_spread == pytest.approx(0.002)
    assert quote.contract is contract
