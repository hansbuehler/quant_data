from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from cdxcore.bs import bs
from scipy.special import ndtr

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.contracts import OptionContract, OptionQuote, OptionRight
from quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.greeks import (
    BlackGreeks,
    black_digital_greeks,
    black_option_greeks,
    interpolate_vega_hedge,
    strike_interpolation,
)


def _quote(strike: float, *, business_dte: float = 1.0) -> OptionQuote:
    return OptionQuote(
        trading_date=date(2023, 1, 17),
        contract=OptionContract(
            contract_id=f"test-{strike}",
            expiry_date=date(2023, 1, 18),
            fixed_strike=strike,
            right=OptionRight.PUT if strike < 1.0 else OptionRight.CALL,
        ),
        business_dte=business_dte,
        atm_annual_vol=0.2,
        strike_over_forward=strike,
        standardized_log_moneyness=0.0,
        mark=0.01,
        half_spread=0.001,
    )


def test_black_digital_vega_matches_finite_difference() -> None:
    strike = 1.01
    vol = 0.2
    root_time = np.sqrt(5.0 / DAYS_PER_YEAR)
    step = 1.0e-5

    def price(annual_vol: float, spot: float = 1.0) -> float:
        d2 = np.log(spot / strike) / (annual_vol * root_time)
        d2 -= 0.5 * annual_vol * root_time
        return float(ndtr(d2))

    greeks = black_digital_greeks(
        strike_over_forward=strike,
        annual_vol=vol,
        business_dte=5.0,
        discount_factor=1.0,
        right=OptionRight.CALL,
    )
    assert greeks.price == pytest.approx(price(vol))
    assert greeks.vega == pytest.approx((price(vol + step) - price(vol - step)) / (2 * step), rel=1e-7)
    assert greeks.delta == pytest.approx((price(vol, 1 + step) - price(vol, 1 - step)) / (2 * step), rel=1e-7)


def test_black_option_greeks_and_strike_distance_hedge() -> None:
    quotes = (_quote(0.99), _quote(1.01))
    bracket = strike_interpolation(
        quotes,
        fixed_strike=0.995,
        target_business_dte=1.0,
    )
    assert tuple(index for index, _ in bracket) == (0, 1)
    assert tuple(weight for _, weight in bracket) == pytest.approx((0.75, 0.25))
    quote_greeks = tuple(
        black_option_greeks(quote, forward_over_spot=1.0, discount_factor=1.0)
        for quote in quotes
    )
    assert all(greeks.vega > 0.0 for greeks in quote_greeks)
    call = quote_greeks[1]
    root_time = np.sqrt(1.0 / DAYS_PER_YEAR)
    step = 1.0e-5

    def vanilla_price(spot: float, annual_vol: float) -> float:
        return float(
            spot * bs.price(
                k=1.01 / spot,
                vol=annual_vol,
                sqrtT=root_time,
                is_call=True,
            )
        )

    assert call.delta == pytest.approx(
        (vanilla_price(1 + step, 0.2) - vanilla_price(1 - step, 0.2)) / (2 * step),
        rel=1e-6,
    )
    assert call.vega == pytest.approx(
        (vanilla_price(1.0, 0.2 + step) - vanilla_price(1.0, 0.2 - step)) / (2 * step),
        rel=1e-6,
    )
    target = BlackGreeks(price=0.4, delta=4.0, vega=-0.1)
    spot, options, delta_residual, vega_residual, cap_hits = interpolate_vega_hedge(
        target=target,
        quote_greeks=quote_greeks,
        bracket=bracket,
        claim_quantity=2.0,
        position_cap=1000.0,
        minimum_vega=1.0e-12,
    )
    assert options[0] / options[1] == pytest.approx(3.0)
    assert spot == pytest.approx(-2.0 * target.delta - sum(
        option * greeks.delta for option, greeks in zip(options, quote_greeks, strict=True)
    ))
    assert delta_residual == pytest.approx(0.0, abs=1.0e-12)
    assert vega_residual == pytest.approx(0.0, abs=1.0e-12)
    assert cap_hits == 0


def test_strike_interpolation_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="outside"):
        strike_interpolation(
            (_quote(0.99), _quote(1.01)),
            fixed_strike=1.02,
            target_business_dte=1.0,
        )


def test_strike_interpolation_uses_first_later_expiry_with_a_bracket() -> None:
    quotes = (
        _quote(0.99),
        _quote(1.01, business_dte=2.0),
        _quote(0.98, business_dte=2.0),
    )
    bracket = strike_interpolation(
        quotes,
        fixed_strike=1.0,
        target_business_dte=1.0,
    )
    assert tuple(index for index, _ in bracket) == (2, 1)
    assert tuple(weight for _, weight in bracket) == pytest.approx((1 / 3, 2 / 3))
