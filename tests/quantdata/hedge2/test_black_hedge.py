"""The weighted European vega hedge must have its own added delta neutralized."""

import numpy as np
import pandas as pd
import pytest
from cdxcore.bs import bs
from scipy.special import ndtr

from quantdata.hedge2.black_hedge import black_digital_exposures, black_digital_hedge
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim


def test_black_bracketing_hedge_neutralizes_delta_and_vega() -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 5.]), forwards=np.array([100., 100.]),
                      reference_forwards=np.array([100., 100.]), discounts=np.ones(2))
    claim = Claim(label="digital", kind="digital", initial_spot=100., strike=100.2, terminal_day=1)
    book = pd.DataFrame({"bdte": [1., 1.], "strike": [100., 101.], "option_type": ["call", "call"]})
    positions = black_digital_hedge(claim=claim, curve=curve, book=book, target_vol=.2,
                                    option_vols=np.full(2, .2), target_side=1)
    assert abs(positions[2] / positions[1] - .25) < 1e-12

    def wealth(spot: float, vol: float) -> float:
        total_vol = vol / np.sqrt(252.)
        d2 = np.log(spot / claim.strike) / total_vol - .5 * total_vol
        option_values = spot * bs.price(k=book["strike"].to_numpy() / spot,
                                        vol=total_vol, is_call=True) / 100.
        return float(ndtr(d2) + positions[0] * spot / 100. + positions[1:] @ option_values)

    assert abs((wealth(100.0001, .2) - wealth(99.9999, .2)) / .0002) < 1e-6
    assert abs((wealth(100., .200001) - wealth(100., .199999)) / .000002) < 1e-6


def test_synthetic_prepaid_forward_is_not_an_observed_lower_strike_bracket() -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 5.]), forwards=np.array([100., 100.]),
                      reference_forwards=np.array([100., 100.]), discounts=np.ones(2))
    claim = Claim(label="digital", kind="digital", initial_spot=100., strike=100., terminal_day=1)
    book = pd.DataFrame({"bdte": [1., 1., 1.], "strike": [0., 101., 102.],
        "option_type": ["call", "call", "call"],
        "instrument_kind": ["zero_strike_call", "european", "european"]})
    with pytest.raises(ValueError, match="surrounding observed"):
        black_digital_hedge(claim=claim, curve=curve, book=book, target_vol=.2,
            option_vols=np.full(3, .2), target_side=1)


@pytest.mark.parametrize("is_call", (True, False))
def test_black_cash_delta_and_vega_match_price_differences_with_nonflat_carry(is_call: bool) -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 5.]), forwards=np.array([100.1, 100.4]),
        reference_forwards=np.array([100.2, 100.5]), discounts=np.array([.999, .996]))
    claim = Claim(label="digital", kind="digital", initial_spot=100., strike=100.2, terminal_day=1, is_call=is_call)
    delta, vega = black_digital_exposures(claim=claim, curve=curve, target_vol=.2)

    def price(spot: float, vol: float) -> float:
        total_vol = vol / np.sqrt(252.)
        probability = ndtr(np.log(100.1 * spot / 100. / claim.strike) / total_vol - .5 * total_vol)
        return .999 * (probability if is_call else 1. - probability)

    assert delta == pytest.approx((price(100.0001, .2) - price(99.9999, .2)) / .0002, abs=1e-8)
    assert vega == pytest.approx((price(100., .200001) - price(100., .199999)) / .000002, abs=1e-8)
