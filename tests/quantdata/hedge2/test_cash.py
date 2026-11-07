"""Observed cash marks, carry rolls and zero-cost expiry accounting."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2.cash import (
    CashCurve,
    MarketDay,
    discounted_daily_gains,
    model_europeans,
    observed_marks,
    select_book,
)
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.pricing import DlvPricer


def _curve() -> CashCurve:
    days = np.array([1., 3., 5., 10.])
    return CashCurve(spot=100., business_dtes=days, forwards=100. * np.exp(.001 * days),
                     reference_forwards=100. * np.exp(.002 * days), discounts=np.exp(-.003 * days))


def test_cash_curve_roll_keeps_source_and_reference_forward_distinct() -> None:
    curve = _curve()
    rolled = curve.roll(1., new_spot=102.)
    forward, reference, discount = rolled.at(2.)
    assert forward == pytest.approx(102. * np.exp(.002))
    assert reference == pytest.approx(102. * np.exp(.004))
    assert discount == pytest.approx(np.exp(-.006))
    np.testing.assert_allclose(rolled.at(0.), [102., 102., 1.])


def test_option_prices_obey_cash_put_call_parity() -> None:
    config = DlvConfig(num_inner_strikes=20)
    inner = shared_inner_strikes(np.full(9, .2), config=config)
    pricer = DlvPricer(support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]],
                       expiries_y=config.expiries_y, dlvsqs=np.full((9, 20), .04))
    values = model_europeans(pricer, _curve(), business_dtes=np.array([3., 3.]),
                             cash_strikes=np.array([99., 99.]), is_call=np.array([True, False]))
    forward, _, discount = _curve().at(3.)
    assert values[0] - values[1] == pytest.approx(discount * (forward - 99.), abs=1e-12)


def test_expiry_mark_has_no_spread_and_no_option_exit_trade() -> None:
    quotes = pd.DataFrame({"expiry": [pd.Timestamp("2024-04-02")], "strike": [100.],
                           "option_type": ["call"], "is_am": [False], "observed_mid": [2.],
                           "observed_half_spread": [.2]})
    book = quotes.copy()
    market = MarketDay(day=date(2024, 4, 2), quotes=quotes,
                       curve=_curve().at_spot(102.))
    prices, spreads = observed_marks(book, market)
    np.testing.assert_array_equal(prices, [2.])
    np.testing.assert_array_equal(spreads, [0.])
    gains = discounted_daily_gains(previous_option_marks=np.array([1.5]), next_option_marks=prices,
                                   next_spot=102., previous_curve=_curve(), episode_initial_spot=100.)
    _, reference, discount = _curve().at(1.)
    np.testing.assert_allclose(gains, [discount * (102. - reference) / 100.,
                                      (discount * 2. - 1.5) / 100.])


def test_hedge_screen_uses_expiry_atm_vol_not_the_cached_nms_column() -> None:
    frame = pd.DataFrame({"expiry": pd.to_datetime(["2024-04-02"] * 4 + ["2024-04-08"] * 4),
                          "strike": [99., 99., 101., 101.] * 2,
                          "option_type": ["put", "call", "put", "call"] * 2,
                          "is_am": False, "bdte": [1.] * 4 + [5.] * 4,
                          "fwd": 100., "nms": 500.})
    market = MarketDay(day=date(2024, 4, 1), quotes=frame, curve=_curve())
    book = select_book(market, atm_vols=np.full(len(frame), .2))
    assert len(book) == 4
    assert book["option_type"].tolist() == ["put", "call", "put", "call"]
    assert np.all(np.abs(book["hedge_nms"]) < 2.)
    single = select_book(market, atm_vols=np.full(len(frame), .2), co_terminal_business_dtes=(1,))
    assert len(single) == 2


def test_itm_book_adds_one_explicit_zero_strike_call_per_expiry() -> None:
    quotes = pd.DataFrame({"expiry": pd.to_datetime(["2024-04-02"] * 4 + ["2024-04-08"] * 4),
        "strike": [99., 99., 101., 101.] * 2, "option_type": ["put", "call", "put", "call"] * 2,
        "is_am": False, "bdte": [1.] * 4 + [5.] * 4, "fwd": 100.,
        "observed_mid": 1., "observed_half_spread": .1})
    market = MarketDay(day=date(2024, 4, 1), quotes=quotes, curve=_curve())
    book = select_book(market, atm_vols=np.full(len(quotes), .2), option_side="itm", include_co_terminal_forwards=True)
    synthetic = book["instrument_kind"] == "zero_strike_call"
    assert len(book) == 6
    assert book.loc[~synthetic, "option_type"].tolist() == ["call", "put", "call", "put"]
    assert book.loc[synthetic, "observed_half_spread"].sum() == 0.
    marked, spreads = observed_marks(book, market)
    np.testing.assert_allclose(marked, book["observed_mid"])
    np.testing.assert_allclose(spreads, book["observed_half_spread"])
    following = MarketDay(day=date(2024, 4, 2), quotes=quotes, curve=_curve().roll(1., new_spot=102.))
    marked, spreads = observed_marks(book, following)
    np.testing.assert_allclose(marked[synthetic], [102., 102. * np.exp(-.008)])
    np.testing.assert_array_equal(spreads[synthetic], 0.)


def test_zero_strike_call_model_mark_is_discounted_forward() -> None:
    curve = _curve()
    prices = model_europeans(None, curve, business_dtes=np.array([1., 5.]),
        cash_strikes=np.array([0., 0.]), is_call=np.array([True, True]))
    forward, _, discount = curve.at(np.array([1., 5.]))
    np.testing.assert_allclose(prices, discount * forward)


def test_ordinary_missing_quote_does_not_become_a_synthetic_forward() -> None:
    quotes = pd.DataFrame({"expiry": [pd.Timestamp("2024-04-02")], "strike": [0.],
        "option_type": ["call"], "is_am": [False], "observed_mid": [100.], "observed_half_spread": [0.]})
    market = MarketDay(day=date(2024, 4, 1), quotes=quotes, curve=_curve())
    with pytest.raises(ValueError, match="explicit synthetic"):
        observed_marks(quotes, market)
