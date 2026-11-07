"""Forward-start and KO Black baselines hedge the separate variance buckets."""

import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedge2.black_exotic_hedge import black_exotic_hedge, black_exotic_price
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim


def test_pre_reset_ratio_has_zero_delta_and_opposite_bucket_hedges() -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 5.]),
        forwards=np.full(3, 100.), reference_forwards=np.full(3, 100.), discounts=np.ones(3))
    book = pd.DataFrame({"bdte": [3., 3., 5., 5.], "strike": [99., 101., 99., 101.],
                         "option_type": ["put", "call", "put", "call"]})
    claim = Claim(label="forward", kind="ratio_straddle", terminal_day=5,
                  observation_day=3, initial_spot=100., strike=1.)
    hedge = black_exotic_hedge(claim=claim, curve=curve, book=book, option_vols=np.full(4, .2),
        total_variances={3: .04 * 3. / DAYS_PER_YEAR, 5: .04 * 5. / DAYS_PER_YEAR}, target_side=-1)
    assert hedge.target_exposures[0] == pytest.approx(0., abs=1e-12)
    assert hedge.target_exposures[1] == pytest.approx(0., abs=1e-12)
    assert hedge.positions[1:3].sum() < 0 < hedge.positions[3:].sum()
    np.testing.assert_allclose(hedge.residual_exposures, 0., atol=1e-12)
    ko = Claim(label="ko", kind="ko", terminal_day=5, observation_day=3,
               barrier=100., initial_spot=100., strike=99., is_call=True)
    ko_hedge = black_exotic_hedge(claim=ko, curve=curve, book=book, option_vols=np.full(4, .2),
        total_variances={3: .04 * 3. / DAYS_PER_YEAR, 5: .04 * 5. / DAYS_PER_YEAR}, target_side=-1)
    assert np.all(np.abs(ko_hedge.target_exposures[1:]) > 1e-3)
    np.testing.assert_allclose(ko_hedge.residual_exposures, 0., atol=1e-12)
    refined = black_exotic_hedge(claim=ko, curve=curve, book=book, option_vols=np.full(4, .2),
        total_variances={3: .04 * 3. / DAYS_PER_YEAR, 5: .04 * 5. / DAYS_PER_YEAR}, target_side=-1,
        variance_bump_fraction=.0005, quadrature_order=768)
    np.testing.assert_allclose(refined.positions, ko_hedge.positions, atol=3e-5, rtol=3e-4)


def test_post_reset_price_holds_realized_reset_spot_fixed() -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 2.]),
        forwards=np.full(2, 100.), reference_forwards=np.full(2, 100.), discounts=np.ones(2))
    claim = Claim(label="f", kind="ratio_straddle", terminal_day=5,
                  observation_day=3, initial_spot=100., strike=1.).observe(elapsed_day=3, spot=99.)
    price = black_exotic_price(claim=claim, curve=curve, elapsed_day=3, variance_buckets=np.array([.0004]))
    moved = black_exotic_price(claim=claim, curve=curve.at_spot(101.), elapsed_day=3,
                               variance_buckets=np.array([.0004]))
    assert moved > price
    assert claim.reset_spot == 99.
    opposite_flag = Claim(label="f", kind="ratio_straddle", terminal_day=5, is_call=False,
        observation_day=3, initial_spot=100., strike=1.).observe(elapsed_day=3, spot=99.)
    assert black_exotic_price(claim=opposite_flag, curve=curve, elapsed_day=3,
        variance_buckets=np.array([.0004])) == pytest.approx(price)


def test_zero_strike_forwards_are_not_black_vega_brackets() -> None:
    from quantdata.hedge2.black_exotic_hedge import _bracket

    book = pd.DataFrame({"bdte": [3., 3., 3.], "strike": [0., 99., 101.],
        "option_type": ["call", "call", "put"],
        "instrument_kind": ["zero_strike_call", "european", "european"]})
    selected, weights = _bracket(book, tenor=3, strike=100.)
    np.testing.assert_array_equal(selected, [1, 2])
    np.testing.assert_allclose(weights, [.5, .5])
    with pytest.raises(ValueError, match="observed strikes surrounding"):
        _bracket(book, tenor=3, strike=98.)


def test_unused_zero_iv_wing_does_not_invalidate_black_variance_hedge() -> None:
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 5.]),
        forwards=np.full(3, 100.), reference_forwards=np.full(3, 100.), discounts=np.ones(3))
    book = pd.DataFrame({"bdte": [3., 3., 5., 5., 5.], "strike": [99., 101., 99., 101., 150.],
        "option_type": ["call"] * 5})
    claim = Claim(label="forward", kind="ratio_straddle", terminal_day=5,
        observation_day=3, initial_spot=100., strike=1.)
    hedge = black_exotic_hedge(claim=claim, curve=curve, book=book, option_vols=np.array([.2, .2, .2, .2, 0.]),
        total_variances={3: .04 * 3. / DAYS_PER_YEAR, 5: .04 * 5. / DAYS_PER_YEAR}, target_side=-1)
    assert hedge.positions[-1] == 0.
    assert np.isfinite(hedge.positions).all()
    np.testing.assert_allclose(hedge.residual_exposures, 0., atol=1e-12)
