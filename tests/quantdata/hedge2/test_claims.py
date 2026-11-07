"""Claim contracts retain cash strikes and observe path events only on their dates."""

import numpy as np
import pytest

from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim, make_claims
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.pricing import DlvPricer


def _setup() -> tuple[DlvPricer, CashCurve]:
    config = DlvConfig(num_inner_strikes=20)
    inner = shared_inner_strikes(np.full(9, .2), config=config)
    pricer = DlvPricer(support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]],
                       expiries_y=config.expiries_y, dlvsqs=np.full((9, 20), .04))
    days = np.array([1., 2., 3., 4., 5., 10., 15., 20., 25.])
    curve = CashCurve(spot=100., business_dtes=days, forwards=np.full(9, 100.),
                      reference_forwards=np.full(9, 100.), discounts=np.ones(9))
    return pricer, curve


def test_enumeration_contains_all_tenor_digitals_and_unique_exotic_nms() -> None:
    claims = make_claims(initial_spot=100., atm_vols_by_day={1: .2, 2: .2, 3: .2, 5: .2})
    assert len(claims) == 48
    for tenor in (1, 3, 5):
        digitals = [claim for claim in claims if claim.kind == "digital" and claim.terminal_day == tenor]
        assert len(digitals) == 10
        assert len([claim for claim in digitals if claim.strike == 100.]) == 2
    assert len([claim for claim in claims if claim.kind == "ko"]) == 9
    assert len({claim.label for claim in claims}) == 48


def test_ko_observes_only_on_day_three_and_keeps_cash_barrier_fixed() -> None:
    claim = Claim(label="ko", kind="ko", initial_spot=100., strike=99.,
                  terminal_day=5, observation_day=3, barrier=100., is_call=True)
    assert not claim.observe(elapsed_day=1, spot=95.).knocked_out
    with pytest.raises(ValueError, match="observation"):
        claim.payoff(102.)
    stopped = claim.observe(elapsed_day=3, spot=99.)
    assert stopped.payoff(105.) == 0.
    alive = claim.observe(elapsed_day=3, spot=101.)
    assert alive.payoff(102.) == pytest.approx(.03)


def test_ratio_reset_uses_actual_spot_not_a_forward_or_new_strike() -> None:
    claim = Claim(label="ratio", kind="ratio_straddle", initial_spot=100., strike=1.02,
                  terminal_day=5, observation_day=3)
    reset = claim.observe(elapsed_day=3, spot=105.)
    assert reset.payoff(106.) == pytest.approx(abs(106. / 105. - 1.02))
    pricer, curve = _setup()
    assert np.isfinite(reset.price(pricer, curve, elapsed_day=3))


def test_cash_digital_call_put_complement_and_zero_time_payoff() -> None:
    pricer, curve = _setup()
    call = Claim(label="call", kind="digital", initial_spot=100., strike=100., terminal_day=3)
    put = Claim(label="put", kind="digital", initial_spot=100., strike=100., terminal_day=3, is_call=False)
    assert call.price(pricer, curve, elapsed_day=0) + put.price(pricer, curve, elapsed_day=0) == pytest.approx(1.)
    assert call.price(pricer, curve, elapsed_day=3) == 0.
    assert put.price(pricer, curve, elapsed_day=3) == 1.
