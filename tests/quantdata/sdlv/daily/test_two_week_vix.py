from __future__ import annotations

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import get_business_days
from quantdata.sdlv.daily.front_spot.jax_model import RiskNeutralState
from quantdata.sdlv.daily.two_week_vix import (
    conditional_total,
    eligible_pair,
    market_term,
    partial_tower,
)
from quantdata.sdlv.daily.vix_options import _partial_day
from quantdata.vixdata.formula import constant_maturity_vix


def test_two_week_selection_uses_raw_friday_books_and_settlement_clocks() -> None:
    day=date(2025, 8, 29)
    sessions=get_business_days(day,date(2025,10,3))
    raw=pd.DataFrame({'exdate':['2025-09-05','2025-09-12','2025-09-17','2025-09-19'],
                      'am_settlement':[0.,0.,0.,1.]})
    pair=eligible_pair(day,raw,sessions)
    assert pair[0][:2] == (date(2025,9,12),False)
    assert pair[1][:2] == (date(2025,9,19),True)
    assert pair[0][2] == pytest.approx(14-15/1440)
    assert pair[1][2] == pytest.approx(21-405/1440)
    assert pair[0][3] == 9
    with pytest.raises(ValueError,match='bracketing'):
        eligible_pair(day,raw.iloc[:2],sessions)


def test_market_term_is_computed_from_raw_option_mids() -> None:
    strikes=np.array([80.,90.,100.,110.,120.])
    calls=np.array([20.1,10.5,3.,.6,.1])
    puts=calls+strikes-100
    raw=pd.DataFrame([dict(exdate='2025-09-12',am_settlement=0.,exercise_style='E',ss_flag='0',
        strike_price=k,cp_flag=side,best_bid=p-.01,best_offer=p+.01)
        for side,prices in [('C',calls),('P',puts)] for k,p in zip(strikes,prices,strict=True)])
    value,audit=market_term(raw,pd.DataFrame({'days':[1.,30.],'rate':[0.,0.]}),
                            (date(2025,9,12),False,14.,10.))
    otm=np.array([.1,.5,3.,.6,.1])
    assert value == pytest.approx(2*np.sum(10*otm/strikes**2))
    assert audit['forward'] == 100
    assert constant_maturity_vix(near_days=7,next_days=21,near_total_variance=.04*7/365,
        next_total_variance=.04*21/365,target_days=14) == pytest.approx(20)


def test_partial_P_with_zero_risk_price_matches_existing_Q() -> None:
    with jax.enable_x64():
        support=jnp.array([.5,.8,1.,1.2,1.5])
        state=RiskNeutralState(jnp.array([[.05,.2,.5,.2,.05]]),jnp.full((1,2,3),.04),jnp.ones(1))
        normals=jnp.array([[[.3,-.2]], [[-.1,.4]]])
        loadings=jnp.full((2,3,2),.1)
        actual=partial_tower(support,state,loadings,normals,jnp.zeros(2),fraction=.2)
        expected=_partial_day(support,state,loadings,normals,fraction=.2)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        shifted=partial_tower(support,state,loadings,normals,jnp.array([.5,.3]),fraction=.2)
        assert not np.allclose(actual,shifted,atol=1e-9)
        np.testing.assert_allclose(shifted.sum(axis=-1),1.,atol=1e-12)


def test_remaining_density_index_and_calendar_annualization() -> None:
    support=np.array([.8,1.,1.2])
    tower=np.array([[[0.,1.,0.],[.25,.5,.25],[.5,0.,.5]]])
    total=conditional_total(support,tower,absolute_business=4.5,calendar_days=14)
    density=np.array([.375,.25,.375])
    expected=-2*(density@np.log(support))+.05**2*(4.5-1.2)/252
    np.testing.assert_allclose(total,expected)
    with pytest.raises(ValueError,match='outside'):
        conditional_total(support,tower,absolute_business=6,calendar_days=14)
