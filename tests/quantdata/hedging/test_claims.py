from __future__ import annotations

import numpy as np
import pytest

from quantdata.hedging.claims import (
    BarrierDirection,
    BarrierOptionClaim,
    DigitalClaim,
    RealizedVarianceClaim,
    normalized_strike_from_nms,
)
from quantdata.hedging.contracts import OptionRight


def test_atm_and_non_atm_strike_conventions() -> None:
    assert normalized_strike_from_nms(
        nms=0.0,
        business_dte=3.0,
        atm_annual_vol=0.2,
    ) == pytest.approx(1.0)
    assert normalized_strike_from_nms(
        nms=-0.75,
        business_dte=3.0,
        atm_annual_vol=0.2,
    ) < 1.0


def test_call_and_put_digitals_include_the_strike() -> None:
    call = DigitalClaim(right=OptionRight.CALL, strike=1.0, maturity_business_days=1)
    put = DigitalClaim(right=OptionRight.PUT, strike=1.0, maturity_business_days=1)
    assert call.payoff([1.0, 1.0]) == 0.0
    assert put.payoff([1.0, 1.0]) == 1.0


def test_realized_variance_is_unannualized_average_simple_return_square() -> None:
    claim = RealizedVarianceClaim(maturity_business_days=2)
    path = np.asarray([1.0, 1.1, 0.99])
    expected = 0.5 * (0.1**2 + (-0.1) ** 2)
    assert claim.payoff(path) == pytest.approx(expected)


def test_barriers_monitor_future_business_day_closes_only() -> None:
    put = BarrierOptionClaim(
        right=OptionRight.PUT,
        strike=1.0,
        barrier=0.95,
        direction=BarrierDirection.DOWN,
        maturity_business_days=2,
    )
    call = BarrierOptionClaim(
        right=OptionRight.CALL,
        strike=1.0,
        barrier=1.05,
        direction=BarrierDirection.UP,
        maturity_business_days=2,
    )
    assert put.payoff([0.90, 0.98, 0.97]) == pytest.approx(0.03)
    assert put.payoff([1.00, 0.95, 0.97]) == 0.0
    assert call.payoff([1.10, 1.04, 1.03]) == pytest.approx(0.03)
    assert call.payoff([1.00, 1.05, 1.03]) == 0.0
