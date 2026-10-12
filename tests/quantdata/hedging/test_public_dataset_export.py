from __future__ import annotations

import math
from datetime import date

import pytest

from quantdata.sdlv.daily.front_spot.hedging.data.export_public_candidate import (
    _finalize_observations,
)


def test_forward_adjusted_return_uses_previous_retained_observation() -> None:
    observations = [
        {
            "date": date(2024, 1, 2),
            "trading_session_is_half_day": False,
            "_source_spot_reference": 100.0,
        },
        {
            "date": date(2024, 1, 4),
            "trading_session_is_half_day": False,
            "_source_spot_reference": 103.0,
        },
    ]
    expiries = [
        {
            "business_dtes": [1.0, 3.0],
            "forward_over_spot": [1.01, 1.03],
            "discount_factors": [0.999, 0.996],
        },
        {
            "business_dtes": [1.0],
            "forward_over_spot": [1.0],
            "discount_factors": [1.0],
        },
    ]

    finalized = _finalize_observations(observations, expiries)

    assert all(value is None for key, value in finalized[0].items() if key not in {
        "date",
        "trading_session_is_half_day",
    })
    expected_carry = math.exp(0.5 * (math.log(1.01) + math.log(1.03)))
    expected_discount = math.exp(0.5 * (math.log(0.999) + math.log(0.996)))
    assert finalized[1]["business_sessions_from_previous_observation"] == 2
    assert finalized[1]["forward_carry_from_previous_observation"] == pytest.approx(
        expected_carry
    )
    assert finalized[1]["discount_factor_from_previous_observation"] == pytest.approx(
        expected_discount
    )
    assert finalized[1]["forward_adjusted_spot_return_into_date"] == pytest.approx(
        1.03 / expected_carry - 1.0
    )
