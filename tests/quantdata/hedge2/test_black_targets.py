"""Checks for the Black benchmark target prices."""

from __future__ import annotations

import numpy as np
import pytest
from cdxcore.bs import bs
from scipy.special import ndtr

from quantdata.hedge2.black_targets import (
    forward_start_straddle_price,
    single_observation_ko_price,
)


def test_forward_start_uses_forward_variance_not_terminal_variance() -> None:
    price = forward_start_straddle_price(
        strike_ratio=1.0, forward_to_reset=1.02, forward_to_terminal=1.04,
        variance_to_reset=0.01, variance_to_terminal=0.03,
        discount_to_terminal=0.98,
    )
    forward_ratio = 1.04 / 1.02
    expected = 0.98 * forward_ratio * (
        bs.price(k=1.0 / forward_ratio, vol=np.sqrt(0.02), is_call=True)
        + bs.price(k=1.0 / forward_ratio, vol=np.sqrt(0.02), is_call=False)
    )
    assert price == pytest.approx(expected)
    with pytest.raises(ValueError, match="Forward variance"):
        forward_start_straddle_price(
            strike_ratio=1.0, forward_to_reset=1.0, forward_to_terminal=1.0,
            variance_to_reset=0.03, variance_to_terminal=0.02,
        )


def test_ko_integrates_joint_terminal_payoff_and_survival() -> None:
    price = single_observation_ko_price(
        strike=0.9, barrier=1.0, is_call=True,
        forward_to_ko=1.0, forward_to_terminal=1.0,
        variance_to_ko=0.04, variance_to_terminal=0.04,
        quadrature_order=192,
    )
    sigma = np.sqrt(0.04)
    expected = ndtr(0.5 * sigma) - 0.9 * ndtr(-0.5 * sigma)
    assert price == pytest.approx(expected, abs=2e-6)


def test_ko_is_not_survival_probability_times_vanilla_price() -> None:
    price = single_observation_ko_price(
        strike=1.0, barrier=1.0, is_call=True,
        forward_to_ko=1.0, forward_to_terminal=1.0,
        variance_to_ko=0.02, variance_to_terminal=0.04,
    )
    vanilla = bs.price(k=1.0, vol=0.2, is_call=True)
    assert 0.5 * vanilla < price < vanilla


def test_ko_at_observation_uses_realized_survival() -> None:
    common = dict(
        strike=1.0, barrier=1.0, is_call=True,
        variance_to_ko=0.0, variance_to_terminal=0.02,
    )
    assert single_observation_ko_price(
        **common, forward_to_ko=0.99, forward_to_terminal=0.99,
    ) == 0.0
    expected = 1.01 * bs.price(k=1.0 / 1.01, vol=np.sqrt(0.02))
    assert single_observation_ko_price(
        **common, forward_to_ko=1.01, forward_to_terminal=1.01,
    ) == pytest.approx(expected)
