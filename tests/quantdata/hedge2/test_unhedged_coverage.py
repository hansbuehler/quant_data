"""Broad coverage keeps unit-cash digitals and does not alter realized means."""

import numpy as np
import pytest

from quantdata.hedge2.claims import Claim
from quantdata.hedge2.unhedged_coverage import unhedged_digital_rows


def test_unhedged_cash_discount_and_strict_strike_convention() -> None:
    claims = tuple(Claim(label=label, kind="digital", terminal_day=1, initial_spot=100.,
        strike=100., is_call=is_call) for label, is_call in (("call", True), ("put", False)))
    rows = unhedged_digital_rows(claims=claims, market_marks=np.array([.4, .59]),
        following_spot=100., discount=.99)
    assert [row["terminal_payoff"] for row in rows] == [0., 0., 1., 1.]
    np.testing.assert_allclose([row["costed_pnl"] for row in rows], [.4, -.4, -.4, .4])
    assert all(row["entry_cost"] == row["hedge_gain"] == 0. for row in rows)


@pytest.mark.parametrize("changes", [{"market_marks": np.array([np.nan])},
    {"market_marks": np.zeros(2)}, {"following_spot": 0.}, {"discount": np.inf}])
def test_unhedged_coverage_rejects_invalid_cash_inputs(changes) -> None:
    claim = Claim(label="call", kind="digital", terminal_day=1, initial_spot=100., strike=100.)
    with pytest.raises(ValueError, match="one-day digitals"):
        unhedged_digital_rows(**{"claims": (claim,), "market_marks": np.array([.4]),
            "following_spot": 101., "discount": .99, **changes})


def test_unhedged_coverage_cannot_substitute_a_multiday_marked_gain() -> None:
    claim = Claim(label="call", kind="digital", terminal_day=3, initial_spot=100., strike=100.)
    with pytest.raises(ValueError, match="one-day digitals"):
        unhedged_digital_rows(claims=(claim,), market_marks=np.array([.4]),
            following_spot=101., discount=.99)
