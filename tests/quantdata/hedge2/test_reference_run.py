"""Analytic references preserve premium units, entry-only costs and visible capacity."""

import numpy as np
import pytest

from quantdata.hedge2.reference_run import score_reference_positions


def test_reference_cash_conversion_charges_entry_not_the_hypothetical_exit() -> None:
    result, allocations = score_reference_positions(positions=np.array([30., -2., 3.]), initial_spot=100.,
        cash_prices=np.array([100., 2., 4.]), cash_gains=np.array([1., -.4, .1]),
        half_spreads=np.array([0., .05, .1]), next_half_spreads=np.array([0., .06, .12]),
        discount=.99, signed_target_gain=.2)
    np.testing.assert_allclose(allocations, [30., -.04, .12])
    assert result["hedge_gain"] == pytest.approx(.311)
    assert result["entry_cost"] == pytest.approx(.004)
    assert result["costed_pnl"] == pytest.approx(.507)
    assert result["hypothetical_exit_cost"] == pytest.approx(.004752)
    assert result["gross_positions"] == pytest.approx(30.16)
    assert result["comparison_box_exceeded"] and result["reference_is_uncapped"]
    assert result["cap_hits"] == 1


def test_unhedged_reference_has_no_cost_or_position() -> None:
    result, allocations = score_reference_positions(positions=np.zeros(1), initial_spot=100.,
        cash_prices=np.array([100.]), cash_gains=np.array([2.]), half_spreads=np.zeros(1),
        next_half_spreads=np.zeros(1), discount=1., signed_target_gain=-.5)
    assert result["costed_pnl"] == -.5
    assert result["entry_cost"] == result["hypothetical_exit_cost"] == 0.
    assert not result["comparison_box_exceeded"]
    np.testing.assert_array_equal(allocations, 0.)


def test_reference_cannot_silently_charge_a_spot_spread() -> None:
    with pytest.raises(ValueError, match="free spot"):
        score_reference_positions(positions=np.ones(1), initial_spot=100., cash_prices=np.array([100.]),
            cash_gains=np.array([2.]), half_spreads=np.array([.01]), next_half_spreads=np.zeros(1),
            discount=1., signed_target_gain=0.)
