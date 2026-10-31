from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.filtered_vix import filtered_term
from quantdata.vixdata.formula import discrete_total_variance, select_vix_strip


def _books() -> dict:
    strikes = np.arange(60, 141, 10, dtype=float) / 100
    puts = np.array([.02, .1, .2, 1, 4, 11, 21, 31, 41]) / 100
    calls = puts + 1.02 - strikes
    return dict(strikes=strikes, call_bids=calls * .99, call_asks=calls * 1.01,
                put_bids=puts * .99, put_asks=puts * 1.01)


def test_filtered_strips_preserve_deleted_raw_zero_bid_stops() -> None:
    books = _books()
    result, constituents = filtered_term(**books, retained=np.ones(9, dtype=bool), raw_put_stop=.7, raw_call_stop=1.3)
    np.testing.assert_allclose([row["normalized_strike"] for row in constituents], [.8, .9, 1, 1.1, 1.2])
    assert result["normalized_forward"] == pytest.approx(1.02)
    assert result["bid_total_variance"] < result["total_variance"] < result["ask_total_variance"]


def test_filtered_weights_recomputed_not_taken_from_full_strip() -> None:
    books = _books()
    mask = np.ones(9, dtype=bool)
    mask[3] = False
    _, constituents = filtered_term(**books, retained=mask, raw_put_stop=.7, raw_call_stop=1.3)
    np.testing.assert_allclose([row["delta_normalized_strike"] for row in constituents], [.2, .15, .1, .1])


def test_no_filter_matches_shared_formula() -> None:
    books = _books()
    result, _ = filtered_term(**books, retained=np.ones(9, dtype=bool), raw_put_stop=None, raw_call_stop=None)
    expected = discrete_total_variance(select_vix_strip(**books, discount_factor=1.), discount_factor=1.)
    assert result["total_variance"] == pytest.approx(expected)


def test_filtered_universe_cannot_be_extrapolated_if_it_loses_a_wing() -> None:
    with pytest.raises(ValueError, match="bracket"):
        filtered_term(**_books(), retained=np.arange(9) < 4, raw_put_stop=None, raw_call_stop=None)
