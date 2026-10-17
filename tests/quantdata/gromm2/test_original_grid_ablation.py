from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.original_grid_ablation import (
    center_bands,
    reconstruct_quote_expiries,
)


def test_reconstruct_quote_expiries_matches_encoded_interpolation() -> None:
    actual = reconstruct_quote_expiries(
        model_times_y=np.asarray([1.0, 2.0, 3.0]) / 252.0,
        lower=np.asarray([0, 0, 1]),
        upper=np.asarray([0, 1, 2]),
        alpha=np.asarray([1.0, 0.5, 0.25]),
        before_first=np.asarray([True, False, False]),
    )
    assert actual == pytest.approx(np.asarray([1.0, 1.5, 2.25]) / 252.0)


def test_center_bands_preserves_market_half_spreads() -> None:
    midpoints = np.asarray([0.1, 0.2])
    market_bids = np.asarray([0.08, 0.17])
    market_asks = np.asarray([0.12, 0.23])
    bids, asks = center_bands(
        synthetic_midpoints=midpoints + 0.01,
        market_bids=market_bids,
        market_asks=market_asks,
    )
    assert 0.5 * (bids + asks) == pytest.approx(midpoints + 0.01)
    assert 0.5 * (asks - bids) == pytest.approx(
        0.5 * (market_asks - market_bids)
    )
