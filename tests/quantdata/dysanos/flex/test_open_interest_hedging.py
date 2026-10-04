from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdata.dysanos.flex.run_open_interest_hedging import (
    _after_close_oi_book,
)


def test_after_close_open_interest_defines_next_session_proxy_weights() -> None:
    current = pd.DataFrame(
        {
            "is_am": [False],
            "is_american": [False],
            "bdte": [10.0],
            "strike": [100.0],
            "call_opt_id": [11],
            "put_opt_id": [12],
            "mkt_call_log_open_interest": [np.log10(11.0)],
            "mkt_put_log_open_interest": [np.log10(91.0)],
            "mkt_call_bid": [1.0],
            "mkt_call_ask": [3.0],
            "mkt_put_bid": [2.0],
            "mkt_put_ask": [4.0],
        }
    )
    following = pd.DataFrame(
        {
            "call_opt_id": [11],
            "put_opt_id": [12],
            "mkt_call_bid": [2.0],
            "mkt_call_ask": [4.0],
            "mkt_put_bid": [4.0],
            "mkt_put_ask": [6.0],
        }
    )

    book = _after_close_oi_book(current, following, spot=100.0)

    np.testing.assert_allclose(book["weights"], (-0.10, -0.90))
    np.testing.assert_allclose(book["remaining_days"], (9.0, 9.0))
    assert book["gross_open_interest"] == pytest.approx(100.0)
    expected_pnl = np.dot(
        (-0.10, -0.90),
        ((3.0 - 1.0) / 100.0, (5.0 - 2.0) / 100.0),
    )
    assert book["realized_unhedged_pnl"] == pytest.approx(expected_pnl)
