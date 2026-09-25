from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.metrics import nominal_expiry_excess_statistics


def test_nominal_expiry_statistics_use_completed_business_days() -> None:
    fit = np.array([1.0, 1.2, 1.4, 0.9])
    bid = np.array([0.9, 0.9, 0.9, 1.0])
    ask = np.array([1.1, 1.1, 1.1, 1.2])
    dtes = np.array([1.00000001, 1.2, 2.00000001, 2.2])

    rows = nominal_expiry_excess_statistics(fit, bid, ask, dtes)

    assert [row["nominal_market_dte"] for row in rows] == [1, 2]
    assert [row["num_quotes"] for row in rows] == [2, 2]
    assert rows[0]["percent_inside_bid_ask"] == pytest.approx(50.0)
    assert rows[0]["mean_excess"] == pytest.approx(0.5)
    assert rows[1]["percent_inside_bid_ask"] == pytest.approx(0.0)
    assert rows[1]["mean_excess"] == pytest.approx(2.0)


def test_nominal_expiry_statistics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per option"):
        nominal_expiry_excess_statistics(
            np.array([1.0]),
            np.array([0.9]),
            np.array([1.1]),
            np.array([1.0, 2.0]),
        )
