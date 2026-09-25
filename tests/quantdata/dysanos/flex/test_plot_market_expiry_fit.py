from __future__ import annotations

import numpy as np
import pytest

from quantdata.dysanos.flex.plot_market_expiry_fit import _series


def test_market_expiry_plot_series_preserves_report_order() -> None:
    report = {
        "by_nominal_market_dte": [
            {"nominal_market_dte": 1, "percent_inside_bid_ask": 80.0},
            {"nominal_market_dte": 2, "percent_inside_bid_ask": 90.0},
        ]
    }

    expiries, values = _series(report, "percent_inside_bid_ask")

    np.testing.assert_array_equal(expiries, (1.0, 2.0))
    np.testing.assert_array_equal(values, (80.0, 90.0))


def test_market_expiry_plot_series_rejects_unsorted_expiries() -> None:
    report = {
        "by_nominal_market_dte": [
            {"nominal_market_dte": 2, "mean_excess": 0.2},
            {"nominal_market_dte": 1, "mean_excess": 0.1},
        ]
    }

    with pytest.raises(ValueError, match="increasing"):
        _series(report, "mean_excess")
