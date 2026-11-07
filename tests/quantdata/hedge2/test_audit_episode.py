"""The saved-position audit uses actual undiscounted current premium marks."""

import numpy as np
import pytest

from quantdata.hedge2.audit_episode import check_premium_limits


def test_actual_current_marks_determine_caps_not_raw_contracts_or_discounted_prices() -> None:
    positions = np.array([[5., 200., -1000.], [500., 500., 500.]])
    prices = np.array([120., 5., 1.])
    hits = check_premium_limits(holdings=positions, current_prices=prices, initial_spot=100.,
        methods=("DLV parametric", "BS delta+vega"), position_cap=10.)
    np.testing.assert_array_equal(hits, [2, 0])
    with pytest.raises(RuntimeError, match="current-premium"):
        check_premium_limits(holdings=positions[:1], current_prices=prices * 1.01, initial_spot=100.,
            methods=("DLV parametric",), position_cap=10.)


def test_zero_price_expired_inventory_is_not_divided_by_a_floor_in_cash_audit() -> None:
    hits = check_premium_limits(holdings=np.array([[0., 1000.]]), current_prices=np.array([100., 0.]),
        initial_spot=100., methods=("DLV Greek",), position_cap=10.)
    np.testing.assert_array_equal(hits, 0)
