"""Static defects are counted without repairing prices or changing cash strikes."""

import numpy as np
from cdxcore.bs import bs

from quantdata.hedging.model_audit.static_arb import call_violations, iv_stability, next_tenor


def test_black_grid_satisfies_nonuniform_strike_and_calendar_tests() -> None:
    strikes = np.array([.8, .97, 1., 1.04, 1.3])
    calls = np.broadcast_to(bs.price(k=strikes, vol=.2, sqrtT=np.sqrt(5. / 252.)), (3, 5))
    longer = np.broadcast_to(bs.price(k=strikes, vol=.2, sqrtT=np.sqrt(10. / 252.)), (3, 5))
    result = call_violations(strikes, calls, longer_prices=longer, zero_strike=np.ones(3))
    assert all(value == 0 for name, value in result.items() if name.endswith("_violations"))
    assert result["calendar_tests"] == 15
    assert result["convexity_tests"] == 12


def test_bad_prices_report_separate_defects_and_missing_values() -> None:
    strikes = np.array([.9, 1., 1.1])
    calls = np.array([[.1, .2, .01], [np.nan, .03, .01]])
    result = call_violations(strikes, calls, longer_prices=calls - .01, zero_strike=np.ones(2))
    assert result["decreasing_violations"] == 1
    assert result["convexity_violations"] == 1
    assert result["calendar_violations"] == 5
    assert result["invalid_price_nodes"] == 1
    assert result["invalid_price_paths"] == 1


def test_last_model_tenor_does_not_invent_a_calendar_expiry() -> None:
    assert next_tenor(5., np.array([1., 2., 3., 5., 7., 10.])) == 7.
    assert next_tenor(10., np.array([1., 2., 3., 5., 7., 10.])) is None


def test_iv_range_counts_remain_distinct_from_static_arbitrage() -> None:
    strikes = np.array([.99, 1., 1.01])
    calls = np.broadcast_to(bs.price(k=strikes, vol=.3, sqrtT=np.sqrt(5. / 252.)), (2, 3))
    result = iv_stability(calls, strikes, 5., np.full((252, 3), .2))
    assert result["outside_historical_range_count"] == 6
    assert result["invalid_iv_count"] == 0
    checks = call_violations(strikes, calls, longer_prices=None, zero_strike=np.ones(2))
    assert all(value == 0 for name, value in checks.items() if name.endswith("_violations"))
