"""Batch decoding must preserve the scalar pricer's irregular grid and payoff timing."""

from datetime import date

import numpy as np
import pytest

from quantdata.hedge2.batch_pricing import (
    decode_batch,
    floating_inner_strikes,
    row_searchsorted_right,
)
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.state import DailyState


def state() -> DailyState:
    config = DlvConfig(num_inner_strikes=20)
    vols = np.full(9, .2)
    inner = shared_inner_strikes(vols, config=config)
    return DailyState(day=date(2024, 4, 19), config=config,
        native=np.c_[np.full((9, 20), np.log(.04)), np.log(vols)].reshape(-1),
        reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])


def test_floating_batch_grid_matches_nonuniform_scalar_geometry() -> None:
    model = state()
    vols = np.array([np.linspace(.1, .2, 9), np.linspace(.5, .15, 9), np.full(9, .2)])
    result = floating_inner_strikes(vols, config=model.config)
    for index, curve in enumerate(vols):
        np.testing.assert_allclose(result[index], shared_inner_strikes(curve, config=model.config), rtol=1e-14)
    assert np.all(np.sum(result == 1., axis=1) == 1)


@pytest.mark.parametrize("digital", [False, True])
def test_batch_option_prices_and_digital_ties_match_scalar_payoff_integrals(digital) -> None:
    model = state()
    rng = np.random.default_rng(171)
    native = model.native + rng.normal(scale=.05, size=(4, len(model.native)))
    batch = decode_batch(model, native)
    days = np.array([0., 1., 2., 2.5, 4., 5., 7., 10., 25.])
    queries = np.broadcast_to(np.array([1., .98, 1., 1.02, 1., 1.04, .95, .9, 1.2]), (4, len(days))).copy()
    queries[:, 3] = batch.support[:, 10]
    calls = np.array([True, False, True, True, False, True, False, True, False])
    result = batch.europeans(business_dtes=days, strikes=queries, is_call=calls, digital=digital)
    for index, values in enumerate(native):
        scalar = model.decode(values).pricer
        np.testing.assert_allclose(batch.support[index], scalar.support, atol=1e-15)
        np.testing.assert_allclose(batch.densities[index], scalar.marginal_densities, atol=1e-14)
        expected = [scalar.european_at(expiry_y=tenor / 252., strikes=np.array([strike]),
            is_call=side, digital=digital)[0] for tenor, strike, side in zip(days, queries[index], calls)]
        np.testing.assert_allclose(result[index], expected, atol=5e-14)


def test_row_binary_search_handles_exact_nodes_and_outside_support() -> None:
    support = np.array([[.5, 1., 2.], [.4, .8, 1.2]])
    queries = np.array([[.1, .5, .75, 1., 2., 5.], [.1, .4, .5, .8, 1.2, 5.]])
    expected = np.stack([np.searchsorted(row, query, side="right") for row, query in zip(support, queries)])
    np.testing.assert_array_equal(row_searchsorted_right(support, queries), expected)


def test_zero_strike_calls_have_unit_normalized_value() -> None:
    model = state()
    batch = decode_batch(model, np.tile(model.native, (3, 1)))
    prices = batch.europeans(business_dtes=np.array([1., 3., 5.]), strikes=np.zeros(3))
    np.testing.assert_allclose(prices, 1., atol=1e-13)


def test_batch_reports_guards_and_rejects_extreme_forward_geometry() -> None:
    model = state()
    values = np.tile(model.native, (2, 1))
    values[0, :2] = [-100., 100.]
    decoded = decode_batch(model, values)
    np.testing.assert_array_equal(decoded.lower_clips, [1, 0])
    np.testing.assert_array_equal(decoded.upper_clips, [1, 0])
    values[1].reshape(9, 21)[:, -1] = 100.
    with pytest.raises(ValueError, match="geometry"):
        decode_batch(model, values)


def test_partial_density_horizon_keeps_full_original_floating_grid() -> None:
    model = state()
    native = np.tile(model.native, (2, 1))
    full = decode_batch(model, native)
    partial = decode_batch(model, native, max_business_dte=2.5)
    assert partial.densities.shape[1] == 3
    np.testing.assert_allclose(partial.support, full.support)
    np.testing.assert_allclose(partial.density_at(2.5), full.density_at(2.5))
