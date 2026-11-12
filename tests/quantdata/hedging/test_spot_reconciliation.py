"""Spot is present exactly once in retained joint replay and cash-strike repricing."""

from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.hedge2.batch_pricing import BatchDlvPricer
from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedge2.terminal_market import cash_europeans_batch
from quantdata.hedging.model_audit.spot_reconciliation import replay_training_changes


@pytest.mark.parametrize("orthogonal", [False, True])
def test_paired_training_replay_restores_spot_once(orthogonal: bool) -> None:
    rng = np.random.default_rng(122)
    count, dimensions, rank = 160, 6, 3
    spot = rng.normal(scale=.02, size=count)
    loading = rng.normal(size=dimensions)
    states = np.zeros((count, dimensions))
    for index in range(1, count):
        states[index] = .7 * states[index - 1] + spot[index] * loading + rng.normal(scale=.03, size=dimensions)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(count))
    fit = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
        session_indices=np.arange(count), as_of_date=dates[-1],
        config=DynamicsConfig(window_transitions=100, rank=rank, spot_orthogonal=orthogonal))
    before, after = states[-101:-1], states[-100:]
    replay, expected = replay_training_changes(before, after, fit)
    np.testing.assert_allclose(replay, expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(replay @ fit.directions, (after - before) @ fit.directions, atol=1e-13)
    if orthogonal:
        omitted = replay - fit.residuals[:, :1] * fit.spot_state_loading
        assert np.max(np.abs(omitted - expected)) > 1e-3


def test_cash_pricer_moves_fixed_strikes_with_spot_and_is_homogeneous() -> None:
    support = np.broadcast_to(np.array([.85, 1., 1.15]), (3, 3))
    decoded = BatchDlvPricer(support=support, densities=np.tile([[[.25, .5, .25]]], (3, 1, 1)),
        business_dtes=np.array([5.]), lower_clips=np.zeros(3, dtype=int), upper_clips=np.zeros(3, dtype=int))
    curve = CashCurve(spot=100., business_dtes=np.array([5.]), forwards=np.array([100.]),
        reference_forwards=np.array([100.]), discounts=np.array([1.]))
    spots = np.array([98., 100., 102.])
    fixed = cash_europeans_batch(decoded=decoded, template=curve, spots=spots,
        business_dtes=np.array([5.]), cash_strikes=np.array([100.]), is_call=np.array([True]))
    assert np.all(np.diff(fixed[:, 0]) > 0.)
    for index, spot in enumerate(spots):
        floating = cash_europeans_batch(decoded=decoded, template=curve,
            spots=np.repeat(spot, 3), business_dtes=np.array([5.]),
            cash_strikes=np.array([spot]), is_call=np.array([True]))
        assert floating[index, 0] / spot == pytest.approx(.0375, abs=1e-14)
