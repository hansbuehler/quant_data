"""Weighted bootstrap repricing preserves the joint law and removes asset alpha."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedge2.scenarios import generate_scenarios
from quantdata.hedge2.state import DailyState


def test_empirical_scenarios_are_reproducible_centered_and_keep_settlement_payoffs() -> None:
    config = DlvConfig(num_inner_strikes=10)
    inner = shared_inner_strikes(np.full(9, .2), config=config)
    native = np.c_[np.full((9, 10), np.log(.04)), np.full(9, np.log(.2))].reshape(-1)
    rng = np.random.default_rng(14)
    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(20))
    samples = native + rng.normal(scale=.001, size=(20, len(native)))
    samples[-1] = native
    dynamics = fit_dynamics(dates=dates, states=samples, spot_returns_into_date=rng.normal(scale=.01, size=20),
                             session_indices=np.arange(20), as_of_date=dates[-1],
                             config=DynamicsConfig(window_transitions=15, rank=2, spot_orthogonal=True))
    state = DailyState(day=dates[-1], native=native, config=config,
                       reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 5., 25.]),
                      forwards=np.full(4, 100.), reference_forwards=np.full(4, 100.), discounts=np.ones(4))
    book = pd.DataFrame({"bdte": [1.], "strike": [100.], "option_type": ["call"]})
    claim = Claim(label="digital", kind="digital", initial_spot=100., strike=100., terminal_day=1)
    inputs = dict(state=state, dynamics=dynamics, curve=curve, book=book, claims=(claim,),
                  initial_spot=100., draw_count=10000, seed=92)
    scenarios = generate_scenarios(**inputs)
    repeated = generate_scenarios(**inputs)
    np.testing.assert_array_equal(scenarios.instrument_gains, repeated.instrument_gains)
    np.testing.assert_allclose(scenarios.weights @ scenarios.instrument_gains, 0., atol=1e-14)
    assert scenarios.draw_count == 10000
    assert scenarios.distinct_innovations == 15
    assert scenarios.effective_innovation_sample_size <= 15.
    assert len(np.unique(scenarios.claim_gains[:, 0])) == 2
