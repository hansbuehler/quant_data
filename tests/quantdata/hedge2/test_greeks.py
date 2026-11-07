"""Cash-contract spot/PCA bumps must preserve the spot-first hedge convention."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantdata.hedge2.cash import CashCurve
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedge2.greeks import (
    calculate_greeks,
    calculate_parametric_greeks,
    factor_risk_covariance,
    parametric_factor_covariance,
)
from quantdata.hedge2.state import DailyState


def test_spot_is_present_and_pca_vegas_include_front_dlvs() -> None:
    config = DlvConfig(num_inner_strikes=20)
    inner = shared_inner_strikes(np.full(9, .2), config=config)
    native = np.c_[np.full((9, 20), np.log(.04)), np.full(9, np.log(.2))].reshape(-1)
    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(30))
    rng = np.random.default_rng(18)
    samples = native + rng.normal(scale=.05, size=(30, len(native)))
    samples[-1] = native
    model = fit_dynamics(dates=dates, states=samples, spot_returns_into_date=rng.normal(scale=.01, size=30),
                          session_indices=np.arange(30), as_of_date=dates[-1],
                          config=DynamicsConfig(window_transitions=25, rank=2))
    state = DailyState(day=dates[-1], native=native, config=config,
                       reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 5., 25.]),
                      forwards=np.full(4, 100.), reference_forwards=np.full(4, 100.), discounts=np.ones(4))
    book = pd.DataFrame({"bdte": [1., 1.], "strike": [100., 100.], "option_type": ["call", "put"]})
    claim = Claim(label="d", kind="digital", terminal_day=1, initial_spot=100., strike=100.)
    result = calculate_greeks(state=state, dynamics=model, curve=curve, book=book, claims=(claim,), initial_spot=100.)
    np.testing.assert_allclose(result.instrument_exposures[0], [1., 0., 0.], atol=1e-13)
    np.testing.assert_allclose(result.instrument_exposures[1] - result.instrument_exposures[2], [1., 0., 0.], atol=1e-13)
    assert np.max(np.abs(result.instrument_exposures[1:, 1:])) > 1e-8
    covariance = factor_risk_covariance(model, spot_orthogonal=False)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_parametric_delta_includes_spot_loading_outside_retained_pcs(monkeypatch) -> None:
    day = date(2024, 1, 2)
    state = SimpleNamespace(day=day, native=np.zeros(2))
    model = SimpleNamespace(as_of_date=day, directions=np.array([[1.], [0.]]),
        daily_factor_scales=np.array([2.]), spot_state_loading=np.array([0., 3.]),
        joint_innovation_covariance=np.array([[.01, .02], [.02, .5]]))
    curve = CashCurve(spot=100., business_dtes=np.array([1., 5.]),
        forwards=np.full(2, 100.), reference_forwards=np.full(2, 100.), discounts=np.ones(2))
    book = pd.DataFrame({"bdte": [1.], "strike": [100.], "option_type": ["call"]})

    def prices(*, native, curve, **kwargs):
        relative_spot = curve.spot / 100.
        return np.array([relative_spot, native[0] + native[1],
                         relative_spot**2 * (1. + native[1])]), 0, 0

    monkeypatch.setattr("quantdata.hedge2.greeks._price_vector", prices)
    fixed = calculate_greeks(state=state, dynamics=model, curve=curve, book=book, claims=(None,), initial_spot=100.)
    conditioned = calculate_parametric_greeks(state=state, dynamics=model, curve=curve,
        book=book, claims=(None,), initial_spot=100.)
    assert conditioned.spot_is_state_conditioned
    np.testing.assert_allclose(conditioned.instrument_exposures[:, 0], [1., 3.], atol=1e-12)
    np.testing.assert_allclose(conditioned.claim_exposures[:, 0], [5.], atol=1e-12)
    np.testing.assert_allclose(conditioned.instrument_exposures[:, 1:], fixed.instrument_exposures[:, 1:])
    assert conditioned.combined_spot_claim_exposures[0] > conditioned.claim_exposures[0, 0]
    expected = model.joint_innovation_covariance / np.outer([1., 2.], [1., 2.])
    np.testing.assert_allclose(parametric_factor_covariance(model), expected)
