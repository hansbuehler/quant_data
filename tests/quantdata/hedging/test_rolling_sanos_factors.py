"""Causality and spot orthogonality of rolling SANOS state factors."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.hedging.data.rolling_sanos_factors import (
    fit_rolling_sanos_factors,
    select_sanos_factor_window,
)
from quantdata.hedging.data.sanos_state_greeks import (
    finite_difference_sanos_state_greeks,
)


def _history() -> tuple[tuple[date, ...], np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(27)
    dates = tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(260))
    spot_returns = random.normal(size=len(dates))
    innovations = random.normal(size=(len(dates), 5))
    state_changes = innovations + 2.0 * spot_returns[:, None] * np.array(
        [1.0, 0.4, 0.0, -0.3, 0.0]
    )
    states = np.cumsum(state_changes, axis=0)
    sessions = np.ones(len(dates), dtype=np.int64)
    sessions[3] = 2
    return dates, states, spot_returns, sessions


def test_rolling_window_uses_only_exact_past_transitions() -> None:
    dates, states, spot_returns, sessions = _history()
    window = select_sanos_factor_window(
        dates=dates,
        raw_states=states,
        spot_returns_into_date=spot_returns,
        sessions_from_previous_date=sessions,
        as_of_date=dates[255],
    )
    assert len(window.endpoint_dates) == 252
    assert window.endpoint_dates[-1] == dates[255]
    assert dates[3] not in window.endpoint_dates
    assert dates[256] not in window.endpoint_dates


def test_spot_orthogonal_pca_retains_spot_surface_loading() -> None:
    dates, states, spot_returns, sessions = _history()
    window = select_sanos_factor_window(
        dates=dates,
        raw_states=states,
        spot_returns_into_date=spot_returns,
        sessions_from_previous_date=sessions,
        as_of_date=dates[-1],
    )
    ordinary = fit_rolling_sanos_factors(window, method="ordinary", rank=3)
    orthogonal = fit_rolling_sanos_factors(
        window, method="spot_orthogonal", rank=3
    )
    assert ordinary.state_loadings.shape == (5, 3)
    assert np.allclose(ordinary.spot_correlated_state_loading, 0.0)
    assert abs(ordinary.spot_factor_covariance[0, 1]) > 0.05
    assert np.allclose(orthogonal.spot_factor_covariance[0, 1:], 0.0, atol=1e-12)
    assert np.linalg.norm(orthogonal.spot_correlated_state_loading) > 1.0
    assert np.allclose(np.cov(orthogonal.factor_scores, rowvar=False), np.eye(3))
    assert np.all(orthogonal.explained_variance_fraction > 0.0)


def test_insufficient_exact_transitions_rejected() -> None:
    dates, states, spot_returns, sessions = _history()
    with pytest.raises(ValueError, match="Insufficient"):
        select_sanos_factor_window(
            dates=dates,
            raw_states=states,
            spot_returns_into_date=spot_returns,
            sessions_from_previous_date=sessions,
            as_of_date=dates[100],
        )


def test_spot_partial_remains_fixed_state_in_both_variants() -> None:
    dates, states, spot_returns, sessions = _history()
    window = select_sanos_factor_window(
        dates=dates,
        raw_states=states,
        spot_returns_into_date=spot_returns,
        sessions_from_previous_date=sessions,
        as_of_date=dates[-1],
    )
    state_weights = np.array([0.7, -0.2, 0.3, 0.1, 0.5])

    def price(spot: float, state: np.ndarray) -> np.ndarray:
        return np.array([2.0 * spot + state_weights @ state])

    for method in ("ordinary", "spot_orthogonal"):
        factors = fit_rolling_sanos_factors(window, method=method, rank=3)
        greeks = finite_difference_sanos_state_greeks(
            price=price,
            spot=1.0,
            raw_state=states[-1],
            factors=factors,
        )
        assert greeks.fixed_state_spot == pytest.approx([2.0])
        assert greeks.factor_exposures[0] == pytest.approx(
            state_weights @ factors.state_loadings
        )
        assert greeks.spot_correlated_state == pytest.approx(
            [state_weights @ factors.spot_correlated_state_loading]
        )
