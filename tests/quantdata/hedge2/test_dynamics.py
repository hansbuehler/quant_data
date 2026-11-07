"""Rolling dynamics tests cover recovery, causality and factor orientation."""

from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics


def _history() -> tuple[tuple[date, ...], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(121)
    states = np.zeros((1000, 4))
    for index in range(1, len(states)):
        states[index] = 0.8 * states[index - 1] + rng.normal(size=4)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(len(states)))
    return dates, states, rng.normal(scale=0.01, size=len(states)), np.arange(len(states))


def test_ar1_recovers_stationary_process() -> None:
    dates, states, spot, sessions = _history()
    fitted = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                          session_indices=sessions, as_of_date=dates[-1],
                          config=DynamicsConfig(window_transitions=900, rank=4))
    np.testing.assert_allclose(fitted.ar_matrix, 0.8 * np.eye(4), atol=0.09)
    assert fitted.constrained_spectral_radius < 1.
    assert fitted.training_rmse < fitted.persistence_rmse
    np.testing.assert_allclose(fitted.residuals.mean(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(fitted.joint_innovation_covariance,
                               np.cov(fitted.residuals, rowvar=False, ddof=0))


def test_future_states_do_not_change_fitted_model() -> None:
    dates, states, spot, sessions = _history()
    kwargs = dict(dates=dates, spot_returns_into_date=spot, session_indices=sessions,
                  as_of_date=dates[500], config=DynamicsConfig(rank=4))
    original = fit_dynamics(states=states, **kwargs)
    changed = states.copy()
    changed[501:] *= 100.
    repeated = fit_dynamics(states=changed, **kwargs)
    np.testing.assert_array_equal(original.ar_matrix, repeated.ar_matrix)
    np.testing.assert_array_equal(original.directions, repeated.directions)
    assert original.training_end == dates[500]


def test_rolling_fit_records_prior_orientation_and_uses_exact_sessions() -> None:
    dates, states, spot, sessions = _history()
    config = DynamicsConfig(rank=4)
    previous = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                            session_indices=sessions, as_of_date=dates[500], config=config)
    fitted = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                          session_indices=sessions, as_of_date=dates[501], config=config,
                          previous=previous)
    assert fitted.initialisation_date == dates[500]
    assert np.all(np.sum(previous.directions * fitted.directions, axis=0) > 0.)
    with pytest.raises(ValueError, match="earlier"):
        fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                     session_indices=sessions, as_of_date=dates[499], config=config,
                     previous=previous)


def test_spot_conditioned_ar_fits_after_removing_its_prescribed_loading() -> None:
    dates, states, spot, sessions = _history()
    spot = spot + .02 * states[:, 0]
    config = DynamicsConfig(rank=4, spot_orthogonal=True)
    fitted = fit_dynamics(dates=dates, states=states, spot_returns_into_date=spot,
                          session_indices=sessions, as_of_date=dates[-1], config=config)
    selected = np.arange(len(states) - config.window_transitions, len(states))
    before = (states[selected - 1] - fitted.state_mean) @ fitted.directions
    after = (states[selected] - fitted.state_mean) @ fitted.directions
    centered_before = before - before.mean(axis=0)
    centered_spot = spot[selected] - spot[selected].mean()
    spot_effect = centered_spot[:, None] * (fitted.spot_state_loading @ fitted.directions)
    target = after - after.mean(axis=0) - spot_effect
    cross = centered_before.T @ centered_before
    ridge = config.regression_ridge * np.trace(cross) / config.rank
    expected = np.linalg.solve(cross + ridge * np.eye(config.rank), centered_before.T @ target).T
    radius = np.max(np.abs(np.linalg.eigvals(expected)))
    if radius > config.max_spectral_radius:
        expected *= config.max_spectral_radius / radius
    np.testing.assert_allclose(fitted.ar_matrix, expected, atol=1e-13)
    raw_errors = after - before @ fitted.ar_matrix.T - fitted.ar_intercept
    np.testing.assert_allclose(fitted.residuals[:, 1:], raw_errors - spot_effect, atol=1e-13)
    assert fitted.training_rmse == pytest.approx(np.sqrt(np.mean(raw_errors**2)))
    assert fitted.spot_conditioned_persistence_rmse == pytest.approx(
        np.sqrt(np.mean((after - before - spot_effect)**2)))


def test_missing_previous_session_does_not_bridge_returns_or_discard_current_state() -> None:
    dates, states, spot, sessions = _history()
    spot[500] = np.nan
    sessions[500:] += 1
    options = dict(dates=dates, spot_returns_into_date=spot, session_indices=sessions,
        config=DynamicsConfig(rank=4))
    previous = fit_dynamics(states=states, as_of_date=dates[499], **options)
    current = fit_dynamics(states=states, as_of_date=dates[500], **options)
    assert current.as_of_date == dates[500]
    assert current.training_end == dates[499]
    assert current.residuals.shape == (252, 5)
    np.testing.assert_array_equal(current.residuals, previous.residuals)
    np.testing.assert_array_equal(current.directions, previous.directions)
    np.testing.assert_allclose(current.current_scores, (states[500] - current.state_mean) @ current.directions)
    assert not np.allclose(current.current_scores, previous.current_scores)
