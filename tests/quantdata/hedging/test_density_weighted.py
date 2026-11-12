"""Density metrics, native-unit inversion and rolling-coordinate alignment."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedging.model_audit.density_weighted import (
    _load,
    _save,
    density_metric,
    fit_weighted,
    recursive_native_paths,
)
from quantdata.hedging.model_audit.metrics import forecast_variance_calibration


def _arguments() -> dict:
    rng = np.random.default_rng(222)
    states = np.zeros((140, 6))
    spot = rng.normal(scale=.02, size=len(states))
    for index in range(1, len(states)):
        states[index] = .7 * states[index - 1] + spot[index] * np.arange(6) + rng.normal(scale=.03, size=6)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(len(states)))
    return dict(states=states, dates=dates, spot_returns_into_date=spot, session_indices=np.arange(len(states)),
                as_of_date=dates[-2], config=DynamicsConfig(window_transitions=100, rank=3, spot_orthogonal=True))


def test_metric_uses_origin_probability_mass_and_unit_forward_vol_weight() -> None:
    masses = np.array([[[.1, .2, .3, .4]], [[.2, .1, .4, .3]]])
    np.testing.assert_allclose(density_metric(masses), [.15, .35, 1.])
    with pytest.raises(ValueError, match="unit masses"):
        density_metric(masses * 2.)


def test_unit_metric_exactly_reproduces_original_estimation() -> None:
    arguments = _arguments()
    original = fit_dynamics(**arguments)
    weighted = fit_weighted(weights=np.ones(6), **arguments)
    np.testing.assert_array_equal(weighted.weighted_fit.directions, original.directions)
    np.testing.assert_array_equal(weighted.weighted_fit.ar_matrix, original.ar_matrix)
    current = arguments["states"][-2]
    expected = current + original.forecast_state_change(current) + original.residuals[:, :1] * original.spot_state_loading
    expected += original.residuals[:, 1:] @ original.directions.T
    np.testing.assert_allclose(weighted.next_native(current), expected, atol=1e-14)


def test_weighted_projection_replays_in_native_units() -> None:
    arguments = _arguments()
    metric = np.array([1e-10, .03, .2, .5, 1., 1.])
    model = fit_weighted(weights=metric, **arguments)
    fit = model.weighted_fit
    indices = np.arange(39, 139)
    before = arguments["states"][indices - 1]
    after = arguments["states"][indices]
    native_directions = model.unscale_changes(fit.directions.T).T
    np.testing.assert_allclose(native_directions.T @ (metric[:, None] * native_directions), np.eye(3), atol=1e-12)
    scores = (before * model.scale - fit.state_mean) @ fit.directions
    fitted_change = ((fit.ar_intercept + scores @ fit.ar_matrix.T - scores) @ fit.directions.T
                     + fit.residuals[:, :1] * fit.spot_state_loading + fit.residuals[:, 1:] @ fit.directions.T)
    replay = model.unscale_changes(fitted_change)
    projected_changes = ((after - before) * model.scale) @ fit.directions
    np.testing.assert_allclose((replay * model.scale) @ fit.directions, projected_changes, atol=1e-12)


def test_zero_mass_freezes_coordinate_without_probability_floor() -> None:
    arguments = _arguments()
    model = fit_weighted(weights=np.array([0., .03, .2, .5, 1., 1.]), **arguments)
    current = arguments["states"][-2]
    np.testing.assert_array_equal(model.next_native(current)[:, 0], np.full(100, current[0]))


def test_rolling_basis_maps_previous_loadings_into_new_metric() -> None:
    arguments = _arguments()
    previous = fit_weighted(weights=np.array([.01, .03, .2, .5, 1., 1.]), **arguments)
    arguments["as_of_date"] = arguments["dates"][-1]
    current = fit_weighted(weights=np.array([.02, .04, .3, .7, 1., 1.]), previous=previous, **arguments)
    assert current.weighted_fit.initialisation_date == previous.weighted_fit.as_of_date
    assert np.isfinite(current.next_native(arguments["states"][-1])).all()


def test_weighted_checkpoint_is_inference_reusable(tmp_path: Path) -> None:
    arguments = _arguments()
    fitted = fit_weighted(weights=np.array([.01, .03, .2, .5, 1., 1.]), **arguments)
    _save(fitted, tmp_path)
    restored = _load(tmp_path)
    np.testing.assert_array_equal(restored.weights, fitted.weights)
    np.testing.assert_array_equal(restored.next_native(arguments["states"][-2]), fitted.next_native(arguments["states"][-2]))


def test_recursive_first_step_uses_the_identical_joint_innovation_and_metric() -> None:
    arguments = _arguments()
    fitted = fit_weighted(weights=np.array([.01, .03, .2, .5, 1., 1.]), **arguments)
    current = arguments["states"][-2]
    values, _ = recursive_native_paths(fitted, current, paths=32, steps=3, seed=122)
    chosen = np.random.default_rng(122).integers(0, len(fitted.weighted_fit.residuals), 32)
    np.testing.assert_allclose(values[:, 0], fitted.next_native(current)[chosen], atol=1e-12)


def test_predictive_variance_uses_forecast_errors_not_unconditional_changes() -> None:
    means = np.array([[-10.], [0.], [10.]])
    actual = means + np.array([[-1.], [1.], [-1.]])
    result = forecast_variance_calibration(means, np.ones((3, 1, 1)), actual)
    assert result["conditional_variance_over_forecast_mse"] == pytest.approx(1.)
    assert result["forecast_error_squared_norm_mean"] == pytest.approx(1.)
