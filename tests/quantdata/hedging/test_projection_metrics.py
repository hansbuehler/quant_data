"""Weighted affine reconstruction and observable-metric native preimages."""

from datetime import date, timedelta

import numpy as np
import pytest

from quantdata.hedge2.dlv import DlvConfig
from quantdata.hedge2.dynamics import DynamicsConfig
from quantdata.hedge2.state import DailyState
from quantdata.hedging.model_audit.projection_metrics import (
    affine_projection,
    fit_projected,
    metric_projection,
    observable_measures,
    projected_scenarios,
)


def test_affine_probability_metric_prioritizes_central_coordinates() -> None:
    random = np.random.default_rng(42)
    values = random.normal(size=(250, 3)) * [10., 1., .01]
    encoder, decoder, _ = affine_projection(values, np.array([1e-8, 1., 0.]), rank=1)
    np.testing.assert_allclose(encoder.T @ decoder, np.eye(1), atol=1e-12)
    reconstruction = (values - values.mean(axis=0)) @ encoder @ decoder.T
    assert np.var(reconstruction[:, 1]) / np.var(values[:, 1]) > .99
    np.testing.assert_array_equal(reconstruction[:, 2], np.zeros(250))


def test_metric_preimages_recover_output_pca_and_native_scores() -> None:
    random = np.random.default_rng(782)
    changes = random.normal(size=(80, 12))
    changes -= changes.mean(axis=0)
    span_encoder, span_decoder, _ = affine_projection(changes, np.linspace(.01, 1., 12), rank=12)
    jacobian = random.normal(size=(12, 7))
    image_jacobian = span_decoder.T @ jacobian
    encoder, decoder, coverage = metric_projection(changes, span_encoder, image_jacobian, rank=4)
    np.testing.assert_allclose(encoder.T @ decoder, np.eye(4), atol=1e-11)
    images = changes @ jacobian
    _, singular, right = np.linalg.svd(images, full_matrices=False)
    expected = images @ right[:4].T @ right[:4]
    np.testing.assert_allclose(changes @ encoder @ decoder.T @ jacobian, expected, atol=1e-10)
    assert coverage == pytest.approx(np.sum(singular[:4]**2) / np.sum(singular**2))


def test_projected_training_replay_preserves_paired_spot_and_factor_law() -> None:
    random = np.random.default_rng(23)
    levels = np.zeros((61, 10))
    spot = random.normal(scale=.01, size=61)
    for index in range(1, len(levels)):
        levels[index] = .8 * levels[index - 1] + random.normal(scale=.03, size=10) + spot[index]
    encoder, decoder, _ = affine_projection(levels, np.linspace(.1, 1., 10), rank=6)
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(61))
    fit = fit_projected(levels=levels, dates=dates, spots=spot, sessions=np.arange(61), encoder=encoder,
        decoder=decoder, config=DynamicsConfig(window_transitions=60, rank=6, spot_orthogonal=True))
    current = levels[-1]
    result = projected_scenarios(current, encoder, decoder, fit)
    expected = (fit.forecast_state_change(current @ encoder) + fit.residuals[:, :1] * fit.spot_state_loading
                + fit.residuals[:, 1:] @ fit.directions.T) @ decoder.T + current
    np.testing.assert_array_equal(result, expected)
    assert fit.training_end == dates[-1]
    centered_spot = spot[1:] - spot[1:].mean()
    full_beta = centered_spot @ np.diff(levels, axis=0) / (centered_spot @ centered_spot)
    perpendicular = full_beta - (full_beta @ encoder) @ decoder.T
    corrected = projected_scenarios(current, encoder, decoder, fit, native_spot_loading=full_beta)
    np.testing.assert_allclose(corrected - result, fit.residuals[:, :1] * perpendicular, atol=1e-14)
    assert np.linalg.norm(perpendicular) > 0.


def test_projected_fit_preserves_exact_session_gaps() -> None:
    random = np.random.default_rng(904)
    levels = random.normal(size=(62, 9))
    dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(62))
    sessions = np.arange(62)
    sessions[31:] += 1
    spots = random.normal(scale=.01, size=62)
    spots[[0, 31]] = np.nan
    encoder, decoder, _ = affine_projection(levels, np.ones(9), rank=6)
    fitted = fit_projected(levels=levels, dates=dates, spots=spots, sessions=sessions,
        encoder=encoder, decoder=decoder, config=DynamicsConfig(window_transitions=60, rank=6))
    assert fitted.residuals.shape == (60, 7)
    assert fitted.training_start == dates[0]
    assert fitted.training_end == dates[-1]


def test_surface_metrics_use_itm_premiums_and_absolute_origin_grid_masses() -> None:
    config = DlvConfig(business_dtes=(1., 3.), num_inner_strikes=10)
    native = np.tile(np.r_[np.full(10, 2. * np.log(.2)), np.log(.2)], 2)
    reference = np.r_[.5, np.linspace(.8, 1.2, 10), 1.5]
    state = DailyState(day=date(2024, 4, 1), native=native, reference_support=reference, config=config)
    pricer = state.decode().pricer
    strikes = np.tile([.99, 1., 1.01], (2, 1))
    premium = np.concatenate([pricer.european_at(expiry_y=day / 252., strikes=row, is_call=row < 1.)
                              for day, row in zip(config.business_dtes, strikes, strict=True)])
    measures, guards = observable_measures(state, native[None], strikes=strikes,
        business_dtes=np.asarray(config.business_dtes), origin_support=pricer.support, origin_premium=premium)
    np.testing.assert_allclose(measures["itm_premium"], np.ones((1, 6)), atol=1e-12)
    masses = measures["probability_changes"].reshape(1, 2, -1)
    np.testing.assert_allclose(masses.sum(axis=2), 1., atol=1e-12)
    np.testing.assert_allclose(masses[0], pricer.marginal_densities, atol=1e-11)
    np.testing.assert_allclose(measures["straddle_sqrt_time"],
        measures["straddle_spot"] / np.repeat(np.sqrt(np.asarray(config.business_dtes) / 252.), 3))
    assert guards["lower_clips"] == guards["upper_clips"] == 0
