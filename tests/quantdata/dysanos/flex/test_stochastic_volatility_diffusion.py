"""Tests for the no-replay latent log-variance diffusion."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import digamma, polygamma

from quantdata.dysanos.flex.stochastic_volatility_diffusion import (
    LogVarianceDiffusionConfig,
    filter_log_variance_diffusion,
    fit_log_variance_diffusion,
    log_chi_square_moments,
)


def test_log_chi_square_moments_are_exact() -> None:
    mean, variance = log_chi_square_moments(200)

    assert mean == pytest.approx(float(digamma(100.0) + math.log(2.0 / 200.0)))
    assert variance == pytest.approx(float(polygamma(1, 100.0)))


def test_stationary_multiplier_has_unit_mean() -> None:
    generator = np.random.default_rng(912)
    config = LogVarianceDiffusionConfig(
        persistence=0.93,
        stationary_log_variance=0.48,
        unconditional_scale=1.0,
    )
    latent = generator.normal(
        scale=math.sqrt(config.stationary_log_variance),
        size=1_000_000,
    )
    multiplier = np.exp(latent - 0.5 * config.stationary_log_variance)

    assert np.mean(multiplier) == pytest.approx(1.0, abs=2.0e-3)
    assert config.log_variance_innovation_standard_deviation == pytest.approx(
        math.sqrt(0.48 * (1.0 - 0.93**2))
    )


def test_filter_resets_segments_and_reports_positive_scales() -> None:
    config = LogVarianceDiffusionConfig(
        persistence=0.8,
        stationary_log_variance=0.3,
        unconditional_scale=1.2,
    )
    radii = np.asarray((0.7, 1.4, 0.7, 1.4))
    starts = np.asarray((True, False, True, False))
    filtered = filter_log_variance_diffusion(
        radii,
        starts,
        dimension=200,
        config=config,
    )

    assert filtered.predicted_means[0] == pytest.approx(0.0)
    assert filtered.predicted_means[2] == pytest.approx(0.0)
    assert filtered.predicted_variances[0] == pytest.approx(0.3)
    assert filtered.predicted_variances[2] == pytest.approx(0.3)
    np.testing.assert_allclose(filtered.filtered_means[:2], filtered.filtered_means[2:])
    assert np.all(filtered.conditional_mean_scales > 0.0)
    assert np.all(np.isfinite(filtered.log_likelihood))


def test_fit_recovers_persistent_synthetic_log_variance() -> None:
    generator = np.random.default_rng(715)
    dimension = 200
    persistence = 0.86
    stationary_variance = 0.36
    unconditional_scale = 0.92
    count = 4_000
    latent = np.empty(count)
    latent[0] = generator.normal(scale=math.sqrt(stationary_variance))
    innovation_std = math.sqrt(stationary_variance * (1.0 - persistence**2))
    for index in range(1, count):
        latent[index] = (
            persistence * latent[index - 1]
            + innovation_std * generator.normal()
        )
    chi_square = generator.chisquare(dimension, size=count) / dimension
    radii = (
        unconditional_scale
        * np.exp(latent - 0.5 * stationary_variance)
        * chi_square
    )
    fit = fit_log_variance_diffusion(
        radii,
        np.asarray([True, *([False] * (count - 1))]),
        dimension=dimension,
    )

    assert fit.success
    assert fit.config.persistence == pytest.approx(persistence, abs=0.04)
    assert fit.config.stationary_log_variance == pytest.approx(
        stationary_variance, abs=0.06
    )
    assert fit.config.unconditional_scale == pytest.approx(
        unconditional_scale, abs=0.04
    )
    assert fit.negative_log_likelihood < fit.scaled_gaussian_negative_log_likelihood


def test_continuous_ou_parameters_reproduce_exact_persistence() -> None:
    config = LogVarianceDiffusionConfig(
        persistence=0.75,
        stationary_log_variance=0.4,
    )
    continuous = config.continuous_ou_parameters(step_sessions=5)

    assert math.exp(-5.0 * continuous["mean_reversion_per_session"]) == pytest.approx(
        config.persistence
    )
    expected_stationary_variance = (
        continuous["log_variance_diffusion_per_sqrt_session"] ** 2
        / (2.0 * continuous["mean_reversion_per_session"])
    )
    assert expected_stationary_variance == pytest.approx(
        config.stationary_log_variance
    )
