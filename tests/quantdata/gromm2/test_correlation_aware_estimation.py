from __future__ import annotations

import numpy as np
import pytest

from quantdata.gromm2.correlation_aware_estimation import (
    latent_loading,
    residual_spot_posterior,
)


def test_residual_spot_posterior_recovers_gaussian_conditional_moments() -> None:
    total_variance = np.asarray([0.09])
    heston_variance = np.asarray([0.04])
    variance_score = np.asarray([0.7])
    correlation = -0.6
    annualized_return = np.asarray([-0.11])
    result = residual_spot_posterior(
        annualized_spot_return=annualized_return,
        total_annual_variance=total_variance,
        heston_variance=heston_variance,
        heston_variance_score=variance_score,
        heston_correlation=correlation,
        residual_variance_floor=1.0e-8,
    )

    residual_variance = 0.05
    noise_variance = 0.04 * (1.0 - correlation**2)
    centered = annualized_return[0] - correlation * 0.2 * variance_score[0]
    observation_variance = residual_variance + noise_variance
    assert result.mean[0] == pytest.approx(
        np.sqrt(residual_variance) * centered / observation_variance
    )
    assert result.variance[0] == pytest.approx(
        noise_variance / observation_variance
    )
    assert result.conditional_return_score[0] == pytest.approx(
        centered / np.sqrt(observation_variance)
    )


def test_residual_spot_posterior_becomes_observed_without_heston_noise() -> None:
    result = residual_spot_posterior(
        annualized_spot_return=np.asarray([0.3, -0.6]),
        total_annual_variance=np.asarray([0.09, 0.09]),
        heston_variance=np.asarray([1.0e-12, 1.0e-12]),
        heston_variance_score=np.asarray([0.0, 0.0]),
        heston_correlation=0.0,
        residual_variance_floor=1.0e-8,
    )
    assert result.mean == pytest.approx(np.asarray([1.0, -2.0]), abs=1.0e-9)
    assert result.variance == pytest.approx(np.zeros(2), abs=1.0e-9)


def test_latent_loading_uses_posterior_second_moment() -> None:
    posterior_mean = np.asarray([0.2, -0.4, 0.6])
    posterior_variance = np.asarray([0.8, 0.6, 0.4])
    true_loading = np.asarray([2.0, -1.0])
    expected_complete_score = posterior_mean**2 + posterior_variance
    innovations = (
        expected_complete_score[:, None]
        / posterior_mean[:, None]
        * true_loading[None]
    )
    fitted = latent_loading(
        innovations,
        posterior_mean,
        posterior_variance,
    )
    assert fitted == pytest.approx(true_loading)
