"""Tests for correlated front-expiry spot sampling."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from quantdata.dysanos.flex.front_spot_copula import (
    _front_quantile_spline,
    fit_gaussian_front_spot_copula,
)


def test_ridge_copula_recovers_the_spot_state_direction() -> None:
    generator = np.random.default_rng(412)
    states = generator.standard_normal((2_000, 8))
    direction = np.asarray((0.35, -0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0))
    scores = states @ direction + np.sqrt(1.0 - direction @ direction) * (
        generator.standard_normal(len(states))
    )

    fit = fit_gaussian_front_spot_copula(
        states,
        scores,
        fit_end="2024-12-31",
    )

    cosine = float(fit.state_loading @ direction) / (
        np.linalg.norm(fit.state_loading) * np.linalg.norm(direction)
    )
    assert cosine > 0.95
    assert 0.05 < fit.explained_fraction < 0.40


def test_front_quantile_spline_is_monotone_and_inverts_the_mixture_cdf() -> None:
    support = np.asarray((0.8, 1.0, 1.2), dtype=np.float64)
    density = np.asarray((0.2, 0.6, 0.2), dtype=np.float64)
    sqrt_variance = 0.08
    gaussian_score = np.linspace(-3.0, 3.0, 121)

    gross = _front_quantile_spline(
        support=support,
        density=density,
        sqrt_variance=sqrt_variance,
        gaussian_score=gaussian_score,
    )

    normalized_support = support / float(density @ support)
    z = (
        np.log(gross)[:, None]
        - np.log(normalized_support)[None, :]
        + 0.5 * sqrt_variance**2
    ) / sqrt_variance
    recovered = ndtr(z) @ density
    assert np.all(np.diff(gross) > 0.0)
    np.testing.assert_allclose(recovered, ndtr(gaussian_score), atol=2.0e-6)
