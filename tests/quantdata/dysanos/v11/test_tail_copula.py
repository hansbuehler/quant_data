"""Checks that scale-mixture tail dependence preserves the front marginal."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quantdata.dysanos.v11.tail_copula import normal_score_from_scale_mixture


def test_zero_log_variance_reduces_to_normal_rescaling() -> None:
    raw = np.array([-2.0, -0.2, 0.0, 0.7, 2.3])
    result = normal_score_from_scale_mixture(
        raw, stationary_log_variance=0.0, unconditional_scale=1.44
    )
    np.testing.assert_allclose(result, raw / 1.2)


def test_scale_mixture_transform_has_normal_spot_marginal() -> None:
    generator = np.random.default_rng(3)
    log_variance = 0.36
    scale = 0.9
    latent = generator.normal(scale=np.sqrt(log_variance), size=60_000)
    factor = np.sqrt(scale * np.exp(latent - 0.5 * log_variance))
    raw = factor * generator.standard_normal(len(factor))
    score = normal_score_from_scale_mixture(
        raw,
        stationary_log_variance=log_variance,
        unconditional_scale=scale,
    )
    assert abs(np.mean(score)) < 0.02
    assert abs(np.std(score) - 1.0) < 0.02
    assert abs(np.mean(norm.cdf(score) < 0.05) - 0.05) < 0.01


def test_conditional_ou_transform_preserves_front_marginal() -> None:
    generator = np.random.default_rng(8)
    stationary_variance = 0.36
    conditional_mean = 0.4
    conditional_variance = 0.13
    latent = generator.normal(
        loc=conditional_mean,
        scale=np.sqrt(conditional_variance),
        size=40_000,
    )
    raw = np.sqrt(np.exp(latent - 0.5 * stationary_variance)) * (
        generator.standard_normal(len(latent))
    )
    score = normal_score_from_scale_mixture(
        raw,
        stationary_log_variance=stationary_variance,
        unconditional_scale=1.0,
        conditional_log_variance_mean=conditional_mean,
        conditional_log_variance_variance=conditional_variance,
    )
    assert abs(np.mean(score)) < 0.02
    assert abs(np.std(score) - 1.0) < 0.02
