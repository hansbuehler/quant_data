from __future__ import annotations

import numpy as np

from quantdata.dysanos.v11.daily_joint_dynamics import (
    fit_daily_joint_gaussian,
    sample_antithetic_joint_innovations,
)


def test_daily_joint_covariance_is_positive_definite() -> None:
    generator = np.random.default_rng(17)
    common = generator.standard_normal((200, 2))
    observations = common @ np.asarray(((1.0, 0.3, -0.2), (0.1, 0.8, 0.7)))
    observations += 0.05 * generator.standard_normal(observations.shape)
    fit = fit_daily_joint_gaussian(observations, factor_rank=2)
    eigenvalues = np.linalg.eigvalsh(fit.covariance)
    assert np.min(eigenvalues) > 0.0
    assert 0.9 < fit.retained_variance_fraction <= 1.0
    sample_covariance = np.cov(observations, rowvar=False)
    assert np.allclose(fit.covariance[:-1, -1], sample_covariance[:-1, -1])


def test_daily_joint_samples_are_antithetic_about_fitted_mean() -> None:
    generator = np.random.default_rng(21)
    observations = generator.standard_normal((100, 4))
    fit = fit_daily_joint_gaussian(observations, factor_rank=4)
    samples = sample_antithetic_joint_innovations(fit, count=20, seed=9)
    assert np.allclose(0.5 * (samples[:10] + samples[10:]), fit.mean[None, :])
