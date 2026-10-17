from __future__ import annotations

import numpy as np

from quantdata.gromm2.background import (
    cir_conditional_moments,
    fit_cir_quasi_likelihood,
    kernel_annual_volatilities,
)


def test_kernel_annual_volatilities_interpolates_total_variance() -> None:
    expiries = np.asarray([1.0, 2.0])
    variances = np.asarray([0.04, 0.18])
    actual = kernel_annual_volatilities(
        expiries,
        variances,
        np.asarray([1.0, 1.5, 3.0]),
        smoothness=0.25,
    )
    expected = np.sqrt(0.25 * np.asarray([0.04, 0.11, 0.18]) / [1.0, 1.5, 2.0])
    assert np.allclose(actual, expected)


def test_cir_fit_recovers_conditional_moment_parameters() -> None:
    generator = np.random.default_rng(20261014)
    observations = 30_000
    step = 1.0 / 252.0
    mean_reversion = 18.0
    long_run_variance = 0.08**2
    volatility_of_variance = 0.45
    origins = np.exp(
        generator.normal(np.log(long_run_variance), 0.25, observations)
    )
    conditional_mean, conditional_variance = cir_conditional_moments(
        origins,
        mean_reversion=mean_reversion,
        long_run_variance=long_run_variance,
        volatility_of_variance=volatility_of_variance,
        step_years=step,
    )
    endpoints = conditional_mean + np.sqrt(conditional_variance) * generator.normal(
        size=observations
    )
    positive = endpoints > 0.0
    estimate = fit_cir_quasi_likelihood(
        origins[positive],
        endpoints[positive],
        step_years=step,
    )
    assert np.isclose(estimate.mean_reversion, mean_reversion, rtol=0.15)
    assert np.isclose(estimate.long_run_variance, long_run_variance, rtol=0.12)
    assert np.isclose(
        estimate.volatility_of_variance,
        volatility_of_variance,
        rtol=0.08,
    )
