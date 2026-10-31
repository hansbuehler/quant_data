"""Stationary sampling must include both spot and residual innovations."""

import numpy as np
import pytest

from quantdata.dysanos.flex.stochastic_volatility_diffusion import LogVarianceDiffusionConfig
from quantdata.dysanos.v11.run_vix_histogram import sample_tail_states, stationary_state_law


def test_stationary_scalar_spot_loading_and_centering() -> None:
    mean, covariance = stationary_state_law(
        transition=np.array([[.8]]), equilibrium=np.array([2.]),
        residual_factor=np.array([[.3]]), spot_loading=np.array([.4]),
        spot_score_mean=.1,
    )
    assert mean[0] == pytest.approx(2 - .04 / .2)
    assert covariance[0, 0] == pytest.approx((.3**2 + .4**2) / (1 - .8**2))


def test_stationary_row_state_orientation() -> None:
    transition = np.array([[.5, .2], [0., .7]])
    equilibrium = np.array([1., 2.])
    factor = np.array([[.4, 0.], [.1, .3]])
    beta = np.array([.2, -.1])
    mean, covariance = stationary_state_law(
        transition=transition, equilibrium=equilibrium,
        residual_factor=factor, spot_loading=beta, spot_score_mean=.15,
    )
    np.testing.assert_allclose(mean, equilibrium + (mean - equilibrium) @ transition - .15 * beta)
    np.testing.assert_allclose(covariance, transition.T @ covariance @ transition + factor @ factor.T + np.outer(beta, beta))
    assert np.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize("a", [1., 1.1, float("nan")])
def test_reject_nonstationary_law(a: float) -> None:
    with pytest.raises(ValueError):
        stationary_state_law(
            transition=np.array([[a]]), equilibrium=np.zeros(1),
            residual_factor=np.eye(1), spot_loading=np.zeros(1), spot_score_mean=0.,
        )


def test_tail_sampler_normal_limit_and_antithetic_pairs() -> None:
    values, audit = sample_tail_states(
        transition=np.array([[.5]]), equilibrium=np.array([2.]),
        residual_factor=np.array([[.3]]), spot_loading=np.array([.4]), spot_score_mean=.1,
        config=LogVarianceDiffusionConfig(persistence=.8, stationary_log_variance=0., unconditional_scale=1.69),
        samples=20_000, burn_in=40, seed=12,
    )
    stationary_mean = 2 - .04 / .5
    np.testing.assert_allclose(values[:10_000] + values[10_000:], 2 * stationary_mean)
    assert np.var(values) == pytest.approx((.3**2 * 1.69 + .4**2) / (1 - .5**2), rel=.04)
    assert audit["initial_state_contraction_norm"] < 1e-8


def test_tail_sampler_preserves_gaussian_spot_score() -> None:
    values, _ = sample_tail_states(
        transition=np.zeros((1, 1)), equilibrium=np.zeros(1),
        residual_factor=np.array([[1e-8]]), spot_loading=np.ones(1), spot_score_mean=0.,
        config=LogVarianceDiffusionConfig(persistence=.85, stationary_log_variance=.6, unconditional_scale=.9),
        samples=40_000, burn_in=1, seed=13,
    )
    assert np.var(values) == pytest.approx(1., abs=.04)
    assert np.mean(values**4) / np.mean(values**2)**2 == pytest.approx(3., abs=.2)


def test_tail_sampler_has_lognormal_surface_tails() -> None:
    values, _ = sample_tail_states(
        transition=np.zeros((1, 1)), equilibrium=np.zeros(1),
        residual_factor=np.ones((1, 1)), spot_loading=np.zeros(1), spot_score_mean=0.,
        config=LogVarianceDiffusionConfig(persistence=.7, stationary_log_variance=.6, unconditional_scale=.9),
        samples=80_000, burn_in=1, seed=14,
    )
    assert np.var(values) == pytest.approx(.9, rel=.06)
    assert np.mean(values**4) / np.mean(values**2)**2 > 4.5


def test_tail_sampler_rejects_insufficient_burn_in() -> None:
    with pytest.raises(ValueError, match="Burn-in"):
        sample_tail_states(
            transition=np.array([[.99]]), equilibrium=np.zeros(1), residual_factor=np.eye(1),
            spot_loading=np.zeros(1), spot_score_mean=0.,
            config=LogVarianceDiffusionConfig(persistence=.7, stationary_log_variance=.6),
            samples=4, burn_in=10, seed=1,
        )
