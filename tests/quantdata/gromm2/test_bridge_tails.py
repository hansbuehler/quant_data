from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from quantdata.gromm2.bridge import (
    advance_front_density,
    front_density_volatility,
    sample_front_index,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('tail_probability', [1.e-40, 1.e-100, 1.e-200])
def test_front_loading_preserves_small_positive_tails(tail_probability: float) -> None:
    density = jnp.asarray([tail_probability, .25, .75, tail_probability])
    remaining = 1. / 252.
    actual = np.asarray(front_density_volatility(density, remaining_time=jnp.asarray(remaining)))
    expected = np.exp(-.5 * ndtri(tail_probability)**2) / np.sqrt(2. * np.pi * remaining)
    np.testing.assert_allclose(actual[[0, -1]], [-expected, expected], rtol=2.e-11, atol=0.)


@pytest.mark.parametrize('tail_probability', [1.e-40, 1.e-100])
@pytest.mark.parametrize('rare_index', [0, 1])
def test_reveal_preserves_small_tail_quantiles(tail_probability: float, rare_index: int) -> None:
    density = np.ones(2) - tail_probability
    density[rare_index] = tail_probability
    actual = advance_front_density(jnp.asarray(density), jnp.asarray(0.),
        remaining_time=jnp.asarray(1.), step_size=jnp.asarray(.5), final_step=jnp.asarray(False))
    expected = ndtr(np.sqrt(2.) * ndtri(tail_probability))
    np.testing.assert_allclose(np.asarray(actual)[rare_index], expected, rtol=2.e-11, atol=0.)


def test_terminal_sampling_can_select_a_tiny_upper_tail() -> None:
    density = jnp.asarray([1., 1.e-40])
    assert int(sample_front_index(density, jnp.asarray(15.))) == 1


def test_terminal_sampling_skips_zero_mass_lower_nodes() -> None:
    density = jnp.asarray([0., .5, .5])
    assert int(sample_front_index(density, jnp.asarray(-40.))) == 1


@pytest.mark.parametrize('rare_index', [0, 1])
def test_tiny_tail_reveal_has_correct_conditional_mean(rare_index: int) -> None:
    probability = 1.e-100
    density = np.ones(2) - probability
    density[rare_index] = probability
    abscissa, weights = np.polynomial.hermite.hermgauss(256)
    batched = jnp.broadcast_to(jnp.asarray(density), (len(abscissa), 2))
    revealed = advance_front_density(batched, jnp.asarray(np.sqrt(2.) * abscissa),
        remaining_time=jnp.asarray(1.), step_size=jnp.asarray(.5), final_step=jnp.asarray(False))
    mean = np.dot(weights / np.sqrt(np.pi), np.asarray(revealed)[:, rare_index])
    np.testing.assert_allclose(mean, probability, rtol=2.e-9, atol=0.)


def test_tiny_interior_bin_has_nonzero_correct_loading() -> None:
    density = jnp.asarray([.25, 1.e-30, .75])
    actual = front_density_volatility(density, remaining_time=jnp.asarray(1.))
    np.testing.assert_allclose(np.asarray(actual)[1], 1.e-30 * ndtri(.25), rtol=2.e-13, atol=0.)


@pytest.mark.parametrize('normal', [-2., 0., 2.])
def test_tiny_interior_bin_reveal_preserves_its_probability(normal: float) -> None:
    density = jnp.asarray([.25, 1.e-30, .75])
    actual = advance_front_density(density, jnp.asarray(normal), remaining_time=jnp.asarray(1.),
        step_size=jnp.asarray(.5), final_step=jnp.asarray(False))
    quantile = ndtri(.25)
    shifted = np.sqrt(2.) * quantile - normal
    expected = 1.e-30 * np.sqrt(2.) * np.exp(.5 * (quantile**2 - shifted**2))
    np.testing.assert_allclose(np.asarray(actual)[1], expected, rtol=2.e-13, atol=0.)


def test_loading_forward_derivative_is_finite_on_fixed_support_boundary() -> None:
    density = jnp.asarray([0., .25, .75, 0.])
    tangent = jnp.asarray([0., 1., -1., 0.])
    _, actual = jax.jvp(lambda values: front_density_volatility(values, remaining_time=jnp.asarray(1.)),
        (density,), (tangent,))
    expected = np.asarray([0., ndtri(.25), -ndtri(.25), 0.])
    np.testing.assert_allclose(actual, expected, rtol=2.e-13, atol=0.)


def test_narrow_prior_bin_can_resolve_near_full_information() -> None:
    density = jnp.asarray([.25, .0002, .7498])
    step = 1. - 1.e-10
    normal = ndtri(.25) / np.sqrt(step)
    actual = advance_front_density(density, jnp.asarray(normal), remaining_time=jnp.asarray(1.),
        step_size=jnp.asarray(step), final_step=jnp.asarray(False))
    np.testing.assert_allclose(np.asarray(actual)[1], .5, rtol=0., atol=1.e-8)
