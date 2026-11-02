"""Direct dense references for the attenuation-aware covariance estimator."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.attenuation_estimation import (
    AttenuationFitConfig,
    AttenuationTrainConfig,
    fit_joint_loadings,
    joint_second_moment_score,
    prepare_estimation_arrays,
)


def test_factor_score_matches_explicit_joint_dense_covariance():
    generator = np.random.default_rng(42)
    parameters = generator.normal(size=(5, 3))
    attenuation = generator.uniform(.01, 2., size=(7, 5))
    observations = generator.normal(size=(7, 5))
    spots = generator.normal(size=7)
    expected = []
    for weight, observed, spot in zip(attenuation, observations, spots):
        joint = np.vstack((np.eye(1, 3), weight[:, None] * parameters / np.sqrt(5.)))
        covariance = joint @ joint.T
        target = np.r_[spot, observed]
        expected.append(np.sum(covariance**2) - 2. * target @ covariance @ target)
    actual = joint_second_moment_score(jnp.asarray(parameters), jnp.asarray(attenuation),
                                      jnp.asarray(observations), jnp.asarray(spots))
    assert float(actual) == pytest.approx(np.mean(expected), rel=1.e-12)
    gradient = jax.grad(joint_second_moment_score)(jnp.asarray(parameters), jnp.asarray(attenuation),
                                                jnp.asarray(observations), jnp.asarray(spots))
    step = 1.e-5
    plus, minus = parameters.copy(), parameters.copy()
    plus[2, 1] += step
    minus[2, 1] -= step
    difference = (joint_second_moment_score(plus, attenuation, observations, spots)
                  - joint_second_moment_score(minus, attenuation, observations, spots)) / (2. * step)
    assert float(gradient[2, 1]) == pytest.approx(float(difference), rel=1.e-7)


def _fixture():
    generator = np.random.default_rng(19)
    return dict(innovations=generator.normal(size=(20, 2, 3)), spot_scores=generator.normal(size=20),
                start_densities=np.broadcast_to(np.asarray([[1.e-30, .2, .8], [.1, .8, .1]]), (20, 2, 3)),
                initial_loadings=generator.normal(size=(2, 3, 2)))


def test_tiny_attenuation_does_not_divide_observed_innovations():
    source = _fixture()
    data = prepare_estimation_arrays(**source, config=AttenuationFitConfig())
    assert data['attenuation_rms'][0] < 1.e-25
    assert np.isfinite(data['initial_parameters']).all()
    assert np.max(np.abs(data['weighted_observations'])) < 5.
    np.testing.assert_allclose(np.mean(data['rms_scaled_attenuation']**2, axis=0), 1., rtol=1.e-14)
    np.testing.assert_allclose(data['weighted_observations'].mean(axis=0), 0., atol=1.e-15)


def test_unidentified_nodes_are_not_silently_assigned_zero_loading():
    source = _fixture()
    source['start_densities'] = source['start_densities'].copy()
    source['start_densities'][:, 0, 0] = 0.
    with pytest.raises(ValueError, match='never active'):
        prepare_estimation_arrays(**source, config=AttenuationFitConfig())
    data = prepare_estimation_arrays(**source, config=AttenuationFitConfig(density_floor=0.))
    np.testing.assert_array_equal(data['attenuation'], 1.)


def test_restart_preserves_optimizer_state_and_matches_uninterrupted_fit(tmp_path):
    data = prepare_estimation_arrays(**_fixture(), config=AttenuationFitConfig())
    first = AttenuationTrainConfig(minimum_epochs=2, maximum_epochs=3, checkpoint_epochs=1,
                                  gradient_tolerance=0., improvement_tolerance=0.)
    full = AttenuationTrainConfig(minimum_epochs=2, maximum_epochs=6, checkpoint_epochs=1,
                                 gradient_tolerance=0., improvement_tolerance=0.)
    report = fit_joint_loadings(data=data, output=tmp_path / 'resume', train=first, identity='fixture')
    assert report['status'] == 'budget_exhausted_not_converged'
    resumed = fit_joint_loadings(data=data, output=tmp_path / 'resume', train=full, identity='fixture')
    direct = fit_joint_loadings(data=data, output=tmp_path / 'direct', train=full, identity='fixture')
    assert resumed['epoch'] == direct['epoch'] == 6
    with np.load(tmp_path / 'resume' / 'checkpoint.npz') as left, np.load(tmp_path / 'direct' / 'checkpoint.npz') as right:
        np.testing.assert_array_equal(left['parameters'], right['parameters'])
        np.testing.assert_array_equal(left['optimizer_1'], right['optimizer_1'])


def test_budget_extension_preserves_decayed_learning_rate_and_plateau_state(tmp_path):
    data = prepare_estimation_arrays(**_fixture(), config=AttenuationFitConfig())
    short = AttenuationTrainConfig(minimum_epochs=2, maximum_epochs=3, checkpoint_epochs=1,
        plateau_epochs=1, improvement_tolerance=1.e6, gradient_tolerance=0.)
    full = replace(short, maximum_epochs=6)
    fit_joint_loadings(data=data, output=tmp_path / 'resume', train=short, identity='plateau')
    resumed = fit_joint_loadings(data=data, output=tmp_path / 'resume', train=full, identity='plateau')
    direct = fit_joint_loadings(data=data, output=tmp_path / 'direct', train=full, identity='plateau')
    assert resumed['learning_rate'] == direct['learning_rate'] < short.learning_rate
    assert resumed['stale_epochs'] == direct['stale_epochs']
    with np.load(tmp_path / 'resume/checkpoint.npz') as left, np.load(tmp_path / 'direct/checkpoint.npz') as right:
        np.testing.assert_array_equal(left['parameters'], right['parameters'])


def test_tighter_convergence_reopens_a_previously_completed_fit(tmp_path):
    data = prepare_estimation_arrays(**_fixture(), config=AttenuationFitConfig())
    loose = AttenuationTrainConfig(minimum_epochs=2, maximum_epochs=3, gradient_tolerance=1.e6)
    first = fit_joint_loadings(data=data, output=tmp_path, train=loose, identity='tolerance')
    assert first['status'] == 'converged' and first['epoch'] == 2
    tight = replace(loose, maximum_epochs=5, gradient_tolerance=0.)
    resumed = fit_joint_loadings(data=data, output=tmp_path, train=tight, identity='tolerance')
    assert resumed['epoch'] == 5
    assert resumed['status'] == 'budget_exhausted_not_converged'
