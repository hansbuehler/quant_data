from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from quantdata.dysanos.grid.torch_simulator import (
    AR1PCA,
    ARSampler,
    ARSamplerConfig,
    GaussianSpotImportanceConfig,
    _fit_continuous_ou_drift,
    _importance_sample_gaussian_spot_paths,
    _importance_sample_spot_given_surface_paths,
    _sample_surface_then_spot_gaussian_noise,
    fit_surface_first_xar1_pca,
    fit_xar1_pca,
)


def test_truncated_ar_innovations_do_not_rescale_discarded_variance_by_default():
    assert ARSamplerConfig(num_alpha=5).rescale_var is False


def test_continuous_ou_fit_is_stable_and_fits_irregular_transitions():
    rng = np.random.default_rng(7)
    true_k = np.array([[0.35, 0.18], [-0.08, 0.22]])
    true_mu = np.array([0.4, -0.3])
    ndt = rng.choice([0.5, 1.0, 1.75], size=120)
    x0 = rng.normal(size=(len(ndt), 2))
    x1 = np.empty_like(x0)
    for i, h in enumerate(ndt):
        x1[i] = true_mu + (x0[i] - true_mu) @ expm(-h * true_k)

    initial_k = np.eye(2) * 0.1
    u0, fitted_k = _fit_continuous_ou_drift(x0, x1, ndt, initial_k)
    fitted_mu = (u0 @ np.linalg.inv(fitted_k))[0]
    predictions = np.empty_like(x0)
    for i, h in enumerate(ndt):
        predictions[i] = fitted_mu + (x0[i] - fitted_mu) @ expm(-h * fitted_k)

    assert np.all(np.linalg.eigvals(fitted_k).real > 0.0)
    assert np.max(np.abs(predictions - x1)) < 1e-6


def test_continuous_ou_fit_handles_differently_scaled_state_coordinates():
    k = np.array([[0.3, 2e-5], [-800.0, 0.4]])
    mu = np.array([2e-3, 4e4])
    ndt = np.resize(np.array([0.5, 1.0, 1.5]), 90)
    states = [np.array([1e-3, 3e4])]
    for h in ndt:
        states.append(mu + (states[-1] - mu) @ expm(-h * k))
    states = np.asarray(states)
    time_y = np.concatenate(([0.0], np.cumsum(ndt)))

    result = fit_xar1_pca(states, time_y, num_alpha=2, use_matrix_exp=True)

    assert np.all(np.linalg.eigvals(result.k).real > 0.0)
    assert np.max(np.abs(result.alpha)) < 1e-5
    assert np.max(np.abs(np.mean(result.alpha, axis=0))) < 1e-10
    assert result.fit_quality.r2 > 1.0 - 1e-10
    assert result.fit_quality.r2_by_state.shape == (2,)
    assert result.fit_quality.rmse < 1e-6
    assert result.fit_quality.rmse_by_state.shape == (2,)
    assert result.fit_quality.sample_mean.shape == (2,)
    assert result.fit_quality.sample_std.shape == (2,)
    assert np.allclose(result.fit_quality.fitted_mean, mu, atol=1e-6)
    assert np.allclose(
        result.fit_quality.mean_error,
        result.fit_quality.fitted_mean - result.fit_quality.sample_mean,
    )
    stats = result.fit_quality.summary_statistics()
    assert np.allclose(
        stats["mean_error_in_std"],
        result.fit_quality.mean_error / result.fit_quality.sample_std,
    )
    assert stats["log_spot_sample_mean"] == result.fit_quality.sample_mean[0]
    assert stats["log_spot_sample_std"] == result.fit_quality.sample_std[0]
    quality_summary = str(result.fit_quality)
    assert "mean fit RMS=" in quality_summary
    assert "log-spot mean: sample=" in quality_summary
    assert "sample std=" in quality_summary


def test_free_sampler_uses_euler_transition_when_matrix_exp_is_disabled():
    sampler = ARSampler.__new__(ARSampler)
    sampler.config = SimpleNamespace(use_matrix_exp=False)
    sampler.model = SimpleNamespace(config=SimpleNamespace(num_state=0))
    sampler.xstate = np.array([[2.0], [2.0]])
    sampler.rng = np.random.default_rng(1)
    sampler.cholesky = np.ones((1, 1))
    sampler.alpha_mean = np.zeros(1)
    sampler.pcaxar1 = AR1PCA(
        U0=np.array([[0.1]]),
        alpha=np.zeros((1, 1)),
        dU=np.zeros((1, 1)),
        k=np.array([[0.2]]),
        ndt=np.ones(1),
        mean_dt=1.0,
        state_center=np.zeros(1),
        state_scale=np.ones(1),
        vars_explained=np.ones(1),
        var_explained=1.0,
    )

    path = sampler.free_sample_xstates(
        num_steps=4, num_samples=1, dt=1.0, bandwidth=1.0
    )

    assert np.allclose(path[0, :, 0], [2.0, 1.7, 1.46, 1.268])

    conditional = sampler.free_sample_xstates(
        num_steps=4,
        num_samples=3,
        dt=1.0,
        bandwidth=1.0,
        random_start=False,
        initial_state=np.array([5.0]),
    )

    assert conditional.shape == (3, 4, 1)
    assert np.allclose(conditional[:, :, 0], [5.0, 4.1, 3.38, 2.804])


def test_sparse_conditional_sampler_starts_exactly_and_uses_offset_lengths():
    sampler = ARSampler.__new__(ARSampler)
    sampler.config = SimpleNamespace(use_matrix_exp=True)
    sampler.model = SimpleNamespace(config=SimpleNamespace(num_state=1))
    sampler.rng = np.random.default_rng(2)
    sampler.cholesky = np.ones((1, 1))
    sampler.pcaxar1 = AR1PCA(
        U0=np.array([[0.0, 0.5]]),
        alpha=np.zeros((2, 1)),
        dU=np.zeros((1, 2)),
        k=np.diag([0.0, 0.5]),
        ndt=np.ones(2),
        mean_dt=1.0,
        state_center=np.zeros(2),
        state_scale=np.ones(2),
        vars_explained=np.ones(2),
        var_explained=1.0,
        surface_first=True,
    )

    paths = sampler.sample_conditional_xstates_at_offsets(
        initial_state=np.array([2.0, 3.0]),
        step_offsets=np.array([0, 2, 5]),
        num_samples=4,
        dt=1.0,
    )

    expected_surface = [3.0, 1.0 + 2.0 * np.exp(-1.0), 1.0 + 2.0 * np.exp(-2.5)]
    assert paths.shape == (4, 3, 2)
    assert np.allclose(paths[:, :, 0], 2.0)
    assert np.allclose(paths[:, :, 1], expected_surface)


def test_gaussian_spot_importance_sampling_recovers_joint_path_moments():
    samples, steps = 80_000, 5
    path_rng = np.random.default_rng(71)
    A = np.array([[0.85, 0.10], [0.0, 0.80]])
    b = np.array([[0.02, -0.01]])
    Q = np.array([[0.04, 0.015], [0.015, 0.03]])
    noise_factor = np.linalg.cholesky(Q)
    reference = np.zeros((samples, steps, 2))
    reference[:, 0] = np.array([0.1, -0.2])
    for step in range(1, steps):
        reference[:, step] = (
            reference[:, step - 1] @ A
            + b
            + path_rng.normal(size=(samples, 2)) @ noise_factor.T
        )

    proposal, log_weight, component, _ = (
        _importance_sample_gaussian_spot_paths(
            reference_paths=reference,
            A=A,
            b=b,
            Q=Q,
            target_steps=(2, 4),
            config=GaussianSpotImportanceConfig(
                spot_std_multiplier=4.0,
                base_probability=0.2,
            ),
            rng=np.random.default_rng(72),
        )
    )
    weight = np.exp(log_weight)
    weighted_mean = np.average(proposal[:, 4], axis=0, weights=weight)
    weighted_variance = np.average(
        np.square(proposal[:, 4] - weighted_mean), axis=0, weights=weight
    )

    assert np.bincount(component, minlength=3) == pytest.approx(
        [0.2 * samples, 0.4 * samples, 0.4 * samples], rel=0.02
    )
    assert weight.mean() == pytest.approx(1.0, abs=0.01)
    assert weighted_mean == pytest.approx(reference[:, 4].mean(axis=0), abs=0.006)
    assert weighted_variance == pytest.approx(
        reference[:, 4].var(axis=0), abs=0.006
    )
    assert proposal[:, 4, 0].std() > 2.0 * reference[:, 4, 0].std()


def test_gaussian_spot_importance_config_rejects_invalid_mixture():
    with pytest.raises(ValueError, match="greater than one"):
        GaussianSpotImportanceConfig(spot_std_multiplier=1.0)
    with pytest.raises(ValueError, match="strictly between"):
        GaussianSpotImportanceConfig(base_probability=0.0)


def test_surface_first_gaussian_noise_has_requested_joint_covariance():
    covariance = np.array(
        [
            [0.09, 0.025, -0.012],
            [0.025, 0.04, 0.01],
            [-0.012, 0.01, 0.03],
        ]
    )
    noise = _sample_surface_then_spot_gaussian_noise(
        covariance=covariance,
        size=(120_000,),
        rng=np.random.default_rng(91),
    )

    assert noise.shape == (120_000, 3)
    assert np.cov(noise, rowvar=False, bias=True) == pytest.approx(
        covariance, abs=9.0e-4
    )


def test_surface_first_fit_leaves_surface_autonomous_and_spot_nonstationary():
    rng = np.random.default_rng(92)
    observations = 1_200
    states = np.zeros((observations, 3))
    surface_A = np.array([[0.88, 0.04], [-0.02, 0.81]])
    surface_mean = np.array([0.3, -0.2])
    surface_factor = np.array([[0.08, 0.0], [0.03, 0.06]])
    for step in range(observations - 1):
        common_noise = rng.normal(size=2)
        surface_noise = common_noise @ surface_factor.T
        states[step + 1, 1:] = (
            surface_mean
            + (states[step, 1:] - surface_mean) @ surface_A
            + surface_noise
        )
        states[step + 1, 0] = (
            states[step, 0]
            + 0.001
            + common_noise @ np.array([0.012, -0.006])
            + 0.02 * rng.normal()
        )
    time_y = np.arange(observations, dtype=float) / 252.0

    result = fit_surface_first_xar1_pca(
        states,
        time_y,
        num_alpha=None,
        use_matrix_exp=True,
    )
    changed_spot = states.copy()
    changed_spot[:, 0] = np.cumsum(rng.normal(size=observations))
    changed_result = fit_surface_first_xar1_pca(
        changed_spot,
        time_y,
        num_alpha=None,
        use_matrix_exp=True,
    )

    assert result.surface_first
    assert result.num_surface_alpha == 2
    assert result.alpha.shape == (observations - 1, 3)
    assert np.allclose(result.k[0], 0.0)
    assert np.allclose(result.k[:, 0], 0.0)
    assert np.all(np.linalg.eigvals(result.k[1:, 1:]).real > 0.0)
    assert np.count_nonzero(np.isclose(np.linalg.eigvals(result.k), 0.0)) == 1
    assert np.allclose(result.k[1:, 1:], changed_result.k[1:, 1:])
    assert np.allclose(result.U0[0, 1:], changed_result.U0[0, 1:])
    reconstructed_residual = result.alpha @ result.dU
    assert np.max(np.abs(np.mean(reconstructed_residual, axis=0))) < 1.0e-10
    assert result.spot_residual_std > 0.0
    assert "fitted mean=N/A (unit root)" in str(result.fit_quality)
    assert "nan" not in str(result.fit_quality).lower()


def test_invariant_surface_distribution_excludes_unit_root_spot():
    sampler = ARSampler.__new__(ARSampler)
    sampler.model = SimpleNamespace(config=SimpleNamespace(num_state=2))
    sampler.rng = np.random.default_rng(93)
    sampler.pcaxar1 = AR1PCA(
        U0=np.array([[0.01, 0.3, -0.1]]),
        alpha=np.tile(np.array([[1.0, -0.5, 0.25], [-1.0, 0.5, -0.25]]), (200, 1)),
        dU=np.array(
            [[0.2, 0.1, 0.0], [0.1, 0.02, 0.08], [0.15, 0.0, 0.05]]
        ),
        k=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.4, 0.1], [0.0, -0.05, 0.3]]
        ),
        ndt=np.ones(400),
        mean_dt=1.0,
        state_center=np.zeros(3),
        state_scale=np.ones(3),
        vars_explained=np.ones(3),
        var_explained=1.0,
        surface_first=True,
        num_surface_alpha=2,
    )

    mean, covariance = sampler.invariant_surface_distribution()
    samples = sampler.sample_invariant_surface_states(100_000)

    assert mean == pytest.approx(
        np.linalg.solve(sampler.pcaxar1.k[1:, 1:].T, np.array([0.3, -0.1]))
    )
    assert samples.mean(axis=0) == pytest.approx(mean, abs=0.004)
    assert np.cov(samples, rowvar=False, bias=True) == pytest.approx(
        covariance, abs=0.002
    )


def test_conditional_spot_importance_preserves_surface_paths_and_moments():
    samples, steps = 100_000, 6
    rng = np.random.default_rng(94)
    A = np.array([[1.0, 0.0], [0.08, 0.82]])
    b = np.array([[0.003, -0.01]])
    Q = np.array([[0.05, 0.018], [0.018, 0.035]])
    reference = np.zeros((samples, steps, 2))
    reference[:, 0] = np.array([0.1, -0.2])
    for step in range(steps - 1):
        noise = _sample_surface_then_spot_gaussian_noise(
            covariance=Q,
            size=(samples,),
            rng=rng,
        )
        reference[:, step + 1] = reference[:, step] @ A + b + noise

    proposal, log_weight, component, _ = (
        _importance_sample_spot_given_surface_paths(
            reference_paths=reference,
            A=A,
            b=b,
            Q=Q,
            target_steps=(3, 5),
            config=GaussianSpotImportanceConfig(
                spot_std_multiplier=4.0,
                base_probability=0.2,
            ),
            rng=np.random.default_rng(95),
        )
    )
    weight = np.exp(log_weight)
    weighted_mean = np.average(proposal[:, 5], axis=0, weights=weight)
    weighted_variance = np.average(
        np.square(proposal[:, 5] - weighted_mean), axis=0, weights=weight
    )

    assert np.array_equal(proposal[..., 1:], reference[..., 1:])
    assert np.bincount(component, minlength=3) == pytest.approx(
        [0.2 * samples, 0.4 * samples, 0.4 * samples], rel=0.02
    )
    assert weight.mean() == pytest.approx(1.0, abs=0.01)
    assert weighted_mean == pytest.approx(reference[:, 5].mean(axis=0), abs=0.006)
    assert weighted_variance == pytest.approx(
        reference[:, 5].var(axis=0), abs=0.008
    )
    assert proposal[:, 5, 0].std() > 2.0 * reference[:, 5, 0].std()
