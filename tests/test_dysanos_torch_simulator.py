import numpy as np
from scipy.linalg import expm
from types import SimpleNamespace

from quantdata.dysanos.grid.torch_simulator import (
    AR1PCA,
    ARSampler,
    _fit_continuous_ou_drift,
    fit_xar1_pca,
)


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
        num_steps=2, num_samples=1, dt=1.0, bandwidth=1.0
    )

    assert np.allclose(path[0, :, 0], [2.0, 1.7])
