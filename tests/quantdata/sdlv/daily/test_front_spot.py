from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from quantdata.sdlv.daily import DATASET_PATH
from quantdata.sdlv.daily.front_spot.audit import _latent_call_prices, _summarize
from quantdata.sdlv.daily.front_spot.estimation import (
    AlignedDlvHistory,
    JointInnovationFit,
    fit_achievable_physical_drift,
    fit_joint_innovations,
)
from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_exercise_probabilities,
    black_mixture_call_prices,
    black_mixture_call_second_moments,
    decode_density_tower,
    normal_interval_probabilities,
    one_day_physical_step,
    one_day_risk_neutral_step,
    one_day_risk_neutral_substepped,
    project_physical_log_drift,
    risk_neutral_dlv_drift,
    sample_front_index,
    stabilized_relative_drift_increment,
    substep_brownian_weights,
)
from quantdata.sdlv.daily.front_spot.paper_experiments import (
    implied_volatility_from_calls,
)


def _small_surface() -> tuple[jax.Array, jax.Array]:
    support = jnp.asarray([0.50, 0.80, 1.00, 1.20, 1.50, 2.00])
    dlvsq = jnp.asarray(
        [
            [0.04, 0.035, 0.03, 0.04],
            [0.045, 0.04, 0.035, 0.045],
            [0.05, 0.045, 0.04, 0.05],
        ]
    )
    return support, dlvsq


def test_daily_dataset_contains_reference_spots() -> None:
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        dates = dataset["dates"]
        spots = np.asarray(dataset["spot_ref"], dtype=np.float64)
    assert spots.shape == dates.shape == (891,)
    assert np.all(np.isfinite(spots) & (spots > 0.0))
    assert spots[0] == pytest.approx(4796.56005859375)
    assert spots[-1] == pytest.approx(6460.259765625)


def test_density_tower_preserves_mass_and_forward() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    np.testing.assert_allclose(np.asarray(tower.sum(axis=-1)), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(tower @ support), 1.0, atol=1.0e-13)
    assert float(jnp.min(tower)) >= 0.0


def test_risk_neutral_drift_cancels_density_drift_nodewise() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = jnp.asarray(
        [
            [[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]],
        ]
    )
    densities, density_volatility, relative_drift = risk_neutral_dlv_drift(
        support,
        tower[0],
        dlvsq[1:],
        loadings,
        dt=1.0 / 252.0,
    )
    cancellation = dlvsq[1:] * (
        relative_drift * densities[:, 1:-1]
        + jnp.sum(loadings * density_volatility[:, 1:-1, :], axis=-1)
    )
    np.testing.assert_allclose(np.asarray(cancellation), 0.0, atol=2.0e-14)


def test_one_day_step_preserves_positive_density_tower() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = jnp.asarray(
        [
            [[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]],
        ]
    )
    result = one_day_risk_neutral_step(
        support,
        RiskNeutralState(tower[0], dlvsq[1:], jnp.asarray(1.0)),
        loadings,
        jnp.asarray([0.25, -0.50]),
        jnp.asarray(0.75),
    )
    np.testing.assert_allclose(
        np.asarray(result.density_tower.sum(axis=-1)), 1.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(result.density_tower @ support),
        np.asarray(result.latent_spot),
        atol=1.0e-13,
    )
    assert bool(jnp.all(jnp.isfinite(result.state.future_dlvsq)))
    assert float(jnp.min(result.state.future_dlvsq)) > 0.0


def test_joint_innovation_fit_orients_first_factor_to_spot() -> None:
    generator = np.random.default_rng(42)
    sample_count = 400
    origin = generator.normal(size=(sample_count, 2, 3))
    spot = generator.normal(size=sample_count)
    orthogonal = generator.normal(size=sample_count)
    loading_spot = np.linspace(0.01, 0.06, 6).reshape(2, 3)
    loading_orthogonal = np.linspace(-0.03, 0.02, 6).reshape(2, 3)
    residual = (
        spot[:, None, None] * loading_spot
        + orthogonal[:, None, None] * loading_orthogonal
    )
    endpoint = 0.1 + 0.9 * origin + residual
    cdf = np.asarray(jax.scipy.special.ndtr(jnp.asarray(spot)))
    history = AlignedDlvHistory(
        dates=tuple(f"2024-01-{index + 1:02d}" for index in range(sample_count)),
        origin_indices=np.arange(sample_count, dtype=np.int64),
        endpoint_indices=np.arange(sample_count, dtype=np.int64),
        origin_log_dlvsq=origin,
        endpoint_log_dlvsq=endpoint,
        spot_gaussian_scores=spot,
        front_cdf_values=cdf,
    )
    fit = fit_joint_innovations(history, factor_rank=2)
    daily_first_loading = fit.annualized_loadings[..., 0] / np.sqrt(252.0)
    np.testing.assert_allclose(daily_first_loading, loading_spot, atol=4.0e-3)
    np.testing.assert_allclose(
        fit.factor_scores.T @ fit.factor_scores / sample_count,
        np.eye(2),
        atol=2.0e-2,
    )


def test_physical_drift_projection_recovers_attainable_component() -> None:
    generator = np.random.default_rng(7)
    sample_count = 12
    origin = generator.normal(size=(sample_count, 2, 3))
    slope = np.full((2, 3), 0.95)
    intercept = np.full((2, 3), 0.01)
    loadings = generator.normal(scale=0.2, size=(2, 3, 2))
    q_drift = generator.normal(scale=0.1, size=(sample_count, 2, 3))
    risk_prices = generator.normal(scale=0.3, size=(sample_count, 2))
    empirical_drift = q_drift + np.einsum("ijn,sn->sij", loadings, risk_prices)
    endpoint = origin + empirical_drift / 252.0
    history = AlignedDlvHistory(
        dates=("2024-01-01",),
        origin_indices=np.arange(sample_count, dtype=np.int64),
        endpoint_indices=np.arange(sample_count, dtype=np.int64),
        origin_log_dlvsq=origin,
        endpoint_log_dlvsq=endpoint,
        spot_gaussian_scores=np.zeros(sample_count),
        front_cdf_values=np.full(sample_count, 0.5),
    )
    fitted_intercept = (
        empirical_drift / 252.0 + origin - slope * origin
    ).mean(axis=0)
    innovations = JointInnovationFit(
        conditional_intercept=fitted_intercept,
        conditional_slope=slope,
        centered_residuals=np.empty((sample_count, 6)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        annualized_loadings=loadings,
        daily_eigenvalues=np.ones(1),
        explained_variance_fraction=1.0,
        factor_scores=np.empty((sample_count, 2)),
    )
    conditional_empirical = 252.0 * (
        fitted_intercept + slope * origin - origin
    )
    target_risk_prices = np.linalg.lstsq(
        loadings.reshape(-1, 2),
        (conditional_empirical - q_drift).reshape(sample_count, -1).T,
        rcond=None,
    )[0].T
    result = fit_achievable_physical_drift(
        history,
        innovations,
        q_drift,
        np.ones_like(q_drift),
        ridge=0.0,
    )
    np.testing.assert_allclose(result.market_price_of_risk, target_risk_prices)
    np.testing.assert_allclose(
        result.achievable_log_drift,
        q_drift + np.einsum("ijn,sn->sij", loadings, target_risk_prices),
    )


def test_jax_physical_projection_matches_numpy_estimator() -> None:
    generator = np.random.default_rng(11)
    sample_count = 5
    expiry_count = 2
    strike_count = 3
    factor_count = 2
    origin = generator.normal(size=(sample_count, expiry_count, strike_count))
    q_drift = generator.normal(scale=0.1, size=origin.shape)
    loadings = generator.normal(
        scale=0.2,
        size=(expiry_count, strike_count, factor_count),
    )
    intercept = generator.normal(scale=0.01, size=(expiry_count, strike_count))
    slope = generator.uniform(0.85, 0.99, size=(expiry_count, strike_count))
    density = generator.uniform(size=(sample_count, expiry_count, strike_count + 2))
    density /= density.sum(axis=-1, keepdims=True)
    weights = density[..., 1:-1]
    weights /= weights.sum(axis=-1, keepdims=True)
    history = AlignedDlvHistory(
        dates=("2024-01-01",),
        origin_indices=np.arange(sample_count, dtype=np.int64),
        endpoint_indices=np.arange(sample_count, dtype=np.int64),
        origin_log_dlvsq=origin,
        endpoint_log_dlvsq=origin,
        spot_gaussian_scores=np.zeros(sample_count),
        front_cdf_values=np.full(sample_count, 0.5),
    )
    innovations = JointInnovationFit(
        conditional_intercept=intercept,
        conditional_slope=slope,
        centered_residuals=np.empty((sample_count, expiry_count * strike_count)),
        spot_score_mean=0.0,
        spot_score_std=1.0,
        annualized_loadings=loadings,
        daily_eigenvalues=np.ones(factor_count - 1),
        explained_variance_fraction=1.0,
        factor_scores=np.empty((sample_count, factor_count)),
    )
    expected = fit_achievable_physical_drift(
        history,
        innovations,
        q_drift,
        weights,
    )
    empirical, achievable, risk_price = project_physical_log_drift(
        jnp.asarray(origin),
        jnp.asarray(q_drift),
        jnp.asarray(density),
        jnp.asarray(loadings),
        jnp.asarray(intercept),
        jnp.asarray(slope),
        dt=1.0 / 252.0,
    )
    np.testing.assert_allclose(np.asarray(empirical), expected.empirical_log_drift)
    np.testing.assert_allclose(np.asarray(achievable), expected.achievable_log_drift)
    np.testing.assert_allclose(
        np.asarray(risk_price), expected.market_price_of_risk
    )


def test_one_day_physical_step_preserves_algebraic_density_invariants() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = jnp.asarray(
        [
            [[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]],
        ]
    )
    result = one_day_physical_step(
        support,
        RiskNeutralState(tower[0], dlvsq[1:], jnp.asarray(1.0)),
        loadings,
        jnp.zeros_like(dlvsq[1:]),
        jnp.ones_like(dlvsq[1:]),
        jnp.asarray([0.25, -0.50]),
        jnp.asarray(0.75),
    )
    np.testing.assert_allclose(
        np.asarray(result.density_tower.sum(axis=-1)), 1.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(result.density_tower @ support),
        np.asarray(result.latent_spot),
        atol=1.0e-13,
    )
    assert bool(jnp.all(jnp.isfinite(result.market_price_of_risk)))
    assert bool(jnp.all(jnp.isfinite(result.state.future_dlvsq)))
    assert float(jnp.min(result.state.future_dlvsq)) > 0.0


def test_single_substep_matches_frozen_one_day_update() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = jnp.asarray(
        [
            [[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]],
        ]
    )
    state = RiskNeutralState(tower[0], dlvsq[1:], jnp.asarray(1.0))
    normal = jnp.asarray([0.25, -0.50])
    expected = one_day_risk_neutral_step(
        support,
        state,
        loadings,
        normal,
        jnp.asarray(0.75),
    )
    result = one_day_risk_neutral_substepped(
        support,
        state,
        loadings,
        normal[None, :],
        jnp.asarray(0.75),
    )
    np.testing.assert_allclose(
        np.asarray(result.state.future_dlvsq),
        np.asarray(expected.state.future_dlvsq),
    )
    np.testing.assert_allclose(
        np.asarray(result.density_tower),
        np.asarray(expected.density_tower),
    )
    np.testing.assert_allclose(np.asarray(result.spot), np.asarray(expected.spot))


def test_substepped_front_sample_uses_full_daily_spot_increment() -> None:
    support, dlvsq = _small_surface()
    tower = decode_density_tower(support, dlvsq, dt=1.0 / 252.0)
    loadings = jnp.asarray(
        [
            [[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]],
        ]
    )
    normals = jnp.asarray(
        [
            [0.20, -0.10],
            [-0.40, 0.30],
            [0.70, -0.20],
            [0.10, 0.40],
        ]
    )
    state = RiskNeutralState(tower[0], dlvsq[1:], jnp.asarray(1.0))
    result = one_day_risk_neutral_substepped(
        support,
        state,
        loadings,
        normals,
        jnp.asarray(0.0),
    )
    weights = substep_brownian_weights(4)
    expected_index = sample_front_index(tower[0], weights @ normals[:, 0])
    np.testing.assert_equal(
        np.asarray(result.sampled_front_index),
        np.asarray(expected_index),
    )


def test_latent_call_cumulative_transform_matches_direct_payoffs() -> None:
    support, dlvsq = _small_surface()
    tower = np.asarray(decode_density_tower(support, dlvsq, dt=1.0 / 252.0))
    expected = tower @ np.maximum(
        np.asarray(support)[:, None] - np.asarray(support)[None, 1:-1],
        0.0,
    )
    np.testing.assert_allclose(
        _latent_call_prices(tower, np.asarray(support)),
        expected,
        atol=2.0e-16,
    )


def test_audit_uses_conservative_predictable_or_empirical_error() -> None:
    values = np.asarray([0.0, 0.0, 0.0, 4.0])
    summary = _summarize(
        np.asarray([values.sum()]),
        np.asarray([(values**2).sum()]),
        len(values),
        np.asarray([True]),
        standard_error_override=np.asarray([0.1]),
    )
    empirical_standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
    assert summary.worst_standard_error == pytest.approx(empirical_standard_error)


def test_black_mixture_moments_have_correct_zero_vol_limit() -> None:
    support = jnp.asarray([0.8, 1.0, 1.2])
    density = jnp.asarray([[0.2, 0.5, 0.3]])
    strikes = jnp.asarray([0.9, 1.1])
    payoff = jnp.maximum(support[:, None] - strikes[None, :], 0.0)
    expected_price = density @ payoff
    expected_second_moment = density @ payoff**2
    expected_exercise_probability = density @ (payoff > 0.0)

    arguments = (
        support,
        density,
        strikes,
        jnp.asarray(1.0),
        jnp.asarray([0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(black_mixture_call_prices(*arguments)),
        np.asarray(expected_price),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(black_mixture_call_second_moments(*arguments)),
        np.asarray(expected_second_moment),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(black_mixture_call_exercise_probabilities(*arguments)),
        np.asarray(expected_exercise_probability),
        atol=1.0e-14,
    )


def test_normal_interval_probabilities_preserve_upper_tail_mass() -> None:
    boundaries = jnp.asarray([-jnp.inf, 0.0, 8.0, 9.0, jnp.inf])
    probabilities = normal_interval_probabilities(boundaries)
    expected_upper_interval = jax.scipy.special.ndtr(-8.0) - jax.scipy.special.ndtr(
        -9.0
    )
    np.testing.assert_allclose(np.asarray(probabilities.sum()), 1.0)
    np.testing.assert_allclose(
        np.asarray(probabilities[-2]),
        np.asarray(expected_upper_interval),
        rtol=1.0e-14,
    )
    assert float(probabilities[-2]) > 0.0


def test_stabilized_relative_drift_is_exact_locally_and_finite_in_tails() -> None:
    relative_drift = jnp.asarray([-1.0e9, -2.0, 0.0, 1.0, 3.0, 1.0e9])
    dt = 0.25
    increment = stabilized_relative_drift_increment(relative_drift, dt)
    multiplier = jnp.exp(increment)
    np.testing.assert_allclose(
        np.asarray(increment[1:4]),
        np.asarray(relative_drift[1:4] * dt),
    )
    assert bool(jnp.all(jnp.isfinite(multiplier)))
    assert bool(jnp.all(multiplier > 0.0))


def test_paper_implied_volatility_inverts_black_calls() -> None:
    forward = np.asarray([[1.0], [1.0]])
    strikes = np.asarray([[0.98, 1.0, 1.02]])
    maturity = np.asarray([[1.0 / 252.0], [5.0 / 252.0]])
    volatility = np.asarray(0.24)
    total_volatility = volatility * np.sqrt(maturity)
    log_moneyness = np.log(forward / strikes)
    d1 = log_moneyness / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    calls = forward * np.asarray(jax.scipy.special.ndtr(d1)) - strikes * np.asarray(
        jax.scipy.special.ndtr(d2)
    )
    recovered = implied_volatility_from_calls(
        forward,
        strikes,
        maturity,
        calls,
    )
    np.testing.assert_allclose(recovered, volatility, atol=1.0e-10)
