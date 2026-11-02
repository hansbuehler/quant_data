from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import (
    decode_density_tower,
    decode_density_tower_from_front,
    density0_from_support,
    infer_dlvsq_from_densities,
    risk_neutral_log_drift,
    risk_neutral_relative_drift,
    solve_sanos_step,
)
from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    expanding_supports,
    transport_tower,
)
from quantdata.gromm2.market import solve_sanos_step_numpy

jax.config.update("jax_enable_x64", True)


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = Gromm2GeometryConfig(num_inner_strikes=20)
    expiries = np.asarray([1.0, 3.0, 8.0]) / 252.0
    supports = expanding_supports(0.2**2 * expiries, config=config)
    return expiries, supports, transport_tower(supports)


@pytest.mark.parametrize("linear_solver", ["thomas", "pcr"])
def test_variable_grid_decoder_preserves_mass_and_mean(linear_solver: str) -> None:
    expiries, supports, transports = _geometry()
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.full((3, supports.shape[1] - 2), 0.18**2)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
            linear_solver=linear_solver,
        )
    )
    assert np.min(tower) >= -1.0e-12
    assert np.allclose(np.sum(tower, axis=1), 1.0, atol=2.0e-10, rtol=0.0)
    assert np.allclose(np.sum(tower * supports, axis=1), 1.0, atol=2.0e-10, rtol=0.0)


@pytest.mark.parametrize("linear_solver", ["thomas", "pcr"])
def test_batched_sanos_solve_matches_independent_float64_solves(linear_solver: str) -> None:
    support = np.exp(np.linspace(np.log(0.25), np.log(2.5), 202))
    initial = np.asarray(density0_from_support(jnp.asarray(support)))
    dlvsq_dt = np.stack(
        (
            np.linspace(0.01, 8.0, 200),
            np.linspace(8.0, 0.01, 200),
        )
    ) / 252.0
    previous = np.broadcast_to(initial, (2, len(support)))

    batched = np.asarray(
        solve_sanos_step(
            jnp.asarray(support),
            jnp.asarray(previous),
            jnp.asarray(dlvsq_dt),
            linear_solver=linear_solver,
        )
    )
    independent = np.stack(
        [
            solve_sanos_step_numpy(
                support,
                previous[index],
                dlvsq_dt[index],
            )
            for index in range(len(previous))
        ]
    )

    assert np.allclose(batched, independent, atol=2.0e-12, rtol=2.0e-12)
    assert np.allclose(np.sum(batched, axis=1), 1.0, atol=2.0e-10, rtol=0.0)


@pytest.mark.parametrize("linear_solver", ["thomas", "pcr"])
def test_sanos_solve_preserves_mass_and_mean_for_matrix_rhs(linear_solver: str) -> None:
    support = np.exp(np.linspace(np.log(0.25), np.log(2.5), 202))
    right_hand_side = np.stack(
        (
            np.linspace(-1.0, 1.0, len(support)),
            np.linspace(2.0, -0.5, len(support)),
        ),
        axis=-1,
    )
    dlvsq_dt = np.linspace(0.01, 8.0, 200) / 252.0

    result = np.asarray(
        solve_sanos_step(
            jnp.asarray(support),
            jnp.asarray(right_hand_side),
            jnp.asarray(dlvsq_dt),
            linear_solver=linear_solver,
        )
    )

    assert np.allclose(
        np.sum(result, axis=0),
        np.sum(right_hand_side, axis=0),
        atol=2.0e-10,
        rtol=0.0,
    )
    assert np.allclose(
        np.sum(result * support[:, None], axis=0),
        np.sum(right_hand_side * support[:, None], axis=0),
        atol=2.0e-10,
        rtol=0.0,
    )


def test_dlv_inversion_recovers_constant_local_variance() -> None:
    expiries, supports, transports = _geometry()
    intervals = np.diff(expiries, prepend=0.0)
    expected = np.full((3, supports.shape[1] - 2), 0.16**2)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(expected),
            jnp.asarray(intervals),
        )
    )
    previous = np.asarray(density0_from_support(jnp.asarray(supports[0])))
    for index in range(3):
        if index > 0:
            previous = transports[index - 1] @ tower[index - 1]
        inferred = infer_dlvsq_from_densities(
            supports[index],
            previous,
            tower[index],
            interval_years=float(intervals[index]),
        )
        assert np.allclose(inferred, expected[index], atol=3.0e-9, rtol=2.0e-7)


@pytest.mark.parametrize("linear_solver", ["thomas", "pcr"])
def test_q_drift_cancels_density_drift_to_first_order(linear_solver: str) -> None:
    expiries, supports, transports = _geometry()
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.full((3, supports.shape[1] - 2), 0.16**2)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
            linear_solver=linear_solver,
        )
    )
    factor_count = 2
    loadings = np.zeros((2, supports.shape[1] - 2, factor_count))
    loadings[..., 0] = 0.12
    loadings[..., 1] = np.linspace(-0.04, 0.04, supports.shape[1] - 2)
    front_loading = np.zeros((supports.shape[1], factor_count))
    densities, density_loadings, drift = risk_neutral_log_drift(
        jnp.asarray(supports),
        jnp.asarray(transports),
        jnp.asarray(tower[0]),
        jnp.asarray(dlvsq[1:]),
        jnp.asarray(intervals[1:]),
        jnp.asarray(loadings),
        jnp.asarray(front_loading),
        linear_solver=linear_solver,
    )
    assert densities.shape == (2, supports.shape[1])
    assert density_loadings.shape == (2, supports.shape[1], factor_count)
    assert drift.shape == loadings.shape[:-1]
    assert np.all(np.isfinite(np.asarray(drift)))


@pytest.mark.parametrize('linear_solver', ['thomas', 'pcr'])
def test_ito_generator_and_loadings_cancel_with_stochastic_front(linear_solver: str) -> None:
    supports = jnp.array([[.6, .8, 1., 1.2, 1.4]] * 3)
    transports = jnp.broadcast_to(jnp.eye(5), (2, 5, 5))
    front = jnp.array([.03, .2, .54, .2, .03])
    variance = jnp.array([[.04, .05, .03], [.05, .07, .03]])
    intervals = jnp.array([5., 10.]) / 252.
    loadings = jnp.array([[[.2, -.1], [.3, .1], [.15, .2]], [[.1, -.3], [.2, .2], [.3, -.1]]])
    front_loading = jnp.stack((front_density_volatility(front, remaining_time=jnp.array(1. / 252.)),
        jnp.zeros(5)), axis=-1)
    state = jnp.concatenate((front, jnp.log(variance).reshape(-1)))

    def decode(values):
        return decode_density_tower_from_front(supports, transports, values[:5],
            jnp.exp(values[5:]).reshape(2, 3), intervals, linear_solver=linear_solver)

    _, expected_loading, drift = risk_neutral_log_drift(supports, transports, front,
        variance, intervals, loadings, front_loading, linear_solver=linear_solver)
    state_drift = jnp.concatenate((jnp.zeros(5), drift.reshape(-1)))
    state_loading = jnp.concatenate((front_loading, loadings.reshape(-1, 2)), axis=0)
    jacobian = jax.jacfwd(decode)(state)
    hessian = jax.jacfwd(jax.jacfwd(decode))(state)
    decoded_loading = jnp.einsum('mni,if->mnf', jacobian, state_loading)
    generator = jnp.einsum('mni,i->mn', jacobian, state_drift)
    generator += .5 * jnp.einsum('mnij,if,jf->mn', hessian, state_loading, state_loading)
    np.testing.assert_allclose(generator, 0., atol=1.e-10, rtol=0.)
    np.testing.assert_allclose(decoded_loading[1:], expected_loading, atol=1.e-10, rtol=0.)


@pytest.mark.parametrize('floor', [0., 1.e-4])
def test_stable_relative_drift_matches_q_restriction_at_positive_density(floor: float) -> None:
    density = jnp.array([.1, .2, .3, .3, .1])
    raw = jnp.array([[1., 2.], [3., 4.], [2., -1.]])
    absolute = jnp.arange(10., dtype=jnp.float64).reshape(5, 2)
    interior = density[1:-1]
    effective = raw * (interior / (interior + floor * interior.max()))[:, None]
    reference = -jnp.sum(effective * absolute[1:-1] / interior[:, None], axis=-1)
    actual = risk_neutral_relative_drift(raw, absolute, density, density_loading_floor=floor)
    np.testing.assert_allclose(actual, reference, rtol=1.e-14)


def test_attenuated_drift_has_analytic_zero_mass_limit_without_inf_times_zero() -> None:
    density = jnp.array([0., 0., 1., 0., 0.])
    raw = jnp.array([[1., 2.], [0., 0.], [0., 0.]])
    absolute = jnp.zeros((5, 2)).at[1].set(jnp.array([.001, .001]))
    actual = risk_neutral_relative_drift(raw, absolute, density, density_loading_floor=1.e-4)
    np.testing.assert_allclose(actual, [-30., 0., 0.], rtol=1.e-14)
    untampered = risk_neutral_relative_drift(raw, absolute, density)
    assert not np.isfinite(np.asarray(untampered)[0])
@pytest.mark.parametrize('linear_solver', ['thomas', 'pcr', 'thomas_unsymmetric', 'pcr_unsymmetric'])
def test_zero_and_extremely_small_variances_solve_the_original_m_matrix(linear_solver):
    from quantdata.gromm2.dlv import _sanos_diagonals, solve_sanos_step

    support = jnp.asarray([.5, .8, 1., 1.2, 1.5, 2.])
    variances = jnp.asarray([.02, 0., 1.e-290, .04])
    lower, diagonal, upper = map(np.asarray, _sanos_diagonals(support, variances))
    matrix = np.diag(diagonal) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    rhs = jnp.asarray([[.1, -.1], [.2, .4], [.3, -.2], [.2, .3], [.1, -.1], [.1, -.3]])
    actual = np.asarray(solve_sanos_step(support, rhs, variances, linear_solver=linear_solver))
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, np.asarray(rhs)), rtol=2.e-14, atol=2.e-14)
    np.testing.assert_allclose(actual.sum(axis=0), np.asarray(rhs).sum(axis=0), atol=2.e-14)
    assert np.isfinite(actual).all()


def test_zero_variance_solver_derivative_is_finite():
    from quantdata.gromm2.dlv import solve_sanos_step

    support = jnp.asarray([.5, .8, 1., 1.2, 1.5])
    rhs = jnp.asarray([.1, .2, .4, .2, .1])
    point = jnp.asarray([.01, 0., .02])
    actual = jax.jacfwd(lambda local: solve_sanos_step(support, rhs, local, linear_solver='pcr'))(point)
    assert np.isfinite(np.asarray(actual)).all()


@pytest.mark.parametrize('linear_solver', ['thomas', 'pcr', 'thomas_unsymmetric', 'pcr_unsymmetric'])
def test_original_solver_preserves_tiny_interior_support_independent_of_batch(linear_solver):
    support = jnp.asarray([.5, .8, 1., 1.2, 1.5])
    variances = jnp.asarray([1.e-300, .01, .02])
    rhs = jnp.asarray([.5, 1.e-300, 0., 0., .5])
    single = solve_sanos_step(support, rhs, variances, linear_solver=linear_solver)
    batch = solve_sanos_step(support, jnp.stack((rhs, rhs)),
        jnp.stack((variances, jnp.asarray([0., .01, .02]))), linear_solver=linear_solver)
    assert float(single[1]) > 0.
    assert float(single[1]) == pytest.approx(1.e-300, rel=1.e-12, abs=0.)
    np.testing.assert_array_equal(single, batch[0])
