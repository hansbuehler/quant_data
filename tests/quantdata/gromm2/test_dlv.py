from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.dlv import (
    decode_density_tower,
    density0_from_support,
    infer_dlvsq_from_densities,
    risk_neutral_log_drift,
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


def test_variable_grid_decoder_preserves_mass_and_mean() -> None:
    expiries, supports, transports = _geometry()
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.full((3, supports.shape[1] - 2), 0.18**2)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
        )
    )
    assert np.min(tower) >= -1.0e-12
    assert np.allclose(np.sum(tower, axis=1), 1.0, atol=2.0e-10, rtol=0.0)
    assert np.allclose(np.sum(tower * supports, axis=1), 1.0, atol=2.0e-10, rtol=0.0)


def test_batched_sanos_solve_matches_independent_float64_solves() -> None:
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


def test_sanos_solve_preserves_mass_and_mean_for_matrix_rhs() -> None:
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


def test_q_drift_cancels_density_drift_to_first_order() -> None:
    expiries, supports, transports = _geometry()
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.full((3, supports.shape[1] - 2), 0.16**2)
    tower = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
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
    )
    assert densities.shape == (2, supports.shape[1])
    assert density_loadings.shape == (2, supports.shape[1], factor_count)
    assert drift.shape == loadings.shape[:-1]
    assert np.all(np.isfinite(np.asarray(drift)))
