from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.dlv import decode_density_tower
from quantdata.gromm2.geometry import (
    Gromm2GeometryConfig,
    expanding_supports,
    transport_tower,
)
from quantdata.gromm2.pricing import price_active_calls
from quantdata.gromm2.simulation import (
    SimulationState,
    localize_loadings_by_density,
    one_day_risk_neutral_step,
)


def _two_expiry_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = Gromm2GeometryConfig(num_inner_strikes=16)
    supports = expanding_supports(np.asarray([0.04 / 252.0, 0.08 / 252.0]), config=config)
    transports = transport_tower(supports)
    expiries = np.asarray([1.0, 2.0]) / 252.0
    intervals = np.diff(expiries, prepend=0.0)
    dlvsq = np.full((2, 16), 0.20**2)
    densities = np.asarray(
        decode_density_tower(
            jnp.asarray(supports),
            jnp.asarray(transports),
            jnp.asarray(dlvsq),
            jnp.asarray(intervals),
        )
    )
    return supports, transports, dlvsq, densities


def test_zero_loading_daily_step_preserves_density_invariants() -> None:
    supports, transports, dlvsq, densities = _two_expiry_state()
    path_count = 8
    factor_count = 2
    state = SimulationState(
        front_density=jnp.broadcast_to(densities[0], (path_count, len(supports[0]))),
        dlvsq=jnp.broadcast_to(dlvsq, (path_count,) + dlvsq.shape),
        background_level=jnp.ones(path_count),
    )
    result = one_day_risk_neutral_step(
        jnp.asarray(supports),
        jnp.asarray(transports),
        jnp.asarray([1.0 / 252.0]),
        state,
        jnp.zeros((1, dlvsq.shape[1], factor_count)),
        jnp.zeros((8, path_count, factor_count)),
        jnp.zeros(path_count),
        remaining_front_days=jnp.asarray(1),
        substeps=8,
    )
    tower = np.asarray(result.density_tower)
    assert np.all(np.isfinite(tower))
    assert np.min(tower) >= 0.0
    assert np.allclose(np.sum(tower, axis=-1), 1.0, atol=2.0e-13)
    assert np.allclose(
        np.einsum("pjn,jn->pj", tower, supports),
        np.asarray(result.latent_spot)[:, None],
        atol=2.0e-12,
    )


def test_background_call_pricing_reduces_to_intrinsic_at_settlement() -> None:
    supports, _, _, densities = _two_expiry_state()
    background = jnp.asarray([0.9, 1.1])
    front = jnp.broadcast_to(jnp.asarray(densities[:1]), (2, 1, supports.shape[1]))
    strikes = jnp.asarray([[0.9, 1.0, 1.1]])
    actual = np.asarray(
        price_active_calls(
            jnp.asarray(supports[:1]),
            front,
            background,
            jnp.asarray([0.0]),
            strikes,
        )
    )
    kernel = np.maximum(
        np.asarray(background)[:, None, None] * supports[0][None, None, :]
        - np.asarray(strikes)[0][None, :, None],
        0.0,
    )
    expected = np.einsum(
        "pkn,pn->pk",
        kernel,
        np.broadcast_to(densities[0], (2, supports.shape[1])),
    )
    assert np.allclose(actual[:, 0], expected, atol=2.0e-13)


def test_density_localization_suppresses_only_vanishing_mass() -> None:
    densities = jnp.asarray([[[0.0, 1.0e-12, 0.25, 0.75, 0.0]]])
    loadings = jnp.ones((1, 1, 3, 2))
    localized = np.asarray(
        localize_loadings_by_density(
            loadings,
            densities,
            relative_floor=1.0e-4,
        )
    )
    assert np.max(localized[..., 0, :]) < 1.0e-7
    assert np.min(localized[..., 1:, :]) > 0.999


def test_zero_floor_disables_attenuation_even_at_zero_mass() -> None:
    densities = jnp.asarray([[[0., 0., .25, .75, 0.]]])
    loadings = jnp.ones((1, 1, 3, 2))
    np.testing.assert_array_equal(localize_loadings_by_density(loadings, densities, relative_floor=0.), loadings)


def test_untamed_step_does_not_compress_the_required_drift(monkeypatch) -> None:
    supports, transports, dlvsq, densities = _two_expiry_state()
    state = SimulationState(jnp.asarray(densities[None, 0]), jnp.asarray(dlvsq[None]), jnp.ones(1))

    def increments(*args, **kwargs):
        shape = (1, 1, dlvsq.shape[-1])
        return jnp.full(shape, 2000.), jnp.zeros(shape), jnp.zeros(shape)

    monkeypatch.setattr('quantdata.gromm2.simulation.fused_risk_neutral_increments', increments)
    arguments = dict(remaining_front_days=jnp.asarray(2), substeps=1,
        factor_scales=jnp.ones((1, 1)), density_loading_floor=0.)
    untamed = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports),
        jnp.asarray([1. / 252.]), state, jnp.zeros((1, dlvsq.shape[-1], 1)),
        jnp.zeros((1, 1, 1)), jnp.zeros(1), drift_taming=False, **arguments)
    tamed = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports),
        jnp.asarray([1. / 252.]), state, jnp.zeros((1, dlvsq.shape[-1], 1)),
        jnp.zeros((1, 1, 1)), jnp.zeros(1), drift_taming=True, **arguments)
    np.testing.assert_allclose(untamed.state.dlvsq[:, 1:], dlvsq[None, 1:] * np.exp(2000. / 252.), rtol=1.e-6)
    assert float(untamed.drift_taming_fraction) == 0.
    assert float(tamed.drift_taming_fraction) == 1.
    assert np.max(np.asarray(tamed.state.dlvsq)) < np.min(np.asarray(untamed.state.dlvsq[:, 1:]))


def test_passive_dlv_variation_matches_known_noise_and_does_not_change_state(monkeypatch) -> None:
    supports, transports, dlvsq, densities = _two_expiry_state()
    count, substeps = 4, 8
    state = SimulationState(jnp.broadcast_to(jnp.asarray(densities[0]), (count, densities.shape[-1])),
        jnp.broadcast_to(jnp.asarray(dlvsq), (count,) + dlvsq.shape), jnp.ones(count))
    normals = np.random.default_rng(15).normal(size=(substeps, count, 2))

    def increments(*args, **kwargs):
        noise = .25 * args[7][:, 0] - .5 * args[7][:, 1]
        shape = (count, 1, dlvsq.shape[-1])
        return jnp.full(shape, .8), jnp.full(shape, .25**2 + .5**2), jnp.broadcast_to(noise[:, None, None], shape)

    monkeypatch.setattr('quantdata.gromm2.simulation.fused_risk_neutral_increments', increments)
    arguments = dict(remaining_front_days=jnp.asarray(2), substeps=substeps, factor_scales=jnp.ones((count, 2)),
                     density_loading_floor=0., drift_taming=False)
    plain = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports),
        jnp.asarray([1. / 252.]), state, jnp.zeros((1, 16, 2)), jnp.asarray(normals), jnp.zeros(count), **arguments)
    measured = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports),
        jnp.asarray([1. / 252.]), state, jnp.zeros((1, 16, 2)), jnp.asarray(normals), jnp.zeros(count),
        collect_dlv_variation=True, **arguments)
    np.testing.assert_array_equal(plain.state.dlvsq, measured.state.dlvsq)
    np.testing.assert_array_equal(plain.density_tower, measured.density_tower)
    expected_noise = (.25 * normals[:, :, 0] - .5 * normals[:, :, 1]).sum(axis=0) / np.sqrt(252. * substeps)
    expected_noise = np.broadcast_to(expected_noise[:, None, None]**2, (count, 1, 16))
    np.testing.assert_allclose(measured.dlv_variation[:, 0], expected_noise, atol=1.e-14)
    np.testing.assert_allclose(measured.dlv_variation[:, 1], (.25**2 + .5**2) / 252., atol=1.e-14)
    assert np.max(measured.dlv_variation[:, 4]) < 2.e-14


def test_authoritative_log_state_survives_underflow_and_recovers(monkeypatch) -> None:
    supports, transports, dlvsq, densities = _two_expiry_state()
    state = SimulationState(jnp.asarray(densities[None, 0]), jnp.asarray(dlvsq[None]), jnp.ones(1))

    def controlled_drift(sign):
        def increments(*args, **kwargs):
            shape = (1, 1, dlvsq.shape[-1])
            return jnp.full(shape, sign * 252000.), jnp.zeros(shape), jnp.zeros(shape)
        return increments

    monkeypatch.setattr('quantdata.gromm2.simulation.fused_risk_neutral_increments', controlled_drift(-1.))
    arguments = dict(remaining_front_days=jnp.asarray(3), substeps=1, factor_scales=jnp.ones((1, 1)),
                     drift_taming=False, collect_dlv_variation=True)
    first = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports), jnp.asarray([1. / 252.]),
        state, jnp.zeros((1, 16, 1)), jnp.zeros((1, 1, 1)), jnp.zeros(1),
        log_dlvsq=jnp.log(state.dlvsq), **arguments)
    assert np.all(np.asarray(first.state.dlvsq[:, 1:]) == 0.)
    assert np.isfinite(np.asarray(first.log_dlvsq)).all()
    assert np.isfinite(np.asarray(first.density_tower)).all()
    assert np.max(np.asarray(first.dlv_variation[:, 4])) < 1.e-12
    monkeypatch.setattr('quantdata.gromm2.simulation.fused_risk_neutral_increments', controlled_drift(1.))
    second = one_day_risk_neutral_step(jnp.asarray(supports), jnp.asarray(transports), jnp.asarray([1. / 252.]),
        first.state, jnp.zeros((1, 16, 1)), jnp.zeros((1, 1, 1)), jnp.zeros(1),
        log_dlvsq=first.log_dlvsq, **arguments)
    np.testing.assert_allclose(second.state.dlvsq, state.dlvsq, rtol=2.e-13, atol=1.e-14)
    assert np.isfinite(np.asarray(second.density_tower)).all()
