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
