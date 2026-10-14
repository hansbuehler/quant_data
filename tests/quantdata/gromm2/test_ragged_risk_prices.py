from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from quantdata.gromm2.bridge import front_density_volatility
from quantdata.gromm2.dlv import (
    decode_density_tower,
    density0_from_support,
    risk_neutral_log_drift,
    solve_sanos_step,
)
from quantdata.gromm2.ragged_risk_prices import _ragged_q_drift_function


def test_ragged_q_drift_matches_rectangular_equation() -> None:
    support = np.asarray([0.4, 0.8, 1.0, 1.3, 2.0])
    supports = np.broadcast_to(support, (3, len(support))).copy()
    transports = np.broadcast_to(np.eye(len(support)), (2, len(support), len(support))).copy()
    interval_years = np.asarray([1.0, 1.0]) / 252.0
    dlvsq = np.asarray(
        [
            [[0.04, 0.05, 0.06], [0.05, 0.06, 0.07], [0.06, 0.07, 0.08]],
            [[0.05, 0.04, 0.05], [0.06, 0.05, 0.06], [0.07, 0.06, 0.07]],
        ]
    )
    initial = density0_from_support(jnp.asarray(supports[0]))
    front = solve_sanos_step(
        jnp.asarray(supports[0]),
        jnp.broadcast_to(initial, (len(dlvsq), len(support))),
        jnp.asarray(dlvsq[:, 0] / 252.0),
    )
    loadings = np.linspace(-0.2, 0.3, 2 * 3 * 2).reshape(2, 3, 2)
    ragged = _ragged_q_drift_function(
        supports=tuple(supports),
        transports=tuple(transports),
        interval_years=interval_years,
        annualized_loadings=loadings.reshape(-1, 2),
        future_offsets=np.asarray([0, 3, 6]),
        front_years=1.0 / 252.0,
        density_loading_floor=1.0e-4,
    )
    ragged_drift, effective, weights = ragged(
        front,
        jnp.asarray(dlvsq[:, 1:].reshape(len(dlvsq), -1)),
    )

    tower = decode_density_tower(
        jnp.asarray(supports),
        jnp.asarray(transports),
        jnp.asarray(dlvsq),
        jnp.asarray([1.0, 1.0, 1.0]) / 252.0,
    )
    interior = tower[:, 1:, 1:-1]
    scale = jnp.max(interior, axis=-1, keepdims=True)
    localization = interior / (interior + 1.0e-4 * scale)
    rectangular_loadings = jnp.asarray(loadings)[None] * localization[..., None]
    front_loading = jnp.zeros(front.shape + (2,))
    front_loading = front_loading.at[..., 0].set(
        front_density_volatility(front, remaining_time=jnp.asarray(1.0 / 252.0))
    )
    _, _, rectangular_drift = risk_neutral_log_drift(
        jnp.asarray(supports),
        jnp.asarray(transports),
        front,
        jnp.asarray(dlvsq[:, 1:]),
        jnp.asarray(interval_years),
        rectangular_loadings,
        front_loading,
    )

    assert np.allclose(np.asarray(ragged_drift), np.asarray(rectangular_drift).reshape(2, -1))
    assert np.allclose(np.asarray(effective), np.asarray(rectangular_loadings).reshape(2, -1, 2))
    assert np.allclose(np.sum(np.asarray(weights), axis=1), 1.0)
