"""Check option Brownian exposures against differentiation of the codebook."""

import jax
import jax.numpy as jnp
import numpy as np

from quantdata.sdlv.daily.front_spot.jax_model import (
    RiskNeutralState,
    black_mixture_call_prices,
    decode_density_tower,
    decode_density_tower_from_front,
    front_density_volatility,
)
from quantdata.sdlv.dhgromm1.exposures import call_exposures

jax.config.update("jax_enable_x64", True)


def test_call_exposures_match_codebook_directional_derivatives() -> None:
    dt = 1.0 / 252.0
    support = jnp.asarray([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    dlv = jnp.asarray([[0.04, 0.035, 0.03, 0.04], [0.045, 0.04, 0.035, 0.045],
                       [0.05, 0.045, 0.04, 0.05]])
    tower = decode_density_tower(support, dlv, dt=dt)
    loadings = jnp.asarray([[[0.10, -0.02], [0.08, 0.01], [0.06, 0.03], [0.04, 0.02]],
                            [[0.07, 0.01], [0.06, 0.02], [0.05, 0.02], [0.03, 0.01]]])
    state = RiskNeutralState(tower[0], dlv[1:], jnp.asarray(1.0))
    strikes = jnp.asarray([0.9, 1.0, 1.1])
    prices, exposures = call_exposures(support=support, state=state, loadings=loadings,
                                       strikes=strikes, remaining_days=3)

    def quote(front, future, background):
        density = decode_density_tower_from_front(support, front, future, dt=dt)
        return black_mixture_call_prices(support, density[-1:], strikes[None, :], background,
                                        jnp.asarray([3.0]))[0]

    np.testing.assert_allclose(prices, quote(*state), rtol=1e-12, atol=1e-12)
    for factor in range(3):
        front_shock = front_density_volatility(state.front_density, dt=dt) if factor == 0 else jnp.zeros_like(state.front_density)
        dlv_shock = state.future_dlvsq * loadings[..., factor] if factor < 2 else jnp.zeros_like(state.future_dlvsq)
        black_shock = state.background_level * 0.05 if factor == 2 else jnp.asarray(0.0)
        _, expected = jax.jvp(quote, tuple(state), (front_shock, dlv_shock, black_shock))
        np.testing.assert_allclose(exposures[:, factor], expected, rtol=1e-10, atol=1e-12)
