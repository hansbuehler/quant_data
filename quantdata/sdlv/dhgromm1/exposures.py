"""Instantaneous exposures of real GROMM v1 option contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.sdlv.daily.front_spot.estimation import BACKGROUND_ANNUAL_VOL
from quantdata.sdlv.daily.front_spot.jax_model import RiskNeutralState, risk_neutral_dlv_drift


def call_exposures(*, support: jax.Array, state: RiskNeutralState, loadings: jax.Array,
                   strikes: jax.Array, remaining_days: int) -> tuple[jax.Array, jax.Array]:
    """Return call marks and exposures to the PCA factors and Black background.

    Strikes are fixed cash strikes; ``remaining_days`` must identify a live
    future row in the current tower. The last exposure column is the
    independent Black Brownian driver. Inputs may carry leading path axes.
    """
    if not 2 <= remaining_days <= state.future_dlvsq.shape[-2] + 1:
        raise ValueError("option expiry must lie in the live future DLV tower")
    dt = 1.0 / DAYS_PER_YEAR
    densities, density_volatility, _ = risk_neutral_dlv_drift(
        support, state.front_density, state.future_dlvsq, loadings, dt=dt, front_horizon=dt)
    row = remaining_days - 2
    density = densities[..., row, :]
    density_vol = density_volatility[..., row, :, :]
    component = state.background_level[..., None, None] * support[..., :, None]
    strikes = jnp.asarray(strikes)
    deviation = BACKGROUND_ANNUAL_VOL * jnp.sqrt(remaining_days * dt)
    first = jnp.log(component / strikes) / deviation + 0.5 * deviation
    second = first - deviation
    kernel = component * jsp.special.ndtr(first) - strikes * jsp.special.ndtr(second)
    prices = jnp.einsum("...n,...nk->...k", density, kernel)
    pca = jnp.einsum("...nf,...nk->...kf", density_vol, kernel)
    background = BACKGROUND_ANNUAL_VOL * jnp.einsum(
        "...n,...nk->...k", density, component * jsp.special.ndtr(first))
    return prices, jnp.concatenate((pca, background[..., None]), axis=-1)
