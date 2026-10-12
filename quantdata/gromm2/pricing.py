"""Normalized option pricing along simulated GROMM2 paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jax.Array


def price_active_calls(
    supports: Array,
    densities: Array,
    background_level: Array,
    remaining_years: Array,
    strikes: Array,
    *,
    background_annual_volatility: float = 0.05,
) -> Array:
    """Price fixed-strike calls for every active expiry and path."""
    time_major_densities = jnp.moveaxis(densities, -2, 0)

    def price_expiry(
        _: None,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[None, Array]:
        support, density, remaining, expiry_strikes = inputs
        latent_spot = background_level[..., None] * support
        intrinsic = jnp.maximum(
            latent_spot[..., None, :] - expiry_strikes[..., :, None], 0.0
        )
        standard_deviation = background_annual_volatility * jnp.sqrt(
            jnp.maximum(remaining, 0.0)
        )

        def black(_: None) -> Array:
            log_ratio = jnp.log(
                latent_spot[..., None, :] / expiry_strikes[..., :, None]
            )
            d1 = log_ratio / standard_deviation + 0.5 * standard_deviation
            d2 = d1 - standard_deviation
            return (
                latent_spot[..., None, :] * jsp.special.ndtr(d1)
                - expiry_strikes[..., :, None] * jsp.special.ndtr(d2)
            )

        kernel = jax.lax.cond(remaining > 0.0, black, lambda _: intrinsic, None)
        return None, jnp.einsum("...kn,...n->...k", kernel, density)

    _, time_major_prices = jax.lax.scan(
        price_expiry,
        None,
        (supports, time_major_densities, remaining_years, strikes),
    )
    return jnp.moveaxis(time_major_prices, 0, -2)
