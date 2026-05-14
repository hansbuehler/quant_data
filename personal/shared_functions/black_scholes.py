from functools import partial

import jax
import jax.numpy as jnp
import polars as pl


# --- Black-Scholes price & vega ---
@jax.jit
def bs_call_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    norm = jax.scipy.stats.norm
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


@jax.jit
def bs_vega(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    return S * jax.scipy.stats.norm.pdf(d1) * jnp.sqrt(T)


@jax.jit
def bs_put_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    norm = jax.scipy.stats.norm
    return K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --- Newton-Raphson IV solvers (vega is identical for calls and puts) ---
@jax.jit
def newton_iv(market_price, S, K, T, r, max_iter=10):
    sigma = jnp.full_like(market_price, 0.3)

    def step(sigma, _):
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        return sigma - (price - market_price) / jnp.where(vega < 1e-8, 1e-8, vega), None

    sigma, _ = jax.lax.scan(step, sigma, None, length=max_iter)
    return sigma


@jax.jit
def newton_iv_put(market_price, S, K, T, r, max_iter=10):
    sigma = jnp.full_like(market_price, 0.3)

    def step(sigma, _):
        price = bs_put_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        return sigma - (price - market_price) / jnp.where(vega < 1e-8, 1e-8, vega), None

    sigma, _ = jax.lax.scan(step, sigma, None, length=max_iter)
    return sigma
