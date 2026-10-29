"""Cash-contract books and real-option PCA baskets for the forward hedge test."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

LEVELS = (0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05)


def primary_contracts() -> dict[str, np.ndarray]:
    """Deduplicate day-zero cohorts against the continuously tradable static book."""
    contracts = [(0, expiry, level, False) for expiry in (10, 20) for level in LEVELS]
    contracts += [(day, day + 10, level, True) for day in range(1, 11) for level in LEVELS]
    starts, expiries, levels, locked = map(np.asarray, zip(*contracts, strict=True))
    return dict(starts=starts.astype(np.int32), expiries=expiries.astype(np.int32),
        levels=levels.astype(np.float64), locked=locked.astype(bool), calls=levels >= 1.0)


def basket_contracts(model_dtes: np.ndarray) -> dict[str, np.ndarray]:
    """Use fixed, positive-strike vanilla contracts that survive the hedge horizon."""
    contracts = [(int(expiry), float(level)) for expiry in model_dtes
        if expiry >= 25 and expiry % 5 == 0 for level in np.linspace(0.9, 1.1, 11)]
    expiries, strikes = map(np.asarray, zip(*contracts, strict=True))
    return dict(expiries=expiries.astype(np.int32), strikes=strikes, calls=strikes >= 1.0)


def vanilla_kernel(latent: jax.Array, strikes: jax.Array, deviation: jax.Array,
                   calls: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return cash-option kernels and spot deltas, including exact zero smoothing."""
    safe = jnp.maximum(deviation, jnp.finfo(latent.dtype).eps)
    first = jnp.log(latent / strikes) / safe + 0.5 * safe
    second = first - safe
    call = latent * jsp.special.ndtr(first) - strikes * jsp.special.ndtr(second)
    put = strikes * jsp.special.ndtr(-second) - latent * jsp.special.ndtr(-first)
    sign = jnp.where(calls, 1.0, -1.0)
    intrinsic = jnp.maximum(sign * (latent - strikes), 0.0)
    kernel = jnp.where(deviation > 0.0, jnp.where(calls, call, put), intrinsic)
    delta = jnp.where(calls, jsp.special.ndtr(first), -jsp.special.ndtr(-first))
    delta = jnp.where(deviation > 0.0, delta, sign * (sign * (latent - strikes) > 0.0))
    return jnp.maximum(kernel, 0.0), delta


def contract_marks(*, supports: jax.Array, densities: jax.Array, background: jax.Array,
                   spot: jax.Array, strikes: jax.Array, expiries: np.ndarray,
                   calls: np.ndarray, model_dtes: np.ndarray, offset: int,
                   day: int, volatility: float) -> jax.Array:
    """Price fixed cash strikes on their original absolute model expiry rows."""
    remaining = jnp.asarray(expiries - day)
    indices = jnp.asarray(np.clip(np.searchsorted(model_dtes, expiries) - offset, 0, len(supports) - 1))
    latent = background[:, None, None] * supports[indices][None]
    deviation = volatility * jnp.sqrt(jnp.maximum(remaining, 0) / 252.0)
    kernel, _ = vanilla_kernel(latent, strikes[..., None], deviation[None, :, None], jnp.asarray(calls)[None, :, None])
    marks = jnp.sum(kernel * densities[:, indices], axis=-1)
    intrinsic = jnp.maximum(jnp.where(jnp.asarray(calls), 1.0, -1.0) * (spot[:, None] - strikes), 0.0)
    return jnp.where(remaining < 0, 0.0, jnp.where(remaining == 0, intrinsic, marks))


def factor_baskets(exposures: np.ndarray, *, residual_factors: int,
                   relative_cutoff: float = 1e-7) -> tuple[np.ndarray, dict[str, object]]:
    """Fit inception PCA exposures using options, then fix real contract weights.

    Weights are L1 normalized for meaningful action caps. They are not reset to
    Brownian gains or recomputed using future data. The live span is a testable
    property, not a consequence of calling these baskets PCA portfolios.
    """
    values = np.asarray(exposures, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("option exposures must be a finite contract-by-factor matrix")
    targets = np.eye(values.shape[1])[:, 1:1 + residual_factors]
    weights = np.linalg.pinv(values.T, rcond=relative_cutoff) @ targets
    norm = np.sum(np.abs(weights), axis=0)
    if np.any(norm <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("the actual option book cannot form all requested factor baskets")
    weights /= norm
    singular = np.linalg.svd(values, compute_uv=False)
    realized = values.T @ weights
    return weights, dict(construction="Fixed inception minimum-norm real-option PCA baskets; unit gross contract notional per basket.",
        singular_values=singular.tolist(), option_loading_rank=int(np.linalg.matrix_rank(values, tol=relative_cutoff * singular[0])),
        inverse_gross_normalization=norm.tolist(), realized_exposures=realized.tolist(),
        target_exposures=(targets / norm).tolist(),
        maximum_exposure_residual=float(np.max(np.abs(realized - targets / norm))),
        relative_singular_cutoff=relative_cutoff)
