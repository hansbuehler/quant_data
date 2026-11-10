"""Barycentric hat front: spot settles anywhere in [K^0, K^{N+1}], not on a node.

At the start of a front interval (remaining time tau) the front density q is written as the hat projection
q^i = E[phi^i(S_T)] of the maximum-entropy terminal law psi = exp(sum_i lambda_i phi^i) (log psi continuous and
piecewise linear; lambda by Newton on the convex dual, tridiagonal Hessian). Spot is the Bass map
S_T = G(W_T) = F_psi^{-1}(Phi(W_T / sqrt(tau))) of the factor-0 Brownian increment W over the interval, with one G
for the whole interval. Given W_s = w after elapsed time s, the terminal law is psi reweighted by
LR(x) = n_{tau-s}(x - w) / n_tau(x), x = G^{-1}(s), so q_s = hat projection of psi LR and dq_s/dw is the projection
of psi LR (x - w) / (tau - s). The cell quadrature points x are fixed for the interval: a step costs one LR
evaluation per point. On settlement q = phi(G(W_T)): two nodes with barycentric weights.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

Array = jax.Array
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(6)
_GL_NODES, _GL_WEIGHTS = (_GL_NODES + 1.) / 2., _GL_WEIGHTS / 2.
_NEWTON_STEPS = 20
_STEP_SIZES = .5 ** np.arange(10)


class HatInterval(NamedTuple):
    """Fixed data of one front interval (per path)."""

    density: Array        # q at the interval start, (..., N+2)
    lam: Array            # log psi at the nodes, (..., N+2)
    width: Array          # cell widths, (..., N+1)
    cdf: Array            # psi mass below each node, normalised, (..., N+2)
    u: Array              # quadrature positions in each cell, (..., N+1, 6)
    psi_weight: Array     # cell mass times quadrature weight, (..., N+1, 6)
    x: Array              # G^{-1} at the quadrature points, (..., N+1, 6)
    reference: Array      # quadrature hat projection of psi (equals density up to quadrature error)
    tau: Array            # remaining time at the interval start


def _series(b: Array, k: int) -> Array:
    fact = np.cumprod(np.r_[1., np.arange(1., 7.)])
    return sum(b**j / (fact[j] * (j + k + 1.)) for j in range(7))


def _cell_integrals(lam: Array, width: Array) -> tuple[Array, Array, Array]:
    """dK int_0^1 u^k psi du per cell, psi = exp(lam_c (1-u) + lam_{c+1} u), written in e^{lam} (no overflow)."""
    left, right = jnp.exp(lam[..., :-1]), jnp.exp(lam[..., 1:])
    b = lam[..., 1:] - lam[..., :-1]
    small = jnp.abs(b) < .05
    t = jnp.where(small, 1., b)
    exact = ((right - left) / t, (right * (t - 1.) + left) / t**2, (right * (t * t - 2. * t + 2.) - 2. * left) / t**3)
    return tuple(width * jnp.where(small, left * _series(b, k), x) for k, x in enumerate(exact))


def _project(cell_left: Array, cell_right: Array) -> Array:
    zero = jnp.zeros_like(cell_left[..., :1])
    return jnp.concatenate((cell_left, zero), axis=-1) + jnp.concatenate((zero, cell_right), axis=-1)


def max_entropy(density: Array, width: Array) -> Array:
    """lambda with int phi^i exp(sum_j lambda_j phi^j) = q^i: Newton on the dual with backtracking."""
    q = density / jnp.sum(density, axis=-1, keepdims=True)
    zero = jnp.zeros_like(width[..., :1])
    hat_width = .5 * (jnp.concatenate((zero, width), -1) + jnp.concatenate((width, zero), -1))
    lam = jnp.clip(jnp.log(jnp.maximum(q, 1e-300) / hat_width), -700., 300.)
    steps = jnp.asarray(_STEP_SIZES, q.dtype)

    def newton(lam, _):
        i0, i1, i2 = _cell_integrals(lam, width)
        grad = _project(i0 - i1, i1) - q
        diag = _project(i0 - 2. * i1 + i2, i2) + 1e-300
        off = i1 - i2
        step = jax.lax.linalg.tridiagonal_solve(jnp.concatenate((zero, off), -1), diag,
                                                jnp.concatenate((off, zero), -1), grad[..., None])[..., 0]
        step = jnp.where(jnp.isfinite(step), step, 0.)
        trial = jnp.clip(lam[..., None, :] - steps[:, None] * step[..., None, :], -700., 300.)   # q = 0 nodes: lam -> -inf
        dual = jnp.sum(_cell_integrals(trial, width[..., None, :])[0], -1) - jnp.sum(trial * q[..., None, :], -1)
        dual = jnp.where(jnp.isfinite(dual), dual, jnp.inf)
        return jnp.take_along_axis(trial, jnp.argmin(dual, -1)[..., None, None], -2)[..., 0, :], None

    lam, _ = jax.lax.scan(newton, lam, None, length=_NEWTON_STEPS)
    return lam


def start_interval(density: Array, supports: Array, *, remaining_time: Array) -> HatInterval:
    """Solve psi once for the interval and fix the quadrature points x = G^{-1}(s)."""
    width = jnp.broadcast_to(jnp.diff(supports, axis=-1), density.shape[:-1] + (density.shape[-1] - 1,))
    lam = max_entropy(density, width)
    i0, i1, _ = _cell_integrals(lam, width)
    total = jnp.sum(i0, axis=-1, keepdims=True)
    cdf = jnp.concatenate((jnp.zeros_like(i0[..., :1]), jnp.cumsum(i0, axis=-1)), axis=-1) / total
    mass = i0 / total                                                                  # psi mass per cell
    p = jnp.asarray(_GL_NODES, density.dtype)                                          # quadrature in the cell-CDF coordinate
    b = (lam[..., 1:] - lam[..., :-1])[..., None]
    small = jnp.abs(b) < 1e-6
    t = jnp.where(small, 1., b)
    u = jnp.where(small, p, jnp.where(b > 0, 1. + jnp.log(jnp.exp(-t) + p * -jnp.expm1(-t)) / t,
                                      jnp.log1p(p * jnp.expm1(t)) / t))
    u = jnp.clip(u, 0., 1.)
    prob = jnp.clip(cdf[..., :-1, None] + mass[..., None] * p, 1e-300, 1. - 1e-16)
    psi_weight = mass[..., None] * jnp.asarray(_GL_WEIGHTS, density.dtype)
    reference = _project(jnp.sum(psi_weight * (1. - u), -1), jnp.sum(psi_weight * u, -1))
    q = density / jnp.sum(density, axis=-1, keepdims=True)
    return HatInterval(q, lam, width, cdf, u, psi_weight,
                       jnp.sqrt(remaining_time) * jsp.special.ndtri(prob), reference, remaining_time)


def hat_density(interval: HatInterval, w: Array, *, elapsed: Array) -> tuple[Array, Array]:
    """Front density q_s and its Brownian loading dq_s/dw given W_s = w (before settlement)."""
    rest = interval.tau - elapsed
    x, w = interval.x, w[..., None, None]
    log_lr = .5 * jnp.log(interval.tau / rest) - .5 * (x - w)**2 / rest + .5 * x**2 / interval.tau
    weight = jnp.exp(jnp.log(interval.psi_weight) + log_lr)                           # psi LR, no overflow in LR alone
    density = _project(jnp.sum(weight * (1. - interval.u), -1), jnp.sum(weight * interval.u, -1)) \
        + interval.density - interval.reference                                       # exact at w = 0, s = 0
    dweight = weight * (x - w) / rest
    loading = _project(jnp.sum(dweight * (1. - interval.u), -1), jnp.sum(dweight * interval.u, -1))
    return jnp.maximum(density, 0.), loading


def settle_density(interval: HatInterval, w: Array) -> Array:
    """phi(G(w)): barycentric weights on the two nodes around S_T = G(w)."""
    prob = jsp.special.ndtr(w / jnp.sqrt(interval.tau))[..., None]
    cell = jnp.clip(jnp.sum(prob > interval.cdf[..., 1:-1], axis=-1, keepdims=True), 0, interval.width.shape[-1] - 1)
    take = lambda v: jnp.take_along_axis(v, cell, axis=-1)
    lam = interval.lam
    a, b = take(lam[..., :-1]), take(lam[..., 1:] - lam[..., :-1])
    i0, _, _ = _cell_integrals(lam, interval.width)
    total = jnp.sum(i0, axis=-1, keepdims=True)
    t = (prob - take(interval.cdf)) * total / (take(interval.width) * jnp.exp(a))       # int_0^u e^{b v} dv = t
    small = jnp.abs(b) < 1e-8
    u = jnp.clip(jnp.where(small, t, jnp.log1p(jnp.where(small, 0., b) * t) / jnp.where(small, 1., b)), 0., 1.)
    nodes = jnp.arange(lam.shape[-1])
    return jnp.where(nodes == cell, 1. - u, 0.) + jnp.where(nodes == cell + 1, u, 0.)
