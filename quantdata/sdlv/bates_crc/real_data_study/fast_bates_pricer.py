r"""Fast, fully-vectorized Bates Fourier pricers for calibration (many
thousands of price evaluations per fit). Re-implemented independently of
`bates_transform.py` (which loops in Python per-u-node in order to call the
general `reserve_ell` integrand) so that the ordinary-Bates hot path used by
every calibrator here is vectorized over both the Fourier frequency `u` and,
within one expiry, the strike `K`.

Cross-checked against the already-verified `bates_transform.fourier_call_price`
in `tests/test_real_data_study_fast_bates_pricer.py` -- this module is a
speed optimisation, not a new pricing model; its outputs must agree with the
slower, previously-proven-correct implementation to double precision on the
ordinary-Bates (no-reserve) case, and are cross-checked against
`bates_transform`'s reserve-aware path for the extended case too.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from quantdata.sdlv.bates_crc.heston_riccati import (
    GaussianJumpParams,
    HestonParams,
    gaussian_bates_jump_exponent,
    heston_bd,
    riccati_A_closed_form,
    riccati_D_closed_form,
)


def _riccati_D_grid(u_col: np.ndarray, tau_row: np.ndarray, heston: HestonParams) -> np.ndarray:
    """D(u,tau) broadcast over a (n_u, n_tau) grid. Inlined rather than
    calling riccati_D_closed_form directly because that function's
    `tau == 0.0` scalar short-circuit raises on an array-valued tau; safe
    here since every tau in this module's use (open-interval quadrature
    nodes strictly inside a positive-length interval) is strictly > 0."""
    b, d, g = heston_bd(u_col, heston)
    e = np.exp(-d * tau_row)
    return (b - d) / heston.xi ** 2 * (1.0 - e) / (1.0 - g * e)

_ALPHA = 1.5


@lru_cache(maxsize=16)
def _u_grid(n_quad: int, u_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable-by-convention quadrature arrays cached by geometry."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half = 0.5 * u_max
    return half * (nodes + 1.0), half * weights


def price_calls_ordinary_bates(heston: HestonParams, jumps: GaussianJumpParams, V_t: float,
                                strikes: np.ndarray, expiries: np.ndarray, *, n_quad: int = 200,
                                u_max: float = 250.0) -> np.ndarray:
    """Vectorized ordinary (no-reserve) Bates call price, pure units
    (forward=1). `strikes`, `expiries` are 1D arrays of equal length,
    strike[i] priced at expiries[i]."""
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    out = np.empty_like(strikes)
    u_re, weights = _u_grid(n_quad, u_max)
    u = u_re - 1j * (_ALPHA + 1.0)  # shape (n_quad,)

    for tau in np.unique(expiries):
        mask = expiries == tau
        if tau <= 0.0:
            out[mask] = np.maximum(1.0 - strikes[mask], 0.0)
            continue
        D = riccati_D_closed_form(u, tau, heston)
        A = riccati_A_closed_form(u, tau, heston)
        j_term = tau * gaussian_bates_jump_exponent(u, jumps)
        phi = np.exp(V_t * D + A + j_term)  # shape (n_quad,)

        k = np.log(strikes[mask])  # shape (n_k,)
        denom = _ALPHA ** 2 + _ALPHA - u_re ** 2 + 1j * (2.0 * _ALPHA + 1.0) * u_re  # (n_quad,)
        integrand = np.exp(-1j * np.outer(u_re, k)) * (phi / denom)[:, None]  # (n_quad, n_k)
        integral = np.sum(weights[:, None] * integrand, axis=0)
        out[mask] = (np.exp(-_ALPHA * k) / math.pi) * integral.real
    return out


def price_calls_extended_bates(heston: HestonParams, current_jumps: GaussianJumpParams,
                                envelope: GaussianJumpParams, V_t: float, strikes: np.ndarray,
                                expiries: np.ndarray, reset_times: tuple, b_used: tuple, *,
                                t: float = 0.0, n_quad: int = 200, u_max: float = 250.0,
                                n_reserve_quad: int = 80) -> np.ndarray:
    """Vectorized extended-Bates-with-reserve call price. Uses the exact
    cancellation identity proven in AUDIT.md Sec. 8/10 (verified again here
    against bates_transform in tests): the current-jump-parameter term
    cancels exactly against the reserve integral's own -j_{p_n} piece, so

        log Phi_t(u,T) = V_t*D(u,tau) + A(u,tau) + tau*j_bar(u)
                          + kappa * sum_k b_used[k] * int_{max(t,t_k)}^{T} D(u,s-t_k) ds

    which needs only ONE jump-exponent evaluation (the envelope, not the
    current params) plus a vectorized-over-u reserve integral, rather than
    bates_transform's per-u Python loop."""
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    out = np.empty_like(strikes)
    u_re, weights = _u_grid(n_quad, u_max)
    u = u_re - 1j * (_ALPHA + 1.0)

    s_nodes_leg, s_weights_leg = np.polynomial.legendre.leggauss(n_reserve_quad)

    for tau in np.unique(expiries):
        mask = expiries == tau
        T = t + tau
        if tau <= 0.0:
            out[mask] = np.maximum(1.0 - strikes[mask], 0.0)
            continue
        D = riccati_D_closed_form(u, tau, heston)
        A = riccati_A_closed_form(u, tau, heston)
        jbar_term = tau * gaussian_bates_jump_exponent(u, envelope)

        reserve_term = np.zeros_like(u)
        for k, t_k in enumerate(reset_times):
            lo = max(t, t_k)
            if lo >= T:
                continue
            half = 0.5 * (T - lo)
            mid = 0.5 * (T + lo)
            s_nodes = mid + half * s_nodes_leg  # (n_reserve_quad,)
            # D(u, s - t_k) for every (u, s) pair: shape (n_quad, n_reserve_quad)
            D_grid = _riccati_D_grid(u[:, None], (s_nodes - t_k)[None, :], heston)
            reserve_term = reserve_term + heston.kappa * b_used[k] * half * (D_grid @ s_weights_leg)

        phi = np.exp(V_t * D + A + jbar_term + reserve_term)

        k_log = np.log(strikes[mask])
        denom = _ALPHA ** 2 + _ALPHA - u_re ** 2 + 1j * (2.0 * _ALPHA + 1.0) * u_re
        integrand = np.exp(-1j * np.outer(u_re, k_log)) * (phi / denom)[:, None]
        integral = np.sum(weights[:, None] * integrand, axis=0)
        out[mask] = (np.exp(-_ALPHA * k_log) / math.pi) * integral.real
    return out
