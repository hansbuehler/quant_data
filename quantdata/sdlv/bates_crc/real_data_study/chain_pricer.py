r"""Coherent DLV finite-state chain x {Black, ordinary Bates, extended Bates}
mixture pricers, vectorized for calibration speed. Reuses
comparison_interface.ChainGeneratorConfig/CoherentChainBlackAdapter's own
(already independently verified: exact zero generator residual, exact
Chapman-Kolmogorov composition) transition-matrix machinery for the X
factor -- not reimplemented here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quantdata.sdlv.bates_crc.comparison_interface import (
    ChainGeneratorConfig,
    CoherentChainBlackAdapter,
)
from quantdata.sdlv.bates_crc.heston_riccati import GaussianJumpParams, HestonParams
from quantdata.sdlv.bates_crc.real_data_study.fast_bates_pricer import (
    price_calls_extended_bates,
    price_calls_ordinary_bates,
)


def _black_call(a: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    intrinsic = np.maximum(a - k, 0.0)
    sw = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sw_safe = np.where(sw > 0.0, sw, 1.0)
        d1 = (np.log(a / k) + 0.5 * w) / sw_safe
        d2 = d1 - sw_safe
    price = a * norm.cdf(d1) - k * norm.cdf(d2)
    return np.where(w > 0.0, price, intrinsic)


def chain_weights(chain_config: ChainGeneratorConfig, expiries: np.ndarray) -> dict:
    """{tau: q_i(tau)} for every unique expiry, q_i(tau) = P[X_tau=x_i | X_0=x_i0]."""
    chain = CoherentChainBlackAdapter(chain_config)
    i0 = chain_config.initial_index
    out = {}
    for tau in np.unique(expiries):
        if tau <= 0.0:
            q = np.zeros(len(chain_config.anchors))
            q[i0] = 1.0
        else:
            q = chain.transition_matrix(float(tau))[i0, :]
        out[float(tau)] = q
    return out


def price_calls_chain_black(chain_config: ChainGeneratorConfig, sigma_Y: float, strikes: np.ndarray,
                             expiries: np.ndarray) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    x = np.asarray(chain_config.anchors, dtype=float)
    weights = chain_weights(chain_config, expiries)
    out = np.zeros_like(strikes)
    for tau in np.unique(expiries):
        mask = expiries == tau
        q = weights[float(tau)]
        w = sigma_Y ** 2 * tau
        k = strikes[mask]
        acc = np.zeros_like(k)
        for qi, xi in zip(q, x):
            if qi <= 0.0:
                continue
            acc = acc + qi * _black_call(np.full_like(k, xi), k, np.full_like(k, w))
        out[mask] = acc
    return out


def price_calls_chain_bates(chain_config: ChainGeneratorConfig, heston: HestonParams, jumps: GaussianJumpParams,
                             V_t: float, strikes: np.ndarray, expiries: np.ndarray, *, n_quad: int = 200,
                             u_max: float = 250.0) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    x = np.asarray(chain_config.anchors, dtype=float)
    weights = chain_weights(chain_config, expiries)
    out = np.zeros_like(strikes)
    for tau in np.unique(expiries):
        mask = expiries == tau
        q = weights[float(tau)]
        k = strikes[mask]
        acc = np.zeros_like(k)
        for qi, xi in zip(q, x):
            if qi <= 0.0:
                continue
            scaled_k = k / xi
            tau_arr = np.full_like(scaled_k, tau)
            comp = xi * price_calls_ordinary_bates(heston, jumps, V_t, scaled_k, tau_arr, n_quad=n_quad, u_max=u_max)
            acc = acc + qi * comp
        out[mask] = acc
    return out


def price_calls_chain_extended_bates(chain_config: ChainGeneratorConfig, heston: HestonParams,
                                      current_jumps: GaussianJumpParams, envelope: GaussianJumpParams, V_t: float,
                                      strikes: np.ndarray, expiries: np.ndarray, reset_times: tuple, b_used: tuple,
                                      *, t: float = 0.0, n_quad: int = 200, u_max: float = 250.0,
                                      n_reserve_quad: int = 80) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    x = np.asarray(chain_config.anchors, dtype=float)
    weights = chain_weights(chain_config, expiries)
    out = np.zeros_like(strikes)
    for tau in np.unique(expiries):
        mask = expiries == tau
        q = weights[float(tau)]
        k = strikes[mask]
        acc = np.zeros_like(k)
        for qi, xi in zip(q, x):
            if qi <= 0.0:
                continue
            scaled_k = k / xi
            tau_arr = np.full_like(scaled_k, tau)
            comp = xi * price_calls_extended_bates(heston, current_jumps, envelope, V_t, scaled_k, tau_arr,
                                                     reset_times, b_used, t=t, n_quad=n_quad, u_max=u_max,
                                                     n_reserve_quad=n_reserve_quad)
            acc = acc + qi * comp
        out[mask] = acc
    return out
