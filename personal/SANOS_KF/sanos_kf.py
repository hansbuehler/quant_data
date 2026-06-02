"""
sanos_kf.py

Online SANOS + Extended Kalman Filter for a single expiry.

The SANOS call-price mixture becomes the nonlinear observation function h(x)
of an EKF so the DLV weights and smoothing vol evolve online with each quote.

State  x ∈ ℝ^{n+1}:
    x[:n]  = log(Σᵢ)   — log squared discrete local vols (DLVs)
    x[n]   = log(σ*)   — log shared smoothing vol

Density (Ω = I, p₀ = uniform 1/n):
    q̃ᵢ = (1/n) / (1 + Σᵢ)   then projected onto {q≥0, Σq=1, ΣqKⁱ=1}

Observation (normalized call price):
    h(x, kˡ, T) = Σᵢ qᵢ · BSCallNorm(anchor_kᵢ, kˡ, σ*²·T)

Jacobian: jax.jacrev (primary) + analytic formula (cross-check / paper appendix).

Closed-form Jacobian (Ω = I):
  For i = 0…n-1 (DLV components):
      ∂h/∂xᵢ = −qᵢ · Σᵢ/(1+Σᵢ) · (cᵢ − h)
  For i = n (σ* component):
      ∂h/∂xₙ = σ* · √T · Σⱼ qⱼ · exp(anchor_kⱼ) · φ(d1ⱼ)

  where cᵢ = BSCallNorm(anchor_kᵢ, kˡ, σ*²T), h = qᵀc, φ = standard normal pdf.

Public API
----------
SanosKFConfig          — hyperparameters
sanos_jacobian_analytic() — closed-form Jacobian (numpy)
run_sanos_kf()         — main EKF loop (single expiry)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from cdxcore.bs import bs as _cdx_bs
from jax.scipy.stats import norm as jax_norm
from scipy.stats import norm as np_norm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# IV inversion (vectorized over per-quote T)
# ---------------------------------------------------------------------------


def _invert_iv_vec(c_model_vec: np.ndarray, eval_ks: np.ndarray, Ts: np.ndarray) -> np.ndarray:
    """
    Invert normalized call prices → IVs with per-quote T (array).

    Mirrors _invert_iv_batch from sanos_surface.py but accepts T as an array,
    needed because T decreases intraday even for a fixed expiry.
    """
    eval_ks = np.asarray(eval_ks, dtype=float)
    c_model_vec = np.asarray(c_model_vec, dtype=float)
    Ts = np.asarray(Ts, dtype=float)

    k_ratio = np.exp(eval_ks)
    call_mask = eval_ks >= 0.0
    prices = np.where(call_mask, c_model_vec, c_model_vec + k_ratio - 1.0)
    intr = np.maximum(0.0, np.where(call_mask, 1.0 - k_ratio, k_ratio - 1.0))
    valid = (
        np.isfinite(prices)
        & np.isfinite(Ts)
        & (Ts > 0.0)
        & (prices > intr + 1e-8)
        & np.where(call_mask, prices < 1.0, prices >= 5e-5)
    )
    ivs = np.full(len(eval_ks), float("nan"))
    if valid.any():
        ivs[valid] = _cdx_bs.implied(
            k=k_ratio[valid],
            prices=prices[valid],
            is_call=call_mask[valid],
            sqrtT=np.sqrt(np.maximum(Ts[valid], 1e-8)),
            price_tol=1e-7,
            on_exceed_bounds="quiet",
        )
    return ivs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SanosKFConfig:
    n_anchors: int = 20
    k_min: float = -0.20
    k_max: float = 0.15
    sigma_star_init: float = 0.20  # initial σ* (smoothing vol)
    Sigma_init: float = 0.25  # initial Σᵢ for all anchors (uniform)
    Q_scale_dlv: float = 1e-3  # process noise scale for log-DLV per trading day
    Q_scale_vol: float = 1e-4  # process noise scale for log-σ* per trading day
    P0_scale: float = 1e-4  # initial covariance diagonal scale
    trading_day_s: float = 23400.0  # trading day length in seconds (6.5 h) for Q scaling
    calls_only: bool = False  # if True, skip put quotes (uses PCP otherwise)


# ---------------------------------------------------------------------------
# JAX primitives — differentiable w.r.t. x
# ---------------------------------------------------------------------------


def _bs_call_norm_jax(
    anchor_ks: jnp.ndarray,
    market_k: float,
    v: float,
) -> jnp.ndarray:
    """
    Normalized BS call prices over anchor strikes (JAX, differentiable).

    BSCallNorm(aᵢ, m, v) = exp(aᵢ)·N(d1ᵢ) − exp(m)·N(d2ᵢ)
    d1ᵢ = (aᵢ − m + v/2) / √v,   d2ᵢ = d1ᵢ − √v

    Returns (n,) array.
    """
    v = jnp.maximum(v, 1e-10)
    sqrt_v = jnp.sqrt(v)
    d1 = (anchor_ks - market_k + v / 2.0) / sqrt_v
    d2 = d1 - sqrt_v
    return jnp.exp(anchor_ks) * jax_norm.cdf(d1) - jnp.exp(market_k) * jax_norm.cdf(d2)


def _obs_fn(
    x: jnp.ndarray,
    anchor_ks: jnp.ndarray,
    market_k: float,
    T: float,
) -> jnp.ndarray:
    """
    SANOS EKF observation function — normalized call price h(x, kˡ, T) ∈ ℝ.

    Fully differentiable w.r.t. x; use jax.jacrev(_obs_fn)(x, ...) for H.
    """
    n = anchor_ks.shape[0]
    Sigma = jnp.exp(x[:n])
    sigma_star = jnp.exp(x[n])
    v = jnp.maximum(sigma_star**2 * T, 1e-10)

    # Density: Ω=I → q̃ᵢ = (1/n)/(1+Σᵢ); soft-normalize (martingale projected separately)
    q_raw = (1.0 / n) / (1.0 + Sigma)
    q = q_raw / q_raw.sum()

    c_vec = _bs_call_norm_jax(anchor_ks, market_k, v)
    return jnp.dot(q, c_vec)


# Pre-compile Jacobian (traces once per shape, then cached by JAX)
_jac_fn = jax.jit(jax.jacrev(_obs_fn))
_obs_jit = jax.jit(_obs_fn)


# ---------------------------------------------------------------------------
# Martingale projection (numpy)
# ---------------------------------------------------------------------------


def _project_q(q_raw: np.ndarray, anchor_K: np.ndarray) -> np.ndarray:
    """
    Project q̃ onto {Σq=1, ΣqKⁱ=1, q≥0}.

    Step 1: analytic affine projection onto {Σq=1, ΣqK=1} via 2×2 system.
    Step 2: clip negatives and renormalize (approximates nonneg projection).

    anchor_K = exp(anchor_ks) — anchor strikes normalized by forward (F=1).
    """
    n = len(anchor_K)
    A = np.stack([np.ones(n), anchor_K])  # (2, n)
    b = np.array([1.0, 1.0])
    AAt = A @ A.T  # (2, 2)
    residual = A @ q_raw - b
    try:
        correction = A.T @ np.linalg.solve(AAt, residual)
    except np.linalg.LinAlgError:
        correction = np.zeros(n)
    q_proj = q_raw - correction
    q_proj = np.maximum(q_proj, 0.0)
    s = q_proj.sum()
    if s < 1e-12:
        return np.full(n, 1.0 / n)
    return q_proj / s


# ---------------------------------------------------------------------------
# Analytic Jacobian (numpy) — paper appendix / cross-check against jax.jacrev
# ---------------------------------------------------------------------------


def sanos_jacobian_analytic(
    x: np.ndarray,
    anchor_ks: np.ndarray,
    market_k: float,
    T: float,
) -> np.ndarray:
    """
    Closed-form Jacobian ∂h/∂x ∈ ℝ^{n+1} (Ω = I).

    Derivation (soft-normalized q, no martingale projection):
        q_j = f_j / S,   f_j = (1/n)/(1+Σⱼ),   S = Σ fⱼ
        ∂q_j/∂xᵢ = −qᵢ · Σᵢ/(1+Σᵢ) · (δᵢⱼ − qⱼ)
        ∂h/∂xᵢ   = Σⱼ cⱼ · ∂q_j/∂xᵢ
                  = −qᵢ · Σᵢ/(1+Σᵢ) · (cᵢ − h)

    For the σ* component (x[n] = log σ*):
        ∂BSCallNorm/∂xₙ = exp(aⱼ) · φ(d1ⱼ) · σ* · √T   (chain rule through v=σ*²T)
        ∂h/∂xₙ = σ* · √T · Σⱼ qⱼ · exp(aⱼ) · φ(d1ⱼ)
    """
    n = len(anchor_ks)
    Sigma = np.exp(x[:n])
    sigma_star = np.exp(x[n])
    v = max(float(sigma_star**2 * T), 1e-10)
    sqrt_v = np.sqrt(v)

    q_raw = (1.0 / n) / (1.0 + Sigma)
    q = q_raw / q_raw.sum()

    d1 = (anchor_ks - market_k + v / 2.0) / sqrt_v
    d2 = d1 - sqrt_v
    c_vec = np.exp(anchor_ks) * np_norm.cdf(d1) - np.exp(market_k) * np_norm.cdf(d2)
    h_val = float(c_vec @ q)

    # DLV Jacobian: H_dlv[i] = −qᵢ·Σᵢ/(1+Σᵢ)·(cᵢ − h)
    H_dlv = -q * Sigma / (1.0 + Sigma) * (c_vec - h_val)

    # σ* Jacobian: H_vol = σ*·√T · Σⱼ qⱼ·exp(aⱼ)·φ(d1ⱼ)
    sqrtT = np.sqrt(max(float(T), 1e-8))
    phi_d1 = np_norm.pdf(d1)
    H_vol = float(sigma_star * sqrtT * (q * np.exp(anchor_ks) * phi_d1).sum())

    return np.append(H_dlv, H_vol)


# ---------------------------------------------------------------------------
# EKF step
# ---------------------------------------------------------------------------


def _ekf_step(
    x: np.ndarray,
    P: np.ndarray,
    anchor_ks: np.ndarray,
    market_k: float,
    T: float,
    z: float,
    R: float,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    One predict + update step of the SANOS EKF.

    Uses jax.jacrev for the Jacobian H = ∂h/∂x.
    Returns (x_new, P_new, z_hat).
    """
    # Predict (random walk)
    P_pred = P + Q

    # Nonlinear observation + linearized Jacobian
    x_jax = jnp.array(x)
    ak_jax = jnp.array(anchor_ks)
    mk = float(market_k)
    Tf = float(T)

    z_hat = float(_obs_jit(x_jax, ak_jax, mk, Tf))
    H = np.array(_jac_fn(x_jax, ak_jax, mk, Tf))  # (n+1,)

    if not (np.isfinite(z_hat) and np.isfinite(H).all()):
        return x, P_pred, float("nan")

    # Update
    innov = z - z_hat
    S = float(H @ P_pred @ H) + R
    if abs(S) < 1e-14:
        return x, P_pred, z_hat

    K_gain = (P_pred @ H) / S  # (n+1,)
    x_new = x + K_gain * innov
    # old
    # P_new = P_pred - np.outer(K_gain, H) @ P_pred
    # P_new = (P_new + P_new.T) * 0.5    # enforce symmetry

    # Joseph form for numerical stability: P_new = (I - K H) P_pred (I - K H)' + K R K'
    IKH = np.eye(len(x)) - np.outer(K_gain, H)
    P_new = IKH @ P_pred @ IKH.T + R * np.outer(K_gain, K_gain)
    P_new = (P_new + P_new.T) * 0.5  # still good practice

    z_hat_post = float(_obs_jit(jnp.array(x_new), jnp.array(anchor_ks), market_k, T))
    return x_new, P_new, z_hat_post


# ---------------------------------------------------------------------------
# Main EKF loop
# ---------------------------------------------------------------------------


def run_sanos_kf(
    df_quotes: pl.DataFrame,
    df_trades: pl.DataFrame,
    cfg: SanosKFConfig,
    return_state: bool = False,
    anchor_ks: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Online SANOS EKF on a single expiry.

    Parameters
    ----------
    df_quotes : output of build_iv_observations() for one expiry.
                Required columns: ts_ns, log_moneyness, T, is_call,
                                  bid_price, ask_price, forward.
                Optional: discount_factor (defaults to 1).
    df_trades : same schema; must have quote_ts_ns for out-of-sample eval.
                Also needs observed_iv.
    cfg       : SanosKFConfig
    anchor_ks : optional array of log-moneyness anchor positions (sorted).
                If None, falls back to np.linspace(cfg.k_min, cfg.k_max, cfg.n_anchors).

    Returns
    -------
    df_quotes_result : df_quotes + sanos_kf_price (normalized call), sanos_kf_iv
    df_trades_result : df_trades + sanos_kf_matched (IV), trade_residual
    """
    if anchor_ks is not None:
        anchor_ks = np.asarray(anchor_ks, dtype=float)
        n = len(anchor_ks)
    else:
        n = cfg.n_anchors
        anchor_ks = np.linspace(cfg.k_min, cfg.k_max, n)
    anchor_K = np.exp(anchor_ks)  # anchor strikes (F-normalized)

    # --- State initialization ---
    x = np.concatenate(
        [
            np.full(n, np.log(cfg.Sigma_init)),
            np.array([np.log(cfg.sigma_star_init)]),
        ]
    )
    # Q_rate: process noise variance per trading day (scaled to per-second below)
    Q_rate = np.diag(
        np.concatenate(
            [
                np.full(n, cfg.Q_scale_dlv),
                np.array([cfg.Q_scale_vol]),
            ]
        )
    )
    P = cfg.P0_scale * np.eye(n + 1)

    # --- Quote arrays ---
    df_q = df_quotes.sort("ts_ns")
    has_df_col = "discount_factor" in df_q.columns

    ts_ns_arr = df_q["ts_ns"].cast(pl.Int64).to_numpy()
    ks = df_q["log_moneyness"].to_numpy().astype(float)
    Ts = df_q["T"].to_numpy().astype(float)
    bids = df_q["bid_price"].to_numpy().astype(float)
    asks = df_q["ask_price"].to_numpy().astype(float)
    fwds = df_q["forward"].to_numpy().astype(float)
    dfs = df_q["discount_factor"].to_numpy().astype(float) if has_df_col else np.ones(len(df_q))
    is_calls = df_q["is_call"].to_numpy().astype(bool)

    n_q = len(df_q)
    z_hats = np.full(n_q, float("nan"))
    x_hist = np.full((n_q, n + 1), float("nan"))

    _q_per_ns = 1.0 / (cfg.trading_day_s * 1e9)  # Q scale factor per nanosecond
    _last_ts = ts_ns_arr[0] if n_q > 0 else 0

    for i in range(n_q):
        scale = fwds[i] * dfs[i]
        if scale <= 0 or not np.isfinite(scale):
            _last_ts = ts_ns_arr[i]
            x_hist[i] = x
            continue

        if cfg.calls_only and not is_calls[i]:
            _last_ts = ts_ns_arr[i]
            x_hist[i] = x
            continue

        # Normalize prices: always convert to call price space via PCP for puts
        bid_norm = bids[i] / scale
        ask_norm = asks[i] / scale
        if not is_calls[i]:
            # PCP (normalized): C = P + 1 − exp(k)
            k = ks[i]
            bid_norm = bid_norm + 1.0 - np.exp(k)
            ask_norm = ask_norm + 1.0 - np.exp(k)

        mid_norm = (bid_norm + ask_norm) / 2.0
        spread_half = (ask_norm - bid_norm) / 2.0
        R = max(float(spread_half) ** 2, 1e-10)

        if (
            not np.isfinite(mid_norm)
            or not np.isfinite(ks[i])
            or not np.isfinite(Ts[i])
            or Ts[i] <= 0.0
        ):
            x_hist[i] = x
            # _last_ts = ts_ns_arr[i]
            continue

        # Scale Q by elapsed wall-clock time so process noise is per-trading-day
        dt_ns = max(float(ts_ns_arr[i] - _last_ts), 0.0)
        Q = Q_rate * (dt_ns * _q_per_ns)
        _last_ts = ts_ns_arr[i]

        x, P, z_hat = _ekf_step(x, P, anchor_ks, ks[i], Ts[i], mid_norm, R, Q)
        z_hats[i] = z_hat
        x_hist[i] = x

    sanos_kf_ivs = _invert_iv_vec(z_hats, ks, Ts)

    df_quotes_result = df_q.with_columns(
        [
            pl.Series("sanos_kf_price", z_hats.tolist()),
            pl.Series("sanos_kf_iv", sanos_kf_ivs.tolist()),
        ]
    )

    # --- Trade evaluation ---
    if df_trades.is_empty() or "quote_ts_ns" not in df_trades.columns:
        return df_quotes_result, df_trades.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("sanos_kf_matched"),
                pl.lit(None, dtype=pl.Float64).alias("trade_residual"),
            ]
        )

    # Join state history onto trades via quote_ts_ns (backward asof)
    th_cols = {f"_th{j}": x_hist[:, j].tolist() for j in range(n + 1)}
    df_state = pl.DataFrame({"ts_ns": df_q["ts_ns"], **th_cols})

    df_t = df_trades.sort("quote_ts_ns").join_asof(
        df_state, left_on="quote_ts_ns", right_on="ts_ns", strategy="backward"
    )

    t_ks = df_t["log_moneyness"].to_numpy().astype(float)
    t_Ts = df_t["T"].to_numpy().astype(float)
    th_mat = np.column_stack([df_t[f"_th{j}"].to_numpy() for j in range(n + 1)])

    ak_jax = jnp.array(anchor_ks)
    matched_call_prices = np.full(len(df_t), float("nan"))

    for i in range(len(df_t)):
        th = th_mat[i]
        k, T = t_ks[i], t_Ts[i]
        if np.isfinite(th).all() and np.isfinite(k) and np.isfinite(T) and T > 0.0:
            matched_call_prices[i] = float(_obs_jit(jnp.array(th), ak_jax, k, T))

    sanos_kf_matched = _invert_iv_vec(matched_call_prices, t_ks, t_Ts)

    obs_iv = df_t["observed_iv"].to_numpy().astype(float)
    trade_residual = obs_iv - sanos_kf_matched

    drop_cols = [f"_th{j}" for j in range(n + 1)]
    df_trades_result = df_t.with_columns(
        [
            pl.Series("sanos_kf_matched", sanos_kf_matched.tolist()),
            pl.Series("trade_residual", trade_residual.tolist()),
        ]
    ).drop(drop_cols)

    if not return_state:
        return df_quotes_result, df_trades_result

    state_cols: dict = {"ts_ns": df_q["ts_ns"]}
    state_cols.update({f"x_{j}": x_hist[:, j].tolist() for j in range(n + 1)})
    df_state = pl.DataFrame(state_cols)
    return df_quotes_result, df_trades_result, df_state
