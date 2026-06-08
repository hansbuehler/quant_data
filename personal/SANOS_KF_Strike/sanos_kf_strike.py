"""
sanos_kf_strike.py

Online SANOS + Extended Kalman Filter for a single expiry, in absolute strike space.
Smoothing parameter σ* = 0: observation is the discrete call payoff, not a BS mixture.

State  x ∈ ℝ^{N-2}:
    x[i] = (Σ^{i+1})²  — squared DLV at interior anchor K[i+1], i = 0, …, N-3

Initial density q̄₀ ∈ ℝ^N:
    Second finite difference of (F₀ − K)⁺ evaluated on the anchor grid.
    Approximates a Dirac delta at F₀; sum = 1, mean = F₀.
    Computed once from the first valid quote's forward and frozen for the run.

    ⚠ Model mean = F₀ (fixed). If the underlying forward drifts materially during
    the session, residuals will show a bias. For intraday single-expiry runs this
    is acceptable; re-initialize if running over multiple hours.

Density at time T:
    q = Q⁻¹(x, T) \\ q̄₀   via SANOS tri-diagonal transition operator (Remark 3.5)

Observation (undiscounted absolute call price):
    h(x, K_market, T) = Σᵢ qᵢ · (Kᵢ − K_market)⁺

Jacobian: jax.jacrev (argnums=0 — differentiates w.r.t. x).

Public API
----------
SanosKFStrikeConfig     — hyperparameters
run_sanos_kf_strike()   — main EKF loop (single expiry)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from cdxcore.bs import bs as _cdx_bs

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SanosKFStrikeConfig:
    Sigma_sq_init: float = 0.04  # initial (Σᵢ)² for all interior anchors
    Q_scale_dlv_sq: float = 1e-6  # process noise for (Σᵢ)² per trading day
    P0_scale: float = 1e-8  # initial covariance diagonal scale
    trading_day_s: float = 23400.0  # trading day length in seconds (6.5 h)
    calls_only: bool = False  # if True, skip put quotes
    strike_round: float = 1.0  # rounding precision (in $ strike units) for deduplication
    dlv_max: float = (
        2.0  # DLV upper bound; if any DLV√(x[i]) hits this after update, row is ignored
    )
    ignore_at_dlv_max: bool = (
        True  # if True, skip KF update when any post-update DLV reaches dlv_max
    )


# ---------------------------------------------------------------------------
# Initial density (numpy — computed once)
# ---------------------------------------------------------------------------


def _q_bar_0_from_strikes(anchor_Ks: np.ndarray, F_0: float) -> np.ndarray:
    """
    Initial risk-neutral density at T→0: point mass at F₀ spread onto the grid.

    Uses Theorem 3.6 / eq. (22): second finite difference of (F₀ − K)⁺.
    Sentinels: K⁰ = 0, K^{N+1} = 2 · K_max.

    Returns q̄₀ ∈ ℝ^N with q̄₀ ≥ 0, sum ≈ 1, (q̄₀ · anchor_Ks).sum() ≈ F₀.
    """
    N = len(anchor_Ks)
    K_ext = np.concatenate([[0.0], anchor_Ks, [2.0 * anchor_Ks[-1]]])

    def put_diff(a, b):
        return np.maximum(F_0 - b, 0.0) - np.maximum(F_0 - a, 0.0)

    q = np.zeros(N)
    for ell in range(N):
        term1 = put_diff(K_ext[ell + 1], K_ext[ell + 2]) / (K_ext[ell + 2] - K_ext[ell + 1])
        term2 = put_diff(K_ext[ell], K_ext[ell + 1]) / (K_ext[ell + 1] - K_ext[ell])
        q[ell] = term1 - term2
    return q


# ---------------------------------------------------------------------------
# JAX primitives — differentiable w.r.t. x (argnums=0)
# ---------------------------------------------------------------------------


def _thomas_solve(
    lower: jnp.ndarray,
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Thomas algorithm for a tridiagonal system A x = rhs, O(N) via two lax.scan passes.

    lower[i] = A[i, i-1]  (lower[0] is unused — no row above row 0)
    upper[i] = A[i, i+1]  (upper[-1] is unused — no row below last)

    JAX-differentiable: lax.scan has defined VJP rules, so jacrev propagates
    through both the forward sweep and back-substitution automatically.
    Assumes diagonal dominance (no pivoting) — guaranteed here because
    Q_inv = I + T·(pos-semi-def tridiagonal) is positive definite.
    """

    def _fwd(carry, inp):
        dm, rm = carry
        a, b, c_prev, d = inp  # lower[j], diag[j], upper[j-1], rhs[j]
        w = a / dm
        return (b - w * c_prev, d - w * rm), (b - w * c_prev, d - w * rm)

    (_, _), (dm_tail, rm_tail) = jax.lax.scan(
        _fwd,
        (diag[0], rhs[0]),
        (lower[1:], diag[1:], upper[:-1], rhs[1:]),
    )
    dm = jnp.concatenate([diag[:1], dm_tail])
    rm = jnp.concatenate([rhs[:1], rm_tail])

    x_last = rm[-1] / dm[-1]

    def _bwd(x_next, inp):
        rm_j, dm_j, c_j = inp
        x_j = (rm_j - c_j * x_next) / dm_j
        return x_j, x_j

    _, x_rest = jax.lax.scan(_bwd, x_last, (rm[:-1], dm[:-1], upper[:-1]), reverse=True)
    return jnp.append(x_rest, x_last)


def _density_from_state(
    x: jnp.ndarray,
    anchor_Ks: jnp.ndarray,
    T: float,
    q_bar_0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve Q_inv(x, T) @ q = q_bar_0 via Thomas algorithm → full density q ∈ ℝ^N.

    Q_inv is tridiagonal: boundary rows are identity, interior rows carry the
    SANOS ω± coefficients (Remark 3.5). Thomas algorithm is O(N); differentiable
    via lax.scan VJP rules.
    """
    N = anchor_Ks.shape[0]
    Ki = anchor_Ks[1 : N - 1]
    Kp = anchor_Ks[2:N]
    Km = anchor_Ks[0 : N - 2]
    denom = Kp - Km
    op = Ki**2 / (denom * (Kp - Ki))
    om = Ki**2 / (denom * (Ki - Km))
    c = x * T

    main = 1.0 + (op + om) * c
    lower = jnp.concatenate([jnp.zeros(1), -om[1:] * c[1:]])
    upper = jnp.concatenate([-op[:-1] * c[:-1], jnp.zeros(1)])
    rhs = (
        q_bar_0[1 : N - 1]
        .at[0]
        .add(om[0] * c[0] * q_bar_0[0])
        .at[-1]
        .add(op[-1] * c[-1] * q_bar_0[N - 1])
    )
    q_int = _thomas_solve(lower, main, upper, rhs)
    return jnp.concatenate([q_bar_0[:1], q_int, q_bar_0[N - 1 : N]])


def _obs_fn_strike(
    x: jnp.ndarray,
    anchor_Ks: jnp.ndarray,
    K_market: float,
    T: float,
    q_bar_0: jnp.ndarray,
) -> jnp.ndarray:
    """
    SANOS EKF observation function — undiscounted absolute call price h(x) ∈ ℝ.
    Differentiable w.r.t. x (use jax.jacrev(..., argnums=0)).
    """
    q = _density_from_state(x, anchor_Ks, T, q_bar_0)
    payoff = jnp.maximum(anchor_Ks - K_market, 0.0)
    return jnp.dot(q, payoff)


_obs_jit_strike = jax.jit(_obs_fn_strike)
_jac_fn_strike = jax.jit(jax.jacrev(_obs_fn_strike, argnums=0))

_jac_raw = jax.jacrev(_obs_fn_strike, argnums=0)


def _ekf_scan_step(carry: tuple, inputs: tuple) -> tuple:
    """
    Single EKF predict+update step for jax.lax.scan.

    Carry: (x, P, ak_jax, q_bar_0, q_dlv_scalar, dlv_max_sq, ignore_at_dlv_max_flag)
      ak_jax, q_bar_0, dlv_max_sq, ignore_at_dlv_max_flag are carried as constants.
    Inputs: (K_i, T_i, z_i, R_i, q_scale_i, valid_i)
    Outputs: updated carry + (x_new, z_out) stacked over steps.
    """
    x, P, ak_jax, q_bar_0, q_dlv_scalar, dlv_max_sq, ignore_at_dlv_max_flag = carry
    K_i, T_i, z_i, R_i, q_scale_i, valid_i = inputs

    n_state = x.shape[0]
    eye_n = jnp.eye(n_state, dtype=jnp.float64)

    P_pred = P + eye_n * (q_dlv_scalar * q_scale_i)

    z_hat = _obs_fn_strike(x, ak_jax, K_i, T_i, q_bar_0)
    H = _jac_raw(x, ak_jax, K_i, T_i, q_bar_0)

    innov = z_i - z_hat
    S = H @ P_pred @ H + R_i
    K_gain = (P_pred @ H) / S

    x_upd = jnp.maximum(x + K_gain * innov, 1e-8)

    IKH = eye_n - jnp.outer(K_gain, H)
    P_upd = IKH @ P_pred @ IKH.T + R_i * jnp.outer(K_gain, K_gain)
    P_upd = (P_upd + P_upd.T) * 0.5

    z_post = _obs_fn_strike(x_upd, ak_jax, K_i, T_i, q_bar_0)

    all_finite = jnp.isfinite(z_hat) & jnp.all(jnp.isfinite(H)) & (jnp.abs(S) > 1e-14)

    # DLV_MAX filter: if any post-update DLV² ≥ dlv_max², treat observation as bad.
    # State is frozen (x unchanged) but process noise still accumulates (P = P_pred),
    # so the next valid observation sees appropriately inflated uncertainty.
    at_dlv_max = ignore_at_dlv_max_flag & jnp.any(x_upd >= dlv_max_sq)

    do_update = valid_i & all_finite & ~at_dlv_max

    x_new = jnp.where(do_update, x_upd, x)
    # Rows skipped due to DLV_MAX accumulate process noise; other invalid rows freeze P.
    P_new = jnp.where(do_update, P_upd, jnp.where(at_dlv_max, P_pred, P))
    z_out = jnp.where(do_update, z_post, jnp.nan)

    return (x_new, P_new, ak_jax, q_bar_0, q_dlv_scalar, dlv_max_sq, ignore_at_dlv_max_flag), (
        x_new,
        z_out,
    )


# Compiled once per unique (n_state, N, n_q) shape combination.
_run_ekf_scan_jit = jax.jit(lambda init, xs: jax.lax.scan(_ekf_scan_step, init, xs))


# ---------------------------------------------------------------------------
# IV inversion (vectorized over quotes)
# ---------------------------------------------------------------------------


def _invert_iv_vec_strike(
    h_vec: np.ndarray,
    K_markets: np.ndarray,
    F_ts: np.ndarray,
    Ts: np.ndarray,
) -> np.ndarray:
    """
    Invert undiscounted absolute call prices → implied vols.

    Normalizes: c_norm = h / F_t,  k_ratio = K_market / F_t.
    Always treats model output as call prices (σ* = 0, no put-call ambiguity).
    """
    h_vec = np.asarray(h_vec, dtype=float)
    K_markets = np.asarray(K_markets, dtype=float)
    F_ts = np.asarray(F_ts, dtype=float)
    Ts = np.asarray(Ts, dtype=float)

    c_norm = h_vec / np.where(F_ts > 0, F_ts, np.nan)
    k_ratio = K_markets / np.where(F_ts > 0, F_ts, np.nan)
    intr = np.maximum(1.0 - k_ratio, 0.0)

    valid = (
        np.isfinite(c_norm)
        & np.isfinite(Ts)
        & (Ts > 0.0)
        & np.isfinite(k_ratio)
        & (k_ratio > 0.0)
        & (c_norm > intr + 1e-8)
        & (c_norm < 1.0)
    )
    ivs = np.full(len(h_vec), float("nan"))
    if valid.any():
        ivs[valid] = _cdx_bs.implied(
            k=k_ratio[valid],
            prices=c_norm[valid],
            is_call=np.ones(int(valid.sum()), dtype=bool),
            sqrtT=np.sqrt(np.maximum(Ts[valid], 1e-8)),
            price_tol=1e-7,
            on_exceed_bounds="quiet",
        )
    return ivs


# ---------------------------------------------------------------------------
# EKF step
# ---------------------------------------------------------------------------


def _ekf_step_strike(
    x: np.ndarray,
    P: np.ndarray,
    anchor_Ks: np.ndarray,
    K_market: float,
    T: float,
    z: float,
    R: float,
    Q: np.ndarray,
    q_bar_0: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    One predict + update step of the SANOS-Strike EKF.

    Returns (x_new, P_new, z_hat_post).
    Post-update: x is clipped to [1e-8, ∞) to keep (Σᵢ)² > 0.
    """
    P_pred = P + Q

    x_jax = jnp.array(x)
    ak_jax = jnp.array(anchor_Ks)
    Kf = float(K_market)
    Tf = float(T)

    z_hat = float(_obs_jit_strike(x_jax, ak_jax, Kf, Tf, q_bar_0))
    H = np.array(_jac_fn_strike(x_jax, ak_jax, Kf, Tf, q_bar_0))  # (N-2,)

    if not (np.isfinite(z_hat) and np.isfinite(H).all()):
        return x, P_pred, float("nan")

    innov = z - z_hat
    S = float(H @ P_pred @ H) + R
    if abs(S) < 1e-14:
        return x, P_pred, z_hat

    K_gain = (P_pred @ H) / S
    x_new = x + K_gain * innov
    x_new = np.maximum(x_new, 1e-8)  # enforce (Σᵢ)² > 0

    # Joseph form for numerical stability
    IKH = np.eye(len(x)) - np.outer(K_gain, H)
    P_new = IKH @ P_pred @ IKH.T + R * np.outer(K_gain, K_gain)
    P_new = (P_new + P_new.T) * 0.5

    z_hat_post = float(_obs_jit_strike(jnp.array(x_new), ak_jax, Kf, Tf, q_bar_0))
    return x_new, P_new, z_hat_post


# ---------------------------------------------------------------------------
# Main EKF loop
# ---------------------------------------------------------------------------


def run_sanos_kf_strike(
    df_quotes: pl.DataFrame,
    df_trades: pl.DataFrame,
    cfg: SanosKFStrikeConfig,
    return_state: bool = False,
    anchor_Ks: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Online SANOS-Strike EKF on a single expiry (absolute strike space, σ* = 0).

    Parameters
    ----------
    df_quotes : output of build_iv_observations() for one expiry.
                Required columns: ts_ns, log_moneyness, T, is_call,
                                  bid_price, ask_price, forward.
                Optional: discount_factor (defaults to 1).
    df_trades : same schema; must have quote_ts_ns and observed_iv.
    cfg       : SanosKFStrikeConfig.
    anchor_Ks : optional sorted array of absolute strike anchors.
                If None, derived from the unique observed strikes in df_quotes
                (rounded to cfg.strike_round to remove floating-point noise).

    Returns
    -------
    df_quotes_result : df_quotes + sanos_kf_price (undiscounted abs call), sanos_kf_iv
    df_trades_result : df_trades + sanos_kf_matched (IV), trade_residual
    [df_state]       : ts_ns, x_0, …, x_{N-3}  (only if return_state=True)

    Notes
    -----
    q̄₀ is frozen from the first valid quote's forward F₀. The model martingale
    mean equals F₀ for the entire run. Forward drift causes residual bias;
    acceptable for intraday single-expiry runs.
    """
    # --- Compute quotes arrays early so we can derive anchor grid from data ---
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

    K_abs_arr = fwds * np.exp(ks)

    if anchor_Ks is not None:
        anchor_Ks = np.asarray(anchor_Ks, dtype=float)
    else:
        valid = np.isfinite(K_abs_arr) & np.isfinite(fwds) & (fwds > 0)
        if not valid.any():
            raise ValueError("No valid quotes to derive anchor grid from data.")
        K_rounded = np.round(K_abs_arr[valid] / cfg.strike_round) * cfg.strike_round
        anchor_Ks = np.sort(np.unique(K_rounded))

    N = len(anchor_Ks)
    n_state = N - 2  # interior anchors only

    # --- State initialization ---
    x = np.full(n_state, cfg.Sigma_sq_init)
    P = cfg.P0_scale * np.eye(n_state)

    # q_bar_0 computed on first valid quote; None until then
    q_bar_0: jnp.ndarray | None = None

    n_q = len(df_q)
    z_hats = np.full(n_q, float("nan"))
    x_hist = np.full((n_q, n_state), float("nan"))

    _q_per_ns = 1.0 / (cfg.trading_day_s * 1e9)

    ak_jax = jnp.array(anchor_Ks)

    # ------------------------------------------------------------------
    # Precompute all scan inputs (numpy, no Python per-quote overhead)
    # ------------------------------------------------------------------

    # Validity: forward/DF, strike in grid, optional calls-only filter
    safe_dfs = np.where(dfs > 0, dfs, 1.0)
    valid_fwd = np.isfinite(fwds) & (fwds > 0) & np.isfinite(dfs) & (dfs > 0)
    valid_strike = (
        np.isfinite(K_abs_arr) & (K_abs_arr >= anchor_Ks[0]) & (K_abs_arr <= anchor_Ks[-1])
    )
    valid_call_filter = ~cfg.calls_only | is_calls

    # Undiscounted call prices (vectorized; put→call via PCP)
    bid_undisc_arr = bids / safe_dfs
    ask_undisc_arr = asks / safe_dfs
    pcp_adj = np.where(~is_calls, fwds - K_abs_arr, 0.0)
    bid_undisc_arr = bid_undisc_arr + pcp_adj
    ask_undisc_arr = ask_undisc_arr + pcp_adj
    mid_undisc_arr = (bid_undisc_arr + ask_undisc_arr) / 2.0
    spread_half_arr = (ask_undisc_arr - bid_undisc_arr) / 2.0
    R_arr = np.maximum(spread_half_arr**2, 1e-10)

    valid_obs = np.isfinite(mid_undisc_arr) & np.isfinite(Ts) & (Ts > 0.0)
    valid_mask = valid_fwd & valid_strike & valid_call_filter & valid_obs

    # q_bar_0: freeze from first valid forward (before the scan)
    first_valid_idxs = np.where(valid_mask)[0]
    if len(first_valid_idxs) > 0:
        q_bar_0 = jnp.array(_q_bar_0_from_strikes(anchor_Ks, fwds[first_valid_idxs[0]]))

    # dt_ns[i] = ts_ns[i] − ts_ns[i−1] (= 0 for i=0)
    # Original loop updates _last_ts on every quote (skipped or not), so
    # dt is always the consecutive difference — precomputable as np.diff.
    dt_ns_arr = np.empty(n_q, dtype=np.float64)
    if n_q > 0:
        dt_ns_arr[0] = 0.0
    if n_q > 1:
        dt_ns_arr[1:] = np.maximum(np.diff(ts_ns_arr.astype(np.float64)), 0.0)
    Q_diag_scale = dt_ns_arr * _q_per_ns  # per-step multiplier for Q diagonal

    # ------------------------------------------------------------------
    # lax.scan EKF loop — single compiled XLA kernel
    # ------------------------------------------------------------------

    if q_bar_0 is not None:
        scan_inputs = (
            jnp.array(K_abs_arr),
            jnp.array(Ts),
            jnp.array(mid_undisc_arr),
            jnp.array(R_arr),
            jnp.array(Q_diag_scale),
            jnp.array(valid_mask),
        )

        init = (
            jnp.array(x, dtype=jnp.float64),
            jnp.array(P, dtype=jnp.float64),
            ak_jax,
            q_bar_0,
            jnp.float64(cfg.Q_scale_dlv_sq),
            jnp.float64(cfg.dlv_max**2),
            jnp.bool_(cfg.ignore_at_dlv_max),
        )

        # _run_ekf_scan_jit is a module-level JIT'd function — compiled once per
        # unique shape combination and reused across all calls.
        (_x_final, _P_final, _, _, _, _, _), (x_hist_jax, z_hats_jax) = _run_ekf_scan_jit(
            init, scan_inputs
        )

        z_hats = np.array(z_hats_jax)
        x_hist = np.array(x_hist_jax)

    sanos_kf_ivs = _invert_iv_vec_strike(z_hats, K_abs_arr, fwds, Ts)

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

    th_cols = {f"_th{j}": x_hist[:, j].tolist() for j in range(n_state)}
    df_state_join = pl.DataFrame({"ts_ns": df_q["ts_ns"], **th_cols})

    df_t = df_trades.sort("quote_ts_ns").join_asof(
        df_state_join, left_on="quote_ts_ns", right_on="ts_ns", strategy="backward"
    )

    t_ks = df_t["log_moneyness"].to_numpy().astype(float)
    t_Ts = df_t["T"].to_numpy().astype(float)
    t_fwds = df_t["forward"].to_numpy().astype(float)
    t_K_abs = t_fwds * np.exp(t_ks)
    th_mat = np.column_stack([df_t[f"_th{j}"].to_numpy() for j in range(n_state)])

    matched_h = np.full(len(df_t), float("nan"))

    if q_bar_0 is not None:
        for i in range(len(df_t)):
            th = th_mat[i]
            K_i, T_i = t_K_abs[i], t_Ts[i]
            if (
                np.isfinite(th).all()
                and np.isfinite(K_i)
                and np.isfinite(T_i)
                and T_i > 0.0
                and anchor_Ks[0] <= K_i <= anchor_Ks[-1]
            ):
                matched_h[i] = float(_obs_jit_strike(jnp.array(th), ak_jax, K_i, T_i, q_bar_0))

    sanos_kf_matched = _invert_iv_vec_strike(matched_h, t_K_abs, t_fwds, t_Ts)

    obs_iv = df_t["observed_iv"].to_numpy().astype(float)
    trade_residual = obs_iv - sanos_kf_matched

    drop_cols = [f"_th{j}" for j in range(n_state)]
    df_trades_result = df_t.with_columns(
        [
            pl.Series("sanos_kf_matched", sanos_kf_matched.tolist()),
            pl.Series("trade_residual", trade_residual.tolist()),
        ]
    ).drop(drop_cols)

    if not return_state:
        return df_quotes_result, df_trades_result

    state_cols: dict = {"ts_ns": df_q["ts_ns"]}
    state_cols.update({f"x_{j}": x_hist[:, j].tolist() for j in range(n_state)})
    df_state_out = pl.DataFrame(state_cols)
    return df_quotes_result, df_trades_result, df_state_out
