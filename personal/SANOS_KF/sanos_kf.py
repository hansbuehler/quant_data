"""
sanos_kf.py

Online SANOS + Extended Kalman Filter for a single expiry.

ONE model: the exact tridiagonal SANOS density (same parameterization as
quantdata.sanos), run in ABSOLUTE STRIKE space on the strike-aligned grid (interior
nodes on the observed strikes within [k_min,k_max], ≤ n_anchors, padded ±1 node).
cfg.ell is in DOLLARS.

State  x = net DLV² over the N-2 interior strike nodes (≥ 0).

Density (per window, re-centered on the LIVE forward F):
    A(x)·d = b with rows {Σd = 1, Σd·K = F, tri-diagonal DLV² equalities} — the
    exactly-determined SANOS system, well-conditioned ∂d/∂x. (Earlier BS-mixture
    heuristics q ∝ (1/n)/(1+Σ) are GONE — they were not the SANOS density and could
    not fit; see the quantdata.sanos tridiagonal operator.)

Observation (undiscounted call price C(K)/df, matched against mid/df):
    η = 0  (cfg.smoothing=False):  h(x,K,F) = Σⱼ dⱼ·(Kⱼ − K)⁺          (intrinsic)
    η > 0  (cfg.smoothing=True) :  h(x,K,F) = Σⱼ dⱼ·[Kⱼ·N(d1ⱼ) − K·N(d2ⱼ)]
                                   d1ⱼ = (log(Kⱼ/K) + v/2)/√v,  v = η·V_ATM
η (= cfg.eta, paper smoothness) sets the smoothing variance v = η·V_ATM (V_ATM =
estimated ATM total variance) that blurs each density node into a lognormal — it
smooths only the OBSERVATION, never the density (exactly how quantdata.sanos applies
its `sqrtVar`). The η→0 limit recovers the intrinsic.
The output sanos_kf_price is divided by F → F-normalized call (unchanged convention).

Design notes:
    - in-filter density is forward-centered every window (Σd·K = live F);
    - process & prior noise correlated across DLV nodes via an RBF kernel
      (cfg.correlated / cfg.ell in dollars), so a quote pulls its neighbours;
    - quotes assimilated in time windows (cfg.batch_seconds): one time-proportional
      predict + one batch update per window, held-out prior = previous window's
      posterior (sanos_kf_price_prior);
    - reject-update guard keeps the density arbitrage-free (cfg.reject_neg_mass).

Jacobian: analytic ∂d/∂x (numpy) through the smoothed/intrinsic payoff.

Public API
----------
SanosKFConfig  — hyperparameters
run_sanos_kf() — main EKF loop (single expiry)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from cdxcore.bs import bs as _cdx_bs
from scipy.stats import norm as np_norm

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
    # eta — SANOS smoothness ∈ [0,1) (paper §3.1, eq 17). The smoothing kernel TOTAL
    # variance is η·V_ATM (V_ATM = ATM total variance, estimated per run from near-ATM
    # IV), i.e. each density node is blurred by a lognormal of vol √η·σ_ATM. η=0 ⇒
    # intrinsic (linear interpolation); η=1 ⇒ ATM-only. Paper default η=0.25 (=0.5²).
    # Applied iff smoothing=True.
    eta: float = 0.25
    Sigma_init: float = 0.25  # initial net DLV² for all interior nodes (uniform)
    Q_scale_dlv: float = 1e-3  # process noise scale for net DLV² per trading day
    P0_scale: float = 1e-4  # initial covariance diagonal scale
    trading_day_s: float = 23400.0  # trading day length in seconds (6.5 h) for Q scaling
    calls_only: bool = False  # if True, fit on call quotes only
    puts_only: bool = False  # if True, fit on put quotes only (mutually exclusive w/ calls_only)
    # correlated (RBF) process & prior noise across DLV anchors so that a quote at one
    # strike pulls its neighbours (and stale wing nodes) too. ell is in DOLLARS.
    correlated: bool = True
    ell: float = 3.0  # RBF correlation length in DOLLARS (absolute strike distance)
    # smoothing — the model is ALWAYS the exact tridiagonal SANOS density (state = net
    # DLV² over the N-2 interior nodes). This flag only gates the OBSERVATION smoothing:
    # True ⇒ price each density node as a lognormal of variance η·V_ATM (paper production
    # model); False ⇒ η forced to 0 ⇒ intrinsic discrete calls (the original exact-solve).
    smoothing: bool = True
    dlv_min: float = 0.001  # DLV floor (v = x + dlv_min²) so ∂d/∂x doesn't vanish
    # exact-solve only: if a batch update yields a density with negative mass beyond
    # this fraction, REJECT the update and carry the previous (valid) state forward —
    # "propagate the old fit". This keeps the state always arbitrage-free and lets us
    # score every snapshot honestly (vs. dropping bad snapshots from the metric).
    # Set to None to disable (accept every update).
    reject_neg_mass: float | None = 0.02
    # Gap 3 — assimilate quotes in time windows of this width (seconds): one
    # time-proportional predict + one batch EKF update per window, held-out prior =
    # previous window's posterior. Required (None raises).
    batch_seconds: float | None = 1.0


# ---------------------------------------------------------------------------
# State correlation (numpy)
# ---------------------------------------------------------------------------


def _rbf(nodes: np.ndarray, ell: float, correlated: bool) -> np.ndarray:
    """(m×m) RBF correlation over node positions; identity if correlated=False/ell<=0.
    ell is in DOLLARS (the nodes are absolute strikes)."""
    m = len(nodes)
    if not (correlated and ell > 0):
        return np.eye(m)
    dist = np.abs(nodes[:, None] - nodes[None, :])
    return np.exp(-0.5 * (dist / ell) ** 2)


# ---------------------------------------------------------------------------
# Exact tridiagonal SANOS density (mimics quantdata.sanos), in absolute strike space
# ---------------------------------------------------------------------------
# The density is the exactly-determined SANOS system (same parameterization as the
# quantdata.sanos tridiagonal operator, here as a single-snapshot linear solve):
#   row 0 : Σ d        = 1          (normalisation)
#   row 1 : Σ d·K      = F          (martingale, absolute strikes / live forward)
#   rows  : DLV² equalities (tri-diagonal, net DLV² state x ≥ 0)
# giving a well-conditioned ∂d/∂x. State x = net DLV² at the N-2 interior anchors.
# η·V_ATM (η=cfg.eta, when cfg.smoothing) only smooths the OBSERVATION (each density
# node prices as a lognormal of variance η·V_ATM instead of an intrinsic), never the
# density itself — see _make_exact_fns. This is exactly how quantdata.sanos applies
# its `sqrtVar` smoothing on top of the fitted density.


def _exact_dens_ddx(x, kappa, box, pay_full, dlv_min, N, fwd=1.0):
    """Solve A(x) d = b for the density d and return (d, ∂d/∂x).

    ∂d/∂x_s = box[s+1]·d[s+1]·A⁻¹[:, s+2] (single-entry ∂A/∂x). Returns
    (None, None) if the linear system is singular."""
    v = np.asarray(x, float) + dlv_min**2
    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0] = 1.0
    b[0] = 1.0
    A[1] = kappa
    b[1] = fwd
    A[2:N] = pay_full[1 : N - 1]
    A[np.arange(2, N), np.arange(1, N - 1)] -= v * box[1 : N - 1]
    b[2:N] = np.maximum(fwd - kappa[1 : N - 1], 0.0)
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None, None
    d = Ainv @ b
    dd_dx = Ainv[:, 2:N] * (box[1 : N - 1] * d[1 : N - 1])[np.newaxis, :]
    return d, dd_dx


def build_exact_grid(df_quotes, cfg):
    """Construct the absolute strike-aligned grid + reference horizon for exact-solve
    mode (interior nodes on observed strikes within [k_min,k_max], subsampled to
    ≤ n_anchors, padded ±1 node). Returns (grid_K, tte_ref). Use this to reconstruct
    the model density for negative-mass scoring (see exact_solve_density)."""
    Kall = (
        df_quotes["strike"].to_numpy().astype(float)
        if "strike" in df_quotes.columns
        else (df_quotes["forward"] * np.exp(df_quotes["log_moneyness"])).to_numpy().astype(float)
    )
    F0 = float(np.nanmedian(df_quotes["forward"].to_numpy().astype(float)))
    lo, hi = F0 * np.exp(cfg.k_min), F0 * np.exp(cfg.k_max)
    Kq = np.unique(np.round(Kall[np.isfinite(Kall)], 4))
    Kq = Kq[(Kq >= lo) & (Kq <= hi)]
    if len(Kq) > cfg.n_anchors:
        Kq = Kq[np.unique(np.linspace(0, len(Kq) - 1, cfg.n_anchors).round().astype(int))]
    grid_K = np.concatenate(([Kq[0] * 0.9], Kq, [Kq[-1] * 1.1]))
    tte_ref = float(np.nanmedian(df_quotes["T"].to_numpy().astype(float)))
    return grid_K, tte_ref


def _estimate_atm_var(df_quotes, tte_ref, n_atm_strikes: int = 8) -> float:
    """Estimate the ATM total variance V_ATM ≈ σ²_ATM·tte_ref for the SANOS smoothing
    kernel (kernel variance = η·V_ATM, paper §3.1).

    σ_ATM = skew-corrected ATM vol: take the per-strike median observed_iv over the
    `n_atm_strikes` DISTINCT strikes nearest the (median) forward — strike-count window, so
    it's robust to the $1 grid and the forward sitting between strikes — and read the OLS
    intercept of iv ≈ a + b·k at k=0 (= the forward; interpolates between strikes, removes
    the linear skew). df_quotes holds calls AND puts (puts_only only gates the filter), so
    these straddle the forward. Falls back to the near-strike median, then a price-based ATM
    approximation (Brenner–Subrahmanyam: normalized call ≈ 0.3989·σ·√T), then 0 (no smoothing)."""
    k = df_quotes["log_moneyness"].to_numpy().astype(float)
    F0 = float(np.nanmedian(df_quotes["forward"].to_numpy().astype(float)))
    K = (
        df_quotes["strike"].to_numpy().astype(float)
        if "strike" in df_quotes.columns
        else df_quotes["forward"].to_numpy().astype(float) * np.exp(k)
    )
    uniqK = np.unique(K[np.isfinite(K) & (K > 0)])
    if uniqK.size == 0:
        return 0.0
    nearK = uniqK[np.argsort(np.abs(uniqK - F0))[: max(int(n_atm_strikes), 2)]]

    # Per-strike (k, median IV) points at the near strikes → OLS intercept at k=0.
    ks: list[float] = []
    ivs: list[float] = []
    if "observed_iv" in df_quotes.columns:
        iv = df_quotes["observed_iv"].to_numpy().astype(float)
        for s in nearK:
            sel = (K == s) & np.isfinite(iv) & (iv > 0)
            if sel.any():
                ks.append(float(np.log(s / F0)))
                ivs.append(float(np.median(iv[sel])))
    ks_arr = np.asarray(ks)
    ivs_arr = np.asarray(ivs)

    sigma_atm = float("nan")
    if ivs_arr.size >= 3 and np.ptp(ks_arr) > 1e-6:
        a, _b = np.linalg.lstsq(np.vstack([np.ones(ivs_arr.size), ks_arr]).T, ivs_arr, rcond=None)[
            0
        ]
        sigma_atm = float(a)
    elif ivs_arr.size >= 1:
        sigma_atm = float(np.median(ivs_arr))

    if not (np.isfinite(sigma_atm) and sigma_atm > 0.0):
        # Price-based fallback (no observed_iv, or degenerate fit) at the near strikes.
        Fq = df_quotes["forward"].to_numpy().astype(float)
        T = df_quotes["T"].to_numpy().astype(float)
        ic = df_quotes["is_call"].to_numpy().astype(bool)
        mid = 0.5 * (
            df_quotes["bid_price"].to_numpy().astype(float)
            + df_quotes["ask_price"].to_numpy().astype(float)
        )
        call = np.where(ic, mid, mid + Fq - K)  # PCP → undiscounted call
        cnorm = call / np.where(Fq > 0, Fq, 1.0)
        sel = np.isin(K, nearK) & np.isfinite(cnorm) & (cnorm > 0) & np.isfinite(T) & (T > 0)
        if not sel.any():
            return 0.0
        sigma_atm = float(np.median(cnorm[sel] / (0.3989 * np.sqrt(T[sel]))))

    if not (np.isfinite(sigma_atm) and sigma_atm > 0.0):
        return 0.0
    return sigma_atm**2 * tte_ref


def exact_solve_density(x_state, grid_K, tte_ref, dlv_min=0.001, fwd=None):
    """Reconstruct the risk-neutral density on the ABSOLUTE strike grid from a
    net-DLV² state (exact-solve mode), for diagnostics / negative-mass scoring.

    Builds the same K grid + box weights as run_sanos_kf's exact mode (pass the SAME
    grid_K and tte_ref it used) and solves A(x) d = b with mean = fwd (defaults to the
    grid midpoint if None). Returns the density d (length len(grid_K)), or None if the
    system is singular. The signed negative mass is Σ min(d, 0) (0 = arbitrage-free)."""
    grid_K = np.asarray(grid_K, float)
    N = len(grid_K)
    if fwd is None:
        fwd = float(0.5 * (grid_K[0] + grid_K[-1]))
    xrow = np.concatenate(([2 * grid_K[0] - grid_K[1]], grid_K, [2 * grid_K[-1] - grid_K[-2]]))
    box = grid_K**2 / (xrow[2:] - xrow[:-2]) * tte_ref
    box[0] = 0.0
    box[-1] = 0.0
    pay_full = np.maximum(grid_K[np.newaxis, :] - grid_K[:, np.newaxis], 0.0)
    d, _ = _exact_dens_ddx(np.asarray(x_state, float), grid_K, box, pay_full, dlv_min, N, fwd=fwd)
    return d


def _make_exact_fns(grid_K, box, pay_full, dlv_min, kernel_var=0.0):
    """Build (obs_scalar, jac_scalar, obs_vec, jac_vec, dens_fn) closures for the
    exact tridiagonal SANOS density. The density `d(x)` is always the exact solve
    (re-centered Σd·K = live fwd each window); `kernel_var` only changes how the
    market quotes are PRICED from that density (the observation payoff):

      kernel_var = 0  → intrinsic    payoff_j(K) = (K_j − K)⁺          (discrete calls)
      kernel_var > 0  → BS-smoothed  payoff_j(K) = K_j·N(d1) − K·N(d2),
                        d1 = (log(K_j/K) + v/2)/√v with TOTAL variance v = kernel_var.

    Per the SANOS paper (§3.1, eq 17) the production model uses Call(K_j, K, η·V_ATM):
    the kernel variance is η·V_ATM (η = smoothness ∈ [0,1), V_ATM = ATM total variance),
    i.e. each density node is blurred by a lognormal of vol √η·σ_ATM. So pass
    kernel_var = η·V_ATM here. h(K) = Σ_j payoff_j(K)·d_j is the undiscounted call
    (= C/df). The kernel_var→0 limit recovers the intrinsic (= linear interpolation)
    exactly. Smoothing the nodes preserves the martingale (Σd·K_j = F) and no-arbitrage,
    so it only blurs the discrete-call kinks."""
    N = len(grid_K)
    n_state = N - 2
    gk_row = grid_K[None, :]  # (1, N)

    if kernel_var and kernel_var > 0.0:
        v = max(float(kernel_var), 1e-12)
        sqrt_v = np.sqrt(v)

        def _pay(K_arr):  # (m, N): BS-call of each density node (spot=K_j) at strike K_i
            K_col = np.asarray(K_arr, float).reshape(-1, 1)
            d1 = (np.log(gk_row / K_col) + 0.5 * v) / sqrt_v
            d2 = d1 - sqrt_v
            return gk_row * np_norm.cdf(d1) - K_col * np_norm.cdf(d2)
    else:

        def _pay(K_arr):  # (m, N): intrinsic discrete-call payoff (K_j − K_i)⁺
            K_col = np.asarray(K_arr, float).reshape(-1, 1)
            return np.maximum(gk_row - K_col, 0.0)

    def _dd(xx, fwd):
        return _exact_dens_ddx(np.asarray(xx), grid_K, box, pay_full, dlv_min, N, fwd=fwd)

    def obs_scalar(xx, K, fwd):
        d, _ = _dd(xx, fwd)
        if d is None:
            return float("nan")
        return float(_pay([K])[0] @ d)

    def jac_scalar(xx, K, fwd):
        d, dd = _dd(xx, fwd)
        if d is None:
            return np.full(n_state, np.nan)
        return _pay([K])[0] @ dd

    def obs_vec(xx, K_arr, fwd):
        d, _ = _dd(xx, fwd)
        if d is None:
            return np.full(len(K_arr), np.nan)
        return _pay(K_arr) @ d

    def jac_vec(xx, K_arr, fwd):
        d, dd = _dd(xx, fwd)
        if d is None:
            return np.full((len(K_arr), n_state), np.nan)
        return _pay(K_arr) @ dd

    def dens_fn(xx, fwd):
        d, _ = _dd(xx, fwd)
        return d

    return obs_scalar, jac_scalar, obs_vec, jac_vec, dens_fn


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
    anchor_ks : deprecated / ignored (anchors are the data's strike grid).

    Returns
    -------
    df_quotes_result : df_quotes + sanos_kf_price (normalized call), sanos_kf_iv
    df_trades_result : df_trades + sanos_kf_matched (IV), trade_residual
    """
    # ONE model: the exact tridiagonal SANOS density on the strike-aligned grid
    # (interior nodes on observed strikes within [k_min,k_max], ≤ n_anchors, ±1 pad).
    # State = net DLV² over the N-2 interior nodes; density re-centers Σd·K = live fwd
    # each window. cfg.smoothing only controls the OBSERVATION: the SANOS production
    # smoothing (paper §3.1) prices each density node as a lognormal of TOTAL variance
    # η·V_ATM (η=cfg.eta, V_ATM = estimated ATM total variance), i.e. smoothing vol
    # √η·σ_ATM; cfg.smoothing=False ⇒ η=0 ⇒ intrinsic discrete calls (exact-solve).
    # The `anchor_ks` arg is ignored (kept for signature back-compat).
    if cfg.batch_seconds is None:
        raise NotImplementedError("run_sanos_kf requires batched mode (cfg.batch_seconds set)")
    grid_K, tte_ref = build_exact_grid(df_quotes, cfg)
    eta = float(cfg.eta) if cfg.smoothing else 0.0
    kernel_var = eta * _estimate_atm_var(df_quotes, tte_ref) if eta > 0.0 else 0.0

    xrow = np.concatenate(([2 * grid_K[0] - grid_K[1]], grid_K, [2 * grid_K[-1] - grid_K[-2]]))
    box = grid_K**2 / (xrow[2:] - xrow[:-2]) * tte_ref
    box[0] = 0.0
    box[-1] = 0.0
    pay_full = np.maximum(grid_K[np.newaxis, :] - grid_K[:, np.newaxis], 0.0)  # (N,N) intrinsic
    obs_scalar, jac_scalar, obs_vec, jac_vec, dens_fn = _make_exact_fns(
        grid_K, box, pay_full, cfg.dlv_min, kernel_var=kernel_var
    )
    x = np.full(len(grid_K) - 2, cfg.Sigma_init)  # flat net DLV² over interior nodes
    var_diag = np.full(len(grid_K) - 2, cfg.Q_scale_dlv)
    C_full = _rbf(grid_K[1:-1], cfg.ell, cfg.correlated)  # ell in DOLLARS

    clip_nonneg = True  # net DLV² state must stay ≥ 0
    reject_tol = cfg.reject_neg_mass  # density-arbitrage reject guard (dens_fn always set)
    n_state = len(x)

    # Q_rate: process noise covariance per trading day (scaled to per-second below).
    # Bug 2: correlated across DLV anchors via the RBF matrix C, Q = D^½ C D^½.
    sd = np.sqrt(var_diag)
    Q_rate = (sd[:, None] * sd[None, :]) * C_full
    P = cfg.P0_scale * C_full

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
    z_hats = np.full(n_q, float("nan"))  # posterior model price (normalized call)
    z_hats_prior = np.full(n_q, float("nan"))  # held-out (prior) model price — Gap 3
    x_hist = np.full((n_q, n_state), float("nan"))

    _q_per_ns = 1.0 / (cfg.trading_day_s * 1e9)  # Q scale factor per nanosecond

    # Absolute strikes + undiscounted call space (C/df) — the observation space.
    # PCP for puts: C/df = P/df + (F − K).
    scale = fwds * dfs
    ok_scale = np.isfinite(scale) & (scale > 0)
    K_abs = fwds * np.exp(ks)
    dfs_safe = np.where(np.isfinite(dfs) & (dfs > 0), dfs, 1.0)
    bid_c = np.where(is_calls, bids / dfs_safe, bids / dfs_safe + fwds - K_abs)
    ask_c = np.where(is_calls, asks / dfs_safe, asks / dfs_safe + fwds - K_abs)
    mid_obs = 0.5 * (bid_c + ask_c)
    R_obs = np.maximum((0.5 * (ask_c - bid_c)) ** 2, 1e-10)

    valid = ok_scale & np.isfinite(mid_obs) & np.isfinite(ks) & np.isfinite(Ts) & (Ts > 0.0)
    if cfg.calls_only and cfg.puts_only:
        raise ValueError("calls_only and puts_only are mutually exclusive")
    if cfg.calls_only:
        valid &= is_calls
    if cfg.puts_only:
        valid &= ~is_calls

    # ── Gap 3: per-snapshot batch filter ──────────────────────────────────────
    # One time-proportional predict + one batch EKF update per window; held-out prior
    # for every quote = the previous window's posterior, so a quote burst no longer
    # inflates the state by quote count. obs/jac take (x, K_abs, live_fwd).
    batch_ns = int(cfg.batch_seconds * 1e9)
    window = ts_ns_arr // batch_ns
    I_n = np.eye(n_state)
    _last_ts = ts_ns_arr[0] if n_q > 0 else 0
    bounds = np.flatnonzero(np.diff(window)) + 1 if n_q else np.array([], dtype=int)
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n_q]))
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        idx = np.arange(s, e)
        idx_v = idx[valid[idx]]

        # predict once per window
        t_w = float(ts_ns_arr[s])
        dt_ns = max(t_w - float(_last_ts), 0.0)
        _last_ts = t_w
        P_pred = P + Q_rate * (dt_ns * _q_per_ns)
        P_pred = 0.5 * (P_pred + P_pred.T)
        x_prior = x.copy()

        if len(idx_v) == 0:  # predict-only window
            x, P = x_prior, P_pred
            x_hist[idx] = x_prior
            continue

        # Live window forward — the density re-centers via Σd·K = fwd_w.
        fwd_w = float(np.nanmedian(fwds[idx_v]))
        K_v = K_abs[idx_v]
        z_hat_p = np.asarray(obs_vec(x_prior, K_v, fwd_w))  # (m,)
        H = np.asarray(jac_vec(x_prior, K_v, fwd_w))  # (m, n_state)
        z_hats_prior[idx_v] = z_hat_p

        fin = np.isfinite(z_hat_p) & np.isfinite(H).all(axis=1)
        if not fin.any():
            x, P = x_prior, P_pred
            x_hist[idx] = x_prior
            continue

        Hf = H[fin]
        innov = mid_obs[idx_v][fin] - z_hat_p[fin]
        Rf = np.diag(R_obs[idx_v][fin])
        S = Hf @ P_pred @ Hf.T + Rf
        try:
            K_gain = P_pred @ Hf.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            x, P = x_prior, P_pred
            x_hist[idx] = x_prior
            continue
        x_cand = x_prior + K_gain @ innov
        if clip_nonneg:  # net DLV² state must stay ≥ 0
            x_cand = np.maximum(x_cand, 0.0)

        # Reject-update guard: if the candidate posterior implies a density with
        # negative mass beyond reject_tol, discard the update and carry the previous
        # valid state forward ("propagate the old fit"), so the state stays
        # arbitrage-free and every snapshot is still scored honestly.
        reject = False
        if reject_tol is not None and dens_fn is not None:
            d_cand = dens_fn(x_cand, fwd_w)
            reject = d_cand is None or (-np.minimum(d_cand, 0.0).sum() > reject_tol)

        if reject:
            x, P = x_prior, P_pred
            z_hats[idx_v] = z_hat_p  # posterior == held-out prior (no update taken)
        else:
            x = x_cand
            IKH = I_n - K_gain @ Hf
            P = IKH @ P_pred @ IKH.T + K_gain @ Rf @ K_gain.T  # Joseph form
            P = 0.5 * (P + P.T)
            z_hats[idx_v] = np.asarray(obs_vec(x, K_v, fwd_w))
        x_hist[idx] = x

    # Observations are absolute undiscounted calls (C/df); convert to the F-normalized
    # convention (price/(F·df)) so the IV inversion and every downstream eval (which
    # multiplies back by F) are unchanged.
    z_hats = z_hats / fwds
    z_hats_prior = z_hats_prior / fwds

    sanos_kf_ivs = _invert_iv_vec(z_hats, ks, Ts)
    sanos_kf_ivs_prior = _invert_iv_vec(z_hats_prior, ks, Ts)

    df_quotes_result = df_q.with_columns(
        [
            pl.Series("sanos_kf_price", z_hats.tolist()),
            pl.Series("sanos_kf_iv", sanos_kf_ivs.tolist()),
            pl.Series("sanos_kf_price_prior", z_hats_prior.tolist()),
            pl.Series("sanos_kf_iv_prior", sanos_kf_ivs_prior.tolist()),
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
    th_cols = {f"_th{j}": x_hist[:, j].tolist() for j in range(n_state)}
    df_state = pl.DataFrame({"ts_ns": df_q["ts_ns"], **th_cols})

    df_t = df_trades.sort("quote_ts_ns").join_asof(
        df_state, left_on="quote_ts_ns", right_on="ts_ns", strategy="backward"
    )

    t_ks = df_t["log_moneyness"].to_numpy().astype(float)
    t_Ts = df_t["T"].to_numpy().astype(float)
    t_fwds = (
        df_t["forward"].to_numpy().astype(float)
        if "forward" in df_t.columns
        else np.full(len(df_t), float(np.nanmedian(fwds)))
    )
    t_K_abs = t_fwds * np.exp(t_ks)  # absolute trade strike
    th_mat = np.column_stack([df_t[f"_th{j}"].to_numpy() for j in range(n_state)])

    matched_call_prices = np.full(len(df_t), float("nan"))

    # obs_scalar(x, K_abs, fwd) returns the undiscounted call (C/df) from the exact
    # density at state `th`; divide by the trade's forward to get the F-normalized call
    # that _invert_iv_vec expects.
    for i in range(len(df_t)):
        th = th_mat[i]
        K, T, fwd = t_K_abs[i], t_Ts[i], t_fwds[i]
        if not (np.isfinite(th).all() and np.isfinite(K) and np.isfinite(T) and T > 0.0):
            continue
        matched_abs = float(obs_scalar(th, K, fwd))
        matched_call_prices[i] = matched_abs / fwd if fwd > 0 else float("nan")

    sanos_kf_matched = _invert_iv_vec(matched_call_prices, t_ks, t_Ts)

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
    df_state = pl.DataFrame(state_cols)
    return df_quotes_result, df_trades_result, df_state
