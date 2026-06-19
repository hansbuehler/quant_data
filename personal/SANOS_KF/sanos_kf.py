"""
sanos_kf.py

Online SANOS + Extended Kalman Filter for a single expiry.

The SANOS call-price mixture becomes the nonlinear observation function h(x)
of an EKF so the DLV weights and smoothing vol evolve online with each quote.

State  x ∈ ℝ^{n+1}:
    x[:n]  = log(Σᵢ)   — log squared discrete local vols (DLVs)
    x[n]   = log(σ*)   — log shared smoothing vol

Density (Ω = I, p₀ = uniform 1/n):
    q̃ᵢ = (1/n) / (1 + Σᵢ)   then projected onto {Σq=1, ΣqKⁱ=1} (forward-centered)

Fixes vs. the original implementation:
    Bug 1 — the in-filter density is now forward-centered every step (martingale
            projection wired into _obs_fn, not just applied for reporting).
    Bug 2 — process & prior noise are correlated across DLV anchors via an RBF
            kernel (cfg.correlated / cfg.ell), so a quote pulls its neighbours.
    Gap 3 — quotes are assimilated in time windows (cfg.batch_seconds): one
            time-proportional predict + one batch update per window, with the
            held-out prior = previous window's posterior (sanos_kf_price_prior).

Modes (cfg.smoothing / cfg.sigma_star_init):
    smoothing=True               — σ* is a free state (BS-mixture, state = n+1).
    smoothing=False, σ*>0        — σ* frozen (BS-mixture, state = n).
    smoothing=False, σ*=0        — EXACT-SOLVE: the SANOS exactly-determined density
                                   {Σd=1, Σdκ=1, DLV² equalities}, state = net DLV²
                                   over the n-2 interior anchors (well-conditioned
                                   ∂d/∂x). This is the SANOS_KF_Strike density ported
                                   into F-normalized strike space (no BS mixture).

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
    calls_only: bool = False  # if True, fit on call quotes only
    puts_only: bool = False  # if True, fit on put quotes only (mutually exclusive w/ calls_only)
    # Bug 2 — correlated (RBF) process & prior noise across DLV anchors so that a
    # quote at one strike pulls its neighbours (and stale wing nodes) too.
    correlated: bool = True
    ell: float = 0.05  # RBF correlation length over anchor log-moneyness
    # smoothing — when True, σ* is a free EKF state (BS-mixture smoothing). When
    # False, σ* is frozen at sigma_star_init (no σ* state, no log(0)).
    # SPECIAL CASE: smoothing=False AND sigma_star_init=0 ⇒ EXACT-SOLVE mode — the
    # density is the exactly-determined SANOS system (well-conditioned ∂d/∂x), state
    # = net DLV² at the N-2 interior anchors, like SANOS_KF_Strike (no BS mixture).
    smoothing: bool = True
    dlv_min: float = 0.001  # DLV floor for exact-solve mode (v = x + dlv_min²)
    # exact-solve only: if a batch update yields a density with negative mass beyond
    # this fraction, REJECT the update and carry the previous (valid) state forward —
    # "propagate the old fit". This keeps the state always arbitrage-free and lets us
    # score every snapshot honestly (vs. dropping bad snapshots from the metric).
    # Set to None to disable (accept every update).
    reject_neg_mass: float | None = 0.02
    # Gap 3 — assimilate quotes in time windows of this width (seconds): one
    # time-proportional predict + one batch EKF update per window, held-out prior =
    # previous window's posterior. None ⇒ legacy per-quote sequential filter.
    batch_seconds: float | None = 1.0


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


def _h_from_state(
    Sigma: jnp.ndarray,
    sigma_star: float,
    anchor_ks: jnp.ndarray,
    market_k: float,
    T: float,
) -> jnp.ndarray:
    """Normalized call price h = qᵀ·BSCallNorm given DLVs Σ and smoothing vol σ*.

    Density: Ω=I → q̃ᵢ = (1/n)/(1+Σᵢ), then affine-projected onto the moment
    constraints {Σq=1, ΣqKⁱ=1} so the filter's density is forward-centered every
    step (martingale / "forward re-centering"). The projection is linear ⇒ fully
    differentiable, so jax.jacrev picks up its effect on H automatically."""
    n = anchor_ks.shape[0]
    v = jnp.maximum(sigma_star**2 * T, 1e-10)
    q_raw = (1.0 / n) / (1.0 + Sigma)
    q = _project_q_moments(q_raw, jnp.exp(anchor_ks))
    c_vec = _bs_call_norm_jax(anchor_ks, market_k, v)
    return jnp.dot(q, c_vec)


def _obs_fn(
    x: jnp.ndarray,
    anchor_ks: jnp.ndarray,
    market_k: float,
    T: float,
) -> jnp.ndarray:
    """Observation with σ* as a free state (smoothing ON). State x = (n+1,):
    x[:n] = log Σ, x[n] = log σ*. Differentiable w.r.t. x → jac via jax.jacrev."""
    n = anchor_ks.shape[0]
    return _h_from_state(jnp.exp(x[:n]), jnp.exp(x[n]), anchor_ks, market_k, T)


def _obs_fn_ns(
    x: jnp.ndarray,
    anchor_ks: jnp.ndarray,
    market_k: float,
    T: float,
    sigma_star: float,
) -> jnp.ndarray:
    """Observation with σ* frozen at a constant (smoothing OFF). State x = (n,) =
    log Σ only — no σ* dimension, no log(0). σ*=0 ⇒ v floored at 1e-10 ⇒ the
    BS-mixture collapses to the density-weighted intrinsic (discrete payoff)."""
    return _h_from_state(jnp.exp(x), sigma_star, anchor_ks, market_k, T)


def _project_q_moments(q_raw: jnp.ndarray, anchor_K: jnp.ndarray) -> jnp.ndarray:
    """Affine projection of q̃ onto {Σq = 1, Σq·K = 1} (F-normalized, F=1).

    Differentiable (no nonneg clip): keeps the EKF observation/Jacobian smooth.
    anchor_K = exp(anchor_ks). Solves the 2×2 normal system for the minimal
    correction q = q̃ − Aᵀ(AAᵀ)⁻¹(A q̃ − b), with A = [1; K], b = [1; 1]."""
    n = anchor_K.shape[0]
    A = jnp.stack([jnp.ones(n), anchor_K])  # (2, n)
    b = jnp.array([1.0, 1.0])
    residual = A @ q_raw - b
    correction = A.T @ jnp.linalg.solve(A @ A.T, residual)
    return q_raw - correction


# Pre-compile Jacobian (traces once per shape, then cached by JAX)
_jac_fn = jax.jit(jax.jacrev(_obs_fn))
_obs_jit = jax.jit(_obs_fn)

# Batched (vmapped) observation + Jacobian: same prior state x, many (market_k, T).
# Used by the per-snapshot batch update (Gap 3) to assimilate a whole 1-s window.
_obs_vmap = jax.jit(jax.vmap(_obs_fn, in_axes=(None, None, 0, 0)))
_jac_vmap = jax.jit(jax.vmap(_jac_fn, in_axes=(None, None, 0, 0)))

# Smoothing-OFF variants (σ* frozen, state = (n,) DLVs). jacrev differentiates
# w.r.t. x (argnum 0) only; sigma_star is a constant broadcast across quotes.
_jac_fn_ns = jax.jit(jax.jacrev(_obs_fn_ns))
_obs_jit_ns = jax.jit(_obs_fn_ns)
_obs_vmap_ns = jax.jit(jax.vmap(_obs_fn_ns, in_axes=(None, None, 0, 0, None)))
_jac_vmap_ns = jax.jit(jax.vmap(_jac_fn_ns, in_axes=(None, None, 0, 0, None)))


def _rbf(nodes: np.ndarray, ell: float, correlated: bool) -> np.ndarray:
    """(m×m) RBF correlation over node positions; identity if correlated=False/ell<=0."""
    m = len(nodes)
    if not (correlated and ell > 0):
        return np.eye(m)
    dist = np.abs(nodes[:, None] - nodes[None, :])
    return np.exp(-0.5 * (dist / ell) ** 2)


def _corr_full(anchor_ks: np.ndarray, ell: float, correlated: bool, smoothing: bool) -> np.ndarray:
    """State correlation matrix: RBF over the n DLV anchors (in log-moneyness), with
    σ* (last index, only present when smoothing=True) left uncorrelated. Identity
    when correlated=False or ell<=0. Shape (n+1) with smoothing, else (n)."""
    n = len(anchor_ks)
    n_state = n + 1 if smoothing else n
    C = np.eye(n_state)
    C[:n, :n] = _rbf(anchor_ks, ell, correlated)
    return C


# ---------------------------------------------------------------------------
# Exact-solve density (SANOS_KF_Strike-style), in F-normalized strike space
# ---------------------------------------------------------------------------
# Used when smoothing=False AND sigma_star_init=0: instead of the BS-mixture
# heuristic q ∝ (1/n)/(1+Σ), the density is the exactly-determined SANOS system
#   row 0 : Σ d        = 1          (normalisation)
#   row 1 : Σ d·κ      = 1          (martingale, κ = K/F)
#   rows  : DLV² equalities (tri-diagonal, net DLV² state x ≥ 0)
# giving a well-conditioned ∂d/∂x (the ill-conditioning of the mixture is what
# made σ*=0 unlearnable). State x = net DLV² at the N-2 interior anchors.


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


def exact_solve_density(x_state, anchor_ks, tte_ref, dlv_min=0.001, fwd=1.0):
    """Reconstruct the F-normalized risk-neutral density on the anchor grid from a
    net-DLV² state (exact-solve mode), for diagnostics / negative-mass scoring.

    Builds the same κ grid + box weights as run_sanos_kf's exact mode (so pass the
    SAME tte_ref it used — the median quote T) and solves A(x) d = b. Returns the
    density d (length len(anchor_ks)), or None if the linear system is singular.
    The signed negative mass is then Σ min(d, 0) (0 = arbitrage-free)."""
    anchor_ks = np.asarray(anchor_ks, float)
    kappa = np.exp(anchor_ks)
    N = len(kappa)
    xrow = np.concatenate(([2 * kappa[0] - kappa[1]], kappa, [2 * kappa[-1] - kappa[-2]]))
    box = kappa**2 / (xrow[2:] - xrow[:-2]) * tte_ref
    box[0] = 0.0
    box[-1] = 0.0
    pay_full = np.maximum(kappa[np.newaxis, :] - kappa[:, np.newaxis], 0.0)
    d, _ = _exact_dens_ddx(np.asarray(x_state, float), kappa, box, pay_full, dlv_min, N, fwd=fwd)
    return d


def _make_exact_fns(kappa, box, pay_full, dlv_min):
    """Build (obs_scalar, jac_scalar, obs_vec, jac_vec) closures with the same
    signatures as the JAX ones, so the EKF loop is mode-agnostic. The observation
    is the normalized call price h(κ_m) = Σ_j d_j·(κ_j − κ_m)⁺ (matches mid/(F·df));
    `anchor_ks`/`T` args are ignored (grid & horizon are baked into box)."""
    N = len(kappa)
    n_state = N - 2

    def _dd(xx):
        return _exact_dens_ddx(np.asarray(xx), kappa, box, pay_full, dlv_min, N)

    def obs_scalar(xx, ak, mk, T):
        d, _ = _dd(xx)
        if d is None:
            return float("nan")
        return float(np.maximum(kappa - np.exp(float(mk)), 0.0) @ d)

    def jac_scalar(xx, ak, mk, T):
        d, dd = _dd(xx)
        if d is None:
            return np.full(n_state, np.nan)
        return np.maximum(kappa - np.exp(float(mk)), 0.0) @ dd

    def obs_vec(xx, ak, mk_arr, T_arr):
        d, _ = _dd(xx)
        km = np.exp(np.asarray(mk_arr, float))
        if d is None:
            return np.full(len(km), np.nan)
        return np.maximum(kappa[None, :] - km[:, None], 0.0) @ d

    def jac_vec(xx, ak, mk_arr, T_arr):
        d, dd = _dd(xx)
        km = np.exp(np.asarray(mk_arr, float))
        if d is None:
            return np.full((len(km), n_state), np.nan)
        return np.maximum(kappa[None, :] - km[:, None], 0.0) @ dd

    def dens_fn(xx):
        d, _ = _dd(xx)
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

    NOTE: this corresponds to the *soft-normalized* density (Σq=1 only) and is kept
    as a paper-appendix reference / sanity check. The live filter now uses the
    forward-centered density (`_project_q_moments`, {Σq=1, ΣqK=1}) and gets its
    Jacobian from `jax.jacrev(_obs_fn)`, so this function no longer matches H exactly.

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
    obs_scalar,
    jac_scalar,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    One predict + update step of the SANOS EKF.

    obs_scalar / jac_scalar are h and ∂h/∂x for one quote (smoothing on/off bound
    by the caller). Returns (x_new, P_new, z_hat).
    """
    # Predict (random walk)
    P_pred = P + Q

    # Nonlinear observation + linearized Jacobian
    x_jax = jnp.array(x)
    ak_jax = jnp.array(anchor_ks)
    mk = float(market_k)
    Tf = float(T)

    z_hat = float(obs_scalar(x_jax, ak_jax, mk, Tf))
    H = np.array(jac_scalar(x_jax, ak_jax, mk, Tf))  # (n_state,)

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

    z_hat_post = float(obs_scalar(jnp.array(x_new), jnp.array(anchor_ks), market_k, T))
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

    # Mode selection. smoothing=False with sigma_star_init=0 ⇒ EXACT-SOLVE (state =
    # net DLV², SANOS exactly-determined density). Otherwise BS-mixture, with σ* a
    # free state (smoothing=True) or frozen at sigma_star_init (smoothing=False).
    ak_j = jnp.array(anchor_ks)
    exact = (not cfg.smoothing) and (float(cfg.sigma_star_init) == 0.0)
    clip_nonneg = exact
    dens_fn = None  # exact-solve density reconstructor (for update rejection)
    reject_tol = cfg.reject_neg_mass if exact else None

    if exact:
        # F-normalized strike grid (κ = K/F, forward = 1) + DLV² box weights. The
        # horizon for box uses a fixed reference tte (median T) — the terminal
        # density is T-independent, so the quote's own T does not enter the solve.
        kappa = anchor_K
        tte_ref = float(np.nanmedian(df_quotes["T"].to_numpy().astype(float)))
        xrow = np.concatenate(([2 * kappa[0] - kappa[1]], kappa, [2 * kappa[-1] - kappa[-2]]))
        box = kappa**2 / (xrow[2:] - xrow[:-2]) * tte_ref
        box[0] = 0.0
        box[-1] = 0.0
        pay_full = np.maximum(kappa[np.newaxis, :] - kappa[:, np.newaxis], 0.0)  # (N,N)
        obs_scalar, jac_scalar, obs_vec, jac_vec, dens_fn = _make_exact_fns(
            kappa, box, pay_full, cfg.dlv_min
        )
        x = np.full(n - 2, cfg.Sigma_init)  # flat net DLV² over interior anchors
        var_diag = np.full(n - 2, cfg.Q_scale_dlv)
        C_full = _rbf(anchor_ks[1:-1], cfg.ell, cfg.correlated)
    elif cfg.smoothing:
        obs_scalar, jac_scalar = _obs_jit, _jac_fn
        obs_vec, jac_vec = _obs_vmap, _jac_vmap
        var_diag = np.concatenate([np.full(n, cfg.Q_scale_dlv), np.array([cfg.Q_scale_vol])])
        x = np.concatenate([np.full(n, np.log(cfg.Sigma_init)), [np.log(cfg.sigma_star_init)]])
        C_full = _corr_full(anchor_ks, cfg.ell, cfg.correlated, smoothing=True)
    else:
        _ss = float(cfg.sigma_star_init)
        obs_scalar = lambda xx, ak, mk, T: _obs_jit_ns(xx, ak, mk, T, _ss)  # noqa: E731
        jac_scalar = lambda xx, ak, mk, T: _jac_fn_ns(xx, ak, mk, T, _ss)  # noqa: E731
        obs_vec = lambda xx, ak, mk, T: _obs_vmap_ns(xx, ak, mk, T, _ss)  # noqa: E731
        jac_vec = lambda xx, ak, mk, T: _jac_vmap_ns(xx, ak, mk, T, _ss)  # noqa: E731
        var_diag = np.full(n, cfg.Q_scale_dlv)
        x = np.full(n, np.log(cfg.Sigma_init))
        C_full = _corr_full(anchor_ks, cfg.ell, cfg.correlated, smoothing=False)
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

    # ── Vectorized per-quote normalization → call-price space (PCP for puts) ──
    scale = fwds * dfs
    ok_scale = np.isfinite(scale) & (scale > 0)
    scale_safe = np.where(ok_scale, scale, 1.0)
    bid_norm = bids / scale_safe
    ask_norm = asks / scale_safe
    put = ~is_calls
    bid_norm = np.where(put, bid_norm + 1.0 - np.exp(ks), bid_norm)
    ask_norm = np.where(put, ask_norm + 1.0 - np.exp(ks), ask_norm)
    mid_norm = 0.5 * (bid_norm + ask_norm)
    R_all = np.maximum((0.5 * (ask_norm - bid_norm)) ** 2, 1e-10)
    valid = ok_scale & np.isfinite(mid_norm) & np.isfinite(ks) & np.isfinite(Ts) & (Ts > 0.0)
    if cfg.calls_only and cfg.puts_only:
        raise ValueError("calls_only and puts_only are mutually exclusive")
    if cfg.calls_only:
        valid &= is_calls
    if cfg.puts_only:
        valid &= ~is_calls

    if cfg.batch_seconds is None:
        # ── Legacy per-quote sequential filter ────────────────────────────────
        _last_ts = ts_ns_arr[0] if n_q > 0 else 0
        for i in range(n_q):
            if not valid[i]:
                _last_ts = ts_ns_arr[i]
                x_hist[i] = x
                continue
            dt_ns = max(float(ts_ns_arr[i] - _last_ts), 0.0)
            Q = Q_rate * (dt_ns * _q_per_ns)
            _last_ts = ts_ns_arr[i]
            x, P, z_hat = _ekf_step(
                x, P, anchor_ks, ks[i], Ts[i], mid_norm[i], R_all[i], Q, obs_scalar, jac_scalar
            )
            if clip_nonneg:  # exact-solve: net DLV² state must stay ≥ 0
                x = np.maximum(x, 0.0)
                z_hat = obs_scalar(x, anchor_ks, ks[i], Ts[i])
            z_hats[i] = z_hat
            x_hist[i] = x
    else:
        # ── Gap 3: per-snapshot batch filter ──────────────────────────────────
        # One time-proportional predict + one batch EKF update per window; held-out
        # prior for every quote = the previous window's posterior, so a quote burst
        # no longer inflates the state by quote count.
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

            ks_w = jnp.array(ks[idx_v])
            Ts_w = jnp.array(Ts[idx_v])
            xj = jnp.array(x_prior)
            z_hat_p = np.asarray(obs_vec(xj, ak_j, ks_w, Ts_w))  # (m,)
            H = np.asarray(jac_vec(xj, ak_j, ks_w, Ts_w))  # (m, n_state)
            z_hats_prior[idx_v] = z_hat_p

            fin = np.isfinite(z_hat_p) & np.isfinite(H).all(axis=1)
            if not fin.any():
                x, P = x_prior, P_pred
                x_hist[idx] = x_prior
                continue

            Hf = H[fin]
            innov = mid_norm[idx_v][fin] - z_hat_p[fin]
            Rf = np.diag(R_all[idx_v][fin])
            S = Hf @ P_pred @ Hf.T + Rf
            try:
                K_gain = P_pred @ Hf.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                x, P = x_prior, P_pred
                x_hist[idx] = x_prior
                continue
            x_cand = x_prior + K_gain @ innov
            if clip_nonneg:  # exact-solve: net DLV² state must stay ≥ 0
                x_cand = np.maximum(x_cand, 0.0)

            # Reject-update guard (exact-solve): if the candidate posterior implies a
            # density with negative mass beyond reject_tol, discard the update and
            # carry the previous valid state forward ("propagate the old fit"), so the
            # state stays arbitrage-free and every snapshot is still scored honestly.
            reject = False
            if reject_tol is not None and dens_fn is not None:
                d_cand = dens_fn(x_cand)
                reject = d_cand is None or (-np.minimum(d_cand, 0.0).sum() > reject_tol)

            if reject:
                x, P = x_prior, P_pred
                z_hats[idx_v] = z_hat_p  # posterior == held-out prior (no update taken)
            else:
                x = x_cand
                IKH = I_n - K_gain @ Hf
                P = IKH @ P_pred @ IKH.T + K_gain @ Rf @ K_gain.T  # Joseph form
                P = 0.5 * (P + P.T)
                z_hats[idx_v] = np.asarray(obs_vec(jnp.array(x), ak_j, ks_w, Ts_w))
            x_hist[idx] = x

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
    th_mat = np.column_stack([df_t[f"_th{j}"].to_numpy() for j in range(n_state)])

    matched_call_prices = np.full(len(df_t), float("nan"))

    for i in range(len(df_t)):
        th = th_mat[i]
        k, T = t_ks[i], t_Ts[i]
        if np.isfinite(th).all() and np.isfinite(k) and np.isfinite(T) and T > 0.0:
            matched_call_prices[i] = float(obs_scalar(jnp.array(th), ak_j, k, T))

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
