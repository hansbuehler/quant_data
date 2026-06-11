"""
Stage 2 DLV² Kalman filter with cross-strike correlated process noise.

Design:
- State x ∈ ℝⁿ = latent DLV² at n interior anchors (always full length).
- Dynamics: random walk x_{t+1} = x_t + w,  w ~ N(0, Q_t),
            Q_t = q_scale · Δt_days · C,  where C is an RBF correlation matrix.
- Observation: selection-H retaining only the m_t fresh nodes each step.
            z_t = H_t x_t + v_t,  v_t ~ N(0, r_diag · I_{m_t}).
- Update: Joseph-form covariance, symmetrized each step.

The off-diagonal mass in Q (not just P₀) regenerates cross-strike coupling every
step, so fresh ATM nodes continuously pull stale wing nodes even across long gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KFConfig:
    q_scale: float = 1e-6
    ell: float = 10.0
    r_diag: float = 1e-4
    p0_scale: float = 1e-6
    p0_correlated: bool = True
    p0_matrix: Optional[np.ndarray] = None  # if set, overrides p0_scale/p0_correlated
    stencil: int = 1  # 1 = permissive, 3 = conservative
    soft_floor: Optional[float] = None  # None = pure linear; 0.0 = production
    trading_day_s: float = 23400.0
    ignore_at_dlv_max: bool = True
    dlv_max_sq_net: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_p0_ss(
    grid_K_nodes: np.ndarray,
    q_scale: float,
    ell: float,
    r_diag: float,
    dt_days_typical: float,
) -> np.ndarray:
    """
    Steady-state posterior covariance for F=I, H=I, R=r_diag·I.

    Uses the eigendecomposition of the RBF correlation matrix C to solve
    n independent scalar Riccati equations exactly.  The result is the
    steady-state P_post the filter would reach after many steps — using it
    as P₀ eliminates the warm-up period entirely.

    Parameters
    ----------
    grid_K_nodes   : (n,) DLV² node strike coordinates
    q_scale        : process noise scale (same as KFConfig.q_scale)
    ell            : RBF length scale (same as KFConfig.ell)
    r_diag         : observation noise variance (same as KFConfig.r_diag)
    dt_days_typical: typical Δt_days per step (e.g. 1/trading_day_s for 1-s batches)

    Returns
    -------
    P_post_ss : (n, n) steady-state posterior covariance matrix
    """
    C = make_corr_matrix(grid_K_nodes, ell)
    lam, V = np.linalg.eigh(C)  # lam sorted ascending; C = V diag(lam) V'
    lam = np.maximum(lam, 0.0)  # guard against floating-point near-zero negatives
    q_eig = q_scale * dt_days_typical * lam  # eigenvalues of Q per step
    # Scalar Riccati: x² - q·x - q·r = 0  →  x = (q + √(q²+4qr)) / 2  (pred ss)
    disc = np.sqrt(q_eig**2 + 4.0 * q_eig * r_diag)
    p_post_ss_eig = (disc - q_eig) / 2.0  # P_post = P_pred - Q per step
    P_post_ss = V @ np.diag(p_post_ss_eig) @ V.T
    return 0.5 * (P_post_ss + P_post_ss.T)


def make_corr_matrix(grid_K_nodes: np.ndarray, ell: float) -> np.ndarray:
    """
    Build an n×n RBF correlation matrix over strike distance.

    grid_K_nodes : (n,) absolute dollar strike coordinates of the DLV² nodes.
    ell          : correlation length in the same units as grid_K_nodes.
    """
    diffs = grid_K_nodes[:, None] - grid_K_nodes[None, :]  # (n, n)
    C = np.exp(-0.5 * (diffs / ell) ** 2)
    np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)
    return C


def make_fresh_node_mask(fresh_anchor_mask: np.ndarray, stencil: int = 1) -> np.ndarray:
    """
    Convert a boolean mask over the N interior anchors to a boolean mask over
    the n = N-2 DLV² nodes.

    fresh_anchor_mask : (N,) bool — which of the N interior anchors (anchor_pure)
                        had a live market quote in this snapshot.
    stencil = 1 (permissive) : node j is fresh iff its own anchor (j+1) was fresh.
    stencil = 3 (conservative): node j is fresh iff anchors j, j+1, j+2 all fresh
                                (i.e. the full butterfly support was live).
    """
    if stencil == 1:
        return fresh_anchor_mask[1:-1].copy()
    elif stencil == 3:
        return fresh_anchor_mask[:-2] & fresh_anchor_mask[1:-1] & fresh_anchor_mask[2:]
    else:
        raise ValueError(f"stencil must be 1 or 3, got {stencil!r}")


def _ts_seconds(r: dict) -> float:
    ts = r["ts"]
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts) / 1e9


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------


def run_dlv_kalman_filter_v2(
    results: list[dict],
    N_dlv: int,
    grid_K_nodes: np.ndarray,
    cfg: KFConfig,
    x0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    """
    Stage 2 DLV² Kalman filter with correlated process noise and selection-H.

    Parameters
    ----------
    results       : list of snapshot dicts from run_intraday_loop.
                    Each dict must have:
                      "ts"               : timestamp
                      "failed"           : bool
                      "dlvsqs"           : (1, n) or None
                      "fresh_anchor_mask": (N,) bool or None
                    (fresh_anchor_mask is None → all nodes treated as fresh,
                     reproducing the legacy behaviour when no mask is tracked.)
    N_dlv         : int — state dimension n = N - 2.
    grid_K_nodes  : (n,) absolute strike coordinates of DLV² nodes (dlv_anchor_pure).
    cfg           : KFConfig
    x0            : (n,) optional initial state vector. If None, the first valid
                    snapshot's dlvsqs are used (original behaviour).

    Returns
    -------
    x_hist       : (T, n) posterior mean DLV²
    x_pred_hist  : (T, n) prior mean
    P_diag_hist  : (T, n) posterior covariance diagonal  (√ = posterior std)
    innov_hist   : (T, n) innovations at fresh nodes, NaN elsewhere
    ts_hist      : (T,)   POSIX seconds
    failed_hist  : (T,)   bool — snapshot fit failed
    n_fresh_hist : (T,)   int  — number of fresh nodes used each step
    """
    n = N_dlv
    T = len(results)
    I_n = np.eye(n)

    C = make_corr_matrix(grid_K_nodes, cfg.ell)
    if cfg.p0_matrix is not None:
        P_init = np.array(cfg.p0_matrix, dtype=float)
    else:
        P_init = cfg.p0_scale * (C if cfg.p0_correlated else I_n)

    # --- find initialisation snapshot: first with ≥1 fresh node ---
    idx0 = None
    for k, r in enumerate(results):
        if r["failed"] or r.get("dlvsqs") is None:
            continue
        mask = r.get("fresh_anchor_mask")
        if mask is None:
            idx0 = k
            break
        fn = make_fresh_node_mask(mask, cfg.stencil)
        if fn.sum() >= 1:
            idx0 = k
            break

    if idx0 is None:
        raise RuntimeError("No valid snapshot found to initialise the filter.")

    x = x0.copy() if x0 is not None else results[idx0]["dlvsqs"][0].copy()
    P = P_init.copy()
    t_prev = _ts_seconds(results[idx0])

    # --- output arrays ---
    x_hist = np.empty((T, n))
    x_pred_hist = np.empty((T, n))
    P_diag_hist = np.empty((T, n))
    innov_hist = np.full((T, n), np.nan)
    ts_hist = np.empty(T)
    failed_hist = np.empty(T, dtype=bool)
    n_fresh_hist = np.zeros(T, dtype=int)

    for k, r in enumerate(results):
        t_k = _ts_seconds(r)
        dt_s = max(t_k - t_prev, 0.0)
        dt_days = dt_s / cfg.trading_day_s

        # ── Predict ──────────────────────────────────────────────────────────
        Q_t = cfg.q_scale * dt_days * C
        x_pred = x.copy()
        P_pred = P + Q_t
        P_pred = 0.5 * (P_pred + P_pred.T)

        ts_hist[k] = t_k
        failed_hist[k] = r["failed"]

        # ── Determine observation set ─────────────────────────────────────────
        do_update = False
        fresh_node = None

        if not r["failed"] and r.get("dlvsqs") is not None:
            mask = r.get("fresh_anchor_mask")
            if mask is None:
                fresh_node = np.ones(n, dtype=bool)  # backward compat: all fresh
            else:
                fresh_node = make_fresh_node_mask(mask, cfg.stencil)

            m = int(fresh_node.sum())
            n_fresh_hist[k] = m

            # DLV_MAX filter: skip update if any observed node is at the ceiling
            z_full = r["dlvsqs"][0].copy()
            at_dlv_max = (
                cfg.ignore_at_dlv_max
                and cfg.dlv_max_sq_net is not None
                and bool(np.any(z_full[fresh_node] >= cfg.dlv_max_sq_net))
            )

            do_update = (m >= 1) and (not at_dlv_max)

        # ── Update or predict-only ────────────────────────────────────────────
        if do_update:
            z = z_full[fresh_node]  # (m,)
            y = z - x_pred[fresh_node]  # innovation (m,)

            # Exploit selection-H structure via index slicing
            P_sub = P_pred[np.ix_(fresh_node, fresh_node)]  # (m, m)
            P_cols = P_pred[:, fresh_node]  # (n, m)

            R_m = cfg.r_diag * np.eye(m)
            S = P_sub + R_m  # (m, m)
            K = P_cols @ np.linalg.inv(S)  # (n, m)

            x = x_pred + K @ y  # (n,)

            # Joseph form: (I - K H) P_pred (I - K H)ᵀ + K R Kᵀ
            # H is the selection matrix: H[j, i] = 1 iff fresh_node[i] is the j-th True
            fresh_idx = np.where(fresh_node)[0]  # (m,)
            KH = np.zeros((n, n))
            KH[:, fresh_idx] = K  # K @ H  (n, n)
            IKH = I_n - KH
            P = IKH @ P_pred @ IKH.T + K @ R_m @ K.T
            P = 0.5 * (P + P.T)

            if cfg.soft_floor is not None:
                x = np.maximum(x, cfg.soft_floor)

            innov_hist[k][fresh_node] = y
        else:
            x = x_pred
            P = P_pred

        x_hist[k] = x
        x_pred_hist[k] = x_pred
        P_diag_hist[k] = np.diag(P)
        t_prev = t_k

    return x_hist, x_pred_hist, P_diag_hist, innov_hist, ts_hist, failed_hist, n_fresh_hist


# ---------------------------------------------------------------------------
# Synthetic self-test  (spec §12)
# ---------------------------------------------------------------------------


def _synthetic_selftest(seed: int = 42, verbose: bool = False) -> None:
    """
    Verifies the cross-strike coupling mechanism on synthetic data.

    Three assertions:
    1. MECHANISM — with only ATM nodes fresh and wing not fresh, the correlated
       filter (RBF Q) continuously moves the wing estimate while the diagonal
       filter (C = I, same selection-H) leaves the wing frozen. Specifically,
       after 200 consecutive ATM-only steps starting from a common init, the
       cumulative absolute movement of wing in the correlated filter must be
       at least 10× the movement in the diagonal filter.
    2. BOUNDED STD — running the correlated filter for 2000 steps with wing
       observed only every 250 steps, the wing posterior std plateaus:
       max(std[500:]) < 3 × std[100].
    3. LEGACY EQUIVALENCE — with C = I (ell→0) and all nodes fresh, the new
       filter reproduces the old-style diagonal filter to float64 precision.
    """
    rng = np.random.default_rng(seed)
    n = 17
    T = 2000
    trading_day_s = 23400.0
    q_scale = 1e-4
    r_diag = 1e-4
    p0_scale = 1e-3

    # --- synthetic strike grid: $454 … $486, spacing $2 ---
    strikes = np.linspace(454.0, 454.0 + 2.0 * (n - 1), n)

    # --- shared observation sequence ---
    # x_truth: smooth sinusoidal ATM movement coupled to wing via ell=10 correlation
    ell = 10.0
    dt_days = 1.0 / trading_day_s
    sigma_step = np.sqrt(q_scale * dt_days)
    C_true = make_corr_matrix(strikes, ell)
    L_chol = np.linalg.cholesky(C_true + 1e-8 * np.eye(n))

    x_truth = np.empty((T, n))
    x_truth[0] = np.full(n, 0.04)
    for t in range(1, T):
        dw = sigma_step * L_chol @ rng.standard_normal(n)
        x_truth[t] = np.maximum(x_truth[t - 1] + dw, 0.0)

    # N = n + 2 anchor grid (n_pad = 1)
    N = n + 2
    obs_noise_std = np.sqrt(r_diag)

    # Use integer nanosecond timestamps so the unmodified _ts_seconds
    # returns the correct seconds value: float(t_ns) / 1e9 = t_s.
    NS_PER_S = int(1e9)

    import dlv_kalman_filter as _self  # for the legacy reference to _ts_seconds

    try:
        # ── assertion 1: mechanism test ────────────────────────────────────────
        # Use 200 consecutive ATM-only steps (wing never fresh).
        # Both filters are initialised identically from the same first observation.
        T1 = 201  # step 0 = init (all fresh); steps 1-200 = ATM only
        fa_all = np.ones(N, dtype=bool)
        fa_atm = np.zeros(N, dtype=bool)
        fa_atm[6:13] = True  # anchor indices 6-12 → DLV nodes 5-11

        results_mech = []
        for t in range(T1):
            z = x_truth[t] + obs_noise_std * rng.standard_normal(n)
            fa = fa_all if t == 0 else fa_atm
            results_mech.append(
                {
                    "ts": t * NS_PER_S,
                    "failed": False,
                    "dlvsqs": z[np.newaxis, :],
                    "fresh_anchor_mask": fa,
                }
            )

        cfg_corr1 = KFConfig(
            q_scale=q_scale,
            ell=ell,
            r_diag=r_diag,
            p0_scale=p0_scale,
            p0_correlated=True,
            stencil=1,
            soft_floor=None,
            trading_day_s=trading_day_s,
            ignore_at_dlv_max=False,
        )
        cfg_diag1 = KFConfig(
            q_scale=q_scale,
            ell=1e-6,
            r_diag=r_diag,  # C ≈ I
            p0_scale=p0_scale,
            p0_correlated=False,
            stencil=1,
            soft_floor=None,
            trading_day_s=trading_day_s,
            ignore_at_dlv_max=False,
        )

        x_c1, _, _, _, _, _, _ = run_dlv_kalman_filter_v2(
            results_mech, N_dlv=n, grid_K_nodes=strikes, cfg=cfg_corr1
        )
        x_d1, _, _, _, _, _, _ = run_dlv_kalman_filter_v2(
            results_mech, N_dlv=n, grid_K_nodes=strikes, cfg=cfg_diag1
        )

        wing_idx = 1
        # cumulative absolute movement of wing posterior mean (steps 1-200)
        move_corr = float(np.sum(np.abs(np.diff(x_c1[1:, wing_idx]))))
        move_diag = float(np.sum(np.abs(np.diff(x_d1[1:, wing_idx]))))

        assert move_corr > 10.0 * max(move_diag, 1e-20), (
            f"Assertion 1 FAILED: correlated wing movement={move_corr:.4e} is not "
            f"10× the diagonal movement={move_diag:.4e} over 200 ATM-only steps"
        )

        # ── assertion 2: posterior std bounded ────────────────────────────────
        # Long run (T steps) with wing only fresh every 250 steps.
        fresh_masks_long = []
        for t in range(T):
            fa = np.zeros(N, dtype=bool)
            fa[6:13] = True
            if t % 250 == 0:
                fa[2] = True
            fresh_masks_long.append(fa)

        results_long = []
        for t in range(T):
            z = x_truth[t] + obs_noise_std * rng.standard_normal(n)
            results_long.append(
                {
                    "ts": t * NS_PER_S,
                    "failed": False,
                    "dlvsqs": z[np.newaxis, :],
                    "fresh_anchor_mask": fresh_masks_long[t],
                }
            )

        cfg_long = KFConfig(
            q_scale=q_scale,
            ell=ell,
            r_diag=r_diag,
            p0_scale=p0_scale,
            p0_correlated=True,
            stencil=1,
            soft_floor=None,
            trading_day_s=trading_day_s,
            ignore_at_dlv_max=False,
        )
        _, _, P_diag_long, _, _, _, _ = run_dlv_kalman_filter_v2(
            results_long, N_dlv=n, grid_K_nodes=strikes, cfg=cfg_long
        )

        std_wing = np.sqrt(np.maximum(P_diag_long[:, wing_idx], 0.0))
        std_at_100 = float(std_wing[100])
        std_max_late = float(std_wing[500:].max())

        assert std_max_late < 3.0 * std_at_100, (
            f"Assertion 2 FAILED: posterior std diverges at wing: "
            f"max(std[500:])={std_max_late:.4e} >= 3×std[100]={std_at_100:.4e}"
        )

        # ── assertion 3: C = I + all-fresh reproduces legacy diagonal filter ──
        results_all_fresh = []
        for t in range(T):
            z = x_truth[t] + obs_noise_std * rng.standard_normal(n)
            results_all_fresh.append(
                {
                    "ts": t * NS_PER_S,
                    "failed": False,
                    "dlvsqs": z[np.newaxis, :],
                    "fresh_anchor_mask": None,  # None → all fresh
                }
            )

        cfg_leg_v2 = KFConfig(
            q_scale=q_scale,
            ell=1e-6,
            r_diag=r_diag,
            p0_scale=p0_scale,
            p0_correlated=False,
            stencil=1,
            soft_floor=None,
            trading_day_s=trading_day_s,
            ignore_at_dlv_max=False,
        )
        x_v2_id, _, P_v2_id, _, _, _, _ = run_dlv_kalman_filter_v2(
            results_all_fresh, N_dlv=n, grid_K_nodes=strikes, cfg=cfg_leg_v2
        )

        # Legacy diagonal filter (inline reference implementation, H = I)
        I_leg = np.eye(n)
        R_leg = r_diag * I_leg
        x_leg = results_all_fresh[0]["dlvsqs"][0].copy()
        P_leg = p0_scale * I_leg
        t_prev_leg = _self._ts_seconds(results_all_fresh[0])
        x_hist_leg = np.empty((T, n))
        P_diag_leg = np.empty((T, n))
        for k, r in enumerate(results_all_fresh):
            t_k = _self._ts_seconds(r)
            dt_d = max(t_k - t_prev_leg, 0.0) / trading_day_s
            Q_k = q_scale * dt_d * I_leg
            xp = x_leg.copy()
            Pp = P_leg + Q_k
            z = r["dlvsqs"][0].copy()
            y = z - xp
            S = Pp + R_leg
            K_g = Pp @ np.linalg.inv(S)
            x_leg = xp + K_g @ y
            IKH = I_leg - K_g
            P_leg = IKH @ Pp @ IKH.T + K_g @ R_leg @ K_g.T
            P_leg = 0.5 * (P_leg + P_leg.T)
            x_hist_leg[k] = x_leg
            P_diag_leg[k] = np.diag(P_leg)
            t_prev_leg = t_k

        assert np.allclose(x_v2_id, x_hist_leg, atol=1e-12), (
            "Assertion 3 FAILED: v2 with C≈I does not match legacy diagonal filter "
            f"(max |Δx|={np.abs(x_v2_id - x_hist_leg).max():.2e})"
        )
        assert np.allclose(P_v2_id, P_diag_leg, atol=1e-12), (
            "Assertion 3 FAILED: P_diag v2 with C≈I does not match legacy "
            f"(max |ΔP|={np.abs(P_v2_id - P_diag_leg).max():.2e})"
        )

    finally:
        pass  # no cleanup needed; no patching was done

    if verbose:
        print("Self-test PASSED")
        print(
            f"  Wing movement corr={move_corr:.4e}  diag={move_diag:.4e}  ratio={move_corr / max(move_diag, 1e-20):.1f}×"
        )
        print(
            f"  Wing std bounded: max(std[500:])={std_max_late:.4e}  3×std[100]={3 * std_at_100:.4e}"
        )
        print(f"  C≈I vs legacy: max |Δx|={np.abs(x_v2_id - x_hist_leg).max():.2e}")


if __name__ == "__main__":
    _synthetic_selftest(verbose=True)
