"""CRC reset: deterministic state step, target-surface freeze, tangent-
parameter update, and the restricted Hull-White inverse calibration
(Task 1B Sec. 9-12).

Follows Gambara & Teichmann (2020), "Consistent Recalibration Models and
Deep Calibration" (arXiv:2006.09455), Algorithm 1, replacing their NN1/NN2
networks with the exact Fourier pricer (generalized_bates.py) and a
deterministic constrained least-squares inverse: "First establish the exact
pricer and a high-accuracy deterministic inverse" (Task 1B Sec. objective).

Does not modify oracle.py or generalized_bates.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, brentq

import oracle as orc
import generalized_bates as gb

# ===========================================================================
# Sec. 8: calibration grid (10 maturities x 13 moneyness) -- a SEPARATE
# object from the protected DLV grid (T=0.25,0.5,1,2; K=60..160).
# Moneyness convention: m = K/F, F = S (r=q=0 so the forward equals spot).
# Cash strikes are derived at the RESET spot and held fixed thereafter.
# ===========================================================================

MONEYNESS_GRID: tuple[float, ...] = tuple(np.linspace(0.8, 1.2, 13))
CALIBRATION_TAUS: tuple[float, ...] = gb.CRC_CALIBRATION_MATURITIES  # 10 maturities, years

PROTECTED_STRIKES: tuple[float, ...] = orc.FIXED_STRIKES
PROTECTED_EXPIRIES: tuple[float, ...] = orc.FIXED_EXPIRIES


def calibration_strikes(S: float) -> np.ndarray:
    """Convert the moneyness grid to cash strikes at spot S (Sec. 8: "convert
    the grid to cash strikes at the reset time and then hold those cash
    contracts fixed throughout the reset comparison")."""
    return S * np.asarray(MONEYNESS_GRID)


# ===========================================================================
# Grid pricing / Black-Scholes IV and vega (self-contained, mirrors
# oracle.black_scholes_call's erf-based normal CDF).
# ===========================================================================


def price_grid(t: float, S: float, V: float, taus: np.ndarray, Ks: np.ndarray,
                heston: orc.HestonParams, extension: gb.HullWhiteJumpExtension) -> np.ndarray:
    """C[j,k] at (T_j = t + taus[j], K_k = Ks[k]), observed from t."""
    out = np.empty((len(taus), len(Ks)))
    for j, tau in enumerate(taus):
        T = t + tau
        for k, K in enumerate(Ks):
            out[j, k] = float(gb.generalized_bates_call_price(t, S, V, T, K, heston, extension))
    return out


def _bs_call(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * tau) / sv
    d2 = d1 - sv
    return S * orc._norm_cdf(d1) - K * orc._norm_cdf(d2)


def _bs_vega(S: float, K: float, tau: float, sigma: float) -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return 0.0
    sv = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * tau) / sv
    return S * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * math.sqrt(tau)


def implied_vol(C: float, S: float, K: float, tau: float, lo: float = 1.0e-6, hi: float = 5.0) -> float:
    intrinsic = max(S - K, 0.0)
    if tau <= 0.0 or not math.isfinite(C) or C <= intrinsic + 1.0e-12 or C >= S - 1.0e-12:
        return float("nan")
    f = lambda sig: _bs_call(S, K, tau, sig) - C
    try:
        return brentq(f, lo, hi, xtol=1.0e-10)
    except ValueError:
        return float("nan")


def implied_vol_grid(C: np.ndarray, S: float, taus: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    out = np.full_like(C, np.nan)
    for j, tau in enumerate(taus):
        for k, K in enumerate(Ks):
            out[j, k] = implied_vol(C[j, k], S, K, tau)
    return out


def vega_grid(S: float, taus: np.ndarray, Ks: np.ndarray, IV: np.ndarray) -> np.ndarray:
    out = np.zeros_like(IV)
    for j, tau in enumerate(taus):
        for k, K in enumerate(Ks):
            sig = IV[j, k]
            out[j, k] = _bs_vega(S, K, tau, sig) if math.isfinite(sig) else 0.0
    return out


# ===========================================================================
# Sec. 9 Step A: deterministic state evolution.
# ===========================================================================


@dataclass(frozen=True)
class StateStep:
    t_pre: float
    t_post: float
    dt: float
    S_pre: float
    V_pre: float
    S_post: float
    V_post: float
    method: str


def deterministic_conditional_mean_state_step(
    t_pre: float, S_pre: float, V_pre: float, dt: float, heston: orc.HestonParams,
) -> StateStep:
    """Advance (X, V) by dt under the OLD model's Heston diffusion via
    CONDITIONAL MEANS (Sec. 9 Step A: "use ... the conditional mean state" --
    the declared choice recorded here, deterministically, no RNG).

    V^+ = theta + (V^- - theta) exp(-kappa dt)                [exact CIR mean]
    X^+ = X^- - (1/2) E[int_t^{t+dt} V_s ds]
        = X^- - (1/2)( theta dt + (V^--theta)(1-exp(-kappa dt))/kappa )

    Jumps contribute exactly zero to E[X_{t+dt}] by construction of the
    compensator (E[compensated jump increment] = 0 for ANY admissible bucket
    law), so this formula does not depend on the jump extension.
    """
    kappa, theta = heston.kappa, heston.theta
    exp_kdt = math.exp(-kappa * dt)
    V_post = theta + (V_pre - theta) * exp_kdt
    mean_integral_V = theta * dt + (V_pre - theta) * (1.0 - exp_kdt) / kappa
    X_post = math.log(S_pre) - 0.5 * mean_integral_V
    S_post = math.exp(X_post)
    return StateStep(
        t_pre=t_pre, t_post=t_pre + dt, dt=dt, S_pre=S_pre, V_pre=V_pre,
        S_post=S_post, V_post=V_post, method="conditional_mean",
    )


# ===========================================================================
# Sec. 9 Step C: deterministic tangent-parameter update.
# ===========================================================================


@dataclass(frozen=True)
class TangentUpdate:
    heston_pre: orc.HestonParams
    heston_post: orc.HestonParams
    relative_changes: dict


def deterministic_tangent_update(
    heston_pre: orc.HestonParams,
    d_theta_rel: float = 0.05,
    d_xi_rel: float = -0.05,
    d_rho_rel: float = -0.05,
) -> TangentUpdate:
    """Sec. 9 Step C: deterministic small update; kappa (and lambda,
    separately, at the caller) held fixed; relative changes capped at 5%,
    matching the paper's numerical experiment; admissibility enforced."""
    for name, rel in [("theta", d_theta_rel), ("xi", d_xi_rel), ("rho", d_rho_rel)]:
        if abs(rel) > 0.05 + 1.0e-12:
            raise ValueError(f"{name} relative change {rel} exceeds the +-5% cap")
    theta_post = heston_pre.theta * (1.0 + d_theta_rel)
    xi_post = heston_pre.xi * (1.0 + d_xi_rel)
    rho_post = heston_pre.rho * (1.0 + d_rho_rel)
    if not (theta_post > 0.0 and xi_post > 0.0 and -1.0 < rho_post < 1.0):
        raise ValueError("post-reset tangent parameters violate basic Heston admissibility")
    if not (2.0 * heston_pre.kappa * theta_post > xi_post ** 2):
        raise ValueError("post-reset tangent parameters violate Feller admissibility (2*kappa*theta > xi^2)")
    heston_post = orc.HestonParams(kappa=heston_pre.kappa, theta=theta_post, xi=xi_post, rho=rho_post)
    return TangentUpdate(
        heston_pre=heston_pre, heston_post=heston_post,
        relative_changes=dict(theta=d_theta_rel, xi=d_xi_rel, rho=d_rho_rel),
    )


# ===========================================================================
# Sec. 9 Step D: inverse Hull-White calibration.
#
# Parameterisation: x = (nu_1..5, log_delta_1..5) in R^10 -- delta_b =
# exp(log_delta_b) is structurally positive (never repaired after the fact).
# Explicit, documented bounds: nu_b in [-1.0, 0.5], delta_b in [0.01, 1.0].
# Regularisation: Tikhonov toward a^- in x-space (gamma * (x - x0_pre)),
# consistent with the spec's gamma*R(a+,a-) term.
# ===========================================================================

NU_BOUNDS = (-1.0, 0.5)
LOG_DELTA_BOUNDS = (math.log(0.01), math.log(1.0))


def _extension_params_to_x(extension: gb.HullWhiteJumpExtension) -> np.ndarray:
    return np.concatenate([np.asarray(extension.nu, dtype=float), np.log(np.asarray(extension.delta, dtype=float))])


def _x_to_extension(x: np.ndarray, lambda_: float, bucket_edges: tuple[float, ...]) -> gb.HullWhiteJumpExtension:
    nu = tuple(float(v) for v in x[:5])
    delta = tuple(float(v) for v in np.exp(x[5:]))
    return gb.HullWhiteJumpExtension(lambda_=lambda_, bucket_edges=bucket_edges, nu=nu, delta=delta)


@dataclass(frozen=True)
class CRCInverseResult:
    extension_pre: gb.HullWhiteJumpExtension
    extension_post: gb.HullWhiteJumpExtension
    x0_multistart: list
    x_opt: np.ndarray
    best_start_index: int
    success: bool
    cost: float
    n_calls_total: int
    gamma: float
    optimizer_max_nfev: int
    calib_strikes: np.ndarray
    calib_taus: np.ndarray
    C_target: np.ndarray
    IV_target: np.ndarray
    C_post: np.ndarray
    IV_post: np.ndarray
    vega_weights: np.ndarray
    fit_mask: np.ndarray
    price_residual: np.ndarray  # (C_post - C_target), full grid, nan where masked out
    iv_residual_bp: np.ndarray  # 1e4*(IV_post - IV_target), full grid


def solve_crc_inverse(
    state_post: StateStep,
    heston_post: orc.HestonParams,
    extension_pre: gb.HullWhiteJumpExtension,
    C_target: np.ndarray,
    calib_strikes: np.ndarray,
    calib_taus: np.ndarray,
    gamma: float = 1.0e-3,
    max_nfev: int = 3000,
    n_multistart: int = 3,
) -> CRCInverseResult:
    """Solve Sec. 9 Step D's inverse problem: recalibrate the jump extension
    so that the post-reset (tangent-updated) model reproduces C_target.

    Deterministic multistart (Sec. 9: "deterministic multistart constrained
    optimiser") -- NO RNG: the multistart perturbations are fixed, declared
    offsets from a^- (x0_multistart[0] = a^- unperturbed, then small
    deterministic nudges), and the best (lowest final cost) is kept.
    """
    S, V, t = state_post.S_post, state_post.V_post, state_post.t_post
    lambda_ = extension_pre.lambda_
    edges = extension_pre.bucket_edges

    IV_target = implied_vol_grid(C_target, S, calib_taus, calib_strikes)
    vega = vega_grid(S, calib_taus, calib_strikes, IV_target)
    intrinsic = np.maximum(S - calib_strikes, 0.0)[None, :]
    fit_mask = np.isfinite(IV_target) & (C_target - intrinsic > 1.0e-4) & (S - C_target > 1.0e-4) & (vega > 1.0e-8)

    x_pre = _extension_params_to_x(extension_pre)
    lb = np.array([NU_BOUNDS[0]] * 5 + [LOG_DELTA_BOUNDS[0]] * 5)
    ub = np.array([NU_BOUNDS[1]] * 5 + [LOG_DELTA_BOUNDS[1]] * 5)

    def residual_fn(x):
        ext = _x_to_extension(x, lambda_, edges)
        C = price_grid(t, S, V, calib_taus, calib_strikes, heston_post, ext)
        price_resid = ((C - C_target) / np.maximum(vega, 1.0e-3))[fit_mask]
        reg_resid = math.sqrt(gamma) * (x - x_pre)
        return np.concatenate([price_resid, reg_resid])

    # Deterministic multistart: unperturbed a^-, then two fixed, declared nudges.
    starts = [
        x_pre.copy(),
        np.clip(x_pre + np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1]), lb, ub),
        np.clip(x_pre - np.array([0.02, 0.02, 0.02, 0.02, 0.02, -0.1, -0.1, -0.1, -0.1, -0.1]), lb, ub),
    ][:n_multistart]

    best = None
    best_idx = 0
    n_calls_total = 0
    for i, x0 in enumerate(starts):
        fit = least_squares(
            residual_fn, x0, bounds=(lb, ub), xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13, max_nfev=max_nfev,
        )
        n_calls_total += fit.nfev
        if best is None or fit.cost < best.cost:
            best = fit
            best_idx = i

    x_opt = best.x
    extension_post = _x_to_extension(x_opt, lambda_, edges)
    C_post = price_grid(t, S, V, calib_taus, calib_strikes, heston_post, extension_post)
    IV_post = implied_vol_grid(C_post, S, calib_taus, calib_strikes)

    return CRCInverseResult(
        extension_pre=extension_pre, extension_post=extension_post,
        x0_multistart=starts, x_opt=x_opt, best_start_index=best_idx,
        success=bool(best.success), cost=float(best.cost), n_calls_total=n_calls_total,
        gamma=gamma, optimizer_max_nfev=max_nfev,
        calib_strikes=calib_strikes, calib_taus=calib_taus,
        C_target=C_target, IV_target=IV_target, C_post=C_post, IV_post=IV_post,
        vega_weights=vega, fit_mask=fit_mask,
        price_residual=np.where(fit_mask, C_post - C_target, np.nan),
        iv_residual_bp=np.where(fit_mask, 1.0e4 * (IV_post - IV_target), np.nan),
    )


# ===========================================================================
# Sec. 10: exact-theoretical-null vs restricted-numerical-approximation
# labelling. Never call an approximate reset an exact theoretical CRC null.
# ===========================================================================

HIGH_ACCURACY_RMS_IV_BP = 1.0
HIGH_ACCURACY_MAX_IV_BP = 5.0


def label_crc_result(rms_iv_bp: float, max_iv_bp: float) -> str:
    if math.isfinite(rms_iv_bp) and math.isfinite(max_iv_bp) and rms_iv_bp <= HIGH_ACCURACY_RMS_IV_BP and max_iv_bp <= HIGH_ACCURACY_MAX_IV_BP:
        return "restricted_crc_high_accuracy"
    return "restricted_crc_approximation"


# ===========================================================================
# Withheld dense strike/maturity grid: NEVER used for calibration, evaluation
# only (Sec. 11: "The withheld dense grid must not be used in calibration").
# Kept moderate in size (9x6=54 points) since each point costs an
# un-jitted Lewis-integral price call (~2-3ms); this is an evaluation-only
# diagnostic, not the full spec-resolution dense mesh used in Task 1.
# ===========================================================================

WITHHELD_STRIKES: tuple[float, ...] = tuple(float(k) for k in np.arange(67.5, 153.0, 10.0))
WITHHELD_TAUS: tuple[float, ...] = (0.35, 0.6, 0.85, 1.1, 1.4, 1.7)


# ===========================================================================
# Sec. 11: reset diagnostics.
# ===========================================================================


def _surface_error_stats(C_post: np.ndarray, C_target: np.ndarray, IV_post: np.ndarray, IV_target: np.ndarray,
                          S: float, mask: np.ndarray) -> dict:
    dC = (C_post - C_target)[mask]
    dIV = (IV_post - IV_target)
    iv_mask = mask & np.isfinite(IV_post) & np.isfinite(IV_target)
    dIV_masked = dIV[iv_mask]
    return dict(
        e_C_max=float(np.max(np.abs(dC))) if dC.size else float("nan"),
        rms_price_error=float(np.sqrt(np.mean(dC ** 2))) if dC.size else float("nan"),
        spot_normalized_price_error=float(np.max(np.abs(dC)) / S) if dC.size else float("nan"),
        max_iv_error_bp=float(1.0e4 * np.max(np.abs(dIV_masked))) if dIV_masked.size else float("nan"),
        rms_iv_error_bp=float(1.0e4 * np.sqrt(np.mean(dIV_masked ** 2))) if dIV_masked.size else float("nan"),
        n_points=int(mask.sum()), n_iv_points=int(iv_mask.sum()),
    )


def surface_preservation_diagnostics(
    state_post: StateStep, heston_pre: orc.HestonParams, extension_pre: gb.HullWhiteJumpExtension,
    heston_post: orc.HestonParams, extension_post: gb.HullWhiteJumpExtension,
) -> dict:
    """Sec. 11 "Surface preservation": errors on the calibration grid (handled
    by CRCInverseResult itself), the protected DLV grid, and a withheld dense
    grid never used in calibration."""
    S, V, t = state_post.S_post, state_post.V_post, state_post.t_post

    protected_taus = np.array([T - t for T in PROTECTED_EXPIRIES if T - t > 0.0])
    protected_Ks = np.asarray(PROTECTED_STRIKES)
    C_target_prot = price_grid(t, S, V, protected_taus, protected_Ks, heston_pre, extension_pre)
    C_post_prot = price_grid(t, S, V, protected_taus, protected_Ks, heston_post, extension_post)
    IV_target_prot = implied_vol_grid(C_target_prot, S, protected_taus, protected_Ks)
    IV_post_prot = implied_vol_grid(C_post_prot, S, protected_taus, protected_Ks)
    intrinsic_prot = np.maximum(S - protected_Ks, 0.0)[None, :]
    mask_prot = (C_target_prot - intrinsic_prot > 1.0e-6) & (S - C_target_prot > 1.0e-6)
    protected = _surface_error_stats(C_post_prot, C_target_prot, IV_post_prot, IV_target_prot, S, mask_prot)

    withheld_taus = np.asarray(WITHHELD_TAUS)
    withheld_Ks = np.asarray(WITHHELD_STRIKES)
    C_target_wh = price_grid(t, S, V, withheld_taus, withheld_Ks, heston_pre, extension_pre)
    C_post_wh = price_grid(t, S, V, withheld_taus, withheld_Ks, heston_post, extension_post)
    IV_target_wh = implied_vol_grid(C_target_wh, S, withheld_taus, withheld_Ks)
    IV_post_wh = implied_vol_grid(C_post_wh, S, withheld_taus, withheld_Ks)
    intrinsic_wh = np.maximum(S - withheld_Ks, 0.0)[None, :]
    mask_wh = (C_target_wh - intrinsic_wh > 1.0e-6) & (S - C_target_wh > 1.0e-6)
    withheld = _surface_error_stats(C_post_wh, C_target_wh, IV_post_wh, IV_target_wh, S, mask_wh)

    return dict(protected_grid=protected, withheld_dense_grid=withheld)


def dlv_state_continuity(
    state_post: StateStep, heston_pre: orc.HestonParams, extension_pre: gb.HullWhiteJumpExtension,
    heston_post: orc.HestonParams, extension_post: gb.HullWhiteJumpExtension,
) -> dict:
    """Sec. 11 "DLV-state continuity": ||q+-q-||, ||ell+-ell-|| on the SAME
    protected finite grid, both surfaces evaluated at the post-reset state.
    An exact surface-preserving reset should leave (q, ell) unchanged."""
    S, V, t = state_post.S_post, state_post.V_post, state_post.t_post
    q_pre_vals, q_post_vals, ell_pre_vals, ell_post_vals = [], [], [], []
    for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES:
        for m_idx in orc.FIXED_BACK_SLAB_INDICES:
            K_minus, K_n, K_plus = PROTECTED_STRIKES[n_idx - 1], PROTECTED_STRIKES[n_idx], PROTECTED_STRIKES[n_idx + 1]
            T_prev, T_curr = PROTECTED_EXPIRIES[m_idx - 1], PROTECTED_EXPIRIES[m_idx]
            if T_curr - t <= 0.0:
                continue
            ell_pre, q_pre = gb.make_generalized_finite_dlv_field(K_minus, K_n, K_plus, T_prev, T_curr, heston_pre, extension_pre)
            ell_post, q_post = gb.make_generalized_finite_dlv_field(K_minus, K_n, K_plus, T_prev, T_curr, heston_post, extension_post)
            qv_pre = float(q_pre(t, S, V))
            qv_post = float(q_post(t, S, V))
            if qv_pre <= orc.Q_MIN_DEFAULT or qv_post <= orc.Q_MIN_DEFAULT:
                continue
            q_pre_vals.append(qv_pre); q_post_vals.append(qv_post)
            ell_pre_vals.append(float(ell_pre(t, S, V))); ell_post_vals.append(float(ell_post(t, S, V)))
    q_pre_arr, q_post_arr = np.array(q_pre_vals), np.array(q_post_vals)
    ell_pre_arr, ell_post_arr = np.array(ell_pre_vals), np.array(ell_post_vals)
    return dict(
        n_nodes=len(q_pre_arr),
        q_l2_diff=float(np.linalg.norm(q_post_arr - q_pre_arr)),
        q_max_abs_diff=float(np.max(np.abs(q_post_arr - q_pre_arr))) if len(q_pre_arr) else float("nan"),
        ell_l2_diff=float(np.linalg.norm(ell_post_arr - ell_pre_arr)),
        ell_max_abs_diff=float(np.max(np.abs(ell_post_arr - ell_pre_arr))) if len(ell_pre_arr) else float("nan"),
        q_l2_norm_pre=float(np.linalg.norm(q_pre_arr)),
        ell_l2_norm_pre=float(np.linalg.norm(ell_pre_arr)),
    )


def characteristic_change_report(tu: TangentUpdate, extension_pre: gb.HullWhiteJumpExtension,
                                  extension_post: gb.HullWhiteJumpExtension, V_post: float) -> dict:
    """Sec. 11 "Characteristic changes": Heston tangent parameters, jump-bucket
    parameters, Brownian loadings (sigma_S=sqrt(V), sigma_V=xi*sqrt(V), rho),
    instantaneous jump compensator kappa_b^J per bucket."""
    hp, hq = tu.heston_pre, tu.heston_post
    sv = math.sqrt(max(V_post, 0.0))
    return dict(
        heston_pre=dict(kappa=hp.kappa, theta=hp.theta, xi=hp.xi, rho=hp.rho),
        heston_post=dict(kappa=hq.kappa, theta=hq.theta, xi=hq.xi, rho=hq.rho),
        relative_changes=tu.relative_changes,
        brownian_loadings_pre=dict(sigma_S=sv, sigma_V=hp.xi * sv, rho=hp.rho),
        brownian_loadings_post=dict(sigma_S=sv, sigma_V=hq.xi * sv, rho=hq.rho),
        jump_bucket_edges=list(extension_pre.bucket_edges),
        nu_pre=list(extension_pre.nu), delta_pre=list(extension_pre.delta),
        nu_post=list(extension_post.nu), delta_post=list(extension_post.delta),
        kappa_J_pre=list(extension_pre.kappa_J), kappa_J_post=list(extension_post.kappa_J),
        lambda_fixed=extension_pre.lambda_,
    )


def post_reset_dlv_residual_report(
    state_post: StateStep, heston_post: orc.HestonParams, extension_post: gb.HullWhiteJumpExtension,
) -> dict:
    """Sec. 11 "Post-reset no-arbitrage residual": the full jump-corrected
    DLV residual using the NEW characteristics, evaluated forward from the
    post-reset state/time, split into same-bucket (theorem premise holds)
    and cross-bucket (structural mismatch, see generalized_bates.py) nodes,
    and separately reporting R_nojump (jump term deliberately omitted)."""
    S, V, t = state_post.S_post, state_post.V_post, state_post.t_post
    same_R, cross_R, nojump_R = [], [], []
    for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES:
        for m_idx in orc.FIXED_BACK_SLAB_INDICES:
            T_curr = PROTECTED_EXPIRIES[m_idx]
            if T_curr - t <= 0.0:
                continue
            r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, t, S, V, heston_post, extension_post)
            if r.excluded_near_boundary or not (r.q > orc.Q_MIN_DEFAULT):
                continue
            if r.cross_bucket_slab:
                cross_R.append(abs(r.R))
            else:
                same_R.append(abs(r.R))
                nojump_R.append(abs(r.R_nojump))
    same_R, cross_R, nojump_R = np.array(same_R), np.array(cross_R), np.array(nojump_R)
    return dict(
        same_bucket_max_R=float(same_R.max()) if len(same_R) else float("nan"),
        same_bucket_median_R=float(np.median(same_R)) if len(same_R) else float("nan"),
        same_bucket_n=len(same_R),
        cross_bucket_max_R=float(cross_R.max()) if len(cross_R) else float("nan"),
        cross_bucket_median_R=float(np.median(cross_R)) if len(cross_R) else float("nan"),
        cross_bucket_n=len(cross_R),
        same_bucket_nojump_median_R=float(np.median(nojump_R)) if len(nojump_R) else float("nan"),
        jump_term_essential_ratio=float(np.median(nojump_R) / max(np.median(same_R), 1e-300)) if len(same_R) else float("nan"),
    )


# ===========================================================================
# Sec. 12: positive controls. Each should produce a DETECTABLE surface jump
# or generator residual -- verifying that the diagnostics above have power,
# not just that the "real" reset happens to look small.
# ===========================================================================


def control_naive_no_recalibration(
    state_post: StateStep, heston_pre: orc.HestonParams, extension_pre: gb.HullWhiteJumpExtension,
    heston_post: orc.HestonParams,
) -> dict:
    """Control 1 (Sec. 12 item 1): change the Heston tangent parameters
    WITHOUT recalibrating the jump extension (extension_post = extension_pre
    unchanged). Should show a much larger surface residual than the
    recalibrated (Step D) result."""
    return surface_preservation_diagnostics(state_post, heston_pre, extension_pre, heston_post, extension_pre)


def control_too_few_buckets(
    state_post: StateStep, heston_post: orc.HestonParams, extension_pre: gb.HullWhiteJumpExtension,
    C_target: np.ndarray, calib_strikes: np.ndarray, calib_taus: np.ndarray,
    gamma: float = 1.0e-3, max_nfev: int = 200,
) -> dict:
    """Control 2 (Sec. 12 item 2): calibrate with only 2 effective degrees of
    freedom (near-term buckets {0,1} share one (nu,delta), far-term buckets
    {2,3,4} share another) instead of the full 5. Should fit worse than the
    unrestricted 5-bucket inverse."""
    S, V, t = state_post.S_post, state_post.V_post, state_post.t_post
    lambda_, edges = extension_pre.lambda_, extension_pre.bucket_edges
    x_pre_full = _extension_params_to_x(extension_pre)
    x0 = np.array([np.mean(x_pre_full[[0, 1]]), np.mean(x_pre_full[[2, 3, 4]]),
                    np.mean(x_pre_full[[5, 6]]), np.mean(x_pre_full[[7, 8, 9]])])
    lb = np.array([NU_BOUNDS[0], NU_BOUNDS[0], LOG_DELTA_BOUNDS[0], LOG_DELTA_BOUNDS[0]])
    ub = np.array([NU_BOUNDS[1], NU_BOUNDS[1], LOG_DELTA_BOUNDS[1], LOG_DELTA_BOUNDS[1]])
    IV_target = implied_vol_grid(C_target, S, calib_taus, calib_strikes)
    vega = vega_grid(S, calib_taus, calib_strikes, IV_target)
    intrinsic = np.maximum(S - calib_strikes, 0.0)[None, :]
    fit_mask = np.isfinite(IV_target) & (C_target - intrinsic > 1.0e-4) & (S - C_target > 1.0e-4) & (vega > 1.0e-8)

    def x2_to_extension(x2):
        nu = (x2[0], x2[0], x2[1], x2[1], x2[1])
        delta = tuple(np.exp([x2[2], x2[2], x2[3], x2[3], x2[3]]))
        return gb.HullWhiteJumpExtension(lambda_=lambda_, bucket_edges=edges, nu=nu, delta=delta)

    def residual_fn(x2):
        ext = x2_to_extension(x2)
        C = price_grid(t, S, V, calib_taus, calib_strikes, heston_post, ext)
        return ((C - C_target) / np.maximum(vega, 1.0e-3))[fit_mask]

    fit = least_squares(residual_fn, x0, bounds=(lb, ub), xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12, max_nfev=max_nfev)
    ext_post = x2_to_extension(fit.x)
    C_post = price_grid(t, S, V, calib_taus, calib_strikes, heston_post, ext_post)
    IV_post = implied_vol_grid(C_post, S, calib_taus, calib_strikes)
    stats = _surface_error_stats(C_post, C_target, IV_post, IV_target, S, fit_mask)
    stats["success"] = bool(fit.success)
    stats["n_free_params"] = 4
    return stats


def control_stopped_early(
    state_post: StateStep, heston_post: orc.HestonParams, extension_pre: gb.HullWhiteJumpExtension,
    C_target: np.ndarray, calib_strikes: np.ndarray, calib_taus: np.ndarray,
) -> CRCInverseResult:
    """Control 3 (Sec. 12 item 3): deliberately stop the optimiser after a
    tiny function-evaluation budget (1 call). Should fit worse than the full
    solve (verified: rms IV error 29.5bp at max_nfev=1 vs 17.8bp converged,
    and clearly better than the zero-recalibration control (40.7bp) --
    max_nfev=2 already recovers most of the gap, so max_nfev=1 is used to
    keep this control meaningfully distinct)."""
    return solve_crc_inverse(
        state_post, heston_post, extension_pre, C_target, calib_strikes, calib_taus,
        gamma=1.0e-3, max_nfev=1, n_multistart=1,
    )


def control_jump_term_omitted(state_post: StateStep, heston_post: orc.HestonParams,
                               extension_post: gb.HullWhiteJumpExtension) -> dict:
    """Control 4 (Sec. 12 item 4): the DLV residual with the jump cross-term
    deliberately omitted -- R_nojump, already computed by
    generalized_finite_grid_node_residual/post_reset_dlv_residual_report.
    Exposed here as its own named control for the report."""
    report = post_reset_dlv_residual_report(state_post, heston_post, extension_post)
    return dict(
        same_bucket_median_R_full=report["same_bucket_median_R"],
        same_bucket_median_R_nojump=report["same_bucket_nojump_median_R"],
        ratio=report["jump_term_essential_ratio"],
    )
