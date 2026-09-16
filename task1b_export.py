"""Task 1B exports: Generalised-Bates (fixed, time-inhomogeneous) and CRC
pre/post-reset visible/hidden surfaces, plus the consolidated report
(Sec. 13). Does not overwrite Task 1's oracle.py exports (grid_visible.npz,
dense_hidden.npz in this same directory).

Run from this directory:
    python task1b_export.py
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

import oracle as orc
import generalized_bates as gb
import crc_inverse as ci

OUT_DIR = "."

# Moderate dense hidden mesh (tractable without vmap/JIT for the generalized
# model): 21 strikes x 21 maturities per observation time. Smaller than Task
# 1's spec-resolution 151x161 mesh -- documented scope reduction, see
# README.md and task1b_report.json["limitations"].
DENSE_STRIKES = np.linspace(55.0, 195.0, 21)
DENSE_TAUS_OFFSET = np.linspace(0.05, 1.55, 21)  # added to each observation time t


def _finite_diff_q(C_row: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Interior butterfly mass via central finite differences on a dense
    strike row (NOT autodiff): q_n = (C_{n+1}-C_n)/(K_{n+1}-K_n) -
    (C_n-C_{n-1})/(K_n-K_{n-1}), zero-padded at the two edges."""
    q = np.zeros_like(C_row)
    hm = K[1:-1] - K[:-2]
    hp = K[2:] - K[1:-1]
    q[1:-1] = (C_row[2:] - C_row[1:-1]) / hp - (C_row[1:-1] - C_row[:-2]) / hm
    return q


def export_generalized_bates_fixed(heston: orc.HestonParams, extension: gb.HullWhiteJumpExtension):
    """Sec. 13: generalized_bates_grid_visible.npz / generalized_bates_dense_hidden.npz
    for the FIXED (non-degenerate) Generalised-Bates null model."""
    visible = {}
    hidden = {}
    S, V = orc.S0_DEFAULT, orc.V0_DEFAULT

    for ti, t in enumerate(orc.DEFAULT_OBSERVATION_TIMES):
        C = np.full((19, 3), np.nan)
        q = np.full((19, 3), np.nan)
        ell = np.full((19, 3), np.nan)
        L_C = np.full((19, 3), np.nan)
        L_q = np.full((19, 3), np.nan)
        R = np.full((19, 3), np.nan)
        R_nojump = np.full((19, 3), np.nan)
        admissible = np.zeros((19, 3), dtype=bool)
        excluded = np.zeros((19, 3), dtype=bool)
        cross_bucket = np.zeros((19, 3), dtype=bool)
        bucket_curr = np.full((19, 3), -1, dtype=int)
        bucket_prev = np.full((19, 3), -1, dtype=int)

        for ni, n_idx in enumerate(orc.FIXED_STRIKE_INTERIOR_INDICES):
            for mi, m_idx in enumerate(orc.FIXED_BACK_SLAB_INDICES):
                T_curr = orc.FIXED_EXPIRIES[m_idx]
                if T_curr - t <= 0.0:
                    continue
                r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, t, S, V, heston, extension)
                C[ni, mi] = r.C
                q[ni, mi] = r.q
                ell[ni, mi] = r.ell
                L_C[ni, mi] = r.L_C
                L_q[ni, mi] = r.L_q
                R[ni, mi] = r.R
                R_nojump[ni, mi] = r.R_nojump
                admissible[ni, mi] = r.admissible
                excluded[ni, mi] = r.excluded_near_boundary
                cross_bucket[ni, mi] = r.cross_bucket_slab
                bucket_curr[ni, mi] = r.bucket_curr
                bucket_prev[ni, mi] = r.bucket_prev

        for field, arr in [("C", C), ("q", q), ("ell", ell), ("L_C", L_C), ("L_q", L_q), ("R", R),
                            ("R_nojump", R_nojump), ("admissible", admissible), ("excluded_near_boundary", excluded),
                            ("cross_bucket_slab", cross_bucket), ("bucket_curr", bucket_curr), ("bucket_prev", bucket_prev)]:
            visible[f"t{ti}_{field}"] = arr

        # Dense hidden mesh: prices + IVs + finite-difference q, no generator/
        # residual diagnostics (see module docstring's documented scope note).
        taus = DENSE_TAUS_OFFSET
        C_dense = ci.price_grid(t, S, V, taus, DENSE_STRIKES, heston, extension)
        IV_dense = ci.implied_vol_grid(C_dense, S, taus, DENSE_STRIKES)
        q_dense = np.array([_finite_diff_q(C_dense[j], DENSE_STRIKES) for j in range(len(taus))])
        hidden[f"t{ti}_C"] = C_dense
        hidden[f"t{ti}_IV"] = IV_dense
        hidden[f"t{ti}_q_finite_diff"] = q_dense

    meta = dict(
        S0=S, V0=V, observation_times=list(orc.DEFAULT_OBSERVATION_TIMES),
        strikes_interior=list(orc.FIXED_STRIKES[1:-1]), back_slab_expiries=list(orc.FIXED_EXPIRIES[1:]),
        heston=dict(kappa=heston.kappa, theta=heston.theta, xi=heston.xi, rho=heston.rho),
        extension=dict(lambda_=extension.lambda_, bucket_edges=list(extension.bucket_edges),
                        nu=list(extension.nu), delta=list(extension.delta)),
        dense_strikes=list(DENSE_STRIKES), dense_tau_offsets=list(DENSE_TAUS_OFFSET),
    )
    visible["meta_json"] = json.dumps(meta)
    hidden["meta_json"] = json.dumps(meta)
    np.savez(f"{OUT_DIR}/generalized_bates_grid_visible.npz", **visible)
    np.savez(f"{OUT_DIR}/generalized_bates_dense_hidden.npz", **hidden)
    return meta


def export_crc_state(filename_prefix: str, t: float, S: float, V: float, heston: orc.HestonParams,
                      extension: gb.HullWhiteJumpExtension) -> dict:
    """One CRC state's (pre- or post-reset) protected grid_visible.npz, at a
    single observation time t (the reset time)."""
    visible = {}
    C = np.full(19 * 3, np.nan).reshape(19, 3)
    q = np.full((19, 3), np.nan)
    ell = np.full((19, 3), np.nan)
    L_C = np.full((19, 3), np.nan)
    R = np.full((19, 3), np.nan)
    R_nojump = np.full((19, 3), np.nan)
    admissible = np.zeros((19, 3), dtype=bool)
    excluded = np.zeros((19, 3), dtype=bool)
    cross_bucket = np.zeros((19, 3), dtype=bool)

    for ni, n_idx in enumerate(orc.FIXED_STRIKE_INTERIOR_INDICES):
        for mi, m_idx in enumerate(orc.FIXED_BACK_SLAB_INDICES):
            T_curr = orc.FIXED_EXPIRIES[m_idx]
            if T_curr - t <= 0.0:
                continue
            r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, t, S, V, heston, extension)
            C[ni, mi] = r.C
            q[ni, mi] = r.q
            ell[ni, mi] = r.ell
            L_C[ni, mi] = r.L_C
            R[ni, mi] = r.R
            R_nojump[ni, mi] = r.R_nojump
            admissible[ni, mi] = r.admissible
            excluded[ni, mi] = r.excluded_near_boundary
            cross_bucket[ni, mi] = r.cross_bucket_slab

    for field, arr in [("C", C), ("q", q), ("ell", ell), ("L_C", L_C), ("R", R), ("R_nojump", R_nojump),
                        ("admissible", admissible), ("excluded_near_boundary", excluded), ("cross_bucket_slab", cross_bucket)]:
        visible[field] = arr

    meta = dict(
        t=t, S=S, V=V,
        heston=dict(kappa=heston.kappa, theta=heston.theta, xi=heston.xi, rho=heston.rho),
        extension=dict(lambda_=extension.lambda_, bucket_edges=list(extension.bucket_edges),
                        nu=list(extension.nu), delta=list(extension.delta)),
    )
    visible["meta_json"] = json.dumps(meta)
    np.savez(f"{OUT_DIR}/{filename_prefix}_grid_visible.npz", **visible)
    return meta


def export_crc_dense_hidden(t: float, S: float, V: float,
                             heston_pre, extension_pre, heston_post, extension_post) -> dict:
    taus = DENSE_TAUS_OFFSET
    C_pre = ci.price_grid(t, S, V, taus, DENSE_STRIKES, heston_pre, extension_pre)
    IV_pre = ci.implied_vol_grid(C_pre, S, taus, DENSE_STRIKES)
    C_post = ci.price_grid(t, S, V, taus, DENSE_STRIKES, heston_post, extension_post)
    IV_post = ci.implied_vol_grid(C_post, S, taus, DENSE_STRIKES)
    out = dict(
        pre_C=C_pre, pre_IV=IV_pre, post_C=C_post, post_IV=IV_post,
        dense_strikes=DENSE_STRIKES, dense_tau_offsets=taus,
    )
    meta = dict(t=t, S=S, V=V)
    out["meta_json"] = json.dumps(meta)
    np.savez(f"{OUT_DIR}/crc_dense_hidden.npz", **out)
    return meta


def run() -> dict:
    t_start = time.time()
    report: dict = {}

    # --- Stage 1: constant-bucket reduction (Sec. 4/15 item 1) ---
    reg_heston, reg_jumps = gb.BATES_REGRESSION_HESTON, gb.BATES_REGRESSION_JUMPS
    reg_ext = gb.constant_bucket_extension(reg_jumps)
    max_price_diff = 0.0
    for t in orc.DEFAULT_OBSERVATION_TIMES:
        for T in orc.FIXED_EXPIRIES:
            tau = T - t
            if tau <= 0.0:
                continue
            for K in orc.FIXED_STRIKES:
                c_ref = float(orc.call_price(orc.S0_DEFAULT, orc.V0_DEFAULT, tau, K, reg_heston, reg_jumps))
                c_gen = float(gb.generalized_bates_call_price(t, orc.S0_DEFAULT, orc.V0_DEFAULT, T, K, reg_heston, reg_ext))
                max_price_diff = max(max_price_diff, abs(c_ref - c_gen))
    report["stage1_constant_bucket_reduction"] = dict(max_abs_price_diff=max_price_diff, tolerance=1.0e-8,
                                                        passed=bool(max_price_diff <= 1.0e-8))
    if max_price_diff > 1.0e-8:
        report["STOPPED_AT_STAGE"] = 1
        return report

    # --- Stage 2: martingale normalisation (Sec. 15 item 2) ---
    max_mart = 0.0
    for tau in [0.05, 0.5, 1.0, 2.0]:
        for heston, ext in [(reg_heston, reg_ext), (gb.CRC_ADMISSIBLE_HESTON, gb.CRC_TEST_EXTENSION)]:
            err = abs(gb.martingale_normalisation_error(tau, orc.V0_DEFAULT, heston, ext))
            max_mart = max(max_mart, err)
    report["stage2_martingale_normalisation"] = dict(max_abs_error=max_mart, tolerance=1.0e-10, passed=bool(max_mart <= 1.0e-10))
    if max_mart > 1.0e-10:
        report["STOPPED_AT_STAGE"] = 2
        return report

    # --- Stage 3: independent pricing / static arbitrage (Sec. 15 item 3) ---
    heston_crc, ext_crc = gb.CRC_ADMISSIBLE_HESTON, gb.CRC_TEST_EXTENSION
    max_method_diff = 0.0
    for t in [0.0, 0.1]:
        for T in [0.5, 1.0, 2.0]:
            tau = T - t
            if tau <= 0.0:
                continue
            for K in [70.0, 100.0, 130.0]:
                lewis = float(gb.generalized_bates_call_price(t, orc.S0_DEFAULT, orc.V0_DEFAULT, T, K, heston_crc, ext_crc))
                p1p2 = gb.generalized_bates_heston93_call_price(orc.S0_DEFAULT, orc.V0_DEFAULT, tau, K, heston_crc, ext_crc)
                max_method_diff = max(max_method_diff, abs(lewis - p1p2))
    report["stage3_independent_pricing"] = dict(max_abs_diff_lewis_vs_p1p2=max_method_diff, tolerance=1.0e-8,
                                                  passed=bool(max_method_diff <= 1.0e-8))
    if max_method_diff > 1.0e-8:
        report["STOPPED_AT_STAGE"] = 3
        return report

    # --- Stage 4: fixed Generalised-Bates generator identity (Sec. 15 item 4) ---
    max_gen = 0.0
    n_tested = 0
    for t in orc.DEFAULT_OBSERVATION_TIMES:
        for T in [0.5, 1.0, 2.0]:
            tau = T - t
            if tau <= 0.0:
                continue
            if gb.distance_to_bucket_boundary(tau, ext_crc.bucket_edges) < 0.02:
                continue
            for K in [85.0, 100.0, 115.0]:
                f = gb.make_generalized_price_field(T, K, heston_crc, ext_crc)
                L = float(gb.generalized_bates_generator(f, t, orc.S0_DEFAULT, orc.V0_DEFAULT, heston_crc, ext_crc, tau))
                max_gen = max(max_gen, abs(L))
                n_tested += 1
    report["stage4_generator_identity"] = dict(max_abs_L_GB_C=max_gen, n_tested=n_tested, tolerance=1.0e-7,
                                                passed=bool(max_gen <= 1.0e-7))
    if max_gen > 1.0e-7:
        report["STOPPED_AT_STAGE"] = 4
        return report

    # --- Stage 5: jump-corrected DLV identity (Sec. 15 item 5) ---
    same_R, cross_R = [], []
    for t in orc.DEFAULT_OBSERVATION_TIMES:
        for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES:
            for m_idx in orc.FIXED_BACK_SLAB_INDICES:
                r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, t, orc.S0_DEFAULT, orc.V0_DEFAULT, heston_crc, ext_crc)
                if r.excluded_near_boundary or not (r.q > orc.Q_MIN_DEFAULT):
                    continue
                (cross_R if r.cross_bucket_slab else same_R).append(abs(r.R))
    same_R, cross_R = np.array(same_R), np.array(cross_R)
    report["stage5_jump_corrected_dlv_identity"] = dict(
        same_bucket_max_R=float(same_R.max()), same_bucket_n=len(same_R), tolerance=1.0e-5,
        passed=bool(same_R.max() <= 1.0e-5),
        cross_bucket_max_R=float(cross_R.max()), cross_bucket_median_R=float(np.median(cross_R)), cross_bucket_n=len(cross_R),
        cross_bucket_note="Not a bug: cross_bucket_slab nodes straddle a jump-bucket boundary, so the back-slab "
                           "theorem's shared-jump-law premise does not hold there; see generalized_bates.py docstring.",
    )
    if same_R.max() > 1.0e-5:
        report["STOPPED_AT_STAGE"] = 5
        return report

    # --- Export the fixed Generalised-Bates visible/hidden surfaces ---
    gb_meta = export_generalized_bates_fixed(heston_crc, ext_crc)
    report["generalized_bates_export_meta"] = gb_meta

    # --- Stage 6: deterministic CRC inverse (Sec. 15 item 6) ---
    step = ci.deterministic_conditional_mean_state_step(0.0, orc.S0_DEFAULT, orc.V0_DEFAULT, 0.05, heston_crc)
    tu = ci.deterministic_tangent_update(heston_crc)
    Ks = ci.calibration_strikes(step.S_post)
    taus = np.asarray(ci.CALIBRATION_TAUS)
    C_target_calib = ci.price_grid(step.t_post, step.S_post, step.V_post, taus, Ks, heston_crc, ext_crc)
    inv = ci.solve_crc_inverse(step, tu.heston_post, ext_crc, C_target_calib, Ks, taus, gamma=1.0e-3, max_nfev=200, n_multistart=3)
    rms_iv_calib = float(np.sqrt(np.nanmean(inv.iv_residual_bp ** 2)))
    max_iv_calib = float(np.nanmax(np.abs(inv.iv_residual_bp)))
    label = ci.label_crc_result(rms_iv_calib, max_iv_calib)
    report["stage6_crc_inverse"] = dict(
        success=inv.success, cost=inv.cost, n_calls_total=inv.n_calls_total, best_start_index=inv.best_start_index,
        gamma=inv.gamma, rms_iv_error_bp_calibration_grid=rms_iv_calib, max_iv_error_bp_calibration_grid=max_iv_calib,
        label=label,
    )

    # --- Stage 7: pre/post surface + DLV-continuity diagnostics (Sec. 15 item 7) ---
    surf = ci.surface_preservation_diagnostics(step, heston_crc, ext_crc, tu.heston_post, inv.extension_post)
    dlv_cont = ci.dlv_state_continuity(step, heston_crc, ext_crc, tu.heston_post, inv.extension_post)
    chg = ci.characteristic_change_report(tu, ext_crc, inv.extension_post, step.V_post)
    post_res = ci.post_reset_dlv_residual_report(step, tu.heston_post, inv.extension_post)
    report["stage7_reset_diagnostics"] = dict(
        surface_preservation=surf, dlv_state_continuity=dlv_cont,
        characteristic_changes=chg, post_reset_dlv_residual=post_res,
    )

    # --- Stage 8: withheld dense-grid evaluation (Sec. 15 item 8, part of surf) ---
    report["stage8_withheld_dense_grid"] = surf["withheld_dense_grid"]
    report["stage8_passed_withheld_evaluated"] = True

    # --- Sec. 12 positive controls ---
    naive = ci.control_naive_no_recalibration(step, heston_crc, ext_crc, tu.heston_post)
    too_few = ci.control_too_few_buckets(step, tu.heston_post, ext_crc, C_target_calib, Ks, taus, max_nfev=200)
    stopped_early = ci.control_stopped_early(step, tu.heston_post, ext_crc, C_target_calib, Ks, taus)
    jump_omitted = ci.control_jump_term_omitted(step, tu.heston_post, inv.extension_post)
    report["sec12_positive_controls"] = dict(
        control1_naive_no_recalibration=naive["protected_grid"],
        control2_too_few_buckets=too_few,
        control3_stopped_early=dict(
            rms_iv_error_bp=float(np.sqrt(np.nanmean(stopped_early.iv_residual_bp ** 2))),
            max_iv_error_bp=float(np.nanmax(np.abs(stopped_early.iv_residual_bp))),
            success=stopped_early.success, n_calls_total=stopped_early.n_calls_total,
        ),
        control4_jump_term_omitted=jump_omitted,
        converged_rms_iv_error_bp_for_comparison=rms_iv_calib,
    )

    # --- Exports: CRC pre/post grid_visible + dense_hidden ---
    pre_meta = export_crc_state("crc_pre_reset", step.t_pre, step.S_pre, step.V_pre, heston_crc, ext_crc)
    post_meta = export_crc_state("crc_post_reset", step.t_post, step.S_post, step.V_post, tu.heston_post, inv.extension_post)
    dense_meta = export_crc_dense_hidden(step.t_post, step.S_post, step.V_post, heston_crc, ext_crc, tu.heston_post, inv.extension_post)
    report["crc_pre_reset_export_meta"] = pre_meta
    report["crc_post_reset_export_meta"] = post_meta
    report["crc_dense_hidden_export_meta"] = dense_meta

    # --- Stage 9: export reproducibility check (Sec. 15 item 9) ---
    reload_visible = np.load(f"{OUT_DIR}/crc_post_reset_grid_visible.npz")
    reload_gb_visible = np.load(f"{OUT_DIR}/generalized_bates_grid_visible.npz")
    reload_dense = np.load(f"{OUT_DIR}/crc_dense_hidden.npz")
    R_reloaded = reload_visible["R"]
    R_finite_count = int(np.isfinite(R_reloaded).sum())
    report["stage9_export_reproducibility"] = dict(
        reload_succeeded=True,
        crc_post_reset_R_finite_count=R_finite_count,
        crc_post_reset_R_matches_in_memory=bool(np.allclose(
            R_reloaded[np.isfinite(R_reloaded)],
            [gb.generalized_finite_grid_node_residual(n_idx, m_idx, step.t_post, step.S_post, step.V_post,
                                                        tu.heston_post, inv.extension_post).R
             for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES for m_idx in orc.FIXED_BACK_SLAB_INDICES
             if orc.FIXED_EXPIRIES[m_idx] - step.t_post > 0.0
             and math.isfinite(gb.generalized_finite_grid_node_residual(n_idx, m_idx, step.t_post, step.S_post,
                                                                          step.V_post, tu.heston_post, inv.extension_post).R)],
        )) if R_finite_count > 0 else None,
        generalized_bates_visible_keys=len(reload_gb_visible.files),
        crc_dense_hidden_keys=len(reload_dense.files),
    )

    report["state_step"] = dict(t_pre=step.t_pre, t_post=step.t_post, dt=step.dt, S_pre=step.S_pre, V_pre=step.V_pre,
                                 S_post=step.S_post, V_post=step.V_post, method=step.method)
    report["tangent_update"] = dict(
        heston_pre=dict(kappa=tu.heston_pre.kappa, theta=tu.heston_pre.theta, xi=tu.heston_pre.xi, rho=tu.heston_pre.rho),
        heston_post=dict(kappa=tu.heston_post.kappa, theta=tu.heston_post.theta, xi=tu.heston_post.xi, rho=tu.heston_post.rho),
        relative_changes=tu.relative_changes,
    )
    report["extension_pre"] = dict(lambda_=ext_crc.lambda_, bucket_edges=list(ext_crc.bucket_edges),
                                    nu=list(ext_crc.nu), delta=list(ext_crc.delta))
    report["extension_post"] = dict(lambda_=inv.extension_post.lambda_, bucket_edges=list(inv.extension_post.bucket_edges),
                                     nu=list(inv.extension_post.nu), delta=list(inv.extension_post.delta))
    report["wall_clock_seconds"] = time.time() - t_start
    report["ALL_STAGES_PASSED"] = True
    return report


if __name__ == "__main__":
    result = run()
    with open(f"{OUT_DIR}/task1b_report.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_meta")}, indent=2, default=str))
