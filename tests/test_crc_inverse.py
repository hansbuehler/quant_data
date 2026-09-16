"""Unit tests for crc_inverse.py: deterministic state step, tangent update,
inverse calibration, reset diagnostics and positive controls (Task 1B
Sec. 9-12).

Run with:  PYTHONPATH=.. pytest tests/test_crc_inverse.py
Note: several tests here call the CRC inverse solver (a real, unmocked
least_squares fit over a 130-point calibration grid) and take real wall-clock
time (order tens of seconds each); this file is not intended for a tight
inner-loop test cycle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oracle as orc
import generalized_bates as gb
import crc_inverse as ci

HESTON_PRE = gb.CRC_ADMISSIBLE_HESTON
EXT_PRE = gb.CRC_TEST_EXTENSION
S0, V0 = orc.S0_DEFAULT, orc.V0_DEFAULT


# ===========================================================================
# Sec. 9 Step A: deterministic state evolution.
# ===========================================================================


def test_state_step_v_reverts_toward_theta():
    step = ci.deterministic_conditional_mean_state_step(0.0, S0, 0.09, 0.05, HESTON_PRE)
    assert HESTON_PRE.theta < step.V_post < 0.09  # mean-reverts down toward theta


def test_state_step_is_identity_at_v_equals_theta():
    step = ci.deterministic_conditional_mean_state_step(0.0, S0, HESTON_PRE.theta, 0.05, HESTON_PRE)
    assert abs(step.V_post - HESTON_PRE.theta) < 1.0e-12


def test_state_step_spot_drops_by_jensen_gap_when_v_above_theta():
    """E[X_{t+dt}] < X_t when V > 0 (the -0.5*E[int V ds] term is strictly
    negative), so the conditional-mean spot is strictly below the starting
    spot -- this is a real (if small) effect, not a bug."""
    step = ci.deterministic_conditional_mean_state_step(0.0, S0, V0, 0.05, HESTON_PRE)
    assert step.S_post < step.S_pre
    assert abs(math.log(step.S_post / step.S_pre) + 0.5 * V0 * 0.05) < 1.0e-3  # close to the V~const approx


# ===========================================================================
# Sec. 9 Step C: deterministic tangent update.
# ===========================================================================


def test_tangent_update_respects_5pct_cap():
    tu = ci.deterministic_tangent_update(HESTON_PRE, d_theta_rel=0.05, d_xi_rel=-0.05, d_rho_rel=-0.05)
    assert abs(tu.heston_post.theta / tu.heston_pre.theta - 1.0) <= 0.05 + 1.0e-12
    assert abs(tu.heston_post.xi / tu.heston_pre.xi - 1.0) <= 0.05 + 1.0e-12
    assert tu.heston_post.kappa == tu.heston_pre.kappa  # kappa held fixed


def test_tangent_update_rejects_change_over_cap():
    with pytest.raises(ValueError):
        ci.deterministic_tangent_update(HESTON_PRE, d_theta_rel=0.10)


def test_tangent_update_enforces_feller():
    """CRC_ADMISSIBLE_HESTON has enough Feller margin (ratio ~1.31) that no
    +-5% move can violate it (worst case within the cap: ratio ~1.13, still
    admissible) -- so this uses a hand-built near-boundary HestonParams to
    exercise the guard rail directly."""
    near_boundary = orc.HestonParams(kappa=2.0, theta=0.04, xi=0.392, rho=-0.7)  # ratio 2*2*0.04/0.392^2 ~= 1.041
    with pytest.raises(ValueError):
        ci.deterministic_tangent_update(near_boundary, d_theta_rel=-0.05, d_xi_rel=0.05, d_rho_rel=0.0)


# ===========================================================================
# Sec. 8: calibration grid / moneyness convention.
# ===========================================================================


def test_calibration_strikes_convention():
    Ks = ci.calibration_strikes(100.0)
    assert len(Ks) == 13
    assert abs(Ks[0] - 80.0) < 1.0e-9  # moneyness 0.8 at S=100
    assert abs(Ks[-1] - 120.0) < 1.0e-9  # moneyness 1.2


def test_calibration_grid_distinct_from_protected_grid():
    assert set(ci.CALIBRATION_TAUS) != set(T for T in ci.PROTECTED_EXPIRIES)


# ===========================================================================
# Black-Scholes IV/vega round trip (self-contained sanity).
# ===========================================================================


def test_implied_vol_round_trip():
    S, K, tau, sigma = 100.0, 95.0, 0.5, 0.25
    C = ci._bs_call(S, K, tau, sigma)
    iv = ci.implied_vol(C, S, K, tau)
    assert abs(iv - sigma) < 1.0e-6


# ===========================================================================
# Sec. 9 Step D + Sec. 10 labelling: full inverse solve (slow).
# ===========================================================================


@pytest.fixture(scope="module")
def full_reset():
    step = ci.deterministic_conditional_mean_state_step(0.0, S0, V0, 0.05, HESTON_PRE)
    tu = ci.deterministic_tangent_update(HESTON_PRE)
    Ks = ci.calibration_strikes(step.S_post)
    taus = np.asarray(ci.CALIBRATION_TAUS)
    C_target = ci.price_grid(step.t_post, step.S_post, step.V_post, taus, Ks, HESTON_PRE, EXT_PRE)
    inv = ci.solve_crc_inverse(step, tu.heston_post, EXT_PRE, C_target, Ks, taus, gamma=1.0e-3, max_nfev=200, n_multistart=3)
    return step, tu, Ks, taus, C_target, inv


def test_crc_inverse_converges(full_reset):
    _, _, _, _, _, inv = full_reset
    assert inv.success
    assert inv.fit_mask.sum() == inv.fit_mask.size  # all 130 calibration points usable


def test_crc_inverse_extension_stays_admissible(full_reset):
    """The optimiser's bounded, log-parameterised delta_b is structurally
    positive -- constructing HullWhiteJumpExtension from x_opt must not raise."""
    _, _, _, _, _, inv = full_reset
    assert all(d > 0.0 for d in inv.extension_post.delta)


def test_crc_inverse_beats_naive_no_recalibration(full_reset):
    """The recalibrated (Step D) fit must be closer to the target than doing
    nothing (Sec. 12 control 1 comparison, sanity direction)."""
    step, tu, Ks, taus, C_target, inv = full_reset
    rms_recal = float(np.sqrt(np.nanmean(inv.iv_residual_bp ** 2)))
    naive = ci.control_naive_no_recalibration(step, HESTON_PRE, EXT_PRE, tu.heston_post)
    rms_naive = naive["protected_grid"]["rms_iv_error_bp"]
    assert rms_recal < rms_naive


def test_crc_result_labelled_restricted_approximation_not_exact(full_reset):
    """Sec. 10: never call a merely-approximate reset an exact theoretical
    CRC null. With this Heston perturbation and 5-bucket family, the fit
    leaves a real residual (tens of bp), so it must be labelled the
    approximation tier, not silently marked high-accuracy."""
    _, _, _, _, _, inv = full_reset
    rms = float(np.sqrt(np.nanmean(inv.iv_residual_bp ** 2)))
    max_ = float(np.nanmax(np.abs(inv.iv_residual_bp)))
    label = ci.label_crc_result(rms, max_)
    assert label == "restricted_crc_approximation"


def test_label_high_accuracy_when_residual_tiny():
    assert ci.label_crc_result(0.5, 3.0) == "restricted_crc_high_accuracy"


def test_label_approximation_when_residual_large():
    assert ci.label_crc_result(50.0, 100.0) == "restricted_crc_approximation"


# ===========================================================================
# Sec. 11: reset diagnostics.
# ===========================================================================


def test_dlv_continuity_nonzero_for_approximate_reset(full_reset):
    """An imperfect (approximation-tier) reset must show a DETECTABLE (q,ell)
    jump -- a truly surface-preserving reset would leave these ~0."""
    step, tu, Ks, taus, C_target, inv = full_reset
    cont = ci.dlv_state_continuity(step, HESTON_PRE, EXT_PRE, tu.heston_post, inv.extension_post)
    assert cont["q_l2_diff"] > 1.0e-4
    assert cont["ell_l2_diff"] > 1.0e-4


def test_post_reset_model_is_itself_a_valid_oracle(full_reset):
    """Sec. 11 "post-reset no-arbitrage residual": regardless of calibration
    quality, the POST-reset (heston_post, extension_post) pair is itself a
    valid Generalised-Bates oracle and must satisfy the jump-corrected DLV
    identity to near machine precision on same-bucket nodes."""
    step, tu, Ks, taus, C_target, inv = full_reset
    report = ci.post_reset_dlv_residual_report(step, tu.heston_post, inv.extension_post)
    assert report["same_bucket_max_R"] < 1.0e-8


def test_jump_term_omission_control_shows_large_gap(full_reset):
    step, tu, Ks, taus, C_target, inv = full_reset
    ctrl = ci.control_jump_term_omitted(step, tu.heston_post, inv.extension_post)
    assert ctrl["ratio"] > 1.0e3  # R_nojump vastly exceeds R (positive control)


# ===========================================================================
# Sec. 12 positive controls: detector power.
# ===========================================================================


def test_control_too_few_buckets_worse_than_full_fit(full_reset):
    step, tu, Ks, taus, C_target, inv = full_reset
    rms_full = float(np.sqrt(np.nanmean(inv.iv_residual_bp ** 2)))
    few = ci.control_too_few_buckets(step, tu.heston_post, EXT_PRE, C_target, Ks, taus, max_nfev=200)
    assert few["rms_iv_error_bp"] > rms_full


def test_control_stopped_early_worse_than_full_fit(full_reset):
    step, tu, Ks, taus, C_target, inv = full_reset
    rms_full = float(np.sqrt(np.nanmean(inv.iv_residual_bp ** 2)))
    early = ci.control_stopped_early(step, tu.heston_post, EXT_PRE, C_target, Ks, taus)
    rms_early = float(np.sqrt(np.nanmean(early.iv_residual_bp ** 2)))
    assert rms_early > rms_full
