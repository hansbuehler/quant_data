"""Unit tests for generalized_bates.py: constant-bucket reduction, pricer
acceptance, martingale normalisation, generator identity and jump-corrected
DLV residual for the time-inhomogeneous, maturity-bucketed Generalised-Bates
model (Task 1B).

Run with:  PYTHONPATH=.. pytest tests/test_generalized_bates.py
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

S0, V0 = orc.S0_DEFAULT, orc.V0_DEFAULT
REG_HESTON = gb.BATES_REGRESSION_HESTON
REG_JUMPS = gb.BATES_REGRESSION_JUMPS
REG_EXT = gb.constant_bucket_extension(REG_JUMPS)
CRC_HESTON = gb.CRC_ADMISSIBLE_HESTON
CRC_EXT = gb.CRC_TEST_EXTENSION

BOUNDARY_TOL = 0.02


# ===========================================================================
# Parameter validation (Sec. 3)
# ===========================================================================


def test_extension_rejects_negative_lambda():
    with pytest.raises(ValueError):
        gb.HullWhiteJumpExtension(lambda_=-0.1, bucket_edges=gb.DEFAULT_BUCKET_EDGES, nu=(0.0,) * 5, delta=(0.1,) * 5)


def test_extension_rejects_wrong_bucket_count():
    with pytest.raises(ValueError):
        gb.HullWhiteJumpExtension(lambda_=0.3, bucket_edges=gb.DEFAULT_BUCKET_EDGES, nu=(0.0,) * 4, delta=(0.1,) * 4)


def test_extension_rejects_nonincreasing_edges():
    bad_edges = (0.0, 0.1, 0.1, 0.5, 1.0, 2.0)
    with pytest.raises(ValueError):
        gb.HullWhiteJumpExtension(lambda_=0.3, bucket_edges=bad_edges, nu=(0.0,) * 5, delta=(0.1,) * 5)


def test_extension_rejects_nonpositive_delta():
    with pytest.raises(ValueError):
        gb.HullWhiteJumpExtension(lambda_=0.3, bucket_edges=gb.DEFAULT_BUCKET_EDGES, nu=(0.0,) * 5, delta=(0.1, 0.1, 0.0, 0.1, 0.1))


def test_extension_rejects_edges_not_starting_at_zero():
    bad_edges = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
    with pytest.raises(ValueError):
        gb.HullWhiteJumpExtension(lambda_=0.3, bucket_edges=bad_edges, nu=(0.0,) * 5, delta=(0.1,) * 5)


def test_default_extension_covers_full_grid_horizon():
    assert REG_EXT.covers(2.0)
    assert CRC_EXT.covers(2.0)


def test_crc_admissible_params_satisfy_feller():
    assert 2.0 * CRC_HESTON.kappa * CRC_HESTON.theta > CRC_HESTON.xi ** 2


def test_regression_params_unchanged_from_oracle_defaults():
    """Sec. 6: the regression baseline is EXACTLY the existing default, not a
    modified copy."""
    assert REG_HESTON == orc.HESTON_DEFAULT
    assert REG_JUMPS == orc.BATES_JUMPS_DEFAULT


# ===========================================================================
# Bucket bookkeeping
# ===========================================================================


@pytest.mark.parametrize(
    "tau,expected_bucket",
    [
        (0.0, 0), (14.0 / 365.0, 0), (14.0 / 365.0 + 1e-9, 1),
        (60.0 / 365.0, 1), (60.0 / 365.0 + 1e-9, 2),
        (120.0 / 365.0, 2), (120.0 / 365.0 + 1e-9, 3),
        (270.0 / 365.0, 3), (270.0 / 365.0 + 1e-9, 4),
        (2.0, 4),
    ],
)
def test_current_bucket_index(tau, expected_bucket):
    assert gb.current_bucket_index(tau, gb.DEFAULT_BUCKET_EDGES) == expected_bucket


def test_current_bucket_index_rejects_out_of_range():
    with pytest.raises(ValueError):
        gb.current_bucket_index(2.5, gb.DEFAULT_BUCKET_EDGES)


@pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 1.999999])
def test_bucket_overlaps_sum_to_tau_when_covered(tau):
    total = float(np.sum(np.asarray(gb.bucket_overlap(tau, gb.DEFAULT_BUCKET_EDGES))))
    assert abs(total - tau) < 1e-12


# ===========================================================================
# Sec. 4: exact reduction to ordinary Bates when all buckets are identical.
# Characteristic functions (complex Fourier args used by the Lewis pricer):
# near machine precision. Prices: <=1e-8 abs.
# ===========================================================================


@pytest.mark.parametrize("tau", [0.05, 0.5, 1.0, 1.8])
def test_reduction_characteristic_function_machine_precision(tau):
    import jax.numpy as jnp

    u = orc.DEFAULT_PRICE_QUADRATURE.u - 0.5j
    psi_ref = orc.characteristic_function(u, tau, V0, REG_HESTON, REG_JUMPS)
    psi_gen = gb.generalized_bates_characteristic_function(u, tau, V0, REG_HESTON, REG_EXT)
    rel = jnp.max(jnp.abs(psi_ref - psi_gen) / jnp.maximum(jnp.abs(psi_ref), 1e-300))
    assert float(rel) < 1e-10


@pytest.mark.parametrize("t", orc.DEFAULT_OBSERVATION_TIMES)
@pytest.mark.parametrize("T", orc.FIXED_EXPIRIES)
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_reduction_price_protected_grid(t, T, K):
    tau = T - t
    if tau <= 0.0:
        pytest.skip("tau<=0")
    c_ref = float(orc.call_price(S0, V0, tau, K, REG_HESTON, REG_JUMPS))
    c_gen = float(gb.generalized_bates_call_price(t, S0, V0, T, K, REG_HESTON, REG_EXT))
    assert abs(c_ref - c_gen) <= 1.0e-8


@pytest.mark.parametrize("K,T", [(72.5, 0.6), (97.5, 1.3), (143.0, 1.9)])
def test_reduction_price_off_grid(K, T):
    c_ref = float(orc.call_price(S0, V0, T, K, REG_HESTON, REG_JUMPS))
    c_gen = float(gb.generalized_bates_call_price(0.0, S0, V0, T, K, REG_HESTON, REG_EXT))
    assert abs(c_ref - c_gen) <= 1.0e-8


def test_reduction_independent_p1p2_method():
    for K, tau in [(70.0, 0.5), (100.0, 0.5), (130.0, 0.5), (100.0, 1.0), (85.0, 1.8)]:
        c_ref = orc.heston93_call_price(S0, V0, tau, K, REG_HESTON, REG_JUMPS)
        c_gen = gb.generalized_bates_heston93_call_price(S0, V0, tau, K, REG_HESTON, REG_EXT)
        assert abs(c_ref - c_gen) <= 1.0e-8


def test_reduction_generator_matches_oracle_bates_generator():
    for t, T, K in [(0.0, 1.0, 100.0), (0.05, 0.5, 90.0), (0.1, 2.0, 120.0)]:
        tau = T - t
        f_gen = gb.make_generalized_price_field(T, K, REG_HESTON, REG_EXT)
        f_ref = orc.make_price_field(T, K, REG_HESTON, REG_JUMPS)
        L_gen = float(gb.generalized_bates_generator(f_gen, t, S0, V0, REG_HESTON, REG_EXT, tau))
        L_ref = float(orc.bates_generator(f_ref, t, S0, V0, REG_HESTON, REG_JUMPS))
        assert abs(L_gen - L_ref) <= 1.0e-8


# ===========================================================================
# Sec. 5/15.2: martingale normalisation psi_GB(-i; tau, V) == 1.
# ===========================================================================


@pytest.mark.parametrize("tau", [0.05, 0.5, 1.0, 2.0])
def test_martingale_normalisation_constant_bucket(tau):
    err = gb.martingale_normalisation_error(tau, V0, REG_HESTON, REG_EXT)
    assert abs(err) < 1.0e-10


@pytest.mark.parametrize("tau", [0.05, 0.5, 1.0, 2.0])
def test_martingale_normalisation_generalized(tau):
    err = gb.martingale_normalisation_error(tau, V0, CRC_HESTON, CRC_EXT)
    assert abs(err) < 1.0e-10


# ===========================================================================
# Sec. 15.3: independent pricing / static-arbitrage checks for the actual
# (non-degenerate) Generalised-Bates model.
# ===========================================================================


@pytest.mark.parametrize("t", [0.0, 0.1])
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_generalized_pricer_agrees_with_independent_method(t, T, K):
    tau = T - t
    if tau <= 0.0:
        pytest.skip("tau<=0")
    lewis = float(gb.generalized_bates_call_price(t, S0, V0, T, K, CRC_HESTON, CRC_EXT))
    p1p2 = gb.generalized_bates_heston93_call_price(S0, V0, tau, K, CRC_HESTON, CRC_EXT)
    assert abs(lewis - p1p2) <= 1.0e-8


@pytest.mark.parametrize("K,tau", [(70.0, 0.5), (100.0, 0.5), (130.0, 0.5), (100.0, 1.0)])
def test_generalized_put_call_parity_independent_put(K, tau):
    call = float(gb.generalized_bates_call_price(0.0, S0, V0, tau, K, CRC_HESTON, CRC_EXT))
    indep_put = gb.generalized_bates_heston93_put_price(S0, V0, tau, K, CRC_HESTON, CRC_EXT)
    parity_put = call - S0 + K
    assert abs(indep_put - parity_put) <= 1.0e-8


def test_generalized_call_price_bounds_and_monotonicity():
    Ks = np.arange(60.0, 161.0, 5.0)
    Cs = np.array([float(gb.generalized_bates_call_price(0.0, S0, V0, 1.0, K, CRC_HESTON, CRC_EXT)) for K in Ks])
    intrinsic = np.maximum(S0 - Ks, 0.0)
    assert np.all(Cs >= intrinsic - 1.0e-6)
    assert np.all(Cs <= S0 + 1.0e-6)
    assert np.all(np.diff(Cs) <= 1.0e-9)  # monotone decreasing in K
    assert np.all(np.diff(Cs, 2) >= -1.0e-9)  # convex in K


def test_generalized_call_price_calendar_monotone():
    Ts = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
    Cs = np.array([float(gb.generalized_bates_call_price(0.0, S0, V0, T, 100.0, CRC_HESTON, CRC_EXT)) for T in Ts])
    assert np.all(np.diff(Cs) >= -1.0e-9)


# ===========================================================================
# Sec. 7 item 6 / Sec. 15.4: fixed Generalised-Bates generator identity
# L_GB C = 0, away from deterministic bucket boundaries.
# ===========================================================================


@pytest.mark.parametrize("t", orc.DEFAULT_OBSERVATION_TIMES)
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
def test_generalized_generator_identity(t, T, K):
    tau = T - t
    if tau <= 0.0:
        pytest.skip("tau<=0")
    if gb.distance_to_bucket_boundary(tau, CRC_EXT.bucket_edges) < BOUNDARY_TOL:
        pytest.skip("tau within boundary_tol of a deterministic bucket boundary")
    f = gb.make_generalized_price_field(T, K, CRC_HESTON, CRC_EXT)
    L = float(gb.generalized_bates_generator(f, t, S0, V0, CRC_HESTON, CRC_EXT, tau))
    assert abs(L) <= 1.0e-7


def test_generator_boundary_exclusion_is_necessary():
    """Regression: tau exactly at the outer bucket-coverage edge (t=0, T=2.0)
    hits a genuine JAX autodiff subgradient tie in jnp.minimum(tau, hi) (the
    averaged subgradient 0.5 instead of the correct one-sided 1.0), and MUST
    be excluded, not silently tolerated by loosening the threshold."""
    t, T, K = 0.0, 2.0, 100.0
    tau = T - t
    assert gb.distance_to_bucket_boundary(tau, CRC_EXT.bucket_edges) < 1e-9
    f = gb.make_generalized_price_field(T, K, CRC_HESTON, CRC_EXT)
    L = float(gb.generalized_bates_generator(f, t, S0, V0, CRC_HESTON, CRC_EXT, tau))
    assert abs(L) > 1.0e-3  # confirms this node IS a genuine problem, not fixed by chance


# ===========================================================================
# Sec. 7 item 7 / Sec. 15.5: jump-corrected finite-grid DLV residual, plus
# the positive control (jump term omitted must NOT vanish alongside R).
# ===========================================================================


@pytest.mark.parametrize("n_idx", [1, 10, 19])
@pytest.mark.parametrize("m_idx", [1, 2, 3])
def test_generalized_finite_grid_residual_small_same_bucket(n_idx, m_idx):
    """The jump-corrected DLV identity, tested where the back-slab's premise
    (a single shared jump law across T_prev and T_curr) actually holds."""
    r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, 0.1, S0, V0, CRC_HESTON, CRC_EXT)
    if r.excluded_near_boundary:
        pytest.skip("near a deterministic bucket boundary")
    if r.cross_bucket_slab:
        pytest.skip("cross-bucket slab: theorem premise (shared jump law) does not hold, see module docstring")
    assert abs(r.L_C) <= 1.0e-6
    assert abs(r.L_q) <= 1.0e-6
    assert abs(r.R) <= 1.0e-5


@pytest.mark.parametrize("n_idx", [1, 10, 19])
@pytest.mark.parametrize("m_idx", [1, 2, 3])
def test_generalized_positive_control_same_bucket(n_idx, m_idx):
    r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, 0.1, S0, V0, CRC_HESTON, CRC_EXT)
    if r.excluded_near_boundary:
        pytest.skip("near a deterministic bucket boundary")
    if r.cross_bucket_slab:
        pytest.skip("cross-bucket slab: theorem premise (shared jump law) does not hold, see module docstring")
    assert abs(r.R_nojump) > 100.0 * max(abs(r.R), 1.0e-15)


def test_cross_bucket_slab_residual_is_a_diagnosed_structural_feature_not_a_bug():
    """Regression/documentation test: when a back slab straddles a bucket
    boundary (T_prev and T_curr fall in different buckets at t), R is O(1e-3)
    -- NOT because the identity is violated by a bug, but because the two
    constituent maturities are, under the per-contract-tau convention (see
    module docstring), genuinely priced under two different jump laws, so
    the back-slab theorem's shared-law premise does not hold for that pair.
    This must NOT be silently passed by loosening the residual tolerance."""
    # (t=0.1, m_idx=1): T_prev=0.25 (bucket 1), T_curr=0.5 (bucket 3) -- straddles.
    r = gb.generalized_finite_grid_node_residual(10, 1, 0.1, S0, V0, CRC_HESTON, CRC_EXT)
    assert r.cross_bucket_slab
    assert abs(r.R) > 1.0e-4  # confirms this is a real, not-accidentally-small, mismatch


def test_generalized_finite_grid_full_sweep_no_nan():
    for t in orc.DEFAULT_OBSERVATION_TIMES:
        for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES:
            for m_idx in orc.FIXED_BACK_SLAB_INDICES:
                r = gb.generalized_finite_grid_node_residual(n_idx, m_idx, t, S0, V0, CRC_HESTON, CRC_EXT)
                assert math.isfinite(r.R), f"NaN/Inf at t={t}, n={n_idx}, m={m_idx}"
                assert math.isfinite(r.R_nojump), f"NaN/Inf (nojump) at t={t}, n={n_idx}, m={m_idx}"
