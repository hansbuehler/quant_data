"""Unit tests for oracle.py: pricer acceptance checks + a small-grid residual test.

Run with:  PYTHONPATH=.. pytest tests/  (or from the project root: pytest)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oracle as orc

HESTON = orc.HESTON_DEFAULT
JUMPS_NONE = orc.JumpParams()
JUMPS_BATES = orc.BATES_JUMPS_DEFAULT
S0, V0 = orc.S0_DEFAULT, orc.V0_DEFAULT


# ===========================================================================
# Pricer acceptance checks (spec Sec. "Pricing")
# ===========================================================================


@pytest.mark.parametrize("K", [70.0, 85.0, 100.0, 115.0, 130.0])
@pytest.mark.parametrize("tau", [0.05, 0.25, 0.75, 1.8])
def test_pricer_agrees_with_independent_method_heston(K, tau):
    lewis = float(orc.call_price(S0, V0, tau, K, HESTON, JUMPS_NONE))
    h93 = orc.heston93_call_price(S0, V0, tau, K, HESTON, JUMPS_NONE)
    assert abs(lewis - h93) <= 1.0e-8


@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("tau", [0.05, 0.5, 1.8])
def test_pricer_agrees_with_independent_method_bates(K, tau):
    lewis = float(orc.call_price(S0, V0, tau, K, HESTON, JUMPS_BATES))
    h93 = orc.heston93_call_price(S0, V0, tau, K, HESTON, JUMPS_BATES)
    assert abs(lewis - h93) <= 1.0e-8


@pytest.mark.parametrize("K,tau", [(70.0, 0.5), (100.0, 0.5), (130.0, 0.5), (100.0, 1.8)])
def test_put_call_parity_independent_put(K, tau):
    """Independent put (own P1/P2 recomputation) vs call-derived parity put."""
    call = float(orc.call_price(S0, V0, tau, K, HESTON, JUMPS_NONE))
    indep_put = orc.heston93_put_price(S0, V0, tau, K, HESTON, JUMPS_NONE)
    parity_put = call - S0 + K
    assert abs(indep_put - parity_put) <= 1.0e-8


def test_xi_to_zero_matches_black_scholes():
    tiny_xi = orc.HestonParams(kappa=1.5, theta=V0, xi=1e-5, rho=-0.7)
    lewis = float(orc.call_price(S0, V0, 0.5, 100.0, tiny_xi, JUMPS_NONE))
    bs = orc.black_scholes_call(S0, 100.0, 0.5, math.sqrt(V0))
    assert abs(lewis - bs) <= 1.0e-6


def test_no_branch_cut_discontinuity_in_tau():
    taus = np.linspace(0.01, 5.0, 100)
    prices = np.array([float(orc.call_price(S0, V0, float(t), 100.0, HESTON, JUMPS_NONE)) for t in taus])
    assert np.all(np.diff(prices) > -1.0e-8)  # monotone increasing in tau (ATM call)
    # smoothness: no jump larger than a modest fraction of the local slope's scale
    second_diff = np.diff(prices, 2)
    assert np.max(np.abs(second_diff)) < 1.0


def test_delta_matches_autodiff_gradient():
    """Cross-check: dC/dS from JAX autodiff should equal the Heston93 P1 (a
    textbook identity: Delta = P1 in this pricing decomposition)."""
    import jax

    K, tau = 100.0, 0.5
    dC_dS = float(jax.grad(lambda S: orc.call_price(S, V0, tau, K, HESTON, JUMPS_NONE))(S0))
    P1, _P2 = orc._heston93_p1_p2(S0, V0, tau, K, HESTON, JUMPS_NONE, n=1000, u_max=800.0)
    assert abs(dC_dS - P1) <= 1.0e-6


# ===========================================================================
# Small-grid residual test (Stage A1/B1 core identity, on a reduced grid for
# test speed -- the full grid is exercised in the notebook / export.py).
# ===========================================================================


@pytest.mark.parametrize("n_idx", [1, 10, 19])
@pytest.mark.parametrize("m_idx", [1, 2, 3])
def test_heston_finite_grid_residual_small(n_idx, m_idx):
    r = orc.finite_grid_node_residual(n_idx, m_idx, 0.1, S0, V0, HESTON, JUMPS_NONE)
    assert abs(r.L_C) <= 1.0e-7
    assert abs(r.L_q) <= 1.0e-7
    assert abs(r.R) <= 1.0e-5


@pytest.mark.parametrize("n_idx", [1, 10, 19])
@pytest.mark.parametrize("m_idx", [1, 2, 3])
def test_bates_finite_grid_residual_small(n_idx, m_idx):
    r = orc.finite_grid_node_residual(n_idx, m_idx, 0.1, S0, V0, HESTON, JUMPS_BATES)
    assert abs(r.L_C) <= 1.0e-7
    assert abs(r.L_q) <= 1.0e-7
    assert abs(r.R) <= 1.0e-5


@pytest.mark.parametrize("n_idx", [1, 10, 19])
@pytest.mark.parametrize("m_idx", [1, 2, 3])
def test_bates_positive_control(n_idx, m_idx):
    """R^{B,nojump} (jump term omitted) must be materially non-zero: if the
    jump term were computed incorrectly (or accidentally omitted), R and
    R_nojump would coincide."""
    r = orc.finite_grid_node_residual(n_idx, m_idx, 0.1, S0, V0, HESTON, JUMPS_BATES)
    assert r.R_nojump is not None
    assert abs(r.R_nojump) > 100.0 * max(abs(r.R), 1.0e-15)


@pytest.mark.parametrize("K,T", [(70.0, 0.6), (100.0, 1.0), (130.0, 1.9)])
def test_heston_continuous_residual_small(K, T):
    r = orc.continuous_node_residual(K, T, 0.1, S0, V0, HESTON, JUMPS_NONE)
    assert abs(r.L_q) <= 1.0e-7
    assert abs(r.R_normalized) <= 1.0e-4


@pytest.mark.parametrize("K,T", [(70.0, 1.0), (100.0, 1.0), (130.0, 1.0)])
def test_bates_continuous_residual_small(K, T):
    r = orc.continuous_node_residual(K, T, 0.1, S0, V0, HESTON, JUMPS_BATES)
    assert abs(r.R_normalized) <= 1.0e-4
    assert r.R_nojump is not None
    assert abs(r.R_nojump) > 100.0 * max(abs(r.R), 1.0e-15)


def test_vectorized_continuous_grid_matches_per_point():
    """The jax.vmap-batched continuous_grid_residuals must agree with the
    per-point continuous_node_residual it was derived from."""
    Ks = np.array([70.0, 100.0, 130.0])
    Ts = np.array([0.6, 1.0, 1.9])
    batched = orc.continuous_grid_residuals(Ks, Ts, 0.1, S0, V0, HESTON, JUMPS_NONE, chunk_size=9)
    for i, K in enumerate(Ks):
        for j, T in enumerate(Ts):
            single = orc.continuous_node_residual(K, T, 0.1, S0, V0, HESTON, JUMPS_NONE)
            assert abs(batched.q[i, j] - single.q) < 1.0e-10
            assert abs(batched.ell[i, j] - single.ell) < 1.0e-10
            assert abs(batched.R[i, j] - single.R) < 1.0e-10


# ===========================================================================
# QE simulation: martingale property, and the specified MC martingale test.
# ===========================================================================


def test_qe_martingale_property_heston():
    step = orc.simulate_paths(S0, V0, 0.25, 12, 100000, HESTON, JUMPS_NONE, seed=1)
    se = step.S.std(ddof=1) / math.sqrt(len(step.S))
    assert abs(step.S.mean() - S0) <= 5.0 * se  # within ~5 std errors


def test_qe_martingale_property_bates():
    step = orc.simulate_paths(S0, V0, 0.25, 12, 100000, HESTON, JUMPS_BATES, seed=2)
    se = step.S.std(ddof=1) / math.sqrt(len(step.S))
    assert abs(step.S.mean() - S0) <= 5.0 * se


def test_mc_martingale_test_contains_zero():
    r = orc.martingale_test(100.0, 1.0, 0.1, S0, V0, 1.0 / 52.0, 4, 50000, HESTON, JUMPS_NONE, seed=42)
    assert r.ci95[0] <= 0.0 <= r.ci95[1]


# ===========================================================================
# Regression: extreme Gauss-Hermite jump-quadrature nodes shifting the spot
# to a degenerate "deep wing" region blew up a finite-difference ell field
# to +-inf, poisoning the weighted jump-integral sum to NaN (found via the
# full-grid export sweep at t=0.0, n_idx=1, m_idx=2 -- not caught by the
# small parametrized grid above, which happened not to hit this combination).
# ===========================================================================


def test_bates_finite_grid_no_nan_regression():
    """This specific (t, n_idx, m_idx) combination produced R=nan before the
    delta_y non-finite guard was added."""
    r = orc.finite_grid_node_residual(1, 2, 0.0, S0, V0, HESTON, JUMPS_BATES)
    assert math.isfinite(r.R)
    assert math.isfinite(r.R_nojump)
    assert abs(r.R) <= 1.0e-5


def test_bates_finite_grid_full_sweep_no_nan():
    """Every (interior strike, back-slab expiry, observation time) combination
    must produce finite R -- the bug above was only caught by sweeping the
    FULL grid, not the small parametrized subset tested elsewhere in this
    file, so this test exercises the full grid explicitly (still fast: the
    finite grid is only 19x3 nodes per observation time)."""
    for t in orc.DEFAULT_OBSERVATION_TIMES:
        for n_idx in orc.FIXED_STRIKE_INTERIOR_INDICES:
            for m_idx in orc.FIXED_BACK_SLAB_INDICES:
                r = orc.finite_grid_node_residual(n_idx, m_idx, t, S0, V0, HESTON, JUMPS_BATES)
                assert math.isfinite(r.R), f"NaN/Inf at t={t}, n={n_idx}, m={m_idx}"
                assert math.isfinite(r.R_nojump), f"NaN/Inf (nojump) at t={t}, n={n_idx}, m={m_idx}"
