"""Unit tests for sanos.py (Task 2): SANOS reconstruction from grid_visible.npz.

Run with:  PYTHONPATH=.. pytest tests/test_sanos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sanos

ALL_SLICES = sanos.load_all_visible_slices()


# ===========================================================================
# Data loading / no-leakage sanity.
# ===========================================================================


def test_load_visible_slice_rejects_hidden_path():
    with pytest.raises(ValueError):
        sanos.load_visible_slice("heston", 0, path="dense_hidden.npz")


def test_all_slices_loaded():
    assert len(ALL_SLICES) == len(sanos.MODELS) * sanos.N_OBS_TIMES


def test_slice_shapes():
    for sl in ALL_SLICES:
        assert sl.K_n.shape == (19,)
        assert sl.T_m.shape == (3,)
        assert sl.C.shape == (19, 3)
        assert sl.q.shape == (19, 3)
        assert sl.ell.shape == (19, 3)


# ===========================================================================
# bs_mixture_call kernel: robust v=0 limit, a<=0 atoms contribute 0.
# ===========================================================================


def test_bs_mixture_call_v0_matches_intrinsic():
    a, K = np.array([90.0, 100.0, 110.0]), np.array([100.0])
    price = sanos.bs_mixture_call(a[:, None], K[None, :], 0.0)
    expected = np.maximum(a[:, None] - K[None, :], 0.0)
    assert np.allclose(price, expected)


def test_bs_mixture_call_zero_anchor_is_worthless():
    price = sanos.bs_mixture_call(0.0, np.array([50.0, 100.0]), 0.25)
    assert np.allclose(price, 0.0)


def test_bs_mixture_call_positive_v_exceeds_intrinsic_atm():
    """Genuine positive time value away from v=0, at the money."""
    price = sanos.bs_mixture_call(100.0, 100.0, 0.04)
    assert price > 0.0  # ATM call always has positive time value for v>0


def test_bs_mixture_call_deep_itm_converges_to_intrinsic_as_v_shrinks():
    a, K = 150.0, 100.0
    p_small = float(sanos.bs_mixture_call(a, K, 1.0e-8))
    assert abs(p_small - (a - K)) < 1.0e-3


# ===========================================================================
# Step 2: exact finite-law closure.
# ===========================================================================


@pytest.mark.parametrize("sl", ALL_SLICES, ids=lambda s: f"{s.model}_t{s.t_index}")
@pytest.mark.parametrize("slab", [0, 1, 2])
def test_exact_closure_reproduces_grid_exactly(sl, slab):
    if sl.T_m[slab] - sl.t <= 0.0:
        pytest.skip("tau<=0")
    K, C, S = sl.K_n, sl.C[:, slab], sl.S
    R = K[-1] + sanos.DEFAULT_RIGHT_ANCHOR_MULTIPLIER * sanos.STRIKE_STEP
    _, _, diag = sanos.exact_finite_law_closure(K, C, S, R)
    assert diag["max_reconstruction_error"] < 1.0e-9


@pytest.mark.parametrize("sl", ALL_SLICES, ids=lambda s: f"{s.model}_t{s.t_index}")
@pytest.mark.parametrize("slab", [0, 1, 2])
def test_exact_closure_mass_and_mean_are_anchor_invariant(sl, slab):
    """Mass=1, mean=S hold EXACTLY regardless of which right anchor is used
    (algebraic identity of the slope-jump construction, not a numerical
    coincidence) -- verified across every tested multiplier."""
    if sl.T_m[slab] - sl.t <= 0.0:
        pytest.skip("tau<=0")
    K, C, S = sl.K_n, sl.C[:, slab], sl.S
    for m in sanos.RIGHT_ANCHOR_MULTIPLIERS:
        R = K[-1] + m * sanos.STRIKE_STEP
        _, _, diag = sanos.exact_finite_law_closure(K, C, S, R)
        assert abs(diag["mass"] - 1.0) < 1.0e-9
        assert abs(diag["mean"] - S) < 1.0e-6


def test_exact_closure_rejects_right_anchor_inside_grid():
    sl = ALL_SLICES[0]
    K, C, S = sl.K_n, sl.C[:, 0], sl.S
    with pytest.raises(ValueError):
        sanos.exact_finite_law_closure(K, C, S, right_anchor=K[-1])


def test_right_anchor_sensitivity_default_multiplier_is_admissible_everywhere():
    """The frozen DEFAULT_RIGHT_ANCHOR_MULTIPLIER=8 must give a valid
    (non-negative) discrete law across the FULL visible grid -- this is
    exactly the property that justified freezing it in sanos.py."""
    for sl in ALL_SLICES:
        for slab in range(3):
            if sl.T_m[slab] - sl.t <= 0.0:
                continue
            R = sl.K_n[-1] + sanos.DEFAULT_RIGHT_ANCHOR_MULTIPLIER * sanos.STRIKE_STEP
            _, _, diag = sanos.exact_finite_law_closure(sl.K_n, sl.C[:, slab], sl.S, R)
            assert diag["valid"], f"{sl.model} t{sl.t_index} slab{slab}: min_weight={diag['min_weight']}"


def test_right_anchor_sensitivity_small_multiplier_frequently_inadmissible():
    """Regression/documentation: multiplier=1 (R=K_max+5) is NOT safe --
    most (model,t,slab) combinations produce a negative tail weight.
    Confirms the sensitivity sweep has real content, not a vacuous pass."""
    n_fail = 0
    n_total = 0
    for sl in ALL_SLICES:
        for slab in range(3):
            if sl.T_m[slab] - sl.t <= 0.0:
                continue
            n_total += 1
            R = sl.K_n[-1] + 1 * sanos.STRIKE_STEP
            _, _, diag = sanos.exact_finite_law_closure(sl.K_n, sl.C[:, slab], sl.S, R)
            if not diag["valid"]:
                n_fail += 1
    assert n_fail >= n_total // 2  # a clear majority, not an edge case


def test_exact_closure_sensitivity_sweep_helper():
    sl = ALL_SLICES[0]
    results = sanos.exact_closure_sensitivity_sweep(sl, slab_index=0)
    assert len(results) == len(sanos.RIGHT_ANCHOR_MULTIPLIERS)
    for r in results:
        assert abs(r.mass - 1.0) < 1.0e-9
        assert abs(r.mean - sl.S) < 1.0e-6
