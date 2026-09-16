"""Task 2: SANOS reconstruction from the Task-1 Heston/Bates oracle export.

Reads ONLY grid_visible.npz. This module must never open dense_hidden.npz --
that file is reserved exclusively for the final evaluation step (Step 6,
implemented in evaluation.py), used only after every choice here (anchors,
smoothing, regularisation, constraints, hyperparameters) has been frozen.
`tests/test_no_hidden_leakage.py` enforces this structurally.

SANOS mixture representation (per spec):
    H_m(K) = sum_i q_i^m * B(a_i^m, K, v_m)
B = Black-Scholes call with spot-like anchor a, strike K, and TOTAL VARIANCE
v (not vol, not vol^2*tau -- v itself), r=q=0. Robust v=0 limit:
B(a,K,0) = (a-K)^+ exactly (no division by sqrt(v)=0).

Four distinct constructions (spec; do not conflate):
  1. Exact finite-law closure (this module, Step 2): v=0, atoms at 21
     anchors (0, the 19 visible strikes, a right tail anchor R), EXACT
     reproduction of the visible grid by construction.
  2. Fixed-kernel linear density lift (Step 3): a DETERMINISTIC v* (not
     grid-calibrated), does NOT generally reproduce the grid exactly.
  3. Fixed-kernel grid-calibrated SANOS (Step 4): v* fixed but ANCHORS/
     WEIGHTS constrained-recalibrated to hit the grid.
  4. State-dependent-kernel SANOS (Step 5): v_m(t,S,V), statically
     arbitrage-free snapshot by snapshot, NOT dynamically validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

VISIBLE_PATH = "grid_visible.npz"

MODELS: tuple[str, ...] = ("heston", "bates")
N_OBS_TIMES = 4
N_SLABS = 3  # back-slab expiries T=0.5, 1.0, 2.0 (m_idx = 1, 2, 3 in oracle.py's convention)


# ===========================================================================
# Data loading (grid_visible.npz ONLY -- see module docstring).
# ===========================================================================


@dataclass(frozen=True)
class VisibleSlice:
    """One (model, observation-time) slice of the visible grid: 19 strikes x
    3 back-slab expiries, plus the state (t, S, V) at which it was observed."""

    model: str
    t_index: int
    t: float
    S: float
    V: float
    K_n: np.ndarray  # (19,)
    T_m: np.ndarray  # (3,)
    C: np.ndarray  # (19, 3)
    q: np.ndarray  # (19, 3)
    ell: np.ndarray  # (19, 3)
    mask: np.ndarray  # (19, 3) bool, admissibility (q>q_min and ell>=0)


def load_visible_slice(model: str, t_index: int, path: str = VISIBLE_PATH) -> VisibleSlice:
    if "hidden" in path:
        raise ValueError("load_visible_slice must never be pointed at a hidden export")
    d = np.load(path)
    prefix = f"{model}_t{t_index}_"
    return VisibleSlice(
        model=model, t_index=t_index, t=float(d[prefix + "t"]), S=float(d[prefix + "S"]), V=float(d[prefix + "V"]),
        K_n=d[prefix + "K_n"], T_m=d[prefix + "T_m"], C=d[prefix + "C"], q=d[prefix + "q"], ell=d[prefix + "ell"],
        mask=d[prefix + "mask"],
    )


def load_all_visible_slices(path: str = VISIBLE_PATH) -> list[VisibleSlice]:
    return [load_visible_slice(model, ti, path) for model in MODELS for ti in range(N_OBS_TIMES)]


# ===========================================================================
# The SANOS mixture kernel B(a, K, v).
# ===========================================================================


def bs_mixture_call(a, K, v):
    """B(a,K,v): Black-Scholes call, anchor a (spot-like), strike K, TOTAL
    VARIANCE v, r=q=0. a<=0 contributes exactly 0 for any K>0, v>=0 (an
    a<=0 atom represents "underlying at or below zero", worthless for any
    positive-strike call). Robust v=0 limit: B(a,K,0)=(a-K)^+ exactly."""
    a = np.asarray(a, dtype=float)
    K = np.asarray(K, dtype=float)
    v = np.asarray(v, dtype=float)
    intrinsic = np.maximum(a - K, 0.0)
    a_pos = np.maximum(a, 1.0e-300)  # guard only; multiplied out below where a<=0
    with np.errstate(divide="ignore", invalid="ignore"):
        sv = np.sqrt(np.maximum(v, 0.0))
        sv_safe = np.where(sv > 0.0, sv, 1.0)
        d1 = (np.log(a_pos / K) + 0.5 * v) / sv_safe
        d2 = d1 - sv
    bs_price = a * norm.cdf(d1) - K * norm.cdf(d2)
    price = np.where(v > 0.0, bs_price, intrinsic)
    price = np.where(a > 0.0, price, 0.0)
    return price


def sanos_surface(anchors: np.ndarray, weights: np.ndarray, K: np.ndarray, v) -> np.ndarray:
    """H(K) = sum_i weights[i] * B(anchors[i], K, v). K may be an array;
    v may be scalar (fixed-kernel) or per-anchor (state-dependent kernel)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    anchors = np.asarray(anchors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        prices = bs_mixture_call(anchors[:, None], K[None, :], v)  # (n_anchors, n_K)
    else:
        prices = bs_mixture_call(anchors[:, None], K[None, :], v[:, None])
    return weights @ prices


# ===========================================================================
# Step 2: exact finite-law closure (v=0), one back slab at a time.
# ===========================================================================

#: Sec.-required sensitivity set: R = K_max + multiplier * strike_step.
RIGHT_ANCHOR_MULTIPLIERS: tuple[int, ...] = (1, 2, 4, 8)
STRIKE_STEP = 5.0

#: Frozen choice for Steps 3-5 (any construction needing a single right-tail
#: treatment), chosen from the Step-2 sensitivity sweep over the VISIBLE
#: grid alone (never touches dense_hidden.npz): multiplier=1 fails
#: non-negativity on 20/24 (model, t, slab) combos, multiplier=2 on 4/24,
#: multiplier=4 is clean but with a razor-thin margin (worst min_weight
#: 9.5e-8 across the full sweep) -- multiplier=8 is the smallest choice
#: tested with a comfortable non-negativity margin. See README.md.
DEFAULT_RIGHT_ANCHOR_MULTIPLIER = 8


@dataclass(frozen=True)
class ExactClosureResult:
    model: str
    t_index: int
    slab_index: int  # 0, 1, 2 (back-slab position, not oracle.py's 1-indexed m_idx)
    right_anchor: float
    right_anchor_multiplier: int
    anchors: np.ndarray  # (21,)
    weights: np.ndarray  # (21,)
    mass: float
    mean: float
    min_weight: float
    valid: bool  # min_weight >= -tol (non-negative discrete density)
    max_reconstruction_error: float  # max_n |H(K_n) - C_n|, must be ~0 by construction


def exact_finite_law_closure(
    K: np.ndarray, C: np.ndarray, S: float, right_anchor: float, tol: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Breeden-Litzenberger discrete closure of a piecewise-linear call curve:
    21 anchors (0, the 19 visible strikes, right_anchor), atoms (v=0) at
    each. Weights are exactly the slope jumps of the augmented curve
    (C(0)=S, C(right_anchor)=0). EXACTLY reproduces C at the 19 visible
    strikes by construction, for ANY right_anchor > K[-1] -- see README.md
    for the algebraic proof (telescoping slope-difference identity)."""
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    if right_anchor <= K[-1]:
        raise ValueError("right_anchor must exceed the largest visible strike")
    X = np.concatenate([[0.0], K, [right_anchor]])
    Ca = np.concatenate([[S], C, [0.0]])
    slopes = np.diff(Ca) / np.diff(X)
    w = np.empty_like(X)
    w[0] = 1.0 + slopes[0]
    w[1:-1] = slopes[1:] - slopes[:-1]
    w[-1] = -slopes[-1]

    recon = sanos_surface(X, w, K, 0.0)
    diagnostics = dict(
        mass=float(w.sum()), mean=float(w @ X), min_weight=float(w.min()),
        valid=bool(w.min() >= -tol), max_reconstruction_error=float(np.max(np.abs(recon - C))),
    )
    return X, w, diagnostics


def frozen_closure(slice_: VisibleSlice, slab_index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """The Step-2 exact closure at the FROZEN DEFAULT_RIGHT_ANCHOR_MULTIPLIER,
    reused unchanged as the anchor/weight basis for Steps 3-4."""
    K, C, S = slice_.K_n, slice_.C[:, slab_index], slice_.S
    R = float(K[-1]) + DEFAULT_RIGHT_ANCHOR_MULTIPLIER * STRIKE_STEP
    return exact_finite_law_closure(K, C, S, R)


def exact_closure_sensitivity_sweep(
    slice_: VisibleSlice, slab_index: int, multipliers: tuple[int, ...] = RIGHT_ANCHOR_MULTIPLIERS,
) -> list[ExactClosureResult]:
    """Sec.-required sensitivity analysis: run the exact closure at >=3
    (here 4) right-anchor choices and report how min_weight (admissibility)
    and the tail masses change. Mass/mean are proved (and verified) to be
    invariant to R; the TAIL SHAPE is NOT -- this is honestly reported, not
    hidden."""
    K, C, S = slice_.K_n, slice_.C[:, slab_index], slice_.S
    K_max = float(K[-1])
    results = []
    for m in multipliers:
        R = K_max + m * STRIKE_STEP
        X, w, diag = exact_finite_law_closure(K, C, S, R)
        results.append(ExactClosureResult(
            model=slice_.model, t_index=slice_.t_index, slab_index=slab_index,
            right_anchor=R, right_anchor_multiplier=m, anchors=X, weights=w, **diag,
        ))
    return results


# ===========================================================================
# Step 3: fixed-kernel linear density lift.
#
# Reuses Step 2's (anchors, weights) EXACTLY (the frozen right-anchor
# closure), unchanged -- only the kernel width v changes, from the atomic
# v=0 to a FIXED, deterministically-chosen v* > 0, the SAME for every
# (model, t, slab). Since weights are not refit, this does NOT generally
# reproduce the grid: H(K_n; v*) != C_n in general. "Linear" refers to the
# map weights -> H(.;v*) being linear (a fixed-kernel mixture), not to any
# claim of exactness.
#
# v* choice (deterministic, NOT grid-calibrated -- a function of the FIXED,
# already-known strike grid alone, never of any specific C/q/ell values):
# one "kernel bandwidth = one strike-step" rule, v* = (STRIKE_STEP/S0)^2,
# with S0 the repo-wide default spot (100.0). This is a standard
# histogram-bin-width-as-KDE-bandwidth heuristic, fixed once, globally.
# ===========================================================================

DEFAULT_S0 = 100.0
DEFAULT_V_STAR: float = (STRIKE_STEP / DEFAULT_S0) ** 2


@dataclass(frozen=True)
class FixedKernelLiftResult:
    model: str
    t_index: int
    slab_index: int
    v_star: float
    anchors: np.ndarray
    weights: np.ndarray
    H_at_strikes: np.ndarray  # (19,), the lifted (smoothed) surface at the visible strikes
    max_abs_price_error: float  # max_n |H(K_n;v*) - C_n|
    rms_price_error: float
    max_abs_price_error_normalized: float  # normalized by S


def fixed_kernel_density_lift(
    slice_: VisibleSlice, slab_index: int, v_star: float = DEFAULT_V_STAR,
) -> FixedKernelLiftResult:
    """Step 3: lift the Step-2 discrete law to a smooth mixture with a FIXED
    kernel width v_star, without refitting weights. Reports the resulting
    (expected, non-zero) mismatch against the visible grid."""
    X, w, _ = frozen_closure(slice_, slab_index)
    K, C, S = slice_.K_n, slice_.C[:, slab_index], slice_.S
    H = sanos_surface(X, w, K, v_star)
    err = H - C
    return FixedKernelLiftResult(
        model=slice_.model, t_index=slice_.t_index, slab_index=slab_index, v_star=v_star,
        anchors=X, weights=w, H_at_strikes=H,
        max_abs_price_error=float(np.max(np.abs(err))),
        rms_price_error=float(np.sqrt(np.mean(err ** 2))),
        max_abs_price_error_normalized=float(np.max(np.abs(err)) / S),
    )
