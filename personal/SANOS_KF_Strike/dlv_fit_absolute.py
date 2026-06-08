"""
Self-contained single-expiry SANOS density fitter.
Absolute-strike variant: K values are in dollars (not K/F); prices are C/df.

Solves for the risk-neutral density d over a fixed strike grid that
minimizes weighted bid-ask excess (and optionally mid-price squared error),
subject to:
  - d >= 0,  sum(d) = 1,  sum(K_i * d_i) = fwd   (valid probability + martingale)
  - DLV box bounds: dC in [min_dlvsq, max_dlvsq] * box * d
  - log-barrier or second-difference smoothness regularizer
  - temporal regularizer anchoring d to the previous snapshot

All prices are normalized by df only (C/df). Strikes are absolute dollar values.
Put-call parity: C/df = P/df + F - K.

DLV² is scale-free: both dC and box scale with F, so their ratio is unchanged
relative to the K/F-normalized version.  The same dlv_min/dlv_max bounds apply.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
from dlv_grid_absolute import dlvbox_weights


class SolverError(Exception):
    pass


def fit_density(
    *,
    bid_norm: np.ndarray,
    ask_norm: np.ndarray,
    pure_K: np.ndarray,
    is_calls: np.ndarray,
    xstrikes_row: np.ndarray,
    grid_K: np.ndarray,
    density0: np.ndarray,
    tte_geom: float,
    fwd: float,
    dlv_min: float = 0.001,
    dlv_max: float = 2.0,
    min_half_spread: float = 1e-5,
    mid_fit_weight: float = 0.1,
    regularizer_weight: float = 1e-6,
    log_dlv_space: bool = True,
    temporal_regularizer_weight: float = 1.0,
    temporal_weight_clip: float = 100.0,
    prev_density: np.ndarray | None = None,
    bid_ask_penalty_weight: float = 0.0,
    solver: str = "CLARABEL",
    max_error: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a single-expiry SANOS risk-neutral density in absolute strike space.

    Parameters
    ----------
    bid_norm, ask_norm : (M,) option prices divided by df  (C/df or P/df)
    pure_K    : (M,) absolute dollar strikes
    is_calls  : (M,) bool, True = call, False = put
    xstrikes_row : (N+2,) full strike grid including boundary anchors (dollars)
    grid_K    : (N,) interior anchors — snapped version from make_density0
    density0  : (N,) initial density (point mass at fwd, from make_density0)
    tte_geom  : model expiry used as dt in dlvbox (slightly above market tte)
    fwd       : forward price for this snapshot — used in martingale and PCP
    prev_density : (N,) density from previous snapshot for temporal regularizer

    Returns
    -------
    density    : (N,) fitted risk-neutral density
    dlvsqs     : (N-2,) DLV squared, net of dlv_min^2
    fit_prices : (M,) fitted option prices in C/df (or P/df) convention
    """
    N = len(grid_K)

    # convert puts to calls via PCP: C/df = P/df + F - K
    opt_call_bid = np.where(is_calls, bid_norm, bid_norm + fwd - pure_K)
    opt_call_ask = np.where(is_calls, ask_norm, ask_norm + fwd - pure_K)
    opt_call_mid = 0.5 * (opt_call_bid + opt_call_ask)

    half_spread = np.maximum(0.5 * (opt_call_ask - opt_call_bid), min_half_spread)
    inv_hspread = 1.0 / half_spread

    # CVXPY density variable
    d = cp.Variable(N, nonneg=True)

    # basic probability constraints — martingale target is fwd (not 1.0)
    constraints = [
        cp.sum(d) == 1.0,
        d <= 1.0,
        cp.sum(cp.multiply(grid_K, d)) == fwd,
    ]

    # DLV box constraint
    min_dlvsq = float(dlv_min**2)
    max_dlvsq = float(dlv_max**2)
    box = dlvbox_weights(xstrikes_row, tte_geom)  # (N,)

    payoff_at_grid = np.maximum(grid_K[np.newaxis, :] - grid_K[:, np.newaxis], 0.0)  # (N, N)
    mcalls_t = payoff_at_grid @ d  # (N,) CVXPY

    # call prices at grid strikes under a point mass at the current fwd:
    #   mcalls_tm1[i] = max(fwd - grid_K[i], 0)
    # Using fwd directly (not density0) ensures dC[0] = 0 at the low boundary
    # anchor for any per-snapshot fwd, which is required by box[0]=0.
    # (density0 is fixed at ref_fwd; any snapshot with fwd ≠ ref_fwd would
    # make the boundary infeasible if we used density0 here.)
    mcalls_tm1 = np.maximum(fwd - grid_K, 0.0)

    dC = mcalls_t - mcalls_tm1  # (N,) CVXPY
    box_d = cp.multiply(box, d)  # (N,) CVXPY
    constraints += [
        dC >= min_dlvsq * box_d,
        dC <= max_dlvsq * box_d,
    ]

    # model call prices at market strikes
    fit_mat = np.maximum(grid_K[np.newaxis, :] - pure_K[:, np.newaxis], 0.0)  # (M, N)
    fit_calls = fit_mat @ d  # (M,) CVXPY

    # objective
    if mid_fit_weight == 1.0:
        objective = cp.sum(cp.multiply(inv_hspread, fit_calls - opt_call_mid) ** 2)
    elif mid_fit_weight == 0.0:
        objective = cp.sum(
            cp.multiply(
                inv_hspread,
                cp.pos(fit_calls - opt_call_ask) + cp.pos(opt_call_bid - fit_calls),
            )
        )
    else:
        err_sqr = cp.sum(cp.multiply(inv_hspread, fit_calls - opt_call_mid) ** 2)
        excess = cp.sum(
            cp.multiply(
                inv_hspread,
                cp.pos(fit_calls - opt_call_ask) + cp.pos(opt_call_bid - fit_calls),
            )
        )
        objective = mid_fit_weight * err_sqr + (1.0 - mid_fit_weight) * excess

    if bid_ask_penalty_weight > 0.0:
        outside = cp.pos(fit_calls - opt_call_ask) + cp.pos(opt_call_bid - fit_calls)
        objective = objective + bid_ask_penalty_weight * cp.sum_squares(outside)

    if regularizer_weight > 0.0:
        if log_dlv_space:
            objective = objective - regularizer_weight * cp.sum(cp.log(d + 1e-10))
        elif N >= 3:
            second_diff = d[:-2] - 2 * d[1:-1] + d[2:]
            objective = objective + regularizer_weight * cp.sum_squares(second_diff)

    if temporal_regularizer_weight > 0.0 and prev_density is not None:
        prev_d = np.asarray(prev_density, dtype=np.float64)
        if prev_d.shape == (N,):
            eps_floor = np.mean(prev_d) / temporal_weight_clip
            inv_prev = 1.0 / np.maximum(prev_d, eps_floor)
            objective = objective + temporal_regularizer_weight * cp.sum_squares(
                cp.multiply(inv_prev, d - prev_d)
            )

    prob = cp.Problem(cp.Minimize(objective), constraints)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Solution may be inaccurate.*", category=UserWarning
        )
        try:
            prob.solve(solver=solver)
        except cp.SolverError as e:
            raise SolverError(str(e)) from e

    if prob.status == "infeasible":
        raise SolverError("Optimization problem is infeasible.")
    if d.value is None:
        raise SolverError(f"Solver returned status '{prob.status}' with no solution.")

    density = d.value

    if max_error is not None:
        fit_c = fit_calls.value
        err = inv_hspread * (
            np.clip(fit_c - opt_call_ask, 0.0) + np.clip(opt_call_bid - fit_c, 0.0)
        )
        if err.max() > max_error:
            raise SolverError(
                f"Max relative error {err.max():.4e} exceeds max_error {max_error:.4e}."
            )

    dlvsqs = _compute_dlvsqs(
        dC_values=dC.value,
        box_d_values=box_d.value,
        min_dlvsq=min_dlvsq,
        max_dlvsq=max_dlvsq,
    )

    fit_call_prices = fit_calls.value
    # PCP back to put prices: P/df = C/df + K - F
    fit_prices = np.where(is_calls, fit_call_prices, fit_call_prices + pure_K - fwd)

    return density, dlvsqs, fit_prices


def _compute_dlvsqs(
    dC_values: np.ndarray,
    box_d_values: np.ndarray,
    min_dlvsq: float,
    max_dlvsq: float,
    eps_dlvsq: float = 1e-6,
) -> np.ndarray:
    """
    Extract DLV^2 from the fitted dC and box*density values.
    Identical to the normalized version — DLV² is scale-free.
    """
    dC = dC_values[1:-1]
    box_d = box_d_values[1:-1]

    dC_zero = dC <= eps_dlvsq
    too_small = dC <= min_dlvsq * box_d
    too_big = dC >= max_dlvsq * box_d

    divisor = np.where(too_small | dC_zero | too_big, 1.0, box_d)
    dlvsqs = np.select(
        [dC_zero, too_small, too_big],
        [min_dlvsq, min_dlvsq, max_dlvsq],
        default=dC / divisor,
    )
    dlvsqs = np.where(np.isfinite(dlvsqs), dlvsqs, min_dlvsq)
    dlvsqs = np.clip(dlvsqs, min_dlvsq, max_dlvsq) - min_dlvsq
    return dlvsqs
