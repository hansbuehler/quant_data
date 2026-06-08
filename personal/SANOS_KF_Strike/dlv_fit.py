"""
Self-contained single-expiry SANOS density fitter.

Solves for the risk-neutral density d over a fixed strike grid that
minimizes weighted bid-ask excess (and optionally mid-price squared error),
subject to:
  - d >= 0,  sum(d) = 1,  sum(K_i * d_i) = 1   (valid probability + martingale)
  - DLV box bounds: dC in [min_dlvsq, max_dlvsq] * box * d
  - log-barrier or second-difference smoothness regularizer
  - temporal regularizer anchoring d to the previous snapshot

All prices are normalized (divided by F * df). Strikes are K/F values.

Difference from quantdata.sanos.fitter.Fitter
----------------------------------------------
This module is self-contained: no quantdata dependency. It targets the
T=1 (single-expiry) case only and uses the simplified call price formula
  C(K) = max(K_i - K, 0) @ d   (fwd_vols=None, no BS smoothing)
rather than the general time-interpolated pricing used by GridSanos.price.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
from dlv_grid import dlvbox_weights


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
    Fit a single-expiry SANOS risk-neutral density.

    Parameters
    ----------
    bid_norm, ask_norm : (M,) normalized option prices (calls or puts)
    pure_K    : (M,) K/F values of the market options
    is_calls  : (M,) bool, True = call, False = put
    xstrikes_row : (N+2,) full strike grid including boundary anchors
    grid_K    : (N,) interior anchors — must be the snapped version from
                make_density0 so that density0[ix] is exactly at 1.0
    density0  : (N,) initial density (point mass at 1.0, from make_density0)
    tte_geom  : model expiry used as dt in dlvbox (slightly above market tte)
    prev_density : (N,) density from previous snapshot for temporal regularizer

    Returns
    -------
    density    : (N,) fitted risk-neutral density
    dlvsqs     : (N-2,) DLV squared, net of dlv_min^2, for the N-2 interior
                 anchors (strips the two boundary anchors where box=0)
    fit_prices : (M,) fitted option prices (same call/put convention as input)
    """
    N = len(grid_K)
    M = len(pure_K)

    # convert puts to calls via PCP for a unified objective
    opt_call_bid = np.where(is_calls, bid_norm, bid_norm + 1.0 - pure_K)
    opt_call_ask = np.where(is_calls, ask_norm, ask_norm + 1.0 - pure_K)
    opt_call_mid = 0.5 * (opt_call_bid + opt_call_ask)

    half_spread = np.maximum(0.5 * (opt_call_ask - opt_call_bid), min_half_spread)
    inv_hspread = 1.0 / half_spread

    # CVXPY density variable
    d = cp.Variable(N, nonneg=True)

    # basic probability constraints
    constraints = [
        cp.sum(d) == 1.0,
        d <= 1.0,
        cp.sum(cp.multiply(grid_K, d)) == 1.0,
    ]

    # DLV box constraint
    # dC[i] = call_t(K_i) - call_0(K_i)
    #        = max(K_j - K_i, 0) @ d  -  max(1 - K_i, 0)
    # where the second term comes from density0 being a point mass at 1.0
    min_dlvsq = float(dlv_min**2)
    max_dlvsq = float(dlv_max**2)
    box = dlvbox_weights(xstrikes_row, tte_geom)  # (N,)

    payoff_at_grid = np.maximum(
        grid_K[np.newaxis, :] - grid_K[:, np.newaxis], 0.0
    )  # (N, N): row i = payoffs of call with strike K_i evaluated at each anchor K_j
    mcalls_t = payoff_at_grid @ d  # (N,) CVXPY: call prices at grid strikes under d

    # call prices at grid strikes under density0 (point mass at 1.0)
    mcalls_tm1 = np.maximum(
        density0 @ np.maximum(grid_K[:, np.newaxis] - grid_K[np.newaxis, :], 0.0), 0.0
    )
    # simplified: since density0 is a point mass at index ix where grid_K[ix] = 1.0:
    #   mcalls_tm1[i] = max(1.0 - grid_K[i], 0.0)
    # The matrix form above is equivalent and handles the general case cleanly.

    dC = mcalls_t - mcalls_tm1  # (N,) CVXPY
    box_d = cp.multiply(box, d)  # (N,) CVXPY
    constraints += [
        dC >= min_dlvsq * box_d,
        dC <= max_dlvsq * box_d,
    ]

    # model call prices at market strikes: C(K) = max(K_i - K, 0) @ d
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

    # regularizer: log-barrier (max-entropy) or second-difference smoothness
    if regularizer_weight > 0.0:
        if log_dlv_space:
            # -lambda * sum(log(d_i)): convex, penalises small densities uniformly in log-space
            objective = objective - regularizer_weight * cp.sum(cp.log(d + 1e-10))
        elif N >= 3:
            second_diff = d[:-2] - 2 * d[1:-1] + d[2:]
            objective = objective + regularizer_weight * cp.sum_squares(second_diff)

    # temporal regularizer: lambda_T * sum((d_i - d_hat_i)^2 / d_hat_i^2)
    # relative L2 in density space ≈ log-DLV continuity for small changes
    if temporal_regularizer_weight > 0.0 and prev_density is not None:
        prev_d = np.asarray(prev_density, dtype=np.float64)
        if prev_d.shape == (N,):
            eps_floor = np.mean(prev_d) / temporal_weight_clip
            inv_prev = 1.0 / np.maximum(prev_d, eps_floor)
            objective = objective + temporal_regularizer_weight * cp.sum_squares(
                cp.multiply(inv_prev, d - prev_d)
            )

    # solve
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

    # optionally check max bid-ask error
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
    fit_prices = np.where(is_calls, fit_call_prices, fit_call_prices + pure_K - 1.0)

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

    Strips the two boundary anchors (where box=0) and returns (N-2,) values,
    clipped to [min_dlvsq, max_dlvsq] and net of min_dlvsq.

    The boundary anchors trivially satisfy dC=0 (their dC constraint is inactive
    since box=0), so stripping them is safe.
    """
    # strip boundary anchors where box=0
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
