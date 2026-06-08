"""
Strike grid construction for single-expiry SANOS density fitting.
Absolute-strike variant: K values are in dollars (not K/F).
"""

from __future__ import annotations

import numpy as np


def make_strike_grid(
    obs_strikes_abs: np.ndarray,
    fwd: float,
    lower_scale: float = 0.9,
    upper_scale: float = 1.1,
    n_pad: int = 1,
) -> dict:
    """
    Build the SANOS strike grid in absolute strike space.

    The grid has:
      - one boundary anchor at lo = lower_scale * obs[0]
      - n_pad padding anchors between lo and the first observed strike
      - all observed strikes (in dollars)
      - n_pad padding anchors between the last observed strike and hi
      - one boundary anchor at hi = upper_scale * obs[-1]

    Parameters
    ----------
    obs_strikes_abs : sorted absolute dollar strikes
    fwd             : current forward price (used for ATM assertions)
    lower_scale, upper_scale : boundary = scale * extreme observed strike
    n_pad : number of padding anchors on each side

    Returns
    -------
    dict with keys:
        xstrikes_row  : (N+2,) full grid including boundary anchors (dollars)
        anchor_pure   : (N,)   interior anchors = xstrikes_row[1:-1]
        interior_pure : (N-2*n_pad,) observed anchors excluding pads
        lo, hi        : float boundary values
    """
    obs_ks = np.sort(np.asarray(obs_strikes_abs, dtype=np.float64))
    lo = obs_ks[0] * lower_scale
    hi = obs_ks[-1] * upper_scale

    lo_pads = np.linspace(lo, obs_ks[0], n_pad + 2)[1:-1]
    hi_pads = np.linspace(obs_ks[-1], hi, n_pad + 2)[1:-1]

    xstrikes_row = np.concatenate([[lo], lo_pads, obs_ks, hi_pads, [hi]])
    anchor_pure = xstrikes_row[1:-1]
    interior_pure = xstrikes_row[1 + n_pad : -1 - n_pad]

    assert anchor_pure[0] < fwd, "first pad must be below ATM (fwd)"
    assert anchor_pure[-1] > fwd, "last pad must be above ATM (fwd)"

    return dict(
        xstrikes_row=xstrikes_row,
        anchor_pure=anchor_pure,
        interior_pure=interior_pure,
        lo=lo,
        hi=hi,
    )


def make_density0(xstrikes_row: np.ndarray, fwd: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Initial density: point mass at the interior anchor closest to fwd.

    Snaps the nearest anchor to exactly fwd **in-place** on xstrikes_row, so
    that any views derived before this call (e.g. anchor_pure from
    make_strike_grid) also see the snapped value.  This ensures the martingale
    constraint grid_K @ d == fwd and the diagnostic anchor_pure @ d are over
    the same K-vector, giving a residual at solver precision (~1e-14 * fwd).

    Parameters
    ----------
    xstrikes_row : (N+2,) full grid including boundary anchors — modified in place
    fwd          : forward price; snap target

    Returns
    -------
    grid_K   : (N,) view of xstrikes_row[1:-1] with one element snapped to fwd
    density0 : (N,) point-mass density
    """
    grid_K = xstrikes_row[1:-1]  # view — snap propagates to xstrikes_row and all its views
    N = len(grid_K)
    assert grid_K[0] < fwd < grid_K[-1], "grid must straddle fwd"
    ix = int(np.abs(grid_K - fwd).argmin())
    grid_K[ix] = fwd  # in-place snap to forward
    density0 = np.zeros(N)
    density0[ix] = 1.0
    return grid_K, density0


def dlvbox_weights(grid_K_full: np.ndarray, dt: float) -> np.ndarray:
    """
    DLV box weights for each interior anchor:
        box[i] = K_i^2 / (K_{i+1,full} - K_{i-1,full}) * dt

    Identical formula to the normalized version — works correctly with
    absolute K values.  Both dC and box scale linearly with F, so their
    ratio (DLV²) is scale-free and the same dlv_min/dlv_max bounds apply.

    The first and last elements are zeroed because the SANOS boundary condition
    pins dlvbox[0] = dlvbox[-1] = 0, forcing dC = 0 at the boundary anchors.

    Parameters
    ----------
    grid_K_full : (N+2,) full grid including boundary anchors (dollars)
    dt          : model time step (tte_geom)

    Returns
    -------
    (N,) box weights, zeroed at indices 0 and -1
    """
    interior = grid_K_full[1:-1]
    box = interior**2 / (grid_K_full[2:] - grid_K_full[:-2]) * dt
    box[0] = 0.0
    box[-1] = 0.0
    return box
