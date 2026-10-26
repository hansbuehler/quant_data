r"""Frozen hyperparameters and the model-selection/held-out date split.

Every value here was fixed using ONLY the two selection dates
(SELECTION_DATES), before any held-out-date calibration was run, per
DATA_PROVENANCE.md Section 3. Do not tune anything here after looking at
held-out results.
"""

from __future__ import annotations

from datetime import date

import numpy as np

# ---------------------------------------------------------------------------
# Dates and tickers (DATA_PROVENANCE.md Sections 2-4).
# ---------------------------------------------------------------------------

SELECTION_DATES = (date(2005, 11, 21), date(2010, 7, 7))
HELD_OUT_DATES = (date(2011, 7, 7), date(2024, 3, 5), date(2024, 12, 26), date(2025, 1, 3))
ALL_DATES = SELECTION_DATES + HELD_OUT_DATES

PRIMARY_TICKER = "SPX"
SECONDARY_TICKER = "AAPL"

DY2_SMOOTHNESS = 0.25

# ---------------------------------------------------------------------------
# Common evaluation contract grid (DATA_PROVENANCE.md Section 5).
# ---------------------------------------------------------------------------

TARGET_TENORS_YEARS = (7 / 365.25, 1 / 12, 2 / 12, 3 / 12, 0.5, 1.0)
TARGET_MONEYNESS = (0.85, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15)
LOG_MONEYNESS_TOLERANCE = 0.03

# ---------------------------------------------------------------------------
# Coherent DLV chain (comparison_interface.ChainGeneratorConfig).
# ---------------------------------------------------------------------------

N_ANCHORS = 7
ANCHOR_HALF_WIDTH = 0.45  # anchors = exp(linspace(-half_width, +half_width, n)), n odd -> middle anchor == 1.0 exactly
NODE_COUNT_SWEEP = (5, 7, 9)  # Section C "number of latent nodes" sweep

CHAIN_KAPPA_BOUNDS = (0.05, 300.0)
CHAIN_SIGMA_Y_BOUNDS = (0.02, 3.0)
ANCHOR_HALF_WIDTH_BOUNDS = (0.03, 0.60)  # fitted, not fixed -- see DATA_PROVENANCE.md / REAL_DATA_FINDINGS.md
                                          # on why a single frozen half-width could not span both short-dated
                                          # ATM curvature and far-OTM long-dated coverage simultaneously


def build_anchors(n: int, half_width: float = ANCHOR_HALF_WIDTH) -> tuple:
    """Symmetric geometric anchor grid with X_0=1.0 EXACTLY at the middle
    index -- required for the chain's already-verified mean-preserving
    martingale property (sum_i q_i * x_i = x_{i0}) to mean sum_i q_i*x_i=1,
    matching the pure-unit forward. n must be odd."""
    if n % 2 == 0:
        raise ValueError(f"n must be odd so 1.0 is exactly an anchor, got {n}")
    anchors = tuple(float(v) for v in np.exp(np.linspace(-half_width, half_width, n)))
    initial_index = n // 2
    assert abs(anchors[initial_index] - 1.0) < 1e-12
    return anchors, initial_index

# ---------------------------------------------------------------------------
# Bates background parameters (heston_riccati.HestonParams / GaussianJumpParams).
# ---------------------------------------------------------------------------

BATES_BOUNDS = dict(
    kappa=(0.1, 15.0),
    theta=(0.005, 0.50),
    xi=(0.05, 3.0),
    rho=(-0.99, 0.99),
    lambda_J=(0.0, 5.0),
    m_J=(-0.5, 0.5),
    delta_J=(0.01, 1.0),
)

N_MULTI_START = 4

# ---------------------------------------------------------------------------
# Extended-Bates reserve sweep (Section C).
# ---------------------------------------------------------------------------

RESERVE_RESET_HORIZON_YEARS = 1 / 12  # hypothetical next reset t_1, one month out
RESERVE_BUDGET_SWEEP = (0.0, 0.01, 0.03, 0.06)  # r_1, variance units; 0.0 = degenerate/ordinary Bates
THETA_BAND_HALF_WIDTH = 0.03  # [theta_min, theta_max] = calibrated theta +/- this
JUMP_BOX_WIDEN_FACTOR = 1.5  # jump-box (Lambda, M, delta_min, delta_max, s_star) widened relative to calibrated jump params by this factor

FOURIER_N_QUAD = 200
FOURIER_U_MAX = 250.0

# Two-tier precision (REAL_DATA_FINDINGS.md Sec. 4): full precision above is
# used for FINAL reported prices/metrics; this reduced tier is used INSIDE
# every least-squares optimisation loop (many thousands of price calls),
# then the converged parameters are re-priced once at full precision.
FOURIER_N_QUAD_OPTIMIZATION = 80
FOURIER_U_MAX_OPTIMIZATION = 150.0
