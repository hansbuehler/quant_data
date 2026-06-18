"""
surface_model.py

Model registry for the spline-state Kalman filter.

Each model is a `Model` instance bundling:
  - observe(theta, k, T)  -> (z_hat, H_jacobian)   used by the EKF update
  - initial_theta(cfg)    -> np.ndarray             starting state
  - q_diag_weights        -> np.ndarray             per-param relative Q scale
                             (used when Q_structure="diagonal")

Available models (swap in the notebook config cell):
  QUAD_SLICE_MODEL   — 3 params, a + b·k + c·k²                       (linear KF, single expiry)
  SVI_SLICE_MODEL    — 5 params, w=a+b·[ρ(k−m)+√((k−m)²+σ²)], IV=√(w/T)  (EKF, single expiry)
  SVI_MODEL          — 5 params, same formula, shared state across T   (EKF, multiple maturities)
  TANH_MODEL         — 3 params, a·√T·(1+tanh(b·k+c·k²))              (EKF, multiple maturities)
  BSPLINE_SLICE_MODEL— 8 params, cubic B-splines on fixed k grid       (linear KF, single expiry)
                       default knots: log-moneyness ∈ [−0.30, +0.15]
                       use make_bspline_slice_model() to customise knots/degree/n_basis

To add a new model: write _observe_xxx / _initial_theta_xxx, then define
XXX_MODEL = Model(name=..., n_params=..., observe=..., initial_theta=..., q_diag_weights=...).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline

# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Model:
    """One parametric form: observation function + initial state."""

    name: str
    n_params: int
    observe: Callable[[np.ndarray, float, float], tuple[float, np.ndarray]]
    initial_theta: Callable[["SurfaceModelConfig"], np.ndarray]
    # Relative per-parameter Q weights used when Q_structure="diagonal".
    # Entry i multiplies Q_scale for parameter i.
    # Rule of thumb: fast-moving params (level) → 1.0, slow params (shape) → 0.05.
    q_diag_weights: np.ndarray = dataclasses.field(default=None)
    # Optional projection applied to θ after every KF update, pulling the state
    # back onto the model's valid parameter domain so the (unconstrained) EKF
    # update cannot diverge into a region where observe() is undefined.
    # None → no projection (state left exactly as the KF update produced it).
    project: Callable[[np.ndarray], np.ndarray] = dataclasses.field(default=None)


# ---------------------------------------------------------------------------
# Model 1 — SVI slice  (5 params, textbook total-variance form)
# w(k) = a + b·[ρ(k−m) + √((k−m)²+σ²)]   (total variance = IV²·T)
# IV(k) = √(w(k) / T)
# Use for a single expiry.
# ---------------------------------------------------------------------------


def _observe_svi_slice(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    Standard Gatheral SVI (total-variance form):
        w(k) = a + b·[ρ·(k−m) + √((k−m)²+σ²)]   (total variance = IV²·T)
        IV(k) = √(w(k) / T)

    θ = [a, b, ρ, m, σ]
    ATM (k=m): w = a + b·σ,  IV = √((a + b·σ) / T)

    Returns (nan, zeros) when w ≤ 0 so the KF skips the update rather than
    taking a sqrt of a negative number.
    """
    a, b, rho, m, sigma = (float(theta[i]) for i in range(5))
    sigma = max(sigma, 1e-6)
    T_safe = max(float(T), 1e-6)

    d = k - m
    sq = np.sqrt(d**2 + sigma**2)
    f = rho * d + sq
    w = a + b * f  # total variance

    if w <= 0:
        return float("nan"), np.zeros(5)

    z_hat = np.sqrt(w / T_safe)  # IV

    # Jacobian: ∂IV/∂θ = (1 / (2·√(w·T))) · ∂w/∂θ
    scale = 1.0 / (2.0 * np.sqrt(w * T_safe))
    H = scale * np.array(
        [
            1.0,  # ∂w/∂a
            f,  # ∂w/∂b
            b * d,  # ∂w/∂ρ
            -b * (rho + d / sq),  # ∂w/∂m
            b * sigma / sq,  # ∂w/∂σ
        ]
    )
    return z_hat, H


def _project_svi(theta: np.ndarray) -> np.ndarray:
    """Project θ = [a, b, ρ, m, σ] back onto the valid SVI domain.

    The EKF update is unconstrained and, around a flat prior with large P0, can
    take a single huge step that lands ρ outside (−1, 1) or drives the total
    variance negative — after which observe() returns NaN, the filter can only
    predict, and θ freezes (all later spline_iv / kalman_price become NaN).
    This clamp keeps every parameter in range so the filter cannot leave the
    SVI manifold:
        b ≥ 0,  σ ≥ σ_min,  ρ ∈ [−0.999, 0.999]
        a ≥ w_floor − b·σ·√(1−ρ²)   ⇒  min_k w(k) ≥ w_floor > 0
    (min_k [ρ·d + √(d²+σ²)] = σ·√(1−ρ²), attained at d = −ρσ/√(1−ρ²).)

    Note: only positivity / sign constraints are imposed. Hard-clamping the
    weakly-identified centre m or width σ to a "sensible" box makes the fit far
    worse, not better — the EKF covariance still treats them as free, so the
    filter fights the clamp every step and drags a/b/ρ into bad values. Their
    drift is controlled through process noise (small q_diag_weights), not here.
    """
    a, b, rho, m, sigma = (float(theta[i]) for i in range(5))
    b = max(b, 0.0)
    sigma = max(sigma, 1e-4)
    rho = min(max(rho, -0.999), 0.999)
    w_floor = 1e-8
    a_min = w_floor - b * sigma * np.sqrt(1.0 - rho**2)
    a = max(a, a_min)
    return np.array([a, b, rho, m, sigma])


def _initial_theta_svi_slice(cfg: "SurfaceModelConfig") -> np.ndarray:
    """a=5e-5, b=5e-4, ρ=−0.7, m=0, σ=0.2
    ATM total variance w ≈ 1.5e-4; for T=0.008 (2-day SPY): ATM IV ≈ 0.137"""
    return np.array([5e-5, 5e-4, -0.7, 0.0, 0.2])


SVI_SLICE_MODEL = Model(
    name="svi_slice",
    n_params=5,
    observe=_observe_svi_slice,
    initial_theta=_initial_theta_svi_slice,
    # level (a) moves constantly with spot; b/ρ move moderately; m/σ near-static
    q_diag_weights=np.array([1.0, 0.2, 0.1, 0.02, 0.02]),
    project=_project_svi,
)


# ---------------------------------------------------------------------------
# Model 2 — SVI multi-expiry  (5 params, textbook total-variance form)
# w(k) = a + b·[ρ(k−m) + √((k−m)²+σ²)]   (total variance = IV²·T)
# IV(k,T) = √(w(k) / T)
# Same formula as SVI_SLICE_MODEL; distinguished by initial_theta (tuned for
# a shared state across maturities) and name.
# ---------------------------------------------------------------------------


def _observe_svi(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    Standard Gatheral SVI (total-variance form), same as SVI_SLICE_MODEL:
        w(k) = a + b·[ρ·(k−m) + √((k−m)²+σ²)]   (total variance)
        IV(k, T) = √(w(k) / T)

    θ = [a, b, ρ, m, σ]
    ATM (k=m): IV = √((a + b·σ) / T), naturally decreasing with T for fixed w.

    Returns (nan, zeros) when w ≤ 0 so the KF skips the update.
    """
    return _observe_svi_slice(theta, k, T)


def _initial_theta_svi(cfg: "SurfaceModelConfig") -> np.ndarray:
    """a=0.001, b=0.005, ρ=−0.7, m=0, σ=0.2
    ATM total variance w ≈ 0.002; for T=0.083 (1-month): ATM IV ≈ 0.155"""
    return np.array([0.001, 0.005, -0.7, 0.0, 0.2])


SVI_MODEL = Model(
    name="svi",
    n_params=5,
    observe=_observe_svi,
    initial_theta=_initial_theta_svi,
    q_diag_weights=np.array([1.0, 0.2, 0.1, 0.02, 0.02]),
    project=_project_svi,
)


# ---------------------------------------------------------------------------
# Model 3 — Quadratic slice  (3 params, no √T)
# IV(k) = a + b·k + c·k²
# Linear in θ → standard KF (no EKF), H = [1, k, k²].
# ---------------------------------------------------------------------------


def _observe_quad_slice(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    IV(k) = a + b·k + c·k²

    θ = [a, b, c]
    a: ATM level,  b: skew slope,  c: smile curvature.
    Linear in θ — z_hat = H·θ exactly, no EKF linearisation error.
    """
    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
    H = np.array([1.0, k, k**2])
    z_hat = a + b * k + c * k**2
    return z_hat, H


def _initial_theta_quad_slice(cfg: "SurfaceModelConfig") -> np.ndarray:
    """Flat 20% smile: a=0.20, b=0, c=0."""
    return np.array([0.20, 0.0, 0.0])


QUAD_SLICE_MODEL = Model(
    name="quad_slice",
    n_params=3,
    observe=_observe_quad_slice,
    initial_theta=_initial_theta_quad_slice,
    q_diag_weights=np.array([1.0, 0.3, 0.1]),
)


# ---------------------------------------------------------------------------
# Model 4 — Tanh smile  (3 params)
# IV(k,T) = a · √T · (1 + tanh(b·k + c·k²))
# ---------------------------------------------------------------------------


def _observe_tanh(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    h(θ) = a · √T · (1 + tanh(b·k + c·k²))

    θ = [a, b, c]
    ATM (k=0): h = a·√T  (always positive for a > 0)
    """
    a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
    sqrtT = np.sqrt(max(T, 1e-6))
    arg = b * k + c * k**2
    t = np.tanh(arg)
    sech2 = 1.0 - t**2
    z_hat = a * sqrtT * (1.0 + t)
    H = np.array(
        [
            sqrtT * (1.0 + t),
            a * sqrtT * sech2 * k,
            a * sqrtT * sech2 * k**2,
        ]
    )
    return z_hat, H


def _initial_theta_tanh(cfg: "SurfaceModelConfig") -> np.ndarray:
    """Flat 20% smile: a=0.40, b=0, c=0.  At T=0.25: h = 0.40·0.5·1 = 0.20"""
    return np.array([0.40, 0.0, 0.0])


TANH_MODEL = Model(
    name="tanh",
    n_params=3,
    observe=_observe_tanh,
    initial_theta=_initial_theta_tanh,
    q_diag_weights=np.array([1.0, 0.2, 0.05]),
)


# ---------------------------------------------------------------------------
# KF hyper-parameters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SurfaceModelConfig:
    model: Model = dataclasses.field(default_factory=lambda: SVI_SLICE_MODEL)
    Q_scale: float = 0.05
    # "scaled_identity" — Q = Q_scale · I  (ignores q_diag_weights)
    # "diagonal"        — Q = diag(Q_scale · model.q_diag_weights)
    Q_structure: str = "diagonal"
    P0_scale: float = 1.0

    @property
    def n_params(self) -> int:
        return self.model.n_params


def build_Q(cfg: SurfaceModelConfig) -> np.ndarray:
    """Process noise covariance Q ∈ ℝ^{n_params × n_params}."""
    n = cfg.n_params
    if cfg.Q_structure == "scaled_identity":
        return cfg.Q_scale * np.eye(n)
    elif cfg.Q_structure == "diagonal":
        if cfg.model.q_diag_weights is not None:
            weights = cfg.model.q_diag_weights
        else:
            weights = np.linspace(1.0, 0.2, n)
        return np.diag(cfg.Q_scale * weights)
    else:
        raise ValueError(f"Unknown Q_structure: {cfg.Q_structure!r}")


def build_P0(cfg: SurfaceModelConfig) -> np.ndarray:
    """Initial state covariance P0 ∈ ℝ^{n_params × n_params}."""
    return cfg.P0_scale * np.eye(cfg.n_params)


# ---------------------------------------------------------------------------
# Model 5 — B-spline smile  (linear KF, no EKF)
# IV(k) = Σᵢ θᵢ · Bᵢ(k)
# H = [B₀(k), …, Bₙ(k)]   — exact, no linearisation error.
# k outside the knot domain is clamped; B-splines form a partition of unity
# so the ATM level is the weighted average of the coefficients over the ATM
# basis functions.
# ---------------------------------------------------------------------------


def make_bspline_slice_model(
    k_min: float = -0.30,
    k_max: float = 0.15,
    n_basis: int = 8,
    degree: int = 3,
    atm_iv: float = 0.20,
) -> Model:
    """
    Cubic B-spline smile: IV(k) = Σᵢ θᵢ · Bᵢ(k).

    Knots are fixed and evenly spaced in log-moneyness. The model is linear in
    θ so the Kalman update is exact (no EKF).  Points outside [k_min, k_max]
    are clamped — the boundary basis function absorbs any extrapolation.

    Parameters
    ----------
    k_min, k_max : log-moneyness domain (default: −0.30 to +0.15 for SPX)
    n_basis      : number of B-spline basis functions (controls flexibility)
    degree       : spline degree; 3 (cubic) is the default
    atm_iv       : flat-smile initialisation for all coefficients
    """
    # Clamped (open) knot vector: (degree+1) repeated at each boundary,
    # with (n_basis − degree − 1) uniformly spaced interior knots.
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_basis={n_basis} too small for degree={degree}; need n_basis > degree")
    interior = np.linspace(k_min, k_max, n_interior + 2)[1:-1]
    t = np.r_[
        np.repeat(k_min, degree + 1),
        interior,
        np.repeat(k_max, degree + 1),
    ]
    # Valid evaluation domain (same as knot span for clamped splines)
    k_lo = float(t[degree])
    k_hi = float(t[-(degree + 1)])

    def _observe(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
        k_c = float(np.clip(k, k_lo, k_hi))
        H = BSpline.design_matrix([k_c], t, degree).toarray()[0]
        z_hat = float(H @ theta)
        return z_hat, H

    return Model(
        name=f"bspline_slice_n{n_basis}_d{degree}",
        n_params=n_basis,
        observe=_observe,
        initial_theta=lambda cfg: np.full(cfg.n_params, atm_iv),
        q_diag_weights=np.ones(n_basis),
    )


# Default: 8 cubic B-splines, log-moneyness ∈ [−0.30, +0.15]  (4 interior knots)
BSPLINE_SLICE_MODEL = make_bspline_slice_model(k_min=-0.30, k_max=0.15, n_basis=8, degree=3)
