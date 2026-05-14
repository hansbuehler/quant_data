"""
surface_model.py

Model registry for the spline-state Kalman filter.

Each model is a `Model` instance bundling:
  - observe(theta, k, T)  -> (z_hat, H_jacobian)   used by the EKF update
  - initial_theta(cfg)    -> np.ndarray             starting state
  - q_diag_weights        -> np.ndarray             per-param relative Q scale
                             (used when Q_structure="diagonal")

Available models (swap in the notebook config cell):
  QUAD_SLICE_MODEL — 3 params, a + b·k + c·k²            (linear KF, single expiry)
  SVI_SLICE_MODEL  — 5 params, a + b·[ρ(k−m)+√((k−m)²+σ²)]  (EKF, single expiry)
  SVI_MODEL        — 5 params, same × √T                 (EKF, multiple maturities)
  TANH_MODEL       — 3 params, a·√T·(1+tanh(b·k+c·k²))  (EKF, multiple maturities)

To add a new model: write _observe_xxx / _initial_theta_xxx, then define
XXX_MODEL = Model(name=..., n_params=..., observe=..., initial_theta=..., q_diag_weights=...).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

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


# ---------------------------------------------------------------------------
# Model 1 — SVI slice  (5 params, no √T)
# IV(k) = a + b·[ρ(k−m) + √((k−m)²+σ²)]
# Use for a single expiry.  H[0]=1 → same effective gain as per-option KF.
# ---------------------------------------------------------------------------


def _observe_svi_slice(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    Pure SVI raw parametrisation (no T scaling):
        IV(k) = a + b·[ρ·(k−m) + √((k−m)²+σ²)]

    θ = [a, b, ρ, m, σ]
    ATM (k=m): IV = a + b·σ

    No √T: for a single expiry H[0]=1, giving the same effective Kalman gain
    as the individual per-option filter.  Adding √T shrinks H by √T and
    suppresses the gain for short-dated options (see SVI_MODEL for multi-expiry).
    """
    a, b, rho, m, sigma = (float(theta[i]) for i in range(5))
    sigma = max(sigma, 1e-6)

    d = k - m
    sq = np.sqrt(d**2 + sigma**2)
    f = rho * d + sq
    z_hat = a + b * f

    H = np.array(
        [
            1.0,  # ∂/∂a  — always 1, full gain on level
            f,  # ∂/∂b
            b * d,  # ∂/∂ρ
            -b * (rho + d / sq),  # ∂/∂m
            b * sigma / sq,  # ∂/∂σ
        ]
    )
    return z_hat, H


def _initial_theta_svi_slice(cfg: "SurfaceModelConfig") -> np.ndarray:
    """a=0.15, b=0.4, ρ=−0.7, m=0, σ=0.2  →  ATM = 0.15 + 0.4·0.2 = 0.23"""
    return np.array([0.15, 0.4, -0.7, 0.0, 0.2])


SVI_SLICE_MODEL = Model(
    name="svi_slice",
    n_params=5,
    observe=_observe_svi_slice,
    initial_theta=_initial_theta_svi_slice,
    # level (a) moves constantly with spot; b/ρ move moderately; m/σ near-static
    q_diag_weights=np.array([1.0, 0.2, 0.1, 0.02, 0.02]),
)


# ---------------------------------------------------------------------------
# Model 2 — SVI with √T scaling  (5 params)
# IV(k,T) = [a + b·(ρ(k−m) + √((k−m)²+σ²))] · √T
# Use when fitting multiple maturities jointly.
# ---------------------------------------------------------------------------


def _observe_svi(theta: np.ndarray, k: float, T: float) -> tuple[float, np.ndarray]:
    """
    SVI raw parametrisation × √T:
        IV(k,T) = [a + b·(ρ·(k−m) + √((k−m)²+σ²))] · √T

    θ = [a, b, ρ, m, σ]
    ATM (k=m): IV = (a + b·σ) · √T

    For short-dated options √T is small, suppressing H and the Kalman gain.
    Prefer SVI_SLICE_MODEL when working on a single expiry.
    """
    a, b, rho, m, sigma = (float(theta[i]) for i in range(5))
    sigma = max(sigma, 1e-6)

    sqrtT = np.sqrt(max(T, 1e-6))
    d = k - m
    sq = np.sqrt(d**2 + sigma**2)
    f = rho * d + sq
    z_hat = (a + b * f) * sqrtT

    H = sqrtT * np.array(
        [
            1.0,
            f,
            b * d,
            -b * (rho + d / sq),
            b * sigma / sq,
        ]
    )
    return z_hat, H


def _initial_theta_svi(cfg: "SurfaceModelConfig") -> np.ndarray:
    """a=0.023, b=0.5, ρ=−0.7, m=0, σ=0.3  →  ATM·√T=0.006 ≈ 0.023"""
    return np.array([0.023, 0.5, -0.7, 0.0, 0.3])


SVI_MODEL = Model(
    name="svi",
    n_params=5,
    observe=_observe_svi,
    initial_theta=_initial_theta_svi,
    q_diag_weights=np.array([1.0, 0.2, 0.1, 0.02, 0.02]),
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
