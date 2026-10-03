r"""Section 2 (background implementation): the Heston Riccati coefficient,
the Gaussian Bates jump exponent, and the continuation return transform,
exactly as specified in the current DYSANOS II manuscript's DLV x
extended-Bates section (eq. eq:bcrc-riccati, eq:bcrc-jump-exponent,
eq:bcrc-transform, eq:bcrc-affine-A).

Riccati equation (eq:bcrc-riccati):
    d/dtau D(u,tau) = (1/2) xi^2 D(u,tau)^2 + (rho*xi*i*u - kappa) D(u,tau)
                       - (1/2)(u^2+i*u),      D(u,0) = 0.

Closed form (standard Heston affine solution, re-derived and verified here
independently against direct numerical ODE integration -- Task 1's
"independent pricing check" -- not merely asserted):
    b(u)  = kappa - i*rho*xi*u
    d(u)  = sqrt(b(u)^2 + xi^2*(u^2+i*u)),   Re d(u) > 0
    g(u)  = (b(u)-d(u)) / (b(u)+d(u))
    D(u,tau) = (b-d)/xi^2 * (1-exp(-d*tau)) / (1-g*exp(-d*tau))
    A(u,tau) = kappa*theta/xi^2 * [(b-d)*tau - 2*log((1-g*exp(-d*tau))/(1-g))]
             = kappa*theta * integral_0^tau D(u,r) dr   (eq:bcrc-affine-A)

Gaussian Bates jump exponent (eq:bcrc-jump-exponent):
    j_p(u) = lambda_J * [exp(i*u*m_J - delta_J^2*u^2/2) - 1
                          - i*u*(exp(m_J+delta_J^2/2) - 1)]
This is the manuscript's own martingale-normalised (eq:bcrc-lk-martingale)
form specialised to a single Gaussian jump-size law: writing
kappa_J = exp(m_J+delta_J^2/2)-1 for the raw jump-size mean, the -i*u*kappa_J
term is exactly the "-i*u*(e^z-1)" compensator in eq:bcrc-lk-martingale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0 or self.xi <= 0.0 or self.theta < 0.0:
            raise ValueError(f"invalid Heston parameters {self}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must lie in [-1,1], got {self.rho}")


@dataclass(frozen=True)
class GaussianJumpParams:
    lambda_J: float  # intensity
    m_J: float        # log-jump mean
    delta_J: float     # log-jump std

    def __post_init__(self) -> None:
        if self.lambda_J < 0.0 or self.delta_J <= 0.0:
            raise ValueError(f"invalid jump parameters {self}")


def heston_bd(u: np.ndarray, heston: HestonParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """b(u), d(u) (branch with Re d >= 0), g(u)."""
    u = np.asarray(u, dtype=complex)
    b = heston.kappa - 1j * heston.rho * heston.xi * u
    d = np.sqrt(b ** 2 + heston.xi ** 2 * (u ** 2 + 1j * u))
    d = np.where(d.real < 0.0, -d, d)
    g = (b - d) / (b + d)
    return b, d, g


def riccati_D_closed_form(u: np.ndarray, tau: float, heston: HestonParams) -> np.ndarray:
    """D(u,tau), closed form. D(u,0)=0 exactly (checked in tests)."""
    if tau == 0.0:
        return np.zeros_like(np.asarray(u, dtype=complex))
    b, d, g = heston_bd(u, heston)
    e = np.exp(-d * tau)
    return (b - d) / heston.xi ** 2 * (1.0 - e) / (1.0 - g * e)


def riccati_A_closed_form(u: np.ndarray, tau: float, heston: HestonParams) -> np.ndarray:
    """A(u,tau) = kappa*theta * integral_0^tau D(u,r) dr, closed form."""
    if tau == 0.0:
        return np.zeros_like(np.asarray(u, dtype=complex))
    b, d, g = heston_bd(u, heston)
    e = np.exp(-d * tau)
    log_term = np.log((1.0 - g * e) / (1.0 - g))
    return heston.kappa * heston.theta / heston.xi ** 2 * ((b - d) * tau - 2.0 * log_term)


def riccati_D_ode_integration(u: complex, tau: float, heston: HestonParams,
                               n_steps: int = 20_000) -> complex:
    """Independent check: direct RK4 integration of the Riccati ODE, NOT
    using the closed-form solution. Used only to verify the closed form
    (Task 1's 'independent pricing check'), not for production pricing."""
    def rhs(D: complex) -> complex:
        return (0.5 * heston.xi ** 2 * D ** 2
                + (heston.rho * heston.xi * 1j * u - heston.kappa) * D
                - 0.5 * (u ** 2 + 1j * u))

    D = 0.0 + 0.0j
    h = tau / n_steps
    for _ in range(n_steps):
        k1 = rhs(D)
        k2 = rhs(D + 0.5 * h * k1)
        k3 = rhs(D + 0.5 * h * k2)
        k4 = rhs(D + h * k3)
        D = D + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return D


def gaussian_bates_jump_exponent(u: np.ndarray, jumps: GaussianJumpParams) -> np.ndarray:
    """j_p(u), eq. (eq:bcrc-jump-exponent). j_p(0)=0 and (checked in tests)
    the martingale normalisation j_p(-i) = 0 hold exactly."""
    u = np.asarray(u, dtype=complex)
    kappa_J = math.exp(jumps.m_J + 0.5 * jumps.delta_J ** 2) - 1.0
    return jumps.lambda_J * (
        np.exp(1j * u * jumps.m_J - 0.5 * jumps.delta_J ** 2 * u ** 2) - 1.0
        - 1j * u * kappa_J
    )


def moment_explosion_time(p: float, heston: HestonParams) -> float:
    """T*(p): the affine moment-explosion time at real order p, per the
    Keller-Ressel criterion already used and verified in this research
    programme's earlier heston_bates_oracle work (Task 8). Reused here as
    the moment-strip check for the Bates background factor: any p with
    T*(p) < current tau means that moment is already infinite and must not
    be relied upon by Fourier damping or a moment-based control variate."""
    a = heston.xi ** 2 / 2.0
    b_p = heston.rho * heston.xi * p - heston.kappa
    c_p = p * (p - 1.0) / 2.0
    disc = 4.0 * a * c_p - b_p ** 2
    if disc <= 0.0:
        return float("inf")
    s = math.sqrt(disc)
    return 2.0 / s * (math.pi / 2.0 - math.atan(b_p / s))


def moment_strip_report(heston: HestonParams, tau: float, p_max: float = 50.0,
                         n_grid: int = 200) -> dict:
    """Reports the largest real order p for which T*(p) > tau (i.e. the
    moment is still finite at this maturity) -- the usable Fourier-damping
    strip upper bound."""
    ps = np.linspace(1.01, p_max, n_grid)
    times = np.array([moment_explosion_time(p, heston) for p in ps])
    usable = ps[times > tau]
    return dict(
        tau=tau, p_max_tested=p_max,
        largest_finite_moment_order=float(usable.max()) if usable.size else None,
        all_finite_in_range=bool(np.all(times > tau)),
    )
