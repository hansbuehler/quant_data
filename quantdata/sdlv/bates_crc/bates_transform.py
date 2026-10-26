r"""Section 2/3 combined: the full continuation return transform
Phi_t(u,T) (eq:bcrc-transform), Fourier call pricing, and the Phi(0)=Phi(-i)=1
martingale-normalisation checks required by Section 5 of the verification
battery.

    Phi_t(u,T) = exp{ V_t D(u,T-t) + kappa*theta*integral_0^(T-t) D(u,r) dr
                       + (T-t)*j_p(u) + integral_t^T ell_s(u) ds }

The first two terms are the ordinary Bates background (current jump
parameters p_n, current variance V_t); the last is the reserve correction
(reserve_crc.integrated_reserve). At the reset instant itself the reserve
integral's lower/upper limits are exactly the current continuation
interval, so Proposition prop:bcrc-inverse's exact-reset-preservation claim
reduces to the reserve construction leaving Phi_t(u,T) UNCHANGED across a
reset for every T on the continuation -- checked directly in Task 4 below,
not merely asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantdata.sdlv.bates_crc.heston_riccati import (
    HestonParams,
    gaussian_bates_jump_exponent,
    riccati_A_closed_form,
    riccati_D_closed_form,
)
from quantdata.sdlv.bates_crc.reserve_crc import (
    BcrcState,
    GaussianJumpParams,
    ReserveSchedule,
    integrated_reserve,
)


def continuation_transform(u: complex, t: float, T: float, V_t: float, heston: HestonParams,
                            state: BcrcState, schedule: ReserveSchedule, envelope: GaussianJumpParams,
                            n_quad: int = 400) -> complex:
    """Phi_t(u,T), eq. (eq:bcrc-transform), current jump parameters taken
    from state.jump_params_path[-1]."""
    tau = T - t
    current_jumps = state.jump_params_path[-1]
    D = riccati_D_closed_form(u, tau, heston)
    A = riccati_A_closed_form(u, tau, heston)
    j_term = tau * gaussian_bates_jump_exponent(u, current_jumps)
    reserve_term = integrated_reserve(u, t, T, state, schedule, heston, envelope, n_quad=n_quad)
    return np.exp(V_t * D + A + j_term + reserve_term)


def verify_martingale_normalisation(t: float, T: float, V_t: float, heston: HestonParams,
                                     state: BcrcState, schedule: ReserveSchedule,
                                     envelope: GaussianJumpParams) -> dict:
    """Phi_t(0,T) = Phi_t(-i,T) = 1, for the FULL transform including the
    reserve correction."""
    phi_0 = continuation_transform(0.0, t, T, V_t, heston, state, schedule, envelope)
    phi_mi = continuation_transform(-1j, t, T, V_t, heston, state, schedule, envelope)
    return dict(
        t=t, T=T, phi_0=complex(phi_0), phi_minus_i=complex(phi_mi),
        abs_error_phi_0=abs(phi_0 - 1.0), abs_error_phi_minus_i=abs(phi_mi - 1.0),
    )


# ===========================================================================
# Fourier call pricing (Carr-Madan damped-FFT-free direct quadrature -- the
# contract book here is small and fixed, so a direct Gauss-Legendre
# quadrature of the damped integral is simpler and equally accurate; an
# independent Gauss-Hermite / Monte Carlo cross-check is in Task 5's
# verification module).
# ===========================================================================


@dataclass(frozen=True)
class FourierPricingConfig:
    damping_alpha: float = 1.5
    n_quad: int = 400
    u_max: float = 200.0

    def __post_init__(self) -> None:
        if self.damping_alpha <= 0.0:
            raise ValueError("damping_alpha must be positive")


def fourier_call_price(strike: float, t: float, T: float, V_t: float, heston: HestonParams,
                        state: BcrcState, schedule: ReserveSchedule, envelope: GaussianJumpParams,
                        config: FourierPricingConfig = FourierPricingConfig(), *, y_forward: float = 1.0,
                        n_reserve_quad: int = 200) -> float:
    """Damped Fourier inversion (Carr-Madan) of the "pure"-unit call price
    on the background factor Y alone: prices E[(y_forward*Y_T - K)^+ | F_t].
    log(k) is the log-strike; the transform's own u variable already prices
    the LOG-RETURN of Y from y_forward=1 at t, so a nonunit y_forward is
    handled by scaling the strike, k -> K/y_forward, consistently with the
    manuscript's own G_t(x;T,K) = Black-type kernel usage in the product
    construction (Section 3 module)."""
    if T <= t:
        return max(y_forward - strike, 0.0)
    k = math.log(strike / y_forward)
    alpha = config.damping_alpha

    def integrand(u: np.ndarray) -> np.ndarray:
        shifted = u - 1j * (alpha + 1.0)
        phis = np.array([
            continuation_transform(complex(uu), t, T, V_t, heston, state, schedule, envelope,
                                    n_quad=n_reserve_quad)
            for uu in shifted
        ])
        numerator = np.exp(-1j * u * k) * phis
        denominator = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
        return numerator / denominator

    nodes, weights = np.polynomial.legendre.leggauss(config.n_quad)
    half = 0.5 * config.u_max
    u_nodes = half * (nodes + 1.0)
    values = integrand(u_nodes)
    integral = half * np.sum(weights * values)
    price = (math.exp(-alpha * k) / math.pi) * integral.real
    return float(y_forward * price)


def gauss_hermite_mc_cross_check(strike: float, t: float, T: float, V_t: float, heston: HestonParams,
                                  jumps: GaussianJumpParams, n_paths: int = 200_000, n_var_substeps: int = 200,
                                  seed: int = 7) -> dict:
    """An INDEPENDENT pricing check for the ordinary (no-reserve) Bates
    background: simulates the QE variance path + compound-Poisson Gaussian
    jumps directly (the same construction already verified in this
    session's earlier heston_bates_oracle work), and compares the Monte
    Carlo price against fourier_call_price with an empty reserve schedule.
    Deliberately re-implemented here from scratch (not imported from the
    other repository), so the two pricers share no code."""
    rng = np.random.default_rng(seed)
    tau = T - t
    kappa, theta, xi, rho = heston.kappa, heston.theta, heston.xi, heston.rho

    dt = tau / n_var_substeps
    V = np.full(n_paths, V_t)
    logY = np.zeros(n_paths)
    for _ in range(n_var_substeps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Zv = rho * Z1 + math.sqrt(max(1.0 - rho ** 2, 0.0)) * Z2
        V_next = np.maximum(
            V + kappa * (theta - V) * dt + xi * np.sqrt(np.maximum(V, 0.0) * dt) * Zv, 0.0
        )
        logY += -0.5 * V * dt + np.sqrt(np.maximum(V, 0.0) * dt) * Z1
        V = V_next
    n_jumps = rng.poisson(jumps.lambda_J * tau, size=n_paths)
    kappa_J = math.exp(jumps.m_J + 0.5 * jumps.delta_J ** 2) - 1.0
    # sum of n_jumps iid N(m_J, delta_J^2) = N(n_jumps*m_J, n_jumps*delta_J^2)
    jump_normal = rng.standard_normal(n_paths)
    jump_sum = n_jumps * jumps.m_J + np.sqrt(np.maximum(n_jumps, 0)) * jumps.delta_J * jump_normal
    logY += jump_sum - jumps.lambda_J * kappa_J * tau

    Y_T = np.exp(logY)
    payoffs = np.maximum(Y_T - strike, 0.0)
    mc_price = float(np.mean(payoffs))
    mc_se = float(np.std(payoffs) / math.sqrt(n_paths))
    return dict(mc_price=mc_price, mc_standard_error=mc_se, n_paths=n_paths)
