"""Heston/Bates dynamic-arbitrage oracle: pricer, generators, DLV diagnostics.

Conventions (fixed throughout; see README.md):
    - Zero interest rate, zero dividend yield (r = q = 0). S is a Q-martingale.
    - Cash strikes K and calendar expiries T, both fixed in calendar time
      (never moneyness or time-to-maturity grids).
    - Double precision (float64) throughout -- see `configure_jax()`.

Heston SDE under Q:
    dS_t/S_t = sqrt(V_t) dW^1_t
    dV_t     = kappa(theta - V_t) dt + xi sqrt(V_t) dW^2_t
    d<W^1,W^2>_t = rho dt

Bates SDE under Q adds compound-Poisson log-jumps to S (same V):
    dS_t/S_t- = sqrt(V_t) dW^1_t + integral (e^y - 1)(mu - nu)(dt, dy)
    nu(dy) = lambda * phi_{mu_J, delta_J}(y) dy

Because C_t(T,K) = E_Q[(S_T-K)^+ | F_t] by construction, Heston/Bates are
"oracles": every identity below that follows from martingality must hold up
to numerical error. A failure indicates a bug in this module, not a finding
about the models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def configure_jax() -> None:
    """Idempotent: ensure float64 is enabled (call at import time is enough)."""
    jax.config.update("jax_enable_x64", True)


configure_jax()


# ===========================================================================
# Parameters
# ===========================================================================


@dataclass(frozen=True)
class HestonParams:
    """Static (no term structure) Heston parameters."""

    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0 or self.theta <= 0.0 or self.xi <= 0.0:
            raise ValueError("kappa, theta, xi must be strictly positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly in (-1, 1)")

    @property
    def feller_ratio(self) -> float:
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


@dataclass(frozen=True)
class JumpParams:
    """Merton-style log-normal jump parameters for Bates. lambda_=0 -> Heston."""

    lambda_: float = 0.0
    mu_j: float = 0.0
    delta_j: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_ < 0.0:
            raise ValueError("lambda_ must be non-negative")
        if self.lambda_ > 0.0 and self.delta_j <= 0.0:
            raise ValueError("delta_j must be strictly positive when lambda_ > 0")

    @property
    def compensator_k(self) -> float:
        """k = E[e^Y] - 1, Y ~ N(mu_j, delta_j^2); 0 when there are no jumps."""
        if self.lambda_ == 0.0:
            return 0.0
        return math.exp(self.mu_j + 0.5 * self.delta_j**2) - 1.0


HESTON_DEFAULT = HestonParams(kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
BATES_JUMPS_DEFAULT = JumpParams(lambda_=0.3, mu_j=-0.10, delta_j=0.15)
S0_DEFAULT = 100.0
V0_DEFAULT = 0.04


# ===========================================================================
# Characteristic function: Little Heston trap (Albrecher et al. 2007),
# extended multiplicatively with the Bates jump component.
# psi(u; tau, V) = E[exp(i u log(S_{t+tau}/S_t)) | V_t = V]
# ===========================================================================


def characteristic_function(
    u: jnp.ndarray,
    tau: float,
    V: float,
    heston: HestonParams,
    jumps: JumpParams,
) -> jnp.ndarray:
    """Heston/Bates characteristic function of log(S_{t+tau}/S_t), r=q=0."""
    u = jnp.asarray(u, dtype=jnp.complex128)
    kappa, theta, xi, rho = heston.kappa, heston.theta, heston.xi, heston.rho
    xi2 = xi * xi
    b = kappa - rho * xi * 1j * u
    d = jnp.sqrt(b * b + xi2 * (1j * u + u * u))
    g = (b - d) / (b + d)
    e = jnp.exp(-d * tau)
    A = (kappa * theta / xi2) * ((b - d) * tau - 2.0 * jnp.log((1.0 - g * e) / (1.0 - g)))
    B = ((b - d) / xi2) * (1.0 - e) / (1.0 - g * e)
    psi = jnp.exp(A + B * V)
    if jumps.lambda_ != 0.0:
        k = jumps.compensator_k
        jump_cf = jnp.exp(1j * u * jumps.mu_j - 0.5 * (u**2) * jumps.delta_j**2)
        psi = psi * jnp.exp(tau * jumps.lambda_ * (jump_cf - 1.0 - 1j * u * k))
    return psi


# ===========================================================================
# Pricer, method 1 (primary, JAX-differentiable): Lewis (2001) single
# integral, Gauss-Legendre quadrature on a truncated domain [0, U_MAX].
#   C = S - sqrt(S*K)/pi * int_0^inf Re[e^{i u k_S} psi(u - i/2; tau, V)] / (u^2 + 1/4) du
# ===========================================================================


class QuadratureNodes(NamedTuple):
    u: jnp.ndarray
    w: jnp.ndarray


def gauss_legendre_quadrature(n: int, u_max: float) -> QuadratureNodes:
    """Gauss-Legendre nodes/weights mapped from [-1,1] to [0, u_max]."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * u_max * (x + 1.0)
    weight = 0.5 * u_max * w
    return QuadratureNodes(jnp.asarray(u), jnp.asarray(weight))


#: Default pricer quadrature. Convergence verified in tests/ and the notebook
#: (Stage 0): n=1000, u_max=800 agrees with n=1600,u_max=800 to < 1e-6 abs.
DEFAULT_PRICE_QUADRATURE = gauss_legendre_quadrature(1000, 800.0)


def call_price(
    S: float,
    V: float,
    tau: float,
    K: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    quadrature: QuadratureNodes = DEFAULT_PRICE_QUADRATURE,
) -> jnp.ndarray:
    """European call price C(t,S,V;T,K) with tau=T-t, via Lewis (2001). r=q=0.

    Differentiable end-to-end in (S, V, tau, K) via JAX.
    """
    k_S = jnp.log(S / K)
    shifted = quadrature.u - 0.5j
    psi = characteristic_function(shifted, tau, V, heston, jumps)
    integrand = jnp.real(jnp.exp(1j * quadrature.u * k_S) * psi) / (quadrature.u**2 + 0.25)
    integral = jnp.sum(quadrature.w * integrand)
    return S - jnp.sqrt(S * K) / jnp.pi * integral


def put_price(
    S: float,
    V: float,
    tau: float,
    K: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    quadrature: QuadratureNodes = DEFAULT_PRICE_QUADRATURE,
) -> jnp.ndarray:
    """European put price via put-call parity (r=q=0): P = C - S + K."""
    return call_price(S, V, tau, K, heston, jumps, quadrature) - S + K


# ===========================================================================
# Pricer, method 2 (independent cross-check only, plain numpy/complex128,
# NOT JAX-differentiable, not used in the residual machinery): original
# Heston (1993) P1/P2 two-integral formula.
#   C = S*P1 - K*P2,  P_j via Gil-Pelaez inversion of a measure-shifted CF.
# Genuinely different derivation from Lewis (2001): validates the pricer
# independently (see tests/test_pricer.py and the notebook's Stage 0).
# ===========================================================================


def _np_characteristic_function(u: np.ndarray, tau: float, V: float, heston: HestonParams, jumps: JumpParams) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    kappa, theta, xi, rho = heston.kappa, heston.theta, heston.xi, heston.rho
    xi2 = xi * xi
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + xi2 * (1j * u + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    A = (kappa * theta / xi2) * ((b - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    B = ((b - d) / xi2) * (1.0 - e) / (1.0 - g * e)
    psi = np.exp(A + B * V)
    if jumps.lambda_ != 0.0:
        k = jumps.compensator_k
        jump_cf = np.exp(1j * u * jumps.mu_j - 0.5 * (u**2) * jumps.delta_j**2)
        psi = psi * np.exp(tau * jumps.lambda_ * (jump_cf - 1.0 - 1j * u * k))
    return psi


def _heston93_p1_p2(
    S: float,
    V: float,
    tau: float,
    K: float,
    heston: HestonParams,
    jumps: JumpParams,
    n: int,
    u_max: float,
) -> tuple[float, float]:
    """Gil-Pelaez digital probabilities P1, P2 for the original Heston (1993)
    formula. IMPORTANT: the threshold is log(K/S), not log(K) -- our
    characteristic function is of the log RETURN log(S_T/S_t), not the log
    price.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * u_max * (x + 1.0)
    weight = 0.5 * u_max * w
    u_c = u.astype(np.complex128)

    log_k = math.log(K / S)
    psi_minus_i = _np_characteristic_function(np.array([-1j]), tau, V, heston, jumps)[0]
    f1 = _np_characteristic_function(u_c - 1j, tau, V, heston, jumps) / psi_minus_i
    f2 = _np_characteristic_function(u_c, tau, V, heston, jumps)
    integrand1 = np.real(np.exp(-1j * u_c * log_k) * f1 / (1j * u_c))
    integrand2 = np.real(np.exp(-1j * u_c * log_k) * f2 / (1j * u_c))
    P1 = 0.5 + (1.0 / math.pi) * np.sum(weight * integrand1)
    P2 = 0.5 + (1.0 / math.pi) * np.sum(weight * integrand2)
    return float(P1), float(P2)


def heston93_call_price(
    S: float,
    V: float,
    tau: float,
    K: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    n: int = 1000,
    u_max: float = 800.0,
) -> float:
    """Independent cross-check pricer: original Heston (1993) P1/P2 formula, C = S*P1 - K*P2."""
    P1, P2 = _heston93_p1_p2(S, V, tau, K, heston, jumps, n, u_max)
    return S * P1 - K * P2


def heston93_put_price(
    S: float,
    V: float,
    tau: float,
    K: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    n: int = 1000,
    u_max: float = 800.0,
) -> float:
    """Independent put price via the SAME P1/P2 decomposition: P = K(1-P2) - S(1-P1).

    Genuinely independent of `put_price` (which is call_price - S + K by
    parity): this recomputes P1, P2 from scratch via Gil-Pelaez inversion, so
    comparing it against `call_price(...) - S + K` is a real parity check,
    not a tautology.
    """
    P1, P2 = _heston93_p1_p2(S, V, tau, K, heston, jumps, n, u_max)
    return K * (1.0 - P2) - S * (1.0 - P1)


# ===========================================================================
# Black-Scholes reference (self-contained, math.erf-based normal CDF), used
# only for the xi->0, lambda=0 pricer acceptance check.
# ===========================================================================


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S: float, K: float, tau: float, sigma: float) -> float:
    d1 = (math.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


# ===========================================================================
# Generic scalar-field machinery on (t, S, V).
#
# Every quantity we differentiate below -- a single fixed-contract price C,
# a finite butterfly q_n^m, a finite DLV coordinate ell_n^m, or their
# continuous-strike analogues -- is a scalar function f(t, S, V) built by
# composing `call_price` calls (with T, K, or a whole strike/expiry grid
# closed over). JAX differentiates through that composition automatically,
# so the generator/Carre-du-champ/jump machinery below is written ONCE,
# generically, against any such f.
#
# Convention: f(t, S, V) uses tau = T - t internally, so f_t = -d/dtau (the
# spec's remark "d_tC = -d_tau C at fixed T" falls out automatically here).
# ===========================================================================

ScalarField = "Callable[[float, float, float], jnp.ndarray]"


def field_derivatives(f, t: float, S: float, V: float):
    """Return (f_t, f_S, f_V, f_SS, f_SV, f_VV) for a scalar field f(t,S,V).

    First derivatives via jax.grad; the (S,V) Hessian block via jax.hessian
    (one joint call, avoiding redundant re-differentiation).
    """
    f_t = jax.grad(f, argnums=0)(t, S, V)
    f_S, f_V = jax.grad(f, argnums=(1, 2))(t, S, V)
    (f_SS, f_SV), (_f_VS, f_VV) = jax.hessian(f, argnums=(1, 2))(t, S, V)
    return f_t, f_S, f_V, f_SS, f_SV, f_VV


def heston_generator(f, t: float, S: float, V: float, heston: HestonParams) -> jnp.ndarray:
    r"""L_H f = f_t + (1/2)VS^2 f_SS + rho*xi*V*S f_SV + (1/2)xi^2 V f_VV + kappa(theta-V) f_V."""
    f_t, _f_S, f_V, f_SS, f_SV, f_VV = field_derivatives(f, t, S, V)
    return (
        f_t
        + 0.5 * V * S**2 * f_SS
        + heston.rho * heston.xi * V * S * f_SV
        + 0.5 * heston.xi**2 * V * f_VV
        + heston.kappa * (heston.theta - V) * f_V
    )


def carre_du_champ(f, g, t: float, S: float, V: float, heston: HestonParams) -> jnp.ndarray:
    r"""Gamma_H(f,g) = V[S^2 f_S g_S + rho*xi*S(f_S g_V + f_V g_S) + xi^2 f_V g_V].

    This is the density of d<f,g>^c_t (continuous/Brownian covariation only;
    the jump contribution is handled separately by `jump_cross_term`). Uses
    the correlation explicitly -- omitting rho here is the most likely
    silent error (per the spec's pitfalls).
    """
    _t1, f_S, f_V, *_ = field_derivatives(f, t, S, V)
    _t2, g_S, g_V, *_ = field_derivatives(g, t, S, V)
    return V * (
        S**2 * f_S * g_S
        + heston.rho * heston.xi * S * (f_S * g_V + f_V * g_S)
        + heston.xi**2 * f_V * g_V
    )


def gauss_hermite_jump_quadrature(n: int, mu_j: float, delta_j: float) -> QuadratureNodes:
    """Nodes y_i and probability weights w_i (sum(w_i)=1) for E_Y[g(Y)], Y~N(mu_j, delta_j^2).

    Standard substitution Y = mu_j + sqrt(2) delta_j x for physicists'-Hermite
    nodes x_i, weights v_i (int e^{-x^2} h(x) dx ~ sum v_i h(x_i)):
    E[g(Y)] ~ (1/sqrt(pi)) sum v_i g(mu_j + sqrt(2) delta_j x_i).
    """
    x, v = np.polynomial.hermite.hermgauss(n)
    y = mu_j + math.sqrt(2.0) * delta_j * x
    w = v / math.sqrt(math.pi)
    return QuadratureNodes(jnp.asarray(y), jnp.asarray(w))


def _jump_quadrature_for(jumps: JumpParams, n: int = 250) -> QuadratureNodes:
    """>=40 Gauss-Hermite nodes per the spec (default 250 -- convergence
    checked in tests/ and the notebook: L_B C shrinks from ~9e-6 at n=40 to
    ~4e-12 at n=250; numpy's hermgauss becomes numerically unstable for
    n gtrsim 300-400, so 250 is chosen with margin below that breakdown)."""
    return gauss_hermite_jump_quadrature(n, jumps.mu_j, jumps.delta_j)


def delta_y(f, t: float, S: float, V: float, y) -> jnp.ndarray:
    r"""Delta_y f := f(t, S e^y, V) - f(t, S, V): exact finite difference at
    jump size(s) y (scalar or array); V does not jump.

    Guards against non-finite f(t, S e^y, V): for a finite-difference field
    (e.g. a DLV ell_n^m built from a butterfly q_n^m in the denominator),
    evaluating at an extreme jump-shifted spot can land in a degenerate
    "deep wing" region where the underlying butterfly is itself near zero,
    blowing the ratio up to +-inf -- even though such extreme Gauss-Hermite
    nodes carry astronomically small weight, one non-finite term poisons the
    whole weighted sum (inf + -inf = nan; observed in practice for Bates at
    some finite-grid nodes). The true continuous jump integral has
    negligible mass in that extreme a tail anyway, so non-finite evaluations
    are treated as a zero contribution (Delta_y f := 0) rather than
    propagating the numerical artifact.
    """
    def one(yy):
        shifted = f(t, S * jnp.exp(yy), V)
        base = f(t, S, V)
        diff = shifted - base
        return jnp.where(jnp.isfinite(diff), diff, 0.0)
    return jax.vmap(one)(jnp.atleast_1d(y))


def jump_generator_term(f, t: float, S: float, V: float, jumps: JumpParams, n_nodes: int = 250) -> jnp.ndarray:
    r"""lambda * E_Y[f(t,Se^Y,V) - f(t,S,V) - S f_S (e^Y-1)], the extra term in L_B beyond L_H."""
    if jumps.lambda_ == 0.0:
        return jnp.asarray(0.0)
    quad = _jump_quadrature_for(jumps, n_nodes)
    _f_t, f_S, *_ = field_derivatives(f, t, S, V)
    df = delta_y(f, t, S, V, quad.u)
    compensator = S * f_S * (jnp.exp(quad.u) - 1.0)
    return jumps.lambda_ * jnp.sum(quad.w * (df - compensator))


def bates_generator(f, t: float, S: float, V: float, heston: HestonParams, jumps: JumpParams, n_nodes: int = 250) -> jnp.ndarray:
    r"""L_B f = L_H f + lambda E_Y[f(t,Se^Y,V) - f(t,S,V) - S f_S (e^Y-1)]."""
    return heston_generator(f, t, S, V, heston) + jump_generator_term(f, t, S, V, jumps, n_nodes)


def jump_cross_term(f, g, t: float, S: float, V: float, jumps: JumpParams, n_nodes: int = 250) -> jnp.ndarray:
    r"""lambda * E_Y[Delta_y f * Delta_y g], the jump contribution to d<f,g>_t
    (needed for the Bates DLV drift residual's extra term)."""
    if jumps.lambda_ == 0.0:
        return jnp.asarray(0.0)
    quad = _jump_quadrature_for(jumps, n_nodes)
    df = delta_y(f, t, S, V, quad.u)
    dg = delta_y(g, t, S, V, quad.u)
    return jumps.lambda_ * jnp.sum(quad.w * df * dg)


# ===========================================================================
# Fixed cash-strike / calendar-expiry grid (spec defaults).
# ===========================================================================

#: K_n = 60, 65, ..., 160 (21 nodes). Interior nodes for finite differences
#: are n=1,...,19 (K=65,...,155): a butterfly needs both neighbours.
FIXED_STRIKES: tuple[float, ...] = tuple(60.0 + 5.0 * i for i in range(21))
FIXED_STRIKE_INTERIOR_INDICES: tuple[int, ...] = tuple(range(1, len(FIXED_STRIKES) - 1))

#: T_1=0.25 (front, excluded from residual testing -- see spec Sec. "Front
#: expiry"), T_2, T_3, T_4 are the back slabs.
FIXED_EXPIRIES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
FIXED_BACK_SLAB_INDICES: tuple[int, ...] = (1, 2, 3)  # m=2,3,4 in 1-indexed spec notation

DEFAULT_OBSERVATION_TIMES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)

Q_MIN_DEFAULT = 1.0e-8


def make_price_field(T: float, K: float, heston: HestonParams, jumps: JumpParams = JumpParams()):
    """f(t,S,V) = C(t,S,V;T,K) = call_price(S,V,T-t,K,...). tau=T-t internally."""
    return lambda t, S, V: call_price(S, V, T - t, K, heston, jumps)


def make_finite_butterfly_field(K_minus: float, K_n: float, K_plus: float, T: float,
                                 heston: HestonParams, jumps: JumpParams = JumpParams()):
    r"""q_n^m(t,S,V) = (C_{n+1}-C_n)/(K_{n+1}-K_n) - (C_n-C_{n-1})/(K_n-K_{n-1}).

    A butterfly MASS (not a density/second derivative) as noted in the spec.
    """
    def q(t, S, V):
        C_plus = call_price(S, V, T - t, K_plus, heston, jumps)
        C_n = call_price(S, V, T - t, K_n, heston, jumps)
        C_minus = call_price(S, V, T - t, K_minus, heston, jumps)
        return (C_plus - C_n) / (K_plus - K_n) - (C_n - C_minus) / (K_n - K_minus)
    return q


def make_finite_dlv_field(K_minus: float, K_n: float, K_plus: float, T_prev: float, T_curr: float,
                           heston: HestonParams, jumps: JumpParams = JumpParams()):
    r"""ell_n^m(t,S,V) = (C_n^m - C_n^{m-1}) / (omega_n^m q_n^m),
    omega_n^m = K_n^2 (T_m - T_{m-1}) / (K_{n+1} - K_{n-1})."""
    q_field = make_finite_butterfly_field(K_minus, K_n, K_plus, T_curr, heston, jumps)
    omega = K_n**2 * (T_curr - T_prev) / (K_plus - K_minus)

    def ell(t, S, V):
        C_curr = call_price(S, V, T_curr - t, K_n, heston, jumps)
        C_prev = call_price(S, V, T_prev - t, K_n, heston, jumps)
        return (C_curr - C_prev) / (omega * q_field(t, S, V))
    return ell, q_field


@dataclass(frozen=True)
class NodeResidual:
    """Diagnostics at one (strike, expiry) DLV node, one observation state."""

    n_index: int
    m_index: int
    K: float
    T: float
    t: float
    C: float  # C(t,S,V;T_m,K_n)
    q: float
    ell: float
    admissible: bool
    L_C: float  # generator residual on the raw fixed-contract price (sanity)
    L_q: float  # generator residual on the butterfly (linear in C, must vanish too)
    R: float  # the DLV drift residual (Heston: R^H: Bates: R^B)
    R_nojump: float | None = None  # Bates only: R^B with the jump term omitted (positive control)


def finite_grid_node_residual(
    n_idx: int,
    m_idx: int,
    t: float,
    S: float,
    V: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    strikes: tuple[float, ...] = FIXED_STRIKES,
    expiries: tuple[float, ...] = FIXED_EXPIRIES,
    q_min: float = Q_MIN_DEFAULT,
    n_jump_nodes: int = 250,
) -> NodeResidual:
    """Compute q, ell, and the DLV drift residual R at one grid node.

    m_idx is a back slab (>=1 in 0-indexed terms, i.e. expiries[m_idx] is
    T_2 or later); uses expiries[m_idx-1] as the previous slab.
    """
    K_minus, K_n, K_plus = strikes[n_idx - 1], strikes[n_idx], strikes[n_idx + 1]
    T_prev, T_curr = expiries[m_idx - 1], expiries[m_idx]

    ell_field, q_field = make_finite_dlv_field(K_minus, K_n, K_plus, T_prev, T_curr, heston, jumps)
    C_field = make_price_field(T_curr, K_n, heston, jumps)

    C_val = float(C_field(t, S, V))
    q_val = float(q_field(t, S, V))
    ell_val = float(ell_field(t, S, V))
    admissible = bool(q_val > q_min and ell_val >= 0.0)

    L_C = float(heston_generator(C_field, t, S, V, heston)) if jumps.lambda_ == 0.0 else float(
        bates_generator(C_field, t, S, V, heston, jumps, n_jump_nodes)
    )
    L_q = float(heston_generator(q_field, t, S, V, heston)) if jumps.lambda_ == 0.0 else float(
        bates_generator(q_field, t, S, V, heston, jumps, n_jump_nodes)
    )

    gamma_over_q = carre_du_champ(ell_field, q_field, t, S, V, heston) / q_field(t, S, V)
    if jumps.lambda_ == 0.0:
        L_ell = heston_generator(ell_field, t, S, V, heston)
        R = float(L_ell + gamma_over_q)
        R_nojump = None
    else:
        L_ell = bates_generator(ell_field, t, S, V, heston, jumps, n_jump_nodes)
        jump_term = jump_cross_term(ell_field, q_field, t, S, V, jumps, n_jump_nodes) / q_field(t, S, V)
        R = float(L_ell + gamma_over_q + jump_term)
        L_ell_heston_only = heston_generator(ell_field, t, S, V, heston)
        R_nojump = float(L_ell_heston_only + gamma_over_q)

    return NodeResidual(
        n_index=n_idx, m_index=m_idx, K=K_n, T=T_curr, t=t, C=C_val,
        q=q_val, ell=ell_val, admissible=admissible,
        L_C=L_C, L_q=L_q, R=R, R_nojump=R_nojump,
    )


# ===========================================================================
# Continuous-strike DLV: q = d^2C/dK^2, ell = 2 dC/dT / (K^2 q) (Dupire local
# variance, zero rates). Same generator/Gamma machinery as the finite grid,
# applied to fields built by differentiating w.r.t. (K,T) instead of by
# finite differences.
# ===========================================================================

DENSE_STRIKE_MESH: np.ndarray = np.arange(50.0, 201.0, 1.0)
DENSE_EXPIRY_MESH: np.ndarray = np.round(np.arange(FIXED_EXPIRIES[1] - 0.1, FIXED_EXPIRIES[-1] + 1e-9, 0.01), 6)


def make_continuous_q_field(K0: float, T0: float, heston: HestonParams, jumps: JumpParams = JumpParams()):
    """q(t,S,V) = d^2 C(t,S,V;T0,K)/dK^2 |_{K=K0}."""
    def q(t, S, V):
        C_of_K = lambda K: call_price(S, V, T0 - t, K, heston, jumps)
        return jax.grad(jax.grad(C_of_K))(K0)
    return q


def make_continuous_dlv_field(K0: float, T0: float, heston: HestonParams, jumps: JumpParams = JumpParams()):
    """ell(t,S,V) = 2 dC/dT|_{T0,K0} / (K0^2 q(t,S,V)); also returns q's field."""
    q_field = make_continuous_q_field(K0, T0, heston, jumps)

    def ell(t, S, V):
        C_of_T = lambda T: call_price(S, V, T - t, K0, heston, jumps)
        dC_dT = jax.grad(C_of_T)(T0)
        return 2.0 * dC_dT / (K0**2 * q_field(t, S, V))
    return ell, q_field


@dataclass(frozen=True)
class ContinuousResidual:
    """Diagnostics at one (K,T) continuous-mesh point, one observation state."""

    K: float
    T: float
    t: float
    q: float
    ell: float
    admissible: bool
    L_q: float
    R: float
    R_normalized: float
    R_nojump: float | None = None


def continuous_node_residual(
    K0: float,
    T0: float,
    t: float,
    S: float,
    V: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    q_min: float = Q_MIN_DEFAULT,
    n_jump_nodes: int = 250,
) -> ContinuousResidual:
    """Compute q, ell, and the continuous DLV drift residual R at one (K,T) mesh point."""
    ell_field, q_field = make_continuous_dlv_field(K0, T0, heston, jumps)

    q_val = float(q_field(t, S, V))
    ell_val = float(ell_field(t, S, V))
    admissible = bool(q_val > q_min and ell_val >= 0.0)

    L_q = float(heston_generator(q_field, t, S, V, heston)) if jumps.lambda_ == 0.0 else float(
        bates_generator(q_field, t, S, V, heston, jumps, n_jump_nodes)
    )

    gamma_over_q = carre_du_champ(ell_field, q_field, t, S, V, heston) / q_field(t, S, V)
    if jumps.lambda_ == 0.0:
        L_ell = heston_generator(ell_field, t, S, V, heston)
        R = float(L_ell + gamma_over_q)
        normalizer = float(jnp.abs(L_ell) + jnp.abs(gamma_over_q))
        R_nojump = None
    else:
        L_ell = bates_generator(ell_field, t, S, V, heston, jumps, n_jump_nodes)
        jump_term = jump_cross_term(ell_field, q_field, t, S, V, jumps, n_jump_nodes) / q_field(t, S, V)
        R = float(L_ell + gamma_over_q + jump_term)
        normalizer = float(jnp.abs(L_ell) + jnp.abs(gamma_over_q) + jnp.abs(jump_term))
        L_ell_heston_only = heston_generator(ell_field, t, S, V, heston)
        R_nojump = float(L_ell_heston_only + gamma_over_q)

    R_normalized = R / normalizer if normalizer > 0.0 else float("nan")

    return ContinuousResidual(
        K=K0, T=T0, t=t, q=q_val, ell=ell_val, admissible=admissible,
        L_q=L_q, R=R, R_normalized=R_normalized, R_nojump=R_nojump,
    )


# ===========================================================================
# Simulation: Andersen (2008) QE scheme for V, with an exact martingale
# correction for S derived from the QE branch's own conditional MGF (so that
# E[S_{t+dt}|F_t]=S_t holds analytically, not just approximately). Bates
# jumps are exact compound-Poisson with the usual compensator (also exactly
# driftless, since jump and diffusion parts are independent). Plain
# numpy -- simulation itself is not differentiated, only priced afterwards.
# ===========================================================================


@dataclass(frozen=True)
class SimulationStep:
    """One QE(+jump) step's terminal state and diagnostics."""

    S: np.ndarray
    V: np.ndarray
    used_exponential_branch: np.ndarray  # bool, per path: psi > psi_c


@dataclass(frozen=True)
class QEInnovations:
    """Explicit driving randomness for one QE(+jump) substep (n paths)."""

    Zv: np.ndarray  # squared-Gaussian branch normal
    U: np.ndarray  # exponential+point-mass branch uniform, in (0,1)
    Zs: np.ndarray  # independent residual normal for log S
    n_jumps: np.ndarray  # Poisson jump count per path (int)
    Zj: np.ndarray  # standard normal used to build the summed jump size


def draw_qe_innovations(n: int, dt: float, jumps: JumpParams, rng: np.random.Generator) -> QEInnovations:
    n_jumps = rng.poisson(jumps.lambda_ * dt, size=n) if jumps.lambda_ > 0.0 else np.zeros(n, dtype=np.int64)
    return QEInnovations(
        Zv=rng.standard_normal(n),
        U=rng.uniform(low=1e-300, high=1.0 - 1e-15, size=n),
        Zs=rng.standard_normal(n),
        n_jumps=n_jumps,
        Zj=rng.standard_normal(n),
    )


def antithetic_pair(innov: QEInnovations) -> QEInnovations:
    """Mirror innovations: negate Gaussians, reflect the uniform, keep the
    SAME jump count (Poisson counts don't have a natural sign-flip) but
    negate the jump-size normal, giving a valid antithetic pairing for both
    the diffusion and the jump part."""
    return QEInnovations(
        Zv=-innov.Zv,
        U=1.0 - innov.U,
        Zs=-innov.Zs,
        n_jumps=innov.n_jumps,
        Zj=-innov.Zj,
    )


def qe_step(
    S: np.ndarray,
    V: np.ndarray,
    dt: float,
    heston: HestonParams,
    jumps: JumpParams,
    innov: QEInnovations,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
) -> SimulationStep:
    """One Andersen (2008) QE step, martingale-corrected, with Bates jumps.

    Derivation of the martingale correction: writing
      log S_{t+dt} = log S_t + K0 + K1 V_t + K2 V_{t+dt} + sqrt(K3 V_t + K4 V_{t+dt}) Z,
    Z ~ N(0,1) independent of V_{t+dt}, E[S_{t+dt}|F_t] = S_t requires
      K0 = -(K1 + K3/2) V_t - log E[exp(w V_{t+dt})],  w = K2 + K4/2,
    with E[exp(w V_{t+dt})] taken under the QE branch's own conditional law:
    closed-form moment generating function in BOTH the squared-Gaussian
    branch (psi<=psi_c) and the exponential+point-mass branch (psi>psi_c).
    This is exact given the QE discretization of V, not an approximation.

    The jump compensator -lambda*k*dt is separately exact (jumps are
    independent of the diffusion; E[exp(sum jumps - lambda*k*dt)] = 1 exactly
    for a compound-Poisson sum of lognormal jumps, regardless of dt).
    """
    kappa, theta, xi, rho = heston.kappa, heston.theta, heston.xi, heston.rho
    gamma2 = 1.0 - gamma1

    exp_kdt = math.exp(-kappa * dt)
    m = theta + (V - theta) * exp_kdt
    s2 = (
        V * xi**2 * exp_kdt / kappa * (1.0 - exp_kdt)
        + theta * xi**2 / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
    )
    psi = s2 / np.clip(m, 1e-300, None) ** 2
    use_exp_branch = psi > psi_c

    # Branch 1: psi <= psi_c (squared-Gaussian).
    psi_safe = np.clip(psi, 1e-12, None)
    b2 = 2.0 / psi_safe - 1.0 + np.sqrt(np.clip(2.0 / psi_safe, 0.0, None) * np.clip(2.0 / psi_safe - 1.0, 0.0, None))
    a = m / (1.0 + b2)
    V_branch1 = a * (np.sqrt(np.clip(b2, 0.0, None)) + innov.Zv) ** 2

    # Branch 2: psi > psi_c (exponential + point mass at 0).
    p = (psi - 1.0) / (psi + 1.0)
    p = np.clip(p, 0.0, 1.0 - 1e-12)
    beta = (1.0 - p) / np.clip(m, 1e-300, None)
    U = innov.U
    with np.errstate(divide="ignore"):
        V_branch2 = np.where(U <= p, 0.0, (1.0 / beta) * np.log((1.0 - p) / np.clip(1.0 - U, 1e-300, None)))

    V_next = np.where(use_exp_branch, V_branch2, V_branch1)

    K1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K3 = gamma1 * dt * (1.0 - rho**2)
    K4 = gamma2 * dt * (1.0 - rho**2)
    w = K2 + 0.5 * K4

    # Martingale-correction K0*, per branch.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom1 = 1.0 - 2.0 * w * a
        log_mgf_1 = w * a * b2 / denom1 - 0.5 * np.log(denom1)
        log_mgf_2 = np.log(p + (1.0 - p) * beta / (beta - w))
    if np.any(denom1[~use_exp_branch] <= 0.0):
        raise RuntimeError(
            "QE martingale correction (branch 1) invalid: 1 - 2*w*a <= 0 for some path; "
            "reduce the time step dt"
        )
    if np.any(beta[use_exp_branch] <= w):
        raise RuntimeError(
            "QE martingale correction (branch 2) invalid: beta <= w for some path; "
            "reduce the time step dt"
        )
    log_mgf = np.where(use_exp_branch, log_mgf_2, log_mgf_1)
    K0_star = -(K1 + 0.5 * K3) * V - log_mgf

    diffusion_var = np.clip(K3 * V + K4 * V_next, 0.0, None)
    log_S_next = np.log(S) + K0_star + K1 * V + K2 * V_next + np.sqrt(diffusion_var) * innov.Zs

    if jumps.lambda_ > 0.0:
        k = jumps.compensator_k
        nj = innov.n_jumps.astype(np.float64)
        jump_increment = nj * jumps.mu_j + np.sqrt(nj) * jumps.delta_j * innov.Zj
        log_S_next = log_S_next - jumps.lambda_ * k * dt + jump_increment

    return SimulationStep(S=np.exp(log_S_next), V=np.clip(V_next, 0.0, None), used_exponential_branch=use_exp_branch)


def simulate_paths(
    S0: float,
    V0: float,
    total_dt: float,
    n_substeps: int,
    n_paths: int,
    heston: HestonParams,
    jumps: JumpParams,
    seed: int,
    antithetic: bool = True,
) -> SimulationStep:
    """Simulate n_paths QE(+jump) paths for total_dt, split into n_substeps QE steps.

    Antithetic: the first half and second half of the paths share mirrored
    (negated Gaussian / reflected uniform) innovations at every substep --
    variance reduction for the martingale test in `martingale_test`.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic simulation requires an even n_paths")
    rng = np.random.default_rng(seed)
    sub_dt = total_dt / n_substeps
    half = n_paths // 2 if antithetic else n_paths

    S = np.full(n_paths, S0, dtype=np.float64)
    V = np.full(n_paths, V0, dtype=np.float64)
    used_exp = np.zeros(n_paths, dtype=bool)
    for _ in range(n_substeps):
        if antithetic:
            base = draw_qe_innovations(half, sub_dt, jumps, rng)
            mirror = antithetic_pair(base)
            innov = QEInnovations(
                Zv=np.concatenate([base.Zv, mirror.Zv]),
                U=np.concatenate([base.U, mirror.U]),
                Zs=np.concatenate([base.Zs, mirror.Zs]),
                n_jumps=np.concatenate([base.n_jumps, mirror.n_jumps]),
                Zj=np.concatenate([base.Zj, mirror.Zj]),
            )
        else:
            innov = draw_qe_innovations(n_paths, sub_dt, jumps, rng)
        step = qe_step(S, V, sub_dt, heston, jumps, innov)
        S, V, used_exp = step.S, step.V, step.used_exponential_branch
    return SimulationStep(S=S, V=V, used_exponential_branch=used_exp)


# ===========================================================================
# MC martingale test: E[C(t+Delta,S_{t+Delta},V_{t+Delta};T,K)] - C(t,S,V;T,K)
# Antithetic sampling + (S_{t+Delta}-S_t) as a control variate (E[control]=0
# exactly, by the martingale property already verified above).
# ===========================================================================


@dataclass(frozen=True)
class MartingaleTestResult:
    K: float
    T: float
    t: float
    total_dt: float
    n_substeps: int
    n_paths: int
    C_t: float
    estimate: float  # E_hat[C(t+Delta,...)] - C(t,...), control-variate-adjusted
    std_error: float
    ci95: tuple[float, float]
    control_variate_beta: float
    raw_estimate: float  # same, without the control variate (for comparison)
    raw_std_error: float


def martingale_test(
    K: float,
    T: float,
    t: float,
    S: float,
    V: float,
    total_dt: float,
    n_substeps: int,
    n_paths: int,
    heston: HestonParams,
    jumps: JumpParams,
    seed: int,
) -> MartingaleTestResult:
    """One-step MC martingale test for the fixed contract (T,K), from (t,S,V).

    Y_i = C(t+Delta, S_i, V_i; T,K) - C(t,S,V;T,K), the pathwise change in the
    fixed-contract price. X_i = S_i - S (control variate, E[X]=0 exactly).
    Reports both the raw sample mean and the control-variate-adjusted one.
    """
    step = simulate_paths(S, V, total_dt, n_substeps, n_paths, heston, jumps, seed, antithetic=True)
    tau_terminal = T - (t + total_dt)
    if tau_terminal <= 0.0:
        raise ValueError("t + total_dt must remain strictly before T")

    C_t = float(call_price(S, V, T - t, K, heston, jumps))
    # Vectorized terminal pricing: call_price is JAX and broadcasts over arrays.
    C_terminal = np.asarray(
        jax.vmap(lambda s, v: call_price(s, v, tau_terminal, K, heston, jumps))(
            jnp.asarray(step.S), jnp.asarray(step.V)
        )
    )
    Y = C_terminal - C_t
    X = step.S - S  # control variate, E[X] = 0 exactly

    n = len(Y)
    raw_mean = float(Y.mean())
    raw_se = float(Y.std(ddof=1) / math.sqrt(n))

    cov_YX = float(np.cov(Y, X, ddof=1)[0, 1])
    var_X = float(np.var(X, ddof=1))
    beta = cov_YX / var_X if var_X > 0.0 else 0.0
    Y_cv = Y - beta * X
    cv_mean = float(Y_cv.mean())
    cv_se = float(Y_cv.std(ddof=1) / math.sqrt(n))

    z95 = 1.959963984540054
    ci95 = (cv_mean - z95 * cv_se, cv_mean + z95 * cv_se)

    return MartingaleTestResult(
        K=K, T=T, t=t, total_dt=total_dt, n_substeps=n_substeps, n_paths=n_paths,
        C_t=C_t, estimate=cv_mean, std_error=cv_se, ci95=ci95,
        control_variate_beta=beta, raw_estimate=raw_mean, raw_std_error=raw_se,
    )


# ===========================================================================
# Vectorized continuous-grid residuals (jax.vmap over the whole (K,T) mesh).
#
# The dense mesh (spec default: K in [50,200] step 1, T step 0.01) has
# ~24k points; a per-point Python loop through jax.hessian is far too slow
# (single-point cost ~0.3-0.6s -> ~2-4 hours for the full mesh). vmap over
# the mesh, jit-compiled once, computes the full spec-resolution mesh in a
# few seconds -- verified to match the per-point results exactly.
# ===========================================================================


@dataclass(frozen=True)
class ContinuousGridResult:
    """Batched continuous-DLV diagnostics over a (K,T) mesh, one observation
    state. Arrays all have shape (len(K_mesh), len(T_mesh)).

    R_cont = L_ell + gamma_over_q + jump_term (jump_term is all-zero for
    Heston): stored separately, per the spec's export requirement ("the L,
    Gamma and jump terms needed to recompute R_cont"), rather than only the
    pre-summed residual.
    """

    K_mesh: np.ndarray
    T_mesh: np.ndarray
    t: float
    C: np.ndarray
    q: np.ndarray
    ell: np.ndarray
    L_q: np.ndarray
    L_ell: np.ndarray
    gamma_over_q: np.ndarray
    jump_term: np.ndarray  # zero array for Heston (lambda=0)
    R: np.ndarray
    R_normalized: np.ndarray
    admissible: np.ndarray
    R_nojump: np.ndarray | None = None


def _continuous_single_point(K0, T0, t, S, V, heston: HestonParams, jumps: JumpParams, n_jump_nodes: int):
    """Same computation as `continuous_node_residual`, returning raw jnp
    scalars (no Python floats/dataclass) so it can be wrapped in jax.vmap."""
    ell_field, q_field = make_continuous_dlv_field(K0, T0, heston, jumps)
    C_val = call_price(S, V, T0 - t, K0, heston, jumps)
    q_val = q_field(t, S, V)
    ell_val = ell_field(t, S, V)

    if jumps.lambda_ == 0.0:
        L_q = heston_generator(q_field, t, S, V, heston)
    else:
        L_q = bates_generator(q_field, t, S, V, heston, jumps, n_jump_nodes)

    gamma_over_q = carre_du_champ(ell_field, q_field, t, S, V, heston) / q_field(t, S, V)
    if jumps.lambda_ == 0.0:
        L_ell = heston_generator(ell_field, t, S, V, heston)
        jump_term = jnp.asarray(0.0)
        R = L_ell + gamma_over_q
        normalizer = jnp.abs(L_ell) + jnp.abs(gamma_over_q)
        R_nojump = jnp.asarray(0.0)
    else:
        L_ell = bates_generator(ell_field, t, S, V, heston, jumps, n_jump_nodes)
        jump_term = jump_cross_term(ell_field, q_field, t, S, V, jumps, n_jump_nodes) / q_field(t, S, V)
        R = L_ell + gamma_over_q + jump_term
        normalizer = jnp.abs(L_ell) + jnp.abs(gamma_over_q) + jnp.abs(jump_term)
        L_ell_heston_only = heston_generator(ell_field, t, S, V, heston)
        R_nojump = L_ell_heston_only + gamma_over_q

    R_normalized = jnp.where(normalizer > 0.0, R / normalizer, jnp.nan)
    return C_val, q_val, ell_val, L_q, L_ell, gamma_over_q, jump_term, R, R_normalized, R_nojump


def continuous_grid_residuals(
    K_mesh: np.ndarray,
    T_mesh: np.ndarray,
    t: float,
    S: float,
    V: float,
    heston: HestonParams,
    jumps: JumpParams = JumpParams(),
    q_min: float = Q_MIN_DEFAULT,
    n_jump_nodes: int = 250,
    chunk_size: int = 250,
) -> ContinuousGridResult:
    """Vectorized continuous DLV residuals over the full outer-product (K,T) mesh.

    Processed in chunks of `chunk_size` points: q needs d^2C/dK^2 (a nested
    grad) and the generator/Gamma need a further Hessian in (S,V), so each
    point's computation graph is effectively 4th-order in the underlying
    pricer: vmapping the FULL spec mesh (~24k points) in one batch peaks
    around 15GB+ RSS (observed OOM at 16GB). Chunking keeps peak memory
    bounded and the JIT compile is cached across chunks (only re-triggered
    once for a final, smaller remainder chunk). Measured on the spec's
    default mesh (~24k points): Heston at chunk_size=500 -> ~8s, ~0.6GB
    peak; Bates (with the extra jump quadrature) at chunk_size=250 -> ~60s,
    ~2.7GB peak.
    """
    K_grid, T_grid = np.meshgrid(np.asarray(K_mesh), np.asarray(T_mesh), indexing="ij")
    shape = K_grid.shape
    K_flat, T_flat = K_grid.reshape(-1), T_grid.reshape(-1)
    n_total = len(K_flat)

    fn = jax.jit(
        jax.vmap(lambda K0, T0: _continuous_single_point(K0, T0, t, S, V, heston, jumps, n_jump_nodes))
    )

    C_parts, q_parts, ell_parts, Lq_parts, Lell_parts, gam_parts, jump_parts = [], [], [], [], [], [], []
    R_parts, Rn_parts, Rj_parts = [], [], []
    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        C, q, ell, L_q, L_ell, gam, jt, R, R_norm, R_nojump = fn(
            jnp.asarray(K_flat[start:stop]), jnp.asarray(T_flat[start:stop])
        )
        C_parts.append(np.asarray(C))
        q_parts.append(np.asarray(q))
        ell_parts.append(np.asarray(ell))
        Lq_parts.append(np.asarray(L_q))
        Lell_parts.append(np.asarray(L_ell))
        gam_parts.append(np.asarray(gam))
        jump_parts.append(np.asarray(jt))
        R_parts.append(np.asarray(R))
        Rn_parts.append(np.asarray(R_norm))
        Rj_parts.append(np.asarray(R_nojump))

    q_np = np.concatenate(q_parts).reshape(shape)
    ell_np = np.concatenate(ell_parts).reshape(shape)
    admissible = (q_np > q_min) & (ell_np >= 0.0)
    return ContinuousGridResult(
        K_mesh=np.asarray(K_mesh), T_mesh=np.asarray(T_mesh), t=t,
        C=np.concatenate(C_parts).reshape(shape),
        q=q_np, ell=ell_np,
        L_q=np.concatenate(Lq_parts).reshape(shape),
        L_ell=np.concatenate(Lell_parts).reshape(shape),
        gamma_over_q=np.concatenate(gam_parts).reshape(shape),
        jump_term=np.concatenate(jump_parts).reshape(shape),
        R=np.concatenate(R_parts).reshape(shape),
        R_normalized=np.concatenate(Rn_parts).reshape(shape),
        admissible=admissible,
        R_nojump=np.concatenate(Rj_parts).reshape(shape) if jumps.lambda_ != 0.0 else None,
    )
