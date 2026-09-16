"""Generalised-Bates: time-inhomogeneous, maturity-bucketed Hull-White jump extension.

Extends the Heston diffusion of `oracle.py` (unchanged, imported not modified)
with a compensated compound-Poisson jump component whose log-jump law is
piecewise-constant across five "maturity buckets" I_1..I_5, following the
Hull-White extension used by Gambara & Teichmann (2020), "Consistent
Recalibration Models and Deep Calibration" (arXiv:2006.09455) in their Heston
CRC example.

Convention (per the Task 1B spec, Sec. 1-2): buckets partition TIME-TO-
MATURITY (tau = T - t), not absolute calendar time. For an option observed at
t with maturity T, tau = T - t, and the jump contribution to its
characteristic function is

    exp( sum_b Delta_b(tau) * chi_b(u) ),   Delta_b(tau) = Leb([0,tau] n I_b)

i.e. each fixed-maturity contract is priced under its OWN "residual-time"
jump schedule, counted backward from its own expiry. This is NOT the same as
a single physical process with calendar-time-varying jump law (which would
make Delta_b depend on t and T separately); it is exactly the formula the
spec gives, and it is what makes the constant-bucket reduction (Sec. 4) exact
by construction: when edges cover [0, tau], sum_b Delta_b(tau) = tau always.

Because C_field objects used in the DLV machinery (oracle.make_finite_dlv_field
etc.) combine TWO different maturities (T_prev, T_curr), there is no single
canonical "current jump law at calendar time t" shared by both -- each price
sub-field is priced under its OWN maturity's residual-time schedule. For the
generator identity on a single fixed contract (Sec. 7 item 6) this is
unambiguous: the active bucket is determined by that contract's own
tau = T - t. For the DLV jump-covariation term (Sec. 7 item 7, Corollary
"Jump correction"), which needs one shared jump-size law for the (q, ell)
covariation at a back slab (T_prev, T_curr), we use the convention that the
back slab's OWN maturity T_curr defines the active bucket (tau_curr = T_curr
- t): this matches the theorem's per-node (n, m) statement, where T_m =
T_curr is the slab being tested. This choice is documented here and in
README.md; it is the natural (and only unambiguous) reading of the theorem
under the per-contract-tau convention this spec specifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import oracle as orc

N_BUCKETS = 5

#: Ten CRC calibration maturities (days/365), per spec Sec. 2.
CRC_CALIBRATION_MATURITIES_DAYS: tuple[int, ...] = (7, 14, 30, 60, 90, 120, 180, 270, 360, 440)
CRC_CALIBRATION_MATURITIES: tuple[float, ...] = tuple(d / 365.0 for d in CRC_CALIBRATION_MATURITIES_DAYS)

#: Five future-time buckets I_1..I_5 in time-to-maturity space (years), per
#: spec Sec. 2. Edges coincide with calibration maturities 14, 60, 120, 270
#: days; the last bucket is extended to 2.0 (not 440/365 ~= 1.205) because the
#: existing DLV export includes T=2.0 and the buckets must cover the whole
#: pricing horizon (Sec. 3 validation: "coverage of the complete pricing
#: horizon").
DEFAULT_BUCKET_EDGES: tuple[float, ...] = (0.0, 14.0 / 365.0, 60.0 / 365.0, 120.0 / 365.0, 270.0 / 365.0, 2.0)


# ===========================================================================
# Parameters
# ===========================================================================


@dataclass(frozen=True)
class HullWhiteJumpExtension:
    """Time-inhomogeneous, maturity-bucketed compensated compound-Poisson jump law.

    lambda_ is the (fixed, bucket-independent) Poisson intensity. nu[b],
    delta[b] parametrise Y_b ~ N(nu[b], delta[b]^2), the log-jump size law
    active while tau = T - t lies in bucket I_b (see module docstring).
    """

    lambda_: float
    bucket_edges: tuple[float, ...]
    nu: tuple[float, ...]
    delta: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.lambda_ < 0.0:
            raise ValueError("lambda_ must be non-negative")
        if len(self.nu) != N_BUCKETS:
            raise ValueError(f"exactly {N_BUCKETS} nu_b values required, got {len(self.nu)}")
        if len(self.delta) != N_BUCKETS:
            raise ValueError(f"exactly {N_BUCKETS} delta_b values required, got {len(self.delta)}")
        if len(self.bucket_edges) != N_BUCKETS + 1:
            raise ValueError(f"bucket_edges must have {N_BUCKETS + 1} entries, got {len(self.bucket_edges)}")
        edges = np.asarray(self.bucket_edges, dtype=float)
        if edges[0] != 0.0:
            raise ValueError("bucket_edges[0] must be 0.0 (buckets partition [0, horizon])")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("bucket_edges must be strictly increasing")
        for b, d in enumerate(self.delta):
            if not (d > 0.0):
                raise ValueError(f"delta[{b}] must be strictly positive, got {d}")
        for b, (nu_b, d) in enumerate(zip(self.nu, self.delta)):
            m = nu_b + 0.5 * d * d
            if not math.isfinite(m) or not math.isfinite(math.exp(m)):
                raise ValueError(f"bucket {b}: E[e^Y_b] = exp(nu_b + delta_b^2/2) is not finite")

    def covers(self, horizon: float) -> bool:
        """True iff the buckets cover every tau in [0, horizon] (Sec. 3 validation)."""
        return self.bucket_edges[-1] >= horizon

    @property
    def kappa_J(self) -> tuple[float, ...]:
        """kappa_b^J = E[e^{Y_b} - 1], one per bucket."""
        return tuple(math.exp(nu_b + 0.5 * d * d) - 1.0 for nu_b, d in zip(self.nu, self.delta))


def constant_bucket_extension(
    jumps: orc.JumpParams, bucket_edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES
) -> HullWhiteJumpExtension:
    """The Sec. 4 degenerate case: all five buckets equal to a single JumpParams."""
    return HullWhiteJumpExtension(
        lambda_=jumps.lambda_,
        bucket_edges=bucket_edges,
        nu=(jumps.mu_j,) * N_BUCKETS,
        delta=(jumps.delta_j,) * N_BUCKETS,
    )


#: Regression baseline: EXACTLY the existing default Heston/Bates parameters
#: (Feller ratio 0.48, deliberately sub-Feller -- see oracle.py/README.md).
#: Kept unchanged for the constant-bucket reduction test (Sec. 4/6): the
#: current Bates oracle stays valid without replacing it.
BATES_REGRESSION_HESTON: orc.HestonParams = orc.HESTON_DEFAULT
BATES_REGRESSION_JUMPS: orc.JumpParams = orc.BATES_JUMPS_DEFAULT

#: Paper-faithful CRC experiment parameters: Feller-ADMISSIBLE (2*kappa*theta
#: > xi^2), per spec Sec. 6. 2*2.0*0.04 = 0.16 > 0.35^2 = 0.1225 (ratio
#: ~1.306). kappa, rho kept close to the regression defaults; xi lowered
#: (0.5 -> 0.35) to buy Feller admissibility without changing the sign/scale
#: of vol-of-vol or correlation qualitatively.
CRC_ADMISSIBLE_HESTON: orc.HestonParams = orc.HestonParams(kappa=2.0, theta=0.04, xi=0.35, rho=-0.7)
assert 2.0 * CRC_ADMISSIBLE_HESTON.kappa * CRC_ADMISSIBLE_HESTON.theta > CRC_ADMISSIBLE_HESTON.xi ** 2

#: A genuinely non-degenerate (non-identical-bucket) Generalised-Bates jump
#: term structure for the Sec. 7 "fixed Generalised-Bates null" test: near-
#: term jump risk more negative/volatile than far-term, tapering smoothly
#: toward the regression Bates values, all admissible (delta_b > 0, finite
#: exponential moments by construction of a Gaussian law).
CRC_TEST_EXTENSION: HullWhiteJumpExtension = HullWhiteJumpExtension(
    lambda_=0.3,
    bucket_edges=DEFAULT_BUCKET_EDGES,
    nu=(-0.16, -0.13, -0.10, -0.08, -0.07),
    delta=(0.22, 0.19, 0.15, 0.13, 0.12),
)


# ===========================================================================
# Bucket overlap Delta_b(tau) and the compensated jump exponent chi_b(u)
# ===========================================================================


def bucket_overlap(tau, bucket_edges: tuple[float, ...]) -> jnp.ndarray:
    """Delta_b(tau) = Leb([0,tau] n I_b) for every bucket b, as an array shape (5,).

    I_1 = [e0,e1], I_b = (e_{b-1}, e_b] for b>1; since we only ever need the
    LENGTH of the overlap with [0,tau], the boundary convention (open/closed)
    does not affect this Lebesgue-measure computation.
    """
    edges = jnp.asarray(bucket_edges)
    lo, hi = edges[:-1], edges[1:]
    return jnp.clip(jnp.minimum(tau, hi) - lo, 0.0, None)


def current_bucket_index(tau: float, bucket_edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES) -> int:
    """Python-level (non-traced) index b such that tau in I_b. I_1=[e0,e1] closed
    both ends; I_b=(e_{b-1},e_b] for b>1. Raises if tau is outside coverage."""
    edges = np.asarray(bucket_edges, dtype=float)
    if tau < edges[0] - 1.0e-12 or tau > edges[-1] + 1.0e-12:
        raise ValueError(f"tau={tau} outside bucket coverage [{edges[0]}, {edges[-1]}]")
    tau_c = min(max(tau, edges[0]), edges[-1])
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if (b == 0 and lo <= tau_c <= hi) or (b > 0 and lo < tau_c <= hi):
            return b
    raise AssertionError("unreachable: bucket edges must partition [edges[0], edges[-1]]")


def distance_to_bucket_boundary(tau: float, bucket_edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES) -> float:
    """min |tau - edge| over ALL edges, including the outer coverage boundary
    edges[0]=0 and edges[-1] (=2.0 by default) -- used to keep generator/
    finite-difference test points away from the non-smooth bucket boundaries
    (Sec. 7: "away from deterministic bucket boundaries").

    The outer boundary matters numerically, not just the interior transitions:
    JAX's autodiff of jnp.minimum(tau, hi) at the exact tie tau==hi returns
    the AVERAGED subgradient 0.5 (verified: d/dtau sum_b Delta_b(tau) jumps
    1.0 -> 0.5 -> 0.0 across tau=2.0-eps, 2.0, 2.0+eps), not the correct
    interior one-sided value 1.0 -- so tau exactly at edges[-1] silently
    breaks the generator identity by a term of order 0.5*chi(u) if not
    excluded. This exact combination occurs at the protected DLV grid's
    (t=0, T=2.0) node (tau=2.0) and is excluded from residual testing for
    that reason (see README.md)."""
    edges = np.asarray(bucket_edges, dtype=float)
    return float(np.min(np.abs(edges - tau)))


def _jump_exponent_chi(u: jnp.ndarray, nu: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """chi_b(u) = E[e^{iuY_b}] - 1 - iu*kappa_b^J, one value per bucket, broadcast
    over the leading shape of u. Returns shape u.shape + (5,)."""
    u = u[..., None]
    cf = jnp.exp(1j * u * nu - 0.5 * (u ** 2) * (delta ** 2))
    kappa_J = jnp.exp(nu + 0.5 * delta ** 2) - 1.0
    return cf - 1.0 - 1j * u * kappa_J


def generalized_jump_log_cf(u: jnp.ndarray, tau: float, extension: HullWhiteJumpExtension) -> jnp.ndarray:
    """log( E[exp(iu * jump part of X_T - X_t)] ) = sum_b Delta_b(tau) chi_b(u)."""
    if extension.lambda_ == 0.0:
        return jnp.zeros_like(u, dtype=jnp.complex128) if hasattr(u, "shape") else jnp.asarray(0.0 + 0.0j)
    nu = jnp.asarray(extension.nu)
    delta = jnp.asarray(extension.delta)
    chi = _jump_exponent_chi(u, nu, delta)
    overlaps = bucket_overlap(tau, extension.bucket_edges)
    return extension.lambda_ * jnp.sum(overlaps * chi, axis=-1)


# ===========================================================================
# Characteristic function and pricer (JAX, differentiable in t/S/V)
# ===========================================================================


def generalized_bates_characteristic_function(
    u: jnp.ndarray, tau: float, V: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension
) -> jnp.ndarray:
    """psi_GB(u; tau, V) = psi_Heston(u; tau, V) * exp(sum_b Delta_b(tau) chi_b(u)).

    Reuses oracle.characteristic_function (Heston-only, jumps=JumpParams())
    for the diffusion part rather than re-deriving A(tau,u), B(tau,u): this
    keeps the validated Little-Heston-trap formula as the single source of
    truth for the diffusion piece.
    """
    psi_H = orc.characteristic_function(u, tau, V, heston, orc.JumpParams())
    return psi_H * jnp.exp(generalized_jump_log_cf(u, tau, extension))


def generalized_bates_call_price(
    t: float,
    S: float,
    V: float,
    T: float,
    K: float,
    heston: orc.HestonParams,
    extension: HullWhiteJumpExtension,
    quadrature: orc.QuadratureNodes = orc.DEFAULT_PRICE_QUADRATURE,
) -> jnp.ndarray:
    """European call price C(t,S,V;T,K) under the Generalised-Bates model, r=q=0.

    Takes t and T as SEPARATE explicit arguments (tau = T - t computed
    INSIDE this function) rather than a single tau argument: this is
    required so that jax.grad(..., argnums=0) w.r.t. t, with T (and hence
    the active jump bucket) held fixed via closure, differentiates correctly
    through the bucket-overlap Delta_b(T - t) -- exactly mirroring
    oracle.make_price_field's `T - t` pattern.
    """
    tau = T - t
    k_S = jnp.log(S / K)
    shifted = quadrature.u - 0.5j
    psi = generalized_bates_characteristic_function(shifted, tau, V, heston, extension)
    integrand = jnp.real(jnp.exp(1j * quadrature.u * k_S) * psi) / (quadrature.u ** 2 + 0.25)
    integral = jnp.sum(quadrature.w * integrand)
    return S - jnp.sqrt(S * K) / jnp.pi * integral


def generalized_bates_put_price(
    t: float,
    S: float,
    V: float,
    T: float,
    K: float,
    heston: orc.HestonParams,
    extension: HullWhiteJumpExtension,
    quadrature: orc.QuadratureNodes = orc.DEFAULT_PRICE_QUADRATURE,
) -> jnp.ndarray:
    """Put via parity: P = C - S + K (r=q=0). NOT independent of the call price
    (see generalized_bates_heston93_put_price for a genuinely independent put)."""
    return generalized_bates_call_price(t, S, V, T, K, heston, extension, quadrature) - S + K


def make_generalized_price_field(T: float, K: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension):
    """f(t,S,V) = C(t,S,V;T,K), for use with oracle.field_derivatives / heston_generator."""
    return lambda t, S, V: generalized_bates_call_price(t, S, V, T, K, heston, extension)


# ===========================================================================
# Independent cross-check pricer: numpy Gil-Pelaez P1/P2, genuinely
# reimplemented (not calling the JAX characteristic function above), mirroring
# oracle.py's _np_characteristic_function / _heston93_p1_p2 / heston93_call_price.
# ===========================================================================


def _np_jump_exponent_chi(u: np.ndarray, nu: np.ndarray, delta: np.ndarray) -> np.ndarray:
    u = u[..., None]
    cf = np.exp(1j * u * nu - 0.5 * (u ** 2) * (delta ** 2))
    kappa_J = np.exp(nu + 0.5 * delta ** 2) - 1.0
    return cf - 1.0 - 1j * u * kappa_J


def _np_bucket_overlap(tau: float, bucket_edges: tuple[float, ...]) -> np.ndarray:
    edges = np.asarray(bucket_edges, dtype=float)
    lo, hi = edges[:-1], edges[1:]
    return np.clip(np.minimum(tau, hi) - lo, 0.0, None)


def _np_generalized_jump_log_cf(u: np.ndarray, tau: float, extension: HullWhiteJumpExtension) -> np.ndarray:
    if extension.lambda_ == 0.0:
        return np.zeros_like(u, dtype=np.complex128)
    nu = np.asarray(extension.nu, dtype=np.float64)
    delta = np.asarray(extension.delta, dtype=np.float64)
    chi = _np_jump_exponent_chi(u, nu, delta)
    overlaps = _np_bucket_overlap(tau, extension.bucket_edges)
    return extension.lambda_ * np.sum(overlaps * chi, axis=-1)


def _np_generalized_bates_characteristic_function(
    u: np.ndarray, tau: float, V: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension
) -> np.ndarray:
    psi_H = orc._np_characteristic_function(u, tau, V, heston, orc.JumpParams())
    return psi_H * np.exp(_np_generalized_jump_log_cf(u, tau, extension))


def _generalized_p1_p2(
    S: float, V: float, tau: float, K: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension,
    n: int, u_max: float,
) -> tuple[float, float]:
    """Gil-Pelaez digital probabilities for the Generalised-Bates model.
    Threshold is log(K/S) -- see oracle._heston93_p1_p2's docstring."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * u_max * (x + 1.0)
    weight = 0.5 * u_max * w
    u_c = u.astype(np.complex128)

    log_k = math.log(K / S)
    psi_minus_i = _np_generalized_bates_characteristic_function(np.array([-1j]), tau, V, heston, extension)[0]
    f1 = _np_generalized_bates_characteristic_function(u_c - 1j, tau, V, heston, extension) / psi_minus_i
    f2 = _np_generalized_bates_characteristic_function(u_c, tau, V, heston, extension)
    integrand1 = np.real(np.exp(-1j * u_c * log_k) * f1 / (1j * u_c))
    integrand2 = np.real(np.exp(-1j * u_c * log_k) * f2 / (1j * u_c))
    P1 = 0.5 + (1.0 / math.pi) * np.sum(weight * integrand1)
    P2 = 0.5 + (1.0 / math.pi) * np.sum(weight * integrand2)
    return float(P1), float(P2)


def generalized_bates_heston93_call_price(
    S: float, V: float, tau: float, K: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension,
    n: int = 1000, u_max: float = 800.0,
) -> float:
    """Independent cross-check pricer (Gil-Pelaez P1/P2), C = S*P1 - K*P2."""
    P1, P2 = _generalized_p1_p2(S, V, tau, K, heston, extension, n, u_max)
    return S * P1 - K * P2


def generalized_bates_heston93_put_price(
    S: float, V: float, tau: float, K: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension,
    n: int = 1000, u_max: float = 800.0,
) -> float:
    """Independent put price via the SAME P1/P2 decomposition (genuinely
    independent of generalized_bates_put_price's call-S+K parity)."""
    P1, P2 = _generalized_p1_p2(S, V, tau, K, heston, extension, n, u_max)
    return K * (1.0 - P2) - S * (1.0 - P1)


# ===========================================================================
# Martingale normalisation: psi_GB(-i; tau, V) must equal 1 (Sec. 5).
# ===========================================================================


def martingale_normalisation_error(
    tau: float, V: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension
) -> complex:
    """psi_GB(-i; tau, V) - 1, should be ~0 (E[e^{X_T - X_t}|F_t] = 1)."""
    val = generalized_bates_characteristic_function(jnp.asarray(-1j, dtype=jnp.complex128), tau, V, heston, extension)
    return complex(val) - 1.0 + 0.0j


# ===========================================================================
# Generator L_GB, jump-generator term and jump cross-term, using the CURRENT
# active bucket (see module docstring for the convention). Reuses
# oracle.heston_generator (diffusion part, unchanged -- correct for ANY
# composed field, see module docstring), oracle.field_derivatives,
# oracle.delta_y and oracle.carre_du_champ (all field-generic, no jump
# dependence baked in) exactly as-is.
# ===========================================================================


def jump_generator_term_bucketed(
    f, t: float, S: float, V: float, extension: HullWhiteJumpExtension, tau_for_bucket: float, n_nodes: int = 250
) -> jnp.ndarray:
    r"""lambda * E_{Y~N(nu_b,delta_b^2)}[f(t,Se^Y,V) - f(t,S,V) - S f_S (e^Y-1)],
    b = current_bucket_index(tau_for_bucket, extension.bucket_edges)."""
    if extension.lambda_ == 0.0:
        return jnp.asarray(0.0)
    b = current_bucket_index(tau_for_bucket, extension.bucket_edges)
    quad = orc.gauss_hermite_jump_quadrature(n_nodes, extension.nu[b], extension.delta[b])
    _f_t, f_S, *_ = orc.field_derivatives(f, t, S, V)
    df = orc.delta_y(f, t, S, V, quad.u)
    compensator = S * f_S * (jnp.exp(quad.u) - 1.0)
    return extension.lambda_ * jnp.sum(quad.w * (df - compensator))


def generalized_bates_generator(
    f, t: float, S: float, V: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension,
    tau_for_bucket: float, n_nodes: int = 250,
) -> jnp.ndarray:
    r"""L_GB f = L_H f + lambda E_Y[f(t,Se^Y,V)-f(t,S,V)-S f_S (e^Y-1)], jump law
    from the bucket active at tau_for_bucket = T - t for the contract/slab
    that defines f (see module docstring for the multi-maturity convention)."""
    return orc.heston_generator(f, t, S, V, heston) + jump_generator_term_bucketed(
        f, t, S, V, extension, tau_for_bucket, n_nodes
    )


def jump_cross_term_bucketed(
    f, g, t: float, S: float, V: float, extension: HullWhiteJumpExtension, tau_for_bucket: float, n_nodes: int = 250
) -> jnp.ndarray:
    r"""lambda * E_Y[Delta_y f * Delta_y g] under the bucket active at tau_for_bucket."""
    if extension.lambda_ == 0.0:
        return jnp.asarray(0.0)
    b = current_bucket_index(tau_for_bucket, extension.bucket_edges)
    quad = orc.gauss_hermite_jump_quadrature(n_nodes, extension.nu[b], extension.delta[b])
    df = orc.delta_y(f, t, S, V, quad.u)
    dg = orc.delta_y(g, t, S, V, quad.u)
    return extension.lambda_ * jnp.sum(quad.w * df * dg)


# ===========================================================================
# Finite-grid DLV: reuses oracle.py's butterfly/DLV field construction
# (make_finite_butterfly_field / make_finite_dlv_field are field-generic --
# they only need a `call_price`-shaped function; here we substitute
# generalized_bates_call_price for oracle.call_price), and its own
# generator/jump machinery above.
# ===========================================================================


def make_generalized_finite_butterfly_field(
    K_minus: float, K_n: float, K_plus: float, T: float, heston: orc.HestonParams, extension: HullWhiteJumpExtension
):
    def q(t, S, V):
        C_plus = generalized_bates_call_price(t, S, V, T, K_plus, heston, extension)
        C_n = generalized_bates_call_price(t, S, V, T, K_n, heston, extension)
        C_minus = generalized_bates_call_price(t, S, V, T, K_minus, heston, extension)
        return (C_plus - C_n) / (K_plus - K_n) - (C_n - C_minus) / (K_n - K_minus)
    return q


def make_generalized_finite_dlv_field(
    K_minus: float, K_n: float, K_plus: float, T_prev: float, T_curr: float,
    heston: orc.HestonParams, extension: HullWhiteJumpExtension,
):
    q_field = make_generalized_finite_butterfly_field(K_minus, K_n, K_plus, T_curr, heston, extension)
    omega = K_n ** 2 * (T_curr - T_prev) / (K_plus - K_minus)

    def ell(t, S, V):
        C_curr = generalized_bates_call_price(t, S, V, T_curr, K_n, heston, extension)
        C_prev = generalized_bates_call_price(t, S, V, T_prev, K_n, heston, extension)
        return (C_curr - C_prev) / (omega * q_field(t, S, V))
    return ell, q_field


@dataclass(frozen=True)
class GeneralizedNodeResidual:
    """Diagnostics at one (strike, expiry) DLV node under the Generalised-Bates
    model. The active jump bucket for the generator/cross-term is taken from
    tau_curr = T_curr - t (see module docstring for the convention).

    cross_bucket_slab: True iff bucket(T_prev - t) != bucket(T_curr - t).
    THIS IS NOT AN ERROR CONDITION but a genuine structural feature of the
    per-contract-tau convention: when the previous- and current-expiry calls
    of a back slab are priced under DIFFERENT active buckets at t, they are
    governed by two DIFFERENT maturity-indexed models (see module docstring),
    so the back-slab theorem's premise -- a single shared jump law governing
    BOTH expiries -- does not hold for that (t, m) pair, and R is expected to
    be O(bucket-parameter-scale), not O(1e-7). Confirmed numerically:
    cross_bucket_slab=False nodes reach ~1e-13-1e-16; cross_bucket_slab=True
    nodes sit at ~1e-3, exactly the scale of the jump-law mismatch between
    the two buckets. See README.md."""

    n_index: int
    m_index: int
    K: float
    T: float
    t: float
    tau_curr: float
    tau_prev: float
    bucket_curr: int
    bucket_prev: int
    cross_bucket_slab: bool
    C: float
    q: float
    ell: float
    admissible: bool
    excluded_near_boundary: bool
    L_C: float
    L_q: float
    R: float
    R_nojump: float


def generalized_finite_grid_node_residual(
    n_idx: int,
    m_idx: int,
    t: float,
    S: float,
    V: float,
    heston: orc.HestonParams,
    extension: HullWhiteJumpExtension,
    strikes: tuple[float, ...] = orc.FIXED_STRIKES,
    expiries: tuple[float, ...] = orc.FIXED_EXPIRIES,
    q_min: float = orc.Q_MIN_DEFAULT,
    n_jump_nodes: int = 250,
    boundary_tol: float = 0.02,
) -> GeneralizedNodeResidual:
    """Generalised-Bates analogue of oracle.finite_grid_node_residual: the
    jump-corrected DLV drift residual R = L_ell + gamma_over_q/q + jump_term/q,
    all evaluated with the active bucket at tau_curr = T_curr - t."""
    K_minus, K_n, K_plus = strikes[n_idx - 1], strikes[n_idx], strikes[n_idx + 1]
    T_prev, T_curr = expiries[m_idx - 1], expiries[m_idx]
    tau_curr = T_curr - t
    tau_prev = T_prev - t

    ell_field, q_field = make_generalized_finite_dlv_field(K_minus, K_n, K_plus, T_prev, T_curr, heston, extension)
    C_field = make_generalized_price_field(T_curr, K_n, heston, extension)

    C_val = float(C_field(t, S, V))
    q_val = float(q_field(t, S, V))
    ell_val = float(ell_field(t, S, V))
    admissible = bool(q_val > q_min and ell_val >= 0.0)
    excluded = distance_to_bucket_boundary(tau_curr, extension.bucket_edges) < boundary_tol

    L_C = float(generalized_bates_generator(C_field, t, S, V, heston, extension, tau_curr, n_jump_nodes))
    L_q = float(generalized_bates_generator(q_field, t, S, V, heston, extension, tau_curr, n_jump_nodes))

    gamma_over_q = orc.carre_du_champ(ell_field, q_field, t, S, V, heston) / q_field(t, S, V)
    L_ell = generalized_bates_generator(ell_field, t, S, V, heston, extension, tau_curr, n_jump_nodes)
    jump_term = jump_cross_term_bucketed(ell_field, q_field, t, S, V, extension, tau_curr, n_jump_nodes) / q_field(t, S, V)
    R = float(L_ell + gamma_over_q + jump_term)
    L_ell_heston_only = orc.heston_generator(ell_field, t, S, V, heston)
    R_nojump = float(L_ell_heston_only + gamma_over_q)

    bucket_curr = current_bucket_index(tau_curr, extension.bucket_edges)
    bucket_prev = current_bucket_index(tau_prev, extension.bucket_edges) if tau_prev > 0.0 else bucket_curr

    return GeneralizedNodeResidual(
        n_index=n_idx, m_index=m_idx, K=K_n, T=T_curr, t=t, tau_curr=tau_curr, tau_prev=tau_prev,
        bucket_curr=bucket_curr, bucket_prev=bucket_prev, cross_bucket_slab=(bucket_curr != bucket_prev),
        C=C_val, q=q_val, ell=ell_val, admissible=admissible, excluded_near_boundary=excluded,
        L_C=L_C, L_q=L_q, R=R, R_nojump=R_nojump,
    )
