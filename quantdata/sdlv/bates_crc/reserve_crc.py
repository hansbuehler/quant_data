r"""Section 3: the exact reserve CRC (DYSANOS II manuscript Section 12,
"A two-sided admissible region with scheduled Levy reserves", Theorem
thm:bcrc-reserve, and the branching Lemma lem:bcrc-branching it rests on).

Branching lemma -- proof status (audited, not assumed; see Task 0 write-up
in ``AUDIT.md`` for the full discussion):

For fixed (kappa,xi,rho) and theta=0 (zero immigration), the Heston affine
transform satisfies E[exp(iu Z_r^{(v,0)})] = exp(v D(u,r)) for the ACTUAL
random variable Z_r^{(v,0)} (the log return, over horizon r, of a
zero-immigration Heston process started at variance v) -- this is standard
affine-model theory (Duffie-Pan-Singleton 2000), not re-derived here, and
does not depend on any positivity property of D itself: it follows from
CIR-with-zero-immigration being a well-posed, pathwise-unique, a.s.
finite-valued SDE for every kappa,xi>0, v>=0 (a standard SDE
existence/uniqueness fact). Consequently, for FIXED r, u -> exp(v D(u,r))
is a genuine characteristic function for every v>=0 (it is the actual
characteristic function of an actual random variable), and the family is
multiplicatively additive in v (exp((v1+v2)D(u,r)) = exp(v1 D)*exp(v2 D),
an algebraic triviality once v->exp(vD) is known to be a characteristic
function at every v). A continuous convolution semigroup of probability
measures indexed by v>=0 has a Levy-Khintchine representation with a
NONNEGATIVE triplet (Sato, "Levy Processes and Infinitely Divisible
Distributions", Theorem 9.8, applied to the semigroup indexed by v rather
than by calendar time) -- this is where the nonnegativity comes from. It is
an ANALYTIC argument, not a finite-frequency numerical sweep, and the
module below treats it as such: ``branching_lemma_sanity_check`` below is
labelled EXACTLY as a supplementary sanity check (a NECESSARY, not
sufficient, condition: |exp(vD(u,r))| <= 1), never as a proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quantdata.sdlv.bates_crc.heston_riccati import (
    GaussianJumpParams,
    HestonParams,
    gaussian_bates_jump_exponent,
    riccati_D_closed_form,
)

# ===========================================================================
# The dominating Gaussian jump measure (eq:bcrc-envelope).
# ===========================================================================


@dataclass(frozen=True)
class JumpBox:
    """The box (eq:bcrc-jump-box) ordinary Gaussian jump parameters must
    stay inside at every reset."""

    Lambda: float
    M: float
    delta_min: float
    delta_max: float
    s_star: float

    def __post_init__(self) -> None:
        if not (0.0 < self.delta_min <= self.delta_max < self.s_star):
            raise ValueError(f"require 0 < delta_min <= delta_max < s_star, got {self}")
        if self.Lambda < 0.0 or self.M < 0.0:
            raise ValueError(f"Lambda, M must be non-negative, got {self}")

    def contains(self, jumps: GaussianJumpParams) -> bool:
        return (
            0.0 <= jumps.lambda_J <= self.Lambda
            and abs(jumps.m_J) <= self.M
            and self.delta_min <= jumps.delta_J <= self.delta_max
        )


def dominating_R_star(box: JumpBox) -> float:
    """R_*, eq. (eq:bcrc-envelope)."""
    return (box.s_star / box.delta_min) * math.exp(
        box.M ** 2 / (2.0 * (box.s_star ** 2 - box.delta_max ** 2))
    )


def dominating_envelope_params(box: JumpBox) -> GaussianJumpParams:
    """nu-bar as a GaussianJumpParams instance (m_J=0, delta_J=s_star,
    lambda_J = Lambda*R_*), so j-bar(u) = gaussian_bates_jump_exponent(u,
    this)."""
    R_star = dominating_R_star(box)
    return GaussianJumpParams(lambda_J=box.Lambda * R_star, m_J=0.0, delta_J=box.s_star)


def verify_envelope_dominates(box: JumpBox, jumps: GaussianJumpParams, *, n_grid: int = 4001) -> dict:
    """Verifies sup_z [phi_{m,delta}(z)/phi_{0,s_*}(z)] <= R_* directly by
    dense grid evaluation of the actual ratio (an analytic closed-form
    bound is also available and checked below) -- this is a genuine
    positivity-of-a-ratio check, not a characteristic-function-positivity
    frequency sweep, so it is not subject to the same caution."""
    if not box.contains(jumps):
        raise ValueError(f"{jumps} lies outside the declared jump box {box}")
    z = np.linspace(-8.0 * box.s_star, 8.0 * box.s_star, n_grid)
    phi_target = np.exp(-0.5 * ((z - jumps.m_J) / jumps.delta_J) ** 2) / (jumps.delta_J * math.sqrt(2.0 * math.pi))
    phi_dom = np.exp(-0.5 * (z / box.s_star) ** 2) / (box.s_star * math.sqrt(2.0 * math.pi))
    ratio = phi_target / phi_dom
    R_star = dominating_R_star(box)
    closed_form_bound = (box.s_star / jumps.delta_J) * math.exp(
        jumps.m_J ** 2 / (2.0 * (box.s_star ** 2 - jumps.delta_J ** 2))
    )
    return dict(
        max_ratio_on_grid=float(ratio.max()), R_star=R_star, closed_form_bound=closed_form_bound,
        dominated_on_grid=bool(ratio.max() <= R_star * (1.0 + 1.0e-9)),
        closed_form_bound_within_R_star=bool(closed_form_bound <= R_star * (1.0 + 1.0e-9)),
    )


def branching_lemma_sanity_check(u_values: tuple[float, ...], r_values: tuple[float, ...],
                                  heston: HestonParams, v_values: tuple[float, ...] = (0.5, 1.0, 3.0)) -> dict:
    """SUPPLEMENTARY sanity check only (see module docstring): verifies the
    NECESSARY condition |exp(v*D(u,r))| <= 1 + tol at finitely many
    (u,r,v). Does NOT, and is not claimed to, prove Levy-measure
    positivity -- that is established analytically (module docstring), not
    by this or any finite sweep."""
    violations = []
    max_abs = 0.0
    for r in r_values:
        for u in u_values:
            D = riccati_D_closed_form(u, r, heston)
            for v in v_values:
                val = abs(np.exp(v * D))
                max_abs = max(max_abs, val)
                if val > 1.0 + 1.0e-8:
                    violations.append(dict(u=u, r=r, v=v, abs_value=val))
    return dict(
        max_abs_value=max_abs, n_violations=len(violations), violations=violations[:10],
        note="supplementary necessary-condition check only; NOT a proof of Levy-measure "
             "positivity (see module docstring for the actual analytic argument)",
    )


# ===========================================================================
# Scheduled reserves: initial state and reset updates.
# ===========================================================================


@dataclass(frozen=True)
class ReserveSchedule:
    """The fixed schedule: reset times, reserve sizes, and theta bounds
    (eq:bcrc-theta-region)."""

    reset_times: tuple[float, ...]  # t_1 < ... < t_N
    reserve_sizes: tuple[float, ...]  # r_1, ..., r_N > 0
    theta_min: float
    theta_max: float

    def __post_init__(self) -> None:
        if len(self.reset_times) != len(self.reserve_sizes):
            raise ValueError("reset_times and reserve_sizes must have the same length")
        if list(self.reset_times) != sorted(self.reset_times):
            raise ValueError("reset_times must be strictly increasing")
        if any(r <= 0.0 for r in self.reserve_sizes):
            raise ValueError("reserve sizes must be strictly positive")
        if not (self.theta_min < self.theta_max):
            raise ValueError("theta_min must be < theta_max")


@dataclass(frozen=True)
class BcrcState:
    """The full CRC state after n resets (n=0: initial). b_used[k] is the
    coefficient b_k^(n) of eq:bcrc-reserve-update (b_k^(n) = r_k - Delta_k
    for k<=n, r_k for k>n)."""

    n_resets_applied: int
    theta_path: tuple[float, ...]  # theta_0, theta_1, ..., theta_n
    b_used: tuple[float, ...]  # length N, current reserve allocation
    jump_params_path: tuple[GaussianJumpParams, ...]  # p_0, p_1, ..., p_n


def initial_state(theta_0: float, jumps_0: GaussianJumpParams, schedule: ReserveSchedule) -> BcrcState:
    if not (schedule.theta_min <= theta_0 <= schedule.theta_max):
        raise ValueError(f"theta_0={theta_0} outside [{schedule.theta_min},{schedule.theta_max}]")
    return BcrcState(
        n_resets_applied=0, theta_path=(theta_0,),
        b_used=tuple(schedule.reserve_sizes), jump_params_path=(jumps_0,),
    )


def admissible_delta_interval(theta_prev: float, r_n: float, schedule: ReserveSchedule) -> tuple[float, float]:
    """[theta_min - theta_prev, min(r_n, theta_max - theta_prev)] -- the
    manuscript's own projection interval, eq. after (eq:bcrc-theta-region)."""
    lo = schedule.theta_min - theta_prev
    hi = min(r_n, schedule.theta_max - theta_prev)
    if lo > hi:
        raise ValueError(
            f"empty admissible interval at theta_prev={theta_prev}, r_n={r_n}: [{lo},{hi}]"
        )
    return lo, hi


def apply_reset(state: BcrcState, *, proposed_delta_n: float, proposed_jumps_n: GaussianJumpParams,
                 box: JumpBox, schedule: ReserveSchedule) -> tuple[BcrcState, dict]:
    """One reset, eq. (eq:bcrc-reserve-update). Projects proposed_delta_n
    into the admissible interval (never silently accepts an infeasible
    proposal) and reports whether projection was needed."""
    n = state.n_resets_applied
    if n >= len(schedule.reset_times):
        raise ValueError("no further scheduled resets remain")
    theta_prev = state.theta_path[-1]
    r_n = schedule.reserve_sizes[n]
    lo, hi = admissible_delta_interval(theta_prev, r_n, schedule)
    delta_n = min(max(proposed_delta_n, lo), hi)
    projected = not (lo <= proposed_delta_n <= hi)

    if not box.contains(proposed_jumps_n):
        raise ValueError(f"proposed jump parameters {proposed_jumps_n} outside declared box {box}")

    theta_n = theta_prev + delta_n
    b_used = list(state.b_used)
    b_used[n] = r_n - delta_n
    if b_used[n] < -1.0e-12:
        raise ValueError(f"reset {n+1} infeasible: b_used={b_used[n]} < 0")
    b_used[n] = max(b_used[n], 0.0)

    new_state = BcrcState(
        n_resets_applied=n + 1,
        theta_path=state.theta_path + (theta_n,),
        b_used=tuple(b_used),
        jump_params_path=state.jump_params_path + (proposed_jumps_n,),
    )
    diagnostics = dict(
        reset_index=n + 1, theta_prev=theta_prev, theta_n=theta_n, delta_n=delta_n,
        proposed_delta_n=proposed_delta_n, projected=projected,
        admissible_interval=(lo, hi), r_n=r_n, b_used_n=b_used[n],
    )
    return new_state, diagnostics


def reserve_ell(u: complex, s: float, state: BcrcState, schedule: ReserveSchedule,
                 heston: HestonParams, envelope: GaussianJumpParams) -> complex:
    """ell_s^(n)(u), eq. (eq:bcrc-reserve-update) (n = state.n_resets_applied),
    valid for s >= t_n (t_0 := 0)."""
    n = state.n_resets_applied
    current_jumps = state.jump_params_path[-1]
    value = gaussian_bates_jump_exponent(u, envelope) - gaussian_bates_jump_exponent(u, current_jumps)
    for k in range(len(schedule.reset_times)):
        t_k = schedule.reset_times[k]
        if s >= t_k:
            value += heston.kappa * state.b_used[k] * riccati_D_closed_form(u, s - t_k, heston)
    return value


def integrated_reserve(u: complex, t_from: float, t_to: float, state: BcrcState, schedule: ReserveSchedule,
                        heston: HestonParams, envelope: GaussianJumpParams, n_quad: int = 400) -> complex:
    """integral_{t_from}^{t_to} ell_s^(n)(u) ds via Gauss-Legendre
    quadrature (ell is smooth in s away from the reset times, which are
    excluded from the open quadrature interval by construction)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half = 0.5 * (t_to - t_from)
    mid = 0.5 * (t_to + t_from)
    s_nodes = mid + half * nodes
    values = np.array([reserve_ell(u, s, state, schedule, heston, envelope) for s in s_nodes])
    return half * np.sum(weights * values)
