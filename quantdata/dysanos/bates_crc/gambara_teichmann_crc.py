"""Paper-aligned generalized-Bates building blocks for CRC Algorithm 1.

This module implements the model used in Gambara--Teichmann, section 7.3:
Heston stochastic variance plus compensated compound-Poisson Gaussian jumps.
The jump mean and standard deviation are piecewise constant in five maturity
buckets, while ``kappa`` and the Poisson intensity remain fixed across CRC
resets.  It is intentionally independent of the repository's DLV-chain Bates
extension.

The online inverse map in Algorithm 1 is a separate numerical object.  A reset
is accepted by :func:`regrid_jump_parameters` only when repricing the fixed
contracts meets the caller's absolute-price tolerance.  Failed approximate
inversions therefore cannot silently be labelled CRC resets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantdata.dysanos.bates_crc.heston_riccati import (
    GaussianJumpParams,
    HestonParams,
    gaussian_bates_jump_exponent,
    riccati_A_closed_form,
    riccati_D_closed_form,
)
from quantdata.dysanos.bates_crc.real_data_study.fast_bates_pricer import (
    _ALPHA,
    _u_grid,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class FiveBucketJumps:
    """Gaussian log-jump laws on five consecutive time-to-maturity buckets."""

    intensity: float
    boundaries: tuple[float, float, float, float]
    means: tuple[float, float, float, float, float]
    stds: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError(f"intensity must be finite and non-negative, got {self.intensity}")
        if len(self.boundaries) != 4 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.boundaries
        ):
            raise ValueError(f"expected four finite positive boundaries, got {self.boundaries}")
        if any(right <= left for left, right in zip(self.boundaries, self.boundaries[1:])):
            raise ValueError(f"boundaries must be strictly increasing, got {self.boundaries}")
        if len(self.means) != 5 or any(not math.isfinite(value) for value in self.means):
            raise ValueError(f"expected five finite jump means, got {self.means}")
        if len(self.stds) != 5 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.stds
        ):
            raise ValueError(f"expected five finite positive jump stds, got {self.stds}")

    def bucket_durations(self, tau: float) -> FloatArray:
        """Elapsed year fractions in each bucket over ``[0, tau]``."""

        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError(f"tau must be finite and non-negative, got {tau}")
        edges = np.asarray((0.0, *self.boundaries, math.inf), dtype=np.float64)
        return np.maximum(np.minimum(tau, edges[1:]) - edges[:-1], 0.0)

    def integrated_exponent(self, u: NDArray[np.complex128], tau: float) -> NDArray[np.complex128]:
        """Return the compensated jump exponent integrated over ``[0,tau]``."""

        u = np.asarray(u, dtype=np.complex128)
        total = np.zeros_like(u)
        for duration, mean, std in zip(self.bucket_durations(tau), self.means, self.stds):
            if duration > 0.0:
                params = GaussianJumpParams(lambda_J=self.intensity, m_J=mean, delta_J=std)
                total += duration * gaussian_bates_jump_exponent(u, params)
        return total


@dataclass(frozen=True, kw_only=True)
class GeneralizedBatesState:
    """State and parameters needed to price one generalized-Bates surface."""

    log_spot: float
    variance: float
    heston: HestonParams
    jumps: FiveBucketJumps

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_spot):
            raise ValueError(f"log_spot must be finite, got {self.log_spot}")
        if not math.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError(f"variance must be finite and non-negative, got {self.variance}")


@dataclass(frozen=True, kw_only=True)
class RegridResult:
    """Auditable result of one Algorithm-1 inverse-map application."""

    jumps: FiveBucketJumps
    prices_before: FloatArray
    prices_after: FloatArray
    heldout_prices_before: FloatArray
    heldout_prices_after: FloatArray
    calibration_max_abs_price_error: float
    heldout_max_abs_price_error: float
    max_abs_price_error: float
    rms_price_error: float
    tolerance: float
    function_evaluations: int
    accepted: bool

    def require_accepted(self) -> FiveBucketJumps:
        """Return the regridded jumps, refusing an approximate failed reset."""

        if not self.accepted:
            raise RuntimeError(
                "generalized-Bates inverse map did not preserve fixed-contract prices: "
                f"max error {self.max_abs_price_error:.6g} exceeds {self.tolerance:.6g}"
            )
        return self.jumps


def paired_maturity_boundaries(maturities: FloatArray) -> tuple[float, float, float, float]:
    """Create five paper-style buckets from exactly ten increasing maturities.

    Each Gaussian pair applies to two adjacent surface maturities.  Boundaries
    lie halfway between maturity nodes 2/3, 4/5, 6/7, and 8/9 (one-based).
    """

    maturities = np.asarray(maturities, dtype=np.float64)
    if maturities.shape != (10,) or np.any(~np.isfinite(maturities)):
        raise ValueError("maturities must contain exactly ten finite values")
    if maturities[0] <= 0.0 or np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities must be positive and strictly increasing")
    return tuple(float(0.5 * (maturities[index] + maturities[index + 1])) for index in (1, 3, 5, 7))


def price_calls_generalized_bates(
    state: GeneralizedBatesState,
    strikes: FloatArray,
    expiries: FloatArray,
    *,
    n_quad: int = 240,
    u_max: float = 600.0,
) -> FloatArray:
    """Price fixed-strike calls in spot currency with zero rates/dividends.

    ``strikes`` are absolute strikes.  Internally the Fourier integral uses
    strike divided by the current spot; multiplying by spot restores currency
    units.  This makes the fixed-contract semantics explicit for path resets.
    """

    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    if strikes.ndim != 1 or strikes.shape != expiries.shape:
        raise ValueError("strikes and expiries must be equal-length one-dimensional arrays")
    if np.any(strikes <= 0.0) or np.any(expiries < 0.0):
        raise ValueError("strikes must be positive and expiries non-negative")

    spot = math.exp(state.log_spot)
    relative_strikes = strikes / spot
    out = np.empty_like(strikes)
    u_real, weights = _u_grid(n_quad, u_max)
    u = u_real - 1j * (_ALPHA + 1.0)
    denominator = _ALPHA**2 + _ALPHA - u_real**2 + 1j * (2.0 * _ALPHA + 1.0) * u_real

    for tau in np.unique(expiries):
        mask = expiries == tau
        if tau == 0.0:
            out[mask] = np.maximum(spot - strikes[mask], 0.0)
            continue
        heston = state.heston
        log_transform = (
            state.variance * riccati_D_closed_form(u, float(tau), heston)
            + riccati_A_closed_form(u, float(tau), heston)
            + state.jumps.integrated_exponent(u, float(tau))
        )
        log_strikes = np.log(relative_strikes[mask])
        integrand = np.exp(-1j * np.outer(u_real, log_strikes)) * (
            np.exp(log_transform) / denominator
        )[:, None]
        integral = np.sum(weights[:, None] * integrand, axis=0)
        out[mask] = spot * np.exp(-_ALPHA * log_strikes) * integral.real / math.pi
    return out


def bates_antithetic_step(
    state: GeneralizedBatesState,
    *,
    dt: float,
    normal_pairs: FloatArray,
    jump_counts: NDArray[np.int64] | None = None,
    jump_normals: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """One full-truncation Bates step while preserving supplied pair order.

    ``normal_pairs`` has shape ``(pairs, 2, 2)`` and must be antithetic along
    axis 1.  Components are the independent spot and variance Gaussian shocks.
    Optional jump inputs have shape ``(pairs, 2)``; they are explicit because
    Poisson antithetics require a chosen coupling rather than sign negation.
    """

    normals = np.asarray(normal_pairs, dtype=np.float64)
    if normals.ndim != 3 or normals.shape[1:] != (2, 2):
        raise ValueError("normal_pairs must have shape (pairs, 2, 2)")
    if not np.allclose(normals[:, 0], -normals[:, 1], rtol=0.0, atol=1.0e-14):
        raise ValueError("normal_pairs must contain adjacent antithetic pairs")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    pair_shape = normals.shape[:2]
    if jump_counts is None:
        counts = np.zeros(pair_shape, dtype=np.int64)
    else:
        counts = np.asarray(jump_counts, dtype=np.int64)
        if counts.shape != pair_shape or np.any(counts < 0):
            raise ValueError(f"jump_counts must have shape {pair_shape} and be non-negative")
    if jump_normals is None:
        jump_z = np.zeros(pair_shape, dtype=np.float64)
    else:
        jump_z = np.asarray(jump_normals, dtype=np.float64)
        if jump_z.shape != pair_shape:
            raise ValueError(f"jump_normals must have shape {pair_shape}")

    heston = state.heston
    variance = max(state.variance, 0.0)
    sqrt_variance_dt = math.sqrt(variance * dt)
    spot_z = normals[:, :, 0]
    variance_z = normals[:, :, 1]
    correlated_spot_z = heston.rho * variance_z + math.sqrt(max(1.0 - heston.rho**2, 0.0)) * spot_z
    next_variance = np.maximum(
        variance + heston.kappa * (heston.theta - variance) * dt + heston.xi * sqrt_variance_dt * variance_z,
        0.0,
    )

    mean = state.jumps.means[0]
    std = state.jumps.stds[0]
    compensator = state.jumps.intensity * (math.exp(mean + 0.5 * std**2) - 1.0)
    aggregate_jump = counts * mean + np.sqrt(counts) * std * jump_z
    next_log_spot = (
        state.log_spot
        + (-0.5 * variance - compensator) * dt
        + sqrt_variance_dt * correlated_spot_z
        + aggregate_jump
    )
    return next_log_spot, next_variance


def regrid_jump_parameters(
    state_before_reset: GeneralizedBatesState,
    heston_after_reset: HestonParams,
    strikes: FloatArray,
    expiries: FloatArray,
    *,
    heldout_strikes: FloatArray,
    heldout_expiries: FloatArray,
    tolerance: float,
    n_quad: int = 240,
    u_max: float = 600.0,
    max_nfev: int = 600,
) -> RegridResult:
    """Numerically apply Algorithm 1 step 6 on a small fixed-contract grid.

    This direct least-squares inverse is suitable for structural checks and
    generating supervised inverse-map data.  It is not fast enough to replace
    the paper's learned inverse at every state of a 100,000-path simulation.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and positive, got {tolerance}")
    if heston_after_reset.kappa != state_before_reset.heston.kappa:
        raise ValueError("Algorithm 1 keeps kappa fixed across a reset")

    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    heldout_strikes = np.asarray(heldout_strikes, dtype=np.float64)
    heldout_expiries = np.asarray(heldout_expiries, dtype=np.float64)
    if heldout_strikes.size == 0 or heldout_strikes.shape != heldout_expiries.shape:
        raise ValueError("a non-empty equal-length held-out contract grid is required")
    prices_before = price_calls_generalized_bates(
        state_before_reset, strikes, expiries, n_quad=n_quad, u_max=u_max
    )
    old_jumps = state_before_reset.jumps
    x0 = np.asarray((*old_jumps.means, *np.log(old_jumps.stds)), dtype=np.float64)

    def decode(values: FloatArray) -> FiveBucketJumps:
        return FiveBucketJumps(
            intensity=old_jumps.intensity,
            boundaries=old_jumps.boundaries,
            means=tuple(float(value) for value in values[:5]),
            stds=tuple(float(value) for value in np.exp(values[5:])),
        )

    def residual(values: FloatArray) -> FloatArray:
        candidate = GeneralizedBatesState(
            log_spot=state_before_reset.log_spot,
            variance=state_before_reset.variance,
            heston=heston_after_reset,
            jumps=decode(values),
        )
        return price_calls_generalized_bates(
            candidate, strikes, expiries, n_quad=n_quad, u_max=u_max
        ) - prices_before

    lower = np.asarray((-0.75,) * 5 + (math.log(0.005),) * 5)
    upper = np.asarray((0.75,) * 5 + (math.log(1.0),) * 5)
    solution = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
    )
    fitted_jumps = decode(solution.x)
    state_after = GeneralizedBatesState(
        log_spot=state_before_reset.log_spot,
        variance=state_before_reset.variance,
        heston=heston_after_reset,
        jumps=fitted_jumps,
    )
    prices_after = price_calls_generalized_bates(
        state_after, strikes, expiries, n_quad=n_quad, u_max=u_max
    )
    heldout_prices_before = price_calls_generalized_bates(
        state_before_reset,
        heldout_strikes,
        heldout_expiries,
        n_quad=n_quad,
        u_max=u_max,
    )
    heldout_prices_after = price_calls_generalized_bates(
        state_after,
        heldout_strikes,
        heldout_expiries,
        n_quad=n_quad,
        u_max=u_max,
    )
    errors = prices_after - prices_before
    heldout_errors = heldout_prices_after - heldout_prices_before
    calibration_maximum = float(np.max(np.abs(errors)))
    heldout_maximum = float(np.max(np.abs(heldout_errors)))
    maximum = max(calibration_maximum, heldout_maximum)
    return RegridResult(
        jumps=fitted_jumps,
        prices_before=prices_before,
        prices_after=prices_after,
        heldout_prices_before=heldout_prices_before,
        heldout_prices_after=heldout_prices_after,
        calibration_max_abs_price_error=calibration_maximum,
        heldout_max_abs_price_error=heldout_maximum,
        max_abs_price_error=maximum,
        rms_price_error=float(np.sqrt(np.mean(np.concatenate((errors, heldout_errors)) ** 2))),
        tolerance=tolerance,
        function_evaluations=int(solution.nfev),
        accepted=bool(solution.success and maximum <= tolerance),
    )
