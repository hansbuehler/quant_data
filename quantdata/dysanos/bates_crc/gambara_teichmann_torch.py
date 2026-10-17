"""Batched Torch valuation for the generalized-Bates CRC model.

The scalar reference implementation in :mod:`gambara_teichmann_crc` remains
the audit oracle.  This module evaluates the same Carr--Madan integral across
many independent path states, which is required for the 100,000-path dynamic
diagnostic.  ``jump_start`` makes the financial timing explicit: before a CRC
reset at the end of a step, the remaining option is valued with the old
time-inhomogeneous jump law over ``[jump_start, jump_start + expiry]``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

_ALPHA = 1.5


@lru_cache(maxsize=16)
def _quadrature_arrays(n_quad: int, u_max: float) -> tuple[np.ndarray, np.ndarray]:
    if n_quad <= 0 or not math.isfinite(u_max) or u_max <= 0.0:
        raise ValueError("quadrature size and upper bound must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half = 0.5 * u_max
    return half * (nodes + 1.0), half * weights


def _bucket_durations(
    boundaries: tuple[float, float, float, float],
    *,
    start: float,
    expiry: float,
) -> tuple[float, float, float, float, float]:
    if (
        not math.isfinite(start)
        or start < 0.0
        or not math.isfinite(expiry)
        or expiry < 0.0
    ):
        raise ValueError("jump interval must have a finite non-negative start and expiry")
    edges = (0.0, *boundaries, math.inf)
    end = start + expiry

    def integrated_until(value: float, left: float, right: float) -> float:
        return max(min(value, right) - left, 0.0)

    return tuple(
        integrated_until(end, left, right)
        - integrated_until(start, left, right)
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    )


def price_calls_generalized_bates_batched(
    *,
    log_spot: torch.Tensor,
    variance: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
    jump_means: torch.Tensor,
    jump_stds: torch.Tensor,
    strikes: torch.Tensor,
    expiry: float,
    kappa: float,
    intensity: float,
    boundaries: tuple[float, float, float, float],
    jump_start: float = 0.0,
    n_quad: int = 240,
    u_max: float = 600.0,
) -> torch.Tensor:
    """Price a strike row for each independent generalized-Bates state.

    State tensors have shape ``(paths,)``, jump tensors ``(paths, 5)``, and
    ``strikes`` either ``(strike_count,)`` or ``(paths, strike_count)``.
    Returned prices are in spot currency and have shape
    ``(paths, strike_count)``.
    """

    state = (log_spot, variance, theta, xi, rho)
    if any(value.ndim != 1 for value in state):
        raise ValueError("generalized-Bates state tensors must be one-dimensional")
    path_count = len(log_spot)
    if any(len(value) != path_count for value in state[1:]):
        raise ValueError("generalized-Bates state tensors must have matching path axes")
    if jump_means.shape != (path_count, 5) or jump_stds.shape != (path_count, 5):
        raise ValueError("jump means and standard deviations must have shape (paths, 5)")
    if strikes.ndim == 1:
        strikes = strikes[None].expand(path_count, -1)
    if strikes.ndim != 2 or strikes.shape[0] != path_count:
        raise ValueError("strikes must have shape (strike_count,) or (paths, strike_count)")
    tensors = (*state, jump_means, jump_stds, strikes)
    if any(value.device != log_spot.device or value.dtype != log_spot.dtype for value in tensors):
        raise ValueError("all valuation tensors must share a floating dtype and device")
    if not log_spot.dtype.is_floating_point:
        raise ValueError("valuation tensors must use a floating dtype")
    if expiry < 0.0 or kappa <= 0.0 or intensity < 0.0:
        raise ValueError("expiry and intensity must be non-negative and kappa positive")
    if expiry == 0.0:
        return torch.clamp(torch.exp(log_spot)[:, None] - strikes, min=0.0)

    real_nodes, real_weights = _quadrature_arrays(n_quad, u_max)
    u_real = torch.as_tensor(real_nodes, dtype=log_spot.dtype, device=log_spot.device)
    weights = torch.as_tensor(real_weights, dtype=log_spot.dtype, device=log_spot.device)
    complex_dtype = torch.complex128 if log_spot.dtype == torch.float64 else torch.complex64
    u = u_real.to(complex_dtype) - 1j * (_ALPHA + 1.0)

    theta_column = theta[:, None]
    xi_column = xi[:, None]
    rho_column = rho[:, None]
    b = kappa - 1j * rho_column * xi_column * u[None]
    d = torch.sqrt(b.square() + xi_column.square() * (u.square() + 1j * u)[None])
    d = torch.where(d.real < 0.0, -d, d)
    g = (b - d) / (b + d)
    exponential = torch.exp(-d * expiry)
    denominator_riccati = 1.0 - g * exponential
    coefficient_d = (
        (b - d)
        / xi_column.square()
        * (1.0 - exponential)
        / denominator_riccati
    )
    logarithm = torch.log(denominator_riccati / (1.0 - g))
    coefficient_a = (
        kappa
        * theta_column
        / xi_column.square()
        * ((b - d) * expiry - 2.0 * logarithm)
    )

    jump_exponent = torch.zeros(
        (path_count, n_quad), dtype=complex_dtype, device=log_spot.device
    )
    durations = _bucket_durations(boundaries, start=jump_start, expiry=expiry)
    for index, duration in enumerate(durations):
        if duration == 0.0:
            continue
        mean = jump_means[:, index, None]
        std = jump_stds[:, index, None]
        compensator = torch.exp(mean + 0.5 * std.square()) - 1.0
        exponent = torch.exp(
            1j * u[None] * mean - 0.5 * std.square() * u.square()[None]
        )
        jump_exponent += duration * intensity * (
            exponent - 1.0 - 1j * u[None] * compensator
        )

    log_transform = variance[:, None] * coefficient_d + coefficient_a + jump_exponent
    spot = torch.exp(log_spot)
    relative_strikes = strikes / spot[:, None]
    if torch.any(relative_strikes <= 0.0):
        raise ValueError("strikes and spot must be positive")
    log_strikes = torch.log(relative_strikes)
    carr_madan_denominator = (
        _ALPHA**2
        + _ALPHA
        - u_real.square()
        + 1j * (2.0 * _ALPHA + 1.0) * u_real
    ).to(complex_dtype)
    characteristic = torch.exp(log_transform) / carr_madan_denominator[None]
    integrand = torch.exp(
        -1j * u_real[None, :, None] * log_strikes[:, None, :]
    ) * characteristic[:, :, None]
    integral = torch.sum(weights[None, :, None] * integrand, dim=1).real
    prices = (
        spot[:, None]
        * torch.exp(-_ALPHA * log_strikes)
        * integral
        / math.pi
    )
    return prices
