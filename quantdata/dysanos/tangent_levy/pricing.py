"""Generic Fourier pricing for finite-grid Tangent Lévy densities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp

from .config import PricingConfig
from .coordinates import normalized_call_intrinsic, trapezoid_weights

CharacteristicFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PreparedFourierPricer:
    """Precomputed equation (2.36) quadrature for a fixed strike vector."""

    log_strikes: np.ndarray
    u: np.ndarray
    kernel: np.ndarray
    simpson_weights: np.ndarray

    @classmethod
    def create(
        cls,
        log_strikes: np.ndarray | float,
        *,
        config: PricingConfig = PricingConfig(),
    ) -> PreparedFourierPricer:
        """Build the fixed phase and Simpson weights once for calibration."""

        x = np.atleast_1d(np.asarray(log_strikes, dtype=np.float64))
        verify_inp(
            x.ndim == 1 and np.all(np.isfinite(x)),
            lambda: f"'log_strikes' must be a finite scalar or vector, got {x.shape}",
        )
        u = np.linspace(0.0, config.integration_upper, config.integration_points)
        denominator = np.square(u) + 0.25
        kernel = np.exp(-1j * x[:, None] * u[None, :]) / denominator[None, :]
        step = u[1] - u[0]
        weights = np.ones_like(u)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        weights *= step / 3.0
        return cls(
            log_strikes=x,
            u=u,
            kernel=kernel,
            simpson_weights=weights,
        )

    def price(self, characteristic_function: CharacteristicFunction) -> np.ndarray:
        """Evaluate calls using a new characteristic function."""

        shifted = self.u - 0.5j
        phi = np.asarray(characteristic_function(shifted), dtype=np.complex128)
        verify_inp(
            phi.shape == self.u.shape and np.all(np.isfinite(phi)),
            lambda: (
                "the characteristic function must return one finite value per "
                f"integration point, got {phi.shape}, expected {self.u.shape}"
            ),
        )
        integral = np.real(self.kernel * phi[None, :]) @ self.simpson_weights
        prices = 1.0 - np.exp(0.5 * self.log_strikes) * integral / np.pi
        verify_inp(
            np.all(np.isfinite(prices)),
            lambda: "Fourier integration produced non-finite call prices",
        )
        return prices


def fourier_call_prices(
    log_strikes: np.ndarray | float,
    characteristic_function: CharacteristicFunction,
    *,
    config: PricingConfig = PricingConfig(),
) -> np.ndarray:
    """Price normalized calls with equation (2.36) of the paper.

    ``characteristic_function(z)`` must return ``E[exp(i z log(S_T/S_t))]``
    for complex ``z``. Spot, rates, and dividends are normalized to one, zero,
    and zero respectively.
    """

    original = np.asarray(log_strikes)
    prepared = PreparedFourierPricer.create(log_strikes, config=config)
    prices = prepared.price(characteristic_function)
    return prices[0] if original.ndim == 0 else prices


def density_characteristic_function(
    *,
    tau: float,
    jump_x: np.ndarray,
    average_density: np.ndarray,
) -> CharacteristicFunction:
    """Return the characteristic function induced by a finite-grid density.

    The density is the maturity-average density ``eta`` from equation (2.15),
    not the forward density ``kappa``. Integrals are deterministic trapezoidal
    quadrature over ``jump_x``; mass outside the grid is intentionally zero.
    """

    jump_x = np.asarray(jump_x, dtype=np.float64)
    eta = np.asarray(average_density, dtype=np.float64)
    verify_inp(
        np.isfinite(tau) and tau >= 0.0,
        lambda: f"'tau' must be finite and nonnegative, got {tau!r}",
    )
    verify_inp(
        eta.shape == jump_x.shape
        and np.all(np.isfinite(eta))
        and np.all(eta >= 0.0),
        lambda: (
            "'average_density' must be finite, nonnegative, and match jump_x; "
            f"got {eta.shape} and {jump_x.shape}"
        ),
    )
    weights = trapezoid_weights(jump_x)
    weighted_eta = weights * eta
    compensator = float(weighted_eta @ np.expm1(jump_x))

    def characteristic(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.complex128)
        jump_transform = np.exp(1j * z[..., None] * jump_x) - 1.0
        exponent = tau * (
            -1j * z * compensator + jump_transform @ weighted_eta
        )
        return np.exp(exponent)

    return characteristic


def density_call_prices(
    *,
    tau: float,
    log_strikes: np.ndarray | float,
    jump_x: np.ndarray,
    average_density: np.ndarray,
    config: PricingConfig = PricingConfig(),
) -> np.ndarray:
    """Price normalized calls for an arbitrary finite-grid average density."""

    if tau == 0.0:
        strikes = np.exp(np.asarray(log_strikes, dtype=np.float64))
        return normalized_call_intrinsic(strikes)
    characteristic = density_characteristic_function(
        tau=tau,
        jump_x=jump_x,
        average_density=average_density,
    )
    return fourier_call_prices(
        log_strikes,
        characteristic,
        config=config,
    )


def average_density_from_forward(
    *,
    tau: np.ndarray,
    forward_density: np.ndarray,
) -> np.ndarray:
    """Integrate forward ``kappa`` into maturity-average ``eta``.

    Parameters
    ----------
    tau:
        Increasing maturity grid starting at zero, shape ``(T,)``.
    forward_density:
        Array with trailing shape ``(T, X)``. Leading axes, such as paths, are
        retained.
    """

    tau = np.asarray(tau, dtype=np.float64)
    kappa = np.asarray(forward_density, dtype=np.float64)
    verify_inp(
        tau.ndim == 1
        and tau.size >= 2
        and tau[0] == 0.0
        and np.all(np.diff(tau) > 0.0),
        lambda: "'tau' must start at zero and be strictly increasing",
    )
    verify_inp(
        kappa.ndim >= 2
        and kappa.shape[-2] == tau.size
        and np.all(np.isfinite(kappa)),
        lambda: (
            "'forward_density' must be finite with trailing maturity size "
            f"{tau.size}, got {kappa.shape}"
        ),
    )
    increments = 0.5 * (kappa[..., 1:, :] + kappa[..., :-1, :])
    increments = increments * np.diff(tau).reshape(
        *((1,) * (kappa.ndim - 2)), -1, 1
    )
    cumulative = np.concatenate(
        [np.zeros_like(kappa[..., :1, :]), np.cumsum(increments, axis=-2)],
        axis=-2,
    )
    eta = np.empty_like(kappa)
    eta[..., 0, :] = kappa[..., 0, :]
    eta[..., 1:, :] = cumulative[..., 1:, :] / tau[1:].reshape(
        *((1,) * (kappa.ndim - 2)), -1, 1
    )
    return eta
