"""Generic Fourier pricing for finite-grid Tangent Lévy densities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp
from scipy.stats import poisson

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

    def price(
        self,
        characteristic_function: CharacteristicFunction,
        *,
        forward_moment: float = 1.0,
    ) -> np.ndarray:
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
        prices = forward_moment - np.exp(0.5 * self.log_strikes) * integral / np.pi
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
    forward_moment: float = 1.0,
) -> np.ndarray:
    """Price normalized calls with equation (2.36) of the paper.

    ``characteristic_function(z)`` must return ``E[exp(i z log(S_T/S_t))]``
    for complex ``z``. Spot, rates, and dividends are normalized to one, zero,
    and zero respectively.
    """

    original = np.asarray(log_strikes)
    prepared = PreparedFourierPricer.create(log_strikes, config=config)
    prices = prepared.price(
        characteristic_function, forward_moment=forward_moment
    )
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
    spacing = np.diff(np.asarray(jump_x, dtype=np.float64))
    if spacing.size and np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-12):
        return lattice_density_call_prices(
            tau=tau,
            log_strikes=log_strikes,
            jump_x=jump_x,
            average_density=average_density,
        )
    characteristic = density_characteristic_function(
        tau=tau,
        jump_x=jump_x,
        average_density=average_density,
    )
    jump_x = np.asarray(jump_x, dtype=np.float64)
    eta = np.asarray(average_density, dtype=np.float64)
    weighted = trapezoid_weights(jump_x) * eta
    intensity = float(np.sum(weighted))
    compensator = float(weighted @ np.expm1(jump_x))
    atom_probability = math.exp(-tau * intensity)
    atom_gross_return = math.exp(-tau * compensator)
    atom_log_return = math.log(atom_gross_return)
    strikes = np.exp(np.asarray(log_strikes, dtype=np.float64))
    prepared = PreparedFourierPricer.create(log_strikes, config=config)
    continuous = prepared.price(
        lambda z: characteristic(z)
        - atom_probability * np.exp(1j * z * atom_log_return),
        forward_moment=1.0 - atom_probability * atom_gross_return,
    )
    atom = atom_probability * np.maximum(atom_gross_return - strikes, 0.0)
    result = continuous + atom
    return result.item() if np.asarray(log_strikes).ndim == 0 else result


def lattice_density_call_prices(
    *,
    tau: float,
    log_strikes: np.ndarray | float,
    jump_x: np.ndarray,
    average_density: np.ndarray,
    fft_size: int = 16_384,
) -> np.ndarray:
    """Price the uniform-grid compound-Poisson law by cyclic convolution.

    The 301-node v2 jump support lies on an exact lattice.  Embedding its jump
    probabilities in a much wider FFT lattice evaluates the compound-Poisson
    distribution, including the no-jump atom, without Fourier payoff-truncation
    error.  The default support spans more than 80 log-return units.
    """

    jump = np.asarray(jump_x, dtype=np.float64)
    eta = np.asarray(average_density, dtype=np.float64)
    verify_inp(
        fft_size >= 1024 and fft_size % 2 == 0,
        lambda: "FFT lattice size must be an even integer of at least 1024",
    )
    spacing = float(jump[1] - jump[0])
    offsets = np.rint(jump / spacing).astype(np.int64)
    verify_inp(
        np.allclose(jump, offsets * spacing, rtol=0.0, atol=1.0e-10)
        and np.max(np.abs(offsets)) < fft_size // 2,
        lambda: "jump grid is not representable on the requested FFT lattice",
    )
    weighted = trapezoid_weights(jump) * eta
    intensity = float(np.sum(weighted))
    verify_inp(intensity > 0.0, lambda: "Levy density must have positive mass")
    jump_probability = np.zeros(fft_size, dtype=np.float64)
    np.add.at(jump_probability, offsets % fft_size, weighted / intensity)
    transform = np.fft.fft(jump_probability)
    probabilities = np.fft.ifft(
        np.exp(tau * intensity * (transform - 1.0))
    ).real
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= np.sum(probabilities)
    indices = np.arange(fft_size, dtype=np.int64)
    signed = np.where(indices <= fft_size // 2, indices, indices - fft_size)
    share_intensity = float(weighted @ np.exp(jump))
    maximum_jumps = int(
        max(
            poisson.ppf(1.0 - 1.0e-14, tau * intensity),
            poisson.ppf(1.0 - 1.0e-14, tau * share_intensity),
        )
    )
    maximum_offset = maximum_jumps * int(np.max(np.abs(offsets)))
    probabilities[np.abs(signed) > maximum_offset] = 0.0
    # FFT roundoff leaves O(1e-18) positive dust across remote lattice cells.
    # Multiplication by exp(x) can turn that dust into a material fake share-
    # measure tail.  Values below this level are numerical noise in float64.
    probabilities[probabilities < 1.0e-17] = 0.0
    probabilities /= np.sum(probabilities)
    compensator = float(weighted @ np.expm1(jump))
    gross = np.exp(signed * spacing - tau * compensator)
    log_strike_values = np.asarray(log_strikes, dtype=np.float64)
    strike_shape = log_strike_values.shape
    strikes = np.exp(np.atleast_1d(log_strike_values).reshape(-1))
    order = np.argsort(gross)
    sorted_gross = gross[order]
    sorted_probability = probabilities[order]
    tail_probability = np.cumsum(sorted_probability[::-1])[::-1]
    tail_gross = np.cumsum((sorted_probability * sorted_gross)[::-1])[::-1]
    first_in_money = np.searchsorted(sorted_gross, strikes, side="right")
    calls = np.zeros_like(strikes)
    active = first_in_money < len(sorted_gross)
    calls[active] = (
        tail_gross[first_in_money[active]]
        - strikes[active] * tail_probability[first_in_money[active]]
    )
    if log_strike_values.ndim == 0:
        return calls.item()
    return calls.reshape(strike_shape)


def lattice_density_call_prices_batched(
    *,
    tau: float,
    log_strikes: np.ndarray,
    jump_x: np.ndarray,
    average_density: np.ndarray,
    fft_size: int = 8_192,
) -> np.ndarray:
    """Price pathwise strikes and densities with batched lattice FFTs.

    The leading axis is the path axis.  Each path has its own Lévy density and
    strike vector, while all paths share the same uniform jump lattice.
    """

    jump = np.asarray(jump_x, dtype=np.float64)
    eta = np.asarray(average_density, dtype=np.float64)
    strike_logs = np.asarray(log_strikes, dtype=np.float64)
    verify_inp(
        eta.ndim == 2
        and eta.shape[1] == jump.size
        and strike_logs.ndim == 2
        and strike_logs.shape[0] == eta.shape[0]
        and np.all(np.isfinite(eta))
        and np.all(eta >= 0.0)
        and np.all(np.isfinite(strike_logs)),
        lambda: (
            "batched densities and log strikes must be finite with shapes "
            f"(P,{jump.size}) and (P,K); got {eta.shape} and {strike_logs.shape}"
        ),
    )
    verify_inp(
        fft_size >= 1024 and fft_size % 2 == 0,
        lambda: "FFT lattice size must be an even integer of at least 1024",
    )
    spacing = float(jump[1] - jump[0])
    offsets = np.rint(jump / spacing).astype(np.int64)
    verify_inp(
        np.allclose(jump, offsets * spacing, rtol=0.0, atol=1.0e-10)
        and np.max(np.abs(offsets)) < fft_size // 2,
        lambda: "jump grid is not representable on the requested FFT lattice",
    )
    weighted = eta * trapezoid_weights(jump)[None]
    intensity = np.sum(weighted, axis=1)
    verify_inp(
        np.all(intensity > 0.0),
        lambda: "every batched Lévy density must have positive mass",
    )
    jump_probability = np.zeros((eta.shape[0], fft_size), dtype=np.float64)
    jump_probability[:, offsets % fft_size] = weighted / intensity[:, None]
    transform = np.fft.fft(jump_probability, axis=1)
    probabilities = np.fft.ifft(
        np.exp(tau * intensity[:, None] * (transform - 1.0)),
        axis=1,
    ).real
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    indices = np.arange(fft_size, dtype=np.int64)
    signed = np.where(indices <= fft_size // 2, indices, indices - fft_size)
    share_intensity = weighted @ np.exp(jump)
    maximum_jumps = np.maximum(
        poisson.ppf(1.0 - 1.0e-14, tau * intensity),
        poisson.ppf(1.0 - 1.0e-14, tau * share_intensity),
    ).astype(np.int64)
    maximum_offset = maximum_jumps * int(np.max(np.abs(offsets)))
    verify_inp(
        np.all(maximum_offset < fft_size // 2),
        lambda: (
            "batched FFT support is too short for the compound-Poisson tail: "
            f"maximum offset {int(np.max(maximum_offset))}, half-size "
            f"{fft_size // 2}"
        ),
    )
    probabilities[
        np.abs(signed)[None] > maximum_offset[:, None]
    ] = 0.0
    probabilities[probabilities < 1.0e-17] = 0.0
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    order = np.argsort(signed)
    base_gross = np.exp(signed[order] * spacing)
    sorted_probability = probabilities[:, order]
    tail_probability = np.cumsum(sorted_probability[:, ::-1], axis=1)[:, ::-1]
    compensator = weighted @ np.expm1(jump)
    gross_scale = np.exp(-tau * compensator)
    tail_gross = np.cumsum(
        (sorted_probability * base_gross[None])[:, ::-1], axis=1
    )[:, ::-1]
    strikes = np.exp(strike_logs)
    first_in_money = np.searchsorted(
        base_gross,
        (strikes / gross_scale[:, None]).reshape(-1),
        side="right",
    ).reshape(strikes.shape)
    active = first_in_money < fft_size
    gather = np.minimum(first_in_money, fft_size - 1)
    calls = gross_scale[:, None] * np.take_along_axis(
        tail_gross, gather, axis=1
    ) - strikes * np.take_along_axis(tail_probability, gather, axis=1)
    calls[~active] = 0.0
    return calls


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
