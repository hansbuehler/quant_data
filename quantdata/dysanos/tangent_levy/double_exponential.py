"""Double-exponential Tangent Lévy density and pricing formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cdxcore.err import verify_inp

from .config import PricingConfig
from .coordinates import normalized_call_intrinsic
from .pricing import PreparedFourierPricer


@dataclass(frozen=True, kw_only=True)
class DoubleExponentialParameters:
    """Kou pure-jump parameters from equations (2.10)--(2.11)."""

    intensity: float
    positive_rate: float
    negative_rate: float
    positive_probability: float

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.intensity) and self.intensity > 0.0,
            lambda: f"'intensity' must be finite and positive, got {self.intensity!r}",
        )
        verify_inp(
            math.isfinite(self.positive_rate) and self.positive_rate > 1.0,
            lambda: (
                "'positive_rate' must be finite and greater than one, got "
                f"{self.positive_rate!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.negative_rate) and self.negative_rate > 0.0,
            lambda: (
                "'negative_rate' must be finite and positive, got "
                f"{self.negative_rate!r}"
            ),
        )
        verify_inp(
            math.isfinite(self.positive_probability)
            and 0.0 < self.positive_probability < 1.0,
            lambda: (
                "'positive_probability' must lie in (0, 1), got "
                f"{self.positive_probability!r}"
            ),
        )

    @property
    def negative_probability(self) -> float:
        """Probability of a negative jump."""

        return 1.0 - self.positive_probability

    @property
    def exponential_jump_mean(self) -> float:
        """Return ``E[exp(Y)-1]`` for one jump."""

        p = self.positive_probability
        return (
            p * self.positive_rate / (self.positive_rate - 1.0)
            + (1.0 - p) * self.negative_rate / (self.negative_rate + 1.0)
            - 1.0
        )


def density(
    jump_x: np.ndarray | float,
    parameters: DoubleExponentialParameters,
) -> np.ndarray:
    """Evaluate the Lévy density ``eta(x)`` in equation (2.11)."""

    x = np.asarray(jump_x, dtype=np.float64)
    verify_inp(
        np.all(np.isfinite(x)),
        lambda: "jump sizes must contain only finite values",
    )
    p = parameters.positive_probability
    result = np.where(
        x >= 0.0,
        p * parameters.positive_rate * np.exp(-parameters.positive_rate * x),
        (1.0 - p)
        * parameters.negative_rate
        * np.exp(parameters.negative_rate * x),
    )
    return parameters.intensity * result


def symmetry_index(parameters: DoubleExponentialParameters) -> float:
    """Return equation (2.17), ``integral (exp(x)-1) eta(x) dx``."""

    return parameters.intensity * parameters.exponential_jump_mean


def continuous_negative_rate(*, positive_rate: float, probability: float) -> float:
    """Derive the negative-tail rate from density continuity at zero."""

    verify_inp(
        math.isfinite(positive_rate) and positive_rate > 1.0,
        lambda: f"'positive_rate' must exceed one, got {positive_rate!r}",
    )
    verify_inp(
        math.isfinite(probability) and 0.0 < probability < 1.0,
        lambda: f"'probability' must lie in (0, 1), got {probability!r}",
    )
    return probability * positive_rate / (1.0 - probability)


def constrained_parameters(
    *,
    intensity: float,
    positive_rate: float,
    target_symmetry_index: float,
) -> DoubleExponentialParameters:
    """Apply equations (2.16)--(2.18) to obtain the two derived parameters."""

    verify_inp(
        math.isfinite(intensity) and intensity > 0.0,
        lambda: f"'intensity' must be finite and positive, got {intensity!r}",
    )
    verify_inp(
        math.isfinite(positive_rate) and positive_rate > 1.0,
        lambda: f"'positive_rate' must exceed one, got {positive_rate!r}",
    )
    verify_inp(
        math.isfinite(target_symmetry_index),
        lambda: f"'target_symmetry_index' must be finite, got {target_symmetry_index!r}",
    )
    tail_excess = positive_rate - 1.0
    ratio = target_symmetry_index / intensity
    denominator = 1.0 + 2.0 * tail_excess - ratio * tail_excess**2
    probability = tail_excess * (1.0 + ratio) / denominator
    verify_inp(
        math.isfinite(probability) and 0.0 < probability < 1.0,
        lambda: (
            "equations (2.16)--(2.18) imply an invalid probability "
            f"{probability!r} for intensity={intensity}, "
            f"positive_rate={positive_rate}, Xi={target_symmetry_index}"
        ),
    )
    negative_rate = continuous_negative_rate(
        positive_rate=positive_rate,
        probability=probability,
    )
    return DoubleExponentialParameters(
        intensity=intensity,
        positive_rate=positive_rate,
        negative_rate=negative_rate,
        positive_probability=probability,
    )


def characteristic_function(
    z: np.ndarray,
    *,
    tau: float,
    parameters: DoubleExponentialParameters,
) -> np.ndarray:
    """Return the martingale characteristic function of ``log(S_T/S_t)``."""

    verify_inp(
        math.isfinite(tau) and tau >= 0.0,
        lambda: f"'tau' must be finite and nonnegative, got {tau!r}",
    )
    z = np.asarray(z, dtype=np.complex128)
    p = parameters.positive_probability
    jump_transform = (
        p * parameters.positive_rate / (parameters.positive_rate - 1j * z)
        + (1.0 - p)
        * parameters.negative_rate
        / (parameters.negative_rate + 1j * z)
    )
    exponent = tau * parameters.intensity * (
        jump_transform - 1.0 - 1j * z * parameters.exponential_jump_mean
    )
    return np.exp(exponent)


def call_prices(
    *,
    tau: float,
    log_strikes: np.ndarray | float,
    parameters: DoubleExponentialParameters,
    config: PricingConfig = PricingConfig(),
) -> np.ndarray:
    """Price normalized calls under the pure-jump double-exponential model."""

    if tau == 0.0:
        return normalized_call_intrinsic(np.exp(np.asarray(log_strikes)))
    atom_probability = math.exp(-tau * parameters.intensity)
    atom_gross_return = math.exp(
        -tau * parameters.intensity * parameters.exponential_jump_mean
    )
    atom_log_return = math.log(atom_gross_return)
    strikes = np.exp(np.asarray(log_strikes, dtype=np.float64))
    prepared = PreparedFourierPricer.create(log_strikes, config=config)
    continuous = prepared.price(
        lambda z: characteristic_function(z, tau=tau, parameters=parameters)
        - atom_probability * np.exp(1j * z * atom_log_return),
        forward_moment=1.0 - atom_probability * atom_gross_return,
    )
    result = continuous + atom_probability * np.maximum(
        atom_gross_return - strikes, 0.0
    )
    return result.item() if np.asarray(log_strikes).ndim == 0 else result
