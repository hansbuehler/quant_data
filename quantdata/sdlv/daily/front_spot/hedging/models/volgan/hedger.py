"""Scenario sampling from a fitted public-data VolGAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.volgan.training import (
    VolganTrainingResult,
)
from quantdata.torch import torch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class VolganGeneratedTransitions:
    """One-day normalized return and next-surface scenarios."""

    forward_log_returns: FloatArray
    next_log_implied_volatilities: FloatArray

    def __post_init__(self) -> None:
        if self.forward_log_returns.ndim != 1:
            raise ValueError("VolGAN generated returns must be a vector")
        if self.next_log_implied_volatilities.ndim != 3:
            raise ValueError("VolGAN generated surfaces must be path-expiry-strike")
        if len(self.forward_log_returns) != len(self.next_log_implied_volatilities):
            raise ValueError("VolGAN generated paths do not align")
        if not np.all(np.isfinite(self.forward_log_returns)) or not np.all(
            np.isfinite(self.next_log_implied_volatilities)
        ):
            raise ValueError("VolGAN generated transitions must be finite")


def sample_volgan_transitions(
    fit: VolganTrainingResult,
    *,
    condition: NDArray[np.floating],
    path_count: int,
    seed: int,
) -> VolganGeneratedTransitions:
    """Sample one-step transitions conditional on one released market state."""
    generator = fit.generator
    config = generator.config
    condition_array = np.asarray(condition, dtype=np.float64)
    if condition_array.shape != (config.condition_dimension,):
        raise ValueError("VolGAN sampling condition has the wrong shape")
    if path_count < 1:
        raise ValueError("VolGAN path_count must be positive")
    with torch.no_grad():
        noise_generator = torch.Generator(device=generator.device)
        noise_generator.manual_seed(seed)
        noise = torch.randn(
            (path_count, config.noise_dimension),
            generator=noise_generator,
            dtype=generator.fdtype,
            device=generator.device,
        )
        conditions = torch.as_tensor(
            np.broadcast_to(condition_array, (path_count, len(condition_array))).copy(),
            dtype=generator.fdtype,
            device=generator.device,
        )
        decoded = generator.decode(generator(noise, conditions)).cpu().numpy()
    current = condition_array[3:].reshape(
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    )
    next_surfaces = current[None] + decoded[:, 1:].reshape(
        path_count,
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    )
    return VolganGeneratedTransitions(
        forward_log_returns=decoded[:, 0] / np.sqrt(252.0),
        next_log_implied_volatilities=next_surfaces,
    )
