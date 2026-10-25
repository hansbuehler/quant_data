"""Recursive remaining-horizon paths from one fitted VolGAN."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly import (
    GeneratedWeeklyPaths,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    VolganLiveCondition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.training import (
    VolganTrainingResult,
)
from quantdata.torch import torch


def sample_volgan_weekly_paths(
    fit: VolganTrainingResult,
    *,
    live: VolganLiveCondition,
    path_count: int,
    step_count: int,
    seed: int,
) -> GeneratedWeeklyPaths:
    """Sample a VolGAN recursively while updating every path's live condition."""
    generator = fit.generator
    config = generator.config
    if live.values.shape != (config.full_condition_dimension,):
        raise ValueError("VolGAN weekly live condition has the wrong shape")
    if path_count < 2:
        raise ValueError("VolGAN weekly sampling requires at least two paths")
    if not 1 <= step_count <= 5:
        raise ValueError("VolGAN weekly step count must lie in [1,5]")

    expiry_count = len(config.grid.expiries_y)
    strike_count = config.grid.strike_count
    trailing_returns = np.broadcast_to(
        live.trailing_forward_log_returns,
        (path_count, 21),
    ).copy()
    current_surfaces = np.broadcast_to(
        live.values[3:].reshape(expiry_count, strike_count),
        (path_count, expiry_count, strike_count),
    ).copy()
    generated_returns = np.empty((path_count, step_count), dtype=np.float64)
    generated_surfaces = np.empty(
        (path_count, step_count, expiry_count, strike_count),
        dtype=np.float64,
    )
    noise_generator = torch.Generator(device=generator.device)
    noise_generator.manual_seed(seed)
    annualization = np.sqrt(252.0)
    with torch.no_grad():
        for step in range(step_count):
            conditions = np.concatenate(
                (
                    annualization * trailing_returns[:, [-1, -2]],
                    np.std(trailing_returns, axis=1, ddof=0)[:, None]
                    * annualization,
                    current_surfaces.reshape(path_count, -1),
                ),
                axis=1,
            )
            noise = torch.randn(
                (path_count, config.noise_dimension),
                generator=noise_generator,
                dtype=generator.fdtype,
                device=generator.device,
            )
            condition_tensor = torch.as_tensor(
                conditions,
                dtype=generator.fdtype,
                device=generator.device,
            )
            decoded = generator.decode_full_targets(
                generator(noise, generator.encode_full_conditions(condition_tensor))
            ).cpu().numpy()
            next_returns = decoded[:, 0] / annualization
            next_surfaces = current_surfaces + decoded[:, 1:].reshape(
                path_count,
                expiry_count,
                strike_count,
            )
            if not np.all(np.isfinite(next_returns)) or not np.all(
                np.isfinite(next_surfaces)
            ):
                raise FloatingPointError("VolGAN weekly recursion became non-finite")
            generated_returns[:, step] = next_returns
            generated_surfaces[:, step] = next_surfaces
            trailing_returns = np.concatenate(
                (trailing_returns[:, 1:], next_returns[:, None]),
                axis=1,
            )
            current_surfaces = next_surfaces
    return GeneratedWeeklyPaths(
        forward_log_returns=generated_returns,
        next_log_implied_volatilities=generated_surfaces,
        grid=config.grid,
        seed=seed,
    )
