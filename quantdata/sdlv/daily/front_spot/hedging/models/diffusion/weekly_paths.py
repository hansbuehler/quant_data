"""Recursive remaining-horizon paths from one fitted sequential diffusion."""

from __future__ import annotations

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    DiffusionLiveContext,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    _sample_diffusion_next_states,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.training import (
    DiffusionTrainingResult,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_weekly import (
    GeneratedWeeklyPaths,
)
from quantdata.torch import torch


def sample_diffusion_weekly_paths(
    fit: DiffusionTrainingResult,
    *,
    live: DiffusionLiveContext,
    path_count: int,
    step_count: int,
    seed: int,
    return_dispersion_tolerance: float | None = None,
) -> GeneratedWeeklyPaths:
    """Sample recursively by appending each generated state to its own context."""
    model = fit.ema_model
    config = model.config
    expected = (
        config.sequence_length - 1,
        config.channel_count,
        len(config.grid.expiries_y),
        len(config.grid.strikes_over_forward),
    )
    if live.sequence.shape != expected:
        raise ValueError("diffusion weekly live context has the wrong shape")
    if path_count < 2:
        raise ValueError("diffusion weekly sampling requires at least two paths")
    if not 1 <= step_count <= 5:
        raise ValueError("diffusion weekly step count must lie in [1,5]")
    if (
        return_dispersion_tolerance is not None
        and return_dispersion_tolerance < 0.0
    ):
        raise ValueError("diffusion return dispersion tolerance must be non-negative")

    contexts = np.broadcast_to(
        live.sequence,
        (path_count, *live.sequence.shape),
    ).copy()
    generated_returns = np.empty((path_count, step_count), dtype=np.float64)
    generated_surfaces = np.empty(
        (
            path_count,
            step_count,
            len(config.grid.expiries_y),
            len(config.grid.strikes_over_forward),
        ),
        dtype=np.float64,
    )
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    for step in range(step_count):
        next_states = _sample_diffusion_next_states(
            fit,
            observed_sequences=contexts,
            generator=generator,
        )
        return_channel = next_states[:, 1]
        dispersion = float(np.max(np.ptp(return_channel, axis=(-2, -1))))
        if (
            return_dispersion_tolerance is not None
            and dispersion > return_dispersion_tolerance
        ):
            raise FloatingPointError(
                "diffusion return channel is not spatially broadcast: "
                f"dispersion={dispersion:.6g}"
            )
        next_returns = np.mean(return_channel, axis=(-2, -1))
        next_surfaces = next_states[:, 0]
        generated_returns[:, step] = next_returns
        generated_surfaces[:, step] = next_surfaces
        broadcast_returns = np.broadcast_to(
            next_returns[:, None, None],
            next_surfaces.shape,
        )
        next_context_state = np.stack(
            (next_surfaces, broadcast_returns),
            axis=1,
        )
        contexts = np.concatenate(
            (contexts[:, 1:], next_context_state[:, None]),
            axis=1,
        )
    return GeneratedWeeklyPaths(
        forward_log_returns=generated_returns,
        next_log_implied_volatilities=generated_surfaces,
        grid=config.grid,
        seed=seed,
    )
