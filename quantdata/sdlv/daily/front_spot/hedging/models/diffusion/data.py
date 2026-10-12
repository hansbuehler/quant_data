"""Sequential diffusion tensors from released SANOS histories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
    SanosFixedGridHistory,
    build_sanos_fixed_grid_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class DiffusionTrainingData:
    """Exact-session 22-day sequences with two public-data channels."""

    start_dates: tuple[date, ...]
    target_dates: tuple[date, ...]
    sequences: FloatArray

    def __post_init__(self) -> None:
        count = len(self.start_dates)
        if len(self.target_dates) != count or any(
            target <= start
            for start, target in zip(self.start_dates, self.target_dates, strict=True)
        ):
            raise ValueError("diffusion sequence dates are inconsistent")
        if self.sequences.ndim != 5 or self.sequences.shape[0] != count:
            raise ValueError("diffusion sequences must be sample-time-channel-expiry-strike")
        if self.sequences.shape[1] != 22 or self.sequences.shape[2] != 2:
            raise ValueError("diffusion sequences require 22 dates and two channels")
        if count < 2 or not np.all(np.isfinite(self.sequences)):
            raise ValueError("diffusion sequences must contain at least two finite samples")


@dataclass(frozen=True, kw_only=True)
class DiffusionLiveContext:
    """Twenty-one observed dates ending at one decision date."""

    as_of_date: date
    forecast_date: date
    sequence: FloatArray

    def __post_init__(self) -> None:
        if self.forecast_date <= self.as_of_date:
            raise ValueError("diffusion forecast date must follow its as-of date")
        if (
            self.sequence.ndim != 4
            or self.sequence.shape[0:2] != (21, 2)
            or not np.all(np.isfinite(self.sequence))
        ):
            raise ValueError("diffusion live context must be a finite 21-day sequence")


def build_diffusion_training_data(
    panel: PublicHedgingPanel,
    *,
    end_date: date,
    observation_count: int = 252,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> DiffusionTrainingData:
    """Build trailing exact-session sequences ending no later than ``end_date``."""
    require_public_hedging_panel(panel)
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=end_date,
        observation_count=observation_count,
        grid=grid,
    )
    return diffusion_sequences_from_history(history)


def build_diffusion_live_context(
    panel: PublicHedgingPanel,
    *,
    as_of_date: date,
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid(),
) -> DiffusionLiveContext:
    """Build a 21-day context ending exactly at ``as_of_date``."""
    require_public_hedging_panel(panel)
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=as_of_date,
        observation_count=22,
        grid=grid,
    )
    return diffusion_live_context_from_history(
        history,
        as_of_date=as_of_date,
        forecast_date=panel.next_one_session_date(as_of_date),
    )


def diffusion_live_context_from_history(
    history: SanosFixedGridHistory,
    *,
    as_of_date: date,
    forecast_date: date,
) -> DiffusionLiveContext:
    """Construct a live context while enforcing its declared decision date."""
    if len(history.dates) < 22:
        raise ValueError("diffusion live conditioning requires at least 22 dates")
    if history.dates[-1] != as_of_date:
        raise ValueError(
            f"diffusion history ends on {history.dates[-1]}, not as-of date {as_of_date}"
        )
    if not np.all(history.sessions_from_previous_observation[-21:] == 1):
        raise ValueError("diffusion live context is not a consecutive-session chain")
    surfaces = history.log_implied_volatilities[-21:]
    returns = history.forward_log_returns_into_date[-21:]
    sequence = np.stack(
        (
            surfaces,
            np.broadcast_to(returns[:, None, None], surfaces.shape),
        ),
        axis=1,
    )
    return DiffusionLiveContext(
        as_of_date=as_of_date,
        forecast_date=forecast_date,
        sequence=sequence,
    )


def diffusion_sequences_from_history(
    history: SanosFixedGridHistory,
) -> DiffusionTrainingData:
    """Convert a fixed-grid history into overlapping 21-to-1 examples."""
    sequences: list[FloatArray] = []
    start_dates: list[date] = []
    target_dates: list[date] = []
    for target_index in range(22, len(history.dates)):
        start_index = target_index - 21
        if not np.all(
            history.sessions_from_previous_observation[start_index : target_index + 1]
            == 1
        ):
            continue
        surfaces = history.log_implied_volatilities[start_index : target_index + 1]
        returns = history.forward_log_returns_into_date[start_index : target_index + 1]
        broadcast_returns = np.broadcast_to(
            returns[:, None, None],
            surfaces.shape,
        )
        sequences.append(np.stack((surfaces, broadcast_returns), axis=1))
        start_dates.append(history.dates[start_index])
        target_dates.append(history.dates[target_index])
    if len(sequences) < 2:
        raise ValueError("history contains fewer than two exact diffusion sequences")
    return DiffusionTrainingData(
        start_dates=tuple(start_dates),
        target_dates=tuple(target_dates),
        sequences=np.stack(sequences),
    )
