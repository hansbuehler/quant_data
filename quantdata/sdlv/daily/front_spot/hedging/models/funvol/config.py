"""Geometry and short CPU training controls for the FuNVol adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)


@version("0.1", dependencies=[GenerativeSurfaceGrid])
@dataclass(frozen=True, kw_only=True)
class FunvolConfig:
    """Single-asset FuNVol geometry on the distributable SANOS IV grid."""

    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid.short_end_nms()
    functional_order: int = 4
    factor_count: int = 3
    lag_count: int = 10
    gru_layers: int = 3
    hidden_size: int = 16
    sessions_per_year: int = 252

    def __post_init__(self) -> None:
        verify_inp(self.functional_order >= 1, lambda: "functional_order must be positive")
        verify_inp(self.factor_count >= 1, lambda: "factor_count must be positive")
        verify_inp(self.lag_count >= 2, lambda: "lag_count must be at least two")
        verify_inp(self.gru_layers >= 1, lambda: "gru_layers must be positive")
        verify_inp(self.hidden_size >= 1, lambda: "hidden_size must be positive")
        verify_inp(self.sessions_per_year >= 1, lambda: "sessions_per_year must be positive")

    @property
    def state_dimension(self) -> int:
        return self.factor_count + 1

    @property
    def surface_size(self) -> int:
        return len(self.grid.expiries_y) * self.grid.strike_count


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class FunvolTrainConfig:
    """CPU training controls with optional dated epoch checkpoints."""

    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    validation_fraction: float = 0.2
    pit_weight: float = 1.0
    seed: int = 20261024

    def __post_init__(self) -> None:
        verify_inp(self.epochs >= 1, lambda: "epochs must be positive")
        verify_inp(self.batch_size >= 2, lambda: "batch_size must be at least two")
        verify_inp(self.learning_rate > 0.0, lambda: "learning_rate must be positive")
        verify_inp(
            0.0 <= self.validation_fraction < 0.5,
            lambda: "validation_fraction must lie in [0,0.5)",
        )
        verify_inp(self.pit_weight >= 0.0, lambda: "pit_weight must be nonnegative")
