"""Configuration for the public-data VolGAN baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)


@version("1.1", dependencies=[GenerativeSurfaceGrid])
@dataclass(frozen=True, kw_only=True)
class VolganModelConfig:
    """Published VolGAN geometry adapted to the released SANOS grid."""

    label: str
    group: str = "hedging/volgan"
    user_version: str = "0.1"
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid()
    noise_dimension: int = 32
    hidden_dimension: int = 16

    def __post_init__(self) -> None:
        group_parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid VolGAN label {self.label!r}",
        )
        verify_inp(
            bool(group_parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
            lambda: f"invalid VolGAN group {self.group!r}",
        )
        verify_inp(self.noise_dimension > 0, lambda: "noise_dimension must be positive")
        verify_inp(self.hidden_dimension > 0, lambda: "hidden_dimension must be positive")

    @property
    def surface_size(self) -> int:
        return len(self.grid.expiries_y) * len(self.grid.strikes_over_forward)

    @property
    def condition_dimension(self) -> int:
        return 3 + self.surface_size

    @property
    def output_dimension(self) -> int:
        return 1 + self.surface_size

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


@version("1.2")
@dataclass(frozen=True, kw_only=True)
class VolganTrainConfig:
    """Operational VolGAN training controls."""

    epochs: int = 10_000
    gradient_matching_epochs: int = 25
    batch_size: int = 100
    generator_learning_rate: float = 1.0e-4
    discriminator_learning_rate: float = 1.0e-4
    optimizer: str = "rmsprop"
    validation_fraction: float = 0.2
    seed: int = 20_261_012
    progress_seconds: float = 240.0
    checkpoint_epochs: int = 100
    compile_models: bool = False
    allow_changed_config_warm_start: bool = False
    just_load: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.epochs >= 1, lambda: "epochs must be positive")
        verify_inp(
            0 <= self.gradient_matching_epochs <= self.epochs,
            lambda: "gradient_matching_epochs must lie in [0,epochs]",
        )
        verify_inp(self.batch_size >= 2, lambda: "batch_size must be at least two")
        verify_inp(self.optimizer in {"rmsprop", "adam"}, lambda: "invalid optimizer")
        for name, value in (
            ("generator_learning_rate", self.generator_learning_rate),
            ("discriminator_learning_rate", self.discriminator_learning_rate),
        ):
            verify_inp(
                math.isfinite(value) and value > 0.0,
                lambda name=name: f"{name} must be finite-positive",
            )
        verify_inp(
            math.isfinite(self.validation_fraction)
            and 0.1 <= self.validation_fraction <= 0.5,
            lambda: "validation_fraction must lie in [0.1,0.5]",
        )
        verify_inp(
            math.isfinite(self.progress_seconds)
            and 1.0 <= self.progress_seconds <= 300.0,
            lambda: "progress_seconds must lie in [1,300]",
        )
        verify_inp(self.checkpoint_epochs >= 1, lambda: "checkpoint_epochs must be positive")
