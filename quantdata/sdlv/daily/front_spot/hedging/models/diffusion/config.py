"""Configuration for sequential volatility diffusion baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from cdxcore.err import verify_inp
from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
from cdxcore.util import is_filename
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)


class DiffusionVariant(StrEnum):
    """Published baseline variants."""

    AD_SEQ_VOL = "ad-seq-vol"
    AD_SEQ_VOL_FT = "ad-seq-vol-ft"


@version("1.3", dependencies=[GenerativeSurfaceGrid])
@dataclass(frozen=True, kw_only=True)
class DiffusionModelConfig:
    """Causal transformer and diffusion geometry."""

    label: str
    group: str = "hedging/diffusion"
    user_version: str = "0.1"
    grid: GenerativeSurfaceGrid = GenerativeSurfaceGrid()
    variant: DiffusionVariant = DiffusionVariant.AD_SEQ_VOL
    sequence_length: int = 22
    channel_count: int = 2
    model_dimension: int = 256
    transformer_layers: int = 8
    attention_heads: int = 4
    feedforward_multiplier: int = 2
    dropout: float = 0.1
    diffusion_steps: int = 200
    lora_rank: int = 4

    def __post_init__(self) -> None:
        group_parts = self.group.split("/")
        verify_inp(
            bool(self.label) and (self.label.startswith("*") or is_filename(self.label)),
            lambda: f"invalid diffusion label {self.label!r}",
        )
        verify_inp(
            bool(group_parts)
            and all(part not in {"", ".", ".."} and is_filename(part) for part in group_parts),
            lambda: f"invalid diffusion group {self.group!r}",
        )
        verify_inp(self.sequence_length == 22, lambda: "published sequence length is 22")
        verify_inp(self.channel_count == 2, lambda: "diffusion requires two channels")
        verify_inp(self.model_dimension > 0, lambda: "model_dimension must be positive")
        verify_inp(
            self.model_dimension % self.attention_heads == 0,
            lambda: "model_dimension must be divisible by attention_heads",
        )
        verify_inp(self.transformer_layers > 0, lambda: "transformer_layers must be positive")
        verify_inp(
            self.feedforward_multiplier > 0,
            lambda: "feedforward_multiplier must be positive",
        )
        verify_inp(
            math.isfinite(self.dropout) and 0.0 <= self.dropout < 1.0,
            lambda: "dropout must lie in [0,1)",
        )
        verify_inp(self.diffusion_steps >= 2, lambda: "diffusion_steps must be at least two")
        verify_inp(
            self.variant is DiffusionVariant.AD_SEQ_VOL or self.lora_rank > 0,
            lambda: "fine-tuned diffusion requires positive lora_rank",
        )

    @property
    def surface_size(self) -> int:
        return len(self.grid.expiries_y) * len(self.grid.strikes_over_forward)

    @property
    def day_dimension(self) -> int:
        return self.channel_count * self.surface_size

    @property
    def filename(self) -> str:
        if self.label.startswith("*"):
            return label_unique_filename48_8(self.label[1:])
        return named_unique_filename48_8(self.label, self)


@version("1.1")
@dataclass(frozen=True, kw_only=True)
class DiffusionTrainConfig:
    """Operational controls for published and rolling diffusion fits."""

    epochs: int = 3_000
    batch_size: int = 256
    learning_rate: float = 7.0e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1_600
    ema_decay: float = 0.995
    validation_fraction: float = 0.2
    arbitrage_penalty: float = 1.0
    return_broadcast_penalty: float = 1.0
    seed: int = 20_261_012
    progress_seconds: float = 240.0
    checkpoint_epochs: int = 10
    compile_model: bool = False
    mixed_precision: bool = True
    allow_changed_config_warm_start: bool = False
    just_load: bool = False

    def __post_init__(self) -> None:
        verify_inp(self.epochs >= 1, lambda: "epochs must be positive")
        verify_inp(self.batch_size >= 1, lambda: "batch_size must be positive")
        verify_inp(
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0,
            lambda: "learning_rate must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: "weight_decay must be finite and non-negative",
        )
        verify_inp(self.warmup_steps >= 0, lambda: "warmup_steps must be non-negative")
        verify_inp(
            math.isfinite(self.ema_decay) and 0.0 < self.ema_decay < 1.0,
            lambda: "ema_decay must lie in (0,1)",
        )
        verify_inp(
            math.isfinite(self.validation_fraction)
            and 0.1 <= self.validation_fraction <= 0.5,
            lambda: "validation_fraction must lie in [0.1,0.5]",
        )
        for name, value in (
            ("arbitrage_penalty", self.arbitrage_penalty),
            ("return_broadcast_penalty", self.return_broadcast_penalty),
        ):
            verify_inp(
                math.isfinite(value) and value >= 0.0,
                lambda name=name: f"{name} must be finite and non-negative",
            )
        verify_inp(
            math.isfinite(self.progress_seconds)
            and 1.0 <= self.progress_seconds <= 300.0,
            lambda: "progress_seconds must lie in [1,300]",
        )
        verify_inp(self.checkpoint_epochs >= 1, lambda: "checkpoint_epochs must be positive")
