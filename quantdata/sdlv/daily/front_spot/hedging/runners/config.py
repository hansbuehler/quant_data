"""Versioned run configurations for generated daily hedge baselines."""

from __future__ import annotations

from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganTrainConfig,
)

GeneratedModelConfig = VolganModelConfig | DiffusionModelConfig
GeneratedTrainConfig = VolganTrainConfig | DiffusionTrainConfig


@version(
    "1.1",
    dependencies=[
        VolganModelConfig,
        VolganTrainConfig,
        DiffusionModelConfig,
        DiffusionTrainConfig,
        GeneratedHedgeConfig,
    ],
)
@dataclass(frozen=True, kw_only=True)
class GeneratedDailyRunConfig:
    """All result-affecting controls for one generated daily baseline."""

    label: str
    model_config: GeneratedModelConfig
    train_config: GeneratedTrainConfig
    hedge_config: GeneratedHedgeConfig
    observation_count: int = 252
    device: str = "cpu"
    precision: int = 32
    base_checkpoint_sha256: str | None = None
    warm_start_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "run label must be non-empty")
        verify_inp(
            (
                isinstance(self.model_config, VolganModelConfig)
                and isinstance(self.train_config, VolganTrainConfig)
            )
            or (
                isinstance(self.model_config, DiffusionModelConfig)
                and isinstance(self.train_config, DiffusionTrainConfig)
            ),
            lambda: "generated model and training configurations do not match",
        )
        verify_inp(
            self.observation_count >= 23,
            lambda: "generated baseline requires at least 23 observations",
        )
        verify_inp(
            self.device in {"cpu", "cuda"},
            lambda: "generated baseline device must be cpu or cuda",
        )
        verify_inp(
            self.precision in {32, 64},
            lambda: "generated baseline precision must be 32 or 64",
        )
        for name, fingerprint in (
            ("base", self.base_checkpoint_sha256),
            ("warm-start", self.warm_start_checkpoint_sha256),
        ):
            if fingerprint is None:
                continue
            verify_inp(
                len(fingerprint) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in fingerprint.lower()
                ),
                lambda name=name: f"{name} checkpoint fingerprint must be a SHA-256",
            )

    def __str__(self) -> str:
        return (
            f"{type(self.model_config).__name__}-"
            f"{self.model_config.user_version}-"
            f"{self.observation_count}obs-"
            f"{self.hedge_config.scenario_count}paths-"
            f"{self.device}{self.precision}"
        )


@version(
    "0.1",
    dependencies=[GrommConfig, GrommTrainConfig, RollingGromm2Config],
)
@dataclass(frozen=True, kw_only=True)
class GrommDailyRunConfig:
    """All result-affecting controls for one dated rolling GROMM2 run."""

    label: str
    model_config: GrommConfig
    rolling_config: RollingGromm2Config
    train_config: GrommTrainConfig
    device: str = "cpu"
    precision: int = 64

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "run label must be non-empty")
        verify_inp(
            self.device in {"cpu", "cuda"},
            lambda: "GROMM device must be cpu or cuda",
        )
        verify_inp(
            self.precision == 64,
            lambda: "GROMM2 benchmark simulation requires 64-bit precision",
        )

    def __str__(self) -> str:
        return (
            f"gromm2-{self.model_config.measure.value}-"
            f"{self.model_config.objective.value}-"
            f"lambda-{self.model_config.risk_parameter:g}-"
            f"{self.model_config.path_count}paths-{self.device}{self.precision}"
        )
