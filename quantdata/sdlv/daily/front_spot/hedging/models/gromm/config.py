"""Configuration for the rolling GROMM hedging model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.hedging.mean_reduction import MeanReduction, MeanReductionConfig
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_calibration import (
    RollingDlvCalibrationConfig,
)


class GrommMeasure(StrEnum):
    """Supported scenario measures."""

    P = "P"
    Q = "Q"


class GrommObjective(StrEnum):
    """Supported deep-hedging objectives."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"


@version(
    "0.4",
    dependencies=[RollingDlvCalibrationConfig, MeanReductionConfig],
)
@dataclass(frozen=True, kw_only=True)
class GrommConfig:
    """Logical and numerical settings for daily GROMM scenarios."""

    label: str = "gromm-daily-smoke"
    group: str = "hedging/gromm"
    user_version: str = "0.1"
    calibration: RollingDlvCalibrationConfig = RollingDlvCalibrationConfig()
    measure: GrommMeasure = GrommMeasure.Q
    objective: GrommObjective = GrommObjective.MEAN_VARIANCE
    risk_parameter: float = 1.0
    path_count: int = 10_000
    batch_size: int = 10_000
    dlv_substeps: int = 8
    time_power: float = 1.0
    seed: int = 20_261_012
    position_cap: float = 1_000.0
    mean_reduction: MeanReduction = MeanReduction.DEMEAN
    mean_reduction_config: MeanReductionConfig = MeanReductionConfig()

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            self.calibration.factor.factor_rank == 16,
            lambda: "GROMM dynamics require exactly 16 rolling factors",
        )
        verify_inp(
            self.calibration.greek_factor_count == 8,
            lambda: "GROMM Greeks require exactly the first eight factors",
        )
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite and positive",
        )
        verify_inp(
            self.path_count >= 4 and self.path_count % 2 == 0,
            lambda: "path_count must be even and at least four",
        )
        verify_inp(
            self.batch_size >= 2
            and self.batch_size % 2 == 0
            and self.path_count % self.batch_size == 0,
            lambda: (
                "batch_size must be even, positive, and divide path_count"
            ),
        )
        verify_inp(
            self.dlv_substeps == 8,
            lambda: "the benchmark requires eight DLV substeps per business day",
        )
        verify_inp(
            math.isfinite(self.time_power) and self.time_power >= 1.0,
            lambda: "time_power must be finite and at least one",
        )
        verify_inp(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            lambda: "seed must be an integer",
        )
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite and positive",
        )
        verify_inp(
            isinstance(self.mean_reduction, MeanReduction),
            lambda: "mean_reduction must be a MeanReduction",
        )


@version("0.2")
@dataclass(frozen=True, kw_only=True)
class GrommTrainConfig:
    """Operational controls for restartable GROMM policy training."""

    maximum_epochs: int = 2_000
    minimum_epochs: int = 100
    patience_epochs: int = 100
    batch_pairs: int = 2_048
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    validation_fraction: float = 0.2
    minimum_relative_improvement: float = 1.0e-6
    seed: int = 20_261_012
    progress_seconds: float = 240.0
    checkpoint_epochs: int = 25
    compile_policy: bool = False
    allow_changed_config_warm_start: bool = False
    just_load: bool = False

    def __post_init__(self) -> None:
        verify_inp(
            self.maximum_epochs >= self.minimum_epochs >= 1,
            lambda: "maximum_epochs must exceed positive minimum_epochs",
        )
        verify_inp(
            self.patience_epochs >= 1,
            lambda: "patience_epochs must be positive",
        )
        verify_inp(self.batch_pairs >= 1, lambda: "batch_pairs must be positive")
        verify_inp(
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0,
            lambda: "learning_rate must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.weight_decay) and self.weight_decay >= 0.0,
            lambda: "weight_decay must be finite and non-negative",
        )
        verify_inp(
            math.isfinite(self.validation_fraction)
            and 0.1 <= self.validation_fraction <= 0.5,
            lambda: "validation_fraction must lie in [0.1,0.5]",
        )
        verify_inp(
            math.isfinite(self.minimum_relative_improvement)
            and self.minimum_relative_improvement >= 0.0,
            lambda: "minimum_relative_improvement must be non-negative",
        )
        verify_inp(
            math.isfinite(self.progress_seconds)
            and 1.0 <= self.progress_seconds <= 300.0,
            lambda: "progress_seconds must lie in [1,300]",
        )
        verify_inp(
            self.checkpoint_epochs >= 1,
            lambda: "checkpoint_epochs must be positive",
        )
