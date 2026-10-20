"""Configuration for the SANOS-LV reference hedge."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorConfig,
)


@version("2.5", dependencies=[RollingDlvFactorConfig])
@dataclass(frozen=True, kw_only=True)
class SanosLvConfig:
    """Numerical controls for the deterministic reference hedge."""

    label: str = "sanos-lv-daily-smoke"
    group: str = "hedging/sanos-lv"
    user_version: str = "0.1"
    position_cap: float = 1_000.0
    exposure_residual_tolerance: float = 1.0e-8
    relative_spot_step: float = 0.01
    weekly_background_grid_size: int = 2049
    weekly_background_stddev_width: float = 10.0
    rolling_factors: RollingDlvFactorConfig = RollingDlvFactorConfig()

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.exposure_residual_tolerance)
            and self.exposure_residual_tolerance > 0.0,
            lambda: "exposure_residual_tolerance must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.relative_spot_step)
            and 0.0 < self.relative_spot_step < 1.0,
            lambda: "relative_spot_step must lie strictly between zero and one",
        )
        verify_inp(
            257 <= self.weekly_background_grid_size <= 4097
            and self.weekly_background_grid_size % 2 == 1,
            lambda: (
                "weekly_background_grid_size must be odd and lie in [257,4097]"
            ),
        )
        verify_inp(
            math.isfinite(self.weekly_background_stddev_width)
            and self.weekly_background_stddev_width >= 6.0,
            lambda: "weekly_background_stddev_width must be finite and at least six",
        )
