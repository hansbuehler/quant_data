"""Configuration for covariance-weighted SANOS Greek hedges."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig


@version("1.0", dependencies=[SanosLvConfig])
@dataclass(frozen=True, kw_only=True)
class SanosParametricConfig:
    """Rolling factor covariance and one-day cost-aware risk controls."""

    label: str = "sanos-parametric-daily"
    group: str = "hedging/sanos-parametric"
    user_version: str = "0.1"
    greeks: SanosLvConfig = SanosLvConfig()
    risk_parameter: float = 10.0
    spot_entry_cost: float = 0.0

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            math.isfinite(self.risk_parameter) and self.risk_parameter > 0.0,
            lambda: "risk_parameter must be finite and positive",
        )
        verify_inp(
            math.isfinite(self.spot_entry_cost)
            and self.spot_entry_cost >= 0.0,
            lambda: "spot_entry_cost must be finite and non-negative",
        )
