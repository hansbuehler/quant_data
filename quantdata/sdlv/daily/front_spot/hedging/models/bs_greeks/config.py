"""Configuration for the Black delta-and-vega benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class BsGreeksConfig:
    """Controls for the static one-session Greek hedge."""

    label: str = "bs-greeks-daily"
    group: str = "hedging/bs-greeks"
    user_version: str = "0.1"
    position_cap: float = 1_000.0
    minimum_vega: float = 1.0e-12

    def __post_init__(self) -> None:
        verify_inp(bool(self.label.strip()), lambda: "label must be non-empty")
        verify_inp(bool(self.group.strip()), lambda: "group must be non-empty")
        verify_inp(
            math.isfinite(self.position_cap) and self.position_cap > 0.0,
            lambda: "position_cap must be finite-positive",
        )
        verify_inp(
            math.isfinite(self.minimum_vega) and self.minimum_vega > 0.0,
            lambda: "minimum_vega must be finite-positive",
        )
