"""Shared open-loop weekly policy controls for recursive generators."""

from __future__ import annotations

from dataclasses import dataclass

from cdxcore.err import verify_inp
from cdxcore.version import version

from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
)


@version("1.0", dependencies=[HistoricalSanosConfig])
@dataclass(frozen=True, kw_only=True)
class GeneratedWeeklyPolicyConfig:
    """Scenario and risk settings shared by generated weekly baselines."""

    scenario_count: int = 100_000
    seed: int = 20_261_012
    hedge: HistoricalSanosConfig = HistoricalSanosConfig()

    def __post_init__(self) -> None:
        verify_inp(
            self.scenario_count >= 2,
            lambda: "generated weekly scenario_count must be at least two",
        )
        verify_inp(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            lambda: "generated weekly seed must be an integer",
        )
