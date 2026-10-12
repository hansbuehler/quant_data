"""Initial-date SANOS-LV spot-delta hedge."""

from __future__ import annotations

from dataclasses import dataclass

from quantdata.sdlv.daily.front_spot.hedging.benchmark.daily import OneDayMarketSample
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig


@dataclass(frozen=True, kw_only=True)
class SanosLvAction:
    """Reference action and cap diagnostic."""

    spot: float
    option: float
    cap_hits: int


def sanos_lv_delta_action(
    sample: OneDayMarketSample,
    *,
    config: SanosLvConfig = SanosLvConfig(),
) -> SanosLvAction:
    """Hedge a long digital with the homogeneous SANOS-LV initial spot delta."""
    unconstrained = -sample.target_spot_delta
    spot = min(max(unconstrained, -config.position_cap), config.position_cap)
    return SanosLvAction(
        spot=spot,
        option=0.0,
        cap_hits=int(spot != unconstrained),
    )
