from __future__ import annotations

from dataclasses import replace

from quantdata.sdlv.daily.front_spot.hedging.benchmark.daily import one_day_market_sample
from quantdata.sdlv.daily.front_spot.hedging.data.fixtures import (
    synthetic_one_day_transition,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.hedger import (
    sanos_lv_delta_action,
)


def test_sanos_lv_action_is_opposite_the_long_claim_delta() -> None:
    sample = one_day_market_sample(synthetic_one_day_transition())
    action = sanos_lv_delta_action(sample)
    assert action.spot == -sample.target_spot_delta
    assert action.option == 0.0
    assert action.cap_hits == 0


def test_sanos_lv_action_reports_cap_hit() -> None:
    sample = one_day_market_sample(synthetic_one_day_transition())
    sample = replace(sample, target_spot_delta=2.0)
    action = sanos_lv_delta_action(sample, config=SanosLvConfig(position_cap=1.0))
    assert action.spot == -1.0
    assert action.cap_hits == 1
