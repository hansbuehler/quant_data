"""Run the SANOS-LV weekly common-pricing CPU smoke test."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import (
    SanosLvConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
    price_weekly_sanos_lv_targets,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)


def run_weekly_pricing_smoke(
    day: PublicMarketDay,
    *,
    config: SanosLvConfig = SanosLvConfig(),
) -> dict[str, object]:
    """Return a JSON-ready common/model price comparison for one date."""
    model_result = price_weekly_sanos_lv_targets(day, config=config)
    model_prices = model_result.by_label()
    common_claims = build_common_weekly_claims(day, config=config)
    common_prices = {
        definition.target.label: float(definition.market_price)
        for definition in common_claims
        if definition.market_price is not None
    }
    if common_prices.keys() != model_prices.keys():
        raise RuntimeError("weekly common and model target labels do not align")
    return {
        "schema": "hedging-weekly-pricing-smoke-v1",
        "origin_date": day.trading_date.isoformat(),
        "background_grid_size": model_result.background_grid_size,
        "background_stddev_width": model_result.background_stddev_width,
        "transition_mass_error": model_result.transition_mass_error,
        "transition_martingale_error": model_result.transition_martingale_error,
        "common_prices": common_prices,
        "dlv_model_prices": model_prices,
    }


def main() -> None:
    """Price and print all weekly targets on one public-candidate date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.dataset)
    result = run_weekly_pricing_smoke(panel.market_day(arguments.origin_date))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
