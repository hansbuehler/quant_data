"""Run the SANOS-LV weekly hedge-policy public-data smoke test."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.hedging.protocols import HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    build_weekly_episode_path,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_runner import (
    WeeklyFittedPolicy,
    WeeklyMarketFit,
    run_weekly_model,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_adapter import (
    SanosLvWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)


def run_weekly_policy_smoke(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    all_targets: bool,
) -> dict[str, object]:
    """Run one frozen-weekly public-data episode with shared fit caches."""
    episode = build_weekly_episode_path(
        panel,
        origin_date=origin_date,
        schedule=HedgeSchedule.FROZEN_WEEKLY,
    )
    claims = build_common_weekly_claims(episode.market_days[0])
    if not all_targets:
        claims = claims[:1]
    adapter = SanosLvWeeklyAdapter()
    market_cache: dict[str, WeeklyMarketFit] = {}
    policy_cache: dict[str, WeeklyFittedPolicy] = {}
    started = time.monotonic()
    records: list[dict[str, object]] = []
    for index, claim in enumerate(claims, start=1):
        run = run_weekly_model(
            panel,
            episode=episode,
            claim=claim,
            adapter=adapter,
            market_fit_cache=market_cache,
            policy_fit_cache=policy_cache,
        )
        records.append(
            {
                "target": claim.target.label,
                "model_price": run.result.breakdown.model_price,
                "market_price": run.result.breakdown.market_price,
                "pure_hedge_error": run.result.breakdown.pure_hedge_error,
                "economic_pnl": run.result.breakdown.economic_pnl,
                "transaction_cost": run.result.transaction_cost,
                "maximum_absolute_spot_position": float(
                    np.max(np.abs(run.spot_positions))
                ),
                "nonzero_option_positions": int(
                    np.count_nonzero(np.abs(run.option_positions) > 1.0e-12)
                ),
            }
        )
        print(
            f"SANOS-LV weekly smoke {index}/{len(claims)} "
            f"target={claim.target.label} elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    return {
        "schema": "hedging-sanos-lv-weekly-policy-smoke-v1",
        "origin_date": origin_date.isoformat(),
        "schedule": HedgeSchedule.FROZEN_WEEKLY.value,
        "contract_count": len(episode.contracts),
        "target_count": len(records),
        "market_fit_count": len(market_cache),
        "policy_fit_count": len(policy_cache),
        "elapsed_seconds": time.monotonic() - started,
        "records": records,
    }


def main() -> None:
    """Run and optionally save the weekly SANOS-LV policy smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_weekly_policy_smoke(
        PublicHedgingPanel(arguments.dataset),
        origin_date=arguments.origin_date,
        all_targets=arguments.all_targets,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
