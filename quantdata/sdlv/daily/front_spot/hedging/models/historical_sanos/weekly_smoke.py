"""Run the historical SANOS weekly policy public-data smoke test."""

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
    run_weekly_model,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_adapter import (
    HistoricalSanosWeeklyAdapter,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_pricing import (
    build_common_weekly_claims,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)


def run_historical_weekly_smoke(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    schedule: HedgeSchedule,
    objective: HistoricalSanosObjective,
) -> dict[str, object]:
    """Fit and evaluate one public-data target under one weekly schedule."""
    episode = build_weekly_episode_path(
        panel,
        origin_date=origin_date,
        schedule=schedule,
    )
    claim = build_common_weekly_claims(episode.market_days[0])[0]
    adapter = HistoricalSanosWeeklyAdapter(
        config=HistoricalSanosConfig(
            label=f"historical-sanos-weekly-{objective.value}",
            objective=objective,
        )
    )
    started = time.monotonic()
    run = run_weekly_model(
        panel,
        episode=episode,
        claim=claim,
        adapter=adapter,
    )
    elapsed = time.monotonic() - started
    print(
        f"historical SANOS weekly smoke complete schedule={schedule.value} "
        f"objective={objective.value} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return {
        "schema": "hedging-historical-sanos-weekly-smoke-v1",
        "origin_date": origin_date.isoformat(),
        "schedule": schedule.value,
        "objective": objective.value,
        "target": claim.target.label,
        "contract_count": len(episode.contracts),
        "fit_count": len(set(run.policy_fit_identities)),
        "fit_as_of_dates": list(run.fit_as_of_dates),
        "elapsed_seconds": elapsed,
        "buyer_prices_per_fit_spot": [
            record.buyer_indifference_per_fit_spot
            for record in run.fit_price_records
        ],
        "seller_prices_per_fit_spot": [
            record.seller_indifference_per_fit_spot
            for record in run.fit_price_records
        ],
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


def main() -> None:
    """Run and optionally save one historical weekly smoke result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument(
        "--schedule",
        type=HedgeSchedule,
        choices=(HedgeSchedule.FROZEN_WEEKLY, HedgeSchedule.RESTARTED_WEEKLY),
        default=HedgeSchedule.FROZEN_WEEKLY,
    )
    parser.add_argument(
        "--objective",
        type=HistoricalSanosObjective,
        choices=tuple(HistoricalSanosObjective),
        default=HistoricalSanosObjective.MEAN_VARIANCE,
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_historical_weekly_smoke(
        PublicHedgingPanel(arguments.dataset),
        origin_date=arguments.origin_date,
        schedule=arguments.schedule,
        objective=arguments.objective,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
