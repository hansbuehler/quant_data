"""CPU-only one-origin FuNVol fixed-contract hedge gate."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.adapter import run_funvol_one_day
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import FunvolTrainConfig
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import GeneratedHedgeConfig
from quantdata.torch import BaseEnvironment


def main() -> None:
    """Fit one date and print the common benchmark accounting record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--observation-count", type=int, default=252)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scenario-count", type=int, default=10_000)
    parser.add_argument("--claim-quantity", type=float, default=1.0)
    arguments = parser.parse_args()
    result = run_funvol_one_day(
        PublicHedgingPanel(arguments.dataset),
        origin_date=arguments.origin_date,
        checkpoint_path=arguments.checkpoint,
        warm_start_checkpoint=arguments.warm_start_checkpoint,
        observation_count=arguments.observation_count,
        train_config=FunvolTrainConfig(epochs=arguments.epochs),
        hedge_config=GeneratedHedgeConfig(scenario_count=arguments.scenario_count),
        environment=BaseEnvironment(precision=64, device="cpu"),
        claim_quantity=arguments.claim_quantity,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
