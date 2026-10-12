"""Run the historical SANOS CPU smoke test."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.adapter import (
    run_historical_sanos_one_day,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)


def main() -> None:
    """Execute and print one deterministic common-schema result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_historical_sanos_one_day(
        PublicHedgingPanel(arguments.dataset),
        origin_date=arguments.origin_date,
    )
    if arguments.output is not None:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
