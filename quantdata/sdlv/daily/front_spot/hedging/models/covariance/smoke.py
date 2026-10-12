"""Run the independent robust-covariance CPU smoke test."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quantdata.hedging.metrics import assert_smoke_fixture
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.adapter import (
    run_covariance_one_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)
_FIXTURE = Path(__file__).with_name("fixtures") / "smoke_expected.json"


def main() -> None:
    """Execute and print one deterministic common-schema result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-fixture-check", action="store_true")
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.dataset)
    config = CovarianceHedgeConfig()
    result = run_covariance_one_day(
        panel,
        origin_date=arguments.origin_date,
        config=config,
    )
    if arguments.origin_date == _DEFAULT_ORIGIN and not arguments.skip_fixture_check:
        assert_smoke_fixture(result, _FIXTURE)
    if arguments.output is not None:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
