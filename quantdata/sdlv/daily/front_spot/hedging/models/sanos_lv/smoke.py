"""Run the SANOS-LV CPU smoke test on the public candidate."""

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
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.adapter import (
    run_sanos_lv_daily_book,
)

_DEFAULT_ORIGIN = date(2023, 1, 17)
_FIXTURE = Path(__file__).with_name("fixtures") / "smoke_expected.json"
_FACTOR_FIXTURE = Path(__file__).with_name("fixtures") / "factor_smoke_expected.json"
_PRIMARY_CLAIM = "digital-call-1d-atm"
_FACTOR_CLAIM = "digital-call-3d-atm"


def main() -> None:
    """Execute and print one deterministic common-schema result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-fixture-check", action="store_true")
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.dataset)
    results = run_sanos_lv_daily_book(panel, origin_date=arguments.origin_date)
    result_by_claim = {result.claim: result for result in results}
    result = result_by_claim[_PRIMARY_CLAIM]
    if arguments.origin_date == _DEFAULT_ORIGIN and not arguments.skip_fixture_check:
        assert_smoke_fixture(result, _FIXTURE)
        assert_smoke_fixture(result_by_claim[_FACTOR_CLAIM], _FACTOR_FIXTURE)
    if arguments.output is not None:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
