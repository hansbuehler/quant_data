"""Summarize one completed public-data daily hedge benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily_results import (
    summarize_completed_daily_run,
)


def main() -> None:
    """Parse, validate, summarize, and persist one complete daily run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-dataset", action="store_true")
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(
        arguments.dataset,
        validate=arguments.validate_dataset,
    )
    report = summarize_completed_daily_run(panel, run_directory=arguments.run)
    output = arguments.output or arguments.run / "results.json"
    report.write_json(output)
    print(
        json.dumps(
            {
                "schema": "hedging-daily-results-summary-v1",
                "run_identity": report.run_identity,
                "result_count": report.result_count,
                "slice_count": len(report.slices),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
