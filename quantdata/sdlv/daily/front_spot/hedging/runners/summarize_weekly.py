"""Summarize one completed overlapping weekly hedging run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.weekly_results import (
    summarize_completed_weekly_run,
)


def main() -> None:
    """Parse one completed weekly run and write its aggregate statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_public_candidate_path(),
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    panel = PublicHedgingPanel(arguments.dataset, validate=True)
    results = summarize_completed_weekly_run(
        panel,
        run_directory=arguments.run_directory,
    )
    output = arguments.output or arguments.run_directory / "results.json"
    results.write_json(output)
    print(
        json.dumps(
            {
                "schema": results.to_dict()["schema"],
                "run_identity": results.run_identity,
                "result_count": results.result_count,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
