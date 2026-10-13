"""Prepare or fit one leakage-free rolling GROMM2 market artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.runners.jax_environment import (
    force_cpu_for_prepare_cli,
)

force_cpu_for_prepare_cli(sys.argv)

from quantdata.sdlv.daily.front_spot.hedging.models.gromm.rolling_gromm2 import (
    RollingGromm2Config,
    fit_rolling_gromm2_market,
    prepare_rolling_gromm2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--stage", choices=("prepare", "fit"), required=True)
    arguments = parser.parse_args()
    config = RollingGromm2Config()
    print(
        json.dumps(
            {
                "stage": f"rolling-gromm2-{arguments.stage}-start",
                "as_of_date": arguments.as_of_date.isoformat(),
                "dataset": str(arguments.dataset.resolve()),
                "artifact_root": str(arguments.artifact_root.resolve()),
                "config": asdict(config),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if arguments.stage == "prepare":
        result = prepare_rolling_gromm2(
            dataset=arguments.dataset,
            artifact_root=arguments.artifact_root,
            as_of_date=arguments.as_of_date,
            config=config,
        )
        payload = {
            "stage": "rolling-gromm2-prepare-complete",
            "as_of_date": arguments.as_of_date.isoformat(),
            "manifest": str(result.manifest.resolve()),
        }
    else:
        result = fit_rolling_gromm2_market(
            dataset=arguments.dataset,
            artifact_root=arguments.artifact_root,
            as_of_date=arguments.as_of_date,
            config=config,
        )
        payload = {
            "stage": "rolling-gromm2-fit-complete",
            "as_of_date": arguments.as_of_date.isoformat(),
            "manifest": str(result.paths.manifest.resolve()),
            "percent_inside_spread": result.percent_inside_spread,
            "explained_variance_ratio": result.explained_variance_ratio,
            "physical_projection_fraction": result.physical_projection_fraction,
            "maximum_daily_risk_price_norm": (
                result.maximum_daily_risk_price_norm
            ),
            "physical_accepted": result.physical_accepted,
        }
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
