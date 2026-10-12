"""Reconcile expensive shared daily inputs before claiming a GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.benchmark.sizing import (
    TargetNotionalConfig,
    daily_target_notional_artifact_path,
    daily_target_notional_calibration_sha256,
    daily_target_notional_file_sha256,
    load_or_calibrate_daily_target_notionals,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)


def main() -> None:
    """Run the full CPU reconciliation and emit its immutable handoff hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cache_directory = arguments.cache_directory
    if cache_directory is None:
        root = os.environ.get("QUANTDATA_ROOT")
        if not root:
            parser.error("QUANTDATA_ROOT or --cache-directory is required")
        cache_directory = Path(root) / "processing" / "hedging" / "target-notionals"

    panel = PublicHedgingPanel(arguments.dataset)
    config = TargetNotionalConfig()
    calibration = load_or_calibrate_daily_target_notionals(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    artifact = daily_target_notional_artifact_path(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    payload = {
        "schema": "hedging-daily-input-preflight-v1",
        "dataset_manifest_sha256": dataset_manifest_sha256(panel),
        "target_notional_artifact": str(artifact),
        "target_notional_artifact_sha256": daily_target_notional_file_sha256(
            artifact
        ),
        "target_notional_calibration_sha256": (
            daily_target_notional_calibration_sha256(calibration)
        ),
        "target_count": len(calibration.target_labels),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="", flush=True)
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(arguments.output)
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
