"""Reconcile expensive shared weekly inputs before production evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalConfig,
    load_or_calibrate_weekly_target_notionals,
    weekly_target_notional_artifact_path,
    weekly_target_notional_calibration_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)


def prepare_weekly_inputs(
    panel: PublicHedgingPanel,
    *,
    cache_directory: Path,
    config: WeeklyTargetNotionalConfig = WeeklyTargetNotionalConfig(),
) -> dict[str, object]:
    """Build the immutable weekly sizing handoff used by all weekly models."""
    calibration = load_or_calibrate_weekly_target_notionals(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    artifact = weekly_target_notional_artifact_path(
        panel,
        cache_directory=cache_directory,
        config=config,
    )
    return {
        "schema": "hedging-weekly-input-preflight-v1",
        "dataset_manifest_sha256": dataset_manifest_sha256(panel),
        "target_notional_artifact": str(artifact),
        "target_notional_artifact_sha256": file_sha256(artifact),
        "target_notional_calibration_sha256": (
            weekly_target_notional_calibration_sha256(calibration)
        ),
        "target_count": len(calibration.target_labels),
        "episode_count": len(calibration.episode_origin_dates),
        "as_of_date": calibration.as_of_date.isoformat(),
        "reference_spot_standard_deviation": (
            calibration.reference_spot_standard_deviation
        ),
        "target_quantities": calibration.quantity_by_label,
    }


def main() -> None:
    """Run the CPU reconciliation and emit authenticated handoff metadata."""
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

    payload = prepare_weekly_inputs(
        PublicHedgingPanel(arguments.dataset, validate=True),
        cache_directory=cache_directory,
    )
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
