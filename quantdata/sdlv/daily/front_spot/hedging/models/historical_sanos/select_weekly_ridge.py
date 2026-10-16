"""Select the weekly Historical-SANOS ridge before production outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly_sizing import (
    WeeklyTargetNotionalConfig,
    load_authenticated_weekly_target_notionals_from_path,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosObjective,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_ridge import (
    DEFAULT_CALIBRATION_ORIGIN,
    DEFAULT_WEEKLY_POSITION_CAP,
    DEFAULT_WEEKLY_RIDGE_CANDIDATES,
    repository_source_commit,
    run_weekly_ridge_calibration,
    write_weekly_ridge_selection,
)


def main() -> None:
    """Run and persist the authenticated weekly ridge study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--target-notional-artifact", type=Path, required=True)
    parser.add_argument("--target-notional-artifact-sha256", required=True)
    parser.add_argument("--origin-date", type=date.fromisoformat, default=DEFAULT_CALIBRATION_ORIGIN)
    parser.add_argument(
        "--ridge-candidate",
        type=float,
        action="append",
        dest="ridge_candidates",
    )
    parser.add_argument("--risk-parameter", type=float, default=0.1)
    parser.add_argument(
        "--position-cap",
        type=float,
        default=DEFAULT_WEEKLY_POSITION_CAP,
    )
    parser.add_argument(
        "--objective",
        choices=tuple(value.value for value in HistoricalSanosObjective),
        default=HistoricalSanosObjective.ENTROPY.value,
    )
    parser.add_argument("--maximum-latent-coordinate", type=float, default=1.0e5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    artifact_sha256 = _file_sha256(arguments.target_notional_artifact)
    if artifact_sha256 != arguments.target_notional_artifact_sha256:
        raise RuntimeError("target-notional artifact SHA-256 mismatch")
    panel = PublicHedgingPanel(arguments.dataset)
    target_notionals = load_authenticated_weekly_target_notionals_from_path(
        panel,
        artifact_path=arguments.target_notional_artifact,
        expected_artifact_sha256=artifact_sha256,
        config=WeeklyTargetNotionalConfig(),
    )
    selection = run_weekly_ridge_calibration(
        panel,
        target_notionals=target_notionals,
        target_notional_artifact_sha256=artifact_sha256,
        origin_date=arguments.origin_date,
        ridge_candidates=(
            DEFAULT_WEEKLY_RIDGE_CANDIDATES
            if arguments.ridge_candidates is None
            else tuple(arguments.ridge_candidates)
        ),
        objective=HistoricalSanosObjective(arguments.objective),
        risk_parameter=arguments.risk_parameter,
        position_cap=arguments.position_cap,
        maximum_latent_coordinate=arguments.maximum_latent_coordinate,
        source_commit=repository_source_commit(require_clean=True),
        progress=lambda message: print(message, flush=True),
    )
    write_weekly_ridge_selection(selection, output=arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "file_sha256": _file_sha256(arguments.output),
                "selection_sha256": selection.sha256,
                "objective": selection.objective.value,
                "risk_parameter": selection.risk_parameter,
                "selected_ridge": selection.selected_ridge,
                "selected_policy": (
                    "null-baseline"
                    if selection.selected_ridge is None
                    else "ridge"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
