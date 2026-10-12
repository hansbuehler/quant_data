"""Run the deterministic GROMM CPU smoke test on the public candidate."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import jax

from quantdata.hedging.metrics import assert_smoke_fixture
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.adapter import (
    run_gromm_one_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommConfig,
    GrommMeasure,
    GrommObjective,
    GrommTrainConfig,
)
from quantdata.torch import BaseEnvironment

_DEFAULT_ORIGIN = date(2023, 1, 17)
_FIXTURE = Path(__file__).with_name("fixtures") / "smoke_expected.json"


def _default_checkpoint_directory() -> Path:
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        raise RuntimeError("QUANTDATA_ROOT is required for GROMM checkpoints")
    return Path(root) / "training" / "hedging" / "gromm" / "cpu-smoke"


def main() -> None:
    """Execute and print one deterministic common-schema GROMM result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--just-load", action="store_true")
    parser.add_argument("--skip-fixture-check", action="store_true")
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    config = GrommConfig(
        label="gromm-cpu-smoke",
        measure=GrommMeasure.Q,
        objective=GrommObjective.MEAN_VARIANCE,
        risk_parameter=10.0,
        path_count=128,
        batch_size=128,
    )
    train_config = GrommTrainConfig(
        maximum_epochs=2_000,
        minimum_epochs=100,
        patience_epochs=100,
        batch_pairs=64,
        learning_rate=3.0e-3,
        minimum_relative_improvement=1.0e-5,
        checkpoint_epochs=25,
        seed=config.seed,
        just_load=arguments.just_load,
    )
    panel = PublicHedgingPanel(arguments.dataset)
    result = run_gromm_one_day(
        panel,
        origin_date=arguments.origin_date,
        checkpoint_directory=(
            _default_checkpoint_directory()
            if arguments.checkpoint_directory is None
            else arguments.checkpoint_directory
        ),
        config=config,
        train_config=train_config,
        environment=BaseEnvironment(precision=64, device="cpu", debug=True),
    )
    if arguments.origin_date == _DEFAULT_ORIGIN and not arguments.skip_fixture_check:
        assert_smoke_fixture(result, _FIXTURE)
    if arguments.output is not None:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
