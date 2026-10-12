"""Run the common one-day diffusion hedge smoke."""

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
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.adapter import (
    run_diffusion_one_day,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
)
from quantdata.torch import BaseEnvironment

_DEFAULT_ORIGIN = date(2023, 1, 17)
_FIXTURE = Path(__file__).with_name("fixtures") / "smoke_expected.json"


def main() -> None:
    """Train and score one deterministic generated-scenario hedge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--origin-date", type=date.fromisoformat, default=_DEFAULT_ORIGIN)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("/tmp/quant_data/diffusion-smoke/checkpoints-v1"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--just-load", action="store_true")
    parser.add_argument("--skip-fixture-check", action="store_true")
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.dataset)
    result = run_diffusion_one_day(
        panel,
        origin_date=arguments.origin_date,
        model_config=DiffusionModelConfig(
            label="diffusion-public-cpu-smoke-v1",
            model_dimension=32,
            transformer_layers=1,
            attention_heads=4,
            diffusion_steps=8,
            dropout=0.0,
        ),
        checkpoint_directory=arguments.checkpoint_directory,
        train_config=DiffusionTrainConfig(
            epochs=3,
            batch_size=32,
            warmup_steps=2,
            checkpoint_epochs=1,
            progress_seconds=60.0,
            mixed_precision=False,
            just_load=arguments.just_load,
        ),
        hedge_config=GeneratedHedgeConfig(scenario_count=64),
        environment=BaseEnvironment(precision=32, device="cpu", debug=True),
    )
    if arguments.origin_date == _DEFAULT_ORIGIN and not arguments.skip_fixture_check:
        assert_smoke_fixture(result, _FIXTURE)
    if arguments.output is not None:
        result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
