"""Run the precommitted 2022 GROMM architecture study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.config import (
    GrommTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.gromm.selection import (
    GrommArchitectureStudyConfig,
    run_gromm_architecture_study,
)
from quantdata.torch import BaseEnvironment


def main() -> None:
    """Generate, fit, select, and persist the architecture evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths-per-case", type=int, default=128)
    parser.add_argument("--maximum-epochs", type=int, default=2_000)
    parser.add_argument("--minimum-epochs", type=int, default=100)
    parser.add_argument("--patience-epochs", type=int, default=100)
    parser.add_argument("--batch-pairs", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--just-load", action="store_true")
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = run_gromm_architecture_study(
        PublicHedgingPanel(arguments.dataset),
        checkpoint_directory=arguments.checkpoint_directory,
        config=GrommArchitectureStudyConfig(
            paths_per_case=arguments.paths_per_case,
        ),
        train_config=GrommTrainConfig(
            maximum_epochs=arguments.maximum_epochs,
            minimum_epochs=arguments.minimum_epochs,
            patience_epochs=arguments.patience_epochs,
            batch_pairs=arguments.batch_pairs,
            checkpoint_epochs=25,
            just_load=arguments.just_load,
        ),
        environment=BaseEnvironment(
            precision=64,
            device=arguments.device,
            debug=True,
        ),
    )
    result.write_json(arguments.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
