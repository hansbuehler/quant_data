"""Small CPU-only FuNVol fit from the distributable SANOS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    build_sanos_fixed_grid_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol import (
    FunvolConfig,
    FunvolTrainConfig,
    fit_funvol,
    sample_funvol_one_day,
)
from quantdata.torch import BaseEnvironment


def main() -> None:
    """Train briefly and print finite one-day spot/surface diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--observation-count", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--path-count", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.dataset)
    config = FunvolConfig()
    history = build_sanos_fixed_grid_history(
        panel,
        end_date=panel.dates[arguments.observation_count - 1],
        observation_count=arguments.observation_count,
        grid=config.grid,
    )
    fit = fit_funvol(
        history,
        config=config,
        train_config=FunvolTrainConfig(epochs=arguments.epochs),
        environment=BaseEnvironment(precision=64, device="cpu"),
        checkpoint_path=arguments.checkpoint,
    )
    returns, surfaces = sample_funvol_one_day(
        fit,
        history,
        path_count=arguments.path_count,
        seed=20261024,
    )
    print(
        json.dumps(
            {
                "dataset_manifest_sha256": panel.dataset_manifest_sha256,
                "as_of_date": str(history.dates[-1]),
                "training_targets": fit.training_target_count,
                "validation_targets": fit.validation_target_count,
                "fpc_explained_variance": fit.fpc.explained_variance_ratio.tolist(),
                "training_loss": fit.training_loss,
                "validation_loss": fit.validation_loss,
                "completed_epochs": fit.completed_epochs,
                "checkpoint": str(fit.checkpoint_path) if fit.checkpoint_path else None,
                "sample_spot_log_return_mean": float(np.mean(returns)),
                "sample_spot_log_return_std": float(np.std(returns)),
                "sample_iv_min": float(np.min(surfaces)),
                "sample_iv_max": float(np.max(surfaces)),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
