"""Recover deterministic scenarios from an already fitted dated diffusion model."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.adapter import (
    _save_generated_scenarios,
    fit_public_diffusion,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    build_diffusion_live_context,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.generated_scenarios import (
    GeneratedHedgeConfig,
)
from quantdata.torch import BaseEnvironment


def export_saved_scenarios(
    *,
    as_of_date: date,
    checkpoint_directory: Path,
    warm_start_checkpoint: Path,
    warm_start_sha256: str,
    output: Path,
    device: str,
    path_count: int,
) -> str:
    """Load an authenticated fitted checkpoint without changing its weights."""
    panel = PublicHedgingPanel(default_public_candidate_path(), validate=False)
    grid = GenerativeSurfaceGrid.short_end_nms()
    model = DiffusionModelConfig(
        label="diffusion-base-short-end-nms35-2022",
        grid=grid,
        variant=DiffusionVariant.AD_SEQ_VOL,
    )
    training = DiffusionTrainConfig(
        epochs=1,
        warmup_steps=0,
        mixed_precision=False,
        just_load=True,
    )
    hedge = GeneratedHedgeConfig(scenario_count=path_count)
    fitted = fit_public_diffusion(
        panel,
        as_of_date=as_of_date,
        model_config=model,
        checkpoint_directory=checkpoint_directory,
        train_config=training,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_warm_start_sha256=warm_start_sha256,
        environment=BaseEnvironment(precision=32, device=device),
    )
    live = build_diffusion_live_context(panel, as_of_date=as_of_date, grid=grid)
    generated = sample_diffusion_transitions(
        fitted.result,
        observed_sequence=live.sequence,
        path_count=path_count,
        seed=hedge.seed,
    )
    fingerprint = _save_generated_scenarios(
        output,
        forward_log_returns=generated.forward_log_returns,
        next_log_implied_volatilities=generated.next_log_implied_volatilities,
        origin_date=as_of_date,
        forecast_date=live.forecast_date,
        dataset_manifest_sha256=panel.dataset_manifest_sha256,
        fit_checkpoint_sha256=file_sha256(fitted.result.checkpoint_path),
        warm_start_checkpoint_sha256=warm_start_sha256,
        base_checkpoint_sha256=None,
        model_config=model,
        train_config=training,
        hedge_config=hedge,
    )
    print(
        f"saved diffusion scenarios {as_of_date} paths={path_count} "
        f"checkpoint={fitted.result.checkpoint_path} sha256={fingerprint}",
        flush=True,
    )
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--path-count", type=int, default=10_000)
    export_saved_scenarios(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
