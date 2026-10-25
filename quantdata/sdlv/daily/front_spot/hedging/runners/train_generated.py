"""Train one auditable pre-evaluation generated-model initializer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.adapter import (
    fit_public_diffusion,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.adapter import (
    fit_public_volgan,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganModelConfig,
    VolganPcaModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.diagnostics import (
    audit_volgan_conditional_dispersion,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.torch import BaseEnvironment

_LAST_PRE_EVALUATION_DATE = date(2022, 12, 30)
_PRE_EVALUATION_OBSERVATIONS = 247


def main() -> None:
    """Train a fixed initializer and write its immutable identity manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument(
        "--surface-grid", choices=("published", "short-end-nms"), default="published"
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=_LAST_PRE_EVALUATION_DATE)
    parser.add_argument("--observation-count", type=int, default=_PRE_EVALUATION_OBSERVATIONS)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--volgan-pca-rank", type=int)
    parser.add_argument("--checkpoint-epochs", type=int)
    parser.add_argument("--progress-seconds", type=float, default=240.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", type=int, choices=(32, 64))
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    arguments = parser.parse_args()
    if arguments.as_of > _LAST_PRE_EVALUATION_DATE:
        parser.error("initializer date must not enter the 2023-2025 evaluation period")

    panel = PublicHedgingPanel(arguments.dataset)
    if arguments.as_of not in panel.dates:
        parser.error(f"dataset does not contain initializer date {arguments.as_of}")
    available = panel.dates.index(arguments.as_of) + 1
    if arguments.observation_count > available:
        parser.error(
            f"initializer has {available} available dates, fewer than "
            f"--observation-count={arguments.observation_count}"
        )
    surface_grid = (
        GenerativeSurfaceGrid.short_end_nms()
        if arguments.surface_grid == "short-end-nms"
        else GenerativeSurfaceGrid()
    )
    if arguments.model == "volgan":
        if arguments.base_checkpoint is not None:
            parser.error("VolGAN does not use --base-checkpoint")
        defaults = VolganTrainConfig()
        epochs = defaults.epochs if arguments.epochs is None else arguments.epochs
        precision = 64 if arguments.precision is None else arguments.precision
        model_config = (
            VolganModelConfig(label=arguments.label, grid=surface_grid)
            if arguments.volgan_pca_rank is None
            else VolganPcaModelConfig(
                label=arguments.label,
                grid=surface_grid,
                pca_rank=arguments.volgan_pca_rank,
            )
        )
        train_config = VolganTrainConfig(
            epochs=epochs,
            gradient_matching_epochs=min(defaults.gradient_matching_epochs, epochs),
            batch_size=(
                defaults.batch_size
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            generator_learning_rate=defaults.generator_learning_rate,
            discriminator_learning_rate=defaults.discriminator_learning_rate,
            validation_fraction=defaults.validation_fraction,
            seed=defaults.seed,
            progress_seconds=arguments.progress_seconds,
            checkpoint_epochs=(
                defaults.checkpoint_epochs
                if arguments.checkpoint_epochs is None
                else arguments.checkpoint_epochs
            ),
            compile_models=arguments.compile_model,
        )
        fitted = fit_public_volgan(
            panel,
            as_of_date=arguments.as_of,
            model_config=model_config,
            checkpoint_directory=arguments.checkpoint_directory,
            observation_count=arguments.observation_count,
            train_config=train_config,
            environment=BaseEnvironment(
                precision=precision,
                device=arguments.device,
                debug=False,
            ),
        )
        training_examples = len(fitted.data.condition_dates)
        result = fitted.result
        variant = "volgan"
        conditional_audit = (
            None
            if model_config.pca_rank is None
            else audit_volgan_conditional_dispersion(
                fitted.data,
                result,
                validation_fraction=train_config.validation_fraction,
            ).as_dict()
        )
    else:
        variant_enum = DiffusionVariant(arguments.model.removeprefix("diffusion-"))
        if variant_enum is DiffusionVariant.AD_SEQ_VOL_FT and arguments.base_checkpoint is None:
            parser.error("fine-tuned diffusion requires --base-checkpoint")
        if variant_enum is DiffusionVariant.AD_SEQ_VOL and arguments.base_checkpoint is not None:
            parser.error("base checkpoint is only valid for fine-tuned diffusion")
        defaults = DiffusionTrainConfig()
        precision = 32 if arguments.precision is None else arguments.precision
        model_config = DiffusionModelConfig(
            label=arguments.label,
            variant=variant_enum,
            grid=surface_grid,
        )
        train_config = DiffusionTrainConfig(
            epochs=defaults.epochs if arguments.epochs is None else arguments.epochs,
            batch_size=(
                defaults.batch_size
                if arguments.batch_size is None
                else arguments.batch_size
            ),
            learning_rate=defaults.learning_rate,
            weight_decay=defaults.weight_decay,
            warmup_steps=defaults.warmup_steps,
            ema_decay=defaults.ema_decay,
            validation_fraction=defaults.validation_fraction,
            arbitrage_penalty=defaults.arbitrage_penalty,
            return_broadcast_penalty=defaults.return_broadcast_penalty,
            seed=defaults.seed,
            progress_seconds=arguments.progress_seconds,
            checkpoint_epochs=(
                defaults.checkpoint_epochs
                if arguments.checkpoint_epochs is None
                else arguments.checkpoint_epochs
            ),
            compile_model=arguments.compile_model,
            mixed_precision=(
                defaults.mixed_precision
                and not arguments.no_mixed_precision
                and arguments.device == "cuda"
            ),
        )
        fitted = fit_public_diffusion(
            panel,
            as_of_date=arguments.as_of,
            model_config=model_config,
            checkpoint_directory=arguments.checkpoint_directory,
            observation_count=arguments.observation_count,
            train_config=train_config,
            base_checkpoint=arguments.base_checkpoint,
            environment=BaseEnvironment(
                precision=precision,
                device=arguments.device,
                debug=False,
            ),
        )
        training_examples = len(fitted.data.start_dates)
        result = fitted.result
        variant = variant_enum.value
        conditional_audit = None

    manifest = {
        "schema": "hedging-generated-initializer-v1",
        "model": variant,
        "surface_grid": arguments.surface_grid,
        "label": arguments.label,
        "dataset_manifest_sha256": dataset_manifest_sha256(panel),
        "as_of_date": arguments.as_of.isoformat(),
        "observation_count": arguments.observation_count,
        "training_examples": training_examples,
        "device": arguments.device,
        "precision": precision,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "checkpoint_path": str(result.checkpoint_path),
        "checkpoint_sha256": file_sha256(result.checkpoint_path),
        "base_checkpoint_path": (
            None
            if arguments.base_checkpoint is None
            else str(arguments.base_checkpoint)
        ),
        "base_checkpoint_sha256": (
            None
            if arguments.base_checkpoint is None
            else file_sha256(arguments.base_checkpoint)
        ),
        "completed_epochs": result.completed_epochs,
        "validation_loss": result.validation_loss,
        "best_validation_loss": result.best_validation_loss,
        "converged": result.converged,
        "conditional_audit": conditional_audit,
    }
    output = (
        arguments.checkpoint_directory / "initializer-manifest.json"
        if arguments.output is None
        else arguments.output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
