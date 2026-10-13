"""Select rolling generated-model update epochs using only 2022 validation dates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    file_sha256,
    remove_completed_checkpoint,
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
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    BUDGET_SELECTION_OBSERVATION_COUNT,
)
from quantdata.torch import BaseEnvironment, torch


@dataclass(frozen=True, kw_only=True)
class EpochBudgetScore:
    """Validation-loss summary for one rolling-update budget."""

    epochs: int
    losses: tuple[float, ...]
    mean: float
    standard_error: float


def select_epoch_budget(
    losses_by_epoch: dict[int, tuple[float, ...]],
) -> tuple[int, tuple[EpochBudgetScore, ...]]:
    """Return the cheapest budget within one standard error of the best mean."""
    if not losses_by_epoch:
        raise ValueError("at least one epoch budget is required")
    scores: list[EpochBudgetScore] = []
    expected_count: int | None = None
    for epochs, losses in sorted(losses_by_epoch.items()):
        if epochs <= 0 or not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("epoch budgets and validation losses must be finite-positive")
        if expected_count is None:
            expected_count = len(losses)
        elif len(losses) != expected_count:
            raise ValueError("epoch budgets do not share the same validation origins")
        values = np.asarray(losses, dtype=np.float64)
        standard_error = (
            0.0
            if len(values) == 1
            else float(np.std(values, ddof=1) / np.sqrt(len(values)))
        )
        scores.append(
            EpochBudgetScore(
                epochs=epochs,
                losses=losses,
                mean=float(np.mean(values)),
                standard_error=standard_error,
            )
        )
    best = min(scores, key=lambda score: (score.mean, score.epochs))
    threshold = best.mean + best.standard_error
    selected = min(score.epochs for score in scores if score.mean <= threshold)
    return selected, tuple(scores)


def main() -> None:
    """Run fixed-source rolling fits on selected 2022 origins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("volgan", "diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--label", required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--validation-start", type=date.fromisoformat, required=True)
    parser.add_argument("--validation-end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--observation-count",
        type=int,
        default=BUDGET_SELECTION_OBSERVATION_COUNT,
    )
    parser.add_argument("--origin-count", type=int, default=8)
    parser.add_argument("--candidate-epochs", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-epochs", type=int)
    parser.add_argument("--progress-seconds", type=float, default=240.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", type=int, choices=(32, 64))
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--retain-fit-checkpoints", action="store_true")
    arguments = parser.parse_args()
    candidates = _parse_epoch_candidates(arguments.candidate_epochs)
    if arguments.validation_start > arguments.validation_end:
        parser.error("validation start follows validation end")
    if arguments.validation_end.year > 2022:
        parser.error("generated-model budget selection must not use post-2022 dates")
    if arguments.observation_count != BUDGET_SELECTION_OBSERVATION_COUNT:
        parser.error(
            "generated-model budget selection requires the precommitted "
            f"{BUDGET_SELECTION_OBSERVATION_COUNT}-observation window"
        )
    if arguments.origin_count < 2:
        parser.error("--origin-count must be at least two")

    panel = PublicHedgingPanel(arguments.dataset)
    eligible = tuple(
        value
        for index, value in enumerate(panel.dates)
        if arguments.validation_start <= value <= arguments.validation_end
        and index + 1 >= arguments.observation_count
    )
    if len(eligible) < arguments.origin_count:
        parser.error(
            f"only {len(eligible)} eligible validation origins for "
            f"--origin-count={arguments.origin_count}"
        )
    origins = _evenly_spaced_dates(eligible, arguments.origin_count)
    source_metadata = _checkpoint_metadata(arguments.warm_start_checkpoint)
    source_sha256 = file_sha256(arguments.warm_start_checkpoint)
    base_sha256 = (
        None
        if arguments.base_checkpoint is None
        else file_sha256(arguments.base_checkpoint)
    )
    source_end_date = date.fromisoformat(str(source_metadata["training_end_date"]))
    if source_end_date >= origins[0]:
        parser.error("warm-start checkpoint must predate every validation origin")

    losses_by_epoch: dict[int, tuple[float, ...]] = {}
    checkpoints_by_epoch: dict[int, tuple[tuple[Path, str], ...]] = {}
    for candidate_index, epochs in enumerate(candidates, start=1):
        losses: list[float] = []
        checkpoints: list[tuple[Path, str]] = []
        for origin_index, origin in enumerate(origins, start=1):
            print(
                json.dumps(
                    {
                        "stage": "generated-budget-selection",
                        "candidate": candidate_index,
                        "candidate_count": len(candidates),
                        "epochs": epochs,
                        "origin": origin.isoformat(),
                        "origin_index": origin_index,
                        "origin_count": len(origins),
                    }
                ),
                flush=True,
            )
            loss, checkpoint = _fit_candidate(
                panel,
                model=arguments.model,
                label=arguments.label,
                origin=origin,
                observation_count=arguments.observation_count,
                epochs=epochs,
                batch_size=arguments.batch_size,
                checkpoint_epochs=arguments.checkpoint_epochs,
                progress_seconds=arguments.progress_seconds,
                checkpoint_directory=(
                    arguments.checkpoint_directory / f"epochs-{epochs}"
                ),
                warm_start_checkpoint=arguments.warm_start_checkpoint,
                base_checkpoint=arguments.base_checkpoint,
                expected_warm_start_sha256=source_sha256,
                expected_base_checkpoint_sha256=base_sha256,
                device=arguments.device,
                precision=arguments.precision,
                compile_model=arguments.compile_model,
                no_mixed_precision=arguments.no_mixed_precision,
            )
            losses.append(loss)
            checkpoints.append((checkpoint, file_sha256(checkpoint)))
        losses_by_epoch[epochs] = tuple(losses)
        checkpoints_by_epoch[epochs] = tuple(checkpoints)
    selected, scores = select_epoch_budget(losses_by_epoch)
    result = {
        "schema": "hedging-generated-epoch-selection-v1",
        "model": arguments.model,
        "label": arguments.label,
        "dataset_manifest_sha256": dataset_manifest_sha256(panel),
        "warm_start_checkpoint": str(arguments.warm_start_checkpoint),
        "warm_start_sha256": source_sha256,
        "base_checkpoint_sha256": base_sha256,
        "warm_start_training_end_date": source_end_date.isoformat(),
        "validation_origins": [value.isoformat() for value in origins],
        "observation_count": arguments.observation_count,
        "one_standard_error_rule": True,
        "selected_epochs": selected,
        "scores": [
            {
                "epochs": score.epochs,
                "losses": score.losses,
                "mean": score.mean,
                "standard_error": score.standard_error,
                "fit_checkpoint_sha256": [
                    fingerprint
                    for _, fingerprint in checkpoints_by_epoch[score.epochs]
                ],
            }
            for score in scores
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not arguments.retain_fit_checkpoints:
        for checkpoints in checkpoints_by_epoch.values():
            for checkpoint, _ in checkpoints:
                if checkpoint.resolve() == arguments.warm_start_checkpoint.resolve():
                    raise RuntimeError("refusing to remove the warm-start checkpoint")
                remove_completed_checkpoint(checkpoint)


def _fit_candidate(
    panel: PublicHedgingPanel,
    *,
    model: str,
    label: str,
    origin: date,
    observation_count: int,
    epochs: int,
    batch_size: int | None,
    checkpoint_epochs: int | None,
    progress_seconds: float,
    checkpoint_directory: Path,
    warm_start_checkpoint: Path,
    base_checkpoint: Path | None,
    expected_warm_start_sha256: str,
    expected_base_checkpoint_sha256: str | None,
    device: str,
    precision: int | None,
    compile_model: bool,
    no_mixed_precision: bool,
) -> tuple[float, Path]:
    if model == "volgan":
        if base_checkpoint is not None:
            raise ValueError("VolGAN does not use a base checkpoint")
        defaults = VolganTrainConfig()
        actual_precision = 64 if precision is None else precision
        fitted = fit_public_volgan(
            panel,
            as_of_date=origin,
            model_config=VolganModelConfig(label=label),
            checkpoint_directory=checkpoint_directory,
            observation_count=observation_count,
            train_config=VolganTrainConfig(
                epochs=epochs,
                gradient_matching_epochs=0,
                batch_size=defaults.batch_size if batch_size is None else batch_size,
                generator_learning_rate=defaults.generator_learning_rate,
                discriminator_learning_rate=defaults.discriminator_learning_rate,
                validation_fraction=defaults.validation_fraction,
                seed=defaults.seed,
                progress_seconds=progress_seconds,
                checkpoint_epochs=(
                    min(defaults.checkpoint_epochs, epochs)
                    if checkpoint_epochs is None
                    else checkpoint_epochs
                ),
                compile_models=compile_model,
            ),
            warm_start_checkpoint=warm_start_checkpoint,
            expected_warm_start_sha256=expected_warm_start_sha256,
            environment=BaseEnvironment(
                precision=actual_precision,
                device=device,
                debug=False,
            ),
        )
        return fitted.result.validation_loss, fitted.result.checkpoint_path

    variant = DiffusionVariant(model.removeprefix("diffusion-"))
    if variant is DiffusionVariant.AD_SEQ_VOL_FT and base_checkpoint is None:
        raise ValueError("fine-tuned diffusion requires a base checkpoint")
    if variant is DiffusionVariant.AD_SEQ_VOL and base_checkpoint is not None:
        raise ValueError("base checkpoint is only valid for fine-tuned diffusion")
    defaults = DiffusionTrainConfig()
    actual_precision = 32 if precision is None else precision
    fitted = fit_public_diffusion(
        panel,
        as_of_date=origin,
        model_config=DiffusionModelConfig(label=label, variant=variant),
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=DiffusionTrainConfig(
            epochs=epochs,
            batch_size=defaults.batch_size if batch_size is None else batch_size,
            learning_rate=defaults.learning_rate,
            weight_decay=defaults.weight_decay,
            warmup_steps=0,
            ema_decay=defaults.ema_decay,
            validation_fraction=defaults.validation_fraction,
            arbitrage_penalty=defaults.arbitrage_penalty,
            return_broadcast_penalty=defaults.return_broadcast_penalty,
            seed=defaults.seed,
            progress_seconds=progress_seconds,
            checkpoint_epochs=(
                min(defaults.checkpoint_epochs, epochs)
                if checkpoint_epochs is None
                else checkpoint_epochs
            ),
            compile_model=compile_model,
            mixed_precision=(
                defaults.mixed_precision
                and not no_mixed_precision
                and device == "cuda"
            ),
        ),
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=warm_start_checkpoint,
        expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
        expected_warm_start_sha256=expected_warm_start_sha256,
        environment=BaseEnvironment(
            precision=actual_precision,
            device=device,
            debug=False,
        ),
    )
    return fitted.result.validation_loss, fitted.result.checkpoint_path


def _parse_epoch_candidates(value: str) -> tuple[int, ...]:
    try:
        candidates = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("candidate epochs must be comma-separated integers") from error
    if not candidates or candidates[0] <= 0:
        raise ValueError("candidate epochs must be positive")
    return candidates


def _evenly_spaced_dates(values: tuple[date, ...], count: int) -> tuple[date, ...]:
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    selected = tuple(values[int(index)] for index in indices)
    if len(set(selected)) != count:
        raise ValueError("validation range is too short for distinct spaced origins")
    return selected


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "training_end_date" not in payload:
        raise ValueError("warm-start checkpoint has no auditable training end date")
    return payload


if __name__ == "__main__":
    main()
