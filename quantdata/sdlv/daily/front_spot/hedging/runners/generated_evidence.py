"""Authenticate generated-model initializers and 2022 budget selections."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    file_sha256,
)
from quantdata.torch import torch

INITIALIZER_SCHEMA = "hedging-generated-initializer-v1"
SELECTION_SCHEMA = "hedging-generated-epoch-selection-v1"


@dataclass(frozen=True, kw_only=True)
class GeneratedSelectionEvidence:
    """Immutable checkpoint and epoch budget selected without evaluation data."""

    model: str
    initializer_checkpoint: Path
    initializer_checkpoint_sha256: str
    selected_epochs: int
    base_checkpoint: Path | None
    base_checkpoint_sha256: str | None
    initializer_manifest_sha256: str
    epoch_selection_sha256: str


@dataclass(frozen=True, kw_only=True)
class _CheckpointEvidence:
    model: str
    training_end_date: date
    completed_epochs: int
    base_checkpoint_sha256: str | None


def load_generated_selection_evidence(
    *,
    initializer_manifest: Path,
    epoch_selection: Path,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
    base_initializer_manifest: Path | None = None,
) -> GeneratedSelectionEvidence:
    """Load and cross-check one pre-evaluation generated-model selection."""
    initializer_path = Path(initializer_manifest)
    selection_path = Path(epoch_selection)
    initializer = _read_json_object(initializer_path)
    selection = _read_json_object(selection_path)
    normalized_model = _normalized_model(expected_model)
    if initializer.get("schema") != INITIALIZER_SCHEMA:
        raise ValueError("unexpected generated initializer schema")
    if _normalized_model(_require_string(initializer, "model")) != normalized_model:
        raise ValueError("generated initializer model does not match the requested model")
    if initializer.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated initializer uses another dataset")
    checkpoint = Path(_require_string(initializer, "checkpoint_path"))
    checkpoint_sha256 = _require_sha256(initializer, "checkpoint_sha256")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("generated initializer checkpoint hash mismatch")
    checkpoint_evidence = _load_checkpoint_evidence(
        checkpoint,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )
    initializer_as_of = date.fromisoformat(_require_string(initializer, "as_of_date"))
    if initializer_as_of != checkpoint_evidence.training_end_date:
        raise ValueError("generated initializer date differs from checkpoint payload")
    if initializer.get("completed_epochs") != checkpoint_evidence.completed_epochs:
        raise ValueError("generated initializer completion differs from checkpoint payload")

    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected generated epoch-selection schema")
    if _normalized_model(_require_string(selection, "model")) != normalized_model:
        raise ValueError("generated epoch selection model does not match")
    if selection.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated epoch selection uses another dataset")
    selected_epochs = selection.get("selected_epochs")
    if not isinstance(selected_epochs, int) or selected_epochs <= 0:
        raise ValueError("generated epoch selection has no positive selected budget")
    if Path(_require_string(selection, "warm_start_checkpoint")) != checkpoint:
        raise ValueError("generated epoch selection uses another initializer path")
    if _require_sha256(selection, "warm_start_sha256") != checkpoint_sha256:
        raise ValueError("generated epoch selection uses another initializer hash")
    claimed_source_end_date = date.fromisoformat(
        _require_string(selection, "warm_start_training_end_date")
    )
    if claimed_source_end_date != checkpoint_evidence.training_end_date:
        raise ValueError("generated selection source date differs from checkpoint")
    if selection.get("one_standard_error_rule") is not True:
        raise ValueError("generated epoch selection did not use one-standard-error rule")
    if selection.get("observation_count") != 252:
        raise ValueError("generated epoch selection does not use 252 observations")
    raw_validation_origins = selection.get("validation_origins")
    if not isinstance(raw_validation_origins, list) or not raw_validation_origins:
        raise ValueError("generated epoch selection has no validation origins")
    validation_origins = tuple(
        date.fromisoformat(str(value)) for value in raw_validation_origins
    )
    if (
        any(value.year > 2022 for value in validation_origins)
        or any(
            validation_origins[index] >= validation_origins[index + 1]
            for index in range(len(validation_origins) - 1)
        )
        or checkpoint_evidence.training_end_date >= validation_origins[0]
    ):
        raise ValueError("generated epoch selection uses evaluation-period data")
    selected_scores = _validated_budget_scores(
        selection.get("scores"),
        validation_count=len(validation_origins),
    )
    recomputed_selection = _one_standard_error_selection(selected_scores)
    if selected_epochs != recomputed_selection:
        raise ValueError("generated epoch budget is not the recomputed one-SE choice")

    selected_base_sha256 = selection.get("base_checkpoint_sha256")
    if selected_base_sha256 is not None:
        selected_base_sha256 = _validate_sha256_value(
            selected_base_sha256,
            name="base_checkpoint_sha256",
        )
    base_checkpoint = None
    base_checkpoint_sha256 = None
    if base_initializer_manifest is not None:
        base = load_generated_initializer_evidence(
            initializer_manifest=base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        base_checkpoint = base.initializer_checkpoint
        base_checkpoint_sha256 = base.initializer_checkpoint_sha256
    if normalized_model == "diffusion-ad-seq-vol-ft":
        initializer_base_path = initializer.get("base_checkpoint_path")
        initializer_base_sha256 = initializer.get("base_checkpoint_sha256")
        if (
            base_checkpoint is None
            or initializer_base_path != str(base_checkpoint)
            or initializer_base_sha256 != base_checkpoint_sha256
            or selected_base_sha256 != base_checkpoint_sha256
            or checkpoint_evidence.base_checkpoint_sha256
            != base_checkpoint_sha256
        ):
            raise ValueError("fine-tuned generated selection has another base checkpoint")
    elif (
        selected_base_sha256 is not None
        or base_checkpoint is not None
        or checkpoint_evidence.base_checkpoint_sha256 is not None
    ):
        raise ValueError("non-fine-tuned generated selection cannot use a base checkpoint")

    return GeneratedSelectionEvidence(
        model=normalized_model,
        initializer_checkpoint=checkpoint,
        initializer_checkpoint_sha256=checkpoint_sha256,
        selected_epochs=selected_epochs,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        initializer_manifest_sha256=file_sha256(initializer_path),
        epoch_selection_sha256=file_sha256(selection_path),
    )


def load_generated_initializer_evidence(
    *,
    initializer_manifest: Path,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
) -> GeneratedSelectionEvidence:
    """Authenticate an initializer when no rolling epoch budget is required."""
    path = Path(initializer_manifest)
    payload = _read_json_object(path)
    normalized_model = _normalized_model(expected_model)
    if payload.get("schema") != INITIALIZER_SCHEMA:
        raise ValueError("unexpected generated initializer schema")
    if _normalized_model(_require_string(payload, "model")) != normalized_model:
        raise ValueError("generated initializer model does not match the requested model")
    if payload.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated initializer uses another dataset")
    checkpoint = Path(_require_string(payload, "checkpoint_path"))
    checkpoint_sha256 = _require_sha256(payload, "checkpoint_sha256")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("generated initializer checkpoint hash mismatch")
    checkpoint_evidence = _load_checkpoint_evidence(
        checkpoint,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )
    as_of_date = date.fromisoformat(_require_string(payload, "as_of_date"))
    if as_of_date != checkpoint_evidence.training_end_date:
        raise ValueError("generated initializer date differs from checkpoint payload")
    if payload.get("completed_epochs") != checkpoint_evidence.completed_epochs:
        raise ValueError("generated initializer completion differs from checkpoint payload")
    return GeneratedSelectionEvidence(
        model=normalized_model,
        initializer_checkpoint=checkpoint,
        initializer_checkpoint_sha256=checkpoint_sha256,
        selected_epochs=0,
        base_checkpoint=None,
        base_checkpoint_sha256=None,
        initializer_manifest_sha256=file_sha256(path),
        epoch_selection_sha256="",
    )


def _normalized_model(model: str) -> str:
    return model if model == "volgan" or model.startswith("diffusion-") else f"diffusion-{model}"


def _load_checkpoint_evidence(
    path: Path,
    *,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
) -> _CheckpointEvidence:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("generated initializer checkpoint must contain a mapping")
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("generated initializer checkpoint format is incompatible")
    if payload.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated initializer checkpoint uses another dataset")
    for name in ("model_code_version", "model_user_version", "model_config_identity"):
        _require_string(payload, name)
    _require_string(payload, "precision")
    training_start = date.fromisoformat(_require_string(payload, "training_start_date"))
    training_end = date.fromisoformat(_require_string(payload, "training_end_date"))
    if training_start > training_end or training_end.year > 2022:
        raise ValueError("generated initializer checkpoint enters evaluation data")
    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        raise ValueError("generated initializer checkpoint has no training config")
    epochs = train_config.get("epochs")
    epoch_index = payload.get("epoch")
    if (
        not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or epochs <= 0
        or not isinstance(epoch_index, int)
        or isinstance(epoch_index, bool)
        or epoch_index + 1 < epochs
    ):
        raise ValueError("generated initializer checkpoint is incomplete")

    normalized_model = _normalized_model(expected_model)
    base_fingerprint = payload.get("base_checkpoint_fingerprint")
    if base_fingerprint is not None:
        base_fingerprint = _validate_sha256_value(
            base_fingerprint,
            name="base_checkpoint_fingerprint",
        )
    if normalized_model == "volgan":
        actual_model = "volgan"
        if not isinstance(payload.get("generator"), dict) or not isinstance(
            payload.get("discriminator"), dict
        ):
            raise ValueError("VolGAN initializer checkpoint has no model state")
        if base_fingerprint is not None:
            raise ValueError("VolGAN initializer cannot use a base checkpoint")
    else:
        variant = _require_string(payload, "variant")
        actual_model = _normalized_model(variant)
        if not isinstance(payload.get("model"), dict) or not isinstance(
            payload.get("ema_model"), dict
        ):
            raise ValueError("diffusion initializer checkpoint has no model state")
        if actual_model == "diffusion-ad-seq-vol-ft" and base_fingerprint is None:
            raise ValueError("fine-tuned diffusion checkpoint has no base fingerprint")
        if actual_model != "diffusion-ad-seq-vol-ft" and base_fingerprint is not None:
            raise ValueError("base diffusion checkpoint unexpectedly has a base")
    if actual_model != normalized_model:
        raise ValueError("generated initializer checkpoint has another model variant")
    return _CheckpointEvidence(
        model=actual_model,
        training_end_date=training_end,
        completed_epochs=epoch_index + 1,
        base_checkpoint_sha256=base_fingerprint,
    )


def _validated_budget_scores(
    value: object,
    *,
    validation_count: int,
) -> tuple[tuple[int, tuple[float, ...], float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("generated epoch selection has no complete score rows")
    rows: list[tuple[int, tuple[float, ...], float, float]] = []
    seen: set[int] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("generated epoch selection score must be an object")
        epochs = raw.get("epochs")
        losses_raw = raw.get("losses")
        fingerprints = raw.get("fit_checkpoint_sha256")
        if (
            not isinstance(epochs, int)
            or isinstance(epochs, bool)
            or epochs <= 0
            or epochs in seen
            or not isinstance(losses_raw, list | tuple)
            or len(losses_raw) != validation_count
            or not isinstance(fingerprints, list)
            or len(fingerprints) != validation_count
        ):
            raise ValueError("generated epoch selection score row is incomplete")
        losses = tuple(float(item) for item in losses_raw)
        if not all(math.isfinite(item) for item in losses):
            raise ValueError("generated epoch selection losses must be finite")
        for fingerprint in fingerprints:
            _validate_sha256_value(fingerprint, name="fit_checkpoint_sha256")
        values = np.asarray(losses, dtype=np.float64)
        mean = float(np.mean(values))
        standard_error = (
            0.0
            if len(values) == 1
            else float(np.std(values, ddof=1) / np.sqrt(len(values)))
        )
        if not math.isclose(float(raw.get("mean", math.nan)), mean, abs_tol=1.0e-12):
            raise ValueError("generated epoch selection mean was not reproduced")
        if not math.isclose(
            float(raw.get("standard_error", math.nan)),
            standard_error,
            abs_tol=1.0e-12,
        ):
            raise ValueError("generated epoch selection standard error was not reproduced")
        seen.add(epochs)
        rows.append((epochs, losses, mean, standard_error))
    return tuple(sorted(rows))


def _one_standard_error_selection(
    scores: tuple[tuple[int, tuple[float, ...], float, float], ...],
) -> int:
    best_epochs, _, best_mean, best_standard_error = min(
        scores,
        key=lambda item: (item[2], item[0]),
    )
    del best_epochs
    threshold = best_mean + best_standard_error
    return min(epochs for epochs, _, mean, _ in scores if mean <= threshold)


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _require_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(payload: dict[str, object], name: str) -> str:
    return _validate_sha256_value(payload.get(name), name=name)


def _validate_sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256")
    return value.lower()
