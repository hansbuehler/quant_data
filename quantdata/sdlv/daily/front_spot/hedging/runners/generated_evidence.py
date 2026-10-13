"""Authenticate generated-model initializers and 2022 budget selections."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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
PRODUCTION_GATE_SCHEMA = "hedging-generated-production-gate-v3"
QUANT_REVIEW_SCHEMA = "hedging-generated-quant-review-v1"
GPU_READINESS_SCHEMA = "hedging-generated-gpu-readiness-v1"
BUDGET_SELECTION_OBSERVATION_COUNT = 126
PRODUCTION_SCOPES = frozenset(
    {
        "daily-public",
        "daily-rolling",
        "frozen-weekly-public",
        "frozen-weekly-rolling",
        "restarted-weekly-public",
        "restarted-weekly-rolling",
    }
)
_SCOPE_BINDING_FIELDS = frozenset(
    {
        "initializer_manifest_sha256",
        "initializer_checkpoint_sha256",
        "selection_source_manifest_sha256",
        "selection_source_checkpoint_sha256",
        "epoch_selection_sha256",
        "selected_epochs",
        "base_initializer_manifest_sha256",
        "base_checkpoint_sha256",
        "selection_base_initializer_manifest_sha256",
        "selection_base_checkpoint_sha256",
    }
)


@dataclass(frozen=True, kw_only=True)
class GeneratedSelectionEvidence:
    """Immutable checkpoint and epoch budget selected without evaluation data."""

    model: str
    initializer_checkpoint: Path
    initializer_checkpoint_sha256: str
    selection_source_manifest_sha256: str | None
    selection_source_checkpoint_sha256: str | None
    selected_epochs: int
    base_checkpoint: Path | None
    base_checkpoint_sha256: str | None
    base_initializer_manifest_sha256: str | None
    selection_base_initializer_manifest_sha256: str | None
    selection_base_checkpoint_sha256: str | None
    initializer_manifest_sha256: str
    epoch_selection_sha256: str


@dataclass(frozen=True, kw_only=True)
class GeneratedProductionGateEvidence:
    """Authenticated review and GPU-readiness approval for one model."""

    model: str
    reviewed_commit: str
    quant_review_path: Path
    quant_review_sha256: str
    gpu_readiness_path: Path
    gpu_readiness_sha256: str
    scope_bindings: dict[str, dict[str, object]]
    gate_sha256: str


@dataclass(frozen=True, kw_only=True)
class _CheckpointEvidence:
    model: str
    training_end_date: date
    completed_epochs: int
    base_checkpoint_sha256: str | None
    model_code_version: str
    model_user_version: str
    model_config_identity: str
    precision: str


def load_generated_production_gate(
    *,
    gate_path: Path,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
    expected_reviewed_commit: str,
) -> GeneratedProductionGateEvidence:
    """Authenticate the explicit production gate for one generated model."""
    path = Path(gate_path)
    payload, gate_sha256 = _read_json_object_with_sha256(path)
    normalized_model = _normalized_model(expected_model)
    if payload.get("schema") != PRODUCTION_GATE_SCHEMA:
        raise ValueError("unexpected generated production-gate schema")
    if _normalized_model(_require_string(payload, "model")) != normalized_model:
        raise ValueError("generated production gate uses another model")
    if payload.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated production gate uses another dataset")
    if payload.get("status") != "pass":
        raise ValueError("generated production gate has not passed")
    reviewed_commit = _require_hex_digest(payload, "reviewed_commit", lengths=(40,))
    if reviewed_commit != expected_reviewed_commit:
        raise ValueError("generated production gate reviews another commit")
    scope_bindings = _validated_scope_bindings(payload.get("scope_bindings"))
    quant_review_path, quant_review_sha256 = _authenticated_gate_artifact(
        payload,
        "quant_review",
        gate_path=path,
        expected_schema=QUANT_REVIEW_SCHEMA,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_reviewed_commit=reviewed_commit,
        expected_scope_bindings=scope_bindings,
    )
    gpu_readiness_path, gpu_readiness_sha256 = _authenticated_gate_artifact(
        payload,
        "gpu_readiness",
        gate_path=path,
        expected_schema=GPU_READINESS_SCHEMA,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_reviewed_commit=reviewed_commit,
        expected_scope_bindings=scope_bindings,
    )
    return GeneratedProductionGateEvidence(
        model=normalized_model,
        reviewed_commit=reviewed_commit,
        quant_review_path=quant_review_path,
        quant_review_sha256=quant_review_sha256,
        gpu_readiness_path=gpu_readiness_path,
        gpu_readiness_sha256=gpu_readiness_sha256,
        scope_bindings=scope_bindings,
        gate_sha256=gate_sha256,
    )


def reviewable_repository_commit(repo_root: Path | None = None) -> str:
    """Return HEAD only when tracked source is clean enough to authenticate."""
    root = (
        Path(__file__).resolve().parents[6]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    head = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reviewed_commit = _validate_hex_digest_value(
        head,
        name="repository HEAD",
        lengths=(40,),
    )
    status = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("tracked repository changes are not reviewable production code")
    return reviewed_commit


def generated_scope_binding(
    *,
    selection: GeneratedSelectionEvidence | None,
    base_initializer: GeneratedSelectionEvidence | None = None,
) -> dict[str, object]:
    """Return the exact pre-evaluation evidence required by one workflow."""
    if selection is None:
        initializer_manifest_sha256 = None
        initializer_checkpoint_sha256 = None
        epoch_selection_sha256 = None
        selected_epochs = None
        selection_source_manifest_sha256 = None
        selection_source_checkpoint_sha256 = None
        selection_base_manifest_sha256 = None
        selection_base_checkpoint_sha256 = None
    else:
        initializer_manifest_sha256 = selection.initializer_manifest_sha256
        initializer_checkpoint_sha256 = selection.initializer_checkpoint_sha256
        epoch_selection_sha256 = selection.epoch_selection_sha256 or None
        selected_epochs = selection.selected_epochs or None
        selection_source_manifest_sha256 = (
            selection.selection_source_manifest_sha256
        )
        selection_source_checkpoint_sha256 = (
            selection.selection_source_checkpoint_sha256
        )
        selection_base_manifest_sha256 = (
            selection.selection_base_initializer_manifest_sha256
        )
        selection_base_checkpoint_sha256 = (
            selection.selection_base_checkpoint_sha256
        )
    selection_production_base_manifest_sha256 = (
        None if selection is None else selection.base_initializer_manifest_sha256
    )
    selection_production_base_checkpoint_sha256 = (
        None if selection is None else selection.base_checkpoint_sha256
    )
    base_manifest_sha256 = (
        selection_production_base_manifest_sha256
        if base_initializer is None
        else base_initializer.initializer_manifest_sha256
    )
    base_checkpoint_sha256 = (
        selection_production_base_checkpoint_sha256
        if base_initializer is None
        else base_initializer.initializer_checkpoint_sha256
    )
    if base_initializer is not None and (
        selection_production_base_manifest_sha256 not in {None, base_manifest_sha256}
        or selection_production_base_checkpoint_sha256
        not in {None, base_checkpoint_sha256}
    ):
        raise ValueError("generated scope uses conflicting production bases")
    return {
        "initializer_manifest_sha256": initializer_manifest_sha256,
        "initializer_checkpoint_sha256": initializer_checkpoint_sha256,
        "selection_source_manifest_sha256": selection_source_manifest_sha256,
        "selection_source_checkpoint_sha256": selection_source_checkpoint_sha256,
        "epoch_selection_sha256": epoch_selection_sha256,
        "selected_epochs": selected_epochs,
        "base_initializer_manifest_sha256": base_manifest_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "selection_base_initializer_manifest_sha256": (
            selection_base_manifest_sha256
        ),
        "selection_base_checkpoint_sha256": selection_base_checkpoint_sha256,
    }


def require_generated_production_scope(
    gate: GeneratedProductionGateEvidence,
    *,
    scope: str,
    selection: GeneratedSelectionEvidence | None,
    base_initializer: GeneratedSelectionEvidence | None = None,
) -> None:
    """Require a gate for this exact workflow and pre-evaluation lineage."""
    if scope not in PRODUCTION_SCOPES:
        raise ValueError(f"unknown generated production scope: {scope}")
    actual = gate.scope_bindings.get(scope)
    if actual is None:
        raise ValueError(f"generated production gate does not cover {scope}")
    expected = generated_scope_binding(
        selection=selection,
        base_initializer=base_initializer,
    )
    if actual != expected:
        raise ValueError(f"generated production gate evidence differs for {scope}")


def load_generated_selection_evidence(
    *,
    initializer_manifest: Path,
    selection_source_initializer_manifest: Path,
    epoch_selection: Path,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
    base_initializer_manifest: Path | None = None,
    selection_base_initializer_manifest: Path | None = None,
) -> GeneratedSelectionEvidence:
    """Load and cross-check one pre-evaluation generated-model selection."""
    initializer_path = Path(initializer_manifest)
    selection_path = Path(epoch_selection)
    initializer, initializer_manifest_sha256 = _read_json_object_with_sha256(
        initializer_path
    )
    selection, epoch_selection_sha256 = _read_json_object_with_sha256(
        selection_path
    )
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

    selection_source = load_generated_initializer_evidence(
        initializer_manifest=selection_source_initializer_manifest,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )
    selection_checkpoint = selection_source.initializer_checkpoint
    selection_checkpoint_sha256 = (
        selection_source.initializer_checkpoint_sha256
    )
    selection_checkpoint_evidence = _load_checkpoint_evidence(
        selection_checkpoint,
        expected_model=normalized_model,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )
    if _checkpoint_model_identity(selection_checkpoint_evidence) != (
        _checkpoint_model_identity(checkpoint_evidence)
    ):
        raise ValueError(
            "generated epoch-selection source uses another model geometry"
        )

    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected generated epoch-selection schema")
    if _normalized_model(_require_string(selection, "model")) != normalized_model:
        raise ValueError("generated epoch selection model does not match")
    if selection.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("generated epoch selection uses another dataset")
    selected_epochs = selection.get("selected_epochs")
    if not isinstance(selected_epochs, int) or selected_epochs <= 0:
        raise ValueError("generated epoch selection has no positive selected budget")
    if Path(_require_string(selection, "warm_start_checkpoint")) != selection_checkpoint:
        raise ValueError("generated epoch selection uses another source initializer")
    if _require_sha256(selection, "warm_start_sha256") != selection_checkpoint_sha256:
        raise ValueError("generated epoch selection uses another source hash")
    claimed_source_end_date = date.fromisoformat(
        _require_string(selection, "warm_start_training_end_date")
    )
    if claimed_source_end_date != selection_checkpoint_evidence.training_end_date:
        raise ValueError("generated selection source date differs from checkpoint")
    if selection.get("one_standard_error_rule") is not True:
        raise ValueError("generated epoch selection did not use one-standard-error rule")
    if selection.get("observation_count") != BUDGET_SELECTION_OBSERVATION_COUNT:
        raise ValueError(
            "generated epoch selection does not use the precommitted "
            f"{BUDGET_SELECTION_OBSERVATION_COUNT}-observation window"
        )
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
        or selection_checkpoint_evidence.training_end_date >= validation_origins[0]
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
    base_initializer_manifest_sha256 = None
    selection_base_initializer_manifest_sha256 = None
    selection_base_checkpoint_sha256 = None
    selection_base = None
    if base_initializer_manifest is not None:
        base = load_generated_initializer_evidence(
            initializer_manifest=base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        base_checkpoint = base.initializer_checkpoint
        base_checkpoint_sha256 = base.initializer_checkpoint_sha256
        base_initializer_manifest_sha256 = base.initializer_manifest_sha256
    if selection_base_initializer_manifest is not None:
        selection_base = load_generated_initializer_evidence(
            initializer_manifest=selection_base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        selection_base_initializer_manifest_sha256 = (
            selection_base.initializer_manifest_sha256
        )
        selection_base_checkpoint_sha256 = (
            selection_base.initializer_checkpoint_sha256
        )
    if normalized_model == "diffusion-ad-seq-vol-ft":
        initializer_base_path = initializer.get("base_checkpoint_path")
        initializer_base_sha256 = initializer.get("base_checkpoint_sha256")
        selection_source_manifest = _read_json_object(
            Path(selection_source_initializer_manifest)
        )
        selection_source_base_path = selection_source_manifest.get(
            "base_checkpoint_path"
        )
        selection_source_base_sha256 = selection_source_manifest.get(
            "base_checkpoint_sha256"
        )
        if (
            base_checkpoint is None
            or selection_base is None
            or initializer_base_path != str(base_checkpoint)
            or initializer_base_sha256 != base_checkpoint_sha256
            or checkpoint_evidence.base_checkpoint_sha256
            != base_checkpoint_sha256
            or selection_source_base_path
            != str(selection_base.initializer_checkpoint)
            or selection_source_base_sha256
            != selection_base_checkpoint_sha256
            or selected_base_sha256 != selection_base_checkpoint_sha256
            or selection_checkpoint_evidence.base_checkpoint_sha256
            != selection_base_checkpoint_sha256
        ):
            raise ValueError("fine-tuned generated selection has another base checkpoint")
        selection_base_checkpoint_evidence = _load_checkpoint_evidence(
            selection_base.initializer_checkpoint,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        base_checkpoint_evidence = _load_checkpoint_evidence(
            base_checkpoint,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        if _checkpoint_model_identity(selection_base_checkpoint_evidence) != (
            _checkpoint_model_identity(base_checkpoint_evidence)
        ):
            raise ValueError(
                "fine-tuned selection and production bases use another geometry"
            )
    elif (
        selected_base_sha256 is not None
        or base_checkpoint is not None
        or selection_base is not None
        or checkpoint_evidence.base_checkpoint_sha256 is not None
        or selection_checkpoint_evidence.base_checkpoint_sha256 is not None
    ):
        raise ValueError("non-fine-tuned generated selection cannot use a base checkpoint")

    return GeneratedSelectionEvidence(
        model=normalized_model,
        initializer_checkpoint=checkpoint,
        initializer_checkpoint_sha256=checkpoint_sha256,
        selection_source_manifest_sha256=(
            selection_source.initializer_manifest_sha256
        ),
        selection_source_checkpoint_sha256=selection_checkpoint_sha256,
        selected_epochs=selected_epochs,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_initializer_manifest_sha256=base_initializer_manifest_sha256,
        selection_base_initializer_manifest_sha256=(
            selection_base_initializer_manifest_sha256
        ),
        selection_base_checkpoint_sha256=selection_base_checkpoint_sha256,
        initializer_manifest_sha256=initializer_manifest_sha256,
        epoch_selection_sha256=epoch_selection_sha256,
    )


def load_generated_initializer_evidence(
    *,
    initializer_manifest: Path,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
    expected_base_initializer_manifest: Path | None = None,
) -> GeneratedSelectionEvidence:
    """Authenticate an initializer when no rolling epoch budget is required."""
    path = Path(initializer_manifest)
    payload, initializer_manifest_sha256 = _read_json_object_with_sha256(path)
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
    base_checkpoint = None
    base_checkpoint_sha256 = None
    base_initializer_manifest_sha256 = None
    if expected_base_initializer_manifest is not None:
        if normalized_model != "diffusion-ad-seq-vol-ft":
            raise ValueError("only a fine-tuned initializer can use an expected base")
        base = load_generated_initializer_evidence(
            initializer_manifest=expected_base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        )
        base_checkpoint = base.initializer_checkpoint
        base_checkpoint_sha256 = base.initializer_checkpoint_sha256
        base_initializer_manifest_sha256 = base.initializer_manifest_sha256
        if (
            payload.get("base_checkpoint_path") != str(base_checkpoint)
            or payload.get("base_checkpoint_sha256") != base_checkpoint_sha256
            or checkpoint_evidence.base_checkpoint_sha256
            != base_checkpoint_sha256
        ):
            raise ValueError("fine-tuned generated initializer has another base")
    return GeneratedSelectionEvidence(
        model=normalized_model,
        initializer_checkpoint=checkpoint,
        initializer_checkpoint_sha256=checkpoint_sha256,
        selection_source_manifest_sha256=None,
        selection_source_checkpoint_sha256=None,
        selected_epochs=0,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_initializer_manifest_sha256=base_initializer_manifest_sha256,
        selection_base_initializer_manifest_sha256=None,
        selection_base_checkpoint_sha256=None,
        initializer_manifest_sha256=initializer_manifest_sha256,
        epoch_selection_sha256="",
    )


def _normalized_model(model: str) -> str:
    return model if model == "volgan" or model.startswith("diffusion-") else f"diffusion-{model}"


def _checkpoint_model_identity(
    evidence: _CheckpointEvidence,
) -> tuple[str, str, str, str, str]:
    return (
        evidence.model,
        evidence.model_code_version,
        evidence.model_user_version,
        evidence.model_config_identity,
        evidence.precision,
    )


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
    model_code_version = _require_string(payload, "model_code_version")
    model_user_version = _require_string(payload, "model_user_version")
    model_config_identity = _require_string(payload, "model_config_identity")
    precision = _require_string(payload, "precision")
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
        model_code_version=model_code_version,
        model_user_version=model_user_version,
        model_config_identity=model_config_identity,
        precision=precision,
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
    payload, _ = _read_json_object_with_sha256(path)
    return payload


def _read_json_object_with_sha256(path: Path) -> tuple[dict[str, object], str]:
    value = Path(path).read_bytes()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload, hashlib.sha256(value).hexdigest()


def _require_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(payload: dict[str, object], name: str) -> str:
    return _validate_sha256_value(payload.get(name), name=name)


def _authenticated_gate_artifact(
    payload: dict[str, object],
    name: str,
    *,
    gate_path: Path,
    expected_schema: str,
    expected_model: str,
    expected_dataset_manifest_sha256: str,
    expected_reviewed_commit: str,
    expected_scope_bindings: dict[str, dict[str, object]],
) -> tuple[Path, str]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"generated production gate has no {name} artifact")
    artifact_path = Path(_require_string(value, "path"))
    if not artifact_path.is_absolute():
        artifact_path = gate_path.parent / artifact_path
    artifact_sha256 = _require_sha256(value, "sha256")
    artifact, actual_artifact_sha256 = _read_json_object_with_sha256(artifact_path)
    if actual_artifact_sha256 != artifact_sha256:
        raise ValueError(f"generated production gate {name} hash mismatch")
    if artifact.get("schema") != expected_schema:
        raise ValueError(f"generated production gate {name} schema is incompatible")
    if artifact.get("status") != "pass":
        raise ValueError(f"generated production gate {name} has not passed")
    if _normalized_model(_require_string(artifact, "model")) != expected_model:
        raise ValueError(f"generated production gate {name} uses another model")
    if artifact.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError(f"generated production gate {name} uses another dataset")
    reviewed_commit = _require_hex_digest(
        artifact,
        "reviewed_commit",
        lengths=(40,),
    )
    if reviewed_commit != expected_reviewed_commit:
        raise ValueError(f"generated production gate {name} reviews another commit")
    artifact_bindings = _validated_scope_bindings(
        artifact.get("scope_bindings")
    )
    if artifact_bindings != expected_scope_bindings:
        raise ValueError(f"generated production gate {name} covers other evidence")
    return artifact_path, artifact_sha256


def _validated_scope_bindings(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("generated production gate has no workflow scopes")
    bindings: dict[str, dict[str, object]] = {}
    for raw_scope, raw_binding in value.items():
        if not isinstance(raw_scope, str) or raw_scope not in PRODUCTION_SCOPES:
            raise ValueError(f"unknown generated production scope: {raw_scope}")
        if not isinstance(raw_binding, dict):
            raise ValueError(f"generated production scope {raw_scope} is not an object")
        if set(raw_binding) != _SCOPE_BINDING_FIELDS:
            raise ValueError(
                f"generated production scope {raw_scope} has incomplete bindings"
            )
        binding: dict[str, object] = {}
        for name in _SCOPE_BINDING_FIELDS - {"selected_epochs"}:
            item = raw_binding.get(name)
            binding[name] = (
                None
                if item is None
                else _validate_sha256_value(item, name=f"{raw_scope}.{name}")
            )
        selected_epochs = raw_binding.get("selected_epochs")
        if selected_epochs is not None and (
            not isinstance(selected_epochs, int)
            or isinstance(selected_epochs, bool)
            or selected_epochs <= 0
        ):
            raise ValueError(f"{raw_scope}.selected_epochs must be positive")
        binding["selected_epochs"] = selected_epochs
        selection_values = tuple(
            binding[name]
            for name in (
                "initializer_manifest_sha256",
                "initializer_checkpoint_sha256",
                "selection_source_manifest_sha256",
                "selection_source_checkpoint_sha256",
                "epoch_selection_sha256",
                "selected_epochs",
            )
        )
        if any(item is None for item in selection_values) and any(
            item is not None for item in selection_values
        ):
            raise ValueError(
                f"generated production scope {raw_scope} has partial selection evidence"
            )
        base_values = tuple(
            binding[name]
            for name in (
                "base_initializer_manifest_sha256",
                "base_checkpoint_sha256",
            )
        )
        if any(item is None for item in base_values) and any(
            item is not None for item in base_values
        ):
            raise ValueError(
                f"generated production scope {raw_scope} has partial base evidence"
            )
        selection_base_values = tuple(
            binding[name]
            for name in (
                "selection_base_initializer_manifest_sha256",
                "selection_base_checkpoint_sha256",
            )
        )
        if any(item is None for item in selection_base_values) and any(
            item is not None for item in selection_base_values
        ):
            raise ValueError(
                f"generated production scope {raw_scope} has partial selection-base evidence"
            )
        bindings[raw_scope] = binding
    return bindings


def _require_hex_digest(
    payload: dict[str, object],
    name: str,
    *,
    lengths: tuple[int, ...],
) -> str:
    return _validate_hex_digest_value(
        payload.get(name),
        name=name,
        lengths=lengths,
    )


def _validate_hex_digest_value(
    value: object,
    *,
    name: str,
    lengths: tuple[int, ...],
) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a hexadecimal digest")
    return value.lower()


def _validate_sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a SHA-256")
    return value.lower()
