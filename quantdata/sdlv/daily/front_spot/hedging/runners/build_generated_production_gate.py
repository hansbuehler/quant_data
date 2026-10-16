"""Assemble an authenticated generated-model production gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    PRODUCTION_GATE_SCHEMA,
    PRODUCTION_READY_STATUSES,
    PRODUCTION_SCOPES,
    GeneratedProductionGateEvidence,
    GeneratedSelectionEvidence,
    generated_scope_binding,
    load_generated_initializer_evidence,
    load_generated_production_gate,
    load_generated_selection_evidence,
    reviewable_repository_commit,
)


def generated_scope_bindings(
    *,
    scopes: tuple[str, ...],
    selection: GeneratedSelectionEvidence,
    base_initializer: GeneratedSelectionEvidence | None = None,
) -> dict[str, dict[str, object]]:
    """Bind public and rolling scopes to their exact pre-evaluation evidence."""
    if not scopes:
        raise ValueError("at least one generated production scope is required")
    if len(set(scopes)) != len(scopes):
        raise ValueError("generated production scopes must be unique")
    unknown = set(scopes) - PRODUCTION_SCOPES
    if unknown:
        raise ValueError(f"unknown generated production scopes: {sorted(unknown)}")
    return {
        scope: generated_scope_binding(
            selection=selection if scope.endswith("-rolling") else None,
            base_initializer=base_initializer,
        )
        for scope in scopes
    }


def write_generated_production_gate(
    *,
    output: Path,
    model: str,
    dataset_manifest_sha256_value: str,
    reviewed_commit: str,
    scope_bindings: dict[str, dict[str, object]],
    quant_review: Path,
    gpu_readiness: Path,
    protected_inputs: dict[str, Path] | None = None,
    forbidden_directories: dict[str, Path] | None = None,
) -> GeneratedProductionGateEvidence:
    """Write and re-authenticate one production gate atomically."""
    destination = Path(output).resolve()
    review_path = Path(quant_review).resolve()
    readiness_path = Path(gpu_readiness).resolve()
    protected = dict(protected_inputs or {})
    protected.update(
        {
            "quant review": review_path,
            "GPU readiness": readiness_path,
        }
    )
    forbidden = dict(forbidden_directories or {})
    _require_distinct_output(destination, protected)
    _require_outside_directories(destination, forbidden)
    readiness_payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not isinstance(readiness_payload, dict):
        raise ValueError("GPU readiness must be a JSON object")
    readiness_status = readiness_payload.get("status")
    if readiness_status not in PRODUCTION_READY_STATUSES:
        raise ValueError("GPU readiness is not production-ready")
    readiness_warnings = readiness_payload.get("warnings", {})
    payload = {
        "schema": PRODUCTION_GATE_SCHEMA,
        "model": model,
        "dataset_manifest_sha256": dataset_manifest_sha256_value,
        "status": readiness_status,
        "warnings": readiness_warnings,
        "reviewed_commit": reviewed_commit,
        "scope_bindings": scope_bindings,
        "quant_review": {
            "path": str(review_path),
            "sha256": file_sha256(review_path),
        },
        "gpu_readiness": {
            "path": str(readiness_path),
            "sha256": file_sha256(readiness_path),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")
    _require_distinct_output(
        lock_path,
        {"production gate output": destination, **protected},
        candidate_label="production gate lock",
    )
    _require_outside_directories(
        lock_path,
        forbidden,
        candidate_label="production gate lock",
    )
    lock = FileLock(str(lock_path))
    with lock:
        _require_distinct_output(destination, protected)
        _require_outside_directories(destination, forbidden)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            load_generated_production_gate(
                gate_path=temporary,
                expected_model=model,
                expected_dataset_manifest_sha256=dataset_manifest_sha256_value,
                expected_reviewed_commit=reviewed_commit,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return load_generated_production_gate(
        gate_path=destination,
        expected_model=model,
        expected_dataset_manifest_sha256=dataset_manifest_sha256_value,
        expected_reviewed_commit=reviewed_commit,
    )


def main() -> None:
    """Load reviewed evidence and assemble an immutable production gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "volgan",
            "diffusion-ad-seq-vol",
            "diffusion-ad-seq-vol-ft",
        ),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--initializer-manifest", type=Path, required=True)
    parser.add_argument("--selection-source-manifest", type=Path, required=True)
    parser.add_argument("--epoch-selection", type=Path, required=True)
    parser.add_argument("--base-initializer-manifest", type=Path)
    parser.add_argument("--selection-base-initializer-manifest", type=Path)
    parser.add_argument(
        "--scope",
        action="append",
        choices=tuple(sorted(PRODUCTION_SCOPES)),
        required=True,
    )
    parser.add_argument("--quant-review", type=Path, required=True)
    parser.add_argument("--gpu-readiness", type=Path, required=True)
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    is_fine_tuned = arguments.model == "diffusion-ad-seq-vol-ft"
    if is_fine_tuned and (
        arguments.base_initializer_manifest is None
        or arguments.selection_base_initializer_manifest is None
    ):
        parser.error(
            "fine-tuned diffusion requires production and selection base manifests"
        )
    if not is_fine_tuned and (
        arguments.base_initializer_manifest is not None
        or arguments.selection_base_initializer_manifest is not None
    ):
        parser.error("only fine-tuned diffusion accepts base manifests")

    panel = PublicHedgingPanel(arguments.dataset, validate=True)
    dataset_sha256 = dataset_manifest_sha256(panel)
    selection_source = load_generated_initializer_evidence(
        initializer_manifest=arguments.selection_source_manifest,
        expected_model=arguments.model,
        expected_dataset_manifest_sha256=dataset_sha256,
        expected_base_initializer_manifest=(
            arguments.selection_base_initializer_manifest
        ),
    )
    selection = load_generated_selection_evidence(
        initializer_manifest=arguments.initializer_manifest,
        selection_source_initializer_manifest=arguments.selection_source_manifest,
        epoch_selection=arguments.epoch_selection,
        expected_model=arguments.model,
        expected_dataset_manifest_sha256=dataset_sha256,
        base_initializer_manifest=arguments.base_initializer_manifest,
        selection_base_initializer_manifest=(
            arguments.selection_base_initializer_manifest
        ),
    )
    base_initializer = None
    selection_base_initializer = None
    if arguments.base_initializer_manifest is not None:
        base_initializer = load_generated_initializer_evidence(
            initializer_manifest=arguments.base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=dataset_sha256,
        )
        selection_base_initializer = load_generated_initializer_evidence(
            initializer_manifest=arguments.selection_base_initializer_manifest,
            expected_model="diffusion-ad-seq-vol",
            expected_dataset_manifest_sha256=dataset_sha256,
        )
    protected_inputs = {
        "dataset manifest": Path(arguments.dataset) / "manifest.json",
        "initializer manifest": arguments.initializer_manifest,
        "initializer checkpoint": selection.initializer_checkpoint,
        "selection-source manifest": arguments.selection_source_manifest,
        "selection-source checkpoint": selection_source.initializer_checkpoint,
        "epoch selection": arguments.epoch_selection,
        "quant review": arguments.quant_review,
        "GPU readiness": arguments.gpu_readiness,
    }
    if arguments.base_initializer_manifest is not None:
        assert base_initializer is not None
        assert selection_base_initializer is not None
        protected_inputs.update(
            {
                "base initializer manifest": arguments.base_initializer_manifest,
                "base initializer checkpoint": (
                    base_initializer.initializer_checkpoint
                ),
                "selection-base initializer manifest": (
                    arguments.selection_base_initializer_manifest
                ),
                "selection-base initializer checkpoint": (
                    selection_base_initializer.initializer_checkpoint
                ),
            }
        )
    _require_distinct_output(Path(arguments.output), protected_inputs)
    scopes = tuple(arguments.scope)
    bindings = generated_scope_bindings(
        scopes=scopes,
        selection=selection,
        base_initializer=base_initializer,
    )
    reviewed_commit = arguments.reviewed_commit or reviewable_repository_commit()
    gate = write_generated_production_gate(
        output=arguments.output,
        model=arguments.model,
        dataset_manifest_sha256_value=dataset_sha256,
        reviewed_commit=reviewed_commit,
        scope_bindings=bindings,
        quant_review=arguments.quant_review,
        gpu_readiness=arguments.gpu_readiness,
        protected_inputs=protected_inputs,
        forbidden_directories={"public dataset": arguments.dataset},
    )
    print(
        json.dumps(
            {
                "gate": str(Path(arguments.output).resolve()),
                "gate_sha256": gate.gate_sha256,
                "model": gate.model,
                "status": gate.status,
                "warnings": gate.warnings,
                "reviewed_commit": gate.reviewed_commit,
                "scopes": sorted(gate.scope_bindings),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _require_distinct_output(
    output: Path,
    inputs: dict[str, Path],
    *,
    candidate_label: str = "production gate output",
) -> None:
    destination = Path(output).resolve()
    for label, raw_path in inputs.items():
        source = Path(raw_path).resolve()
        aliases = source == destination
        if not aliases and source.exists() and destination.exists():
            aliases = source.samefile(destination)
        if aliases:
            raise ValueError(f"{candidate_label} aliases {label}: {source}")


def _require_outside_directories(
    output: Path,
    directories: dict[str, Path],
    *,
    candidate_label: str = "production gate output",
) -> None:
    destination = Path(output).resolve()
    for label, raw_directory in directories.items():
        directory = Path(raw_directory).resolve()
        if destination == directory or destination.is_relative_to(directory):
            raise ValueError(f"{candidate_label} lies inside {label}: {directory}")


if __name__ == "__main__":
    main()
