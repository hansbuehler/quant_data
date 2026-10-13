from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.runners.build_generated_production_gate import (
    generated_scope_bindings,
    write_generated_production_gate,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.generated_evidence import (
    GPU_READINESS_SCHEMA,
    QUANT_REVIEW_SCHEMA,
    GeneratedSelectionEvidence,
)


def _selection(*, model: str, stem: str) -> GeneratedSelectionEvidence:
    return GeneratedSelectionEvidence(
        model=model,
        initializer_checkpoint=Path(f"/{stem}.pt"),
        initializer_checkpoint_sha256="1" * 64,
        selection_source_manifest_sha256="2" * 64,
        selection_source_checkpoint_sha256="3" * 64,
        selected_epochs=30,
        base_checkpoint=None,
        base_checkpoint_sha256=None,
        base_initializer_manifest_sha256=None,
        selection_base_initializer_manifest_sha256=None,
        selection_base_checkpoint_sha256=None,
        initializer_manifest_sha256="4" * 64,
        epoch_selection_sha256="5" * 64,
    )


def test_scope_bindings_distinguish_public_and_rolling() -> None:
    selection = _selection(model="diffusion-ad-seq-vol", stem="diffusion")
    bindings = generated_scope_bindings(
        scopes=("daily-public", "daily-rolling"),
        selection=selection,
    )

    assert bindings["daily-public"]["selected_epochs"] is None
    assert bindings["daily-public"]["initializer_checkpoint_sha256"] is None
    assert bindings["daily-rolling"]["selected_epochs"] == 30
    assert bindings["daily-rolling"]["initializer_checkpoint_sha256"] == "1" * 64


def test_scope_bindings_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        generated_scope_bindings(
            scopes=("daily-public", "daily-public"),
            selection=_selection(model="volgan", stem="volgan"),
        )


def test_scope_bindings_pin_fine_tuning_bases() -> None:
    base = _selection(model="diffusion-ad-seq-vol", stem="diffusion")
    fine_tuned = replace(
        _selection(model="diffusion-ad-seq-vol-ft", stem="diffusion-ft"),
        base_checkpoint=base.initializer_checkpoint,
        base_checkpoint_sha256=base.initializer_checkpoint_sha256,
        base_initializer_manifest_sha256=base.initializer_manifest_sha256,
        selection_base_initializer_manifest_sha256="7" * 64,
        selection_base_checkpoint_sha256="8" * 64,
    )

    bindings = generated_scope_bindings(
        scopes=("frozen-weekly-public", "frozen-weekly-rolling"),
        selection=fine_tuned,
        base_initializer=base,
    )

    public = bindings["frozen-weekly-public"]
    rolling = bindings["frozen-weekly-rolling"]
    assert public["initializer_checkpoint_sha256"] is None
    assert public["base_checkpoint_sha256"] == base.initializer_checkpoint_sha256
    assert rolling["initializer_checkpoint_sha256"] == "1" * 64
    assert rolling["base_checkpoint_sha256"] == base.initializer_checkpoint_sha256
    assert rolling["selection_base_checkpoint_sha256"] == "8" * 64


def test_gate_writer_authenticates_review_and_readiness(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public", "daily-rolling"),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )
    output = tmp_path / "production-gate.json"

    gate = write_generated_production_gate(
        output=output,
        model=model,
        dataset_manifest_sha256_value=dataset_sha256,
        reviewed_commit=commit,
        scope_bindings=bindings,
        quant_review=review,
        gpu_readiness=readiness,
    )

    assert gate.gate_sha256 == file_sha256(output)
    assert set(gate.scope_bindings) == {"daily-public", "daily-rolling"}


def test_gate_writer_rejects_stale_readiness(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps(
            {
                "schema": GPU_READINESS_SCHEMA,
                **common,
                "reviewed_commit": "b" * 40,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviews another commit"):
        write_generated_production_gate(
            output=tmp_path / "production-gate.json",
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
        )


def test_gate_writer_preserves_aliased_review(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )
    original_review = review.read_bytes()

    with pytest.raises(ValueError, match="output aliases quant review"):
        write_generated_production_gate(
            output=review,
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
        )

    assert review.read_bytes() == original_review


def test_gate_writer_rejects_hardlink_alias(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    output = tmp_path / "production-gate.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )
    os.link(review, output)
    original_review = review.read_bytes()

    with pytest.raises(ValueError, match="output aliases quant review"):
        write_generated_production_gate(
            output=output,
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
        )

    assert review.read_bytes() == original_review


def test_gate_writer_preserves_protected_lineage_input(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    lineage = tmp_path / "initializer-manifest.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )
    lineage.write_text('{"immutable": true}\n', encoding="utf-8")
    original_lineage = lineage.read_bytes()

    with pytest.raises(ValueError, match="output aliases initializer manifest"):
        write_generated_production_gate(
            output=lineage,
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
            protected_inputs={"initializer manifest": lineage},
        )

    assert lineage.read_bytes() == original_lineage


def test_gate_writer_preserves_review_at_lock_path(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    output = tmp_path / "production-gate.json"
    review = tmp_path / "production-gate.json.lock"
    readiness = tmp_path / "gpu-readiness.json"
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )
    original_review = review.read_bytes()

    with pytest.raises(ValueError, match="lock aliases quant review"):
        write_generated_production_gate(
            output=output,
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
        )

    assert review.read_bytes() == original_review


def test_gate_writer_rejects_output_inside_dataset(tmp_path: Path) -> None:
    model = "volgan"
    dataset_sha256 = "6" * 64
    commit = "a" * 40
    bindings = generated_scope_bindings(
        scopes=("daily-public",),
        selection=_selection(model=model, stem="volgan"),
    )
    common = {
        "model": model,
        "dataset_manifest_sha256": dataset_sha256,
        "status": "pass",
        "reviewed_commit": commit,
        "scope_bindings": bindings,
    }
    review = tmp_path / "quant-review.json"
    readiness = tmp_path / "gpu-readiness.json"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    review.write_text(
        json.dumps({"schema": QUANT_REVIEW_SCHEMA, **common}),
        encoding="utf-8",
    )
    readiness.write_text(
        json.dumps({"schema": GPU_READINESS_SCHEMA, **common}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output lies inside public dataset"):
        write_generated_production_gate(
            output=dataset / "payload.npz",
            model=model,
            dataset_manifest_sha256_value=dataset_sha256,
            reviewed_commit=commit,
            scope_bindings=bindings,
            quant_review=review,
            gpu_readiness=readiness,
            forbidden_directories={"public dataset": dataset},
        )
