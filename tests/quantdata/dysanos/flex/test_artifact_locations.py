from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quantdata.dysanos.flex.pipeline_monitor import _resolve_completion_artifact
from quantdata.dysanos.flex.run_joint_hierarchical_q1 import (
    _resolve_training_artifact,
)


def test_training_artifact_rebases_with_unchanged_digest(tmp_path: Path) -> None:
    training_root = tmp_path / "new-training-root"
    migrated = (
        training_root
        / "flex-shared-strikes"
        / "joint-hierarchical"
        / "v1"
        / "checkpoints"
        / "model.pt"
    )
    migrated.parent.mkdir(parents=True)
    content = b"portable-checkpoint-content"
    migrated.write_bytes(content)
    recorded = Path(
        "/old-machine/training/flex-shared-strikes/joint-hierarchical/"
        "v1/checkpoints/model.pt"
    )

    resolved = _resolve_training_artifact(
        recorded,
        hashlib.sha256(content).hexdigest(),
        training_root=training_root,
    )

    assert resolved == migrated


def test_training_artifact_rebase_rejects_digest_change(tmp_path: Path) -> None:
    training_root = tmp_path / "new-training-root"
    migrated = training_root / "flex-shared-strikes" / "model.pt"
    migrated.parent.mkdir(parents=True)
    migrated.write_bytes(b"wrong-content")

    with pytest.raises(ValueError, match="checksum-valid migrated"):
        _resolve_training_artifact(
            "/old-machine/training/flex-shared-strikes/model.pt",
            hashlib.sha256(b"expected-content").hexdigest(),
            training_root=training_root,
        )


def test_completion_artifact_rebases_with_unchanged_digest(tmp_path: Path) -> None:
    training_root = tmp_path / "new-training-root"
    migrated = training_root / "flex-shared-strikes" / "reports" / "proof.json"
    migrated.parent.mkdir(parents=True)
    content = b'{"status":"complete"}\n'
    migrated.write_bytes(content)

    resolved = _resolve_completion_artifact(
        "/old-machine/training/flex-shared-strikes/reports/proof.json",
        hashlib.sha256(content).hexdigest(),
        training_root=training_root,
    )

    assert resolved == migrated
