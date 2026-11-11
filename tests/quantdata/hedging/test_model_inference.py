"""Inference retention must preserve exact EMA weights and authenticate every file."""

import json
from pathlib import Path

import pytest

from quantdata.hedging.model_audit.inference import (
    export_checkpoint,
    fingerprint,
    load_export,
    model_from_export,
    tensor_identity,
)
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import GenerativeSurfaceGrid
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.torch import BaseEnvironment, torch


def _checkpoint(directory: Path, *, day: str = "2024-04-01") -> tuple[Path, DiffusionModelConfig, dict]:
    config = DiffusionModelConfig(label="test-inference", grid=GenerativeSurfaceGrid.short_end_nms(),
        variant=DiffusionVariant.AD_SEQ_VOL_FT, model_dimension=8, transformer_layers=1, attention_heads=2)
    network = SequentialVolatilityDenoiser(BaseEnvironment(precision=32, device="cpu"), config=config,
                                          channel_mean=torch.zeros(2), channel_scale=torch.ones(2))
    state = network.state_dict()
    checkpoint = directory / f"source-{day}.pt"
    torch.save({"ema_model": state, "model": {name: tensor + 1 for name, tensor in state.items()},
        "model_config_identity": config.filename, "model_code_version": config.version.unique_id48,
        "model_user_version": config.user_version, "training_end_date": day,
        "optimizer": {"unused": torch.ones(10)}}, checkpoint)
    return checkpoint, config, state


def test_tensor_identity_includes_dtype_and_names() -> None:
    assert tensor_identity({"first": torch.ones(2)}) != tensor_identity({"other": torch.ones(2)})
    assert tensor_identity({"first": torch.ones(2)}) != tensor_identity({"first": torch.ones(2, dtype=torch.float64)})


def test_export_preserves_ema_and_drops_optimizer(tmp_path: Path) -> None:
    source, config, original = _checkpoint(tmp_path)
    entry = export_checkpoint(source, expected_sha256=fingerprint(source), config=config, directory=tmp_path / "inference")
    manifest, tensors = load_export(Path(entry["manifest"]), expected_sha256=entry["manifest_sha256"])
    assert "optimizer" not in manifest["metadata"]
    for name, tensor in tensors.items():
        assert torch.equal(tensor, original[name])
    restored = model_from_export(Path(entry["manifest"]), environment=BaseEnvironment(precision=32, device="cpu"))
    assert all(buffer.device.type == "cpu" for buffer in restored.buffers())
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert source.exists()


def test_export_reuses_identical_frozen_backbone(tmp_path: Path) -> None:
    source, config, original = _checkpoint(tmp_path)
    first = export_checkpoint(source, expected_sha256=fingerprint(source), config=config, directory=tmp_path / "inference")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    saved["training_end_date"] = "2024-04-02"
    saved["ema_model"]["output_adapter.up.weight"] += 1.
    second_source = tmp_path / "second.pt"
    torch.save(saved, second_source)
    second = export_checkpoint(second_source, expected_sha256=fingerprint(second_source), config=config, directory=tmp_path / "inference")
    before = json.loads(Path(first["manifest"]).read_text())
    after = json.loads(Path(second["manifest"]).read_text())
    assert before["backbone"] == after["backbone"]
    assert before["adapter"]["path"] != after["adapter"]["path"]


def test_corrupt_export_is_rejected(tmp_path: Path) -> None:
    source, config, original = _checkpoint(tmp_path)
    entry = export_checkpoint(source, expected_sha256=fingerprint(source), config=config, directory=tmp_path / "inference")
    manifest_path = Path(entry["manifest"])
    manifest = json.loads(manifest_path.read_text())
    with Path(manifest["backbone"]["path"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="blob checksum"):
        load_export(manifest_path, expected_sha256=entry["manifest_sha256"])


def test_export_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source, config, original = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="checksum changed"):
        export_checkpoint(source, expected_sha256="wrong", config=config, directory=tmp_path / "inference")
