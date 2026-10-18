"""Regression checks for strict DYSANOS 1.1 temporal split helpers."""

from datetime import date

import pytest
import torch

from quantdata.dysanos.flex.training import FlexTrainConfig
from quantdata.dysanos.v11.run_strict_pre2025_direct import (
    _continuation_checkpoint,
    _decoder_train_config_for_checkpoint,
    _train_count,
)


def test_train_count_accepts_contiguous_2025_suffix() -> None:
    assert _train_count(
        (date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2))
    ) == 2


def test_train_count_rejects_noncontiguous_suffix() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _train_count((date(2024, 12, 31), date(2025, 1, 2), date(2024, 12, 30)))


def test_continuation_retains_weights_but_resets_optimizer(tmp_path) -> None:
    source = tmp_path / "source.pt"
    target = tmp_path / "continuation.pt"
    torch.save(
        {
            "phase": "market",
            "model_state_dict": {"weight": torch.tensor([1.25])},
            "training_state": {
                "completed_epochs": 600,
                "resume_signature": {"num_epochs": 600},
                "optimizer_state_dicts": [{"step": 600}],
            },
            "training_audits": {"supervised": {"converged": True}},
            "market_history": [{"epoch": 599}],
        },
        source,
    )
    assert _continuation_checkpoint(source, target) == target
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["phase"] == "supervised_complete"
    assert payload["training_state"] is None
    assert payload["market_history"] == []
    assert torch.equal(payload["model_state_dict"]["weight"], torch.tensor([1.25]))
    assert _continuation_checkpoint(source, target) == target


def test_continuation_rejects_incomplete_prior_schedule(tmp_path) -> None:
    source = tmp_path / "source.pt"
    torch.save(
        {
            "phase": "market",
            "training_state": {
                "completed_epochs": 599,
                "resume_signature": {"num_epochs": 600},
            },
        },
        source,
    )
    with pytest.raises(ValueError, match="completed prior schedule"):
        _continuation_checkpoint(source, tmp_path / "continuation.pt")


def test_continuation_accepts_absent_training_audits(tmp_path) -> None:
    source = tmp_path / "source.pt"
    target = tmp_path / "continuation.pt"
    torch.save(
        {
            "phase": "market",
            "training_state": {
                "completed_epochs": 600,
                "resume_signature": {"num_epochs": 600},
            },
            "training_audits": None,
        },
        source,
    )
    _continuation_checkpoint(source, target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["training_audits"] == {}


def test_continuation_reuses_completed_decoder_without_source_hash(tmp_path) -> None:
    source = tmp_path / "source.pt"
    target = tmp_path / "continuation.pt"
    torch.save({"phase": "market"}, source)
    torch.save(
        {
            "phase": "complete",
            "training_audits": {
                "supervised": {"converged": True},
                "market": {"converged": True},
            },
        },
        target,
    )
    before = target.read_bytes()
    assert _continuation_checkpoint(source, target) == target
    assert target.read_bytes() == before


def test_continuation_rejects_completed_decoder_with_failed_audit(tmp_path) -> None:
    source = tmp_path / "source.pt"
    target = tmp_path / "continuation.pt"
    torch.save({"phase": "market"}, source)
    torch.save(
        {
            "phase": "complete",
            "training_audits": {"market": {"converged": False}},
        },
        target,
    )
    with pytest.raises(ValueError, match="lacks passing audits"):
        _continuation_checkpoint(source, target)


def test_completed_decoder_is_explicitly_loaded_for_joint_restart(tmp_path) -> None:
    checkpoint = tmp_path / "decoder.pt"
    config = FlexTrainConfig(num_epochs=250)
    assert _decoder_train_config_for_checkpoint(config, checkpoint) is config
    torch.save({"phase": "market"}, checkpoint)
    assert _decoder_train_config_for_checkpoint(config, checkpoint) is config
    torch.save({"phase": "complete"}, checkpoint)
    loaded_config = _decoder_train_config_for_checkpoint(config, checkpoint)
    assert loaded_config.just_load is True
    assert loaded_config.num_epochs == config.num_epochs
