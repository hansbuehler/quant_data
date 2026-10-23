"""Checkpoint admission tests for prespecified DYSANOS 1.1 epoch caps."""

from dataclasses import asdict

import pytest
import torch

from quantdata.dysanos.flex.config import JointDirectDynamicsConfig
from quantdata.dysanos.flex.joint_direct import JointDirectTrainConfig
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.v11.capped_model import load_capped_joint_model
from quantdata.torch import BaseEnvironment


def test_loads_fully_saved_nonconverged_epoch_cap_without_relabeling(tmp_path) -> None:
    checkpoint = tmp_path / "joint.pt"
    config = JointDirectDynamicsConfig(label="cap-test", state_size=2)
    train = JointDirectTrainConfig(
        num_epochs=3,
        warmup_epochs=0,
        dynamics_ramp_epochs=0,
        convergence_window=2,
    )
    model = JointDirectDynamicsModel(
        BaseEnvironment(precision=32, device="cpu", debug=False),
        num_days=4,
        config=config,
    )
    payload = {
        "schema": "flex-joint-direct-dynamics-v1",
        "phase": "training",
        "model_config": asdict(config),
        "model_config_version": config.version.unique_id48,
        "model_user_version": config.user_version,
        "train_config": asdict(train),
        "model_state_dict": model.state_dict(),
        "completed_epochs": 3,
        "history": [
            {"epoch": float(epoch), "joint_objective": value, "dynamics_ramp": 1.0}
            for epoch, value in enumerate((1.0, 0.9, 0.8), start=1)
        ],
    }
    torch.save(payload, checkpoint)
    restored, report = load_capped_joint_model(checkpoint, device="cpu", maximum_epochs=3)
    assert restored.daily_states.shape == (4, 2)
    assert report["phase"] == "training"
    assert report["completed_epochs"] == 3
    assert report["convergence"]["converged"] is False

    payload["completed_epochs"] = 2
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="epoch count"):
        load_capped_joint_model(checkpoint, device="cpu", maximum_epochs=3)
