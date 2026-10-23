"""Load a DYSANOS 1.1 joint model saved at its prespecified epoch cap."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import torch

from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _convergence_audit,
)
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.presentation_hedging import _checkpoint_config
from quantdata.torch import BaseEnvironment


def load_capped_joint_model(
    checkpoint: Path,
    *,
    device: str,
    maximum_epochs: int = 600,
) -> tuple[JointDirectDynamicsModel, dict[str, object]]:
    """Load a completed or capped fit, retaining its true convergence status.

    A ``training``-phase checkpoint is accepted only if its full stated epoch
    schedule was saved.  This loader does not change the checkpoint phase or
    imply the strict loss-convergence test passed.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError("not a Joint Direct checkpoint")
    phase = payload.get("phase")
    if phase not in {"training", "complete"}:
        raise ValueError(f"unsupported joint checkpoint phase {phase!r}")
    training_values = payload.get("train_config")
    history = payload.get("history")
    if not isinstance(training_values, dict) or not isinstance(history, list):
        raise ValueError("joint checkpoint lacks its training configuration/history")
    training_config = JointDirectTrainConfig(**training_values)
    epochs = int(payload.get("completed_epochs", -1))
    if epochs != len(history) or epochs > maximum_epochs or epochs <= 0:
        raise ValueError("joint checkpoint has an invalid epoch count")
    if phase == "training" and epochs != training_config.num_epochs:
        raise ValueError("joint checkpoint was interrupted before its epoch cap")
    config = _checkpoint_config(payload)
    if (
        payload.get("model_config_version") != config.version.unique_id48
        or payload.get("model_user_version") != config.user_version
    ):
        raise ValueError("joint checkpoint has stale model configuration")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("joint checkpoint has no model weights")
    states = state_dict.get("daily_state_embedding.weight")
    if not isinstance(states, torch.Tensor) or states.ndim != 2:
        raise ValueError("joint checkpoint has no daily latent states")
    audit = _convergence_audit(history, config=training_config)
    if phase == "complete" and audit.get("converged") is not True:
        raise ValueError("complete checkpoint lacks a passing convergence audit")
    env = BaseEnvironment(precision=32, device=device, debug=False)
    model = JointDirectDynamicsModel(env, num_days=len(states), config=config)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    model.eval()
    return model, {
        "checkpoint": str(checkpoint.resolve()),
        "phase": phase,
        "completed_epochs": epochs,
        "convergence": audit,
        "training_dates": len(states),
    }
