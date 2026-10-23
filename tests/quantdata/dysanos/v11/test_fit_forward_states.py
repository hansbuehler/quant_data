"""Frozen-decoder initialization checks for forward DYSANOS states."""

import pytest
import torch

from quantdata.dysanos.flex.config import JointDirectDynamicsConfig
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.v11.fit_forward_states import (
    ForwardStateFitConfig,
    _project_native_initial_states,
)
from quantdata.torch import BaseEnvironment


def test_projection_recovers_known_holdout_states_without_changing_decoder() -> None:
    model = JointDirectDynamicsModel(
        BaseEnvironment(precision=32, device="cpu", debug=False),
        num_days=2,
        config=JointDirectDynamicsConfig(label="projection-test", state_size=3),
    )
    desired = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]])
    before = model.raw_decoder.I0.weight.detach().clone()
    native = model.raw_states_from_states(desired).detach()
    projected = _project_native_initial_states(model, native)
    torch.testing.assert_close(projected.float(), desired, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(model.raw_decoder.I0.weight, before)


def test_forward_state_config_rejects_more_than_600_epochs() -> None:
    with pytest.raises(ValueError, match="invalid forward"):
        ForwardStateFitConfig(maximum_epochs=601)
