"""Conditional normal-innovation architecture invariants."""

import pytest
import torch

from quantdata.dysanos.v11.conditional_sequence import (
    ConditionalInnovationModel,
    ConditionalSequenceConfig,
    baseline_normal_nll,
    conditional_normal_nll,
)
from quantdata.torch import BaseEnvironment


@pytest.mark.parametrize("architecture", ["lstm", "selective_ssm"])
def test_conditional_networks_produce_finite_white_predictions(
    architecture: str,
) -> None:
    env = BaseEnvironment(precision=32, device="cpu", debug=True)
    config = ConditionalSequenceConfig(
        label=f"test-{architecture}",
        architecture=architecture,
        input_size=5,
        hidden_size=7,
        mean_rank=2,
        state_size=4,
        sequence_length=3,
    )
    basis = torch.eye(4, dtype=env.fdtype)[:, :2]
    model = ConditionalInnovationModel(env, config=config, white_mean_basis=basis)
    sequence = torch.randn(6, 3, 5, dtype=env.fdtype)
    spot = torch.randn(6, dtype=env.fdtype)
    target = torch.randn(6, 4, dtype=env.fdtype)
    predicted, log_scale = model(sequence, spot)
    assert predicted.shape == target.shape
    assert log_scale.shape == (6,)
    loss = conditional_normal_nll(target, predicted, log_scale)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    assert baseline_normal_nll(target) > 0.0
