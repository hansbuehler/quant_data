"""Sequence overlays start at the original Gaussian innovation law."""

import pytest
import torch

from quantdata.dysanos.v11.vix_sequence import (
    VixSequenceConfig,
    VixSequenceModel,
    quantile_distribution_loss,
)
from quantdata.torch import BaseEnvironment


@pytest.mark.parametrize("architecture", ["lstm", "selective_ssm"])
def test_sequence_starts_at_original_noise_and_has_gradients(architecture: str) -> None:
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    config = VixSequenceConfig(label="test-vix", architecture=architecture, input_size=3,
                               factor_rank=2, state_size=4, sequence_length=5)
    model = VixSequenceModel(env, config=config, white_basis=torch.eye(4, dtype=torch.float64)[:, :2])
    epsilon = torch.randn(8, 4, dtype=torch.float64)
    context = torch.randn(8, 5, 3, dtype=torch.float64)
    white, penalty = model(context, torch.randn(8, dtype=torch.float64), epsilon)
    torch.testing.assert_close(white, epsilon)
    assert penalty.item() == 0
    white.square().sum().backward()
    assert model.readout.I0.weight.grad is not None
    assert torch.isfinite(model.readout.I0.weight.grad).all()


def test_quantile_loss_detects_level_shift() -> None:
    values = torch.arange(10, dtype=torch.float64)[:, None].requires_grad_()
    probabilities = torch.linspace(.1, .9, 7, dtype=torch.float64)
    target = torch.quantile(values.detach(), probabilities, dim=0)
    loss = quantile_distribution_loss(values, probabilities=probabilities, target_quantiles=target,
                                      target_scale=torch.ones(1, dtype=torch.float64))
    assert loss.item() == pytest.approx(0)
    shifted = quantile_distribution_loss(values, probabilities=probabilities, target_quantiles=target + 2,
                                         target_scale=torch.ones(1, dtype=torch.float64))
    assert shifted.item() == pytest.approx(4)
    shifted.backward()
    assert values.grad.sum().item() == pytest.approx(-4)
