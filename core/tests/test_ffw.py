from __future__ import annotations

import torch
import pytest

from openqai.core.nn.torch import BaseEnvironment
from openqai.core.nn.torch.ffw import FeedForward, FeedForwardConfig


def test_feedforward_config_defaults_and_validation() -> None:
    config = FeedForwardConfig()

    assert config.activation == "Softplus"
    assert config.num_depth == 0
    assert config.num_width == 16
    assert isinstance(config.get_activation(), torch.nn.Softplus)


def test_feedforward_forward_with_list_input() -> None:
    env = BaseEnvironment(precision=32, device=torch.device("cpu"))
    model = FeedForward(env, num_input=2, num_output=1, config=FeedForwardConfig(num_depth=0))

    with torch.no_grad():
        model.I0.weight.fill_(1.0)
        model.I0.bias.zero_()

    output = model([torch.tensor(1.0), torch.tensor(2.0)])

    assert output.shape == torch.Size([1])
    assert torch.isclose(output.squeeze(), torch.tensor(3.0))


def test_feedforward_forward_with_tensor_input_and_mask() -> None:
    env = BaseEnvironment(precision=32, device=torch.device("cpu"))
    model = FeedForward(env, num_input=2, num_output=1, config=FeedForwardConfig(num_depth=0))

    with torch.no_grad():
        model.I0.weight.fill_(1.0)
        model.I0.bias.zero_()

    tensor_input = torch.tensor([1.0, 2.0])
    mask = torch.tensor(True)

    output = model(tensor_input, mask=mask)

    assert output.shape == torch.Size([1])
    assert torch.isclose(output.squeeze(), torch.tensor(3.0))


@pytest.mark.parametrize("activation", ["ReLU", "Softplus"])
def test_feedforward_smoke_test_matches_notebook_pattern(activation: str) -> None:
    env = BaseEnvironment(device=torch.device("cpu"), debug=True)
    config = FeedForwardConfig(num_depth=1, num_width=32, activation=activation)
    model = FeedForward(env=env, num_input=3, num_output=1, config=config, use_final_bias=True)
    x = torch.randn(2, 3, dtype=torch.float32, device=env.device)

    y = model(x)

    assert y.shape == torch.Size([2, 1])
    assert y.device == env.device
    assert torch.isfinite(y).all()