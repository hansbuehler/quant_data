from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from quantdata.torch import (
    BaseEnvironment,
    BaseModel,
    MetricTacker,
    check,
    get,
    pop,
)
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


class _DummyModel(BaseModel):
    def __init__(self, env: BaseEnvironment):
        super().__init__(env)
        self.linear = torch.nn.Linear(2, 1, bias=True)

    def forward(self, data, training: bool = False, return_what=BaseModel.ReturnWhat.LOSS_WEIGHT):
        return torch.tensor(1.0), torch.tensor(1.0)


def test_environment_and_base_model_verify() -> None:
    env = BaseEnvironment(precision=32, device=torch.device("cpu"), debug=True)
    model = _DummyModel(env)

    assert env.fdtype == torch.float32
    assert env.idtype == torch.int32
    assert model.device.type == "cpu"

    model.verify(lambda: torch.tensor(True), "no-op")
    with pytest.raises(RuntimeError, match="failed check"):
        model.verify(lambda: torch.tensor(False), "failed check")


def test_check_get_pop_and_broadcast_helpers() -> None:
    tensor = torch.ones((2, 3), dtype=torch.float32)
    data = {"x": tensor.clone(), "flag": torch.tensor([True, False])}

    assert check(tensor, "x", shape=(2, 3), dtype=torch.float32) is tensor
    assert torch.equal(get(data, "x", shape=2, dtype=torch.float32), tensor)
    assert torch.equal(pop(data, "x", optional=False), tensor)
    assert "x" not in data
    assert get({}, "missing", optional=True) is None
    assert check(None, "maybe", optional=True) is None


@pytest.mark.skipif(sys.platform == "win32", reason="grad diagnostics test is disabled on Windows")
def test_grad_diagnostics_and_metric_tracker() -> None:
    model = _DummyModel(BaseEnvironment(precision=32, device="cpu", debug=True))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    assert model.compute_grad_norm().item() == pytest.approx((3.0 / 2.0) ** 0.5)
    model.debug_detect_grad_issues()

    model.linear.weight.grad = torch.full_like(model.linear.weight, float("inf"))
    with pytest.raises(RuntimeError, match="non-finite grad"):
        model.debug_detect_grad_issues()

    tracker = MetricTacker()
    tracker.record(1, {"loss": 0.5, "accuracy": 0.9})
    tracker.record_metric(2, "accuracy", 0.95)

    state = tracker.state_dict()
    assert state["loss"]["epochs"].dtype == np.int32
    assert tracker["accuracy"]["values"] == [0.9, 0.95]

    restored = MetricTacker().load_state_dict(state)
    assert restored.get("loss")["values"] == pytest.approx([0.5])


def test_torch_backbone_and_feedforward() -> None:
    env = BaseEnvironment(precision=32, device="cpu")
    backbone = BackboneTorch(env)

    array = backbone.to_array([1.0, 2.0], dtype=backbone.float32)
    assert isinstance(array, torch.Tensor)
    assert torch.equal(
        backbone.to_array(np.asarray([1.0, 2.0], dtype=np.float32), dtype=backbone.float32),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    assert np.array_equal(backbone.to_numpy(array), np.asarray([1.0, 2.0], dtype=np.float32))
    assert backbone.are_broadcastable((2, 1), (1, 3))
    assert backbone.stack([array, array], axis=0).shape == (2, 2)
    assert torch.equal(backbone.stop_gradient(array).detach(), array)

    config = FeedForwardConfig(num_depth=0)
    model = FeedForward(env, num_input=2, num_output=1, config=config)
    with torch.no_grad():
        model.I0.weight.fill_(1.0)
        model.I0.bias.zero_()

    output = model([torch.tensor(1.0), torch.tensor(2.0)])
    assert torch.isclose(output.squeeze(), torch.tensor(3.0))

    assert isinstance(config.get_activation(), torch.nn.Softplus)