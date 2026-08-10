from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from quantdata.dysanos.grid.gendysanos.norm_continuation import (
    NormContinuationAction,
    NormContinuationConfig,
    clipped_radius_pnl_derivative,
    continuation_loss,
    continuation_pnl,
    exterior_l1_penalty,
    radial_l1_clip,
)
from quantdata.dysanos.grid.gendysanos.required_runs import RequiredCase
from quantdata.dysanos.grid.gendysanos.run_expiry_arbitrage_experiments import (
    _find_cache,
)
from quantdata.torch import BaseEnvironment, BaseModel
from quantdata.torch.ffw import FeedForward, FeedForwardConfig


def _config(*, utility: str = "powerexp") -> NormContinuationConfig:
    return NormContinuationConfig(
        label="test-norm-continuation",
        source_cache_uid="test-cache",
        state_features=2,
        trading_times=2,
        expiries=1,
        instruments_per_expiry=3,
        risk_aversion=2.0,
        utility=utility,
        radii=(1.0, 2.0, 4.0),
        exterior_penalty=0.25,
        network_config=FeedForwardConfig(
            num_depth=1,
            num_width=5,
            activation="SiLU",
            use_bias=True,
        ),
    )


def _model(*, utility: str = "powerexp") -> NormContinuationAction:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    return NormContinuationAction(
        env,
        config=_config(utility=utility),
        state_mean=torch.zeros((2, 2), dtype=env.fdtype),
        state_whitener=torch.eye(2, dtype=env.fdtype).expand(2, -1, -1).clone(),
    )


def test_cache_lookup_ignores_complete_manifests_from_old_schemas(
    tmp_path: Path,
) -> None:
    case = RequiredCase(
        generator="pca-iv",
        trading_frequency="weekly",
        return_horizon="expiry",
        hedge="delta",
        cost_rate=1.0e-4,
        path_count=1_000,
    )
    configuration = {
        "generator": case.generator,
        "trading_frequency": case.trading_frequency,
    }
    for schema in ("1.0", "1.3"):
        uid = f"cache-{schema.replace('.', '-')}"
        directory = tmp_path / uid
        directory.mkdir()
        (directory / "block.npz").touch()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "schema_version": schema,
                    "cache_uid": uid,
                    "configuration": configuration,
                    "num_samples": case.path_count,
                    "blocks": [{"start": 0, "stop": case.path_count, "file": "block.npz"}],
                }
            ),
            encoding="utf-8",
        )
    result = _find_cache(case, tmp_path)
    assert result.cache_uid == "cache-1-3"


def test_radial_clip_enforces_l1_radius() -> None:
    raw = torch.tensor(
        [[[[2.0, -1.0, 1.0]], [[0.1, -0.2, 0.3]]]], dtype=torch.float64
    )
    action, raw_l1, clipped = radial_l1_clip(raw, 1.0)
    assert torch.allclose(raw_l1, torch.tensor([[4.0, 0.6]], dtype=torch.float64))
    assert clipped.tolist() == [[True, False]]
    assert action[0, 0].abs().sum().item() == pytest.approx(1.0)
    assert torch.equal(action[0, 1], raw[0, 1])


def test_tensor_radius_uses_one_compilable_projection_graph() -> None:
    raw = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float64)
    action, raw_l1, clipped = radial_l1_clip(
        raw, torch.tensor(1.5, dtype=torch.float64)
    )
    assert raw_l1.item() == pytest.approx(3.0)
    assert clipped.item() is True
    assert action.abs().sum().item() == pytest.approx(1.5)


def test_exterior_penalty_restores_radial_gradient_outside_ball() -> None:
    raw = torch.tensor([[[[2.0, 1.0]]]], dtype=torch.float64, requires_grad=True)
    action, raw_l1, _ = radial_l1_clip(raw, 1.0)
    utility = action.sum()
    utility_gradient = torch.autograd.grad(utility, raw, retain_graph=True)[0]
    assert torch.equal(utility_gradient, torch.zeros_like(utility_gradient))

    penalty = exterior_l1_penalty(raw_l1, 1.0).sum()
    penalty_gradient = torch.autograd.grad(penalty, raw)[0]
    assert torch.equal(penalty_gradient, torch.full_like(penalty_gradient, 4.0))


def test_increasing_radius_restores_utility_radial_gradient() -> None:
    raw = torch.tensor([[[[2.0, 1.0]]]], dtype=torch.float64, requires_grad=True)
    clipped_action, _, _ = radial_l1_clip(raw, 1.0)
    clipped_gradient = torch.autograd.grad(clipped_action.sum(), raw, retain_graph=True)[0]
    interior_action, _, clipped = radial_l1_clip(raw, 4.0)
    interior_gradient = torch.autograd.grad(interior_action.sum(), raw)[0]
    assert torch.equal(clipped_gradient, torch.zeros_like(clipped_gradient))
    assert clipped.item() is False
    assert torch.equal(interior_gradient, torch.ones_like(interior_gradient))


def test_mechanical_radius_pnl_derivative_matches_finite_difference() -> None:
    raw = torch.tensor([[[[2.0, -1.0]], [[0.2, -0.1]]]], dtype=torch.float64)
    returns = torch.tensor([[[[0.4, -0.2]], [[0.4, -0.2]]]], dtype=torch.float64)
    costs = torch.full_like(returns, 0.01)
    radius = 1.0
    action, _, clipped = radial_l1_clip(raw, radius)
    pnl = continuation_pnl(action, returns, costs)
    derivative = clipped_radius_pnl_derivative(pnl, clipped, radius)

    step = 1.0e-6
    upper = continuation_pnl(radial_l1_clip(raw, radius + step)[0], returns, costs)
    lower = continuation_pnl(radial_l1_clip(raw, radius - step)[0], returns, costs)
    finite_difference = (upper - lower) / (2.0 * step)
    assert torch.allclose(derivative, finite_difference, atol=1.0e-10)
    assert derivative[0, 0] != 0.0
    assert derivative[0, 1] == 0.0


def test_model_uses_independent_project_feedforward_networks() -> None:
    model = _model()
    assert isinstance(model, BaseModel)
    assert all(isinstance(network, FeedForward) for network in model.networks)
    assert all(network.env is model.env for network in model.networks)
    state = torch.randn(7, 2, 2, dtype=model.fdtype)
    action, raw_l1, clipped = model(state, 1.0)
    assert action.shape == (7, 2, 1, 3)
    assert raw_l1.shape == clipped.shape == (7, 2)
    assert torch.equal(action, torch.zeros_like(action))

    action[:, 0].sum().backward()
    assert any(parameter.grad is not None for parameter in model.networks[0].parameters())
    assert all(
        parameter.grad is None or torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in model.networks[1].parameters()
    )


@pytest.mark.parametrize("utility", ["powerexp", "entropy"])
def test_zero_action_has_finite_value_and_nonzero_learning_gradient(utility: str) -> None:
    model = _model(utility=utility)
    state = torch.zeros(16, 2, 2, dtype=model.fdtype)
    returns = torch.ones(16, 2, 1, 3, dtype=model.fdtype) * 0.1
    costs = torch.full_like(returns, 1.0e-4)
    weight = torch.ones(16, dtype=model.fdtype)

    action, raw_l1, clipped = model(state, 1.0)
    assert torch.equal(action, torch.zeros_like(action))
    assert torch.equal(raw_l1, torch.zeros_like(raw_l1))
    assert not clipped.any()
    assert torch.equal(
        exterior_l1_penalty(raw_l1, 1.0),
        torch.zeros(2, dtype=model.fdtype),
    )

    loss, values = continuation_loss(
        model,
        state,
        returns,
        costs,
        weight,
        radius=1.0,
    )
    assert loss.item() == pytest.approx(0.0, abs=1.0e-15)
    assert torch.equal(values["objective"], torch.zeros_like(values["objective"]))
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


@pytest.mark.parametrize("utility", ["powerexp", "entropy"])
def test_continuation_loss_is_finite_and_differentiable(utility: str) -> None:
    model = _model(utility=utility)
    state = torch.randn(32, 2, 2, dtype=model.fdtype)
    returns = torch.randn(32, 2, 1, 3, dtype=model.fdtype) * 0.1
    costs = torch.full_like(returns, 1.0e-4)
    weight = torch.ones(32, dtype=model.fdtype)
    loss, values = continuation_loss(
        model,
        state,
        returns,
        costs,
        weight,
        radius=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert values["objective"].shape == (2,)
    assert values["penalty"].shape == (2,)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
