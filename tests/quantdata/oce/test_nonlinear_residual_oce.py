"""Tests for weighted-orthogonal nonlinear OCE policies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdata.oce.nonlinear_residual_oce import (
    OrthogonalPolicyConfig,
    OrthogonalResidualPolicy,
    build_policy_population,
    evaluate_nonlinear_policy,
    solve_nonlinear_oce,
)
from quantdata.oce.simple_ou_oce import MarketConfig, simulate_market
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_neural_residual_is_weighted_orthogonal_to_affine_basis() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = OrthogonalPolicyConfig(
        label="orthogonality-test",
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5),
    )
    model = OrthogonalResidualPolicy(env, config=config, instrument_count=3)
    state = torch.linspace(-2.0, 2.0, 31, dtype=torch.float64)
    probability = torch.linspace(1.0, 2.0, 31, dtype=torch.float64)
    probability /= torch.sum(probability)

    with torch.no_grad():
        model.residual_network.In[-1].weight.normal_()
        action, residual = model(state, probability)
    basis = torch.stack((torch.ones_like(state), state), dim=1)
    moment = basis.T @ (probability[:, None] * residual)

    assert action.shape == (31, 3)
    assert torch.max(torch.abs(moment)).item() < 1.0e-13


def test_projection_is_differentiable() -> None:
    env = BaseEnvironment(precision=64, device="cpu")
    config = OrthogonalPolicyConfig(
        label="gradient-test",
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    model = OrthogonalResidualPolicy(env, config=config, instrument_count=2)
    state = torch.linspace(-1.5, 1.5, 25, dtype=torch.float64)
    probability = torch.full((25,), 1.0 / 25.0, dtype=torch.float64)
    returns = torch.stack((torch.sin(state), torch.cos(2.0 * state)), dim=1)

    with torch.no_grad():
        model.residual_network.In[-1].weight.normal_(std=0.2)
    action, _ = model(state, probability)
    loss = torch.sum(action * returns)
    loss.backward()

    assert model.residual_network.I0.weight.grad is not None
    assert torch.linalg.vector_norm(model.residual_network.I0.weight.grad).item() > 0.0


def test_nonlinear_solver_starts_from_and_can_improve_affine_policy() -> None:
    paths = simulate_market(MarketConfig(paths=750, seed=31))
    state, instrument_return = build_policy_population(paths)
    config = OrthogonalPolicyConfig(
        label="small-solve",
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=6),
    )
    result = solve_nonlinear_oce(
        state,
        instrument_return,
        config=config,
        maximum_iterations=80,
        gradient_tolerance=1.0e-7,
    )

    assert result.oce >= result.initial_oce - 1.0e-10
    assert result.residual_weighted_affine_moment < 1.0e-12
    assert result.action.shape == instrument_return.shape
    assert np.sum(result.density) == pytest.approx(1.0)
    assert np.all(result.density > 0.0)

    valuation = evaluate_nonlinear_policy(result, state, instrument_return)
    assert valuation["oce"] == pytest.approx(result.oce, abs=1.0e-10)


def test_incremental_q_resolve_starts_at_zero_action_and_oce() -> None:
    paths = simulate_market(MarketConfig(paths=500, seed=37))
    state, instrument_return = build_policy_population(paths)
    config = OrthogonalPolicyConfig(
        label="incremental-q-test",
        network=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=4),
    )
    p_result = solve_nonlinear_oce(
        state,
        instrument_return,
        config=config,
        maximum_iterations=30,
    )
    q_result = solve_nonlinear_oce(
        state,
        instrument_return,
        config=config,
        base_density=p_result.density,
        projection_density=np.ones(len(state)),
        initial_model=p_result.model,
        incremental_from_initial=True,
        maximum_iterations=1,
    )

    assert q_result.initial_oce == pytest.approx(0.0, abs=1.0e-12)
    assert q_result.initial_gradient_l2_norm == pytest.approx(
        p_result.gradient_l2_norm, rel=1.0e-8, abs=1.0e-10
    )
