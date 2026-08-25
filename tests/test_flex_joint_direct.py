"""Tests for jointly learned Direct SANOS factors and stable dynamics."""

from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    JointDirectDynamicsConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _convergence_audit,
    _save_checkpoint,
    factor_gauge_penalty,
    initialize_joint_model,
    joint_dynamics_loss,
    project_native_raw_to_joint_states,
    standardize_affine_direct_model,
)
from quantdata.dysanos.flex.model import (
    JointDirectDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.torch import BaseEnvironment


def _geometry() -> SharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    return SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=3,
        num_inner_strikes=6,
    )


def _source(env: BaseEnvironment, *, num_days: int = 40) -> LatentStateSanosModel:
    generator = torch.Generator(device="cpu").manual_seed(1729)
    model = LatentStateSanosModel(
        env,
        num_days=num_days,
        config=PcaDecoderConfig(
            label="joint-direct-source-test",
            state_size=4,
            geometry=_geometry(),
        ),
    )
    with torch.no_grad():
        model.daily_states.copy_(
            torch.randn(
                model.daily_states.shape,
                generator=generator,
                dtype=env.fdtype,
            )
        )
        model.raw_decoder.I0.weight.copy_(
            torch.randn(
                model.raw_decoder.I0.weight.shape,
                generator=generator,
                dtype=env.fdtype,
            )
        )
        model.raw_decoder.I0.bias.copy_(
            torch.randn(
                model.raw_decoder.I0.bias.shape,
                generator=generator,
                dtype=env.fdtype,
            )
        )
    return model


def test_affine_whitening_preserves_raw_states_and_fixes_gauge() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    initialization = standardize_affine_direct_model(source)
    original = source.raw_states_from_states(source.daily_states)
    transformed = (
        initialization.states @ initialization.decoder_weight.transpose(0, 1)
        + initialization.decoder_bias
    )
    torch.testing.assert_close(transformed, original)
    torch.testing.assert_close(
        torch.mean(initialization.states, dim=0),
        torch.zeros(4, dtype=env.fdtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    covariance = initialization.states.T @ initialization.states / len(
        initialization.states
    )
    torch.testing.assert_close(
        covariance,
        torch.eye(4, dtype=env.fdtype),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_joint_generator_is_real_stationary_and_differentiable() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    initialization = standardize_affine_direct_model(source)
    model = JointDirectDynamicsModel(
        env,
        num_days=source.num_days,
        config=JointDirectDynamicsConfig(
            label="joint-direct-model-test",
            state_size=4,
            geometry=_geometry(),
            minimum_mean_reversion_per_week=1.0e-3,
        ),
    )
    origin = np.arange(source.num_days - 5, dtype=np.int64)
    endpoint = origin + 5
    initialize_joint_model(
        model,
        initialization,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    generator = model.mean_reversion_generator()
    transition = model.weekly_transition()
    eigenvalues = torch.linalg.eigvals(generator)
    assert torch.min(eigenvalues.real) >= 1.0e-3 - 1.0e-10
    assert torch.max(torch.abs(torch.linalg.eigvals(transition))) < 1.0

    loss = joint_dynamics_loss(
        model,
        torch.as_tensor(origin),
        torch.as_tensor(endpoint),
    ) + 0.1 * factor_gauge_penalty(model.daily_states)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (
        model.daily_states,
        model.damping_factor,
        model.rotation_raw,
        model.equilibrium_mean,
        model.innovation_cholesky_raw,
    ):
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_projection_recovers_states_on_joint_affine_manifold() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    initialization = standardize_affine_direct_model(source)
    model = JointDirectDynamicsModel(
        env,
        num_days=source.num_days,
        config=JointDirectDynamicsConfig(
            label="joint-direct-projection-test",
            state_size=4,
            geometry=_geometry(),
        ),
    )
    origin = np.arange(source.num_days - 5, dtype=np.int64)
    initialize_joint_model(
        model,
        initialization,
        origin_indices=origin,
        endpoint_indices=origin + 5,
    )
    raw = model.raw_states_from_states(model.daily_states).detach()
    projected = project_native_raw_to_joint_states(model, raw)
    torch.testing.assert_close(
        projected,
        model.daily_states.detach().cpu(),
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_checkpoint_accepts_environment_string_device(tmp_path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = JointDirectDynamicsModel(
        env,
        num_days=8,
        config=JointDirectDynamicsConfig(
            label="joint-direct-checkpoint-test",
            state_size=4,
            geometry=_geometry(),
        ),
    )
    # CUDA environments currently preserve explicit indexed devices as strings.
    # Reproduce that interface without requiring a GPU in this unit test.
    env.device = "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "joint.pt"
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        [],
        checkpoint,
        train_config=JointDirectTrainConfig(
            num_epochs=2,
            warmup_epochs=0,
            dynamics_ramp_epochs=0,
        ),
        market_fingerprints=tuple(str(index) for index in range(8)),
        source_decoder_sha256="a" * 64,
        phase="training",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["phase"] == "training"


def test_convergence_audit_compares_only_fully_weighted_epochs() -> None:
    history = []
    for epoch in range(1, 81):
        ramp = min(1.0, epoch / 20.0)
        # The ramp makes early combined values artificially smaller than every
        # value under the final objective.  The fixed-weight tail is converged.
        objective = 1.0 + 0.2 * ramp
        if epoch >= 20:
            objective = 1.2 + 1.0e-4 * (80 - epoch)
        history.append(
            {
                "epoch": float(epoch),
                "dynamics_ramp": ramp,
                "joint_objective": objective,
            }
        )
    audit = _convergence_audit(
        history,
        config=JointDirectTrainConfig(
            num_epochs=80,
            warmup_epochs=0,
            dynamics_ramp_epochs=20,
            convergence_window=20,
        ),
    )
    assert audit["converged"] is True
    assert audit["comparable_start_epoch"] == 20
    assert audit["best_epoch"] == 80
