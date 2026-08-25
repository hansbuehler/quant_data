"""Focused tests for frozen Direct-200 quote-capacity calibration."""

from __future__ import annotations

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    FlexObjectiveConfig,
    JointDirectDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.direct_validation_capacity import (
    DirectValidationCapacityTrainConfig,
    _capacity_convergence_audit,
    _checkpoint_payload,
    _learning_rate_multiplier,
    _restore_checkpoint,
    frozen_capacity_decoder_sha256,
    frozen_joint_state_sha256,
    make_frozen_capacity_model,
)
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.torch import BaseEnvironment


def _source(env: BaseEnvironment) -> JointDirectDynamicsModel:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=2,
        num_inner_strikes=6,
    )
    generator = torch.Generator(device="cpu").manual_seed(8128)
    model = JointDirectDynamicsModel(
        env,
        num_days=8,
        config=JointDirectDynamicsConfig(
            label="direct-validation-capacity-source-test",
            state_size=3,
            geometry=geometry,
        ),
    )
    with torch.no_grad():
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
        model.damping_factor.copy_(
            torch.randn(
                model.damping_factor.shape,
                generator=generator,
                dtype=env.fdtype,
            )
        )
    return model


def test_capacity_model_copies_and_freezes_decoder_and_not_dynamics() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    generator = torch.Generator(device="cpu").manual_seed(1731)
    states = torch.randn((5, 3), generator=generator, dtype=env.fdtype)
    source_digest = frozen_joint_state_sha256(source)
    model = make_frozen_capacity_model(source, states, env=env)
    decoder_digest = frozen_capacity_decoder_sha256(model)

    torch.testing.assert_close(
        model.raw_states_from_states(states),
        source.raw_states_from_states(states),
    )
    assert not any(
        parameter.requires_grad for parameter in model.raw_decoder.parameters()
    )
    optimizer = torch.optim.SparseAdam([model.daily_states], lr=1.0e-2)
    indices = torch.tensor([0, 2, 4], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    model.raw_states_for_days(indices).square().mean().backward()
    optimizer.step()

    assert frozen_joint_state_sha256(source) == source_digest
    assert frozen_capacity_decoder_sha256(model) == decoder_digest
    assert not torch.equal(model.daily_states.detach(), states)


def test_capacity_learning_rate_warms_then_decays_to_floor() -> None:
    config = DirectValidationCapacityTrainConfig(
        num_epochs=8,
        warmup_epochs=2,
        minimum_learning_rate_fraction=0.1,
    )
    values = [_learning_rate_multiplier(epoch, config) for epoch in range(8)]
    assert values[0] == 0.55
    assert values[1] == 1.0
    assert values[2] == 1.0
    assert values[-1] == 0.1
    assert all(left >= right for left, right in zip(values[1:], values[2:]))


def test_capacity_checkpoint_restores_states_and_frozen_identity(tmp_path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    states = torch.arange(15, dtype=env.fdtype).reshape(5, 3) / 10.0
    model = make_frozen_capacity_model(source, states, env=env)
    config = DirectValidationCapacityTrainConfig(
        num_epochs=4,
        warmup_epochs=0,
        convergence_window=2,
    )
    objective = FlexObjectiveConfig()
    optimizer = torch.optim.SparseAdam([model.daily_states], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    source_digest = frozen_joint_state_sha256(source)
    decoder_digest = frozen_capacity_decoder_sha256(model)
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        [{"epoch": 1.0, "objective": 2.0}],
        phase="training",
        config=config,
        objective_config=objective,
        market_fingerprints=("a", "b", "c", "d", "e"),
        source_joint_sha256="1" * 64,
        source_native_sha256="2" * 64,
        frozen_source_state_sha256=source_digest,
        frozen_decoder_sha256=decoder_digest,
        numpy_rng_state=np.random.default_rng(17).bit_generator.state,
    )
    checkpoint = tmp_path / "capacity.pt"
    torch.save(payload, checkpoint)
    with torch.no_grad():
        model.daily_states.zero_()
    restored = _restore_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint,
        config=config,
        objective_config=objective,
        market_fingerprints=("a", "b", "c", "d", "e"),
        source_joint_sha256="1" * 64,
        source_native_sha256="2" * 64,
        frozen_source_state_sha256=source_digest,
        frozen_decoder_sha256=decoder_digest,
    )
    assert restored is not None
    assert restored[0] == 1
    torch.testing.assert_close(model.daily_states, states)


def test_capacity_convergence_requires_a_flat_trailing_window() -> None:
    config = DirectValidationCapacityTrainConfig(
        num_epochs=12,
        warmup_epochs=0,
        convergence_window=4,
        convergence_relative_tolerance=1.0e-3,
    )
    history = [
        {"epoch": float(epoch), "objective": 1.0 + 1.0e-5 * (12 - epoch)}
        for epoch in range(1, 13)
    ]
    audit = _capacity_convergence_audit(history, config=config)
    assert audit["converged"] is True
