"""Tests for jointly learned expiry-causal SANOS factors and P dynamics."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import (
    JointHierarchicalDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.joint_hierarchical import (
    _CONTINUATION_CHECKPOINT_SCHEMA,
    JointHierarchicalTrainConfig,
    QuoteOptimalStateContinuationConfig,
    QuoteOptimalStateTrainConfig,
    _restore_checkpoint,
    _save_checkpoint,
    _save_state_checkpoint,
    hierarchical_dynamics_loss,
    hierarchical_factor_gauge_penalty,
    initialize_joint_hierarchical_model,
    load_quote_state_continuation_initialization,
    standardize_affine_hierarchical_model,
)
from quantdata.dysanos.flex.model import (
    JointHierarchicalDynamicsModel,
    LatentStateSanosModel,
)
from quantdata.torch import BaseEnvironment


def _geometry() -> SharedStrikeCappedUnboundedDlvFlexGeometryConfig:
    return SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=3,
        num_inner_strikes=6,
    )


def _source(env: BaseEnvironment, *, num_days: int = 40) -> LatentStateSanosModel:
    generator = torch.Generator(device="cpu").manual_seed(7127)
    model = LatentStateSanosModel(
        env,
        num_days=num_days,
        config=TriangularPcaDecoderConfig(
            label="joint-hierarchical-source-test",
            state_size=6,
            state_sizes_by_expiry=(2, 2, 2),
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


def _joint_model(
    env: BaseEnvironment,
    *,
    num_days: int = 40,
) -> JointHierarchicalDynamicsModel:
    return JointHierarchicalDynamicsModel(
        env,
        num_days=num_days,
        config=JointHierarchicalDynamicsConfig(
            label="joint-hierarchical-model-test",
            state_size=6,
            state_sizes_by_expiry=(2, 2, 2),
            geometry=_geometry(),
        ),
    )


def test_blockwise_whitening_preserves_raw_states_and_decoder_causality() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    initialization = standardize_affine_hierarchical_model(source)
    original = source.raw_states_from_states(source.daily_states)
    transformed = (
        initialization.states @ initialization.decoder_weight.T
        + initialization.decoder_bias
    )
    torch.testing.assert_close(transformed, original)
    for block in (slice(0, 2), slice(2, 4), slice(4, 6)):
        values = initialization.states[:, block]
        torch.testing.assert_close(
            torch.mean(values, dim=0),
            torch.zeros(2, dtype=env.fdtype),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            values.T @ values / len(values),
            torch.eye(2, dtype=env.fdtype),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    block_width = _geometry().expiry_block_state_size
    assert torch.count_nonzero(
        initialization.decoder_weight[:block_width, 2:]
    ) == 0
    assert torch.count_nonzero(
        initialization.decoder_weight[block_width : 2 * block_width, 4:]
    ) == 0


def test_decoder_and_matrix_exponential_preserve_expiry_causality() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = _joint_model(env)
    with torch.no_grad():
        model.raw_decoder.I0.weight.normal_()
        model.causal_coupling_raw.normal_(std=0.2)
        model.damping_factor.normal_(std=0.2)
        model.rotation_raw.normal_(std=0.2)
    state = torch.randn((3, 6), dtype=env.fdtype)
    changed = state.clone()
    changed[:, 4:] += 7.0
    raw = model.raw_states_from_states(state)
    changed_raw = model.raw_states_from_states(changed)
    block_width = _geometry().expiry_block_state_size
    torch.testing.assert_close(raw[:, : 2 * block_width], changed_raw[:, : 2 * block_width])

    generator = model.mean_reversion_generator()
    transition = model.weekly_transition()
    forbidden = model.generator_strict_upper_mask.T
    torch.testing.assert_close(
        generator * forbidden,
        torch.zeros_like(generator),
    )
    torch.testing.assert_close(
        transition * forbidden,
        torch.zeros_like(transition),
        atol=1.0e-12,
        rtol=0.0,
    )
    predicted = state @ transition
    changed_predicted = changed @ transition
    torch.testing.assert_close(predicted[:, :4], changed_predicted[:, :4])


def test_hierarchical_generator_is_stationary_and_loss_is_differentiable() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    source = _source(env)
    initialization = standardize_affine_hierarchical_model(source)
    model = _joint_model(env)
    origin = np.arange(source.num_days - 5, dtype=np.int64)
    endpoint = origin + 5
    initialize_joint_hierarchical_model(
        model,
        initialization,
        origin_indices=origin,
        endpoint_indices=endpoint,
    )
    eigenvalues = torch.linalg.eigvals(model.mean_reversion_generator())
    assert torch.min(eigenvalues.real) >= 1.0e-3 - 1.0e-10
    assert torch.max(torch.abs(torch.linalg.eigvals(model.weekly_transition()))) < 1.0
    innovation = model.innovation_cholesky()
    torch.testing.assert_close(innovation.triu(1), torch.zeros_like(innovation))

    loss = hierarchical_dynamics_loss(
        model,
        torch.as_tensor(origin),
        torch.as_tensor(endpoint),
    ) + 0.1 * hierarchical_factor_gauge_penalty(
        model.daily_states,
        model.config.state_sizes_by_expiry,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (
        model.daily_states,
        model.damping_factor,
        model.rotation_raw,
        model.causal_coupling_raw,
        model.equilibrium_mean,
        model.innovation_cholesky_raw,
    ):
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_epoch(
    tmp_path,
) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = _joint_model(env, num_days=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = JointHierarchicalTrainConfig(
        num_epochs=2,
        warmup_epochs=0,
        dynamics_ramp_epochs=0,
    )
    checkpoint = tmp_path / "joint-hierarchical.pt"
    history = [{"epoch": 1.0, "joint_objective": 2.0}]
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        checkpoint,
        train_config=config,
        market_fingerprints=tuple(str(index) for index in range(8)),
        source_decoder_sha256="a" * 64,
        phase="training",
    )
    original = model.daily_states.detach().clone()
    with torch.no_grad():
        model.daily_states.add_(10.0)
    restored = _restore_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint,
        train_config=config,
        market_fingerprints=tuple(str(index) for index in range(8)),
        source_decoder_sha256="a" * 64,
    )
    assert restored is not None
    epoch, restored_history, phase = restored
    assert epoch == 1
    assert restored_history == history
    assert phase == "training"
    torch.testing.assert_close(model.daily_states, original)


def _write_exhausted_quote_state_checkpoint(
    tmp_path,
    *,
    phase: str = "training",
):
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = _joint_model(env, num_days=2)
    source_model_path = tmp_path / "joint-source.pt"
    torch.save({"source": "immutable"}, source_model_path)
    config = QuoteOptimalStateTrainConfig(
        num_epochs=4,
        warmup_epochs=0,
        convergence_window=2,
        convergence_relative_tolerance=1.0e-3,
        compile_sanos_solver=False,
    )
    history = [
        {
            "epoch": float(index + 1),
            "objective": value,
            "dlvsq_cap_penalty": 0.01,
            "total": value + 0.01,
            "learning_rate": 1.0e-4,
        }
        for index, value in enumerate((2.0, 1.5, 1.2, 1.0))
    ]
    optimizer = torch.optim.SparseAdam([model.daily_states], lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state_checkpoint = tmp_path / "quote-states.pt"
    _save_state_checkpoint(
        model,
        optimizer,
        scheduler,
        history,
        state_checkpoint,
        train_config=config,
        market_fingerprints=("first", "second"),
        source_model_sha256=hashlib.sha256(source_model_path.read_bytes()).hexdigest(),
        rng=np.random.default_rng(13),
        phase=phase,
    )
    markets = (
        type("Market", (), {"training_fingerprint": "first"})(),
        type("Market", (), {"training_fingerprint": "second"})(),
    )
    return model, markets, source_model_path, state_checkpoint, config


def test_continuation_accepts_only_exhausted_finite_nonplateau_source(
    tmp_path,
) -> None:
    model, markets, source_model, state_checkpoint, config = (
        _write_exhausted_quote_state_checkpoint(tmp_path)
    )
    initialization = load_quote_state_continuation_initialization(
        model,
        markets,  # type: ignore[arg-type]
        source_state_checkpoint_path=state_checkpoint,
        source_model_path=source_model,
        base_train_config=config,
    )
    assert initialization.states.shape == (2, 6)
    assert initialization.source_completed_epochs == 4
    assert initialization.source_convergence["converged"] is False
    assert (
        initialization.source_convergence["reason"]
        == "trailing_window_not_converged"
    )
    assert initialization.provenance["optimizer_scheduler_reset"] is True


def test_continuation_rejects_completed_base_checkpoint(tmp_path) -> None:
    model, markets, source_model, state_checkpoint, config = (
        _write_exhausted_quote_state_checkpoint(tmp_path, phase="complete")
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_quote_state_continuation_initialization(
            model,
            markets,  # type: ignore[arg-type]
            source_state_checkpoint_path=state_checkpoint,
            source_model_path=source_model,
            base_train_config=config,
        )


def test_continuation_checkpoint_has_new_schema_and_provenance(tmp_path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model = _joint_model(env, num_days=2)
    optimizer = torch.optim.SparseAdam([model.daily_states], lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "continuation.pt"
    provenance = {
        "source_state_checkpoint_sha256": "a" * 64,
        "optimizer_scheduler_reset": True,
    }
    _save_state_checkpoint(
        model,
        optimizer,
        scheduler,
        [],
        checkpoint,
        train_config=QuoteOptimalStateContinuationConfig(),
        market_fingerprints=("first", "second"),
        source_model_sha256="b" * 64,
        rng=np.random.default_rng(17),
        phase="training",
        checkpoint_schema=_CONTINUATION_CHECKPOINT_SCHEMA,
        continuation_provenance=provenance,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == _CONTINUATION_CHECKPOINT_SCHEMA
    assert payload["continuation_provenance"] == provenance
    assert payload["completed_epochs"] == 0
