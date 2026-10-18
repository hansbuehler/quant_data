"""Tests for jointly learned Direct SANOS factors and stable dynamics."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    FlexMarketConfig,
    FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig,
    JointDirectDynamicsConfig,
    PcaDecoderConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _convergence_audit,
    _learning_rate_multiplier,
    _restart_train_config_identity,
    _restore_checkpoint,
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
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _geometry_from_completion,
    _market_config_from_completion,
    install_all_history_warm_start,
    install_direct_decoder_initialization,
)
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _market_cache_geometry,
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


def test_all_history_market_config_inherits_native_exclusions() -> None:
    expected = FlexMarketConfig(excluded_business_dtes=(0.2,))
    restored = _market_config_from_completion(
        {"market_config": asdict(expected)}
    )

    assert restored == expected


def test_all_history_restores_exact_front_term_geometry() -> None:
    expected = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=8,
        num_model_expiries=4,
        explicit_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
        min_eta=0.03,
        max_eta=0.42,
    )

    restored = _geometry_from_completion(
        {"geometry": asdict(expected)},
        requested_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
    )

    assert restored == expected


def test_all_history_rejects_expiry_grid_different_from_completion() -> None:
    expected = FrontTermEtaSharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_inner_strikes=8,
        num_model_expiries=4,
        explicit_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
    )

    with np.testing.assert_raises_regex(ValueError, "expiry grid differs"):
        _geometry_from_completion(
            {"geometry": asdict(expected)},
            requested_model_business_dtes=(1.0, 3.0, 5.0, 10.0),
        )


def test_all_history_market_cache_retains_explicit_expiry_grid() -> None:
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=4,
        explicit_model_business_dtes=(1.0, 2.0, 5.0, 10.0),
    )

    cached = _market_cache_geometry(geometry)

    assert type(cached).__name__ == "FlexGeometryConfig"
    assert cached.explicit_model_business_dtes == (1.0, 2.0, 5.0, 10.0)
    np.testing.assert_array_equal(
        cached.model_business_dtes,
        geometry.model_business_dtes,
    )


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


def test_joint_restart_identity_ignores_operational_day_batch_size() -> None:
    first = JointDirectTrainConfig(day_batch_size=16)
    second = JointDirectTrainConfig(day_batch_size=128)

    assert _restart_train_config_identity(first) == _restart_train_config_identity(
        second
    )


def test_joint_checkpoint_extends_epochs_without_restarting_schedule(tmp_path) -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    model_config = JointDirectDynamicsConfig(
        label="joint-direct-epoch-extension-test",
        state_size=4,
        geometry=_geometry(),
    )
    original = JointDirectTrainConfig(
        num_epochs=2,
        warmup_epochs=0,
        dynamics_ramp_epochs=0,
    )
    extended = replace(original, num_epochs=4)
    fingerprints = tuple(str(index) for index in range(8))
    checkpoint = tmp_path / "joint.pt"
    model = JointDirectDynamicsModel(env, num_days=8, config=model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _learning_rate_multiplier(epoch, original)
    )
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        [{"epoch": 1.0}, {"epoch": 2.0}],
        checkpoint,
        train_config=original,
        market_fingerprints=fingerprints,
        source_decoder_sha256="a" * 64,
        phase="training",
    )

    target = JointDirectDynamicsModel(env, num_days=8, config=model_config)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-4)
    schedule_config = extended
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer, lambda epoch: _learning_rate_multiplier(epoch, schedule_config)
    )
    restored = _restore_checkpoint(
        target,
        target_optimizer,
        target_scheduler,
        checkpoint,
        train_config=extended,
        market_fingerprints=fingerprints,
        source_decoder_sha256="a" * 64,
    )
    assert restored is not None
    assert restored[0] == 2
    assert restored[2:] == ("training", 2)
    assert _restart_train_config_identity(original) == _restart_train_config_identity(
        extended
    )
    assert _learning_rate_multiplier(3, replace(extended, num_epochs=restored[3])) == (
        original.minimum_learning_rate_fraction
    )
    schedule_config = replace(extended, num_epochs=restored[3])
    target_optimizer.step()
    target_scheduler.step()
    np.testing.assert_allclose(
        target_optimizer.param_groups[0]["lr"],
        1.0e-4 * original.minimum_learning_rate_fraction,
    )

    continued = tmp_path / "continued.pt"
    _save_checkpoint(
        target,
        target_optimizer,
        target_scheduler,
        [{"epoch": float(epoch)} for epoch in range(1, 4)],
        continued,
        train_config=extended,
        market_fingerprints=fingerprints,
        source_decoder_sha256="a" * 64,
        phase="training",
        schedule_horizon=restored[3],
    )
    continued_payload = torch.load(continued, map_location="cpu", weights_only=False)
    assert continued_payload["schedule_horizon"] == 2
    assert continued_payload["train_config"]["num_epochs"] == 4

    with np.testing.assert_raises_regex(ValueError, "only extend"):
        _restore_checkpoint(
            target,
            target_optimizer,
            target_scheduler,
            checkpoint,
            train_config=replace(original, num_epochs=1),
            market_fingerprints=fingerprints,
            source_decoder_sha256="a" * 64,
        )


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


def test_all_history_warm_start_appends_2025_states() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    config = JointDirectDynamicsConfig(
        label="joint-direct-all-history-warm-start-test",
        state_size=4,
        geometry=_geometry(),
    )
    source = JointDirectDynamicsModel(env, num_days=2, config=config)
    target = JointDirectDynamicsModel(env, num_days=3, config=config)
    generator = torch.Generator(device="cpu").manual_seed(1931)
    source_2025_states = torch.randn(
        (1, config.state_size),
        generator=generator,
        dtype=env.fdtype,
    )
    source_sha256 = "a" * 64
    installed = install_all_history_warm_start(
        target,
        source_joint_payload={
            "schema": "flex-joint-direct-dynamics-v1",
            "phase": "complete",
            "model_state_dict": source.state_dict(),
        },
        source_joint_sha256=source_sha256,
        source_2025_payload={
            "schema": "flex-direct-2025-quote-capacity-states-v1",
            "source_joint_checkpoint_sha256": source_sha256,
            "validation_dates": ["2025-01-02"],
            "quote_optimal_states": source_2025_states,
        },
        dates=(date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2)),
    )

    expected_states = torch.cat((source.daily_states.detach(), source_2025_states), dim=0)
    torch.testing.assert_close(target.daily_states, expected_states)
    for name, value in target.state_dict().items():
        if name != "daily_state_embedding.weight":
            torch.testing.assert_close(value, source.state_dict()[name])
    assert installed["num_2022_2024_states"] == 2
    assert installed["num_2025_states"] == 1
    assert installed["maximum_state_roundoff"] == 0.0


def test_complete_direct_decoder_initializes_all_history_joint_model() -> None:
    env = BaseEnvironment(precision=64, device="cpu", debug=True)
    geometry = _geometry()
    source = _source(env, num_days=8)
    target = JointDirectDynamicsModel(
        env,
        num_days=8,
        config=JointDirectDynamicsConfig(
            label="joint-direct-complete-decoder-initialization-test",
            state_size=4,
            geometry=geometry,
        ),
    )
    dates = tuple(date(2024, 11, 4) + timedelta(weeks=index) for index in range(8))
    fingerprints = tuple(f"market-{index}" for index in range(8))
    days = tuple(
        SimpleNamespace(training_fingerprint=fingerprint)
        for fingerprint in fingerprints
    )
    initialization = install_direct_decoder_initialization(
        target,
        source_payload={
            "schema_version": 2,
            "phase": "complete",
            "training_audits": {
                "supervised": {"converged": True},
                "market": {"converged": True},
            },
            "market_fingerprints": fingerprints,
            "model_identity": {"model_config": asdict(source.config)},
            "model_state_dict": source.state_dict(),
        },
        geometry=geometry,
        days=days,  # type: ignore[arg-type]
        dates=dates,
        env=env,
    )

    torch.testing.assert_close(
        target.raw_states_from_states(target.daily_states),
        source.raw_states_from_states(source.daily_states),
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert initialization["mode"] == "complete_all_history_affine_direct_decoder"
    assert torch.max(
        torch.abs(torch.linalg.eigvals(target.weekly_transition()))
    ) < 1.0
