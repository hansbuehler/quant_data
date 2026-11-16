"""Identity checks for the independent original-code rolling adapters."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
import torch

from quantdata.dysanos.flex.config import FlexTrainConfig
from quantdata.dysanos.flex.joint_direct import (
    JointDirectTrainConfig,
    _restart_train_config_identity,
)
from quantdata.dysanos.flex.training import TrainingConvergenceError
from quantdata.hedging.model_audit.legacy_rolling import (
    fit_joint_to_convergence,
    paper_geometry,
    same_origin_load_config,
)
from quantdata.hedging.model_audit.legacy_v2 import checked_native


def test_paper_geometry_matches_current_front_term_eta_model():
    geometry = paper_geometry()
    assert geometry.native_state_size == 3620
    np.testing.assert_allclose(geometry.model_expiries_y * 252., [1, 2, 3, 4, 5, *range(10, 71, 5)])


def test_cpu_restart_may_change_compilation_and_logging_not_the_objective():
    original = JointDirectTrainConfig()
    operational = replace(original, compile_sanos_solver=False, checkpoint_every=1, report_every=1)
    assert _restart_train_config_identity(original) == _restart_train_config_identity(operational)
    assert _restart_train_config_identity(original) != _restart_train_config_identity(
        replace(operational, dynamics_weight=.2))


@pytest.mark.parametrize("phase,just_load", [("complete", True), ("training", False)])
def test_same_origin_completed_stage_is_loaded_not_warm_started(tmp_path, phase, just_load):
    checkpoint = tmp_path / "native.pt"
    torch.save({"phase": phase}, checkpoint)
    training = FlexTrainConfig()
    selected = same_origin_load_config(training, checkpoint)
    assert selected.just_load == just_load
    assert not selected.warm_start
    assert not training.just_load


@pytest.mark.parametrize("day,smoothness,passed", [
    ("2024-04-02", .1, True), ("2024-04-01", .25, True), ("2024-04-01", .1, False)])
def test_native_inputs_reject_wrong_date_source_or_failed_reconstruction(tmp_path, day, smoothness, passed):
    path = tmp_path / "native.npz"
    np.savez(path, date=day, source_smoothness=smoothness, reconstruction_passed=passed,
        raw_u=np.zeros(4), decoded_densities=np.ones((1, 3)) / 3., support_strikes=np.array([.5, 1., 1.5]))
    with pytest.raises(ValueError, match="Unverified native market state"):
        checked_native(path, date(2024, 4, 1))


def test_native_input_is_checked_without_modifying_source(tmp_path):
    path = tmp_path / "native.npz"
    np.savez(path, date="2024-04-01", source_smoothness=.1, reconstruction_passed=True,
        raw_u=np.zeros(4), decoded_densities=np.ones((1, 3)) / 3., support_strikes=np.array([.5, 1., 1.5]))
    original = path.read_bytes()
    native, densities, support = checked_native(path, date(2024, 4, 1))
    assert native.shape == (4,)
    assert densities.sum() == pytest.approx(1.)
    assert support[1] == 1.
    assert path.read_bytes() == original


def test_joint_continuation_keeps_schedule_and_original_accuracy_threshold(tmp_path, monkeypatch):
    import json

    from quantdata.dysanos.flex import joint_direct

    checkpoint = tmp_path / "joint.pt"
    payload = {"train_config": {"num_epochs": 300}, "completed_epochs": 300, "schedule_horizon": 300,
        "scheduler_state_dict": {"_last_lr": [1e-5, 3e-6, 3e-5]},
        "history": [{"epoch": index + 1, "joint_objective": 1. - .001 * index, "dynamics_ramp": 1.}
            for index in range(300)]}
    torch.save(payload, checkpoint)
    calls = []

    def trainer(model, market, **kwargs):
        config = kwargs["train_config"]
        calls.append(config.num_epochs)
        assert config.convergence_relative_tolerance == .002
        assert kwargs["checkpoint_path"] == checkpoint
        if len(calls) == 1:
            raise TrainingConvergenceError("exact original convergence warning")
        return [], {"converged": True}

    monkeypatch.setattr(joint_direct, "fit_joint_direct_model", trainer)
    _, audit = fit_joint_to_convergence(None, (), dates=(), objective=None, training=JointDirectTrainConfig(),
        checkpoint=checkpoint, decoder_sha256="unchanged", environment=None)
    assert calls == [300, 400]
    assert audit["converged"]
    record = json.loads((tmp_path / "joint-continuations/epoch-300.json").read_text())
    assert record["exact_warning"] == "exact original convergence warning"
    assert record["original_schedule_horizon"] == 300
    assert torch.load(checkpoint, weights_only=False)["schedule_horizon"] == 300


def test_joint_continuation_does_not_accept_invalid_history(tmp_path, monkeypatch):
    from quantdata.dysanos.flex import joint_direct

    checkpoint = tmp_path / "joint.pt"
    torch.save({"train_config": {"num_epochs": 300}, "completed_epochs": 300, "history": []}, checkpoint)

    def trainer(*args, **kwargs):
        raise TrainingConvergenceError("nonfinite or insufficient history")

    monkeypatch.setattr(joint_direct, "fit_joint_direct_model", trainer)
    with pytest.raises(TrainingConvergenceError, match="insufficient history"):
        fit_joint_to_convergence(None, (), dates=(), objective=None, training=JointDirectTrainConfig(),
            checkpoint=checkpoint, decoder_sha256="unchanged", environment=None)


def test_paper_cpu_queue_never_waits_for_or_claims_gpu(tmp_path, monkeypatch):
    from quantdata.hedging.model_audit import legacy_queue

    calls = []
    monkeypatch.setattr(legacy_queue.sys, "argv", ["legacy_queue", "--root", str(tmp_path),
        "--reference", str(tmp_path), "--families", "dysanos1"])
    monkeypatch.setattr(legacy_queue, "origins", lambda _: ["2024-04-01"])
    monkeypatch.setattr(legacy_queue.signal, "signal", lambda *_: None)
    monkeypatch.setattr(legacy_queue, "run_owned", lambda command, **kwargs: calls.append((command, kwargs)))

    def unexpected_gpu_wait(*args, **kwargs):
        pytest.fail("CPU training must not wait for a GPU")

    monkeypatch.setattr(legacy_queue, "wait_gpu", unexpected_gpu_wait)
    legacy_queue.main()
    assert len(calls) == 2
    assert "--prepare-only" in calls[0][0]
    assert "--prepare-only" not in calls[1][0]
    for command, kwargs in calls:
        assert "gpuctl" not in command
        assert command[command.index("--device") + 1] == "cpu"
        assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["environment"]["JAX_PLATFORMS"] == "cpu"


def test_gromm_queue_keeps_managed_gpu_routing(tmp_path, monkeypatch):
    from quantdata.hedging.model_audit import legacy_queue

    calls, waits = [], []
    monkeypatch.setattr(legacy_queue.sys, "argv", ["legacy_queue", "--root", str(tmp_path),
        "--reference", str(tmp_path), "--families", "gromm1"])
    monkeypatch.setattr(legacy_queue, "origins", lambda _: ["2024-04-01"])
    monkeypatch.setattr(legacy_queue.signal, "signal", lambda *_: None)
    monkeypatch.setattr(legacy_queue, "run_owned", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(legacy_queue, "wait_gpu", lambda *args, **kwargs: waits.append(kwargs))
    legacy_queue.main()
    assert len(waits) == 1
    assert calls[1][0][:4] == ["gpuctl", "run", "--gpus", "1"]
    assert "--device" not in calls[1][0]
