from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.gromm2 import joint_hedging as joint
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@pytest.fixture(autouse=True)
def _resources(monkeypatch):
    monkeypatch.setattr(joint, "check_resource_headroom", lambda: None)
    torch.set_num_threads(1)


def _config(**kwargs) -> joint.OptionsPolicyConfig:
    values = dict(label="joint-test", feature_count=3, option_expiry_days=(1, 3), hidden_size=4,
                  deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    return joint.OptionsPolicyConfig(**(values | kwargs))


def _model(config=None) -> joint.JointOptionsHedge:
    return joint.JointOptionsHedge(BaseEnvironment(device="cpu", precision=64, debug=True),
        config=config or _config(), normalization=(np.zeros(3), np.ones(3)))


def _inputs():
    features = torch.zeros((4, 4, 3), dtype=torch.float64)
    prices = torch.ones((4, 5, 3), dtype=torch.float64)
    prices[:, :, 0] = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
    prices[:, :, 1] = torch.tensor([0.1, 0.3, 0.0, 0.0, 0.0])
    prices[:, :, 2] = torch.tensor([0.2, 0.25, 0.3, 0.4, 0.0])
    return features, prices


def test_options_only_has_no_spot_action_and_shared_memory_has_separate_heads() -> None:
    model = _model()
    assert model.config.instrument_count == 2
    assert model.empty.memory is model.claim.memory
    assert model.empty.affine is not model.claim.affine
    assert model.empty.deep is not model.claim.deep
    output = model(*_inputs())
    assert output.empty.positions.shape == (4, 4, 2)
    assert not torch.count_nonzero(output.empty.positions)
    assert len(list(model.parameters())) == len({id(value) for value in model.parameters()})


def test_turnover_cost_charges_entry_but_not_cash_settlement_or_expiry_reset() -> None:
    model = _model()
    with torch.no_grad():
        model.empty.affine.I0.bias[0] = 10.0 * np.arctanh(0.2)
    output = model(*_inputs()).empty
    torch.testing.assert_close(output.gross_gain, torch.full((4,), 0.4, dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.full((4,), 2.0e-5, dtype=torch.float64))
    torch.testing.assert_close(output.trading_gain, output.gross_gain - 2.0e-5)
    assert torch.isfinite(output.penalty).all()


def test_both_portfolios_supply_gradients_to_the_same_recurrent_weights() -> None:
    model = _model()
    with torch.no_grad():
        model.empty.affine.I0.weight.fill_(0.02)
        model.claim.affine.I0.weight.fill_(0.03)
    features, prices = _inputs()
    features[:, :, 0] = torch.arange(4, dtype=torch.float64)
    output = model(features, prices)
    loss, _ = joint.joint_objective(model, output, torch.full((4,), 0.02), torch.ones(4))
    loss.backward()
    assert model.empty.memory.weight_ih.grad is not None
    assert torch.count_nonzero(model.empty.memory.weight_ih.grad)
    assert model.empty.translation.grad is not None and model.claim.translation.grad is not None


def test_weighted_oce_uniform_minibatch_gradients_use_global_probability_mass() -> None:
    pnl = torch.tensor([-0.2, 0.1, 0.4, 0.8], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.05, 0.15, 0.3, 0.5], dtype=torch.float64)
    translation = torch.tensor(0.0, dtype=torch.float64)
    full = joint.weighted_oce(pnl, translation, 1.0, 4.0 * weights)
    parts = sum(joint.weighted_oce(pnl[part], translation, 1.0, 4.0 * weights[part]) for part in (slice(0, 2), slice(2, 4))) / 2.0
    torch.testing.assert_close(full, parts)
    expected = (weights * joint.power_exponential_utility(pnl, 1.0)).sum()
    torch.testing.assert_close(full, expected)


def test_risk_sign_and_indifference_price_are_cash_translation_consistent() -> None:
    weights = np.full(8, 1.0 / 8.0)
    zero_risk, _ = joint.optimized_weighted_risk(np.zeros(8), weights, 1.0)
    claim_risk, _ = joint.optimized_weighted_risk(np.full(8, -0.02), weights, 1.0)
    assert claim_risk - zero_risk == pytest.approx(0.02)


def test_high_risk_aversion_has_finite_joint_loss_and_gradients() -> None:
    model = _model(_config(risk_aversion=100.0))
    output = model(*_inputs())
    loss, _ = joint.joint_objective(model, output, torch.full((4,), 0.02), torch.ones(4))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_cli_passes_requested_risk_aversion(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def run_experiment(directory, **kwargs):
        captured.update(directory=directory, **kwargs)

    monkeypatch.setattr(joint, "run_joint_experiment", run_experiment)
    monkeypatch.setattr("sys.argv", ["joint_hedging", "--run-directory", str(tmp_path), "--risk-aversions", "100"])
    joint.main()
    assert captured["directory"] == tmp_path
    assert captured["risk_aversions"] == (100.0,)


def _data(seed=1, split="train"):
    generator = np.random.default_rng(seed)
    features, prices = _inputs()
    features = generator.normal(size=features.shape)
    prices = prices.numpy().copy()
    prices[:, :, 0] += generator.normal(scale=0.005, size=prices.shape[:2])
    return joint.ScenarioData(features, prices, np.full(4, 0.02), (1, 3),
        {"model": {"test": 1}, "split": split}, f"sha-{seed}", (0.99, 1.01), (False, True))


def _train_config(**overrides):
    defaults = dict(device="cpu", precision=64, batch_size=2, evaluation_batch_size=2,
                    maximum_epochs=2, plot_every_epochs=1, cpu_threads=1)
    return joint.JointTrainConfig(**(defaults | overrides))


def test_joint_training_saves_train_and_validation_losses_and_prices(tmp_path: Path) -> None:
    weights = np.full(4, 0.25)
    result = joint.train_joint(_data(), _data(seed=2, split="validation"), weights, weights,
        directory=tmp_path, config=_config(), train_config=_train_config())
    history = json.loads((tmp_path / "training_history.json").read_text())["epochs"]
    assert len(history) == 3
    assert history[0]["zero_action_baseline"]
    assert all(np.isfinite(row["validation_objective"]) for row in history)
    assert (tmp_path / "training_loss.png").exists()
    assert result["validation"]["DH_price"] == pytest.approx(
        result["validation"]["claim"]["DH"] - result["validation"]["empty"]["DH"])
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert checkpoint["identity"]["training_weights_sha256"] == checkpoint["identity"]["validation_weights_sha256"]
    assert result["training"]["selection"].startswith("Best joint validation loss")


def test_validation_plateau_controls_learning_rate_and_early_stopping(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(joint, "evaluate_joint", lambda *args, **kwargs: {"objective": 1.0, "penalty": 0.0,
        "DH_price": 0.02, "economic_objective": 1.0, "empty": {"loss_at_training_translation": 0.0},
        "claim": {"loss_at_training_translation": 1.0}})
    weights = np.full(4, 0.25)
    result = joint.train_joint(_data(), _data(seed=2, split="validation"), weights, weights,
        directory=tmp_path, config=_config(), train_config=_train_config(maximum_epochs=8,
            learning_rate_patience=2, convergence_patience=3))
    assert result["training"]["early_stopped"]
    assert result["training"]["epoch"] == 3
    assert result["training"]["best_epoch"] == 0
    history = json.loads((tmp_path / "training_history.json").read_text())["epochs"]
    assert history[2]["learning_rate"] < history[1]["learning_rate"]


def test_joint_mid_epoch_checkpoint_resume_reproduces_uninterrupted_weights(tmp_path: Path, monkeypatch) -> None:
    weights = np.full(4, 0.25)
    kwargs = dict(train=_data(), validation=_data(seed=2, split="validation"), train_weights=weights,
        validation_weights=weights, config=_config(), train_config=_train_config())
    original_step = torch.optim.Adam.step

    def stopping_step(*args, **step_kwargs):
        result = original_step(*args, **step_kwargs)
        signal.raise_signal(signal.SIGINT)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", stopping_step)
        with pytest.raises(InterruptedError):
            joint.train_joint(**kwargs, directory=tmp_path / "resumed")
    joint.train_joint(**kwargs, directory=tmp_path / "resumed")
    joint.train_joint(**kwargs, directory=tmp_path / "uninterrupted")
    resumed = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    full = torch.load(tmp_path / "uninterrupted" / "checkpoint.pt", weights_only=False)
    for name in full["model"]:
        torch.testing.assert_close(resumed["model"][name], full["model"][name], rtol=0.0, atol=0.0)
    assert resumed["training_state"]["history"] == full["training_state"]["history"]


def test_original_q_is_default_and_never_reads_preconditioning_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(joint, "load_scenarios", lambda path, expected_split: _data(
        seed=1 if expected_split == "train" else 2, split=expected_split))

    def forbidden_weights(*args, **kwargs):
        raise AssertionError("Original Q must not load entropy weights")

    monkeypatch.setattr(joint, "load_path_weights", forbidden_weights)
    calls = []

    def fit(train, validation, train_weights, validation_weights, **kwargs):
        np.testing.assert_array_equal(train_weights, np.full(4, 0.25))
        np.testing.assert_array_equal(validation_weights, np.full(4, 0.25))
        assert not kwargs["preconditioned"]
        assert kwargs["train_config"].batch_size == 256
        calls.append(kwargs)
        return {"test": True}

    monkeypatch.setattr(joint, "train_joint", fit)
    report = joint.run_joint_experiment(tmp_path, risk_aversions=(1.0,))
    assert len(calls) == 1
    assert not report["preconditioned"]


def test_nonuniform_weights_cannot_be_used_when_preconditioning_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicitly enabling"):
        joint.train_joint(_data(), _data(seed=2), np.asarray([0.1, 0.1, 0.4, 0.4]), np.full(4, 0.25),
            directory=tmp_path, config=_config(), train_config=_train_config())


@pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="requires an allocated GPU")
def test_allocated_gpu_joint_forward_backward_preserves_costs_and_option_geometry() -> None:
    env = joint.training_environment(joint.JointTrainConfig(device="cuda"))
    model = joint.JointOptionsHedge(env, config=_config(), normalization=(np.zeros(3), np.ones(3)))
    features, prices = _inputs()
    features, prices = features.to(device=env.device), prices.to(device=env.device)
    output = model(features, prices)
    loss, _ = joint.joint_objective(model, output, torch.full((4,), 0.02, device=env.device), torch.ones(4, device=env.device))
    loss.backward()
    assert torch.isfinite(loss)
    assert output.empty.positions.shape[-1] == 2
    assert model.claim.affine.I0.weight.grad.device.type == "cuda"
