from __future__ import annotations

import copy
import json
import signal
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from quantdata.gromm2 import hedging_policy as hedging
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@pytest.fixture(autouse=True)
def _cpu_test_resources(monkeypatch):
    monkeypatch.setattr(hedging, "check_resource_headroom", lambda: None)
    torch.set_num_threads(1)


def _config(**kwargs) -> hedging.HedgingPolicyConfig:
    values = dict(label="test-forward", feature_count=4, option_expiry_days=(1, 4), hidden_size=3,
                  deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    return hedging.HedgingPolicyConfig(**(values | kwargs))


def _model(config: hedging.HedgingPolicyConfig | None = None) -> hedging.HedgingPolicy:
    config = config or _config()
    return hedging.HedgingPolicy(BaseEnvironment(device="cpu", precision=64, debug=True), config=config,
                                 normalization_mean=np.zeros(config.feature_count), normalization_scale=np.ones(config.feature_count))


def _data(*, seed: int = 123, split: str = "train") -> hedging.ScenarioData:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(8, 4, 4)).astype(np.float32)
    prices = np.ones((8, 5, 3), dtype=np.float64)
    prices[:, :, 0] += np.cumsum(generator.normal(scale=0.02, size=(8, 5)), axis=1)
    prices[:, :, 1:] = 0.1 + generator.uniform(0.0, 0.02, size=(8, 5, 2))
    prices[:, 2:, 1] = 0.0
    payoff = np.abs(prices[:, 4, 0] / prices[:, 2, 0] - 1.0)
    return hedging.ScenarioData(features, prices, payoff, (1, 4),
        {"split": split, "config": {"paths": 8, "seed": seed}, "model": {"origin": "2025-08-29"}}, f"sha-{seed}", (0.99, 1.01), (False, True))


def _training(**kwargs) -> hedging.TrainConfig:
    values = dict(maximum_epochs=2, minimum_epochs=1, batch_size=4, evaluation_batch_size=4, cpu_threads=1, precision=64)
    return hedging.TrainConfig(**(values | kwargs))


@pytest.mark.parametrize("risk_aversion", [1.0, 10.0])
def test_power_exponential_values_and_gradients(risk_aversion: float) -> None:
    values = torch.tensor([-2.0, -0.1, 0.0, 0.1, 3.0], dtype=torch.float64, requires_grad=True)
    actual = hedging.power_exponential_utility(values, risk_aversion)
    expected = torch.where(values >= 0.0, (1.0 - torch.exp(-risk_aversion * values)) / risk_aversion,
                           values - 0.5 * risk_aversion * values.square())
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    derivative = torch.where(values >= 0.0, torch.exp(-risk_aversion * values), 1.0 - risk_aversion * values)
    torch.testing.assert_close(values.grad, derivative)


def test_power_exponential_inactive_branches_do_not_overflow() -> None:
    values = torch.tensor([-1.0e10, 0.0, 1.0e30], dtype=torch.float32, requires_grad=True)
    utility = hedging.power_exponential_utility(values, 10.0)
    utility.sum().backward()
    assert torch.isfinite(utility).all()
    assert torch.isfinite(values.grad).all()
    assert float(utility[-1].detach()) == pytest.approx(0.1)
    assert values.grad[1] == 1.0


def test_oce_translation_invariance_and_first_order_condition() -> None:
    pnl = np.asarray([-0.3, -0.1, 0.02, 0.05, 0.4])
    value, translation, _ = hedging.optimized_oce(pnl, 10.0)
    shifted, shifted_translation, _ = hedging.optimized_oce(pnl + 2.5, 10.0)
    assert shifted == pytest.approx(value + 2.5, abs=1.0e-12)
    assert shifted_translation == pytest.approx(translation + 2.5, abs=1.0e-12)
    centered = pnl - translation
    derivative = np.exp(-10.0 * np.maximum(centered, 0.0)) - 10.0 * np.minimum(centered, 0.0)
    assert derivative.mean() == pytest.approx(1.0, abs=1.0e-11)
    assert hedging.optimized_oce(np.full(4, 0.123), 1.0)[:2] == pytest.approx((0.123, 0.123))


def test_supplied_oce_translation_gradient_matches_definition() -> None:
    pnl = torch.tensor([-0.2, 0.1, 0.4], dtype=torch.float64)
    translation = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    value = hedging.oce_value(pnl, translation, 1.0)
    value.backward()
    centered = pnl - translation.detach()
    expected = 1.0 - torch.where(centered >= 0.0, torch.exp(-centered), 1.0 - centered).mean()
    torch.testing.assert_close(translation.grad, expected)


def test_training_feature_normalization_is_population_based() -> None:
    features = np.asarray([[[1.0, 7.0], [2.0, 7.0]], [[4.0, 7.0], [5.0, 7.0]]])
    mean, scale = hedging.feature_normalization(features, floor=1.0e-4)
    np.testing.assert_allclose(mean, [3.0, 7.0])
    np.testing.assert_allclose(scale, [np.sqrt(2.5), 1.0e-4])
    with pytest.raises(FloatingPointError):
        hedging.feature_normalization(features * np.nan, floor=1.0e-4)


def test_initial_policy_actions_gains_and_penalty_are_exactly_zero() -> None:
    data = _data()
    model = _model()
    output = model(torch.from_numpy(data.features), torch.from_numpy(data.prices))
    for value in output:
        assert torch.count_nonzero(value) == 0
    assert output.trading_gain.dtype == torch.float64


def test_future_features_cannot_change_past_positions() -> None:
    model = _model()
    with torch.no_grad():
        model.affine.I0.weight.fill_(0.03)
        model.deep.In[-1].weight.fill_(0.02)
    data = _data()
    features, prices = torch.from_numpy(data.features), torch.from_numpy(data.prices)
    original = model(features, prices)
    perturbed = features.clone()
    perturbed[:, 2:] += 100.0
    alternative = model(perturbed, prices)
    torch.testing.assert_close(original.positions[:, :2], alternative.positions[:, :2], rtol=0.0, atol=0.0)
    assert not torch.equal(original.positions[:, 2:], alternative.positions[:, 2:])


def test_both_action_heads_use_the_same_current_hidden_and_position_state() -> None:
    model = _model()
    width = model.config.feature_count + model.config.hidden_size + model.config.instrument_count
    assert model.affine.num_input == model.deep.num_input == width
    with torch.no_grad():
        model.affine.I0.weight.fill_(0.02)
        model.deep.I0.weight.fill_(0.02)
        model.deep.In[-1].weight.fill_(0.02)
    combined = torch.ones((2, width), dtype=torch.float64, requires_grad=True)
    for head in (model.affine, model.deep):
        derivative = torch.autograd.grad(head(combined).sum(), combined, retain_graph=True)[0]
        assert torch.all(derivative > 0.0)
    inputs = {"affine": [], "deep": []}
    affine_hook = model.affine.register_forward_pre_hook(lambda _, args: inputs["affine"].append(args[0].detach().clone()))
    deep_hook = model.deep.register_forward_pre_hook(lambda _, args: inputs["deep"].append(args[0].detach().clone()))
    data = _data()
    output = model(torch.from_numpy(data.features), torch.from_numpy(data.prices))
    affine_hook.remove()
    deep_hook.remove()
    for affine_input, deep_input in zip(inputs["affine"], inputs["deep"], strict=True):
        torch.testing.assert_close(affine_input, deep_input, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(inputs["affine"][0][:, model.config.feature_count:model.config.feature_count + model.config.hidden_size]) == 0
    assert torch.count_nonzero(inputs["affine"][1][:, model.config.feature_count:model.config.feature_count + model.config.hidden_size]) > 0
    torch.testing.assert_close(inputs["affine"][1][:, -1], output.positions[:, 0, -1] / model.config.position_cap)


def test_cash_settlement_is_paid_once_and_expired_positions_are_reset() -> None:
    model = _model()
    with torch.no_grad():
        model.affine.I0.bias[1] = model.config.position_cap * np.arctanh(2.0 / model.config.position_cap)
    features = torch.zeros((4, 4, 4), dtype=torch.float64)
    prices = torch.ones((4, 5, 3), dtype=torch.float64)
    prices[:, :, 1] = torch.tensor([0.1, 0.4, 0.0, 0.0, 0.0])
    received = []
    hook = model.affine.register_forward_pre_hook(lambda _, args: received.append(args[0].detach().clone()))
    output = model(features, prices)
    hook.remove()
    torch.testing.assert_close(output.trading_gain, torch.full((4,), 0.6, dtype=torch.float64))
    torch.testing.assert_close(output.positions[:, 0, 1], torch.full((4,), 2.0, dtype=torch.float64))
    assert torch.count_nonzero(output.positions[:, 1:, 1]) == 0
    assert torch.count_nonzero(received[1][:, -2]) == 0


def test_cap_and_soft_penalty_are_numerical_not_transaction_costs() -> None:
    model = _model()
    with torch.no_grad():
        model.affine.I0.bias.fill_(100.0)
    data = _data()
    output = model(torch.from_numpy(data.features), torch.from_numpy(data.prices))
    assert torch.all(output.positions.abs() <= model.config.position_cap)
    assert torch.all(output.penalty > 0.0)
    assert torch.all(output.cap_fraction > 0.0)
    expected_gain = (output.positions * torch.diff(torch.from_numpy(data.prices), dim=1)).sum(dim=(1, 2))
    torch.testing.assert_close(output.trading_gain, expected_gain)


def test_pair_standard_error_uses_complete_paths_not_time_observations() -> None:
    values = np.asarray([1.0, 3.0, 4.0, 6.0, 9.0, 11.0])
    result = hedging.pair_statistics(values)
    assert result["mean"] == pytest.approx(np.mean([2.0, 5.0, 10.0]))
    assert result["standard_error"] == pytest.approx(np.std([2.0, 5.0, 10.0], ddof=1) / np.sqrt(3))
    with pytest.raises(ValueError):
        hedging.pair_statistics(values[:-1])


def _write_scenarios(path: Path, data: hedging.ScenarioData, *, completed: int = 8) -> None:
    with h5py.File(path, "w") as stream:
        stream.attrs["identity"] = json.dumps(data.identity)
        stream.attrs["completed_paths"] = completed
        for name in ("features", "prices", "payoff", "option_expiry_days", "option_strikes", "option_is_call"):
            stream.create_dataset(name, data=getattr(data, name))


def test_scenario_loader_rejects_partial_wrong_split_or_expired_marks(tmp_path: Path) -> None:
    path = tmp_path / "train.h5"
    data = _data()
    _write_scenarios(path, data, completed=4)
    with pytest.raises(ValueError, match="incomplete"):
        hedging.load_scenarios(path, expected_split="train")
    _write_scenarios(path, data)
    with pytest.raises(ValueError, match="labelled"):
        hedging.load_scenarios(path, expected_split="validation")
    loaded = hedging.load_scenarios(path, expected_split="train")
    assert loaded.prices.dtype == np.float64
    assert len(loaded.sha256) == 64
    data.prices[:, 3, 1] = 0.01
    _write_scenarios(path, data)
    with pytest.raises(ValueError, match="expired"):
        hedging.load_scenarios(path, expected_split="train")


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    data, config = _data(), _config()
    normalization = hedging.feature_normalization(data.features, floor=config.normalization_floor)
    common = dict(data=data, config=config, claim_price=float(data.payoff.mean()), normalization=normalization)
    resumed_dir, full_dir = tmp_path / "resumed", tmp_path / "full"
    hedging.fit_policy(**common, directory=resumed_dir, train_config=_training(maximum_epochs=1))
    resumed, report = hedging.fit_policy(**common, directory=resumed_dir, train_config=_training(maximum_epochs=2))
    full, _ = hedging.fit_policy(**common, directory=full_dir, train_config=_training(maximum_epochs=2))
    resumed_checkpoint = torch.load(resumed_dir / "checkpoint.pt", weights_only=False)
    full_checkpoint = torch.load(full_dir / "checkpoint.pt", weights_only=False)
    for name in resumed.state_dict():
        torch.testing.assert_close(resumed_checkpoint["model"][name], full_checkpoint["model"][name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(resumed.state_dict()[name], full.state_dict()[name], rtol=0.0, atol=0.0)
    assert resumed_checkpoint["training_state"]["history"] == full_checkpoint["training_state"]["history"]
    assert report["stop_reason"] == "maximum_epoch_safety_limit_not_certified_converged"
    assert not report["converged"]
    read_only, loaded_report = hedging.fit_policy(**common, directory=resumed_dir, train_config=_training(), just_load=True)
    assert loaded_report["best_epoch"] == report["best_epoch"]
    for name in resumed.state_dict():
        torch.testing.assert_close(read_only.state_dict()[name], resumed.state_dict()[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("stop_stage", ["optimizer-update", "full-training-evaluation"])
def test_mid_epoch_stop_resumes_without_repeating_optimizer_updates(tmp_path: Path, monkeypatch, stop_stage: str) -> None:
    data, config = _data(), _config()
    kwargs = dict(data=data, config=config, train_config=_training(), claim_price=float(data.payoff.mean()),
        normalization=hedging.feature_normalization(data.features, floor=config.normalization_floor))
    interrupted_dir, full_dir = tmp_path / "interrupted", tmp_path / "uninterrupted"
    with monkeypatch.context() as patch:
        if stop_stage == "optimizer-update":
            original_step = torch.optim.Adam.step

            def stopping_step(*args, **step_kwargs):
                result = original_step(*args, **step_kwargs)
                signal.raise_signal(signal.SIGTERM)
                return result

            patch.setattr(torch.optim.Adam, "step", stopping_step)
        else:
            def stopping_evaluation(*args, **evaluation_kwargs):
                evaluation_kwargs["progress"].stop.set()
                raise InterruptedError("test stop during evaluation")

            patch.setattr(hedging, "_full_training_objective", stopping_evaluation)
        with pytest.raises(InterruptedError):
            hedging.fit_policy(**kwargs, directory=interrupted_dir)
    stopped = torch.load(interrupted_dir / "checkpoint.pt", weights_only=False)
    assert stopped["training_state"]["next_batch"] == (4 if stop_stage == "optimizer-update" else 8)
    hedging.fit_policy(**kwargs, directory=interrupted_dir)
    hedging.fit_policy(**kwargs, directory=full_dir)
    resumed = torch.load(interrupted_dir / "checkpoint.pt", weights_only=False)
    full = torch.load(full_dir / "checkpoint.pt", weights_only=False)
    for name in full["model"]:
        torch.testing.assert_close(resumed["model"][name], full["model"][name], rtol=0.0, atol=0.0)
    assert resumed["training_state"]["history"] == full["training_state"]["history"]


def test_full_training_plateau_reviews_learning_rate_before_convergence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hedging, "_full_training_objective", lambda *args, **kwargs: (1.0, 0.0))
    data, config = _data(), _config()
    _, report = hedging.fit_policy(data, directory=tmp_path / "plateau", config=config,
        train_config=_training(maximum_epochs=8, learning_rate=1.0e-3, minimum_learning_rate=5.0e-4,
            learning_rate_patience=2, convergence_patience=2), claim_price=float(data.payoff.mean()),
        normalization=hedging.feature_normalization(data.features, floor=config.normalization_floor))
    assert report["converged"]
    assert report["epoch"] == 3
    history = json.loads((tmp_path / "plateau" / "training_history.json").read_text())["epochs"]
    assert [row["learning_rate"] for row in history] == [1.0e-3, 1.0e-3, 5.0e-4]


def test_checkpoint_data_mismatch_raises_without_deleting_it(tmp_path: Path) -> None:
    data, config = _data(), _config()
    normalization = hedging.feature_normalization(data.features, floor=config.normalization_floor)
    kwargs = dict(directory=tmp_path / "logical", config=config, train_config=_training(maximum_epochs=1), claim_price=float(data.payoff.mean()), normalization=normalization)
    hedging.fit_policy(data, **kwargs)
    checkpoint = kwargs["directory"] / "checkpoint.pt"
    original = checkpoint.read_bytes()
    data.sha256 = "different-content"
    with pytest.raises(ValueError, match="identity changed"):
        hedging.fit_policy(data, **kwargs)
    assert checkpoint.read_bytes() == original


def test_frozen_evaluation_preserves_policy_and_optimizes_only_metric_translation() -> None:
    model = _model(_config(liability_units=1.0))
    before = copy.deepcopy(model.state_dict())
    data = _data(split="validation")
    metrics, scores = hedging.evaluate_policy(model, data, claim_price=0.02, batch_size=4)
    np.testing.assert_allclose(scores["pnl"], 0.02 - data.payoff)
    assert metrics["oce_improvement_over_unhedged"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["oce_improvement_over_unhedged_statistics"]["standard_error"] == pytest.approx(0.0, abs=1.0e-14)
    assert metrics["near_cap_absolute_position_threshold"] == 0.95 * model.config.position_cap
    for name in before:
        torch.testing.assert_close(model.state_dict()[name], before[name], rtol=0.0, atol=0.0)


def test_end_to_end_validation_is_opened_only_after_all_policies_are_frozen(tmp_path: Path, monkeypatch) -> None:
    _write_scenarios(tmp_path / "train.h5", _data())
    _write_scenarios(tmp_path / "validation.h5", _data(seed=124, split="validation"))
    events = []
    original_load, original_fit = hedging.load_scenarios, hedging.fit_policy

    def load(path, **kwargs):
        events.append(path.name)
        return original_load(path, **kwargs)

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        events.append(kwargs["config"].label)
        return result

    monkeypatch.setattr(hedging, "load_scenarios", load)
    monkeypatch.setattr(hedging, "fit_policy", fit)
    report = hedging.run_hedging_experiment(tmp_path, train_config=_training(maximum_epochs=1),
        risk_aversions=(1.0,), policy_config=_config())
    assert events[0] == "train.h5"
    assert events[-1] == "validation.h5"
    assert len(report["policies"]) == 2
    assert report["policies"][1]["training"]["initialization"] == "corresponding-empty-policy"
    assert "claim_minus_empty_hedge" in report["policies"][1]["validation_metrics"]
    training_mean = np.mean(_data().features, axis=(0, 1))
    for policy in report["policies"]:
        checkpoint = torch.load(Path(policy["directory"]) / "checkpoint.pt", weights_only=False)
        np.testing.assert_allclose(checkpoint["best_state"]["normalization_mean"].numpy(), training_mean, atol=1.0e-7)
    assert (tmp_path / "hedging_report.json").is_file()


@pytest.mark.parametrize("overrides", [{"position_cap": 0.0}, {"risk_aversion": 0.0}, {"feature_count": 0}, {"group": "../bad"}])
def test_invalid_policy_parameters_are_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_incompatible_user_version_is_preserved_before_cold_start(tmp_path: Path) -> None:
    data, config = _data(), _config()
    normalization = hedging.feature_normalization(data.features, floor=config.normalization_floor)
    kwargs = dict(directory=tmp_path / "versioned", train_config=_training(maximum_epochs=1),
        claim_price=float(data.payoff.mean()), normalization=normalization)
    hedging.fit_policy(data, config=config, **kwargs)
    with pytest.warns(UserWarning, match="version changed"):
        hedging.fit_policy(data, config=replace(config, user_version="2"), **kwargs)
    assert len(list(kwargs["directory"].glob("checkpoint-incompatible-*.pt"))) == 1


def test_resource_stop_enforces_own_process_graceful_deadline(monkeypatch) -> None:
    progress = hedging._Progress(label="resource-test", interval=60.0)
    waits, killed = [], []
    monkeypatch.setattr(progress.closed, "wait", lambda seconds: waits.append(seconds) or False)
    monkeypatch.setattr(hedging.os, "kill", lambda process, sig: killed.append((process, sig)))
    progress._resource_stop(RuntimeError("STOP resource pressure"))
    assert progress.stop.is_set() and progress.checkpoint_requested.is_set()
    assert waits == [60.0]
    assert killed == [(hedging.os.getpid(), signal.SIGKILL)]


def test_failed_checkpoint_save_is_not_retried_under_pressure(tmp_path: Path, monkeypatch) -> None:
    attempts = []

    def failed_save(*args, **kwargs):
        attempts.append(args[0])
        raise OSError("simulated storage failure")

    monkeypatch.setattr(hedging, "_atomic_torch_save", failed_save)
    data, config = _data(), _config()
    with pytest.raises(OSError, match="storage failure"):
        hedging.fit_policy(data, directory=tmp_path / "failed-save", config=config, train_config=_training(maximum_epochs=1),
            claim_price=float(data.payoff.mean()), normalization=hedging.feature_normalization(data.features, floor=config.normalization_floor))
    assert len(attempts) == 1


@pytest.mark.parametrize("visibility", [None, "", "  ", "-1", "none", "void"])
def test_cuda_request_without_allocation_never_probes_or_uses_gpu(monkeypatch, visibility: str | None) -> None:
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    calls = []
    for method in ("is_available", "device_count", "set_device", "manual_seed", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, method, lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(RuntimeError, match="gpuctl allocation"):
        hedging.training_environment(_training(device="cuda"))
    assert not calls


def test_cpu_environment_never_queries_cuda(monkeypatch) -> None:
    calls = []
    for method in ("is_available", "device_count", "set_device", "manual_seed", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, method, lambda *args, **kwargs: calls.append((args, kwargs)))
    env = hedging.training_environment(_training())
    assert env.device.type == "cpu"
    assert not calls


def test_allocated_cuda_environment_targets_only_visible_device_zero(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-by-gpuctl")
    calls = []
    sentinel = type("AllocatedEnvironment", (), {"device": torch.device("cuda", 0)})()

    def environment(**kwargs):
        calls.append(("environment", kwargs))
        return sentinel

    monkeypatch.setattr(hedging, "BaseEnvironment", environment)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("set_device", device)))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(("seed", seed)))
    env = hedging.training_environment(_training(device="cuda"))
    assert env is sentinel
    assert calls[0][1]["device"] == torch.device("cuda", 0)
    assert calls[1] == ("set_device", torch.device("cuda", 0))
    assert calls[2] == ("seed", _training().seed)


def test_device_is_operational_not_financial_checkpoint_identity(tmp_path: Path) -> None:
    data, config = _data(), _config()
    normalization = hedging.feature_normalization(data.features, floor=config.normalization_floor)
    directory = tmp_path / "device"
    _, report = hedging.fit_policy(data, directory=directory, config=config, train_config=_training(maximum_epochs=1),
        claim_price=float(data.payoff.mean()), normalization=normalization)
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=False)
    assert "device" not in checkpoint["identity"]["config"]
    assert checkpoint["torch_cuda_rng"] is None
    assert checkpoint["runtime_device"] == report["device"] == "cpu"
