"""Execution-only training options preserve cash, gradients and restart state."""

import signal
from copy import deepcopy
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal import evaluate_terminal_policy
from quantdata.hedge2.terminal_policy import PolicyBook, TerminalPolicy, TerminalPolicyConfig
from quantdata.hedge2.terminal_torch import TensorTerminalPaths
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    make_terminal_evaluator,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_policy import sample_data
from tests.quantdata.hedge2.test_terminal_tree import two_step_tree


@pytest.fixture
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("objective,risk", [("mean_variance", 10.), ("mean_variance", np.inf),
    ("power_exponential", 10.)])
@pytest.mark.parametrize("ties", [False, True])
def test_compiled_evaluation_preserves_wealth_costs_positions_and_all_gradients(
        objective, risk, ties, reset_compiler) -> None:
    original = two_step_tree()
    gains = np.repeat(original.gains * .05, 2, axis=2)
    gains[:, 1, 1] = 0.
    stops = np.zeros((4, 2), dtype=bool)
    stops[:2, 0] = True
    paths = replace(original, gains=gains, costs=np.full_like(gains, .002),
        live_contracts=np.array([[True, True], [True, False], [False, False]]),
        terminate_after_step=stops, claim_gain=original.claim_gain * .1)
    prices = np.ones_like(gains)
    prices[:, 0, 1] = .1
    prices[:, 1, 1] = 0.
    book = PolicyBook(prices=prices, asset_descriptors=np.array([[1., 0., 0., 0.], [0., 0., 1., -.1]]),
        initial_business_dtes=np.array([0., 1.]))
    book.validate_histories(paths)
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    torch.manual_seed(145)
    eager_policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(label="compiled-equality",
        feature_count=1, fee_preconditioning=True))
    if not ties:
        with torch.no_grad():
            for parameter in eager_policy.parameters():
                parameter.normal_(0., .02)
    compiled_policy = deepcopy(eager_policy)
    initial = np.zeros(2) if ties else np.array([.1, -.2])
    tensor_paths, tensor_book = TensorTerminalPaths.from_paths(paths, env=env), book.tensor(env)
    arguments = dict(paths=tensor_paths, book=tensor_book, initial_inventory=torch.tensor(initial),
        objective=objective, risk_aversion=risk, target_side=-1)
    eager = make_terminal_evaluator(policy=eager_policy, **arguments)
    compiled = make_terminal_evaluator(policy=compiled_policy, **arguments,
        compile_evaluation=True, backend="aot_eager")
    expected, expected_risk = eager()
    actual, actual_risk = compiled()
    for name in ("wealth", "hedge_gain", "entry_cost", "positions", "step_gains", "step_costs"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=1e-11, rtol=1e-10)
    for name in ("value", "mean_wealth", "std_wealth", "cash_shift", "cash_first_order_residual"):
        expected_value, actual_value = getattr(expected_risk, name), getattr(actual_risk, name)
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(actual_value, expected_value, atol=1e-11, rtol=1e-10)
    expected_risk.value.backward()
    actual_risk.value.backward()
    for (name, parameter), (compiled_name, compiled_parameter) in zip(
            eager_policy.named_parameters(), compiled_policy.named_parameters(), strict=True):
        assert name == compiled_name
        torch.testing.assert_close(compiled_parameter.grad, parameter.grad, atol=1e-10, rtol=1e-9)
    with torch.no_grad():
        def numpy_actor(features, inventory, step):
            return eager_policy(torch.tensor(features, dtype=env.fdtype), torch.tensor(inventory, dtype=env.fdtype),
                step, book=tensor_book, costs=tensor_paths.costs[:, step]).numpy()

        replay = evaluate_terminal_policy(paths=paths, initial_inventory=initial,
            policy=numpy_actor, target_side=-1, position_cap=None)
    np.testing.assert_allclose(actual.wealth.detach().numpy(), replay.wealth, atol=1e-12)


def test_compiled_evaluation_keeps_nonfinite_runtime_rejection(reset_compiler) -> None:
    paths, book = sample_data()
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(label="compiled-finite", feature_count=1))
    tensor_paths = TensorTerminalPaths.from_paths(paths, env=env)
    evaluate = make_terminal_evaluator(policy=policy, paths=tensor_paths, book=book.tensor(env),
        initial_inventory=torch.zeros(1, dtype=env.fdtype), objective="mean_variance", risk_aversion=10.,
        target_side=-1, compile_evaluation=True, backend="aot_eager")
    evaluate()
    with torch.no_grad():
        policy.affine.I0.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        evaluate()


@pytest.mark.parametrize("seconds", [0., -1., np.nan, np.inf, True, "slow"])
def test_invalid_checkpoint_cadence_is_rejected(seconds) -> None:
    with pytest.raises(ValueError, match="Checkpoint cadence"):
        TerminalTrainConfig(checkpoint_seconds=seconds)


def test_full_evaluation_compilation_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit compilation flag"):
        TerminalTrainConfig(compile_evaluation=True)
    assert not TerminalTrainConfig().compile_evaluation
    assert TerminalTrainConfig().checkpoint_seconds is None


def test_checkpoint_cadence_changes_only_writes_not_training_or_optimizer_state(monkeypatch, tmp_path) -> None:
    import quantdata.hedge2.terminal_train as module

    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="checkpoint-cadence", feature_count=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    training = TerminalTrainConfig(maximum_updates=4, minimum_updates=1, check_every=1, stable_checks=100)
    writes = []
    original = module._atomic_checkpoint

    def save(path, values):
        writes.append(int(values["updates"]))
        original(path, values)

    monkeypatch.setattr(module, "_atomic_checkpoint", save)
    reference = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "legacy")
    assert writes == [1, 2, 3, 4, 4]
    writes.clear()
    candidate = train_terminal_policy(**arguments, train_config=replace(training, checkpoint_seconds=3600.),
        output_directory=tmp_path / "timed")
    assert writes == [4]
    assert candidate.history == reference.history
    for name, value in candidate.policy.state_dict().items():
        torch.testing.assert_close(value, reference.policy.state_dict()[name], rtol=0., atol=0.)
    expected = torch.load(reference.checkpoint_path, map_location="cpu", weights_only=True)
    actual = torch.load(candidate.checkpoint_path, map_location="cpu", weights_only=True)
    for field in ("model", "best_model", "optimizer", "scheduler", "torch_rng"):
        def compare(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0., atol=0.)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                assert len(left) == len(right)
                for first, second in zip(left, right, strict=True):
                    compare(first, second)
            else:
                assert left == right

        compare(actual[field], expected[field])


def test_long_periodic_cadence_still_saves_on_graceful_stop_and_resumes(tmp_path) -> None:
    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="timed-resume", feature_count=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    training = TerminalTrainConfig(maximum_updates=8, minimum_updates=1, check_every=1,
        stable_checks=100, checkpoint_seconds=3600.)
    whole = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "whole")
    notified = []

    def progress(message):
        if not notified:
            notified.append(message)
            signal.raise_signal(signal.SIGTERM)

    stopped = train_terminal_policy(**arguments, train_config=training,
        output_directory=tmp_path / "split", progress=progress)
    assert stopped.status == "checkpointed_stop"
    assert stopped.updates < whole.updates
    saved = torch.load(stopped.checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["status"] == "checkpointed_stop"
    resumed = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "split")
    assert resumed.updates == whole.updates
    assert resumed.history == whole.history
    for name, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, whole.policy.state_dict()[name], rtol=0., atol=0.)


def test_elapsed_checkpoint_timer_and_forced_stable_save(monkeypatch, tmp_path) -> None:
    import quantdata.hedge2.terminal_train as module

    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="checkpoint-clock", feature_count=1, use_deep=False),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    clock = [0.]
    writes = []
    original = module._atomic_checkpoint

    def save(path, values):
        writes.append((int(values["updates"]), values["status"]))
        original(path, values)

    def progress(message):
        clock[0] += 2.

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module, "_atomic_checkpoint", save)
    training = TerminalTrainConfig(maximum_updates=4, minimum_updates=1, check_every=1,
        stable_checks=100, checkpoint_seconds=1., progress_seconds=1.)
    clock[0] = 2.
    train_terminal_policy(**arguments, train_config=training,
        output_directory=tmp_path / "timer", progress=progress)
    assert writes == [(2, "training_limit"), (3, "training_limit"),
        (4, "training_limit"), (4, "training_limit")]
    writes.clear()
    stable = replace(training, maximum_updates=3, minimum_updates=1, stable_checks=1,
        checkpoint_seconds=3600., relative_loss_tolerance=1e6,
        dollar_action_tolerance=1e6, gradient_tolerance=1e6)
    fit = train_terminal_policy(**arguments, train_config=stable,
        output_directory=tmp_path / "stable")
    assert fit.status == "training_stable"
    assert writes == [(1, "training_stable"), (1, "training_stable")]


def test_compiled_training_updates_preserve_checkpoint_resume(monkeypatch, tmp_path, reset_compiler) -> None:
    import quantdata.hedge2.terminal_train as module

    original = module.make_terminal_evaluator

    def aot_evaluate(**arguments):
        return original(**arguments, backend="aot_eager")

    monkeypatch.setattr(module, "make_terminal_evaluator", aot_evaluate)
    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="compiled-resume", feature_count=1, use_deep=False),
        env=BaseEnvironment(device="cpu", precision=64, debug=False), objective="power_exponential",
        risk_aversion=10., target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    training = TerminalTrainConfig(maximum_updates=4, minimum_updates=1, check_every=1,
        stable_checks=100, checkpoint_seconds=3600., compile=True, compile_evaluation=True)
    eager = train_terminal_policy(**arguments,
        train_config=replace(training, compile=False, compile_evaluation=False),
        output_directory=tmp_path / "eager")
    whole = train_terminal_policy(**arguments, train_config=training,
        output_directory=tmp_path / "compiled")
    for name, value in whole.policy.state_dict().items():
        torch.testing.assert_close(value, eager.policy.state_dict()[name], rtol=1e-9, atol=1e-11)
    notified = []

    def progress(message):
        if not notified:
            notified.append(message)
            signal.raise_signal(signal.SIGTERM)

    stopped = train_terminal_policy(**arguments, train_config=training,
        output_directory=tmp_path / "split", progress=progress)
    assert stopped.status == "checkpointed_stop"
    resumed = train_terminal_policy(**arguments, train_config=training,
        output_directory=tmp_path / "split")
    assert resumed.history == whole.history
    for name, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, whole.policy.state_dict()[name], rtol=0., atol=0.)
