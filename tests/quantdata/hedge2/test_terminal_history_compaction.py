"""History sharing changes actor work, never paths, cash flows or risk weights."""

from copy import deepcopy
from dataclasses import fields, replace
from datetime import date

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal import TerminalPaths
from quantdata.hedge2.terminal_policy import PolicyBook, TerminalPolicy, TerminalPolicyConfig
from quantdata.hedge2.terminal_torch import TensorTerminalPaths
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    make_terminal_evaluator,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_policy import sample_data


def branching_inputs(*, fee_preconditioning: bool = False, use_deep: bool = True):
    rng = np.random.default_rng(917)
    histories = np.array([[0, 1, 3], [0, 1, 3], [0, 1, 4], [0, 1, 4],
                          [0, 2, 5], [0, 2, 5], [0, 2, 6], [0, 2, 6]])
    costs = np.broadcast_to(np.array([0., .002, .003, .004]), (8, 3, 4)).copy()
    costs *= 1. + histories[:, :, None] / 10.
    terminated = np.zeros((8, 3), dtype=bool)
    terminated[:4, 0] = True
    live = np.ones((4, 4), dtype=bool)
    live[2:, 2] = False
    live[-1, 1:] = False
    paths = TerminalPaths(gains=rng.normal(0., .04, (8, 3, 4)), costs=costs,
        features=np.stack((histories / 10., histories**2 / 100.), axis=-1), history_ids=histories,
        live_contracts=live, terminate_after_step=terminated, claim_gain=rng.normal(0., .2, 8),
        weights=np.arange(1., 9.) / 36.)
    prices = np.broadcast_to(np.array([1., .3, .25, .4]), (8, 3, 4)).copy()
    prices *= 1. + histories[:, :, None] / 100.
    prices[:, 1:, 3] = 0.
    book = PolicyBook(prices=prices,
        asset_descriptors=np.array([[1., 0., 0., 0.], [0., 0., 1., .1],
                                    [0., 0., 0., -.05], [0., 0., 1., .2]]),
        initial_business_dtes=np.array([0., 3., 2., 3.]))
    env = BaseEnvironment(device="cpu", precision=64, debug=True)
    torch.manual_seed(918)
    policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(label="history-test", feature_count=2,
        fee_preconditioning=fee_preconditioning, use_deep=use_deep))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(0., .05)
    tensors = TensorTerminalPaths.from_paths(paths, env=env)
    policy.rebase_normalization(tensors.features, tensors.weights)
    return paths, book, policy, torch.tensor([.2, -.03, .04, .1], dtype=env.fdtype)


@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
@pytest.mark.parametrize("fee_preconditioning", [False, True])
@pytest.mark.parametrize("use_deep", [False, True])
def test_compaction_preserves_every_path_value_and_parameter_gradient(objective, fee_preconditioning, use_deep) -> None:
    paths, book, original, inventory = branching_inputs(fee_preconditioning=fee_preconditioning, use_deep=use_deep)
    evaluations, gradients = [], []
    for compact in (False, True):
        policy = deepcopy(original)
        buffers = {name: value.clone() for name, value in policy.named_buffers()}
        evaluate = make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
            book=book.tensor(policy.env), initial_inventory=inventory, objective=objective, risk_aversion=10.,
            target_side=-1, history_ids=paths.history_ids if compact else None)
        result, risk = evaluate()
        risk.value.backward()
        evaluations.append((result, risk))
        gradients.append({name: parameter.grad.clone() for name, parameter in policy.named_parameters()})
        for name, value in policy.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0., atol=0.)
    for value in fields(evaluations[0][0]):
        torch.testing.assert_close(getattr(evaluations[0][0], value.name),
            getattr(evaluations[1][0], value.name), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(evaluations[0][1].value, evaluations[1][1].value, rtol=1e-12, atol=1e-12)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-10, atol=1e-11)


def test_actor_rows_are_shared_but_every_future_path_still_enters_wealth(monkeypatch) -> None:
    paths, book, policy, inventory = branching_inputs()
    calls = []
    forward = policy.forward

    def counted(features, current_inventory, step, **kwargs):
        calls.append(len(features))
        return forward(features, current_inventory, step, **kwargs)

    monkeypatch.setattr(policy, "forward", counted)
    evaluate = make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
        book=book.tensor(policy.env), initial_inventory=inventory, objective="mean_variance", risk_aversion=10.,
        target_side=-1, history_ids=paths.history_ids)
    result, _ = evaluate()
    assert calls == [1, 2, 4]
    assert result.positions.shape == paths.gains.shape
    assert result.wealth.shape == (8,)
    assert result.wealth[0] != result.wealth[1]


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_one_optimizer_update_matches_the_full_actor(optimizer) -> None:
    paths, book, original, inventory = branching_inputs()
    models = []
    for compact in (False, True):
        policy = deepcopy(original)
        evaluate = make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
            book=book.tensor(policy.env), initial_inventory=inventory, objective="mean_variance", risk_aversion=10.,
            target_side=-1, history_ids=paths.history_ids if compact else None)
        fitter = (torch.optim.Adam(policy.parameters(), lr=1e-3) if optimizer == "adam" else
            torch.optim.LBFGS(policy.parameters(), lr=.1, max_iter=1, line_search_fn="strong_wolfe"))

        def closure():
            fitter.zero_grad(set_to_none=True)
            loss = evaluate()[1].value
            loss.backward()
            return loss

        fitter.step(closure)
        models.append(policy)
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[name], rtol=1e-10, atol=1e-11)


def test_compacted_whole_evaluation_can_use_the_compiler_without_changing_gradients() -> None:
    paths, book, original, inventory = branching_inputs(use_deep=False)
    gradients, risks = [], []
    for compiled in (False, True):
        policy = deepcopy(original)
        evaluate = make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
            book=book.tensor(policy.env), initial_inventory=inventory, objective="mean_variance", risk_aversion=10.,
            target_side=-1, history_ids=paths.history_ids, compile_policy=compiled,
            compile_evaluation=compiled, backend="aot_eager")
        risk = evaluate()[1].value
        risk.backward()
        risks.append(risk.detach())
        gradients.append([parameter.grad.clone() for parameter in policy.parameters()])
    torch.testing.assert_close(risks[0], risks[1], rtol=1e-12, atol=1e-12)
    for original_gradient, compiled_gradient in zip(*gradients, strict=True):
        torch.testing.assert_close(original_gradient, compiled_gradient, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("field", ["features", "costs", "prices"])
def test_approximately_shared_inputs_are_not_silently_merged(field) -> None:
    paths, book, policy, inventory = branching_inputs()
    if field == "prices":
        values = book.prices.copy()
        values[1, 0, 1] += 5e-13
        book = replace(book, prices=values)
    else:
        values = getattr(paths, field).copy()
        values[1, 0, 0] += 5e-13
        paths = replace(paths, **{field: values})
    with pytest.raises(ValueError, match="exactly equal predictable inputs"):
        make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
            book=book.tensor(policy.env), initial_inventory=inventory, objective="mean_variance", risk_aversion=10.,
            target_side=-1, history_ids=paths.history_ids)


@pytest.mark.parametrize("invalid", ["root", "merge", "float", "active"])
def test_invalid_history_information_is_rejected(invalid) -> None:
    paths, book, policy, inventory = branching_inputs()
    histories = paths.history_ids.copy()
    tensors = TensorTerminalPaths.from_paths(paths, env=policy.env)
    if invalid == "root":
        histories[1, 0] = 12
    elif invalid == "merge":
        histories[:, 2] = 7
    elif invalid == "float":
        histories = histories.astype(float)
    else:
        terminated = tensors.terminate_after_step.clone()
        terminated[0, 0] = False
        tensors = replace(tensors, terminate_after_step=terminated)
    with pytest.raises(ValueError):
        make_terminal_evaluator(policy=policy, paths=tensors, book=book.tensor(policy.env),
            initial_inventory=inventory, objective="mean_variance", risk_aversion=10., target_side=-1,
            history_ids=histories)


@pytest.mark.parametrize("module", [torch.nn.Dropout(.1), torch.nn.BatchNorm1d(2), torch.nn.RReLU()])
def test_stochastic_or_batch_dependent_modules_are_rejected(module) -> None:
    paths, book, policy, inventory = branching_inputs()
    policy.add_module("unsupported", module)
    with pytest.raises(ValueError, match="deterministic pointwise"):
        make_terminal_evaluator(policy=policy, paths=TensorTerminalPaths.from_paths(paths, env=policy.env),
            book=book.tensor(policy.env), initial_inventory=inventory, objective="mean_variance", risk_aversion=10.,
            target_side=-1, history_ids=paths.history_ids)


def test_compacted_lbfgs_restart_preserves_exact_state_and_history(tmp_path) -> None:
    paths, book = sample_data()
    training = TerminalTrainConfig(optimizer="lbfgs", compact_histories=True, maximum_updates=12,
        minimum_updates=1, check_every=2, stable_checks=100, lbfgs_learning_rate=.1)
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="compact-restart", feature_count=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    whole = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "whole")
    train_terminal_policy(**arguments, train_config=replace(training, maximum_updates=6), output_directory=tmp_path / "split")
    split = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "split")
    assert split.history == whole.history
    for name, value in whole.policy.state_dict().items():
        torch.testing.assert_close(split.policy.state_dict()[name], value, rtol=0., atol=0.)
    with pytest.raises(ValueError, match="numerical protocol"):
        train_terminal_policy(**arguments, train_config=replace(training, compact_histories=False),
            output_directory=tmp_path / "split")


def test_compaction_is_opt_in_and_requires_a_boolean() -> None:
    assert not TerminalTrainConfig().compact_histories
    with pytest.raises(ValueError, match="explicit boolean"):
        TerminalTrainConfig(compact_histories=1)
