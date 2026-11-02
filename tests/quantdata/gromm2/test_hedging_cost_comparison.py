from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import quantdata.gromm2.hedging_cost_comparison as cost_comparison
from quantdata.gromm2.hedging_cost_comparison import (
    frozen_market_model,
    frozen_policy_fingerprint,
    transaction_cost_quotes,
)
from quantdata.gromm2.hedging_policy import ScenarioData
from quantdata.gromm2.market_hedging import (
    MarketBookConfig,
    MarketBookPolicy,
    UnitTurnoverBookConfig,
    UnitTurnoverBookPolicy,
)
from quantdata.gromm2.portfolio_hedging import LIABILITIES, IndependentHedges
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


def test_constant_costs_shift_risks_and_quotes_by_exact_cash_amounts():
    payoff = np.linspace(0.01, 0.05, 8)
    gains = {name: np.zeros(8) for name in LIABILITIES}
    costs = {name: np.full(8, value) for name, value in {"zero": 0.001, "short": 0.003, "long": 0.002}.items()}
    result = transaction_cost_quotes(payoff, gains, costs, risk_aversion=100.0)
    before, after = (result[key]["statistics"] for key in ("with_costs", "costs_removed"))
    assert before["ask"]["mean"] - after["ask"]["mean"] == pytest.approx(0.002)
    assert after["bid"]["mean"] - before["bid"]["mean"] == pytest.approx(0.001)
    assert after["ask"]["standard_error"] == pytest.approx(before["ask"]["standard_error"])
    assert after["bid"]["standard_error"] == pytest.approx(before["bid"]["standard_error"])


def test_random_path_costs_are_not_replaced_by_their_average():
    payoff = np.linspace(0.0, 0.1, 8)
    gains = {name: np.zeros(8) for name in LIABILITIES}
    costs = {name: np.zeros(8) for name in LIABILITIES}
    costs["short"] = 0.05 * payoff
    exact = transaction_cost_quotes(payoff, gains, costs, risk_aversion=100.0)
    averaged = transaction_cost_quotes(payoff, gains,
        {name: np.full(8, value.mean()) for name, value in costs.items()}, risk_aversion=100.0)
    assert abs(exact["costs_removed"]["statistics"]["ask"]["mean"]
        - averaged["costs_removed"]["statistics"]["ask"]["mean"]) > 1e-5


def test_negative_or_incomplete_cost_vectors_are_rejected():
    gains = {name: np.zeros(8) for name in LIABILITIES}
    with pytest.raises(ValueError, match="all three"):
        transaction_cost_quotes(np.ones(8), gains, {}, risk_aversion=100.0)
    costs = {name: np.zeros(8) for name in LIABILITIES}
    costs["long"][0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        transaction_cost_quotes(np.ones(8), gains, costs, risk_aversion=100.0)


@pytest.mark.parametrize("unit_costs", [False, True])
def test_best_weights_normalization_and_actions_are_loaded_unchanged_without_optimizer(unit_costs):
    config_class = UnitTurnoverBookConfig if unit_costs else MarketBookConfig
    policy_class = UnitTurnoverBookPolicy if unit_costs else MarketBookPolicy
    config = config_class(label="frozen-cost-test", kind="factors", feature_count=3,
        option_expiry_days=(20,) * 3, option_start_days=(0,) * 3, basket_weights=((1.0,),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    env = BaseEnvironment(device="cpu", precision=64, debug=False)
    source = IndependentHedges(env, config=config,
        normalization=(np.ones(3), np.full(3, 2.0)), policy_class=policy_class)
    with torch.no_grad():
        for index, policy in enumerate(source.policies.values()):
            policy.affine.I0.bias.fill_(0.1 * (index + 1))
    checkpoint = dict(identity=dict(config=asdict(config), code_version=config.version.unique_id48,
        user_version=config.user_version), report=dict(train_config=dict(precision=64)),
        state=dict(policies={name: dict(best_state=copy.deepcopy(policy.state_dict()))
            for name, policy in source.policies.items()}))
    restored = frozen_market_model(checkpoint)
    features = torch.zeros((4, 20, 3), dtype=torch.float64)
    prices = torch.ones((4, 21, 3), dtype=torch.float64)
    with torch.no_grad():
        for name, policy in restored.policies.items():
            assert not policy.training
            torch.testing.assert_close(policy(features, prices).positions, source.policies[name](features, prices).positions,
                rtol=0.0, atol=0.0)
            torch.testing.assert_close(policy.normalization_mean, torch.ones(3, dtype=torch.float64))
            torch.testing.assert_close(policy.normalization_scale, torch.full((3,), 2.0, dtype=torch.float64))


@pytest.fixture
def frozen_cost_run(tmp_path: Path, monkeypatch):
    config = UnitTurnoverBookConfig(label="resumable-cost-test", kind="factors", feature_count=3,
        option_expiry_days=(20,) * 3, option_start_days=(0,) * 3, basket_weights=((1.0,),),
        hidden_size=4, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))
    model = IndependentHedges(BaseEnvironment(device="cpu", precision=64, debug=False), config=config,
        normalization=(np.zeros(3), np.ones(3)), policy_class=UnitTurnoverBookPolicy)
    checkpoint = dict(identity=dict(config=asdict(config), code_version=config.version.unique_id48,
        user_version=config.user_version, training_scenarios_sha256="train-source", validation_scenarios_sha256="validation-source"),
        model=model.state_dict(), report=dict(train_config=dict(precision=64), scenario_model=dict(simulation_case="black0"),
            stop_reason="validation_plateau"),
        state=dict(epoch=7, policies={name: dict(best_epoch=1, best_state=copy.deepcopy(policy.state_dict()))
            for name, policy in model.policies.items()}))
    directory = tmp_path / "policy"
    directory.mkdir()
    torch.save(checkpoint, directory / "checkpoint.pt")
    data = {split: ScenarioData(features=np.zeros((8, 20, 3)), prices=np.ones((8, 21, 3)),
        payoff=np.linspace(0.0, 0.02, 8), option_expiry_days=config.option_expiry_days,
        identity=dict(split=split), sha256=f"{split}-source") for split in ("train", "validation")}
    monkeypatch.setattr(cost_comparison, "check_resource_headroom", lambda: None)
    monkeypatch.setattr(cost_comparison, "load_market_scenarios", lambda path, *, split, **kwargs: (data[split], {}))
    return directory, data


def test_frozen_fingerprint_ignores_training_progress_but_not_selected_weights(frozen_cost_run):
    directory, _ = frozen_cost_run
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected = frozen_policy_fingerprint(checkpoint)
    changed = copy.deepcopy(checkpoint)
    changed["state"]["epoch"] = 100
    changed["report"]["stop_reason"] = "validation_plateau"
    changed["optimizers"] = {"training_progress": 123}
    for policy in changed["state"]["policies"].values():
        policy["best_epoch"] = 99
        policy["best_state"] = dict(reversed(list(policy["best_state"].items())))
    assert frozen_policy_fingerprint(changed) == expected
    changed["state"]["policies"]["short"]["best_state"]["normalization_mean"][0] += .01
    assert frozen_policy_fingerprint(changed) != expected


@pytest.mark.parametrize("field", ["precision", "case", "source", "version"])
def test_frozen_fingerprint_includes_every_valuation_identity(frozen_cost_run, field):
    directory, _ = frozen_cost_run
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected = frozen_policy_fingerprint(checkpoint)
    if field == "precision":
        checkpoint["report"]["train_config"]["precision"] = 32
    elif field == "case":
        checkpoint["report"]["scenario_model"]["simulation_case"] = "black5"
    elif field == "source":
        checkpoint["identity"]["training_scenarios_sha256"] = "different-source"
    else:
        checkpoint["identity"]["user_version"] = "different-version"
    assert frozen_policy_fingerprint(checkpoint) != expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None])
def test_frozen_fingerprint_rejects_invalid_selected_tensors(frozen_cost_run, invalid):
    directory, _ = frozen_cost_run
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = checkpoint["state"]["policies"]["zero"]["best_state"]
    if invalid is None:
        state["translation"] = None
    else:
        state["translation"].fill_(invalid)
    with pytest.raises(ValueError, match="finite tensor"):
        frozen_policy_fingerprint(checkpoint)


def test_frozen_cost_resume_skips_completed_split_and_keeps_checkpoint_immutable(frozen_cost_run, tmp_path: Path, monkeypatch):
    directory, data = frozen_cost_run
    original = (directory / "checkpoint.pt").read_bytes()
    output = tmp_path / "costs"
    interruption = True
    calls = []
    iterator_class = cost_comparison.MarketBatchIterator

    def load(path, *, split, **kwargs):
        if split == "train" and interruption:
            raise InterruptedError("test interruption between complete splits")
        return data[split], {}

    def iterator(scenarios, **kwargs):
        calls.append(scenarios.identity["split"])
        return iterator_class(scenarios, **kwargs)

    monkeypatch.setattr(cost_comparison, "load_market_scenarios", load)
    monkeypatch.setattr(cost_comparison, "MarketBatchIterator", iterator)
    with pytest.raises(InterruptedError, match="test interruption"):
        cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
            batch_size=4, cpu_threads=1)
    partial = json.loads((output / "cost_comparison.json").read_text())
    assert partial["complete"] is False
    assert set(partial["splits"]) == {"validation"}
    validation_archive = (output / "validation_frozen_gains.npz").read_bytes()
    interruption = False
    result = cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
        batch_size=8, cpu_threads=1, resume=True)
    assert result["complete"] is True
    assert set(result["splits"]) == {"train", "validation"}
    assert calls == ["validation", "train"]
    assert (output / "validation_frozen_gains.npz").read_bytes() == validation_archive
    assert (directory / "checkpoint.pt").read_bytes() == original
    assert (output / "frozen_checkpoint.pt").read_bytes() == original
    repeated = cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
        batch_size=8, cpu_threads=1, resume=True)
    assert repeated == result
    assert calls == ["validation", "train"]


@pytest.mark.parametrize("changed_file", ["checkpoint", "archive"])
def test_frozen_cost_resume_rejects_changed_inputs_or_committed_results(frozen_cost_run, tmp_path: Path, changed_file: str):
    directory, _ = frozen_cost_run
    output = tmp_path / "costs"
    cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
        splits=("validation",), batch_size=8, cpu_threads=1)
    frozen_bytes = (output / "frozen_checkpoint.pt").read_bytes()
    target = directory / "checkpoint.pt" if changed_file == "checkpoint" else output / "validation_frozen_gains.npz"
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
            splits=("validation",), batch_size=8, cpu_threads=1, resume=True)
    assert (output / "frozen_checkpoint.pt").read_bytes() == frozen_bytes


def test_frozen_cost_resume_can_extend_a_complete_one_split_evaluation(frozen_cost_run, tmp_path: Path):
    directory, _ = frozen_cost_run
    output = tmp_path / "costs"
    first = cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
        splits=("validation",), batch_size=8, cpu_threads=1)
    assert first["complete"] is True
    extended = cost_comparison.evaluate_frozen_costs(directory, run_directory=tmp_path, output_directory=output,
        splits=("train",), batch_size=8, cpu_threads=1, resume=True)
    assert extended["requested_splits"] == ["validation", "train"]
    assert extended["complete"] is True
    assert extended["splits"]["validation"] == first["splits"]["validation"]
