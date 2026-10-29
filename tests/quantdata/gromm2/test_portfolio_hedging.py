from __future__ import annotations

import json
import signal

import numpy as np
import pytest
import torch

from quantdata.gromm2 import portfolio_hedging as portfolio
from quantdata.gromm2.hedging_policy import ScenarioData
from quantdata.gromm2.joint_hedging import JointTrainConfig
from quantdata.torch import BaseEnvironment
from quantdata.torch.ffw import FeedForwardConfig


@pytest.fixture(autouse=True)
def _resources(monkeypatch):
    monkeypatch.setattr(portfolio, "check_resource_headroom", lambda: None)
    torch.set_num_threads(1)


def _config():
    return portfolio.BornOptionConfig(label="test-independent", feature_count=3,
        option_expiry_days=(1, 3), option_start_days=(0, 1), hidden_size=4,
        risk_aversion=100.0, deep=FeedForwardConfig(activation="SiLU", num_depth=1, num_width=5))


def _model():
    return portfolio.IndependentHedges(BaseEnvironment(device="cpu", precision=64, debug=True),
        config=_config(), normalization=(np.zeros(3), np.ones(3)))


def _data(seed=1, split="train"):
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(8, 4, 3))
    prices = np.ones((8, 5, 3))
    prices[:, :, 0] = [1.0, 1.1, 1.2, 1.3, 1.4]
    prices[:, :, 1] = [0.1, 0.3, 0.0, 0.0, 0.0]
    prices[:, :, 2] = [0.0, 0.2, 0.3, 0.4, 0.0]
    return ScenarioData(features, prices, np.full(8, 0.02), (1, 3),
        {"model": {"test": 1}, "split": split}, f"sha-{seed}")


def _training(**overrides):
    return JointTrainConfig(**(dict(device="cpu", precision=64, batch_size=4,
        evaluation_batch_size=4, maximum_epochs=2, cpu_threads=1) | overrides))


def test_all_three_networks_and_optimizers_have_independent_parameters() -> None:
    model = _model()
    identities = [{id(value) for value in policy.parameters()} for policy in model.policies.values()]
    for first in range(3):
        for second in range(first + 1, 3):
            assert not identities[first].intersection(identities[second])


def test_new_contract_birth_price_is_not_earned_and_entry_cost_is_charged() -> None:
    model = _model()
    policy = model.policies["short"]
    with torch.no_grad():
        policy.affine.I0.bias[1] = 10.0 * np.arctanh(0.2)
    data = _data()
    output = policy(torch.as_tensor(data.features), torch.as_tensor(data.prices))
    torch.testing.assert_close(output.positions[:, 0, 1], torch.zeros(8, dtype=torch.float64))
    torch.testing.assert_close(output.gross_gain, torch.full((8,), 0.4, dtype=torch.float64))
    torch.testing.assert_close(output.transaction_cost, torch.full((8,), 2.2e-5, dtype=torch.float64))


def test_constant_payoff_quotes_and_three_error_bracket_are_exact() -> None:
    payoff = np.full(8, 0.02)
    report = portfolio.portfolio_quote_statistics(payoff, {name: np.zeros(8) for name in portfolio.LIABILITIES}, risk_aversion=100.0)
    assert report["statistics"]["ask"]["mean"] == pytest.approx(0.02)
    assert report["statistics"]["bid"]["mean"] == pytest.approx(0.02)
    assert report["bid_ask_three_se_bracket"] == pytest.approx([0.02, 0.02])
    assert report["statistics"]["ask"]["standard_error"] == 0.0


def test_training_records_separate_best_epochs_and_both_quote_sides(tmp_path) -> None:
    result = portfolio.train_independent_portfolios(_data(), _data(2, "validation"), directory=tmp_path,
        config=_config(), train_config=_training())
    report = json.loads((tmp_path / "training_report.json").read_text())
    assert report["epoch"] == 2
    assert set(report["policies"]) == {"zero", "short", "long"}
    assert report["parameter_sharing"].startswith("None")
    assert result["validation"]["statistics"]["expectation"]["pair_count"] == 4
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert set(checkpoint["optimizers"]) == set(portfolio.LIABILITIES)
    assert (tmp_path / "validation_frozen_pnl.npz").exists()


def test_complete_three_policy_minibatch_resume_is_exact(tmp_path, monkeypatch) -> None:
    original = torch.optim.Adam.step

    def stop_step(*args, **kwargs):
        output = original(*args, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return output

    kwargs = dict(train=_data(), validation=_data(2, "validation"), config=_config(), train_config=_training())
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", stop_step)
        with pytest.raises(InterruptedError):
            portfolio.train_independent_portfolios(**kwargs, directory=tmp_path / "resumed")
    portfolio.train_independent_portfolios(**kwargs, directory=tmp_path / "resumed")
    portfolio.train_independent_portfolios(**kwargs, directory=tmp_path / "full")
    resumed = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    full = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)
    for name, values in full["model"].items():
        torch.testing.assert_close(resumed["model"][name], values, rtol=0.0, atol=0.0)
    assert resumed["state"]["history"] == full["state"]["history"]
