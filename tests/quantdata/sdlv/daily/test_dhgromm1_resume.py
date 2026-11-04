"""Continuation preserves Adam, fixed reference weights and minibatch order."""

from pathlib import Path

import numpy as np
import pytest
import torch

from quantdata.sdlv.dhgromm1.hedge import HedgeConfig, oce_value
from quantdata.sdlv.dhgromm1.training import (
    TrainConfig,
    load_checkpoint,
    load_frozen_policy,
    train_policy,
)


@pytest.mark.parametrize("claim_sign", [1, -1])
def test_resumed_training_matches_uninterrupted_epochs(tmp_path: Path, claim_sign: int) -> None:
    rng = np.random.default_rng(43)
    prices = rng.uniform(.5, 1.5, (12, 3, 2)).astype(np.float32)
    returns = rng.normal(0, .05, (12, 2, 2)).astype(np.float32)
    payoff = rng.uniform(0, .03, 12)
    config = HedgeConfig(label="resume-test", instruments=2, horizon_days=2, hidden_size=3)
    folders = {name: tmp_path / name for name in ("full", "first", "resumed")}
    for folder in folders.values():
        folder.mkdir()

    def fit(folder: Path, epochs: int, resume: Path | None = None):
        return train_policy(config=config, train=TrainConfig(epochs=epochs, batch_size=4, learning_rate=1e-4,
                            seed=37, patience=0), prices=prices, returns=returns, payoff=payoff,
                            claim_sign=claim_sign, output=folder, price_mean=np.zeros(2),
                            price_scale=np.ones(2), resume_checkpoint=resume)

    fit(folders["full"], 3)
    fit(folders["first"], 1)
    source = folders["first"] / "resume-test.checkpoint.pt"
    legacy = load_checkpoint(source, config=config)
    legacy.pop("pair_order")  # Exercise recovery of the earlier checkpoint format.
    legacy_path = folders["first"] / "legacy.pt"
    torch.save(legacy, legacy_path)
    resumed = fit(folders["resumed"], 2, legacy_path)
    expected = load_checkpoint(folders["full"] / "resume-test.checkpoint.pt", config=config)
    actual = load_checkpoint(folders["resumed"] / "resume-test.checkpoint.pt", config=config)
    assert actual["epoch"] == expected["epoch"] == 2
    assert resumed.fit_summary["additional_epochs_run"] == 2
    assert actual["selected_epoch"] == expected["selected_epoch"]
    assert actual["best_loss"] == expected["best_loss"]
    for group in ("state_dict", "best_state"):
        for key in actual[group]:
            torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)
    np.testing.assert_array_equal(actual["pair_order"], expected["pair_order"])
    for key, state in actual["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, expected["optimizer"]["state"][key][name], rtol=0, atol=0)
    frozen = load_frozen_policy(folders["resumed"] / "resume-test.checkpoint.pt", config=config, device="cpu")
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert frozen.fit_summary["additional_epochs_run"] == 0
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, actual["best_state"][key], rtol=0, atol=0)


def test_continuation_rejects_incompatible_model_version(tmp_path: Path) -> None:
    path = tmp_path / "incompatible.pt"
    torch.save(dict(model_code_version="obsolete", user_version="0.1"), path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path, config=HedgeConfig(label="test"))


def test_weights_only_warm_start_reevaluates_candidate_and_resets_optimizer(tmp_path: Path) -> None:
    rng = np.random.default_rng(49)
    prices = rng.uniform(.8, 1.2, (12, 3, 2)).astype(np.float32)
    returns = rng.normal(0, .02, (12, 2, 2)).astype(np.float32)
    payoff = rng.uniform(0, .02, 12)
    config = HedgeConfig(label="warm-test", instruments=2, horizon_days=2, hidden_size=3)
    source, target = tmp_path / "source", tmp_path / "warm"
    source.mkdir()
    target.mkdir()
    options = dict(config=config, prices=prices, returns=returns, payoff=payoff, claim_sign=1,
                   price_mean=np.zeros(2), price_scale=np.ones(2))
    train_policy(**options, output=source, train=TrainConfig(epochs=1, batch_size=4, seed=41, learning_rate=1e-4))
    checkpoint = source / "warm-test.checkpoint.pt"
    saved = load_checkpoint(checkpoint, config=config)
    # None of these old-sample operational statistics belong to the new fit.
    saved.update(best_loss=-12345., optimizer="must not be loaded", epoch=77, numpy_rng="old sample")
    torch.save(saved, checkpoint)
    returns = returns * 2
    options["returns"] = returns
    initial = load_frozen_policy(checkpoint, config=config, device="cpu")
    with torch.no_grad():
        output = initial(torch.as_tensor(prices), returns=torch.as_tensor(returns))
        expected_loss = float(-oce_value(output.gain + torch.as_tensor(payoff, dtype=torch.float32),
                                         initial.translation, config.risk_aversion) + output.penalty.mean())
    policy = train_policy(**options, output=target, warm_start_checkpoint=checkpoint,
        train=TrainConfig(epochs=1, batch_size=4, seed=41, learning_rate=1e-30, patience=0))
    result = load_checkpoint(target / "warm-test.checkpoint.pt", config=config)
    assert policy.fit_summary["initial_training_loss"] == pytest.approx(expected_loss, abs=1e-8)
    assert policy.fit_summary["start_epoch"] == 0
    assert policy.fit_summary["epochs_run"] == 1
    assert policy.fit_summary["warm_start"]["source_selected_epoch"] == saved["selected_epoch"]
    assert result["best_loss"] == pytest.approx(expected_loss, abs=1e-8)
    for state in result["optimizer"]["state"].values():
        assert float(state["step"]) == 3  # Three fresh minibatches, not the old epoch count.
    for key, value in result["state_dict"].items():
        torch.testing.assert_close(value, saved["best_state"][key], rtol=0, atol=0)
