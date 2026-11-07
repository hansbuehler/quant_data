"""Daily terminal decisions preserve actual inventory and causal policy initialization."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2 import terminal_decision
from quantdata.hedge2.terminal_decision import fit_terminal_decision
from quantdata.hedge2.terminal_market import TerminalSimulationConfig, generate_terminal_paths
from quantdata.hedge2.terminal_policy import TerminalPolicy, TerminalPolicyConfig
from quantdata.hedge2.terminal_train import TerminalPolicyFit, TerminalTrainConfig
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_market import inputs


def decision_inputs():
    arguments = inputs()
    arguments["dynamics"].training_start = date(2023, 4, 19)
    arguments["dynamics"].training_end = arguments["state"].day
    sessions = get_business_days(arguments["state"].day, date(2024, 5, 1))
    arguments["book"]["expiry"] = pd.to_datetime([sessions[int(value)] for value in arguments["book"].bdte])
    arguments["book"]["is_am"] = False
    simulations = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=6))
    return dict(simulations=simulations, contracts=arguments["book"], claim=arguments["claims"][0], claim_index=0,
        state=arguments["state"], dynamics=arguments["dynamics"], initial_inventory=np.array([.2, .1, -.1, 0., .3]),
        model_config=TerminalPolicyConfig(label="decision", feature_count=5, use_deep=False),
        train_config=TerminalTrainConfig(maximum_updates=1, minimum_updates=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10., target_side=-1)


def later_inputs(arguments, *, new_inception=False):
    state = replace(arguments["state"], day=date(2024, 4, 22))
    model = deepcopy(arguments["dynamics"])
    model.as_of_date = model.training_end = state.day
    model.daily_factor_scales *= 1.5
    contracts = arguments["contracts"].copy()
    curve = inputs()["curve"]
    if new_inception:
        curve = replace(curve, spot=102., forwards=1.02 * curve.forwards, reference_forwards=1.02 * curve.reference_forwards)
        original = arguments["claim"]
        claim = replace(original, initial_spot=102., strike=original.strike * (1. if original.kind == "ratio_straddle" else 1.02),
            barrier=1.02 * original.barrier)
        sessions = get_business_days(state.day, date(2024, 5, 6))
        contracts["expiry"] = pd.to_datetime([sessions[int(value)] for value in contracts.bdte])
    else:
        curve = curve.roll(1., new_spot=102.)
        claim = arguments["claim"]
        contracts["bdte"] -= 1.
    simulations = generate_terminal_paths(state=state, dynamics=model, curve=curve, book=contracts,
        claims=(claim,), initial_spot=claim.initial_spot, current_claim_marks=np.array([.5]),
        elapsed_day=0 if new_inception else 1, config=TerminalSimulationConfig(paths=6), seed=12)
    return {**arguments, "state": state, "dynamics": model, "simulations": simulations,
        "contracts": contracts, "claim": claim, "claim_index": 0}


def fake_fit(arguments, *, status="training_stable"):
    policy = TerminalPolicy(env=arguments["env"], config=arguments["model_config"])
    with torch.no_grad():
        policy.affine.I0.bias.fill_(1.)
    policy.as_of_date = arguments["as_of_date"]
    policy.fit_context = {name: arguments[name] for name in ("objective", "risk_aversion", "target_side", "target_key")}
    directory = arguments["output_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    return TerminalPolicyFit(policy=policy, status=status, updates=1, initial_loss=1., best_training_loss=.5,
        history=(), checkpoint_path=directory / "unused.pt", initialisation_date=(None if arguments["previous_policy"] is None
            else arguments["previous_policy"].as_of_date))


def test_carried_inventory_and_cash_conversion_reach_the_daily_root(monkeypatch, tmp_path: Path) -> None:
    arguments, recorded = decision_inputs(), []

    def fit(**values):
        recorded.append(values)
        return fake_fit(values)

    monkeypatch.setattr(terminal_decision, "train_terminal_policy", fit)
    result = fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)
    np.testing.assert_array_equal(recorded[0]["initial_inventory"], arguments["initial_inventory"])
    np.testing.assert_allclose(result.premium_positions, 10. * np.tanh(.1), rtol=1e-14)
    assert not result.accepted_for_execution
    with np.load(tmp_path / "root_decision.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["cash_unit_holdings"], result.positions / 100.)
        np.testing.assert_array_equal(saved["initial_inventory"], arguments["initial_inventory"])


def test_small_simulation_cannot_be_promoted_to_production(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 10000"):
        fit_terminal_decision(**decision_inputs(), output_directory=tmp_path)


def test_current_model_can_use_last_valid_paired_history_without_inventing_an_increment(monkeypatch, tmp_path: Path) -> None:
    arguments = decision_inputs()
    arguments["dynamics"].training_end = date(2024, 4, 18)
    monkeypatch.setattr(terminal_decision, "train_terminal_policy", lambda **values: fake_fit(values))
    result = fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)
    assert result.state.day == arguments["dynamics"].as_of_date
    with np.load(tmp_path / "root_decision.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
    assert metadata["latest_fitted_transition"] == "2024-04-18"
    assert metadata["latest_transition_session_lag"] == 1


def test_future_ending_training_history_is_rejected_before_fitting(tmp_path: Path) -> None:
    arguments = decision_inputs()
    arguments["dynamics"].training_end = date(2024, 4, 22)
    with pytest.raises(ValueError, match="today's causal model"):
        fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)


def test_daily_restart_maps_new_factors_and_keeps_nonzero_inventory(monkeypatch, tmp_path: Path) -> None:
    arguments, recorded = decision_inputs(), []

    def fit(**values):
        recorded.append(values)
        return fake_fit(values)

    monkeypatch.setattr(terminal_decision, "train_terminal_policy", fit)
    previous = fit_terminal_decision(**arguments, output_directory=tmp_path / "prior", diagnostic_only=True)
    restarted = later_inputs(arguments)
    restarted["initial_inventory"] = previous.positions.copy()
    result = fit_terminal_decision(**restarted, previous_decision=previous,
        output_directory=tmp_path / "next", diagnostic_only=True)
    assert recorded[-1]["previous_policy"] is previous.fit.policy
    np.testing.assert_array_equal(recorded[-1]["initial_inventory"], previous.positions)
    assert recorded[-1]["feature_mapping"].elapsed_step_offset == 1
    assert result.fit.initialisation_date == previous.state.day
    assert result.elapsed_day == 1


def test_missing_preceding_episode_policy_is_not_silently_cold_started(tmp_path: Path) -> None:
    arguments = later_inputs(decision_inputs())
    with pytest.raises(ValueError, match="no silent cold start"):
        fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)


def test_training_limit_is_preserved_as_an_unaccepted_candidate(monkeypatch, tmp_path: Path) -> None:
    arguments = decision_inputs()
    monkeypatch.setattr(terminal_decision, "train_terminal_policy", lambda **values: fake_fit(values, status="training_limit"))
    result = fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)
    assert result.fit.status == "training_limit"
    assert not result.accepted_for_execution
    assert (tmp_path / "root_decision.npz").is_file()


def test_independent_adequacy_result_is_saved_with_the_root_not_silently_replaced(monkeypatch, tmp_path: Path) -> None:
    arguments = decision_inputs()
    monkeypatch.setattr(terminal_decision, "train_terminal_policy", lambda **values: fake_fit(values))
    recorded = []
    original = terminal_decision.assess_zero_action_reference

    def inspect_reference(**values):
        result = original(**values)
        recorded.append((values, result))
        return result

    monkeypatch.setattr(terminal_decision, "assess_zero_action_reference", inspect_reference)
    result = fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)
    np.testing.assert_array_equal(recorded[0][0]["initial_inventory"], arguments["initial_inventory"])
    with np.load(tmp_path / "root_decision.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
    assert metadata["zero_action_reference_audit"] == recorded[0][1]
    assert not result.accepted_for_execution


def test_stability_alone_cannot_promote_a_dominated_production_policy(monkeypatch, tmp_path: Path) -> None:
    arguments = decision_inputs()
    original = arguments["simulations"]
    path_fields = ("instrument_gains", "raw_instrument_gains", "removed_conditional_mean", "costs",
        "instrument_prices", "spots", "state_features", "history_ids", "claim_gains", "claim_payoffs",
        "terminate_after_step", "weights", "innovation_indices", "lower_clips", "upper_clips")
    arguments["simulations"] = replace(original, **{name: np.repeat(getattr(original, name), 1667, axis=0)
        for name in path_fields})
    arguments["dynamics"] = deepcopy(arguments["dynamics"])
    arguments["dynamics"].residuals = np.repeat(arguments["dynamics"].residuals[:1], 252, axis=0)
    monkeypatch.setattr(terminal_decision, "train_terminal_policy", lambda **values: fake_fit(values))
    monkeypatch.setattr(terminal_decision, "audit_fitted_policy", lambda **values: {"risk": 1e6})
    result = fit_terminal_decision(**arguments, output_directory=tmp_path)
    assert result.fit.status == "training_stable"
    assert not result.accepted_for_execution
    with np.load(tmp_path / "root_decision.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
    assert not metadata["zero_action_reference_audit"]["no_worse_than_zero_reference"]
    assert not metadata["accepted_for_execution"]


@pytest.mark.parametrize("claim_index", [0, 1, 2])
def test_previous_origin_weights_are_initialized_with_new_numeraire_features(monkeypatch, tmp_path: Path, claim_index: int) -> None:
    arguments, recorded = decision_inputs(), []
    arguments.update(claim=inputs()["claims"][claim_index], claim_index=claim_index)

    def fit(**values):
        recorded.append(values)
        return fake_fit(values)

    monkeypatch.setattr(terminal_decision, "train_terminal_policy", fit)
    prior = fit_terminal_decision(**arguments, output_directory=tmp_path / "prior", diagnostic_only=True)
    result = fit_terminal_decision(**later_inputs(arguments, new_inception=True), previous_origin_decision=prior,
        output_directory=tmp_path / "new-origin", diagnostic_only=True)
    mapping = recorded[-1]["feature_mapping"]
    assert result.fit.initialisation_date == prior.state.day
    assert mapping.elapsed_step_offset == 0
    assert mapping.matrix[-1, -2] == pytest.approx(np.log(102. / 100.) if claim_index == 2 else 0.)
    assert recorded[-1]["previous_policy"] is prior.fit.policy


def test_restart_rejects_changed_cash_strike_with_the_same_nms_label(monkeypatch, tmp_path: Path) -> None:
    arguments = decision_inputs()
    monkeypatch.setattr(terminal_decision, "train_terminal_policy", lambda **values: fake_fit(values))
    previous = fit_terminal_decision(**arguments, output_directory=tmp_path / "prior", diagnostic_only=True)
    current = later_inputs(arguments)
    current["claim"] = replace(current["claim"], strike=101.)
    with pytest.raises(ValueError, match="same episode"):
        fit_terminal_decision(**current, previous_decision=previous, output_directory=tmp_path / "next", diagnostic_only=True)


def test_restart_rejects_an_incorrect_original_spot_unit(monkeypatch, tmp_path: Path) -> None:
    arguments = later_inputs(decision_inputs())
    prices = arguments["simulations"].instrument_prices.copy()
    prices[:, 0, 0] *= 2.
    arguments["simulations"] = replace(arguments["simulations"], instrument_prices=prices)
    with pytest.raises(ValueError, match="claim units"):
        fit_terminal_decision(**arguments, output_directory=tmp_path, diagnostic_only=True)
