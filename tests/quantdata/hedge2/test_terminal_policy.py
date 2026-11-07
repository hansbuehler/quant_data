"""Terminal policies use current information, premium units and causal rolling weights."""

import os
from copy import deepcopy
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
import torch

from quantdata.hedge2.terminal import terminal_risk
from quantdata.hedge2.terminal_features import TerminalFeatureMapping, rolling_feature_mapping
from quantdata.hedge2.terminal_policy import PolicyBook, TerminalPolicy, TerminalPolicyConfig
from quantdata.hedge2.terminal_train import (
    TerminalTrainConfig,
    audit_fitted_policy,
    load_terminal_policy,
    subset_terminal_data,
    train_terminal_policy,
)
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_market import inputs as market_inputs
from tests.quantdata.hedge2.test_terminal_tree import two_step_tree


def sample_data():
    original = two_step_tree()
    paths = replace(original, gains=.1 * original.gains, costs=np.zeros_like(original.costs),
        claim_gain=.1 * original.claim_gain)
    prices = np.ones_like(paths.gains)
    prices[:, 1, 0] += .1 * paths.features[:, 1, 0]
    book = PolicyBook(prices=prices, asset_descriptors=np.array([[1., 0., 0., 0.]]),
        initial_business_dtes=np.array([0.]))
    return paths, book


def randomized_policy(*, feature_count=1):
    torch.manual_seed(81)
    env = BaseEnvironment(device="cpu", precision=64, debug=True)
    policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(label="test-policy", feature_count=feature_count))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.normal_(0., .1)
    return policy


def test_policy_is_pointwise_and_future_prices_do_not_change_current_action() -> None:
    paths, book = sample_data()
    policy = randomized_policy()
    env = policy.env
    features = torch.tensor(paths.features[:, 0], dtype=env.fdtype)
    inventory = torch.zeros((4, 1), dtype=env.fdtype)
    original = policy(features, inventory, 0, book=book.tensor(env), costs=torch.zeros_like(inventory))
    changed = replace(book, prices=book.prices.copy())
    changed.prices[:, 1] *= 20.
    actual = policy(features, inventory, 0, book=changed.tensor(env), costs=torch.zeros_like(inventory))
    torch.testing.assert_close(actual, original, rtol=0., atol=0.)
    torch.testing.assert_close(original, original[:1].expand_as(original), rtol=0., atol=0.)
    subset = replace(book, prices=book.prices[[2, 0]])
    selected = policy(features[[2, 0]], inventory[[2, 0]], 0,
        book=subset.tensor(env), costs=torch.zeros((2, 1), dtype=env.fdtype))
    torch.testing.assert_close(selected, original[[2, 0]], rtol=0., atol=1e-15)


def test_normalization_update_preserves_raw_input_function_for_both_networks() -> None:
    policy = randomized_policy()
    paths, book = sample_data()
    env = policy.env
    features = torch.tensor(paths.features[:, 1], dtype=env.fdtype)
    inventory = torch.tensor([[.1], [.2], [-.3], [.4]], dtype=env.fdtype)
    costs = torch.zeros_like(inventory)
    original = policy(features, inventory, 1, book=book.tensor(env), costs=costs)
    policy.rebase_normalization(torch.tensor(paths.features + 2., dtype=env.fdtype),
        torch.tensor([.1, .2, .3, .4], dtype=env.fdtype))
    actual = policy(features, inventory, 1, book=book.tensor(env), costs=costs)
    torch.testing.assert_close(actual, original, rtol=1e-12, atol=1e-12)


def test_current_premium_caps_hold_without_a_silent_price_floor_or_gross_cap() -> None:
    env = BaseEnvironment(device="cpu", precision=64)
    policy = TerminalPolicy(env=env, config=TerminalPolicyConfig(label="cap-test", feature_count=1, use_deep=False))
    with torch.no_grad():
        policy.affine.I0.bias.fill_(1000.)
    book = PolicyBook(prices=np.array([[[1., 1e-12, 0.]]]),
        asset_descriptors=np.array([[1., 0., 0., 0.], [0., 0., 1., -.1], [0., 0., 0., .1]]),
        initial_business_dtes=np.array([0., 2., 2.]))
    positions = policy(torch.zeros((1, 1), dtype=env.fdtype), torch.zeros((1, 3), dtype=env.fdtype), 0,
        book=book.tensor(env), costs=torch.zeros((1, 3), dtype=env.fdtype))
    np.testing.assert_allclose(positions.detach().numpy(), [[10., 1e13, 0.]], rtol=1e-12)
    assert torch.isfinite(positions).all()
    assert policy.affine.env is env


def test_price_table_cannot_reveal_an_unobserved_next_branch() -> None:
    paths, book = sample_data()
    changed = book.prices.copy()
    changed[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match="not available"):
        replace(book, prices=changed).validate_histories(paths)


def test_path_selection_keeps_complete_histories_and_prices_together() -> None:
    paths, book = sample_data()
    selected, selected_book = subset_terminal_data(paths, book, np.array([3, 0]))
    np.testing.assert_array_equal(selected.gains, paths.gains[[3, 0]])
    np.testing.assert_array_equal(selected.history_ids, paths.history_ids[[3, 0]])
    np.testing.assert_array_equal(selected_book.prices, book.prices[[3, 0]])
    with pytest.raises(ValueError, match="distinct valid"):
        subset_terminal_data(paths, book, np.array([0, 0]))


@pytest.mark.parametrize("objective", ["mean_variance", "power_exponential"])
def test_policy_optimizes_terminal_risk_and_replays_without_anticipation(tmp_path, objective) -> None:
    paths, book = sample_data()
    config = TerminalPolicyConfig(label="terminal-fit", feature_count=1, use_deep=False)
    fit = train_terminal_policy(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=config, train_config=TerminalTrainConfig(maximum_updates=300, minimum_updates=50,
            learning_rate=.02, check_every=10), env=BaseEnvironment(device="cpu", precision=64),
        objective=objective, risk_aversion=10., target_side=-1, as_of_date=date(2024, 4, 19),
        target_key="synthetic", output_directory=tmp_path)
    unhedged = terminal_risk(-paths.claim_gain, paths.weights, objective=objective, risk_aversion=10.)
    assert fit.best_training_loss < unhedged.value * .05
    replay = audit_fitted_policy(policy=fit.policy, paths=paths, book=book, initial_inventory=np.zeros(1),
        objective=objective, risk_aversion=10., target_side=-1)
    assert replay["maximum_replay_error"] < 1e-12
    assert replay["std_wealth"] < .01
    loaded = load_terminal_policy(fit.checkpoint_path, env=BaseEnvironment(device="cpu", precision=64))
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, fit.policy.state_dict()[name], rtol=0., atol=0.)


def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rng_state(tmp_path, device="cpu") -> None:
    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="resume-fit", feature_count=1),
        env=BaseEnvironment(device=device, precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic")
    training = TerminalTrainConfig(maximum_updates=12, minimum_updates=1, check_every=2, stable_checks=100)
    uninterrupted = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "whole")
    partial = train_terminal_policy(**arguments, train_config=replace(training, maximum_updates=6),
        output_directory=tmp_path / "split")
    assert partial.status == "training_limit"
    resumed = train_terminal_policy(**arguments, train_config=training, output_directory=tmp_path / "split")
    assert resumed.updates == uninterrupted.updates == 12
    assert resumed.history == uninterrupted.history
    for name, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, uninterrupted.policy.state_dict()[name], rtol=0., atol=0.)


@pytest.mark.skipif(os.environ.get("HEDGE2_GPU_TEST") != "1", reason="Requires an explicitly gpuctl-allocated GPU")
def test_gpu_checkpoint_resume_preserves_optimizer_and_host_rng_state(tmp_path) -> None:
    test_checkpoint_resume_matches_uninterrupted_optimizer_and_rng_state(tmp_path, device="cuda")


def test_changed_training_samples_are_not_silently_resumed(tmp_path) -> None:
    paths, book = sample_data()
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1),
        model_config=TerminalPolicyConfig(label="identity-fit", feature_count=1, use_deep=False),
        train_config=TerminalTrainConfig(maximum_updates=1, minimum_updates=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic", output_directory=tmp_path)
    train_terminal_policy(**arguments)
    arguments["paths"] = replace(paths, claim_gain=paths.claim_gain + .01)
    with pytest.raises(ValueError, match="different simulations"):
        train_terminal_policy(**arguments)


def test_changed_native_axes_are_explicitly_mapped_on_the_new_model_span() -> None:
    arguments = market_inputs()
    previous_state, previous_model = arguments["state"], arguments["dynamics"]
    root_change = np.zeros_like(previous_state.native)
    root_change[1] = .01
    current_state = replace(previous_state, day=date(2024, 4, 22), native=previous_state.native + root_change)
    current_model = deepcopy(previous_model)
    current_model.as_of_date = current_state.day
    current_model.directions[0, 0] = 1. / np.sqrt(2.)
    current_model.directions[1, 0] = 1. / np.sqrt(2.)
    current_model.daily_factor_scales = np.array([.2])
    mapping = rolling_feature_mapping(previous_state=previous_state, previous_model=previous_model,
        current_state=current_state, current_model=current_model, previous_root_spot=100., current_root_spot=102.,
        elapsed_step_offset=1)
    native = current_state.native + .3 * .2 * current_model.directions[:, 0]
    native[10] += .02
    current = np.array([np.log(105. / 102.), (native - current_state.native) @ current_model.directions[:, 0] / .2,
        .02, 1., .03])
    previous = np.array([np.log(105. / 100.), (native - previous_state.native) @ previous_model.directions[:, 0] / .1,
        .02, 1., .03])
    np.testing.assert_allclose(mapping.matrix @ current + mapping.offset, previous, atol=1e-13)
    assert mapping.old_subspace_projection_residual > 0.
    assert mapping.elapsed_step_offset == 1


def test_feature_rotation_is_absorbed_by_both_first_layers() -> None:
    policy = randomized_policy(feature_count=2)
    env = policy.env
    paths, book = sample_data()
    features = torch.tensor([[-.2, .3], [.1, .4], [.5, -.3], [.4, -.1]], dtype=env.fdtype)
    matrix = np.array([[0., 2.], [-.5, 0.]])
    offset = np.array([.1, -.2])
    old_features = features @ torch.tensor(matrix.T, dtype=env.fdtype) + torch.tensor(offset, dtype=env.fdtype)
    inventory, costs = torch.zeros((4, 1), dtype=env.fdtype), torch.zeros((4, 1), dtype=env.fdtype)
    original = policy(old_features, inventory, 1, book=book.tensor(env), costs=costs)
    policy.rebase_normalization(torch.stack([features, features + .1], dim=1),
        torch.full((4,), .25, dtype=env.fdtype), mapping=TerminalFeatureMapping(matrix=matrix, offset=offset))
    actual = policy(features, inventory, 1, book=book.tensor(env), costs=costs)
    torch.testing.assert_close(actual, original, rtol=1e-12, atol=1e-12)


def test_rolling_warm_start_requires_mapping_and_records_earlier_weight_date(tmp_path) -> None:
    paths, book = sample_data()
    config = TerminalPolicyConfig(label="rolling-fit", feature_count=1, use_deep=False)
    arguments = dict(paths=paths, book=book, initial_inventory=np.zeros(1), model_config=config,
        train_config=TerminalTrainConfig(maximum_updates=1, minimum_updates=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, target_key="synthetic")
    prior = train_terminal_policy(**arguments, as_of_date=date(2024, 4, 19), output_directory=tmp_path / "prior")
    with pytest.raises(ValueError, match="explicit recorded"):
        train_terminal_policy(**arguments, as_of_date=date(2024, 4, 22), previous_policy=prior.policy,
            output_directory=tmp_path / "missing-map")
    current = train_terminal_policy(**arguments, as_of_date=date(2024, 4, 22), previous_policy=prior.policy,
        feature_mapping=TerminalFeatureMapping(matrix=np.eye(1), offset=np.zeros(1)), output_directory=tmp_path / "current")
    assert current.initialisation_date == date(2024, 4, 19)
    saved = torch.load(current.checkpoint_path, weights_only=True)
    assert saved["feature_mapping"]["matrix"] == [[1.]]


def test_user_version_mismatch_is_detected_before_model_reconstruction(tmp_path) -> None:
    paths, book = sample_data()
    config = TerminalPolicyConfig(label="version-fit", feature_count=1, use_deep=False)
    fit = train_terminal_policy(paths=paths, book=book, initial_inventory=np.zeros(1), model_config=config,
        train_config=TerminalTrainConfig(maximum_updates=1, minimum_updates=1),
        env=BaseEnvironment(device="cpu", precision=64), objective="mean_variance", risk_aversion=10.,
        target_side=-1, as_of_date=date(2024, 4, 19), target_key="synthetic", output_directory=tmp_path)
    before = fit.checkpoint_path.read_bytes()
    with pytest.warns(UserWarning, match="out of date"), pytest.raises(ValueError, match="preserved"):
        load_terminal_policy(fit.checkpoint_path, env=BaseEnvironment(device="cpu", precision=64),
            expected_config=replace(config, user_version="0.2"))
    assert fit.checkpoint_path.read_bytes() == before
