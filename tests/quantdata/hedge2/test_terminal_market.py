"""Full-horizon simulation, conditional means and path-observation cash flows."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.hedge2 import terminal_market
from quantdata.hedge2.cash import CashCurve, model_europeans
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal import evaluate_terminal_policy
from quantdata.hedge2.terminal_market import TerminalSimulationConfig, generate_terminal_paths


def inputs():
    config = DlvConfig(num_inner_strikes=10)
    vols = np.full(9, .2)
    inner = shared_inner_strikes(vols, config=config)
    state = DailyState(day=date(2024, 4, 19), config=config,
        native=np.c_[np.full((9, 10), np.log(.04)), np.log(vols)].reshape(-1),
        reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])
    directions = np.zeros((len(state.native), 1))
    directions[0, 0] = 1.
    beta = np.zeros(len(state.native))
    beta[10] = .2
    model = SimpleNamespace(as_of_date=state.day, state_mean=state.native.copy(),
        current_scores=np.zeros(1), directions=directions, ar_intercept=np.zeros(1),
        ar_matrix=np.array([[.8]]), daily_factor_scales=np.array([.1]),
        spot_state_loading=beta, residuals=np.array([[-.02, .001], [-.01, -.002], [.01, .002], [.02, -.001]]))
    days = np.array([1., 3., 5., 25.])
    curve = CashCurve(spot=100., business_dtes=days, forwards=100. * np.exp(.001 * days),
        reference_forwards=100. * np.exp(.002 * days), discounts=np.exp(-.003 * days))
    book = pd.DataFrame({"bdte": [3., 5., 3., 5.], "strike": [100., 100., 0., 0.],
        "option_type": ["call"] * 4, "instrument_kind": ["european", "european", "zero_strike_call", "zero_strike_call"],
        "observed_half_spread": [.1, .1, 0., 0.]})
    book["observed_mid"] = model_europeans(state.decode().pricer, curve,
        business_dtes=book.bdte.to_numpy(), cash_strikes=book.strike.to_numpy(), is_call=np.ones(4, dtype=bool))
    claims = (Claim(label="digital", kind="digital", terminal_day=5, initial_spot=100., strike=100.),
        Claim(label="ko", kind="ko", terminal_day=5, observation_day=3, initial_spot=100., strike=99., barrier=100.),
        Claim(label="ratio", kind="ratio_straddle", terminal_day=5, observation_day=3, initial_spot=100., strike=1.))
    return dict(state=state, dynamics=model, curve=curve, book=book, claims=claims,
        initial_spot=100., current_claim_marks=np.array([.5, .02, .01]), seed=81)


def test_terminal_market_paths_reproduce_draws_and_observation_payoffs() -> None:
    arguments = inputs()
    first = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=12, integration_parent_batch=3))
    second = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=12, integration_parent_batch=1))
    np.testing.assert_array_equal(first.innovation_indices, second.innovation_indices)
    np.testing.assert_allclose(first.instrument_gains, second.instrument_gains, atol=1e-13)
    np.testing.assert_allclose(first.claim_payoffs[:, 0], first.spots[:, -1] > 100.)
    np.testing.assert_allclose(first.claim_payoffs[:, 1],
        (first.spots[:, 3] > 100.) * np.maximum(first.spots[:, -1] - 99., 0.) / 100.)
    np.testing.assert_allclose(first.claim_payoffs[:, 2], np.abs(first.spots[:, -1] / first.spots[:, 3] - 1.))
    np.testing.assert_array_equal(first.terminate_after_step[:, 2, 1], first.spots[:, 3] <= 100.)
    assert np.max(first.conditional_mean_residual) < 1e-13
    for index, claim in enumerate(arguments["claims"]):
        paths = first.for_claim(claim, index)
        np.testing.assert_array_equal(paths.features[:, :3, -2:], 0.)
    ratio = first.for_claim(arguments["claims"][2], 2)
    np.testing.assert_allclose(ratio.features[:, 3:, -1], np.log(first.spots[:, 3:4] / 100.) * np.ones((1, 2)))


def test_conditional_demeaning_is_state_dependent_not_a_single_future_path_mean() -> None:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=32, integration_parent_batch=4))
    np.testing.assert_allclose(result.raw_instrument_gains - result.removed_conditional_mean, result.instrument_gains)
    assert np.ptp(result.removed_conditional_mean[:, 1, 1]) > 1e-7
    indices, first = np.unique(result.innovation_indices[:, 0], return_index=True)
    assert len(indices) == 4
    np.testing.assert_allclose(result.instrument_gains[first, 0].mean(axis=0), 0., atol=1e-13)
    np.testing.assert_allclose(result.state_features[:, 0], 0., atol=1e-13)


def test_raw_option_gains_telescope_to_settlement_without_exit_or_rebase() -> None:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, discount_to_current=.95,
        config=TerminalSimulationConfig(paths=8, integration_parent_batch=2))
    book = arguments["book"]
    for index, row in enumerate(book.itertuples()):
        tenor = int(row.bdte)
        payoff = np.maximum(result.spots[:, tenor] - row.strike, 0.)
        expected = .95 * (np.exp(-.003 * tenor) * payoff - row.observed_mid) / 100.
        np.testing.assert_allclose(result.raw_instrument_gains[:, :, index + 1].sum(axis=1), expected, atol=1e-13)
        np.testing.assert_array_equal(result.raw_instrument_gains[:, tenor:, index + 1], 0.)
        expected_costs = .95 * np.exp(-.003 * np.arange(tenor)) * row.observed_half_spread / 100.
        np.testing.assert_allclose(result.costs[:, :tenor, index + 1],
            np.broadcast_to(expected_costs, (len(result.spots), tenor)))


def test_knockout_policy_terminates_after_observation_period_gain() -> None:
    arguments = inputs()
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=24, integration_parent_batch=4))
    claim = arguments["claims"][1]
    paths = result.for_claim(claim, 1)
    evaluated = evaluate_terminal_policy(paths=paths, initial_inventory=np.zeros(5),
        policy=lambda features, inventory, step: np.ones_like(inventory), target_side=-1)
    knocked_out = result.terminate_after_step[:, 2, 1]
    assert knocked_out.any() and (~knocked_out).any()
    np.testing.assert_array_equal(evaluated.positions[knocked_out, 3:], 0.)
    np.testing.assert_array_equal(evaluated.step_costs[knocked_out, 3:], 0.)
    np.testing.assert_allclose(evaluated.step_gains[knocked_out, 2], result.instrument_gains[knocked_out, 2].sum(axis=1))


def test_daily_restart_preserves_observed_reset_spot_and_remaining_horizon() -> None:
    arguments = inputs()
    claim = replace(arguments["claims"][2], reset_spot=104., observation_done=True)
    arguments.update(claims=(claim,), current_claim_marks=np.array([.02]), elapsed_day=3)
    arguments["book"] = arguments["book"].iloc[[1, 3]].copy()
    arguments["book"]["bdte"] = 2.
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=8, integration_parent_batch=2))
    assert result.instrument_gains.shape == (8, 2, 3)
    np.testing.assert_allclose(result.claim_payoffs[:, 0], np.abs(result.spots[:, -1] / 104. - 1.))
    paths = result.for_claim(claim, 0)
    np.testing.assert_allclose(paths.features[:, :, -1], np.log(104. / 100.))
    np.testing.assert_array_equal(paths.features[:, :, -2], 1.)


def test_mismatched_future_model_date_cannot_be_used() -> None:
    arguments = inputs()
    arguments["dynamics"].as_of_date = date(2024, 4, 22)
    with pytest.raises(ValueError, match="causal inputs"):
        generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=2))


def test_conditional_integration_prices_identical_histories_only_once(monkeypatch) -> None:
    arguments = inputs()
    original = terminal_market.decode_batch
    decoded_count = 0

    def counted_decode(state, native, **kwargs):
        nonlocal decoded_count
        decoded_count += len(native)
        return original(state, native, **kwargs)

    monkeypatch.setattr(terminal_market, "decode_batch", counted_decode)
    result = generate_terminal_paths(**arguments, config=TerminalSimulationConfig(paths=128, integration_parent_batch=8))
    expected_parents = np.array([len(np.unique(row)) for row in result.history_ids.T])
    np.testing.assert_array_equal(result.integrated_parent_counts, expected_parents)
    assert result.integrated_parent_counts[0] == 1
    assert result.integrated_parent_counts[1] == 4
    assert decoded_count == 4 * expected_parents[:-1].sum()


def test_nonflat_carry_and_conditional_means_match_independent_scalar_branch_replay() -> None:
    arguments = inputs()
    days = np.array([1., 2., 3., 4., 5., 25.])
    root_curve = CashCurve(spot=100., business_dtes=days,
        forwards=100. * np.array([1.001, 1.004, 1.003, 1.008, 1.007, 1.025]),
        reference_forwards=100. * np.array([1.002, 1.003, 1.007, 1.009, 1.011, 1.033]),
        discounts=np.array([.999, .995, .994, .992, .991, .965]))
    arguments["curve"] = root_curve
    result = generate_terminal_paths(**arguments, discount_to_current=.93,
        config=TerminalSimulationConfig(paths=3, integration_parent_batch=2))
    model, state, book = arguments["dynamics"], arguments["state"], arguments["book"]
    for path in range(3):
        native = state.native.copy()
        spot = root_curve.spot
        current_prices = np.r_[spot, book.observed_mid.to_numpy()] / 100.
        for step in range(5):
            scores = (native - model.state_mean) @ model.directions
            forecast = native + model.directions @ (model.ar_intercept + model.ar_matrix @ scores - scores)
            reference_before = root_curve.at(float(step))[1]
            _, reference_after, discount_after = root_curve.at(float(step + 1))
            discount_before = root_curve.at(float(step))[2]
            financed_forward = spot * reference_after / reference_before
            branch_prices = []
            remaining = book.bdte.to_numpy() - step - 1
            for innovation in model.residuals:
                following_spot = financed_forward * (1. + innovation[0])
                following_native = forecast + model.directions @ innovation[1:] + model.spot_state_loading * innovation[0]
                following_curve = root_curve.roll(float(step + 1), new_spot=following_spot)
                following_prices = np.zeros(len(book))
                live = remaining > 0
                if live.any():
                    following_prices[live] = model_europeans(state.decode(following_native).pricer, following_curve,
                        business_dtes=remaining[live], cash_strikes=book.strike.to_numpy()[live],
                        is_call=np.ones(live.sum(), dtype=bool))
                settling = remaining == 0
                following_prices[settling] = np.maximum(following_spot - book.strike.to_numpy()[settling], 0.)
                branch_prices.append(np.r_[following_spot, following_prices] / 100.)
            branch_prices = np.array(branch_prices)
            selected = branch_prices[result.innovation_indices[path, step]]
            current_discount = .93 * discount_before
            next_discount = discount_after / discount_before
            raw = current_discount * (next_discount * selected - current_prices)
            raw[0] = current_discount * next_discount * (selected[0] - financed_forward / 100.)
            mean = current_discount * (next_discount * branch_prices.mean(axis=0) - current_prices)
            mean[0] = current_discount * next_discount * (branch_prices[:, 0].mean() - financed_forward / 100.)
            live = np.r_[True, book.bdte.to_numpy() > step]
            np.testing.assert_allclose(result.raw_instrument_gains[path, step], raw * live, atol=1e-12)
            np.testing.assert_allclose(result.removed_conditional_mean[path, step], mean * live, atol=1e-12)
            np.testing.assert_allclose(result.instrument_prices[path, step + 1], selected, atol=1e-12)
            innovation = model.residuals[result.innovation_indices[path, step]]
            native = forecast + model.directions @ innovation[1:] + model.spot_state_loading * innovation[0]
            spot, current_prices = 100. * selected[0], selected
    expected = .93 * (root_curve.at(5.)[2] * result.claim_payoffs - arguments["current_claim_marks"])
    np.testing.assert_allclose(result.claim_gains, expected, atol=1e-13)
