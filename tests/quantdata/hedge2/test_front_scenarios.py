"""Independent cash repricing of opt-in calibrated-front simulation candidates."""

from dataclasses import asdict, replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.cash import CashCurve, model_europeans
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.dlv import DlvConfig, shared_inner_strikes
from quantdata.hedge2.dynamics import DynamicsConfig, fit_dynamics
from quantdata.hedge2.front_joint import EmpiricalFrontCopula, GaussianFrontCopula
from quantdata.hedge2.front_scenarios import generate_front_scenarios
from quantdata.hedge2.front_spot import DiscreteFrontLaw
from quantdata.hedge2.objective_run import save_verified_array_inputs
from quantdata.hedge2.state import DailyState


def inputs(kind):
    config = DlvConfig(num_inner_strikes=10)
    inner = shared_inner_strikes(np.full(9, .2), config=config)
    native = np.c_[np.full((9, 10), np.log(.04)), np.full(9, np.log(.2))].reshape(-1)
    rng = np.random.default_rng(14)
    dates = tuple(get_business_days(date(2024, 1, 2), date(2024, 2, 2))[:20])
    samples = native + rng.normal(scale=.001, size=(20, len(native)))
    samples[-1] = native
    dynamics = fit_dynamics(dates=dates, states=samples, spot_returns_into_date=rng.normal(scale=.01, size=20),
        session_indices=np.arange(20), as_of_date=dates[-1],
        config=DynamicsConfig(window_transitions=15, rank=2, spot_orthogonal=True))
    state = DailyState(day=dates[-1], native=native, config=config,
        reference_support=np.r_[.5 * inner[0], inner, 1.5 * inner[-1]])
    curve = CashCurve(spot=100., business_dtes=np.array([1., 3., 5., 25.]),
        forwards=np.array([100.1, 100.3, 100.5, 102.5]),
        reference_forwards=np.array([100.2, 100.6, 101., 105.]), discounts=np.exp(-.001 * np.array([1., 3., 5., 25.])))
    book = pd.DataFrame({"bdte": [1., 3., 5.], "strike": [100., 99., 0.],
        "option_type": ["call", "put", "call"]})
    claims = (Claim(label="call", kind="digital", initial_spot=100., strike=100., terminal_day=1),
        Claim(label="put", kind="digital", initial_spot=100., strike=100., terminal_day=1, is_call=False),
        Claim(label="3D-call", kind="digital", initial_spot=100., strike=101., terminal_day=3),
        Claim(label="ko", kind="ko", initial_spot=100., strike=99., terminal_day=5,
            observation_day=3, barrier=100.),
        Claim(label="ratio", kind="ratio_straddle", initial_spot=100., strike=1., terminal_day=5,
            observation_day=3))
    coupling = kind(historical_scores=np.linspace(-2., 2., len(dynamics.residuals)),
        factor_residuals=dynamics.residuals[:, 1:], as_of_date=state.day)
    marks = np.array([claim.price(state.decode().pricer, curve, elapsed_day=0) for claim in claims])
    return dict(state=state, dynamics=dynamics, coupling=coupling, curve=curve, book=book,
        claims=claims, initial_spot=100., current_claim_marks=marks, draw_count=41, seed=92, diagnostic_only=True)


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_front_scenarios_match_independent_scalar_cash_and_target_repricing(kind) -> None:
    arguments = inputs(kind)
    result = generate_front_scenarios(**arguments, pricing_batch_size=7)
    state, model, curve = (arguments[name] for name in ("state", "dynamics", "curve"))
    book, claims = arguments["book"], arguments["claims"]
    current = state.decode().pricer
    baseline = model_europeans(current, curve, business_dtes=book.bdte.to_numpy(),
        cash_strikes=book.strike.to_numpy(), is_call=(book.option_type == "call").to_numpy())
    marks = np.array([claim.price(current, curve, elapsed_day=0) for claim in claims])
    _, reference, discount = curve.at(1.)
    raw = []
    target = []
    for index, spot in enumerate(result.next_spots):
        native = (state.native + model.forecast_state_change(state.native)
            + model.spot_state_loading * result.physical_spot_returns[index]
            + model.directions @ result.factor_residuals[index])
        following = state.decode(native).pricer
        next_curve = curve.roll(1., new_spot=float(spot))
        options = model_europeans(following, next_curve, business_dtes=book.bdte.to_numpy() - 1.,
            cash_strikes=book.strike.to_numpy(), is_call=(book.option_type == "call").to_numpy())
        raw.append(np.r_[discount * (spot - reference), discount * options - baseline] / 100.)
        target.append(discount * np.array([claim.observe(elapsed_day=1, spot=float(spot)).price(
            following, next_curve, elapsed_day=1) for claim in claims]) - marks)
    np.testing.assert_allclose(result.instrument_gains + result.removed_instrument_mean, raw, atol=5e-13)
    np.testing.assert_allclose(result.claim_gains, target, atol=5e-13)
    np.testing.assert_allclose(result.weights @ result.instrument_gains, 0., atol=1e-15)
    np.testing.assert_allclose(result.removed_instrument_mean, result.weights @ np.asarray(raw), atol=1e-15)
    assert result.draw_count == 41
    assert abs(result.front_relative_mean_residual) < 1e-14
    assert result.spot_law == arguments["coupling"].law_kind
    assert result.spot_law_identity == arguments["coupling"].identity


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_batch_size_cannot_change_draws_or_weighted_scenario_law(kind, tmp_path) -> None:
    arguments = inputs(kind)
    first = generate_front_scenarios(**arguments, pricing_batch_size=3)
    second = generate_front_scenarios(**arguments, pricing_batch_size=41)
    np.testing.assert_array_equal(first.next_spots, second.next_spots)
    np.testing.assert_array_equal(first.factor_residuals, second.factor_residuals)
    np.testing.assert_array_equal(first.innovation_indices, second.innovation_indices)
    np.testing.assert_array_equal(first.innovation_frequencies, second.innovation_frequencies)
    np.testing.assert_allclose(first.instrument_gains, second.instrument_gains, atol=1e-13)
    np.testing.assert_allclose(first.claim_gains, second.claim_gains, atol=1e-13)
    assert first.innovation_frequencies.sum() == 41
    assert first.forecast_identity == second.forecast_identity
    assert first.population_innovation_covariance[0, 0] == pytest.approx(
        first.front_variance / float(arguments["curve"].at(1.)[1])**2)
    if kind is EmpiricalFrontCopula:
        assert first.innovation_indices.shape == (first.distinct_innovations, 2)
        np.testing.assert_array_equal(first.factor_residuals,
            arguments["dynamics"].residuals[first.historical_residual_indices, 1:])
    else:
        assert first.distinct_innovations == 41
        assert len(first.historical_residual_indices) == 0
    path = tmp_path / "front-scenarios.npz"
    save_verified_array_inputs(path, asdict(first))
    save_verified_array_inputs(path, asdict(first))


@pytest.mark.parametrize("kind", [EmpiricalFrontCopula, GaussianFrontCopula])
def test_historical_raw_spot_volatility_is_not_reused_for_front_sampling(kind) -> None:
    arguments = inputs(kind)
    arguments["claims"] = arguments["claims"][:2]
    arguments["current_claim_marks"] = arguments["current_claim_marks"][:2]
    first = generate_front_scenarios(**arguments)
    model = arguments["dynamics"]
    arguments["dynamics"] = replace(model, residuals=np.c_[1000. * model.residuals[:, 0], model.residuals[:, 1:]])
    second = generate_front_scenarios(**arguments)
    np.testing.assert_array_equal(first.next_spots, second.next_spots)
    np.testing.assert_array_equal(first.factor_residuals, second.factor_residuals)
    np.testing.assert_array_equal(first.instrument_gains, second.instrument_gains)


def test_one_day_digital_settlement_is_complementary_and_claim_alpha_is_not_removed() -> None:
    arguments = inputs(EmpiricalFrontCopula)
    arguments["claims"] = arguments["claims"][:2]
    arguments["curve"] = replace(arguments["curve"], forwards=np.full(4, 100.),
        reference_forwards=np.full(4, 100.), discounts=np.ones(4))
    arguments["draw_count"] = 1
    arguments["current_claim_marks"] = np.array([claim.price(arguments["state"].decode().pricer,
        arguments["curve"], elapsed_day=0) for claim in arguments["claims"]])
    result = generate_front_scenarios(**arguments)
    np.testing.assert_allclose(result.claim_gains.sum(axis=1), 0., atol=1e-15)
    assert np.max(np.abs(result.claim_gains)) > .1
    np.testing.assert_array_equal(result.instrument_gains, 0.)
    front = DiscreteFrontLaw.from_pricer(pricer=arguments["state"].decode().pricer, parity_forward=100.)
    assert result.front_variance == front.variance


def test_mismatched_coupling_date_and_expired_hedge_book_are_rejected() -> None:
    arguments = inputs(EmpiricalFrontCopula)
    arguments["coupling"] = replace(arguments["coupling"], as_of_date=date(2023, 1, 1))
    with pytest.raises(ValueError, match="same dated surface"):
        generate_front_scenarios(**arguments)
    arguments = inputs(EmpiricalFrontCopula)
    arguments["book"].loc[0, "bdte"] = 0.
    with pytest.raises(ValueError, match="Expired hedge contracts"):
        generate_front_scenarios(**arguments)


def test_common_market_target_marks_are_preserved_instead_of_replaced_by_dlv_prices() -> None:
    arguments = inputs(EmpiricalFrontCopula)
    original = generate_front_scenarios(**arguments)
    adjustment = np.array([.01, .02, .01, .03, .02])
    arguments["current_claim_marks"] += adjustment
    changed = generate_front_scenarios(**arguments)
    np.testing.assert_array_equal(original.instrument_gains, changed.instrument_gains)
    np.testing.assert_allclose(changed.claim_gains, original.claim_gains - adjustment[None, :], atol=1e-15)
    assert original.forecast_identity != changed.forecast_identity
    arguments["seed"] += 1
    assert changed.forecast_identity != generate_front_scenarios(**arguments).forecast_identity


def test_small_forecasts_cannot_be_mistaken_for_production() -> None:
    arguments = inputs(EmpiricalFrontCopula)
    arguments["diagnostic_only"] = False
    with pytest.raises(ValueError, match="Production front scenarios require"):
        generate_front_scenarios(**arguments)
