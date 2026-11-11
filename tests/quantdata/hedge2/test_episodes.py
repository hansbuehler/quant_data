"""A fixed episode numeraire and contract set must survive daily rebalancing."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.cash import CashCurve, MarketDay
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.episodes import HedgeRule, terminal_pnl


def test_episode_terminal_accounting_does_not_rebase_each_day_or_charge_exit() -> None:
    pnl = terminal_pnl(side=-1, initial_price=.4, terminal_payoff=1., discount_to_terminal=.99,
                       hedge_gains=.6, entry_costs=.001)
    assert pnl == pytest.approx(.009)
    unhedged = terminal_pnl(side=-1, initial_price=.4, terminal_payoff=1., discount_to_terminal=.99,
                            hedge_gains=0., entry_costs=0.)
    assert pnl - unhedged == pytest.approx(.599)


def test_episode_rules_reject_unagreed_objectives_and_infinite_power_risk() -> None:
    HedgeRule(method="DLV parametric", side=1, objective="mean_variance", risk_aversion=np.inf)
    with pytest.raises(ValueError):
        HedgeRule(method="DLV statistical", side=-1, objective="power_exponential", risk_aversion=np.inf)
    with pytest.raises(ValueError):
        HedgeRule(method="DLV statistical", side=0, objective="mean_variance", risk_aversion=10.)


@pytest.mark.parametrize("mixed", [False, True])
def test_greek_methods_reject_synthetic_forwards_before_loading_data(tmp_path, mixed) -> None:
    from quantdata.hedge2.episodes import run_episode

    rules = (HedgeRule(method="DLV Greek", side=-1),)
    if mixed:
        rules += (HedgeRule(method="DLV parametric", side=-1,
            objective="mean_variance", risk_aversion=10.),)
    with pytest.raises(ValueError, match="Greek-only hedges exclude synthetic"):
        run_episode(origin=date(2023, 1, 9), family="digital-1D", store=None, archive=None,
            daily_directory=tmp_path, dynamics_directory=tmp_path, rules=rules,
            option_side="itm", include_co_terminal_forwards=True, optimized_premium_cap=10.)


def test_knockout_ends_hedging_at_observation_without_charging_exit(monkeypatch, tmp_path) -> None:
    from quantdata.hedge2.episodes import EpisodeCheckpointedStop, run_episode

    origin = date(2023, 1, 9)
    sessions = get_business_days(origin, origin + timedelta(days=15))[:6]
    book = pd.DataFrame({"expiry": pd.to_datetime([sessions[3], sessions[5]]),
        "strike": [95., 95.], "option_type": ["call", "call"], "is_am": [False, False],
        "bdte": [3., 5.], "observed_mid": [10., 10.], "observed_half_spread": [.1, .1]})
    markets = {}
    for elapsed, day in enumerate(sessions):
        curve = CashCurve(spot=100. + elapsed, business_dtes=np.array([1., 3., 5., 25.]),
            forwards=np.full(4, 100. + elapsed), reference_forwards=np.full(4, 100. + elapsed),
            discounts=np.ones(4))
        markets[day] = MarketDay(day=day, quotes=book.copy(), curve=curve)
    source = SimpleNamespace(interpolate_atm_vols=lambda values: np.full_like(np.asarray(values), .2))
    archive = SimpleNamespace(load=lambda day: (source, {}))
    claim = Claim(label="ko", kind="ko", terminal_day=5, observation_day=3, initial_spot=100.,
                  strike=100., barrier=100., is_call=False)
    monkeypatch.setattr("quantdata.hedge2.episodes.make_claims", lambda **kwargs: (claim,))
    monkeypatch.setattr("quantdata.hedge2.episodes.select_book", lambda *args, **kwargs: book.copy())
    monkeypatch.setattr(Claim, "price", lambda self, *args, **kwargs: .2 if not self.knocked_out else 0.)
    state = SimpleNamespace(decode=lambda: SimpleNamespace(pricer=object()))
    monkeypatch.setattr("quantdata.hedge2.episodes.DailyState.load", lambda *args: state)
    monkeypatch.setattr("quantdata.hedge2.episodes.load_dynamics", lambda path: SimpleNamespace(
        training_start=date(2022, 1, 3), training_end=date.fromisoformat(path.stem), spot_state_loading=np.zeros(2)))
    monkeypatch.setattr("quantdata.hedge2.episodes.calculate_greeks", lambda **kwargs: SimpleNamespace(
        instrument_exposures=np.ones((1 + len(kwargs["book"]), 2)), claim_exposures=np.ones((1, 2))))
    monkeypatch.setattr("quantdata.hedge2.episodes.factor_risk_covariance", lambda *args, **kwargs: np.eye(2))
    solves = []

    def solve(**kwargs):
        solves.append(1)
        positions = np.r_[0., np.ones(len(kwargs["costs"]) - 1)]
        cost = float(kwargs["costs"] @ np.abs(positions - kwargs["previous_positions"]))
        return SimpleNamespace(positions=positions, entry_cost=cost, full_entry_cost=cost,
                               risk_residual_norm=0., solver="test", status="test", warnings=())

    monkeypatch.setattr("quantdata.hedge2.episodes.solve_greek_hedge", solve)
    dynamics = tmp_path / "dynamics"
    (dynamics / "days").mkdir(parents=True)
    for day in sessions[:-1]:
        (dynamics / "days" / f"{day.isoformat()}.npz").touch()
    result = run_episode(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
        archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
        rules=(HedgeRule(method="DLV Greek", side=-1),), checkpoint_directory=tmp_path / "steps")
    record = result.summary.iloc[0]
    assert len(solves) == 3
    assert record.hedge_end_date == sessions[3].isoformat()
    assert record.termination_reason == "knockout"
    assert record.entry_cost == pytest.approx(.002)
    assert record.terminal_hypothetical_exit_cost == pytest.approx(.001)
    assert record.terminal_payoff == 0.
    later = result.trades.loc[result.trades.elapsed_day >= 3]
    np.testing.assert_array_equal(later[["entry_cost", "hedge_gain"]], 0.)
    assert len(list((tmp_path / "steps").glob("*/completed_step.npz"))) == 5
    with np.load(tmp_path / "steps" / "step-02" / "completed_step.npz") as saved:
        assert bool(saved["terminated"][0])
        np.testing.assert_array_equal(saved["holdings_after_hedge"], 0.)
    restored = run_episode(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
        archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
        rules=(HedgeRule(method="DLV Greek", side=-1),), checkpoint_directory=tmp_path / "steps")
    assert len(solves) == 3
    pd.testing.assert_frame_equal(restored.summary, result.summary)
    pd.testing.assert_frame_equal(restored.trades, result.trades)
    with pytest.raises(EpisodeCheckpointedStop):
        run_episode(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
            archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
            rules=(HedgeRule(method="DLV Greek", side=-1),), checkpoint_directory=tmp_path / "interrupted",
            stop_requested=lambda: True)
    resumed = run_episode(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
        archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
        rules=(HedgeRule(method="DLV Greek", side=-1),), checkpoint_directory=tmp_path / "interrupted")
    pd.testing.assert_frame_equal(resumed.summary, result.summary)
    pd.testing.assert_frame_equal(resumed.trades, result.trades)
    (dynamics / "days" / f"{origin.isoformat()}.npz").write_bytes(b"changed fitted model")
    with pytest.raises(ValueError, match="inputs.*changed"):
        run_episode(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
            archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
            rules=(HedgeRule(method="DLV Greek", side=-1),), checkpoint_directory=tmp_path / "steps")


def test_current_itm_black_reference_preserves_uncapped_positions_and_checkpoint(monkeypatch, tmp_path) -> None:
    from quantdata.hedge2.episodes import run_episode

    origin = date(2023, 1, 9)
    sessions = get_business_days(origin, origin + timedelta(days=15))[:6]
    book = pd.DataFrame({"expiry": pd.to_datetime([sessions[3], sessions[3], sessions[5], sessions[5]]),
        "strike": [0., 95., 0., 95.], "option_type": ["call"] * 4, "is_am": [False] * 4,
        "bdte": [3., 3., 5., 5.], "observed_mid": [100., 10., 100., 10.],
        "observed_half_spread": [0., .1, 0., .1],
        "instrument_kind": ["zero_strike_call", "european"] * 2})
    markets = {}
    for elapsed, day in enumerate(sessions):
        curve = CashCurve(spot=100. + elapsed, business_dtes=np.array([1., 3., 5., 25.]),
            forwards=np.full(4, 100. + elapsed), reference_forwards=np.full(4, 100. + elapsed),
            discounts=np.ones(4))
        markets[day] = MarketDay(day=day, quotes=book.copy(), curve=curve)
    calls = []

    def price(**kwargs):
        assert np.all(kwargs["strikes"] > 0.)
        calls.append(kwargs["strikes"])
        return np.zeros_like(kwargs["strikes"]), np.full_like(kwargs["strikes"], .2)

    source = SimpleNamespace(interpolate_atm_vols=lambda values: np.full_like(np.asarray(values), .2), price=price)
    monkeypatch.setattr("quantdata.hedge2.episodes.surface_implied_vols", lambda source, **kwargs:
        price(strikes=kwargs["strikes"])[1])
    archive = SimpleNamespace(load=lambda day: (source, {}))
    claim = Claim(label="ko", kind="ko", terminal_day=5, observation_day=3, initial_spot=100.,
                  strike=100., barrier=100., is_call=False)
    monkeypatch.setattr("quantdata.hedge2.episodes.make_claims", lambda **kwargs: (claim,))

    def selected_book(*args, **kwargs):
        assert kwargs["option_side"] == "itm"
        assert kwargs["include_co_terminal_forwards"] is True
        return book.copy()

    monkeypatch.setattr("quantdata.hedge2.episodes.select_book", selected_book)
    monkeypatch.setattr(Claim, "price", lambda self, *args, **kwargs: .2 if not self.knocked_out else 0.)
    state = SimpleNamespace(decode=lambda: SimpleNamespace(pricer=object()))
    monkeypatch.setattr("quantdata.hedge2.episodes.DailyState.load", lambda *args: state)
    monkeypatch.setattr("quantdata.hedge2.episodes.load_dynamics", lambda path: SimpleNamespace(
        training_start=date(2022, 1, 3), training_end=date.fromisoformat(path.stem)))
    monkeypatch.setattr("quantdata.hedge2.episodes.black_exotic_hedge", lambda **kwargs: SimpleNamespace(
        positions=np.r_[2000., (kwargs["book"]["strike"].to_numpy() > 0.).astype(float)]))
    dynamics = tmp_path / "dynamics"
    (dynamics / "days").mkdir(parents=True)
    for day in sessions[:-1]:
        (dynamics / "days" / f"{day.isoformat()}.npz").touch()
    arguments = dict(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
        archive=archive, daily_directory=tmp_path, dynamics_directory=dynamics,
        rules=(HedgeRule(method="BS delta+vega", side=-1),), checkpoint_directory=tmp_path / "steps",
        option_side="itm", include_co_terminal_forwards=True, reference_premium_cap=10.)
    result = run_episode(**arguments)
    record = result.summary.iloc[0]
    assert calls
    assert record.reference_is_uncapped
    assert record.cap_days == 0
    assert record.premium_comparison_breach_days == 3
    assert record.maximum_absolute_premium_position == pytest.approx(2040.)
    assert record.entry_cost == pytest.approx(.002)
    assert record.terminal_hypothetical_exit_cost == pytest.approx(.001)
    np.testing.assert_array_equal(result.positions[0][0], [2000., 0., 1., 0., 1.])
    restored = run_episode(**arguments)
    pd.testing.assert_frame_equal(restored.summary, result.summary)
    with pytest.raises(ValueError, match="inputs.*changed"):
        run_episode(**(arguments | {"reference_premium_cap": 20.}))
    (tmp_path / "config.json").write_text('{"decoder": "changed"}\n')
    with pytest.raises(ValueError, match="inputs.*changed"):
        run_episode(**arguments)


@pytest.mark.parametrize("rule", [HedgeRule(method="DLV Greek", side=-1),
    HedgeRule(method="DLV parametric", side=-1, objective="mean_variance", risk_aversion=10.),
    HedgeRule(method="DLV parametric", side=1, objective="power_exponential", risk_aversion=10.)])
def test_current_premium_optimizers_preserve_inventory_fees_and_free_knockout(
        rule, monkeypatch, tmp_path) -> None:
    from quantdata.hedge2.episodes import run_episode

    origin = date(2023, 1, 9)
    sessions = get_business_days(origin, origin + timedelta(days=15))[:6]
    tenors = np.array([1., 3., 5., 25.])
    book = pd.DataFrame({"expiry": pd.to_datetime([sessions[3], sessions[5]]),
        "strike": [95., 95.], "option_type": ["call", "call"], "is_am": [False, False],
        "bdte": [3., 5.], "observed_mid": [10., 10.], "observed_half_spread": [.1, .1]})
    markets = {day: MarketDay(day=day, quotes=book.copy(), curve=CashCurve(spot=100. + elapsed,
        business_dtes=tenors, forwards=np.full(4, 100. + elapsed),
        reference_forwards=np.full(4, 100. + elapsed), discounts=.99**tenors))
        for elapsed, day in enumerate(sessions)}
    source = SimpleNamespace(interpolate_atm_vols=lambda values: np.full_like(np.asarray(values), .2))
    claim = Claim(label="ko", kind="ko", terminal_day=5, observation_day=3, initial_spot=100.,
        strike=100., barrier=100., is_call=False)
    monkeypatch.setattr("quantdata.hedge2.episodes.make_claims", lambda **kwargs: (claim,))
    monkeypatch.setattr("quantdata.hedge2.episodes.select_book", lambda *args, **kwargs: book.copy())
    monkeypatch.setattr(Claim, "price", lambda self, *args, **kwargs: .2 if not self.knocked_out else 0.)
    monkeypatch.setattr("quantdata.hedge2.episodes.DailyState.load", lambda *args:
        SimpleNamespace(decode=lambda: SimpleNamespace(pricer=object())))
    monkeypatch.setattr("quantdata.hedge2.episodes.load_dynamics", lambda path: SimpleNamespace(
        training_start=date(2022, 1, 3), training_end=date.fromisoformat(path.stem), spot_state_loading=np.zeros(2)))

    def exposures(**kwargs):
        assert kwargs["elapsed_day"] < 3
        return SimpleNamespace(instrument_exposures=np.r_[np.array([[1., 0.]]),
            np.tile([.3, .4], (len(kwargs["book"]), 1))], claim_exposures=np.array([[.4, .5]]))

    monkeypatch.setattr("quantdata.hedge2.episodes.calculate_greeks", exposures)
    monkeypatch.setattr("quantdata.hedge2.episodes.calculate_parametric_greeks", exposures)
    monkeypatch.setattr("quantdata.hedge2.episodes.factor_risk_covariance", lambda *args, **kwargs: np.diag([.01, .04]))
    monkeypatch.setattr("quantdata.hedge2.episodes.parametric_factor_covariance", lambda *args, **kwargs: np.diag([.01, .04]))
    dynamics = tmp_path / "dynamics"
    (dynamics / "days").mkdir(parents=True)
    for day in sessions[:-1]:
        (dynamics / "days" / f"{day.isoformat()}.npz").touch()
    arguments = dict(origin=origin, family="ko", store=SimpleNamespace(load=markets.__getitem__),
        archive=SimpleNamespace(load=lambda day: (source, {})), daily_directory=tmp_path,
        dynamics_directory=dynamics, rules=(rule,), checkpoint_directory=tmp_path / "steps",
        option_side="itm", include_co_terminal_forwards=rule.method != "DLV Greek", optimized_premium_cap=.15)
    result = run_episode(**arguments)
    record = result.summary.iloc[0]
    assert record.termination_reason == "knockout"
    assert not record.reference_is_uncapped
    assert record.optimized_premium_cap == .15
    assert record.maximum_absolute_premium_position <= .15 * (1. + 1e-8)
    np.testing.assert_array_equal(result.trades.loc[result.trades.elapsed_day >= 3,
        ["entry_cost", "hedge_gain"]], 0.)
    for elapsed in range(3):
        step = tmp_path / "steps" / f"step-{elapsed:02d}"
        with np.load(step / "risk_inputs.npz", allow_pickle=False) as saved:
            position = result.positions[elapsed][0]
            old = saved["previous_positions"][0]
            cash_cost = saved["costs"] @ np.abs(position - old)
            assert cash_cost == pytest.approx(result.trades.iloc[elapsed].entry_cost, abs=1e-12)
            assert saved["normalized_premium_prices"][0] == pytest.approx((100. + elapsed) / 100.)
            assert np.max(np.abs(position * saved["normalized_premium_prices"])) <= .15 * (1. + 1e-8)
            if elapsed:
                np.testing.assert_allclose(old, result.positions[elapsed - 1][0])
    restored = run_episode(**arguments)
    pd.testing.assert_frame_equal(restored.summary, result.summary)
    pd.testing.assert_frame_equal(restored.trades, result.trades)
    with pytest.raises(ValueError, match="inputs.*changed"):
        run_episode(**(arguments | {"optimized_premium_cap": .2}))


def test_one_day_greek_keeps_the_entire_short_dated_book_and_charges_entry_only(monkeypatch, tmp_path) -> None:
    from quantdata.hedge2.episodes import run_episode

    origin = date(2023, 1, 6)
    sessions = get_business_days(origin, origin + timedelta(days=15))[:6]
    book = pd.DataFrame({"expiry": pd.to_datetime([sessions[1], sessions[3], sessions[5]]),
        "strike": [95.] * 3, "option_type": ["call"] * 3, "is_am": [False] * 3,
        "bdte": [1., 3., 5.], "observed_mid": [10.] * 3, "observed_half_spread": [.1] * 3})
    tenors = np.array([1., 3., 5., 25.])
    markets = {day: MarketDay(day=day, quotes=book.copy(), curve=CashCurve(spot=100. + elapsed,
        business_dtes=tenors, forwards=np.full(4, 100. + elapsed),
        reference_forwards=np.full(4, 100. + elapsed), discounts=np.ones(4)))
        for elapsed, day in enumerate(sessions[:2])}
    source = SimpleNamespace(interpolate_atm_vols=lambda values: np.full_like(np.asarray(values), .2))
    claim = Claim(label="digital-1D-call", kind="digital", terminal_day=1,
        initial_spot=100., strike=100., is_call=True)
    monkeypatch.setattr("quantdata.hedge2.episodes.make_claims", lambda **kwargs: (claim,))
    selected = []

    def select(*args, **kwargs):
        selected.append(kwargs)
        return book.copy()

    monkeypatch.setattr("quantdata.hedge2.episodes.select_book", select)
    monkeypatch.setattr(Claim, "price", lambda self, pricer, curve, *, elapsed_day: .4)
    monkeypatch.setattr("quantdata.hedge2.episodes.DailyState.load", lambda *args:
        SimpleNamespace(decode=lambda: SimpleNamespace(pricer=object())))
    monkeypatch.setattr("quantdata.hedge2.episodes.load_dynamics", lambda path: SimpleNamespace(
        training_start=date(2022, 1, 3), training_end=origin, spot_state_loading=np.zeros(2)))
    monkeypatch.setattr("quantdata.hedge2.episodes.calculate_greeks", lambda **kwargs: SimpleNamespace(
        instrument_exposures=np.array([[1., 0.], [.1, 0.], [.2, 0.], [.3, .4]]),
        claim_exposures=np.array([[.4, .5]])))
    monkeypatch.setattr("quantdata.hedge2.episodes.factor_risk_covariance", lambda *args, **kwargs: np.eye(2))
    dynamics = tmp_path / "dynamics"
    (dynamics / "days").mkdir(parents=True)
    (dynamics / "days" / f"{origin.isoformat()}.npz").touch()
    result = run_episode(origin=origin, family="digital-1D", store=SimpleNamespace(load=markets.__getitem__),
        archive=SimpleNamespace(load=lambda day: (source, {})), daily_directory=tmp_path,
        dynamics_directory=dynamics, rules=(HedgeRule(method="DLV Greek", side=-1),),
        checkpoint_directory=tmp_path / "steps", option_side="itm", include_co_terminal_forwards=False,
        optimized_premium_cap=10.)
    assert selected[0]["co_terminal_business_dtes"] is None
    assert selected[0]["option_side"] == "itm"
    assert selected[0]["include_co_terminal_forwards"] is False
    assert len(result.initial_book) == 3
    assert len(result.positions) == 1
    assert result.positions[0][0, 3] == pytest.approx(1.25)
    record = result.summary.iloc[0]
    assert record.terminal_date == sessions[1].isoformat()
    assert record.terminal_payoff == 1.
    assert record.terminal_hypothetical_exit_cost > 0.
    assert record.costed_pnl == pytest.approx(-.6 + record.hedge_gain - record.entry_cost)
    assert record.entry_cost == pytest.approx(.00125)
