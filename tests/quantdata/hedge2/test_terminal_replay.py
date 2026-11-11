"""Remaining-horizon policies trade daily fresh models without future-market inputs."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2 import terminal_replay
from quantdata.hedge2.audit_terminal_replay import audit_terminal_replay
from quantdata.hedge2.cash import MarketDay, model_europeans, observed_marks
from quantdata.hedge2.claims import Claim
from quantdata.hedge2.terminal_cache import ForecastCheckpointStop
from quantdata.hedge2.terminal_market import TerminalSimulationConfig
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig
from quantdata.hedge2.terminal_replay import load_origin_decision, replay_terminal_episode
from quantdata.hedge2.terminal_train import TerminalTrainConfig
from quantdata.torch import BaseEnvironment
from tests.quantdata.hedge2.test_terminal_market import inputs


def replay_inputs(monkeypatch, root: Path, *, kind="digital", horizon=3):
    original = inputs()
    origin = original["state"].day
    sessions = get_business_days(origin, origin + timedelta(days=15))[:horizon + 1]
    tenors = (horizon,) if kind == "digital" else (3, 5)
    book = original["book"].loc[original["book"].bdte.isin(tenors)].copy().reset_index(drop=True)
    book["expiry"] = pd.to_datetime([sessions[int(tenor)] for tenor in book.bdte])
    book["is_am"], book["date"] = False, pd.Timestamp(origin)
    claim = Claim(label="target", kind=kind, terminal_day=horizon, initial_spot=100.,
        strike=1. if kind == "ratio_straddle" else 100., is_call=kind != "ko",
        observation_day=0 if kind == "digital" else 3, barrier=100. if kind == "ko" else 0.)
    markets, states, models, events = {}, {}, {}, []
    daily, dynamics = root / "daily", root / "dynamics"
    for directory in (daily, dynamics):
        (directory / "days").mkdir(parents=True)
    (daily / "config.json").write_text(json.dumps(asdict(original["state"].config)))
    for elapsed, day in enumerate(sessions):
        spot = 100. + elapsed
        curve = replace(original["curve"], spot=spot,
            forwards=original["curve"].forwards * spot / 100.,
            reference_forwards=original["curve"].reference_forwards * spot / 100.)
        quoted = book.loc[book.bdte >= elapsed].copy()
        quoted["bdte"] -= elapsed
        quoted["date"] = pd.Timestamp(day)
        quoted["observed_mid"] = model_europeans(original["state"].decode().pricer, curve,
            business_dtes=quoted.bdte.to_numpy(), cash_strikes=quoted.strike.to_numpy(),
            is_call=(quoted.option_type == "call").to_numpy())
        markets[day] = MarketDay(day=day, quotes=quoted, curve=curve)
        states[day] = replace(original["state"], day=day)
        model = deepcopy(original["dynamics"])
        model.as_of_date = model.training_end = day
        model.training_start = date(2023, 1, 3)
        models[day] = model
        for directory in (daily, dynamics):
            (directory / "days" / f"{day}.npz").write_text(day.isoformat())

    def load_market(day):
        events.append(("market", day))
        return markets[day]

    source = SimpleNamespace(interpolate_atm_vols=lambda values: np.full_like(np.asarray(values, dtype=float), .2))
    archive = SimpleNamespace(load=lambda day: (source, {"source_date": day.isoformat(), "source_smoothness": .1}))
    monkeypatch.setattr(terminal_replay, "make_claims", lambda **kwargs: (claim,))
    monkeypatch.setattr(terminal_replay, "select_book", lambda *args, **kwargs: book.copy())
    monkeypatch.setattr(terminal_replay.DailyState, "load", lambda directory, day: states[day])
    monkeypatch.setattr(terminal_replay, "load_dynamics", lambda path: models[date.fromisoformat(path.stem)])
    wrapped = terminal_replay.fit_terminal_decision

    def fit(**kwargs):
        events.append(("fit", kwargs["state"].day, kwargs["initial_inventory"].copy(), kwargs["claim"]))
        return wrapped(**kwargs)

    monkeypatch.setattr(terminal_replay, "fit_terminal_decision", fit)
    arguments = dict(origin=origin, claim_label="target", store=SimpleNamespace(load=load_market), archive=archive,
        daily_directory=daily, dynamics_directory=dynamics, output_directory=root / "result",
        objective="mean_variance", risk_aversion=10., target_side=-1,
        env=BaseEnvironment(device="cpu", precision=64),
        train_config=TerminalTrainConfig(maximum_updates=2, minimum_updates=1, check_every=1, stable_checks=100),
        simulation_config=TerminalSimulationConfig(paths=8, integration_parent_batch=2),
        model_config=TerminalPolicyConfig(label="replay", feature_count=5, use_deep=False), diagnostic_only=True)
    return arguments, markets, events, sessions


def test_replay_uses_each_days_model_and_actual_inventory_before_observing_next_close(monkeypatch, tmp_path: Path) -> None:
    arguments, markets, events, sessions = replay_inputs(monkeypatch, tmp_path)
    result = replay_terminal_episode(**arguments)
    fits = [event for event in events if event[0] == "fit"]
    assert result.status == "complete"
    assert [event[1] for event in fits] == sessions[:-1]
    assert result.origin_decision.state.day == sessions[0]
    assert result.origin_decision.elapsed_day == 0
    assert not result.summary["all_fits_stable"]
    assert result.summary["diagnostic_only"]
    for step, day in enumerate(sessions[:-1]):
        fit_index = next(index for index, event in enumerate(events) if event[:2] == ("fit", day))
        next_index = next(index for index, event in enumerate(events) if event[:2] == ("market", sessions[step + 1]))
        assert fit_index < next_index
        with np.load(arguments["output_directory"] / f"step-{step:02d}" / "realized.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(fits[step][2], saved["previous_positions"])
            if step:
                with np.load(arguments["output_directory"] / f"step-{step-1:02d}" / "realized.npz", allow_pickle=False) as prior:
                    np.testing.assert_array_equal(saved["previous_positions"], prior["holdings_after"][saved["columns"]])
    assert np.any(fits[1][2] != 0.)
    assert result.trades.fit_status.eq("training_limit").all()


@pytest.mark.parametrize("dependency", ("terminal_adequacy.py", "terminal_schedule.py"))
def test_replay_resume_binds_reference_gate_and_training_schedule_code(monkeypatch, tmp_path: Path, dependency: str) -> None:
    arguments, _, _, _ = replay_inputs(monkeypatch, tmp_path)
    replay_terminal_episode(**arguments)
    saved = json.loads((arguments["output_directory"] / "config.json").read_text())
    assert dependency in saved["implementation_sha256"]
    original = Path.read_bytes

    def changed_source(path):
        content = original(path)
        return content + b"\n" if path.name == dependency else content

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    with pytest.raises(ValueError, match="Terminal replay scope"):
        replay_terminal_episode(**arguments)


def test_saved_cashflows_replay_independently_without_forecast_demeaning_or_exit_fees(monkeypatch, tmp_path: Path) -> None:
    arguments, markets, _, sessions = replay_inputs(monkeypatch, tmp_path)
    result = replay_terminal_episode(**arguments)
    initial_book = pd.read_parquet(arguments["output_directory"] / "initial_book.parquet")
    discount, total_gain, total_cost = 1., 0., 0.
    previous = np.zeros(1 + len(initial_book))
    for elapsed, current_day in enumerate(sessions[:-1]):
        current, following = markets[current_day], markets[sessions[elapsed + 1]]
        active = np.flatnonzero(initial_book.bdte.to_numpy() > elapsed + 1e-6)
        book = initial_book.iloc[active]
        marks, spreads = observed_marks(book, current)
        next_marks, _ = observed_marks(book, following)
        _, reference, one_day_discount = current.curve.at(1.)
        with np.load(arguments["output_directory"] / f"step-{elapsed:02d}" / "realized.npz", allow_pickle=False) as saved:
            positions = saved["positions"]
            raw = discount * np.r_[one_day_discount * (following.curve.spot - reference),
                one_day_discount * next_marks - marks] / markets[sessions[0]].curve.spot
            total_gain += float(positions @ raw)
            total_cost += float(discount * np.abs(positions[1:] - previous[1 + active]) @ spreads / 100.)
            previous = saved["holdings_after"]
        discount *= one_day_discount
    # The short digital's premium is the inception zero-smoothing DLV value (LV mode).
    forward, _, discount_to_terminal = markets[sessions[0]].curve.at(3.)
    premium = float(discount_to_terminal * inputs()["state"].decode().pricer.european_at(expiry_y=3. / DAYS_PER_YEAR,
        strikes=np.array([100. / forward]), is_call=True, digital=True)[0])
    expected_unhedged = premium - discount
    assert result.summary["hedge_gain"] == pytest.approx(total_gain, abs=1e-13)
    assert result.summary["entry_cost"] == pytest.approx(total_cost, abs=1e-13)
    assert result.summary["unhedged_pnl"] == pytest.approx(expected_unhedged, abs=1e-13)
    assert result.summary["costed_pnl"] == pytest.approx(expected_unhedged + total_gain - total_cost, abs=1e-13)


def test_daily_checkpoint_resume_does_not_retrain_completed_steps(monkeypatch, tmp_path: Path) -> None:
    arguments, _, events, _ = replay_inputs(monkeypatch, tmp_path)
    stopped = replay_terminal_episode(**arguments, stop_requested=lambda: len([event for event in events if event[0] == "fit"]) >= 1)
    assert stopped.status == "checkpointed_stop"
    assert stopped.summary["completed_steps"] == 1
    resumed = replay_terminal_episode(**arguments)
    assert resumed.status == "complete"
    assert len([event for event in events if event[0] == "fit"]) == 3
    repeated = replay_terminal_episode(**arguments)
    assert repeated.summary == resumed.summary
    assert len([event for event in events if event[0] == "fit"]) == 3


def test_sides_reuse_identical_forecasts_without_sharing_actual_inventory(monkeypatch, tmp_path: Path) -> None:
    arguments, _, events, _ = replay_inputs(monkeypatch, tmp_path)
    generate = terminal_replay.generate_terminal_paths
    generated = []

    def track(**kwargs):
        generated.append(kwargs["state"].day)
        return generate(**kwargs)

    monkeypatch.setattr(terminal_replay, "generate_terminal_paths", track)
    arguments["shared_forecast_directory"] = tmp_path / "shared"
    short = replay_terminal_episode(**arguments)
    short_inventory = [event[2].copy() for event in events if event[0] == "fit"]
    long_directory = tmp_path / "long"
    long = replay_terminal_episode(**{**arguments, "target_side": 1, "output_directory": long_directory})
    assert short.status == long.status == "complete"
    assert len(generated) == 3
    fits = [event for event in events if event[0] == "fit"]
    for step in range(3):
        first = arguments["output_directory"] / f"step-{step:02d}" / "forecast.npz"
        second = long_directory / f"step-{step:02d}" / "forecast.npz"
        assert first.stat().st_ino == second.stat().st_ino
        record = json.loads(first.with_suffix(".json").read_text())
        assert record["identity"]["simulation_version"] == TerminalSimulationConfig.version.unique_id48
        assert {"terminal_market.py", "dlv.py", "cash.py", "state.py"} <= set(record["identity"]["implementation_sha256"])
        with np.load(long_directory / f"step-{step:02d}" / "realized.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(fits[3 + step][2], saved["previous_positions"])
    assert not np.array_equal(short_inventory[1], fits[4][2])
    assert audit_terminal_replay(directory=long_directory, store=arguments["store"])["status"] == "passed"


def test_waiting_forecast_stop_is_checkpointed_not_a_numerical_failure(monkeypatch, tmp_path: Path) -> None:
    arguments, _, events, _ = replay_inputs(monkeypatch, tmp_path)

    def stop(*args, **kwargs):
        raise ForecastCheckpointStop("Stop requested")

    monkeypatch.setattr(terminal_replay, "_saved_forecast", stop)
    result = replay_terminal_episode(**arguments)
    assert result.status == "checkpointed_stop"
    assert result.summary["completed_steps"] == 0
    assert not any(event[0] == "fit" for event in events)


def test_knocked_out_episode_stops_at_day_three_and_clears_all_inventory_for_free(monkeypatch, tmp_path: Path) -> None:
    arguments, _, events, sessions = replay_inputs(monkeypatch, tmp_path, kind="ko", horizon=5)
    result = replay_terminal_episode(**arguments)
    assert result.status == "complete"
    assert result.summary["hedge_end_date"] == sessions[3].isoformat()
    assert result.summary["terminal_payoff"] == 0.
    assert len([event for event in events if event[0] == "fit"]) == 3
    assert not any(event[:2] == ("market", sessions[4]) for event in events)
    with np.load(arguments["output_directory"] / "step-02" / "realized.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["holdings_after"], 0.)


def test_ratio_reset_is_actual_day_three_spot_in_each_later_daily_fit(monkeypatch, tmp_path: Path) -> None:
    arguments, markets, events, sessions = replay_inputs(monkeypatch, tmp_path, kind="ratio_straddle", horizon=5)
    result = replay_terminal_episode(**arguments)
    assert result.status == "complete"
    fits = [event for event in events if event[0] == "fit"]
    assert all(event[3].reset_spot is None for event in fits[:3])
    assert all(event[3].reset_spot == markets[sessions[3]].curve.spot for event in fits[3:])
    expected = abs(markets[sessions[5]].curve.spot / markets[sessions[3]].curve.spot - 1.)
    assert result.summary["terminal_payoff"] == pytest.approx(expected)


def test_unfinished_production_fit_cannot_trade_or_observe_next_market(monkeypatch, tmp_path: Path) -> None:
    arguments, _, events, sessions = replay_inputs(monkeypatch, tmp_path)
    arguments["diagnostic_only"] = False
    monkeypatch.setattr(terminal_replay, "fit_terminal_decision", lambda **kwargs: SimpleNamespace(
        accepted_for_execution=False, fit=SimpleNamespace(status="training_limit")))
    result = replay_terminal_episode(**arguments)
    assert result.status == "training_incomplete"
    assert result.trades.empty
    assert not any(event[:2] == ("market", sessions[1]) for event in events)


def test_replay_resume_rejects_changed_realized_evidence(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, _ = replay_inputs(monkeypatch, tmp_path)
    replay_terminal_episode(**arguments)
    path = arguments["output_directory"] / "step-01" / "realized.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        replay_terminal_episode(**arguments)


def test_next_overlapping_origin_loads_inception_not_the_final_future_policy(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, sessions = replay_inputs(monkeypatch, tmp_path)
    replay_terminal_episode(**arguments)
    loaded = load_origin_decision(arguments["output_directory"], env=arguments["env"], daily_directory=arguments["daily_directory"])
    assert loaded.state.day == sessions[0]
    assert loaded.fit.policy.as_of_date == sessions[0]
    assert loaded.elapsed_day == 0


def test_missing_following_market_preserves_current_inception_for_causal_warm_start(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, sessions = replay_inputs(monkeypatch, tmp_path)
    original = arguments["store"].load

    def load(day):
        if day == sessions[1]:
            raise FileNotFoundError("Missing following observed market")
        return original(day)

    arguments["store"].load = load
    with pytest.raises(FileNotFoundError, match="following observed market"):
        replay_terminal_episode(**arguments)
    record = json.loads((arguments["output_directory"] / "episode.json").read_text())
    assert record["next_step"] == 0
    assert record["trades"] == []
    loaded = load_origin_decision(arguments["output_directory"], env=arguments["env"], daily_directory=arguments["daily_directory"])
    assert loaded.state.day == sessions[0]
    assert loaded.elapsed_day == 0


@pytest.mark.parametrize("missing_step", (0, 1))
def test_resume_after_missing_market_keeps_the_decided_policy_immutable(monkeypatch, tmp_path: Path, missing_step) -> None:
    arguments, _, events, sessions = replay_inputs(monkeypatch, tmp_path)
    original = arguments["store"].load

    def load(day):
        if day == sessions[missing_step + 1]:
            raise ValueError("Missing observed mids on following date")
        return original(day)

    arguments["store"].load = load
    with pytest.raises(ValueError, match="Missing observed mids"):
        replay_terminal_episode(**arguments)
    record = json.loads((arguments["output_directory"] / "episode.json").read_text())
    pending = record["pending_decision"]
    checkpoint = Path(pending["files"]["checkpoint"]["path"])
    before = checkpoint.read_bytes()
    arguments["store"].load = original
    arguments["train_config"] = replace(arguments["train_config"], maximum_updates=4)
    resumed = replay_terminal_episode(**arguments)
    assert resumed.status == "complete"
    assert checkpoint.read_bytes() == before
    assert len([event for event in events if event[0] == "fit"]) == 3
    assert audit_terminal_replay(directory=arguments["output_directory"], store=arguments["store"])["status"] == "passed"


@pytest.mark.parametrize("kind,horizon", [("digital", 3), ("ko", 5), ("ratio_straddle", 5)])
def test_independent_audit_matches_actual_market_cash_and_target_lifecycle(monkeypatch, tmp_path: Path, kind, horizon) -> None:
    arguments, _, _, _ = replay_inputs(monkeypatch, tmp_path, kind=kind, horizon=horizon)
    replay_terminal_episode(**arguments)
    result = audit_terminal_replay(directory=arguments["output_directory"], store=arguments["store"])
    assert result["status"] == "passed"
    assert result["maximum_cash_error"] < 1e-12
    assert result["diagnostic_only"]


def test_changed_decoder_config_cannot_restore_an_old_origin_policy(monkeypatch, tmp_path: Path) -> None:
    arguments, _, _, _ = replay_inputs(monkeypatch, tmp_path)
    replay_terminal_episode(**arguments)
    path = arguments["daily_directory"] / "config.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="input or checkpoint changed"):
        load_origin_decision(arguments["output_directory"], env=arguments["env"], daily_directory=arguments["daily_directory"])
