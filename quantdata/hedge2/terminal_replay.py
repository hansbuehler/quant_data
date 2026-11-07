"""Daily fresh-model, actual-inventory replay of a terminal-risk hedge episode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import (
    MarketDay,
    MarketStore,
    discounted_daily_gains,
    observed_marks,
    select_book,
)
from quantdata.hedge2.claims import Claim, make_claims
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive, surface_digitals
from quantdata.hedge2.state import DailyState
from quantdata.hedge2.terminal_cache import ForecastCheckpointStop, saved_terminal_forecast
from quantdata.hedge2.terminal_decision import TerminalDecision, fit_terminal_decision
from quantdata.hedge2.terminal_market import (
    TerminalMarketPaths,
    TerminalSimulationConfig,
    generate_terminal_paths,
)
from quantdata.hedge2.terminal_policy import TerminalPolicyConfig
from quantdata.hedge2.terminal_train import (
    TerminalPolicyFit,
    TerminalTrainConfig,
    load_terminal_policy,
)
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True, kw_only=True)
class TerminalReplayResult:
    status: str
    summary: dict[str, object]
    trades: pd.DataFrame
    origin_decision: TerminalDecision | None


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.dtype.str.encode() + str(contiguous.shape).encode() + contiguous.tobytes()).hexdigest()


def _book_digest(book: pd.DataFrame) -> str:
    return _array_digest(pd.util.hash_pandas_object(book, index=False).to_numpy())


def _market_digest(market: MarketDay) -> str:
    parts = [_book_digest(market.quotes)]
    parts.extend(_array_digest(np.asarray(value)) for value in asdict(market.curve).values())
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _atomic_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _save_book(path: Path, book: pd.DataFrame) -> None:
    if path.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(path), book, check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError("A saved frozen terminal hedge book changed") from error
        return
    temporary = path.with_suffix(".partial.parquet")
    book.to_parquet(temporary, index=False)
    temporary.replace(path)


def _saved_forecast(directory: Path, *, identity: dict[str, object],
                    generate: Callable[[], TerminalMarketPaths], cache_directory: Path | None = None,
                    stop_requested: Callable[[], bool] | None = None) -> TerminalMarketPaths:
    return saved_terminal_forecast(directory, identity=identity, generate=generate,
        cache_directory=cache_directory, progress=lambda message: print(message, flush=True),
        stop_requested=stop_requested)


def _decision_record(decision: TerminalDecision, *, contracts_path: Path,
                     daily_directory: Path, dynamics_directory: Path) -> dict[str, object]:
    fit = decision.fit
    inputs = {"state": daily_directory / "days" / f"{decision.state.day}.npz",
        "daily_configuration": daily_directory / "config.json",
        "dynamics": dynamics_directory / "days" / f"{decision.state.day}.npz",
        "contracts": contracts_path, "checkpoint": fit.checkpoint_path}
    return {"files": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in inputs.items()},
        "date": decision.state.day.isoformat(), "root_spot": decision.root_spot,
        "elapsed_day": decision.elapsed_day, "claim": asdict(decision.claim),
        "positions": decision.positions.tolist(), "premium_positions": decision.premium_positions.tolist(),
        "accepted_for_execution": decision.accepted_for_execution,
        "fit": {"status": fit.status, "updates": fit.updates, "initial_loss": fit.initial_loss,
            "best_training_loss": fit.best_training_loss,
            "initialisation_date": None if fit.initialisation_date is None else fit.initialisation_date.isoformat()}}


def _restore_decision(record: dict[str, object], *, env: BaseEnvironment,
                      daily_directory: Path) -> TerminalDecision:
    for description in record["files"].values():
        if hashlib.sha256(Path(description["path"]).read_bytes()).hexdigest() != description["sha256"]:
            raise ValueError("A completed terminal decision input or checkpoint changed")
    day = date.fromisoformat(record["date"])
    checkpoint = Path(record["files"]["checkpoint"]["path"])
    policy = load_terminal_policy(checkpoint, env=env)
    details = record["fit"]
    fit = TerminalPolicyFit(policy=policy, status=details["status"], updates=details["updates"],
        initial_loss=details["initial_loss"], best_training_loss=details["best_training_loss"], history=(),
        checkpoint_path=checkpoint, initialisation_date=(None if details["initialisation_date"] is None
            else date.fromisoformat(details["initialisation_date"])))
    claim = Claim(**record["claim"])
    return TerminalDecision(fit=fit, state=DailyState.load(daily_directory, day),
        dynamics=load_dynamics(Path(record["files"]["dynamics"]["path"])), root_spot=record["root_spot"],
        elapsed_day=record["elapsed_day"], claim_label=claim.label, initial_spot=claim.initial_spot, claim=claim,
        contracts=pd.read_parquet(record["files"]["contracts"]["path"]),
        positions=np.array(record["positions"]), premium_positions=np.array(record["premium_positions"]),
        accepted_for_execution=record["accepted_for_execution"])


def load_origin_decision(directory: Path, *, env: BaseEnvironment, daily_directory: Path) -> TerminalDecision:
    """Load only inception weights for the next overlapping origin, never a later fit."""
    state = json.loads((directory / "episode.json").read_text())
    record = state["origin_decision"]
    if record is None or record["elapsed_day"] != 0:
        raise ValueError("This episode has no saved inception decision to initialize the next origin")
    return _restore_decision(record, env=env, daily_directory=daily_directory)


def replay_terminal_episode(*, origin: date, claim_label: str, store: MarketStore, archive: SourceArchive,
                            daily_directory: Path, dynamics_directory: Path, output_directory: Path,
                            objective: str, risk_aversion: float, target_side: int, env: BaseEnvironment,
                            train_config: TerminalTrainConfig = TerminalTrainConfig(),
                            simulation_config: TerminalSimulationConfig = TerminalSimulationConfig(),
                            model_config: TerminalPolicyConfig | None = None,
                            previous_origin_decision: TerminalDecision | None = None,
                            diagnostic_only: bool = False,
                            stop_requested: Callable[[], bool] | None = None,
                            shared_forecast_directory: Path | None = None) -> TerminalReplayResult:
    """Execute only today's root action, then observe the next actual close.

    A new causal model is loaded at every actual hedge date. Within today's
    simulated remaining horizon that model is frozen. Original strikes,
    barrier/reset and frozen contracts survive the daily model changes.
    Realized P&L uses RAW licensed-market gains, not forecast alpha-adjusted
    gains. Costs are actual entry turnover; terminal/KO exit fees are zero.

    Return the INCEPTION policy for the next overlapping origin's warm start,
    not the future-dated final policy. Every forecast, fitted policy and
    completed daily cashflow is retained outside Git. Unfinished production
    fits stop before execution; diagnostic runs retain explicit candidate
    status rather than silently calling such fits converged.
    """
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Terminal replay outputs must be outside Git working trees")
    if origin < date(2023, 1, 1) or origin > date(2025, 12, 31):
        raise ValueError("The agreed out-of-sample episode origins are in 2023--2025")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        first = store.load(origin)
        source, source_metadata = archive.load(origin)
        possibilities = make_claims(initial_spot=first.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
        matches = tuple(claim for claim in possibilities if claim.label == claim_label)
        if len(matches) != 1:
            raise ValueError("Choose one of the agreed fixed-NMS target definitions")
        original = matches[0]
        horizon = original.terminal_day
        sessions = get_business_days(origin, origin + timedelta(days=15))[:horizon + 1]
        if len(sessions) != horizon + 1 or sessions[-1] > date(2025, 12, 31):
            raise ValueError("The whole episode must remain inside the requested evaluation period")
        vols = np.asarray(source.interpolate_atm_vols(first.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
        tenors = None if horizon == 1 else ((horizon,) if original.kind == "digital" else (3, 5))
        initial_book = select_book(first, atm_vols=vols, co_terminal_business_dtes=tenors,
            option_side="itm", include_co_terminal_forwards=True)
        initial_state = DailyState.load(daily_directory, origin)
        initial_model = load_dynamics(dynamics_directory / "days" / f"{origin}.npz")
        configuration = model_config or TerminalPolicyConfig(label="terminal-policy", feature_count=initial_model.directions.shape[1] + 4)
        if original.kind == "digital":
            forward, _, discount = first.curve.at(float(horizon))
            initial_price = float(discount * surface_digitals(source, expiry_y=horizon / DAYS_PER_YEAR,
                strikes=np.array([original.strike / forward]), is_call=original.is_call)[0])
        else:
            initial_price = original.price(initial_state.decode().pricer, first.curve, elapsed_day=0)
        modules = ("terminal_replay.py", "terminal_decision.py", "terminal_train.py", "terminal_policy.py",
            "terminal_features.py", "terminal_torch.py", "terminal_market.py", "terminal.py", "cash.py",
            "claims.py", "pricing.py", "batch_pricing.py", "source.py", "state.py", "rolling.py", "dynamics.py",
            "terminal_cache.py", "terminal_adequacy.py", "terminal_schedule.py", "dlv.py")
        identity = {"origin": origin.isoformat(), "claim": asdict(original), "initial_price": initial_price,
            "objective": objective, "risk_aversion": risk_aversion, "target_side": target_side,
            "model_config": asdict(configuration), "simulation_config": asdict(simulation_config),
            "simulation_version": TerminalSimulationConfig.version.unique_id48,
            "precision": str(env.fdtype), "device_type": env.device.type, "diagnostic_only": diagnostic_only,
            "daily_directory": str(daily_directory.resolve()), "dynamics_directory": str(dynamics_directory.resolve()),
            "daily_configuration_sha256": hashlib.sha256((daily_directory / "config.json").read_bytes()).hexdigest(),
            "market_sources": {name: str(getattr(store, name).resolve()) for name in (
                "market_directory", "licensed_fit_directory", "raw_quote_directory") if getattr(store, name, None) is not None},
            "initial_book_sha256": _book_digest(initial_book), "source_model": source_metadata,
            "previous_origin_date": None if previous_origin_decision is None else previous_origin_decision.state.day.isoformat(),
            "previous_origin_checkpoint_sha256": None if previous_origin_decision is None else hashlib.sha256(
                previous_origin_decision.fit.checkpoint_path.read_bytes()).hexdigest(),
            "costs": "option entry half-spreads x1; spot zero; no terminal/KO exit fee",
            "same_close_execution_idealization": True,
            "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in modules}}
        identity = json.loads(json.dumps(identity))
        config_path = output_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != identity:
            raise ValueError("Terminal replay scope, inputs or implementation changed")
        _atomic_state(config_path, identity)
        _save_book(output_directory / "initial_book.parquet", initial_book)
        checkpoint = output_directory / "episode.json"
        state = {"next_step": 0, "claim": asdict(original), "holdings": np.zeros(1 + len(initial_book)).tolist(),
            "hedge_gain": 0., "entry_cost": 0., "hypothetical_exit_metric": 0., "discount_to_current": 1.,
            "trades": [], "market_hashes": {}, "previous_decision": None, "origin_decision": None,
            "pending_decision": None,
            "terminal_payoff": None, "status": "running", "fit_statuses": []}
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            for day, expected in state["market_hashes"].items():
                if _market_digest(store.load(date.fromisoformat(day))) != expected:
                    raise ValueError("Previously used observed market inputs changed on replay resume")
            for row in state["trades"]:
                directory = output_directory / f"step-{row['elapsed_day']:02d}"
                paths = {"realized_inputs_sha256": directory / "realized.npz",
                    "policy_checkpoint_sha256": directory / "policy" / row["policy_checkpoint_filename"]}
                for key, path in paths.items():
                    if not path.resolve().is_relative_to(resolved) or hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                        raise ValueError("A completed terminal replay artifact changed")
                forecast = json.loads((directory / "forecast.json").read_text())
                if hashlib.sha256((directory / "forecast.npz").read_bytes()).hexdigest() != forecast["sha256"]:
                    raise ValueError("A completed terminal forecast changed")
        previous = None if state["previous_decision"] is None else _restore_decision(
            state["previous_decision"], env=env, daily_directory=daily_directory)
        inception = None if state["origin_decision"] is None else _restore_decision(
            state["origin_decision"], env=env, daily_directory=daily_directory)
        for elapsed in range(state["next_step"], horizon):
            if state["terminal_payoff"] is not None or stop_requested is not None and stop_requested():
                break
            current = store.load(sessions[elapsed])
            state["market_hashes"][current.day.isoformat()] = _market_digest(current)
            claim = Claim(**state["claim"])
            daily = DailyState.load(daily_directory, current.day)
            dynamics_path = dynamics_directory / "days" / f"{current.day}.npz"
            model = load_dynamics(dynamics_path)
            surface, _ = archive.load(current.day)
            active = np.flatnonzero(initial_book["bdte"].to_numpy(dtype=float) > elapsed + 1e-6)
            columns = np.r_[0, 1 + active]
            book = initial_book.iloc[active].copy().reset_index(drop=True)
            book["bdte"] = book["bdte"].to_numpy(dtype=float) - elapsed
            marks, spreads = observed_marks(book, current)
            book["observed_mid"], book["observed_half_spread"] = marks, spreads
            directory = output_directory / f"step-{elapsed:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            _save_book(directory / "book.parquet", book)
            if claim.kind == "digital":
                forward, _, discount = current.curve.at(float(horizon - elapsed))
                mark = float(discount * surface_digitals(surface, expiry_y=(horizon - elapsed) / DAYS_PER_YEAR,
                    strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0])
            else:
                mark = claim.price(daily.decode().pricer, current.curve, elapsed_day=elapsed)
            forecast_identity = {"date": current.day.isoformat(), "elapsed_day": elapsed,
                "state_sha256": hashlib.sha256(
                (daily_directory / "days" / f"{current.day}.npz").read_bytes()).hexdigest(),
                "simulation_version": TerminalSimulationConfig.version.unique_id48,
                "implementation_sha256": {name: identity["implementation_sha256"][name] for name in (
                    "terminal_market.py", "terminal.py", "pricing.py", "batch_pricing.py", "state.py",
                    "dlv.py", "cash.py", "claims.py", "rolling.py", "dynamics.py")},
                "daily_configuration_sha256": identity["daily_configuration_sha256"],
                "dynamics_sha256": hashlib.sha256(dynamics_path.read_bytes()).hexdigest(),
                "market_sha256": _market_digest(current), "book_sha256": _book_digest(book), "claim": asdict(claim),
                "mark": mark, "discount_to_current": state["discount_to_current"],
                "configuration": asdict(simulation_config), "seed": origin.toordinal() * 10 + elapsed}
            try:
                simulations = _saved_forecast(directory, identity=forecast_identity,
                    cache_directory=shared_forecast_directory, stop_requested=stop_requested,
                    generate=lambda: generate_terminal_paths(state=daily, dynamics=model, curve=current.curve,
                        book=book, claims=(claim,), initial_spot=first.curve.spot,
                        current_claim_marks=np.array([mark]), elapsed_day=elapsed,
                        discount_to_current=state["discount_to_current"], seed=forecast_identity["seed"],
                        config=simulation_config, progress=lambda message: print(message, flush=True)))
            except ForecastCheckpointStop:
                break
            if stop_requested is not None and stop_requested():
                break
            inventory = np.array(state["holdings"])[columns]
            print(f"terminal replay origin={origin} date={current.day} step={elapsed+1}/{horizon} "
                f"claim={claim_label} objective={objective} lambda={risk_aversion} side={target_side}", flush=True)
            pending = state.get("pending_decision")
            if pending is None:
                decision = fit_terminal_decision(simulations=simulations, contracts=book, claim=claim, claim_index=0,
                    state=daily, dynamics=model, initial_inventory=inventory, model_config=configuration,
                    train_config=train_config, env=env, objective=objective, risk_aversion=risk_aversion,
                    target_side=target_side, output_directory=directory / "policy", previous_decision=previous,
                    previous_origin_decision=previous_origin_decision if elapsed == 0 else None, diagnostic_only=diagnostic_only)
            else:
                decision = _restore_decision(pending, env=env, daily_directory=daily_directory)
                if decision.elapsed_day != elapsed or decision.state.day != current.day or decision.claim != claim:
                    raise ValueError("A pending terminal decision does not belong to the current episode step")
            if (not diagnostic_only and not decision.accepted_for_execution) or decision.fit.status == "checkpointed_stop":
                state["status"] = ("numerical_review_required" if decision.fit.status == "training_stable"
                    else "training_incomplete" if decision.fit.status != "checkpointed_stop" else "checkpointed_stop")
                _atomic_state(checkpoint, state)
                return TerminalReplayResult(status=state["status"], summary={"completed_steps": elapsed,
                    "pending_fit_status": decision.fit.status, "diagnostic_only": diagnostic_only},
                    trades=pd.DataFrame(state["trades"]), origin_decision=inception)
            record = _decision_record(decision, contracts_path=directory / "book.parquet",
                daily_directory=daily_directory, dynamics_directory=dynamics_directory)
            state["pending_decision"] = record
            if elapsed == 0:
                state["origin_decision"] = record
                inception = decision
            _atomic_state(checkpoint, state)
            following = store.load(sessions[elapsed + 1])
            next_marks, next_spreads = observed_marks(book, following)
            realized = discounted_daily_gains(previous_option_marks=marks, next_option_marks=next_marks,
                next_spot=following.curve.spot, previous_curve=current.curve,
                episode_initial_spot=first.curve.spot, discount_to_previous=state["discount_to_current"])
            costs = state["discount_to_current"] * np.r_[0., spreads] / first.curve.spot
            positions = decision.positions
            entry_cost = float(costs @ np.abs(positions - inventory))
            gain = float(positions @ realized)
            next_discount = state["discount_to_current"] * float(current.curve.at(1.)[2])
            exit_metric = float(next_discount * np.abs(positions[1:]) @ next_spreads / first.curve.spot)
            after = claim.observe(elapsed_day=elapsed + 1, spot=following.curve.spot)
            ended = elapsed + 1 == horizon or after.kind == "ko" and after.knocked_out
            holdings = np.zeros(1 + len(initial_book))
            holdings[columns] = positions
            holdings[1 + np.flatnonzero(initial_book["bdte"].to_numpy(dtype=float) <= elapsed + 1 + 1e-6)] = 0.
            if ended:
                holdings[:] = 0.
            np.savez_compressed(directory / "realized.npz", columns=columns, previous_positions=inventory,
                positions=positions, realized_raw_gains=realized, costs=costs, next_half_spreads=next_spreads,
                current_option_marks=marks, next_option_marks=next_marks, holdings_after=holdings)
            state.update(next_step=elapsed + 1, claim=asdict(after), holdings=holdings.tolist(),
                hedge_gain=state["hedge_gain"] + gain, entry_cost=state["entry_cost"] + entry_cost,
                hypothetical_exit_metric=state["hypothetical_exit_metric"] + exit_metric,
                discount_to_current=next_discount, previous_decision=record,
                pending_decision=None,
                origin_decision=record if elapsed == 0 else state["origin_decision"], status="running")
            state["market_hashes"][following.day.isoformat()] = _market_digest(following)
            state["fit_statuses"].append(decision.fit.status)
            state["trades"].append({"origin": origin.isoformat(), "date": current.day.isoformat(),
                "next_date": following.day.isoformat(), "elapsed_day": elapsed, "claim": claim.label,
                "objective": objective, "risk_aversion": risk_aversion, "side": target_side,
                "hedge_gain": gain, "entry_cost": entry_cost, "hypothetical_exit_cost": exit_metric,
                "fit_status": decision.fit.status, "updates": decision.fit.updates,
                "maximum_premium_position": float(np.abs(decision.premium_positions).max()),
                "gross_premium_position": float(np.abs(decision.premium_positions).sum()),
                "forecast_zero_live_model_prices": int(simulations.zero_live_model_prices.sum()),
                "cap_hits": int(np.sum(np.abs(decision.premium_positions) >= configuration.instrument_cap * (1. - 1e-6))),
                "knockout": bool(after.knocked_out), "accepted_for_execution": decision.accepted_for_execution,
                "policy_checkpoint_filename": decision.fit.checkpoint_path.name,
                "policy_checkpoint_sha256": hashlib.sha256(decision.fit.checkpoint_path.read_bytes()).hexdigest(),
                "realized_inputs_sha256": hashlib.sha256((directory / "realized.npz").read_bytes()).hexdigest()})
            if ended:
                state["terminal_payoff"] = after.payoff(following.curve.spot)
            _atomic_state(checkpoint, state)
            previous, inception = decision, decision if elapsed == 0 else inception
        completed = state["terminal_payoff"] is not None
        status = "complete" if completed else "checkpointed_stop"
        state["status"] = status
        _atomic_state(checkpoint, state)
        summary = {"status": status, "origin": origin.isoformat(), "scheduled_expiry": sessions[-1].isoformat(),
            "hedge_end_date": sessions[state["next_step"]].isoformat(), "claim": claim_label,
            "objective": objective, "risk_aversion": risk_aversion, "side": target_side,
            "initial_price": initial_price, "terminal_payoff": state["terminal_payoff"],
            "discount_to_terminal": state["discount_to_current"], "hedge_gain": state["hedge_gain"],
            "entry_cost": state["entry_cost"], "hypothetical_exit_metric": state["hypothetical_exit_metric"],
            "hypothetical_exit_metric_definition": "sum of daily hypothetical liquidation fees; never charged",
            "unhedged_pnl": None, "costed_pnl": None, "completed_steps": state["next_step"],
            "diagnostic_only": diagnostic_only, "all_fits_stable": bool(state["fit_statuses"]) and all(
                value == "training_stable" for value in state["fit_statuses"]),
            "same_close_execution_idealization": True, "no_terminal_exit_cost": True}
        summary["cap_hit_days_pct"] = (100. * sum(row["cap_hits"] > 0 for row in state["trades"]) / len(state["trades"])
            if state["trades"] else 0.)
        if completed:
            unhedged = target_side * (state["discount_to_current"] * state["terminal_payoff"] - initial_price)
            summary.update(unhedged_pnl=unhedged, costed_pnl=unhedged + state["hedge_gain"] - state["entry_cost"])
        _atomic_state(output_directory / "summary.json", summary)
        pd.DataFrame(state["trades"]).to_parquet(output_directory / "trades.parquet", index=False)
        return TerminalReplayResult(status=status, summary=summary, trades=pd.DataFrame(state["trades"]), origin_decision=inception)
