"""Fixed-contract hedge episodes with daily causal re-estimation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.black_exotic_hedge import black_exotic_hedge
from quantdata.hedge2.black_hedge import black_digital_hedge
from quantdata.hedge2.cash import (
    MarketStore,
    discounted_daily_gains,
    observed_marks,
    select_book,
    zero_strike_mask,
)
from quantdata.hedge2.certification import certify_candidate
from quantdata.hedge2.claims import Claim, make_claims
from quantdata.hedge2.greek_hedge import GreekBoxInfeasible, solve_greek_hedge
from quantdata.hedge2.greeks import (
    calculate_greeks,
    calculate_parametric_greeks,
    factor_risk_covariance,
    parametric_factor_covariance,
)
from quantdata.hedge2.optimization import ScenarioOptimizer
from quantdata.hedge2.parametric import ParametricOptimizer
from quantdata.hedge2.premium import (
    GreekBudgetInfeasible,
    PremiumParametricOptimizer,
    PremiumUnits,
    solve_premium_greek,
)
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.scenarios import generate_scenarios
from quantdata.hedge2.source import SourceArchive, surface_implied_vols
from quantdata.hedge2.state import DailyState


@dataclass(frozen=True, kw_only=True)
class HedgeRule:
    method: str
    side: int
    objective: str = "none"
    risk_aversion: float = 0.
    greek_budget: float = 1.

    def __post_init__(self) -> None:
        if (self.method not in ("unhedged", "BS delta+vega", "DLV Greek", "DLV parametric", "DLV statistical")
                or self.side not in (-1, 1) or not 0 <= self.greek_budget <= 1):
            raise ValueError("Unknown episode hedge rule, side or Greek budget")
        if self.method in ("DLV parametric", "DLV statistical") and (
                self.objective not in ("mean_variance", "power_exponential")
                or self.risk_aversion <= 0 or np.isnan(self.risk_aversion)
                or self.objective == "power_exponential" and not np.isfinite(self.risk_aversion)):
            raise ValueError("Episode optimization needs an agreed objective and risk aversion")


@dataclass(frozen=True, kw_only=True)
class EpisodeResult:
    summary: pd.DataFrame
    trades: pd.DataFrame
    initial_book: pd.DataFrame
    positions: tuple[np.ndarray, ...]
    active_indices: tuple[np.ndarray, ...]
    diagnostics: tuple[dict[str, object], ...]
    input_hashes: dict[str, str] | None = None


class EpisodeCheckpointedStop(Exception):
    """A requested graceful stop, with a complete saved daily-step checkpoint."""


def terminal_pnl(*, side: int, initial_price: float, terminal_payoff: float,
                 discount_to_terminal: float, hedge_gains: float, entry_costs: float) -> float:
    if (side not in (-1, 1) or discount_to_terminal <= 0 or entry_costs < 0
            or not all(np.isfinite(value) for value in (initial_price, terminal_payoff,
                       discount_to_terminal, hedge_gains, entry_costs))):
        raise ValueError("Episode accounting requires finite values, a side and nonnegative entry costs")
    return side * (discount_to_terminal * terminal_payoff - initial_price) + hedge_gains - entry_costs


def run_episode(*, origin: date, family: str, store: MarketStore, archive: SourceArchive,
                daily_directory: Path, dynamics_directory: Path, rules: tuple[HedgeRule, ...],
                claim_labels: tuple[str, ...] | None = None,
                checkpoint_directory: Path | None = None,
                stop_requested: Callable[[], bool] | None = None,
                objective_gap_tolerance: float = 1e-4,
                option_side: str = "otm", include_co_terminal_forwards: bool = False,
                reference_premium_cap: float | None = None,
                optimized_premium_cap: float | None = None) -> EpisodeResult:
    """Daily receding one-step hedges, not a globally optimized multi-day policy.

    All cash contracts and their selected side are frozen at inception. Each
    day's rolling model may use observations only through that day. Wealth,
    costs and risk aversion stay in inception-notional discounted units.
    Expiring options settle without an exit trade; live adjustments pay
    turnover half-spreads. Source/model gaps raise explicit errors.
    Current ITM/forward references are analytic and uncapped; an optional
    premium-dollar threshold reports comparison breaches without clipping.
    Greek/parametric limits use current undiscounted premium dollars when
    optimized_premium_cap is supplied; physical inventory and cash retain
    their original episode units across daily re-estimation.
    Greek-only matching excludes synthetic zero-strike calls. Run it in a
    separate book from scenario methods that retain co-terminal forwards.
    """
    horizons = {"digital-1D": 1, "digital-3D": 3, "digital-5D": 5, "ko": 5, "ratio_straddle": 5}
    if family not in horizons or not rules:
        raise ValueError("Choose a supported target family and at least one hedge rule")
    if (option_side not in ("otm", "itm") or not isinstance(include_co_terminal_forwards, bool)
            or reference_premium_cap is not None and (not np.isfinite(reference_premium_cap) or reference_premium_cap <= 0)
            or optimized_premium_cap is not None and (not np.isfinite(optimized_premium_cap) or optimized_premium_cap <= 0)):
        raise ValueError("Episode book side, forwards and reference premium threshold must be valid")
    if include_co_terminal_forwards and any(rule.method == "DLV Greek" for rule in rules):
        raise ValueError("Greek-only hedges exclude synthetic co-terminal forwards; use a separate book")
    if (option_side != "otm" or include_co_terminal_forwards or reference_premium_cap is not None) and any(
            rule.method not in ("unhedged", "BS delta+vega") for rule in rules) and optimized_premium_cap is None:
        raise ValueError("Current ITM/forward optimizers require an explicit current-premium limit")
    if optimized_premium_cap is not None and any(rule.method == "DLV statistical" for rule in rules):
        raise ValueError("Current multi-day statistical hedges require the terminal-horizon controller")
    horizon = horizons[family]
    sessions = get_business_days(origin, origin + timedelta(days=15))[:horizon + 1]
    if sessions[0] != origin or len(sessions) != horizon + 1 or sessions[-1] > date(2025, 12, 31):
        raise ValueError("An episode must have its full horizon inside the agreed period")
    markets = tuple(store.load(day) for day in sessions)
    first = markets[0]
    source, _ = archive.load(origin)
    claims = tuple(claim for claim in make_claims(initial_spot=first.curve.spot,
        atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
        if (claim.kind == family or family == f"digital-{claim.terminal_day}D" and claim.kind == "digital")
        and (claim_labels is None or claim.label in claim_labels))
    if not claims or claim_labels is not None and set(claim_labels) != {claim.label for claim in claims}:
        raise ValueError("No matching claims, or a requested label is outside the family")
    for day in sessions[:-1]:
        if not (dynamics_directory / "days" / f"{day.isoformat()}.npz").exists():
            raise FileNotFoundError(f"Missing causal rolling dynamics on episode date {day}")
    tenors = (None if horizon == 1 else (horizon,)) if family.startswith("digital") else (3, 5)
    atm = np.asarray(source.interpolate_atm_vols(first.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
    initial_book = select_book(first, atm_vols=atm, co_terminal_business_dtes=tenors,
        option_side=option_side, include_co_terminal_forwards=include_co_terminal_forwards)
    for market in markets:
        active = initial_book.loc[pd.to_datetime(initial_book["expiry"]).dt.date >= market.day]
        observed_marks(active, market)
    initial_state = DailyState.load(daily_directory, origin)
    initial_pricer = initial_state.decode().pricer
    # Every hedged claim, digitals included, is valued with the zero-smoothing DLV in LV mode.
    initial_prices = [float(claim.price(initial_pricer, first.curve, elapsed_day=0)) for claim in claims]
    input_hashes = {}
    if checkpoint_directory is not None:
        observation_digest = hashlib.sha256()
        for market in markets:
            active_book = initial_book.loc[pd.to_datetime(initial_book["expiry"]).dt.date >= market.day]
            for values in observed_marks(active_book, market):
                observation_digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
            for values in asdict(market.curve).values():
                observation_digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
        paths = [directory / "days" / f"{day.isoformat()}.npz"
                 for day in sessions[:-1] for directory in (daily_directory, dynamics_directory)]
        paths.append(daily_directory / "config.json")
        if hasattr(archive, "model_path"):
            paths.extend(archive.model_path(day) for day in sessions[:-1])
        input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.exists()}
        input_hashes["observed_marks_spreads_and_carry"] = observation_digest.hexdigest()
        input_hashes["frozen_book"] = hashlib.sha256(pd.util.hash_pandas_object(initial_book, index=True).to_numpy().tobytes()).hexdigest()
    initial_identity = {"origin": origin.isoformat(), "family": family, "rules": [asdict(rule) for rule in rules],
                        "claim_labels": [claim.label for claim in claims], "initial_prices": initial_prices,
                        "initial_spot": float(first.curve.spot), "input_hashes": input_hashes,
                        "option_side": option_side, "include_co_terminal_forwards": include_co_terminal_forwards,
                        "reference_premium_cap": reference_premium_cap}
    if optimized_premium_cap is not None:
        initial_identity["optimized_premium_cap"] = optimized_premium_cap
    keys = [(target, rule) for target in range(len(claims)) for rule in rules]
    holdings = np.zeros((len(keys), 1 + len(initial_book)))
    cumulative_gains = np.zeros(len(keys))
    cumulative_costs = np.zeros(len(keys))
    cumulative_exit_metric = np.zeros(len(keys))
    cap_days = np.zeros(len(keys), dtype=int)
    premium_breach_days = np.zeros(len(keys), dtype=int)
    maximum_premium_positions = np.zeros(len(keys))
    maximum_gross_premium_positions = np.zeros(len(keys))
    terminated = np.zeros(len(keys), dtype=bool)
    hedge_end_dates = [sessions[-1].isoformat()] * len(keys)
    termination_reasons = ["scheduled_expiry"] * len(keys)
    terminal_exit_metrics = np.zeros(len(keys))
    rows, saved_positions, saved_indices, diagnostics = [], [], [], []
    discount_to_current = 1.
    first_step = 0
    checkpoint = None if checkpoint_directory is None else checkpoint_directory / "episode_checkpoint.npz"
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"]))
            if identity != initial_identity:
                raise ValueError("Episode checkpoint inputs, frozen book, inception marks, rules or products changed")
            first_step = int(saved["next_step"])
            if not 0 <= first_step <= horizon:
                raise ValueError("Episode checkpoint step is outside the contractual horizon")
            holdings = saved["holdings"]
            if holdings.shape != (len(keys), 1 + len(initial_book)) or not np.isfinite(holdings).all():
                raise ValueError("Saved episode holdings do not match the frozen contract book")
            cumulative_gains, cumulative_costs = saved["cumulative_gains"], saved["cumulative_costs"]
            cumulative_exit_metric, cap_days = saved["cumulative_exit_metric"], saved["cap_days"]
            premium_breach_days = saved["premium_breach_days"]
            maximum_premium_positions = saved["maximum_premium_positions"]
            maximum_gross_premium_positions = saved["maximum_gross_premium_positions"]
            terminated, terminal_exit_metrics = saved["terminated"], saved["terminal_exit_metrics"]
            hedge_end_dates = json.loads(str(saved["hedge_end_dates_json"]))
            termination_reasons = json.loads(str(saved["termination_reasons_json"]))
            claims = tuple(Claim(**item) for item in json.loads(str(saved["claims_json"])))
            discount_to_current = float(saved["discount_to_current"])
            rows, diagnostics = json.loads(str(saved["rows_json"])), json.loads(str(saved["diagnostics_json"]))
            saved_positions = [saved[f"positions_{step}"] for step in range(first_step)]
            saved_indices = [saved[f"indices_{step}"] for step in range(first_step)]
        print(f"episode restored origin={origin} family={family} completed_steps={first_step}/{horizon}", flush=True)
    for elapsed, current in enumerate(markets[:-1]):
        if elapsed < first_step:
            continue
        print(f"episode origin={origin} family={family} step={elapsed+1}/{horizon} "
              f"portfolios={len(keys)} date={current.day}", flush=True)
        following = markets[elapsed + 1]
        source, _ = archive.load(current.day)
        state = DailyState.load(daily_directory, current.day)
        model = load_dynamics(dynamics_directory / "days" / f"{current.day.isoformat()}.npz")
        if model.training_start < date(2022, 1, 1) or model.training_end > current.day:
            raise ValueError("Episode dynamics violate the causal history window")
        active = np.flatnonzero(initial_book["bdte"].to_numpy(dtype=float) > elapsed + 1e-6)
        columns = np.r_[0, 1 + active]
        book = initial_book.iloc[active].copy().reset_index(drop=True)
        book["bdte"] = book["bdte"].to_numpy(dtype=float) - elapsed
        marks, spreads = observed_marks(book, current)
        next_marks, next_spreads = observed_marks(book, following)
        costs = discount_to_current * np.r_[0., spreads / first.curve.spot]
        normalized_premium_prices = np.r_[current.curve.spot, marks] / first.curve.spot
        needed = {rule.method for portfolio, (_, rule) in enumerate(keys) if not terminated[portfolio]}
        premium_units = (PremiumUnits(normalized_prices=normalized_premium_prices,
            normalized_costs=costs, position_cap=optimized_premium_cap)
            if optimized_premium_cap is not None and needed & {"DLV Greek", "DLV parametric"} else None)
        realized = discounted_daily_gains(previous_option_marks=marks, next_option_marks=next_marks,
            next_spot=following.curve.spot, previous_curve=current.curve,
            episode_initial_spot=first.curve.spot, discount_to_previous=discount_to_current)
        _, _, one_day_discount = current.curve.at(1.)
        arguments = dict(state=state, dynamics=model, curve=current.curve, book=book,
                         claims=claims, initial_spot=first.curve.spot, elapsed_day=elapsed)
        greeks = calculate_greeks(**arguments) if "DLV Greek" in needed else None
        parametric_greeks = calculate_parametric_greeks(**arguments) if "DLV parametric" in needed else None
        if parametric_greeks is not None:
            if premium_units is None:
                parametric = ParametricOptimizer(exposures=discount_to_current * parametric_greeks.instrument_exposures,
                    covariance=parametric_factor_covariance(model), costs=costs)
            else:
                parametric = PremiumParametricOptimizer(exposures=discount_to_current * parametric_greeks.instrument_exposures,
                    covariance=parametric_factor_covariance(model), units=premium_units)
        if "DLV statistical" in needed:
            scenarios = generate_scenarios(**arguments, seed=origin.toordinal() * 10 + elapsed)
            statistical = ScenarioOptimizer(instrument_gains=discount_to_current * scenarios.instrument_gains,
                                             weights=scenarios.weights, costs=costs)
        step_directory = None
        if checkpoint_directory is not None:
            step_directory = checkpoint_directory / f"step-{elapsed:02d}"
            step_directory.mkdir(parents=True, exist_ok=True)
            book.to_parquet(step_directory / "book.parquet", index=False)
            (step_directory / "claims.json").write_text(json.dumps([asdict(claim) for claim in claims], indent=2) + "\n")
            risk_inputs = {"costs": costs, "realized_gains": realized, "active_columns": columns,
                           "previous_positions": holdings[:, columns], "discount_to_current": discount_to_current,
                           "claim_labels": np.array([claim.label for claim in claims])}
            if premium_units is not None:
                risk_inputs.update(normalized_premium_prices=normalized_premium_prices,
                    optimized_premium_cap=optimized_premium_cap)
            if greeks is not None:
                risk_inputs.update(greek_instrument_exposures=discount_to_current * greeks.instrument_exposures,
                    greek_claim_exposures=discount_to_current * greeks.claim_exposures)
            if "DLV statistical" in needed:
                risk_inputs.update(instrument_gains=statistical.gains, weights=statistical.weights,
                    claim_gains=discount_to_current * scenarios.claim_gains,
                    removed_instrument_mean=scenarios.removed_instrument_mean)
            if parametric_greeks is not None:
                risk_inputs.update(instrument_exposures=discount_to_current * parametric_greeks.instrument_exposures,
                    claim_exposures=discount_to_current * parametric_greeks.claim_exposures,
                    covariance=parametric_factor_covariance(model))
            np.savez_compressed(step_directory / "risk_inputs.npz", **risk_inputs)
        if "BS delta+vega" in needed:
            option_vols = np.asarray(source.interpolate_atm_vols(book["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR)).copy()
            ordinary = ~zero_strike_mask(book)
            if ordinary.any():
                ordinary_days = book.loc[ordinary, "bdte"].to_numpy(dtype=float)
                ordinary_vols = surface_implied_vols(source, expiries_y=ordinary_days / DAYS_PER_YEAR,
                    strikes=book.loc[ordinary, "strike"].to_numpy(dtype=float) / current.curve.at(ordinary_days)[0],
                )
                option_vols[ordinary] = ordinary_vols
            remaining = horizon - elapsed
            raw_variances = np.array([float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR))**2
                                      * tenor / DAYS_PER_YEAR for tenor in range(1, remaining + 1)])
            coherent_variances = np.maximum.accumulate(raw_variances)
            total_variances = {tenor: float(value) for tenor, value in enumerate(coherent_variances, start=1)}
        risk_scales = (np.sqrt(np.maximum(np.diag(factor_risk_covariance(model,
            spot_orthogonal=bool(np.any(model.spot_state_loading)))), 1e-20))
            if "DLV Greek" in needed else None)
        step_positions = []
        observed_claims = tuple(claim.observe(elapsed_day=elapsed + 1, spot=following.curve.spot) for claim in claims)
        for portfolio, (target, rule) in enumerate(keys):
            claim = claims[target]
            previous = holdings[portfolio, columns]
            solver, status, warnings = "none", "not applicable", ()
            gap = 0.
            training_objective, training_mean, training_std, cash_residual = None, None, None, None
            certificate_solver, certificate_lower_bound, certificate_upper_bound = None, None, None
            certificate = {}
            mandatory_turnover_cost, optimizer_cost = 0., None
            greek_full_entry_cost, greek_residual_norm = None, None
            contract_position_override = None
            was_active = not terminated[portfolio]
            if not was_active:
                position = np.zeros(len(costs))
                status = "knocked out; discounted cash only"
            elif rule.method == "unhedged":
                position = np.zeros(len(costs))
            elif rule.method == "BS delta+vega":
                if claim.kind == "digital":
                    forward = current.curve.at(float(horizon - elapsed))[0]
                    target_vol = surface_implied_vols(source, expiries_y=np.array([(horizon-elapsed) / DAYS_PER_YEAR]),
                        strikes=np.array([claim.strike / forward]))
                    position = black_digital_hedge(claim=claim, curve=current.curve, book=book,
                        target_vol=float(target_vol[0]), option_vols=np.asarray(option_vols),
                        target_side=rule.side, elapsed_day=elapsed)
                else:
                    position = black_exotic_hedge(claim=claim, curve=current.curve, book=book,
                        option_vols=np.asarray(option_vols), total_variances=total_variances,
                        target_side=rule.side, elapsed_day=elapsed).positions
                solver, status = "Black finite/analytic Greeks", "analytic"
            elif rule.method == "DLV Greek":
                greek_arguments = dict(exposures=discount_to_current * greeks.instrument_exposures,
                    target_exposures=discount_to_current * greeks.claim_exposures[target],
                    previous_positions=previous, risk_scales=risk_scales,
                    target_side=rule.side, budget_fraction=rule.greek_budget)
                if premium_units is None:
                    hedge = solve_greek_hedge(**greek_arguments, costs=costs)
                    position = hedge.positions
                else:
                    try:
                        adapted = solve_premium_greek(**greek_arguments, units=premium_units)
                    except (GreekBudgetInfeasible, GreekBoxInfeasible) as error:
                        context = {"origin": origin.isoformat(), "date": current.day.isoformat(),
                            "family": family, "claim": claim.label, **asdict(rule),
                            "elapsed_day": elapsed, "portfolio_index": portfolio,
                            "claim_labels": [item.label for item in claims],
                            "input_hashes": input_hashes,
                            "executed_at_failure": False, "terminal_performance_available": False,
                            "error_type": type(error).__name__, "exact_error": str(error)}
                        evidence = {"previous_contract_positions": previous,
                            "normalized_premium_prices": normalized_premium_prices, "costs": costs,
                            "active_columns": columns, "target_exposures": greek_arguments["target_exposures"],
                            "instrument_exposures": greek_arguments["exposures"]}
                        if isinstance(error, GreekBudgetInfeasible):
                            context.update(full_entry_cost=error.full_entry_cost,
                                mandatory_turnover_cost=error.mandatory_turnover_cost,
                                discretionary_full_cost=error.discretionary_full_cost,
                                turnover_budget=error.turnover_budget, shortfall=error.shortfall)
                        else:
                            values = asdict(error.certificate)
                            for name in ("primal_positions", "normalized_separator"):
                                evidence[name] = values.pop(name)
                            context["box_certificate"] = values
                            context["original_solver_status"] = error.solver_status
                            context["original_solver_message"] = error.solver_message
                        if step_directory is not None:
                            prefix = step_directory / f"unavailable-{portfolio:04d}"
                            np.savez_compressed(prefix.with_suffix(".npz"), **evidence)
                            prefix.with_suffix(".json").write_text(json.dumps(context, indent=2) + "\n")
                            context["failure_evidence"] = str(prefix)
                        error.portfolio_context = context
                        raise
                    hedge, position = adapted.result, adapted.contract_positions
                    mandatory_turnover_cost = adapted.mandatory_turnover_cost
                solver, status, warnings = hedge.solver, hedge.status, hedge.warnings
                optimizer_cost = hedge.entry_cost
                greek_full_entry_cost, greek_residual_norm = hedge.full_entry_cost, hedge.risk_residual_norm
            else:
                if rule.method == "DLV parametric":
                    parametric_result = parametric.solve(target_exposures=discount_to_current * parametric_greeks.claim_exposures[target],
                        objective=rule.objective, risk_aversion=rule.risk_aversion,
                        target_side=rule.side, previous_positions=previous)
                    if premium_units is None:
                        hedge = parametric_result.hedge
                    else:
                        hedge = parametric_result.result.hedge
                        contract_position_override = parametric_result.contract_positions
                        mandatory_turnover_cost = parametric_result.mandatory_turnover_cost
                else:
                    target_scenarios = discount_to_current * scenarios.claim_gains[:, target]
                    hedge = statistical.solve(claim_gains=target_scenarios, objective=rule.objective,
                        risk_aversion=rule.risk_aversion, target_side=rule.side, previous_positions=previous)
                    hedge, certificate = certify_candidate(optimizer=statistical, candidate=hedge,
                        claim_gains=target_scenarios, objective=rule.objective, risk_aversion=rule.risk_aversion,
                        target_side=rule.side, previous_positions=previous, gap_tolerance=objective_gap_tolerance,
                        certificate_directory=None if step_directory is None else step_directory / f"certificate-{portfolio:04d}")
                position = hedge.positions if contract_position_override is None else contract_position_override
                solver, status, warnings = hedge.solver, hedge.status, hedge.warnings
                optimizer_cost = hedge.entry_cost
                training_objective, training_mean, training_std = hedge.training_objective, hedge.training_mean, hedge.training_std
                cash_residual = hedge.cash_first_order_residual
                gap = certificate.get("certified_objective_gap", hedge.convex_first_order_gap_bound)
                warnings = (*warnings, *certificate.get("dual_warnings", ()))
                certificate_solver = certificate.get("dual_solver")
                certificate_lower_bound = certificate.get("dual_lower_bound")
                certificate_upper_bound = certificate.get("primal_upper_bound")
                if step_directory is not None and certificate.get("original_gap", 0.) > objective_gap_tolerance:
                    (step_directory / f"certificate-{portfolio:04d}" / "refinement.json").write_text(json.dumps(certificate, indent=2) + "\n")
                if gap > objective_gap_tolerance:
                    if step_directory is not None:
                        np.savez_compressed(step_directory / "review_candidate.npz", positions=position,
                            previous_positions=previous, target_index=target)
                        (step_directory / "review.json").write_text(json.dumps({"claim": claim.label, **asdict(rule),
                            "solver": solver, "status": status, "warnings": list(warnings), "objective_gap": gap,
                            "training_objective": training_objective, "certificate_lower_bound": certificate_lower_bound,
                            "certificate_upper_bound": certificate_upper_bound}, indent=2) + "\n")
                    raise RuntimeError(f"Multi-day objective certificate needs review: {origin} {current.day} "
                                       f"{claim.label} {rule}, gap={gap}")
            if not np.isfinite(position).all():
                raise ValueError("Non-finite episode portfolio")
            cost = float(costs @ np.abs(position - previous))
            if optimizer_cost is not None and not np.isclose(cost, optimizer_cost, rtol=1e-8, atol=1e-12):
                raise RuntimeError("Episode contract turnover disagrees with optimizer premium-unit cost")
            gain = float(position @ realized)
            exit_metric = float(discount_to_current * one_day_discount
                * np.abs(position[1:]) @ next_spreads / first.curve.spot)
            cumulative_gains[portfolio] += gain
            cumulative_costs[portfolio] += cost
            cumulative_exit_metric[portfolio] += exit_metric
            premium_positions = position * normalized_premium_prices
            if premium_units is not None and rule.method.startswith("DLV "):
                if np.any(np.abs(premium_positions) > optimized_premium_cap * (1. + 1e-8)):
                    raise RuntimeError("Episode post-trade current-premium limit failed its independent check")
                caps = int(np.sum(np.abs(premium_positions) >= optimized_premium_cap * (1. - 1e-6)))
            else:
                caps = (int(np.sum(np.abs(position) >= 1000. * (1. - 1e-6)))
                        if rule.method.startswith("DLV ") else 0)
            cap_days[portfolio] += int(caps > 0)
            maximum_premium = float(np.max(np.abs(premium_positions), initial=0.))
            gross_premium = float(np.abs(premium_positions).sum())
            breaches = (int(np.sum(np.abs(premium_positions) > reference_premium_cap * (1. + 1e-12)))
                        if reference_premium_cap is not None else 0)
            premium_breach_days[portfolio] += int(breaches > 0)
            maximum_premium_positions[portfolio] = max(maximum_premium_positions[portfolio], maximum_premium)
            maximum_gross_premium_positions[portfolio] = max(maximum_gross_premium_positions[portfolio], gross_premium)
            holdings[portfolio, columns] = position
            knocked_out_now = was_active and observed_claims[target].kind == "ko" and observed_claims[target].knocked_out
            if knocked_out_now:
                terminated[portfolio] = True
                hedge_end_dates[portfolio] = following.day.isoformat()
                termination_reasons[portfolio] = "knockout"
                terminal_exit_metrics[portfolio] = exit_metric
                holdings[portfolio] = 0.
            elif was_active and elapsed + 1 == horizon:
                terminal_exit_metrics[portfolio] = exit_metric
            step_positions.append(position)
            rows.append({"origin": origin.isoformat(), "date": current.day.isoformat(), "elapsed_day": elapsed,
                "claim": claim.label, **asdict(rule), "entry_cost": cost, "hedge_gain": gain,
                "hypothetical_exit_cost": exit_metric, "cap_hits": caps, "objective_gap": gap,
                "premium_comparison_breaches": breaches, "reference_premium_cap": reference_premium_cap,
                "optimized_premium_cap": optimized_premium_cap, "mandatory_turnover_cost": mandatory_turnover_cost,
                "greek_full_entry_cost": greek_full_entry_cost, "greek_residual_norm": greek_residual_norm,
                "maximum_absolute_premium_position": maximum_premium, "gross_premium_positions": gross_premium,
                "training_objective": training_objective, "training_mean_net": training_mean,
                "training_std": training_std, "cash_first_order_residual": cash_residual,
                "certificate_solver": certificate_solver, "certificate_lower_bound": certificate_lower_bound,
                "certificate_upper_bound": certificate_upper_bound,
                "original_objective_gap": certificate.get("original_gap", gap), "refined": certificate.get("polished", False),
                "solver": solver, "status": status, "warnings": list(warnings),
                "claim_active_before_step": was_active, "knockout_exit_at_mid": knocked_out_now,
                "discount_to_current": discount_to_current, "gross_positions": float(np.abs(position).sum())})
        saved_positions.append(np.asarray(step_positions))
        saved_indices.append(columns)
        if step_directory is not None:
            pd.DataFrame(rows[-len(keys):]).to_parquet(step_directory / "trades.parquet", index=False)
            np.savez_compressed(step_directory / "completed_step.npz", positions=np.asarray(step_positions),
                holdings_after_hedge=holdings, cumulative_gains=cumulative_gains,
                cumulative_costs=cumulative_costs, terminated=terminated)
        diagnostics.append({"date": current.day.isoformat(), "active_contracts": len(book),
            "frozen_initial_contracts": len(initial_book), "training_start": model.training_start.isoformat(),
            "training_end": model.training_end.isoformat(), "objective_semantics": "daily receding one-step hedge",
            "spot_cost": 0., "option_half_spread_multiplier": 1.,
            "optimized_premium_cap": optimized_premium_cap,
            "atm_total_variance_max_correction": (float(np.max(coherent_variances - raw_variances))
                                                  if "BS delta+vega" in needed else None)})
        print(f"episode origin={origin} family={family} completed_steps={elapsed+1}/{horizon} "
              f"cost_sum={cumulative_costs.sum():.6g}", flush=True)
        discount_to_current *= float(one_day_discount)
        claims = observed_claims
        settled = np.flatnonzero(initial_book["bdte"].to_numpy(dtype=float) <= elapsed + 1. + 1e-6)
        holdings[:, 1 + settled] = 0.
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".partial.npz")
            np.savez_compressed(temporary, identity_json=json.dumps(initial_identity), next_step=elapsed + 1,
                holdings=holdings, cumulative_gains=cumulative_gains, cumulative_costs=cumulative_costs,
                cumulative_exit_metric=cumulative_exit_metric, cap_days=cap_days, terminated=terminated,
                premium_breach_days=premium_breach_days, maximum_premium_positions=maximum_premium_positions,
                maximum_gross_premium_positions=maximum_gross_premium_positions,
                terminal_exit_metrics=terminal_exit_metrics, hedge_end_dates_json=json.dumps(hedge_end_dates),
                termination_reasons_json=json.dumps(termination_reasons), claims_json=json.dumps([asdict(claim) for claim in claims]),
                discount_to_current=discount_to_current, rows_json=json.dumps(rows), diagnostics_json=json.dumps(diagnostics),
                **{f"positions_{step}": values for step, values in enumerate(saved_positions)},
                **{f"indices_{step}": values for step, values in enumerate(saved_indices)})
            temporary.replace(checkpoint)
        if stop_requested is not None and stop_requested():
            if checkpoint is None:
                raise ValueError("Graceful stopping requires a checkpoint directory")
            raise EpisodeCheckpointedStop(f"Checkpointed {origin} {family} after step {elapsed + 1}/{horizon}")
    summary = []
    for portfolio, (target, rule) in enumerate(keys):
        claim = claims[target]
        payoff = claim.payoff(markets[-1].curve.spot)
        summary.append({"origin": origin.isoformat(), "terminal_date": sessions[-1].isoformat(),
            "contractual_terminal_date": sessions[-1].isoformat(), "hedge_end_date": hedge_end_dates[portfolio],
            "termination_reason": termination_reasons[portfolio],
            "claim": claim.label, **asdict(rule), "initial_price": initial_prices[target],
            "initial_price_source": "zero-smoothing DLV (LV mode)",
            "terminal_payoff": payoff, "discount_to_terminal": discount_to_current,
            "hedge_gain": cumulative_gains[portfolio], "entry_cost": cumulative_costs[portfolio],
            "hypothetical_exit_cost_sum": cumulative_exit_metric[portfolio], "cap_days": int(cap_days[portfolio]),
            "premium_comparison_breach_days": int(premium_breach_days[portfolio]),
            "maximum_absolute_premium_position": maximum_premium_positions[portfolio],
            "maximum_gross_premium_positions": maximum_gross_premium_positions[portfolio],
            "reference_premium_cap": reference_premium_cap,
            "optimized_premium_cap": optimized_premium_cap,
            "reference_is_uncapped": rule.method in ("unhedged", "BS delta+vega"),
            "terminal_hypothetical_exit_cost": terminal_exit_metrics[portfolio],
            "costed_pnl": terminal_pnl(side=rule.side, initial_price=initial_prices[target],
                terminal_payoff=payoff, discount_to_terminal=discount_to_current,
                hedge_gains=cumulative_gains[portfolio], entry_costs=cumulative_costs[portfolio])})
    return EpisodeResult(summary=pd.DataFrame(summary), trades=pd.DataFrame(rows), initial_book=initial_book,
                          positions=tuple(saved_positions), active_indices=tuple(saved_indices),
                          diagnostics=tuple(diagnostics), input_hashes=input_hashes)
