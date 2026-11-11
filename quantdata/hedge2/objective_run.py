"""Resumable matched-date one-day parametric and statistical hedge experiments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import signal
import time
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import MarketStore, discounted_daily_gains, observed_marks, select_book
from quantdata.hedge2.certification import certify_candidate
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.front_joint import EmpiricalFrontCopula, GaussianFrontCopula
from quantdata.hedge2.front_scenarios import generate_front_scenarios
from quantdata.hedge2.front_spot import DiscreteFrontLaw
from quantdata.hedge2.greeks import calculate_parametric_greeks, parametric_factor_covariance
from quantdata.hedge2.historical import build_historical_returns
from quantdata.hedge2.optimization import ScenarioOptimizer
from quantdata.hedge2.parametric import ParametricOptimizer
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.scenarios import generate_scenarios
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.state import DailyState


def save_checkpoint(path: Path, rows: list[dict[str, object]], positions: list[np.ndarray]) -> None:
    """One atomic file keeps row identities aligned with portfolio arrays."""
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, positions=np.asarray(positions), rows_json=json.dumps(rows))
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    if not path.exists():
        return [], []
    with np.load(path, allow_pickle=False) as saved:
        rows = json.loads(str(saved["rows_json"]))
        positions = list(saved["positions"])
    if len(rows) != len(positions):
        raise ValueError(f"Checkpoint row/portfolio alignment failed: {path}")
    return rows, positions


def save_verified_array_inputs(path: Path, values: dict[str, object]) -> None:
    """A resumed portfolio may not be combined with regenerated different inputs."""
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                    saved[name].dtype != value.dtype or not np.array_equal(saved[name], value)
                    for name, value in arrays.items()):
                raise ValueError(f"Saved risk inputs changed before resume: {path}")
        return
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_verified_book(path: Path, book: pd.DataFrame) -> None:
    if path.exists():
        if not pd.read_parquet(path).equals(book):
            raise ValueError(f"The fixed hedge book changed before resume: {path}")
        return
    temporary = path.with_suffix(".partial.parquet")
    book.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_verified_metadata(path: Path, metadata: dict[str, object]) -> None:
    serialized = json.dumps(metadata, indent=2, default=str) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(serialized):
            raise ValueError(f"Saved economic/source metadata changed before resume: {path}")
        return
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(serialized)
    temporary.replace(path)


def _identity(*, output_directory: Path, configuration: dict[str, object]) -> None:
    package = Path(__file__).parent
    names = ("objective_run.py", "certification.py", "scenarios.py", "objectives.py", "optimization.py",
             "parametric.py", "polish.py", "dual.py", "greeks.py", "claims.py", "cash.py", "state.py",
             "pricing.py", "source.py", "rolling.py", "dynamics.py", "historical.py", "dlv.py", "period.py",
             "front_joint.py", "front_scenarios.py", "front_spot.py", "front_scores.py", "batch_pricing.py",
             "terminal_market.py", "terminal.py")
    configuration["implementation_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}
    path = output_directory / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError("The output belongs to a different code, input or experiment identity")
        return
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(configuration, indent=2) + "\n")
    temporary.replace(path)
    with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(package / name, arcname=name)


def run_objectives(*, method: str, source_root: Path, licensed_fit_directory: Path,
                   raw_quote_directory: Path, market_directory: Path, daily_directory: Path | None,
                   dynamics_directory: Path | None, output_directory: Path, dates: tuple[date, ...],
                   objective_gap_tolerance: float = 1e-4, diagnostic_only: bool = False,
                   itm_premium_book: bool = False, front_scores_path: Path | None = None,
                   front_surface_law: str | None = None) -> dict[str, object]:
    arguments = dict(locals())
    output_directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_directory / "run.lock"), timeout=0)
    with lock:
        return _run_objectives_locked(**arguments, run_lock=lock)


def _run_objectives_locked(*, method: str, source_root: Path, licensed_fit_directory: Path,
                           raw_quote_directory: Path, market_directory: Path, daily_directory: Path | None,
                           dynamics_directory: Path | None, output_directory: Path, dates: tuple[date, ...],
                           objective_gap_tolerance: float, diagnostic_only: bool,
                           itm_premium_book: bool, front_scores_path: Path | None,
                           front_surface_law: str | None, run_lock: FileLock) -> dict[str, object]:
    if method not in ("parametric", "statistical", "historical") or not dates or tuple(sorted(set(dates))) != dates:
        raise ValueError("Choose a hedge method and nonempty, strictly increasing origin dates")
    if daily_directory is None:
        raise ValueError("Every method requires daily DLV states to value the hedged claims")
    if method != "historical" and dynamics_directory is None:
        raise ValueError("DLV methods require rolling dynamics")
    if (front_scores_path is None) != (front_surface_law is None):
        raise ValueError("Choose both a front-score panel and an explicit surface-shock law, or neither")
    if front_surface_law not in (None, "empirical_rank", "joint_gaussian") or method == "historical" and front_surface_law is not None:
        raise ValueError("Front surface law must be empirical_rank or joint_gaussian for a model-based hedge")
    if any(day < date(2023, 1, 1) or day > date(2025, 12, 31) for day in dates):
        raise ValueError("The agreed evaluation period is 2023--2025, initialized with 2022 history")
    output_directory.mkdir(parents=True, exist_ok=True)
    front_scores = None
    front_scores_sha256 = None
    if front_scores_path is not None:
        payload = front_scores_path.read_bytes()
        front_scores_sha256 = hashlib.sha256(payload).hexdigest()
        front_scores = pd.read_parquet(io.BytesIO(payload))
        for name in ("date", "pair_start"):
            front_scores[name] = pd.to_datetime(front_scores[name]).dt.date
    configuration = {"method": method, "source_root": str(source_root),
        "licensed_fit_directory": str(licensed_fit_directory), "raw_quote_directory": str(raw_quote_directory),
        "market_directory": str(market_directory), "daily_directory": None if daily_directory is None else str(daily_directory),
        "dynamics_directory": None if dynamics_directory is None else str(dynamics_directory), "dates": [day.isoformat() for day in dates],
        "objective_gap_tolerance": objective_gap_tolerance, "diagnostic_only": diagnostic_only,
        "option_cost_multiplier": 1., "spot_cost": 0., "position_cap": 10. if itm_premium_book else 1000.,
        "itm_premium_book": itm_premium_book,
        "front_surface_law": front_surface_law,
        "front_scores_path": None if front_scores_path is None else str(front_scores_path),
        "front_scores_sha256": front_scores_sha256,
        "position_basis": "current_cash_premium_dollars" if itm_premium_book else "episode_spot_normalized_contracts",
        "include_co_terminal_forwards": itm_premium_book, "same_close_execution_idealization": True}
    configuration["model_configuration_sha256"] = {name: hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
        for name, directory in (("daily_decoder", daily_directory), ("dynamics", dynamics_directory)) if directory is not None}
    configuration["claim_valuation"] = "zero-smoothing DLV (LV mode)"
    _identity(output_directory=output_directory, configuration=configuration)
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    started = time.monotonic()
    stopping = False

    def request_stop(signum, frame) -> None:
        nonlocal stopping
        stopping = True
        print(f"stop requested signal={signum}; saving the current cell before exit", flush=True)

    previous_handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)}
    all_rows = []
    try:
        with run_lock:
            for day_index, day in enumerate(dates):
                directory = output_directory / day.isoformat()
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = directory / "checkpoint.npz"
                rows, positions = load_checkpoint(checkpoint)
                completed = {(row["claim"], row["side"], row["objective"], row["risk_aversion"]) for row in rows}
                if len(completed) != len(rows):
                    raise ValueError(f"Duplicate checkpoint cells on {day}")
                if rows and not all(row["optimization_certified"] for row in rows):
                    raise RuntimeError(f"An uncertified candidate on {day} needs investigation before resuming")
                if len(rows) == 180:
                    manifest_path = directory / "completed.json"
                    if not manifest_path.exists():
                        raise ValueError(f"Completed cells need an input-integrity manifest before resume: {directory}")
                    manifest = json.loads(manifest_path.read_text())
                    for name, digest in manifest["artifacts_sha256"].items():
                        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                            raise ValueError(f"Completed day artifact changed: {directory / name}")
                    all_rows.extend(rows)
                    continue
                market = store.load(day)
                following_day = get_business_days(day, day + timedelta(days=10))[1]
                following = store.load(following_day)
                source, source_provenance = archive.load(day)
                state = DailyState.load(daily_directory, day)
                if method != "historical":
                    model_path = dynamics_directory / "days" / f"{day.isoformat()}.npz"
                    model = load_dynamics(model_path)
                    if (model.training_start < date(2022, 1, 1) or model.training_end > day
                            or model.as_of_date != day or len(model.residuals) != 252):
                        raise ValueError("Rolling model violates the agreed causal training period")
                    coupling = (None if front_scores is None else
                        {"empirical_rank": EmpiricalFrontCopula, "joint_gaussian": GaussianFrontCopula}[front_surface_law]
                        .from_dynamics(model=model, scores=front_scores))
                vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
                book = select_book(market, atm_vols=vols, option_side="itm" if itm_premium_book else "otm",
                    include_co_terminal_forwards=itm_premium_book)
                claims = tuple(claim for claim in make_claims(initial_spot=market.curve.spot,
                    atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR))
                                    for tenor in (1, 2, 3, 5)})
                    if claim.kind == "digital" and claim.terminal_day == 1)
                parity = market.curve.at(1.)[0]
                # Hedged claims are valued with the zero-smoothing DLV in LV mode, like every claim type.
                state_pricer = state.decode().pricer
                common_target_marks = np.array([claim.price(state_pricer, market.curve, elapsed_day=0) for claim in claims])
                save_verified_book(directory / "book.parquet", book)
                costs = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float) / market.curve.spot]
                cash_prices = np.r_[market.curve.spot, book["observed_mid"].to_numpy(dtype=float)]
                if not np.isfinite(cash_prices).all() or np.any(cash_prices <= 0):
                    raise ValueError("Positive observed prices are required without premium floors")
                unit_conversion = market.curve.spot / cash_prices if itm_premium_book else np.ones(len(costs))
                costs *= unit_conversion
                next_marks, exit_spreads = observed_marks(book, following)
                realized = discounted_daily_gains(previous_option_marks=book["observed_mid"].to_numpy(dtype=float),
                    next_option_marks=next_marks, next_spot=following.curve.spot, previous_curve=market.curve,
                    episode_initial_spot=market.curve.spot)
                realized *= unit_conversion
                metadata = {"date": day.isoformat(), "source_provenance": source_provenance,
                    "claims": [asdict(claim) for claim in claims], "instruments": len(costs),
                    "scenario_seed": day.toordinal(), "realized_coverage": "explicit supplied complete-book date cohort",
                    "position_basis": configuration["position_basis"],
                    "option_side": "itm" if itm_premium_book else "otm",
                    "target_mark": "zero-smoothing DLV (LV mode), not an observed digital quote"}
                dated_files = {f"{kind}_{when}": root / suffix / f"{selected.isoformat()}.{extension}"
                    for kind, root, suffix, extension in (
                        ("observed_quotes", market_directory, "days", "parquet"),
                        ("licensed_carry", licensed_fit_directory, "SPX", "pdq"),
                        ("licensed_raw_quotes", raw_quote_directory, "SPX", "pdq"))
                    for when, selected in (("current", day), ("following", following_day))}
                dated_files.update(daily_state=daily_directory / "days" / f"{day.isoformat()}.npz",
                    daily_decoder_configuration=daily_directory / "config.json")
                if method != "historical":
                    dated_files["dynamics"] = model_path
                    if front_scores_path is not None:
                        dated_files["front_score_panel"] = front_scores_path
                metadata["input_sha256"] = {name: {"path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in dated_files.items()}
                if method != "historical":
                    metadata.update(dynamics_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        training_start=model.training_start.isoformat(), training_end=model.training_end.isoformat(),
                        latest_transition_ends_today=model.training_end == day,
                        latest_transition_session_lag=len(get_business_days(model.training_end, day)) - 1)
                    if coupling is not None:
                        training_scores = front_scores.loc[(front_scores["pair_start"] >= model.training_start)
                            & (front_scores["date"] <= model.training_end)]
                        metadata.update(front_surface_law=coupling.law_kind,
                            front_dependence_identity=coupling.identity,
                            front_score_count=len(training_scores),
                            front_short_expiry_extrapolations=int(training_scores["source_short_expiry_extrapolation"].sum()),
                            historical_score_mean=float(coupling.historical_scores.mean()),
                            historical_score_std_population=float(coupling.historical_scores.std()),
                            target_mark_convention="common zero-smoothing DLV marks, as in the scenario repricing")
                if method in ("statistical", "historical"):
                    if method == "historical":
                        scenarios = build_historical_returns(archive=archive, licensed_fit_directory=licensed_fit_directory,
                            daily_directory=daily_directory,
                            as_of_date=day, current_spot=market.curve.spot, book=book, claims=claims,
                            return_basis="premium" if itm_premium_book else "spot_notional",
                            progress=lambda message: print(message, flush=True))
                        metadata.update(training_start=str(scenarios.pair_start_dates[0]),
                            training_end=str(scenarios.pair_end_dates[-1]), transitions=len(scenarios.weights),
                            conditional_simulation=False, scenario_seed=None,
                            contract_mapping=scenarios.contract_mapping, target_convention=scenarios.target_convention,
                            return_convention=scenarios.return_convention,
                            mean_removal="uniform historical instrument gains only; raw gains and means retained")
                    else:
                        arguments = dict(state=state, dynamics=model, curve=market.curve,
                            book=book, claims=claims, initial_spot=market.curve.spot, seed=day.toordinal())
                        scenarios = (generate_scenarios(**arguments) if coupling is None else
                            generate_front_scenarios(**arguments, coupling=coupling, diagnostic_only=diagnostic_only))
                        if itm_premium_book:
                            scenarios = replace(scenarios,
                                instrument_gains=scenarios.instrument_gains * unit_conversion[None, :],
                                removed_instrument_mean=scenarios.removed_instrument_mean * unit_conversion)
                        metadata.update(draw_count=scenarios.draw_count, distinct_innovations=scenarios.distinct_innovations,
                            effective_innovation_sample_size=scenarios.effective_innovation_sample_size,
                            mean_removal="weighted asset gains after nonlinear repricing")
                        if coupling is not None:
                            metadata.update(forecast_identity=scenarios.forecast_identity,
                                forecast_scope=scenarios.forecast_scope,
                                mean_removal="Monte Carlo sample-frequency asset centering; not exact branch-integrated conditional expectation")
                    save_verified_array_inputs(directory / "scenarios.npz", asdict(scenarios))
                    optimizer = ScenarioOptimizer(instrument_gains=scenarios.instrument_gains,
                        weights=scenarios.weights, costs=costs, position_cap=configuration["position_cap"])
                    targets = scenarios.claim_gains.T
                else:
                    greeks = calculate_parametric_greeks(state=state, dynamics=model, curve=market.curve,
                        book=book, claims=claims, initial_spot=market.curve.spot)
                    covariance = (parametric_factor_covariance(model) if coupling is None else
                        parametric_factor_covariance(model, front_coupling=coupling,
                            front=DiscreteFrontLaw.from_pricer(pricer=state_pricer, parity_forward=float(parity)),
                            reference_forward=float(market.curve.at(1.)[1])))
                    greeks = replace(greeks, instrument_exposures=greeks.instrument_exposures * unit_conversion[:, None],
                        combined_spot_instrument_exposures=(None if greeks.combined_spot_instrument_exposures is None
                            else greeks.combined_spot_instrument_exposures * unit_conversion))
                    optimizer = ParametricOptimizer(exposures=greeks.instrument_exposures, covariance=covariance,
                        costs=costs, position_cap=configuration["position_cap"])
                    targets = greeks.claim_exposures
                    saved_greeks = {name: value for name, value in asdict(greeks).items() if value is not None}
                    save_verified_array_inputs(directory / "risk_inputs.npz", dict(covariance=covariance, **saved_greeks))
                    metadata.update(normal_risk="exact truncated-normal power OCE; full covariance MV",
                                    spot_linearization="sum of finite fixed-u spot and full-beta surface partials")
                    if coupling is not None:
                        metadata["normal_risk"] = "Gaussian risk approximation with front-calibrated physical spot/factor covariance"
                save_verified_array_inputs(directory / "realized_inputs.npz", {
                    "costs": costs, "realized_gains": realized, "exit_spreads": exit_spreads,
                    "following_spot": following.curve.spot, "current_spot": market.curve.spot,
                    "cash_holdings_multiplier": unit_conversion / market.curve.spot,
                    "current_midpoints": book["observed_mid"].to_numpy(dtype=float)})
                save_verified_metadata(directory / "inputs.json", metadata)
                forward, _, discount = market.curve.at(1.)
                date_started = time.monotonic()
                print(f"{method} date={day} origins={day_index}/{len(dates)} cells={len(rows)}/180 "
                      f"instruments={len(costs)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
                for target_index, claim in enumerate(claims):
                    market_price = float(common_target_marks[target_index])
                    target_gain = float(discount * claim.payoff(following.curve.spot) - market_price)
                    for side in (-1, 1):
                        for objective in ("mean_variance", "power_exponential"):
                            risks = (1., 10., 100., 1000., np.inf) if objective == "mean_variance" else (1., 10., 100., 1000.)
                            for risk in risks:
                                key = (claim.label, side, objective, risk)
                                if key in completed:
                                    continue
                                if method in ("statistical", "historical"):
                                    candidate = optimizer.solve(claim_gains=targets[target_index], objective=objective,
                                                                risk_aversion=risk, target_side=side)
                                    hedge, certificate = certify_candidate(optimizer=optimizer, candidate=candidate,
                                        claim_gains=targets[target_index], objective=objective, risk_aversion=risk,
                                        target_side=side, gap_tolerance=objective_gap_tolerance,
                                        certificate_directory=directory / "certificates" / f"{target_index:02d}-{side}-{objective}-{risk}")
                                else:
                                    result = optimizer.solve(target_exposures=targets[target_index], objective=objective,
                                                             risk_aversion=risk, target_side=side)
                                    hedge = result.hedge
                                    certificate = {"effective_mv_aversion": result.effective_mv_aversion,
                                        "aversion_equation_residual": result.aversion_equation_residual,
                                        "gaussian_risk_method": result.gaussian_risk_method,
                                        "certified_objective_gap": hedge.convex_first_order_gap_bound,
                                        "optimization_certified": hedge.convex_first_order_gap_bound <= objective_gap_tolerance}
                                row = {"date": day.isoformat(), "next_date": following_day.isoformat(),
                                    "claim": claim.label, "side": side,
                                    "method": "SANOS historical" if method == "historical" else f"DLV {method}",
                                    "objective": objective, "risk_aversion": risk, "market_target_price": market_price,
                                    "target_gain": side * target_gain, "hedge_gain": float(hedge.positions @ realized),
                                    **{name: value for name, value in asdict(hedge).items() if name != "positions"},
                                    "costed_pnl": side * target_gain + float(hedge.positions @ realized) - hedge.entry_cost,
                                    "hypothetical_exit_cost": float(discount * (np.abs(hedge.positions[1:]) * unit_conversion[1:]) @ exit_spreads / market.curve.spot),
                                    "gross_positions": float(np.abs(hedge.positions).sum()), **certificate}
                                rows.append(row)
                                positions.append(hedge.positions)
                                save_checkpoint(checkpoint, rows, positions)
                                if not row["optimization_certified"]:
                                    raise RuntimeError(f"Objective certificate needs review: {day} {key}, "
                                                       f"gap={row['certified_objective_gap']}")
                                if stopping:
                                    return {"status": "checkpointed_stop", "last_date": day.isoformat(),
                                            "cells_on_last_date": len(rows), "elapsed_seconds": time.monotonic()-started}
                            elapsed = time.monotonic() - started
                            estimated_remaining = (len(dates) - day_index - 1) * (time.monotonic() - date_started)
                            print(f"{method} date={day} origins={day_index}/{len(dates)} cells={len(rows)}/180 "
                                  f"elapsed_s={elapsed:.1f} remaining_s~{estimated_remaining:.0f}", flush=True)
                pd.DataFrame(rows).to_parquet(directory / "results.parquet", index=False)
                np.savez_compressed(directory / "positions.npz", positions=np.asarray(positions))
                saved_names = ("checkpoint.npz", "results.parquet", "positions.npz", "book.parquet",
                    "inputs.json", "realized_inputs.npz", "risk_inputs.npz" if method == "parametric" else "scenarios.npz")
                save_verified_metadata(directory / "completed.json", {"certified_cells": len(rows),
                    "artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in saved_names}})
                all_rows.extend(rows)
            final = pd.DataFrame(all_rows)
            final.to_parquet(output_directory / "results.parquet", index=False)
            summary = {"status": "complete", "method": method, "dates": len(dates), "cells": len(final),
                "certified_cells": int(final["optimization_certified"].sum()),
                "maximum_objective_gap": float(final["certified_objective_gap"].max()),
                "elapsed_seconds": time.monotonic()-started, "diagnostic_only": diagnostic_only,
                "complete_case_cohort": True, "entry_only_costs": True}
            (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return summary
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("parametric", "statistical", "historical"), required=True)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory",
                 "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--daily-directory", type=Path)
    parser.add_argument("--dynamics-directory", type=Path)
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--dates", type=date.fromisoformat, nargs="+")
    dates.add_argument("--cohort-config", type=Path)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--itm-premium-book", action="store_true")
    parser.add_argument("--front-scores-path", type=Path)
    parser.add_argument("--front-surface-law", choices=("empirical_rank", "joint_gaussian"))
    args = vars(parser.parse_args())
    cohort = args.pop("cohort_config")
    args["dates"] = tuple(date.fromisoformat(value) for value in json.loads(cohort.read_text())["dates"]) if cohort else tuple(args["dates"])
    print(json.dumps(run_objectives(**args), indent=2), flush=True)


if __name__ == "__main__":
    main()
