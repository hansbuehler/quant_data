"""Dated current-Greek mixed baskets and independent observed one-day cash replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.black_hedge import black_digital_hedge
from quantdata.hedge2.cash import (
    MarketStore,
    discounted_daily_gains,
    observed_marks,
    select_book,
    zero_strike_mask,
)
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.factor_portfolios import premium_to_cash_holdings
from quantdata.hedge2.greek_hedge import solve_greek_hedge
from quantdata.hedge2.greeks import calculate_greeks
from quantdata.hedge2.local_portfolios import (
    LocalPortfolioConfig,
    fit_local_portfolios,
    hedge_in_local_basis,
    minimize_basis_costs,
)
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.source import SourceArchive, surface_implied_vols
from quantdata.hedge2.state import DailyState


def run_local_pilot(*, source_configuration: Path, output_directory: Path, dates: tuple[date, ...],
                    config: LocalPortfolioConfig = LocalPortfolioConfig()) -> dict[str, object]:
    """Recompute today's Greeks; never regress historical latent factor payouts.

    Each date uses its as-of-close rolling model and actual ITM midpoint
    book, not later models or later contract selection. Realized next-day
    P&L is descriptive on pilot dates, not rolling performance evidence.
    The eight baskets may mix all PC and spot exposures. Separate spot
    completes the nine-axis basis. Combined hedge caps are reported rather
    than inferred from the individual basket bounds.
    """
    if not dates or tuple(sorted(set(dates))) != dates:
        raise ValueError("Mixed-basket pilot dates must be distinct and chronological")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        sources = json.loads(source_configuration.read_text())
        implementation = sorted(Path(__file__).parent.glob("*.py"))
        inputs = {}
        for day in dates:
            next_day = get_business_days(day, day + timedelta(days=10))[1]
            for label, root in (("daily_state", Path(sources["daily_directory"])),
                                ("dynamics", Path(sources["dynamics_directory"]))):
                inputs[f"{day}/{label}_config"] = root / "config.json"
                inputs[f"{day}/{label}"] = root / "days" / f"{day.isoformat()}.npz"
            for quote_day in (day, next_day):
                inputs[f"{quote_day}/market"] = Path(sources["market_directory"]) / "days" / f"{quote_day.isoformat()}.parquet"
                for label in ("licensed_fit_directory", "raw_quote_directory"):
                    inputs[f"{quote_day}/{label}"] = Path(sources[label]) / "SPX" / f"{quote_day.isoformat()}.pdq"
        identity = {"source_configuration": str(source_configuration),
            "source_configuration_sha256": hashlib.sha256(source_configuration.read_bytes()).hexdigest(),
            "dates": [day.isoformat() for day in dates], "config": asdict(config),
            "config_version": LocalPortfolioConfig.version.unique_id48,
            "basis": "current price Greeks; mixed PC/spot loadings; not historical factor regression",
            "limits": "per basket only; combined hedge violations are separately reported",
            "option_side": "itm", "spot_cost": 0., "entry_spread_multiplier": 1.,
            "same_close_execution_idealization": True, "diagnostic_only": True,
            "inputs_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
            "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}}
        identity_path = output_directory / "config.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Mixed-basket output belongs to different sources, dates, limits or code")
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        if not (output_directory / "implementation.zip").exists():
            with ZipFile(output_directory / "implementation.zip", "w", compression=ZIP_DEFLATED) as saved:
                for path in implementation:
                    saved.write(path, arcname=path.name)
        store = MarketStore(market_directory=Path(sources["market_directory"]),
            licensed_fit_directory=Path(sources["licensed_fit_directory"]),
            raw_quote_directory=Path(sources["raw_quote_directory"]))
        archive = SourceArchive(root=Path(sources["source_root"]))
        daily_directory, dynamics_directory = Path(sources["daily_directory"]), Path(sources["dynamics_directory"])
        rows, basis_rows, started = [], [], time.monotonic()
        for day_index, day in enumerate(dates):
            directory = output_directory / day.isoformat()
            directory.mkdir(parents=True, exist_ok=True)
            completion = directory / "completed.json"
            if completion.exists():
                for name, digest in json.loads(completion.read_text())["artifacts_sha256"].items():
                    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Completed local-basket artifact changed: {name}")
                rows.extend(pd.read_parquet(directory / "hedges.parquet").to_dict("records"))
                basis_rows.extend(pd.read_parquet(directory / "baskets.parquet").to_dict("records"))
                continue
            market = store.load(day)
            source, source_metadata = archive.load(day)
            state = DailyState.load(daily_directory, day)
            dynamics_path = dynamics_directory / "days" / f"{day.isoformat()}.npz"
            dynamics = load_dynamics(dynamics_path)
            if (dynamics.as_of_date != day or dynamics.training_end > day
                    or dynamics.training_start < date(2022, 1, 1) or dynamics.residuals.shape != (252, 9)):
                raise ValueError("Mixed baskets require today's causal 252-transition eight-PC model")
            vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
            book = select_book(market, atm_vols=vols, option_side="itm", include_co_terminal_forwards=True)
            claims = tuple(claim for claim in make_claims(initial_spot=market.curve.spot,
                atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
                if claim.kind == "digital" and claim.terminal_day == 1)
            greeks = calculate_greeks(state=state, dynamics=dynamics, curve=market.curve,
                book=book, claims=claims, initial_spot=market.curve.spot)
            prices = np.r_[market.curve.spot, book["observed_mid"].to_numpy(dtype=float)]
            if not np.isfinite(prices).all() or np.any(prices <= 0):
                raise ValueError("Current premium units require positive observed prices without floors")
            scales = np.r_[np.sqrt(dynamics.joint_innovation_covariance[0, 0]), np.ones(8)]
            exposure_map = scales[:, None] * greeks.instrument_exposures.T * market.curve.spot / prices[None, :]
            target_exposures = greeks.claim_exposures * scales[None, :]
            cash_fees = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float)]
            costs = cash_fees / prices
            fit = fit_local_portfolios(price_factor_greeks=exposure_map[1:].T,
                proportional_costs=costs, config=config)
            if not fit.converged:
                raise RuntimeError(f"Local-basket block optimization needs further convergence: {fit.block_improvement_bounds}")
            refined = minimize_basis_costs(exposure_map=exposure_map, premium_positions=fit.premium_positions,
                proportional_costs=costs, limits=config.limits)
            basis = refined.premium_positions
            next_day = get_business_days(day, day + timedelta(days=10))[1]
            following = store.load(next_day)
            next_marks, next_spreads = observed_marks(book, following)
            daily_gains = discounted_daily_gains(previous_option_marks=prices[1:], next_option_marks=next_marks,
                next_spot=following.curve.spot, previous_curve=market.curve, episode_initial_spot=market.curve.spot)
            _, reference, discount = market.curve.at(1.)
            cash_gains = np.r_[discount * (following.curve.spot - reference), discount * next_marks - prices[1:]]
            if np.max(np.abs(cash_gains / market.curve.spot - daily_gains)) > 1e-12:
                raise RuntimeError("Independent current-book cash gains disagree with normalized accounting")
            option_vols = np.asarray(source.interpolate_atm_vols(book["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR)).copy()
            ordinary = ~zero_strike_mask(book)
            option_vols[ordinary] = surface_implied_vols(source,
                expiries_y=book.loc[ordinary, "bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR,
                strikes=book.loc[ordinary, "strike"].to_numpy(dtype=float) / book.loc[ordinary, "fwd"].to_numpy(dtype=float))
            forward = market.curve.at(1.)[0]
            dated_rows, portfolio_positions, replay_errors = [], [], []
            for claim_index, claim in enumerate(claims):
                mark = float(claim.price(state.decode().pricer, market.curve, elapsed_day=0))
                target_gain = float(discount * claim.payoff(following.curve.spot) - mark)
                target_vol = surface_implied_vols(source, expiries_y=np.array([1. / DAYS_PER_YEAR]),
                    strikes=np.array([claim.strike / forward]))
                for side in (-1, 1):
                    mixed = hedge_in_local_basis(exposure_map=exposure_map, basis_positions=basis,
                        target_exposures=target_exposures[claim_index], target_side=side)
                    full = solve_greek_hedge(exposures=exposure_map.T,
                        target_exposures=target_exposures[claim_index], costs=costs,
                        target_side=side, position_cap=config.limits.instrument_cap)
                    black = black_digital_hedge(claim=claim, curve=market.curve, book=book,
                        target_vol=float(target_vol[0]), option_vols=np.asarray(option_vols), target_side=side)
                    candidates = (("unhedged", np.zeros(len(prices))),
                        ("BS delta+vega", black * prices / market.curve.spot),
                        ("DLV Greek full book", full.positions),
                        ("DLV Greek mixed baskets", mixed.premium_positions))
                    for method, positions in candidates:
                        cash_holdings = premium_to_cash_holdings(positions[:, None], prices)[:, 0]
                        entry_cost = float(costs @ np.abs(positions))
                        hedge_gain = float(cash_holdings @ cash_gains)
                        replay_gain = float((cash_holdings * market.curve.spot) @ daily_gains)
                        replay_fee = float(np.abs(cash_holdings) @ cash_fees)
                        replay_errors.append(max(abs(hedge_gain - replay_gain), abs(entry_cost - replay_fee)))
                        residual = exposure_map @ positions + side * target_exposures[claim_index]
                        dated_rows.append({"date": day.isoformat(), "next_date": next_day.isoformat(),
                            "claim": claim.label, "side": side, "method": method,
                            "target_mark": mark, "dlv_target_price": float(greeks.claim_prices[claim_index]),
                            "target_gain": side * target_gain, "hedge_gain": hedge_gain, "entry_cost": entry_cost,
                            "costed_pnl": side * target_gain + hedge_gain - entry_cost,
                            "hypothetical_exit_cost": float(discount * np.abs(cash_holdings[1:]) @ next_spreads),
                            "minimum_dollar_position": float(positions.min()), "maximum_dollar_position": float(positions.max()),
                            "spot_dollar_position": float(positions[0]),
                            "maximum_absolute_option_dollars": float(np.abs(positions[1:]).max(initial=0.)),
                            "gross_dollar_positions": float(np.abs(positions).sum()),
                            "instrument_cap_hits": int((np.abs(positions) >= config.limits.instrument_cap * (1. - 1e-6)).sum()),
                            "instrument_cap_exceeded": bool(np.max(np.abs(positions)) > config.limits.instrument_cap + 1e-8),
                            "gross_cap_exceeded": bool(np.abs(positions).sum() > config.limits.gross_cap + 1e-8),
                            "greek_residual_norm": float(np.linalg.norm(residual)),
                            "basket_coefficients": json.dumps(mixed.basket_holdings.tolist()) if method == "DLV Greek mixed baskets" else "[]"})
                        portfolio_positions.append(positions)
            dated_baskets = [{"date": day.isoformat(), "basket": index + 1,
                "minimum_dollar_position": float(basis[:, index].min()),
                "maximum_dollar_position": float(basis[:, index].max()),
                "gross_dollar_positions": float(np.abs(basis[:, index]).sum()),
                "entry_cost": float(refined.entry_costs[index]),
                "numerical_cost_gap": float(refined.cost_gaps[index]),
                "scaled_exposure_residual": float(refined.scaled_exposure_residuals[index])} for index in range(8)]
            book.to_parquet(directory / "book.parquet", index=False)
            pd.DataFrame(dated_rows).to_parquet(directory / "hedges.parquet", index=False)
            pd.DataFrame(dated_baskets).to_parquet(directory / "baskets.parquet", index=False)
            positions_frame = book[["expiry", "strike", "option_type", "instrument_kind"]].copy()
            positions_frame = pd.concat([pd.DataFrame([{"instrument_kind": "spot"}]), positions_frame], ignore_index=True)
            for index in range(8):
                positions_frame[f"basket_{index+1}_dollars"] = basis[:, index]
                positions_frame[f"basket_{index+1}_cash_holdings"] = basis[:, index] / prices
            positions_frame.to_csv(directory / "dollar_positions.csv", index=False)
            np.savez_compressed(directory / "portfolios.npz", original_basis=fit.premium_positions,
                basis_positions=basis, basis_cash_holdings=premium_to_cash_holdings(basis, prices),
                greek_loadings=refined.greek_loadings, exposure_map=exposure_map, current_cash_prices=prices,
                proportional_costs=costs, axis_scales=scales, target_exposures=target_exposures,
                hedge_positions=np.asarray(portfolio_positions), pc_lower_clips=greeks.pc_lower_clips,
                pc_upper_clips=greeks.pc_upper_clips, log_volumes=fit.log_volumes)
            pd.DataFrame(refined.greek_loadings, index=["spot"] + [f"PC{index+1}" for index in range(8)],
                columns=[f"basket{index+1}" for index in range(8)]).to_csv(directory / "mixed_greek_loadings.csv")
            dated_summary = {"date": day.isoformat(), "source_model": source_metadata,
                "dynamics_sha256": hashlib.sha256(dynamics_path.read_bytes()).hexdigest(),
                "training_start": dynamics.training_start.isoformat(), "training_end": dynamics.training_end.isoformat(),
                "instruments": len(prices), "sweeps": fit.sweeps, "converged": fit.converged,
                "PC_loading_condition_number": fit.condition_number,
                "augmented_condition_number": mixed.augmented_condition_number,
                "maximum_block_improvement_bound": float(fit.block_improvement_bounds.max()),
                "maximum_cash_replay_error": max(replay_errors),
                "cost_refinement_raw_messages": refined.solver_messages,
                "cost_bounds_qualification": "numerical gaps with recorded exposure residuals; not exact feasible-primal certificates",
                "statistic_scope": "dated current-Greek span and cash replay, not rolling OOS performance"}
            if max(replay_errors) > 1e-10:
                raise RuntimeError("The mixed-basket pilot failed independent dollar cash replay")
            (directory / "summary.json").write_text(json.dumps(dated_summary, indent=2) + "\n")
            names = ("book.parquet", "hedges.parquet", "baskets.parquet", "dollar_positions.csv", "portfolios.npz",
                     "mixed_greek_loadings.csv", "summary.json")
            marker = {"artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
            temporary = completion.with_suffix(".partial.json")
            temporary.write_text(json.dumps(marker, indent=2) + "\n")
            temporary.replace(completion)
            rows.extend(dated_rows)
            basis_rows.extend(dated_baskets)
            print(f"local Greek baskets dates={day_index+1}/{len(dates)} date={day} baskets=8 "
                f"hedges={len(dated_rows)} condition={fit.condition_number:.3g} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        pd.DataFrame(rows).to_parquet(output_directory / "hedges.parquet", index=False)
        pd.DataFrame(basis_rows).to_parquet(output_directory / "baskets.parquet", index=False)
        summary = {"dates": len(dates), "baskets": len(basis_rows), "hedge_records": len(rows),
            "elapsed_seconds": time.monotonic()-started, "diagnostic_only": True,
            "same_close_execution_idealization": True, "basis_not_historical_regression": True}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    arguments = vars(parser.parse_args())
    arguments["dates"] = tuple(arguments["dates"])
    print(json.dumps(run_local_pilot(**arguments), indent=2), flush=True)


if __name__ == "__main__":
    main()
