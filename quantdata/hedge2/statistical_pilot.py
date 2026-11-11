"""A full objective/risk/side diagnostic on a few dated nonlinear scenario books."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import MarketStore, discounted_daily_gains, observed_marks, select_book
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.optimization import ScenarioOptimizer
from quantdata.hedge2.rolling import load_dynamics
from quantdata.hedge2.scenarios import generate_scenarios
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.state import DailyState


def run_statistical_pilot(*, source_root: Path, licensed_fit_directory: Path, raw_quote_directory: Path,
                          market_directory: Path, daily_directory: Path, dynamics_directory: Path,
                          output_directory: Path, dates: tuple[date, ...]) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    started = time.monotonic()
    all_rows = []
    for day in dates:
        day_directory = output_directory / day.isoformat()
        day_directory.mkdir(parents=True, exist_ok=True)
        market = store.load(day)
        following = store.load(get_business_days(day, day + timedelta(days=10))[1])
        source, _ = SourceArchive(root=source_root).load(day)
        state = DailyState.load(daily_directory, day)
        model = load_dynamics(dynamics_directory / "days" / f"{day.isoformat()}.npz")
        vols = np.asarray(source.interpolate_atm_vols(market.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
        book = select_book(market, atm_vols=vols)
        claims = tuple(claim for claim in make_claims(initial_spot=market.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
            if claim.kind == "digital" and claim.terminal_day == 1)
        scenarios = generate_scenarios(state=state, dynamics=model, curve=market.curve, book=book,
                                       claims=claims, initial_spot=market.curve.spot, seed=day.toordinal())
        np.savez_compressed(day_directory / "scenarios.npz", instrument_gains=scenarios.instrument_gains,
                            claim_gains=scenarios.claim_gains, weights=scenarios.weights,
                            removed_instrument_mean=scenarios.removed_instrument_mean,
                            lower_clips=scenarios.lower_clips, upper_clips=scenarios.upper_clips,
                            draw_count=scenarios.draw_count, distinct_innovations=scenarios.distinct_innovations,
                            effective_innovation_sample_size=scenarios.effective_innovation_sample_size)
        book.to_parquet(day_directory / "book.parquet", index=False)
        costs = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float) / market.curve.spot]
        optimizer = ScenarioOptimizer(instrument_gains=scenarios.instrument_gains, weights=scenarios.weights, costs=costs)
        prices, exit_spreads = observed_marks(book, following)
        realized = discounted_daily_gains(previous_option_marks=book["observed_mid"].to_numpy(dtype=float),
                                          next_option_marks=prices, next_spot=following.curve.spot,
                                          previous_curve=market.curve, episode_initial_spot=market.curve.spot)
        forward, _, discount = market.curve.at(1.)
        rows, holdings = [], []
        total_cells = len(claims) * 2 * 9
        print(f"statistical pilot date={day} scenarios={scenarios.draw_count} "
              f"distinct={scenarios.distinct_innovations} instruments={len(costs)} cells={total_cells}", flush=True)
        for target_index, claim in enumerate(claims):
            market_price = float(claim.price(state.decode().pricer, market.curve, elapsed_day=0))
            target_gain = float(discount * claim.payoff(following.curve.spot) - market_price)
            for side in (-1, 1):
                for objective in ("mean_variance", "power_exponential"):
                    risks = (1., 10., 100., 1000., np.inf) if objective == "mean_variance" else (1., 10., 100., 1000.)
                    for risk in risks:
                        hedge = optimizer.solve(claim_gains=scenarios.claim_gains[:, target_index],
                                                 objective=objective, risk_aversion=risk, target_side=side)
                        rows.append({"date": day.isoformat(), "claim": claim.label, "side": side,
                                     "objective": objective, "risk_aversion": risk,
                                     "training_objective": hedge.training_objective,
                                     "training_mean": hedge.training_mean, "training_std": hedge.training_std,
                                     "entry_cost": hedge.entry_cost,
                                     "costed_pnl": side * target_gain + float(hedge.positions @ realized) - hedge.entry_cost,
                                     "hypothetical_exit_cost": float(discount * np.abs(hedge.positions[1:]) @ exit_spreads / market.curve.spot),
                                     "cap_hits": hedge.cap_hits, "gross_positions": float(np.abs(hedge.positions).sum()),
                                     "cash_first_order_residual": hedge.cash_first_order_residual,
                                     "convex_first_order_gap_bound": hedge.convex_first_order_gap_bound,
                                     "solver": hedge.solver, "status": hedge.status,
                                     "warnings": json.dumps(hedge.warnings), "iterations": hedge.iterations})
                        holdings.append(hedge.positions)
                    elapsed = time.monotonic() - started
                    print(f"statistical pilot {day} cells={len(rows)}/{total_cells} "
                          f"target={claim.label} side={side} objective={objective} elapsed_s={elapsed:.1f}", flush=True)
                    temporary = day_directory / "results.partial.parquet"
                    pd.DataFrame(rows).to_parquet(temporary, index=False)
                    temporary.replace(day_directory / "results.parquet")
                    np.savez_compressed(day_directory / "positions.npz", positions=np.asarray(holdings))
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_parquet(output_directory / "results.parquet", index=False)
    summary = {"dates": [day.isoformat() for day in dates], "cells": len(all_rows),
               "elapsed_seconds": time.monotonic() - started,
               "innovation_law": "joint empirical residual bootstrap; 10000 or 10*instruments draws, compressed exactly by frequencies",
               "asset_mean_removal": "weighted demeaning after nonlinear repricing; target gains are not demeaned",
               "pnl_units": "fixed episode initial spot/notional; lambda has no hidden basis-point divisor",
               "diagnostic_dates_not_performance_evidence": True}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "licensed-fit-directory", "raw-quote-directory", "market-directory", "daily-directory", "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(run_statistical_pilot(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
