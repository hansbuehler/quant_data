"""Certify saved statistical candidates, refine high-gap power cells, and solve pure variance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.cash import MarketStore, discounted_daily_gains, observed_marks
from quantdata.hedge2.dual import certify_power
from quantdata.hedge2.objectives import power_oce
from quantdata.hedge2.optimization import ScenarioOptimizer
from quantdata.hedge2.polish import polish_power


def certify(*, input_directory: Path, output_directory: Path, market_directory: Path,
            licensed_fit_directory: Path, raw_quote_directory: Path,
            objective_gap_tolerance: float = 1e-4) -> dict[str, object]:
    if not np.isfinite(objective_gap_tolerance) or objective_gap_tolerance <= 0:
        raise ValueError("The objective-unit certificate tolerance must be finite and positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parent
    configuration = {"input_directory": str(input_directory), "market_directory": str(market_directory),
                     "licensed_fit_directory": str(licensed_fit_directory), "raw_quote_directory": str(raw_quote_directory),
                     "objective_gap_tolerance": objective_gap_tolerance,
                     "implementation_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                        for name in ("objectives.py", "optimization.py", "polish.py", "dual.py", "certify_pilot.py")}}
    (output_directory / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    all_rows = []
    started = time.monotonic()
    for input_file in sorted(input_directory.glob("*/results.parquet")):
        day = date.fromisoformat(input_file.parent.name)
        frame = pd.read_parquet(input_file)
        with np.load(input_file.with_name("positions.npz"), allow_pickle=False) as saved:
            positions = saved["positions"]
        with np.load(input_file.with_name("scenarios.npz"), allow_pickle=False) as saved:
            gains, target_gains, weights = saved["instrument_gains"], saved["claim_gains"], saved["weights"]
        book = pd.read_parquet(input_file.with_name("book.parquet"))
        market = store.load(day)
        following = store.load(get_business_days(day, day + timedelta(days=10))[1])
        next_marks, exit_spreads = observed_marks(book, following)
        realized = discounted_daily_gains(previous_option_marks=book["observed_mid"].to_numpy(dtype=float),
            next_option_marks=next_marks, next_spot=following.curve.spot, previous_curve=market.curve,
            episode_initial_spot=market.curve.spot)
        _, _, discount = market.curve.at(1.)
        costs = np.r_[0., book["observed_half_spread"].to_numpy(dtype=float) / market.curve.spot]
        optimizer = ScenarioOptimizer(instrument_gains=gains, weights=weights, costs=costs)
        labels = list(dict.fromkeys(frame["claim"]))
        rows, holdings = [], []
        for index, original in frame.iterrows():
            row = original.to_dict()
            position = positions[index]
            claim_gains = target_gains[:, labels.index(row["claim"])]
            original_target_gain = row["costed_pnl"] - float(position @ realized) + row["entry_cost"]
            row["original_status"] = row["status"]
            row["original_gap"] = row["convex_first_order_gap_bound"]
            row["polished"] = False
            if row["objective"] == "mean_variance" and np.isinf(row["risk_aversion"]):
                hedge = optimizer.solve(claim_gains=claim_gains, objective="mean_variance",
                                         risk_aversion=np.inf, target_side=row["side"])
                position = hedge.positions
                row.update(solver=hedge.solver, status=hedge.status,
                           convex_first_order_gap_bound=hedge.convex_first_order_gap_bound,
                           warnings=json.dumps([*json.loads(row["warnings"]), *hedge.warnings]))
                row["minimum_variance_is_cost_free"] = True
            elif row["objective"] == "power_exponential" and row["original_gap"] > objective_gap_tolerance:
                refined = polish_power(instrument_gains=gains, claim_gains=claim_gains,
                    weights=weights, costs=costs, initial_positions=position,
                    risk_aversion=row["risk_aversion"], target_side=row["side"])
                position = refined.positions
                row.update(polished=True, solver="SLSQP direct cash-eliminated refinement",
                           status=refined.solver_message, convex_first_order_gap_bound=refined.after_gap,
                           objective_improvement=refined.before_objective - refined.after_objective,
                           polish_iterations=refined.iterations)
            projected = np.clip(position, -1000., 1000.)
            row["cap_roundoff_projection"] = float(np.max(np.abs(projected - position), initial=0.))
            position = projected
            wealth = gains @ position + row["side"] * claim_gains
            mean = float(weights @ wealth)
            variance = float(weights @ (wealth - mean)**2)
            cost = float(costs @ np.abs(position))
            row.update(training_mean=mean - cost, training_std=np.sqrt(variance), entry_cost=cost,
                       costed_pnl=original_target_gain + float(position @ realized) - cost,
                       gross_positions=float(np.abs(position).sum()),
                       cap_hits=int(np.sum(np.abs(position) >= 1000. * (1. - 1e-6))),
                       hypothetical_exit_cost=float(discount * np.abs(position[1:]) @ exit_spreads / market.curve.spot))
            if row["objective"] == "power_exponential":
                oce = power_oce(wealth, weights, risk_aversion=row["risk_aversion"])
                row["training_objective"] = -oce.certainty_equivalent + cost
                row["cash_first_order_residual"] = oce.cash_first_order_residual
                gradient = -gains.T @ (weights * oce.marginal_utility)
                penalty = costs
            elif np.isinf(row["risk_aversion"]):
                row["training_objective"] = .5 * variance
                gradient = gains.T @ (weights * (wealth - mean))
                penalty = np.zeros_like(costs)
            else:
                row["training_objective"] = -mean + .5 * row["risk_aversion"] * variance + cost
                gradient = row["risk_aversion"] * gains.T @ (weights * (wealth - mean))
                penalty = costs
            choices = np.array([-np.full_like(position, 1000.), np.zeros_like(position), np.full_like(position, 1000.)])
            tangent_gap = np.sum(position * gradient + penalty * np.abs(position)
                                 - np.min(choices * gradient + penalty * np.abs(choices), axis=0))
            row["convex_first_order_gap_bound"] = max(float(tangent_gap), 0.)
            row["certified_objective_gap"] = row["convex_first_order_gap_bound"]
            if row["objective"] == "power_exponential" and row["certified_objective_gap"] > objective_gap_tolerance:
                dual = certify_power(instrument_gains=gains, claim_gains=claim_gains,
                    weights=weights, costs=costs, positions=position, risk_aversion=row["risk_aversion"],
                    target_side=row["side"], gap_tolerance=objective_gap_tolerance)
                row.update(dual_lower_bound=dual.dual_lower_bound, dual_objective_gap=dual.objective_gap,
                           dual_solver=dual.solver, dual_status=dual.status, dual_warnings=json.dumps(dual.warnings))
                row["certified_objective_gap"] = min(row["certified_objective_gap"], dual.objective_gap)
            row["objective_gap_tolerance"] = objective_gap_tolerance
            row["optimization_certified"] = bool(row["certified_objective_gap"] <= objective_gap_tolerance)
            rows.append(row)
            holdings.append(position)
            if (index + 1) % 30 == 0:
                print(f"certification date={day} cells={index+1}/{len(frame)} "
                      f"certified={sum(item['optimization_certified'] for item in rows)} "
                      f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
        day_output = output_directory / day.isoformat()
        day_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(day_output / "results.parquet", index=False)
        np.savez_compressed(day_output / "positions.npz", positions=np.asarray(holdings))
        all_rows.extend(rows)
    final = pd.DataFrame(all_rows)
    final.to_parquet(output_directory / "results.parquet", index=False)
    summary = {"cells": len(final), "certified_cells": int(final["optimization_certified"].sum()),
               "uncertified_cells": int((~final["optimization_certified"]).sum()),
               "polished_cells": int(final["polished"].sum()),
               "maximum_gap": float(final["certified_objective_gap"].max()),
               "elapsed_seconds": time.monotonic() - started,
               "risk_aversion_infinity": "exact cost-free variance objective, actual trading costs still charged in realized P&L"}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-directory", "output-directory", "market-directory", "licensed-fit-directory", "raw-quote-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(certify(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
