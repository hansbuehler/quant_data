"""Historical one-session hedging backtest for ATM one-day SPX options."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.verbose import Context

from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.rolling_hedging import (
    mean_variance_hedge,
    oce_hedge,
    one_session_pairs,
    paired_block_bootstrap_improvement,
    summarize_hedge_estimates,
    summarize_realized_pnl,
)
from quantdata.widb.day import enrich_day


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 8, 29))
    parser.add_argument("--minimum-scenarios", type=int, default=250)
    parser.add_argument("--risk-aversion", type=float, default=50.0)
    return parser.parse_args()


def _atm_one_day_row(frame: Any) -> Any:
    valid_books = np.ones(len(frame), dtype=np.bool_)
    for side in ("call", "put"):
        bid = frame[f"mkt_{side}_bid"].to_numpy(dtype=np.float64)
        ask = frame[f"mkt_{side}_ask"].to_numpy(dtype=np.float64)
        valid_books &= np.isfinite(bid) & np.isfinite(ask) & (bid >= 0.0) & (ask > bid)
    eligible = frame.loc[
        (~frame["is_am"])
        & (~frame["is_american"])
        & np.isclose(frame["bdte"], 1.0)
        & valid_books
    ]
    if eligible.empty:
        raise ValueError("market day has no European PM one-day expiry")
    location = np.argmin(np.abs(eligible["pure_strike"].to_numpy() - 1.0))
    return eligible.iloc[int(location)]


def main() -> None:
    arguments = _arguments()
    if arguments.minimum_scenarios < 3:
        raise ValueError("at least three historical scenarios are required")
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(arguments.checkpoint)
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    spots = history.get("spot_refs")
    if not isinstance(dates, list):
        raise ValueError("invalid target-history payload")
    if not isinstance(spots, torch.Tensor):
        raise ValueError("target history has no spot references")
    spot_values = spots.double().numpy()
    pair_origin, pair_endpoint = one_session_pairs(dates)
    historical_spot_gross = spot_values[pair_endpoint] / spot_values[pair_origin]
    rows: list[dict[str, object]] = []
    fitted_estimates: dict[str, dict[str, list[Any]]] = {
        f"{position}_{option}": {
            method: []
            for method in ("mean_variance", "entropy", "power-exp")
        }
        for option in ("call", "put")
        for position in ("buyer", "seller")
    }
    for index in range(len(dates) - 1):
        current_date = date.fromisoformat(dates[index])
        next_date = date.fromisoformat(dates[index + 1])
        if current_date < arguments.start or current_date > arguments.end:
            continue
        matching_pair = np.flatnonzero(
            (pair_origin == index) & (pair_endpoint == index + 1)
        )
        if len(matching_pair) != 1:
            continue
        available = pair_endpoint <= index
        if int(np.sum(available)) < arguments.minimum_scenarios:
            continue
        # A one-day contract must mature at the next retained market close.
        market = enrich_day("SPX", current_date, verbose=Context.quiet)
        try:
            option = _atm_one_day_row(market)
        except ValueError:
            continue
        expiry = np.datetime64(option.expiry, "D")
        if expiry != np.datetime64(next_date):
            continue
        spot = float(spot_values[index])
        strike = float(option.strike) / spot
        call_mid = 0.5 * (float(option.mkt_call_bid) + float(option.mkt_call_ask)) / spot
        put_mid = 0.5 * (float(option.mkt_put_bid) + float(option.mkt_put_ask)) / spot
        gross_scenarios = historical_spot_gross[available]
        centered_spot_change = gross_scenarios - np.mean(gross_scenarios)
        actual_gross = float(spot_values[index + 1] / spot_values[index])
        scenario_payoffs = {
            "call": np.maximum(gross_scenarios - strike, 0.0),
            "put": np.maximum(strike - gross_scenarios, 0.0),
        }
        actual_payoffs = {
            "call": max(actual_gross - strike, 0.0),
            "put": max(strike - actual_gross, 0.0),
        }
        mids = {"call": call_mid, "put": put_mid}
        row: dict[str, object] = {
            "date": dates[index],
            "next_date": dates[index + 1],
            "strike_over_spot": strike,
            "scenarios": len(gross_scenarios),
            "actual_spot_change": actual_gross - 1.0,
        }
        for option_type in ("call", "put"):
            for position_name, position_sign in (("buyer", 1.0), ("seller", -1.0)):
                label = f"{position_name}_{option_type}"
                scenario_change = position_sign * (
                    scenario_payoffs[option_type] - mids[option_type]
                )
                actual_unhedged = position_sign * (actual_payoffs[option_type] - mids[option_type])
                estimates = {
                    "mean_variance": mean_variance_hedge(scenario_change, centered_spot_change)
                }
                for utility in ("entropy", "power-exp"):
                    estimates[utility] = oce_hedge(
                        scenario_change,
                        centered_spot_change,
                        utility=utility,
                        risk_aversion=arguments.risk_aversion,
                        initial_spot_position=(estimates["mean_variance"].spot_position),
                    )
                result: dict[str, object] = {
                    "initial_mid_over_spot": mids[option_type],
                    "realized_unhedged_pnl": actual_unhedged,
                }
                for method, estimate in estimates.items():
                    fitted_estimates[label][method].append(estimate)
                    result[method] = estimate.to_dict()
                    result[f"realized_{method}_pnl"] = actual_unhedged + estimate.spot_position * (
                        actual_gross - 1.0
                    )
                row[label] = result
        rows.append(row)
        print(f"ATM 1D hedge {dates[index]}->{dates[index + 1]}", flush=True)
    if not rows:
        raise RuntimeError("no one-day ATM hedge observations were produced")
    summary: dict[str, object] = {}
    for option_type in ("call", "put"):
        for position_name in ("buyer", "seller"):
            label = f"{position_name}_{option_type}"
            label_summary = {
                "unhedged": summarize_realized_pnl(
                    np.asarray([row[label]["realized_unhedged_pnl"] for row in rows])
                )
            }
            unhedged = np.asarray([row[label]["realized_unhedged_pnl"] for row in rows])
            for method in ("mean_variance", "entropy", "power-exp"):
                candidate = np.asarray([row[label][f"realized_{method}_pnl"] for row in rows])
                label_summary[method] = summarize_realized_pnl(candidate)
                label_summary[method]["fit_audit"] = summarize_hedge_estimates(
                    fitted_estimates[label][method]
                )
                if len(rows) >= 3:
                    label_summary[method]["versus_unhedged"] = paired_block_bootstrap_improvement(
                        unhedged, candidate
                    )
            summary[label] = label_summary
    report = {
        "schema": "dysanos-atm-one-day-hedging-v1",
        "status": "complete",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "date_range": [arguments.start.isoformat(), arguments.end.isoformat()],
        "risk_aversion": arguments.risk_aversion,
        "minimum_scenarios_per_date": arguments.minimum_scenarios,
        "scenario_definition": (
            "All adjacent full-session spot returns whose endpoint is known by "
            "close t; forward changes are sample-centered for hedge fitting."
        ),
        "realized_payoff_proxy": (
            "Next retained close spot is used as the one-day PM settlement proxy."
        ),
        "costs": "Initial marks are bid/ask midpoints; hedge transaction costs are omitted.",
        "evaluation_scope": (
            "Chronological historical-scenario hedge evaluation. Initial option "
            "marks and realized payoffs are market observations; the Direct-200 "
            "checkpoint is recorded for experiment identity but does not determine "
            "a one-day option's terminal cash payoff."
        ),
        "summary": summary,
        "rows": rows,
    }
    write_json_atomic(report, arguments.output)
    print(json.dumps({"output": str(arguments.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
