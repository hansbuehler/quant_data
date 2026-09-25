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

from quantdata.dysanos.flex.front_spot_copula import (
    antithetic_normals,
    fit_front_spot_copula_from_history,
    sample_correlated_front_scenarios,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest, write_json_atomic
from quantdata.dysanos.flex.presentation_hedging import (
    exact_daily_transition,
    load_presentation_model,
)
from quantdata.dysanos.flex.rolling_hedging import (
    mean_variance_hedge,
    oce_hedge,
    one_session_pairs,
    paired_block_bootstrap_improvement,
    paired_block_bootstrap_oce_improvement,
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
    parser.add_argument("--num-scenarios", type=int, default=2_048)
    parser.add_argument("--risk-aversion", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=91_733)
    parser.add_argument("--device", default="cuda:0")
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
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    del env
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_tensor = history.get("states")
    spots = history.get("spot_refs")
    if not isinstance(dates, list) or not isinstance(states_tensor, torch.Tensor):
        raise ValueError("invalid target-history payload")
    if not isinstance(spots, torch.Tensor):
        raise ValueError("target history has no spot references")
    spot_values = spots.double().numpy()
    states = states_tensor.double().numpy()
    daily_transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    daily_transition_np = daily_transition.cpu().double().numpy()
    daily_factor_np = daily_factor.cpu().double().numpy()
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=spot_values,
        dates=dates,
        daily_transition=daily_transition_np,
        daily_factor=daily_factor_np,
        fit_before=arguments.start,
    )
    if copula_fit.observations < arguments.minimum_scenarios:
        raise ValueError("spot-copula estimation has too few pre-evaluation transitions")
    scenario_state_normals, scenario_spot_normals = antithetic_normals(
        count=arguments.num_scenarios,
        dimension=model.config.state_size,
        seed=arguments.seed,
    )
    pair_origin, pair_endpoint = one_session_pairs(dates)
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
        entry_quotes = {
            "call": {
                "bid": float(option.mkt_call_bid) / spot,
                "ask": float(option.mkt_call_ask) / spot,
            },
            "put": {
                "bid": float(option.mkt_put_bid) / spot,
                "ask": float(option.mkt_put_ask) / spot,
            },
        }
        _, gross_scenarios = sample_correlated_front_scenarios(
            model=model,
            current_state=states[index],
            daily_transition=daily_transition_np,
            daily_factor=daily_factor_np,
            fit=copula_fit,
            state_normals=scenario_state_normals,
            independent_spot_normals=scenario_spot_normals,
        )
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
        row: dict[str, object] = {
            "date": dates[index],
            "next_date": dates[index + 1],
            "strike_over_spot": strike,
            "scenarios": arguments.num_scenarios,
            "actual_spot_change": actual_gross - 1.0,
        }
        for option_type in ("call", "put"):
            for position_name, position_sign in (("buyer", 1.0), ("seller", -1.0)):
                label = f"{position_name}_{option_type}"
                entry_side = "ask" if position_sign > 0.0 else "bid"
                entry_price = entry_quotes[option_type][entry_side]
                scenario_change = position_sign * (
                    scenario_payoffs[option_type] - entry_price
                )
                actual_unhedged = position_sign * (
                    actual_payoffs[option_type] - entry_price
                )
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
                    "entry_side": entry_side,
                    "entry_price_over_spot": entry_price,
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
                    if method in {"entropy", "power-exp"}:
                        label_summary[method]["oce_versus_unhedged"] = (
                            paired_block_bootstrap_oce_improvement(
                                unhedged,
                                candidate,
                                utility=method,
                                risk_aversion=arguments.risk_aversion,
                            )
                        )
            summary[label] = label_summary
    report = {
        "schema": "dysanos-atm-one-day-hedging-v2",
        "status": "complete",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "date_range": [arguments.start.isoformat(), arguments.end.isoformat()],
        "risk_aversion": arguments.risk_aversion,
        "minimum_copula_fit_observations": arguments.minimum_scenarios,
        "num_model_scenarios_per_date": arguments.num_scenarios,
        "seed": arguments.seed,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "transition_diagnostics": transition_diagnostics,
        "scenario_definition": (
            "Antithetic Gaussian Direct-200 state shocks. The contemporaneous "
            "spot Gaussian score uses the pre-2025 ridge-estimated state copula "
            "and is mapped through a monotone PCHIP inverse of the current "
            "front-expiry SANOS CDF. Spot changes are sample-centered for fitting."
        ),
        "realized_payoff_proxy": (
            "Next retained close spot is used as the one-day PM settlement proxy."
        ),
        "costs": (
            "Option buyers enter at the ask and sellers at the bid; the one-day "
            "contract settles at payoff. Spot-hedge transaction costs are omitted."
        ),
        "evaluation_scope": (
            "Chronological model-scenario hedge evaluation against next-session "
            "market outcomes. Entry quotes and realized payoffs are observed; "
            "the fitted model determines the hedge scenarios."
        ),
        "summary": summary,
        "rows": rows,
    }
    write_json_atomic(report, arguments.output)
    print(json.dumps({"output": str(arguments.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
