"""Backtest one-session hedges of an after-close open-interest proxy."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cdxcore.verbose import Context

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.front_spot_copula import (
    antithetic_normals,
    fit_front_spot_copula_from_history,
    sample_correlated_front_scenarios,
)
from quantdata.dysanos.flex.market_encoder import (
    checkpoint_digest,
    write_json_atomic,
)
from quantdata.dysanos.flex.presentation_hedging import (
    _decode_surface,
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
    parser.add_argument("--risk-aversion", type=float, default=50.0)
    parser.add_argument("--minimum-scenarios", type=int, default=250)
    parser.add_argument("--num-scenarios", type=int, default=2_048)
    parser.add_argument("--scenario-batch-size", type=int, default=32)
    parser.add_argument("--instrument-batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=91_733)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _mid_map(frame: Any, side: str) -> dict[int, float]:
    identifier = frame[f"{side}_opt_id"].to_numpy(dtype=np.int64)
    bid = frame[f"mkt_{side}_bid"].to_numpy(dtype=np.float64)
    ask = frame[f"mkt_{side}_ask"].to_numpy(dtype=np.float64)
    return {
        int(option_id): float(0.5 * (bid_value + ask_value))
        for option_id, bid_value, ask_value in zip(identifier, bid, ask, strict=True)
        if np.isfinite(bid_value)
        and np.isfinite(ask_value)
        and 0.0 <= bid_value < ask_value
    }


def _after_close_oi_book(
    current: Any,
    following: Any,
    *,
    spot: float,
) -> dict[str, np.ndarray | float | int]:
    """Build the dealer-short proxy from close-t OI and match close-(t+1) mids."""
    next_mid = {side: _mid_map(following, side) for side in ("call", "put")}
    strikes: list[float] = []
    remaining_days: list[float] = []
    is_call: list[bool] = []
    open_interest: list[float] = []
    initial_bid: list[float] = []
    realized_mid: list[float] = []
    for row in current.itertuples(index=False):
        if bool(row.is_am) or bool(row.is_american):
            continue
        if not 2.0 <= float(row.bdte) <= 70.0:
            continue
        normalized_strike = float(row.strike) / spot
        if not 0.75 <= normalized_strike <= 1.25:
            continue
        for side, call_flag in (("call", True), ("put", False)):
            option_id = int(getattr(row, f"{side}_opt_id"))
            if option_id not in next_mid[side]:
                continue
            log_oi = float(getattr(row, f"mkt_{side}_log_open_interest"))
            if not np.isfinite(log_oi):
                continue
            oi = max(10.0**log_oi - 1.0, 0.0)
            if oi < 1.0:
                continue
            bid = float(getattr(row, f"mkt_{side}_bid"))
            ask = float(getattr(row, f"mkt_{side}_ask"))
            if not (np.isfinite(bid) and np.isfinite(ask) and 0.0 <= bid < ask):
                continue
            if bid <= 0.0:
                continue
            strikes.append(normalized_strike)
            remaining_days.append(float(row.bdte) - 1.0)
            is_call.append(call_flag)
            open_interest.append(oi)
            initial_bid.append(bid / spot)
            realized_mid.append(next_mid[side][option_id] / spot)
    if not strikes:
        raise ValueError("no OI-weighted options survived the one-session match")
    oi_values = np.asarray(open_interest, dtype=np.float64)
    weights = -oi_values / np.sum(oi_values)
    current_values = np.asarray(initial_bid, dtype=np.float64)
    next_values = np.asarray(realized_mid, dtype=np.float64)
    return {
        "strikes": np.asarray(strikes, dtype=np.float64),
        "remaining_days": np.asarray(remaining_days, dtype=np.float64),
        "is_call": np.asarray(is_call, dtype=np.bool_),
        "weights": weights,
        "initial_bid": current_values,
        "realized_unhedged_pnl": float(weights @ (next_values - current_values)),
        "gross_open_interest": float(np.sum(oi_values)),
        "options": len(strikes),
    }


def _scenario_portfolio_change(
    *,
    model: Any,
    future_states: np.ndarray,
    future_spots: np.ndarray,
    book: dict[str, np.ndarray | float | int],
    batch_size: int,
    instrument_batch_size: int,
) -> np.ndarray:
    strikes_np = np.asarray(book["strikes"])
    remaining_np = np.asarray(book["remaining_days"])
    calls_np = np.asarray(book["is_call"])
    weights_np = np.asarray(book["weights"])
    initial_np = np.asarray(book["initial_bid"])
    output = np.empty(len(future_states), dtype=np.float64)
    dtype, device = model.fdtype, model.device
    strikes = torch.as_tensor(strikes_np, dtype=dtype, device=device)
    remaining = torch.as_tensor(remaining_np, dtype=dtype, device=device)
    call_mask = torch.as_tensor(calls_np, dtype=torch.bool, device=device)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    initial = torch.as_tensor(initial_np, dtype=dtype, device=device)
    with torch.inference_mode():
        for first in range(0, len(future_states), batch_size):
            last = min(first + batch_size, len(future_states))
            states = torch.as_tensor(future_states[first:last], dtype=dtype, device=device)
            spots = torch.as_tensor(future_spots[first:last], dtype=dtype, device=device)
            surface = _decode_surface(model, states)
            batch_change = torch.zeros(last - first, dtype=dtype, device=device)
            for option_first in range(0, len(strikes), instrument_batch_size):
                option_last = min(option_first + instrument_batch_size, len(strikes))
                option_strikes = strikes[option_first:option_last]
                normalized_strikes = option_strikes[None, :] / spots[:, None]
                expiries = remaining[option_first:option_last][None, :].expand_as(
                    normalized_strikes
                ) / DAYS_PER_YEAR
                calls = spots[:, None] * model.sanos_decoder.price_calls_batch(
                    surface,
                    expiries_y=expiries,
                    strikes=normalized_strikes,
                )
                puts = calls - spots[:, None] + option_strikes[None, :]
                values = torch.where(
                    call_mask[None, option_first:option_last], calls, puts
                )
                batch_change += (
                    values - initial[None, option_first:option_last]
                ) @ weights[option_first:option_last]
            output[first:last] = batch_change.cpu().numpy()
    return output


def main() -> None:
    arguments = _arguments()
    if arguments.scenario_batch_size <= 0 or arguments.instrument_batch_size <= 0:
        raise ValueError("scenario and instrument batch sizes must be positive")
    env, model = load_presentation_model(arguments.checkpoint, device=arguments.device)
    del env
    history = torch.load(arguments.target_history, map_location="cpu", weights_only=False)
    dates = history.get("dates")
    states_tensor = history.get("states")
    spots_tensor = history.get("spot_refs")
    if not isinstance(dates, list) or not isinstance(states_tensor, torch.Tensor):
        raise ValueError("invalid target-history payload")
    if not isinstance(spots_tensor, torch.Tensor):
        raise ValueError("target history has no spot references")
    states = states_tensor.double().numpy()
    spots = spots_tensor.double().numpy()
    transition, daily_factor, transition_diagnostics = exact_daily_transition(model)
    transition_np = transition.cpu().double().numpy()
    daily_factor_np = daily_factor.cpu().double().numpy()
    pair_origin, pair_endpoint = one_session_pairs(dates)
    copula_fit, copula_diagnostics = fit_front_spot_copula_from_history(
        model=model,
        states=states,
        spots=spots,
        dates=dates,
        daily_transition=transition_np,
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

    rows: list[dict[str, object]] = []
    fitted_estimates: dict[str, list[Any]] = {
        method: [] for method in ("mean_variance", "entropy", "power-exp")
    }
    for current_index in range(len(dates) - 1):
        current_date = date.fromisoformat(dates[current_index])
        next_date = date.fromisoformat(dates[current_index + 1])
        if current_date < arguments.start or current_date > arguments.end:
            continue
        matching_pair = np.flatnonzero(
            (pair_origin == current_index) & (pair_endpoint == current_index + 1)
        )
        if len(matching_pair) != 1:
            continue
        current_market = enrich_day("SPX", current_date, verbose=Context.quiet)
        next_market = enrich_day("SPX", next_date, verbose=Context.quiet)
        book = _after_close_oi_book(
            current_market,
            next_market,
            spot=float(spots[current_index]),
        )
        future_states, future_spots = sample_correlated_front_scenarios(
            model=model,
            current_state=states[current_index],
            daily_transition=transition_np,
            daily_factor=daily_factor_np,
            fit=copula_fit,
            state_normals=scenario_state_normals,
            independent_spot_normals=scenario_spot_normals,
        )
        portfolio_change = _scenario_portfolio_change(
            model=model,
            future_states=future_states,
            future_spots=future_spots,
            book=book,
            batch_size=arguments.scenario_batch_size,
            instrument_batch_size=arguments.instrument_batch_size,
        )
        scenario_spot_change = future_spots - 1.0
        centered_spot_change = scenario_spot_change - np.mean(scenario_spot_change)
        estimates = {
            "mean_variance": mean_variance_hedge(portfolio_change, centered_spot_change),
        }
        for utility in ("entropy", "power-exp"):
            estimates[utility] = oce_hedge(
                portfolio_change,
                centered_spot_change,
                utility=utility,
                risk_aversion=arguments.risk_aversion,
                initial_spot_position=estimates["mean_variance"].spot_position,
            )
        actual_spot_change = float(spots[current_index + 1] / spots[current_index] - 1.0)
        unhedged = float(book["realized_unhedged_pnl"])
        row: dict[str, object] = {
            "date": dates[current_index],
            "next_date": dates[current_index + 1],
            "scenarios": arguments.num_scenarios,
            "options": int(book["options"]),
            "gross_open_interest": float(book["gross_open_interest"]),
            "actual_spot_change": actual_spot_change,
            "realized_unhedged_pnl": unhedged,
            "scenario_unhedged_mean": float(np.mean(portfolio_change)),
            "scenario_unhedged_standard_deviation": float(np.std(portfolio_change, ddof=1)),
        }
        for name, estimate in estimates.items():
            fitted_estimates[name].append(estimate)
            row[name] = estimate.to_dict()
            row[f"realized_{name}_pnl"] = unhedged + estimate.spot_position * actual_spot_change
        rows.append(row)
        print(
            f"OI hedge {dates[current_index]}->{dates[current_index + 1]} "
            f"options={book['options']} scenarios={arguments.num_scenarios}",
            flush=True,
        )
    if not rows:
        raise RuntimeError("no chronological OI hedge observations were produced")
    option_counts = np.asarray([row["options"] for row in rows], dtype=np.int64)
    gross_open_interest = np.asarray(
        [row["gross_open_interest"] for row in rows], dtype=np.float64
    )
    summary = {
        "unhedged": summarize_realized_pnl(
            np.asarray([row["realized_unhedged_pnl"] for row in rows])
        )
    }
    for name in ("mean_variance", "entropy", "power-exp"):
        candidate = np.asarray([row[f"realized_{name}_pnl"] for row in rows])
        summary[name] = summarize_realized_pnl(candidate)
        summary[name]["fit_audit"] = summarize_hedge_estimates(
            fitted_estimates[name]
        )
        if len(rows) >= 3:
            summary[name]["versus_unhedged"] = paired_block_bootstrap_improvement(
                np.asarray([row["realized_unhedged_pnl"] for row in rows]),
                candidate,
            )
            if name in {"entropy", "power-exp"}:
                summary[name]["oce_versus_unhedged"] = (
                    paired_block_bootstrap_oce_improvement(
                        np.asarray([row["realized_unhedged_pnl"] for row in rows]),
                        candidate,
                        utility=name,
                        risk_aversion=arguments.risk_aversion,
                    )
                )
    report = {
        "schema": "dysanos-after-close-open-interest-hedging-v2",
        "status": "complete",
        "convention": {
            "information_timing": (
                "The date-t open-interest field is treated as an after-close "
                "proxy and assigned only to the portfolio hedged from close t "
                "to close t+1; its publication timing is not independently "
                "verified."
            ),
            "position_sign": (
                "Dealer short every retained call and put; OI is unsigned, so "
                "this is a transparent proxy and not observed net dealer inventory."
            ),
            "normalization": "Positions sum to minus one unit of gross open interest.",
            "realized_valuation": (
                "The dealer-short proxy is formed at the date-t option bids and "
                "the surviving positions are marked at date-(t+1) midpoints."
            ),
            "costs": (
                "The option entry half-spread is included through bid execution; "
                "next-close liquidation and spot-hedge transaction costs are omitted."
            ),
            "scenario_law": (
                "Antithetic Gaussian Direct-200 state shocks. The contemporaneous "
                "spot Gaussian score uses the pre-2025 ridge-estimated state copula "
                "and is mapped through a monotone PCHIP inverse of the current "
                "front-expiry SANOS CDF."
            ),
        },
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint),
        "target_history": str(arguments.target_history),
        "risk_aversion": arguments.risk_aversion,
        "minimum_copula_fit_observations": arguments.minimum_scenarios,
        "num_model_scenarios_per_date": arguments.num_scenarios,
        "scenario_batch_size": arguments.scenario_batch_size,
        "instrument_batch_size": arguments.instrument_batch_size,
        "seed": arguments.seed,
        "spot_copula_fit": copula_fit.to_dict(),
        "spot_copula_diagnostics": copula_diagnostics,
        "date_range": [arguments.start.isoformat(), arguments.end.isoformat()],
        "book_coverage": {
            "observations": len(rows),
            "minimum_contracts": int(np.min(option_counts)),
            "median_contracts": float(np.median(option_counts)),
            "maximum_contracts": int(np.max(option_counts)),
            "minimum_gross_open_interest": float(np.min(gross_open_interest)),
            "median_gross_open_interest": float(np.median(gross_open_interest)),
            "maximum_gross_open_interest": float(np.max(gross_open_interest)),
        },
        "transition_diagnostics": transition_diagnostics,
        "summary": summary,
        "rows": rows,
    }
    write_json_atomic(report, arguments.output)
    print(json.dumps({"output": str(arguments.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
