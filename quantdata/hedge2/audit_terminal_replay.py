"""Independently recompute actual-market cashflows of completed terminal-policy episodes."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import get_business_days
from quantdata.hedge2.cash import MarketStore, observed_marks


def audit_terminal_replay(*, directory: Path, store: MarketStore) -> dict[str, object]:
    """Audit raw cash P&L, not the fitted pricing law or statistical performance."""
    configuration = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("A partial terminal episode cannot be presented as completed cash evidence")
    original = configuration["claim"]
    horizon = original["terminal_day"]
    origin = date.fromisoformat(configuration["origin"])
    sessions = get_business_days(origin, origin + timedelta(days=15))[:horizon + 1]
    book = pd.read_parquet(directory / "initial_book.parquet")
    trades = pd.read_parquet(directory / "trades.parquet")
    previous = np.zeros(1 + len(book))
    discount, gain_total, cost_total, exit_total, maximum_error = 1., 0., 0., 0., 0.
    observed_spot, knocked_out = None, False
    for elapsed, row in enumerate(trades.itertuples()):
        if elapsed != row.elapsed_day or row.date != sessions[elapsed].isoformat() or knocked_out:
            raise ValueError("Terminal trades must be consecutive and stop after actual knockout")
        current, following = store.load(sessions[elapsed]), store.load(sessions[elapsed + 1])
        active = np.flatnonzero(book.bdte.to_numpy(dtype=float) > elapsed + 1e-6)
        columns = np.r_[0, 1 + active]
        marks, spreads = observed_marks(book.iloc[active], current)
        next_marks, next_spreads = observed_marks(book.iloc[active], following)
        _, reference, one_day_discount = current.curve.at(1.)
        raw = discount * np.r_[one_day_discount * (following.curve.spot - reference),
            one_day_discount * next_marks - marks] / original["initial_spot"]
        costs = discount * np.r_[0., spreads] / original["initial_spot"]
        with np.load(directory / f"step-{elapsed:02d}" / "realized.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["columns"], columns):
                raise ValueError("Terminal replay changed the fixed contract columns")
            position = saved["positions"]
            if position.shape != columns.shape or not np.isfinite(position).all():
                raise ValueError("Saved terminal holdings must be finite and match the fixed contracts")
            gain = float(position @ raw)
            fee = float(costs @ np.abs(position - previous[columns]))
            next_discount = discount * float(one_day_discount)
            exit_metric = float(next_discount * np.abs(position[1:]) @ next_spreads / original["initial_spot"])
            if original["kind"] != "digital" and elapsed + 1 == original["observation_day"]:
                observed_spot = following.curve.spot
                if original["kind"] == "ko":
                    knocked_out = observed_spot <= original["barrier"] if original["is_call"] else observed_spot >= original["barrier"]
            after = np.zeros_like(previous)
            after[columns] = position
            after[1 + np.flatnonzero(book.bdte.to_numpy(dtype=float) <= elapsed + 1 + 1e-6)] = 0.
            if knocked_out or elapsed + 1 == horizon:
                after[:] = 0.
            errors = (np.max(np.abs(saved["realized_raw_gains"] - raw)), np.max(np.abs(saved["costs"] - costs)),
                np.max(np.abs(saved["previous_positions"] - previous[columns])), np.max(np.abs(saved["holdings_after"] - after)),
                abs(gain - row.hedge_gain), abs(fee - row.entry_cost), abs(exit_metric - row.hypothetical_exit_cost))
            maximum_error = max(maximum_error, *(float(error) for error in errors))
            previous = after
        discount = next_discount
        gain_total += gain
        cost_total += fee
        exit_total += exit_metric
    if not len(trades) or len(trades) != horizon and not knocked_out:
        raise ValueError("A completed terminal episode needs settlement or observed knockout")
    terminal_spot = store.load(sessions[len(trades)]).curve.spot
    if original["kind"] == "digital":
        payoff = float(terminal_spot > original["strike"] if original["is_call"] else terminal_spot <= original["strike"])
    elif original["kind"] == "ko":
        payoff = (0. if knocked_out else max(terminal_spot - original["strike"] if original["is_call"]
            else original["strike"] - terminal_spot, 0.) / original["initial_spot"])
    else:
        if observed_spot is None:
            raise ValueError("The ratio straddle lacks its actual reset observation")
        payoff = abs(terminal_spot / observed_spot - original["strike"])
    unhedged = configuration["target_side"] * (discount * payoff - configuration["initial_price"])
    expected = unhedged + gain_total - cost_total
    maximum_error = max(maximum_error, abs(payoff - summary["terminal_payoff"]),
        abs(unhedged - summary["unhedged_pnl"]), abs(expected - summary["costed_pnl"]),
        abs(gain_total - summary["hedge_gain"]), abs(cost_total - summary["entry_cost"]),
        abs(exit_total - summary["hypothetical_exit_metric"]))
    if not np.isfinite(maximum_error) or maximum_error > 1e-10:
        raise RuntimeError(f"Independent actual-market terminal cash replay failed: maximum_error={maximum_error}")
    return {"status": "passed", "maximum_cash_error": maximum_error, "completed_steps": len(trades),
        "raw_realized_gains": True, "entry_only_costs": True, "no_terminal_or_KO_exit_fee": True,
        "diagnostic_only": summary["diagnostic_only"], "not_a_pricing_or_generalization_test": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    arguments = parser.parse_args()
    sources = json.loads((arguments.input_directory / "config.json").read_text())["market_sources"]
    store = MarketStore(**{name: Path(path) for name, path in sources.items()})
    result = audit_terminal_replay(directory=arguments.input_directory, store=store)
    (arguments.input_directory / "cash_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
