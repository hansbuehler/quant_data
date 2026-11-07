"""Independent trade/cash replay for saved fixed-contract multi-day episodes."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import MarketStore, observed_marks
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.source import SourceArchive


def check_premium_limits(*, holdings: np.ndarray, current_prices: np.ndarray,
                         initial_spot: float, methods: tuple[str, ...],
                         position_cap: float) -> np.ndarray:
    """Replay post-trade limits from actual marks, independently of optimizer units."""
    if (holdings.ndim != 2 or current_prices.shape != (holdings.shape[1],)
            or len(methods) != len(holdings) or initial_spot <= 0 or position_cap <= 0
            or not np.isfinite(initial_spot) or not np.isfinite(position_cap)
            or not np.isfinite(holdings).all() or not np.isfinite(current_prices).all()
            or np.any(current_prices < 0.)):
        raise ValueError("Premium audit needs aligned finite holdings, prices and positive scales")
    premium = holdings * current_prices[None, :] / initial_spot
    optimized = np.array([method.startswith("DLV ") for method in methods])
    if np.any(np.abs(premium[optimized]) > position_cap * (1. + 1e-8)):
        raise RuntimeError("Saved optimized episode violates its actual current-premium position limit")
    return np.where(optimized, np.sum(np.abs(premium) >= position_cap * (1. - 1e-6), axis=1), 0)


def audit_episodes(*, input_directory: Path, output_directory: Path) -> dict[str, object]:
    config = json.loads((input_directory / "config.json").read_text())
    store = MarketStore(market_directory=Path(config["market_directory"]),
        licensed_fit_directory=Path(config["licensed_fit_directory"]),
        raw_quote_directory=Path(config["raw_quote_directory"]))
    archive = SourceArchive(root=Path(config["source_root"]))
    records = []
    for path in sorted(input_directory.glob("*/*/summary.parquet")):
        origin = date.fromisoformat(path.parent.parent.name)
        terminal = pd.read_parquet(path)
        trades = pd.read_parquet(path.with_name("trades.parquet"))
        book = pd.read_parquet(path.with_name("initial_book.parquet"))
        end = date.fromisoformat(terminal["terminal_date"].iloc[0])
        sessions = get_business_days(origin, origin + timedelta(days=15))
        sessions = [day for day in sessions if day <= end]
        first = store.load(origin)
        source, _ = archive.load(origin)
        claims = {claim.label: claim for claim in make_claims(initial_spot=first.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})}
        positions = np.zeros((len(terminal), 1 + len(book)))
        total_gain = np.zeros(len(terminal))
        total_cost = np.zeros(len(terminal))
        discount = 1.
        terminal_exits = np.zeros(len(terminal))
        ended = np.zeros(len(terminal), dtype=bool)
        maximum_step_error = 0.
        cap_days = np.zeros(len(terminal), dtype=int)
        with np.load(path.with_name("positions.npz"), allow_pickle=False) as saved:
            for elapsed, day in enumerate(sessions[:-1]):
                current, following = store.load(day), store.load(sessions[elapsed + 1])
                active = np.flatnonzero(pd.to_datetime(book["expiry"]).dt.date.to_numpy() > day)
                columns = np.r_[0, active + 1]
                np.testing.assert_array_equal(saved[f"active_{elapsed}"], columns)
                holdings = saved[f"step_{elapsed}"]
                if holdings.shape != (len(terminal), len(columns)) or not np.isfinite(holdings).all():
                    raise ValueError("Saved episode holdings are non-finite or misaligned")
                current_marks, spreads = observed_marks(book.iloc[active], current)
                next_marks, next_spreads = observed_marks(book.iloc[active], following)
                _, reference, one_day_discount = current.curve.at(1.)
                gains = discount * np.r_[one_day_discount * (following.curve.spot - reference),
                    one_day_discount * next_marks - current_marks] / first.curve.spot
                costs = discount * np.abs(holdings[:, 1:] - positions[:, columns][:, 1:]) @ spreads / first.curve.spot
                hedge_gains = holdings @ gains
                daily = trades.loc[trades["elapsed_day"] == elapsed].reset_index(drop=True)
                if not np.isfinite(daily[["entry_cost", "hedge_gain"]].to_numpy()).all():
                    raise ValueError("Non-finite saved episode trade cash flows")
                if len(daily) != len(terminal) or not daily["claim"].equals(terminal["claim"]):
                    raise ValueError("Saved portfolio rows lost their target identities")
                if config.get("optimized_premium_cap") is not None:
                    cap_hits = check_premium_limits(holdings=holdings,
                        current_prices=np.r_[current.curve.spot, current_marks], initial_spot=first.curve.spot,
                        methods=tuple(daily["method"]), position_cap=config["optimized_premium_cap"])
                    np.testing.assert_array_equal(daily["cap_hits"], cap_hits)
                    cap_days += cap_hits > 0
                error = max(float(np.max(np.abs(daily["entry_cost"].to_numpy() - costs))),
                            float(np.max(np.abs(daily["hedge_gain"].to_numpy() - hedge_gains))))
                maximum_step_error = max(maximum_step_error, error)
                total_gain += hedge_gains
                total_cost += costs
                positions[:, columns] = holdings
                for portfolio, row in enumerate(terminal.itertuples()):
                    claim = claims[row.claim]
                    if claim.kind == "ko" and elapsed + 1 == claim.observation_day:
                        knocked_out = (following.curve.spot <= claim.barrier if claim.is_call
                                       else following.curve.spot >= claim.barrier)
                        if knocked_out:
                            terminal_exits[portfolio] = discount * one_day_discount * np.abs(holdings[portfolio, 1:]) @ next_spreads / first.curve.spot
                            positions[portfolio] = 0.
                            ended[portfolio] = True
                            if row.hedge_end_date != following.day.isoformat() or row.termination_reason != "knockout":
                                raise ValueError("Knocked-out claim did not terminate at its actual observation")
                    if elapsed + 1 == len(sessions) - 1 and not ended[portfolio]:
                        terminal_exits[portfolio] = discount * one_day_discount * np.abs(holdings[portfolio, 1:]) @ next_spreads / first.curve.spot
                settled = np.flatnonzero(pd.to_datetime(book["expiry"]).dt.date.to_numpy() <= following.day)
                positions[:, 1 + settled] = 0.
                discount *= float(one_day_discount)
        terminal_spot = store.load(end).curve.spot
        payoffs = []
        for row in terminal.itertuples():
            claim = claims[row.claim]
            if claim.kind == "digital":
                payoff = float(terminal_spot > claim.strike if claim.is_call else terminal_spot <= claim.strike)
            else:
                observed_spot = store.load(sessions[claim.observation_day]).curve.spot
                if claim.kind == "ko":
                    survives = observed_spot > claim.barrier if claim.is_call else observed_spot < claim.barrier
                    vanilla = max(terminal_spot - claim.strike if claim.is_call else claim.strike - terminal_spot, 0.)
                    payoff = float(survives) * vanilla / first.curve.spot
                else:
                    payoff = abs(terminal_spot / observed_spot - claim.strike)
            payoffs.append(payoff)
        expected = terminal["side"].to_numpy() * (discount * np.asarray(payoffs) - terminal["initial_price"].to_numpy())
        expected += total_gain - total_cost
        if not np.isfinite(terminal[["costed_pnl", "terminal_payoff"]].to_numpy()).all() or not np.isfinite(expected).all():
            raise ValueError("Non-finite saved terminal episode evidence")
        terminal_error = float(np.max(np.abs(expected - terminal["costed_pnl"].to_numpy())))
        if config.get("optimized_premium_cap") is not None:
            np.testing.assert_array_equal(terminal["cap_days"], cap_days)
        payoff_error = float(np.max(np.abs(np.asarray(payoffs) - terminal["terminal_payoff"].to_numpy())))
        exit_error = float(np.max(np.abs(terminal_exits - terminal["terminal_hypothetical_exit_cost"].to_numpy())))
        if max(maximum_step_error, terminal_error, payoff_error, exit_error) > 1e-10:
            raise RuntimeError(f"Episode replay failed on {origin}: step={maximum_step_error}, terminal={terminal_error}")
        records.append({"origin": origin.isoformat(), "family": path.parent.name, "portfolios": len(terminal),
                        "maximum_step_error": maximum_step_error, "terminal_error": terminal_error,
                        "payoff_error": payoff_error, "settled_options_have_no_exit_trade": True})
    if not records:
        raise ValueError("No completed episodes are available to audit")
    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_directory / "replay.parquet", index=False)
    summary = {"completed_episodes": len(records), "portfolios": sum(row["portfolios"] for row in records),
        "maximum_cash_error": max(max(row["maximum_step_error"], row["terminal_error"], row["payoff_error"]) for row in records),
        "input_directory": str(input_directory), "optimized_premium_cap": config.get("optimized_premium_cap")}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    print(json.dumps(audit_episodes(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
