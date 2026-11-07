"""Independently replay saved one-day portfolios against licensed quotes and cash flows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.cash import MarketStore, observed_marks
from quantdata.hedge2.claims import make_claims
from quantdata.hedge2.source import SourceArchive, surface_digitals


def completed_audit_dates(*, input_directory: Path, configuration: dict[str, object],
                          completed_only: bool) -> tuple[str, ...]:
    """Capture completed dates once; never treat a running partial day as evidence."""
    if not completed_only:
        return tuple(configuration["dates"])
    objective_layout = configuration.get("method") in ("parametric", "statistical", "historical")
    selected = []
    for value in configuration["dates"]:
        marker = (input_directory / value / "completed.json" if objective_layout
                  else input_directory / f"{value}.complete.json")
        if not marker.exists():
            continue
        record = json.loads(marker.read_text())
        hashes = record.get("artifacts_sha256", record)
        if not hashes or any(Path(name).name != name or not isinstance(digest, str) for name, digest in hashes.items()):
            raise ValueError(f"Invalid completed-day evidence marker: {marker}")
        for name, digest in hashes.items():
            if hashlib.sha256((marker.parent / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Completed-day audit artifact changed: {marker.parent / name}")
        selected.append(value)
    if not selected:
        raise ValueError("No complete dated artifacts are available for a partial-run audit")
    return tuple(selected)


def audit_daily(*, input_directory: Path, output_directory: Path,
                completed_only: bool = False) -> dict[str, object]:
    configuration = json.loads((input_directory / "config.json").read_text())
    dates = completed_audit_dates(input_directory=input_directory, configuration=configuration,
        completed_only=completed_only)
    store = MarketStore(market_directory=Path(configuration["market_directory"]),
        licensed_fit_directory=Path(configuration["licensed_fit_directory"]),
        raw_quote_directory=Path(configuration["raw_quote_directory"]))
    archive = SourceArchive(root=Path(configuration["source_root"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for index, value in enumerate(dates):
        day = date.fromisoformat(value)
        objective_layout = configuration.get("method") in ("parametric", "statistical", "historical")
        path = input_directory / value / "results.parquet" if objective_layout else input_directory / f"{value}.parquet"
        book_path = input_directory / value / "book.parquet" if objective_layout else input_directory / f"{value}-book.parquet"
        position_path = input_directory / value / "positions.npz" if objective_layout else input_directory / f"{value}-positions.npz"
        frame, book = pd.read_parquet(path), pd.read_parquet(book_path)
        with np.load(position_path, allow_pickle=False) as saved:
            positions = saved["positions"]
        if positions.shape != (len(frame), 1 + len(book)) or not np.isfinite(positions).all():
            raise ValueError(f"Saved hedge row/portfolio geometry failed on {day}")
        current = store.load(day)
        following = store.load(get_business_days(day, day + timedelta(days=10))[1])
        marks, spreads = observed_marks(book, current)
        next_marks, next_spreads = observed_marks(book, following)
        quote_error = max(float(np.max(np.abs(marks - book["observed_mid"].to_numpy(dtype=float)))),
                          float(np.max(np.abs(spreads - book["observed_half_spread"].to_numpy(dtype=float)))))
        cash_prices = np.r_[current.curve.spot, marks]
        denominator = (cash_prices if configuration.get("itm_premium_book", False)
                       else np.full(len(cash_prices), current.curve.spot))
        cash_holdings = positions / denominator[None, :]
        forward, reference, discount = current.curve.at(1.)
        gains = np.r_[discount * (following.curve.spot - reference), discount * next_marks - marks]
        costs = np.abs(cash_holdings[:, 1:]) @ spreads
        hedge_gains = cash_holdings @ gains
        exit_costs = discount * np.abs(cash_holdings[:, 1:]) @ next_spreads
        source, _ = archive.load(day)
        claims = {claim.label: claim for claim in make_claims(initial_spot=current.curve.spot,
            atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
            if claim.kind == "digital" and claim.terminal_day == 1}
        initial_prices = {label: float(discount * surface_digitals(source, expiry_y=1. / DAYS_PER_YEAR,
            strikes=np.array([claim.strike / forward]), is_call=claim.is_call)[0]) for label, claim in claims.items()}
        target_gains = np.array([row.side * (discount * claims[row.claim].payoff(following.curve.spot)
            - initial_prices[row.claim]) for row in frame.itertuples()])
        differences = {
            "market_target_price": frame["market_target_price"].to_numpy() - np.array([initial_prices[label] for label in frame["claim"]]),
            "target_gain": frame["target_gain"].to_numpy() - target_gains,
            "hedge_gain": frame["hedge_gain"].to_numpy() - hedge_gains,
            "entry_cost": frame["entry_cost"].to_numpy() - costs,
            "hypothetical_exit_cost": frame["hypothetical_exit_cost"].to_numpy() - exit_costs,
            "costed_pnl": frame["costed_pnl"].to_numpy() - (target_gains + hedge_gains - costs)}
        if not all(np.isfinite(values).all() for values in differences.values()):
            raise ValueError(f"Non-finite saved cash-flow evidence on {day}")
        maxima = {name: float(np.max(np.abs(values))) for name, values in differences.items()}
        if quote_error > 1e-10 or max(maxima.values()) > 1e-9:
            raise RuntimeError(f"Independent cash/quote replay failed on {day}: {maxima}, quote_error={quote_error}")
        rows.append({"date": value, "portfolios": len(frame), "quote_error_cash": quote_error,
                     **{f"max_{name}_error": error for name, error in maxima.items()},
                     **{f"sha256_{artifact.name}": hashlib.sha256(artifact.read_bytes()).hexdigest()
                        for artifact in (path, book_path, position_path)}})
        if (index + 1) % 50 == 0:
            print(f"cash replay dates={index+1}/{len(dates)} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    pd.DataFrame(rows).to_parquet(output_directory / "replay.parquet", index=False)
    numeric = pd.DataFrame(rows).select_dtypes(include="number")
    error_columns = [name for name in numeric if name.endswith("_error")]
    summary = {"dates": len(rows), "portfolios": sum(row["portfolios"] for row in rows),
        "requested_dates": len(configuration["dates"]), "partial_run_audit": len(dates) < len(configuration["dates"]),
        "maximum_normalized_cash_flow_error": float(numeric[error_columns].to_numpy().max()),
        "maximum_quote_error_cash": float(numeric["quote_error_cash"].max()),
        "elapsed_seconds": time.monotonic()-started, "input_directory": str(input_directory),
        "solver_warning_history": ("Objective records preserve exact solver warnings."
            if configuration.get("method") in ("parametric", "statistical", "historical") else
            "Analytic references do not use numerical hedge optimizers." if configuration.get("method") == "references" else
            "Initial completed Greek records retain statuses but not every original warning. Cash replay does not reconstruct lost warnings.")}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true")
    print(json.dumps(audit_daily(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
