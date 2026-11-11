"""Unhedged one-day digitals without the full European-book coverage restriction."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from quantdata.biz_calendar import DAYS_PER_YEAR, get_business_days
from quantdata.hedge2.claims import Claim, make_claims
from quantdata.hedge2.historical import licensed_curve
from quantdata.hedge2.metrics import summarize_products
from quantdata.hedge2.source import SourceArchive
from quantdata.hedge2.state import DailyState


def unhedged_digital_rows(*, claims: tuple[Claim, ...], market_marks: np.ndarray,
                         following_spot: float, discount: float) -> list[dict[str, object]]:
    """Keep a digital's unit cash payout, both sides and raw discounted settlement."""
    marks = np.asarray(market_marks, dtype=float)
    if (not claims or marks.shape != (len(claims),) or not np.isfinite(marks).all()
            or not np.isfinite([following_spot, discount]).all() or min(following_spot, discount) <= 0
            or any(claim.kind != "digital" or claim.terminal_day != 1 for claim in claims)):
        raise ValueError("Coverage returns require live one-day digitals, finite marks and positive spot/discount")
    rows = []
    for claim, mark in zip(claims, marks, strict=True):
        payoff = claim.payoff(following_spot)
        for side in (-1, 1):
            gain = side * (discount * payoff - float(mark))
            rows.append({"claim": claim.label, "side": side, "method": "unhedged",
                "market_target_price": float(mark), "terminal_payoff": payoff,
                "target_gain": gain, "costed_pnl": gain, "entry_cost": 0., "hedge_gain": 0.})
    return rows


def run_unhedged_coverage(*, source_configuration: Path, coverage_path: Path,
                         output_directory: Path) -> dict[str, object]:
    """Compare the fitted-origin population with its ex-post complete-book subset.

    Claims are valued with the zero-smoothing DLV (LV mode); saved SANOS
    models only set claim strikes. Licensed spot/carry inputs settle them. Missing European quote books do not exclude an otherwise usable
    spot settlement. Actual returns are never centered or volatility-scaled.
    Coverage labels affect reporting only, not prices, strikes or payoffs.
    """
    sources = json.loads(source_configuration.read_text())
    coverage = pd.read_parquet(coverage_path)
    selected = coverage.loc[coverage["group"] == "digital-1D"].set_index("origin")
    if selected.empty or selected.index.has_duplicates:
        raise ValueError("Coverage must identify each one-day fitted origin exactly once")
    dynamics = Path(sources["dynamics_directory"])
    daily_directory = Path(sources["daily_directory"])
    dates = [path.stem for path in sorted((dynamics / "days").glob("*.npz"))
        if date(2023, 1, 1) <= date.fromisoformat(path.stem) <= date(2025, 12, 31)]
    if not dates or set(dates) != set(selected.index):
        raise ValueError("Broad unhedged coverage must use every fitted origin in the same population")
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Coverage artifacts must remain outside Git working trees")
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        if (output_directory / "config.json").exists():
            raise ValueError("Preserve the existing coverage run; use a new uniquely identified output directory")
        implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("unhedged_coverage.py", "historical.py", "claims.py", "source.py", "cash.py", "metrics.py",
                         "state.py", "pricing.py", "dlv.py")}
        identity = {"dates": dates, "source_configuration": str(source_configuration.resolve()),
            "source_configuration_sha256": hashlib.sha256(source_configuration.read_bytes()).hexdigest(),
            "coverage_path": str(coverage_path.resolve()), "coverage_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
            "implementation_sha256": implementation,
            "sample": "all fitted origins; observed spot and carry required, not complete European quote books",
            "units": "unit cash digital payout; zero trading cost; raw realized returns",
            "daily_directory": str(daily_directory), "claim_valuation": "zero-smoothing DLV (LV mode)"}
        (output_directory / "config.json").write_text(json.dumps(identity, indent=2) + "\n")
        archive = SourceArchive(root=Path(sources["source_root"]))
        licensed = Path(sources["licensed_fit_directory"])
        rows, exclusions, inputs, spot_rows = [], [], [], []
        started = time.monotonic()
        for date_index, value in enumerate(dates):
            day = date.fromisoformat(value)
            following = get_business_days(day, day + timedelta(days=10))[1]
            paths = [licensed / "SPX" / f"{selected_day}.pdq" for selected_day in (day, following)]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                exclusions.append({"date": value, "next_date": following.isoformat(), "missing_inputs": json.dumps(missing)})
                continue
            curve, following_curve = (licensed_curve(licensed, selected_day) for selected_day in (day, following))
            source, metadata = archive.load(day)
            _, reference, discount = curve.at(1.)
            claims = tuple(claim for claim in make_claims(initial_spot=curve.spot,
                atm_vols_by_day={tenor: float(source.interpolate_atm_vols(tenor / DAYS_PER_YEAR)) for tenor in (1, 2, 3, 5)})
                if claim.kind == "digital" and claim.terminal_day == 1)
            pricer = DailyState.load(daily_directory, day).decode().pricer
            marks = np.array([claim.price(pricer, curve, elapsed_day=0) for claim in claims])
            complete = bool(selected.loc[value, "eligible"])
            for row in unhedged_digital_rows(claims=claims, market_marks=marks,
                    following_spot=following_curve.spot, discount=float(discount)):
                rows.append({"date": value, "next_date": following.isoformat(), "complete_book": complete, **row})
            spot_rows.append({"date": value, "complete_book": complete,
                "forward_spot_return": float(following_curve.spot / reference - 1.)})
            inputs.append({"date": value, "source_model": metadata,
                "licensed_inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})
            if date_index == 0 or (date_index + 1) % 50 == 0:
                elapsed = time.monotonic() - started
                print(f"unhedged coverage dates={date_index+1}/{len(dates)} rows={len(rows)} "
                    f"elapsed_s={elapsed:.1f} remaining_s~{elapsed*(len(dates)-date_index-1)/(date_index+1):.0f}", flush=True)
        frame, spots = pd.DataFrame(rows), pd.DataFrame(spot_rows)
        if frame.empty:
            raise ValueError("No licensed spot settlements remain for the fitted population")
        samples = pd.concat([frame.assign(sample="all_usable_origins"),
            frame.loc[frame["complete_book"]].assign(sample="complete_book"),
            frame.loc[~frame["complete_book"]].assign(sample="outside_complete_book")], ignore_index=True)
        statistics = summarize_products(samples, group_columns=("sample", "side", "claim"))
        diagnostics = []
        for label, selected_spots in (("all_usable_origins", spots),
                ("complete_book", spots.loc[spots["complete_book"]]),
                ("outside_complete_book", spots.loc[~spots["complete_book"]])):
            returns = selected_spots["forward_spot_return"].to_numpy(dtype=float)
            diagnostics.append({"sample": label, "dates": len(returns),
                "mean_absolute_spot_return": float(np.abs(returns).mean()) if len(returns) else None,
                "spot_return_sample_std": float(returns.std(ddof=1)) if len(returns) > 1 else None})
        outputs = {"results.parquet": frame, "per_product.parquet": statistics, "spot_returns.parquet": spots,
            "spot_sample_diagnostics.parquet": pd.DataFrame(diagnostics),
            "exclusions.parquet": pd.DataFrame(exclusions, columns=("date", "next_date", "missing_inputs"))}
        for name, table in outputs.items():
            destination = output_directory / name
            temporary = destination.with_suffix(".partial.parquet")
            table.to_parquet(temporary, index=False)
            temporary.replace(destination)
        (output_directory / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        summary = {"status": "complete", "fitted_origins": len(dates), "usable_origins": frame["date"].nunique(),
            "complete_book_origins": frame.loc[frame["complete_book"], "date"].nunique(),
            "outside_complete_book_origins": frame.loc[~frame["complete_book"], "date"].nunique(),
            "missing_spot_origins": len(exclusions), "rows": len(frame),
            "quote_filter_not_applied": True, "elapsed_seconds": time.monotonic() - started}
        summary = json.loads(json.dumps(summary, default=int))
        summary["output_sha256"] = {name: hashlib.sha256((output_directory / name).read_bytes()).hexdigest()
            for name in (*outputs, "inputs.json")}
        (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-configuration", "coverage-path", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(run_unhedged_coverage(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
