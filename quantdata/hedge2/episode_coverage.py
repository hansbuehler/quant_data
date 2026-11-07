"""Disclose fixed-book multi-day quote coverage before running hedge optimizers."""

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
from quantdata.hedge2.cash import MarketStore, observed_marks, select_book
from quantdata.hedge2.source import SourceArchive


def coverage(*, source_root: Path, market_directory: Path, licensed_fit_directory: Path,
             raw_quote_directory: Path, dynamics_directory: Path, output_directory: Path,
             option_side: str = "otm", include_co_terminal_forwards: bool = False) -> dict[str, object]:
    """Freeze inception contracts; disclose later gaps rather than pruning the book.

    ITM/premium experiments explicitly select their option side and whether
    their method permits synthetic forwards. Greek-only books exclude them.
    Coverage must match the declared book rather than a different cohort.
    """
    resolved = output_directory.resolve()
    if any((parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Quote-coverage outputs must be outside Git working trees")
    if output_directory.exists():
        raise FileExistsError("Quote coverage needs a new uniquely identified output directory")
    if option_side not in ("otm", "itm"):
        raise ValueError("Choose the ITM or OTM inception book")
    store = MarketStore(market_directory=market_directory, licensed_fit_directory=licensed_fit_directory,
                         raw_quote_directory=raw_quote_directory)
    archive = SourceArchive(root=source_root)
    origins = [date.fromisoformat(path.stem) for path in sorted((dynamics_directory / "days").glob("*.npz"))]
    rows = []
    started = time.monotonic()
    output_directory.mkdir(parents=True)
    configuration = {"source_root": str(source_root.resolve()), "market_directory": str(market_directory.resolve()),
        "licensed_fit_directory": str(licensed_fit_directory.resolve()), "raw_quote_directory": str(raw_quote_directory.resolve()),
        "dynamics_directory": str(dynamics_directory.resolve()), "option_side": option_side,
        "include_co_terminal_forwards": include_co_terminal_forwards,
        "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("episode_coverage.py", "cash.py", "source.py")}}
    (output_directory / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    for index, origin in enumerate(origins):
        for group, horizon, tenors in (("digital-1D", 1, None), ("digital-3D", 3, (3,)),
                                      ("digital-5D", 5, (5,)), ("exotic-3D-5D", 5, (3, 5))):
            row = {"origin": origin.isoformat(), "group": group, "eligible": False, "reason": "", "inception_options": 0,
                "ordinary_options": 0, "synthetic_forwards": 0}
            try:
                sessions = get_business_days(origin, origin + timedelta(days=15))[:horizon + 1]
                for day in sessions[:-1]:
                    if not (dynamics_directory / "days" / f"{day.isoformat()}.npz").exists():
                        raise FileNotFoundError(f"Missing rolling model {day}")
                source, _ = archive.load(origin)
                first = store.load(origin)
                vols = np.asarray(source.interpolate_atm_vols(first.quotes["bdte"].to_numpy(dtype=float) / DAYS_PER_YEAR))
                book = select_book(first, atm_vols=vols, co_terminal_business_dtes=tenors,
                    option_side=option_side, include_co_terminal_forwards=include_co_terminal_forwards)
                row["inception_options"] = len(book)
                row["synthetic_forwards"] = int((book["instrument_kind"] == "zero_strike_call").sum()) if "instrument_kind" in book else 0
                row["ordinary_options"] = len(book) - row["synthetic_forwards"]
                for day in sessions:
                    active = book.loc[pd.to_datetime(book["expiry"]).dt.date >= day]
                    observed_marks(active, store.load(day))
                row["eligible"] = True
            except (FileNotFoundError, ValueError) as error:
                message = str(error)
                if not isinstance(error, FileNotFoundError) and not message.startswith(("Missing observed mids", "No PM contracts")):
                    raise
                row["reason"] = message
            rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"episode coverage origins={index+1}/{len(origins)} eligible_groups={sum(row['eligible'] for row in rows)} "
                  f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_directory / "coverage.parquet", index=False)
    summary = {"candidate_origins": len(origins), "eligible_by_group": frame.groupby("group")["eligible"].sum().to_dict(),
        "option_side": option_side, "include_co_terminal_forwards": include_co_terminal_forwards,
        "first_eligible_by_group": frame.loc[frame["eligible"]].groupby("group")["origin"].first().to_dict(),
        "elapsed_seconds": time.monotonic()-started,
        "selection": "Causal inception books; ex-post full-book complete cases. No future-dependent instrument selection.",
        "missing_quote_note": "The licensed raw-day cache discards bids <=0.011 before joining call and put records; later zero/tiny-bid observations may therefore be unavailable."}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "market-directory", "licensed-fit-directory", "raw-quote-directory",
                 "dynamics-directory", "output-directory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--option-side", choices=("otm", "itm"), default="otm")
    parser.add_argument("--include-co-terminal-forwards", action="store_true")
    print(json.dumps(coverage(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
