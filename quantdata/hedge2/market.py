"""Join the public SANOS benchmark to its licensed observed quotes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_ID = "widb,surface,eta=0.1,all e01f35f1"
BENCHMARK_COLUMNS = (
    "date", "expiry", "strike", "option_type", "is_am", "bdte", "nms",
    "fwd", "df", "fit_price", "fit_vol",
)
SOURCE_COLUMNS = (
    "date", "expiry", "strike", "is_am", "mkt_call_bid", "mkt_call_ask",
    "mkt_put_bid", "mkt_put_ask",
)
OUTPUT_COLUMNS = (
    "date", "expiry", "strike", "option_type", "is_am", "bdte", "nms",
    "fwd", "df", "fit_price", "fit_vol", "observed_bid", "observed_ask",
    "observed_mid", "observed_half_spread",
)


@dataclass(frozen=True, kw_only=True)
class MarketPaths:
    benchmark_directory: Path
    licensed_fit_directory: Path
    ticker: str = "SPX"

    def __post_init__(self) -> None:
        if self.benchmark_directory.name != BENCHMARK_ID:
            raise ValueError(f"Expected the all-option 0.1 benchmark: {BENCHMARK_ID}")
        if not self.benchmark_directory.is_dir():
            raise FileNotFoundError(self.benchmark_directory)
        if not self.licensed_fit_directory.is_dir():
            raise FileNotFoundError(self.licensed_fit_directory)
        if not self.ticker.isalnum():
            raise ValueError("Ticker must be alphanumeric")

    def yearly_file(self, year: int) -> Path:
        files = list((self.benchmark_directory / self.ticker).glob(f"{year}-*-all-0.10.parquet"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one benchmark file for {year}; found {files}")
        return files[0]

    def licensed_file(self, day: date) -> Path:
        return self.licensed_fit_directory / self.ticker / f"{day.isoformat()}.pdq"


def load_benchmark_year(paths: MarketPaths, year: int) -> pd.DataFrame:
    frame = pd.read_parquet(paths.yearly_file(year), columns=list(BENCHMARK_COLUMNS))
    if frame.empty or not (frame["date"].dt.year == year).all():
        raise ValueError(f"Invalid benchmark dates in {year}")
    if not set(frame["option_type"].unique()) <= {"call", "put"}:
        raise ValueError(f"Invalid option types in {year}")
    return frame


def load_market_day(paths: MarketPaths, day: date, benchmark_year: pd.DataFrame) -> pd.DataFrame:
    """Return observed midpoint and SANOS fit as distinct fields for one date."""
    fitted = benchmark_year.loc[benchmark_year["date"] == pd.Timestamp(day)].copy()
    if fitted.empty:
        raise ValueError(f"No benchmark rows on {day}")
    source_path = paths.licensed_file(day)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = pd.read_parquet(source_path, columns=list(SOURCE_COLUMNS))
    if source.empty or not (source["date"] == pd.Timestamp(day)).all():
        raise ValueError(f"Invalid licensed rows on {day}")

    key = ["date", "expiry", "strike", "is_am"]
    if source.duplicated(key).any():
        raise ValueError(f"Duplicate licensed contracts on {day}")
    quoted = pd.concat(
        [
            source[key].assign(
                option_type=option_type,
                observed_bid=source[f"mkt_{option_type}_bid"].to_numpy(),
                observed_ask=source[f"mkt_{option_type}_ask"].to_numpy(),
            )
            for option_type in ("call", "put")
        ],
        ignore_index=True,
    )
    key.append("option_type")
    if fitted.duplicated(key).any() or quoted.duplicated(key).any():
        raise ValueError(f"Duplicate benchmark contracts on {day}")
    joined = fitted.merge(quoted, on=key, how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"Benchmark/source mismatch on {day}: {counts}")
    joined = joined.drop(columns="_merge")
    bid = joined["observed_bid"].to_numpy(dtype=np.float64)
    ask = joined["observed_ask"].to_numpy(dtype=np.float64)
    if not (np.isfinite(bid).all() and np.isfinite(ask).all() and (bid >= 0).all() and (ask > bid).all()):
        raise ValueError(f"Invalid observed bid/ask on {day}")
    if not np.isfinite(joined["fit_price"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Invalid fitted price on {day}")
    joined["observed_mid"] = 0.5 * (bid + ask)
    joined["observed_half_spread"] = 0.5 * (ask - bid)
    return joined.loc[:, OUTPUT_COLUMNS].sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def prepare_market_panel(paths: MarketPaths, output_directory: Path, start: date, end: date) -> dict[str, object]:
    """Resumably write private day files, with a compact progress record."""
    if end < start:
        raise ValueError("End date precedes start date")
    if output_directory.is_relative_to(paths.benchmark_directory) or output_directory.is_relative_to(paths.licensed_fit_directory):
        raise ValueError("Output must not be placed in either source dataset")
    output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    completed = 0
    skipped = 0
    total = 0
    for year in range(start.year, end.year + 1):
        benchmark_year = load_benchmark_year(paths, year)
        days = [value.date() for value in benchmark_year["date"].drop_duplicates() if start <= value.date() <= end]
        total += len(days)
        for day in days:
            output_file = output_directory / "days" / f"{day.isoformat()}.parquet"
            if output_file.is_file():
                skipped += 1
                continue
            output_file.parent.mkdir(parents=True, exist_ok=True)
            frame = load_market_day(paths, day, benchmark_year)
            temporary_file = output_file.with_suffix(".partial.parquet")
            frame.to_parquet(temporary_file, index=False)
            temporary_file.replace(output_file)
            completed += 1
            if completed % 25 == 0:
                elapsed = time.monotonic() - start_time
                print(f"market days new={completed} skipped={skipped} seen={completed + skipped} elapsed_s={elapsed:.1f}", flush=True)
    result = {
        "benchmark_id": BENCHMARK_ID,
        "ticker": paths.ticker,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_dates": total,
        "written_dates": completed,
        "previously_written_dates": skipped,
        "elapsed_seconds": time.monotonic() - start_time,
        "price_convention": "observed bid/ask midpoint; SANOS fitted price retained separately",
        "license": "private source quotes; do not distribute",
    }
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-directory", type=Path, required=True)
    parser.add_argument("--licensed-fit-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    paths = MarketPaths(
        benchmark_directory=args.benchmark_directory,
        licensed_fit_directory=args.licensed_fit_directory,
    )
    print(json.dumps(prepare_market_panel(paths, args.output_directory, args.start, args.end), indent=2))


if __name__ == "__main__":
    main()
