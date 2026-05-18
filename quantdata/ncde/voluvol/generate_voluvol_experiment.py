from __future__ import annotations

import argparse
from datetime import date

from cdxcore.verbose import Context

from quantdata.biz_calendar import DateRange
from quantdata.ncde.voluvol.generate import ExperimentLoader, ExperimentScope
from quantdata.pool import PoolConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cached NCDE voluvol data for a ticker/date range."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to generate.")
    parser.add_argument("--start", default="2019-05-14", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-05-14", help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--forecast-horizon-min",
        type=float,
        default=15.0,
        help="Forward target horizon in minutes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    scope = ExperimentScope(
        tickers=[args.ticker],
        date_range=DateRange(start, end),
        forecast_horizon_min=args.forecast_horizon_min,
    )
    days = scope.date_range.get_business_days()

    print(
        f"Generating voluvol cache for {args.ticker} from {start} to {end} "
        f"for {len(days)} business days with horizon {args.forecast_horizon_min} minutes."
    )
    print(f"First business day: {days[0]}")
    print(f"Last business day: {days[-1]}")
    print(f"Workers: {args.workers}")

    experiment = ExperimentLoader(
        scope,
        pool_config=PoolConfig(num_workers=args.workers, threading=False),
        verbose=Context.all,
    )
    print(f"Experiment cache ready with {len(experiment)} samples.")


if __name__ == "__main__":
    main()
