#!/usr/bin/env python
"""Download FLEX-ready DYSANOS market data from WRDS.

Examples
--------
Download markets for additional tickers and validate the existing SPX cache:

    python -m quantdata.dysanos.download SPX AAPL RUT

The default interval is 2022-01-03 through the latest likely-valid date in
``quantdata.widb.bad.BAD``. SPX is checked cache-only by default; use
``--cache-only-tickers`` to apply the same validation mode to other symbols.
Non-cache-only tickers use the normal read-or-generate mode and may connect to
WRDS on a cache miss.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from datetime import date

from cdxcore.subdir import CacheMustExistError
from cdxcore.verbose import Context

from quantdata.biz_calendar import DateRange
from quantdata.dysanos.flex.config import FlexGeometryConfig, FlexMarketConfig
from quantdata.dysanos.flex.market import load_market_day, market_process_count
from quantdata.pool import PoolConfig
from quantdata.widb.bad import BAD

DEFAULT_START = date(2022, 1, 3)
DEFAULT_END = BAD.LAST_LIKELY_VALID_DATE


def _load_market_job(
    ticker: str,
    trading_date: date,
    *,
    geometry: FlexGeometryConfig,
    market_config: FlexMarketConfig,
    cache_only: bool,
) -> str:
    """Load one FLEX market, optionally requiring its cache to exist."""
    cache_mode = "cacheonly" if cache_only else "on"
    try:
        market = load_market_day(
            ticker,
            trading_date,
            geometry=geometry,
            config=market_config,
            override_cache_mode=cache_mode,
            verbose=Context.quiet,
        )
    except CacheMustExistError:
        if not cache_only:
            raise
        return "cache-missing"
    return "ready" if market is not None else "no-market"


def download(
    tickers: Sequence[str],
    *,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    workers: int = 4,
    include_half_days: bool = True,
    cache_only_tickers: Sequence[str] = (),
) -> int:
    """Generate FLEX-ready market caches and validate selected caches.

    Returns non-zero if a requested cache-only market is missing.
    """
    normalized_tickers = tuple(ticker.strip().upper() for ticker in tickers)
    if not normalized_tickers or any(not ticker for ticker in normalized_tickers):
        raise ValueError("Provide one or more non-empty ticker symbols")
    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError(f"Ticker symbols must be unique: {normalized_tickers}")
    normalized_cache_only = {"SPX"} & set(normalized_tickers)
    normalized_cache_only.update(
        ticker.strip().upper() for ticker in cache_only_tickers if ticker.strip()
    )
    unknown_cache_only = normalized_cache_only - set(normalized_tickers)
    if unknown_cache_only:
        raise ValueError(
            "Cache-only tickers must also be requested: "
            f"{sorted(unknown_cache_only)}"
        )
    if start > end:
        raise ValueError(f"Start date {start} must not be after end date {end}")

    dates = DateRange(
        start=start,
        end=end,
        exclude_half_days=not include_half_days,
    ).get_business_days()
    geometry = FlexGeometryConfig()
    market_config = FlexMarketConfig()
    pool = PoolConfig(
        num_workers=market_process_count(workers),
        threading=False,
    ).pool()

    failed_validation = False
    for ticker in normalized_tickers:
        counts: Counter[str] = Counter()
        jobs = {
            trading_date: pool.delayed(_load_market_job)(
                ticker,
                trading_date,
                geometry=geometry,
                market_config=market_config,
                cache_only=ticker in normalized_cache_only,
            )
            for trading_date in dates
        }
        for index, (trading_date, result) in enumerate(pool.parallel(jobs), start=1):
            counts[result] += 1
            if index % 50 == 0 or index == len(dates):
                print(
                    f"{ticker} {index}/{len(dates)} through {trading_date}: "
                    f"ready={counts['ready']} no-market={counts['no-market']} "
                    f"cache-missing={counts['cache-missing']}",
                    flush=True,
                )
        print(
            f"{ticker}: ready={counts['ready']} no-market={counts['no-market']} "
            f"cache-missing={counts['cache-missing']}",
            flush=True,
        )
        failed_validation |= counts["cache-missing"] > 0

    return int(failed_validation)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected an ISO date (YYYY-MM-DD), received {value!r}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate FLEX-ready DYSANOS market data from WRDS.",
        epilog=(
            "Example: python -m quantdata.dysanos.download SPX AAPL RUT\n"
            f"Default dates: {DEFAULT_START} through {DEFAULT_END}."
        ),
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="WRDS ticker symbols; SPX is cache-only by default",
    )
    parser.add_argument("--start", type=_parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=_parse_date, default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-only-tickers",
        default="",
        help="Comma-separated symbols to validate without WRDS fallback (default: SPX)",
    )
    parser.add_argument(
        "--exclude-half-days",
        action="store_true",
        help="Skip NYSE early-close sessions (included by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return download(
            arguments.tickers,
            start=arguments.start,
            end=arguments.end,
            workers=arguments.workers,
            include_half_days=not arguments.exclude_half_days,
            cache_only_tickers=arguments.cache_only_tickers.split(","),
        )
    except (ValueError, RuntimeError) as error:
        _parser().error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
