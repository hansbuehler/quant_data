"""
Get contracts for underylings
"""

from __future__ import annotations

import math
import threading
import warnings as warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl
from cdxcore.err import verify
from cdxcore.jcpool import JCPool
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import TrackTime, fmt_date, fmt_digits, fmt_list
from cdxcore.verbose import Context

# rom aae.locations import locations, SubDir
# from aae.market import nyse
# from aae.poly import PolyHandle, POLY_START_DATE
# from aae.poly.fetch import fetch_day_summary, has_any_quotes, has_any_trades
# from aae.tickers.calendar import polygon_splits
from cdxcore.version import version
from prettytable import PrettyTable

from quantdata.biz_calendar import (
    DAYS_PER_YEAR,
    NMS,
    TZ_NY,
    DateRange,
    add_business_days,
    count_business_days_between,
    get_business_days,
)
from quantdata.massive import MassiveClient, SubDir
from quantdata.pool import PoolConfig

_cache = MassiveClient.locations.processing("contracts", fmt=SubDir.POLARS_PARQUET)

# -----------------------------
# Rejection codes
# -----------------------------


class REJECTED:
    USED = 0  # contract is used (not rejected)
    EXPIRY = 1  # contract expires too far in the future (days_to_exp > contracts_filter.max_dte)
    MONEYNESS = 2  # contract is too far out of the money (scaled_mns > contracts_filter.max_mns)
    VOLUME = 3  # contract had insufficient volume traded the previous day (this is cash volume)
    PREV_VOLUME_MISSING = -1  # prev day volume data (for filtering) is missing.
    PREV_NO_QUOTES_OR_TRADES = -2  # no trades and no quote data on the previous day (assuming we won't trade tickers if neither quotes nor trades are available)
    NO_TODAY_QUOTES = -3  # no quotes today (assuming we won't trade until we saw a quote)


SCHEMA = {
    "approx_cash_vol_prev_k": pl.Float32,
    "approx_cash_vol_today_k": pl.Float32,
    "cfi": pl.String,
    "contract_type": pl.String,
    "correction": pl.Int32,
    "days_to_exp": pl.Int16,
    "exercise_style": pl.String,
    "exp_date": pl.Date,
    "date": pl.Date,
    "prev_bus_date": pl.Date,
    "primary_exchange": pl.String,
    "reject_code": pl.Int8,
    "scaled_mns": pl.Float32,
    "shares_per_contract": pl.Int32,
    "spot_adj_prev_close": pl.Float32,
    "spot_prev_close": pl.Float32,
    "spot_close": pl.Float32,
    "spot_open": pl.Float32,
    "spot_adj_close": pl.Float32,
    "spot_ref_prev_close": pl.Float32,
    "strike_price": pl.Float32,
    "ticker": pl.String,
    "underlying_ticker": pl.String,
    "has_prev_quotes": pl.Boolean,
    "has_prev_trades": pl.Boolean,
    "has_today_quotes": pl.Boolean,
    "has_today_trades": pl.Boolean,
}

# -----------------------------
# Config
# -----------------------------


@version("0.1")
@dataclass(frozen=True)
class ContractsFilter:
    max_dte: int = 7  #: In business days
    max_mns: float = 2.0  #: Maximum normalized strike, i.e. | log(K/S_prev)*sqrt((1+e)/(T+e)/0.1 | ~ | log(K/S_prev)/(0.1*sqrt(T)) |, where S_prev is the previous close adjusted for stock splits and dividends to match today's unadjusted price, and T is time to expiry in years. This is a measure of moneyness that accounts for both distance to the money and time to expiry.
    min_opt_vol_k: float = 0.0  #: Previous-day approx cash vol threshold (USD 1000's)

    def __str__(self):
        return (
            f"max_dte={self.max_dte},max_mns={self.max_mns:.3g},min_volu_k={self.min_opt_vol_k:.3g}"
        )


# -----------------------------
# Core filtering logic
# -----------------------------


@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.2",
    dependencies=[
        MassiveClient.list_all_contracts_expiring_after_day,
        MassiveClient.fetch_day_summary,
        ContractsFilter,
    ],
    uid=lambda underlying, d, contracts_filter, exclude_half_days: (
        f"{named_unique_filename48_8(str(contracts_filter), contracts_filter, exclude_half_days)}/{underlying}/contracts/{d}"
    ),
    exclude_args=["pool_config", "client", "verbose"],
)
def filter_contracts(
    underlying: str,
    d: date,
    contracts_filter: ContractsFilter,
    exclude_half_days: bool,
    *,
    pool_config: PoolConfig = PoolConfig(),
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Returns:
      - eligible_contracts: list of contract dicts (subset)
      - all_contracts: full list with added fields:
            reject_code (int),
            spot_prev_close (float),
            scaled_mns (float),
            approx_cash_vol_prev_m (float),
            approx_cash_vol_today_m (float)
    Rules:
      1) expiry filter: days_to_exp in [0, contracts_filter.max_dte]
      2) moneyness filter: |log(K/S_prev)|/sqrt(T)/0.1 <= contracts_filter.max_mns, T = days_to_exp / DAYS_PER_YEAR
      3) volume filter on PREVIOUS day: approx_cash_vol_prev_k >= contracts_filter.min_opt_vol_k
    """
    client = MassiveClient()

    """
    Handle stock splits:
        Assume stock of 1000 was split 10:1 into 100.
        Strike today 99 is now 9.9, but the previous close was 100, so the moneyness is 0.1/0.1 = 1.0 instead of 0.01/0.1 = 0.1.
        We therefore scale yesterdays adjusted close with today's adjusted factor.

        F_today = unadjusted / adjusted
        prev_close = adjustted_prev_close * F_today
    """

    context = PrettyObject()
    context.prev_d = add_business_days(
        d, -1, exclude_half_days=exclude_half_days
    )  # previous business day (NB: if d is a Monday then this is the previous Friday, so we correctly skip the weekend)
    context.close_adj_prev = client.fetch_day_summary(
        underlying, context.prev_d, adjusted=True, can_be_none=False
    )["close"]
    context.close_unadj_prev = client.fetch_day_summary(
        underlying, context.prev_d, adjusted=True, can_be_none=False
    )["close"]
    context.close_adj = client.fetch_day_summary(underlying, d, adjusted=True, can_be_none=False)[
        "close"
    ]
    ocv_today_unadj = client.fetch_day_summary(underlying, d, adjusted=False, can_be_none=False)
    context.close_unadj = ocv_today_unadj["close"]
    context.open_unadj = ocv_today_unadj["open"]
    F_today = context.close_unadj / context.close_adj
    context.ref_prev_close = context.close_adj_prev * F_today

    with verbose.write_t(
        lambda: f"\rFetching all contracts for {underlying} expiring on or after {d}... ", end=""
    ) as tme:
        all_contracts = client.list_all_contracts_expiring_after_day(
            underlying, d, max_dte=contracts_filter.max_dte
        )
        verbose.write(lambda: f"found {len(all_contracts)} contracts. This took {tme}.", head=False)

    def process(contract, d, exclude_half_days, context):
        timing = TrackTime("Process")
        """ multi-threaded processing of a single contract, returning the contract dict with added fields and reject_code """
        client = MassiveClient()

        copy_fields = [
            "cfi",
            "contract_type",
            "correction",
            "exercise_style",
            "expiration_date",
            "primary_exchange",
            "shares_per_contract",
            "strike_price",
            "ticker",
            "underlying_ticker",
        ]
        c = {k: contract.get(k, None) for k in copy_fields}

        opt_ticker = contract["ticker"]
        exp_date = datetime.strptime(contract["expiration_date"], "%Y-%m-%d").date()
        c["exp_date"] = exp_date
        c["date"] = d
        c["prev_bus_date"] = context.prev_d
        c["reject_code"] = REJECTED.USED  # assume used unless rejected
        c["spot_adj_prev_close"] = context.close_adj_prev
        c["spot_prev_close"] = context.close_unadj_prev
        c["spot_close"] = context.close_unadj
        c["spot_open"] = context.open_unadj
        c["spot_adj_close"] = context.close_adj
        c["spot_ref_prev_close"] = (
            context.ref_prev_close
        )  # this is the previous close, adjusted to match today's unadjusted spot

        # ---- expiry / DTE
        c["days_to_exp"] = count_business_days_between(
            d, exp_date, exclude_half_days=exclude_half_days
        )  # NB: if d==exp_date then this function returns 0
        if c["days_to_exp"] < 0:
            raise RuntimeError(
                f"Cannot compute days to expiry: no days between {d} and {exp_date} ?"
            )

        K = float(c["strike_price"])
        T = float(c["days_to_exp"] + 1) / float(DAYS_PER_YEAR)

        c["scaled_mns"] = abs(NMS.to_nms(K / context.ref_prev_close, T))
        if c["days_to_exp"] > contracts_filter.max_dte:
            c["reject_code"] = REJECTED.EXPIRY
            return c, timing

        if c["scaled_mns"] > contracts_filter.max_mns:
            c["reject_code"] = REJECTED.MONEYNESS
            return c, timing

        timing.lap_time("Prep")

        def approx_cash_vol_k(oc: dict | None) -> float | None:
            if oc is None:
                return None
            opx, cpx, vol = oc["open"], oc["close"], oc["volume"]
            return float(vol) * 0.5 * (float(opx) + float(cpx)) / 1000.0

        # ---- volume filter first: cheapest check that rejects the most contracts
        with timing("Fetch open close volume prev"):
            oc_prev = client.fetch_day_summary(
                opt_ticker, context.prev_d, adjusted=False, can_be_none=True
            )

        c["approx_cash_vol_prev_k"] = approx_cash_vol_k(oc_prev)

        if oc_prev is None:
            c["approx_cash_vol_today_k"] = None
            c["has_prev_quotes"] = None
            c["has_prev_trades"] = None
            c["has_today_quotes"] = None
            c["has_today_trades"] = None
            c["reject_code"] = REJECTED.PREV_VOLUME_MISSING
            return c, timing

        if c["approx_cash_vol_prev_k"] < contracts_filter.min_opt_vol_k:
            c["approx_cash_vol_today_k"] = None
            c["has_prev_quotes"] = None
            c["has_prev_trades"] = None
            c["has_today_quotes"] = None
            c["has_today_trades"] = None
            c["reject_code"] = REJECTED.VOLUME
            return c, timing

        # ---- prev quotes/trades check
        with timing("Fetch quotes & trades"):
            has_prev_quotes, has_prev_trades = client.has_any_quotes_or_trades(
                opt_ticker, context.prev_d
            )
            c["has_prev_quotes"] = has_prev_quotes
            c["has_prev_trades"] = has_prev_trades

        if not has_prev_quotes and not has_prev_trades:
            c["approx_cash_vol_today_k"] = None
            c["has_today_quotes"] = None
            c["has_today_trades"] = None
            c["reject_code"] = REJECTED.PREV_NO_QUOTES_OR_TRADES
            return c, timing

        # ---- today quotes check
        with timing("Fetch quotes & trades"):
            has_today_quotes, has_today_trades = client.has_any_quotes_or_trades(opt_ticker, d)
            c["has_today_quotes"] = has_today_quotes
            c["has_today_trades"] = has_today_trades

        if not has_today_quotes:
            c["approx_cash_vol_today_k"] = None
            c["reject_code"] = REJECTED.NO_TODAY_QUOTES
            return c, timing

        # ---- today volume (diagnostic only, not used for rejection)
        with timing("Fetch open close volume today"):
            oc_today = client.fetch_day_summary(opt_ticker, d, adjusted=False, can_be_none=True)

        c["approx_cash_vol_today_k"] = approx_cash_vol_k(oc_today)

        return c, timing

    jcpool = JCPool(num_workers=pool_config.num_workers, threading=pool_config.threading)
    df = []
    tme = verbose.timer()
    timing = None
    for c, ctiming in jcpool.parallel(
        jcpool.delayed(process)(ct, d, exclude_half_days, context) for ct in all_contracts
    ):
        timing = ctiming if timing is None else (timing.add(ctiming))
        if tme.interval_test(1):
            verbose.report(
                1,
                lambda: (
                    f"\rFiltering {len(df) + 1}/{len(all_contracts)} contracts: {int((len(df) + 1) * 100 // len(all_contracts))}%: {c['ticker']}..."
                ),
                end="",
            )
        df.append(c)
    if len(df) == 0:
        return None

    df = pl.from_dicts(df, infer_schema_length=None)

    # convert to scheme

    df = df.drop(["expiration_date"])
    if not "correction" in df.columns:
        df = df.with_columns(pl.lit(None).alias("correction"))
    assert set(df.columns) == set(SCHEMA.keys()), (
        f"Contract schema mismatch for {underlying} on {fmt_date(d)}: {sorted(set(df.columns))} != {sorted(set(SCHEMA.keys()))}"
    )
    df = df.with_columns([pl.col(col).cast(dtype) for col, dtype in SCHEMA.items()])

    active = df.filter(pl.col("reject_code") == REJECTED.USED)

    timestr = ""
    for topic, tinfo in timing.items():
        timestr += f"{topic}: {tinfo.average_lap_seconds:.2f}s, "
    timestr = timestr[:-2]

    verbose.write(
        lambda: (
            f"\rFiltered {fmt_digits(len(df))} contacts for {underlying} on {fmt_date(d)}, "
            + f"yielding {fmt_digits(active.height)} active contracts. This took {tme}. "
            + f"Timing: {timestr}."
        ),
        head=False,
    )

    return df


@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.2",
    dependencies=[filter_contracts],
    uid=lambda ticker, date_range, contracts_filter: (
        f"{named_unique_filename48_8(str(contracts_filter), contracts_filter, date_range.exclude_half_days)}/{ticker}/contracts/{date_range}"
    ),
    exclude_args=["pool_config", "client", "verbose", "refresh"],
)
def filter_contracts_between(
    ticker: str,
    date_range: DateRange,
    contracts_filter: ContractsFilter,
    *,
    refresh: bool = False,
    pool_config: PoolConfig = PoolConfig(),
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    """
    Filter option contracts for a given underlying ticker between start and end for a given ticker.

    Parameters
    ----------
    ticker : str
        Underlying ticker symbol, e.g. "AAPL" for Apple.
    date_range : :class:`quantdata.biz_calenddar.DateRange`
        Date range for which to filter contracts. Only contracts with expiration_date >= date_range.start and expiration_date <= date_range.end are returned.
    contracts_filter : :class:`quantdata.massive.options.ContractsFilter`
        Filter parameters for contract filtering. See :class:`ContractsFilter` for details.
    refresh : bool, default False
        If True, refresh data (this will only refresh the top level files, i.e. the files for individual dates, but it will not refresh the cached results of individual contracts).
    pool_config : :class:`quantdata.massive.options.PoolConfig`, default ``PoolConfig()``
        Configuration for parallel processing. See :class:`PoolConfig` for details.
    verbose : :class:`quantdata.massive.options.Context`, default ``Context.quiet``
        Verbose context for logging. See :class:`Context` for details.

    Returns
    -------
    pl.DataFrame
        DataFrame with filtered contracts for each date in the date range. The DataFrame has the same schema as the output of :func:`filter_contracts`, with an additional column "date" indicating the date for which the contracts were filtered.

    """

    dfs = []
    dates = date_range.get_business_days()
    tme = verbose.timer()
    for id, d in enumerate(dates):
        if tme.interval_test(1):
            verbose.write(
                lambda: (
                    f"\rFiltering contracts for {ticker} on {fmt_date(d)} ... {int(id * 100 // len(dates))}% done."
                )
            )
        df = filter_contracts(
            ticker,
            d,
            contracts_filter=contracts_filter,
            exclude_half_days=date_range.exclude_half_days,
            pool_config=pool_config,
            override_cache_mode="update" if refresh else None,
            verbose=verbose(1),
        )
        if df is None:
            continue
        if not "correction" in df.columns:
            df = df.with_columns(pl.lit(None).alias("correction"))

        existing_cols = set(df.columns) & set(SCHEMA.keys())
        df = df.select(existing_cols)

        for col, dtype in SCHEMA.items():
            if col not in existing_cols:
                df = df.with_columns(pl.lit(None).alias(col).cast(dtype))
            else:
                df = df.with_columns(pl.col(col).cast(dtype))
        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError(
            f"No contracts found for {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}"
        )

    with verbose.write_t(
        lambda: (
            f"Concatenating {len(dfs)} contracts for {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}... "
        ),
        end="",
    ) as tme:
        dfs = pl.concat(dfs)
        verbose.write(lambda: f"done. This took {tme}.")
    return dfs


# ================================================================
# Command-line interface
# python -m curve_bins -tickers TSLA AAPL GS JPM NVDA -etfs
# ================================================================

import click
from cdxcore.err import verify_inp

from quantdata.biz_calendar import to_date
from quantdata.massive import cli_defaults


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option(
    "--period",
    nargs=2,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=cli_defaults.period,
    metavar="START END",
    help="Period YYYY-MM-DD YYYY-MM-DD",
)
@click.option("--halfdays", is_flag=True, help="Include half days")
@click.option("--workers", default=cli_defaults.workers, help="Number of workers", type=int)
@click.option("--threading", is_flag=True, help="Use only threading, no multiprocessing")
@click.option(
    "--refresh", is_flag=True, help="Refresh data (this will only refresh the top level files)"
)
@click.option("--quiet", is_flag=True, help="Quiet mode, no logging")
@click.option("--max_dte", default=25, help="Maximum business days to expiry", type=int)
@click.option("--max_nms", default=3.0, help="Maximum normalized moneyness", type=float)
@click.option(
    "--min_opt", default=25, help="Minimum traded option volume in thousand $", type=float
)
def main(
    tickers: list[str],
    period: list[datetime.datetime],
    halfdays: bool,
    workers: int,
    threading: bool,
    quiet: bool,
    refresh: bool,
    max_dte: int,
    max_nms: float,
    min_opt: float,
):
    verify_inp(len(tickers) > 0, "At least one ticker must be specified")
    verify_inp(period is not None and len(period) == 2, "A start and end date must be specified")

    start, end = period

    date_range = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=not halfdays)
    pool_config = PoolConfig(num_workers=workers, threading=threading)
    contracts_filter = ContractsFilter(max_dte=max_dte, max_mns=max_nms, min_opt_vol_k=min_opt)

    verbose = Context.all
    verbose.write(
        lambda: (
            f"Filtering contracts for tickers {fmt_list(tickers)} from {fmt_date(start)} to {fmt_date(end)}, "
            + f"with half days {'included' if halfdays else 'excluded'}, using filter {contracts_filter}, with {pool_config.num_workers} {'threads' if pool_config.threading else 'processes'}."
        )
    )
    for it, ticker in enumerate(tickers):
        verbose.write(
            lambda: (
                f"\rProcessing {fmt_digits(len(tickers))} tickers: {int(it * 100 // len(tickers))}% done. Now processing {ticker} from {fmt_date(start)} to {fmt_date(end)}."
            )
        )
        date_range = DateRange(start, end)
        _ = filter_contracts_between(
            ticker,
            date_range,
            contracts_filter=contracts_filter,
            cache_generate_only=not refresh,
            override_cache_mode="update" if refresh else None,
            refresh=refresh,
            pool_config=pool_config,
            verbose=verbose(1),
        )
    verbose.write("All done")


if __name__ == "__main__":
    main()
