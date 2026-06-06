"""
quantdata.massive.options.contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Managing contract information per underyling.
The main functions in this module are:

* :func:`filter_contracts`
* :func:`filter_contracts_between`

Each retrieves, filters, and caches contrats for a given underlying.
"""

from __future__ import annotations

import warnings as warnings
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl
from cdxcore.err import verify
from cdxcore.jcpool import JCPool
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import TrackTime, fmt_date, fmt_digits, fmt_list
from cdxcore.verbose import Context

from cdxcore.version import version
from prettytable import PrettyTable

from quantdata.biz_calendar import (
    DAYS_PER_YEAR,
    NMS,
    NMS5,
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
    """ Reasons why a contract was rejected from the final list """
    USED = 0  # contract is used (not rejected)
    EXPIRY = 1  # contract expires too far in the future (days_to_exp > contracts_filter.max_dte)
    MONEYNESS = 2  # contract is too far out of the money (nms > contracts_filter.max_mns)
    VOLUME = 3  # contract had insufficient volume traded the previous day (this is cash volume)
    PREV_VOLUME_MISSING = -1  # prev day volume data (for filtering) is missing.
    PREV_NO_QUOTES_OR_TRADES = -2  # no trades and no quote data on the previous day (assuming we won't trade tickers if neither quotes nor trades are available)
    NO_TODAY_QUOTES = -3  # no quotes today (assuming we won't trade until we saw a quote)
    NO_TODAY_TRADES = -4 # no trades today (if contracts_filter.drop_no_trades_today is True)

SCHEMA = {
    "approx_cash_vol_prev_k": pl.Float32,     # previous day cash volume in thousands of USD (volume * mid price / 1000)
    "approx_cash_vol_today_k": pl.Float32,    # today cash volume in thousands of USD (volume * mid price / 1000)  
    "cfi": pl.String,                         # cfi indetifier, e.g. "OC" for option contract
    "contract_type": pl.String,               # contract type, e.g. "call" or "put" 
    "days_to_exp": pl.Int16,                  # full business days to expiry (that means 0DTE have 0 here)
    "exercise_style": pl.String,              # exercise style, e.g. "AMERICAN" or "EUROPEAN"
    "exp_date": pl.Date,                      # expiration date of the contract
    "date": pl.Date,                          # today
    "prev_bus_date": pl.Date,                 # previous business day (used for volume filtering, previous spot etc)
    "primary_exchange": pl.String,            # https://massive.com/docs/rest/options/contracts/contract-overview
    "reject_code": pl.Int8,                   # Why this option was flagged as rejected; c.f. :class:`REJECTED` for possible values and their meaning
    "nms": pl.Float32,                        # :class:`NMS`-computed normalized strike
    "nms5": pl.Float32,                       # :class:`NMS5`-computed normalized strike
    "shares_per_contract": pl.Int32,          # How many shared per contract, usually 100. All prices are normalized to 1, hence this is for reference only.
    "spot_adj_prev_close": pl.Float32,        # Previous close adjusted spot price (i.e. previous close adjusted for stock splits and dividends to match today's unadjusted price, which is the relevant price for moneyness and volume filtering)
    "spot_prev_close": pl.Float32,            # Previous close unadjusted spot price 
    "spot_close": pl.Float32,                 # Today's close unadjusted spot price
    "spot_open": pl.Float32,                  # Today's open unadjusted spot price
    "spot_adj_close": pl.Float32,             # Today's close adjusted spot price (i.e. today's close adjusted for stock splits and dividends to match previous close)
    "spot_ref_prev_close": pl.Float32,        # Reference previous close spot price, adjusted to today's spot. This can be used to scale today's prices without look-ahead bias.
    "strike_price": pl.Float32,               # Strike price of the option
    "ticker": pl.String,                      # Option ticker symbol
    "underlying_ticker": pl.String,           # Underlying asset ticker symbol
    "has_prev_quotes": pl.Boolean,            # Whether previous day quotes are available
    "has_prev_trades": pl.Boolean,            # Whether previous day trades are available
    "has_today_quotes": pl.Boolean,           # Whether today's quotes are available
    "has_today_trades": pl.Boolean,           # Whether today's trades are available
    "opt_open": pl.Float32,                   # Unadjusted option open price today if available
    "opt_close": pl.Float32,                  # Unadjusted option close price today if available
    "opt_prev_open": pl.Float32,              # Unadjusted option open price previous day if available
    "opt_prev_close": pl.Float32,             # Unadjusted option close price previous day if available
}
""" Schema of the :class:`polars.DataFrame` returned by :func:`filter_contracts` and :func:`filter_contracts_between`. 
See also `massive documentation <https://massive.com/docs/rest/options/contracts/contract-overview>`_ for the original field definitions.
"""

# -----------------------------
# Config
# -----------------------------


@version("0.11")
@dataclass(frozen=True)
class ContractsFilter:
    """
    Configuration on to filter the options retrieved via :func:`filter_contracts` and :func:`filter_contracts_between`.
    """
    max_dte: int = 7              #: In business days
    max_mns: float|None = 2.0     #: Maximum absolute normalized strike, i.e. $| log(K/S_\mathrm{prev})*sqrt((1+e)/(T+e)/0.1 | ~ | log(K/S_\mathrm{prev})/(0.1*sqrt(T)) |$, where $S_\mathrm{prev}$ is the previous close adjusted for stock splits and dividends to match today's unadjusted price, and T is time to expiry in years. Here $e$ is one hour. This is a measure of moneyness that accounts for both distance to the money and time to expiry.
    max_mns5 : float|None = None  #: Maximum normalized strike, i.e. $| log(K/S_\mathrm{prev})*sqrt((1+g)/(T+g)/0.1 |$, where $S_\mathrm{prev}$ is the previous close adjusted for stock splits and dividends to match today's unadjusted price, and T is time to expiry in years. Here $g$ is five days which tends to make this measure more robust than NMS alone. This is a measure of moneyness that accounts for both distance to the money and time to expiry.
    min_opt_vol_k: float = 0.0    #: Previous-day approx cash vol threshold (USD 1000's)

    def __str__(self):
        return (
            f"max_dte={self.max_dte}"+\
            ('' if self.max_mns is None else f',max_mns={self.max_mns:.3g}')+
            ('' if self.max_mns5 is None else f',max_mns5={self.max_mns5:.3g}')+
            f',min_opt_vol_k={self.min_opt_vol_k:.3g}'
        )

# -----------------------------
# Core filtering logic
# -----------------------------

@_cache(fmt=SubDir.POLARS_PARQUET).cache(
    "0.23",
    dependencies=[
        MassiveClient.list_all_contracts_expiring_after_day,
        MassiveClient.fetch_day_summary,
        ContractsFilter,
    ],
    uid=lambda underlying, d, contracts_filter, exclude_half_days, drop_no_trades_today: (
        f"{named_unique_filename48_8(f"{contracts_filter} {'xch' if exclude_half_days else ''} {'dnt' if drop_no_trades_today else ''}", contracts_filter, exclude_half_days, drop_no_trades_today)}/{underlying}/contracts/{d}"
    ),
    exclude_args=["pool_config", "client", "verbose"],
)
def filter_contracts(
    underlying: str,
    d: date,
    contracts_filter: ContractsFilter,
    exclude_half_days: bool,
    *,
    drop_no_trades_today: bool = False,
    pool_config: PoolConfig|JCPool = PoolConfig(),
    verbose: Context = Context.quiet,
) -> pl.DataFrame:
    r"""
    Filters and processes option contracts for a given underlying on a given date.

    Applies filtering rules based on expiry, moneyness, and trading volume, and returns
    a data frame with enriched contract information including reject codes and calculated metrics. This function also computes
    a correctly cash-valued adjused ``spot_ref_prev_close`` which can be used as non-forward looking normalizer
    for all cash values "today".

    Filtering Rules implemented in :class:`ContractsFilter`:

    1. Expiry filter: business days to expiry in ``[0, contracts_filter.max_dte]``.
    2. Moneyness filter: $|\log(K/S_{prev})|/\sqrt{T}/0.1$ <= contracts_filter.max_mns where ``T = days_to_exp / DAYS_PER_YEAR``.
    3. Volume filter (previous day): ``approx_cash_vol_prev_k >= contracts_filter.min_opt_vol_k`` (in 1000$).

    The returned :class:`polars.DataFrame` includes all contracts, with a "reject_code" column indicating the reason for rejection based on the above rules.
    To obtain the filtered options, use::

        from quantdata.massive.options.contracts import filter_contracts, ContractsFilter, REJECTED
        df = filter_contracts(...).filter(pl.col("reject_code") == REJECTED.USED)

    **Performance Warning**

    This function downloads the adjusted and unadjusted daily data, and trade and quote activity for each option.
    That is much slower than just obtaining the list of contracts.

    Parameters
    ----------
    underlying : str
        The underlying ticker symbol.
    d : date
        The date for which to retrieve contracts.
    contracts_filter : :class:`ContractsFilter`
        Filter configuration specifying max_dte, max_mns, and min_opt_vol_k thresholds.
    exclude_half_days : bool
        If True, exclude half trading days when calculating business days to expiry.
        This parameter is not contained in :class:`ContractsFilter` because it is
        structural to the model development.
    drop_no_trades_today : bool, default ``False``
        If True, drop options which do not trade 'today'.
        This parameter is not contained in :class:`ContractsFilter` because it is
        structural to the model development. In particular, the model logic must
        be such this feature is not forward-looking (for example by ensuring any
        model analytics only start once the first trade was observed).
    pool_config : PoolConfig, optional
        Configuration for thread pool processing. Defaults to PoolConfig().
    verbose : Context, optional
        Verbosity context for logging. Defaults to Context.quiet.

    Returns
    -------
    Result: :class:`polars.DataFrame`
        Data frame with schema :data:`SCHEMA` of filtered contracts with added fields:

        - reject_code (int): Rejection reason code
        - spot_prev_close (float): Previous close adjusted spot price
        - nms (float): Scaled moneyness metric using :class:`NMS`
        - nms5 (float): Scaled moneyness metric using :class:`NMS5`
        - approx_cash_vol_prev_m (float): Previous day approx cash volume (USD millions)
        - approx_cash_vol_today_m (float): Today approx cash volume (USD millions)
        - days_to_exp (int): Business days to expiration
        - spot_ref_prev_close (float): Previous close adjusted to match today's spot

    """
    client = MassiveClient()

    """
    Handle stock splits:
        Assume stock of 1000 was split 10:1 into 100.
        Strike today 99 is now 9.9, but the previous close was 100, so the moneyness is 0.1/0.1 = 1.0 instead of 0.01/0.1 = 0.1.
        We therefore scale yesterday's adjusted close with today's adjustment factor.

        F_today = unadjusted / adjusted
        prev_close = adjusted_prev_close * F_today
    """

    context = PrettyObject()
    context.prev_d = add_business_days( d, -1, exclude_half_days=exclude_half_days )  # previous business day (NB: if d is a Monday then this is the previous Friday, so we correctly skip the weekend)
    context.close_adj_prev = client.fetch_day_summary( underlying, context.prev_d, adjusted=True, can_be_none=False )["close"]
    context.close_unadj_prev = client.fetch_day_summary( underlying, context.prev_d, adjusted=True, can_be_none=False )["close"]
    context.close_adj = client.fetch_day_summary(underlying, d, adjusted=True, can_be_none=False)[ "close" ]
    ocv_today_unadj = client.fetch_day_summary(underlying, d, adjusted=False, can_be_none=False)
    context.close_unadj = ocv_today_unadj["close"]
    context.open_unadj = ocv_today_unadj["open"]
    F_today = context.close_unadj / context.close_adj
    context.ref_prev_close = context.close_adj_prev * F_today

    with verbose.write_t(lambda: f"\rFetching all contracts for {underlying} expiring on or after {d}... ", end="" ) as tme:
        all_contracts = client.list_all_contracts_expiring_after_day( underlying, d, max_dte=contracts_filter.max_dte )
        all_contracts = pl.DataFrame(all_contracts)
        expected_columns = {'cfi',
                            'contract_type',                        'exercise_style',
                            'expiration_date',
                            'primary_exchange',
                            'shares_per_contract',
                            'strike_price',
                            'ticker',
                            'underlying_ticker'}
        assert set(all_contracts.columns) >= expected_columns, f"{underlying} {d}: Missing expected columns in contracts data: {expected_columns - set(all_contracts.columns)}. This is unexpected, please investigate."
        if len(all_contracts) == 0:
            raise RuntimeError(f"{underlying} {d}: No contracts found with expiries on or after {d}.")  
        verbose.write(lambda: f"found {len(all_contracts)} contracts with expories form  This took {tme}.", head=False)

    all_contracts = all_contracts.with_columns(
            pl.col("expiration_date").str.strptime(pl.Date, "%Y-%m-%d", strict=True).alias("exp_date"),
            pl.lit(d).alias("date"),
            pl.lit(context.prev_d).alias("prev_bus_date"),
            pl.lit(REJECTED.USED).alias("reject_code").cast(pl.Int8),  # assume used unless rejected
            pl.lit(context.close_adj_prev).alias("spot_adj_prev_close"),
            pl.lit(context.close_unadj_prev).alias("spot_prev_close"),
            pl.lit(context.close_unadj).alias("spot_close"),
            pl.lit(context.open_unadj).alias("spot_open"),
            pl.lit(context.close_adj).alias("spot_adj_close"),
            pl.lit(context.ref_prev_close).alias("spot_ref_prev_close")  # this is the previous close, adjusted to match today's unadjusted spot     
        ).with_columns(
            pl.col("exp_date").map_batches(
                lambda exp_dates: pl.Series([
                    count_business_days_between(d, exp_date, exclude_half_days=exclude_half_days)
                    for exp_date in exp_dates
                ])
            ).alias("days_to_exp")
        ).drop("expiration_date", "underlying_ticker")

    neg_dte = all_contracts.filter(pl.col("days_to_exp") < 0)
    if len(neg_dte) > 0:
        raise RuntimeError(f"{underlying} {d}: found negative business days to expiry for dates {sorted(fmt_list(neg_dte['exp_date'].unique().to_list()))}. This should have been excluded by the Massive request filter.")

    days_to_exp = all_contracts["days_to_exp"].cast(pl.Float32).to_numpy()
    K = all_contracts["strike_price"].cast(pl.Float32).to_numpy()
    T = (days_to_exp + 1) / DAYS_PER_YEAR
    nms = NMS.to_nms(K / context.ref_prev_close, T)
    nms5 = NMS5.to_nms(K / context.ref_prev_close, T)
    rejected = all_contracts['reject_code'].to_numpy()

    import numpy as np
    rejected = np.where( days_to_exp > contracts_filter.max_dte, REJECTED.EXPIRY, rejected )
    if not contracts_filter.max_mns is None:
        rejected = np.where( (np.abs(nms) > contracts_filter.max_mns) & ~np.isnan(nms) & (rejected == REJECTED.USED), REJECTED.MONEYNESS, rejected )
    if not contracts_filter.max_mns5 is None:
        rejected = np.where( (np.abs(nms5) > contracts_filter.max_mns5) & ~np.isnan(nms5) & (rejected == REJECTED.USED), REJECTED.MONEYNESS, rejected )

    all_contracts = all_contracts.with_columns(
        pl.Series(rejected).alias("reject_code"),
        pl.Series(nms).cast(pl.Float32).alias("nms"),
        pl.Series(nms5).cast(pl.Float32).alias("nms5"),
    )

    # per-option data
    # ---------------

    def fetch_option_data( opt_ticker, d, prev_d, drop_no_trades_today ):

        timing = TrackTime("Process")
        c      = dict(opt_ticker=opt_ticker, reject_code=REJECTED.USED)  # start with used and change to rejected if any check fails
        client = MassiveClient()

        def approx_cash_vol_k(oc: dict | None) -> float | None:
            opx, cpx, vol = oc["open"], oc["close"], oc["volume"]
            return float(vol) * 0.5 * (float(opx) + float(cpx)) / 1000.0

        # ---- volume filter first: cheapest check that rejects the most contracts
        with timing("Fetch open close volume prev"):
            oc_prev = client.fetch_day_summary( opt_ticker, prev_d, adjusted=False, can_be_none=True )
            c["approx_cash_vol_prev_k"] = approx_cash_vol_k(oc_prev) if not oc_prev is None else None

        # ---- prev quotes/trades check
        with timing("Fetch quotes & trades"):
            has_prev_quotes, has_prev_trades = client.has_any_quotes_or_trades(  opt_ticker, prev_d )
            c["has_prev_quotes"] = has_prev_quotes
            c["has_prev_trades"] = has_prev_trades

        # ---- today quotes check
        with timing("Fetch quotes & trades"):
            has_today_quotes, has_today_trades = client.has_any_quotes_or_trades(opt_ticker, d)
            c["has_today_quotes"] = has_today_quotes
            c["has_today_trades"] = has_today_trades

        # ---- today volume (diagnostic only, not used for rejection)
        with timing("Fetch open close volume today"):
            oc_today = client.fetch_day_summary(opt_ticker, d, adjusted=False, can_be_none=True)
            assert not oc_today is None or not has_today_trades, f"If there are trades today, there should be a day summary with volume data. This is unexpected, please investigate: opt_ticker={opt_ticker}, d={d}, has_today_trades={has_today_trades}"
        with timing("Fetch open close volume yesterday"):
            oc_yesterday = None
            try:
                oc_yesterday = client.fetch_day_summary(opt_ticker, prev_d, adjusted=False, can_be_none=False) if has_prev_trades else None
            except Exception as e:
                if has_prev_trades:
                    raise type(e)(f"Failed to fetch yesterday's open/close/volume for {opt_ticker} on {prev_d}: {e} despite has_prev_trades={has_prev_trades}") from e

        c["approx_cash_vol_today_k"] = approx_cash_vol_k(oc_today) if not oc_today is None else None
        c['opt_open'] = oc_today['open'] if not oc_today is None else None
        c['opt_close'] = oc_today['close'] if not oc_today is None else None
        c['opt_prev_open'] = oc_yesterday['open'] if oc_yesterday is not None else None
        c['opt_prev_close'] = oc_yesterday['close'] if oc_yesterday is not None else None

        if not has_prev_quotes or not has_prev_trades:
            c["reject_code"] = REJECTED.PREV_NO_QUOTES_OR_TRADES

        elif oc_prev is None:
            c["reject_code"] = REJECTED.PREV_VOLUME_MISSING

        elif c["approx_cash_vol_prev_k"] < contracts_filter.min_opt_vol_k:
            c["reject_code"] = REJECTED.VOLUME

        elif not has_today_quotes:
            c["reject_code"] = REJECTED.NO_TODAY_QUOTES

        elif not has_today_trades and drop_no_trades_today:
            c["reject_code"] = REJECTED.NO_TODAY_TRADES

        elif not has_today_quotes:
            c["reject_code"] = REJECTED.NO_TODAY_QUOTES

        elif not has_today_trades and drop_no_trades_today:
            c["reject_code"] = REJECTED.NO_TODAY_TRADES

        return opt_ticker, c, timing

    opt_tickers = all_contracts.filter(pl.col("reject_code") == REJECTED.USED)["ticker"].to_list()

    jcpool = pool_config.pool()
    used_ticker_data = []
    tme = verbose.timer()
    timing = None
    for opt_ticker, opt_data, ctiming in jcpool.parallel( jcpool.delayed(fetch_option_data)(opt_ticker, d, prev_d=context.prev_d, drop_no_trades_today=drop_no_trades_today ) for opt_ticker in opt_tickers ):
        timing = ctiming if timing is None else (timing.add(ctiming))
        if tme.interval_test(1):
            verbose.report( 1, lambda: f"\rFiltering {len(used_ticker_data) + 1}/{len(all_contracts)} contracts: {int((len(used_ticker_data) + 1) * 100 // len(all_contracts))}%: {opt_ticker}...", end="" )
        used_ticker_data.append(opt_data)
    if len(used_ticker_data) == 0:
        return None

    used_ticker_data = pl.DataFrame(used_ticker_data)
    expted_columns = {'opt_ticker','reject_code','approx_cash_vol_prev_k','approx_cash_vol_today_k','has_prev_quotes','has_prev_trades','has_today_quotes','has_today_trades', 'opt_open', 'opt_close', 'opt_prev_open', 'opt_prev_close'}
    assert set(used_ticker_data.columns) == expted_columns, f"{underlying} {d}: Missing expected columns in used ticker data: {expted_columns - set(used_ticker_data.columns)}. This is unexpected, please investigate."

    all_contracts = all_contracts.join( used_ticker_data, left_on="ticker", right_on="opt_ticker", how="left" )
    assert "reject_code_right" in all_contracts.columns, f"{underlying} {d}: Missing 'reject_code_right' column in all_contracts after join. This is unexpected, please investigate."
    assert not "opt_ticker" in all_contracts.columns, f"{underlying} {d}: 'opt_ticker' column should not be in all_contracts after join. This is unexpected, please investigate."

    all_contracts = all_contracts.with_columns(
        pl.when(pl.col("reject_code_right").is_not_null())
        .then(pl.col("reject_code_right"))
        .otherwise(pl.col("reject_code"))
        .alias("reject_code"),
        pl.lit(underlying).alias("underlying_ticker")
    ).drop("reject_code_right")

    assert set(all_contracts.columns) >= set(SCHEMA.keys()), f"DataFrame is missing columns from SCHEMA: {sorted(set(SCHEMA.keys()) - set(all_contracts.columns))}"
    assert set(SCHEMA.keys()) >= set(all_contracts.columns) >= set(SCHEMA.keys()), f"DataFrame has superfluous columns: {sorted(set(all_contracts.columns) - set(SCHEMA.keys()))}"
    all_contracts = all_contracts.with_columns([pl.col(col).cast(dtype) for col, dtype in SCHEMA.items()])

    all_contracts = all_contracts.sort("exp_date","strike_price","contract_type")
    active = all_contracts.filter(pl.col("reject_code") == REJECTED.USED)
    

    timestr = ""
    for topic, tinfo in timing.items():
        timestr += f"{topic}: {tinfo.average_lap_seconds:.2f}s, "
    timestr = timestr[:-2]

    all_expiries = all_contracts["exp_date"].unique().to_list()

    verbose.write(
        lambda: (
            f"\rFiltered {fmt_digits(len(all_contracts))} contacts for {underlying} on {fmt_date(d)}, "
            + f"yielding {fmt_digits(active.height)} active contracts. This took {tme}. "
            + (f"Found {len(all_expiries)} expiries ranging from {all_expiries[0]} to {all_expiries[-1]}. " if len(all_expiries) > 0 else f"Found one expiry, {all_expiries[0]}. ")
            + f"Timing: {timestr}."
        ),
        head=False,
    )
    return all_contracts

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
from quantdata.massive import _cli_defaults


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option(
    "--period",
    nargs=2,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=_cli_defaults.period,
    metavar="START END",
    help="Period YYYY-MM-DD YYYY-MM-DD",
)
@click.option("--halfdays", is_flag=True, help="Include half days")
@click.option("--workers", default=_cli_defaults.workers, help="Number of workers", type=int)
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
