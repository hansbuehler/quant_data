"""
Generating one clean day of data
"""
from __future__ import annotations

from pandas import arrays
from yfinance import tickers

from quantdata.widb import WIDBClient, interpolate_zero_curve
from quantdata.biz_calendar import count_business_days_between, DAYS_PER_YEAR, AM_EXIRY_SUBTRACT, to_date, NMS
from quantdata.widb.client import WIDBClient
from quantdata.widb.bad import BAD
from quantdata.pool import PoolConfig

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, CacheMustExistError
from cdxcore.util import fmt_date, fmt_list, fmt_dict, fmt_big_byte_number
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import named_unique_filename48_8, unique_hash8
from cdxcore.jcpool import JCPool

import datetime as dt
import numpy as np
import pandas as pd
import math as math
import warnings
from prettytable import PrettyTable
from dataclasses import dataclass, field

_processing = WIDBClient.locations.processing("day",fmt=SubDir.PANDAS_PARQUET)
_training   = WIDBClient.locations.training("block",fmt=SubDir.PANDAS_PARQUET)

def _create_empty_df( schema ):
    return pd.DataFrame({c: pd.Series(dtype=dtype) for c,dtype in schema.items()})

# ---------------------------------------------------------------------------
# Raw data, day
# Minimal analytics and clean up.
# ---------------------------------------------------------------------------

SCHEMA_RAW_DAY = {
    'ticker' : np.dtype('O'),
    'date' : np.dtype("datetime64[s]"),  # [D] not supported in storage
    'spot' : np.float32,
    'call_symbol' : np.dtype("O"),
    'put_symbol' : np.dtype("O"),
    'call_opt_id' : np.int32,
    'put_opt_id' : np.int32,
    'expiry_id' : np.int32,
    'is_american' : np.bool_,
    'is_am' : np.bool_,
    'strike' : np.float32,
    'fwd_ref' : np.float32,
    'df' : np.float32,
    'expiry' : np.dtype("datetime64[s]"),  # [D] not supported in storage``
    'bdte' : np.float32, # this is a float because of AM options.
    'tte_y' : np.float32, 
    'mkt_call_ask' : np.float32, 
    'mkt_call_bid' : np.float32, 
    'mkt_call_log_volume' : np.float32, 
    'mkt_call_log_open_interest' : np.float32, 
    'mkt_put_ask' : np.float32,
    'mkt_put_bid' : np.float32, 
    'mkt_put_log_volume' : np.float32,
    'mkt_put_log_open_interest' : np.float32, 
}

@version("0.2")
@dataclass(frozen=True)
class RawDayFilter:
    min_bid : float = 0.011  # minimum bid value, typically above one tick.

    def __post_init__(self):
        verify_inp(self.min_bid >= 0., lambda : f"'min_bid' must be non-negative, found {self.min_bid:.3f}")

    def __str__(self):
        return f"bid={self.min_bid:.3f}"

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.5",
            dependencies=[interpolate_zero_curve, WIDBClient],
            uid=lambda ticker, d, raw_day_filter : f"{named_unique_filename48_8(f'raw {str(raw_day_filter)}',raw_day_filter)}/{ticker}/{fmt_date(d)}",
            exclude_args=['verbose','raise_if_bad'])
def collect_raw_day( ticker : str, 
                     d : dt.date, 
                     raw_day_filter : RawDayFilter, *, 
                     raise_if_bad : bool = True,
                     verbose : Context, secid : int|None = None ) -> pd.DataFrame:
    """
    Collect raw data with minimal analytics for a given ticker and date.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    d : :class:`datetime.date`
        Date for which to collect data
    config : :class:`RawDayFilter`
        Configuration for data collection
    raise_if_bad : bool, default ``True``
        If ``True``, raises an error if the date is in the BAD data sets. If ``False``, returns an empty data set instead. (In no other condition will an empty data set returned)
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging

    Returns
    -------
    Daily data: :class:`pandas.DataFrame`
        Cleaned DataFrame with all relevant data for the given ticker and date, with the schema :attr:`SCHEMA_RAW_DAY`.
        This data set is not empty unless ``raise_if_bad`` is True and the date/ticker is in the BAD data sets.

    Raises
    ------
    Not found: :class:`LookupError`
        If no data was downloaded from IvyDB for the given ticker and date.
    No data: :class:`EOFError`
        If no data was available for the given ticker and date.
    Bad day: :class:`IndexError`
        If this date and ticker are known to have bad data; see :class:`quantdata.widb.bad.BAD` for details.
    Bad data: :class:`RuntimeError`
        If data was downloaded but found to be bad after basic cleaning steps.
        This can happen for example if the forward price is highly suspicious, or if all options had zero or NA bid prices. Please check the date range and calendar settings,
        and consider adding the date/ticker to the BAD data sets if the issue is confirmed.
    """

    if d in BAD.DAYS_ALL:
        if not raise_if_bad:
            return _create_empty_df(SCHEMA_RAW_DAY)
        raise IndexError(f"{d} is in the BAD.DAYS_ALL set. No data will be downloaded for this date. Please check the date range and calendar settings. BAD.DAYS_ALL={BAD.DAYS_ALL}")
    if d in  BAD.DAYS_TICKER.get(ticker, set()):
        if not raise_if_bad:
            return _create_empty_df(SCHEMA_RAW_DAY)
        raise IndexError(f"{d} is in the BAD.DAYS_TICKER set for {ticker}. No data will be downloaded for this date. Please check the date range and calendar settings. BAD.DAYS_TICKER[{ticker}]={BAD.DAYS_TICKER.get(ticker, set())}")

    conn        = WIDBClient()
    secid       = conn.lookup_secid(ticker)

    with verbose.write_t(f"Collecting raw market data for {ticker} on {d}.") as tme:
        try:    
            spot    = conn.spot_price(secid, d, verbose=verbose(1))
        except LookupError as e:
            raise LookupError(f"Failed to fetch spot price for {ticker} on {d}. "+\
                            f"This likely means either the corresponding secid {secid} is not covered by Option Metrics, or {d} is not a business day for this underlying. Please check the date range and calendar settings.") from e
        zero_curve  = conn.zero_curve(d, verbose=verbose(1))
        opts_pd     = conn.options(secid, d, verbose=verbose(1))
        fwd_table   = conn.forwards(secid, d, verbose=verbose(1))

        assert isinstance(zero_curve, pd.DataFrame), f"Internal error: expected zero_curve to be a pandas DataFrame, found {type(zero_curve)}"
        assert isinstance(opts_pd, pd.DataFrame), f"Internal error: expected opts_pd to be a pandas DataFrame, found {type(opts_pd)}"
        assert isinstance(fwd_table, pd.DataFrame), f"Internal error: expected fwd_table to be a pandas DataFrame, found {type(fwd_table)}"

        if opts_pd.empty:
            raise EOFError(f"No option data found for {ticker} on {d}. Please check the date range and calendar settings.") 
        if zero_curve.empty:
            raise EOFError(f"No zero-curve data for {d}")
        if spot is None:
            raise EOFError(f"No spot price for {ticker} on {d}")
        if fwd_table.empty:
            raise EOFError(f"No forward price data for {ticker} on {d}")
    
    # Remove bad options
    # ------------------
    # exclude known bad options, either in totality (by symbol) or per date

    status_removed = []
    n_total = len(opts_pd)

    def remove_bad( status_removed, opts_pd ):
        bad_options   =  BAD.OPTIONS.get(ticker,set()) | BAD.OPTIONS_BY_DATE.get(ticker,{}).get(d, set())
        if len(bad_options) > 0:
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~opts_pd["symbol"].astype(str).isin(bad_options)].copy()
            n_removed = n_before - len(opts_pd) 
            if opts_pd.empty:
                raise EOFError(f"No option data found for {ticker} on {d} after having removed {n_removed} bad options {fmt_list(sorted(bad_options))}. Please check BAD.OPTIONS and the date range.") 
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} 'bad' options")
        return status_removed, opts_pd
    status_removed, opts_pd = remove_bad(status_removed, opts_pd)

    # Remove same day expiry
    # ----------------------

    def remove_same_day_expiry( status_removed, opts_pd ):
        opts_pd['exdate']    = pd.to_datetime(opts_pd['exdate']).dt.date
        same_day_expiry_mask = opts_pd['exdate'] == d
        if same_day_expiry_mask.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~same_day_expiry_mask].copy()
            n_removed = n_before - len(opts_pd)
            if opts_pd.empty:
                raise EOFError(f"{ticker} {d}: all option rows expired today and were dropped.")
            if n_removed > 0:   
                status_removed.append(f"{n_removed}/{n_before} options with same-day expiry") 
        return status_removed, opts_pd
    status_removed, opts_pd = remove_same_day_expiry(status_removed, opts_pd)   
        
    # remove NA quotes
    # ----------------

    def remove_na_quotes( status_removed, opts_pd ):
        na_quotes = opts_pd["best_bid"].isna() | opts_pd["best_offer"].isna()
        if na_quotes.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~na_quotes].copy()
            n_removed = n_before - len(opts_pd) 
            if opts_pd.empty:
                raise EOFError(f"No option data found for {ticker} on {d} after removing {n_removed}/{n_before} options with NA bid or ask. Please check the date range and calendar settings.")
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} options with NA bid or ask")
        return status_removed, opts_pd
    status_removed, opts_pd = remove_na_quotes(status_removed, opts_pd)

    # remove crossed quotes
    # ---------------------

    def remove_crossed_quotes( status_removed, opts_pd ):
        crossed_quotes = (
            ( opts_pd["best_bid"].fillna(0.0).astype(float)
            >= opts_pd["best_offer"].fillna(0.0).astype(float) )
        )
        if crossed_quotes.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~crossed_quotes].copy()
            n_removed = n_before - len(opts_pd) 
            if opts_pd.empty:
                raise EOFError(
                    f"No option data found for {ticker} on {d} after removing {n_removed}/{n_before} options with bid >= ask."
                )
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} options with bid >= ask")
        return status_removed, opts_pd
    status_removed, opts_pd = remove_crossed_quotes(status_removed, opts_pd)

    # remove zero quotes
    # ------------------

    def remove_zero( status_removed, opts_pd ):
        zero_quotes = (
            opts_pd["best_bid"].fillna(0.0).astype(float) <= raw_day_filter.min_bid
        ) 
        if zero_quotes.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~zero_quotes].copy()
            n_removed = n_before - len(opts_pd) 
            if opts_pd.empty:
                raise EOFError(
                    f"No option data found for {ticker} on {d} after removing {n_removed}/{n_before} options with bid <= {raw_day_filter.min_bid}."
                )
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} options with bid <= {raw_day_filter.min_bid}")
        return status_removed, opts_pd
    status_removed, opts_pd = remove_zero(status_removed, opts_pd)

    # remove bad settlement
    # ---------------------
        
    def remove_bad_ss( status_removed, opts_pd ):
        n_ss_bad = opts_pd['ss_flag'].values != "0"
        if n_ss_bad.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~n_ss_bad].copy()#
            n_removed = n_before - len(opts_pd)
            if opts_pd.empty:
                raise EOFError(
                    f"No option data found for {ticker} on {d} after removing {n_removed}/{n_before} options with ss_flag != '0'."
                )
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} options with bad settlement (ss_flag != '0')")
        return status_removed, opts_pd
    status_removed, opts_pd = remove_bad_ss(status_removed, opts_pd)            

    # implied df
    # ----------

    def df_ref( opts_pd, zero_curve ):
        opts_pd = opts_pd.assign(
            discount_factor_ref=lambda frame: interpolate_zero_curve(
                zero_curve,
                np.array(
                    [
                        ( pd.Timestamp(exdate).date() - d).days 
                        for exdate in frame["exdate"]
                    ],
                    dtype=float,
                ),
            ))
        assert np.min( opts_pd["discount_factor_ref"].values) > 0., f"Internal error {ticker} {d}: all implied discount factors must be positive"
        return opts_pd
    opts_pd = df_ref(opts_pd, zero_curve)

    # computed implied forward
    # ------------------------

    def forward_ref( opts_pd, fwd_table ):
        # Forward prices by expiry (and AM/PM settlement).
        # fwd_table is small (one row per expiry/settlement), opts_pd is large.
        # Use a MultiIndex reindex (O(n)) rather than a per-row lookup.
        # Note that IvyDB forward prices are highyl suspicious.
        # We treat them as "forward_ref" only
        assert not fwd_table.empty, "Internal error"
        if "am_settlement" not in fwd_table.columns:
            fwd_table["am_settlement"] = 0.0
        # Keep only what we need for mapping (avoids dtype edge-cases/warnings).
        fwd_table["exdate"] = pd.to_datetime(fwd_table["exdate"]).dt.date
        fwd_table["am_settlement"] = fwd_table["am_settlement"].fillna(0.0).astype(float)
        fwd_table["forward_price"] = fwd_table["forward_price"].astype(float)   

        # remove today since that day was removed also from the options table
        fwd_table         = fwd_table.loc[fwd_table["exdate"] != d]
        bad_forward_mask  = fwd_table["forward_price"] == spot / 1_000_000.0
        if bad_forward_mask.any():
            w = f"{ticker} {d}: found {bad_forward_mask.sum()} forward_ref values equal to spot/1_000_000 ({spot/1_000_000.0}). Replacing those with spot ({spot})."
            verbose.write(fwd_table)
            verbose.write(w)
            warnings.warn(w)
            fwd_table.loc[bad_forward_mask, "forward_price"] = float(spot)

        if (fwd_table['forward_price'].min() < spot/2.) or (fwd_table['forward_price'].max() > spot*2.):
            err = f"{ticker} {d}: suspicious forward prices in fwd_table, min={fwd_table['forward_price'].min()}, max={fwd_table['forward_price'].max()} spot={spot}. Fix or add the date/expiry to the BAD data sets."
            verbose.write(err)
            verbose.write(fwd_table)
            raise RuntimeError(err)      

        # Enforce one forward per (exdate, am_settlement)
        dup_mask = fwd_table.duplicated(subset=["exdate", "am_settlement"], keep=False)
        if dup_mask.any():
            dup_keys = (
                fwd_table.loc[dup_mask, ["exdate", "am_settlement"]]
                .drop_duplicates()
                .head(10)
                .to_records(index=False)
                .tolist()
            )
            raise RuntimeError(
                f"{ticker} {d}: forward table has duplicate keys (exdate, am_settlement). "
                f"Expected exactly one forward per key. Sample duplicates: {dup_keys}. Fix or add the date/expiry to the BAD data sets."
            )
            
        # write forward_ref into option table
        fwd_ref_by_key = fwd_table.set_index(["exdate", "am_settlement"])["forward_price"].astype(float)
        opts_pd["forward_ref"] = fwd_ref_by_key.reindex(
            pd.MultiIndex.from_frame(opts_pd[["exdate", "am_settlement"]], names=["exdate", "am_settlement"])
        ).to_numpy(dtype=float)

        return opts_pd
    opts_pd = forward_ref(opts_pd, fwd_table)

    # Enrich
    # --------------

    def enrich( opts_pd ):
        opts_pd["log_volume"] = np.log10(1+opts_pd["volume"].fillna(0.0).astype(float))
        opts_pd["log_open_interest"] = np.log10(1+opts_pd["open_interest"].fillna(0.0).astype(float))
        opts_pd['is_call'] = opts_pd["cp_flag"].astype("string").str.upper().eq("C")
        opts_pd['is_american'] = opts_pd["exercise_style"].fillna("E").astype("string").str.upper().eq("A")
        opts_pd['is_am']   = opts_pd["am_settlement"].fillna(0.0).astype(float) == 1.0
        opts_pd['strike']  = opts_pd["strike_price"].astype(float)

        opts_pd.drop(columns=["volume", "open_interest", "cp_flag", "exercise_style", "am_settlement", "strike_price"], inplace=True)

        # find opt calls for the same exdate, am_settlement, strike 
        opt_calls = opts_pd[opts_pd["is_call"]].copy()
        opt_puts  = opts_pd[~opts_pd["is_call"]].copy()
        opt_calls = opt_calls.sort_values(by=["exdate", "is_am", "strike", "is_american"])
        opt_puts  = opt_puts.sort_values(by=["exdate", "is_am", "strike", "is_american"])
        del opts_pd

        opt_calls = opt_calls.rename(columns={
            "symbol": "call_symbol",
            "optionid": "call_opt_id",
            "best_bid": "call_bid",
            "best_offer": "call_ask",
            "log_volume": "call_log_volume",
            "log_open_interest": "call_log_open_interest",
        })
        opt_puts  = opt_puts.rename(columns={
            "symbol": "put_symbol",
            "optionid": "put_opt_id",
            "best_bid": "put_bid",
            "best_offer": "put_ask",
            "log_volume": "put_log_volume",
            "log_open_interest": "put_log_open_interest",
        })
        opt_calls = opt_calls[["exdate", "is_am", "strike", "is_american", "call_symbol", "call_opt_id", "call_bid", "call_ask", "call_log_volume", "call_log_open_interest"]+\
                              ["discount_factor_ref", "forward_ref"]]
        opt_puts  = opt_puts[["exdate", "is_am", "strike", "is_american", "put_symbol", "put_opt_id", "put_bid", "put_ask", "put_log_volume", "put_log_open_interest"]]

        opts_pd = pd.merge(
            opt_calls,
            opt_puts,
            on=["exdate", "is_am", "is_american", "strike"],
            suffixes=("", ""),
        )

        # write into a "dict"
        # then convert to panda

        r = PrettyObject()
        r.spot = np.float32(spot)
        r.date = np.array(d, dtype="datetime64[s]")
        r.ticker = ticker
        r.call_symbol = opts_pd["call_symbol"].astype(str).values.tolist()
        r.put_symbol = opts_pd["put_symbol"].astype(str).values.tolist()
        r.call_opt_id = opts_pd["call_opt_id"].to_numpy(dtype=np.int32)
        r.put_opt_id = opts_pd["put_opt_id"].to_numpy(dtype=np.int32)
        r.is_american = opts_pd["is_american"].to_numpy(dtype=bool)
        r.is_am = opts_pd["is_am"].to_numpy(dtype=bool)
        r.expiry = np.array(opts_pd["exdate"], dtype="datetime64[s]")
        r.strike = opts_pd["strike"].values.astype(np.float32)
        r.fwd_ref = opts_pd["forward_ref"].values.astype(np.float32)
        r.df = opts_pd["discount_factor_ref"].values.astype(np.float32)

        r.mkt_call_bid = opts_pd["call_bid"].values.astype(np.float32)
        r.mkt_put_bid = opts_pd["put_bid"].values.astype(np.float32)
        r.mkt_call_ask = opts_pd["call_ask"].values.astype(np.float32)
        r.mkt_put_ask = opts_pd["put_ask"].values.astype(np.float32)
        r.mkt_call_log_volume = opts_pd["call_log_volume"].values.astype(np.float32)
        r.mkt_put_log_volume = opts_pd["put_log_volume"].values.astype(np.float32)
        r.mkt_call_log_open_interest = opts_pd["call_log_open_interest"].values.astype(np.float32)
        r.mkt_put_log_open_interest = opts_pd["put_log_open_interest"].values.astype(np.float32)

        # Option ID as string symbol
        expiry_dates = pd.to_datetime(r.expiry)
        r.expiry_id = (
            expiry_dates.year * 100000
            + expiry_dates.month * 1000
            + expiry_dates.day * 10
            + np.where(r.is_am, 1, 0)
        ).astype(np.int32)

        r.bdte = count_business_days_between(d, expiry_dates.to_numpy(dtype="datetime64[D]")).astype(np.float32)  # returns integers
        r.bdte = r.bdte - np.where( r.is_am, AM_EXIRY_SUBTRACT, 0. ).astype(np.float32) # adjust for AM settlement if needed
        assert np.min(r.bdte) > 0., f"Internal error: {ticker} {d}: found non-positive business days to expiry after adjusting for AM  treatment: min={np.min(r.bdte)}. Please contact the authors with an example. {expiry_dates} / {r.bdte}"
        r.tte_y = r.bdte / DAYS_PER_YEAR
        return pd.DataFrame(dict(r))

    result = enrich(opts_pd)
    if set(result.columns) != set(SCHEMA_RAW_DAY):
        too_much = set(result.columns) - set(SCHEMA_RAW_DAY)
        too_little = set(SCHEMA_RAW_DAY) - set(result.columns)
        raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame has unexpected columns. Superflous: {fmt_list(too_much)}. Missing: {fmt_list(too_little)}. Please contact the authors with an example. ")

    for c,expected_dtype in SCHEMA_RAW_DAY.items():
        actual_dtype = result[c].dtype
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' has unexpected dtype. Expected {expected_dtype}, found {actual_dtype}. Please contact the authors with an example.")

    verbose.write(f"{ticker} {d}: collected {len(result)}/{n_total} options. Removed: {fmt_list(status_removed) if len(status_removed) > 0 else 'none'}.")
    return result

# ---------------------------------------------------------------------------
# Enriched data, day
# Optimal forward computation, implied volatilities, (pure) greeks.
# Also translates everything in calls.
# ---------------------------------------------------------------------------

SCHEMA_ENRICHED_DAY = SCHEMA_RAW_DAY | {
    'fwd' : np.dtype("float32"),          # implied forward from put-call parity
    'nms' : np.dtype("float32"),          # normalized moneyness = strike / forward
    'call_ask_ref' : np.dtype("float32"), # actual synthetic call ask
    'call_bid_ref' : np.dtype("float32"), # actual synthetic call bid
    'call_ask' : np.dtype("float32"),     # implied from pure call after implied vol fitting
    'call_bid' : np.dtype("float32"),     # implied from pure call after implied vol fitting
    'ask_vol' : np.dtype("float32"),
    'bid_vol' : np.dtype("float32"),
    'log_volume' : np.dtype("float32"),
    'log_open_interest' : np.dtype("float32"),
    'ask_pure_price' : np.dtype("float32"),
    'bid_pure_price' : np.dtype("float32"),
    'ask_pure_delta' : np.dtype("float32"),
    'bid_pure_delta' : np.dtype("float32"),
    'ask_pure_gamma' : np.dtype("float32"),
    'bid_pure_gamma' : np.dtype("float32"),
    'bid_pure_vega' : np.dtype("float32"),
    'ask_pure_vega' : np.dtype("float32"),
    'bid_pure_theta' : np.dtype("float32"),
    'ask_pure_theta' : np.dtype("float32"),
    'ask_pure_dk' : np.dtype("float32"),
    'bid_pure_dk' : np.dtype("float32")
}

@version("0.2")
@dataclass(frozen=True)
class ExpiryFilter:
    """ How to filter expiries to be able to compute implied forwards """
    num_min_either_traded : int = 10
    num_min_both_traded : int = 4

    def __post_init__(self):
        verify_inp(self.num_min_either_traded >= 0, lambda : f"'num_min_either_traded' must be non-negative, found {self.num_min_either_traded}")
        verify_inp(self.num_min_both_traded >= 0, lambda : f"'num_min_both_traded' must be non-negative, found {self.num_min_both_traded}")

    def __str__(self):
        return f"eth={self.num_min_either_traded},bth={self.num_min_both_traded}"

def _uid(ticker, d, raw_day_filter, expiry_filter) -> str:
    return f"{named_unique_filename48_8(f'enrich {str(expiry_filter)},{str(raw_day_filter)}',expiry_filter,raw_day_filter)}"+\
           f"/{ticker}/{fmt_date(d)}"

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.3",
            dependencies=[collect_raw_day, ExpiryFilter],
            uid=_uid,
            exclude_args=['verbose','data','raise_if_bad','refresh_cache'])
def enrich_day( ticker : str, d : dt.date, data : pd.DataFrame|None = None, *, 
                raw_day_filter : RawDayFilter = RawDayFilter(), 
                expiry_filter : ExpiryFilter = ExpiryFilter(), 
                refresh_cache : bool = False,
                raise_if_bad : bool = True,
                verbose : Context = Context.quiet ):
    """
    Enrich a day's data by:

    * Compute synthetic minimal spread call options.
    * Compute an implied forward from inverse-spread weighted bid/ask call prices.
    * Compute implied volatilities and (pure) greeks.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    d : :class:`datetime.date`
        Date for which to collect data
    data : :class:`pandas.DataFrame` | None, default ``None``
        Raw data for the given ticker and date. If ``None``, data will be generated
        using :func:`collect_raw_day`.
    raw_day_filter : :class:`RawDayFilter`
        Configuration for data collection
    expiry_filter : :class:`ExpiryFilter`
        Filter for expiries to be able to compute implied forwards
    refresh_cache : bool, default ``False``
        If ``True``, forces refreshing the cache for the raw day data.
    raise_if_bad : bool, default ``True``
        If ``True``, raises an error if the date is in the BAD data sets.
        If ``False``, returns an empty data set instead. (In no other condition will an empty data set returned)
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging

    Returns
    -------
    Daily data: :class:`pandas.DataFrame`
        Cleaned DataFrame with all relevant data for the given ticker and date, creating the schema :attr:`SCHEMA_ENRICHED_DAY`.

    Raises
    ------
    Not found: :class:`LookupError`
        If no data was downloaded from IvyDB for the given ticker and date.
    No data: :class:`EOFError`
        If no data was available for the given ticker and date.
    Bad day: :class:`IndexError`
        If this date and ticker are known to have bad data; see :class:`quantdata.widb.bad.BAD` for details.
    Bad data: :class:`RuntimeError`
        If data was downloaded but found to be bad after basic cleaning steps.
        This can happen for example if the forward price is highly suspicious, or if all options had zero or NA bid prices. Please check the date range and calendar settings,
        and consider adding the date/ticker to the BAD data sets if the issue is confirmed.
    """

    data         = data if data is not None else collect_raw_day(ticker, d, raw_day_filter, verbose=verbose, override_cache_mode="update" if refresh_cache else None, raise_if_bad = raise_if_bad)

    if data.empty:
        return _create_empty_df(SCHEMA_ENRICHED_DAY)

    expiry_ids   = data["expiry_id"].to_numpy()
    unq_ids      = np.unique(expiry_ids)
    num_total    = len(data)
    num_expiries = len(unq_ids)

    put_log_vol  = data['mkt_put_log_volume'].to_numpy().astype(np.float64)
    call_log_vol = data['mkt_call_log_volume'].to_numpy().astype(np.float64)
    put_log_oi   = data['mkt_put_log_open_interest'].to_numpy().astype(np.float64)
    call_log_oi  = data['mkt_call_log_open_interest'].to_numpy().astype(np.float64)

    either_cp   = (put_log_vol>0)|(call_log_vol>0)
    both_cp     = (put_log_vol>0)&(call_log_vol>0)
    fwd_ref     = data['fwd_ref'].to_numpy().astype(np.float64)
    df_ref      = data['df'].to_numpy().astype(np.float64)
    strike      = data['strike'].to_numpy().astype(np.float64)
    expiries    = data['expiry'].to_numpy()
    is_am       = data['is_am'].to_numpy()

    mkt_call_ask = data['mkt_call_ask'].to_numpy().astype(np.float64)
    mkt_call_bid = data['mkt_call_bid'].to_numpy().astype(np.float64)
    mkt_put_ask  = data['mkt_put_ask'].to_numpy().astype(np.float64)
    mkt_put_bid  = data['mkt_put_bid'].to_numpy().astype(np.float64)

    dffwd_ask    = ( mkt_call_ask - mkt_put_bid ) + df_ref*strike   # C-P=DF(FWD-strike)
    dffwd_bid    = ( mkt_call_bid - mkt_put_ask ) + df_ref*strike
    dffwd_mid    = 0.5*(dffwd_ask + dffwd_bid)
    dffwd_wght   = np.where( both_cp, 1./np.maximum( dffwd_ask - dffwd_bid, 0.001 ), 0. )

    # compute dffwd_mean grouped by expiry_id
    arrays                 = PrettyObject()    
    arrays.fwd             = np.zeros_like( dffwd_ask, dtype=np.float32 )
    arrays.nms             = np.zeros_like( dffwd_ask, dtype=np.float32 )
    arrays.is_valid        = np.zeros_like( arrays.fwd, dtype=bool )
    arrays.call_ask        = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.call_bid        = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.call_ask_ref    = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.call_bid_ref    = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_vol         = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_vol         = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.log_volume      = np.log10( np.pow(10.,call_log_vol.astype(np.float64))+np.pow(10.,put_log_vol.astype(np.float64))-1 ).astype(np.float32)
    arrays.log_open_interest = np.log10( np.pow(10.,call_log_oi.astype(np.float64))+np.pow(10.,put_log_oi.astype(np.float64))-1 ).astype(np.float32)
    arrays.ask_pure_price  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_price  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_pure_delta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_delta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_pure_gamma  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_gamma  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_vega   = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_pure_vega   = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_theta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_pure_theta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_pure_dk     = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_pure_dk     = np.zeros_like( arrays.fwd, dtype=np.float32 )

    # project
    dropped_expiries = []

    for expiry_id in unq_ids:
        mask = expiry_ids == expiry_id
        if np.sum( both_cp[mask] ) < expiry_filter.num_min_both_traded:
            arrays.fwd[mask] = fwd_ref[mask]
            iexpory = expiries[mask][0]
            iiam    = is_am[mask][0]
            dropped_expiries.append(f"{fmt_date(to_date(iexpory))}{'am' if iiam else 'pm'}")
            continue
        if np.sum(either_cp[mask]) < expiry_filter.num_min_either_traded:
            arrays.fwd[mask] = fwd_ref[mask]
            iexpory = expiries[mask][0]
            iiam    = is_am[mask][0]
            dropped_expiries.append(f"{fmt_date(to_date(iexpory))}{'am' if iiam else 'pm'}")
            continue
        dffwd_t      = np.sum(dffwd_mid[mask] * dffwd_wght[mask]) / np.sum(dffwd_wght[mask]) # weighted forward for this expiry_id
        df_t         = df_ref[mask][0] # same for all rows 
        fwd_t        = dffwd_t / df_t
        tte_yt       = data['tte_y'].to_numpy()[mask][0]
        sqrtT_t      = math.sqrt(tte_yt)
        arrays.fwd[mask]      = fwd_t
        arrays.is_valid[mask] = True

        # synthetic call price from put-call parity using the projected forward, for comparison with the market call price. 
        sy_call_ask        = mkt_put_ask[mask] + dffwd_t - df_t * strike[mask] 
        sy_call_bid        = mkt_put_bid[mask] + dffwd_t - df_t * strike[mask]

        call_mid           = 0.5 * ( np.minimum( sy_call_ask, mkt_call_ask[mask]) + np.maximum( sy_call_bid, mkt_call_bid[mask]) )
        call_spread        = np.minimum( mkt_call_ask[mask] - mkt_call_bid[mask], sy_call_ask - sy_call_bid )
        new_call_ask       = call_mid + 0.5*call_spread
        new_call_bid       = call_mid - 0.5*call_spread
        pure_strike        = strike[mask] / fwd_t
        pure_call_ask      = new_call_ask / dffwd_t
        pure_call_bid      = new_call_bid / dffwd_t
        ask_vols           = bs.implied( pure_strike, pure_call_ask, sqrtT = sqrtT_t, eps=1E-4 )
        bid_vols           = bs.implied( pure_strike, pure_call_bid, sqrtT = sqrtT_t, eps=1E-4 )

        arrays.call_ask_ref[mask]   = new_call_ask
        arrays.call_bid_ref[mask]   = new_call_bid
        arrays.ask_vol[mask]        = ask_vols
        arrays.bid_vol[mask]        = bid_vols

        bid_greeks               = bs( k=pure_strike, vol=bid_vols, sqrtT = sqrtT_t, eps=1E-4, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )
        ask_greeks               = bs( k=pure_strike, vol=ask_vols, sqrtT = sqrtT_t, eps=1E-4, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )

        def copy(which, src):
            for k,v in src.items():
                arrays[f"{which}_pure_{k}"][mask] = v
        copy("bid", bid_greeks) 
        copy("ask", ask_greeks)

        arrays.call_ask[mask]   = ask_greeks.price * dffwd_t
        arrays.call_bid[mask]   = bid_greeks.price * dffwd_t
        arrays.nms[mask]        = NMS.to_nms( pure_strike, tte_yt )

    data = data.copy(deep=False)

    # drop anything?
    drop = set(data.columns) - set(SCHEMA_ENRICHED_DAY)
    if len(drop) > 0:
        data.drop(columns=drop, inplace=True)

    # copy over
    for r, v in arrays.items():
        data[r] = v
    data = data.loc[data["is_valid"]].drop(columns="is_valid")
    data['date'] = data['date'].astype("datetime64[s]")     # seems we need this for windows
    data['expiry'] = data['expiry'].astype("datetime64[s]") # seems we need this for windows

    if set(data.columns) != set(SCHEMA_ENRICHED_DAY):
        too_much = set(data.columns) - set(SCHEMA_ENRICHED_DAY)
        too_little = set(SCHEMA_ENRICHED_DAY) - set(data.columns)
        raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame has unexpected columns. Superflous: {fmt_list(too_much)}. Missing: {fmt_list(too_little)}. Please contact the authors with an example. ")

    for c,expected_dtype in SCHEMA_ENRICHED_DAY.items():
        actual_dtype = data[c].dtype
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' has unexpected dtype. Expected {expected_dtype}, found {actual_dtype}. Please contact the authors with an example.")

    if len(dropped_expiries) > 0:
        verbose.write(lambda : f"{ticker} {d}: Enriched {len(data)} of {num_total} raw options across {num_expiries} expiries, of which the following {len(dropped_expiries)} were dropped: {fmt_list(dropped_expiries)}.")
    else:
        assert len(data) == num_total, f"Internal error: {ticker} {d}: no expiries were dropped but the number of rows changed from {num_total} to {len(data)}. Please contact the authors with an example."
        verbose.write(lambda : f"{ticker} {d}: Enriched {len(data)} raw options across {num_expiries} expiries. No expiries were dropped.")

    return data

# ---------------------------------------------------------------------------
# Enriched data, day
# Optimal forward computation, implied volatilities, (pure) greeks.
# Also translates everything in calls.
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_EXCLUDE_FIELDS = {
    'ticker',
    'call_symbol', 
    'put_symbol', 
    'call_opt_id', 
    'put_opt_id', 
    'date', 
    'expiry', 
    'expiry_id',
    'is_american', 
    'is_am',
    'mkt_call_ask',
    'mkt_call_bid',
    'mkt_call_log_volume',
    'mkt_call_log_open_interest',
    'mkt_put_ask',
    'mkt_put_bid',
    'mkt_put_log_volume',
    'mkt_put_log_open_interest',
    }

@version("0.2")
@dataclass(frozen=True)
class SampleDaysConfig:
    max_num_options: int = 1000                 #: Maximum number of options. Options are sorted by NMS, expiry and call/put, and only the top max_num_options are kept. This is applied after all other filters.
    max_business_days_to_expiry: int = 252*3    #: Maximum business days to expiry. 
    max_nms: float = 4.                         #: Maximum |NMS|. 
    min_vegaSqrtVar: float = 0.001              #: Minimum vega * sqrt(tte), i.e. the derivative in sqrtVar: BSsq(K,sqrtVar) = BSvol(K,sqrtVar/sqrt(t),t). This is a measure of how much the option price moves with volatility, normalized by time to expiry. It is used to filter out options that are too close to expiry or too far OTM, which can have very low vega and lead to noisy implied volatilities.
    min_log_volume: float = 1.0                 #: Minimum log volume. 
    min_log_open_interest: float = 1.0          #: Minimum log open interest. 
    remove_am: bool = False                     #: Whether to remove options that settled at the AM price. 
    exclude_fields : set[str] = field(default_factory=lambda: set(DEFAULT_SAMPLE_EXCLUDE_FIELDS)) #: List of fields of :attr:`SCHEMA_ENRICHED_DAY` to exclude from the final data set. The default is :attr:`DEFAULT_SAMPLE_EXCLUDE_FIELDS`, which excludes fields that are not commonly used for modeling and can be easily retrieved from the option symbol or other fields. If a field is in both :attr:`exclude_fields` and :attr:`include_fields`, it will be included (i.e. :attr:`include_fields` takes precedence over :attr:`exclude_fields`).
    include_fields : set[str] = field(default_factory=set) #: List of fields of :attr:`SCHEMA_ENRICHED_DAY` to include in the final data set. If non-empty, only these fields will be included in the final data set.

    def __post_init__(self):
        verify_inp(self.max_num_options >= 10, lambda : f"'max_num_options' must be at least 10, found {self.max_num_options}")
        verify_inp(self.max_business_days_to_expiry > 0, lambda : f"'max_business_days_to_expiry' must be positive, found {self.max_business_days_to_expiry}")
        verify_inp(self.max_nms > 0, lambda : f"'max_nms' must be positive, found {self.max_nms}")
        verify_inp(self.min_vegaSqrtVar >= 0, lambda : f"'min_vegaSqrtVar' must be non-negative, found {self.min_vegaSqrtVar}")
        verify_inp(self.min_log_volume >= 0, lambda : f"'min_log_volume' must be non-negative, found {self.min_log_volume}")
        verify_inp(self.min_log_open_interest >= 0, lambda : f"'min_log_open_interest' must be non-negative, found {self.min_log_open_interest}")

        available_fields = set(SCHEMA_ENRICHED_DAY)
        verify_inp( set(self.exclude_fields) <= available_fields, lambda : f"'exclude_fields' must be a subset of {available_fields}; unknown fields found: {sorted(set(self.exclude_fields) - available_fields)}")
        verify_inp( set(self.include_fields) <= available_fields, lambda : f"'include_fields' must be a subset of {available_fields}; unknown fields found: {sorted(set(self.include_fields) - available_fields)}")

        flds = {f for f in SCHEMA_ENRICHED_DAY if f in self.include_fields} if len(self.include_fields) > 0 else set(SCHEMA_ENRICHED_DAY)
        if len(self.exclude_fields) > 0:
            flds = flds - set(self.exclude_fields)
        verify_inp( len(flds) > 0, lambda : f"No fields left after applying include/exclude filters. Include_fields: {sorted(self.include_fields)}, exclude_fields: {sorted(self.exclude_fields)}, available fields={sorted(SCHEMA_ENRICHED_DAY)}")

    def __str__(self):
        if len(self.include_fields) > 0 or len(self.exclude_fields) > 0:
            field_id = ",fld=" + unique_hash8( set(self.include_fields) | set(self.exclude_fields) )
        else:
            field_id = ""
        return f"nopt={self.max_num_options},bdte={self.max_business_days_to_expiry},nms={self.max_nms},lvolu={self.min_log_volume},loi={self.min_log_open_interest},rmam={self.remove_am},vegaSqrtVar={self.min_vegaSqrtVar}{field_id}"
    
    def _filter(self, df : pd.DataFrame) -> pd.DataFrame:
        """ Remove all relevant data """

        assert set(df.columns) >= set(SCHEMA_ENRICHED_DAY), f"Internal error: DataFrame has missing columns. Missing: {fmt_list(set(SCHEMA_ENRICHED_DAY) - set(df.columns))}. Please contact the authors with an example."
        assert not df.isna().any().any(), f"Internal error: filter resulted in NaN values. Please contact the authors with an example."

        if not df.empty:
            df = df[df['bdte'] <= self.max_business_days_to_expiry]
            df = df[np.abs(df['nms']) <= self.max_nms]
            df = df[df['log_volume'] >= self.min_log_volume]  if self.min_log_volume > 0. else df
            df = df[df['log_open_interest'] >= self.min_log_open_interest] if self.min_log_open_interest > 0. else df
            if self.remove_am:
                df = df[~df['is_am']]

            if not df.empty and len(df) > self.max_num_options:
                df['max_nms'] = np.abs(df['nms'])
                df = df.sort_values(by=["max_nms", "tte_y"], ascending=[True, True])
                df = df.head(self.max_num_options)
                df.drop(columns=["max_nms"], inplace=True)
                assert len(df) == self.max_num_options, f"Internal error: after filtering by max_num_options, the number of options is not equal to max_num_options. Please contact the authors with an example."

            if df.empty:
                raise RuntimeError(f"Internal error: filter resulted in an empty data set. Please contact the authors with an example.")

        df  = df.loc[:, self.include_fields] if len(self.include_fields) > 0 else df
        if len(self.exclude_fields) > 0:
            df.drop(columns=self.exclude_fields, inplace=True)
        assert len(df.columns) > 0, f"Internal error: no columns left after applying include/exclude filters. Include_fields: {sorted(self.include_fields)}, exclude_fields: {sorted(self.exclude_fields)}, available fields={sorted(SCHEMA_ENRICHED_DAY)}. Please contact the authors with an example."
        assert not df.isna().any().any(), f"Internal error: filter resulted in NaN values. Please contact the authors with an example."
        return df

def _uid( 
        ticker: str,
        date_range : DateRange, 
        raw_day_filter : RawDayFilter, 
        expiry_filter : ExpiryFilter, 
        sample_days_config : SampleDaysConfig ):
    return f"{named_unique_filename48_8(f'sample {str(raw_day_filter)},{str(expiry_filter)}',raw_day_filter,expiry_filter)}/"+\
           f"{named_unique_filename48_8(str(sample_days_config),sample_days_config)}/"+\
           f"{ticker}/{str(date_range)}"

@_training(fmt=SubDir.PYTREE_HDF5).cache(
        "0.1",
        dependencies=[enrich_day],
        uid = _uid,
        exclude_args=['verbose','dates','refresh_cache','raise_if_bad'])
def collect_days( ticker : str,
                  date_range : DateRange, *,
                  raw_day_filter : RawDayFilter, 
                  expiry_filter : ExpiryFilter, 
                  sample_days_config : SampleDaysConfig,
                  pool_config : PoolConfig = PoolConfig.off(),
                  refresh_cache : bool = False, 
                  verbose : Context = Context.all ):
    """
    Collect enriched data for a list of tickers and a date range, and write it into a single HDF5 file.
    """

    dates = date_range.get_business_days()

    verbose.write(lambda: f"Collecting enriched data for ticker {ticker} for {len(dates)} business days from {fmt_date(dates[0])} to {fmt_date(dates[-1])} {'excluding' if date_range.exclude_half_days else 'including'} half days.")
    verbose.report(2,lambda: f"Sample Days Config: {sample_days_config}")
    verbose.report(2,lambda: f"Raw Day Filter: {raw_day_filter}")
    verbose.report(2,lambda: f"Expiry filter: {expiry_filter}")
    verbose.report(2,lambda: f"Pool config: {pool_config}")

    # which data do we have
    # ---------------------


    def generate():

        thread_pool = JCPool(num_workers=pool_config.num_workers, threading=True)
        existing = {}
        missing = []

        if refresh_cache:
            verbose.verbose("Regenerating all files.")
            missing = dates
        else:
            with verbose.write_t(lambda: f"Checking which enriched daily data files for ticker {ticker} and the {len(dates)} business days from {fmt_date(dates[0])} to {fmt_date(dates[-1])} {'excluding' if date_range.exclude_half_days else 'including'} half days exist in the cache... ", end='') as tme:
                def process(d):
                    try:
                        df = enrich_day( ticker, d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=Context.quiet, 
                                         raise_if_bad=False, 
                                         refresh_cache=False, 
                                         override_cache_mode="cacheonly" )
                    except CacheMustExistError:
                        return None
                    return sample_days_config._filter(df)

                for d, df in thread_pool.parallel( { d: thread_pool.delayed(process)(d) for d in dates } ):
                    if df is None:
                        missing.append(d)
                    else:   
                        existing[d] = df

                if len(missing) > 0:
                    verbose.write(lambda: f"found {len(missing)}/{len(dates)} business days with missing daily data. Regenerating those. This took {tme}.", head=False)
                else:
                    verbose.write(lambda: f"all daily data files exist and have been loaded. This took {tme}.", head=False)

        if len(missing) > 0:
            # generate missing data with parallel processing
            pool = JCPool(num_workers=pool_config.num_workers, threading=False)

            def process( ticker, d, raw_day_filter, expiry_filter, refresh_cache ):
                r = enrich_day( ticker, d, raw_day_filter=raw_day_filter, 
                                       expiry_filter=expiry_filter, 
                                       refresh_cache=refresh_cache, 
                                       raise_if_bad=False, 
                                       verbose=Context.quiet, 
                                       override_cache_mode="update", 
                                       cache_generate_only=True )
                return d, r

            tme = verbose.timer()
            cnt = 0
            for d, status in pool.parallel( pool.delayed(process)( ticker, d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, refresh_cache=refresh_cache ) for d in missing ):
                cnt += 1
                if tme.interval_test(1.):
                    verbose.write(lambda: f"\r{ticker}: generating enriched data for {cnt}/{len(missing)} missing business days using {pool_config.num_workers} processes, now {d}. So far this took {tme}... ", end='')
            verbose.write(lambda: f"\rGenerated all {len(missing)} missing business days for {ticker} using {pool_config.num_workers} processes. This took {tme}.")

            pool.terminate()

            with verbose.write_t(lambda: f"Reading the generated enriched data for the {len(missing)} missing business days for {ticker} in the main process... ", end='') as tme:
                # read them back
                def process(d):
                    try:
                        df = enrich_day( ticker, d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, override_cache_mode="cacheonly", refresh_cache=False, raise_if_bad=False, verbose=Context.quiet )
                    except CacheMustExistError as e:
                        raise RuntimeError(f"Internal error: {ticker} {d}: expected enriched data to be available in the cache after generation, but it was not found. Please contact the authors with an example. Original error message: {str(e)}")
                    df = sample_days_config._filter(df)
                    return df

                for d, df in thread_pool.parallel( { d: thread_pool.delayed(process)(d) for d in missing } ):
                    assert not df is None
                    existing[d] = df

                verbose.write(lambda: f"done; this took {tme}.", head=False)
        del missing

        # clear empty days
        max_num_options = 0
        results = {}
        exdates = sorted(existing)
        for d in exdates:
            df = existing[d]
            if not df.empty:
                max_num_options = max(max_num_options, len(df))
                results[d] = df

        if len(results) < len(existing):
            removed = sorted( set(existing) - set(results) )
            verbose.report(1,lambda: f"The following {len(existing)-len(results)} business days where removed because they had no data after filtering: {fmt_list(removed)}.")

        return results, max_num_options

    results, max_num_options = generate()
    num_steps = len(results)

    # allocate memory
    # ===============

    with verbose.write_t(lambda: f"Generating sample for {num_steps} business days and at most {max_num_options} options per day for ticker {ticker}... ", end='') as tme:

        def allocate():
            df0 = results[list(results.keys())[0]] # touch one df to get the schema
            r = PrettyObject()
            r.is_valid = np.zeros((num_steps,max_num_options), dtype=bool )
            r.ticker = ticker

            for c in df0.columns:
                v = df0[c].to_numpy()
                assert len(v.shape) == 1, f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' has unexpected shape. Expected 1D array, found shape {v.shape}. Please contact the authors with an example."
                r[c] = np.zeros((num_steps, max_num_options), dtype=v.dtype) if len(v) > 0 else np.zeros((num_steps, 1), dtype=v.dtype)

            for i, d in enumerate(sorted(results)):
                df = results[d]
                ln = len(df)
                r.is_valid[i,:ln] = True
                for c in df.columns:
                    v = df[c].to_numpy()
                    r[c][i,:len(v)] = v

            for k,v in r.items():
                if k!="ticker" and v.dtype != np.dtype("O"):
                    try:
                        assert np.all( np.isfinite(v) ), f"Internal error: {ticker}: array for column '{k}' contains non-finite values. Please contact the authors with an example."
                    except TypeError as e:
                        raise RuntimeError(f"Internal error: {ticker}: failed to check for finite values in column '{k}' with dtype {v.dtype}. Please contact the authors with an example. Original error message: {str(e)}")

            return r
        
        r = allocate()

        verbose.write(lambda: f"done; this tool {tme}.", head=False)

        if verbose.shall_report(2):
            nsize = 0

            tbl = PrettyTable()
            tbl.field_names = ["Field", "Shape", "Dtype", "Size", "Mean", "Std", "Min", "Max" ]

            for k, v in r.items():
                if k == "ticker":
                    continue

                try:
                    tbl.add_row([k, v.shape, v.dtype, fmt_big_byte_number(v.nbytes), f"{np.mean(v[r.is_valid]):.4g}", f"{np.std(v[r.is_valid]):.4g}", f"{np.min(v[r.is_valid]):.4g}", f"{np.max(v[r.is_valid]):.4g}" ])
                except Exception:
                    tbl.add_row([k, v.shape, v.dtype, fmt_big_byte_number(v.nbytes), "", "", "", ""])
                nsize += v.nbytes   

            tbl.add_row(["TOTAL", "", "", fmt_big_byte_number(nsize), "", "", "", ""])
            verbose.report(2, tbl)

    return dict(r)  # for HDF5 

# ================================================================
# Command-line interface
# *** something wired is broken with the command line version ***
# ================================================================

import click
from quantdata.widb import cli_defaults
from quantdata.biz_calendar import to_date, DateRange
from quantdata.pool import PoolConfig
from cdxcore.err import verify_inp
from cdxcore.util import fmt_list

def main_works():
    tickers          = ["SPX"]

    start, end       = ( dt.date(2015, 1, 1), dt.date(2025, 1, 1) )
    date_range       = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=True)
    pool_config      = PoolConfig(num_workers=4, threading=False)
    raw_day_filter   = RawDayFilter()   
    expiry_filter    = ExpiryFilter()
    sample_days_config = SampleDaysConfig()
    refresh = False

    dates            = date_range.get_business_days()

    verbose = Context.all
    verbose.write(lambda: f"Enriching data for tickers {fmt_list(tickers)} from {fmt_date(start)} to {fmt_date(end)} {'excluding' if date_range.exclude_half_days else 'including'} half days. Total business days: {len(dates)}.")
    verbose.report(1,lambda: f"Sample Days Config: {sample_days_config}")
    verbose.report(1,lambda: f"Raw Day Filter: {raw_day_filter}")
    verbose.report(1,lambda: f"Expiry filter: {expiry_filter}")
    verbose.report(1,lambda: f"Pool config: {pool_config}")
    for it, ticker in enumerate(tickers):  
        verbose.report(1,lambda: f"\rProcessing ticker {it+1}/{len(tickers)} {ticker} from {fmt_date(start)} to {fmt_date(end)}.")
        collect_days(ticker, date_range, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter,
                     sample_days_config=sample_days_config, pool_config=pool_config, refresh_cache=refresh, verbose=verbose(2) )
    verbose.write(f"All {len(tickers)} tickers processed.")

@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--period",nargs=2, type=click.DateTime(formats=["%Y-%m-%d"]), default=cli_defaults.period, metavar="START END", help="Period YYYY-MM-DD YYYY-MM-DD")
@click.option('--halfdays', is_flag=True, help="Include half days")
@click.option('--workers', default=cli_defaults.workers, help='Number of workers', type=int)
@click.option('--threading', is_flag=True, help="Use only threading, no multiprocessing")
@click.option('--refresh', is_flag=True, help="Refresh data (this will only refresh the top level files)")
@click.option('--debug', is_flag=True, help="Enable debug mode with more verbose output and checks")

def main(tickers : list[str],
        period : list[dt.datetime], 
        halfdays : bool, 
        workers : int, 
        threading : bool, 
        refresh : bool,
        debug : bool):
    
    if debug:
        main_works()
        return
    
    verify_inp( len(tickers) > 0, "At least one ticker must be specified")
    verify_inp( period is not None and len(period) == 2, "A start and end date must be specified")

    start, end       = period
    date_range       = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=not halfdays)
    pool_config      = PoolConfig(num_workers=workers, threading=threading)
    raw_day_filter   = RawDayFilter()   
    expiry_filter    = ExpiryFilter()

    dates            = date_range.get_business_days()

    verbose = Context.all
    verbose.write(lambda: f"Enriching data for tickers {fmt_list(tickers)} from {fmt_date(start)} to {fmt_date(end)} {'excluding' if date_range.exclude_half_days else 'including'} half days. Total business days: {len(dates)}. Pool config: {pool_config}. Raw Day Filter: {raw_day_filter}. Expiry filter: {expiry_filter}.")
    for it, ticker in enumerate(tickers):  
        verbose.report(1,lambda: f"\rProcessing {len(tickers)} tickers: {int(it*100//len(tickers))}% done. Now processing {ticker} from {fmt_date(start)} to {fmt_date(end)}.")
        _refresh_all_days( ticker, dates, raw_day_filter, expiry_filter, pool_config, refresh_cache=refresh, verbose=verbose(2), raise_if_bad=False )
    verbose.write(f"All {len(tickers)} tickers processed.")

if __name__ == "__main__":
    main()
