"""
Generating one day of WRDS Option Metrics data for one ticker
=============================================================

Key functions:

* :func:`collect_raw_day` - Collect raw option data with minimal analytics for a ticker and date
* :func:`enrich_day` - Enrich raw data with implied volatilities, greeks, and synthetic call options

Summary
-------
This module provides tools to process daily option data from WRDS Option Metrics (IvyDB):

1. **collect_raw_day**: Downloads spot, forward, discount factor, and option data. Cleans bad options 
   and computes business days to expiry and implied discount factors. Returns a raw DataFrame.

2. **enrich_day**: Builds on raw data by computing implied forwards from put-call parity, synthetic 
   minimal-spread call options, implied volatilities, and option greeks (delta, gamma, vega, theta).

Both functions use filtering configurations to control data quality and processing parameters.

Usage Example
-------------
    >>> from quantdata.widb.day import collect_raw_day, enrich_day, RawDayFilter, ExpiryFilter
    >>> from quantdata.biz_calendar import to_date
    >>> from cdxcore.verbose import Context
    >>> import datetime as dt
    >>>
    >>> # Define date and filters
    >>> d = to_date("2022-01-03")
    >>> ticker = "SPX"
    >>> raw_filter = RawDayFilter(min_bid=0.015)
    >>> enriched_filter = ExpiryFilter(num_min_both_traded=5)
    >>>
    >>> # Collect and enrich data
    >>> raw_data = collect_raw_day(ticker, d, raw_filter, verbose=Context())
    >>> enriched_data = enrich_day(ticker, d, raw_data, raw_day_filter=raw_filter, 
    ...                             expiry_filter=enriched_filter, verbose=Context())
    >>>
    >>> # Access results
    >>> print(enriched_data[['strike', 'fwd', 'ask_vol', 'bid_vol', 'pure_ask_delta']].head())

This file's ``main`` can also be used to generate data manually:

```bash

    python -m day SPX RUT NDX --period 2020-05-01 2022-01-31
```

"""
from __future__ import annotations

from quantdata.widb import WIDBClient, interpolate_zero_curve
from quantdata.biz_calendar import NMS5, count_business_days_between, DAYS_PER_YEAR, AM_EXIRY_SUBTRACT, to_date, NMS, prev_business_day, DateRange
from quantdata.widb.client import WIDBClient
from quantdata.widb.bad import BAD
from quantdata.pool import PoolConfig
from quantdata.utils.pd import create_empty_df, verify_schema

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.util import fmt_date, fmt_list
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir, CacheMode
from cdxcore.uniquehash import named_unique_filename48_8

import datetime as dt
import numpy as np
import pandas as pd
import math as math
import warnings
from dataclasses import dataclass, field    

_processing = WIDBClient.locations.processing("day",fmt=SubDir.PANDAS_PARQUET)
_IMPLIED_VOL_CALC_EPS = 1E-8
_GREEK_CALC_EPS = 1E-8
_MIN_IMPLIED_VOL = 0.01
_MAX_IMPLIED_VOL = 5.

# ---------------------------------------------------------------------------
# Raw data, day
# Minimal analytics and clean up.
# ---------------------------------------------------------------------------

SCHEMA_RAW_DAY = {
    'ticker' : np.dtype('O'),
    'date' : np.dtype("datetime64[ms]"),  # [D] not supported in storage
    'spot' : np.float32,
    'spot_prev' : np.float32,
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
    'expiry' : np.dtype("datetime64[ms]"),  # [D] not supported in storage``
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

RAW_PER_DAY_FIELDS = {'ticker','date','spot','spot_prev'}
RAW_PER_EXPIRY_FIELDS = {'expiry_id','expiry','bdte','tte_y','is_am','fwd_ref','df'}

IRREGULAR_EXPIRES = {
    dt.date(2025,1,9)    # SPX 2025-01-10 expiry was announced on 2025-01-09, which is the same day as the 2025-01-10 expiry, leading to data issues. We consider this an irregular expiry and drop it by default. If you want to keep it, please set RawDayFilter.drop_irregular_expiries to False and add the date to the BAD data sets if you confirm the data quality is sufficient.    
}

@version("0.5")
@dataclass(frozen=True)
class RawDayFilter:
    min_bid         : float = 0.011  # minimum bid value, typically above one tick.
    drop_irregular_expiries : bool = True    # drop irregular expiries, ie holidays which were announced shortly 

    def __post_init__(self):
        verify_inp(self.min_bid >= 0., lambda : f"'min_bid' must be non-negative, found {self.min_bid:.3f}")

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"bid={self.min_bid:.3f}"

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.7",
            dependencies=[interpolate_zero_curve, WIDBClient],
            uid=lambda ticker, d, raw_day_filter : f"raw/{named_unique_filename48_8(str(raw_day_filter),raw_day_filter)}/{ticker}/{fmt_date(d)}",
            exclude_args=['verbose','raise_if_bad_day'])
def collect_raw_day( ticker : str, 
                     d : dt.date, 
                     raw_day_filter : RawDayFilter, *, 
                     raise_if_bad_day : bool = True,
                     verbose : Context = Context.all, secid : int|None = None ) -> pd.DataFrame:
    """
    Collect raw data with minimal analytics for a given ticker and date.

    The function returns a panda data frame with columns :attr:`SCHEMA_RAW_DAY`.

    The function:

    * Downloads spot, forward, discount and option data.
    * Removes and bad days/tickers/options using :attr:`quantdata.widb.bad.BAD`.
    * Removes options with same-day expiry, NA or zero bid prices, and bad settlement (ss_flag != '0').
      By default the function also removed irregular expiries, which are holidays that were announced shortly before the expiry date, leading to data issues.
    * Computes the implied discount factor from the zero curve and the refernece forward price from the forward table (which is off as the forward table is 15min delayed).
    * Computes business days to expiry and a unique expiry ID (which includes AM/PM).

    Typically the result of this function is processed with :func:`enrich_day`.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    d : :class:`datetime.date`
        Date for which to collect data
    config : :class:`RawDayFilter`
        Configuration for data collection
    raise_if_bad_day : bool, default ``True``
        If ``True``, raises an error if the date is in the BAD data sets. If ``False``, returns an empty data set instead. (In no other condition will an empty data set returned)
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging

    Returns
    -------
    Daily data: :class:`pandas.DataFrame`
        Cleaned DataFrame with all relevant data for the given ticker and date, with the schema :attr:`SCHEMA_RAW_DAY`.
        This data set is not empty unless ``raise_if_bad_day`` is True and the date/ticker is in the BAD data sets.

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
    verify_inp(isinstance(raw_day_filter, RawDayFilter), lambda : f"'raw_day_filter' must be an instance of RawDayFilter, found {type(raw_day_filter)}")

    if d >= BAD.LAST_LIKELY_VALID_DATE:
        raise ValueError(f"{d} is on or after the BAD.LAST_LIKELY_VALID_DATE ({BAD.LAST_LIKELY_VALID_DATE}). Data for this date is likely to be missing from Option Metrics. "+\
                          "If you are sure this is not the case, please update widb/bad.py.")

    if d in BAD.DAYS_ALL:
        if not raise_if_bad_day:
            return create_empty_df(SCHEMA_RAW_DAY)
        raise IndexError(f"{d} is in the BAD.DAYS_ALL set. No data will be downloaded for this date. Please check the date range and calendar settings. BAD.DAYS_ALL={BAD.DAYS_ALL}")
    if d in  BAD.DAYS_TICKER.get(ticker, set()):
        if not raise_if_bad_day:
            return create_empty_df(SCHEMA_RAW_DAY)
        raise IndexError(f"{d} is in the BAD.DAYS_TICKER set for {ticker}. No data will be downloaded for this date. Please check the date range and calendar settings. BAD.DAYS_TICKER[{ticker}]={BAD.DAYS_TICKER.get(ticker, set())}")

    conn        = WIDBClient()
    secid       = conn.lookup_secid(ticker)

    with verbose.write_t(f"Collecting raw market data for {ticker} on {d}.") as tme:
        try:
            spot    = conn.spot_price(secid, d, verbose=verbose(1))
        except LookupError as e:
            raise LookupError(f"Failed to fetch spot price for {ticker} on {d}. "+\
                            f"This likely means either the corresponding secid {secid} is not covered by Option Metrics, or {d} is not a business day for this underlying. Please check the date range and calendar settings.") from e

        try:
            prev_d  = prev_business_day(d, count_start=False)
            spot_d  = conn.spot_price(secid, prev_d, verbose=verbose(1))
        except LookupError as e:
            raise LookupError(f"Failed to fetch spot price for {ticker} on previous business day {prev_d} (relative to {d}).") from e

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

    def remove_irregular_expiries( status_removed, opts_pd ):
        if raw_day_filter.drop_irregular_expiries:
            opts_pd['exdate']    = pd.to_datetime(opts_pd['exdate']).dt.date
            irregular_expiry_mask = opts_pd['exdate'].isin(IRREGULAR_EXPIRES)
            if irregular_expiry_mask.any():
                n_before = len(opts_pd)
                opts_pd = opts_pd.loc[~irregular_expiry_mask].copy()
                n_removed = n_before - len(opts_pd)
                if opts_pd.empty:
                    raise EOFError(f"{ticker} {d}: all option rows had irregular expiries and were dropped.")
                if n_removed > 0:   
                    status_removed.append(f"{n_removed}/{n_before} options with irregular expiries") 
        return status_removed, opts_pd
    
    status_removed, opts_pd = remove_irregular_expiries(status_removed, opts_pd)
        
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
        r.spot_prev = np.float32(spot_d)
        r.date = np.array(d, dtype="datetime64[ms]")
        r.ticker = ticker
        r.call_symbol = opts_pd["call_symbol"].astype(str).values.tolist()
        r.put_symbol = opts_pd["put_symbol"].astype(str).values.tolist()
        r.call_opt_id = opts_pd["call_opt_id"].to_numpy(dtype=np.int32)
        r.put_opt_id = opts_pd["put_opt_id"].to_numpy(dtype=np.int32)
        r.is_american = opts_pd["is_american"].to_numpy(dtype=bool)
        r.is_am = opts_pd["is_am"].to_numpy(dtype=bool)
        r.expiry = np.array(opts_pd["exdate"], dtype="datetime64[ms]")
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
            + np.where(r.is_am, 0, 1)
        ).astype(np.int32)

        r.bdte = count_business_days_between(d, expiry_dates.to_numpy(dtype="datetime64[D]"))  # returns integers

        if np.min(r.bdte) <= 0:
            # historically end of month expiries might fall on a Saturday AM
            # we check that if BDTE is zero, those are the cases
            bad_expiries = np.unique( r.expiry[r.bdte <= 0] )
            bad_is_am    = np.unique( r.is_am[r.bdte <= 0] )
            bad_bdte     = np.unique( r.bdte[r.bdte <= 0] )

            for be, bam, bb in zip(bad_expiries, bad_is_am, bad_bdte):
                be = to_date(be)
                assert bb == 0, f"Internal error: expected non-positive business days to expiry to be exactly zero, found {bb} for expiry {be} with is_am={bam}. Please contact the authors with an example."
                assert be.weekday() == 5, f"Internal error: expected bad expiry {be} to be a Saturday, found weekday {be.weekday()}. Please contact the authors with an example."
                assert bam == True, f"Internal error: expected bad expiry {be} to be an AM settlement, found is_am={bam}. Please contact the authors with an example."

        r.bdte = np.clip( r.bdte.astype(np.float32) - np.where(r.is_am, AM_EXIRY_SUBTRACT, 0.0), min=0. ).astype(np.float32)
        r.tte_y = r.bdte / DAYS_PER_YEAR
        return pd.DataFrame(dict(r))

    result = enrich(opts_pd)
    verify_schema(result, SCHEMA_RAW_DAY, lambda : f"{ticker} {d}")

    # check that common fields are so.    
    for c in RAW_PER_DAY_FIELDS:
        column_data = result[c]
        if not (column_data.nunique() == 1):
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' is expected to have exactly one unique value, found {column_data.nunique()}. Please contact the authors with an example.")

    num_expiries   = result["expiry_id"].nunique()
    for c in RAW_PER_EXPIRY_FIELDS:
        grouped_unique = result.groupby("expiry_id", sort=False)[c].nunique(dropna=False)
        if not (grouped_unique == 1).all():
            bad_expiry_ids = grouped_unique[grouped_unique != 1].index.tolist()[:10]
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' is expected to have a single value within each expiry_id, but this failed for expiry_id values {bad_expiry_ids}. Please contact the authors with an example.")

    verbose.write(f"{ticker} {d}: collected {len(result)}/{n_total} options. Removed: {fmt_list(status_removed) if len(status_removed) > 0 else 'none'}.")
    return result

# ---------------------------------------------------------------------------
# Enriched data, day
# Optimal forward computation, implied volatilities, (pure) greeks.
# Also translates everything in calls.
# ---------------------------------------------------------------------------

SCHEMA_ENRICHED_DAY = SCHEMA_RAW_DAY | {
    'fwd' : np.dtype("float32"),          # implied forward from put-call parity
    'nms' : np.dtype("float32"),          # normalized log-moneyness using :class:`NMS5` 
    'nms_atm' : np.dtype("float32"),      # normalized log-moneyness using :class:`NMS5` with the ATM volatility
    'call_ask' : np.dtype("float32"),     # implied from pure call after implied vol fitting
    'call_bid' : np.dtype("float32"),     # implied from pure call after implied vol fitting
    'ask_vol' : np.dtype("float32"),
    'bid_vol' : np.dtype("float32"),
    'log_volume' : np.dtype("float32"),
    'log_open_interest' : np.dtype("float32"),
    'pure_strike' : np.dtype("float32"),
    'pure_ask_price' : np.dtype("float32"),
    'pure_bid_price' : np.dtype("float32"),
    'pure_ask_delta' : np.dtype("float32"),
    'pure_bid_delta' : np.dtype("float32"),
    'pure_ask_gamma' : np.dtype("float32"),
    'pure_bid_gamma' : np.dtype("float32"),
    'pure_bid_vega' : np.dtype("float32"),
    'pure_ask_vega' : np.dtype("float32"),
    'pure_bid_theta' : np.dtype("float32"),
    'pure_ask_theta' : np.dtype("float32"),
    'pure_ask_dk' : np.dtype("float32"),
    'pure_bid_dk' : np.dtype("float32")
}

ENRICHED_PER_DAY_FIELDS = RAW_PER_DAY_FIELDS
ENRICHED_PER_EXPIRY_FIELDS = RAW_PER_EXPIRY_FIELDS|{'fwd'}

@version("0.5")
@dataclass(frozen=True)
class ExpiryFilter:
    """ How to filter expiries to be able to compute implied forwards """
    num_min_either_traded : int = 10       # Drop expiries with less than this number of either calls or puts traded for a given strike.
    num_min_both_traded   : int = 4        # Drop expiries with less than this number of both calls and puts traded for a given strike.q
    min_pure_spread       : float = 1E-6
    min_implied_vol       : float = 0.01
    max_implied_vol       : float = 5.0
    normalize             : bool = False

    def __post_init__(self):
        verify_inp(self.num_min_either_traded >= 0, lambda : f"'num_min_either_traded' must be non-negative, found {self.num_min_either_traded}")
        verify_inp(self.num_min_both_traded >= 0, lambda : f"'num_min_both_traded' must be non-negative, found {self.num_min_both_traded}")
        verify_inp(self.min_pure_spread > 0, lambda : f"'min_pure_spread' must be positive, found {self.min_pure_spread}")
        verify_inp(self.min_implied_vol > 0, lambda : f"'min_implied_vol' must be positive, found {self.min_implied_vol}")
        verify_inp(self.max_implied_vol > self.min_implied_vol, lambda : f"'max_implied_vol' must be greater than 'min_implied_vol', found max_implied_vol={self.max_implied_vol} and min_implied_vol={self.min_implied_vol}")

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"{'nrm,' if self.normalize else 'nat,'}eth={self.num_min_either_traded},bth={self.num_min_both_traded},mps={self.min_pure_spread},miv={self.min_implied_vol},xiv={self.max_implied_vol}"

def _uid(ticker, d, raw_day_filter, expiry_filter) -> str:
    return f"enrich/{named_unique_filename48_8(f'{str(expiry_filter)},{str(raw_day_filter)}',expiry_filter,raw_day_filter)}"+\
           f"/{ticker}/{fmt_date(d)}"

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.8",
            dependencies=[collect_raw_day, ExpiryFilter],
            uid=_uid,
            exclude_args=['verbose','raw_data','raise_if_bad_day','refresh_cache'])
def enrich_day( ticker : str, d : dt.date, raw_data : pd.DataFrame|None = None, *, 
                raw_day_filter : RawDayFilter = RawDayFilter(), 
                expiry_filter : ExpiryFilter = ExpiryFilter(), 
                override_cache_mode : CacheMode|None = None,
                raise_if_bad_day : bool = True,
                verbose : Context = Context.all ) -> pd.DataFrame:
    """
    Enrich a day's data.

    * Compute synthetic minimal spread call options.
    * Compute an implied forward from inverse-spread weighted bid/ask call prices.
    * Compute implied volatilities and (pure) greeks.

    The returned daya data frame has the schema :attr:`SCHEMA_ENRICHED_DAY`, which includes all columns from :attr:`SCHEMA_RAW_DAY`.

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
    override_cache_mode : :class:`CacheMode` | None, default ``None``
        If not ``None``, overrides the cache mode for the raw day data. Set to ``"update"`` to
        refresh the underlying raw cache.
    raise_if_bad_day : bool, default ``True``
        If ``True``, raises an error if the date is in the BAD data sets.
        If ``False``, returns an empty data set instead. (In no other condition will an empty data set be returned)
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
    if not raw_data is None:
        verify_inp( isinstance(raw_data, pd.DataFrame), lambda : f"Expected raw_data to be a pandas DataFrame or None, found {type(raw_data)}")
        data = raw_data
    else:
        data = collect_raw_day(ticker, d, raw_day_filter, verbose=verbose, override_cache_mode=override_cache_mode, raise_if_bad_day = raise_if_bad_day)
    del raw_data

    if data.empty:
        return create_empty_df(SCHEMA_ENRICHED_DAY)
        verify_schema( data , SCHEMA_RAW_DAY, lambda : f"{ticker} {d}")

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
    arrays.nms_atm         = np.zeros_like( dffwd_ask, dtype=np.float32 )
    arrays.is_valid        = np.zeros_like( arrays.fwd, dtype=bool )
    arrays.call_ask        = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.call_bid        = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.ask_vol         = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.bid_vol         = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.log_volume      = np.log10( np.pow(10.,call_log_vol.astype(np.float64))+np.pow(10.,put_log_vol.astype(np.float64))-1 ).astype(np.float32)
    arrays.log_open_interest = np.log10( np.pow(10.,call_log_oi.astype(np.float64))+np.pow(10.,put_log_oi.astype(np.float64))-1 ).astype(np.float32)
    arrays.pure_strike     = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_price  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_price  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_delta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_delta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_gamma  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_gamma  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_vega   = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_vega   = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_theta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_theta  = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_ask_dk     = np.zeros_like( arrays.fwd, dtype=np.float32 )
    arrays.pure_bid_dk     = np.zeros_like( arrays.fwd, dtype=np.float32 )
    
    # project
    dropped_expiries    = []
    dropped_strikes     = []
    num_dropped_strikes = 0

    for expiry_id in unq_ids:
        mask            = expiry_ids == expiry_id
        iexpory         = expiries[mask][0]
        iiam            = is_am[mask][0]
        strid           = f"{fmt_date(to_date(iexpory))}{'am' if iiam else 'pm'}"
        is_valid        = arrays.is_valid[mask] # starts False

        if np.sum( both_cp[mask] ) < expiry_filter.num_min_both_traded:
            dropped_expiries.append(f"{strid} (insufficient traded strikes)")
            continue
        if np.sum(either_cp[mask]) < expiry_filter.num_min_either_traded:
            dropped_expiries.append(f"{strid} (insufficient traded options)")
            continue
        dffwd_t          = np.sum(dffwd_mid[mask] * dffwd_wght[mask]) / np.sum(dffwd_wght[mask]) # weighted forward for this expiry_id
        if np.min(dffwd_t) < 1E-6:
            dropped_expiries.append(f"{strid} (cannot compute implied forward)")
            continue

        df_t             = df_ref[mask][0] # same for all rows 
        fwd_t            = dffwd_t / df_t
        tte_yt           = data['tte_y'].to_numpy()[mask][0]
        sqrtT_t          = math.sqrt(tte_yt)
        arrays.fwd[mask] = fwd_t
        is_valid[:]      = True

        # synthetic call price from put-call parity using the projected forward, for comparison with the market call price. 
        sy_call_ask      = mkt_put_ask[mask] + dffwd_t - df_t * strike[mask] 
        sy_call_bid      = mkt_put_bid[mask] + dffwd_t - df_t * strike[mask]

        sy_call_spread   = sy_call_ask - sy_call_bid
        bad_options      = (sy_call_ask <= 0.) | (sy_call_bid <= 0.) | (sy_call_spread < expiry_filter.min_pure_spread*dffwd_t)
        issues           = ""
        if np.any(bad_options):
            issues              = f"{np.sum(bad_options)} strikes with p/c issues; "
            num_dropped_strikes += np.sum(bad_options)
            is_valid[:]         &= ~bad_options

        call_mid           = 0.5 * ( np.minimum( sy_call_ask, mkt_call_ask[mask]) + np.maximum( sy_call_bid, mkt_call_bid[mask]) )
        call_spread        = np.minimum( mkt_call_ask[mask] - mkt_call_bid[mask], sy_call_ask - sy_call_bid )
        new_call_ask       = call_mid + 0.5*call_spread
        new_call_bid       = call_mid - 0.5*call_spread
        pure_strike        = strike[mask] / fwd_t
        pure_call_ask      = new_call_ask / dffwd_t
        pure_call_bid      = new_call_bid / dffwd_t
        bad_options        = (pure_call_ask - pure_call_bid) < expiry_filter.min_pure_spread
        if np.any(bad_options):
            issues              = issues + f"{np.sum(bad_options)} strikes with invalid spread; "
            num_dropped_strikes += np.sum(bad_options)
            is_valid[:]         &= ~bad_options

        if not np.any(is_valid):
            dropped_expiries.append(f"{strid} (too many serious forward/prices issues to recover)")
            is_valid[:] = False
            continue
        if np.sum( both_cp[mask][is_valid] ) < expiry_filter.num_min_both_traded:
            dropped_expiries.append(f"{strid} (insufficient traded strikes post filtering)")
            is_valid[:] = False
            continue
        if np.sum(either_cp[mask][is_valid]) < expiry_filter.num_min_either_traded:
            dropped_expiries.append(f"{strid} (insufficient traded options post filtering)")
            is_valid[:] = False
            continue

        # implied vol calculation will be within 10% of spread
        # which means that bid should remain below ask.
        # However, for extreme options the implied vol calculation
        # may reach a boundary. In this case bid/ask spreads
        # are not guranteed to be preserved.

        spread                   = pure_call_ask - pure_call_bid
        price_tol                = np.minimum( spread*0.1, 1E-6)
        ask_vols                 = bs.implied( pure_strike, pure_call_ask, sqrtT = sqrtT_t, 
                                               price_tol=price_tol,
                                               mask=is_valid, 
                                               eps=_IMPLIED_VOL_CALC_EPS, 
                                               vol_min=expiry_filter.min_implied_vol, 
                                               vol_max=expiry_filter.max_implied_vol )
        bid_vols                 = bs.implied( pure_strike, pure_call_bid, sqrtT = sqrtT_t, 
                                               price_tol=price_tol, 
                                               mask=is_valid, 
                                               eps=_IMPLIED_VOL_CALC_EPS, 
                                               vol_min=expiry_filter.min_implied_vol, 
                                               vol_max=expiry_filter.max_implied_vol )
        atm_vol                  = np.interp( 1., pure_strike, (ask_vols+bid_vols)*0.5 )
        ask_vols[~is_valid]      = expiry_filter.min_implied_vol*1.1
        bid_vols[~is_valid]      = expiry_filter.min_implied_vol

        arrays.pure_strike[mask] = pure_strike
        arrays.ask_vol[mask]     = ask_vols
        arrays.bid_vol[mask]     = bid_vols

        bid_greeks               = bs( k=pure_strike, vol=bid_vols, sqrtT = sqrtT_t, eps=_GREEK_CALC_EPS, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )
        ask_greeks               = bs( k=pure_strike, vol=ask_vols, sqrtT = sqrtT_t, eps=_GREEK_CALC_EPS, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )

        diff_del = 0
        ask_diff = ask_greeks.price - pure_call_ask
        diff_err =( np.abs(ask_diff) > price_tol*1.1 ) & is_valid
        if np.any(diff_err):
            ask_vol_at_err = ask_vols[diff_err]
            bnd_vol        = ( ask_vol_at_err == expiry_filter.min_implied_vol ) | ( ask_vol_at_err == expiry_filter.max_implied_vol )
            assert np.all(bnd_vol), f"Internal error: {ticker} {d}: found {np.sum(diff_err)} options with implied vol calculation errors above tolerance, but not at the boundaries. Please contact the authors with an example."
            diff_del       = np.sum(diff_err)
            is_valid       &= ~diff_err

        bid_diff = bid_greeks.price - pure_call_bid
        diff_err = (np.abs(bid_diff) > price_tol*1.1) & is_valid
        if np.any(diff_err):    
            bid_vol_at_err = bid_vols[diff_err]
            bnd_vol        = ( bid_vol_at_err == expiry_filter.min_implied_vol ) | ( bid_vol_at_err == expiry_filter.max_implied_vol )
            assert np.all(bnd_vol), f"Internal error: {ticker} {d}: found {np.sum(diff_err)} options with implied vol calculation errors above tolerance, but not at the boundaries. Please contact the authors with an example."
            diff_del       += np.sum(diff_err)
            is_valid       &= ~diff_err

        if diff_del>0:
            issues         = issues + f"{diff_del} strikes at implied vol boundaries; "
            num_dropped_strikes += diff_del

        # copy residual over

        def copy(which, src):
            for k,v in src.items():
                v[~is_valid] = 0.0
                arrays[f"pure_{which}_{k}"][mask] = v
        copy("bid", bid_greeks) 
        copy("ask", ask_greeks)

        arrays.call_ask[mask]   = ask_greeks.price * dffwd_t
        arrays.call_bid[mask]   = bid_greeks.price * dffwd_t
        arrays.nms[mask]        = NMS5.to_nms( pure_strike, tte_yt )
        arrays.nms_atm[mask]    = NMS5.to_nms( pure_strike, tte_yt, vol=atm_vol )

        assert np.all(arrays.call_bid[mask][is_valid] > 0.), f"Internal error: {ticker} {d}: found non-positive call bid price after enrichment for expiry {strid}. Please contact the authors with an example."
        assert np.all(arrays.call_ask[mask][is_valid] > arrays.call_bid[mask][is_valid]), f"Internal error: {ticker} {d}: found non-positive bid-ask spread {strid}. Please contact the authors with an example."

        assert np.any( is_valid ), "Internal error: no valid options found after enrichment."
        arrays.is_valid[mask] = is_valid ##??? why do I need to do that???
        assert np.all( is_valid == arrays.is_valid[mask] ), "Internal error: mismatch between is_valid and arrays.is_valid"

        if len(issues) > 0:
            dropped_strikes.append(f"{fmt_date(to_date(iexpory))}{'am' if iiam else 'pm'} (Dropped {issues[:-2]}: {np.sum(is_valid)}/{len(is_valid)} strikes left)")

    data = data.copy(deep=False)

    # drop anything?
    drop = set(data.columns) - set(SCHEMA_ENRICHED_DAY)
    if len(drop) > 0:
        data.drop(columns=drop, inplace=True)

    # copy over
    for r, v in arrays.items():
        data[r] = v
    # remove all rows where is_valid is False
    if not np.all(arrays.is_valid):
        data = data.loc[data["is_valid"]].drop(columns="is_valid")

    df_bad = data[data['pure_ask_price']<=data['pure_bid_price']]
    if not df_bad.empty:
        print(df_bad[['expiry_id', 'strike', 'pure_ask_price', 'pure_bid_price']])
        assert df_bad.empty, f"Internal error: {ticker} {d}: found {len(df_bad)} options with pure ask price below pure bid price -- this should not happen here!"

    num_expires_left = data["expiry_id"].nunique()
    if len(dropped_expiries) > 0:
        dropped_expiries = f"Dropped the following {len(dropped_expiries)} of {num_expiries} input expiries: {fmt_list(dropped_expiries)}. "
    if len(dropped_strikes) > 0:
        dropped_strikes = f"Dropped {num_dropped_strikes} strikes among the remaining {num_expires_left} expiries: {fmt_list(dropped_strikes)}. "

    verify_schema(data, SCHEMA_ENRICHED_DAY, lambda : f"{ticker} {d}")

    if data.empty:
        verbose.write(lambda : f"{ticker} {d}: no valid options found after enrichment:")
        for de in dropped_expiries:
            verbose.report(2,de)
        for ds in dropped_strikes:
            verbose.report(2,ds)
        if raise_if_bad_day:
            assert len(dropped_expiries) > 0 or len(dropped_strikes) > 0, "Internal error -- No info??"
            raise RuntimeError(f"Internal error: {ticker} {d}: no valid options found after enrichment.")
        warnings.warn(f"{ticker} {d}: no valid options found after enrichment:{dropped_expiries}{dropped_strikes}")
        return data

    # check that common fields are so.    
    for c in ENRICHED_PER_DAY_FIELDS:
        column_data = data[c]
        if not (column_data.nunique() == 1):
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' is expected to have exactly one unique value, found {column_data.nunique()}. Please contact the authors with an example.")

    num_expiries   = data["expiry_id"].nunique()
    for c in ENRICHED_PER_EXPIRY_FIELDS:
        grouped_unique = data.groupby("expiry_id", sort=False)[c].nunique(dropna=False)
        if not (grouped_unique == 1).all():
            bad_expiry_ids = grouped_unique[grouped_unique != 1].index.tolist()[:10]
            raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' is expected to have a single value within each expiry_id, but this failed for expiry_id values {bad_expiry_ids}. Please contact the authors with an example.")

    verbose.write(lambda : f"{ticker} {d}: Enriched {len(data)} of {num_total} raw options across {num_expires_left} of {num_expiries} input expiries.")
    if len(dropped_expiries) > 0:
        verbose.report(1,dropped_expiries)
    if len(dropped_strikes) > 0:
        verbose.report(1,dropped_strikes)
    return data

# ================================================================
# Command-line interface
# ================================================================

import click
from quantdata.widb import _cli_defaults

@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--period",nargs=2, type=click.DateTime(formats=["%Y-%m-%d"]), default=_cli_defaults.period, metavar="START END", help="Period YYYY-MM-DD YYYY-MM-DD")
@click.option('--halfdays', is_flag=True, help="Include half days")
@click.option('--workers', default=_cli_defaults.workers, help='Number of workers', type=int)
@click.option('--threading', is_flag=True, help="Use only threading, no multiprocessing")
@click.option('--refresh', is_flag=True, help="Refresh data (this will only refresh the top level files)")

def main(tickers : list[str],
        period : list[dt.datetime], 
        halfdays : bool, 
        workers : int, 
        threading : bool, 
        refresh : bool):

    verify_inp( len(tickers) > 0, "At least one ticker must be specified")
    verify_inp( period is not None and len(period) == 2, "A start and end date must be specified")

    from cdxcore.jcpool import JCPool

    start, end         = period
    date_range         = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=not halfdays)
    pool_config        = PoolConfig(num_workers=workers, threading=threading)
    raw_day_filter     = RawDayFilter()   
    expiry_filter      = ExpiryFilter()
    dates              = sorted( set(date_range.get_business_days())-set(BAD.DAYS_ALL) )

    verbose = Context.all
    verbose.write(lambda: f"Enriching data for tickers {fmt_list(tickers)} from {fmt_date(start)} to {fmt_date(end)} {'excluding' if date_range.exclude_half_days else 'including'} half days. Total business days: {len(dates)}.")
    verbose.report(1,lambda: f"Raw Day Filter: {raw_day_filter}")
    verbose.report(1,lambda: f"Expiry filter: {expiry_filter}")
    verbose.report(1,lambda: f"Pool config: {pool_config}")

    pool = JCPool(num_workers=pool_config.num_workers, threading=pool_config.threading)

    for it, ticker in enumerate(tickers):  

        def processing( ticker, d, raw_day_filter, expiry_filter, refresh ):
            enrich_day( ticker=ticker, d=d, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, override_cache_mode="update" if refresh else "on", raise_if_bad_day=False, verbose=Context.quiet, cache_generate_only=True )
            return 0

        with verbose.write_t(lambda: f"\rProcessing ticker {it+1}/{len(tickers)} {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}:", end='') as tme:
            cnt = 0
            for d, _ in pool.parallel( { d: pool.delayed(processing)(ticker, d, raw_day_filter, expiry_filter, refresh) for d in dates } ):
                cnt += 1
                if tme.interval_test(1.):
                    verbose.write(f"\rProcessing ticker {it+1}/{len(tickers)} '{ticker}' date {cnt}/{len(dates)} {fmt_date(d)}. So far this took {tme}...", end='')
            verbose.write(f"\rProcessed ticker {it+1}/{len(tickers)} '{ticker}' for all {len(dates)} dates. This took {tme}.")

    verbose.write(f"All {len(tickers)} tickers processed.")

if __name__ == "__main__":
    main()
