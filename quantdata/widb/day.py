"""
Generating one clean day of data
"""
from __future__ import annotations

from pandas import arrays

from quantdata.widb import WIDBClient, interpolate_zero_curve
from quantdata.biz_calendar import count_business_days_between, DAYS_PER_YEAR, AM_EXIRY_SUBTRACT, to_date
from quantdata.widb.client import WIDBClient
from quantdata.widb.bad import BAD

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir
from cdxcore.util import fmt_date, fmt_list, fmt_dict
from cdxcore.bs import bs
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import named_unique_filename48_8

import datetime as dt
import numpy as np
import pandas as pd
import polars as pl
import math as math
import warnings
from dataclasses import dataclass

_cache = WIDBClient.locations.processing("day",fmt=SubDir.PANDAS_PARQUET)

# ---------------------------------------------------------------------------
# Day's work
# ---------------------------------------------------------------------------

RAW_DAY_COLUMNS = {
    'ticker',
    'date',
    'spot',
    'call_symbol',
    'put_symbol',
    'call_opt_id',
    'put_opt_id',
    'expiry_id',
    'is_american',
    'is_am',
    'strike',
    'fwd_ref',
    'df',
    'expiry',
    'bdte',
    'tte_y', 
    'mkt_call_ask', 
    'mkt_call_bid', 
    'mkt_call_log_volume', 
    'mkt_call_log_open_interest', 
    'mkt_put_ask',
    'mkt_put_bid', 
    'mkt_put_log_volume',
    'mkt_put_log_open_interest', 
}

@version("0.1")
@dataclass(frozen=True)
class CollectConfig:
    min_bid : float = 0.011  # minimum bid value, typically above one tick.

    def __post_init__(self):
        verify_inp(self.min_bid >= 0., lambda : f"'min_bid' must be non-negative, found {self.min_bid:.3f}")

    def __str__(self):
        return f"min_bid {self.min_bid:.3f}"

@_cache(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.2",
            dependencies=[interpolate_zero_curve, WIDBClient],
            uid=lambda ticker, d, collect_config : f"{named_unique_filename48_8(f'day {str(collect_config)}',collect_config)}/{ticker}/{fmt_date(d)}",
            exclude_args=['verbose','secid'])
def collect_raw_day( ticker : str, 
                     d : dt.date, 
                     collect_config : CollectConfig, *, 
                     verbose : Context, secid : int|None = None ) -> pd.DataFrame:
    """
    Collect raw data with minimal analytics for a given ticker and date.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    d : :class:`datetime.date`
        Date for which to collect data
    config : :class:`CollectConfig`
        Configuration for data collection
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging

    Returns
    -------
    Daily data: :class:`pandas.DataFrame`
        Cleaned DataFrame with all relevant data for the given ticker and date, containing columns listed in :attr:`RAW_DAY_COLUMNS`.

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
        raise IndexError(f"{d} is in the BAD.DAYS_ALL set. No data will be downloaded for this date. Please check the date range and calendar settings. BAD.DAYS_ALL={BAD.DAYS_ALL}")
    if d in  BAD.DAYS_TICKER.get(ticker, set()):
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
            opts_pd["best_bid"].fillna(0.0).astype(float) <= collect_config.min_bid
        ) 
        if zero_quotes.any():
            n_before = len(opts_pd)
            opts_pd = opts_pd.loc[~zero_quotes].copy()
            n_removed = n_before - len(opts_pd) 
            if opts_pd.empty:
                raise EOFError(
                    f"No option data found for {ticker} on {d} after removing {n_removed}/{n_before} options with bid <= {collect_config.min_bid}."
                )
            if n_removed > 0:
                status_removed.append(f"{n_removed}/{n_before} options with bid <= {collect_config.min_bid}")
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
        r.spot = float(spot)
        r.date = d
        r.ticker = ticker
        r.call_symbol = opts_pd["call_symbol"].astype(str).values.tolist()
        r.put_symbol = opts_pd["put_symbol"].astype(str).values.tolist()
        r.call_opt_id = opts_pd["call_opt_id"].to_numpy(dtype=np.int32)
        r.put_opt_id = opts_pd["put_opt_id"].to_numpy(dtype=np.int32)
        r.is_american = opts_pd["is_american"].to_numpy(dtype=bool)
        r.is_am = opts_pd["is_am"].to_numpy(dtype=bool)
        r.expiry = pd.to_datetime(opts_pd["exdate"]).dt.date.values.astype("datetime64[D]")
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
    if set(result.columns) != RAW_DAY_COLUMNS:
        too_much = set(result.columns) - RAW_DAY_COLUMNS
        too_little = RAW_DAY_COLUMNS - set(result.columns)
        raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame has unexpected columns. Superflous: {fmt_list(too_much)}. Missing: {fmt_list(too_little)}. Please contact the authors with an example. ")

    verbose.write(f"{ticker} {d}: collected {len(result)}/{n_total} options. Removed: {fmt_list(status_removed) if len(status_removed) > 0 else 'none'}.")
    return result

ENRICHED_DAY_COLUMNS = RAW_DAY_COLUMNS | {
    'fwd',          # implied forward from put-call parity
    'call_ask_ref', # actual synthetic call ask
    'call_bid_ref', # actual synthetic call bid
    'call_ask',     # implied from pure call after implied vol fitting
    'call_bid',     # implied from pure call after implied vol fitting
    'ask_vol',
    'bid_vol',
    'log_volume',
    'log_oi',
    'ask_pure_price',
    'bid_pure_price',
    'ask_pure_delta',
    'bid_pure_delta',
    'ask_pure_gamma',
    'bid_pure_gamma',
    'bid_pure_vega',
    'ask_pure_vega',
    'bid_pure_theta',
    'ask_pure_theta',
    'ask_pure_dk',
    'bid_pure_dk'
}

@version("0.2")
@dataclass(frozen=True)
class ExpiryFilter:
    """ How to filter expiries to be able to compute implied forwards """
    num_min_either_traded : int = 10
    num_min_both_traded : int = 4

def _uid(ticker, d, collect_config, expiry_filter) -> str:
    return f"{named_unique_filename48_8(f'enriched {str(expiry_filter)}',expiry_filter,collect_config)}"+\
           f"/{ticker}/{fmt_date(d)}"

@_cache(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.2",
            dependencies=[collect_raw_day, ExpiryFilter],
            uid=_uid,
            exclude_args=['verbose','data'])
def enrich_day( ticker : str, d : dt.date, data : pd.DataFrame|None = None, *, 
                collect_config : CollectConfig, 
                expiry_filter : ExpiryFilter, 
                verbose : Context = Context.quiet ):
    """
    Enrich a day's data by:

    * Compute synthetic minimal spread call options.
    * Compute an implied forward from inverse-spread weighted bid/ask call prices.
    * Compute implied volatilities and pure greeks.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    d : :class:`datetime.date`
        Date for which to collect data
    data : :class:`pandas.DataFrame` | None, default ``None``
        Raw data for the given ticker and date. If ``None``, data will be generated
        using :func:`collect_raw_day`.
    collect_config : :class:`CollectConfig`
        Configuration for data collection
    expiry_filter : :class:`ExpiryFilter`
        Filter for expiries to be able to compute implied forwards
    verbose : :class:`cdxcore.verbose.Context`
        Verbose context for logging

    Returns
    -------
    Daily data: :class:`pandas.DataFrame`
        Cleaned DataFrame with all relevant data for the given ticker and date, containing columns
        listed in :attr:`ENRICHED_DAY_COLUMNS`.

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

    data         = data if data is not None else collect_raw_day(ticker, d, collect_config, verbose=verbose)
    expiry_ids   = data["expiry_id"].to_numpy()
    unq_ids      = np.unique(expiry_ids)
    num_total    = len(data)
    num_expiries = len(unq_ids)

    put_log_vol  = data['mkt_put_log_volume'].to_numpy()
    call_log_vol = data['mkt_call_log_volume'].to_numpy()
    put_log_oi   = data['mkt_put_log_open_interest'].to_numpy()
    call_log_oi  = data['mkt_call_log_open_interest'].to_numpy()

    either_cp   = (put_log_vol>0)|(call_log_vol>0)
    both_cp     = (put_log_vol>0)&(call_log_vol>0)
    fwd_ref     = data['fwd_ref'].to_numpy()
    df_ref      = data['df'].to_numpy()
    strike      = data['strike'].to_numpy()
    expiries    = data['expiry'].to_numpy()
    is_am       = data['is_am'].to_numpy()

    mkt_call_ask = data['mkt_call_ask'].to_numpy()
    mkt_call_bid = data['mkt_call_bid'].to_numpy()
    mkt_put_ask  = data['mkt_put_ask'].to_numpy()
    mkt_put_bid  = data['mkt_put_bid'].to_numpy()

    dffwd_ask    = ( mkt_call_ask - mkt_put_bid ) + df_ref*strike   # C-P=DF(FWD-strike)
    dffwd_bid    = ( mkt_call_bid - mkt_put_ask ) + df_ref*strike
    dffwd_mid    = 0.5*(dffwd_ask + dffwd_bid)
    dffwd_wght   = np.where( both_cp, 1./np.maximum( dffwd_ask - dffwd_bid, 0.001 ), 0. )

    # compute dffwd_mean grouped by expiry_id
    arrays            = PrettyObject()    
    arrays.fwd        = np.zeros_like( dffwd_ask )
    arrays.is_valid   = np.zeros_like( arrays.fwd, dtype=bool )
    arrays.call_ask   = np.zeros_like( arrays.fwd )
    arrays.call_bid   = np.zeros_like( arrays.fwd )
    arrays.call_ask_ref   = np.zeros_like( arrays.fwd )
    arrays.call_bid_ref   = np.zeros_like( arrays.fwd )
    arrays.ask_vol    = np.zeros_like( arrays.fwd )
    arrays.bid_vol    = np.zeros_like( arrays.fwd )
    arrays.log_volume = np.log10( np.pow(10.,call_log_vol)+np.pow(10.,put_log_vol)-1 )
    arrays.log_oi     = np.log10( np.pow(10.,call_log_oi)+np.pow(10.,put_log_oi)-1 )
    arrays.ask_pure_price  = np.zeros_like( arrays.fwd )
    arrays.bid_pure_price  = np.zeros_like( arrays.fwd )
    arrays.ask_pure_delta  = np.zeros_like( arrays.fwd )
    arrays.bid_pure_delta  = np.zeros_like( arrays.fwd )
    arrays.ask_pure_gamma  = np.zeros_like( arrays.fwd )
    arrays.bid_pure_gamma  = np.zeros_like( arrays.fwd )
    arrays.bid_pure_vega   = np.zeros_like( arrays.fwd )
    arrays.ask_pure_vega   = np.zeros_like( arrays.fwd )
    arrays.bid_pure_theta  = np.zeros_like( arrays.fwd )
    arrays.ask_pure_theta  = np.zeros_like( arrays.fwd )
    arrays.ask_pure_dk     = np.zeros_like( arrays.fwd )
    arrays.bid_pure_dk     = np.zeros_like( arrays.fwd )

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

    data = data.copy(deep=False)
    drop = set(data.columns) - set(ENRICHED_DAY_COLUMNS)
    if len(drop) > 0:
        data.drop(columns=drop, inplace=True)
    for r, v in arrays.items():
        data[r] = v
    data = data.loc[data["is_valid"]].drop(columns="is_valid")

    if set(data.columns) != ENRICHED_DAY_COLUMNS:
        too_much = set(data.columns) - ENRICHED_DAY_COLUMNS
        too_little = ENRICHED_DAY_COLUMNS - set(data.columns)
        raise RuntimeError(f"Internal error: {ticker} {d}: enriched DataFrame has unexpected columns. Superflous: {fmt_list(too_much)}. Missing: {fmt_list(too_little)}. Please contact the authors with an example. ")

    if len(dropped_expiries) > 0:
        verbose.write(lambda : f"{ticker} {d}: Enriched {len(data)} of {num_total} raw options across {num_expiries} expiries, of which the following {len(dropped_expiries)} were dropped: {fmt_list(dropped_expiries)}.")
    else:
        assert len(data) == num_total, f"Internal error: {ticker} {d}: no expiries were dropped but the number of rows changed from {num_total} to {len(data)}. Please contact the authors with an example."
        verbose.write(lambda : f"{ticker} {d}: Enriched {len(data)} raw options across {num_expiries} expiries. No expiries were dropped.")
    return data

