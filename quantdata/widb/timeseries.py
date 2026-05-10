"""
Sampling daily data into a time series.
"""
from __future__ import annotations

from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig
from quantdata.widb.day import enrich_day, SCHEMA_ENRICHED_DAY, RawDayFilter, ExpiryFilter
from quantdata.widb.arbfree import MkArbFree, mk_arb_free_day, SCHEMA_ARB_FREE_DAY_EXPIRY, SCHEMA_ARB_FREE_DAY_SURFACE, ENRICHED_PER_DAY_FIELDS

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.subdir import CacheMode, CacheMustExistError
from cdxcore.util import fmt_date, fmt_list, fmt_big_byte_number, fmt_dict

from cdxcore.err import verify_inp, verify
from cdxcore.pretty import PrettyObject
from cdxcore.jcpool import JCPool

import datetime as dt
import numpy as np
import pandas as pd
import math as math
from prettytable import PrettyTable
from dataclasses import dataclass, field
import warnings as warnings

# ======================================================================
# Sample days for a given ticker
# ======================================================================

@version("0.61")
@dataclass(frozen=True)
class TimeseriesConfig:
    """
    Create a time series with a maximum number of valid options per time step.
    """
    max_num_options: int = 1000                 #: Maximum number of options. Options are sorted by NMS, expiry and call/put, and only the top max_num_options are kept. This is applied after all other filters.#
    max_business_days_to_expiry: int = 252*3    #: Maximum business days to expiry. 
    max_nms: float = 4.                         #: Maximum |NMS|. 
    min_vegaSqrtVar: float = 0.001              #: Minimum vega * sqrt(tte), i.e. the derivative in sqrtVar: BSsq(K,sqrtVar) = BSvol(K,sqrtVar/sqrt(t),t). This is a measure of how much the option price moves with volatility, normalized by time to expiry. It is used to filter out options that are too close to expiry or too far OTM, which can have very low vega and lead to noisy implied volatilities.
    min_log_volume: float = 1.0                 #: Minimum log volume. 
    min_log_open_interest: float = 1.0          #: Minimum log open interest. 
    remove_am: bool = False                     #: Whether to remove options that settled at the AM price. 
    min_opt_per_expiry: int = 10                #: Minimum number of options per expiry. This is applied *before* the max_num_options filter, and can lead to fewer than max_num_options being returned if there are many expiries with few options.

    def __post_init__(self):
        verify_inp(self.max_num_options >= 10, lambda : f"'max_num_options' must be at least 10, found {self.max_num_options}")
        verify_inp(self.max_business_days_to_expiry > 0, lambda : f"'max_business_days_to_expiry' must be positive, found {self.max_business_days_to_expiry}")
        verify_inp(self.max_nms > 0, lambda : f"'max_nms' must be positive, found {self.max_nms}")
        verify_inp(self.min_vegaSqrtVar >= 0, lambda : f"'min_vegaSqrtVar' must be non-negative, found {self.min_vegaSqrtVar}")
        verify_inp(self.min_log_volume >= 0, lambda : f"'min_log_volume' must be non-negative, found {self.min_log_volume}")
        verify_inp(self.min_log_open_interest >= 0, lambda : f"'min_log_open_interest' must be non-negative, found {self.min_log_open_interest}")
        verify_inp(self.min_opt_per_expiry > 0, lambda : f"'min_opt_per_expiry' must be positive, found {self.min_opt_per_expiry}")

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"nopt={self.max_num_options},bdte={self.max_business_days_to_expiry},nms={self.max_nms},lvolu={self.min_log_volume},loi={self.min_log_open_interest},rmam={self.remove_am},vegaSV={self.min_vegaSqrtVar},mxpp={self.min_opt_per_expiry}"
    
    def _filter(self, ticker : str, d : dt.date, df : pd.DataFrame) -> pd.DataFrame:
        """ Remove all relevant data """

        assert set(df.columns) >= set(SCHEMA_ENRICHED_DAY), f"Internal error {ticker} {d}: DataFrame has missing columns. Missing: {fmt_list(set(SCHEMA_ENRICHED_DAY) - set(df.columns))}. Please contact the authors with an example."
        assert not df.isna().any().any(), f"Internal error {ticker} {d}: filter resulted in NaN values. Please contact the authors with an example."
        n_init = len(df)

        if not df.empty:
            df = df[df['bdte'] <= self.max_business_days_to_expiry]
            df = df[np.abs(df['nms']) <= self.max_nms]
            df = df[df['log_volume'] >= self.min_log_volume]  if self.min_log_volume > 0. else df
            df = df[df['log_open_interest'] >= self.min_log_open_interest] if self.min_log_open_interest > 0. else df
            if self.remove_am:
                df = df[~df['is_am']]

            # filter expiries
            if not df.empty:
                expiry_counts = df['expiry_id'].value_counts()
                valid_expiries = expiry_counts[expiry_counts >= self.min_opt_per_expiry].index
                df = df[df['expiry_id'].isin(valid_expiries)]
                if df.empty:
                    expiry_counts = expiry_counts.to_dict()
                    expiry_counts = { d: expiry_counts[d] for d in sorted(expiry_counts) }
                    warnings.warn(f"{ticker} {d}: after filtering no expiry had {self.min_opt_per_expiry} options left. This means this sample is not eligible. Here are the options per filtered expiry: {fmt_dict(expiry_counts)}.")
                    return df

            # remove excess options while keeping min_opt_per_expiry per expiry, and the closes to ATM
            if not df.empty and len(df) > self.max_num_options:
                nms = df['nms'].to_numpy()
                abs_nms = np.abs(nms)

                # Reuse per-expiry positional indices instead of repeatedly filtering the full frame.
                reserved_positions = []
                remaining_positions = []

                for expiry_id, expiry_positions in df.groupby('expiry_id', sort=False).indices.items():
                    expiry_positions = np.asarray(expiry_positions, dtype=np.intp)

                    assert len(expiry_positions) >= self.min_opt_per_expiry, f"Internal error: expiry_id {expiry_id} has fewer options than 'min_opt_per_expiry' after filtering. This should not happen, as we already filtered out expiries with too few options. Please contact the authors with an example."
                    if len(expiry_positions) == self.min_opt_per_expiry:
                        reserved_positions.append(expiry_positions)
                        continue

                    expiry_nms = nms[expiry_positions]
                    selected_mask = np.zeros((len(expiry_positions),), dtype=bool)
                    selected_positions = []

                    # Add smallest positive NMS.
                    pos_local = np.flatnonzero(expiry_nms > 0.)
                    if len(pos_local) > 0:
                        best_pos_local = pos_local[np.argmin(expiry_nms[pos_local])]
                        selected_mask[best_pos_local] = True
                        selected_positions.append(expiry_positions[best_pos_local:best_pos_local+1])

                    # Add highest negative NMS.
                    neg_local = np.flatnonzero(expiry_nms < 0.)
                    if len(neg_local) > 0:
                        best_neg_local = neg_local[np.argmax(expiry_nms[neg_local])]
                        if not selected_mask[best_neg_local]:
                            selected_mask[best_neg_local] = True
                            selected_positions.append(expiry_positions[best_neg_local:best_neg_local+1])

                    # Add (min_opt_per_expiry - 2) remaining options with smallest abs(NMS).
                    remaining_count = self.min_opt_per_expiry - np.sum(selected_mask)
                    if remaining_count > 0:
                        remaining_local = np.flatnonzero(~selected_mask)
                        if len(remaining_local) > 0:
                            remaining_order = remaining_local[np.argsort(abs_nms[expiry_positions[remaining_local]], kind='stable')]
                            extra_local = remaining_order[:remaining_count]
                            selected_mask[extra_local] = True
                            selected_positions.append(expiry_positions[extra_local])

                    reserved_positions.append(np.concatenate(selected_positions) if selected_positions else np.empty((0,), dtype=np.intp))

                    expiry_remaining = expiry_positions[~selected_mask]
                    if len(expiry_remaining) > 0:
                        remaining_positions.append(expiry_remaining)

                reserved_positions = np.concatenate(reserved_positions) if reserved_positions else np.empty((0,), dtype=np.intp)
                remaining_positions = np.concatenate(remaining_positions) if remaining_positions else np.empty((0,), dtype=np.intp)

                # Sort remaining by abs_nms and limit.
                remaining_slots = max(0, self.max_num_options - len(reserved_positions))
                if remaining_slots > 0 and len(remaining_positions) > 0:
                    remaining_positions = remaining_positions[np.argsort(abs_nms[remaining_positions], kind='stable')[:remaining_slots]]
                else:
                    remaining_positions = np.empty((0,), dtype=np.intp)

                # Combine reserved and remaining.
                df = df.iloc[np.concatenate([reserved_positions, remaining_positions])]
                assert len(df) <= self.max_num_options, f"Internal error {ticker} {d}: after filtering by max_num_options, the number of options exceeds max_num_options. Please contact the authors with an example."

            if df.empty:
                raise RuntimeError(f"Internal error {ticker} {d}: filter resulted in an empty data set. Please contact the authors with an example.")

        assert not df.isna().any().any(), f"Internal error {ticker} {d}: filter resulted in NaN values. Please contact the authors with an example."
        return df

def SCHEMA_TIME_SERIES( mk_arb_free: MkArbFree|None ) -> list[str]:  
    """
    Returns the schema for the time series object depending on the :class:`MkArbFree` configuration.
    """
    if mk_arb_free is None:
        source_schema = SCHEMA_ENRICHED_DAY
    elif mk_arb_free.no_arb_mode == "expiry":
        source_schema = SCHEMA_ARB_FREE_DAY_EXPIRY
    elif mk_arb_free.no_arb_mode == "surface":
        source_schema = SCHEMA_ARB_FREE_DAY_SURFACE
    else:
        raise ValueError(f"Unknown no_arb_mode: {mk_arb_free.no_arb_mode}")
    return source_schema


def _enrich_or_mk_arb_free( *,
    ticker: str,
    d: dt.date,
    raw_day_filter: RawDayFilter,
    expiry_filter: ExpiryFilter,
    mk_arb_free: MkArbFree,
    override_cache_mode: CacheMode | None,
    raise_if_bad: bool,
    cache_generate_only : bool,
    return_time_taken : bool,
    warn_if_arb : bool,
    proceed_upon_bad_data : bool,
    verbose: Context ):
    """ wrapper """

    tme = verbose.timer()

    if proceed_upon_bad_data:
        try:
            return _enrich_or_mk_arb_free(
                ticker=ticker, d=d,
                raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, mk_arb_free=mk_arb_free,
                override_cache_mode=override_cache_mode, raise_if_bad=raise_if_bad, cache_generate_only=cache_generate_only,
                return_time_taken=return_time_taken, warn_if_arb=warn_if_arb, proceed_upon_bad_data=False, verbose=verbose )
        except Exception as exc:
            warnings.warn(f"{ticker} {d}: encountered an error while generating data for this day [will proceed] -- {str(exc)}.")
            r = None
    else:
        if mk_arb_free is None:
            r = enrich_day( ticker=ticker, d=d,
                            raw_day_filter=raw_day_filter, 
                            expiry_filter=expiry_filter, 
                            raise_if_bad=raise_if_bad, 
                            verbose=verbose, 
                            override_cache_mode=override_cache_mode, 
                            cache_generate_only=cache_generate_only )
        else:
            r = mk_arb_free_day(
                            ticker=ticker, d=d,
                            raw_day_filter=raw_day_filter, 
                            expiry_filter=expiry_filter, 
                            mk_arb_free=mk_arb_free,
                            raise_if_bad=raise_if_bad, 
                            verbose=verbose, 
                            warn_if_arb=warn_if_arb,
                            override_cache_mode=override_cache_mode, 
                            cache_generate_only=cache_generate_only )
            
    if return_time_taken:
        return r, tme.seconds
    return r

@version("0.1",dependencies=[enrich_day, mk_arb_free_day, TimeseriesConfig] )
def build_time_series(   
                  ticker : str,
                  date_range : DateRange, *,
                  include_fields : set[str]|None,
                  exclude_fields : set[str]|None, 
                  raw_day_filter : RawDayFilter, 
                  expiry_filter : ExpiryFilter, 
                  mk_arb_free : MkArbFree|None,
                  timeseries_config : TimeseriesConfig,
                  pool_config : PoolConfig = PoolConfig.off(),
                  proceed_upon_bad_data : bool = False,
                  warn_if_arb : bool = True,
                  override_cache_mode : CacheMode|None = None,
                  verbose : Context = Context.all ) -> dict[str:np.array|str|type]:
    """
    Collect enriched data for a list of tickers and a date range.

    This function collects daily data according to its ``raw_day_filter`` for the specified fields (which are usually defined by the target model).

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPX"
    date_range : :class:`DateRange`
        Date range for which to collect data. Only business days will be collected, and the date range can be configured to include or exclude half days.
    include_fields : set[str] | None
        Fields to include. If ``None``, all fields are included by default.
    exclude_fields : set[str] | None
        Fields to exclude, if ``include_fields`` is ``None``.
    raw_day_filter : :class:`RawDayFilter`
        Configuration for data collection for the raw day data. This will be passed to :func:`enrich_day` for each day.
    expiry_filter : :class:`ExpiryFilter`
        Filter for expiries to be able to compute implied forwards. This will be passed to :func:`enrich_day` for each day.
    mk_arb_free : :class:`MkArbFree` | None
        If not ``None``, the generated time series will be made arbitrage-free in the sense of :func:`mk_arb_free_day`.
    timeseries_config : :class:`TimeseriesConfig`
        Configuration for filtering and sampling the enriched day data.
    pool_config : :class:`PoolConfig`, default ``PoolConfig.off()``
        Configuration for parallel processing. By default, no parallel processing is used.
    proceed_upon_bad_data : bool, default ``False``
        If ``False``, processing stops upon encountering an exception when creating the data for a given day. If ``True``, processing continues but
        building the time series will fail because of the missing date.
    warn_if_arb : bool, default ``True``
        If ``True``, warnings about detected arbitrage opportunities in the data are shown. This only has an effect if ``mk_arb_free`` is not ``None``.
        Set to ``False`` to turn off reporting.
    override_cache_mode : :class:`cdxcore.subdir.CacheMode` | None, default ``None``
        If not ``None``, overrides the cache mode for the enriched day data. Set to ``"update"`` to refresh the underlying raw cache.
    verbose : :class:`cdxcore.verbose.Context`, default ``Context.all``
        Verbose context for logging.

    Returns
    -------
    Time series: dict[str:np.array|str|type]
        A dictionary with the requested fields, up to the maximum number of options specified in ``timeseries_config``, for each business day in the date range.
        The fields are returned as numpy arrays, and the date field is returned as a string in "YYYY-MM-DD" format
        The data is filtered and sampled according to the specified configurations.
    """

    dates = date_range.get_business_days()

    verbose.write(lambda: f"Collecting {'' if mk_arb_free is None else 'arbitrage-free '}data for ticker {ticker} for {len(dates)} business days from {fmt_date(dates[0])} to {fmt_date(dates[-1])} {'excluding' if date_range.exclude_half_days else 'including'} half days.")
    verbose.report(2,lambda: f"Timeseries config: {timeseries_config}")
    if not mk_arb_free is None:
        verbose.report(2,lambda: f"MkArbFree config: {mk_arb_free}")
    verbose.report(2,lambda: f"Raw Day filter: {raw_day_filter}")
    verbose.report(2,lambda: f"Expiry filter: {expiry_filter}")
    verbose.report(2,lambda: f"Pool config: {pool_config}")

    source_schema = SCHEMA_TIME_SERIES(mk_arb_free)
    source_fields = set(source_schema)

    def get_fields():
        available_fields = source_fields|{'is_valid'}
        if not exclude_fields is None:
            verify_inp( set(exclude_fields) <= available_fields, lambda : f"'exclude_fields' must be a subset of {sorted(available_fields)}; unknown fields found: {sorted(set(exclude_fields) - available_fields)}")
        if not include_fields is None:
            verify_inp( set(include_fields) <= available_fields, lambda : f"'include_fields' must be a subset of {sorted(available_fields)}; unknown fields found: {sorted(set(include_fields) - available_fields)}")

        flds = {f for f in source_fields if f in include_fields} if include_fields is not None else set(source_fields)
        if exclude_fields is not None:
            flds = flds - set(exclude_fields)
        verify_inp( len(flds) > 0, lambda :  "No fields left after applying include/exclude filters. "+\
                                            f"Included fields: {sorted(include_fields) if include_fields is not None else 'all'}; "+\
                                            f"excluded fields: {sorted(exclude_fields) if exclude_fields is not None else 'none'}; "+
                                            f"available fields: {sorted(available_fields)}")
        
        return sorted(flds)

    fields = get_fields()
    
    def generate():

        def filter_fields( d, df ):
            if df is None or df.empty:
                return df
            df  = df.loc[:, fields] 
            assert set(df.columns) == set(fields), f"Internal error: after filtering, the DataFrame columns do not match the expected fields. Expected: {sorted(fields)}, found: {sorted(df.columns)}. Please contact the authors with an example."
            return df

        thread_pool = JCPool(num_workers=pool_config.num_workers, threading=True)
        existing = {}
        missing = []

        if not override_cache_mode is None and not override_cache_mode.read:
            if override_cache_mode.write:
                verbose.write("Regenerating all files.")
            else:
                verbose.write("Regenerating all files without writing them to cache.")
            missing = dates
        else:
            with verbose.write_t(lambda: f"Checking which {'' if mk_arb_free is None else 'arbitrage-free '}enriched daily data files for ticker {ticker} and the {len(dates)} business days from {fmt_date(dates[0])} to {fmt_date(dates[-1])} {'excluding' if date_range.exclude_half_days else 'including'} half days exist in the cache... ", end='') as tme:
                def process(d):
                    try:
                        df = _enrich_or_mk_arb_free(
                                        ticker=ticker, d=d, 
                                        raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, mk_arb_free=mk_arb_free,
                                        verbose=Context.quiet, 
                                        raise_if_bad=False, 
                                        override_cache_mode="cacheonly",
                                        return_time_taken=False,
                                        warn_if_arb=warn_if_arb,
                                        proceed_upon_bad_data=False,
                                        cache_generate_only=False )
                    except CacheMustExistError:
                        return None
                    df = timeseries_config._filter(ticker,d,df)
                    df = filter_fields(d, df)
                    return df

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
            if not override_cache_mode is None and not override_cache_mode.write:
                raise RuntimeError(f"{ticker}: {len(missing)} business days are missing from cache (cache generation is off).")

            # generate missing data with parallel processing
            pool = JCPool(num_workers=pool_config.num_workers, threading=False)

            def process( ticker, d, raw_day_filter, expiry_filter, override_cache_mode, warn_if_arb, proceed_upon_bad_data ):
                r, t = _enrich_or_mk_arb_free(
                                       ticker=ticker, d=d,
                                       raw_day_filter=raw_day_filter, 
                                       expiry_filter=expiry_filter,
                                       mk_arb_free=mk_arb_free,
                                       raise_if_bad=False, 
                                       verbose=Context.quiet, 
                                       return_time_taken=True,
                                       warn_if_arb=warn_if_arb,
                                       proceed_upon_bad_data=proceed_upon_bad_data,
                                       override_cache_mode=override_cache_mode, 
                                       cache_generate_only=True )
                return d, t

            tme = verbose.timer()
            cnt = 0
            for d, seconds in pool.parallel( pool.delayed(process)( ticker, d, 
                                                                   raw_day_filter=raw_day_filter, 
                                                                   expiry_filter=expiry_filter, 
                                                                   override_cache_mode=override_cache_mode,
                                                                   warn_if_arb=warn_if_arb,
                                                                   proceed_upon_bad_data=proceed_upon_bad_data ) for d in missing ):
                cnt += 1
                if tme.interval_test(1.):
                    verbose.write(lambda: f"\r{ticker}: generating {'' if mk_arb_free is None else 'arbitrage-free '}enriched data for {cnt}/{len(missing)} missing business days using {pool_config.num_workers} processes. Processed {d} which took {seconds:.2f}s. Total run time so far: {tme}... ", end='')
            verbose.write(lambda: f"\rGenerated all {len(missing)} missing business days for {ticker} using {pool_config.num_workers} processes. This took {tme}.")

            pool.terminate()

            with verbose.write_t(lambda: f"Reading the generated {'' if mk_arb_free is None else 'arbitrage-free '}enriched data for the {len(missing)} missing business days for {ticker} in the main process... ", end='') as tme:
                # read them back
                def process(d):
                    try:
                        df = _enrich_or_mk_arb_free(
                                        ticker=ticker, d=d,
                                        raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, mk_arb_free=mk_arb_free, 
                                        override_cache_mode="cacheonly", 
                                        raise_if_bad=False,
                                        cache_generate_only=False,
                                        warn_if_arb=warn_if_arb,
                                        proceed_upon_bad_data=False,
                                        return_time_taken=False,
                                        verbose=Context.quiet )
                    except CacheMustExistError as e:
                        raise RuntimeError(f"Internal error: {ticker} {d}: expected {'' if mk_arb_free is None else 'arbitrage-free '}enriched data to be available in the cache after generation, but it was not found. Please contact the authors with an example. Original error message: {str(e)}")
                    assert not df is None, f"Internal error: {ticker} {d}: {'' if mk_arb_free is None else 'arbitrage-free '}enrich_day returned None after generation. Please contact the authors with an example."
                    df = timeseries_config._filter(ticker, d, df)
                    df = filter_fields(d, df)
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
            df0   = results[list(results.keys())[0]] # touch one df to get the schema
            dates = sorted(results)
            r = PrettyObject()
            r.is_valid = np.zeros((num_steps,max_num_options), dtype=bool )
            r.ticker = ticker
            r.dates  = dates
            per_day_fields = set(ENRICHED_PER_DAY_FIELDS)

            for c in df0.columns:
                v = df0[c].to_numpy()
                assert len(v.shape) == 1, f"Internal error: {ticker}: {'' if mk_arb_free is None else 'arbitrage-free '}enriched DataFrame column '{c}' has unexpected shape. Expected 1D array, found shape {v.shape}. Please contact the authors with an example."

                if len(v) == 0 and not c in per_day_fields:
                    per_day_fields.add(c)
                r[c] = np.zeros((num_steps, 1 if c in per_day_fields else max_num_options), dtype=v.dtype)

            for i, d in enumerate(dates):
                df = results.pop(d)
                ln = len(df)
                r.is_valid[i,:ln] = True
                for c in df.columns:
                    v = df[c].to_numpy()
                    if c in per_day_fields:
                        assert np.all( v[0] == v ), f"Internal error: {ticker} {d}: {'' if mk_arb_free is None else 'arbitrage-free '}enriched DataFrame column '{c}' must have a unique value. Found {v} vs {v[v[0] != v][:5]}."
                        r[c][i,0] = v[0]
                    else:
                        r[c][i,:len(v)] = v

            for k,v in r.items():
                if k in ["ticker", "dates"]:
                    continue
                if v.dtype != np.dtype("O"):
                    try:
                        assert np.all( np.isfinite(v) ), f"Internal error: {ticker}: array for column '{k}' contains non-finite values. Please contact the authors with an example."
                    except TypeError as e:
                        raise RuntimeError(f"Internal error: {ticker}: failed to check for finite values in column '{k}' with dtype {v.dtype}. Please contact the authors with an example. Original error message: {str(e)}")
            return r
        
        r = allocate()

        verbose.write(lambda: f"done; this took {tme}.", head=False)
        if verbose.shall_report(2):
            verbose.report(2, str(table_of_time_series(r)))
    return r

def table_of_time_series( r : dict[str:np.array|str|type] ) -> PrettyTable:
    """
    Summarizes the sampled data from :func:`build_time_series` in a :class:`prettytable.PrettyTable`
    which can be displayed in a Jupyter notebook or printed in the console.

    The table includes the shape, dtype, size in bytes, mean, std, min and max for each field in the sampled data.
    """
    nsize = 0
    tbl = PrettyTable()
    tbl.field_names = ["Field", "Shape", "Dtype", "Size", "Mean", "Std", "Min", "Max" ]
    is_valid = r['is_valid']

    for k, v in r.items():
        if k == "ticker":
            continue
        if isinstance(v, np.ndarray):
            vlid = v[is_valid] if k!="is_valid" and v.shape==is_valid.shape else v
            try:
                vlid = vlid.astype(np.float64)
            except Exception as e:
                tbl.add_row( [k, v.shape, v.dtype, fmt_big_byte_number(v.nbytes), "", "", "", ""] )
                nsize += v.nbytes   
                continue

            m,s,mi,mx = np.mean(vlid), np.std(vlid), np.min(vlid), np.max(vlid)
            tbl.add_row( [k, v.shape, v.dtype, fmt_big_byte_number(v.nbytes), f"{m:.4g}", f"{s:.4g}", f"{mi:.4g}", f"{mx:.4g}" ] )
            nsize += v.nbytes   
        elif isinstance(v, list):
            tbl.add_row([k, (len(v),), type(v[0]).__name__ if len(v) > 0 else "?", "", "", "", "", ""])
        else:
            tbl.add_row([k, str(v)[:10], "", "", "", "", "", ""])
    tbl.add_row(["TOTAL", "", "", fmt_big_byte_number(nsize), "", "", "", ""])
    return tbl
