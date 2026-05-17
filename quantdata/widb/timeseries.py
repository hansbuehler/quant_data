"""
Timeseries of daily data per ticker
===================================

Sampling daily data into a time series using :func:`build_time_series`.
This includes filtering and sampling the enriched day data for each day according to
the specified configuration, and optionally making the data arbitrage-free using :func:`mk_arb_free_day`.
"""
from __future__ import annotations

from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig
from quantdata.widb.bad import BAD
from quantdata.widb.bad import BAD
from quantdata.widb.day import enrich_day, RawDayFilter, ExpiryFilter, WIDBClient, SubDir
from quantdata.widb.options_filter import MkArbFree, mk_arb_free_day, OptionsFilter, filter_options_day, _filter_options_day_num_options, SCHEMA_OPTION_DAY

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.subdir import CacheMode, CacheMustExistError
from cdxcore.util import fmt_date, fmt_list, fmt_big_byte_number, fmt_dict
from cdxcore.uniquehash import named_unique_filename48_8

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

_processing_timeseries = WIDBClient.locations.processing("timeseries",fmt=SubDir.PANDAS_PARQUET)

@version("1.0", dependencies=[RawDayFilter, ExpiryFilter, MkArbFree, OptionsFilter] )
@dataclass(frozen=True)
class TimeSeriesDefinition:
    """
    User-definition of a time series for experimentation.
    """
    raw_day_filter : RawDayFilter  = RawDayFilter()  # Configuration for data collection for the raw day data for basic data cleaning using :func:`raw_collect_raw_day_day`.
    expiry_filter : ExpiryFilter   = ExpiryFilter()  # Filter for expiries to be able to compute implied forwards. This will be passed to :func:`enrich_day` for each day.
    mk_arb_free : MkArbFree|None   = MkArbFree()     # Optional linear no-arbitrage filter which generates per expiry a no-arb ``fit1`` between bid/ask, and for the entire surface ``fitX`` which  
                                                     # may not fit within bid/ask. This function also generates a ``mask mask_arb_free`` field indicating which options were
                                                     # fitted successfully (those which were not showed actionable static arbitrage on a given expiry).            
    options_filter : OptionsFilter = OptionsFilter() # How many options to extract and how, every day using :func:`filter_options_day`.

    def __str__(self):
        return f"{self.raw_day_filter},{self.expiry_filter},{self.mk_arb_free},{self.options_filter}"

def _at_uid(ticker, date_range, time_series_def ) -> str:
    ts_id  = named_unique_filename48_8(str(time_series_def), time_series_def)
    return f"{ts_id}/{ticker}/{str(date_range)}"

@_processing_timeseries(fmt=SubDir.PICKLE).cache( "0.1", dependencies=[_filter_options_day_num_options, TimeSeriesDefinition],uid=_at_uid )
def _assemble_time_series(*,
    ticker : str,
    date_range : DateRange, 
    time_series_def : TimeSeriesDefinition,
    pool_config : PoolConfig = PoolConfig.off(),
    skip_bad_days : bool = False,
    override_cache_mode : CacheMode|None = None,
    verbose : Context = Context.all ) -> PrettyObject:
    """
    Local wrapper to pre-compute daily data for a time series.
    """

    dates = sorted( set(date_range.get_business_days())-set(BAD.DAYS_ALL) )
    tme   = verbose.timer()

    verbose.write(lambda: f"Generating daily data for {ticker} time series from {dates[0]} to {dates[-1]} {'excluding' if date_range.exclude_half_days else 'including'} half days:")
    verbose.report(1,lambda: f"Options filter: {time_series_def.options_filter}")
    if not time_series_def.mk_arb_free is None:
        verbose.report(1,lambda: f"MkArbFree config: {time_series_def.mk_arb_free}")
    verbose.report(1,lambda: f"Raw Day filter: {time_series_def.raw_day_filter}")
    verbose.report(1,lambda: f"Expiry filter: {time_series_def.expiry_filter}")
    verbose.report(1,lambda: f"Pool config: {pool_config}")

    pool = JCPool(num_workers=pool_config.num_workers, threading=pool_config.threading)

    cnt         = 0
    max_options = 0
    fails       = []
    for d, cnt_options in pool.parallel( { d: pool.delayed(_filter_options_day_num_options)( 
                                        ticker=ticker,
                                        d=d,
                                        raw_day_filter=time_series_def.raw_day_filter,
                                        expiry_filter=time_series_def.expiry_filter,
                                        mk_arb_free=time_series_def.mk_arb_free,
                                        options_filter=time_series_def.options_filter,
                                        raise_if_bad_day=not skip_bad_days,
                                        override_cache_mode=override_cache_mode,
                                        verbose=None
                                         )  for d in dates } ):
        cnt += 1
        if cnt_options is None:
            fails.append(d)
        else:
            max_options = max(max_options, cnt_options)
        if tme.interval_test(1.):
            verbose.write(lambda: f"\rGenerating daily filtered options data for {ticker} date {cnt}/{len(dates)} {fmt_date(d)} @ {math.ceil(100 * cnt / len(dates))}%. Total run time so far: {tme}{'' if not fails else f', so far {len(fails)} failures'}... ", end='')

    verbose.write(lambda: f"\rGenerated daily filtered options data {len(dates)} dates for {ticker} from {dates[0]} to {dates[-1]} {'excluding' if date_range.exclude_half_days else 'including'} half days. Maximum number of options is {int(max_options)}{'' if not fails else f', with {len(fails)} failures'}. This took {tme}.")
    if fails:
        raise RuntimeError(f"Failed to generate daily filtered options data for {ticker} for the following dates: {fmt_list(sorted(fails))}.")
    return max_options

def _bt_uid(ticker, date_range, include_fields, exclude_fields, time_series_def ) -> str:
    assert isinstance(time_series_def, TimeSeriesDefinition), f"Internal error: expected 'time_series_def' to be an instance of TimeSeriesDefinition, but got {type(time_series_def)}. Please contact the authors with an example."
    ts_id  = named_unique_filename48_8(str(time_series_def), time_series_def)
    fl_id  = f"incl_{fmt_list(sorted(include_fields))}," if not include_fields is None else "incl_all,"
    fl_id += f"excl_{fmt_list(sorted(exclude_fields))}" if not exclude_fields is None else "excl_none"
    fl_id  = named_unique_filename48_8(fl_id, include_fields, exclude_fields)
    return f"{ts_id}/{ticker}/{str(date_range)}/{fl_id}"

_PER_DAY_FIELDS = {'ticker','date','spot','spot_prev','spot_ref','spot_ret','time_y'} # not all fields are present, depending on settings

@_processing_timeseries(fmt=SubDir.PYTREE_HDF5).cache( "0.1", dependencies=[filter_options_day, TimeSeriesDefinition],uid=_bt_uid )
def build_time_series(   
                  ticker : str,
                  date_range : DateRange, *,
                  include_fields : set[str]|None,
                  exclude_fields : set[str]|None, 
                  time_series_def : TimeSeriesDefinition,
                  pool_config : PoolConfig = PoolConfig.off(),
                  skip_bad_days : bool = False,
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
    time_series_def : :class:`TimeSeriesDefinition`
        Definition of the time series to be built, including the filters for data collection and sampling,
        Includes:
        
        * ``raw_day_filter`` of :class:`RawDayFilter`: 
            Configuration for data collection for the raw day data for basic data cleaning using :func:`raw_collect_raw_day_day`.
        * ``expiry_filter`` of :class:`ExpiryFilter`:
            Filter for expiries to be able to compute implied forwards. This will be passed to :func:`enrich_day` for each day.
        * ``mk_arb_free`` of :class:`MkArbFree` | None:
            optional linear no-arbitrage filter which generates per expiry a no-arb ``fit1`` between bid/ask, and for the entire surface ``fitX`` which  
            may not fit within bid/ask. This function also generates a ``mask mask_arb_free`` field indicating which options were
            fitted successfully (those which were not showed actionable static arbitrage on a given expiry).            
        * ``options_filter`` of :class:`OptionsFilter`:
            How many options to extract and how, every day using :func:`filter_options_day`.

    pool_config : :class:`PoolConfig`, default ``PoolConfig.off()``
        Configuration for parallel processing. By default, no parallel processing is used.
    skip_bad_days : bool, default ``False``
        If ``False``, processing stops upon encountering an exception when creating the data for a given day. If ``True``, processing continues but
        building the time series will fail because of the missing date.
    override_cache_mode : :class:`cdxcore.subdir.CacheMode` | None, default ``None``
        If not ``None``, overrides the cache mode for the enriched day data. Set to ``"update"`` to refresh the underlying raw cache.
    verbose : :class:`cdxcore.verbose.Context`, default ``Context.all``
        Verbose context for logging.

    Returns
    -------
    Time series: dict[str:np.array|str|type]
        A dictionary with the requested fields, up to the maximum number of options specified in ``options_filter``, for each business day in the date range.
        The fields are returned as numpy arrays, and the date field is returned as a string in "YYYY-MM-DD" format
        The data is filtered and sampled according to the specified configurations.
    """

    dates            = sorted( set(date_range.get_business_days())-set(BAD.DAYS_ALL) )
    source_schema    = SCHEMA_OPTION_DAY(time_series_def.mk_arb_free, time_series_def.options_filter.normalize)
    source_fields    = set(source_schema)                   # what the source offers
    available_fields = source_fields|{'is_valid','time_y'}  # source + what we will generate

    def get_fields():
        if not exclude_fields is None:
            verify_inp( not 'is_valid' in exclude_fields, lambda : f"'is_valid' cannot be excluded, as it is used for masking invalid options.")
            verify_inp( set(exclude_fields) <= available_fields, lambda : f"'exclude_fields' unknown fields found: {sorted(set(exclude_fields) - available_fields)}. Available fields not yet selected: {sorted(available_fields - set(exclude_fields))}")
        if not include_fields is None:
            verify_inp( 'is_valid' in include_fields, lambda : f"'is_valid' must be included in the fields to include, as it is used for masking invalid options.")
            verify_inp( set(include_fields) <= available_fields, lambda : f"'include_fields' unknown fields found: {sorted(set(include_fields) - available_fields)}. Available fields not yet selected: {sorted(available_fields - set(include_fields))}")

        flds = {f for f in available_fields if f in include_fields} if include_fields is not None else available_fields
        if exclude_fields is not None:
            flds = flds - set(exclude_fields)
        verify_inp( len(flds) > 0, lambda :  "No fields left after applying include/exclude filters. "+\
                                            f"Included fields: {sorted(include_fields) if include_fields is not None else 'all'}; "+\
                                            f"excluded fields: {sorted(exclude_fields) if exclude_fields is not None else 'none'}; "+
                                            f"available fields: {sorted(available_fields)}")
        return set(flds)
    requested_fields = get_fields()  # what the caller wants. Must include 'is_valid' for masking.
    
    # make sure all data exists
    # this is in multi-processing mode
    # --------------------------------

    _ = _assemble_time_series(
                        ticker=ticker,
                        date_range=date_range,
                        time_series_def=time_series_def,
                        pool_config=pool_config,
                        skip_bad_days=skip_bad_days,
                        override_cache_mode=override_cache_mode,
                        verbose=verbose
                    )

    verbose.write(lambda: f"Aggregating {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}data for ticker {ticker} for {len(dates)} business days from {fmt_date(dates[0])} to {fmt_date(dates[-1])} {'excluding' if date_range.exclude_half_days else 'including'} half days:")
    verbose.report(1,lambda: f"Options filter: {time_series_def.options_filter}")
    if not time_series_def.mk_arb_free is None:
        verbose.report(1,lambda: f"MkArbFree config: {time_series_def.mk_arb_free}")
    verbose.report(1,lambda: f"Raw Day filter: {time_series_def.raw_day_filter}")
    verbose.report(1,lambda: f"Expiry filter: {time_series_def.expiry_filter}")
    verbose.report(1,lambda: f"Pool config: {pool_config}")
    verbose.report(1,lambda: f"Requested fields: {fmt_list(sorted(requested_fields))}")

    # load data into current process
    # All data must exist
    # --------------------------------

    def load_data():

        thread_pool = JCPool(num_workers=pool_config.num_workers, threading=True)
        results = {} # by date
        max_options = 0

        with verbose.write_t(lambda: f"Reading the generated {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enriched data for {ticker} in the main process... ", end='') as tme:
            # read them back
            def process(d):
                try:
                    df = filter_options_day(
                                    ticker=ticker, d=d,
                                    raw_day_filter=time_series_def.raw_day_filter, 
                                    expiry_filter=time_series_def.expiry_filter,
                                    mk_arb_free=time_series_def.mk_arb_free, 
                                    options_filter=time_series_def.options_filter,
                                    override_cache_mode="cacheonly", 
                                    raise_if_bad_day=False,
                                    cache_generate_only=False,
                                    verbose=Context.quiet )
                except CacheMustExistError as e:
                    raise RuntimeError(f"Internal error: {ticker} {d}: expected {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enriched data to be available in the cache after generation, but it was not found. Please contact the authors with an example. Original error message: {str(e)}")
                assert not df is None, f"Internal error: {ticker} {d}: {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enrich_day returned None after generation. Please contact the authors with an example."                
                assert set(df.columns) == set(source_fields), f"Internal error: {ticker} {d}: {sorted(source_fields)}, but found {sorted(df.columns)}."
                assert not 'is_valid' in df.columns, f"Internal error: {ticker} {d}: 'is_valid' should not be included in the enriched day data"
                assert not 'time_y' in df.columns, f"Internal error: {ticker} {d}: 'time_y' should not be included in the enriched day data"
                return df

            for d, df in thread_pool.parallel( { d: thread_pool.delayed(process)(d) for d in dates } ):
                results[d] = df
                if tme.interval_test(1.):
                    verbose.write(lambda: f"\rReading data for {ticker} date {fmt_date(d)} {len(results)}/{len(dates)}. Total run time so far: {tme}... ", end='')
                max_options = max(max_options, len(df))

            max_options = int(max_options)
            verbose.write(lambda: f"\rRead the generated {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enriched data for {ticker} in the main process; this took {tme}. Maximum number of options is {max_options}.", end='')

        return results, max_options

    results, max_num_options = load_data()
    num_steps = len(results)

    # allocate memory
    # ===============

    with verbose.write_t(lambda: f"Generating time series for {num_steps} business days and at most {max_num_options} options per day for ticker {ticker}... ", end='') as tme:

        def allocate():
            df0        = results[list(results.keys())[0]] # touch one df to get the schema
            dates      = sorted(results)
            r          = PrettyObject()
            r.ticker   = ticker
            r.dates    = dates
            r.is_valid = np.zeros((num_steps,max_num_options), dtype=np.bool_ )

            if 'time_y' in requested_fields:
                r.time_y = np.array( [ ( d - dates[0] ).total_seconds() / ( 24 * 3600 ) for d in dates ], dtype=np.float32 ).reshape((num_steps,1))

            src_request = sorted(requested_fields-{'is_valid','time_y'})
            for c in src_request:
                v = df0[c].to_numpy()
                assert len(v.shape) == 1, f"Internal error: {ticker}: {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enriched DataFrame column '{c}' has unexpected shape. Expected 1D array, found shape {v.shape}. Please contact the authors with an example."
                assert len(v) > 0 or c in _PER_DAY_FIELDS, f"Internal error: {ticker}: {'' if time_series_def.mk_arb_free is None else 'arbitrage-free '}enriched DataFrame column '{c}' has no values. If this is a per-option field, it should have at least one value. If this is a per-day field, it should be present but can be empty. Please contact the authors with an example."
                r[c] = np.zeros((num_steps, 1 if c in _PER_DAY_FIELDS else max_num_options), dtype=v.dtype)

            for i, d in enumerate(dates):
                df = results.pop(d)
                ln = len(df)
                r.is_valid[i,:ln] = True
                for c in src_request:
                    v = df[c].to_numpy()
                    if c in _PER_DAY_FIELDS:
                        assert np.all( v[0] == v ), f"Internal error: {ticker} {d}: DataFrame column '{c}' must have a unique value. Found {v} vs {v[v[0] != v][:5]}."
                        r[c][i,0] = v[0]
                    else:
                        r[c][i,:len(v)] = v

            for k,v in r.items():
                if k in ["ticker", "dates"]:
                    continue
                if v.dtype == np.bool_:
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

    return dict(r) # for HDF5

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
