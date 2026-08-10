"""
Options Filter
==============

Provides daily option data subject to some filter and optional normalization.
"""
from __future__ import annotations

from quantdata.widb.day import enrich_day, SCHEMA_ENRICHED_DAY, RawDayFilter, ExpiryFilter, SubDir, _processing
from quantdata.widb.arbfree import MkArbFree, mk_arb_free_day, SCHEMA_ARB_FREE_DAY_EXPIRY, SCHEMA_ARB_FREE_DAY_SURFACE

from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.subdir import CacheMode
from cdxcore.util import fmt_date, fmt_list, fmt_big_byte_number, fmt_dict
from cdxcore.uniquehash import named_unique_filename48_8
from quantdata.utils.pd import create_empty_df, verify_schema

from cdxcore.err import verify_inp, verify

import datetime as dt
import numpy as np
import pandas as pd
import math as math
from dataclasses import dataclass, field
import warnings as warnings

# ======================================================================
# Sample days for a given ticker
# ======================================================================

@version("0.62")
@dataclass(frozen=True)
class OptionsFilter:
    """
    Create a time series with a maximum number of valid options per time step.
    """
    normalize : bool = True                     # Normalize output data: 
                                                # * Drop ``spot``
                                                # * Add ``spot_ret``: the return of the spot price compared to the previous day, i.e. ``spot / spot_prev - 1``.
                                                # * Divide the following fields by spot: ``spot_prev``, ``call_ask``, ``call_bid``, ``fwd``, ``strike``, ``mkt_call_bid``, ``mkt_put_bid``, ``mkt_call_ask``, ``mkt_put_ask``.
    max_num_options: int = 1000                 #: Maximum number of options. Options are sorted by NMS5, expiry and call/put, and only the top max_num_options are kept. This is applied after all other filters.#
    max_business_days_to_expiry: int = 252*3    #: Maximum business days to expiry.
    max_nms: float|tuple[float,float] = (-5.,2.) #: Maximum |NMS5| or a tuple with NMS5_lower<0<NMS5_upper.
    nms_use_atm : bool = True                   #: Whether to use NMS5 with the ATM volatility instead of the actual NMS5. 
    min_vegaSqrtVar: float = 0.001              #: Minimum vega * sqrt(tte), i.e. the derivative in sqrtVar: BSsq(K,sqrtVar) = BSvol(K,sqrtVar/sqrt(t),t). This is a measure of how much the option price moves with volatility, normalized by time to expiry.
    min_log_volume: float = 1.0                 #: Minimum log volume. 
    min_log_open_interest: float = 1.0          #: Minimum log open interest. 
    remove_am: bool = False                     #: Whether to remove options that settled at the AM price. 
    min_opt_per_expiry: int = 10                #: Minimum number of options per expiry. This is applied *before* the max_num_options filter, and can lead to fewer than max_num_options being returned if there are many expiries with few options.    

    def __post_init__(self):
        verify_inp(self.max_num_options >= 10, lambda : f"'max_num_options' must be at least 10, found {self.max_num_options}")
        verify_inp(self.max_business_days_to_expiry > 0, lambda : f"'max_business_days_to_expiry' must be positive, found {self.max_business_days_to_expiry}")
        if isinstance(self.max_nms, tuple):
            verify_inp(len(self.max_nms) == 2, lambda : f"'max_nms' tuple must have length 2, found {len(self.max_nms)}")
            verify_inp(self.max_nms[0] < 0 < self.max_nms[1], lambda : f"'max_nms' tuple must have negative first element and positive second element, found {self.max_nms}")       
        else:
            verify_inp(self.max_nms > 0, lambda : f"'max_nms' must be positive, found {self.max_nms}")
        verify_inp(self.min_vegaSqrtVar >= 0, lambda : f"'min_vegaSqrtVar' must be non-negative, found {self.min_vegaSqrtVar}")
        verify_inp(self.min_log_volume >= 0, lambda : f"'min_log_volume' must be non-negative, found {self.min_log_volume}")
        verify_inp(self.min_log_open_interest >= 0, lambda : f"'min_log_open_interest' must be non-negative, found {self.min_log_open_interest}")
        verify_inp(self.min_opt_per_expiry > 0, lambda : f"'min_opt_per_expiry' must be positive, found {self.min_opt_per_expiry}")

    def __str__(self):#by convention use a some what readable string with comma separated key=value pairs
        return f"nopt={self.max_num_options},bdte={self.max_business_days_to_expiry},nms{'at' if self.nms_use_atm else ''}={self.max_nms},lvolu={self.min_log_volume},loi={self.min_log_open_interest},rmam={self.remove_am},vegaSV={self.min_vegaSqrtVar},mxpp={self.min_opt_per_expiry}"
    
    def _filter(self, ticker : str, d : dt.date, df : pd.DataFrame) -> pd.DataFrame:
        """
        Filter data frame
        """

        assert set(df.columns) >= set(SCHEMA_ENRICHED_DAY), f"Internal error {ticker} {d}: DataFrame has missing columns. Missing: {fmt_list(set(SCHEMA_ENRICHED_DAY) - set(df.columns))}. Please contact the authors with an example."
        assert not df.isna().any().any(), f"Internal error {ticker} {d}: filter resulted in NaN values. Please contact the authors with an example."
        n_init = len(df)

        assert not df.empty, "Internal error - this should have been filtered earlier"
        df = df[df['bdte'] <= self.max_business_days_to_expiry]

        if self.normalize:
            df = normalize_day(ticker, d, df)

        min_nms = -self.max_nms if isinstance(self.max_nms, (int, float)) else self.max_nms[0]
        max_nms = self.max_nms if isinstance(self.max_nms, (int, float)) else self.max_nms[1]

        if self.nms_use_atm:
            df = df[(df['nms_atm'] >= min_nms) & (df['nms_atm'] <= max_nms)]
        else:
            df = df[(df['nms'] >= min_nms) & (df['nms'] <= max_nms)]
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
                warnings.warn(f"{ticker} {d}: after filtering no expiry had the minimum of {self.min_opt_per_expiry} options per expiry left. This means this sample is not eligible. Here are the options per filtered expiry: {fmt_dict(expiry_counts)}.")
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

@version("0.3", dependencies=[enrich_day] )
def normalize_day(ticker : str, d : dt.date, df : pd.DataFrame) -> pd.DataFrame:
    """
    Create a normalized version of a day.

    * Drop ``spot``
    * Add ``spot_ret``: the return of the spot price compared to the previous day, i.e. ``spot / spot_prev - 1``.
    * Divide the following fields by spot: ``spot_prev``, ``call_ask``, ``call_bid``, ``fwd``, ``strike``, ``mkt_call_bid``, ``mkt_put_bid``, ``mkt_call_ask``, ``mkt_put_ask``.
    """
    df = df.copy()
    spot      = df['spot'].to_numpy()
    spot_prev = df['spot_prev'].to_numpy()

    if np.min(spot) <= 0.:
        raise ValueError(f"{ticker} {d}: Expected positive spot, got {np.min(spot):.4g} for ticker {ticker}.")
    if np.min(spot_prev) <= 0.:
        raise ValueError(f"{ticker} {d}: Expected positive spot_prev, got {np.min(spot_prev):.4g} for ticker {ticker}.")

    df.drop("spot", axis=1, inplace=True)  
    df['spot_ret']  = spot / spot_prev - 1.
    df['spot_ref']  = spot
    df['spot_prev'] /= spot
    df['call_ask']  /= spot
    df['call_bid']  /= spot
    df['fwd']       /= spot
    df['strike']    /= spot

    df['mkt_call_bid'] /= spot
    df['mkt_put_bid']  /= spot
    df['mkt_call_ask'] /= spot
    df['mkt_put_ask']  /= spot
    return df

def SCHEMA_OPTION_DAY( mk_arb_free: MkArbFree|None, normalized : bool ) -> list[str]:  
    """
    Returns the schema for an option day series object depending on the :class:`MkArbFree` configuration and whether it is normalized or not.
    """
    if mk_arb_free is None:
        source_schema = SCHEMA_ENRICHED_DAY
    elif mk_arb_free.no_arb_mode == "expiry":
        source_schema = SCHEMA_ARB_FREE_DAY_EXPIRY
    elif mk_arb_free.no_arb_mode == "surface":
        source_schema = SCHEMA_ARB_FREE_DAY_SURFACE
    else:
        raise ValueError(f"Unknown no_arb_mode: {mk_arb_free.no_arb_mode}")
    
    if normalized:
        source_schema = dict(source_schema)
        del source_schema['spot']
        source_schema['spot_ref'] = np.float32
        source_schema['spot_ret'] = np.float32

    return source_schema

@version( "0.2", dependencies=[enrich_day, mk_arb_free_day, OptionsFilter, normalize_day] )
def filter_options_day(
    ticker: str,
    d: dt.date,
    source_data : pd.DataFrame | None = None,
    *,
    raw_day_filter: RawDayFilter = RawDayFilter(),
    expiry_filter: ExpiryFilter = ExpiryFilter(),
    mk_arb_free: MkArbFree = MkArbFree(),
    options_filter : OptionsFilter = OptionsFilter(),
    # operational
    raise_if_bad_day: bool = True,
    # pass on caching parameters
    override_cache_mode: CacheMode | None = None,
    cache_generate_only : bool = False,
    # verbose
    verbose: Context = Context.all,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Filter options according to ``options_filter``.
    
    This includes filtering by moneyness, time to expiry, volume, open interest, and other criteria defined in :class:`OptionsFilter`.
    This function itself is not cached, but it since it is basically a thin layer on top of :func:`mk_arb_free_day` and the filtering logic is fast,
    caching is implicit.
    """
    verify_inp(isinstance(options_filter, OptionsFilter), lambda : f"'options_filter' must be an instance of OptionsFilter, found {type(options_filter)}")
    verify_inp(mk_arb_free is None or isinstance(mk_arb_free, MkArbFree), lambda : f"'mk_arb_free' must be None or an instance of MkArbFree, found {type(mk_arb_free)}")
    verify_inp(isinstance(raw_day_filter, RawDayFilter), lambda : f"'raw_day_filter' must be an instance of RawDayFilter, found {type(raw_day_filter)}")
    verify_inp(isinstance(expiry_filter, ExpiryFilter), lambda : f"'expiry_filter' must be an instance of ExpiryFilter, found {type(expiry_filter)}")

    schema = SCHEMA_OPTION_DAY(mk_arb_free, options_filter.normalize)

    if not source_data is None:
        verify_inp( isinstance(source_data, pd.DataFrame), f"{ticker} {d}: expected 'source_data' to be a pandas DataFrame or None, got {type(source_data)}.")
        data = source_data
    else:
        data = mk_arb_free_day(
                        ticker=ticker, d=d,
                        raw_day_filter=raw_day_filter, 
                        expiry_filter=expiry_filter, 
                        mk_arb_free=mk_arb_free,
                        raise_if_bad_day=raise_if_bad_day, 
                        verbose=verbose,
                        # pass on caching parameters
                        override_cache_mode=override_cache_mode, 
                        cache_generate_only=cache_generate_only )

        if data is None:
            assert raise_if_bad_day, "Internal error: empty data should only be returned when 'raise_if_bad_day' is False. Please check the enriched day data for details."
            return None
        
    if data.empty:
        return create_empty_df(schema)

    data = options_filter._filter(ticker, d, data)
    verify_schema(data, schema, lambda : f"{ticker} {d}")
    return data

@_processing(fmt=SubDir.PICKLE).cache(
    "0.1",
    dependencies=[filter_options_day],
    uid=lambda ticker, d, raw_day_filter, expiry_filter, mk_arb_free, options_filter :\
        f"options/{named_unique_filename48_8(f'{str(expiry_filter)},{str(raw_day_filter)},{str(mk_arb_free)},{str(options_filter)}', expiry_filter, raw_day_filter, mk_arb_free, options_filter)}"+\
        f"/{ticker}/{d}"
    )
def _filter_options_day_num_options( *,
    ticker: str,
    d: dt.date,
    raw_day_filter: RawDayFilter = RawDayFilter(),
    expiry_filter: ExpiryFilter = ExpiryFilter(),
    mk_arb_free: MkArbFree = MkArbFree(),
    options_filter : OptionsFilter = OptionsFilter(),
    # operational
    raise_if_bad_day: bool = True,
    # pass on caching parameters
    override_cache_mode: CacheMode | None = None,
    cache_generate_only : bool = False,
    # verbose
    verbose: Context|None = Context.quiet
) -> int:
    """
    Dummy wrapper around :func:`filter_options_day` to cache the number of options after filtering.
    """
    r = filter_options_day(
        ticker=ticker, d=d, 
        raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, mk_arb_free=mk_arb_free, options_filter=options_filter,
        raise_if_bad_day=raise_if_bad_day, 
        override_cache_mode=override_cache_mode, cache_generate_only=cache_generate_only,
        verbose=verbose if not verbose is None else Context.quiet
    )
    return len(r) if not r is None else None
