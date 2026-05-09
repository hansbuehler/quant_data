"""
Generate arbitrage-free option prices
"""
from __future__ import annotations

from quantdata.utils.arbfree import MkArbFree
from quantdata.widb import WIDBClient, interpolate_zero_curve
from quantdata.biz_calendar import count_business_days_between, DAYS_PER_YEAR, AM_EXIRY_SUBTRACT, to_date, NMS, prev_business_day, DateRange
from quantdata.widb.client import WIDBClient
from quantdata.widb.bad import BAD
from quantdata.pool import PoolConfig
from quantdata.utils.pd import create_empty_df, verify_schema
from quantdata.widb.day import SCHEMA_ENRICHED_DAY, _processing, RawDayFilter, ExpiryFilter, enrich_day, ENRICHED_PER_DAY_FIELDS, ENRICHED_PER_EXPIRY_FIELDS, _IMPLIED_VOL_CALC_EPS, _GREEK_CALC_EPS


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

# ---------------------------------------------------------------------------
# Enriched data, day
# Optimal forward computation, implied volatilities, (pure) greeks.
# Also translates everything in calls.
# ---------------------------------------------------------------------------

SCHEMA_ARB_FREE = SCHEMA_ENRICHED_DAY|{
    "pure_fit1_price" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fit1_delta" : np.dtype("float32"),   # the fitted price on the input strikes using the second step of the algorithm (fully no-arb fit)
    "pure_fit1_gamma" : np.dtype("float32"),   # the fitted price on the input strikes using the second step of the algorithm (fully no-arb fit)
    "pure_fit1_vega" : np.dtype("float32"),   # the fitted price on the input strikes using the second step of the algorithm (fully no-arb fit)
    "pure_fit1_theta" : np.dtype("float32"),   # the fitted price on the input strikes using the second step of the algorithm (fully no-arb fit)
    "pure_fit1_dk" : np.dtype("float32"),   # the fitted price on the input strikes using the second step of the algorithm (fully no-arb fit)
    "pure_fitX_price" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fitX_delta" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fitX_gamma" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fitX_vega" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fitX_theta" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
    "pure_fitX_dk" : np.dtype("float32"),   # the fitted price on the input strikes using the first step of the algorithm (linear fit within bid-ask)
}

def _uid(ticker, d, raw_day_filter, expiry_filter, mk_arb_free) -> str:
    return f"{named_unique_filename48_8(f'arbfree {str(expiry_filter)},{str(raw_day_filter)},{str(mk_arb_free)}',expiry_filter,raw_day_filter,mk_arb_free)}"+\
           f"/{ticker}/{fmt_date(d)}"

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
            "0.3",
            dependencies=[enrich_day, MkArbFree],
            uid=_uid,
            exclude_args=['verbose','data','raise_if_bad','refresh_cache'])
def arb_free_day( ticker : str, d : dt.date, enriched_data : pd.DataFrame|None = None, *, 
                  raw_day_filter : RawDayFilter = RawDayFilter(), 
                  expiry_filter : ExpiryFilter = ExpiryFilter(), 
                  mk_arb_free : MkArbFree = MkArbFree(),
                  override_cache_mode : CacheMode|None = None,
                  raise_if_bad : bool = True,
                  verbose : Context = Context.quiet ) -> pd.DataFrame:
    """
    """

    data         = enriched_data if enriched_data is not None else enrich_day(ticker, d, raw_day_filter, verbose=verbose, override_cache_mode=override_cache_mode, raise_if_bad = raise_if_bad)

    if data.empty:
        return create_empty_df(SCHEMA_ENRICHED_DAY)
    verify_schema( data , SCHEMA_ENRICHED_DAY, lambda : f"{ticker} {d}")

    per_expiry    = PrettyObject({k: [] for k in sorted( set(SCHEMA_ENRICHED_DAY)-ENRICHED_PER_DAY_FIELDS) })

    expiry_ids    = data["expiry_id"].to_numpy()
    unq_ids       = np.unique(expiry_ids)
    expiry_fields = sorted( ENRICHED_PER_EXPIRY_FIELDS )

    for uid in unq_ids:
        mask = expiry_ids == uid
        for c in per_expiry:
            assert not c in ENRICHED_PER_DAY_FIELDS
            v = data[c][mask].values
            if c in expiry_fields:
                assert v.all( v[0] == v ), f"Internal error: {ticker} {d}: enriched DataFrame column '{c}' has different values for the same expiry."
                per_expiry[c].append(v[0])
            else:
                per_expiry[c].append(v)

    T = len(unq_ids)

    bystrike_fields = sorted( set(SCHEMA_ENRICHED_DAY)-ENRICHED_PER_DAY_FIELDS-ENRICHED_PER_EXPIRY_FIELDS )
    for t in range(T):
        # sort all data by strike
        pure_strike = per_expiry.pure_strike[t]
        ixs         = np.argsort( pure_strike )
        for c in bystrike_fields:
            per_expiry[c][t] = per_expiry[c][t][ixs]
    del t

    af = mk_arb_free.fit( expiries=np.asarray( per_expiry.tty ),
                          strikes=per_expiry.pure_strike,
                          bids=per_expiry.pure_bid_price,
                          asks=per_expiry.pure_ask_price,
                          handle_arb="warn",
                          verbose=verbose )
    
    af_results = PrettyObject({k : [] for k in sorted(set(SCHEMA_ARB_FREE)-ENRICHED_PER_DAY_FIELDS) })

    cnt_all_strikes = 0

    for aft in range(len(af.expiries)):
        srct      = aft.input_index[aft]
        expiry    = af_results['expiries'][aft]
        tte_y     = per_expiry.tte_y[srct]
        assert np.isclose(expiry, tte_y, rtol=0.0, atol=1e-12), f"Internal error: {ticker} {d}: expiry mismatch between input and fitted surface. Please contact the authors with an example."
        del tte_y

        # expiry only fields
        for c in expiry_fields:
            af_results[c].append(per_expiry[c][srct])

        # copy by field
        sixs      = aft.sixs[aft]
        for c in bystrike_fields:
            src = per_expiry[c][srct]
            src = src if sixs is None else src[sixs]
            af_results[c].append(src)

        # copy new vales
        pure_strike = af_results['strike'][aft]
        pure_fit1   = af_results['fit1'][aft]
        pure_fitX   = af_results['fitX'][aft]

        sqrtT_t     = math.sqrt(expiry)
        fit1_vols   = bs.implied( pure_strike, pure_fit1, sqrtT=sqrtT_t, eps=_IMPLIED_VOL_CALC_EPS )
        fitX_vols   = bs.implied( pure_strike, pure_fitX, sqrtT=sqrtT_t, eps=_IMPLIED_VOL_CALC_EPS )

        fit1_greeks = bs( k=pure_strike, vol=fit1_vols, sqrtT = sqrtT_t, eps=_GREEK_CALC_EPS, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )
        fitX_greeks = bs( k=pure_strike, vol=fitX_vols, sqrtT = sqrtT_t, eps=_GREEK_CALC_EPS, what=bs.PRICE|bs.DELTA|bs.GAMMA|bs.VEGA|bs.THETA|bs.DK )

        af_results["fit1_vol"].append(fit1_vols)
        af_results["fitX_vol"].append(fitX_vols)

        def copy(which, src):
            for k,v in src.items():
                af_results[f"pure_{which}_{k}"].append(v)
        copy("fit1", fit1_greeks)
        copy("fitX", fitX_greeks)

        cnt_all_strikes += pure_strike.shape[0]





