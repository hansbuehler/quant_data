"""
Arbitrage-Free Market Data Filter
=================================

The function :func:`mk_arb_free_day` in this module takes 
attempts to fit an arbitrage-free fit through the bid/ask prices of the options in a given day.
It removes strikes and even expiries whenever necessary to achieve this goal.

Massaging data this way is a pre-requisite for any attempts to fit a strictly arbitrage-free model
to WIDB data.

Example
-------
    >>> import datetime as dt
    >>> from quantdata.widb.arbfree import mk_arb_free_day
    >>> from quantdata.utils.arbfree import MkArbFree
    >>>
    >>> # Generate arbitrage-free prices for SPX on a specific date
    >>> df = mk_arb_free_day("SPX", dt.date(2023, 1, 15))
    >>>
    >>> # With custom arbitrage-free fitting configuration
    >>> mk_arb_free_config = MkArbFree(no_arb_mode="surface")
    >>> df = mk_arb_free_day("SPX", dt.date(2023, 1, 15), mk_arb_free=mk_arb_free_config)
    >>> print(df[["pure_strike", "mask_arb_free", "pure_fit1_price", "pure_fit1_delta"]])

This file's ``main`` can also be used to generate data manually:

```bash

    python -m arbfree SPX RUT NDX --period 2020-05-01 2022-01-31
```

"""
from __future__ import annotations

import datetime as dt
import math
import warnings

from cdxcore.pretty import PrettyObject
import numpy as np
import pandas as pd

from quantdata import env
from quantdata.sanos.arbfree import MkArbFree
from quantdata.widb.options_filter import OptionsFilter, filter_options_day
from quantdata.utils.pd import create_empty_df, verify_schema
from quantdata.widb.day import (
    ENRICHED_PER_DAY_FIELDS,
    ENRICHED_PER_EXPIRY_FIELDS,
    ExpiryFilter,
    RawDayFilter,
    SCHEMA_ENRICHED_DAY,
    _GREEK_CALC_EPS,
    _IMPLIED_VOL_CALC_EPS,
    _processing,
    enrich_day,
)

from cdxcore.bs import bs
from cdxcore.subdir import CacheMode, SubDir
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.util import fmt_date, fmt_list, verify_inp
from cdxcore.verbose import Context

_sanos_data_processing = env.locations("sanos").processing

@_sanos_data_processing(fmt=SubDir.PICKLE).cache(
    "0.1",
    dependencies=[filter_options_day],
    uid=lambda ticker, d, raw_day_filter, expiry_filter, sanos, options_filter :\
        f"fit_sanos_day/{named_unique_filename48_8(f'{str(expiry_filter)},{str(raw_day_filter)},{str(options_filter)}', expiry_filter, raw_day_filter, options_filter)}/"+\
        f"{named_unique_filename48_8(str(sanos), sanos)}/"+\
        f"/{ticker}/{d}"
    )
def fit_sanos_day(
    ticker: str,
    d: dt.date,
    sanos: MkArbFree,
    *,
    raw_day_filter: RawDayFilter = RawDayFilter(),
    expiry_filter: ExpiryFilter = ExpiryFilter(),
    options_filter : OptionsFilter = OptionsFilter(),
    raise_if_bad_day: bool = True,
    verbose: Context = Context.all ) -> pd.DataFrame:
    """
    Fit SANOS to a given day
    """

    # get filtered data.
    # Do run linear fit
    ddata = filter_options_day( ticker, d, raw_day_filter=raw_day_filter, mk_arb_free=MkArbFree(None), expiry_filter=expiry_filter, options_filter=options_filter, raise_if_bad_day=raise_if_bad_day, verbose=verbose )
    if ddata is None or ddata.empty:
        return None

    # fit SANOS
    is_valid = ddata['is_valid'].to_numpy().astype(bool,copy=False)
    strikes  = ddata['pure_strike'].to_numpy()[is_valid]
    asks     = ddata['pure_ask_price'].to_numpy()[is_valid]
    bids     = ddata['pure_bid_price'].to_numpy()[is_valid]
    expiries = ddata['tty_d'].to_numpy()[is_valid]

    market = PrettyObject()
    market.strikes = []
    market.asks = []
    market.bids = []
    market.expiries = []

    utty = np.unique(market.expiries)
    for u in utty:
        mask = market.expiries == u
        if np.sum(mask) == 0:
            continue
        market.expiries.append(u)
        market.strikes.append(strikes[mask])
        market.asks.append(asks[mask])  
        market.bids.append(bids[mask])

    r = PrettyObject()  
    tme            = verbose.timer()
    r.sanos        = sanos( strikes=market.strikes, expiries=market.expiries, bids=market.bids, asks=market.asks, message_context=lambda : f"{ticker} {d}", verbose=verbose )
    r.market_error = r.sanos.compute_market_fit( *market, and_vols = False )
    r.focus_error  = r.sanos.compute_market_fit( expiries=r.sanos.expiries,
                                                 strikes=[ market.strikes[t] for t in r.sanos.input_index], 
                                                 bids=[ market.bids[t] for t in r.sanos.input_index],
                                                 asks=[ market.asks[t] for t in r.sanos.input_index],
                                                 masks=r.sanos.masks, and_vols=False )

    verbose.write(lambda : f"Fitted {'linear' if sanos.smoothness is None else f'{sanos.smoothness:.2f}-smooth'} SANOS to {ticker} {d}. This took {tme}.")
    verbose.report(1,lambda : f"Market fit: {r.market_error.pcnt_fit*100:.1f}% were fitted within bid/ask. The mean and median error for those options not fitted were {r.market_error.mean_err:.2f} and {r.market_error.median_err:.2f}, respectively.")
    verbose.report(1,lambda : f"Target fit: {r.market_error.pcnt_fit*100:.1f}% of options actually calibrated to were fitted within bid/ask. The mean and median error for those options not fitted were {r.market_error.mean_err:.2f} and {r.market_error.median_err:.2f}, respectively.")

    return r



