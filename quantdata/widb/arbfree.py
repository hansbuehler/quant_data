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

import numpy as np
import pandas as pd

from quantdata.utils.arbfree import MkArbFree
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

SCHEMA_ARB_FREE_DAY_EXPIRY = SCHEMA_ENRICHED_DAY | {
    "mask_arb_free": np.dtype("bool"),
    "fit1_vol": np.dtype("float32"),
    "pure_fit1_price": np.dtype("float32"),
    "pure_fit1_delta": np.dtype("float32"),
    "pure_fit1_gamma": np.dtype("float32"),
    "pure_fit1_vega": np.dtype("float32"),
    "pure_fit1_theta": np.dtype("float32"),
    "pure_fit1_dk": np.dtype("float32"),
}
""" Fields and their types when arbitrage mode is ``"expiry"``. """

SCHEMA_ARB_FREE_DAY_SURFACE = SCHEMA_ARB_FREE_DAY_EXPIRY | {
    "fitX_vol": np.dtype("float32"),
    "pure_fitX_price": np.dtype("float32"),
    "pure_fitX_delta": np.dtype("float32"),
    "pure_fitX_gamma": np.dtype("float32"),
    "pure_fitX_vega": np.dtype("float32"),
    "pure_fitX_theta": np.dtype("float32"),
    "pure_fitX_dk": np.dtype("float32"),
}
""" Fields and their types when arbitrage mode is ``"surface"``. """

def _uid(ticker, d, raw_day_filter, expiry_filter, mk_arb_free) -> str:
    return (
        f"arbfree/{named_unique_filename48_8(f'{str(expiry_filter)},{str(raw_day_filter)},{str(mk_arb_free)}', expiry_filter, raw_day_filter, mk_arb_free)}"
        f"/{ticker}/{fmt_date(d)}"
    )

@_processing(fmt=SubDir.PANDAS_PARQUET).cache(
    "0.6",
    dependencies=[enrich_day, MkArbFree],
    uid=_uid,
    exclude_args=["verbose", "enriched_data", "raise_if_bad_day", "refresh_cache"],
)
def mk_arb_free_day(
    ticker: str,
    d: dt.date,
    enriched_data: pd.DataFrame | None = None,
    *,
    raw_day_filter: RawDayFilter = RawDayFilter(),
    expiry_filter: ExpiryFilter = ExpiryFilter(),
    mk_arb_free: MkArbFree|None = MkArbFree(),
    override_cache_mode: CacheMode | None = None,
    raise_if_bad_day: bool = True,
    return_arb_free_densities : bool = False,
    warn_if_arb : bool = True,
    verbose: Context = Context.all,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Generate an arbitrage-free fit for a single enriched WIDB day.

    The returned DataFrame keeps the full input-strike rows for retained expiries
    and adds mask/fitted-price/Greek columns aligned to those input strikes.
    Expiries and strikes dropped by MkArbFree are removed from the returned frame.
    The underlying mechanism is described in detail in :class:`MkArbFree`.

    The schema of the data frame returned depend on the settings in ``mk_arb_free.no_arb_mode``.
    
    * ``expriy``: the dataframe has schema :data:`SCHEMA_ARB_FREE_DAY_EXPIRY`. which includes all columns :data:`SCHEMA_ENRICHED_DAY` plus the following additional columns:

      - ``mask_arb_free``: boolean mask indicating which strikes were retained as arbitrage-free (True) vs dropped (False).
      - ``pure_fit1_price``: fitted arbitrage-free price within bid/ask where ``mask_arb_free`` is True.
      - ``pure_fit1_delta``, ``pure_fit1_gamma``, ``pure_fit1_vega``, ``pure_fit1_theta``, ``pure_fit1_dk``: Greeks corresponding to ``pure_fit1_price``.
    
    * ``surface``: the dataframe has schema :data:`SCHEMA_ARB_FREE_DAY_SURFACE`, which includes all columns from ``SCHEMA_ARB_FREE_DAY_EXPIRY`` plus:
      
      - ``pure_fitX_price``: surface arbitrage-free closest price to ``fit1`` prices. Up to machine precision, existence of a ``fitX`` within bid/ask is equivalent to the market being free of arbitrage.
      - ``pure_fitX_delta``, ``pure_fitX_gamma``, ``pure_fitX_vega``, ``pure_fitX_theta``, ``pure_fitX_dk``: Greeks corresponding to ``pure_fitX_price``.
    
    Returns
    -------
    Result: :class:`pandas.DataFrame`|None or (:class:`pandas.DataFrame`|None, dict|None) if ``return_arb_free`` is True.`
        This function may return None if ``raise_if_bad_day`` is False and the day is deemed bad as defined by :data:`bad.BAD.DAYS_ALL`.
    """
    verify_inp( mk_arb_free.no_arb_mode in ['expiry', 'surface'], lambda: f"{ticker} {d}: arb_free_day requires mk_arb_free.no_arb_mode='surface' for full fit outputs; found no_arb_mode '{mk_arb_free.no_arb_mode}'." )
    mode_surface = mk_arb_free.no_arb_mode == 'surface'
    expected_schema = SCHEMA_ARB_FREE_DAY_SURFACE if mode_surface else SCHEMA_ARB_FREE_DAY_EXPIRY

    if not enriched_data is None:
        verify_inp( isinstance(enriched_data, pd.DataFrame), f"{ticker} {d}: expected 'enriched_data' to be a pandas DataFrame or None, got {type(enriched_data)}.")
        data = enriched_data
    else:
        data = enrich_day(
                    ticker,
                    d,
                    raw_day_filter=raw_day_filter,
                    expiry_filter=expiry_filter,
                    override_cache_mode=override_cache_mode,
                    raise_if_bad_day=raise_if_bad_day,
                    verbose=verbose,
                )
        if data is None:
            assert raise_if_bad_day, "Internal error: empty data should only be returned when 'raise_if_bad_day' is False. Please check the enriched day data for details."
            if not return_arb_free_densities:
                return None
            return None, None
        del enriched_data

    if mk_arb_free is None:
        if not return_arb_free_densities:
            return data
        return data, None

    if data.empty:
        if not return_arb_free_densities:
            return create_empty_df(expected_schema)
        return create_empty_df(expected_schema), None
    verify_schema(data, SCHEMA_ENRICHED_DAY, lambda: f"{ticker} {d}")

    tme = verbose.timer()
    verbose.write(lambda: f"{ticker} {d}: Generating arb-free price surface.")

    expiry_ids        = data["expiry_id"].to_numpy()
    unique_expiry_ids = np.unique(expiry_ids)
    grouped_frames: list[pd.DataFrame] = []
    expiries_d : list[dt.date] = []
    is_ams : list[bool] = []
    expiries: list[float] = []
    strikes: list[np.ndarray] = []
    bids: list[np.ndarray] = []
    asks: list[np.ndarray] = []

    assert data[data['pure_ask_price']<=data['pure_bid_price']].empty, f"{ticker} {d}: found {len(data[data['pure_ask_price']<=data['pure_bid_price']])} options with ask price below or equal to bid price. These options should have been filtered out by the expiry filter. Please check the enriched day data for details."

    for expiry_id in unique_expiry_ids:
        group = data.loc[expiry_ids == expiry_id].sort_values("pure_strike", kind="stable").reset_index(drop=True)
        grouped_frames.append(group)
        expiries.append(float(group["tte_y"].iloc[0]))
        expiries_d.append(group["expiry"].iloc[0])
        is_ams.append(group["is_am"].iloc[0])
        strikes.append(group["pure_strike"].to_numpy(dtype=np.float64))
        bids.append(group["pure_bid_price"].to_numpy(dtype=np.float64))
        asks.append(group["pure_ask_price"].to_numpy(dtype=np.float64))

    assert np.min( np.diff(unique_expiry_ids) ) > 0, f"{ticker} {d}: expiry ids are expected to be strictly increasing. Found non-increasing expiry ids: {fmt_list(unique_expiry_ids)}."
    if np.min( np.diff(expiries) ) <= 0:
        text = ""
        for xid, expiry, date, is_am, ks in zip(unique_expiry_ids, expiries, expiries_d, is_ams, strikes):
            text += f"Expiry ID {xid} has expiry {expiry} on date {date} (AM: {is_am}) with {len(ks)} strikes\n"
        warnings.warn("*** Error:\n"+text)
        assert np.min( np.diff(expiries) ) > 0, f"{ticker} {d}: expiries are expected to be strictly increasing."

    arb_result = mk_arb_free(
        expiries=np.asarray(expiries, dtype=np.float64),
        strikes=strikes,
        bids=bids,
        asks=asks,
        handle_arb="warn" if warn_if_arb else "ignore",
        message_context=lambda: f"{ticker} {d}",
        verbose=verbose(1),
    )

    if len(arb_result.expiries) == 0:
        verbose.write(lambda: f"{ticker} {d}: MkArbFree dropped all expiries. This took {tme}.")
        if not return_arb_free_densities:
            return create_empty_df(expected_schema)
        return create_empty_df(expected_schema), None

    fit1_all = arb_result.fits1
    assert not fit1_all is None, f"Internal error: {ticker} {d}: MkArbFree returned None for fits; expected arrays of shape (num_expiries, num_strikes_per_expiry)."

    if mode_surface:
        fitx_all = arb_result.fitsX
        assert not fitx_all is None, f"Internal error: {ticker} {d}: MkArbFree returned None for fitsX; expected arrays of shape (num_expiries, num_strikes_per_expiry)."
    else:
        fitx_all = None

    result_frames: list[pd.DataFrame] = []
    dropped_strikes = 0
    output_indices = list(arb_result.input_index) if arb_result.input_index is not None else slice(None)

    for output_index, source_index in enumerate(output_indices):
        group = grouped_frames[source_index].copy()
        pure_strike = group["pure_strike"].to_numpy(dtype=np.float64)
        mask_t = np.asarray(arb_result.masks[output_index], dtype=bool)
        fit1_t = np.asarray(fit1_all[output_index], dtype=np.float64)
        fitx_t = np.asarray(fitx_all[output_index], dtype=np.float64) if mode_surface else None

        assert mask_t.shape == pure_strike.shape,  f"Internal error: {ticker} {d}: arb mask shape {mask_t.shape} does not match strike shape {pure_strike.shape}."
        assert fit1_t.shape == pure_strike.shape, f"Internal error: {ticker} {d}: fit1 shape {fit1_t.shape} does not match strike shape {pure_strike.shape}."
        if mode_surface:
            assert fitx_t.shape == pure_strike.shape, f"Internal error: {ticker} {d}: fitX shape {fitx_t.shape} does not match strike shape {pure_strike.shape}."

        sqrt_t    = math.sqrt(float(arb_result.expiries[output_index]))
        fit1_vols = bs.implied(pure_strike, fit1_t, sqrtT=sqrt_t, eps=_IMPLIED_VOL_CALC_EPS)
        fitx_vols = bs.implied(pure_strike, fitx_t, sqrtT=sqrt_t, eps=_IMPLIED_VOL_CALC_EPS) if mode_surface else None
        fit1_greeks = bs(
            k=pure_strike,
            vol=fit1_vols,
            sqrtT=sqrt_t,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        fitx_greeks = bs(
            k=pure_strike,
            vol=fitx_vols,
            sqrtT=sqrt_t,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        ) if mode_surface else None

        group["mask_arb_free"] = mask_t
        group["fit1_vol"] = fit1_vols.astype(np.float32)
        group["pure_fit1_price"] = fit1_t.astype(np.float32)
        for greek_name, values in fit1_greeks.items():
            group[f"pure_fit1_{greek_name}"] = np.asarray(values, dtype=np.float32)

        if mode_surface:
            group["fitX_vol"] = fitx_vols.astype(np.float32)
            group["pure_fitX_price"] = fitx_t.astype(np.float32)
            for greek_name, values in fitx_greeks.items():
                group[f"pure_fitX_{greek_name}"] = np.asarray(values, dtype=np.float32)

        dropped_strikes += int((~mask_t).sum())
        result_frames.append(group)

    result = pd.concat(result_frames, axis=0, ignore_index=True)
    result = result.sort_values(["expiry_id", "pure_strike"], kind="stable", ignore_index=True)

    for column, dtype in expected_schema.items():
        if result[column].dtype != dtype:
            result[column] = result[column].astype(dtype, copy=False)

    verify_schema(result, expected_schema, lambda: f"{ticker} {d}")

    for column in ENRICHED_PER_DAY_FIELDS:
        if result[column].nunique() != 1:
            raise RuntimeError(
                f"Internal error: {ticker} {d}: arb-free DataFrame column '{column}' is expected to have exactly one unique value. Found values: {fmt_list(result[column].unique())}."
            )

    for column in ENRICHED_PER_EXPIRY_FIELDS:
        values_per_expiry = result.groupby("expiry_id")[column].nunique()
        if (values_per_expiry != 1).any():
            raise RuntimeError(
                f"Internal error: {ticker} {d}: arb-free DataFrame column '{column}' is expected to have exactly one unique value per expiry. Found expiries with multiple values: {fmt_list(values_per_expiry[values_per_expiry != 1].index.tolist())}."
            )

    dropped_expiries = len(unique_expiry_ids) - len(output_indices)
    verbose.write( lambda: f"{ticker} {d}: generated arbitrage-free price surface. Dropped {dropped_expiries} expiries and {dropped_strikes} strikes of the remaining expiries. This took {tme}.")
    result.attrs['arb_free_timing'] = tme.seconds
    return result

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
@click.option('--arb-warnings-off', is_flag=True, help="Turn off warnings about detected arbitrage opportunities in the data.")
@click.option('--proceed', is_flag=True, help="Proceed upon encountering bad data.")
@click.option(
    "--no-arb-mode",
    "arb_mode",
    type=click.Choice(["surface", "expiry"], case_sensitive=False),
    default="surface",
    show_default=True,
)
def main(tickers : list[str],
        period : list[dt.datetime], 
        halfdays : bool, 
        workers : int, 
        threading : bool, 
        refresh : bool,
        arb_warnings_off : bool,
        proceed : bool,
        arb_mode : str
        ):

    verify_inp( len(tickers) > 0, "At least one ticker must be specified")
    verify_inp( period is not None and len(period) == 2, "A start and end date must be specified")

    from quantdata.widb.timeseries import build_time_series, OptionsFilter
    from quantdata.biz_calendar import DateRange, to_date
    from quantdata.pool import PoolConfig
    from cdxcore.jcpool import JCPool

    start, end         = period
    date_range         = DateRange(start=to_date(start), end=to_date(end), exclude_half_days=not halfdays)
    pool_config        = PoolConfig(num_workers=workers, threading=threading)
    raw_day_filter     = RawDayFilter()   
    expiry_filter      = ExpiryFilter()
    mk_arb_free        = MkArbFree(no_arb_mode=arb_mode)

    dates              = date_range.get_business_days()

    verbose = Context.all
    verbose.write(lambda: f"Computing Arb-Free fits for tickers {fmt_list(tickers)} from {fmt_date(start)} to {fmt_date(end)} {'excluding' if date_range.exclude_half_days else 'including'} half days. Total business days: {len(dates)}.")
    verbose.report(1,lambda: f"Arb-free config: {mk_arb_free}")
    verbose.report(1,lambda: f"Raw Day Filter: {raw_day_filter}")
    verbose.report(1,lambda: f"Expiry filter: {expiry_filter}")
    verbose.report(1,lambda: f"Pool config: {pool_config}")
    
    pool = JCPool(num_workers=pool_config.num_workers, threading=pool_config.threading)

    for it, ticker in enumerate(tickers):  
        def processing( *, ticker, d, mk_arb_free, raw_day_filter, expiry_filter, refresh ):
            mk_arb_free_day( ticker=ticker, d=d, mk_arb_free=mk_arb_free, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, override_cache_mode="update" if refresh else "on", raise_if_bad_day=False, verbose=Context.quiet, cache_generate_only=True )
            return 0

        with verbose.write_t(lambda: f"\rProcessing ticker {it+1}/{len(tickers)} {ticker} from {fmt_date(date_range.start)} to {fmt_date(date_range.end)}:", end='') as tme:
            cnt = 0
            for d, _ in pool.parallel( { d: pool.delayed(processing)(ticker=ticker, d=d, mk_arb_free=mk_arb_free, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, refresh=refresh) for d in dates } ):
                cnt += 1
                if tme.interval_test(1.):
                    verbose.write(f"\rProcessing ticker {it+1}/{len(tickers)} '{ticker}' date {cnt}/{len(dates)} {fmt_date(d)}. So far this took {tme}...", end='')
            verbose.write(f"\rProcessed ticker {it+1}/{len(tickers)} '{ticker}' for all {len(dates)} dates. This took {tme}.")

    verbose.write(f"All {len(tickers)} tickers processed.")

if __name__ == "__main__":
    main()
