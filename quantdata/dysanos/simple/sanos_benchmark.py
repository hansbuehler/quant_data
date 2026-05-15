"""
Dysanos Non-ML Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^

"""

import math

from cdxcore import bs

from quantdata.sanos.open import OpenSanos
from quantdata.sanos.grid import GridGeometryGenerator, GridSanos
from quantdata.sanos.fitter import Fitter
from quantdata.dysanos.simple.sampler import SimpleSampler, SimpleConfig, SamplingConfig, ExpiryFilter, RawDayFilter, SampleDaysConfig, DateRange, PoolConfig, sample_simple_days
from datetime import date


from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, CacheMode, version
from cdxcore.pretty import PrettyObject, verify_inp
from cdxcore.jcpool import JCPool
from dataclasses import dataclass

from quantdata.sanos.open import OpenGeometryGenerator
from quantdata.sanos.utils import Eps, extract_atm_fwd_vols
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
import numpy as np

@version("0.1")
@dataclass(frozen=True)
class SanosBenchmarkConfig:
    label          : str   #: Unique label
    open_generator : OpenGeometryGenerator = OpenGeometryGenerator()  # how to generate open geometries
    grid_generator : GridGeometryGenerator = GridGeometryGenerator()  # how to generate grid geometries
    fwdvol_frac    : float = 0.25
    eps            : Eps = Eps(1E-3)
    debug          : bool = False
    no_warnings    : bool = False
    
FIELDS_SANOS_BENCHMARK = {

}

@version("0.1", dependencies=[SanosBenchmarkConfig, sample_simple_days])
def sanos_benchmark_day( 
                 ticker : str,
                 d: date,
                 isample : int,
                 data : PrettyObject,
                 benchmark_config : SanosBenchmarkConfig,
                 verbose : Context = Context.all ) -> dict[str, np.ndarray]:        
    """
    For a given day perform:

    1) Fit "open" SANOS to the prevaling market data and strikes. 
    2) Fit "grid" SANOS to the same market data and strikes, but with the grid geometry. This is the benchmark to beat for any ML SANOS fit based on the same grid geometry.
    """

    with verbose.write_t(lambda : f"Processing {ticker} date {d} ({isample+1}/{len(data.dates)})") as tme:
        verbose1 = verbose(1)

        # market data & geometry
        # ----------------------

        market               = PrettyObject()
        is_valid             = data.is_valid[isample,:]
        arb_free_mask        = data.arb_free_mask[isample,is_valid]
        arb_free_mask        = arb_free_mask if np.any(~arb_free_mask) else None    
        M                    = len(is_valid)
        Mx                   = sum(is_valid)
        d                    = data.dates[isample]
        market.strikes       = data.pure_strike[isample,is_valid]
        market.tte_y         = data.tte_y[isample,is_valid]
        market.bid_vol       = data.bid_vol[isample,is_valid]
        market.ask_vol       = data.ask_vol[isample,is_valid]
        market.bid           = data.bid_pure_price[isample,is_valid]
        market.ask           = data.ask_pure_price[isample,is_valid]

        market.target_strikes = market.strikes[arb_free_mask] if arb_free_mask is not None else market.strikes
        market.target_bid     = market.bid[arb_free_mask] if arb_free_mask is not None else market.bid
        market.target_ask     = market.ask[arb_free_mask] if arb_free_mask is not None else market.ask
        market.target_tte_y   = market.tte_y[arb_free_mask] if arb_free_mask is not None else market.tte_y

        # limit maximum expiry
        max_expiry      = benchmark_config.grid_generator.horizon_t
        verify_inp( max_expiry > market.tte_y[0], lambda: f"{ticker} {d}: Maximum expiry of the grid geometry must be greater than the first market expiry, found max_expiry {max_expiry:.4g} and first market expiry {market.tte_y[0]:.4g}" )
        if max_expiry < market.tte_y[-1]:
            is_valid_expiry = market.tte_y <= max_expiry
            n_drop          = np.sum(~is_valid_expiry)
            verbose1.write(lambda : f"Limiting market data to a maximum expiry of {max_expiry*DAYS_PER_YEAR:.4f}d years. Dropping {n_drop}/{len(market.tte_y)} options.")
            for k,v in market.items():
                market[k] = v[is_valid_expiry]

        # open first
        # ----------
        # the output are benchmark fits: this is the benchmark to beat for anything (arb-free, TBF)

        def open_sanos():
            result        = PrettyObject()
            open_geometry = benchmark_config.open_generator.create(
                                expiries=market.tte_y,
                                pure_strikes=market.strikes,
                                dtype=np.float64)
            open_fwd_vols = extract_atm_fwd_vols( market.strikes, market.tte_y, market.ask_vol, fwdvol_min=open_geometry.fwdvol_min )
            open_fitter   = Fitter( geometry = open_geometry,
                                    eps = benchmark_config.eps, 
                                    debug=benchmark_config.debug, 
                                    max_error = None, 
                                    mid_fit_weight = 0.0001, 
                                    no_warnings=benchmark_config.no_warnings )

            arb_free_mask = market.arb_free_mask if np.any(~market.arb_free_mask) else None

            open_dlvsqs,\
            result.open_target_fit,\
            open_densities = \
                open_fitter(fwd_vols     = open_fwd_vols*benchmark_config.fwdvol_frac,
                            opt_strikes  = market.target_strikes,
                            opt_bid      = market.target_bid,
                            opt_ask      = market.target_ask,
                            opt_tte      = market.target_tte_y,
                            opt_is_calls = None,
                            opt_mask     = None,
                            return_only_dlvsqs = False,
                            verbose      = Context.all ) 
            result.open_target_fit_vols = bs.implied(market.target_strikes, result.open_target_fit, sqrtT = np.sqrt(market.target_tte_y) )

            if arb_free_mask is None:
                result.open_fit = result.open_target_fit
                result.open_fit_vols = result.open_target_fit_vols
            else:
                open_sanos = OpenSanos( geometry = open_geometry, eps=benchmark_config.eps, no_warnings=benchmark_config.no_warnings )
                result.open_fit = open_sanos.price(
                                        fwd_vols    = open_fwd_vols,
                                        densities   = open_densities,
                                        #
                                        opt_strikes = market.strikes,
                                        opt_tte     = market.tte_y,
                                        opt_mask    = None,
                                        opt_is_call = None
                                    )
                result.open_fit_vols = bs.implied(market.strikes, result.open_fit, sqrtT = np.sqrt(market.tte_y) )
            del open_dlvsqs, open_densities
            return result

        result = open_sanos()

        # grid next
        # ---------
        # output is the lower bound for any ML SANOS fit based on the same grid geometry
        # we also compute the required interpolation data here.

        def grid_sanos():
            result             = PrettyObject()
            grid_geometry      = benchmark_config.grid_generator.create(return_inner=False, dtype=np.float64) 
            grid_fwd_vols      = extract_atm_fwd_vols( market.strikes, market.tte_y, market.ask_vol, fwdvol_min=grid_geometry.fwdvol_min )
            grid_fitter        = Fitter( geometry = grid_geometry,
                                        eps = benchmark_config.eps, 
                                        debug=benchmark_config.debug, 
                                        max_error = None, 
                                        mid_fit_weight = 0.0001, 
                                        no_warnings=benchmark_config.no_warnings )

            grid_dlvsqs,\
            result.grid_target_fit,\
            grid_densities = \
                grid_fitter(fwd_vols     = grid_fwd_vols*benchmark_config.fwdvol_frac,
                            opt_strikes  = market.target_strikes,
                            opt_bid      = market.target_bid,
                            opt_ask      = market.target_ask,
                            opt_tte      = market.target_tte_y,
                            opt_is_calls = None,
                            opt_mask     = None,
                            return_only_dlvsqs = False,
                            verbose      = Context.all ) 
            result.grid_target_fit_vols = bs.implied(market.target_strikes, result.grid_target_fit, sqrtT = np.sqrt(market.target_tte_y) )
            del grid_dlvsqs, grid_densities

            if arb_free_mask is None:
                result.grid_fit = result.grid_target_fit
                result.grid_fit_vols = result.grid_target_fit_vols
            else:
                grid_sanos = GridSanos( geometry = grid_geometry, eps=benchmark_config.eps, no_warnings=benchmark_config.no_warnings )
                result.grid_fit = grid_sanos.price(
                                        fwd_vols    = grid_fwd_vols,
                                        densities   = grid_densities,
                                        #
                                        opt_strikes = market.strikes,
                                        opt_tte     = market.tte_y,
                                        opt_mask    = None,
                                        opt_is_call = None
                                    )
                result.grid_fit_vols = bs.implied(market.strikes, result.grid_fit, sqrtT = np.sqrt(market.tte_y) )
                return result

        result.update( grid_sanos() )
        result.is_valid = is_valid
        verbose.write(lambda : f"done; this took {tme}.", head=False)

    return result

def create_sanos_benchmark( ticker : str,
                            data : dict[str, np.ndarray],
                            benchmark_config : SanosBenchmarkConfig,
                            simple_config : SimpleConfig, *,
                            pool_config : PoolConfig = PoolConfig.off(),
                            override_cache_mode : CacheMode|None = None,
                            verbose : Context = Context.all ):
    """
    Create SANOS Fitting Benchmark
    """
    assert set(data) == simple_config.config.features()|{'dates'}, f"Internal error: data mismatch: expected features {sorted(simple_config.config.features())} and 'dates', found {sorted(set(data))}"
    num_samples = len(data['dates'])

    with verbose.write_t(lambda : f"Creating SANOS Benchmark for {ticker} with label {benchmark_config.label}: loading data.") as tme:
    
        pool = JCPool(num_workers=pool_config.workers, threading=pool_config.threading)

        def process( *, ticker, d, isample, sample_data, benchmark_config, override_cache_mode ):
            r = process_day( ticker = ticker,
                                d = sample_data["date"],
                                sample_data = sample_data,
                                benchmark_config = benchmark_config,
                                override_cache_mode = override_cache_mode,
                                verbose = Context.quiet )
            return ticker, d, isample, r
        def mk_sample_data( isample ):
            r = dict()
            for k, vi in data.items():
                if k!="dates":
                    r[k] = vi[isample,:]
            return r
        
        results = [None]*num_samples
        for ticker, d, isample, sample_ret in pool.parallel( pool.delayed( process )( ticker=ticker,
                                                                                      d=d,
                                                                                      isample=isample,
                                                                                      sample_data=mk_sample_data(isample),
                                                                                      benchmark_config=benchmark_config, 
                                                                                      override_cache_mode=override_cache_mode )
                                                                                      for isample, d in enumerate(data['dates']) ):
            verbose.report(1,lambda : f"\Generated SANOS benchmark for {ticker} {d} ({isample+1}/{num_samples}) done {math.ceil((isample+1)/num_samples*100)}%. So far this took {tme}... ", end="")
            results[isample] = sample_ret
        assert np.all( [r is not None for r in results] ), "Internal error: some samples were not processed successfully, this should never happen. Please report this to the developers."

        verbose.write(f"\rGenerated SANOS benchmark for {num_samples} samples for ticker {ticker}. This took {tme}.", end="")


    



    data = PrettyObject(
            sample_simple_days( 
                ticker = ticker,
                simple_config = simple_config,
                pool_config = pool_config,
                override_cache_mode = override_cache_mode,
                verbose = verbose(1)
            )
        )
        
    S = len(data.dates)

    for t, d in enumerate(data.dates):
        with verbose.write(lambda : f"\nProcessing {ticker} date {d} ({t+1}/{S})... ", end=""):
        
            

            d.open_geometry = benchmark_config.open_generator.create_geometry(d)

    verbose.write(lambda : f"Creating SANOS Benchmark for {ticker} with label {benchmark_config.label}: loading data."

