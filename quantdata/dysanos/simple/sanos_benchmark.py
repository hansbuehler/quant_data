"""
Fit SANOS to the market using linear programming.
"""






from cdxcore import bs

from quantdata.sanos.open import OpenSanos
from quantdata.sanos.grid import GridGeometryGenerator, GridSanos
from quantdata.sanos.fitter import Fitter
from quantdata.dysanos.simple.sampler import SimpleSampler, SimpleConfig, SamplingConfig, ExpiryFilter, RawDayFilter, SampleDaysConfig, DateRange, PoolConfig, sample_simple_days
from datetime import date


from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, CacheMode, version
from cdxcore.pretty import PrettyObject, verify_inp
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

    open_grid_nms  : tuple[float,float] = (-3., 3.)  #: Range of NMS strikes for the open geometry grid generation. This is a tuple of (lower_scale, upper_scale) where the lowest strike is scaled by lower_scale and the highest strike is scaled by upper_scale. The scaling is applied to the normalized strikes generated from the NMS formula.

    fwdvol_frac    : float = 0.25
    eps            : Eps = Eps(1E-3)
    debug          : bool = False
    no_warnings    : bool = False
    
@version("0.1", dependencies=[SanosBenchmarkConfig, sample_simple_days])
def process_day( ticker : str,
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

        market          = PrettyObject()
        is_valid        = data.is_valid[isample,:]
        M               = len(is_valid)
        Mx              = sum(is_valid)
        d               = data.dates[isample]
        market.strikes  = data.pure_strike[isample,is_valid]
        market.tte_y    = data.tte_y[isample,is_valid]
        market.bid_vol  = data.bid_vol[isample,is_valid]
        market.ask_vol  = data.ask_vol[isample,is_valid]
        market.bid      = data.bid_pure_price[isample,is_valid]
        market.ask      = data.ask_pure_price[isample,is_valid]

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

        open_geometry = benchmark_config.open_generator.create(
                            expiries=market.tte_y,
                            pure_strikes=market.strikes,
                            dtype=np.float64)
        open_fwd_vols = extract_atm_fwd_vols( market.strikes, market.tte_y, market.ask_vol, fwdvol_min=open_geometry.fwdvol_min )
        open_fitter   = Fitter( geometry = open_geometry, eps = benchmark_config.eps, debug=benchmark_config.debug, max_error = None, mid_fit_weight = 0.0001, no_warnings=benchmark_config.no_warnings )

        result        = PrettyObject()

        open_dlvsqs,\
        result.open_fit,\
        open_densities = \
            open_fitter(fwd_vols     = open_fwd_vols*benchmark_config.fwdvol_frac,
                        opt_strikes  = market.strikes,
                        opt_bid      = market.bid,
                        opt_ask      = market.ask,
                        opt_tte      = market.tte_y,
                        opt_is_calls = None,
                        opt_mask     = None,
                        return_only_dlvsqs = False,
                        verbose      = Context.all ) 
        result.open_fit_vols = bs.implied(market.strikes, result.open_fit, sqrtT = np.sqrt(market.tte_y) )
        del open_dlvsqs, open_densities

        # grid next
        # ---------
        # output is the lower bound for any ML SANOS fit based on the same grid geometry
        # we also compute the required interpolation data here.

        grid_geometry,\
        grid_inner_strikes = benchmark_config.grid_generator.create(return_inner=True, dtype=np.float64) 
        grid_fwd_vols      = extract_atm_fwd_vols( market.strikes, market.tte_y, market.ask_vol, fwdvol_min=grid_geometry.fwdvol_min )
        grid_fitter        = Fitter( geometry = grid_geometry, eps = benchmark_config.eps, debug=benchmark_config.debug, max_error = None, mid_fit_weight = 0.0001, no_warnings=benchmark_config.no_warnings )

        grid_dlvsqs,\
        result.grid_fit,\
        grid_densities = \
            grid_fitter(fwd_vols     = grid_fwd_vols*benchmark_config.fwdvol_frac,
                        opt_strikes  = market.strikes,
                        opt_bid      = market.bid,
                        opt_ask      = market.ask,
                        opt_tte      = market.tte_y,
                        opt_is_calls = None,
                        opt_mask     = None,
                        return_only_dlvsqs = False,
                        verbose      = Context.all ) 
        result.grid_fit_vols = bs.implied(market.strikes, result.grid_fit, sqrtT = np.sqrt(market.tte_y) )
        del grid_dlvsqs, grid_densities

        intp = grid_geometry.interpolation_weights( opt_tte = market.tte_y )
        assert set(intp.keys()) == {"lix", "dweight", "vweight", "vextra"}, f"Unexpected keys in interpolation weights: {intp.keys()}"
        result.grid_intp_lix     = intp["lix"]
        result.grid_intp_dweight = intp["dweight"]
        result.grid_intp_vweight = intp["vweight"]
        result.grid_intp_vextra  = intp["vextra"]
        del intp

        for k in result:
            v = result[k]
            assert isinstance(v, np.ndarray), f"Expected result {k} to be a numpy array, got type {type(v)}"
            assert v.shape == (Mx,), f"Expected result {k} to have shape {(Mx,)}, got {v.shape}"
            v_ = np.zeros((M,),dtype=v.dtype)
            v_[is_valid] = v
            result[k] = v_

        result.is_valid = is_valid#
        verbose.write(lambda : f"done; this took {tme}.", head=False)

    return result


"""
        # grid to open
        # ------------
        # Price the (interior) grid options with "open" (ie some hopefully smooth interpolation)

        assert grid_inner_strikes.shape[0] == grid_geometry.T, f"Expected grid_inner_strikes to have shape (T, num_inner_strikes), got {grid_inner_strikes.shape}"
        regrid = PrettyObject()
        regrid.expiries  = grid_geometry.expiries
        regrid.strikes   = grid_inner_strikes

        regrid_tte       = np.full( grid_inner_strikes.shape, regrid.expiries[:,None], grid_inner_strikes.dtype )
        regrid_intp      = grid_geometry.interpolation_weights( opt_tte = regrid_tte.reshape( (-1, )) )
        for k,v in regrid_intp.items():
            regrid["intp_"+k] = v.reshape( regrid.strikes.shape )

        regrid.prices    = open_sanos.price( 
                fwd_vols     = open_fwd_vols*benchmark_config.fwdvol_frac,
                densities    = open_densities,
                opt_strikes  = regrid.strikes.reshape( (-1, )),
                opt_tte      = regrid_tte.reshape( (-1, )), # isn't there a nicer numpy idom?
                opt_mask     = None,
                opt_is_call  = None,
                opt_interp   = regrid_intp,
            )
        regrid.prices     = regrid.prices.reshape( regrid.strikes.shape )

        _,\
        result.grid_fit,\
        grid_densities = \
            grid_fitter(fwd_vols     = grid_fwd_vols*benchmark_config.fwdvol_frac,
                        opt_strikes  = market.strikes,
                        opt_bid      = market.bid,
                        opt_ask      = market.ask,
                        opt_tte      = market.tte_y,
                        opt_is_calls = None,
                        opt_mask     = None,
                        return_only_dlvsqs = False,
                        verbose      = Context.all ) 

        grid_fitter        = Fitter( geometry = grid_geometry, eps = benchmark_config.eps, debug=benchmark_config.debug, max_error = None, mid_fit_weight = 0.0001, no_warnings=benchmark_config.no_warnings )



        model_data = PrettyObject()
        model_data.open_fwd_vols = open_fwd_vols
"""


def create_sanos_benchmark( ticker : str,
                            benchmark_config : SanosBenchmarkConfig,
                            simple_config : SimpleConfig, *,
                            pool_config : PoolConfig = PoolConfig.off(),
                            override_cache_mode : CacheMode|None = None,
                            verbose : Context = Context.all ):
    """
    Create SANOS Fitting Benchmark
    """
    verbose.write(lambda : f"Creating SANOS Benchmark for {ticker} with label {benchmark_config.label}: loading data."

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

