"""
Sanos Non-ML Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^
"""
from __future__ import annotations

from quantdata.widb.timeseries import TimeSeriesDefinition, filter_options_day, build_time_series, DateRange, PoolConfig
from quantdata.sanos.arbfree import MkArbFree, _compute_dlvs
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS5
from quantdata.dysanos.simple import _locations_simple

from cdxcore.bs import bs
from cdxcore.uniquehash import named_unique_filename48_8
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, CacheMode, version
from cdxcore.pretty import PrettyObject, verify_inp
from cdxcore.jcpool import JCPool

from datetime import date
from dataclasses import dataclass, field
import numpy as np
import warnings
import scipy as scipy


DEFAULT_EXPIRIES = np.array([1./255., 3./255., 5./255., 10./255., 20./255., 40./255.,0.5,1.], dtype=np.float64)
DEFAULT_NUMS     = (-4,2)
_EPS_PRICE       = 1E-6

_cache = _locations_simple.experiments("sanos_benchmark")

@version("0.1")
@dataclass(frozen=True)
class SanosBenchmarkConfig:
    smoothness     : float = 0.1
    grid_expiries  : np.array = field( default_factory=lambda: DEFAULT_EXPIRIES )
    grid_nms       : tuple = (-4,2)
    grid_num_nms   : int = 40
    vol_spread_fac : float = 1.01
    quantile_range : float = 0.01
    num_qnt_strikes: int = 200

    def __post_init__(self):
        verify_inp( self.smoothness is None or (isinstance(self.smoothness, (float,int,np.number)) and self.smoothness>=0), lambda : f"Expected 'smoothness' to be a non-negative number or None, but found {str(self.smoothness)}" )
        verify_inp( isinstance(self.grid_expiries, np.ndarray) and self.grid_expiries.ndim==1 and self.grid_expiries[0]>0. and np.all(np.diff(self.grid_expiries)>0), lambda : f"Expected 'grid_expiries' to be a 1D numpy array of positive increasing numbers, but found {str(self.grid_expiries)[:100]}" )
        verify_inp( isinstance(self.grid_nms, tuple) and len(self.grid_nms)==2 and self.grid_nms[0] < 0. and self.grid_nms[1] > 0., lambda : f"Expected 'grid_nms' to be a tuple with two elements (nm_min, nm_max) where nm_min<0 and nm_max>0, but found {self.grid_nms}" )
        verify_inp( self.grid_num_nms>=5, lambda : f"Expected 'grid_num_nms' to be an integer greater than or equal to 5, but found {self.grid_num_nms}" )
        verify_inp( self.vol_spread_fac>0., lambda : f"Expected 'vol_spread_fac' to be a positive number, but found {self.vol_spread_fac}" )

    def __str__(self):
        return f"{f'{self.smoothness:.3g}' if self.smoothness is not None else 'lin'},xp={self.grid_expiries[0]:.2g}-{self.grid_expiries[-1]:.2g},nms=({self.grid_nms[0]:.1f},{self.grid_nms[1]:.1f}),num={self.grid_num_nms},vsopr{self.vol_spread_fac:.4g}"

    def _generate_grid_nms(self):
        dist = float(self.grid_nms[1] - self.grid_nms[0])
        Nlo  = min(max(2, float(self.grid_num_nms*abs(self.grid_nms[0]))//dist),self.grid_num_nms-3)
        Nhi = self.grid_num_nms - Nlo - 1

        assert Nlo>=2 and Nhi>=2, f"Internal error: error {Nlo} {Nhi} for grid_nms {self.grid_nms} and grid_num_nms {self.grid_num_nms}."
        grid_nms = np.concatenate( ( np.linspace(self.grid_nms[0], 0., int(Nlo), endpoint=False), np.linspace(0., self.grid_nms[1], int(Nhi)+1) ) )
        assert len(grid_nms)==self.grid_num_nms, f"Internal error: generated number of moneyness levels {len(grid_nms)} does not match expected {self.grid_num_nms} for grid_nms {self.grid_nms} and grid_num_nms {self.grid_num_nms}."
        return grid_nms

def _bd_uid(ticker, d, time_series_def : TimeSeriesDefinition, benchmark_config : SanosBenchmarkConfig, ) -> str:
    """ Returns a relative UID for a time series based on its definition and the requested fields. """
    assert isinstance(time_series_def, TimeSeriesDefinition), f"Internal error: expected 'time_series_def' to be an instance of TimeSeriesDefinition, but got {type(time_series_def)}. Please contact the authors with an example."
    assert isinstance(benchmark_config, SanosBenchmarkConfig), f"Internal error: expected 'benchmark_config' to be an instance of SanosBenchmarkConfig, but got {type(benchmark_config)}. Please contact the authors with an example."
    ts_id  = named_unique_filename48_8(str(time_series_def), time_series_def)
    bm_id  = named_unique_filename48_8(str(benchmark_config), benchmark_config)
    return f"{ts_id}/{bm_id}/{ticker}/{d}"

#@_cache(fmt=SubDir.PYTREE_HDF5).cache(
#            "0.32", dependencies=[SanosBenchmarkConfig, build_time_series, TimeSeriesDefinition],
#            uid=_bd_uid)
@version("0.1") 
def sanos_benchmark_day( 
                 ticker : str,
                 d: date,
                 data : PrettyObject,
                 time_series_def : TimeSeriesDefinition,
                 benchmark_config : SanosBenchmarkConfig,
                 verbose : Context = Context.all ) -> dict[str, np.ndarray]:        
    """
    For a given day perform:

    1) Fit "open" SANOS to the prevaling market data and strikes. 
    2) Fit "grid" SANOS to the same market data and strikes, but with the grid geometry. This is the benchmark to beat for any ML SANOS fit based on the same grid geometry.
    """

    str_smth = 'linear' if benchmark_config.smoothness is None else f'{benchmark_config.smoothness:.3g}-smooth'
    grid_nms = benchmark_config._generate_grid_nms()

    # market data & geometry
    # ----------------------

    is_valid               = data['is_valid']
    mask_arb_free          = data['mask_arb_free'][is_valid]
    mask_arb_free          = mask_arb_free if np.any(~mask_arb_free) else None    
    mask_arb_free = None

    market                 = PrettyObject()
    market.mask_arb_free   = mask_arb_free
    market.strikes         = data['pure_strike'][is_valid]
    market.expiries        = data['tte_y'][is_valid]
    market.bid             = data['pure_bid_price'][is_valid]
    market.ask             = data['pure_ask_price'][is_valid]

    # we only train our smooth model
    # to the strikes the linear model
    # managed to fit to.
    market.target_strikes  = market.strikes[mask_arb_free] if mask_arb_free is not None else market.strikes
    market.target_bid      = market.bid[mask_arb_free] if mask_arb_free is not None else market.bid
    market.target_ask      = market.ask[mask_arb_free] if mask_arb_free is not None else market.ask
    market.target_expiries = market.expiries[mask_arb_free] if mask_arb_free is not None else market.expiries

    # transform into grids
    # --------------------

    af_target = PrettyObject()
    af_target.strikes  = []
    af_target.expiries = []
    af_target.asks     = []
    af_target.bids     = []

    uttys = np.unique(market.target_expiries)
    for tty in uttys:
        """
        if tty > benchmark_config.grid_expiries[-1] and\
           len(af_target.expiries) > 0 and\
           af_target.expiries[-1] >= benchmark_config.grid_expiries[-1]:
            continue
        """
        mask = market.target_expiries == tty
        strikes_t = market.target_strikes[mask]
        if len(strikes_t)<=3:
            continue
        ixs = np.argsort(strikes_t)

        af_target.expiries.append( tty )
        af_target.strikes.append( strikes_t[ixs] )
        af_target.bids.append( market.target_bid[mask][ixs] )
        af_target.asks.append( market.target_ask[mask][ixs] )

    af_target.expiries = np.array(af_target.expiries)

    # fit a smooth SANOS model to the market
    # --------------------------------------

    with verbose.write_t(lambda : f"\rFitting {str_smth} SANOS for {ticker} {d}... ", end="") as tme:
        mk_arb_free     = MkArbFree(smoothness=benchmark_config.smoothness, surface_expiry_linear=True )
        af_output       = mk_arb_free( strikes = af_target.strikes, expiries = af_target.expiries, bids = af_target.bids, asks = af_target.asks, verbose=verbose.quiet )

        # price full market
        market.fit      = af_output.price( expiries = market.expiries, strikes = market.strikes, and_vols=False, mode="surface" )
        half_spread     = np.maximum( 0.5*(market.ask -market.bid), mk_arb_free.min_half_spread )
        market.fit_err  = (np.maximum( 0., market.fit - market.ask ) + np.maximum(0., market.bid - market.fit) ) / half_spread

        # compute & report fit
        fit_err         = market.fit_err[mask_arb_free] if not mask_arb_free is None else market.fit_err 
        pcnt_fit        = float(np.sum(fit_err > 0.)) / float(len(fit_err))
        median_fit_err  = np.median(fit_err[fit_err>0.] ) if np.any(fit_err>0.) else 0.
        verbose.write(f"done; median error is {median_fit_err:.2f} half-spreads for the {pcnt_fit*100:.1f}% of options which were not fitted within the bid-ask spread. This took {tme}.", head=False )

    # use the smooth SANOS model to generate prices
    # for a grid of options - then fit another
    # SANOS model to that grid.

    # C(1,K) = \sum q_i C(S_i,K)
    # P(1,k) = \sum q_i P[S_i,k]

    # NMS grod
    # --------

    with verbose.write_t(lambda : f"\rTraining {str_smth} SANOS defined NMS grid... ", end="") as tme:
        if benchmark_config.grid_expiries[-1] > af_target.expiries[-1]:
            warnings.warn(f"Warning {str_smth} SANOS {ticker} {d}: the maximum grid expiry {benchmark_config.grid_expiries[-1]:.4g} is greater than the maximum market expiry {af_target.expiries[-1]:.4g}.")

        M    = len( benchmark_config.grid_expiries )
        grid = PrettyObject()
        grid.expiries   = np.asarray( benchmark_config.grid_expiries )
        grid.atms       = af_output.interpolate_atm_vols( grid.expiries )
        assert np.all(np.isfinite(grid.atms)), f"Internal error: generated non-finite ATM vols {grid.atms} for grid expiries {grid.expiries} from input expiries {af_output.expiries} and variances {af_output.variances}."
        grid.strikes    = NMS5.from_nms( z=grid_nms[None,:], t=grid.expiries[:,None], vol=grid.atms[:,None] )
        grid.mids,\
        grid.vols       = af_output.price( expiries=grid.expiries[:,None], strikes=grid.strikes, and_vols=True, mode="surface" )

        # fit and price market
        af_grid         = mk_arb_free( expiries = grid.expiries, 
                                       strikes =  [ grid.strikes[t,:] for t in range(M) ],
                                       mids = [ grid.mids[t,:] for t in range(M) ], 
                                       xstrikes = [ grid.xstrikes[t,:] for t in range(M) ],
                                       vols = grid.atms,
                                       verbose=verbose.quiet )
        grid.fits       = [ af_grid.price( expiries=grid.expiries[t], strikes=grid.strikes[t,:], and_vols=False, mode="surface" ) for t in range(M) ]
        grid.fits       = np.stack( grid.fits, axis=0 )
        assert grid.fits.shape == grid.mids.shape, f"Internal error: expected grid.fits to have shape {grid.mids.shape} but found {grid.fits.shape} for grid expiries {grid.expiries} and strikes {grid.strikes}."
        assert grid.dlvs.shape == grid.mids.shape, f"Internal error: expected grid.dlvs to have shape {grid.mids.shape} but found {grid.dlvs.shape} for grid expiries {grid.expiries} and strikes {grid.strikes}."
        market.grid_fit = af_grid.price( expiries=market.expiries, strikes=market.strikes, and_vols=False, mode="surface" )
        half_spread     = np.maximum( 0.5*(market.ask -market.bid), mk_arb_free.min_half_spread )
        market.grid_err = (np.maximum( 0, market.grid_fit - market.ask ) + np.maximum(0., market.bid - market.grid_fit) ) / half_spread

        # compute & report fit

        target_grid     = market.grid_err[mask_arb_free] if not mask_arb_free is None else market.grid_err
        grid.pcnt_err   = np.sum( target_grid > 0.) / len(target_grid)
        grid.median_err = np.median( target_grid[target_grid>0.] ) if np.any(target_grid>0.) else 0.
        verbose.write(f"done; median error to market is {grid.median_err:.2f} half-spreads for the {grid.pcnt_err*100:.1f}% of options which were not within the bid-ask spread. (This model was not fitted to all options.) This took {tme}.", head=False )


    with verbose.write_t(lambda : f"\rComputing {str_smth} quantiles... ", end="") as tme:
        if benchmark_config.grid_expiries[-1] > af_target.expiries[-1]:
            warnings.warn(f"Warning {str_smth} SANOS {ticker} {d}: the maximum grid expiry {benchmark_config.grid_expiries[-1]:.4g} is greater than the maximum market expiry {af_target.expiries[-1]:.4g}.")

        grid.quantile_values  = np.linspace(benchmark_config.quantile_range, 1.-benchmark_config.quantile_range, 
                                      benchmark_config.grid_num_nms )
        grid.quantiles  = np.zeros((M,benchmark_config.grid_num_nms), dtype=grid.strikes.dtype)

        for t, expiry in enumerate(grid.expiries):
            lower_k       = grid.strikes[t,0]/2.
            upper_k       = grid.strikes[t,-1]*1.5
            quant_strikes = np.exp( np.linspace( np.log(lower_k), np.log(upper_k), benchmark_config.num_qnt_strikes ) )
            assert quant_strikes[-1]*2 < 10, f"Internal error: expected maximum quantile strike {quant_strikes[-1]:.4g} to be less than 10 for expiry {expiry}"

            range_calls   = af_output.price( expiries=expiry, strikes=quant_strikes, and_vols=False, mode="surface" )
            P             = 1.+np.diff(range_calls,prepend=1.)/np.diff(quant_strikes, prepend=0.)
            assert P.shape==(benchmark_config.num_qnt_strikes, ), f"Internal error: expected call price differences to have shape {(benchmark_config.num_qnt_strikes,)} but found {P.shape} for expiry {expiry}."
            if benchmark_config.quantile_range < P[0]:
                warnings.warn(f"Warning {str_smth} SANOS {ticker} {d} expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d the minimum quantile {benchmark_config.quantile_range:.4g} is smaller than the minimum call price {P[0]:.4g} for the generated grid quantile strikes fron {quant_strikes[0]:.4g} to {quant_strikes[-1]:.4g}.")
            if 1.-benchmark_config.quantile_range > P[-1]:
                warnings.warn(f"Warning {str_smth} SANOS {ticker} {d} expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d the maximum quantile {1.-benchmark_config.quantile_range:.4g} is greater than the maximum call price {P[-1]:.4g} for the generated grid quantile strikes fron {quant_strikes[0]:.4g} to {quant_strikes[-1]:.4g}.")
            P             = np.concatenate( ([0.], P, [1.]) )
            X             = np.concatenate( ([0.], quant_strikes, [10.]) )
            grid.quantiles[t,:] \
                          = scipy.interpolate.PchipInterpolator(X,P)( grid.quantile_values )

        grid.mids,\
        grid.vols       = af_output.price( expiries=grid.expiries[:,None], strikes=grid.strikes, and_vols=True, mode="surface" )


    # convert back
    # ------------

    full_market = PrettyObject()
    full_market.fit      = np.zeros( is_valid.shape, dtype=market.fit.dtype )
    full_market.fit_err  = np.zeros( is_valid.shape, dtype=market.fit.dtype )
    full_market.grid_fit = np.zeros( is_valid.shape, dtype=market.fit.dtype )
    full_market.grid_err = np.zeros( is_valid.shape, dtype=market.fit.dtype )
    full_market.fit[is_valid] = market.fit
    full_market.fit_err[is_valid] = market.fit_err
    full_market.grid_fit[is_valid] = market.grid_fit
    full_market.grid_err[is_valid] = market.grid_err

    full_market.pcnt_err = pcnt_fit
    full_market.median_err = median_fit_err
    assert set(grid) == {'expiries','atms', 'strikes', 'mids', 'quantiles', 'vols', 'bids', 'asks', 'fits', 'pcnt_err', 'median_err'}, f"Internal error: expected grid to have keys 'expiries','atms', 'strikes', 'mids', 'vols', 'bids', 'asks', 'fits', 'pcnt_err', 'median_err', but found {sorted(set(grid))}"

    return dict( market=dict(full_market), grid=dict(grid) )


def _b_uid(ticker, date_range, time_series_def : TimeSeriesDefinition, benchmark_config : SanosBenchmarkConfig, ) -> str:
    """ Returns a relative UID for a time series based on its definition and the requested fields. """
    assert isinstance(time_series_def, TimeSeriesDefinition), f"Internal error: expected 'time_series_def' to be an instance of TimeSeriesDefinition, but got {type(time_series_def)}. Please contact the authors with an example."
    assert isinstance(benchmark_config, SanosBenchmarkConfig), f"Internal error: expected 'benchmark_config' to be an instance of SanosBenchmarkConfig, but got {type(benchmark_config)}. Please contact the authors with an example."
    ts_id  = named_unique_filename48_8(str(time_series_def), time_series_def)
    bm_id  = named_unique_filename48_8(str(benchmark_config), benchmark_config)
    return f"{ts_id}/sanos bm {bm_id}/{ticker}/{date_range}"

@_cache(fmt=SubDir.PYTREE_HDF5).cache(
            "0.1", dependencies=[SanosBenchmarkConfig, sanos_benchmark_day, TimeSeriesDefinition],
            uid=_b_uid)
def sanos_benchmark( 
                 ticker : str,
                 date_range : DateRange,
                 time_series : dict,
                 time_series_def : TimeSeriesDefinition,
                 benchmark_config : SanosBenchmarkConfig,
                 pool_config : PoolConfig = PoolConfig.off(),
                 override_cache_mode : CacheMode | None = None,
                 verbose : Context = Context.all ) -> dict[str, np.ndarray]:      

    num_dates = len(time_series['dates'])
    pool      = JCPool(num_workers=pool_config.num_workers, threading=pool_config.threading)
    quietprc  = pool_config.num_workers > 0 and not pool_config.threading

    slice_fields = ['is_valid', 'mask_arb_free', 'pure_strike', 'tte_y', 'pure_bid_price', 'pure_ask_price']


    def slice_data( isample ):
        r = { k: time_series[k][isample,:] for k in slice_fields }
        return r


    def process( ticker, d, ddata, time_series_def, benchmark_config : SanosBenchmarkConfig ):
        return sanos_benchmark_day( ticker=ticker,
                                    d=d,
                                    data=PrettyObject(ddata),
                                    time_series_def=time_series_def,
                                    benchmark_config=benchmark_config,
                                    verbose=Context.quiet if quietprc else Context.all )

    results = {}
    
    with verbose.write_t(lambda : f"Creating SANOS Benchmark for {ticker} from {time_series['dates'][0]} to {time_series['dates'][-1]} with {num_dates} samples:") as tme:
        for d, dresults in pool.parallel( { d : pool.delayed( process )( ticker=ticker, d=d, ddata=slice_data(isample), time_series_def=time_series_def, benchmark_config=benchmark_config ) for isample, d in enumerate(time_series['dates']) } ):
            results[d] = dresults           
            if tme.interval_test(2.) and quietprc:
                verbose.write(lambda : f"\rProcessed date {len(results)}/{num_dates} {d} for {ticker}. Market % error {dresults['market']['pcnt_err']:.2f} median {dresults['market']['median_err']:.3g}, Grid % error {dresults['grid']['pcnt_err']:.2f} median {dresults['grid']['median_err']:.3g}. So far this took {tme}...", end='')

    verbose.write(lambda : f"\rProcessed all {num_dates} for {ticker} from {time_series['dates'][0]} to {time_series['dates'][-1]}. This took {tme}.")

    # allocate one big frame
    # ----------------------

    ignore_fields = {'pcnt_err', 'median_err'}

    result0 = results[time_series['dates'][0]]
    grid    = PrettyObject()
    for k, v in result0['grid'].items():
        if not k in ignore_fields:
            grid[k] = np.zeros( (num_dates,) + v.shape, dtype=v.dtype )
    for k, v in result0['market'].items():
        if not k in ignore_fields:
            assert k not in time_series, f"Internal error: expected market result key {k} to not be in input data keys {sorted(time_series)}, but found it. Please contact the authors with an example."
            time_series[k] = np.zeros( (num_dates,) + v.shape, dtype=v.dtype )

    for isample, d in enumerate(time_series['dates']):
        for k, v in results[d]['grid'].items():
            if k not in ignore_fields:
                grid[k][isample,...] = v
        for k, v in results[d]['market'].items():
            if k not in ignore_fields:
                time_series[k][isample,...] = v

    return dict( time_series=time_series, grid=dict(grid) )
