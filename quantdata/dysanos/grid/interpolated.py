"""
Dysanos Grid
^^^^^^^^^^^^
"""

from cdxcore.bs import bs
from cdxcore.dynaplot import figure, colors_tableau
from cdxcore.err import verify_inp
from cdxcore.subdir import SubDir
from cdxcore.uniquehash import unique_hash8, named_unique_filename48_8
from cdxcore.util import fmt_list
from cdxcore.version import version
from cdxcore.verbose import Context
from cdxcore.pretty import PrettyObject

import numpy as np
import math
from dataclasses import dataclass, field    
import scipy as scipy
from datetime import date
from collections.abc import Callable
import warnings

from quantdata.biz_calendar import DAYS_PER_YEAR, NormalizedStrikes, to_date, NMS5, NMS, DateRange
from quantdata.dysanos.grid import env, SubDir, version, _locations_dysanos_grid
from quantdata.sanos.sanos import SanosFitter
from quantdata.sanos.sanos_jax import fit_dlvsqs
from quantdata.pool import PoolConfig
from quantdata.widb.options_filter import MkArbFree, RawDayFilter, ExpiryFilter, OptionsFilter, filter_options_day

_cache = _locations_dysanos_grid.training("interpolated")

#DEFAULT_EXPIRIES = np.array([1./255., 3./255., 5./255., 10./255., 20./255., 40./255.,0.5,1.], dtype=np.float64)
DEFAULT_EXPIRIES = np.array([2./255., 5./255., 10./255., 20./255., 40./255.,0.5,1.], dtype=np.float64)
DEFAULT_NUMS     = (-2,1)


@version("0.1")
@dataclass(frozen=True)
class GridConfig:
    """
    Extracts from real market data a standardized grid of option prices.

    This standardized grid can then be used in subsequent calculations to fit a uniform SANOS model.
    """

    # grid definition
    grid_expiries    : np.array = field( default_factory=lambda: DEFAULT_EXPIRIES )   # expiries, in years
    grid_range       : tuple = DEFAULT_NUMS  # range for the grid
    grid_num_strikes : int = 20              # number of strikes in the grid
    mode             : str = "nms"           # mode for the grid: "nms" or "logK"
    nms_eps          : float = NMS.EPS       # 1 hour

    # interpolation
    smoothness       : float = 0.25          # SANOS smoothness for interpolation
    min_dlv          : float = 0.01          # Minimum DLV used for linear programming.
    max_dlv          : float = 2.            # Maximum DLV used for linear programming.

    # weights
    min_vega_sqrt    : float = 1E-4          # minimum sqrt vega 

    def __post_init__(self):
        verify_inp( self.mode in ("nms", "logK"), lambda : f"Expected mode to be 'nms' or 'logK' but found {self.mode}.")
        verify_inp( self.grid_num_strikes >= 2, lambda : f"Expected grid_num_strikes to be at least 2 but found {self.grid_num_strikes}.")
        verify_inp( self.grid_expiries[0] > 0. and np.all(np.diff(self.grid_expiries) > 0.), lambda : f"Expected grid expiries to be positive and increasing but found {self.grid_expiries}.")
        verify_inp( len(self.grid_range) == 2 and self.grid_range[0] < 0 and self.grid_range[1] > 0, lambda : f"Expected grid_range to be a tuple of two numbers (lo, hi) with lo<0 and hi>0 but found {self.grid_range}.")
        verify_inp( self.smoothness is None or (self.smoothness >= 0. and self.smoothness < 1.), lambda : f"Expected smoothness to be None or a number between 0 and 1 but found {self.smoothness:.3g}.")
        verify_inp( self.min_dlv >= 0. and self.max_dlv > self.min_dlv, lambda : f"Expected min_dlv to be non-negative and max_dlv to be greater than min_dlv but found min_dlv={self.min_dlv:.3g} and max_dlv={self.max_dlv:.3g}." )
        verify_inp( self.min_vega_sqrt >= 0., lambda : f"Expected min_vega_sqrt to be non-negative but found {self.min_vega_sqrt:.3g}." )
    
    def __str__(self):
        expiries = "("
        for exp in self.grid_expiries:
            expiries += f"{exp*DAYS_PER_YEAR:.1f},"
        expiries = expiries[:-1] + ")"
        str_smoothness = f"sth={self.smoothness:.2f}" if self.smoothness is not None else "lin"
        return f"{self.mode},rg=({self.grid_range[0]},{self.grid_range[1]}),nm={self.grid_num_strikes},xp={expiries[:40]};{str_smoothness},mid={self.min_dlv:.3g},mxd={self.max_dlv:.3g},msqv={self.min_vega_sqrt:.3g}"

    @property
    def T(self) -> int:
        return len(self.grid_expiries)
    @property
    def N(self) -> int:
        return self.grid_num_strikes
    @property
    def TN(self) -> tuple:
        return (self.T, self.N)
    
def _uid_int_grid( ticker, d, raw_day_filter, expiry_filter, options_filter, mk_arb_free, grid_config ):
    return f"{named_unique_filename48_8(f"grid {grid_config}", grid_config, raw_day_filter, expiry_filter, options_filter, mk_arb_free )}/{ticker}/{d}"      

@_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.1", dependencies=[GridConfig, MkArbFree, SanosFitter, OptionsFilter, RawDayFilter, ExpiryFilter, MkArbFree, filter_options_day],
    uid=_uid_int_grid)
def interpolate_grid( 
                ticker : str, d : date, *,
                options_filter : OptionsFilter = OptionsFilter(),
                raw_day_filter : RawDayFilter = RawDayFilter(), 
                expiry_filter: ExpiryFilter = ExpiryFilter(),
                mk_arb_free : MkArbFree = MkArbFree(),                 
                grid_config : GridConfig = GridConfig(),
                message_context : Callable|str|None = None,
                verbose : Context|None = Context.all ) -> dict|None:
    """
    Interpolate the market for a given target grid        
    """

    verbose = verbose if not verbose is None else Context.quiet

    def message(with_cols : bool = True) -> str:
        my_context = f"{ticker} {d}"
        if message_context is None:
            msg = my_context
        else:
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg != "":
                msg += " " + my_context
            else:
                msg = my_context
        msg = msg[0].upper() + msg[1:]
        return msg if not with_cols else msg + ": "

    def _generate_grid(expiries, vols):
        """ Generate grid which contains '1' """

        NMS = NormalizedStrikes(grid_config.nms_eps)

        # generate strikes which contain '1'
        dist = float(grid_config.grid_range[1] - grid_config.grid_range[0])
        assert dist > 0., f"Internal error: expected grid_range to have positive distance but found {grid_config.grid_range}."
        Nlo  = min(max(2, float(grid_config.grid_num_strikes*abs(grid_config.grid_range[0]))//dist),grid_config.grid_num_strikes-3)
        Nhi = grid_config.grid_num_strikes - Nlo - 1
        assert Nlo>=2 and Nhi>=2, f"{message()}Internal error: error {Nlo} {Nhi} for grid_range {grid_config.grid_range} and grid_num_strikes {grid_config.grid_num_strikes}."
        raw_strikes = np.concatenate( ( np.linspace(grid_config.grid_range[0], 0., int(Nlo), endpoint=False), np.linspace(0., grid_config.grid_range[1], int(Nhi)+1) ) )
        assert len(raw_strikes)==grid_config.N, f"{message()}Internal error: generated number of moneyness levels {len(raw_strikes)} does not match expected {grid_config.N}."

        if grid_config.mode == "logK":
            strikes = np.full( (grid_config.T, grid_config.N), np.exp(raw_strikes)[None,:] )
            nms     = NMS.to_nms( k=strikes, t=expiries[:,None], vol=vols[:,None], eps_vol = vols[0] ) 
        else:
            assert grid_config.mode == "nms", f"{message()}Internal error: expected grid_config.mode to be 'nms' or 'logK' but found {grid_config.mode}."
            nms     = np.full( (grid_config.T, grid_config.N), raw_strikes[None,:] )
            strikes = NMS.from_nms( z=nms, t=expiries[:,None], vol=vols[:,None], eps_vol = vols[0] ) 
        assert np.max( np.diff(strikes[:,0]) ) <= 0., f"{message()}Internal error: expected lower strikes to be decreasing but found {np.max( np.diff(strikes[:,0]) )}."
        assert np.min( np.diff(strikes[:,-1]) ) >= 0., f"{message()}Internal error: expected upper strikes to be increasing but found {np.min( np.diff(strikes[:,-1]) )}."

        return nms, strikes

    # load data
    # ---------

    data    = filter_options_day(ticker,d, mk_arb_free=mk_arb_free, options_filter=options_filter, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter, verbose=verbose, raise_if_bad_day=False )
    if data is None or len(data) == 0:
        return {}

    strikes = data['pure_strike'].to_numpy()
    ixs     = np.argsort(strikes)
    strikes = strikes[ixs]
    tte_y   = data['tte_y'].to_numpy()[ixs]
    bid     = data['pure_bid_price'].to_numpy()[ixs]
    ask     = data['pure_ask_price'].to_numpy()[ixs]
    del ixs

    # split into per-expiry
    # ---------------------

    market = PrettyObject()
    market.uid      = unique_hash8( raw_day_filter, expiry_filter, options_filter, mk_arb_free )
    market.expiries = np.unique(tte_y)
    market.strikes  = []
    market.bids     = []
    market.asks     = []

    assert len(market.expiries) > 0, f"{message()}No expiries found after filtering for ticker {ticker} on date {d} with raw_day_filter={raw_day_filter}, expiry_filter={expiry_filter}, options_filter={options_filter} and mk_arb_free={mk_arb_free}. {data}"

    for expiry in market.expiries:
        mask = tte_y == expiry
        market.strikes.append( strikes[mask] )
        market.bids.append( bid[mask] )
        market.asks.append( ask[mask] )

    assert set(market) >= {'expiries', 'strikes', 'bids', 'asks'}, f"{message()}Expected market to have keys 'expiries', 'strikes', 'bids', 'asks' but found {set(market)}"

    str_smth = f"{grid_config.smoothness:.2f}-smooth" if grid_config.smoothness is not None else "linear"

    # fit a smooth SANOS model to the market
    # --------------------------------------

    with verbose.write_t(lambda : f"\r{message()}Fitting {str_smth} SANOS to {sum( len(s) for s in market.strikes )} strikes... ", end="") as tme:
        mk_arb_free  = MkArbFree(smoothness=grid_config.smoothness, surface_expiry_linear=True, min_dlv=grid_config.min_dlv, max_dlv=grid_config.max_dlv )
        af_output    = mk_arb_free( strikes = market.strikes, expiries = market.expiries, bids = market.bids, asks = market.asks, verbose=verbose.quiet )
        fit_result   = af_output.compute_market_fit( expiries=market.expiries, strikes=market.strikes, bids=market.bids, asks=market.asks, and_vols=False )
        verbose.write(f"done; median error is {fit_result.median_err:.2f} half-spreads for the {fit_result.pcnt_fit*100:.1f}% of options which were not fitted within bid-ask spread. This took {tme}.", head=False )

    # NMS grid
    # --------

    nms_grid = PrettyObject()

    with verbose.write_t(lambda : f"\rTraining {str_smth} SANOS on NMS {grid_config.grid_range} {grid_config.mode} grid:") as tme:
        if grid_config.grid_expiries[-1] > market.expiries[-1]:
            warnings.warn(f"{message()}Warning {str_smth} SANOS: the maximum grid expiry {grid_config.grid_expiries[-1]:.4g} is greater than the maximum market expiry {market.expiries[-1]:.4g}.")

        T                   = len( grid_config.grid_expiries )
        N                   = grid_config.grid_num_strikes
        nms_grid.expiries   = np.asarray( grid_config.grid_expiries )
        atm_vols            = af_output.interpolate_atm_vols( nms_grid.expiries )
        nms_grid.nms,\
        strikes             = _generate_grid(nms_grid.expiries, atm_vols)
        nms_grid.variances  = (atm_vols**2) * nms_grid.expiries

        nms_grid.mids,\
        nms_grid.mid_vols   = af_output.price( expiries=nms_grid.expiries[:,None], strikes=strikes, and_vols=True, mode="surface" )
        nms_grid.mid_vega,\
        nms_grid.mid_delta  = bs(k=strikes, sqrtT=np.sqrt( nms_grid.expiries )[:,None], vol=nms_grid.mid_vols, what=bs.VEGA|bs.DELTA )['vega','delta']
        nms_grid.xstrikes   = np.zeros((T,grid_config.grid_num_strikes+2), dtype=strikes.dtype)
        nms_grid.xstrikes[:,1:-1] = strikes
        nms_grid.xstrikes[:,0]    = strikes[:,0]/2.
        nms_grid.xstrikes[:,-1]   = strikes[:,-1]*1.5
        vega_sqrt                 = nms_grid.mid_vega / np.sqrt(nms_grid.expiries[:,None])
        has_vega                  = vega_sqrt >= grid_config.min_vega_sqrt
        nms_grid.weights          = np.where( has_vega, 1./np.where( has_vega, vega_sqrt, 1.), 0. )
        del strikes

        # fit and price market4
        sanos           = SanosFitter(smoothness=grid_config.smoothness, min_dlv=grid_config.min_dlv, max_dlv=grid_config.max_dlv, _raise_on_eps=False )
        sn_grid         = sanos(    expiries = nms_grid.expiries, 
                                    strikes  = [ nms_grid.xstrikes[t,1:-1] for t in range(T) ],
                                    mids     = [ nms_grid.mids[t,:] for t in range(T) ], 
                                    xstrikes = [ nms_grid.xstrikes[t,:] for t in range(T) ],
                                    vols     = atm_vols,
                                    message_context = lambda : message(with_cols=False),
                                    verbose=verbose(1))
        nms_grid.fits,\
        nms_grid.fit_vols    = sn_grid( expiries=nms_grid.expiries[:,None], strikes=nms_grid.xstrikes[:,1:-1], and_vols=True )
        sn_grid.__dict__["_eps_price"] = 1E-6
        assert isinstance( nms_grid.fit_vols, np.ndarray ) , f"{message()}Internal error: expected nms_grid.fit_vols to be a numpy array but found {type(nms_grid.fit_vols)}."
        nms_grid.fit_dlvsqs  = sn_grid.dlvsqs( min_dlv=0., max_dlv=None, on_all_strikes=False ) # do not use DLV bounds here!
        err_vols             = np.median( np.abs( nms_grid.fit_vols - nms_grid.mid_vols ) )
        assert isinstance( nms_grid.fit_dlvsqs, list), f"{message()}Internal error: expected nms_grid.fit_dlvsqs to be a list but found {type(nms_grid.fit_dlvsqs)}."
        nms_grid.fit_dlvsqs  = np.stack( nms_grid.fit_dlvsqs, axis=0 )
        verbose.write(f"Trained {str_smth} SANOS on NMS {grid_config.grid_range} {grid_config.mode} grid; this took {tme}. This median match to implied vols was {err_vols:.2f}.")

    # grid
    assert nms_grid.expiries.shape == (T,), f"{message()}Internal error: expected nms_grid.expiries to have shape {(T,)} but found {nms_grid.expiries.shape}."
    assert nms_grid.variances.shape == (T,), f"{message()}Internal error: expected nms_grid.variances to have shape {(T,)} but found {nms_grid.variances.shape}."
    assert nms_grid.xstrikes.shape == (T,N+2), f"{message()}Internal error: expected nms_grid.strikes to have shape {(T,N)} but found {nms_grid.strikes.shape}."
    assert nms_grid.nms.shape == (T,N), f"{message()}Internal error: expected nms_grid.nms to have shape {(T,N)} but found {nms_grid.nms.shape}."

    # interpolated market
    assert nms_grid.mids.shape == (T,N), f"{message()}Internal error: expected nms_grid.mids to have shape {(T,N)} but found {nms_grid.mids.shape}."
    assert nms_grid.mid_vols.shape == (T,N), f"{message()}Internal error: expected nms_grid.mid_vols to have shape {(T,N)} but found {nms_grid.mid_vols.shape}."
    assert nms_grid.mid_vega.shape == (T,N), f"{message()}Internal error: expected nms_grid.mid_vega to have shape {(T,N)} but found {nms_grid.mid_vega.shape}."
    assert nms_grid.mid_delta.shape == (T,N), f"{message()}Internal error: expected nms_grid.mid_delta to have shape {(T,N)} but found {nms_grid.mid_delta.shape}."
    assert nms_grid.weights.shape == (T,N), f"{message()}Internal error: expected nms_grid.weights to have shape {(T,N)} but found {nms_grid.weights.shape}."

    # fitted with the grid SANOS
    assert nms_grid.fits.shape == (T,N), f"{message()}Internal error: expected nms_grid.fits to have shape {(T,N)} but found {nms_grid.fits.shape}."
    assert nms_grid.fit_vols.shape == (T,N), f"{message()}Internal error: expected nms_grid.fit_vols to have shape {(T,N)} but found {nms_grid.fit_vols.shape}."
    assert nms_grid.fit_dlvsqs.shape == (T,N), f"{message()}Internal error: expected nms_grid.fit_dlvsqs to have shape {(T,N)} but found {nms_grid.fit_dlvsqs.shape}."

    return dict(nms_grid)

def _uid_int_ts( ticker, date_range, raw_day_filter, expiry_filter, options_filter, mk_arb_free, grid_config ):
    return f"{named_unique_filename48_8(f"ts {grid_config}", grid_config, raw_day_filter, expiry_filter, options_filter, mk_arb_free )}/{ticker}/{date_range}"      

@_cache(fmt=SubDir.PYTREE_HDF5).cache(
    "0.16", dependencies=[interpolate_grid],
    uid=_uid_int_ts)
def interpolate_time_series( 
                 ticker : str, 
                 date_range : DateRange, *,
                 grid_config : GridConfig = GridConfig(),
                 options_filter : OptionsFilter = OptionsFilter(),
                 raw_day_filter : RawDayFilter = RawDayFilter(), 
                 expiry_filter: ExpiryFilter = ExpiryFilter(),
                 mk_arb_free : MkArbFree = MkArbFree(),
                 pool_config : PoolConfig = PoolConfig.off(),
                 verbose: Context = Context.all ):

    pool  = pool_config.pool()
    dates = date_range.get_business_days()

    results = {}
    cnt = 0
    tme = verbose.timer()

    verbose.write(lambda : f"Generating interpolated time series for {ticker} for {len(dates)} dates from {dates[0]} to {dates[-1]} on {grid_config.mode} grid:")
    verbose.report(1,lambda : f"Day filter: {raw_day_filter}")
    verbose.report(1,lambda : f"Expiry filter: {expiry_filter}")
    verbose.report(1,lambda : f"Options filter: {options_filter}")
    verbose.report(1,lambda : f"Market Arb free settings: {mk_arb_free}")
    verbose.report(1,lambda : f"Grid config: {grid_config}")
    with verbose.write_t(lambda : f"\rLoading market data for {ticker} for {len(dates)} dates from {dates[0]} to {dates[-1]} on NMS grid with {str(pool_config)} pool:") as tme:

        for d, nms_grid in pool.parallel( { d: pool.delayed(interpolate_grid)( 
                                                                        ticker=ticker, d=d,
                                                                        grid_config=grid_config, 
                                                                        mk_arb_free=mk_arb_free, options_filter=options_filter, raw_day_filter=raw_day_filter, expiry_filter=expiry_filter,
                                                                        verbose=verbose(1) if pool.is_no_pool else None ) for d in dates } ):
            cnt +=1 
            if nms_grid is None or len(nms_grid) == 0:
                verbose.report(1,f"\rFitting {ticker} date {cnt}/{len(dates)} {d}: no data after filtering, skipping.")
            else:
                verbose.report(1, f"\rFitting {ticker} date {cnt}/{len(dates)} {d}: done; so far this took {tme}...", end='\n' if pool.is_no_pool else '' )
                results[d] = PrettyObject(nms_grid)

        verbose.write(lambda : f"\rLoaded market data for {ticker} for {cnt}/{len(dates)} dates from {dates[0]} to {dates[-1]} on NMS grid with {str(pool_config)} pool; this took {tme}.")

    # transform into one big array

    dates = sorted(results)
    results = { d: results[d] for d in dates }    

    r = PrettyObject()
    r.dates      = np.array(dates, dtype="datetime64[D]") # (S,)
    r.expiries   = np.stack( [ results[d].expiries for d in dates ], axis=0 ) # (S, T)
    r.variances  = np.stack( [ results[d].variances for d in dates ], axis=0 ) # (S, T)
    r.xstrikes   = np.stack( [ results[d].xstrikes for d in dates ], axis=0 ) # (S, T, N+2)
    r.nms        = np.stack( [ results[d].nms for d in dates ], axis=0 ) # (S, T, N)

    r.mids       = np.stack( [ results[d].mids for d in dates ], axis=0 ) # (S, T, N)
    r.mid_vols   = np.stack( [ results[d].mid_vols for d in dates ], axis=0 ) # (S, T, N)
    r.mid_vega   = np.stack( [ results[d].mid_vega for d in dates ], axis=0 ) # (S, T, N)
    r.mid_delta  = np.stack( [ results[d].mid_delta for d in dates ], axis=0 ) # (S, T, N)
    r.weights    = np.stack( [ results[d].weights for d in dates ], axis=0 ) # (S, T, N)

    r.fits       = np.stack( [ results[d].mids for d in dates ], axis=0 ) # (S, T, N)
    r.fit_vols   = np.stack( [ results[d].mid_vols for d in dates ], axis=0 ) # (S, T, N)
    r.fit_dlvsqs = np.stack( [ results[d].fit_dlvsqs for d in dates ], axis=0 ) # (S, T, N)

    S  = len(dates)
    T, N = grid_config.TN

    assert r.expiries.shape == (S, T), f"Internal error: expected expiries to have shape {(S,T)} but found {r.expiries.shape}."
    assert r.variances.shape == (S, T), f"Internal error: expected variances to have shape {(S,T)} but found {r.variances.shape}."
    assert r.xstrikes.shape == (S, T, N+2), f"Internal error: expected xstrikes to have shape {(S,T,N+2)} but found {r.xstrikes.shape}."    
    assert r.nms.shape == (S, T, N), f"Internal error: expected nms to have shape {(S,T,N)} but found {r.nms.shape}."

    assert r.mids.shape == (S, T, N), f"Internal error: expected mids to have shape {(S,T,N)} but found {r.mids.shape}."
    assert r.mid_vols.shape == (S, T, N), f"Internal error: expected mid_vols to have shape {(S,T,N)} but found {r.mid_vols.shape}."
    assert r.mid_vega.shape == (S, T, N), f"Internal error: expected mid_vega to have shape {(S,T,N)} but found {r.mid_vega.shape}."
    assert r.mid_delta.shape == (S, T, N), f"Internal error: expected mid_delta to have shape {(S,T,N)} but found {r.mid_delta.shape}."
    assert r.weights.shape == (S, T, N), f"Internal error: expected weights to have shape {(S,T,N)} but found {r.weights.shape}."

    assert r.fits.shape == (S, T, N), f"Internal error: expected fits to have shape {(S,T,N)} but found {r.fits.shape}."
    assert r.fit_vols.shape == (S, T, N), f"Internal error: expected fit_vols to have shape {(S,T,N)} but found {r.fit_vols.shape}."
    assert r.fit_dlvsqs.shape == (S, T, N), f"Internal error: expected fit_dlvsqs to have shape {(S,T,N)} but found {r.fit_dlvsqs.shape}."

    verbose.write(f"\rFitted {ticker} for {S} dates from {dates[0]} to {dates[-1]} on NMS grid; this took {tme}.")
    return dict(r)

