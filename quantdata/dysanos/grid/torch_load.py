
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import unique_hash8
from cdxcore.version import version
from cdxcore.bs import bs
from quantdata.dysanos.grid.interpolated import GridConfig, interpolate_grid, interpolate_time_series, Context, OptionsFilter, RawDayFilter, ExpiryFilter, MkArbFree
from quantdata.biz_calendar import DateRange
from quantdata.pool import PoolConfig
from datetime import date
from quantdata.torch import BaseEnvironment, torch
import numpy as np
from collections.abc import Callable

@version("0.1", dependencies=[GridConfig, OptionsFilter, RawDayFilter, ExpiryFilter, MkArbFree, interpolate_time_series])
def load(mode : str, start=date(2025, 1, 1), *, grid_num_strikes : int = 20, num_workers : int = 8, verbose : Context = Context.all ) -> PrettyObject:
    r = PrettyObject()
    r.ticker      = "SPX"
    r.date_range  = DateRange(start=start, end=date(2025, 8, 30))
    r.grid_config = GridConfig(smoothness=0.25, mode=mode, grid_num_strikes=grid_num_strikes)
    r.options_filter = OptionsFilter()
    r.raw_day_filter = RawDayFilter()
    r.expiry_filter = ExpiryFilter()
    r.mk_arb_free = MkArbFree()
    r.unique_id = unique_hash8(r)
    r.unique_str = f"{r.ticker} {r.date_range} {r.grid_config},{r.options_filter},{r.raw_day_filter},{r.expiry_filter},{r.mk_arb_free}"
    r.pool_config = PoolConfig(num_workers=num_workers, threading=False)
    r.ts = PrettyObject( interpolate_time_series(   "SPX", 
                                                  DateRange(start=start, end=date(2025, 8, 30)), 
                                                  options_filter=r.options_filter,
                                                  raw_day_filter=r.raw_day_filter,
                                                  expiry_filter=r.expiry_filter,
                                                  mk_arb_free=r.mk_arb_free,
                                                  grid_config=r.grid_config, 
                                                  pool_config=r.pool_config, 
                                                  verbose=verbose ) )

    return r

@version("0.1", dependencies=[load])
def preprocess( env : BaseEnvironment,
                smoothness : float|None,
                ts : dict, *,
                message_context: str|Callable[[], str] | None = None ) -> dict:
    """
    Preprocessing training data set

    This function returns a full formed training data set to be passed to the model, all on torch.
    The ``env`` environent defines the device, debugging, and the floating point precision.
    """
    def message(with_cols : bool = True) -> str:
        if message_context is None:
            return ""
        if callable(message_context):
            msg = message_context()
        else:
            msg = str(message_context)
        if msg == "":
            return ""
        msg = msg[0].upper() + msg[1:]
        if not with_cols:
            return msg
        return msg + ": "

    assert smoothness is None or smoothness >= 0., f"{message()}Expected smoothness to be None or non-negative, but found {smoothness}"

    expiries: np.ndarray  = ts['expiries']
    mids:np.ndarray       = ts['mids']
    mid_vols: np.ndarray  = ts['mid_vols']
    mid_vega: np.ndarray   = ts['mid_vega']
    mid_delta: np.ndarray  = ts['mid_delta']
    xstrikes: np.ndarray  = ts['xstrikes']
    nms : np.ndarray      = ts['nms']
    variances: np.ndarray = ts['atm_vols']**2 * expiries
    weights: np.ndarray|None = ts.get('weights', None)

    S, T, N  = mids.shape        
    assert xstrikes.shape == (S, T, N+2), f"Expected xstrikes to have shape {(S, T, N+2)}, but found {xstrikes.shape}"
    assert expiries.shape == (S, T,), f"Expected expiries to have shape {(S, T)}, but found {expiries.shape}"
    assert variances.shape == (S, T), f"Expected variances to have shape {(S, T)}, but found {variances.shape}"
    assert nms.shape == (S, T, N), f"Expected nms to have shape {(S, T, N)}, but found {nms.shape}"
    assert mid_vols.shape == (S, T, N), f"Expected mid_vols to have shape {(S, T, N)}, but found {mid_vols.shape}"
    assert mid_vega.shape == (S, T, N), f"Expected mid_vega to have shape {(S, T, N)}, but found {mid_vega.shape}"
    assert mid_delta.shape == (S, T, N), f"Expected mid_delta to have shape {(S, T, N)}, but found {mid_delta.shape}"
    assert weights is None or weights.shape == mids.shape, lambda : f"{message()}Expected weights to hshape {(S, T, N)}, but found {weights.shape}"

    assert np.min(xstrikes) > 0., f"{message()}All 'xstrikes' must be positive, but found min {np.min(xstrikes)}"
    assert np.min( np.diff(xstrikes, axis=-1) ) > 0., f"{message()}All 'xstrikes' must be increasing, but found min diff {np.min( np.diff(xstrikes, axis=-1) )}"
    assert np.min(expiries) > 0., f"{message()}All 'expiries' must be positive, but found min {np.min(expiries)}"
    assert np.min( np.diff(expiries, axis=-1) ) > 0, f"{message()}All 'expiries' must be increasing, but found min diff {np.min( np.diff(expiries, axis=-1) )}"
    assert np.min(variances) >= 0., f"{message()}All 'variances' must be non-negative, but found min {np.min(variances)}"
    assert np.min( np.diff(variances, axis=-1) ) > 0, f"{message()}All 'variances' must be increasing, but found min diff {np.min( np.diff(variances, axis=-1) )}"
    assert weights is None or np.min(np.sum(weights, axis=-1)) > 0., f"{message()}At least one weight must be positive for each expiry, but found min sum {np.min(np.sum(weights, axis=-1))}"  

    r = PrettyObject()

    # fit
    # compute transition 1
    # --------------------
    dt          = np.diff( expiries, prepend=0. )[:,:,None] 
    box         = xstrikes[:,:,1:-1]**2 * dt / ( xstrikes[:,:,2:] - xstrikes[:,:,:-2])
    w_plus      = box / ( xstrikes[:,:,2:] - xstrikes[:,:,1:-1] )
    w_minus     = box / ( xstrikes[:,:,1:-1] - xstrikes[:,:,:-2] )
    assert np.min( w_plus ) >= 0., f"{message()}computed 'w_plus' must be non-negative, but found min {np.min(w_plus)}"
    assert np.min( w_minus ) >= 0., f"{message()}computed 'w_minus' must be non-negative, but found min {np.min(w_minus)}"

    r.iw_minus = torch.tensor( w_minus, device=env.device, dtype=env.fdtype )
    r.iw_plus = torch.tensor( w_plus, device=env.device, dtype=env.fdtype )
    del w_minus, w_plus, box, dt

    # L transition
    # ------------
    L = np.maximum( 0., xstrikes[:,:-1,None,:] - xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
    L = np.diff( np.diff( L, axis=2, prepend=1. ) / np.diff( xstrikes[:,1:,:,None], axis=2, prepend=0. ), axis=2, append=0. )
    r.L = torch.tensor( L, device=env.device, dtype=env.fdtype )
    assert r.L.shape == (S, T-1, N+2, N+2), f"Expected L to have shape {(S, T-1, N+2, N+2)}, but found {r.L.shape}"
    del L

    # objective
    # ---------
    def np_call( spots, strikes, variances, expiries ):                
        intr = np.maximum( 0., spots - strikes )
        if smoothness is None or smoothness == 0.:
            return intr
        vol     = np.sqrt( smoothness * variances/expiries )
        assert np.all( np.isfinite(vol) ), f"Internal error: {message()}infinite vol"
        return bs.price( k=strikes/spots, vol=vol, sqrtT=np.sqrt(expiries), eps=1E-6 )*spots

    r.CX       = torch.tensor( np_call( spots=xstrikes[:,:,None,:], strikes=xstrikes[:,:,1:-1,None], variances=variances[:,:,None,None], expiries=expiries[:,:,None,None] ), device=env.device, dtype=env.fdtype )
    assert r.CX.shape == (S, T, N, N+2), f"Expected CX to have shape {(S, T, N, N+2)}, but found {r.CX.shape}"

    # rest
    r.xdensity0 = torch.tensor( np.diff( np.diff( np.maximum( 0., 1. - xstrikes[:,0,:] ), prepend=1. ) / np.diff(xstrikes[:,0,:], prepend=0.), append=0. ), device=env.device, dtype=env.fdtype )
    r.weights   = torch.tensor( weights, device=env.device, dtype=env.fdtype ) if not weights is None else None
    r.mids      = torch.tensor( mids, device=env.device, dtype=env.fdtype ) 
    r.mid_vols  = torch.tensor( mid_vols, device=env.device, dtype=env.fdtype )
    r.mid_vega  = torch.tensor( mid_vega, device=env.device, dtype=env.fdtype )
    r.mid_delta = torch.tensor( mid_delta, device=env.device, dtype=env.fdtype )
    r.expiries  = torch.tensor( expiries, device=env.device, dtype=env.fdtype )
    r.xstrikes  = torch.tensor( xstrikes, device=env.device, dtype=env.fdtype )
    r.nms       = torch.tensor( nms, device=env.device, dtype=env.fdtype )

    for k, v in r.items():
        if not v is None:
            assert torch.all( torch.isfinite(v) ), f"{message()}Found non-finite value in {k}"
            assert v.device == r.nms.device, f"{message()}Expected {k} to be on device {r.nms.device}, but found {v.device}"

    return dict(r)