"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

import warnings as warnings

import numpy as np
from dataclasses import dataclass, field
from cdxcore.err import error, verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from sympy import true
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import Backbone, BackboneNumpy

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, get, pop
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.dysanos.grid.torch_market import GridMarketDefinition, PoolConfig
from dataclasses import asdict, dataclass, field

_bbtorch = BackboneTorch()

@version("0.1", dependencies=[GridMarketDefinition, BaseEnvironment])
@dataclass(frozen=True)
class SanosModelConfig:
    label : str = "SanosModel"        # Name of the cache file
    user_version : str = "0.1"        # Version to trigger updates
    smoothness : float|None = None    # Model smoothness

    def __post_init__(self):
        verify_inp( self.smoothness is None or self.smoothness >= 0., f"Expected smoothness to be None or non-negative, but found {self.smoothness}" )

    def __str__(self) -> str:
        return f"{self.label} {self.user_version} " + ( f"{self.smoothness:.2f}-smooth" if not self.smoothness is None else "linear" )

    def create(self, env : BaseEnvironment, *, S, T, N) -> type:
        """ Create Model """
        raise NotImplementedError("This method should be implemented in subclasses to create a specific model instance.")

    def torch_data(self, 
                   env : BaseEnvironment,
                   market : PrettyObject, *,
                   pool_config : PoolConfig = PoolConfig.off(),
                   message_context: str|callable[[], str] | None = None,
                   verbose : Context = Context.all ) -> dict:
        """
        Preprocessing training data set.

        This function returns a full formed training data set to be passed to the model, all on torch.
        The ``env`` environent defines the device, debugging, and the floating point precision.

        Parameters
        ----------
        env : :class:`BaseEnvironment`  
            The torch environment to use for the model.
        market : :class:`PrettyObject`
            The market definition or market data to use for the model. Contains with ``grid_market_def`` an attribute of type :class:`GridMarketDefinition`.
        pool_config : :class:`PoolConfig`, default :meth:`PoolConfig.off`.
            The pool configuration to use for loading the market data. Default is :meth:`PoolConfig.off`.
        message_context : str|callable[[], str]|None, default ``None``
            The message context to use for error messages. If None, no additional message context will be used. 
        verbose : :class:`Context`, default :attr:`Context.all`
            The verbosity context to use for logging.
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

        assert self.smoothness is None or self.smoothness >= 0., f"{message()}Expected smoothness to be None or non-negative, but found {self.smoothness}"

        market = market.load( pool_config=pool_config, verbose=verbose ) if isinstance(market, GridMarketDefinition) else market 
        assert hasattr(market, 'grid_market_def'), f"{message()}Expected market to have attribute 'grid_market_def', but found {fmt_list(sorted(market))}"

        expiries: np.ndarray     = market['expiries']
        mids:np.ndarray          = market['mids']
        mid_vols: np.ndarray     = market['mid_vols']
        mid_vega: np.ndarray     = market['mid_vega']
        mid_delta: np.ndarray    = market['mid_delta']
        xstrikes: np.ndarray     = market['xstrikes']
        nms : np.ndarray         = market['nms']
        variances: np.ndarray    = market['atm_vols']**2 * expiries
        weights: np.ndarray|None = market.get('weights', None)

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
            if self.smoothness is None or self.smoothness == 0.:
                return intr
            vol     = np.sqrt( self.smoothness * variances/expiries )
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