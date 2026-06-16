"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import warnings as warnings

from flax.nnx import Pytree
import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from dataclasses import dataclass, field
from cdxcore.err import error, verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
import scipy
from sympy import true
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import Backbone, BackboneNumpy
import math

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, get, pop
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch

_bbtorch = BackboneTorch()

@version("0.1")
@dataclass(frozen=True)
class SanosCompressorConfig:
    smoothness : float|None = None     # Model smoothness

    min_dlv: float = 0.01              # Min DLV
    max_dlv: float = 2.                # Max DLV
    min_vega_sqrt : float = 1E-4       # minimum vega/sqrtT

    num_state : int = 10
    state_config : FeedForward.Config = FeedForward.Config()
    learn_config : FeedForward.Config|None = None

    def __str__(self) -> str:
        str_smooth = f"{self.smoothness:.2f}-smooth" if not self.smoothness is None else "linear"
        s = f"{str_smooth}@{self.num_state} midlv={self.min_dlv:.2f},mxdlv={self.max_dlv:.2f},vsq={self.min_vega_sqrt:3g} st={self.state_config}"
        if not self.learn_config is None:
            s = s + f",lrn={self.learn_config}"
        return s   

    @version("0.1")
    def preprocess(self, 
                 ts : dict, *,
                 tdevice : torch.device|str|None = None,
                 message_context: str|callable[[], str] | None = None ) -> dict:
        """
        Preprocessing training data set
        This function returns a full formed training data set to be passed to the model.
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

        r.iw_minus = torch.tensor( w_minus, device=tdevice )
        r.iw_plus = torch.tensor( w_plus, device=tdevice )
        del w_minus, w_plus, box, dt

        # L transition
        # ------------
        L = np.maximum( 0., xstrikes[:,:-1,None,:] - xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
        L = np.diff( np.diff( L, axis=2, prepend=1. ) / np.diff( xstrikes[:,1:,:,None], axis=2, prepend=0. ), axis=2, append=0. )
        r.L = torch.tensor( L, device=tdevice )
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

        r.CX       = torch.tensor( np_call( spots=xstrikes[:,:,None,:], strikes=xstrikes[:,:,1:-1,None], variances=variances[:,:,None,None], expiries=expiries[:,:,None,None] ), device=tdevice )
        assert r.CX.shape == (S, T, N, N+2), f"Expected CX to have shape {(S, T, N, N+2)}, but found {r.CX.shape}"

        # rest
        r.xdensity0 = torch.tensor( np.diff( np.diff( np.maximum( 0., 1. - xstrikes[:,0,:] ), prepend=1. ) / np.diff(xstrikes[:,0,:], prepend=0.), append=0. ), device=tdevice )
        r.weights   = torch.tensor( weights, device=tdevice ) if not weights is None else None
        r.mids      = torch.tensor( mids, device=tdevice ) 
        r.mid_vols  = torch.tensor( mid_vols, device=tdevice )
        r.mid_vega  = torch.tensor( mid_vega, device=tdevice )
        r.mid_delta = torch.tensor( mid_delta, device=tdevice )
        r.expiries  = torch.tensor( expiries, device=tdevice )
        r.xstrikes  = torch.tensor( xstrikes, device=tdevice )
        r.nms       = torch.tensor( nms, device=tdevice )

        for k, v in r.items():
            if not v is None:
                assert torch.all( torch.isfinite(v) ), f"{message()}Found non-finite value in {k}"
                assert v.device == r.nms.device, f"{message()}Expected {k} to be on device {r.nms.device}, but found {v.device}"

        return dict(r)

class SanosCompressor(BaseModel):
    
    NUM_FEATURES = 6

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosCompressorConfig = SanosCompressorConfig() ):
        super().__init__(env)

        self.config = config

        # define a dlv parameter per sample
        # or a map from features to state
        if config.learn_config is not None:
            num_input  = T*N*self.NUM_FEATURES
            num_output = config.num_state
            self.learn_state = FeedForward( env, num_input=num_input, num_output=num_output, config=config.learn_config )
            self.dlv_state   = None

        else:
            self.learn_state = None
            self.dlv_state   = torch.nn.Parameter(torch.randn(S, config.num_state, device=env.device), requires_grad=True)

        # define the map state (S, X) -> raw_dlv (S, T, N)
        self.raw_dlv_from_state = FeedForward( env, num_input=config.num_state, num_output=T * N, config=config.state_config )  

    def __call__(self, data : Pytree, training : bool = True ) -> float | Pytree:

        data     = dict(data)
        mids     = data.pop("mids")
        S, T, N  = mids.shape

        min_dlvsq = self.config.min_dlv**2
        max_dlvsq = self.config.max_dlv**2

        xstrikes  = pop(data, "xstrikes", (S, T, N+2) )
        expiries  = pop(data, "expiries", (S, T) )
        nms       = pop(data, "nms", (S, T, N) )
        mid_vols  = pop(data, "mid_vols", (S, T, N) )
        xdensity0 = pop(data, "xdensity0", (S, N+2))
        mid_vega  = pop(data, "mid_vega", (S, T, N) )
        mid_delta = pop(data, "mid_delta", (S, T, N) )

        iw_minus  = pop(data, "iw_minus", (S, T, N) )
        iw_plus   = pop(data, "iw_plus", (S, T, N) )
        L         = pop(data, "L", (S, T-1, N+2, N+2) )
        CX        = pop(data, "CX", (S, T, N, N+2) )
        weights   = pop(data, "weights", (S, T, N), optional=True )

        # compute DLVs
        # ------------

        if not self.learn_state is None:
            logK      = torch.log( xstrikes[:,:,1:-1] )
            features  = torch.stack( [ logK, expiries[:,:,None], mid_vols, nms, mid_vega, mid_delta ], axis=-1 )
            assert features.shape == (S, T, N, self.NUM_FEATURES), f"Expected features to have shape {(S, T, N, self.NUM_FEATURES)}, but found {features.shape}"
            features  = features.reshape( S, -1 )
            state     = self.learn_state( features )
            features  = None
        else:
            state     = self.dlv_state
        assert state.shape == (S, self.config.num_state), f"Expected state to have shape {(S, self.config.num_state)}, but found {state.shape}"  

        dlvs_raw = self.raw_dlv_from_state( state )
        assert dlvs_raw.shape == (S, T*N), f"Expected raw DLVs to have shape {(S, T*N)}, but found {dlvs_raw.shape}"
        dlvs_raw = dlvs_raw.reshape( S, T, N )
        dlvsqs   = torch.sigmoid(dlvs_raw) * (max_dlvsq - min_dlvsq) + min_dlvsq

        # solve for the densities
        # -----------------------

        q_tm1        = xdensity0
        error        = 0.
        wsum         = 0.

        xdensities   = [] if not training else None
        calls        = [] if not training else None
        dlvsqs_out   = [] if not training else None
        error_ts     = [] if not training else None
        wsum_ts      = [] if not training else None

        for t in range(T):
            iw_minus_t  = iw_minus[:,t,:]
            iw_plus_t   = iw_plus[:,t,:]
            CX_t        = CX[:,t,:,:]
            mids_t      = mids[:,t,:]
            vegaSqrt    = mid_vega[:,t,:] / torch.sqrt( expiries[:,t][:,None] ) 
            weights_t   = weights[:,t,:] if not weights is None else torch.where( vegaSqrt<self.config.min_vega_sqrt, 0., 1./torch.clip( vegaSqrt, min=self.config.min_vega_sqrt))

            # move density forward
            if t>0:
                L_t     = L[:,t-1,:,:] 
                q_tm1   = torch.sum( L_t * q_tm1[:,None,:], dim=-1 ) # TODO: make faster

            self.verify( lambda : torch.min( q_tm1 ) >= -1E-8, lambda : f"Negative density values found at expiry #{t}, before solving DLV equation: min density value is {np.min(q_tm1)}.")

            # solve DLV equation
            dlvsqs_t    = dlvsqs[:,t,:]
            w_minus_t   = dlvsqs_t * iw_minus_t
            w_plus_t    = dlvsqs_t * iw_plus_t
            q_t         = solve_sanos_tridiagonal( w_minus=w_minus_t, w_plus=w_plus_t, rhs=q_tm1, backbone=_bbtorch )
            assert q_t.shape == xstrikes[:,t,:].shape, f"Expiry #{t}: shape error in DLV equation solution, expected {xstrikes[:,t,:].shape} but found {q_t.shape}"

            # compute market call prices
            C_t          = torch.sum( CX_t * q_t[:,None,:], dim=-1 ) # TODO: make faster
            assert C_t.shape == (S,N), f"Expiry #{t}: expected shape {(S,N)} for model call prices C_t, but found {C_t.shape}"

            # add errors
            error_t       = weights_t * ( C_t - mids_t ) 
            error_t       = torch.where( torch.abs(error_t) <= 1., error_t**2, 2.*torch.abs(error_t) - 1. )
            error         = error + torch.sum( error_t )
            wsum          = wsum + torch.sum( weights_t )

            if not training:
                xdensities.append( q_t )
                calls.append( C_t )
                dlvsqs_out.append( dlvsqs_t )
                error_ts.append( error_t )
                wsum_ts.append( weights_t )
            q_tm1 = q_t 

        error = error / wsum
        if training:
            return error
        with torch.no_grad():
            xdensities = torch.stack( xdensities, axis=1 ) # (S, T, N+2)
            calls = torch.stack( calls, axis=1 ) # (S, T, N)
            dlvsqs_out = torch.stack( dlvsqs_out, axis=1 ) # (S, T, N)
            error_ts = torch.stack( error_ts, axis=1 ) # (S, T, N)
            wsum_ts = torch.stack( wsum_ts, axis=1 ) # (S, T, N)   

        assert xdensities.shape == (S, T, N+2), f"Expected xdensities to have shape {(S, T, N+2)}, but found {xdensities.shape}"
        assert calls.shape == (S, T, N), f"Expected calls to have shape {(S, T, N)}, but found {calls.shape}"
        assert dlvsqs_out.shape == (S, T, N), f"Expected dlvsqs_out to have shape {(S, T, N)}, but found {dlvsqs_out.shape}"
        assert error_ts.shape == (S, T, N), f"Expected error_ts to have shape {(S, T, N)}, but found {error_ts.shape}"
        assert wsum_ts.shape == (S, T, N), f"Expected wsum_ts to have shape {(S, T, N)}, but found {wsum_ts.shape}"
        return dict(loss=error, xdensities=xdensities, calls=calls, dlvsqs=dlvsqs_out, errors=error_ts, weights=wsum_ts)



