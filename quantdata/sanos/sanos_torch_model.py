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

from quantdata.sanos.sanos_tools import SanosPricer, SanosLPSolver, make_variances, warn_and_fix_density
from quantdata.sanos.sanos import SanosFitter
from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.torch import BaseEnvironment, BaseModel, torch, get, pop
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch

_bbtorch = BackboneTorch()

def sanos_preprocessor( 
                 smoothness : float|None,
                 expiries: np.ndarray, 
                 mids: np.ndarray, 
                 mid_vols: np.ndarray,
                 xstrikes: np.ndarray, *,
                 nms : np.ndarray,
                 variances: np.ndarray,
                 weights: np.ndarray|None = None,
                 tdevice : torch.device|str|None = None,
                 message_context: str|callable[[], str] | None = None ):
    """
    Preprocessing training data set
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

    S, T, N  = mids.shape
    assert xstrikes.shape == (S, T, N+2), f"Expected xstrikes to have shape {(S, T, N+2)}, but found {xstrikes.shape}"
    assert expiries.shape == (S, T,), f"Expected expiries to have shape {(S, T)}, but found {expiries.shape}"
    assert variances.shape == (S, T), f"Expected variances to have shape {(S, T)}, but found {variances.shape}"
    assert nms.shape == (S, N), f"Expected nms to have shape {(S, N)}, but found {nms.shape}"
    assert mid_vols.shape == (S, T, N), f"Expected mid_vols to have shape {(S, T, N)}, but found {mid_vols.shape}"
    assert weights is None or weights.shape == mids.shape, lambda : f"{message()}Expected weights to hshape {(S, T, N)}, but found {weights.shape}"

    assert np.min(xstrikes) > 0., f"{message()}All 'xstrikes' must be positive, but found min {np.min(xstrikes)}"
    assert np.min( np.diff(xstrikes, axis=-1) ) > 0., f"{message()}All 'xstrikes' must be increasing, but found min diff {np.min( np.diff(xstrikes, axis=-1) )}"
    assert np.min(expiries) > 0., f"{message()}All 'expiries' must be positive, but found min {np.min(expiries)}"
    assert np.min( np.diff(expiries, axis=-1) ) > 0, f"{message()}All 'expiries' must be increasing, but found min diff {np.min( np.diff(expiries, axis=-1) )}"
    assert np.min(variances) >= 0., f"{message()}All 'variances' must be non-negative, but found min {np.min(variances)}"
    assert np.min( np.diff(variances, axis=-1) ) > 0, f"{message()}All 'variances' must be increasing, but found min diff {np.min( np.diff(variances, axis=-1) )}"
    assert weights is None or np.min(np.sum(weights, axis=-1)) > 0., f"{message()}At least one weight must be positive for each expiry, but found min sum {np.min(np.sum(weights, axis=-1))}"  

    self = PrettyObject()

    # fit
    # compute transition 1
    # --------------------
    dt          = np.diff( expiries, prepend=0. )[:,:,None] 
    box         = xstrikes[:,:,1:-1]**2 * dt / ( xstrikes[:,:,2:] - xstrikes[:,:,:-2])
    w_plus      = box / ( xstrikes[:,:,2:] - xstrikes[:,:,1:-1] )
    w_minus     = box / ( xstrikes[:,:,1:-1] - xstrikes[:,:,:-2] )
    assert np.min( w_plus ) >= 0., f"{message()}computed 'w_plus' must be non-negative, but found min {np.min(w_plus)}"
    assert np.min( w_minus ) >= 0., f"{message()}computed 'w_minus' must be non-negative, but found min {np.min(w_minus)}"

    self.iw_minus = torch.tensor( w_minus, device=tdevice )
    self.iw_plus = torch.tensor( w_plus, device=tdevice )
    del w_minus, w_plus, box, dt

    # L transition
    # ------------
    L = np.maximum( 0., xstrikes[:,:-1,None,:] - xstrikes[:,1:,:,None] ) # will be multiplied from the right with q_tm1
    L = np.diff( np.diff( L, axis=2, prepend=1. ) / np.diff( xstrikes[:,1:,:,None], axis=2, prepend=0. ), axis=2, append=0. )
    self.L = torch.tensor( L, device=tdevice )
    assert self.L.shape == (S, T-1, N+2, N+2), f"Expected L to have shape {(S, T-1, N+2, N+2)}, but found {self.L.shape}"
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

    self.CX       = torch.tensor( np_call( spots=xstrikes[:,:,None,:], strikes=xstrikes[:,:,1:-1,None], variances=variances[:,:,None,None], expiries=expiries[:,:,None,None] ), device=tdevice )
    assert self.CX.shape == (S, T, N, N+2), f"Expected CX to have shape {(S, T, N, N+2)}, but found {self.CX.shape}"

    # rest
    self.xdensity0 = torch.tensor( np.diff( np.diff( np.maximum( 0., 1. - xstrikes[:,0,:] ), prepend=1. ) / np.diff(xstrikes[:,0,:], prepend=0.), append=0. ), device=tdevice )
    self.weights   = torch.tensor( weights, device=tdevice ) if not weights is None else None
    self.mids      = torch.tensor( mids, device=tdevice ) 
    self.mid_vols  = torch.tensor( mid_vols, device=tdevice )
    self.expiries  = torch.tensor( expiries, device=tdevice )
    self.xstrikes  = torch.tensor( xstrikes, device=tdevice )
    self.nms       = torch.tensor( nms, device=tdevice )
    return dict(self)

@dataclass(frozen=True)
class SanosCompressorConfig:
    min_dlv: float = 0.01
    max_dlv: float = 2.

    num_state : int = 10
    state_config : FeedForward.Config = FeedForward.Config()
    learn_config : FeedForward.Config|None = None

class SanosCompressor(BaseModel):
    
    NUM_FEATURES = 4

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
            self.dlv_state   = torch.nn.Parameter(torch.randn(S, config.num_state) )

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
        nms       = pop(data, "nms", (S, N) )
        mid_vols  = pop(data, "mid_vols", (S, T, N) )
        xdensity0 = pop(data, "xdensity0", (S, N+2))

        iw_minus  = pop(data, "iw_minus", (S, T, N) )
        iw_plus   = pop(data, "iw_plus", (S, T, N) )
        L         = pop(data, "L", (S, T-1, N+2, N+2) )
        CX        = pop(data, "CX", (S, T, N, N+2) )
        weights   = pop(data, "weights", (S, T, N), optional=True )

        # compute DLVs
        # ------------

        if not self.learn_state is None:
            logK      = torch.log( xstrikes[:,:,1:-1] )
            features  = torch.stack( [ logK, expiries[:,:,None], mid_vols, nms[:,None,:] ], axis=-1 )
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

        for t in range(T):
            iw_minus_t  = iw_minus[:,t,:]
            iw_plus_t   = iw_plus[:,t,:]
            CX_t        = CX[:,t,:,:]
            mids_t      = mids[:,t,:]
            weights_t   = weights[:,t,:] if not weights is None else None

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
            error_t       = weights_t * ( C_t - mids_t ) if not weights is None else ( C_t - mids_t )
            error_t       = torch.where( torch.abs(error_t) <= 1., error_t**2, 2.*torch.abs(error_t) - 1. )
            error         = error + torch.sum( error_t )
            wsum          = wsum + torch.sum( weights_t ) if not weights is None else wsum + len(error_t)

            if not training:
                xdensities.append( q_t )
                calls.append( C_t )
                dlvsqs_out.append( dlvsqs_t )
            q_tm1 = q_t 

        error = error / wsum
        if training:
            return error
        with torch.no_grad():
            xdensities = torch.stack( xdensities, axis=1 ) # (S, T, N+2)
            calls = torch.stack( calls, axis=1 ) # (S, T, N)
            dlvsqs_out = torch.stack( dlvsqs_out, axis=1 ) # (S, T, N)
        assert xdensities.shape == (S, T, N+2), f"Expected xdensities to have shape {(S, T, N+2)}, but found {xdensities.shape}"
        assert calls.shape == (S, T, N), f"Expected calls to have shape {(S, T, N)}, but found {calls.shape}"
        assert dlvsqs_out.shape == (S, T, N), f"Expected dlvsqs_out to have shape {(S, T, N)}, but found {dlvsqs_out.shape}"
        return error, dict(xdensities=xdensities, calls=calls, dlvsqs=dlvsqs_out)



