"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from quantdata.dysanos.grid.torch_load import preprocess
import logging

_bbtorch = BackboneTorch()

@version("0.1", dependencies=[preprocess])
@dataclass(frozen=True)
class SanosCompressorConfig:
    label : str = "SanosCompressor"
    user_version : str = "0.1"
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
                   env : BaseEnvironment,
                   ts : dict, *,
                   message_context: str|callable[[], str] | None = None ) -> dict:
        """
        Preprocess training data set
        """
        return preprocess( env=env, smoothness=self.smoothness, ts=ts, message_context=message_context )   
    
class SanosCompressor(BaseModel):
    
    NUM_FEATURES = 6

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosCompressorConfig = SanosCompressorConfig() ):
        super().__init__(env)

        torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(
            recompiles=True,
            graph_breaks=True,
            dynamo=logging.WARNING,
        )

        self.config = config
        np.random.seed(123123123)

        # define a dlv parameter per sample
        # or a map from features to state
        if config.learn_config is not None:
            self.learn_state = FeedForward( env, num_input=T*N*self.NUM_FEATURES, num_output=config.num_state, config=config.learn_config )
            self.dlv_state   = None

        else:
            self.learn_state = None
            state0           = np.random.randn(1, config.num_state)
            state0           = np.full( (S, config.num_state), state0, dtype=np.float32 )
            self.dlv_state   = torch.nn.Parameter(torch.tensor(state0, device=env.device, dtype=env.fdtype), requires_grad=True)

        # define the map state (S, X) -> raw_dlv (S, T, N)
        self.raw_dlv_from_state = FeedForward( env, num_input=config.num_state, num_output=T * N, config=config.state_config )  

    def __call__(self, data : Pytree, sample_indices : np.array|None = None, training : bool = True ) -> float | Pytree:

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
            # learn map from features to state directly
            logK      = torch.log( xstrikes[:,:,1:-1] )
            features  = torch.stack( [ logK, logK*0.+expiries[:,:,None], mid_vols, nms, mid_vega, mid_delta ], axis=-1 )
            assert features.shape == (S, T, N, self.NUM_FEATURES), f"Expected features to have shape {(S, T, N, self.NUM_FEATURES)}, but found {features.shape}"
            features  = features.reshape( S, -1 )
            state     = self.learn_state( features )
            features  = None
        else:
            state     = self.dlv_state
            state     = state if sample_indices is None else state[sample_indices,:]
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
                xdensities.append( torch.detach(q_t) )
                calls.append( torch.detach(C_t) )
                dlvsqs_out.append( torch.detach(dlvsqs_t) )
                error_ts.append( torch.detach(error_t) )
                wsum_ts.append( torch.detach(weights_t) )
            q_tm1 = q_t 

        error = error / wsum
        if training:
            return error
        with torch.no_grad():
            state = torch.detach(state)
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
        assert state.shape == (S, self.config.num_state), f"Expected state to have shape {(S, self.config.num_state)}, but found {state.shape}"
        return dict(loss=error, xdensities=xdensities, calls=calls, dlvsqs=dlvsqs_out, errors=error_ts, weights=wsum_ts, state=state)



