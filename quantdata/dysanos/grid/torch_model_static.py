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
from quantdata.dysanos.grid.torch_model import SanosModelConfig, SanosModel
from quantdata.dysanos.grid.torch_market import GridMarketDefinition
import logging

_bbtorch = BackboneTorch()

@version("0.1", dependencies=[SanosModelConfig, GridMarketDefinition])
@dataclass(frozen=True)
class SanosCompressorConfig(SanosModelConfig):
    min_dlv: float = 0.01              # Min DLV
    max_dlv: float = 2.                # Max DLV

    num_state : int = 10               # states
    qr_state  : bool = False 
    state_config : FeedForward.Config = FeedForward.Config() # decoder
    learn_config : FeedForward.Config|None = None # encoder or None for direct state parameterization

    def __post_init__(self):
        super().__post_init__()
        verify_inp( self.min_dlv >= 0., f"Expected min_dlv to be non-negative, but found {self.min_dlv}" )
        verify_inp( self.min_dlv < self.max_dlv, f"Expected min_dlv to be less than max_dlv, but found min_dlv={self.min_dlv} and max_dlv={self.max_dlv}" )
        verify_inp( self.num_state > 0, f"Expected num_state to be positive, but found {self.num_state}" )

    def __str__(self) -> str:        
        s = f"{self.label_str}@{self.num_state} midlv={self.min_dlv:.2f},mxdlv={self.max_dlv:.2f},vsq={self.min_vega_sqrt:.3g} st={self.state_config}"
        if not self.learn_config is None:
            s = s + f",lrn={self.learn_config}"
        return s   
    
    def create(self, env : BaseEnvironment, *, S, T, N) -> SanosCompressor:
        """ Create Model """
        return SanosCompressor( env, S=S, T=T, N=N, config=self )
    
class SanosCompressor(SanosModel):
    
    Config = SanosCompressorConfig
    NUM_FEATURES = 6

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosCompressorConfig ):
        super().__init__(env)

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

    def compute_dlvsqs( self, data : Pytree, sample_indices : np.array|None = None, training : bool = True ) -> torch.Tensor:
        """ Compute DLVsq grid """
        data      = dict(data)
        mid_vols  = data.pop("mid_vols")
        S, T, N   = mid_vols.shape
        xstrikes  = pop(data, "xstrikes", (S, T, N+2) )
        expiries  = pop(data, "expiries", (S, T) )
        nms       = pop(data, "nms", (S, T, N) )
        mid_vega  = pop(data, "mid_vega", (S, T, N) )
        mid_delta = pop(data, "mid_delta", (S, T, N) )

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

        dlvs_raw  = self.raw_dlv_from_state( state )
        assert dlvs_raw.shape == (S, T*N), f"Expected raw DLVs to have shape {(S, T*N)}, but found {dlvs_raw.shape}"
        dlvs_raw  = dlvs_raw.reshape( S, T, N )
        min_dlvsq = self.config.min_dlv**2
        max_dlvsq = self.config.max_dlv**2
        dlvsqs    = torch.sigmoid(dlvs_raw) * (max_dlvsq - min_dlvsq) + min_dlvsq

        if training:
            return dlvsqs, None
        
        return dlvsqs, dict(state=state.detach())

