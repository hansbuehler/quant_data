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

@version("0.1")
@dataclass(frozen=True)
class SanosCompressorConfig(SanosModelConfig):
    learn_config : FeedForward.Config|None = None # encoder or None for direct state parameterization

    def __post_init__(self):
        super().__post_init__()

    def __str__(self) -> str:
        return f"{self.label_str} {self.obj_str} {self.nms_str} {self.minmax_str} {self.state_str}" + ( f",lrn={self.learn_config}" if not self.learn_config is None else "" )
    
    def create(self, env : BaseEnvironment, *, S, T, N) -> SanosCompressor:
        """ Create Model """
        return SanosCompressor( env, S=S, T=T, N=N, config=self )

@version("0.1")
class SanosCompressor(SanosModel):
    
    Config = SanosCompressorConfig

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosCompressorConfig ):
        super().__init__(env,T=T,config=config)

        self.config = config

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

    def compute_state( self, data : Pytree, *, debug_sample_indices : np.array|None = None, training : bool = False ) -> tuple[torch.Tensor, dict|None]:
        """ Compute DLVsq grid """
        S, T, N          = data['mid_vols'].shape

        # compute DLVs
        # ------------

        if not self.learn_state is None:
            # learn map from features to state directly
            features  = self.build_default_features(data, training=training)
            self.verify( lambda : torch.all( torch.isfinite(features) ), "Infinite values found in features" )
            state     = self.learn_state( features )
            features  = None
        else:
            state     = self.dlv_state
            state     = state if debug_sample_indices is None else state[debug_sample_indices,:]
        assert state.shape == (S, self.config.num_state), f"Expected state to have shape {(S, self.config.num_state)}, but found {state.shape}"  
        return state, {}
