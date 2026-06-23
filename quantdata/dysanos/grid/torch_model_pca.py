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
class SanosPCACompressorConfig(SanosModelConfig):
    num_pca   : int = 5                # PCA of the state
    learn_config : FeedForward.Config|None = None # encoder or None for direct state parameterization
    use_qr : bool = True               # whether to orthogonalize the PCA basis using QR decomposition

    def __post_init__(self):
        super().__post_init__()
        verify_inp( self.num_pca > 0 and self.num_pca <= self.num_state, f"Expected num_pca to be positive and less than or equal to num_state, but found {self.num_pca}" )
        
    def __str__(self) -> str:
        return f"{self.label_str}{' qr' if self.use_qr else ''} {self.obj_str} {self.nms_str} {self.minmax_str} pca={self.num_pca} {self.state_str}" + ( f",lrn={self.learn_config}" if not self.learn_config is None else "" )
        
    def create(self, env : BaseEnvironment, *, S, T, N) -> SanosPCACompressor:
        """ Create Model """
        return SanosPCACompressor( env, S=S, T=T, N=N, config=self )

@version("0.1")
class SanosPCACompressor(SanosModel):
    
    Config = SanosPCACompressorConfig

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosPCACompressorConfig ):
        super().__init__(env,T=T,config=config)

        self.config = config
        np.random.seed(123123123)
        n           = config.num_nms

        # h_t = u_0 + \sum_{s<=t} \alpha_s du_s
        self.u_du  = torch.nn.Parameter(torch.tensor(np.random.randn(1+config.num_pca, config.num_state), device=env.device, dtype=env.fdtype), requires_grad=True)

        # define the map state (S, X) -> raw_dlv (S, T, N)

        if config.learn_config is not None:
            self.learn_alpha = FeedForward( env, num_input=T*N*self.NUM_FEATURES, num_output=config.num_pca, config=config.learn_config )
            self.dlv_alpha   = None

        else:
            self.learn_alpha = None
            alpha0           = np.random.randn(1, config.num_pca)
            alpha0           = np.full( (S, config.num_pca), alpha0, dtype=np.float32 )
            self.dlv_alpha   = torch.nn.Parameter(torch.tensor(alpha0, device=env.device, dtype=env.fdtype), requires_grad=True)

    def compute_state( self, data : Pytree, *, debug_sample_indices : np.array|None = None, training : bool = False ) -> tuple[torch.Tensor, dict|None]:
        """ Compute DLVsq grid """
        S, T, N          = data["mid_vols"].shape

        # compute alphas
        # --------------

        if not self.learn_alpha is None:
            # learn map from features to alpha directly
            features  = self.build_default_features(data, training=training)
            alpha     = self.learn_alpha( features )
            features  = None
            logK      = None
        else:
            alpha     = self.dlv_alpha
            alpha     = alpha if debug_sample_indices is None else alpha[debug_sample_indices,:]
        assert alpha.shape == (S, self.config.num_pca), f"Expected alpha to have shape {(S, self.config.num_pca)}, but found {alpha.shape}"  

        # compute h, dh
        # -------------

        assert self.u_du.shape == (1+self.config.num_pca, self.config.num_state), f"Expected h_dh to have shape {(1+self.config.num_pca, self.config.num_state)}, but found {self.u_du.shape}"
        u        = self.u_du[0,:][None,:]
        du       = self.u_du[1:,:]
        if self.config.use_qr:
            duT, _   = torch.linalg.qr(du.T, mode="reduced")
            du       = duT.T
            duT      = None
            assert du.shape == (self.config.num_pca,self.config.num_state, ), f"Expected duT to have shape {(self.config.num_pca, self.config.num_state)}, but found {du.shape}"
        dh       = alpha @ du
        assert dh.shape == (S, self.config.num_state), f"Expected dh to have shape {(S, self.config.num_state)}, but found {dh.shape}"
        state    = u + dh 
        assert state.shape == (S, self.config.num_state), f"Expected state to have shape {(S, self.config.num_state)}, but found {state.shape}"
        dh = None
        u  = None
        du = None

        return state, dict(alpha=alpha)

