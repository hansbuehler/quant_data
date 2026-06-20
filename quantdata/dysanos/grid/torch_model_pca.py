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

@version("0.1", dependencies=[GridMarketDefinition, BaseEnvironment])
@dataclass(frozen=True)
class SanosPCACompressorConfig(SanosModelConfig):
    min_dlv: float = 0.01              # Min DLV
    max_dlv: float = 2.                # Max DLV

    num_state : int = 20               # state
    num_pca   : int = 5                # PCA of the state
    state_config : FeedForward.Config = FeedForward.Config()  # decoder
    learn_config : FeedForward.Config|None = None # encoder or None for direct state parameterization
    use_qr : bool = True               # whether to orthogonalize the PCA basis using QR decomposition

    def __post_init__(self):
        super().__post_init__()
        verify_inp( self.min_dlv >= 0., f"Expected min_dlv to be non-negative, but found {self.min_dlv}" )
        verify_inp( self.min_dlv < self.max_dlv, f"Expected min_dlv to be less than max_dlv, but found min_dlv={self.min_dlv} and max_dlv={self.max_dlv}" )
        verify_inp( self.num_state > 0, f"Expected num_state to be positive, but found {self.num_state}" )
        verify_inp( self.num_pca > 0 and self.num_pca <= self.num_state, f"Expected num_pca to be positive and less than or equal to num_state, but found {self.num_pca}" )
        
    def __str__(self) -> str:        
        s = f"{self.label_str}@{self.num_state}@{self.num_pca}{'qr' if self.use_qr else ''} midlv={self.min_dlv:.2f},mxdlv={self.max_dlv:.2f},vsq={self.min_vega_sqrt:3g} st={self.state_config}"
        if not self.learn_config is None:
            s = s + f",lrn={self.learn_config}"
        return s   
    
    def create(self, env : BaseEnvironment, *, S, T, N) -> SanosPCACompressor:
        """ Create Model """
        return SanosPCACompressor( env, S=S, T=T, N=N, config=self )

class SanosPCACompressor(SanosModel):
    
    Config = SanosPCACompressorConfig
    NUM_FEATURES = 6

    def __init__( self, env : BaseEnvironment, *,
                  S,
                  T,
                  N,
                  config : SanosPCACompressorConfig ):
        super().__init__(env)

        self.config = config
        np.random.seed(123123123)

        # h_t = u_0 + \sum_{s<=t} \alpha_s du_s
        self.u_du  = torch.nn.Parameter(torch.tensor(np.random.randn(1+config.num_pca, config.num_state), device=env.device, dtype=env.fdtype), requires_grad=True)

        # define the map state (S, X) -> raw_dlv (S, T, N)
        self.raw_dlv_from_state = FeedForward( env, num_input=config.num_state, num_output=T * N, config=config.state_config )  

        if config.learn_config is not None:
            self.learn_alpha = FeedForward( env, num_input=T*N*self.NUM_FEATURES, num_output=config.num_pca, config=config.learn_config )
            self.dlv_alpha   = None

        else:
            self.learn_alpha = None
            alpha0           = np.random.randn(1, config.num_pca)
            alpha0           = np.full( (S, config.num_pca), alpha0, dtype=np.float32 )
            self.dlv_alpha   = torch.nn.Parameter(torch.tensor(alpha0, device=env.device, dtype=env.fdtype), requires_grad=True)

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

        # compute alphas
        # --------------

        if not self.learn_alpha is None:
            # learn map from features to alpha directly
            logK      = torch.log( xstrikes[:,:,1:-1] )
            features  = torch.stack( [ logK, logK*0.+expiries[:,:,None], mid_vols, nms, mid_vega, mid_delta ], axis=-1 )
            assert features.shape == (S, T, N, self.NUM_FEATURES), f"Expected features to have shape {(S, T, N, self.NUM_FEATURES)}, but found {features.shape}"
            features  = features.reshape( S, -1 )
            alpha     = self.learn_alpha( features )
            features  = None
        else:
            alpha     = self.dlv_alpha
            alpha     = alpha if sample_indices is None else alpha[sample_indices,:]
        assert alpha.shape == (S, self.config.num_pca), f"Expected alpha to have shape {(S, self.config.num_pca)}, but found {alpha.shape}"  

        # compute h, dh
        # -------------

        assert self.u_du.shape == (1+self.config.num_pca, self.config.num_state), f"Expected h_dh to have shape {(1+self.config.num_pca, self.config.num_state)}, but found {self.u_du.shape}"
        duT, _   = torch.linalg.qr(self.u_du[1:,:].T, mode="reduced")  
        assert duT.shape == (self.config.num_state, self.config.num_pca), f"Expected duT to have shape {(self.config.num_state, self.config.num_pca)}, but found {duT.shape}"
        dh       = alpha @ duT.T # todo: add orthogonalization
        assert dh.shape == (S, self.config.num_state), f"Expected dh to have shape {(S, self.config.num_state)}, but found {dh.shape}"
        state    = self.u_du[0,:][None,:] + dh 
        assert state.shape == (S, self.config.num_state), f"Expected state to have shape {(S, self.config.num_state)}, but found {state.shape}"

        dlvs_raw = self.raw_dlv_from_state( state )
        assert dlvs_raw.shape == (S, T*N), f"Expected raw DLVs to have shape {(S, T*N)}, but found {dlvs_raw.shape}"
        dlvs_raw = dlvs_raw.reshape( S, T, N )
        min_dlvsq = self.config.min_dlv**2
        max_dlvsq = self.config.max_dlv**2
        dlvsqs   = torch.sigmoid(dlvs_raw) * (max_dlvsq - min_dlvsq) + min_dlvsq

        if training:
            return dlvsqs, None
        
        return dlvsqs, dict(state=state.detach(), alpha=alpha.detach())
