from dataclasses import dataclass
import math
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
import numpy as np
import torch

@dataclass(frozen=True)
class NormalSamplerConfig:
    num_alpha : int|float|None = 0.9   # number of PCA factors as integer or float for % vriance cut off
    rescale_var : bool = True
    seed : int = 113123231
    bandwidth : float = 0.05    # 0: exact resampling / replay, 0.01: tiny jitter, 0.05 default

class NormalSampler(object):
    def __init__(self, market, model, tdata, *, config : NormalSamplerConfig = NormalSamplerConfig(), verbose : Context = Context.all ):
        self.market = market
        self.tdata  = tdata
        self.model  = model
        self.model_output = model(tdata, training=False)
        self.model_output = { k: v.detach().cpu().numpy() for k,v in self.model_output.items() if not v is None }
        self.config       = config
        self.rng          = np.random.default_rng(seed=config.seed)
        self.state        = self.model_output['state']
        assert self.state.shape == (len(market.dates), self.model.config.num_state), f"Internal error {self.state.shape} != {(len(market.dates), self.model.config.num_state)}"

        # PCA
        # ---

        T         = len(market.dates)
        num_state = self.state.shape[-1]
        self.u0   = np.mean(self.state, axis=0, keepdims=True)
        Xc        = self.state - self.u0
        U, SD, Vt = np.linalg.svd(Xc, full_matrices=False)
        assert len(SD) == num_state, "Internal shape error"
        ev        = SD**2
        total_var = np.sum(ev)
        self.vars_explained = np.cumsum(ev)/total_var 

        if config.num_alpha is None or config.num_alpha == -1:
            self.num_alpha = num_state 
        elif isinstance( config.num_alpha, int ):
            verify_inp( config.num_alpha > 0 and config.num_alpha <= num_state, lambda : f"'num_alpha' must be positive and if it is an integer, less than the number of states ({num_state}). Found {config.num_alpha}" )#
            self.num_alpha = config.num_alpha
        else:
            verify_inp( isinstance( config.num_alpha, float ), lambda : f"'num_alpha' must be a float or int, found {type(config.num_alpha)}" )
            verify_inp( config.num_alpha > 0. and config.num_alpha <= 1., lambda : f"'num_alpha' as float must be positive and not exceed 1. Found {config.num_alpha}")
            if config.num_alpha >= 1.:
                self.num_alpha = num_state
            elif config.num_alpha <= self.vars_explained[0]:
                self.num_alpha = 1
            else:
                self.num_alpha = min( np.searchsorted(self.vars_explained, config.num_alpha), num_state )
                assert self.num_alpha >= 0, "Internal error"

        # rescale
        self.var_explained = self.vars_explained[self.num_alpha-1] # this is pre-rescaling
        rescale            = 1. if not config.rescale_var else 1./math.sqrt( self.var_explained )
        self.du            = Vt[:self.num_alpha]               
        self.alpha         = U[:, :self.num_alpha] * SD[:self.num_alpha] * rescale
        
        assert self.u0.shape == (1,num_state), f"Internal error u0 {self.u0.shape} != {(1,num_state)}"
        assert self.du.shape == (self.num_alpha,num_state), f"Internal error du {self.du.shape} != {(self.num_alpha,num_state)}"
        assert self.alpha.shape == (T,self.num_alpha), f"Internal error alpha {self.alpha.shape} != {(T,self.num_alpha)}"
        assert self.vars_explained.shape == (num_state,), f"Internal error vars_explained s{self.vars_explained.shape} != {(num_state,)}"
        
        # decomposition error
        self.decomp_error = self.alpha

        verbose.write( lambda : f"Model {model.qualified_label}: using {self.num_alpha} factors for {num_state} states over {T} dates. The {self.num_alpha} factors explain {self.var_explained:.0%} of variance.")

        # Prepare simulation
        # ------------------

        self.cholesky = np.linalg.cholesky(np.cov(self.alpha, rowvar=False) + 1e-12 * np.eye(self.num_alpha))

    def diff_state( self, index : int, offset : int = 1 ) -> np.ndarray:
        """
        Returns the state at time ``index_next`` starting in ``index`` based on PCA differences.        
        """
        base       = self.alpha[index+offset,:] - self.alpha[index,:]
        return self.state[index,:] + base @ self.du

    def sample_1(self, 
                 index : int,
                 num_samples : int
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of 1-day scenarios starting in the state of ``index``.
        """
        
        bandwidth  = self.config.bandwidth 
        T, A       = self.alpha.shape
        num_state  = self.model.config.num_state
        idx        = self.rng.integers(0, A, size=num_samples)
        dalpha     = self.alpha[idx,:]  - self.alpha[index][None, :]
        assert dalpha.shape == (num_samples, self.num_alpha), "Internal error"
        noise      = self.rng.normal(size=(num_samples, self.num_alpha)) @ self.cholesky.T if bandwidth > 0. else 0.
        new_states = self.state[index,:][None,:] + ( dalpha + bandwidth * noise ) @ self.du

        # ref state - the actual alpha-decode index+1
        base       = self.alpha[index+1] - self.alpha[index]
        ref_state  = self.state[index,:][None,:] + base @ self.du

        assert new_states.shape == (num_samples, self.state.shape[1]), "Internal error"
        return new_states


