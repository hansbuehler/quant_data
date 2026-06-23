from dataclasses import dataclass
import math
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from flax import config
import numpy as np
import torch
from quantdata.biz_calendar import DAYS_PER_YEAR, count_business_days_between

@dataclass(frozen=True)
class NormalSamplerConfig:
    num_alpha : int|float|None = 0.9   # number of PCA factors as integer or float for % vriance cut off
    rescale_var : bool = True
    seed : int = 113123231
    bandwidth : float = 0.05    # 0: exact resampling / replay, 0.01: tiny jitter, 0.05 default
    xvar : bool = True

class NormalSampler(object):
    def __init__(self, market, model, tdata, *, config : NormalSamplerConfig = NormalSamplerConfig(), verbose : Context = Context.all ):
        self.market       = market
        self.tdata        = tdata
        self.model        = model
        self.config       = config
        self.model_output = model(tdata, training=False)
        self.model_output = { k: v.detach().cpu().numpy() for k,v in self.model_output.items() if not v is None }
        self.rng          = np.random.default_rng(seed=config.seed)
        self.state        = self.model_output['state']
        self.dState       = self.state[1:,:] - self.state[:-1,:]
        self.expiries     = self.market['expiries']
        self.time_y       = self.market['time_y']
        assert self.state.shape == (len(market.dates), self.model.config.num_state), f"Internal error {self.state.shape} != {(len(market.dates), self.model.config.num_state)}"

        # PCA XAR1
        # ---------
        # dX_t = k ( U - X_t ) dt + ( alpha + noise ) @ Y sqrt(nt)

        Tm1           = self.dState.shape[0]
        num_state     = self.model.config.num_state
        self.pcaxar1  = fit_xar1_pca(self.state, self.time_y, num_alpha=config.num_alpha, rescale_var=config.rescale_var)
        num_alpha     = self.pcaxar1.alpha.shape[1]

        assert self.pcaxar1.U0.shape == (1,num_state), f"Internal error u0 {self.pcaxar1.U0.shape} != {(1,num_state)}"
        assert self.pcaxar1.dU.shape == (num_alpha,num_state), f"Internal error du {self.pcaxar1.dU.shape} != {(num_alpha,num_state)}"
        assert self.pcaxar1.k.shape == (1,num_state), f"Internal error k {self.pcaxar1.k.shape} != {(1,num_state)}"
        assert self.pcaxar1.alpha.shape == (Tm1,num_alpha), f"Internal error alpha {self.pcaxar1.alpha.shape} != {(Tm1,num_alpha)}"
        assert self.pcaxar1.ndt.shape == (Tm1,), f"Internal error ndt {self.pcaxar1.ndt.shape} != {(Tm1,)}"
        assert self.pcaxar1.vars_explained.shape == (num_state,), f"Internal error vars_explained s{self.pcaxar1.vars_explained.shape} != {(num_state,)}"
        
        verbose.write( lambda : f"Model {model.qualified_label}: using {num_alpha} factors for {num_state} states over {Tm1+1} dates. "+\
                                f"The {num_alpha} factors explain {self.pcaxar1.var_explained:.1%} of variance.")

        # Prepare simulation
        # ------------------
        self.cholesky = np.linalg.cholesky(np.cov(self.pcaxar1.alpha, rowvar=False) + 1e-12 * np.eye(self.pcaxar1.alpha.shape[1]))

    def pred_state( self, index : int, offset : int = 1 ) -> np.ndarray:
        """
        Returns the state at time ``index+offset`` starting in ``index`` based on PCA differences.        
        """
        verify_inp( 0 <= index < self.state.shape[0]-1, lambda : f"Index {index} must be in [0,{self.state.shape[0]-1}]")
        verify_inp( offset >= 0, lambda : f"Offset {offset} must be non-negative")
        verify_inp( index+offset <= self.pcaxar1.ndt.shape[0], lambda : f"Index+offset {index+offset} must be less than {self.pcaxar1.ndt.shape[0]}")
        if offset == 0:
            return self.state[index,:]
        ndt = self.pcaxar1.ndt[index] if offset == 1 else np.sum(self.pcaxar1.ndt[index:index+offset])
        return self.state[index,:] + self.pcaxar1.alpha[index,:] @ self.pcaxar1.dU * np.sqrt(ndt)

    def sample_1(self, 
                 index : int,
                 num_samples : int
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of 1-day scenarios starting in the state of ``index``.
        """
        
        assert index<self.dState.shape[0], f"Index {index} must be less than {self.dState.shape[0]}"
        ndt        = self.pcaxar1.ndt[index]
        Tm1, A     = self.pcaxar1.alpha.shape  
        idx        = self.rng.integers(0, Tm1, size=num_samples)
        alpha      = self.pcaxar1.alpha[idx,:]
        assert alpha.shape == (num_samples, self.pcaxar1.alpha.shape[1]), "Internal error"
        noise      = self.rng.normal(size=(num_samples, self.pcaxar1.alpha.shape[1])) @ self.cholesky.T if self.config.bandwidth  > 0. else 0.
        noise      = ( alpha + self.config.bandwidth * noise ) @ self.pcaxar1.dU * np.sqrt(ndt)
        k          = self.pcaxar1.k
        Xt         = self.state[index,:][None,:]
        new_states = Xt + k * ndt * (self.pcaxar1.U0 - Xt) + noise
        assert new_states.shape == (num_samples, self.state.shape[1]), "Internal error"
        return new_states

    def sample_N(self, 
                 index : int,
                 num_steps : int,
                 num_samples : int = 1,
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of ``num_steps``-day scenarios starting in the state of ``index``.
        Returns a sequece of new states in shape ( nun_samples, num_steps, num_state )
        """
        num_alpha  = self.pcaxar1.alpha.shape[1]
        num_state  = self.model.config.num_state
        assert num_steps > 0, f"num_steps {num_steps} must be positive"
        assert num_samples > 0, f"num_samples {num_samples} must be positive"
        assert index<self.dState.shape[0], f"Index {index} must be less than {self.dState.shape[0]}"
        assert num_steps <= self.dState.shape[0]-index, f"Index {index} with num_steps {num_steps} must be less than {self.dState.shape[0]}"    
        assert index+num_steps <= self.dState.shape[0], f"Index {index} with num_steps {num_steps} must be less than {self.dState.shape[0]}"
        ndt        = self.pcaxar1.ndt[index:index+num_steps]
        Tm1, A     = self.pcaxar1.alpha.shape  
        idx        = self.rng.integers(0, Tm1, size=(num_samples*num_steps,))
        alpha      = self.pcaxar1.alpha[idx,:].reshape((num_samples, num_steps, num_alpha))
        h = self.config.bandwidth
        if h > 0.:
            z = self.rng.normal(size=(num_samples, num_steps, num_alpha)) @ self.cholesky.T
            alpha = math.sqrt(max(0., 1. - h*h)) * alpha + h * z
        noise = (alpha @ self.pcaxar1.dU) * np.sqrt(ndt[None, :, None])
        assert noise.shape == (num_samples, num_steps, num_state), "Internal error"
        k          = self.pcaxar1.k[None,:,:]
        U0         = self.pcaxar1.U0[None,:,:]
        Xt         = self.state[index,:][None,None,:] # starting state
        assert Xt.shape == ( 1, 1, num_state), "Internal error"
        assert k.shape == (1, 1, num_state), "Internal error"
        assert U0.shape == (1, 1, num_state), "Internal error"

        new_states     = np.zeros((num_samples, num_steps, num_state), dtype=Xt.dtype )
        for j in range(num_steps):
            Xt     = Xt + k * ndt[j] * (U0 - Xt) + noise[:,j,:]
            new_states[:,j,:] = Xt
        return new_states#*0 + self.state[index,:][None,None,:]

def fit_xar1_pca(X, tte_y, *, num_alpha=0.9, rescale_var=False):
    r"""
    Fits
    $$
        dX_t = k * dt * (U0 - X_t) + \sqrt{dt} * Y_t
    $$
    with
    $$
        Y_t = \alpha_t @ dU
    $$
    """
    T, N = X.shape
    verify_inp(tte_y.shape == (T,), lambda: f"tte_y must have shape {(T,)}, found {tte_y.shape}")

    dt = np.diff(tte_y)

    # If tte_y is time-to-expiry, this may be negative.
    verify_inp(np.all(dt > 0.), lambda: f"dt must be positive. Found min dt {np.min(dt)}")

    ndt = dt / np.mean(dt)

    Xt = X[:-1, :]
    dX = np.diff(X, axis=0)
    dXdt = dX / ndt[:, None]

    Xbar = np.mean(Xt, axis=0, keepdims=True)
    Ybar = np.mean(dXdt, axis=0, keepdims=True)

    Xc = Xt - Xbar
    Yc = dXdt - Ybar

    beta = np.sum(Xc * Yc, axis=0, keepdims=True) / np.sum(Xc * Xc, axis=0, keepdims=True)

    # dXdt = a + beta * Xt + noise
    # beta = -k
    k = -beta

    verify_inp(np.all(k > 0.), lambda: f"Estimated mean reversion k must be positive. Found min {np.min(k)}")

    a = Ybar - beta * Xbar
    U0 = a / k

    fitted_dXdt = a + beta * Xt

    # dX = ndt * fitted_dXdt + sqrt(ndt) * eps
    # hence eps = (dX - ndt * fitted_dXdt) / sqrt(ndt)
    residual = (dX - ndt[:, None] * fitted_dXdt) / np.sqrt(ndt)[:, None]

    U, SD, Vt = np.linalg.svd(residual, full_matrices=False)
    ev = SD**2
    vars_explained = np.cumsum(ev) / np.sum(ev)

    if num_alpha is None or num_alpha == -1:
        K_alpha = N
    elif isinstance(num_alpha, int):
        verify_inp(0 < num_alpha <= N, lambda: f"'num_alpha' must be in [1,{N}], found {num_alpha}")
        K_alpha = num_alpha
    else:
        verify_inp(isinstance(num_alpha, float), lambda: f"'num_alpha' must be a float or int, found {type(num_alpha)}")
        verify_inp(0. < num_alpha <= 1., lambda: f"'num_alpha' as float must be in (0,1], found {num_alpha}")
        K_alpha = min(np.searchsorted(vars_explained, num_alpha) + 1, N)

    var_explained = vars_explained[K_alpha - 1]
    scale = 1. if not rescale_var else 1. / math.sqrt(var_explained)

    dU = Vt[:K_alpha]
    alpha = U[:, :K_alpha] * SD[:K_alpha] * scale

    return PrettyObject(
        U0=U0,dU=dU, k=k,alpha=alpha,ndt=ndt,vars_explained=vars_explained,var_explained=var_explained)

       
