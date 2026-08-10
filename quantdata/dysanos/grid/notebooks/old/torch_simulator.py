from dataclasses import dataclass
import math
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from flax import config
import numpy as np
import torch
from quantdata.dysanos.grid.torch_pricer import SanosPricer
from quantdata.biz_calendar import DAYS_PER_YEAR, count_business_days_between

@dataclass(frozen=True)
class ARSamplerConfig:
    num_alpha : int|float|None = 0.95   # number of PCA factors as integer or float for % vriance cut off
    rescale_var : bool = True
    seed : int = 113123231

class ARSampler(object):

    Config : ARSamplerConfig
    
    def __init__(self, market,
                       model,
                       torch_data, *, 
                       config : ARSamplerConfig = ARSamplerConfig(), 
                       verbose : Context = Context.all ):
        self.market       = market
        self.torch_data   = torch_data
        self.model        = model
        self.config       = config
        self.model_output = model(torch_data, training=False)
        self.pricer       = SanosPricer.create( smoothness = model.smoothness, model_output = self.model_output, debug = self.model.debug )
        self.model_output = { k: v.detach().cpu().numpy() for k,v in self.model_output.items() }
        self.rng          = np.random.default_rng(seed=config.seed)
        self.spot_ref     = self.market['spot_ref']
        self.spot_ref     = self.spot_ref / self.spot_ref[0]
        self.xstate       = np.concatenate( [ self.spot_ref[:,None], self.model_output['state'] ], axis=1 ) # (T, num_state+1)
        self.expiries     = self.market['expiries']
        self.time_y       = self.market['time_y']
        self.spot_ref     = self.xstate[:,0]
        T                 = len(market.dates)
        assert self.xstate.shape == (T, self.model.config.num_state+1), f"Internal error {self.xstate.shape} != {(T, self.model.config.num_state+1)}"
        assert self.spot_ref.shape == (T,), f"Internal error {self.spot_ref.shape} != {(T,)}"

        # PCA XAR1
        # ---------
        # dX_t = k ( U - X_t ) dt + ( alpha + noise ) @ Y sqrt(nt)

        num_xstate    = self.model.config.num_state+1
        self.pcaxar1  = fit_xar1_pca( self.xstate,
                                      self.time_y, 
                                      num_alpha=config.num_alpha, 
                                      rescale_var=config.rescale_var )
        num_alpha     = self.pcaxar1.alpha.shape[1]

        assert self.pcaxar1.U0.shape == (1,num_xstate), f"Internal error u0 {self.pcaxar1.U0.shape} != {(1,num_xstate)}"
        assert self.pcaxar1.dU.shape == (num_alpha,num_xstate), f"Internal error du {self.pcaxar1.dU.shape} != {(num_alpha,num_xstate)}"
        assert self.pcaxar1.k.shape == (num_xstate,num_xstate), f"Internal error k {self.pcaxar1.k.shape} != {(num_xstate,num_xstate)}"
        assert self.pcaxar1.alpha.shape == (T-1,num_alpha), f"Internal error alpha {self.pcaxar1.alpha.shape} != {(T-1,num_alpha)}"
        assert self.pcaxar1.ndt.shape == (T-1,), f"Internal error ndt {self.pcaxar1.ndt.shape} != {(T-1,)}"
        assert self.pcaxar1.vars_explained.shape == (num_xstate,), f"Internal error vars_explained s{self.pcaxar1.vars_explained.shape} != {(num_xstate,)}"
        
        verbose.write( lambda : f"Model {model.qualified_label}: using {num_alpha} factors for {num_xstate} states (incl. spot) over {T} dates. "+\
                                f"The {num_alpha} factors explain {self.pcaxar1.var_explained:.1%} of variance.")

        # Prepare simulation
        # ------------------
        alpha_hist = self.pcaxar1.alpha
        alpha_mean = np.mean(alpha_hist, axis=0)
        alpha_cov  = np.cov(alpha_hist, rowvar=False, bias=True)

        self.alpha_mean = alpha_mean
        self.cholesky   = np.linalg.cholesky( alpha_cov + 1e-12 * np.eye(self.num_alpha) )
        
        eig_k = np.linalg.eigvals(self.pcaxar1.k)

        rho = max(
            np.max(np.abs(np.linalg.eigvals(
                np.eye(self.num_xstate) - dt * self.pcaxar1.k
            )))
            for dt in self.pcaxar1.ndt
        )

        """ Print some statistics about the fitted PCA-AR1 model. """
        print("alpha mean:", self.pcaxar1.alpha.mean(axis=0))
        print("min real eigenvalue(k):", eig_k.real.min())
        print("max transition spectral radius:", rho, flush=True)

    @property
    def num_alpha(self) -> int:
        return self.pcaxar1.alpha.shape[1]
    
    @property
    def num_xstate(self) -> int:
        return self.model.config.num_state+1

    def pred_xstate( self, index : int, offset : int = 1 ) -> np.ndarray:
        """
        Returns the state at time ``index+offset`` starting in ``index`` based on PCA differences.        
        """
        verify_inp( 0 <= index < self.xstate.shape[0]-1, lambda : f"Index {index} must be in [0,{self.xstate.shape[0]-1}]")
        verify_inp( offset >= 0, lambda : f"Offset {offset} must be non-negative")
        verify_inp( index+offset <= self.pcaxar1.ndt.shape[0], lambda : f"Index+offset {index+offset} must be less than {self.pcaxar1.ndt.shape[0]}")
        if offset == 0:
            return self.xstate[index,:]
        ndt = self.pcaxar1.ndt[index] if offset == 1 else np.sum(self.pcaxar1.ndt[index:index+offset])
        return self.xstate[index,:] + self.pcaxar1.alpha[index,:] @ self.pcaxar1.dU * np.sqrt(ndt)

    def sample_xstates(self, 
                 index : int,
                 num_steps : int, *,
                 num_samples : int = 0,
                 bandwidth : float = 0.,
                 replay : bool = False
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of ``num_steps``-day scenarios starting in the state of ``index``.
        Returns a matrix of form (num_samples, num_steps, num_xstate) 

        Parameters
        ----------
        index : int
            The index of the starting state in the time series.
        num_steps : int
            The number of steps to simulate. This includes ``index``.
        num_samples : int, default 0
            The number of independent samples to generate.
            Set to zero to generate one sample, and return a matrix of shape (num_steps, num_xstate) instead of (num_samples, num_steps, num_xstate).
        bandwidth : float, default 0.
            The bandwidth for the PCA noise. A value of 0 means no noise, while a value of 1 means full noise. Must be in [0,1].
        replay : bool, default False
            If True, the PCA factors are replayed from the historical data, instead of being sampled.
            That means with bandwidth=0, the historical path is generated.
        """
        verify_inp( num_steps > 0, lambda: f"num_steps {num_steps} must be positive")
        verify_inp( num_samples > 0, lambda: f"num_samples {num_samples} must be positive")
        verify_inp( index+num_steps <= self.xstate.shape[0], lambda: f"Index {index} with num_steps {num_steps} must not exceed {self.xstate.shape[0]}")
        verify_inp( bandwidth >= 0. and bandwidth <= 1. , lambda: f"bandwidth {bandwidth} must be in [0,1]")

        num_alpha  = self.num_alpha
        num_xstate = self.num_xstate
        num_steps  = num_steps-1

        if num_steps == 0:
            return self.xstate[index,:][None,None,:] if num_samples == 1 else\
                   np.repeat(self.xstate[index,:][None,None,:], num_samples, axis=0)

        ndt        = self.pcaxar1.ndt[index:index+num_steps]
        Tm1        = self.pcaxar1.alpha.shape[0]
        if not replay:
            idx        = self.rng.integers(0, Tm1, size=(num_samples*num_steps,))
            alpha      = self.pcaxar1.alpha[idx,:].reshape((num_samples, num_steps, num_alpha))
        else:
            alpha      = self.pcaxar1.alpha[index:index+num_steps,:][None,:,:]
            if num_samples > 1:
                alpha = np.repeat(alpha, num_samples, axis=0)

        if bandwidth > 0.:
            z     = self.rng.normal(size=(num_samples, num_steps, num_alpha)) @ self.cholesky.T
            alpha = self.alpha_mean + math.sqrt(max(0., 1. - bandwidth**2)) * (alpha - self.alpha_mean) + bandwidth * z

        noise      = (alpha @ self.pcaxar1.dU) * np.sqrt(ndt[None, :, None])
        assert noise.shape == (num_samples, num_steps, num_xstate), "Internal error"
        k          = self.pcaxar1.k[:,:]
        U0         = self.pcaxar1.U0
        X0         = self.xstate[index,:][None,:]
        assert X0.shape == ( 1, num_xstate), f"Internal error {X0.shape} != {( 1, num_xstate)}"
        assert k.shape == (num_xstate, num_xstate), f"Internal error {k.shape} != {(num_xstate, num_xstate)}"
        assert U0.shape == (1, num_xstate), f"Internal error {U0.shape} != {(1, num_xstate)}"

        new_xstates         = np.zeros((num_samples, num_steps+1, num_xstate), dtype=X0.dtype )
        new_xstates[:,0,:]  = X0
        Xt                  = X0
        for j in range(num_steps):
            Xt      = Xt + ( U0 - Xt @ k ) * ndt[j] + noise[:,j,:]
            assert Xt.shape == (num_samples, num_xstate), f"Internal error {Xt.shape} != {(num_samples, num_xstate)}"
            new_xstates[:,j+1,:] = Xt
        return new_xstates

def fit_xar1_pca(X, time_y, *, num_alpha=0.9, rescale_var=False):
    r"""
    Fits
    $$
        dS_t, dX_t = ( U0 - k X_t) dt + \sqrt{dt} * Y_t + \beta dS
    $$
    and
    $$
        Y_t = \alpha_t @ dU
    $$
    """
    T, N = X.shape
    verify_inp(time_y.shape == (T,), lambda: f"time_y must have shape {(T,)}, found {time_y.shape}")

    # dX_t    = (U0 - X_t k ) * ndt + \sqrt{ndt} * N_t 
    # dX_t/dt = (U0 - X_t k ) + N_t / \sqrt{ndt} 

    dX    = np.diff(X, axis=0)
    dt    = np.diff(time_y)
    verify_inp(np.all(dt > 0.), lambda: f"dt must be positive. Found min dt {np.min(dt)}")
    ndt   = dt / np.mean(dt)
    dXdt  = dX / ndt[:, None]

    X     = X[:-1,:]
    Xbar  = np.mean(X, axis=0, keepdims=True)
    dXbar = np.mean(dXdt, axis=0, keepdims=True)

    Xc    = X - Xbar
    dXc   = dXdt - dXbar

    slope = np.linalg.lstsq(Xc, dXc, rcond=None)[0]
    k     = - slope
    assert k.shape == (N, N), f"Internal error: expected k to have shape {(N, N)}, but found {k.shape}"

    U0    = dXbar + Xbar @ k  
    hatZ  = U0 - X @ k
    res   = (dX - ndt[:, None] * hatZ) / np.sqrt(ndt)[:, None]

    U, SD, V = np.linalg.svd(res, full_matrices=False)

    ev   = SD**2
    vars_explained = np.cumsum(ev) / np.sum(ev)

    if num_alpha is None or num_alpha == -1:
        num_alpha = N
    elif isinstance(num_alpha, int):
        verify_inp(0 < num_alpha <= N, lambda: f"'num_alpha' must be in [1,{N}], found {num_alpha}")
    else:
        verify_inp(isinstance(num_alpha, float), lambda: f"'num_alpha' must be a float or int, found {type(num_alpha)}")
        verify_inp(0. < num_alpha <= 1., lambda: f"'num_alpha' as float must be in (0,1], found {num_alpha}")
        num_alpha = min(np.searchsorted(vars_explained, num_alpha) + 1, N)

    var_explained = vars_explained[num_alpha - 1]
    scale = 1. if not rescale_var else 1. / math.sqrt(var_explained)
    alpha = U[:, :num_alpha] * SD[:num_alpha] * scale
    dU    = V[:num_alpha,:]

    assert alpha.shape == (T-1, num_alpha), f"Internal error: expected alpha to have shape {(T-1, num_alpha)}, but found {alpha.shape}"
    assert dU.shape == (num_alpha, N), f"Internal error: expected dU to have shape {(num_alpha, N)}, but found {dU.shape}"

    return PrettyObject(
        U0=U0,
        dU=dU,
        k=k,
        alpha=alpha,
        ndt=ndt,
        vars_explained=vars_explained,
        var_explained=var_explained)

       
