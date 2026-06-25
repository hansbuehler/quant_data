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
    use_matrix_exp : bool = True       # if True, use exact affine matrix-exponential AR transition

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
        spot_ref          = self.market['spot_ref']
        self.log_nspot    = np.log( spot_ref / spot_ref[0] )
        self.xstate       = np.concatenate( [ self.log_nspot[:,None], self.model_output['state'] ], axis=1 ) # (T, num_state+1)
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
                                      rescale_var=config.rescale_var,
                                      use_matrix_exp=config.use_matrix_exp )
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
        
        if False:
            eig_k       = np.linalg.eigvals(self.pcaxar1.k)
            rho_euler   = np.max(np.abs(1. - self.pcaxar1.ndt[:,None] * eig_k[None,:]))
            rho_exp     = np.max(np.abs(np.exp(-self.pcaxar1.ndt[:,None] * eig_k[None,:])))

            """ Print some statistics about the fitted PCA-AR1 model. """
            print("alpha mean:", self.pcaxar1.alpha.mean(axis=0))
            print("min real eigenvalue(k):", eig_k.real.min())
            print("max Euler transition spectral radius:", rho_euler)
            print("max matrix-exp transition spectral radius:", rho_exp, flush=True)

    @property
    def num_alpha(self) -> int:
        return self.pcaxar1.alpha.shape[1]
    
    @property
    def num_xstate(self) -> int:
        return self.model.config.num_state+1

    """
    def pred_xstate( self, index : int, offset : int = 1 ) -> np.ndarray:
        verify_inp( 0 <= index < self.xstate.shape[0]-1, lambda : f"Index {index} must be in [0,{self.xstate.shape[0]-1}]")
        verify_inp( offset >= 0, lambda : f"Offset {offset} must be non-negative")
        verify_inp( index+offset <= self.pcaxar1.ndt.shape[0], lambda : f"Index+offset {index+offset} must be less than {self.pcaxar1.ndt.shape[0]}")
        if offset == 0:
            return self.xstate[index,:]
        ndt = self.pcaxar1.ndt[index] if offset == 1 else np.sum(self.pcaxar1.ndt[index:index+offset])
        return self.xstate[index,:] + self.pcaxar1.alpha[index,:] @ self.pcaxar1.dU * np.sqrt(ndt)
    """
    def sample_xstates(self, 
                 index       : int,
                 num_steps   : int, *,
                 num_samples : int = 0,
                 bandwidth   : float = 0.,
                 replay      : bool = False,
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of ``num_steps``-day extedned state scenarios starting in the state of ``index``.

        Parameters
        ----------
        index : int
            The index of the starting state in the time series.
        num_steps : int
            The number of steps to simulate. This includes ``index``; so if ``num_steps=1``, the returned state is just the
            state at ``index``.
        num_samples : int, default 0
            The number of independent samples to generate.
            Set to zero to generate one sample, and return a matrix of shape (num_steps, num_xstate) instead of (num_samples, num_steps, num_xstate).
        bandwidth : float, default 0.
            The bandwidth for the PCA noise. A value of 0 means no noise, while a value of 1 means full noise. Must be in [0,1].
        replay : bool, default False
            If True, the PCA factors are replayed from the historical data, instead of being sampled.
            That means with bandwidth=0, the historical path is generated.

        Returns
        -------
        Result : :class:`np.ndarray`
            Returns a matrix of form ``(num_samples, num_steps, num_xstate)`` if ``num_samples>0`` else ``(num_steps, num_xstate)``.
        """
        verify_inp( num_steps > 0, lambda: f"num_steps {num_steps} must be positive")
        verify_inp( num_samples >= 0, lambda: f"num_samples {num_samples} must be non-negative")
        verify_inp( 0 <= index < self.xstate.shape[0], lambda: f"index {index} must be in [0,{self.xstate.shape[0]-1}]")
        verify_inp( index+num_steps <= self.xstate.shape[0], lambda: f"Index {index} with num_steps {num_steps} must not exceed {self.xstate.shape[0]}")
        verify_inp( bandwidth >= 0. and bandwidth <= 1. , lambda: f"bandwidth {bandwidth} must be in [0,1]")
 
        squeeze_sample = num_samples == 0
        num_samples    = 1 if squeeze_sample else num_samples
        num_alpha      = self.num_alpha
        num_xstate     = self.num_xstate
        num_steps      = num_steps-1

        if num_steps == 0:
            new_xstates = np.repeat(self.xstate[index,:][None,None,:], num_samples, axis=0)
            return new_xstates[0] if squeeze_sample else new_xstates

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
        Xt                  = np.repeat(X0, num_samples, axis=0)
        transition_cache    = {}
        for j in range(num_steps):
            if self.config.use_matrix_exp:
                h = float(ndt[j])
                A, b = transition_cache.get(h, (None, None))
                if A is None:
                    A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=h)
                    transition_cache[h] = (A, b)
                Xt = Xt @ A + b + noise[:,j,:]
            else:
                Xt = Xt + ( U0 - Xt @ k ) * ndt[j] + noise[:,j,:]
            assert Xt.shape == (num_samples, num_xstate), f"Internal error {Xt.shape} != {(num_samples, num_xstate)}"
            new_xstates[:,j+1,:] = Xt
        return new_xstates[0,...] if squeeze_sample else new_xstates

    def sample_log_nspot_states(self, 
                 index       : int,
                 num_steps   : int, *,
                 num_samples : int = 0,
                 bandwidth   : float = 0.,
                 replay      : bool = False,
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate ``num_samples`` of ``num_steps``-day log normalized spot/state scenarios starting in the state of ``index``.

        Parameters
        ----------
        index : int
            The index of the starting state in the time series.
        num_steps : int
            The number of steps to simulate. This includes ``index``; so if ``num_steps=1``, the returned state is just the
            state at ``index``.
        num_samples : int, default 0
            The number of independent samples to generate.
            Set to zero to generate one sample, and return a matrix of shape (num_steps, num_xstate) instead of (num_samples, num_steps, num_xstate).
        bandwidth : float, default 0.
            The bandwidth for the PCA noise. A value of 0 means no noise, while a value of 1 means full noise. Must be in [0,1].
        replay : bool, default False
            If True, the PCA factors are replayed from the historical data, instead of being sampled.
            That means with bandwidth=0, the historical path is generated.
        """
        xstate = self.sample_xstates(index=index, num_steps=num_steps, num_samples=num_samples, bandwidth=bandwidth, replay=replay)
        return ( xstate[:,:,0], xstate[:,:,1:] ) if num_samples > 0 else ( xstate[:,0], xstate[:,1:] )

def _xar1_matrix_exp_terms(U0: np.ndarray, k: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact affine transition terms for dX/dt = U0 - X @ k using row-vector convention.

    Returns ``A`` and ``b`` such that ``X_next = X @ A + b`` over normalized
    time step ``h``.  The augmented exponential avoids inverting ``k`` and is
    therefore safe when ``k`` is singular or nearly singular.
    """
    from scipy.linalg import expm

    verify_inp( h >= 0., lambda: f"h must be non-negative; found {h}")
    verify_inp( k.ndim == 2 and k.shape[0] == k.shape[1], lambda: f"k must be square; found {k.shape}")
    n = k.shape[0]
    verify_inp( U0.shape == (1,n) or U0.shape == (n,), lambda: f"U0 must have shape {(1,n)} or {(n,)}; found {U0.shape}")

    dtype = np.result_type(U0, k, float)
    G = np.zeros((n+1, n+1), dtype=dtype)
    G[:n,:n] = -k
    G[n,:n]  = U0.reshape(n)

    M = expm(h * G)
    A = M[:n,:n]
    b = M[n:n+1,:n]
    return A, b

def _xar1_matrix_exp_drift(X: np.ndarray, U0: np.ndarray, k: np.ndarray, ndt: np.ndarray) -> np.ndarray:
    """
    Exact deterministic AR drift for an array of starting states and time steps.
    """
    verify_inp( X.ndim == 2, lambda: f"X must be two-dimensional; found {X.shape}")
    verify_inp( ndt.shape == (X.shape[0],), lambda: f"ndt must have shape {(X.shape[0],)}; found {ndt.shape}")

    out   = np.empty_like(X)
    cache = {}
    for t, h in enumerate(ndt):
        h = float(h)
        A, b = cache.get(h, (None, None))
        if A is None:
            A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=h)
            cache[h] = (A, b)
        out[t:t+1,:] = X[t:t+1,:] @ A + b
    return out

def fit_xar1_pca(X, time_y, *, num_alpha=0.9, rescale_var=False, use_matrix_exp=False):
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
    verify_inp(isinstance(use_matrix_exp, bool), lambda: f"use_matrix_exp must be a bool; found {type(use_matrix_exp)}")

    # dX_t    = (U0 - X_t k ) * ndt + \sqrt{ndt} * N_t 
    # dX_t/dt = (U0 - X_t k ) + N_t / \sqrt{ndt} 

    dX    = np.diff(X, axis=0)
    dt    = np.diff(time_y)
    verify_inp(np.all(dt > 0.), lambda: f"dt must be positive. Found min dt {np.min(dt)}")
    dtbar  =np.mean(dt)
    ndt   = dt / dtbar
    dXdt  = dX / ndt[:, None]

    X0    = X[:-1,:]
    Xbar  = np.mean(X0, axis=0, keepdims=True)
    dXbar = np.mean(dXdt, axis=0, keepdims=True)

    Xc    = X0 - Xbar
    dXc   = dXdt - dXbar

    slope = np.linalg.lstsq(Xc, dXc, rcond=None)[0]
    k     = - slope
    assert k.shape == (N, N), f"Internal error: expected k to have shape {(N, N)}, but found {k.shape}"

    U0    = dXbar + Xbar @ k
    if use_matrix_exp:
        X1_drift = _xar1_matrix_exp_drift(X0, U0=U0, k=k, ndt=ndt)
        res      = (X[1:,:] - X1_drift) / np.sqrt(ndt)[:, None]
    else:
        hatZ     = U0 - X0 @ k
        res      = (dX - ndt[:, None] * hatZ) / np.sqrt(ndt)[:, None]

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
        mean_dt=dtbar,
        vars_explained=vars_explained,
        var_explained=var_explained)

       
