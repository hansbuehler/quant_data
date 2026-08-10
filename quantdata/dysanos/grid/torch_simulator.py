import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject
from cdxcore.uniquehash import unique_hash8
from cdxcore.verbose import Context
from cdxcore.version import version

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_model import SanosModel
from quantdata.dysanos.grid.torch_pricer import SanosPricer
from quantdata.dysanos.simple import sampler


@version("0.3")
@dataclass(frozen=True)
class ARSamplerConfig:
    num_alpha      : int|float|None = 0.95   # number of PCA factors as integer or float for % vriance cut off
    # Retaining fewer residual factors is a low-rank diffusion approximation.
    # Do not inflate those factors to restore discarded variance: that changes
    # the fitted covariance and makes factor-count comparisons inconsistent.
    rescale_var    : bool = False
    seed           : int = 113123231
    use_matrix_exp : bool = True       # if True, use exact affine matrix-exponential AR transition
    surface_first  : bool = True       # fit stationary h before adding log-spot s
    
    def __str__(self) -> str:
        return (
            f"nalp={self.num_alpha},{'rescl' if self.rescale_var else 'nat'},"
            f"{'mxp' if self.use_matrix_exp else 'eul'},"
            f"{'sf' if self.surface_first else 'joint'},seed={self.seed}"
        )
    
    @property
    def rng_current_state_uid(self) -> np.random.Generator:
        """ Utility to mimic the same function in ARSampler, but for the config object. """
        return unique_hash8( np.random.default_rng(seed=self.seed).bit_generator.state )


@version("0.1")
@dataclass(frozen=True, kw_only=True)
class GaussianSpotImportanceConfig:
    """Path-level proposal which broadens selected future log-spot marginals.

    A positive ``base_probability`` component retains the original Gaussian
    path law.  Every other mixture component changes one future log-spot ``s``
    marginal to the same mean and a wider standard deviation.  In the
    surface-first model the complete surface path is retained and the scalar
    conditional spot residual is bridged; the legacy joint model uses a full
    Gaussian state bridge.
    """

    spot_std_multiplier: float = 8.0
    base_probability: float = 0.1
    target_action_count: int = 3

    def __post_init__(self) -> None:
        verify_inp(
            math.isfinite(self.spot_std_multiplier)
            and self.spot_std_multiplier > 1.0,
            lambda: (
                "'spot_std_multiplier' must be finite and greater than one, "
                f"got {self.spot_std_multiplier}"
            ),
        )
        verify_inp(
            math.isfinite(self.base_probability)
            and 0.0 < self.base_probability < 1.0,
            lambda: (
                "'base_probability' must lie strictly between zero and one, "
                f"got {self.base_probability}"
            ),
        )
        verify_inp(
            isinstance(self.target_action_count, int)
            and self.target_action_count > 0,
            lambda: (
                "'target_action_count' must be a positive integer, got "
                f"{self.target_action_count!r}"
            ),
        )


@dataclass(frozen=True, kw_only=True)
class GaussianSpotImportanceSample:
    """One exact Gaussian importance sample and direct-P control moments."""

    paths: np.ndarray
    log_weights: np.ndarray
    proposal_components: np.ndarray
    target_steps: tuple[int, ...]
    target_log_spot_std: np.ndarray
    reference_log_spot_sum: np.ndarray
    reference_log_spot_square_sum: np.ndarray
    reference_realized_variance_sum: float
    reference_realized_variance_square_sum: float

@dataclass(frozen=True)
class AR1FitQuality:
    """One-step deterministic-drift fit diagnostics."""
    r2            : float
    r2_by_state   : np.ndarray
    rmse          : float
    rmse_by_state : np.ndarray
    sample_mean   : np.ndarray
    sample_std    : np.ndarray
    fitted_mean   : np.ndarray
    mean_error    : np.ndarray

    def summary_statistics(self) -> dict:
        """Return scale-independent mean-fit statistics and log-spot details."""
        mean_error_in_std = np.divide(
            self.mean_error,
            self.sample_std,
            out=np.full_like(self.mean_error, np.nan, dtype=float),
            where=self.sample_std > 0.,
        )
        finite = np.isfinite(mean_error_in_std)
        return {
            "mean_error_in_std": mean_error_in_std,
            "mean_error_rms_in_std": (
                float(np.sqrt(np.mean(mean_error_in_std[finite] ** 2)))
                if np.any(finite) else math.nan
            ),
            "mean_error_max_abs_in_std": (
                float(np.max(np.abs(mean_error_in_std[finite])))
                if np.any(finite) else math.nan
            ),
            "log_spot_sample_mean": float(self.sample_mean[0]),
            "log_spot_fitted_mean": float(self.fitted_mean[0]),
            "log_spot_sample_std": float(self.sample_std[0]),
            "log_spot_mean_error_in_std": float(mean_error_in_std[0]),
        }

    def __str__(self) -> str:
        stats = self.summary_statistics()
        if math.isfinite(stats["log_spot_fitted_mean"]):
            spot_fit = (
                f"fitted={stats['log_spot_fitted_mean']:.4g}, "
                f"error={stats['log_spot_mean_error_in_std']:.3g} std"
            )
        else:
            spot_fit = "fitted mean=N/A (unit root)"
        return (
            f"one-step R²={self.r2:.1%}, RMSE={self.rmse:.4g}; "
            f"mean fit RMS={stats['mean_error_rms_in_std']:.3g} std, "
            f"max={stats['mean_error_max_abs_in_std']:.3g} std; "
            f"log-spot mean: sample={stats['log_spot_sample_mean']:.4g}, "
            f"sample std={stats['log_spot_sample_std']:.4g}, "
            f"{spot_fit}"
        )

@dataclass(frozen=True)
class AR1PCA:
    r"""
    dh_t = (U0 - k h_t)dt + \alpha' dU \sqrt{dt}
    """
    U0     : np.ndarray
    alpha  : np.ndarray
    dU     : np.ndarray
    k      : np.ndarray
    ndt    : np.ndarray          # dt/mean(dt)
    mean_dt: float               # mean(dt)
    state_center: np.ndarray|None     # coordinate center used for stable simulation
    state_scale : np.ndarray|None     # coordinate scale used for stable simulation

    # explained
    vars_explained : np.ndarray
    var_explained  : float 
    fit_quality    : AR1FitQuality|None = None
    surface_first  : bool = False
    num_surface_alpha: int|None = None
    surface_var_explained: float|None = None
    spot_surface_r2: float|None = None
    spot_residual_std: float|None = None

@version(
    "1.1",
    dependencies=[
        ARSamplerConfig,
        GaussianSpotImportanceConfig,
        SanosPricer,
        SanosModel,
    ],
)
class ARSampler(object):

    Config : ARSamplerConfig
    
    def __init__(self, market,
                       model,
                       torch_data=None, *,
                       surface_states: np.ndarray | None = None,
                       config : ARSamplerConfig = ARSamplerConfig(), 
                       verbose : Context = Context.all ):
        self.market       = market
        self.torch_data   = torch_data
        self.model        = model
        self.config       = config
        verify_inp(
            (torch_data is None) != (surface_states is None),
            "provide exactly one of 'torch_data' and 'surface_states'",
        )
        if surface_states is None:
            torch_output = model(torch_data, training=False)
            self.pricer = SanosPricer.create(
                smoothness=model.smoothness,
                model_output=torch_output,
                debug=model.debug,
            )
            self.model_output = {
                name: value.detach().cpu().numpy()
                for name, value in torch_output.items()
            }
        else:
            state = np.asarray(surface_states)
            verify_inp(
                state.shape == (len(market.dates), model.config.num_state)
                and np.issubdtype(state.dtype, np.floating)
                and np.all(np.isfinite(state)),
                lambda: (
                    "'surface_states' must be a finite floating array with shape "
                    f"{(len(market.dates), model.config.num_state)}, got "
                    f"{state.shape}/{state.dtype}"
                ),
            )
            # Path generation needs only the fitted surface coordinates. The
            # full training output/pricer would retain multi-GB objective
            # tensors which are unrelated to the AR fit.
            self.pricer = None
            self.model_output = {"state": state}
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
        # dX_t = ( U0 - k X_t ) dt + ( alpha + noise ) @ Y sqrt(nt)

        num_xstate    = self.model.config.num_state+1
        fit_function = (
            fit_surface_first_xar1_pca
            if config.surface_first
            else fit_xar1_pca
        )
        self.pcaxar1: AR1PCA = fit_function(
            self.xstate,
            self.time_y,
            num_alpha=config.num_alpha,
            rescale_var=config.rescale_var,
            use_matrix_exp=config.use_matrix_exp,
        )
        num_alpha     = self.pcaxar1.alpha.shape[1]

        assert self.pcaxar1.U0.shape == (1,num_xstate), f"Internal error u0 {self.pcaxar1.U0.shape} != {(1,num_xstate)}"
        assert self.pcaxar1.dU.shape == (num_alpha,num_xstate), f"Internal error du {self.pcaxar1.dU.shape} != {(num_alpha,num_xstate)}"
        assert self.pcaxar1.k.shape == (num_xstate,num_xstate), f"Internal error k {self.pcaxar1.k.shape} != {(num_xstate,num_xstate)}"
        assert self.pcaxar1.alpha.shape == (T-1,num_alpha), f"Internal error alpha {self.pcaxar1.alpha.shape} != {(T-1,num_alpha)}"
        assert self.pcaxar1.ndt.shape == (T-1,), f"Internal error ndt {self.pcaxar1.ndt.shape} != {(T-1,)}"
        assert self.pcaxar1.vars_explained.shape == (num_xstate,), f"Internal error vars_explained s{self.pcaxar1.vars_explained.shape} != {(num_xstate,)}"

        eig_k = np.linalg.eigvals(self.pcaxar1.k)
        if self.pcaxar1.surface_first:
            eig_surface = np.linalg.eigvals(self.pcaxar1.k[1:, 1:])
            tolerance = 100.0 * np.finfo(float).eps
            assert (
                np.all(np.isfinite(eig_k))
                and np.all(eig_surface.real > 0.0)
                and np.min(np.abs(eig_k)) <= tolerance
            ), (
                "Unstable surface-first generator: expected one zero spot "
                f"eigenvalue and a stable surface block, found {eig_k}"
            )
        else:
            assert np.all(np.isfinite(eig_k)) and np.all(eig_k.real > 0.), \
                f"Unstable fitted k: expected all eigenvalues to have positive real parts, found {eig_k}"
        
        factor_description = (
            f"{self.pcaxar1.num_surface_alpha} surface factors plus one "
            "conditional spot factor"
            if self.pcaxar1.surface_first
            else f"{num_alpha} joint factors"
        )
        verbose.write(
            lambda: (
                f"Model {model.qualified_label}: using {factor_description} "
                f"for {num_xstate} states over {T} dates. The retained factors "
                f"explain {self.pcaxar1.var_explained:.1%} of joint innovation "
                f"variance; {self.pcaxar1.fit_quality}."
            )
        )

        # Prepare simulation
        # ------------------
        alpha_hist      = self.pcaxar1.alpha
        self.alpha_mean = np.mean(alpha_hist, axis=0)
        alpha_cov       = np.cov(alpha_hist, rowvar=False, bias=True)
        self.cholesky   = np.linalg.cholesky( alpha_cov + 1e-12 * np.eye(self.num_alpha) )
        
    @property
    def num_alpha(self) -> int:
        return self.pcaxar1.alpha.shape[1]
    
    @property
    def num_xstate(self) -> int:
        return self.model.config.num_state+1
    
    def reset_rng(self):
        """ Reset the random number generator to the original seed. """
        self.rng = np.random.default_rng(seed=self.config.seed)

    def invariant_surface_distribution(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and covariance of the stationary ML-SANOS state block.

        This distribution exists when the fitted surface generator has all
        eigenvalues in the open right half-plane.  Log-spot is deliberately
        excluded because it has a unit root in the surface-first model.
        """

        from scipy.linalg import solve_continuous_lyapunov

        verify_inp(
            self.pcaxar1.surface_first,
            "the invariant surface distribution requires surface_first=True",
        )
        surface_k = self.pcaxar1.k[1:, 1:]
        eig_k = np.linalg.eigvals(surface_k)
        verify_inp(
            np.all(np.isfinite(eig_k)) and np.all(eig_k.real > 0.0),
            lambda: (
                "the surface generator is not stationary; eigenvalues are "
                f"{eig_k}"
            ),
        )
        alpha_cov = np.atleast_2d(
            np.cov(self.pcaxar1.alpha, rowvar=False, bias=True)
        )
        state_cov = self.pcaxar1.dU.T @ alpha_cov @ self.pcaxar1.dU
        surface_cov_rate = state_cov[1:, 1:]
        covariance = solve_continuous_lyapunov(
            surface_k.T,
            surface_cov_rate,
        )
        covariance = 0.5 * (covariance + covariance.T)
        values = np.linalg.eigvalsh(covariance)
        tolerance = 1.0e3 * np.finfo(values.dtype).eps * max(
            1.0, float(np.max(np.abs(values)))
        )
        verify_inp(
            values[0] >= -tolerance,
            lambda: (
                "invariant surface covariance is not positive semidefinite; "
                f"minimum eigenvalue {values[0]:g}"
            ),
        )
        mean = np.linalg.solve(
            surface_k.T,
            self.pcaxar1.U0[0, 1:],
        )
        return mean, covariance

    def sample_invariant_surface_states(self, num_samples: int) -> np.ndarray:
        """Draw Gaussian states from the invariant surface distribution."""

        verify_inp(
            isinstance(num_samples, int) and num_samples > 0,
            lambda: f"'num_samples' must be a positive integer, got {num_samples!r}",
        )
        mean, covariance = self.invariant_surface_distribution()
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0.0)
        factor = vectors * np.sqrt(values)[None, :]
        samples = mean[None, :] + self.rng.normal(
            size=(num_samples, mean.size)
        ) @ factor.T
        if not np.all(np.isfinite(samples)):
            raise FloatingPointError("invariant surface sampling produced non-finite states")
        return samples

    @property
    def rng_current_state_uid(self) -> np.random.Generator:
        """ Identifies current 'seed'. Used for caching samples """
        return unique_hash8( self.rng.bit_generator.state )

    def report_system(self):
        eig_k       = np.linalg.eigvals(self.pcaxar1.k)
        rho_euler   = np.max(np.abs(1. - self.pcaxar1.ndt[:,None] * eig_k[None,:]))
        rho_exp     = np.max(np.abs(np.exp(-self.pcaxar1.ndt[:,None] * eig_k[None,:])))

        """ Print some statistics about the fitted PCA-AR1 model. """
        print("alpha mean:", self.pcaxar1.alpha.mean(axis=0))
        print("min real eigenvalue(k):", eig_k.real.min())
        print("max Euler transition spectral radius:", rho_euler)
        print("max matrix-exp transition spectral radius:", rho_exp, flush=True)
        print("u0", self.pcaxar1.U0)

    def hist_sample_xstates(self, 
                 index       : int,
                 num_steps   : int, *,
                 num_samples : int = 0,
                 bandwidth   : float = 0.,
                 replay      : bool = True,
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` of ``num_steps`` days of extended state scenarios starting in the state of ``index``
        and using the PCA factors from the historical data. The PCA factors can be replayed or sampled, and noise can be added to the PCA factors.
        If ``replay=True`` and ``bandwidth=0`` this replays historic factor loadings.

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
        replay : bool, default True
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

    def hist_sample_log_nspot_states(self, 
                 index       : int,
                 num_steps   : int, *,
                 num_samples : int = 0,
                 bandwidth   : float = 0.,
                 replay      : bool = True,
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
        replay : bool, default True
            If True, the PCA factors are replayed from the historical data, instead of being sampled.
            That means with bandwidth=0, the historical path is generated.

        Returns
        -------
        State : tuple[ np.ndarray, np.ndarray ]
            log-spot and states
        """
        xstate = self.hist_sample_xstates(index=index, num_steps=num_steps, num_samples=num_samples, bandwidth=bandwidth, replay=replay)
        return ( xstate[:,:,0], xstate[:,:,1:] ) if num_samples > 0 else ( xstate[:,0], xstate[:,1:] )

    def free_sample_xstates(self, 
                 num_steps       : int, 
                 num_samples     : int = 1, 
                 dt              : float = 1./DAYS_PER_YEAR, *,
                 index           : int = None, 
                 bandwidth       : float = 0.,
                 random_start    : bool = True,
                 max_internal_dt : float|None = None,
                 initial_state   : np.ndarray|None = None,
                  ) -> np.ndarray:
        """
        Generate ``num_samples`` paths on a grid with spacing ``dt``.

        The PCA innovation law is calibrated at approximately ``pcaxar1.mean_dt``.
        To simulate a larger output step robustly, each interval is therefore split
        into internal substeps no larger than ``max_internal_dt``.  Every internal
        substep uses an independently sampled PCA loading and the exact affine
        matrix-exponential drift.  Only the states on the requested output grid are
        returned.

        Parameters
        ----------
        num_steps : int
            Number of returned states.  The initial state is included, so
            ``num_steps=1`` returns only the starting state.
        num_samples : int, default 1
            Number of independent paths.  Set to zero to generate one path and
            remove the leading sample dimension from the result.
        dt : float, default ``1/DAYS_PER_YEAR``
            Time between returned states, in the same units as ``time_y``.
        index : int or None
            Starting-state cut-off.  If ``random_start=False``, start exactly at
            ``xstate[index]``.  If ``random_start=True``, sample starting states
            from ``xstate[:index+1]``.  With ``index=None``, use the full history.
        bandwidth : float, default 0
            Gaussian smoothing of the empirical PCA loadings.  Zero uses pure
            historical resampling; one uses a Gaussian with the same covariance.
        random_start : bool, default True
            Whether to sample starting states up to ``index``.
        max_internal_dt : float or None, default None
            Maximum internal simulation step used when ``bandwidth < 1``.
            ``None`` uses the mean historical step ``pcaxar1.mean_dt``.  In full
            normal mode (``bandwidth == 1``), the exact Gaussian OU transition is
            sampled directly over ``dt`` and this argument is ignored.
        initial_state : numpy.ndarray or None, default None
            Exact raw-coordinate state from which every path starts. This is
            useful for conditional Monte Carlo. It cannot be combined with
            ``index``; ``random_start`` has no effect when it is supplied.

        Returns
        -------
        Result : :class:`np.ndarray`
            Shape ``(num_samples, num_steps, num_xstate)`` when
            ``num_samples>0``; otherwise ``(num_steps, num_xstate)``.
        """
        verify_inp( num_steps > 0, lambda: f"num_steps {num_steps} must be positive")
        verify_inp( num_samples >= 0, lambda: f"num_samples {num_samples} must be non-negative")
        verify_inp( bandwidth >= 0. and bandwidth <= 1., lambda: f"bandwidth {bandwidth} must be in [0,1]")
        verify_inp( np.isfinite(dt) and dt > 0., lambda: f"dt {dt} must be finite and positive")
        verify_inp(
            initial_state is not None or index is not None or random_start,
            "'random_start' cannot be False if neither 'index' nor "
            "'initial_state' is supplied",
        )
        verify_inp(
            initial_state is None or index is None,
            "'initial_state' cannot be combined with 'index'",
        )

        Tm1 = self.pcaxar1.alpha.shape[0]
        verify_inp(
            index is None or (0 <= index < Tm1),
            lambda: f"index {index} must be in [0,{Tm1-1}]",
        )

        if bandwidth < 1.:
            if max_internal_dt is None:
                max_internal_dt = float(self.pcaxar1.mean_dt)*10.
            verify_inp(
                np.isfinite(max_internal_dt) and max_internal_dt > 0.,
                lambda: f"max_internal_dt {max_internal_dt} must be finite and positive",
            )

        squeeze_sample = num_samples == 0
        num_samples    = 1 if squeeze_sample else num_samples
        num_alpha      = self.num_alpha
        num_xstate     = self.num_xstate

        # Work in the normalized time used to fit the XAR(1) model.
        ndt = float(dt / self.pcaxar1.mean_dt)
        internal_ndt = None if bandwidth == 1. else _split_xar1_step( ndt, float(max_internal_dt / self.pcaxar1.mean_dt) )

        if initial_state is not None:
            initial_state = np.asarray(initial_state)
            verify_inp(
                initial_state.shape == (num_xstate,)
                and np.issubdtype(initial_state.dtype, np.floating)
                and np.all(np.isfinite(initial_state)),
                lambda: (
                    "'initial_state' must be a finite floating vector with "
                    f"shape {(num_xstate,)}, got "
                    f"{initial_state.shape}/{initial_state.dtype}"
                ),
            )
            X0 = np.repeat(initial_state[None, :], num_samples, axis=0)
            alpha_pool_size = Tm1
        elif index is None:
            X0              = self.xstate[self.rng.integers(0, Tm1, size=num_samples), :]
            alpha_pool_size = Tm1
        else:
            X0 = (
                self.xstate[index, :][None, :]
                if not random_start
                else self.xstate[self.rng.integers(0, index + 1, size=num_samples), :]
            )
            alpha_pool_size = index + 1

        # Simulate in standardized coordinates.  The raw decoder states differ
        # greatly in scale, making matrix exponentials and Lyapunov covariance
        # calculations poorly conditioned when performed after mapping back.
        center     = self.pcaxar1.state_center 
        scale      = self.pcaxar1.state_scale 
        k_raw      = self.pcaxar1.k
        U0_raw     = self.pcaxar1.U0
        k          = k_raw * scale[:, None] / scale[None, :] if self.config.use_matrix_exp else k_raw
        U0         = (U0_raw - center[None, :] @ k_raw) / scale[None, :] if self.config.use_matrix_exp else U0_raw
        dU         = self.pcaxar1.dU / scale[None, :] if self.config.use_matrix_exp else self.pcaxar1.dU
        X0         = (X0 - center[None, :]) / scale[None, :] if self.config.use_matrix_exp else X0
        alpha_hist = self.pcaxar1.alpha

        assert X0.shape == (1, num_xstate) or X0.shape == (num_samples, num_xstate),  f"Internal X0 error {X0.shape}"
        assert k.shape == (num_xstate, num_xstate), f"Internal k error {k.shape} != {(num_xstate, num_xstate)}"
        assert U0.shape == (1, num_xstate), f"Internal U0 error {U0.shape} != {(1, num_xstate)}"
        assert dU.shape == (num_alpha, num_xstate), f"Internal dU error {dU.shape} != {(num_alpha, num_xstate)}"

        new_xstates        = np.zeros((num_samples, num_steps, num_xstate), dtype=X0.dtype)
        new_xstates[:, 0]  = X0
        Xt                 = new_xstates[:, 0, :]

        if not self.config.use_matrix_exp:
            # Original vectorized Euler path.  Generate the entire innovation
            # tensor at once and take exactly one transition per output step.
            # This deliberately bypasses exact-OU covariance construction and
            # empirical internal substeps.
            num_transitions = num_steps - 1
            gaussian_weight = bandwidth
            historical_weight = math.sqrt(max(0., 1. - bandwidth**2))
            if bandwidth < 1.:
                idx = self.rng.integers(  0, alpha_pool_size, size=(num_samples, num_transitions) )
                alpha = alpha_hist[idx, :]
                if bandwidth > 0.:
                    z = self.rng.normal( size=(num_samples, num_transitions, num_alpha) ) @ self.cholesky.T
                    alpha = ( self.alpha_mean + historical_weight * (alpha - self.alpha_mean) + gaussian_weight * z )
            else:
                z = self.rng.normal( size=(num_samples, num_transitions, num_alpha) )
                alpha = self.alpha_mean + z @ self.cholesky.T

            noise = (alpha @ dU) * math.sqrt(ndt)
            assert noise.shape == (num_samples, num_transitions, num_xstate), \
                f"Internal Euler noise error {noise.shape}"
            A = np.eye(num_xstate, dtype=k.dtype) - ndt * k
            b = ndt * U0
            
            for j in range(num_transitions):
                Xt = Xt @ A + b + noise[:, j, :]
                assert Xt.shape == (num_samples, num_xstate), \
                    f"Internal Xt error {Xt.shape} != {(num_samples, num_xstate)}"
                new_xstates[:, j + 1, :] = Xt

            new_xstates = center[None, None, :] + new_xstates * scale[None, None, :]
            return new_xstates[0, ...] if squeeze_sample else new_xstates

        if bandwidth == 1.:
            # Exact Gaussian OU transition and its integrated covariance.
            A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=ndt)
            verify_inp(
                np.all(np.isfinite(A)) and np.all(np.isfinite(b)),
                lambda: "Non-finite matrix-exponential transition; "
                        "the fitted k may be unstable for this time step",
            )
            alpha_cov = self.cholesky @ self.cholesky.T
            state_cov = dU.T @ alpha_cov @ dU
            Lh = _xar1_gaussian_step_factor(k=k, C=state_cov, h=ndt, A=A)
            Q = Lh @ Lh.T
            noise = _sample_surface_then_spot_gaussian_noise(
                covariance=Q,
                size=(num_samples, num_steps - 1),
                rng=self.rng,
            )

            assert noise.shape == (num_samples, num_steps - 1, num_xstate), \
                f"Internal Gaussian noise error {noise.shape}"

            for j in range(num_steps - 1):
                Xt = Xt @ A + b + noise[:, j, :]
                assert Xt.shape == (num_samples, num_xstate), \
                    f"Internal Xt error {Xt.shape} != {(num_samples, num_xstate)}"
                new_xstates[:, j + 1, :] = Xt

            new_xstates = center[None, None, :] + new_xstates * scale[None, None, :]
            return new_xstates[0, ...] if squeeze_sample else new_xstates

        transition_cache   = {}
        gaussian_weight    = bandwidth
        historical_weight  = math.sqrt(max(0., 1. - bandwidth**2))

        # With any empirical component left, preserve the PCA innovation law at
        # its fitted time scale.  Sequential application automatically damps early
        # shocks by the remaining matrix-exponential transitions.
        for j in range(num_steps - 1):
            for h in internal_ndt:
                h = float(h)
                A, b = transition_cache.get(h, (None, None))
                if A is None:
                    A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=h)
                    verify_inp(
                        np.all(np.isfinite(A)) and np.all(np.isfinite(b)),
                        lambda: "Non-finite state transition; the fitted k may "
                                "be unstable for this time step",
                    )
                    transition_cache[h] = (A, b)

                idx   = self.rng.integers(0, alpha_pool_size, size=num_samples)
                alpha = alpha_hist[idx, :]

                if bandwidth > 0.:
                    z = self.rng.normal(size=(num_samples, num_alpha)) @ self.cholesky.T
                    alpha = (
                        self.alpha_mean
                        + historical_weight * (alpha - self.alpha_mean)
                        + gaussian_weight * z
                    )

                noise = (alpha @ dU) * math.sqrt(h)
                assert noise.shape == (num_samples, num_xstate), \
                    f"Internal noise error {noise.shape} != {(num_samples, num_xstate)}"

                Xt = Xt @ A + b + noise
                assert Xt.shape == (num_samples, num_xstate), \
                    f"Internal Xt error {Xt.shape} != {(num_samples, num_xstate)}"

            new_xstates[:, j + 1, :] = Xt

        new_xstates = center[None, None, :] + new_xstates * scale[None, None, :]
        return new_xstates[0, ...] if squeeze_sample else new_xstates

    def sample_conditional_xstates_at_offsets(
        self,
        *,
        initial_state: np.ndarray,
        step_offsets: np.ndarray,
        num_samples: int,
        dt: float = 1.0 / DAYS_PER_YEAR,
    ) -> np.ndarray:
        """Draw exact Gaussian states at selected offsets from one state.

        This is the sparse-time conditional counterpart of
        :meth:`free_sample_xstates`. It avoids constructing every intervening
        daily state when only option-expiry states are required.
        """

        verify_inp(
            self.config.use_matrix_exp and self.pcaxar1.surface_first,
            "sparse conditional sampling requires the surface-first exact "
            "Gaussian model",
        )
        verify_inp(
            isinstance(num_samples, int) and num_samples > 0,
            lambda: f"'num_samples' must be positive, got {num_samples!r}",
        )
        verify_inp(
            np.isfinite(dt) and dt > 0.0,
            lambda: f"'dt' must be finite-positive, got {dt!r}",
        )
        initial_state = np.asarray(initial_state)
        offsets = np.asarray(step_offsets)
        states = self.num_xstate
        verify_inp(
            initial_state.shape == (states,)
            and np.issubdtype(initial_state.dtype, np.floating)
            and np.all(np.isfinite(initial_state)),
            lambda: (
                "'initial_state' must be a finite floating vector with shape "
                f"{(states,)}, got {initial_state.shape}/{initial_state.dtype}"
            ),
        )
        verify_inp(
            offsets.ndim == 1
            and offsets.size >= 2
            and np.issubdtype(offsets.dtype, np.integer)
            and offsets[0] == 0
            and np.all(np.diff(offsets) > 0),
            lambda: (
                "'step_offsets' must be an increasing integer vector starting "
                f"at zero, got {offsets}"
            ),
        )

        center = np.asarray(self.pcaxar1.state_center, dtype=np.float64)
        scale = np.asarray(self.pcaxar1.state_scale, dtype=np.float64)
        verify_inp(
            center.shape == scale.shape == (states,)
            and np.all(np.isfinite(center))
            and np.all(np.isfinite(scale))
            and np.all(scale > 0.0),
            "invalid state normalization for conditional sampling",
        )
        raw_k = np.asarray(self.pcaxar1.k, dtype=np.float64)
        k = raw_k * scale[:, None] / scale[None, :]
        U0 = (
            np.asarray(self.pcaxar1.U0, dtype=np.float64)
            - center[None, :] @ raw_k
        ) / scale[None, :]
        dU = np.asarray(self.pcaxar1.dU, dtype=np.float64) / scale[None, :]
        alpha_covariance = self.cholesky @ self.cholesky.T
        covariance_rate = dU.T @ alpha_covariance @ dU

        one_step_h = float(dt / self.pcaxar1.mean_dt)
        one_step_A, one_step_b = _xar1_matrix_exp_terms(
            U0=U0, k=k, h=one_step_h
        )
        one_step_factor = _xar1_gaussian_step_factor(
            k=k,
            C=covariance_rate,
            h=one_step_h,
            A=one_step_A,
        )
        one_step_covariance = one_step_factor @ one_step_factor.T

        result = np.empty(
            (num_samples, offsets.size, states), dtype=np.float64
        )
        Xt = np.repeat(
            ((initial_state.astype(np.float64) - center) / scale)[None, :],
            num_samples,
            axis=0,
        )
        result[:, 0] = Xt
        for output_index, step_count in enumerate(np.diff(offsets), start=1):
            # Compose the exact one-day transition instead of evaluating the
            # generic Van Loan block exponential over the entire interval.
            # The surface-first model contains a spot unit root, for which the
            # long-horizon block exponential is unnecessarily ill-conditioned.
            A = np.eye(states, dtype=np.float64)
            b = np.zeros((1, states), dtype=np.float64)
            covariance = np.zeros((states, states), dtype=np.float64)
            for _ in range(int(step_count)):
                covariance = (
                    one_step_A.T @ covariance @ one_step_A
                    + one_step_covariance
                )
                b = b @ one_step_A + one_step_b
                A = A @ one_step_A
            covariance = 0.5 * (covariance + covariance.T)
            noise = _sample_surface_then_spot_gaussian_noise(
                covariance=covariance,
                size=(num_samples,),
                rng=self.rng,
            )
            Xt = Xt @ A + b + noise
            result[:, output_index] = Xt

        result *= scale[None, None, :]
        result += center[None, None, :]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("conditional state sample is non-finite")
        return result

    def importance_sample_xstates(
        self,
        *,
        num_steps: int,
        num_samples: int,
        dt: float,
        target_steps: tuple[int, ...],
        config: GaussianSpotImportanceConfig,
        proposal_rng: np.random.Generator,
    ) -> GaussianSpotImportanceSample:
        """Sample exact Gaussian paths under a log-spot tail proposal.

        For the surface-first model, the original surface path is retained and
        only the spot residual conditional on that complete path is widened.
        For the legacy joint fit, the original draw supplies the Gaussian state
        residual conditional on the selected future log spot.  In both cases
        the returned weight is the likelihood ratio of the original path law
        to the *full* proposal mixture, not merely to the selected component.
        """

        verify_inp(
            self.config.use_matrix_exp,
            "Gaussian path importance sampling requires use_matrix_exp=True",
        )
        verify_inp(
            self.pcaxar1.surface_first
            or self.config.num_alpha is None
            or self.num_alpha == self.num_xstate,
            lambda: (
                "Gaussian path importance sampling requires the full state "
                f"innovation dimension, got {self.num_alpha}/{self.num_xstate}"
            ),
        )
        verify_inp(
            isinstance(proposal_rng, np.random.Generator),
            lambda: f"'proposal_rng' must be a NumPy Generator, got {type(proposal_rng)}",
        )
        verify_inp(
            bool(target_steps)
            and tuple(sorted(set(target_steps))) == target_steps
            and target_steps[0] > 0
            and target_steps[-1] < num_steps,
            lambda: (
                "target steps must be unique, increasing, positive, and below "
                f"{num_steps}; got {target_steps}"
            ),
        )

        reference = self.free_sample_xstates(
            num_steps=num_steps,
            num_samples=num_samples,
            dt=dt,
            bandwidth=1.0,
        )
        reference_log_spot = reference[..., 0]
        reference_log_spot_sum = reference_log_spot.sum(axis=0)
        reference_log_spot_square_sum = np.square(reference_log_spot).sum(axis=0)
        reference_rv = np.square(np.diff(reference_log_spot, axis=1)).sum(axis=1)
        reference_rv_sum = float(reference_rv.sum())
        reference_rv_square_sum = float(np.square(reference_rv).sum())
        center = self.pcaxar1.state_center
        scale = self.pcaxar1.state_scale
        verify_inp(
            center is not None and scale is not None,
            "matrix-exponential importance sampling requires state normalization",
        )
        # Standardize the direct-P draw in place. At million-path scale an
        # additional full (sample, time, state) temporary costs tens of GB.
        # The direct-P control moments above are the only raw-coordinate
        # quantities needed after this point.
        standardized_reference = reference
        standardized_reference -= center[None, None, :]
        standardized_reference /= scale[None, None, :]

        ndt = float(dt / self.pcaxar1.mean_dt)
        k = self.pcaxar1.k * scale[:, None] / scale[None, :]
        U0 = (
            self.pcaxar1.U0 - center[None, :] @ self.pcaxar1.k
        ) / scale[None, :]
        dU = self.pcaxar1.dU / scale[None, :]
        A, b = _xar1_matrix_exp_terms(U0=U0, k=k, h=ndt)
        alpha_cov = self.cholesky @ self.cholesky.T
        state_cov = dU.T @ alpha_cov @ dU
        noise_factor = _xar1_gaussian_step_factor(
            k=k,
            C=state_cov,
            h=ndt,
            A=A,
        )
        Q = noise_factor @ noise_factor.T
        importance_function = (
            _importance_sample_spot_given_surface_paths
            if self.pcaxar1.surface_first
            else _importance_sample_gaussian_spot_paths
        )
        standardized, log_weights, components, target_std = (
            importance_function(
                reference_paths=standardized_reference,
                A=A,
                b=b,
                Q=Q,
                target_steps=target_steps,
                config=config,
                rng=proposal_rng,
            )
        )
        standardized *= scale[None, None, :]
        standardized += center[None, None, :]
        paths = standardized
        if not np.all(np.isfinite(paths)) or not np.all(np.isfinite(log_weights)):
            raise FloatingPointError("Gaussian importance sample is non-finite")

        return GaussianSpotImportanceSample(
            paths=paths.astype(reference.dtype, copy=False),
            log_weights=log_weights,
            proposal_components=components,
            target_steps=target_steps,
            target_log_spot_std=target_std * float(scale[0]),
            reference_log_spot_sum=reference_log_spot_sum,
            reference_log_spot_square_sum=reference_log_spot_square_sum,
            reference_realized_variance_sum=reference_rv_sum,
            reference_realized_variance_square_sum=reference_rv_square_sum,
        )


def _sample_surface_then_spot_gaussian_noise(
    *,
    covariance: np.ndarray,
    size: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample joint Gaussian noise by drawing surface, then conditional spot.

    The extended state is ordered as ``(s, h)``, where ``s`` is log-spot and
    ``h`` contains the autonomous ML-SANOS implied-volatility parameters. The
    construction is valid for singular surface covariance matrices and makes
    the sampling order explicit without changing the joint Gaussian law.
    """

    from scipy.linalg import eigh

    verify_inp(
        covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
        and covariance.shape[0] >= 2,
        lambda: f"'covariance' must be square with at least two states, got {covariance.shape}",
    )
    verify_inp(
        isinstance(size, tuple)
        and all(isinstance(value, int) and value >= 0 for value in size),
        lambda: f"'size' must be a tuple of non-negative integers, got {size!r}",
    )
    Q = 0.5 * (covariance + covariance.T)
    verify_inp(np.all(np.isfinite(Q)), "'covariance' contains non-finite values")
    surface_covariance = Q[1:, 1:]
    values, vectors = eigh(surface_covariance)
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 1.0e3 * np.finfo(float).eps * scale
    verify_inp(
        values[0] >= -tolerance,
        lambda: f"surface covariance is not positive semidefinite: {values[0]:g}",
    )
    keep = values > tolerance
    surface_factor = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    surface_noise = (
        rng.normal(size=(*size, int(np.sum(keep)))) @ surface_factor.T
    )

    surface_inverse = np.linalg.pinv(
        surface_covariance,
        rcond=1.0e3 * np.finfo(float).eps,
        hermitian=True,
    )
    spot_given_surface = surface_inverse @ Q[1:, 0]
    conditional_variance = float(
        Q[0, 0] - Q[0, 1:] @ spot_given_surface
    )
    variance_tolerance = 1.0e3 * np.finfo(float).eps * max(
        1.0, abs(float(Q[0, 0]))
    )
    verify_inp(
        conditional_variance >= -variance_tolerance,
        lambda: (
            "conditional spot variance is negative: "
            f"{conditional_variance:g}"
        ),
    )
    conditional_std = math.sqrt(max(conditional_variance, 0.0))
    spot_noise = surface_noise @ spot_given_surface
    if conditional_std > 0.0:
        spot_noise += conditional_std * rng.normal(size=size)

    result = np.empty((*size, Q.shape[0]), dtype=Q.dtype)
    result[..., 0] = spot_noise
    result[..., 1:] = surface_noise
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("surface-first Gaussian sampling produced non-finite noise")
    return result


def _xar1_gaussian_step_factor(
    k: np.ndarray,
    C: np.ndarray,
    h: float,
    A: np.ndarray|None = None,
) -> np.ndarray:
    r"""
    Exact Gaussian noise factor for the row-vector OU transition.

    For

        dX_t = (U0 - X_t @ k) dt + dW_t @ B,
        C    = B.T @ B,

    return ``L`` such that ``L @ L.T = Q_h``, where

        Q_h = integral_0^h exp(-s k.T) C exp(-s k) ds.

    A stable Lyapunov formulation is used whenever ``k`` is mean reverting;
    Van Loan's block exponential is retained as a finite-horizon fallback.
    An eigendecomposition, rather than Cholesky, permits singular low-rank PCA
    covariance matrices.
    """
    from scipy.linalg import eigh, eigvals, expm, solve_continuous_lyapunov

    verify_inp(np.isfinite(h) and h >= 0., lambda: f"h must be finite and non-negative; found {h}")
    verify_inp(k.ndim == 2 and k.shape[0] == k.shape[1], lambda: f"k must be square; found {k.shape}")
    n = k.shape[0]
    verify_inp(C.shape == (n, n), lambda: f"C must have shape {(n,n)}; found {C.shape}")
    verify_inp(np.all(np.isfinite(k)) and np.all(np.isfinite(C)), lambda: "k and C must contain only finite values")

    dtype = np.result_type(k, C, float)
    k = np.asarray(k, dtype=dtype)
    C = np.asarray(C, dtype=dtype)
    C = 0.5 * (C + C.T)

    if h == 0.:
        return np.zeros((n, 0), dtype=dtype)

    if A is None:
        A = expm(-h * k)
    else:
        A = np.asarray(A, dtype=dtype)
        verify_inp(A.shape == (n, n), lambda: f"A must have shape {(n,n)}; found {A.shape}")

    eig_k = eigvals(k)
    stability_tol = 100. * np.finfo(float).eps * max(1., np.linalg.norm(k, ord=2))

    if np.min(eig_k.real) > stability_tol:
        # Stationary covariance P solves k.T P + P k = C, and
        # Q_h = P - A.T P A.  This avoids the exponentially growing block
        # present in the Van Loan construction and is robust for large h.
        P = solve_continuous_lyapunov(k.T, C)
        Q = P - A.T @ P @ A
    else:
        # Finite-horizon fallback valid even when no stationary covariance exists.
        # For F=-k.T, Van Loan gives Q_h = M12 exp(F.T h) = M12 A.
        G = np.zeros((2*n, 2*n), dtype=dtype)
        G[:n, :n] = -k.T
        G[:n, n:] = C
        G[n:, n:] = k
        M = expm(h * G)
        Q = M[:n, n:] @ A

    Q = 0.5 * (Q + Q.T)
    verify_inp(np.all(np.isfinite(Q)), lambda: "Non-finite Gaussian transition covariance")

    values, vectors = eigh(Q)
    scale = max(1., float(np.max(np.abs(values))))
    tolerance = 1000. * np.finfo(float).eps * scale
    verify_inp(
        values[0] >= -tolerance,
        lambda: f"Gaussian transition covariance is not PSD; minimum eigenvalue {values[0]}",
    )

    values = np.maximum(values, 0.)
    keep = values > tolerance
    return vectors[:, keep] * np.sqrt(values[keep])[None, :]


def _importance_sample_spot_given_surface_paths(
    *,
    reference_paths: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    target_steps: tuple[int, ...],
    config: GaussianSpotImportanceConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Broaden conditional log-spot tails while preserving every surface path.

    The surface transition is autonomous.  At each step its innovation is
    sampled under the original law, and the simultaneous spot innovation is
    decomposed into its conditional mean given that surface innovation plus an
    independent scalar Gaussian residual.  Importance sampling changes only
    those scalar residuals.  Consequently the simulated surface marginal and
    every realized surface path are exactly unchanged.
    """

    from scipy.special import logsumexp

    verify_inp(
        reference_paths.ndim == 3 and reference_paths.shape[-1] > 1,
        lambda: (
            "'reference_paths' must have shape (S, steps, states>1), got "
            f"{reference_paths.shape}"
        ),
    )
    samples, steps, states = reference_paths.shape
    verify_inp(
        A.shape == Q.shape == (states, states) and b.shape == (1, states),
        lambda: f"invalid affine Gaussian geometry {A.shape}/{b.shape}/{Q.shape}",
    )
    verify_inp(
        tuple(sorted(set(target_steps))) == target_steps
        and target_steps[0] > 0
        and target_steps[-1] < steps,
        lambda: f"invalid target steps {target_steps} for path length {steps}",
    )
    if not all(np.all(np.isfinite(value)) for value in (reference_paths, A, b, Q)):
        raise ValueError("conditional Gaussian importance inputs contain non-finite values")

    autonomy_tolerance = 1.0e3 * np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(A, ord=2))
    )
    verify_inp(
        np.max(np.abs(A[0, 1:])) <= autonomy_tolerance,
        lambda: (
            "surface transition is not autonomous: current spot loads into "
            f"the next surface by {A[0, 1:]}"
        ),
    )
    surface_covariance = 0.5 * (Q[1:, 1:] + Q[1:, 1:].T)
    surface_inverse = np.linalg.pinv(
        surface_covariance,
        rcond=1.0e3 * np.finfo(float).eps,
        hermitian=True,
    )
    spot_given_surface = surface_inverse @ Q[1:, 0]
    residual_variance = float(
        Q[0, 0] - Q[0, 1:] @ spot_given_surface
    )
    variance_tolerance = 1.0e3 * np.finfo(float).eps * max(
        1.0, abs(float(Q[0, 0]))
    )
    verify_inp(
        residual_variance > variance_tolerance,
        lambda: (
            "conditional spot residual variance must be positive for tail "
            f"importance sampling, got {residual_variance:g}"
        ),
    )
    residual_std = math.sqrt(residual_variance)

    innovations = (
        reference_paths[:, 1:, :]
        - reference_paths[:, :-1, :] @ A
        - b[None, :, :]
    )
    surface_innovations = innovations[..., 1:]
    spot_conditional_mean = surface_innovations @ spot_given_surface
    residual_z = (
        innovations[..., 0] - spot_conditional_mean
    ) / residual_std

    # Conditional on the initial state and complete surface path, only the
    # scalar residual shocks remain random.  Propagate their zero-residual
    # conditional mean and their target-step standard deviations.
    conditional_mean = np.empty((samples, steps), dtype=np.float64)
    conditional_mean[:, 0] = reference_paths[:, 0, 0]
    for step in range(steps - 1):
        conditional_mean[:, step + 1] = (
            conditional_mean[:, step] * A[0, 0]
            + reference_paths[:, step, 1:] @ A[1:, 0]
            + b[0, 0]
            + spot_conditional_mean[:, step]
        )

    target_count = len(target_steps)
    bridge_coefficients = np.zeros((target_count, steps - 1), dtype=np.float64)
    target_variance = np.empty(target_count, dtype=np.float64)
    for target_index, target_step in enumerate(target_steps):
        powers = np.arange(target_step - 1, -1, -1)
        coefficients = residual_std * np.power(A[0, 0], powers)
        variance = float(coefficients @ coefficients)
        verify_inp(
            np.isfinite(variance) and variance > variance_tolerance,
            lambda: f"invalid conditional spot variance at step {target_step}: {variance}",
        )
        target_variance[target_index] = variance
        bridge_coefficients[target_index, :target_step] = coefficients / math.sqrt(variance)

    target_std = np.sqrt(target_variance)
    reference_target_z = (
        reference_paths[:, target_steps, 0]
        - conditional_mean[:, target_steps]
    ) / target_std[None, :]
    reconstructed_target_z = residual_z @ bridge_coefficients.T
    assert np.allclose(reference_target_z, reconstructed_target_z, atol=1.0e-8)

    base_probability = config.base_probability
    component_probability = (1.0 - base_probability) / target_count
    probabilities = np.full(target_count + 1, component_probability)
    probabilities[0] = base_probability
    components = rng.choice(
        target_count + 1,
        size=samples,
        replace=True,
        p=probabilities,
    ).astype(np.int16, copy=False)
    proposal_residual_z = residual_z.astype(np.float64, copy=True)
    multiplier = config.spot_std_multiplier
    for component in range(1, target_count + 1):
        selected = np.flatnonzero(components == component)
        if selected.size == 0:
            continue
        target_index = component - 1
        proposed_target_z = multiplier * rng.standard_normal(selected.size)
        difference = (
            proposed_target_z
            - reference_target_z[selected, target_index]
        )
        proposal_residual_z[selected] += (
            difference[:, None]
            * bridge_coefficients[target_index][None, :]
        )

    proposal = reference_paths.astype(np.float64, copy=True)
    for step in range(steps - 1):
        proposal[:, step + 1, 0] = (
            proposal[:, step] @ A[:, 0]
            + b[0, 0]
            + spot_conditional_mean[:, step]
            + residual_std * proposal_residual_z[:, step]
        )
    assert np.array_equal(proposal[..., 1:], reference_paths[..., 1:])

    proposal_target_z = (
        proposal[:, target_steps, 0]
        - conditional_mean[:, target_steps]
    ) / target_std[None, :]
    log_component_ratio = (
        -math.log(multiplier)
        + 0.5
        * np.square(proposal_target_z)
        * (1.0 - 1.0 / multiplier**2)
    )
    log_terms = np.concatenate(
        (
            np.full((samples, 1), math.log(base_probability)),
            math.log(component_probability) + log_component_ratio,
        ),
        axis=1,
    )
    log_weights = -logsumexp(log_terms, axis=1)
    if not np.all(np.isfinite(proposal)) or not np.all(np.isfinite(log_weights)):
        raise FloatingPointError("conditional spot proposal or likelihood is non-finite")
    return proposal, log_weights, components, target_std


def _importance_sample_gaussian_spot_paths(
    *,
    reference_paths: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    target_steps: tuple[int, ...],
    config: GaussianSpotImportanceConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply an exact future-log-spot Gaussian-mixture proposal.

    ``reference_paths`` must already be a draw from the affine Gaussian path
    law conditional on its first state.  For each target coordinate ``Y``, the
    identity

    ``X | Y=y = X_P + Cov(X,Y)/Var(Y) * (y-Y_P)``

    turns that draw into an exact Gaussian bridge.  The proposal is a mixture
    of the original path law and one widened marginal for every target step.
    """

    from scipy.special import logsumexp

    verify_inp(
        reference_paths.ndim == 3 and reference_paths.shape[-1] > 1,
        lambda: (
            "'reference_paths' must have shape (S, steps, states>1), got "
            f"{reference_paths.shape}"
        ),
    )
    samples, steps, states = reference_paths.shape
    verify_inp(
        A.shape == Q.shape == (states, states) and b.shape == (1, states),
        lambda: f"invalid affine Gaussian geometry {A.shape}/{b.shape}/{Q.shape}",
    )
    verify_inp(
        tuple(sorted(set(target_steps))) == target_steps
        and target_steps[0] > 0
        and target_steps[-1] < steps,
        lambda: f"invalid target steps {target_steps} for path length {steps}",
    )
    if not all(np.all(np.isfinite(value)) for value in (reference_paths, A, b, Q)):
        raise ValueError("Gaussian importance inputs contain non-finite values")

    powers = np.empty((steps, states, states), dtype=np.float64)
    variances = np.empty_like(powers)
    powers[0] = np.eye(states)
    variances[0] = 0.0
    for step in range(1, steps):
        powers[step] = powers[step - 1] @ A
        variances[step] = A.T @ variances[step - 1] @ A + Q

    target_count = len(target_steps)
    target_variance = np.asarray(
        [variances[step, 0, 0] for step in target_steps], dtype=np.float64
    )
    variance_floor = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(variances)))
    )
    verify_inp(
        np.all(target_variance > variance_floor),
        lambda: (
            "future log-spot marginal variance is zero or numerically singular: "
            f"{target_variance}"
        ),
    )

    betas = np.empty((target_count, steps, states), dtype=np.float64)
    for target_index, target_step in enumerate(target_steps):
        target_covariance = variances[target_step]
        for step in range(steps):
            if step <= target_step:
                covariance = variances[step] @ powers[target_step - step]
                betas[target_index, step] = covariance[:, 0]
            else:
                covariance = powers[step - target_step].T @ target_covariance
                betas[target_index, step] = covariance[:, 0]
        betas[target_index] /= target_variance[target_index]
        assert np.allclose(betas[target_index, target_step, 0], 1.0)

    target_lookup = {step: index for index, step in enumerate(target_steps)}
    deterministic = reference_paths[:, 0].astype(np.float64, copy=True)
    target_mean = np.empty((samples, target_count), dtype=np.float64)
    for step in range(1, steps):
        deterministic = deterministic @ A + b
        target_index = target_lookup.get(step)
        if target_index is not None:
            target_mean[:, target_index] = deterministic[:, 0]

    base_probability = config.base_probability
    component_probability = (1.0 - base_probability) / target_count
    probabilities = np.full(target_count + 1, component_probability)
    probabilities[0] = base_probability
    components = rng.choice(
        target_count + 1,
        size=samples,
        replace=True,
        p=probabilities,
    ).astype(np.int16, copy=False)
    proposal = reference_paths.astype(np.float64, copy=True)
    target_std = np.sqrt(target_variance)
    multiplier = config.spot_std_multiplier
    for component in range(1, target_count + 1):
        selected = np.flatnonzero(components == component)
        if selected.size == 0:
            continue
        target_index = component - 1
        target_step = target_steps[target_index]
        proposed_z = multiplier * rng.standard_normal(selected.size)
        proposed_spot = (
            target_mean[selected, target_index]
            + target_std[target_index] * proposed_z
        )
        difference = (
            proposed_spot - reference_paths[selected, target_step, 0]
        )
        proposal[selected] += (
            difference[:, None, None] * betas[target_index][None, :, :]
        )

    target_value = proposal[:, target_steps, 0]
    standardized_target = (target_value - target_mean) / target_std[None, :]
    log_component_ratio = (
        -math.log(multiplier)
        + 0.5
        * np.square(standardized_target)
        * (1.0 - 1.0 / multiplier**2)
    )
    log_terms = np.concatenate(
        (
            np.full((samples, 1), math.log(base_probability)),
            math.log(component_probability) + log_component_ratio,
        ),
        axis=1,
    )
    log_weights = -logsumexp(log_terms, axis=1)
    if not np.all(np.isfinite(proposal)) or not np.all(np.isfinite(log_weights)):
        raise FloatingPointError("Gaussian bridge proposal or likelihood is non-finite")
    verify_inp(
        float(np.max(log_weights))
        <= -math.log(base_probability) + 1.0e-10,
        "importance likelihood exceeds its base-mixture bound",
    )
    return proposal, log_weights, components, target_std

def _split_xar1_step(h: float, max_h: float) -> np.ndarray:
    """Split one normalized XAR(1) step into full substeps plus a remainder."""
    verify_inp(np.isfinite(h) and h > 0., lambda: f"h must be finite and positive; found {h}")
    verify_inp(np.isfinite(max_h) and max_h > 0., lambda: f"max_h must be finite and positive; found {max_h}")

    # Keeping full native-size substeps is preferable to equal subdivision:
    # the empirical alpha distribution was calibrated at that native scale.
    tol = 32. * np.finfo(float).eps * max(1., abs(h), abs(max_h))
    n_full = int(math.floor((h + tol) / max_h))
    remainder = h - n_full * max_h

    if abs(remainder) <= tol:
        remainder = 0.
    elif remainder < 0.:
        # Can only occur through floating-point round-off near an exact multiple.
        n_full -= 1
        remainder = h - n_full * max_h

    parts = [max_h] * n_full
    if remainder > 0.:
        parts.append(remainder)
    if not parts:
        parts.append(h)

    out = np.asarray(parts, dtype=float)
    assert np.all(out > 0.)
    assert abs(float(np.sum(out)) - h) <= 4. * tol
    return out

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

def _fit_daily_ou_drift(
    X0: np.ndarray,
    X1: np.ndarray,
    ndt: np.ndarray,
    initial_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a continuous OU generator from the dominant daily transition.

    Most observations have one normalized daily interval.  An affine least-
    squares transition at that interval followed by one matrix logarithm is
    both much faster and more robust than optimizing a matrix exponential at
    every calibration iteration.  If its logarithm is not a real stable
    generator, retain the Euler regression and shift it minimally into the
    stable half-plane.
    """
    from scipy.linalg import logm

    verify_inp(
        X0.ndim == 2 and X1.shape == X0.shape,
        lambda: f"Invalid OU observations {X0.shape}, {X1.shape}",
    )
    verify_inp(
        ndt.shape == (X0.shape[0],),
        lambda: f"ndt must have shape {(X0.shape[0],)}, found {ndt.shape}",
    )
    n = X0.shape[1]
    verify_inp(
        initial_k.shape == (n, n),
        lambda: f"initial_k must have shape {(n, n)}, found {initial_k.shape}",
    )

    unique_h, h_counts = np.unique(ndt, return_counts=True)
    common_h = float(unique_h[np.argmax(h_counts)])
    common = ndt == common_h
    if np.count_nonzero(common) >= n + 2:
        design = np.concatenate(
            (np.ones((np.count_nonzero(common), 1)), X0[common]), axis=1
        )
        if np.linalg.matrix_rank(design) == n + 1:
            affine = np.linalg.lstsq(design, X1[common], rcond=None)[0]
            candidate_complex = -logm(affine[1:, :]) / common_h
            imaginary_scale = float(np.max(np.abs(candidate_complex.imag)))
            real_scale = max(1., float(np.max(np.abs(candidate_complex.real))))
            candidate_k = candidate_complex.real
            try:
                candidate_mu = np.linalg.solve(
                    (np.eye(n) - affine[1:, :]).T, affine[0, :]
                )
            except np.linalg.LinAlgError:
                candidate_mu = np.full(n, np.nan)
            eig_k = np.linalg.eigvals(candidate_k)
            if (
                np.all(np.isfinite(candidate_complex))
                and np.all(np.isfinite(candidate_mu))
                and imaginary_scale <= 1e-8 * real_scale
                and np.all(eig_k.real > 0.)
            ):
                return candidate_mu[None, :] @ candidate_k, candidate_k

    dXdt = (X1 - X0) / ndt[:, None]
    Xbar = np.mean(X0, axis=0, keepdims=True)
    dXbar = np.mean(dXdt, axis=0, keepdims=True)
    centered_k = -np.linalg.lstsq(X0 - Xbar, dXdt - dXbar, rcond=None)[0]
    candidate_k = centered_k if np.all(np.isfinite(centered_k)) else initial_k.copy()
    stability_floor = math.sqrt(np.finfo(candidate_k.dtype).eps)
    minimum_real_eigenvalue = float(np.min(np.linalg.eigvals(candidate_k).real))
    if minimum_real_eigenvalue <= stability_floor:
        candidate_k = candidate_k + (
            stability_floor - minimum_real_eigenvalue
        ) * np.eye(n, dtype=candidate_k.dtype)
    return dXbar + Xbar @ candidate_k, candidate_k

def _fit_continuous_ou_drift(
    X0: np.ndarray,
    X1: np.ndarray,
    ndt: np.ndarray,
    initial_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an exact, stable continuous-time OU drift to irregular observations.

    The generator is parameterized as ``k = L L.T + S``, with ``S``
    skew-symmetric.  Its symmetric part is therefore positive definite, which
    makes ``exp(-h k)`` contractive for every positive horizon ``h``.  The
    equilibrium ``mu`` is fitted directly and ``U0 = mu @ k``.
    """
    from scipy.optimize import minimize

    verify_inp(X0.ndim == 2 and X1.shape == X0.shape, lambda: f"Invalid OU observations {X0.shape}, {X1.shape}")
    verify_inp(ndt.shape == (X0.shape[0],), lambda: f"ndt must have shape {(X0.shape[0],)}, found {ndt.shape}")
    n = X0.shape[1]
    verify_inp(initial_k.shape == (n, n), lambda: f"initial_k must have shape {(n,n)}, found {initial_k.shape}")

    # Obtain an exact discrete-to-continuous initial estimate from the most
    # common observation interval.  For X1 = c + X0 B, the corresponding OU
    # generator is k = -log(B)/h.  This puts the nonlinear irregular-time fit
    # close to its optimum and avoids hundreds of expensive refinement steps.
    unique_h, h_counts = np.unique(ndt, return_counts=True)
    common_h = unique_h[np.argmax(h_counts)]
    common = ndt == common_h
    initial_mu = None
    if np.count_nonzero(common) >= n + 2:
        from scipy.linalg import logm

        design = np.concatenate((np.ones((np.count_nonzero(common), 1)), X0[common]), axis=1)
        affine = np.linalg.lstsq(design, X1[common], rcond=None)[0]
        discrete_k = -logm(affine[1:, :]) / float(common_h)
        imaginary_scale = float(np.max(np.abs(discrete_k.imag)))
        real_scale = max(1., float(np.max(np.abs(discrete_k.real))))
        if np.all(np.isfinite(discrete_k)) and imaginary_scale <= 1e-8 * real_scale:
            initial_k = discrete_k.real
            try:
                candidate_mu = np.linalg.solve((np.eye(n) - affine[1:, :]).T, affine[0, :])
                if np.all(np.isfinite(candidate_mu)):
                    initial_mu = candidate_mu
            except np.linalg.LinAlgError:
                pass

    # Project the initial estimate onto the interior of the stable
    # parameterization while retaining its rotational component.
    sym = 0.5 * (initial_k + initial_k.T)
    values, vectors = np.linalg.eigh(sym)
    floor = 1e-6 * max(1., float(np.linalg.norm(initial_k, ord=2)))
    sym = (vectors * np.maximum(values, floor)[None, :]) @ vectors.T
    L0 = np.linalg.cholesky(sym)
    S0 = 0.5 * (initial_k - initial_k.T)
    mu0 = np.mean(np.concatenate((X0, X1[-1:, :]), axis=0), axis=0) if initial_mu is None else initial_mu

    lower = np.tril_indices(n)
    upper = np.triu_indices(n, 1)
    diag_positions = np.flatnonzero(lower[0] == lower[1])

    def pack(mu: np.ndarray, L: np.ndarray, S: np.ndarray) -> np.ndarray:
        lp = L[lower].copy()
        lp[diag_positions] = np.log(lp[diag_positions])
        return np.concatenate((mu, lp, S[upper]))

    def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = p[:n]
        lp = p[n:n + len(lower[0])].copy()
        lp[diag_positions] = np.exp(lp[diag_positions])
        L = np.zeros((n, n), dtype=p.dtype)
        L[lower] = lp
        sp = p[n + len(lower[0]):]
        S = np.zeros((n, n), dtype=p.dtype)
        S[upper] = sp
        S[(upper[1], upper[0])] = -sp
        return mu, L @ L.T + S

    unique_h, inverse_h = np.unique(ndt, return_inverse=True)
    tdtype = torch.float64
    tx0 = torch.as_tensor(X0, dtype=tdtype)
    tx1 = torch.as_tensor(X1, dtype=tdtype)
    th = torch.as_tensor(unique_h, dtype=tdtype)
    thinverse = torch.as_tensor(inverse_h, dtype=torch.long)
    tsqrt_h = torch.as_tensor(np.sqrt(ndt)[:, None], dtype=tdtype)
    tlower = (torch.as_tensor(lower[0]), torch.as_tensor(lower[1]))
    tupper = (torch.as_tensor(upper[0]), torch.as_tensor(upper[1]))
    tdiag_mask = torch.as_tensor(lower[0] == lower[1])

    def objective_and_gradient(p: np.ndarray) -> tuple[float, np.ndarray]:
        """One batched matrix-exponential evaluation and its exact gradient."""
        tp = torch.tensor(p, dtype=tdtype, requires_grad=True)
        mu = tp[:n]
        lp = tp[n:n + len(lower[0])]
        lp = torch.where(tdiag_mask, torch.exp(lp), lp)
        L = torch.zeros((n, n), dtype=tdtype)
        L[tlower] = lp
        sp = tp[n + len(lower[0]):]
        S = torch.zeros((n, n), dtype=tdtype)
        S[tupper] = sp
        S[(tupper[1], tupper[0])] = -sp
        k = L @ L.T + S

        transitions = torch.matrix_exp(-th[:, None, None] * k[None, :, :])
        centered = tx0 - mu
        prediction = mu + torch.bmm(centered[:, None, :], transitions[thinverse]).squeeze(1)
        standardized_error = (tx1 - prediction) / tsqrt_h
        objective = 0.5 * torch.mean(standardized_error.square())
        objective.backward()
        return float(objective.detach()), tp.grad.detach().numpy()

    p0 = pack(mu0, L0, S0)
    diagonal_parameters = n + diag_positions
    bounds = [(None, None)] * len(p0)
    for ix in diagonal_parameters:
        bounds[ix] = (-20., 10.)
    result = minimize(
        objective_and_gradient,
        p0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-7, "maxls": 20},
    )
    verify_inp(
        np.all(np.isfinite(result.x)) and np.isfinite(result.fun),
        lambda: f"Continuous-time OU calibration produced a non-finite result: {result.message}",
    )
    if not result.success:
        warnings.warn(
            "Continuous-time OU calibration stopped before the optimizer's "
            f"convergence criterion ({result.message}); using the best stable "
            f"iterate after {result.nit} iterations with objective {result.fun:.6g}."
        )
    mu, k = unpack(result.x)
    eig_k = np.linalg.eigvals(k)
    verify_inp(
        np.all(np.isfinite(eig_k)) and np.all(eig_k.real > 0.),
        lambda: f"Continuous-time OU calibration returned an unstable generator with eigenvalues {eig_k}",
    )
    return mu[None, :] @ k, k

def fit_xar1_pca(X, time_y, *, num_alpha=0.9, rescale_var=False, use_matrix_exp=False):
    r"""
    Fits
    $$
        dX_t = ( U0 - k X_t) dt + \sqrt{dt} * Y_t
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

    state_center = np.zeros(N, dtype=X.dtype)
    state_scale = np.ones(N, dtype=X.dtype)
    if use_matrix_exp:
        # Decoder state coordinates can have vastly different scales.  Fit the
        # continuous-time OU in standardized coordinates, then map its generator
        # back with an exact similarity transform.  This preserves transitions
        # and eigenvalues without forcing the optimizer to handle raw magnitudes.
        state_center = np.mean(X, axis=0)
        state_scale = np.std(X, axis=0)
        scale_floor = math.sqrt(np.finfo(X.dtype).eps) * max(1., float(np.max(state_scale)))
        state_scale = np.maximum(state_scale, scale_floor)
        Z = (X - state_center[None, :]) / state_scale[None, :]
        k_z_initial = k * state_scale[:, None] / state_scale[None, :]
        # Estimate the dominant daily affine transition and take one matrix
        # logarithm rather than solving a dense nonlinear problem repeatedly.
        U0_z, k_z = _fit_daily_ou_drift(
            Z[:-1, :], Z[1:, :], ndt, initial_k=k_z_initial
        )
        k = k_z * state_scale[None, :] / state_scale[:, None]
        U0 = U0_z * state_scale[None, :] + state_center[None, :] @ k
        X1_drift = _xar1_matrix_exp_drift(X0, U0=U0, k=k, ndt=ndt)
        res      = (X[1:,:] - X1_drift) / np.sqrt(ndt)[:, None]
    else:
        U0       = dXbar + Xbar @ k
        hatZ     = U0 - X0 @ k
        X1_drift = X0 + ndt[:, None] * hatZ
        res      = (X[1:,:] - X1_drift) / np.sqrt(ndt)[:, None]

    # Report fit quality in the original state coordinates.  R² measures the
    # deterministic one-step drift fit; PCA explained variance below measures
    # how much of the remaining innovation variance is retained for simulation.
    fit_error = X[1:,:] - X1_drift
    squared_error = np.sum(fit_error**2, axis=0)
    centered_target = X[1:,:] - np.mean(X[1:,:], axis=0, keepdims=True)
    total_squares = np.sum(centered_target**2, axis=0)
    r2_by_state = np.full(N, np.nan, dtype=float)
    nonconstant = total_squares > 0.
    r2_by_state[nonconstant] = 1. - squared_error[nonconstant] / total_squares[nonconstant]
    total_squares_all = float(np.sum(total_squares))
    sample_mean = np.mean(X, axis=0)
    sample_std = np.std(X, axis=0)
    fitted_mean = np.linalg.lstsq(k.T, U0.reshape(N), rcond=None)[0]
    fit_quality = AR1FitQuality(
        r2=(1. - float(np.sum(squared_error)) / total_squares_all)
            if total_squares_all > 0. else math.nan,
        r2_by_state=r2_by_state,
        rmse=float(np.sqrt(np.mean(fit_error**2))),
        rmse_by_state=np.sqrt(np.mean(fit_error**2, axis=0)),
        sample_mean=sample_mean,
        sample_std=sample_std,
        fitted_mean=fitted_mean,
        mean_error=fitted_mean - sample_mean,
    )

    # PCA describes innovations around the fitted deterministic OU drift.  Any
    # finite-sample residual mean must not be resampled as an additional drift.
    res = res - np.mean(res, axis=0, keepdims=True)
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

    return AR1PCA(
        U0=U0,
        dU=dU,
        k=k,
        alpha=alpha,
        ndt=ndt,
        mean_dt=dtbar,
        state_center=state_center,
        state_scale=state_scale,
        vars_explained=vars_explained,
        var_explained=var_explained,
        fit_quality=fit_quality)


def fit_surface_first_xar1_pca(
    X: np.ndarray,
    time_y: np.ndarray,
    *,
    num_alpha: int | float | None = 0.9,
    rescale_var: bool = False,
    use_matrix_exp: bool = False,
) -> AR1PCA:
    r"""Fit stationary surface dynamics before adding the log-spot driver.

    The extended state is ``X=(s,h)``: the first coordinate ``s`` is log-spot
    and the remaining coordinates ``h`` are ML-SANOS implied-volatility
    parameters. The surface block is fitted without spot.
    Log-spot has a constant drift and no mean reversion.  Its innovation is
    regressed on the retained surface PCA scores; one orthogonal residual spot
    factor completes the driver.  Thus ``num_alpha`` counts surface factors,
    while the returned joint model has one additional spot factor.
    """

    verify_inp(
        X.ndim == 2 and X.shape[1] >= 2,
        lambda: (
            "surface-first fitting requires X=(s,h)=(log_spot,surface_state) with "
            f"at least two columns, got {X.shape}"
        ),
    )
    T, N = X.shape
    surface = X[:, 1:]
    surface_fit = fit_xar1_pca(
        surface,
        time_y,
        num_alpha=num_alpha,
        rescale_var=rescale_var,
        use_matrix_exp=use_matrix_exp,
    )
    ndt = surface_fit.ndt
    sqrt_ndt = np.sqrt(ndt)

    # A constant log-spot drift makes its standardized innovation mean zero.
    # The surface generator is autonomous, so only the surface block can have
    # an invariant distribution.
    dlog_spot = np.diff(X[:, 0])
    spot_drift = float(np.sum(dlog_spot) / np.sum(ndt))
    U0 = np.zeros((1, N), dtype=X.dtype)
    U0[0, 0] = spot_drift
    U0[0, 1:] = surface_fit.U0
    k = np.zeros((N, N), dtype=X.dtype)
    k[1:, 1:] = surface_fit.k

    if use_matrix_exp:
        X1_drift = _xar1_matrix_exp_drift(X[:-1], U0=U0, k=k, ndt=ndt)
    else:
        X1_drift = X[:-1] + ndt[:, None] * (U0 - X[:-1] @ k)
    residual = (X[1:] - X1_drift) / sqrt_ndt[:, None]
    residual -= np.mean(residual, axis=0, keepdims=True)

    surface_alpha = surface_fit.alpha
    spot_residual = residual[:, 0]
    spot_loading = np.linalg.lstsq(
        surface_alpha,
        spot_residual,
        rcond=None,
    )[0]
    idiosyncratic_spot = spot_residual - surface_alpha @ spot_loading
    idiosyncratic_spot -= np.mean(idiosyncratic_spot)

    surface_factors = surface_alpha.shape[1]
    alpha = np.concatenate(
        (surface_alpha, idiosyncratic_spot[:, None]),
        axis=1,
    )
    dU = np.zeros((surface_factors + 1, N), dtype=X.dtype)
    dU[:surface_factors, 0] = spot_loading
    dU[:surface_factors, 1:] = surface_fit.dU
    dU[-1, 0] = 1.0

    spot_total = float(np.sum(np.square(spot_residual)))
    surface_total = float(np.sum(np.square(residual[:, 1:])))
    total = spot_total + surface_total
    surface_cumulative = surface_fit.vars_explained * surface_total
    vars_explained = np.concatenate(
        (
            np.asarray([spot_total / total]),
            (spot_total + surface_cumulative) / total,
        )
    )
    var_explained = float(vars_explained[surface_factors])
    spot_surface_r2 = (
        1.0 - float(np.sum(np.square(idiosyncratic_spot))) / spot_total
        if spot_total > 0.0
        else math.nan
    )

    fit_error = X[1:] - X1_drift
    squared_error = np.sum(np.square(fit_error), axis=0)
    centered_target = X[1:] - np.mean(X[1:], axis=0, keepdims=True)
    total_squares = np.sum(np.square(centered_target), axis=0)
    r2_by_state = np.full(N, np.nan, dtype=float)
    nonconstant = total_squares > 0.0
    r2_by_state[nonconstant] = (
        1.0 - squared_error[nonconstant] / total_squares[nonconstant]
    )
    total_squares_all = float(np.sum(total_squares))
    sample_mean = np.mean(X, axis=0)
    sample_std = np.std(X, axis=0)
    assert surface_fit.fit_quality is not None
    fitted_mean = np.concatenate(
        (np.asarray([math.nan]), surface_fit.fit_quality.fitted_mean)
    )
    mean_error = fitted_mean - sample_mean
    fit_quality = AR1FitQuality(
        r2=(
            1.0 - float(np.sum(squared_error)) / total_squares_all
            if total_squares_all > 0.0
            else math.nan
        ),
        r2_by_state=r2_by_state,
        rmse=float(np.sqrt(np.mean(np.square(fit_error)))),
        rmse_by_state=np.sqrt(np.mean(np.square(fit_error), axis=0)),
        sample_mean=sample_mean,
        sample_std=sample_std,
        fitted_mean=fitted_mean,
        mean_error=mean_error,
    )

    assert surface_fit.state_center is not None
    assert surface_fit.state_scale is not None
    state_center = np.concatenate(
        (np.asarray([np.mean(X[:, 0])]), surface_fit.state_center)
    )
    state_scale = np.concatenate(
        (np.asarray([np.std(X[:, 0])]), surface_fit.state_scale)
    )
    scale_floor = math.sqrt(np.finfo(X.dtype).eps) * max(
        1.0, float(np.max(state_scale))
    )
    state_scale = np.maximum(state_scale, scale_floor)

    assert alpha.shape == (T - 1, surface_factors + 1)
    assert dU.shape == (surface_factors + 1, N)
    assert vars_explained.shape == (N,)
    assert np.max(np.abs(np.mean(alpha, axis=0))) < 1.0e-8
    return AR1PCA(
        U0=U0,
        alpha=alpha,
        dU=dU,
        k=k,
        ndt=ndt,
        mean_dt=surface_fit.mean_dt,
        state_center=state_center,
        state_scale=state_scale,
        vars_explained=vars_explained,
        var_explained=var_explained,
        fit_quality=fit_quality,
        surface_first=True,
        num_surface_alpha=surface_factors,
        surface_var_explained=surface_fit.var_explained,
        spot_surface_r2=spot_surface_r2,
        spot_residual_std=float(np.std(idiosyncratic_spot)),
    )

    
def plot_sampler(sampler):
    from cdxcore.dynaplot import colors_tableau, figure, m_o_m
    from IPython.display import Markdown, display
    from matplotlib.ticker import MaxNLocator

    market    = sampler.market
    model     = sampler.model
    dates     = market.dates
    num_alpha = sampler.num_alpha

    with figure(row_size=4, col_size=6, columns=6 ) as fig:
        display(Markdown(f"## {model.qualified_label}: using {sampler.num_alpha} Alphas for {sampler.num_xstate} Extended States"))
        display(Markdown(f"### Real and simulated states, discrete to normal randomness"))

        r = sampler.pcaxar1
        xstate = sampler.xstate
        ndt    = r.ndt[:,None]
        dX     = ( r.U0 - sampler.xstate[:-1,:] @ r.k ) * ndt + r.alpha @ r.dU * np.sqrt(ndt)
        recX   = np.concatenate( [ sampler.xstate[:1,:], sampler.xstate[:1,:] + np.cumsum( dX, axis=0 ) ], axis=0 )
        assert recX.shape == xstate.shape, f"{recX.shape} != {xstate.shape}"

        rec        = np.zeros((len(dates), xstate.shape[1], 5))
        rec[...,0] = xstate
        rec[...,1] = sampler.hist_sample_xstates( index=0, num_steps=len(dates), num_samples=0, bandwidth=0., replay=True )
        rec[...,2] = sampler.hist_sample_xstates( index=0, num_steps=len(dates), num_samples=0, bandwidth=0.1, replay=False )
        rec[...,3] = sampler.hist_sample_xstates( index=0, num_steps=len(dates), num_samples=0, bandwidth=0.5, replay=False )
        rec[...,4] = sampler.hist_sample_xstates( index=0, num_steps=len(dates), num_samples=0, bandwidth=1., replay=False )

        labels = ["Market", "0%", "10%", "50%", "100%"]
        
        ax_ = None
        ax = fig.add_subplot(f"State 0 (log-spot)")
        for j in range(rec.shape[-1]):
            ax.plot( dates, rec[:,0,j], "o" if j==0 else "-", label=labels[j])
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend(loc="upper right")
        for i in range(1,xstate.shape[1]):
            ax = fig.add_subplot(f"State {i}")
            for j, label in enumerate(labels):
                ax.plot( dates, rec[:,i,j], "o" if j==0 else "-", )
            ax.tick_params(axis="x", labelrotation=90)
    
    with figure(row_size=6, col_size=4, columns=6 ) as fig:
        display(Markdown(f"### {model.qualified_label} factors"))
        U0   = sampler.pcaxar1.U0
        dU   = sampler.pcaxar1.dU
        colors = colors_tableau()
        x    = np.arange(U0.shape[1])
        ax0  = fig.add_subplot("U0")
        ax0.bar( x, U0[0,:], width=0.2, color="black")
        for i in range( dU.shape[0] ):
            ax = fig.add_subplot(f"dU[{i}]")
            ax.bar( x, dU[i,:], width=0.2, color=next(colors))
            ax.sharex(ax0)
        ax0.set_ylim( m_o_m( U0, dU ) )
    
    with figure(row_size=5, col_size=5, columns=None ) as fig:
        display(Markdown(f"### {model.qualified_label} weights"))
        mev = sampler.pcaxar1.vars_explained
        
        ax = fig.add_subplot("Explained")
        fig.render()
        ax.plot( np.arange(0,len(mev)), mev*100., ":", color="orange", label="All")
        ax.plot( np.arange(0,num_alpha), mev[:num_alpha]*100., "-", color="red",label=f"First {num_alpha}")
        ax.set_ylabel("% of total variance explained")
        ax.set_xlabel("Number of Eigenvalues")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    with figure(row_size=6, col_size=4, columns=6 ) as fig:
        display(Markdown(f"### {model.qualified_label} weights"))
        
        ax_ = None
        colors = colors_tableau()
        alpha  = sampler.pcaxar1.alpha
        
        for i in range(alpha.shape[1]):
            ax = fig.add_subplot(f"Alpha {i}")
            ax.plot( dates[:-1], alpha[:,i], "*:", label="Model", color=next(colors))
            ax.tick_params(axis="x", labelrotation=90)
            if not ax_ is None:
                ax.sharey(ax_)
            else:
                ax_ = ax
        ax_.set_ylim( m_o_m(alpha) )

    display(Markdown(f"### {model.qualified_label} fit quality"))
    display(sampler.pcaxar1.fit_quality)
