"""
Grid Sanos Fitter

SANOS itself is a rather generic framework which can accomodate any number of model strikes and expires, potentially depending on market data for example when used in "interpolatnion mode" as in https://arxiv.org/pdf/2601.11209.
Grid SANOS, on the other hand, specifies SANOS on a fixed strike and expiry grid for a machine learned representation of an option surface. 

The framework here represents a fixed grid approach. The core class is :class:`dys.grid_sanos.GridGeometry`, which represents the geometry of the grid.

The ``GridFitter`` class here provides functionality to solve for market DLVs given input option data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
import math as math
from cdxcore.verbose import Context
from cdxcore.err import verify_inp
import warnings as warnings

from polars import Config

from quantdata.sanos._geometry import Geometry
from quantdata.utils.backbone import BackboneNumpy

bbnp = BackboneNumpy()

@dataclass
class Eps:
    prices    : float = 1E-6     #: Tolerance for option price errors.
    theta     : float = 1E-6     #: Tolerance for theta errors during DLV computation.
    gamma     : float = 1E-6     #: Tolerance for gamma errors during DLV computation.
    densities : float = 1E-6     #: Tolerance for density errors.
    dlvsq     : float = 1E-6     #: Tolerance for DLV squared errors being positive.

    def __post_init__(self):
        verify_inp( self.prices >= 0., f"Tolerance for prices must be non-negative, found {self.prices}" )
        verify_inp( self.theta >= 0., f"Tolerance for theta must be non-negative, found {self.theta}" )
        verify_inp( self.gamma >= 0., f"Tolerance for gamma must be non-negative, found {self.gamma}" )
        verify_inp( self.densities >= 0., f"Tolerance for densities must be non-negative, found {self.densities}" )
        verify_inp( self.dlvsq >= 0., f"Tolerance for dlvsq must be non-negative, found {self.dlvsq}" )

    def set_all(self, eps : float ):
        verify_inp( eps >= 0., f"Tolerance must be non-negative, found {eps}" )
        self.prices = eps
        self.theta = eps
        self.gamma = eps
        self.densities = eps
        self.dlvsq = eps

@dataclass(frozen=True)
class Fitter:
    """
    SANOS fixed grid DLV fitter.

    This class provides functionality to compute implied discrete local volatilities squaredgiven input option data,
    using the Grid SANOS framework.
    The main method is :meth:`__call__`, which takes in forward volatilities, option strikes, prices and times to expiry, and returns the implied DLVs squared. The method uses convex optimization to fit the model prices to the market prices while ensuring no-arbitrage conditions are satisfied, and then computes the DLVs from the fitted option prices.
    """

    geometry       : Geometry  #: Geometry of the SANOS problem
    eps            : Eps = field(default_factory=Eps) #: Tolerances for various error checks during fitting and DLV computation.
    cvxpy_solver   : str = "CLARABEL" #:  Solver to use for the convex optimization problem in :meth:`fit_dlvsqs`, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options : dict = field(default_factory=dict) #: Options to pass to the CVXPY solver.
    max_inv_spread : float = 1000. #: Maximum weight for tight bid-ask spreads; the optimization problem will weight the errors by the inverse of the spread, but this parameter caps the maximum weight to avoid numerical issues with very tight spreads.
    mid_fit_weight : float = 0.1 #: Weight for fitting to the mid prices of options.
    debug          : bool = False #: Whether to enable debug mode, which performs additional checks and logs more information during fitting. This can be useful for development and debugging, but may slow down the computation.

    def __post_init__(self):
        assert self.max_inv_spread > 0., f"max_inv_spread must be positive, found {self.max_inv_spread}"
        assert self.mid_fit_weight >= 0., f"mid_fit_weight must be non-negative, found {self.mid_fit_weight}"

    def __call__(self,  fwd_vols : np.array|float|None,
                        opt_strikes : np.array,
                        opt_bid : np.array,
                        opt_ask : np.array,
                        opt_tte : np.array,*,
                        opt_is_calls : np.array|None = None,    
                        opt_mask : np.array|None = None,
                        max_error : float|None = 1E-3,
                        return_only_dlvsqs : bool = True,
                        always_return_lists : bool = False,
                        verbose : Context = Context.quiet ) -> np.array | tuple[np.array, np.array, np.array] | list[np.array] | tuple[np.array, list[np.array], list[np.array]]:
        """
        Fit discrete local vols squared for given fwd_vols to option prices, expires and strikes.
        **Note that this funtion returns DLV's squared.**

        The returned DLVs is net of ``config.dlv_min**2``
        to ensure reconstruction is consistent with the fit.

        **Returning Arrays vs Lists**
        
        This function operates differently on geometry strikes given as a rectangular numpy grid or a list of numpy arrays of strikes.
        The former is more efficient in which case the returned DLV squared (and, if requested, the densities) will
        also returned as a grid. In the latter case, the returned DLV squared and densities will be lists of arrays of different lengths per expiry,
        corresponding to the different number of strikes per expiry. You can always return lists using ``always_return_lists=True``.

        Parameters
        ----------
        fwd_vols : array of shape (T,)
            Forward volatilities for each time step. ``T`` must match ``self.T``, the number of non-zero model expiries.
        opt_strikes : array of shape (M,)
            Strikes of the options to price.
        opt_bid : array of shape (M,)
            Bid prices of the options to price.
        opt_ask : array of shape (M,)
            Ask prices of the options to price.
        opt_tte : array of shape (M,)
            Time to expiry of the options to price.
        opt_is_calls : array of shape (M,) | None, default ``None``
            Boolean array indicating whether each option is a call (True) or put (False). If ``None``, all options are assumed to be calls.
        opt_mask : array of shape (M,) | None, default ``None``
            Mask for which options to price.

        max_error : float | None, default 1E-3
            Maximum allowed error between model and market prices; the function will raise a :class:`RuntimeError`
            if violated.
            This error indicates by how much the model price exceeds bid/ask, measured in spreads. It is the sum of:

            * ``Relu( model_price - opt_ask ) / spread``
            * ``Relu( opt_bid - model_price ) / spread``

            Set to ``None`` to disable this check.

        return_only_dlvsqs : bool, default ``True``
            Whether to return only the implied DLVs squared,
            or also the fitted option prices and densities for debugging purposes.
            If ``True``, only the implied DLVs squared are returned;
            if ``False``, a tuple of (dlvsq, prices, densities) is returned.
        always_return_lists : bool, default ``False``
            Whether to always return lists of arrays for DLVs and densities, even when the geometry strikes are given as a numpy grid.
            This can be useful for consistency in downstream code, but may be less efficient.
        verbose : :class:`cdxcore.verbose.Context`, default ``Context.quiet``        
            Verbose context for logging; this is used to log the optimization progress.
        
        Returns
        -------
        dlvsq : array of shape (T,N-2) or list of T arrays of shape (N_t-2,)
            Implied discrete local volatilities, squared, for each time step and strike outside the
            boundaries. c.f. https://arxiv.org/pdf/2601.11209.
            
            The returned DLVs is net of ``config.dlv_min**2``
            to ensure reconstruction is consistent with the fit.
            
        (prices, densities) : tuple of arrays
            If ``return_only_dlvsqs`` is False, also return the fitted option prices
            and model densities for debugging purposes.
            Note that the ``prices`` are of the same dimension as the input data, while
            the model ``densities`` are of dimension ``(T,N)`` or a list of T arrays of shape (N_t,)
            and represent the DLV model densities.
        """
        return self.fit_prepared( 
                                fwd_vols=fwd_vols,
                                opt_strikes=opt_strikes,
                                opt_bid=opt_bid,
                                opt_ask=opt_ask,
                                opt_tte=opt_tte,
                                opt_is_calls=opt_is_calls,
                                opt_mask=opt_mask,
                                opt_lix=None,
                                opt_dweight=None, 
                                opt_vweight=None,
                                opt_vextra=None,
                                max_error=max_error,
                                return_only_dlvsqs=return_only_dlvsqs,
                                always_return_lists=always_return_lists,
                                verbose=verbose )

    def fit_prepared(self,  
                    fwd_vols : np.array|float|None,
                    opt_strikes : np.array,
                    opt_bid : np.array,
                    opt_ask : np.array,
                    opt_tte : np.array,*,
                    opt_is_calls : np.array | None = None,    
                    opt_mask : np.array | None = None,
                    opt_lix : np.array | None = None, 
                    opt_dweight : np.array | None = None, 
                    opt_vweight : np.array | None = None,
                    opt_vextra : np.array | None = None,
                    max_error : float|None = 1E-3,
                    return_only_dlvsqs : bool = True,
                    always_return_lists : bool = False,
                    verbose : Context = Context.quiet ) -> np.array | tuple[np.array, np.array, np.array] | list[np.array] | tuple[np.array, list[np.array], list[np.array]]:
        """
        Extedended version of :meth:`__call__` which allows passing in pre-computed time interpolation weights and indices,
        which can speed up the computation if fitting needs to be done repeatedly for the same set of options but different forward volatilities.
        See :func:`time_interpolation_weights` on how to compute these.

        Additional Parameters
        ---------------------
        
        opt_lix : array of shape (M,) | None, default ``None``
            Per option, the lower index into ``self.xtimes`` such that ``xtimes[lix]<t``. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_dweight : array of shape (M,) | None, default ``None``
            Per option, the density interpolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_vweight : array of shape (M,) | None, default ``None``
            Per option, the volatility interpolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_vextra : array of shape (M,) | None, default ``None``
            Per option, the volatility extrapolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        
        Returns
        -------
        Results : type
            See :meth:`__call__` for details.
        """
        T  = self.geometry.T
        N  = self.geometry.strikes.shape[1]-2 if isinstance(self.geometry.strikes, np.ndarray) else None

        if isinstance(fwd_vols, float):
            fwd_vols = np.full( (T,), fwd_vols, dtype=opt_strikes.dtype )

        verify_inp( len(opt_tte.shape) == 1, f"Expected opt_tte to be a 1D array, found shape {opt_tte.shape} (this function does not support batch dimensions)" )
        verify_inp( fwd_vols is None or fwd_vols.shape == (T,), f"Expected fwd_vols to have shape {(T,)}, found {fwd_vols.shape}" )            
        verify_inp( opt_strikes.shape == opt_bid.shape == opt_ask.shape == opt_tte.shape, f"Expected 'opt_strikes', 'opt_bid', 'opt_ask' and 'opt_tte' to have the same shape; found {opt_strikes.shape}, {opt_bid.shape}, {opt_ask.shape}, {opt_tte.shape}" )
        verify_inp( opt_is_calls is None or opt_is_calls.shape == opt_tte.shape, f"Expected 'opt_is_calls' to have the same shape as 'opt_tte'; found {opt_is_calls.shape} and {opt_tte.shape}" )
        verify_inp( fwd_vols is None or np.min( fwd_vols ) >= 0., f"Forward volatilities must not be negative, found min {np.min(fwd_vols)}" )

        # mask by removing options
        # ------------------------

        if not opt_mask is None:
            opt_strikes = opt_strikes[opt_mask]
            opt_bid     = opt_bid[opt_mask]
            opt_ask     = opt_ask[opt_mask]
            opt_tte     = opt_tte[opt_mask]
            opt_is_calls= opt_is_calls[opt_mask] if opt_is_calls is not None else None
            opt_lix     = opt_lix[opt_mask] if opt_lix is not None else None
            opt_dweight = opt_dweight[opt_mask] if opt_dweight is not None else None
            opt_vweight = opt_vweight[opt_mask] if opt_vweight is not None else None
            opt_vextra  = opt_vextra[opt_mask] if opt_vextra is not None else None

        maskedM = len(opt_strikes)

        # compute sqrt var per option
        # ---------------------------

        opt_sqrtVars = self.geometry.interpolate_sqrtVars( fwd_vols, opt_lix, opt_vweight, opt_vextra ) if fwd_vols is not None else None
        verify_inp( opt_sqrtVars is None or (opt_sqrtVars.shape == (maskedM,) and np.all( np.isfinite(opt_sqrtVars) ) and np.all( opt_sqrtVars >= 0. )), f"Internal error: opt_sqrtVars must have shape {(maskedM,)} and be non-negative and finite, found shape {opt_sqrtVars.shape} and min {np.min(opt_sqrtVars)} and max {np.max(opt_sqrtVars)}" )

        # DLV SANOS
        # ---------

        spread       = opt_ask - opt_bid
        opt_mid      = 0.5*(opt_ask + opt_bid)
        verify_inp( np.min(opt_bid) > -self.config.eps.prices, f"Option bid must be positive, found min {np.min(opt_bid)} [tolerance 'prices' {self.config.eps.prices}]" )
        verify_inp( np.min(spread) > -self.config.eps.prices, f"Option bid-ask spread must be positive, found min {np.min(spread)} [tolerance 'prices' {self.config.eps.prices}]" )

        opt_call_mid = opt_mid if opt_is_calls is None else np.where(opt_is_calls, opt_mid, opt_mid + 1. - opt_strikes )
        opt_call_bid = opt_bid if opt_is_calls is None else np.where(opt_is_calls, opt_bid, opt_bid + 1. - opt_strikes )
        opt_call_ask = opt_ask if opt_is_calls is None else np.where(opt_is_calls, opt_ask, opt_ask + 1. - opt_strikes ) 
        del opt_bid, opt_ask, opt_mid

        # Optimization over densities
        # ---------------------------

        xtimes      = self.geometry.xtimes
        assert len(xtimes) == T+1, f"Internal error: expected xtimes to have length T+1, got {len(xtimes)} with T={T}"
        xstrikes    = self.geometry.xstrikes
        all_dens    = None
        mxstrikes   = None # (T+1,N)
        conditions  = []
        Ns          = [ N ]*T if not N is None else [ xstrikes[t+1].shape[0]-2 for t in range(T) ]

        if not N is None:
            all_dens    = cp.Variable( (T,N), nonneg=True )
            densities   = [ all_dens[t,:] for t in range(T) ]            
            mxstrikes   = [ xstrikes[t,1:-1].astype(np.float6) for t in range(T+1) ]
            conditions  = [ cp.sum(all_dens, axis=1) == 1. ] # density conditions
        else:
            all_dens    = None
            densities   = [ cp.Variable( (T,xstrikes[t].shape[0]-2), nonneg=True ) for t in range(T) ]
            mxstrikes   = [ xstrikes[t][1:-1].astype(np.float64) for t in range(T+1) ]
            conditions  = [ cp.sum(d) == 1. for d in densities ] # density conditions

        all_mcalls_t     = []
        all_mcalls_tm1   = []

        # martingale property for X
        # -------------------------
        # see https://arxiv.org/pdf/2601.11209
        #

        for t in range(0,T):
            mdensity_t   = densities[t]
            mstrikes_tm1 = mxstrikes[t]
            mstrikes_t   = mxstrikes[t+1]

            mcalls_of_t   = np.maximum( mstrikes_t[None,:] - mstrikes_t[:,None], 0. ) @ mdensity_t
            assert mcalls_of_t.shape == (N,), f"Expected mcalls_of_t to have shape {(N,)}, found {mcalls_of_t.shape}"
            all_mcalls_t.append( mcalls_of_t )

            if t>0:
                density_tm1   = densities[t-1]
                mcalls_of_tm1 = np.maximum( mstrikes_tm1[None,:] - mstrikes_t[:,None], 0. ) @ density_tm1
                assert mcalls_of_tm1.shape == mstrikes_tm1.shape, f"Expected mcalls_of_tm1 to have shape {mstrikes_tm1.shape}, found {mcalls_of_tm1.shape}"
            else:
                mcalls_of_tm1 = np.maximum( 0., 1. - mstrikes_t ) # at time 0, the call price is just the intrinsic value since the forward price is normalized to 1.
            conditions   += [ mdensity_t @ mstrikes_t == 1., mcalls_of_t >= mcalls_of_tm1 ]
            all_mcalls_tm1.append( mcalls_of_tm1 )

            # dlvs
            # We note that dC_t = 1/2 DLV^2 K^2 dT C''(K)
            # where C''(K) = 1/2 ( k[2:] - K[:-2]) p(k)
            # hence
            # dC_t = DLV^2 K^2/( k[2:] - K[:-2]) dt p(K)

            dC_t  = ( mcalls_of_t - mcalls_of_tm1 )[1:-1]
            dt    = self.geometry.xtimes[t+1] - self.geometry.xtimes[t]
            Ksqdt = mstrikes_t[1:-1]**2 / ( mstrikes_t[2:] - mstrikes_t[:-2]) * dt

            if self.geometry.dlv_min > 0.:
                conditions += [ dC_t >= cp.multiply( self.geometry.dlv_min**2 * Ksqdt, mdensity_t[1:-1] ) ]
            if not self.geometry.dlv_max is None:
                conditions += [ dC_t <= cp.multiply( self.geometry.dlv_max**2 * Ksqdt, mdensity_t[1:-1] ) ]
            del dC_t, dt, Ksqdt

        del mdensity_t, mstrikes_t, mcalls_of_t, density_tm1, mstrikes_tm1, mcalls_of_tm1

        # computing arbitrary model prices
        # --------------------------------
        # 'opt_lix' is an index in T+1 xtimes; mstrikes has T expiries.

        def call_prices( *, spots, strikes, sqrtVars ):
            if sqrtVars is None:
                return np.maximum( spots - strikes, strikes.dtype.type(0.) )
            else:
                return bbnp.bs_price(spot=spots,
                                     strikes=strikes,
                                     sqrtVars=sqrtVars,
                                     high_precision = True,
                                     debug_verify_bounds = self.debug
                                )

        def compute_fit_calls():
            mstrikes     = self.geometry.strikes
            if isinstance(mstrikes, np.ndarray):
                assert not N is None, "Internal error: expected N to be not None"
                assert not all_dens is None, "Internal error: expected not all_dens to be None"
                N            = mstrikes.shape[1]
                eff_lix      = np.clip(opt_lix-1, min=0)
                calls_prev   = call_prices(spots=mstrikes[eff_lix,:],
                                          strikes=opt_strikes[:,None], 
                                          sqrtVars=opt_sqrtVars[:,None]
                                        ) * opt_dweight[:,None]
                assert calls_prev.shape == (maskedM,N), f"Expected calls_prev to have shape {(maskedM,N)}, got {calls_prev.shape}"
                calls_prev  = cp.sum( cp.multiply( calls_prev, all_dens[eff_lix,:] ), axis=-1 )
                assert calls_prev.shape == (maskedM,), f"Expected calls_prev to have shape {(maskedM,)}, got {calls_prev.shape}"

                calls_next   = call_prices(spots=mstrikes[opt_lix,:],
                                          strikes=opt_strikes[:,None], 
                                          sqrtVars=opt_sqrtVars[:,None]
                                        ) * ( 1.- opt_dweight[:,None] )
                calls_next    = cp.sum( cp.multiply( calls_next, all_dens [opt_lix,:] ), axis=-1 )
                fit_calls     = calls_prev + calls_next 
                del calls_prev, calls_next, mstrikes, eff_lix

            else:
                assert all_dens is None, "Internal error: expected all_dens to be None"
                assert N is None, "Internal error: expected N to be None"
                unique__lix = np.unique(opt_lix)
                fit_calls   = np.zeros_like( opt_call_mid )
                mstrikes    = self.geometry.strikes

                for lix in unique__lix:
                    opt_lix_mask = (opt_lix == lix)
                    lix_strikes  = mstrikes[opt_lix_mask][:,None]
                    lix_sqrtVars = opt_sqrtVars[opt_lix_mask][:,None]
                    lix_weight   = opt_dweight[opt_lix_mask,None]
                    lixM         = np.sum(opt_lix_mask)

                    eff_lix      = max(0, lix-1)
                    eff_spot     = mstrikes[eff_lix,:]
                    eff_N        = eff_spot.shape[0]
                    calls_prev   = call_prices(spots=eff_spot[None,:],
                                               strikes=lix_strikes, 
                                               sqrtVars=lix_sqrtVars,
                                            ) * lix_weight
                    assert calls_prev.shape == (lixM,eff_N), f"Expected calls_prev to have shape {(lixM,eff_N)}, got {calls_prev.shape}"
                    calls_prev  = cp.sum( cp.multiply( calls_prev, densities[eff_lix] ), axis=-1 )
                    assert calls_prev.shape == (lixM,), f"Expected calls_prev to have shape {(lixM,)}, got {calls_prev.shape}"
                    del eff_lix, eff_spot, eff_N

                    lix_spot     = mstrikes[lix,:]
                    lix_N        = lix_spot.shape[0]
                    calls_next   = call_prices( spots=lix_spot[None,:],
                                                strikes=lix_strikes, 
                                                sqrtVars=lix_sqrtVars,
                                            ) * ( 1.- lix_weight )
                    assert calls_next.shape == (lixM,lix_N), f"Expected calls_next to have shape {(lixM,lix_N)}, got {calls_next.shape}"
                    calls_next    = cp.sum( cp.multiply( calls_next, densities[opt_lix] ), axis=-1 )
                    assert calls_next.shape == (lixM,), f"Expected calls_next to have shape {(lixM,)}, got {calls_next.shape}"
                    fit_calls[opt_lix_mask] = calls_prev + calls_next
                    del calls_prev, calls_next, mstrikes
                    del lix_spot, lix_N
                    del lix_strikes, lix_sqrtVars, lix_weight, lixM, opt_ask
            return fit_calls
        
        fit_calls     = compute_fit_calls()
        inv_spread    = np.minimum( self.config.max_inv_spread, 1./spread )
        err_sqr       = cp.multiply( self.config.mid_fit_weight, cp.multiply( inv_spread, fit_calls - opt_call_mid )**2 ) if self.config.mid_fit_weight > 0. else 0.
        err_ask       = cp.multiply( inv_spread, cp.pos( fit_calls - opt_call_ask ) )
        err_bid       = cp.multiply( inv_spread, cp.pos( opt_call_bid - fit_calls ) )
        objective     = err_sqr + err_ask + err_bid 

        with verbose.write_t(lambda : f"Solving for implied DLVs (squared)... ", end='') as tme:
            prob        = cp.Problem( cp.Minimize( cp.sum( objective ) ), conditions )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.", category=UserWarning)
                try:
                    prob.solve( solver=self.config.cvxpy_solver, **self.config.cvxpy_solver_options )
                except cp.SolverError as e:
                    raise e
            
            fit_calls  = fit_calls.value
            densities  = [ d.value for d in densities ] if all_dens is None else all_dens.value

            err_sqr    = np.max( err_sqr.value ) if self.config.mid_fit_weight > 0. else 0.
            err_ask    = np.max( err_ask.value )
            err_bid    = np.max( err_bid.value )
            err        = max( err_ask, err_bid )

            if not max_error is None and err > max_error:
                if prob.status == "optimal":    
                    raise RuntimeError(f"Solver {self.config.cvxpy_solver} failed to find a solution within the specified error tolerance: maximum weighted error {err:.4e} (ask {err_ask:.4e}, bid {err_bid:.4e}, sqr {err_sqr:.4g}) exceeds boundary {max_error:.4e}.")
                raise RuntimeError(f"Solver {self.config.cvxpy_solver} failed to find a solution within the specified error tolerance: maximum weighted error {err:.4e} (ask {err_ask:.4e}, bid {err_bid:.4e}, sqr {err_sqr:.4g}) exceeds boundary {max_error:.4e}. Solver returned status '{prob.status}'.")
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"Solver {self.config.cvxpy_solver} returned '{prob.status}' status. However, maximum weighted error is {err:.4e} below the boundary {max_error:.4e}, so we continue.")
            verbose.write(f"done: maximum weighted error is {err:.4g}. this tool {tme}.", head=False)

        # computing DLVS
        # we use the call prices computed earlier
        with verbose.write_t(lambda : f"Computing DLV squared's... ", end='') as tme:
            all_mcalls_t   = [ tc.value for tc in all_mcalls_t ]
            all_mcalls_tm1 = [ ( tmc.value if not isinstance(tmc, np.ndarray) else tmc ) for tmc in all_mcalls_tm1 ] # all_mcalls_tm1[0] is not a cvxpy expression, so we keep it as a numpy array 
            dts            = np.diff( self.geometry.xtimes )
            dlvsq          = np.zeros( (T,N-2), dtype=opt_call_bid.dtype ) if not N is None else []
            mstrikes      = self.geometry.strikes
            
            for t in range(T):
                # extract calls defined on the same strike range in t
                Nt        = Ns[t]
                strikes_t = mstrikes[t,:] if not N is None else mstrikes[t]
                calls_t   = all_mcalls_t[t]
                calls_tm1 = all_mcalls_tm1[t]
                dT        = dts[t]

                Theta     = ( calls_t[1:-1] - calls_tm1[1:-1] ) / dT
                assert np.min(Theta) >= -self.config.eps.theta, f"Call price must increase with time to expiry, found min Theta {np.min(Theta)} at time {t} [tolerance 'theta' {self.config.eps.theta}]"
                assert np.all(np.isfinite(Theta)), f"Theta contains non-finite values at time {t}"
                Theta     = np.maximum( Theta, 0. ) 
                dC        = np.diff( calls_t ) / np.diff( strikes_t )
                p         = np.diff( dC )
                assert np.min(p) >= -self.config.eps.densities, f"Density error: found {np.min(p)} at time {t} [tolerance 'densities' {self.config.eps.densities}]"
                assert np.all(np.isfinite(p)), f"Density contains non-finite values at time {t}"
                Gamma     = p / ( strikes_t[2:] - strikes_t[:-2] )
                assert np.all(np.isfinite(Gamma)), f"Gamma contains non-finite values at time {t}"
                GammaKsq  = strikes_t[1:-1]**2 * Gamma
                assert np.min(GammaKsq) >= -self.config.eps.gamma, f"Gamma error: found {np.min(GammaKsq)} at time {t} [tolerance 'gamma' {self.config.eps.gamma}]"
                GammaKsq  = np.maximum( GammaKsq, 0. )
                valid     = (GammaKsq > 1E-8) & (Theta > 1E-8)
                Theta     = np.where( valid, Theta, 0. )
                GammaKsq  = np.where( valid, GammaKsq, 1. )
                dlvsq_t   = Theta / GammaKsq
                assert dlvsq_t.shape == (N-2,), f"Expected dlvsq_t to have shape {(N-2,)}, got {dlvsq_t.shape}"
                assert np.all(np.isfinite(dlvsq_t)), f"DLV squared contains non-finite values at time {t}"
                assert np.min(dlvsq_t) > -self.config.eps.dlvsq, f"DLV squared must be positive, found min {np.min(dlvsq_t)} [tolerance 'dlvsq' {self.config.eps.dlvsq}]"

                # consistency step:
                # The __call__ function will add dlv_min**2 to the input dlvsqs,
                # so here we substract it to be consistent
                dlvsq_t   = np.maximum( dlvsq_t-self.geometry.dlv_min**2, 0. )

                if N is None:
                    dlvsq[t,:]  = dlvsq_t
                else:
                    dlvsq.append( dlvsq_t )
            verbose.write("done; this tool {tme}.", head=False)

        if always_return_lists:
            dlvsq = [ dlvsq[t,:] for t in range(T) ] if not N is None else dlvsq
            densities = [ densities[t] for t in range(T) ] if all_dens is None else [ all_dens[t,:] for t in range(T) ]

        if return_only_dlvsqs:
            return dlvsq
        
        fit_prices = np.where( opt_is_calls, fit_calls, fit_calls - 1. + opt_strikes ) if opt_is_calls is not None else fit_calls    
        assert fit_prices.shape == (maskedM,), f"Expected final fit_prices to have shape {(maskedM,)}, got {fit_prices.shape}"

        if not opt_mask is None:
            fit_prices_ = np.zeros( opt_mask.shape, dtype=fit_prices.dtype )
            fit_prices_[opt_mask] = fit_prices
            fit_prices = fit_prices_
            del fit_prices_

        assert densities.shape == (T,N), f"Expected final densities to have shape {(T,N)}, got {densities.shape}"
        return dlvsq, fit_prices, densities




