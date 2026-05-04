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
import warnings as warnings

from dys.grid_sanos.geometry import GridGeometry
from dys.grid_sanos.backbone import BackboneNumpy

bbnp = BackboneNumpy()

@dataclass(frozen=True)
class GridFitterConfig:
    eps_prices: float = 1E-6     #: Tolerance for option price errors.
    eps_theta : float = 1E-6     #: Tolerance for theta errors during DLV computation.
    eps_gamma : float = 1E-6     #: Tolerance for gamma errors during DLV computation.
    eps_densities : float = 1E-6 #: Tolerance for density errors.
    eps_dlvsq : float = 1E-6     #: Tolerance for DLV squared errors being positive.

    cvxpy_solver : str = "CLARABEL" #:  Solver to use for the convex optimization problem in :meth:`fit_dlvsqs`, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options : dict = field(default_factory=dict) #: Options to pass to the CVXPY solver.

    max_inv_spread : float = 1000. #: Maximum weight for tight bid-ask spreads; the optimization problem will weight the errors by the inverse of the spread, but this parameter caps the maximum weight to avoid numerical issues with very tight spreads.
    mid_fit_weight : float = 0.1 #: Weight for fitting to the mid prices of options.

    dlv_max : float|None = 5.
    dlv_floor : float = 0.01 #: Minimum value for discrete local volatilities. The square of this will be added to input DLV squared.
    fwdvol_floor : float = 0.01 #: Minimum allowed value for forward volatilities. This will be added to forward volatilities.

    def __post_init__(self):
        assert self.eps_prices >= 0., f"Tolerance for prices must be positive, found {self.eps_prices}"
        assert self.eps_theta >= 0., f"Tolerance for theta must be positive, found {self.eps_theta}"
        assert self.eps_gamma >= 0., f"Tolerance for gamma must be positive, found {self.eps_gamma}"
        assert self.eps_densities >= 0., f"Tolerance for densities must be positive, found {self.eps_densities}"
        assert self.eps_dlvsq >= 0., f"Tolerance for dlvsq must be positive, found {self.eps_dlvsq}"

        assert self.max_inv_spread > 0., f"max_inv_spread must be positive, found {self.max_inv_spread}"
        assert self.mid_fit_weight >= 0., f"mid_fit_weight must be non-negative, found {self.mid_fit_weight}"
        assert self.dlv_floor >= 0., f"dlv_floor must be non-negative, found {self.dlv_floor}"
        assert self.dlv_max is None or self.dlv_max > self.dlv_floor, f"'dlv_max' must exceed 'dlv_floor'"
        assert self.fwdvol_floor >= 0., f"fwdvol_floor must be non-negative, found {self.fwdvol_floor}"

class GridFitter(object):
    """
    SANOS fixed grid DLV fitter.

    This class provides functionality to compute implied discrete local volatilities squaredgiven input option data,
    using the Grid SANOS framework.
    The main method is :meth:`__call__`, which takes in forward volatilities, option strikes, prices and times to expiry, and returns the implied DLVs squared. The method uses convex optimization to fit the model prices to the market prices while ensuring no-arbitrage conditions are satisfied, and then computes the DLVs from the fitted option prices.

    Parameters
    ---------
    geometry : :class:`dys.grid_sanos.geometry.GridGeometry`
        The geometry of the grid, including strikes and expiries.
    config : :class:`dys.grid_sanos.fitter.GridFitterConfig`
        Configuration parameters for the fitter, including solver options and tolerances.
    debug : bool, default ``False``
        Whether to enable debug mode, which performs additional checks and uses higher precision in some computations. This can be useful for diagnosing issues with the optimization or the DLV computation, but may slow down the computations.        
    """

    Config = GridFitterConfig

    def __init__(self, geometry : GridGeometry, config : GridFitterConfig, debug : bool = False ):
        self.geometry = geometry
        self.config   = config
        self.debug    = debug

    def __call__(self,   fwd_vols : np.array,
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
                         max_error : float = 1E-3,
                         return_only_dlvsqs : bool = True,
                         verbose : Context = Context.quiet ) -> np.array | tuple[np.array, np.array, np.array]:
        """
        Compute implied discrete local vols squared for given fwd_vols, option prices, time to expiry and strikes.
        **Note that this funtion returns DLV's squared.**

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

        opt_lix : array of shape (M,) | None, default ``None``
            Per option, the lower index into ``self.xtimes`` such that ``xtimes[lix]<t``. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_dweight : array of shape (M,) | None, default ``None``
            Per option, the density interpolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_vweight : array of shape (M,) | None, default ``None``
            Per option, the volatility interpolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
        opt_vextra : array of shape (M,) | None, default ``None``
            Per option, the volatility extrapolation weight. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.

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
        verbose : Context, default ``Context.quiet``        
            Verbose context for logging; this is used to log the optimization progress.
        
        Returns
        -------
        dlvsq : array of shape (T,N-2)
            Implied discrete local volatilities, squared, for each time step and strike outside the
            boundaries. c.f. https://arxiv.org/pdf/2601.11209
            
        (prices, densities) : tuple of arrays
            If ``return_only_dlvsqs`` is False, also return the fitted option prices
            and model densities for debugging purposes.
            Note that the ``prices`` are of the same dimension as the input data, while
            the model ``densities`` are of dimension ``(T,N)`` and represent the DLV model
            densities.
        """
        T, N       = self.geometry.TN
        assert len(opt_tte.shape) == 1, f"Expected opt_tte to be a 1D array, got shape {opt_tte.shape} (this function does not support batch dimensions)"

        if isinstance(fwd_vols, float):
            fwd_vols = np.full( (T,), fwd_vols, dtype=np.float64 )

        assert fwd_vols.shape == (T,), f"Expected fwd_vols to have shape {(T,)}, got {fwd_vols.shape} (this function does not support batch dimensions)"            
        assert opt_strikes.shape == opt_bid.shape == opt_ask.shape == opt_tte.shape, f"Expected 'opt_bid' and 'opt_ask' to have the same shape, got {opt_bid.shape} and {opt_ask.shape}"
        assert opt_is_calls is None or opt_is_calls.shape == opt_tte.shape, f"Expected 'opt_is_calls' to have the same shape as 'opt_tte', got {opt_is_calls.shape} and {opt_tte.shape}"
        assert np.min( fwd_vols ) >= 0., f"Forward volatilities must not be negative, found min {np.min(fwd_vols)}"

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

        if opt_lix is None and opt_dweight is None and opt_vweight is None and opt_vextra is None:
            opt_lix, opt_dweight, opt_vweight, opt_vextra = self.geometry.time_interpolation_weights( opt_tte=opt_tte, opt_mask=None )
        else:
            assert opt_lix is not None and opt_dweight is not None and opt_vweight is not None and opt_vextra is not None, "opt_lix, opt_dweight, opt_vweight and opt_vextra must be all None or all not None"
            assert opt_lix.shape == opt_dweight.shape == opt_vweight.shape == opt_vextra.shape == opt_tte.shape, f"Expected opt_lix, opt_dweight, opt_vweight and opt_vextra to have the same shape as opt_tte, got {opt_lix.shape}, {opt_dweight.shape}, {opt_vweight.shape}, {opt_vextra.shape} and {opt_tte.shape}"
        assert np.all( 0. <= opt_dweight ) and np.all( opt_dweight <= 1. ), f"opt_dweight must be between 0 and 1, found min {np.min(opt_dweight)} and max {np.max(opt_dweight)}"
        assert np.all( 0. <= opt_vweight ) and np.all( opt_vweight <= 1. ), f"opt_vweight must be between 0 and 1, found min {np.min(opt_vweight)} and max {np.max(opt_vweight)}"
        assert np.all( opt_vextra >= 0. ), f"opt_vextra must be non-negative, found min {np.min(opt_vextra)}"
    
        xtimes    = self.geometry.xtimes
        assert len(xtimes) == T+1, f"Expected xtimes to have length T+1, got {len(xtimes)} with T={T}"
        assert np.min(opt_lix) >= 0, f"opt_lix must be at least 0, found min {np.min(opt_lix)}"
        assert np.max(opt_lix) < T, f"opt_lix must be at most T-1, found max {np.max(opt_lix)} with T={T}"

        if self.geometry.vol_interp == "sqrt_lin_vol":
            totalSqrtVar = np.cumsum( (fwd_vols+self.config.fwdvol_floor) * self.geometry.dsqrtTs )
            assert totalSqrtVar.shape == (T,) and np.all( np.isfinite(totalSqrtVar) ) and np.all( totalSqrtVar >= 0. ), f"Internal error: totalSqrtVar must have shape {(T,)} and be non-negative and finite, found shape {totalSqrtVar.shape} and min {np.min(totalSqrtVar)} and max {np.max(totalSqrtVar)}"

            prev_sqrtVar = np.where( opt_lix == 0, 0., totalSqrtVar[np.clip(opt_lix-1,min=0)] )
            opt_sqrtVars = opt_vweight * prev_sqrtVar + (1.-opt_vweight) * totalSqrtVar[opt_lix] + opt_vextra * (fwd_vols[-1]+self.config.fwdvol_floor) 
        else:
            assert self.geometry.vol_interp == "lin_var", f"Internal error: vol_interp must be 'sqrt_lin_vol' or 'lin_var', found '{self.geometry.vol_interp}'"
            totalVar = np.cumsum( (fwd_vols+self.config.fwdvol_floor)**2 * self.geometry.dts )
            assert totalVar.shape == (T,) and np.all( np.isfinite(totalVar) ) and np.all( totalVar >= 0. ), f"Internal error: totalVar must have shape {(T,)} and be non-negative and finite, found shape {totalVar.shape} and min {np.min(totalVar)} and max {np.max(totalVar)}"

            prev_var     = np.where( opt_lix == 0, 0., totalVar[np.clip(opt_lix-1,min=0)] )
            opt_sqrtVars = np.sqrt( opt_vweight * prev_var + (1.-opt_vweight) * totalVar[opt_lix] + opt_vextra * ((fwd_vols[-1]+self.config.fwdvol_floor)**2) )
        assert opt_sqrtVars.shape == (maskedM,) and np.all( np.isfinite(opt_sqrtVars) ) and np.all( opt_sqrtVars >= 0. ), f"Internal error: opt_sqrtVars must have shape {(maskedM,)} and be non-negative and finite, found shape {opt_sqrtVars.shape} and min {np.min(opt_sqrtVars)} and max {np.max(opt_sqrtVars)}"

        # DLV SANOS
        # ---------

        mstrikes     = self.geometry.strikes
        spread       = opt_ask - opt_bid
        opt_mid      = 0.5*(opt_ask + opt_bid)
        assert np.min(opt_bid) > -self.config.eps_prices, f"Option bid must be positive, found min {np.min(opt_bid)} [tolerance 'prices' {self.config.eps_prices}]"
        assert np.min(spread) > -self.config.eps_prices, f"Option bid-ask spread must be positive, found min {np.min(spread)} [tolerance 'prices' {self.config.eps_prices}]"

        opt_call_mid = opt_mid if opt_is_calls is None else np.where(opt_is_calls, opt_mid, opt_mid + 1. - opt_strikes )
        opt_call_bid = opt_bid if opt_is_calls is None else np.where(opt_is_calls, opt_bid, opt_bid + 1. - opt_strikes )
        opt_call_ask = opt_ask if opt_is_calls is None else np.where(opt_is_calls, opt_ask, opt_ask + 1. - opt_strikes ) 
        del opt_bid, opt_ask, opt_mid

        # Optimization over densities
        # ---------------------------

        densities   = cp.Variable( (T,N), nonneg=True )
        train_dlvsqs= cp.Variable( (T,N-2), nonneg=True )
        conditions  = [ cp.sum(densities, axis=1) == 1. ] # density conditions
        t_mcalls    = []
        tm1_mcalls  = []

        # martingale property for X
        # -------------------------
        # see https://arxiv.org/pdf/2601.11209
        #

        for t in range(0,T):
            mdensity_t   = densities[t,:]
            mstrikes_t   = mstrikes[t,:]

            mcalls_of_t   = np.maximum( mstrikes_t[None,:] - mstrikes_t[:,None], 0. ) @ mdensity_t
            assert mcalls_of_t.shape == (N,), f"Expected mcalls_of_t to have shape {(N,)}, got {mcalls_of_t.shape}"
            t_mcalls.append( mcalls_of_t )

            if t>0:
                density_tm1   = densities[t-1,:]
                mstrikes_tm1  = mstrikes[t-1,:]
                mcalls_of_tm1 = np.maximum( mstrikes_tm1[None,:] - mstrikes_t[:,None], 0. ) @ density_tm1
                assert mcalls_of_tm1.shape == (N,), f"Expected mcalls_of_tm1 to have shape {(N,)}, got {mcalls_of_tm1.shape}"

            else:
                mcalls_of_tm1 = np.maximum( 0., 1. - mstrikes_t ) # at time 0, the call price is just the intrinsic value since the forward price is normalized to 1.
            conditions   += [ mdensity_t @ mstrikes_t == 1., mcalls_of_t >= mcalls_of_tm1 ]
            tm1_mcalls.append( mcalls_of_tm1 )

            # dlvs
            # We note that dC_t = 1/2 DLV^2 K^2 dT C''(K)
            # where C''(K) = 1/2 ( k[2:] - K[:-2]) p(k)
            # hence
            # dC_t = DLV^2 K^2/( k[2:] - K[:-2]) dt p(K)

            dC_t  = ( mcalls_of_t - mcalls_of_tm1 )[1:-1]
            dt    = self.geometry.times[t] - ( self.geometry.times[t-1] if t > 0 else 0.)
            Ksqdt = mstrikes_t[1:-1]**2 / ( mstrikes_t[2:] - mstrikes_t[:-2]) * dt

            if self.config.dlv_floor > 0.:
                conditions += [ dC_t >= cp.multiply( self.config.dlv_floor**2 * Ksqdt, mdensity_t[1:-1] ) ]
            if not self.config.dlv_max is None:
                conditions += [ dC_t <= cp.multiply( self.config.dlv_max**2 * Ksqdt, mdensity_t[1:-1] ) ]
            del dC_t, dt, Ksqdt

        del mdensity_t, mstrikes_t, mcalls_of_t, density_tm1, mstrikes_tm1, mcalls_of_tm1

        # computing arbitrary model prices
        # --------------------------------
        # 'opt_lix' is an index in T+1 xtimes; mstrikes has T expiries.

        eff_lix      = np.clip(opt_lix-1, min=0)
        calls_prev   = bbnp.bs_price(spot=mstrikes[eff_lix,:],
                                     strikes=opt_strikes[:,None], 
                                     sqrtVars=opt_sqrtVars[:,None],
                                     high_precision = True,
                                     debug_verify_bounds = self.debug
                                ) * opt_dweight[:,None]
        assert calls_prev.shape == (maskedM,N), f"Expected calls_prev to have shape {(maskedM,N)}, got {calls_prev.shape}"
        calls_prev  = cp.sum( cp.multiply( calls_prev, densities[eff_lix,:] ), axis=-1 )
        assert calls_prev.shape == (maskedM,), f"Expected calls_prev to have shape {(maskedM,)}, got {calls_prev.shape}"

        #eff_lix      = np.clip(eff_lix+1, max=T-1)
        calls_next   = bbnp.bs_price(spot=mstrikes[opt_lix,:],
                                     strikes=opt_strikes[:,None], 
                                     sqrtVars=opt_sqrtVars[:,None],
                                     high_precision = True,
                                     debug_verify_bounds = self.debug
                                ) * ( 1.- opt_dweight[:,None] )
        calls_next    = cp.sum( cp.multiply( calls_next, densities[opt_lix,:] ), axis=-1 )

        fit_calls     = calls_prev + calls_next 
        del calls_prev, calls_next

        inv_spread    = np.minimum( self.config.max_inv_spread, 1./spread )
        err_sqr       = cp.multiply( self.config.mid_fit_weight, cp.multiply( inv_spread, fit_calls - opt_call_mid )**2 ) if self.config.mid_fit_weight > 0. else 0.
        err_ask       = cp.multiply( inv_spread, cp.pos( fit_calls - opt_call_ask ) )
        err_bid       = cp.multiply( inv_spread, cp.pos( opt_call_bid - fit_calls ) )
        objective     = err_sqr + err_ask + err_bid + 0.00001 * cp.sum_squares( fdlvsq )

        with verbose.write_t(lambda : f"Solving for implied DLVs (squared) for {T} expiries and {N} strikes per expory... ", end='') as tme:
            prob        = cp.Problem( cp.Minimize( cp.sum( objective ) ), conditions )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.", category=UserWarning)
                try:
                    prob.solve( solver=self.config.cvxpy_solver, **self.config.cvxpy_solver_options )
                except cp.SolverError as e:
                    raise e
            
            fit_calls  = fit_calls.value
            densities  = densities.value

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
            t_mcalls     = [ tc.value for tc in t_mcalls ]
            tm1_mcalls   = [ ( tmc.value if not isinstance(tmc, np.ndarray) else tmc ) for tmc in tm1_mcalls ] # tm1_mcalls[0] is not a cvxpy expression, so we keep it as a numpy array 
            dts          = self.geometry.dts
            dlvsq        = np.zeros( (T,N-2), dtype=opt_call_bid.dtype )

            for t in range(T):
                # extract calls defined on the same strike range in t
                strikes_t = mstrikes[t,:]
                calls_t   = t_mcalls[t]
                calls_tm1 = tm1_mcalls[t]
                dT        = dts[t]

                Theta     = ( calls_t[1:-1] - calls_tm1[1:-1] ) / dT
                assert np.min(Theta) >= -self.config.eps_theta, f"Call price must increase with time to expiry, found min Theta {np.min(Theta)} at time {t} [tolerance 'theta' {self.config.eps_theta}]"
                assert np.all(np.isfinite(Theta)), f"Theta contains non-finite values at time {t}"
                Theta     = np.maximum( Theta, 0. ) 
                dC        = np.diff( calls_t ) / np.diff( strikes_t )
                p         = np.diff( dC )
                assert np.min(p) >= -self.config.eps_densities, f"Density error: found {np.min(p)} at time {t} [tolerance 'densities' {self.config.eps_densities}]"
                assert np.all(np.isfinite(p)), f"Density contains non-finite values at time {t}"
                Gamma     = p / ( strikes_t[2:] - strikes_t[:-2] )
                assert np.all(np.isfinite(Gamma)), f"Gamma contains non-finite values at time {t}"
                GammaKsq  = strikes_t[1:-1]**2 * Gamma
                assert np.min(GammaKsq) >= -self.config.eps_gamma, f"Gamma error: found {np.min(GammaKsq)} at time {t} [tolerance 'gamma' {self.config.eps_gamma}]"
                GammaKsq  = np.maximum( GammaKsq, 0. )
                valid     = (GammaKsq > 1E-8) & (Theta > 1E-8)
                Theta     = np.where( valid, Theta, 0. )
                GammaKsq  = np.where( valid, GammaKsq, 1. )
                dlvsq_t   = Theta / GammaKsq
                assert dlvsq_t.shape == (N-2,), f"Expected dlvsq_t to have shape {(N-2,)}, got {dlvsq_t.shape}"
                assert np.all(np.isfinite(dlvsq_t)), f"DLV squared contains non-finite values at time {t}"
                assert np.min(dlvsq_t) > -self.config.eps_dlvsq, f"DLV squared must be positive, found min {np.min(dlvsq_t)} [tolerance 'dlvsq' {self.config.eps_dlvsq}]"
                dlvsq[t,:]  = np.maximum( dlvsq_t, 0. )
            verbose.write("done; this tool {tme}.", head=False)

        # consistency step:
        # The __call__ function will add dlv_floor**2 to the input dlvsqs,
        # so here we substract it to be consistent
        dlvsq = np.clip( dlvsq-self.config.dlv_floor**2, min=0. ) 

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



















