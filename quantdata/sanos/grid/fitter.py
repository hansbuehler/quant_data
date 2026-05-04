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

from quantdata.sanos.grid import GridGeometry, GridSanos, Eps
from quantdata.utils.backbone import BackboneNumpy

bbnp = BackboneNumpy()

@dataclass(frozen=True)
class Fitter:
    """
    SANOS fixed grid DLV fitter.

    This class provides functionality to compute implied discrete local volatilities squared given input option data,
    using the Grid SANOS framework.
    The main method is :meth:`__call__`, which takes in forward volatilities, option strikes, prices and times to expiry, and returns the implied DLVs squared
    The method uses convex optimization to fit the model prices to the market prices while ensuring no-arbitrage conditions are satisfied, and then computes the DLVs from the fitted option prices.
    """

    geometry             : GridGeometry  #: GridGeometry of the SANOS problem
    eps                  : Eps = field(default_factory=Eps) #: Tolerances for various error checks during fitting and DLV computation.
    cvxpy_solver         : str = "CLARABEL" #:  Solver to use for the convex optimization problem in :meth:`fit_dlvsqs`, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options : dict = field(default_factory=dict) #: Options to pass to the CVXPY solver.
    min_half_spread      : float = 1E-5 #: Maximum weight for tight bid-ask spreads; the optimization problem will weight the errors by the inverse of the spread, but this parameter caps the maximum weight to avoid numerical issues with very tight spreads.
    mid_fit_weight       : float = 0.1 #: Weight for fitting to the mid prices of options. Set to 1. to only fit the mid price.
    debug                : bool = False #: Whether to enable debug mode, which performs additional checks and logs more information during fitting. This can be useful for development and debugging, but may slow down the computation.
    max_error            : float|None = 1E-3 #: max{ (fit-ask)^+ + (bid-fit)^+} / half_spread > max_error, raise a :class:`RuntimeError`. Set to None to disable this check.
    no_warnings          : bool = False #: turn off warnings

    def __post_init__(self):
        verify_inp( self.min_half_spread > 0., f"min_half_spread must be positive, found {self.min_half_spread}" )
        verify_inp( 0. <= self.mid_fit_weight <= 1., f"mid_fit_weight must be between 0 and 1, found {self.mid_fit_weight}" )

    @property
    def T(self) -> int:
        return self.geometry.T
    @property
    def N(self) -> int|None:
        return self.geometry.N
    @property
    def TN(self) -> tuple[int, int|None]:
        return self.geometry.TN

    def __call__(self,  fwd_vols     : np.array|float|None,
                        opt_strikes  : np.array,
                        opt_bid      : np.array,
                        opt_ask      : np.array,
                        opt_tte      : np.array,*,
                        opt_is_calls : np.array|None = None,    
                        opt_mask     : np.array|None = None,
                        return_only_dlvsqs : bool = True,
                        verbose : Context = Context.quiet ) -> np.array | tuple[np.array, np.array, np.array] | list[np.array] | tuple[np.array, list[np.array], list[np.array]]:
        """
        Fit discrete local vols squared for given fwd_vols to option prices, expires and strikes.
        **Note that this funtion returns DLV's squared.**

        The returned DLVs is net of ``geometry.dlv_min**2``
        to ensure reconstruction is consistent with the fit.

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
        return_only_dlvsqs : bool, default ``True``
            Whether to return only the implied DLVs squared,
            or also the fitted option prices and densities for debugging purposes.
            If ``True``, only the implied DLVs squared are returned;
            if ``False``, a tuple of (dlvsq, prices, densities) is returned.
        verbose : :class:`cdxcore.verbose.Context`, default ``Context.quiet``        
            Verbose context for logging; this is used to log the optimization progress.
        
        Returns
        -------
        dlvsq : array of shape (T,N-2)
            Implied discrete local volatilities, squared, for each time step and strike outside the
            boundaries. c.f. https://arxiv.org/pdf/2601.11209.
            
            The returned DLVs is net of ``geometry.dlv_min**2``
            to ensure reconstruction is consistent with the fit.
            
        (prices, densities) : tuple of arrays
            If ``return_only_dlvsqs`` is False, also return the fitted option prices
            and model densities for debugging purposes.
            Note that the ``prices`` are of the same dimension as the input data, while
            the model ``densities`` are of dimension ``(T,N)``
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
                                opt_interpolate=None,
                                return_only_dlvsqs=return_only_dlvsqs,
                                verbose=verbose )

    def fit_prepared(self,  
                    fwd_vols : np.array|float|None,
                    opt_strikes : np.array,
                    opt_bid : np.array,
                    opt_ask : np.array,
                    opt_tte : np.array,*,
                    opt_is_calls : np.array | None = None,    
                    opt_mask : np.array | None = None,
                    opt_interpolate : dict[str|np.ndarray]|None = None,
                    return_only_dlvsqs : bool = True,
                    verbose : Context = Context.quiet ) -> np.array | tuple[np.array, np.array, np.array] | list[np.array] | tuple[np.array, list[np.array], list[np.array]]:
        """
        Extedended version of :meth:`__call__` which allows passing in pre-computed time interpolation weights and indices,
        which can speed up the computation if fitting needs to be done repeatedly for the same set of options but different forward volatilities.
        See :func:`time_interpolation_weights` on how to compute these.

        Additional Parameters
        ---------------------
        
        opt_lix : array of shape (M,) | None, default ``None``
            Per option, the lower index into ``self.xexpiries`` such that ``xexpiries[lix]<t``. This can be computed using the ``time_interpolation_weights`` function. If ``None``, this will be computed on the fly.
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
        T, N = self.TN

        if not fwd_vols is None:
            if hasattr(fwd_vols, "shape"):
                verify_inp( fwd_vols.shape == (T,), lambda: f"Expected fwd_vols to have shape {(T,)}, found {fwd_vols.shape}" )
                verify_inp( np.min( fwd_vols ) >= 0., lambda: f"Forward volatilities must not be negative, found min {np.min(fwd_vols)}" )
            elif fwd_vols == 0.:
                fwd_vols = None
            else:
                verify_inp( fwd_vols > 0., lambda: "'fwd_vols' cannot be negative")
                fwd_vols = np.full( (T,), fwd_vols, dtype=self.geometry.xexpiries.dtype )

        verify_inp( len(opt_tte.shape) == 1, f"Expected opt_tte to be a 1D array, found shape {opt_tte.shape} (this function does not support batch dimensions)" )
        verify_inp( fwd_vols is None or fwd_vols.shape == (T,), lambda : f"Expected fwd_vols to have shape {(T,)}, found {fwd_vols.shape}" )            
        verify_inp( opt_strikes.shape == opt_bid.shape == opt_ask.shape == opt_tte.shape, lambda : f"Expected 'opt_strikes', 'opt_bid', 'opt_ask' and 'opt_tte' to have the same shape; found {opt_strikes.shape}, {opt_bid.shape}, {opt_ask.shape}, {opt_tte.shape}" )
        verify_inp( opt_is_calls is None or opt_is_calls.shape == opt_tte.shape, lambda : f"Expected 'opt_is_calls' to have the same shape as 'opt_tte'; found {opt_is_calls.shape} and {opt_tte.shape}" )
        verify_inp( opt_mask is None or opt_mask.shape == opt_tte.shape, lambda : f"Expected 'opt_mask' to have the same shape as 'opt_tte'; found {opt_mask.shape} and {opt_tte.shape}" )

        # mask by removing options
        # ------------------------

        if not opt_mask is None:
            opt_strikes = opt_strikes[opt_mask]
            opt_bid         = opt_bid[opt_mask]
            opt_ask         = opt_ask[opt_mask]
            opt_tte         = opt_tte[opt_mask]
            opt_is_calls    = opt_is_calls[opt_mask] if opt_is_calls is not None else None
            opt_interpolate = { k: v[opt_mask] for k,v in opt_interpolate.items() } if opt_interpolate is not None else None

        if not self.no_warnings and np.max( opt_tte ) > self.geometry.xexpiries[-1]:
            warnings.warn(f"Found option with expiry {np.max( opt_tte ):.4g} beyond the last model expiry {self.geometry.xexpiries[-1]:.4g}. "+\
                           "While functionaly feasible, this is likely to cause fitting issues. Expand expiry range.")

        opt_interpolate = opt_interpolate if not opt_interpolate is None else self.geometry.time_interpolation_weights( opt_tte, None )
        M = len(opt_strikes)

        # compute sqrt var per option
        # ---------------------------

        if fwd_vols is not None:
            sanos           = GridSanos(geometry=self.geometry, bb=bbnp, debug=self.debug)
            opt_sqrtVars    = sanos.interpolate_sqrtVars( fwd_vols, opt_interpolate ) 
            verify_inp( opt_sqrtVars is None or (opt_sqrtVars.shape == (M,) and np.all( np.isfinite(opt_sqrtVars) ) and np.all( opt_sqrtVars >= 0. )), f"Internal error: opt_sqrtVars must have shape {(M,)} and be non-negative and finite, found shape {opt_sqrtVars.shape} and min {np.min(opt_sqrtVars)} and max {np.max(opt_sqrtVars)}" )
        else:
            opt_sqrtVars = None

        # DLV SANOS
        # ---------

        half_spread  = 0.5*(opt_ask - opt_bid)
        opt_mid      = 0.5*(opt_ask + opt_bid)
        verify_inp( np.min(opt_bid) > -self.eps.prices, f"Option bid must be positive, found min {np.min(opt_bid)} [tolerance 'prices' {self.eps.prices}]" )
        verify_inp( np.min(half_spread) > -self.eps.prices, f"Option half spread must be positive, found min {np.min(half_spread)} [tolerance 'prices' {self.eps.prices}]" )

        opt_call_mid = opt_mid if opt_is_calls is None else np.where(opt_is_calls, opt_mid, opt_mid + 1. - opt_strikes )
        opt_call_bid = opt_bid if opt_is_calls is None else np.where(opt_is_calls, opt_bid, opt_bid + 1. - opt_strikes )
        opt_call_ask = opt_ask if opt_is_calls is None else np.where(opt_is_calls, opt_ask, opt_ask + 1. - opt_strikes ) 
        del opt_bid, opt_ask, opt_mid

        # Optimization over densities
        # ---------------------------

        xexpiries   = np.asarray( self.geometry.xexpiries, dtype=np.float64 )
        dts         = np.diff( xexpiries )
        xstrikes    = np.asarray( self.geometry.xstrikes, dtype=np.float64 )
        mxstrikes   = xstrikes[:,1:-1]
        density0    = np.asarray( self.geometry.density0, dtype=np.float64 )
 
        densities   = cp.Variable( (T,N), nonneg=True )
        conditions  = [ cp.sum(densities, axis=1) == 1., cp.sum( cp.multiply(  mxstrikes[1:,:], densities ), axis=1 ) == 1. ] # density conditions

        # martingale property for X
        # -------------------------
        # see https://arxiv.org/pdf/2601.11209
        
        _0 = np.float64(0.)
        _1 = np.float64(1.)

        # dlvs
        # We note that dC_t = 1/2 DLV^2 K^2 dT C''(K) where C''(K) = 1/2 ( k[2:] - K[:-2]) p(k)
        # hence
        #    dC_t = DLV^2 { K^2/( k[2:] - K[:-2]) dt} p(K)
        # We precompute
        #    dlvbox = K^2/( k[2:] - K[:-2]) dt

        dlvbox       = xstrikes[1:,1:-1]**2 / ( xstrikes[1:,2:] - xstrikes[1:,:-2]) * dts[:,None]
        assert dlvbox.shape == (T,N), f"Expected dlvbox to have shape {(T,N)}, found {dlvbox.shape}"
        dlvbox[:,0]  = 0. # no DLV at the boundary points
        dlvbox[:,-1] = 0. # this generates sufficient freedom for sum(d)=1 and E[d]=1
        assert self.geometry.dlv_max > self.geometry.dlv_min, f"'dlv_max' must exceed 'dlv_min', found {self.geometry.dlv_max} vs {self.geometry.dlv_min}"
        min_dlvsq    = np.float64( self.geometry.dlv_min**2 )
        max_dlvsq    = np.float64( self.geometry.dlv_max**2 )

        # Call_t(k) = \sum_i p_t^i ( k^i_t - k )^+
        # We are computinh
        # mcall_t^i   := Call_t( k^i_t )
        # mcall_tm1^i := Call_{t-1}( k^i_t )
        # Note that density0 is deterministic, hence requires different handling

        dCs = []
        boxs = []

        for t in range(T):
            mcalls_t    = np.maximum( mxstrikes[t+1,None,:] - mxstrikes[t+1,:,None], _0 )  # -> (N,N)
            assert mcalls_t.shape == (N,N), f"Expected shape {(N,N)}, found {mcalls_t.shape}"
            density_t   = densities[t,:] 
            mcalls_t    = mcalls_t @ density_t
            assert mcalls_t.shape == (N,), f"Expected mcalls_t to have shape {(N,)}, found {mcalls_t.shape}"

            density_tm1 = densities[t-1,:] if t > 0 else density0
            mcalls_tm1  = np.maximum( mxstrikes[t,None,:] - mxstrikes[t+1,:,None], 0. )  # (N,N)
            assert mcalls_tm1.shape == (N,N), f"Expected shape {(N,N)}, found {mcalls_tm1.shape}"
            mcalls_tm1  = mcalls_tm1 @ density_tm1
            assert mcalls_tm1.shape == (N,), f"Expected mcalls_tm1 to have shape {(N,)}, found {mcalls_tm1.shape}"

            dC_t         = mcalls_t - mcalls_tm1    
            box_t        = cp.multiply( dlvbox[t,:], density_t ) 
            lower_t      = min_dlvsq * box_t
            upper_t      = max_dlvsq * box_t
            assert lower_t.shape == (N,), f"Expected lower_t to have shape {(N,)}, found {lower_t.shape}"
            assert upper_t.shape == (N,), f"Expected upper_t to have shape {(N,)}, found {upper_t.shape}"
            conditions  += [ dC_t >= 0. ] #[ dC_t >= lower_t, dC_t <= upper_t ] # call price must be increasing in time to expiry for fixed strike, with an upper bound given by the maximum DLV squared.

            dCs.append( dC_t )
            boxs.append( box_t )
            del upper_t, lower_t, box_t, dC_t, mcalls_t, mcalls_tm1

        # computing model prices
        # ----------------------
        # 'opt_lix' is an index in T+1 xexpiries; mstrikes has T expiries.

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

        opt_lix      = opt_interpolate['lix']
        opt_dweight  = opt_interpolate['dweight']
        ulixs        = np.unique(opt_lix)
        mstrikes     = xstrikes[1:,1:-1]
        half_spread  = np.maximum( 0.5*(opt_call_ask - opt_call_bid), opt_call_ask.dtype.type(self.min_half_spread) )
        inv_hspread  = 1./half_spread
        fitted       = {}

        objective    = 0.      

        for uix in ulixs:
            uix          = int(uix)
            umask        = opt_lix == uix
            uM           = np.sum(umask)
            tty          = opt_tte[umask][0]
            
            # compute call prices
            eff_lix     = max(uix-1, 0)
            prev_mstrk  = mstrikes[eff_lix,:]
            next_mstrk  = mstrikes[uix,:]
            upot_strks  = opt_strikes[umask]

            if not self.no_warnings:
                min_k = np.min( upot_strks )
                max_k = np.max( upot_strks )
                if uix > 1 and min_k <= prev_mstrk[1]:
                    warnings.warn(f"Found option with expiry {tty:.4g} (in years) whose strike {min_k:.4g} is at or below the second-lowest model strike {prev_mstrk[1]:.4g} for the previous expiry used "+\
                                "to compute the model option price. This is likely causing errors. Widen strike range.")
                if uix > 1 and max_k >= prev_mstrk[-2]:
                    warnings.warn(f"Found option with expiry {tty:.4g} (in years) whose strike {max_k:.4g} is at or above the second-highest model strike {prev_mstrk[-2]:.4g} for the previous expiry used "+\
                                "to compute the model option price. This is likely causing errors. Widen strike range.")
                if min_k <= next_mstrk[1]:
                    warnings.warn(f"Found option with expiry {tty:.4g} (in years) whose strike {min_k:.4g} is at or below the second-lowest model strike {next_mstrk[1]:.4g} for the next expiry used "+\
                                "to compute the model option price. This is likely causing errors. Widen strike range.")
                if max_k >= next_mstrk[-2]:
                    warnings.warn(f"Found option with expiry {tty:.4g} (in years) whose strike {max_k:.4g} is at or above the second-highest model strike {next_mstrk[-2]:.4g} for the next expiry used "+\
                                "to compute the model option price. This is likely causing errors. Widen strike range.")

            calls_prev  = call_prices( spots=prev_mstrk[None,:],
                                       strikes=upot_strks[:,None], 
                                       sqrtVars=opt_sqrtVars[umask,None] if opt_sqrtVars is not None else None
                                    ) * opt_dweight[umask,None]
            assert calls_prev.shape == (uM,N), f"Expected calls_prev to have shape {(uM,N)}, got {calls_prev.shape}"
            call_prev   = calls_prev @ densities[eff_lix,:]
            assert call_prev.shape == (uM,), f"Expected call_prev to have shape {(uM,)}, got {call_prev.shape}"

            calls_next   = call_prices( spots=next_mstrk[None,:],
                                        strikes=upot_strks[:,None], 
                                        sqrtVars=opt_sqrtVars[umask,None] if opt_sqrtVars is not None else None
                                    ) * ( 1.- opt_dweight[umask,None] )
            assert calls_next.shape == (uM,N), f"Expected calls_next to have shape {(uM,N)}, got {calls_next.shape}"
            calls_next    = calls_next @ densities[uix]
            assert calls_next.shape == (uM,), f"Expected calls_next to have shape {(uM,)}, got {calls_next.shape}"
            fit_calls = call_prev + calls_next

            fitted[uix] = ( fit_calls, umask )

            # add up to objective
            err_sqr       = cp.sum( cp.multiply( inv_hspread[umask], fit_calls - opt_call_mid[umask] )**2 ) if self.mid_fit_weight > 0. else None
            excess        = cp.sum( cp.multiply( inv_hspread[umask], cp.pos( fit_calls - opt_call_ask[umask] ) + cp.pos( opt_call_bid[umask] - fit_calls ) ) ) if self.mid_fit_weight < 1. else None

            if self.mid_fit_weight == 0.:
                objective = objective + excess
            elif self.mid_fit_weight == 1.:
                objective = objective + err_sqr
            else:
                objective = objective + self.mid_fit_weight * err_sqr + (1.- self.mid_fit_weight) * excess
            del calls_prev, calls_next, eff_lix, uM, umask

        assert len(objective.shape) == 0, f"Expected objective to be a scalar, found shape {objective.shape}"

        # optimize
        # --------

        with verbose.write_t(lambda : f"Solving for implied DLVs (squared) for {len(opt_strikes)} options. Mean half spread is {np.mean(half_spread):.4e}... ", end='') as tme:
            prob        = cp.Problem( cp.Minimize( objective ), conditions )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.", category=UserWarning)
                try:
                    prob.solve( solver=self.cvxpy_solver, **self.cvxpy_solver_options )
                except cp.SolverError as e:
                    raise e
            
            densities  = densities.value
            assert np.all( np.isfinite(densities) ), f"Expected fitted call prices to be finite, found min {np.min(fit_calls)} and max {np.max(fit_calls)}"

            fit_calls    = np.zeros_like( opt_strikes )
            for fc, um in fitted.values():
                fit_calls[um] = fc.value
            assert np.all( np.isfinite(fit_calls) ), f"Expected fitted call prices to be finite, found min {np.min(fit_calls)} and max {np.max(fit_calls)}"

            err_sqr    = err_sqr.value if self.mid_fit_weight > 0. else 0.
            objective  = objective.value
            excess     = inv_hspread * ( np.clip( fit_calls - opt_call_ask, min=0. ) + np.clip( opt_call_bid - fit_calls, min=0. ) ) 

            if np.any( excess>0. ):
                bad        = excess[excess>0.]
                max_       = np.max(bad) if len(bad) > 0 else 0.
                q95, q99, q999 = np.quantile(bad, [0.95, 0.99, 0.999])
                bad_str    = f"relative excess error conditional mean {np.mean(bad):.4e}, std {np.std(bad):.4e}, 95th percentile {q95:.4e}, 99th percentile {q99:.4e}, 99.9th percentile {q999:.4e} and max {max_:.4e}"
                del q95, q99, q999, bad
            else:
                bad_str    = "zero relative excess error"
                max_       = 0.

            if not self.max_error is None:
                if max_ > self.max_error:
                    slv_str = f"successfully returned but" if prob.status == "optimal" else f"return with status '{prob.status}' and"
                    err_str = f"Solver {self.cvxpy_solver} {slv_str} failed to find a solution within the specified 'max_error' tolerance. "+\
                            f"Maximum relative error {max_:.4e} exceeds `max_error` {self.max_error:.4e}. " +\
                            f"Objective value: {objective:.4e}, {bad_str} "+\
                            "You can turn this check off by setting 'max_error' to None." 
                    raise RuntimeError( err_str )
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                verbose.write(lambda : f"done; solver {self.cvxpy_solver} returned status '{prob.status}' with {bad_str}. This took {tme}.", head=False)
            else:
                verbose.write(lambda : f"done with {bad_str}. This took {tme}.", head=False)

            del prob, objective, excess, inv_hspread, half_spread, err_sqr, bad_str, max_

        # computing DLVS
        # we use the call prices computed earlier
        # dlvs
        # We note that dC_t = 1/2 DLV^2 K^2 dT C''(K)
        # where C''(K) = 1/2 ( k[2:] - K[:-2]) p(k)
        # hence
        # dC_t = DLV^2 K^2/( k[2:] - K[:-2]) dt p(K)

        assert np.all( np.isfinite(densities) ), f"Expected fitted call prices to be finite, found min {np.min(fit_calls)} and max {np.max(fit_calls)}"
        assert np.min( densities ) > -self.eps.densities, f"Expected densities to be non-negative, found min {np.min(densities)} [tolerance 'densities' {self.eps.densities}]"
        assert np.max( np.abs( np.sum( densities, axis=1 ) - 1. ) ) <= self.eps.densities, f"Expected sum of densities to be 1, found {np.sum(densities, axis=1)} [tolerance 'densities' {self.eps.densities}]"
        assert np.max( np.abs( np.sum( densities*mstrikes, axis=1 ) - 1. ) ) <= self.eps.densities, f"Expected sum of densities*strikes to be 1, found {np.sum(densities, axis=1)} [tolerance 'densities' {self.eps.densities}]"

        with verbose.write_t(lambda : f"Computing DLV squared's... ", end='') as tme:
            dCs     = [ dC_t.value for dC_t in dCs ]
            boxs    = [ box_t.value for box_t in boxs ]
            dlvsqs  = np.zeros( (T,N-2), dtype=fit_calls.dtype )

            box_     = self.geometry.strikes[:,1:-1]**2 / ( self.geometry.strikes[:,2:] - self.geometry.strikes[:,:-2] ) * dts[:,None]
            boxs_    = [ box_[t,:] * densities[t,1:-1] for t in range(T) ] # we recompute the box here to ensure consistency with the fitted densities, which can be slightly off from the optimization constraints due to solver tolerances. This ensures that the DLVs are computed consistently with the fitted densities.

            for a, b in zip(boxs, boxs_):
                assert np.max( np.abs( a[1:-1]-b ) )<1E-6, f"Box error difference {np.max( np.abs( a-b ) ):.4g}"


            for t in range(T):
                dC_t         = dCs[t][1:-1]
                density_box  = boxs[t][1:-1]  # supoosed to be density*box
    
                assert np.all( np.isfinite(dC_t) ), f"Expected dC_t to be finite, found min {np.min(dC_t)} and max {np.max(dC_t)}"
                assert np.all( np.isfinite(density_box) ), f"Expected density_box to be finite, found min {np.min(density_box)} and max {np.max(density_box)}"
                assert np.min( dC_t ) > -self.eps.prices, f"Expected dC_t to be non-negative, found min {np.min(dC_t)} [tolerance 'prices' {self.eps.prices}]"
                assert np.min( density_box ) > -self.eps.densities, f"Expected density_box to be non-negative, found min {np.min(density_box)} [tolerance 'densities' {self.eps.densities}]"

                dC_zero      = np.abs(dC_t) <= self.eps.dlvsq
                too_small    = dC_t <= min_dlvsq * density_box
                too_big      = dC_t >= max_dlvsq * density_box

                """
                ick = ( density_box < 1E-6 ) & ~dC_zero
                if np.any(ick):
                    print(t)
                    print( dC_t[ick] )
                    print( density_box[ick] )
                """
                divisor      = np.where( too_small|dC_zero|too_big, _1, density_box )  # dC_t can be positive while p_t is zero ... in this case we cannot determine a correct DLV - any will do.
                dlvsqs[t,:]  = np.select( [dC_zero, too_small, too_big ], [min_dlvsq, min_dlvsq, max_dlvsq], default=dC_t / divisor )
            dlvsqs = np.where( np.isfinite(dlvsqs), dlvsqs, min_dlvsq ) # if density_box is zero and dC_t is zero, we set DLV to minimum; if density_box is zero and dC_t is positive, we set DLV to minimum as well, since any DLV would be consistent with the fit in this case due to the zero density.
            verbose.write(lambda : f"done; this tool {tme}.", head=False)

        # note that the ``GridSanos.density`` function adds min_dlvsq
        # to the input dlvsqs to ensure that the derivatives do not
        # die in zero. We subtract this here
        dlvsqs = np.clip( dlvsqs - min_dlvsq, min=0. )

        if __debug__ and self.debug:
            _grid_sanos    = GridSanos( geometry=self.geometry, bb=bbnp, debug=self.debug )
            test_densities = _grid_sanos.densities( dlvsqs )
#            assert np.max( np.abs( test_densities - densities ) ) < self.eps.densities, f"DLV fit error: max absolute difference between densities computed from fitted *and clipped* DLVs and optimized densities is {np.max( np.abs( test_densities - densities ) ):.4e}."

        if return_only_dlvsqs:
            return dlvsqs
        
        fit_prices = np.where( opt_is_calls, fit_calls, fit_calls - 1. + opt_strikes ) if opt_is_calls is not None else fit_calls    
        assert fit_prices.shape == (M,), f"Expected final fit_prices to have shape {(M,)} but found {fit_prices.shape}"

        if not opt_mask is None:
            fit_prices_ = np.zeros( opt_mask.shape, dtype=fit_prices.dtype )
            fit_prices_[opt_mask] = fit_prices
            fit_prices = fit_prices_
            del fit_prices_

        assert densities.shape == (T,N), f"Expected final densities to have shape {(T,N)}, got {densities.shape}"
        return dlvsqs, fit_prices, densities

