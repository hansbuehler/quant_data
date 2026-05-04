"""
Grid Sanos

Implementation of the SANOS model from https://arxiv.org/pdf/2601.11209
for an input grid of model strikes and expories.
The model can then produce call prices as function of forward volatilities and discrete local volatilities. This works for ``numpy``
and ``torch`` so far, with ``jax`` support coming soon.

The model also provides as ``fit_dlvs`` function which can be used to
fit DLVs given option prices and forward volatities.
"""

from __future__ import annotations

from dys.grid_sanos.geometry import GridGeometry
from dys.grid_sanos.backbone import Backbone, BackboneNumpy
from dys.grid_sanos.tridiag import solve_sanos_tridiagonal
from cdxcore.verbose import Context
from cdxcore.pretty import PrettyObject
from dataclasses import dataclass
import warnings
import numpy as nnp

class GridSanos(object):
    """
    Standard Grid Sanos with learnable foward volatilities and dynamic option request grid:
    the model can price any call option based just on its pure strikes and time to expory.

    Parameters
    ----------
    geometry : :class:`dys.grid_sanos.GridGeometry`
        Geometry of the model, which includes the grid of strikes and expiries, and the transition operators for the interpolation step.

    backbone : :class:`dys.grid_sanos.Backbone`, default ``BackboneNumpy()``
        Backbone to use for the model; this controls the array library used for the computations, and also provides
        utility functions for debugging and verification. The default is a numpy backbone, which does not require any additional dependencies.

    debug : bool, default ``False``
        Whether to enable debug mode. If True, the model will perform additional checks and print debug information.
        This is heavy; avoid in a production setting.

    cvxpy_solver : str, default ``CLARABEL``
        Solver to use for the convex optimization problem in :meth:`fit_dlvsqs`, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.

    cvxpy_solver_options : dict | None, default ``None``
        Options for the convex optimization solver; this can be used to pass additional parameters to the solver.

    """
    Geometry = GridGeometry
    
    @dataclass(frozen=True)
    class Config:
        """ Configuration """
        vol_floor             : float = float(Backbone.EPS_VOL) #: floor for forward volatilities; this prevents numerical issues when forward volatilities are very close to 0.
        dlv_floor             : float = 1E-4 #: floor for discrete local volatilities; this prevents numerical issues when DLVs are very close to 0.
        max_inv_spread        : float = 10. #: maximum inverse spread for weighting the objective in imply_dlvs; this prevents numerical issues when bid-ask spreads are very small.
        mid_fit_weight        : float = 0.1 #: weight of mid price fit in the objective; this prevents the optimizer from putting all the weight on one side of the spread when the spread is large.
        cvxpy_solver          : str = "CLARABEL" #: cvxpy solver to use for the convex optimization problem in :meth:`fit_dlvsqs`
        cvxpy_solver_options  : dict|None = None #: options for the cvxpy solver.

        def __post_init__(self):
            assert self.vol_floor >= 0., f"'vol_floor' must be non-negative, got {self.vol_floor}"
            assert self.dlv_floor >= 0., f"'dlv_floor' must be non-negative, got {self.dlv_floor}"
            assert self.max_inv_spread > 0., f"'max_inv_spread' must be positive, got {self.max_inv_spread}"
            assert self.mid_fit_weight >= 0., f"'mid_fit_weight' must be non-negative, got {self.mid_fit_weight}"

    def __init__(self, geometry: GridGeometry,
                       backbone : Backbone = BackboneNumpy(), *,
                       config : Config = Config(),
                       debug : bool = False
                       ):
        """ Initialize """

        super().__init__()
        self.geometry = geometry    #: :class:`dys.grid_sanos.GridGeometry`geometry of the model
        self.backbone = backbone    #: :class:`dys.grid_sanos.Backbone` backbone for the model (numpy, torch, jax, etc.)
        self.config   = config      #: :class:`dys.grid_sanos.GridSanos.Config` configuration for the model.

        # debugging, error testing etc
        self.debug    = debug #: turn on debugging. This is expensive, in particular during machine learning.
        self.debug_return_bardensities = False #: whether to return the "bar" densities at each time step for debugging purposes; these are the densities before solving the tridiagonal system, and should be close to the final densities.
        self.debug_internal_tests      = False #: whether to run internal tests on the interpolation and tridiagonal solver; these tests are somewhat expensive, so they are disabled by default.

        # tolerances
        self.eps = PrettyObject( #: tolerances for various checks and numerical issues; these are used in the tests and verifications, and also in the optimization objective to prevent numerical issues when values are close to their boundaries.
            weights = 1E-6, # tolerance for weights being between 0 and 1; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when weights are very close to 0 or 1.
            densities = 1E-6, # tolerance for densities being positive and having mass 1; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when densities are very close to 0 or not summing to 1.
            prices = 1E-6, # tolerance for option prices being positive and respecting bid-ask spread; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when option prices are very close to 0 or not respecting the bid-ask spread.
            dlvsq = 1E-6, # tolerance for discrete local volatilities squared being positive; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when DLVs are very close to 0 or negative.
            theta = 1E-6, # tolerance for theta being positive; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when theta is very close to 0 or negative.
            gamma = 1E-6, # tolerance for gamma being positive; this is used in the tests and verifications, and also in the optimization objective to prevent numerical issues when gamma is very close to 0 or negative.
        )

        # precompute geometry tensors
        self._dts       = backbone.to_array( self.geometry.dts, dtype=backbone.float32 )
        self._dsqrtTs   = backbone.to_array( self.geometry.dsqrtTs, dtype=backbone.float32 )
        self._xtimes    = backbone.to_array( self.geometry.xtimes, dtype=backbone.float32 )
        self._mstrikes  = backbone.to_array( self.geometry.strikes, dtype=backbone.float32 )
        self._mstrikes0 = backbone.to_array( self.geometry.strikes0, dtype=backbone.float32 )
        self._density0  = backbone.to_array( self.geometry.density0, dtype=backbone.float32 )
        wn, wp, L      = self.geometry.transition_operators()
        self._wn        = backbone.to_array( wn, dtype=backbone.float32 )
        self._wp        = backbone.to_array( wp, dtype=backbone.float32 )
        self._L         = backbone.to_array( L, dtype=backbone.float32 )
        del wn, wp, L

        T, N  = self.TN
        assert self._dts.shape == (T,), f"dts must have shape {(T,)}, got {self._dts.shape}"
        assert self._xtimes.shape == (T+1,), f"xtimes must have shape {(T+1,)}, got {self._xtimes.shape}"
        assert self._mstrikes.shape == (T,N), f"mstrikes must have shape {(T,N)}, got {self._mstrikes.shape}"
        assert self._wn.shape == (T,N-2) and self._wp.shape == (T,N-2), f"wn and wp must have shape {(T,N-2)}, got {self._wn.shape} and {self._wp.shape}"
        assert self._L.shape == (T,N,N), f"L must have shape {(T,N,N)}, got {self._L.shape}"
        assert self._density0.shape == (N,), f"density0 must have shape {(N,)}, got {self._density0.shape}"

    @property
    def TN(self) -> int:
        """ Number of expiries and strikes """
        return len(self.geometry.times), len(self.geometry.nms)   

    def verify(self, condition: callable, message : str|callable ):
        """ Verify a condition on the backbone, raise an error with message if not satisfied. This function is a NOOP if ``debug`` is False. """
        if self.debug:
            self.backbone.verify( condition(), message )

    @property
    def mstrikes(self) -> Backbone.Array:
        """ Model strikes """
        return self._mstrikes
    
    @property
    def mxtimes(self) -> Backbone.Array:
        """ Model times including leading zero """
        return self._xtimes
    
    def set_eps(self, eps):
        """ Set flat tolerance for the model """
        for k in self.eps:
            self.eps[k] = eps

    def _compute_aux( self, fwd_vols : Backbone.Array,
                            opt_tte  : Backbone.Array, 
                            opt_mask : Backbone.Array|None = None
                            ) -> Backbone.Array:
        r"""
        Compute the weights for the forward variance calculation, given option times to expiry. This is a separate function for better readability and debuggability.
            dVj  = forward variance from T_{j-1} to T_j
            V(t) = \sum_j dV_j ( { t - T_{j-1} }^+ } - { t - T_j }^+ )/{ T_j - T_{j-1} }
        """

        T, N      = self.TN
        M         = opt_tte.shape[-1]
        head      = opt_tte.shape[:-1]
        hext      = (1,)*len(head)     
        bb        = self.backbone
        _0        = bb.zeros((),dtype=bb.float32)
        self.verify( lambda : bb.min(fwd_vols) >= 0., lambda : f"Forward volatilities must be non-negative, found min {bb.min(fwd_vols):.4g}" )
        fwd_vols  = bb.clip( fwd_vols, min=0.) + self.config.vol_floor

        xtimes    = self._xtimes.reshape( hext + self._xtimes.shape )
        div       = xtimes[...,1:,None] - xtimes[...,:-1,None]
        weights   = bb.relu( ( opt_tte[...,None,:] - xtimes[...,:-1,None] ) / div ) - bb.relu( ( opt_tte[...,None,:] - xtimes[...,1:,None] ) / div )
        weights   = bb.clip( weights, min=0., max=1. )
        del div

        assert bb.are_broadcastable(weights.shape, head+(T,M)), f"Expected weights to have shape {opt_tte.shape[:-1]+(T,M)}, got {weights.shape}"
        if self.debug:
            good = (-self.eps.weights<=bb.min(weights)) & (bb.max(weights) <=1.+self.eps.weights)
            good = bb.where( opt_mask, good, True ) if opt_mask is not None else good
            self.verify( lambda : good, lambda : f"Weights should be between 0 and 1, found min {bb.min(weights)} and max {bb.max(weights)} [tolerance 'weights' {self.eps.weights}]" )

        if self.geometry.vol_interp == "lin_var":
            dts       = self._dts.reshape( hext + self._dts.shape )
            dvars     = (fwd_vols**2) * dts
            assert bb.min( dts ) > 0., f"Time steps must be positive, found min {bb.min(dts)}"
            assert bb.are_broadcastable(dvars.shape, head+(T,)), f"Expected dvars to have shape {head+(T,)} or compatible, got {dvars.shape}"
            del dts

            vars       = bb.sum( dvars[...,:,None] * weights, axis=-2 )
            sqrtVars   = bb.where(opt_mask, bb.sqrt( vars ), _0 ) if opt_mask else bb.sqrt( vars )

            self.verify( lambda : bb.isfinite(vars), lambda : "Forward variances contain non-finite values" )
            self.verify( lambda : bb.min(vars) >= 0., lambda : "Forward variances contain negative values" )
            assert bb.are_broadcastable(sqrtVars.shape, head+(M,)), f"Expected sqrtVars to have shape {opt_tte.shape[:-1]+(T,M)}, got {sqrtVars.shape}"
            del dvars
        
        else:
            assert self.geometry.vol_interp == "sqrt_lin_vol", f"Invalid vol_interp {self.geometry.vol_interp}"
            sqrtT      = bb.sqrt( xtimes )
            dsqrtTs    = self._dsqrtTs.reshape( hext + self._dsqrtTs.shape )
            dvols      = fwd_vols * dsqrtTs
            assert bb.min( dsqrtTs ) > 0., f"Time steps must be positive, found min {bb.min(dsqrtTs)}"
            del dsqrtTs

            div        = sqrtT[...,1:,None] - sqrtT[...,:-1,None]
            sq_weights = bb.relu( ( opt_tte[...,None,:] - sqrtT[...,:-1,None] ) / div ) - bb.relu( ( opt_tte[...,None,:] - sqrtT[...,1:,None] ) / div )
            sq_weights = bb.clip( sq_weights, min=0., max=1. )
            del div
            if self.debug:
                good = (-self.eps.weights<=bb.min(sq_weights)) & (bb.max(sq_weights) <=1.+self.eps.weights)
                good = bb.where( opt_mask, good, True ) if opt_mask is not None else good
                self.verify( lambda : good, lambda : f"Weights should be between 0 and 1, found min {bb.min(sq_weights)} and max {bb.max(sq_weights)} [tolerance 'weights' {self.eps.weights}]" )

            sqrtVars  = bb.sum( dvols[...,None] * sq_weights, axis=-2 )
            sqrtVars  = bb.where(opt_mask, sqrtVars, _0 ) if not opt_mask is None else sqrtVars
            assert bb.are_broadcastable(sqrtVars.shape, head+(M,)), f"Expected sqrtVars to have shape {opt_tte.shape[:-1]+(T,M)}, got {sqrtVars.shape}"
            del dvols, sq_weights
        assert weights.shape[-2:] == (T,M), f"Expected weights to have shape {head+(T,M)}, got {weights.shape}" 

        self.verify( lambda : (weights >= -self.eps.weights) & (weights - 1. <= self.eps.weights), lambda : f"Weights must be between 0 and 1; found {bb.min(weights)} and {bb.max(weights)} [tolerance 'weights' {self.eps.weights}]" )
        self.verify( lambda : bb.isfinite(sqrtVars), lambda : "Weights contain non-finite values" )

        pweights = bb.empty( head+(T,M), dtype=bb.float32 )
        pweights[...,1:,:] = bb.clip( weights[...,:-1,:] - weights[...,1:,:], min=0., max=1. )
        pweights[...,0,:]  = bb.clip( 1. - bb.sum( pweights[...,1:,:], axis=-2 ), min=0., max=1. ) #TODO: needs checking; can also be pre-computed for a given set of options.

        if self.debug:
            self.verify( lambda : (pweights >= -self.eps.weights) & (pweights - 1. <= self.eps.weights), lambda : f"pweights must be between 0 and 1; found {bb.min(pweights)} and {bb.max(pweights)} [tolerance 'weights' {self.eps.weights}]" )
            def test_weights_sum():
                """ test that only one or two weights are non-zero for eachh option """
                sums = bb.sum( pweights > self.eps.weights*10., axis=-2 )
                return bb.any( (sums < 1) | (sums > 2) )
            if test_weights_sum():
                pweights    = bb.to_numpy(pweights).reshape((-1,T,M)).transpose(0,2,1)
                assert pweights.shape[-2:] == (M,T), f"Expected pweights to have shape {(M,T)}, got {pweights.shape}"
                pweights    = pweights.reshape((-1,T))
                num_nonzero = nnp.sum(pweights > self.eps.weights*10., axis=-1)
                if nnp.any( num_nonzero==0 ):
                    print("Found options with zero effective weight:")
                    indices = nnp.arange(pweights.shape[0])[num_nonzero==0][:5]
                    for ix in indices:
                        print(f"Option {ix}: weights {pweights[ix,:]}")
                if nnp.any( num_nonzero>2 ):
                    print("Found options with more than two effective weights:")
                    indices = nnp.arange(pweights.shape[0])[num_nonzero>2][:5]
                    for ix in indices:
                        print(f"Option {ix}: weights {pweights[ix,:]}")
                raise RuntimeError( f"The number of positive weights must be 1 or 2; found min {bb.min(bb.sum(pweights > self.eps.weights*10., axis=-2))}, "+\
                                    f"max {bb.max(bb.sum(pweights > self.eps.weights*10., axis=-2))} #of weights above 0. [tolerance 'weights' {self.eps.weights*10.}]" )
        return pweights, sqrtVars

    def __call__( self, fwd_vols         : Backbone.Array,
                        dlvsqs           : Backbone.Array, 
                        opt_strikes      : Backbone.Array,
                        opt_tte          : Backbone.Array, *,
                        opt_is_calls     : Backbone.Array | None = None, 
                        opt_mask         : Backbone.Array | None = None,
                        return_densities : bool = False ) -> Backbone.Array:
        """
        Compute the grid of option prices given forward volatilities and discrete local volatilities using https://arxiv.org/pdf/2601.11209.

        Parameters
        ----------
        fwd_vols : array of shape (...,T)
            Forward volatilities for each time step.

        dlvsqs : array of shape (...,T,N-2)
            Discrete local volatilities, squared, for each time step and strike outside the boundaries. c.f. https://arxiv.org/pdf/2601.11209.

        opt_strikes : array of shape (...,M)
            Strikes of the options to price.
        
        opt_tte : array of shape (...,M)    
            Time to expiry of the options to price.
        
        opt_mask : array of shape (...,M) | None, default ``None``
            Mask for which options to price; if False, the model will skip the computations for that option and return 0. This can be used to speed up computations when only a subset of options is needed.

        opt_is_calls : array of shape (...,M)
            Boolean array indicating whether each option is a call (True) or put (False).

        return_densities : bool, default ``False``
            Whether to return the densities at each time step;
            if False, only return option prices. 

        Returns
        -------
        prices : array of shape (...,M)
            Prices of the options with given strikes and times to expiry

        densities : array of shape (...,T,N)
            Densities at each time step; only returned if ``return_densities`` is True.
        """
        bb       = self.backbone
        head     = dlvsqs.shape[:-2]
        hext     = (1,)*len(head)     
        T, N     = self.TN

        assert dlvsqs.shape[-2:] == (T,N-2), f"Expected dlvsqs shape to end with size {(T,N-2)}, found {dlvsqs.shape} (recall that the boundary strikes do not require DLVs)"
        assert fwd_vols.shape[-1] in [1,T], f"Expected fwd_vols shape to end with size 1 or {T}, found {fwd_vols.shape}"
        assert bb.are_broadcastable( opt_tte.shape[:-1], head ), f"Expected opt_tte shape to start with {head} or compatible shape, found {opt_tte.shape}"
        assert bb.are_broadcastable( opt_strikes.shape[:-1], head ), f"Expected opt_strikes shape to start with {head} or compatible shape, found {opt_strikes.shape}"
        assert bb.are_broadcastable( opt_strikes.shape, opt_tte.shape ), f"Expected opt_strikes and opt_tte to have compatible shapes, found {opt_strikes.shape} and {opt_tte.shape}"

        if not opt_is_calls is None: assert bb.are_broadcastable( opt_strikes.shape, opt_is_calls.shape ), f"Expected opt_strikes and opt_is_calls to have compatible shapes, found {opt_strikes.shape} and {opt_is_calls.shape}"
        if not opt_mask is None: assert bb.are_broadcastable( opt_strikes.shape, opt_mask.shape ), f"Expected opt_strikes and opt_mask to have compatible shapes, found {opt_strikes.shape} and {opt_mask.shape}"

        M         = opt_tte.shape[-1]
        _text     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert len(head) <= len(_text), f"Too many batch dimensions: expected at most {len(_text)}, got {len(head)}"
        einhext   = _text[:len(head)] 
        wn        = self._wn.reshape( hext+self._wn.shape )
        wp        = self._wp.reshape( hext+self._wp.shape )
        L         = self._L.reshape( hext+self._L.shape )
        dts       = self._dts.reshape( hext+(T,) )
        mstrikes  = self._mstrikes.reshape( hext+(T,N) )
        density0  = bb.empty( head+(N,), dtype=bb.float32 )
        density0[:] = self._density0.reshape( hext+(N,) )

        self.verify( lambda : (density0 >= -self.eps.weights) & (density0 - 1. <= self.eps.weights), lambda : f"Density 0 must be between 0 and 1; found min {bb.min(density0)} and max {bb.max(density0)} [tolerance 'weights' {self.eps.weights}]" )
        self.verify( lambda : bb.abs( bb.sum( density0, axis=-1 ) - 1. ) < self.eps.densities, lambda : f"Density 0 must have positive mass, found sum {bb.max(bb.abs( bb.sum( density0, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

        # compute forward variance
        # ------------------------

        pweights, sqrtVars = self._compute_aux( fwd_vols=fwd_vols, opt_mask=opt_mask, opt_tte=opt_tte )

        # compute densities q
        # -------------------

        # temporary buffers
        cp_       = bb.zeros(head + (N - 1,), dtype=opt_tte.dtype )
        y_        = bb.zeros(head + (N,), dtype=opt_tte.dtype )

        calls         = None
        mdensities    = []
        bardensities  = [] if ( self.debug_return_bardensities or self.debug_internal_tests ) else None

        for t in range(T):
            # solve for density in t
            # ----------------------

            dlvar_t  = (dlvsqs[...,t,:]+self.config.dlv_floor**2) * dts[...,t,None]  # local variance at time t, shape (...,N-2)
            assert dlvar_t.shape == head+(N-2,), f"Expected dlvar_t to have shape {head+(N-2,)}, got {dlvar_t.shape}"
            
            mstrikes_t = mstrikes[...,t,:]
            w_minus    = dlvar_t * wn[...,t,:]
            w_plus     = dlvar_t * wp[...,t,:]

            prev_mdensity = mdensities[t-1] if t>0 else density0
            mdensity_t    = bb.einsum( f"{einhext}mn, {einhext}n -> {einhext}m", L[...,t,:,:], prev_mdensity )  # interpolate density to time t            
            bardensities.append( mdensity_t ) if not bardensities is None else None

            def test():
                """ tests that the linear interpolation operator works """
                prev_strikes = mstrikes[...,t-1,:] if t>0 else self._mstrikes0.reshape( hext+(N,) )
                c_x = bb.sum( bb.relu( prev_strikes[...,None,:] - mstrikes_t[...,:,None] ) * prev_mdensity[...,None,:], axis=-1 )
                c_y = bb.sum( bb.relu( mstrikes_t[...,None,:] - mstrikes_t[...,:,None] ) * mdensity_t[...,None,:], axis=-1 )
                self.verify( lambda : bb.max( bb.abs( c_x-c_y ) )  < 1e-6, lambda : f"Error preserving call prices at time {t}: {bb.max( bb.abs( c_x-c_y ) )}" )
            if self.debug_internal_tests:
                test()

            del prev_mdensity
            assert mdensity_t.shape == density0.shape, f"Expected interpolated mdensity_t to have shape {density0.shape}, got {mdensity_t.shape}"
            self.verify( lambda : (mdensity_t >= -self.eps.densities) & (mdensity_t - 1. <= self.eps.densities), lambda : f"Interpolated density at {t} must be between 0 and 1; found min {bb.min(mdensity_t)} and max {bb.max(mdensity_t)} [tolerance 'densities' {self.eps.densities }]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

            mdensity_t   = solve_sanos_tridiagonal(bb, w_minus=w_minus, w_plus=w_plus, rhs=mdensity_t, cp_=cp_, y_=y_) # keep

            def test():
                """ tests that the tridiagonal solver works by going forward one step and checking that we can recover the same DLVs """
                import numpy as nnp
                assert not bardensities is None, "debug_return_bardensities must be True to run this test"
                bt_  = bb.to_numpy( mdensity_t )
                bd_  = bb.to_numpy( bardensities[-1] )
                wm_  = bb.to_numpy( w_minus )
                wp_  = bb.to_numpy( w_plus )
                Pinv = nnp.zeros(head+(N,N), dtype=bd_.dtype )
                Pinv[...,0,0] = 1.
                Pinv[...,-1,-1] = 1.
                for i in range(1,N-1):
                    Pinv[...,i-1,i] = - wm_[...,i-1] 
                    Pinv[...,i,i]   = 1. + wm_[...,i-1] + wp_[...,i-1] 
                    Pinv[...,i+1,i] = - wp_[...,i-1]

                md_    = nnp.einsum( f"{einhext}mn, {einhext}n -> {einhext}m", nnp.linalg.inv(Pinv), bd_ )
                import numpy as nnp
                assert nnp.max(nnp.abs(bt_ - md_)) < 1e-6, f"Error solving tridiagonal system at time {t}: {nnp.max(nnp.abs(bt_ - md_))}" 
            if self.debug_internal_tests:
                test()

            assert mdensity_t.shape == density0.shape, f"Density at time {t} has wrong shape: expected {density0.shape}, got {mdensity_t.shape}"
            self.verify( lambda : bb.isfinite(mdensity_t), lambda : f"Density at time {t} contains non-finite values" )
            self.verify( lambda : (mdensity_t >= -self.eps.densities) & (mdensity_t - 1. <= self.eps.densities), lambda : f"Interpolated density at {t} must be between 0 and 1; found min {bb.min(mdensity_t)} and max {bb.max(mdensity_t)} [tolerance 'densities' {self.eps.densities}]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            del dlvar_t, w_minus, w_plus

            # compute call option prices
            # ---------------------
            pweight_t = pweights[...,t,:]  # have to fix weights before first t

            # fun fact: if we fix the forward vols ahead of training
            # we can pre-compute the weights and BS option prices
            # and feed them as data.
            spots_t    = mstrikes[...,t,None,:]
            calls_t    = bb.bs_price(spot=spots_t,
                                     strikes=opt_strikes[...,:,None], 
                                     sqrtVars=sqrtVars[...,:,None],
                                     mask=opt_mask[...,:,None] if opt_mask is not None else None,
                                     high_precision = True,
                                     debug_verify_bounds = self.debug
                                     )
            assert calls_t.shape == head+(M,N), f"Expected calls_t to have shape {head+(M,N)}, got {calls_t.shape}"      

            calls_t = bb.einsum( f"{einhext}mn, {einhext}n -> {einhext}m", calls_t, mdensity_t )
            assert calls_t.shape == head+(M,), f"Expected calls_t to have shape {head+(M,)}, got {calls_t.shape}"

            if self.debug:
                intr_t = bb.stop_gradient( bb.relu( 1. - opt_strikes ) ) # intrinsic value of options at time t, shape (...,M,N)
                self.verify( lambda : calls_t - intr_t >= -self.eps.prices, lambda : f"Option prices at time {t} must exceed intrinsic, found min {bb.min(calls_t-intr_t)} [tolerance 'prices' {self.eps.prices}]" )
                self.verify( lambda : calls_t - 1. <= self.eps.prices, lambda : f"Option prices at time {t} must be less than or equal to 1 (since forward price is normalized to 1), found max {bb.max(calls_t-1.)} [tolerance 'prices' {self.eps.prices}]" )
                del intr_t

            if calls is None:
                calls = calls_t * pweight_t
            else:
                calls = calls + calls_t * pweight_t

            mdensities.append(mdensity_t)
            del mdensity_t, pweight_t, calls_t, spots_t

        # validate final prices
        # ---------------------

        assert calls.shape == head+(M,), f"Expected final calls to have shape {head+(M,)}, got {calls.shape}"
        prices = calls if opt_is_calls is None else bb.where( opt_is_calls, calls, calls - 1. + opt_strikes ) # put-call parity; recall that forward price is normalized to 1
        prices = bb.where( opt_mask, prices, 0. ) if opt_mask is not None else prices
        del calls

        if self.debug:
            if opt_is_calls is None:
                intr = bb.relu( 1. - opt_strikes )
                max_ = 1.
                self.verify( lambda : ( prices - intr )[opt_mask] >= -self.eps.prices, lambda : f"Call prices must exceed intrinsic, found min {bb.min(prices-intr)} [tolerance 'prices' {self.eps.prices}]" )
                self.verify( lambda : ( prices - max_ )[opt_mask] <= self.eps.prices, lambda : f"Call prices must be less than or equal to 1 (since forward price is normalized to 1), found max {bb.max(prices -max_)} [tolerance 'prices' {self.eps.prices}]" )
            else:
                intr = bb.relu( bb.where(opt_is_calls,  1. - opt_strikes, opt_strikes - 1.)) 
                max_ = bb.where(opt_is_calls, 1., opt_strikes)
                self.verify( lambda : ( prices - intr )[opt_mask] >= -self.eps.prices, lambda : f"Option prices must exceed intrinsic, found min {bb.min((prices-intr)[opt_is_calls])} calls, {bb.min((prices-intr)[~opt_is_calls])} puts [tolerance 'prices' {self.eps.prices}]" )
                self.verify( lambda : ( prices - max_ )[opt_mask] <= self.eps.prices, lambda : f"Option prices must be less than or equal to max, found max {bb.max((prices -max_)[opt_is_calls])} calls, {bb.max((prices -max_)[~opt_is_calls])} puts [tolerance 'prices' {self.eps.prices}]" )
            del intr, max_

        if not return_densities:
            return prices
        
        # return densities
        # we return both the density at each time step, and
        # the previous "bar" density (TODO: change that)
        densities = bb.stack( mdensities, axis=-2 ) # shape (...,T,N)
        bardensities = bb.stack( bardensities, axis=-2 ) if not bardensities is None else None # shape (...,T,N)
        assert densities.shape == head+(T,N), f"Expected final densities to have shape {head+(T,N)}, got {densities.shape}"
        assert bardensities is None or bardensities.shape == head+(T,N), f"Expected final bardensities to have shape {head+(T,N)}, got {bardensities.shape}"

        def test():
            """ tests that the DLVs are consistent with the densities by computing the DLVs from the densities and comparing to the input DLVs """
            import numpy as nnp
            assert not bardensities is None, "debug_return_bardensities must be True to run this test"
            def relu(x):
                return nnp.maximum(x.dtype.type(0.),x)
            # test dlv
            mstrikes_t     = bb.to_numpy( mstrikes[...,t,:] )
            prev_d_at_t    = bb.to_numpy( bardensities[...,t,:] )
            mdensity_t     = bb.to_numpy( densities[...,t,:] )
            prev_call_at_t = nnp.sum( relu( mstrikes_t[...,None,:] - mstrikes_t[...,:,None] ) * prev_d_at_t[...,None,:], axis=-1 )
            call_t         = nnp.sum( relu( mstrikes_t[...,None,:] - mstrikes_t[...,:,None] ) * mdensity_t[...,None,:], axis=-1 )
            dts_t          = bb.to_numpy( dts[...,t] )

            Theta  = (call_t[...,1:-1] - prev_call_at_t[...,1:-1]) / dts_t[...,None]
            assert nnp.min(Theta) >= -self.eps.theta, f"Expected Theta to be positive, found min {nnp.min(Theta)} [tolerance 'theta' {self.eps.theta}]"
            dC     = nnp.diff(call_t, axis=-1) / nnp.diff(mstrikes_t, axis=-1)
            Gamma  = nnp.diff(dC, axis=-1) / ( mstrikes_t[...,2:] - mstrikes_t[...,:-2] )
            assert nnp.min(Gamma) >= -self.eps.gamma, f"Expected Gamma  to be non-negative, found min {nnp.min(Gamma)} [tolerance 'gamma' {self.eps.gamma}]"
            ok     = (Theta > 1E-8) & (Gamma > 1E-8)
            dlvsq_ = nnp.where(ok, Theta, 0.) / nnp.where( ok, mstrikes_t[...,1:-1]**2 * Gamma, 1. )
            assert nnp.min(dlvsq_) >= -self.eps.dlvsq, f"Expected dlvsq to be non-negative, found min {nnp.min(dlvsq_)} [tolerance 'dlvsq' {self.eps.dlvsq}]"
            dlvsqs_t = bb.to_numpy( dlvsqs[...,t,:] )+self.config.dlv_floor**2 
            if nnp.max( nnp.abs( dlvsq_ - dlvsqs_t ) ) > 1E-2:
                print(f"Error in DLV calculation at time {t}: max abs error {nnp.max( nnp.abs( dlvsq_ - dlvsqs_t ) )}")
                print("\ninput dlvar:\n", dlvsqs_t )
                print("\ntest dlvar:\n", dlvsq_ )
                print("\noks:\n", ok)
                print("\n*** Please contact authors ***")
        if self.debug_internal_tests:
            test()

        if bardensities is None or not self.debug_return_bardensities:
            return prices, densities
        return prices, densities, bardensities

    def fit_dlvsqs( self, fwd_vols : Backbone.Array,
                          opt_strikes : Backbone.Array,
                          opt_bid : Backbone.Array,
                          opt_ask : Backbone.Array,
                          opt_tte : Backbone.Array,*,
                          opt_is_calls : Backbone.Array | None = None,    
                          opt_mask : Backbone.Array | None = None,
                          max_error : float = 1E-3,
                          return_only_dlvsqs : bool = True,
                          verbose : Context = Context.quiet ) -> Backbone.Array:
        """
        Compute implied discrete local vols for given fwd_vols, option prices, time to expiry and strikes.
        This is a standalone numpy function for one set if options. It cannot be run for a batch of options. In this case, please loop manually.

        Parameters
        ----------
        fwd_vols : array of shape (T,)
            Forward volatilities for each time step
        opt_strikes : array of shape (M,)
            Strikes of the options to price
        opt_bid : array of shape (M,)
            Bid prices of the options to price
        opt_ask : array of shape (M,)
            Ask prices of the options to price
        opt_tte : array of shape (M,)
            Time to expiry of the options to price
        opt_is_calls : array of shape (M,)
            Boolean array indicating whether each option is a call (True) or put (False). If None, all options are assumed to be calls.
        opt_mask : array of shape (M,) | None, default ``None``
            Mask for which options to price; if False, the model will skip the computations for that option and return 0. This can be used to speed up computations when only a subset of options is needed.
        max_error : float | None, default 1E-3
            Maximum allowed error between model and market prices; the function will raise a RuntimeError if violated.
            This error indicates by how much the model price exceeds bid/ask, measured in spreads. It is the sum of:

            * ``Relu( model_price - opt_ask ) / spread``
            * ``Relu( opt_bid - model_price ) / spread``

            Set to ``None`` to disable this check.
        return_only_dlvsqs : bool, default True
            Whether to return only the implied DLVs squared, or also the fitted option prices and densities for debugging purposes. If True, only the implied DLVs squared are returned; if False, a tuple of (dlvsq, prices, densities) is returned.
        verbose : Context, default Context.quiet
            Verbose context for logging; this is used to log the optimization progress.
        
        Returns
        -------
        dlvsq : array of shape (T,N-2)
            Implied discrete local volatilities, squared, for each time step and strike outside the boundaries. c.f. https://arxiv.org/pdf/2601.11209
            
        (prices, densities) : tuple of arrays
            If ``return_only_dlvsqs`` is False, also return the fitted option prices and densities for debugging purposes.
        """

        with verbose.write_t("Importing 'cvxpy'... ", end='') as tme:
            import cvxpy as cp
            import numpy as nnp
            verbose.write(f"done; this tool {tme}.", head=False)

        bb          = self.backbone
        T, N        = self.TN
        M           = opt_tte.shape[-1]

        assert fwd_vols.shape == (T,), f"Expected fwd_vols to have shape {(T,)}, got {fwd_vols.shape} (this function does not support batch dimensions)"            
        assert len(opt_tte.shape) == 1, f"Expected opt_tte to have shape (M,), got {opt_tte.shape} (this function does not support batch dimensions)"
        assert opt_bid.shape == opt_ask.shape == opt_tte.shape == opt_strikes.shape, f"Expected opt_bid, opt_ask, opt_tte and opt_strikes to have the same shape, got {opt_bid.shape}, {opt_ask.shape}, {opt_tte.shape} and {opt_strikes.shape}"
        assert opt_is_calls is None or opt_is_calls.shape == opt_strikes.shape, f"Expected opt_is_calls to have the same shape as opt_strikes, got {opt_is_calls.shape} and {opt_strikes.shape}"
        assert opt_mask is None or opt_mask.shape == opt_strikes.shape, f"Expected opt_mask to have the same shape as opt_strikes, got {opt_mask.shape} and {opt_strikes.shape}"
        opt_mask     = bb.to_numpy( opt_mask ) if ( not opt_mask is None and not nnp.all(opt_mask) ) else None

        pweights, sqrtVars = self._compute_aux( fwd_vols=fwd_vols[None,:], opt_tte=opt_tte[None,:], opt_mask=opt_mask[None,:] if opt_mask is not None else None ) # add batch dimension for _compute_aux
        assert sqrtVars.shape == (1,M), f"Expected sqrtVars to have shape {(1,M)}, got {sqrtVars.shape}"
        assert pweights.shape == (1,T,M), f"Expected pweights to have shape {(1,T,M)}, got {pweights.shape}"
        sqrtVars     = sqrtVars[0,:]  # remove batch dimension
        pweights     = pweights[0,:,:]  # remove batch dimension

        mstrikes     = self.mstrikes
        fwd_vols     = bb.to_numpy( fwd_vols )
        opt_bid      = bb.to_numpy( opt_bid )
        opt_ask      = bb.to_numpy( opt_ask )
        opt_strikes  = bb.to_numpy( opt_strikes )
        opt_is_calls = bb.to_numpy( opt_is_calls ) if not opt_is_calls is None else None
        pweights     = bb.to_numpy( pweights )
        sqrtVars     = bb.to_numpy( sqrtVars )
        spread       = opt_ask - opt_bid
        opt_mid      = 0.5*(opt_ask + opt_bid)
        assert opt_mask is None or nnp.any(opt_mask), f"At least one option must be priced; all of 'opt_mask' is False"
        spread       = nnp.where( opt_mask, spread, 1. ) if opt_mask is not None else spread # if an option is masked, we set its spread to 1 so that it does not contribute to the optimization objective and we do not run into issues with zero spreads.
        assert nnp.min(opt_bid) > -self.eps.prices, f"Option bid must be positive, found min {nnp.min(opt_bid)} [tolerance 'prices' {self.eps.prices}]"
        assert nnp.min(spread) > -self.eps.prices, f"Option bid-ask spread must be positive, found min {nnp.min(spread)} [tolerance 'prices' {self.eps.prices}]"

        opt_call_mid = opt_mid if opt_is_calls is None else nnp.where(opt_is_calls, opt_mid, opt_mid + 1. - opt_strikes )
        opt_call_bid = opt_bid if opt_is_calls is None else nnp.where(opt_is_calls, opt_bid, opt_bid + 1. - opt_strikes )
        opt_call_ask = opt_ask if opt_is_calls is None else nnp.where(opt_is_calls, opt_ask, opt_ask + 1. - opt_strikes ) 
        del opt_bid, opt_ask, opt_mid

        # Optimization over densities
        # ---------------------------

        densities   = [ cp.Variable( N, nonneg=True ) for t in range(T) ] # density at each time step, shape (T,N)  
        conditions  = [ d >= 0. for d in densities ] + [ cp.sum(d) == 1. for d in densities ] # density conditions
        t_calls     = []
        tm1_calls   = []

        # martingale property for X
        # -------------------------
        # see https://arxiv.org/pdf/2601.11209

        for t in range(0,T):
            mdensity_t   = densities[t]
            mstrikes_t   = mstrikes[t,:]

            calls_of_t   = nnp.maximum( mstrikes_t[None,:] - mstrikes_t[:,None], 0. ) @ mdensity_t
            assert calls_of_t.shape == (N,), f"Expected calls_of_t to have shape {(N,)}, got {calls_of_t.shape}"
            t_calls.append( calls_of_t )

            if t>0:
                density_tm1  = densities[t-1]
                mstrikes_tm1 = mstrikes[t-1,:]
                calls_of_tm1 = nnp.maximum( mstrikes_tm1[None,:] - mstrikes_t[:,None], 0. ) @ density_tm1
                assert calls_of_tm1.shape == (N,), f"Expected calls_of_tm1 to have shape {(N,)}, got {calls_of_tm1.shape}"
            else:
                calls_of_tm1 = nnp.maximum( 0., 1. - mstrikes_t ) # at time 0, the call price is just the intrinsic value since the forward price is normalized to 1.
            conditions   += [ mdensity_t @ mstrikes_t == 1., calls_of_t >= calls_of_tm1 ]
            tm1_calls.append( calls_of_tm1 )
        del mdensity_t, mstrikes_t, calls_of_t, density_tm1, mstrikes_tm1, calls_of_tm1

        # computing arbitrary model prices
        # --------------------------------

        fit_calls    = None
        npbb         = BackboneNumpy()  # use numpy backbone for BS price calculation in cvxpy

        for t in range(T):
            spots_t     = mstrikes[t,:]
            mdensity_t  = densities[t]

            if not nnp.any(pweights[t,:] > 0.):
                verbose.write(f"Warning: no options with positive weight at time {t}; skipping price calculation for this time step.", head=False)
                continue

            calls_t   = npbb.bs_price(  spot=spots_t[None,:],
                                        strikes=opt_strikes[:,None], 
                                        sqrtVars=sqrtVars[:,None],
                                        high_precision = True,
                                        debug_verify_bounds = self.debug
                                        )  
            calls_t = nnp.where( opt_mask[:,None], calls_t, 0. ) if opt_mask is not None else calls_t
            calls_t = calls_t @ mdensity_t  # shape (M,)

            assert calls_t.shape == (M,), f"Expected calls_t to have shape {(M,)}, got {calls_t.shape}"
            calls_t = cp.multiply( pweights[t,:], calls_t )

            assert calls_t.shape == (M,), f"Expected weighted calls_t to have shape {(M,)}, got {calls_t.shape}"
            fit_calls = fit_calls + calls_t if not fit_calls is None else calls_t 
        del spots_t, mdensity_t, calls_t

        inv_spread  = nnp.minimum( self.config.max_inv_spread, 1./spread )
        inv_spread  = nnp.where( opt_mask, inv_spread, 0. ) if opt_mask is not None else inv_spread
        err_sqr     = cp.multiply( self.config.mid_fit_weight, cp.multiply( inv_spread, fit_calls - opt_call_mid )**2 )
        err_ask     = cp.multiply( inv_spread, cp.pos( fit_calls - opt_call_ask ) )
        err_bid     = cp.multiply( inv_spread, cp.pos( opt_call_bid - fit_calls ) )
        objective   = err_sqr + err_ask + err_bid

        with verbose.write_t(lambda : f"Solving for implied DLVs (squared) for {T} expiries and {N} strikes per expory... ", end='') as tme:
            prob        = cp.Problem( cp.Minimize( cp.sum( objective ) ), conditions )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.", category=UserWarning)
                try:
                    prob.solve( solver=self.config.cvxpy_solver, **( self.config.cvxpy_solver_options if self.config.cvxpy_solver_options is not None else {} ) )
                except cp.SolverError as e:
                    raise e
            
            fit_calls  = fit_calls.value
            densities  = [ d.value for d in densities ]

            if not opt_mask is None:
                assert nnp.max(err_sqr.value[~opt_mask]) == 0., f"Expected squared error for masked options to be 0, found max {nnp.max(err_sqr.value[~opt_mask])}"
                assert nnp.max(err_ask.value[~opt_mask]) == 0., f"Expected ask error for masked options to be 0, found max {nnp.max(err_ask.value[~opt_mask])}"
                assert nnp.max(err_bid.value[~opt_mask]) == 0., f"Expected bid error for masked options to be 0, found max {nnp.max(err_bid.value[~opt_mask])}"
                err_sqr    = nnp.max( err_sqr.value[opt_mask] )
                err_ask    = nnp.max( err_ask.value[opt_mask] )
                err_bid    = nnp.max( err_bid.value[opt_mask] )
            else:
                err_sqr    = nnp.max( err_sqr.value )
                err_ask    = nnp.max( err_ask.value )
                err_bid    = nnp.max( err_bid.value )
            err        = max( err_ask, err_bid )

            if not max_error is None and err > max_error:
                if prob.status == "optimal":    
                    raise RuntimeError(f"Solver {self.config.cvxpy_solver} failed to find a solution within the specified error tolerance: maximum weighted error {err:.4e} (ask {err_ask:.4e}, bid {err_bid:.4e}, sqr {err_sqr:.4g}) exceeds boundary {max_error:.4e}.")
                raise RuntimeError(f"Solver {self.config.cvxpy_solver} failed to find a solution within the specified error tolerance: maximum weighted error {err:.4e} (ask {err_ask:.4e}, bid {err_bid:.4e}, sqr {err_sqr:.4g}) exceeds boundary {max_error:.4e}. Solver returned status '{prob.status}'.")
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"Solver returned '{prob.status}' status. However, maximum weighted error is {err:.4e} below the boundary {max_error:.4e}, so we continue.")
            verbose.write(f"done: maximum weighted error is {err:.4g}. this tool {tme}.", head=False)

        # computing DLVS
        # we use the call prices computed earlier

        with verbose.write_t(lambda : f"Computing DLVs (squared)... ", end='') as tme:
            t_calls      = [ tc.value for tc in t_calls ]
            tm1_calls    = [ ( tmc.value if not isinstance(tmc, nnp.ndarray) else tmc ) for tmc in tm1_calls ] # tm1_calls[0] is not a cvxpy expression, so we keep it as a numpy array 
            dts          = self.geometry.dts
            dlvsq        = nnp.zeros( (T,N-2), dtype=opt_call_bid.dtype )

            for t in range(T):
                # extract calls defined on the same strike range in t
                strikes_t = mstrikes[t,:]
                calls_t   = t_calls[t]
                calls_tm1 = tm1_calls[t]
                dT        = dts[t]

                # DEBUG TODO: remove
                def test():
                    """ tests that the call prices are consistent with the densities by computing the call prices from the densities and comparing to the input call prices """
                    strikes_t = mstrikes[t,:]
                    d_t = densities[t]
                    c_t = nnp.maximum( 0., strikes_t[None,:] - strikes_t[:,None] ) @ d_t
                    assert nnp.max(nnp.abs(c_t - calls_t)) < 1e-8, f"Error in call price calculation at time {t}: {nnp.max(nnp.abs(c_t - calls_t))}"
                    if t>0:
                        d_tm1  = densities[t-1].value
                        barc_t = nnp.maximum( 0., mstrikes[t-1,None,:] - strikes_t[:,None] ) @ d_tm1
                        assert nnp.max(nnp.abs(barc_t - calls_tm1)) < 1e-8, f"Error in call price calculation at time {t}: {nnp.max(nnp.abs(barc_t - calls_tm1))}"
                #test()

                Theta     = ( calls_t[1:-1] - calls_tm1[1:-1] ) / dT
                assert nnp.min(Theta) >= -self.eps.theta, f"Call price must increase with time to expiry, found min Theta {nnp.min(Theta)} at time {t} [tolerance 'theta' {self.eps.theta}]"
                assert nnp.all(nnp.isfinite(Theta)), f"Theta contains non-finite values at time {t}"
                Theta     = nnp.maximum( Theta, 0. ) 
                dC        = nnp.diff( calls_t ) / nnp.diff( strikes_t )
                p         = nnp.diff( dC )
                assert nnp.min(p) >= -self.eps.densities, f"Density error: found {nnp.min(p)} at time {t} [tolerance 'densities' {self.eps.densities}]"
                assert nnp.all(nnp.isfinite(p)), f"Density contains non-finite values at time {t}"
                Gamma     = p / ( strikes_t[2:] - strikes_t[:-2] )
                assert nnp.all(nnp.isfinite(Gamma)), f"Gamma contains non-finite values at time {t}"
                GammaKsq  = strikes_t[1:-1]**2 * Gamma
                assert nnp.min(GammaKsq) >= -self.eps.gamma, f"Gamma error: found {nnp.min(GammaKsq)} at time {t} [tolerance 'gamma' {self.eps.gamma}]"
                GammaKsq  = nnp.maximum( GammaKsq, 0. )
                valid     = (GammaKsq > 1E-8) & (Theta > 1E-8)
                Theta     = nnp.where( valid, Theta, 0. )
                GammaKsq  = nnp.where( valid, GammaKsq, 1. )
                dlvsq_t   = Theta / GammaKsq
                assert dlvsq_t.shape == (N-2,), f"Expected dlvsq_t to have shape {(N-2,)}, got {dlvsq_t.shape}"
                assert nnp.all(nnp.isfinite(dlvsq_t)), f"DLV squared contains non-finite values at time {t}"
                assert nnp.min(dlvsq_t) > -self.eps.dlvsq, f"DLV squared must be positive, found min {nnp.min(dlvsq_t)} [tolerance 'dlvsq' {self.eps.dlvsq}]"
                dlvsq[t,:]  = nnp.maximum( dlvsq_t, 0. )
            verbose.write("done; this tool {tme}.", head=False)

        # consistency step:
        # The __call__ function will add dlv_floor**2 to the input dlvsqs,
        # so here we substract it to be consistent
        dlvsq = nnp.clip( dlvsq-self.config.dlv_floor**2, min=0. ) 

        if return_only_dlvsqs:
            return dlvsq
        
        fit_prices = nnp.where( opt_is_calls, fit_calls, fit_calls - 1. + opt_strikes ) if opt_is_calls is not None else fit_calls    
        densities  = bb.stack( densities, axis=-2 ) # shape (...,T,N)
        assert densities.shape == (T,N), f"Expected final densities to have shape {(T,N)}, got {densities.shape}"
        assert fit_prices.shape == (M,), f"Expected final fit_prices to have shape {(M,)}, got {fit_prices.shape}"
        return dlvsq, fit_prices, densities
    