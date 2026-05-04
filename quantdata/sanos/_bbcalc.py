"""
SANOS

Model geometry
"""

from dataclasses import dataclass
import math
import numpy as np
from cdxcore.version import version
from cdxcore.err import verify_inp, verify
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.utils.backbone import Backbone, BackboneNumpy
from quantdata.sanos._geometry import Geometry
from quantdata.sanos.tridiag import solve_sanos_tridiagonal

class SanosBackbone(object):
    """
    SANOS Calculations which can be shared across backbones.
    Calculations in this class depend on market data.
    """

    def __init__(self, geometry : Geometry,
                       bb       : Backbone|None = None,
                       debug    : bool = False ) -> None:
        self.geometry = geometry
        self.bb       = bb if not bb is None else BackboneNumpy()
        self.debug    = debug

        self.bb_xstrikes = bb.to_array( geometry.xstrikes )
        self.bb_xtimes   = bb.to_array( geometry.xtimes )
        self.bb_density0 = bb.to_array( geometry.density0 )
        self.bb_calls0   = bb.relu( 1. - self.bb_xstrikes[0,1:-1] )

        wn, wp, L        = geometry.transition_operators() # these are lists of numpy arrays of np.float64
        self.bb_wn       = [ bb.to_array( wn_t ) for wn_t in wn ]
        self.bb_wp       = [ bb.to_array( wp_t ) for wp_t in wp ]
        self.bb_L        = [ bb.to_array( L_t ) for L_t in L ]

        if __debug__:
            # checking this just in case
            strikes0 = geometry.strikes[0]
            density0 = geometry.density0
            ix_1     = np.argmin( np.abs( strikes0 - 1. ) )
            assert strikes0.shape == density0.shape, f"Internal error: expected geometry.strikes[0] and geometry.density0 to have the same shape, found {strikes0.shape} and {density0.shape}"
            assert strikes0[ix_1] == 1., f"Internal error: expected strike 1 to be in geometry.strikes[0], found {strikes0}"
            assert density0[ix_1] == 1., f"Internal error: expected density 1 to be in geometry.density0, found {density0}"
            assert np.abs( np.sum(density0) - 1. ) < 1e-10, f"Internal error: expected geometry.density0 to sum to 1, found sum {np.sum(density0)}"

    def verify( self, condition : callable, error : callable|str ):
        """ Raise ``error()`` if ``conditipon()`` is False, in debug mode only """
        if self.debug:
            verify( condition(), error )

    @property
    def grid_mode(self) -> bool:
        return self.geometry.grid_mode

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.geometry.T

    @property
    def N(self) -> int|None:
        """ Returns the common number of strikes accross expiries, if that exists, or None otherwise. """
        return self.geometry.N
    
    @property
    def bb_strikes(self) -> Backbone.ndarray|list[Backbone.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        if self.grid_mode:
            return self.bb_xstrikes[1:,1:-1]
        else:
            return [ bb_strikes_t[1:-1] for bb_strikes_t in self.bb_xstrikes[1:] ]

    def densities( self, dlvsqs : Backbone.Array|list[Backbone.Array] ) -> list[Backbone.Array]:
        """
        Compute densities from DLVs.
        This function is batch compatible.

        Parameters
        ----------
        dlvsqs : array of shape (HEAD,T,N-2) or list of T arrays of shape (HEAD,N-2)
            The DLVs squared, i.e. the local variance divided by the time step, at each model time and strike. If a list is provided, it must have length T and each element must have shape (HEAD,N-2).    
        
        Returns
        -------
        Densites : list of Backbone.Array
            List of ``T`` densities at each model time, not including the initial density at time 0 (which can be accessd via ``bb_density0``). 
        """

        bb          = self.bb
        head        = dlvsqs.shape[:-2]
        hext        = (1,)*len(head)     
        T           = self.T
        N           = self.N # will be None if not in grid mode
        _text       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert len(head) <= len(_text), f"Too many batch dimensions: expected at most {len(_text)}, got {len(head)}"
        einhext     = _text[:len(head)] 

        self.verify( lambda : (self.bb_density0 >= -self.eps.weights) & (self.bb_density0 - 1. <= self.eps.weights), lambda : f"Density 0 must be between 0 and 1; found min {bb.min(self.bb_density0)} and max {bb.max(self.bb_density0)} [tolerance 'weights' {self.eps.weights}]" )
        self.verify( lambda : bb.abs( bb.sum( self.bb_density0, axis=-1 ) - 1. ) < self.eps.densities, lambda : f"Density 0 must have positive mass, found sum {bb.max(bb.abs( bb.sum( self.bb_density0, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

        mxdensities  = [ self.bb_density0 ]
        dts          = bb.diff(self.xtimes) # in bb array
        mstrikes     = self.strikes # in bb array
        mstrikes     = [ mstrikes[t,:] for t in range(T) ] if not self.grid_mode else mstrikes

        if hasattr(dlvsqs, "shape"):
            assert self.geometry.grid_mode, f"Expected 'dlvsqs' to be a Backbone.Array only in grid mode, found type {type(dlvsqs)} and grid_mode {self.geometry.grid_mode}"
            assert dlvsqs.shape[-2:] == (T,N-2), f"Expected 'dlvsqs' to have shape ending with {(T,N-2)}, found {dlvsqs.shape}"
            dlvsqs = [ dlvsqs[...,t,:] for t in range(T) ]
        else:
            assert len(dlvsqs) == T, f"Expected 'dlvsqs' to have length {T}, found {len(dlvsqs)}"
            for t in range(T):
                assert dlvsqs[t].shape[-1] == N-2, f"Expected each element of 'dlvsqs' to have shape ending with {(N-2,)}, found {dlvsqs[t].shape}"

        bb_wn        = [ bb.reshape( wn, hext+wn.shape ) for wn in self.bb_wn ]
        bb_wp        = [ bb.reshape( wp, hext+wp.shape ) for wp in self.bb_wp ]
        bb_L         = [ bb.reshape( L, hext+L.shape ) for L in self.bb_L ]
    
        for t in range(T):
            # solve for density in t
            # ----------------------
            mstrikes_t   = mstrikes[t]
            Nt           = mstrikes_t.shape[-1]
            dlvsqs_t     = dlvsqs[t]
            dlvar_t      = (dlvsqs_t+self.config.dlv_floor**2) * dts[t]  # local variance at time t, shape (...,N-2)
            assert dlvar_t.shape[-1] == Nt-2, f"Expected dlvar_t shape to end in {Nt-2}, found {dlvar_t.shape}"
            
            w_minus      = dlvar_t * bb_wn[t]
            w_plus       = dlvar_t * bb_wp[t]

            mdensity_tm1 = mxdensities[-1]
            mdensity_t   = bb.einsum( f"{einhext}mn, {einhext}n -> {einhext}m", bb_L[t], mdensity_tm1 )  # interpolate density to time t
            mdensity_tm1 = None # free memory; del does not work in torch compile

            self.verify( lambda : (mdensity_t >= -self.eps.densities) & (mdensity_t - 1. <= self.eps.densities), lambda : f"Interpolated density at {t} must be between 0 and 1; found min {bb.min(mdensity_t)} and max {bb.max(mdensity_t)} [tolerance 'densities' {self.eps.densities }]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            self.verify( lambda : bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ) <= self.eps.densities, lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

            mdensity_t  = solve_sanos_tridiagonal(bb, w_minus=w_minus, w_plus=w_plus, rhs=mdensity_t ) # keep
            mxdensities.append( mdensity_t )

        return mxdensities[1:]
    
    def dlvsqs(self, densities : list[Backbone.Array] ) -> Backbone.Array|list[Backbone.Array]:
        """
        Computes the sqrt-variance for a list of options given forward volatilites and current model times.

        Parameters
        ----------
        fwd_vols : array of shape (HEAD,T,) or float
            Forward volatilities for each model time. If a single float is provided, it will be broadcasted to all model times.
        opt_lix : array of shape (HEAD,M,)
            Per option, the lower index into ``geometry.xtimes`` such that ``xtimes[lix]<t``. This can be computed using :meth:`time_interpolation_weights` function.
        opt_vweight : array of shape (HEAD,M,)
            Per option, the volatility interpolation weight. This can be computed using :meth:`time_interpolation_weights`.
        opt_vextra : array of shape (HEAD,M,)
            Per option, the volatility extrapolation weight. This can be computed using :meth:`time_interpolation_weights`.

        """
        bb          = self.bb
        densities   = [ self.bb_density0 ] + [ bb.to_array( d ) for d in densities ] # convert to bb arrays if not already
        dts         = [ bb.diff( self.geometry.xtimes ) ]
        T           = self.T  
        dlvsq       = bb.zeros( (T,N-2), dtype=dts.dtype ) if self.grid_mode else None else []
        mstrikes    = self.bb_strikes
        

        
        for t in range(T):
            # extract calls defined on the same strike range in t
            strikes_t = mstrikes[...,t,:] if self.grid_mode else mstrikes[...,t]
            Nt        = strikes_t.shape[-1]
            Nt        = Ns[t]
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


    def price(  self,
                fwd_vols : Backbone.Array|float|None,
                densities : Backbone.Array|list[Backbone.Array],
                opt_strikes : Backbone.Array,
                opt_tte : Backbone.Array,*,
                opt_is_calls : Backbone.Array| None,    
                opt_mask : Backbone.Array|None,
                opt_lix : Backbone.Array, 
                opt_dweight : Backbone.Array, 
                opt_vweight : Backbone.Array,
                opt_vextra : Backbone.Array
            ):
        """
        Prices a number of options.

        This function uses the Black-Scholes model to price options based on the provided densities and other parameters. 
        This function is torch compilable.

        Returns
        -------
        Results : type
            See :meth:`__call__` for details.
        """
        bb       = self.bb
        T        = self.T
        N        = self.N # will be None if self.xstriks are a numpy array
        head     = opt_strikes.shape[:-1]
        hext     = (1,)*len(head)     

        if len(opt_strikes.shape) != 1:
            if self.grid_mode:
                assert isinstance( densities, Backbone.Array), f"Expected 'densities' to be a Backbone.Array in grid mode, found type {type(densities)}"
                assert len(densities.shape) == len(opt_strikes.shape)+1, f"Expected 'densities' to have one more dimension than 'opt_strikes' in grid mode, found {len(densities.shape)} and {len(opt_strikes.shape)}"
            else:
                for d in densities:
                    assert len(d.shaope) == len(opt_strikes.shape), f"Expected each element of 'densities' to have the same dimension as 'opt_strikes' in list mode, found {len(d.shape)} and {len(opt_strikes.shape)}"

        if not fwd_vols is None:
            if isinstance( fwd_vols, (float, int, np.floating)):
                fwd_vols = bb.full( hext+(T,), fwd_vols, dtype=opt_strikes.dtype )
            self.verify( lambda : bb.min( fwd_vols ) >= 0., lambda : f"Forward volatilities must not be negative, found min {bb.min(fwd_vols)}" )

        assert fwd_vols is None or fwd_vols.shape[-1] == T, f"Expected fwd_vols to have shape last dimension {T}, found {fwd_vols.shape}"            
        assert opt_strikes.shape == opt_tte.shape, f"Expected 'opt_strikes' and 'opt_tte' to have the same shape; found {opt_strikes.shape} and {opt_tte.shape}"
        assert opt_is_calls is None or opt_is_calls.shape == opt_tte.shape, f"Expected 'opt_is_calls' to have the same shape as 'opt_tte'; found {opt_is_calls.shape} and {opt_tte.shape}"
 
        # compute sqrt var per option
        # ---------------------------

        opt_sqrtVars = self.interpolate_sqrtVars( fwd_vols=fwd_vols, opt_lix=opt_lix, opt_vweight=opt_vweight, opt_vextra=opt_vextra )
        assert opt_sqrtVars.shape == opt_lix.shape, f"Expected 'opt_sqrtVars' to have the same shape as 'opt_lix', found {opt_sqrtVars.shape} and {opt_lix.shape}"
        self.verify( lambda : bb.all( np.isfinite(opt_sqrtVars) ) and bb.all( opt_sqrtVars >= 0. ), lambda : f"Internal error: opt_sqrtVars must be non-negative and finite, found shape and min {bb.min(opt_sqrtVars)} and max {bb.max(opt_sqrtVars)}" )
        self.verify( lambda : bb.all( 0. <= opt_dweight ) and bb.all( opt_dweight <= 1. ), lambda : f"'opt_dweight' must be between 0 and 1, found min {bb.min(opt_dweight)} and max {bb.max(opt_dweight)}")

        # computing arbitrary model prices
        # --------------------------------
        # 'opt_lix' is an index in T+1 xtimes; mstrikes has T expiries.

        mstrikes     = self.strikes # is bb array
        T            = self.mstrikes.shape[0]
        M            = opt_strikes.shape[-1]

        if self.grid_mode:
            assert densities.shape == mstrikes.shape, f"Expected densities to have the same shape as geometry.strikes, found {densities.shape} and {mstrikes.shape}"
            assert not N is None, "Internal error: expected N to be not None"
            N            = mstrikes.shape[-1]
            eff_lix      = bb.clip(opt_lix-1, min=0)
            calls_prev   = bb.bs_price( spot=mstrikes[...,eff_lix,:],
                                        strikes=opt_strikes[...,:,None], 
                                        sqrtVars=opt_sqrtVars[...,:,None],
                                        mask=opt_mask[...,:,None],
                                        high_precision = True,
                                        debug_verify_bounds = self.debug
                                    ) * opt_dweight[...,:,None]
            assert calls_prev.shape[-2:] == (M,N), f"Expected calls_prev shape to end with {(M,N)}, got {calls_prev.shape}"
            calls_prev = calls_prev @ densities[...,eff_lix,:,None]
            assert calls_prev.shape[-2:] == (M,1), f"Expected calls_prev to have shape {(M,1)}, got {calls_prev.shape}"
            calls_prev = calls_prev[...,0] # remove last dim

            calls_next   = bb.bs_price(spot=mstrikes[...,opt_lix,:],
                                        strikes=opt_strikes[...,:,None], 
                                        sqrtVars=opt_sqrtVars[...,:,None],
                                        mask=opt_mask[...,:,None],
                                        high_precision = True,
                                        debug_verify_bounds = self.debug
                                    ) * ( 1.- opt_dweight[...,:,None] )
            calls_next    = calls_next @ densities[..., opt_lix,:,None]
            assert calls_next.shape[-2:] == (M,1), f"Expected calls_next to have shape {(M,1)}, got {calls_next.shape}"
            calls_next = calls_next[...,0] # remove last dim
            fit_calls  = calls_prev + calls_next 
            self.verify( lambda : bb.all( np.isfinite(fit_calls) ), lambda : "fit_calls not finite")
            if opt_is_calls is None:
                return fit_calls
            # C-P=1-K => P=C+K-1
            return bb.where( opt_is_calls | ~opt_mask, fit_calls, fit_calls + opt_strikes - 1. )

        # list mode
        # ---------
        # At the moment only works in non-batch mode.
        # Might also be buggy in torch.

        assert len( opt_strikes.shape) == 1, f"List mode only supports non-batch option data; expected opt_strikes to be 1D, found shape {opt_strikes.shape}"

        assert len(densities) == T, f"Expected densities to have length {T}, found {len(densities)}"
        assert N is None, "Internal error: expected N to be None"
        unique__lix = bb.unique(opt_lix, axis=-1)
        fit_calls   = bb.zeros_like( opt_strikes )
        mstrikes    = self.bb_strikes

        for lix in unique__lix:
            opt_lix_mask = (opt_lix == lix)
            
            lix_strikes  = mstrikes[opt_lix_mask][:,None]
            lix_sqrtVars = opt_sqrtVars[opt_lix_mask][:,None]
            lix_weight   = opt_dweight[opt_lix_mask,None]
            lix_mask     = opt_mask[opt_lix_mask,None] if not opt_mask is None else None    
            lixM         = np.sum(opt_lix_mask)

            eff_lix      = max(0, lix-1)
            eff_spot     = mstrikes[eff_lix,:]
            eff_N        = eff_spot.shape[0]
            calls_prev   = bb.bs_price( spot=eff_spot[None,:],
                                        strikes=lix_strikes, 
                                        sqrtVars=lix_sqrtVars,
                                        mask=lix_mask,
                                        high_precision = True,
                                        debug_verify_bounds = self.debug
                                    ) * lix_weight
            assert calls_prev.shape == (lixM,eff_N), f"Expected calls_prev to have shape {(lixM,eff_N)}, got {calls_prev.shape}"
            calls_prev  = calls_prev @ densities[eff_lix,:,None]
            assert calls_prev.shape == (lixM,1), f"Expected calls_prev to have shape {(lixM,)}, got {calls_prev.shape}"
            calls_prev = calls_prev[...,0]
            eff_lix = None
            eff_spot = None
            eff_N = None

            lix_spot     = mstrikes[lix,:]
            lix_N        = lix_spot.shape[0]
            calls_next   = bb.bs_price(spot=lix_spot[None,:],
                                        strikes=lix_strikes, 
                                        sqrtVars=lix_sqrtVars,
                                        mask=lix_mask,
                                        high_precision = True,
                                        debug_verify_bounds = self.debug
                                    ) * ( 1.- lix_weight )
            assert calls_next.shape == (lixM,lix_N), f"Expected calls_next to have shape {(lixM,lix_N)}, got {calls_next.shape}"
            calls_next    = calls_next @ densities[..., opt_lix,:,None]
            assert calls_next.shape == (lixM,1), f"Expected calls_next to have shape {(lixM,)}, got {calls_next.shape}"
            calls_next = calls_next[...,0]
            fit_calls[opt_lix_mask] = calls_prev + calls_next
            calls_prev = None
            calls_next = None
            mstrikes = None
            lix_spot = None
            lix_N = None
            lix_strikes = None
            lix_sqrtVars = None
            lix_weight = None
            lixM = None
        if opt_is_calls is None:
            return fit_calls
        # C-P=1-K => P=C+K-1
        return bb.where( opt_is_calls | ~opt_mask, fit_calls, fit_calls + opt_strikes - 1. )
    
