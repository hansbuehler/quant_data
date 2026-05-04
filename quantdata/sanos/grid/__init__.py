"""
Grid Sanos

SANOS itself is a rather generic framework which can accomodate any number of model strikes and expires, potentially depending on market data for example when used in "interpolatnion mode" as in https://arxiv.org/pdf/2601.11209.
Grid SANOS, on the other hand, specifies SANOS on a fixed strike and expiry grid for a machine learned representation of an option surface. 

The framework here represents a fixed grid approach. The core class is :class:`dys.grid_sanos.GridGeometry`, which represents the geometry of the grid.

**Main components**

* :class:`GridGeometry`: represents the expiries and strikes of the grid, but no market data.
* :class:`GridSanos`: represents the SANOS framework on the grid, including market data and transition operators.
* :class:`Fitter`: fits the SANOS model to market data.

```python

    geometry = GridGeometry.create( ... )  

    fitter   = Fitter( geometry, ... )
    sanos    = fitter( market_data )
    prices   = sanos.price( strikes, expiries )


````
"""

from __future__ import annotations

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.utils.backbone import Backbone, BackboneNumpy
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
import math as math
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
import warnings as warnings
from collections.abc import Sequence

@dataclass(frozen=True)
class GridGeometry:
    xexpiries  : np.ndarray            #: model expiries in years, starting from 0, of size (T+1,) where T is the number of non-zero expiries. The first ``xexpiry`` must be zero.
    xstrikes   : np.ndarray            #: internal model strikes
    dlv_min    : float = 0.01          #: minimum discrete local volatility (dlv) value.
    dlv_max    : float = 5.0           #: maximum discrete local volatility (dlv) value.
    fwdvol_min : float = 0.0           #: minimum backbone forward volatility value.
    vol_interp : str = "lin_sqrt_time"  #: Term volatility interpolation method. Must be a 'lin_var' or 'lin_sqrt_time'
    backbone   : Backbone = field(default_factory=BackboneNumpy) #: Backbone for numerical computations. Default is numpy.

    def __post_init__(self):
        verify_inp( self.dlv_min > 0., lambda : f"'dlv_min' must be positive, but found {self.dlv_min:.4g}") 
        verify_inp( self.dlv_max > self.dlv_min, lambda : f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min >= 0., lambda : f"'fwdvol_min' must be non-negative; found {self.fwdvol_min:.4g}" )
        verify_inp( self.vol_interp in {"lin_sqrt_time", "lin_var"}, lambda : f"'vol_interp' must be either 'lin_sqrt_time' or 'lin_var', but found {self.vol_interp}" )

        verify_inp( len(self.xexpiries.shape) == 1 and len(self.xexpiries) > 1, lambda : "'xexpiries' must be non-empty 1D array of at least two expiries, starting in 0" )
        verify_inp( self.xexpiries[0] == 0., lambda : f"First expiry must be zero; found {self.xexpiries[0]}" )
        verify_inp( np.min( np.diff(self.xexpiries) ) > 0., lambda : "'xexpiries' must be strictly increasing" )

        verify_inp( len(self.xstrikes.shape) == 2 and self.xstrikes.shape[0] == len(self.xexpiries), "'strikes' must be a list of 1D arrays or a 2D array with the same number of rows as 'expiries'")
        verify_inp( self.xstrikes.shape[1] >= 5, lambda: f"'strikes' must have at least 5 elements in its second dimension, found shape {self.xstrikes.shape}" )
        verify_inp( np.all( np.diff(self.xstrikes,axis=1) > 0), lambda: f"'strikes' must be strictly increasing in its second dimension, found {np.min(np.diff(self.xstrikes,axis=1)):.4g}")
        verify_inp( np.all( self.xstrikes[:,1] < 1.) and np.all( self.xstrikes[:,-2] > 1.), lambda: f"'strikes' in its second dimension must include positive values below and above 1, found boundary values {self.xstrikes[:,1]} and {self.xstrikes[:,-2]}" )
        verify_inp( np.all( np.diff( self.xstrikes[:,0] ) <= 0.), lambda: f"The lowest 'strikes' must not increase in time, {self.xstrikes[:,0]:.4g}, {np.diff( self.xstrikes[:,0] ):.4g}")
        verify_inp( np.all( np.diff( self.xstrikes[:,-1] ) >= 0.), lambda: f"The highest 'strikes' must not decrease in time, {self.xstrikes[:,-1]:.4g}, {np.diff( self.xstrikes[:,-1] ):.4g}")

        # strikes and density at time 0
        # we insert a "1" here.
        # we compute the density dynamically.
        strikes0       = self.xstrikes[0,1:-1]
        N1             = len(strikes0)
        assert strikes0[0] < 1. < strikes0[-1], f"Internal error: strikes should include 1, but found {strikes0}."
        ix_1           = min(N1-1,max(0,np.abs(strikes0 - 1.).argmin()))
        strikes0[ix_1] = 1.
        density0       = np.zeros_like( strikes0 )
        density0[ix_1] = 1.
        self.__dict__['_density0'] = density0

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.xexpiries.shape[0]-1

    @property
    def N(self) -> int:
        """ Returns the common number of strikes accross expiries, if that exists, or None otherwise. """
        return self.xstrikes.shape[1]-2

    @property
    def TN(self) -> tuple[int, int]:
        """ Returns the number of expiries and strikes, i.e. use ``T, N = geometry.TN`` to unpack. """
        return self.T, self.N

    @property
    def strikes(self) -> np.ndarray:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.xstrikes[1:,1:-1]

    @property
    def expiries(self) -> np.ndarray:
        """ Returns the model expiries excluding the zero expiry """
        return self.xexpiries[1:]

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density """
        return self._density0   

    @staticmethod
    def create_grid_nms( *, initial_t : float = 1./DAYS_PER_YEAR,
                            horizon_t : float = 5., 
                            num_times : int = 11,
                            nms_range : float | tuple[float,float] = 1.,
                            num_nms   : int = 11,
                            dlv_min   : float = 0.01,
                            dlv_max   : float = 5.,
                            fwdvol_min  : float = 0.01,
                            vol_interp: str = "lin_sqrt_time",
                            lower_scale : float = 0.5,
                            upper_scale : float = 1.5,
                            dtype : np.dtype = np.float64
            ):
        """
        Creates a GridGeometry with reduced parameters

        Parameters
        ----------
        initial_t : float, default ``1./DAYS_PER_YEAR``
            Initial time of the grid, in years. Default is 1 day.
        horizon_t : float, default ``5.``
            Horizon time of the grid, in years. Default is 5 years.
        num_times : int, default ``11``
            Number of time points in the grid.
        nms_range : float or tuple of two floats, default ``1.``
            Range of NMS for the strikes. If a single float is provided, the range will be symmetric around 0.
        num_nms : int, default ``11``
            Number of NMS points for the strikes. Default is 11.
        vol_interp : str, default ``"lin_sqrt_time"``
            Volatility interpolation method. Must be 'lin_sqrt_time' or 'lin_var'. Default is 'lin_sqrt_time'.

        Returns
        -------
        geometry : :class:`dys.grid_sanos.GridGeometry`
            The created geometry.
        """

        verify_inp( initial_t > 0., f"initial_t must be positive, found {initial_t:.4g}" )
        verify_inp( horizon_t > initial_t, f"horizon_t must be greater than initial_t, found {horizon_t:.4g} <= {initial_t:.4g}" )
        verify_inp( num_times >= 2, f"num_times must be at least 2, found {num_times}" )
        verify_inp( isinstance(nms_range, (float, tuple)), f"nms_range must be a float or a tuple of two floats, found {type(nms_range)}" )
        if isinstance(nms_range, tuple):    
            verify_inp( len(nms_range) == 2, f"nms_range tuple must have length 2, found {len(nms_range)}" )
            verify_inp( nms_range[0] < 0. and nms_range[1] > 0., f"nms_range must include negative and positive values, found {nms_range}" )
        else:
            verify_inp( nms_range > 0., f"nms_range must be positive, found {nms_range:.4g}" )
        verify_inp( num_nms >= 3, f"num_nms must be at least 3, found {num_nms}" )

        nms_lo           = float(-nms_range if isinstance(nms_range, float) else nms_range[0])
        nms_hi           = float(+nms_range if isinstance(nms_range, float) else nms_range[1])
        verify_inp( nms_lo < 0. and nms_hi > 0., f"'nms_lo' must be negative and 'nms_hi' must be positive, found nms_lo {nms_lo:.4g} and nms_hi {nms_hi:.4g}" )    

        xexpiries        = np.zeros( (num_times+1,), dtype=dtype )  
        xexpiries[1:]    = np.exp( np.linspace(math.log(initial_t), math.log(horizon_t), num_times) )
        num_xnms         = num_nms//6
        num_inms         = num_nms - 3*num_xnms

        nms              = np.linspace( nms_lo, nms_hi, num_inms )
        inner_strikes    = NMS.from_nms( nms[None,:], t=xexpiries[:,None] )

        xstrikes         = np.zeros((num_times+1,num_nms+2), dtype=dtype )
        xstrikes[:,:num_xnms*2+1] = np.linspace(lower_scale, 1., 2*num_xnms+2)[None,:-1] * inner_strikes[:,0][:,None] 
        xstrikes[:,-num_xnms-1:] = np.linspace(1., upper_scale, num_xnms+2)[None,1:] * inner_strikes[:,-1][:,None] 
        xstrikes[:,2*num_xnms+1:-num_xnms-1] = inner_strikes
        xstrikes[:,-1] *= math.sqrt( upper_scale )

        strike_diff = np.diff( xstrikes, axis=1 )
        assert np.min( strike_diff ) > 0., f"Per time step: strikes must increase. Found {np.min(strike_diff):.4g}"
        return GridGeometry(
            xexpiries  = xexpiries,
            xstrikes   = xstrikes,
            dlv_min    = dlv_min,
            dlv_max    = dlv_max,
            fwdvol_min = fwdvol_min ,
            vol_interp = vol_interp,
        )

    # -------------------------------------------
    # Numerical utilities
    # -------------------------------------------

    def time_interpolation_weights( self,
                                    opt_tte  : np.ndarray, 
                                    opt_mask : np.ndarray|None = None,
                                    dtype    : np.dtype|None = None ) -> dict[str, np.ndarray]:
        """
        Computes option time interpolation weights for variance and densities.
        These do not depend on market data and should be computed ahead of time.

        * ``lix``: per option the lower index into ``self.xexpiries`` such that such that ``self.xexpiries[opt_lix-1] < opt_tte <= self.expiries[opt_lix]``. Note that ``0<opt_lix<len(self.xexpiries)``.
        * ``dweight``: per option the lower interpolation weight for density interpolation.
        * ``vweight``: per option the lower interpolation weight for variance interpolation (can be linear in variance or in sqrt time).
        * ``vextra``: per option extrapolation with final volatility (linear or sqrt)

        Call price BS sqrt-variance is computed as follows in ``lin_sqrt_time`` interpolation mode:

            * if ``t_i < xexpiries[0]`` then ``lix_i = 0`` and 
                ``sqrtVar_i = (1-vweight_i) * sqrtVars[lix_i]``.

            * if ``xexpiries[0] <= t_i <= xexpiries[-1]`` then ``0 < lix_i <= T-1`` and
                ``sqrtVar_i = vweight_i * sqrtVars[lix_i-1] + (1-vweight_i) * sqrtVars[lix_i]``.

            * if ``t_i > xexpiries[-1]`` then we set ``lix_i == T-1`` and
                ``sqrtVar_i = (1 - vweight_i) * sqrtVars[T-1] + vextra_i * vol_extra`` where ``vweight_i = 0``.

        Once set, we also interpolate in model densities.

            * if ``t_i < xexpiries[0]`` then ``opt_lix_i = 0`` and
                ``call_i = (1-dweight_i) * call[opt_lix_i]`` with ``dweight_i = 0``.

            * if ``xexpiries[0] <= t_i <= xexpiries[-1]`` then ``0 < opt_lix_i <= T-1`` and
                ``call_i = dweight_i * call[lix_i-1] + (1-dweight_i) * call[lix_i]``.

            * if ``t_i > xexpiries[-1]`` then we set ``lix_i == T-1`` and
                ``call_i = (1 - dweight_i) * call[T-1]`` with ``dweight_i = 0``.

        Note that ``0<lix<len(self.xexpiries)``.
        """
        T        = self.T
        shape    = opt_tte.shape
        dtype    = dtype if not dtype is None else opt_tte.dtype

        # we use the mask to set opt_tte to 1; from thereon proceed as usual
        opt_mask = opt_mask if ( not opt_mask is None and not np.all(opt_mask) ) else None
        opt_tte  = np.where( opt_mask, opt_tte, 1. ) if opt_mask is not None else opt_tte 
        opt_tte  = np.asarray( opt_tte, dtype=np.float64 )
        assert np.min( opt_tte ) > 0., f"Time to expiry must be positive, found min {np.min(opt_tte)}"
        opt_tte  = np.reshape( opt_tte, (-1,)) # no-copy operatiob
        M        = len(opt_tte)

        # find indices ixtte such that xexpiries[ixtte-1] < opt_tte <= xexpiries[ixtte+1]
        ttes     = np.unique( opt_tte )
        sqrtTtes = np.sqrt( ttes )
        times    = np.asarray(self.expiries, dtype=np.float64)
        sqrtT    = np.sqrt( times )
        ixtte    = np.searchsorted( times, ttes, side='left' )
        assert np.min(ixtte) >= 0, f"Internal error - times[0]>0, so searchsorted must return indices of at least 0, found min {np.min(ixtte)}. Please contact authors."
        assert np.max(ixtte) <= T, f"Internal error - searchsorted must return indices of at most len(times), found max {np.max(ixtte)}. Please contact authors."
        assert ixtte.shape == ttes.shape, f"Expected ixtte to have shape {ttes.shape}, got {ixtte.shape}"

        # compute linear and sqrt-linear weights
        assert self.vol_interp in {"lin_var","lin_sqrt_time"}, f"Internal error: vol_interp must be 'lin_var' or 'lin_sqrt_time', found '{self.vol_interp}'"

        # create weights for ixteel-1. 
        # note that in case of extrapolation ixteel=T-1, hence the weight for ixteel-1 will be 0.
        # dweight is the weight for the previous density call price.
        ixtte_cap   = np.minimum( ixtte, T-1 )
        ixtte_m1    = np.maximum( ixtte-1, 0 )
        prev_times  = np.where( ixtte == 0, 0., times[ixtte_m1] )
        divisor     = np.where( ixtte == T, 1., times[ixtte_cap] - prev_times )
        assert np.min( divisor) > 0., f"Internal error: divisor for time interpolation weights must be positive, found min {np.min(divisor)}. Please contact authors."
        dweight     = np.select( [ixtte==0, ixtte==T], # dweight is always linear
                                 [0.,       0.],
                                 ( times[ixtte_cap] - ttes ) / divisor )
        
        if self.vol_interp == "lin_sqrt_time":
            prev_sqrtT  = np.where( ixtte == 0, 0., sqrtT[ixtte_m1] )
            divisor     = np.where( ixtte == T, 1., sqrtT[ixtte_cap] - prev_sqrtT )
            vweight     = np.where( ixtte == T, 0., ( sqrtT[ixtte_cap] - sqrtTtes ) / divisor )
            vextra      = np.where( ixtte < T, 0., sqrtTtes - prev_sqrtT )
            ixtte       = np.clip( ixtte, max=T-1 )
        else:
            vweight     = np.where( ixtte == T, 0., ( times[ixtte_cap] - ttes ) / divisor ) 
            vextra      = np.where( ixtte < T, 0., ttes - prev_times )
            ixtte       = np.clip( ixtte, max=T-1 )

        opt_lix     = np.zeros((M,), dtype=np.int32 )
        opt_dweight = np.zeros((M,), dtype=dtype )
        opt_vweight = np.zeros((M,), dtype=dtype )
        opt_vextra  = np.zeros((M,), dtype=dtype )

        for ixtte, tte, dw, vw, vx in zip(ixtte, ttes, dweight, vweight, vextra):
            where               = opt_tte == tte
            opt_lix[where]      = ixtte
            opt_dweight[where]  = dw
            opt_vweight[where]  = vw
            opt_vextra[where]   = vx

        opt_lix     = opt_lix.reshape(shape) 
        opt_dweight = opt_dweight.reshape(shape)
        opt_vweight = opt_vweight.reshape(shape)
        opt_vextra  = opt_vextra.reshape(shape)
        return dict(lix=opt_lix, dweight=opt_dweight, vweight=opt_vweight, vextra=opt_vextra)

    # -------------------------------------
    # Numerical utilities
    # -------------------------------------

    def transition_operators(self, *, eps: float = 1E-8 ) -> dict[str, np.ndarray]:
        r"""
        Computes the DLV transition operators for the grid geometry, as described in https://arxiv.org/pdf/2601.11209, Remark 3.5 and Section 3.4.
        These do not depend on market data.

        Returns
        -------
        Transition operators: dict[str, np.ndarray]
            A dictionary containing:
            
            * ``wn : np.ndarray``: the $w^+_t$ transition weights, of shape ``(T,N-2)``.
            * ``wp : np.ndarray``: the $w^-_t$ transition weights, of shape ``(T,N-2)``.
            * ``L : np.ndarray``: the interpolation matrices of shape ``(T,N,N)``.
        """
        dtype    =  self.xstrikes.dtype
        xstrikes = np.asarray(self.xstrikes, dtype=np.float64) # we compute the transition operators in higher precision to avoid numerical issues, then cast back to float32 at the end
        T, N     = xstrikes.shape[0]-1, xstrikes.shape[1]-2
        # Omega in https://arxiv.org/pdf/2601.11209 in Remark 3.5
        strikes  = xstrikes[1:,1:-1]    
        wx       = (strikes[:,1:-1]**2) / (strikes[:,2:] - strikes[:,:-2])
        wn       = wx / (strikes[:,1:-1] - strikes[:,:-2])
        wp       = wx / (strikes[:,2:] - strikes[:,1:-1])
        wx       = None
        assert wn.shape == (T,N-2) and wp.shape == (T,N-2), f"wn and wp must have shape {(T,N-2)}, got {wn.shape} and {wp.shape}"
        assert np.min(wn) >= 0. and np.min(wp) >= 0., f"Transition weights must be non-negative, found min {np.min(wn)} and {np.min(wp)}"

        # interpolation matrx IL
        # https://arxiv.org/pdf/2601.11209 (22)
        # we compute this in higher precision
        xstrikes_t  = xstrikes[1:,:]
        strikes_tm1 = xstrikes[:-1,1:-1]

        divup = np.asarray( xstrikes_t[:,2:,None] - xstrikes_t[:,1:-1,None] ,dtype=np.float64 )
        divdn = np.asarray( xstrikes_t[:,1:-1,None] - xstrikes_t[:, :-2,None] ,dtype=np.float64 )
        assert np.all( divup > 0. ), f"Internal error: divup must be positive, found min {np.min(divup)}"
        assert np.all( divdn > 0. ), f"Internal error: divdn must be positive, found min {np.min(divdn)}"

        LUp = ( np.maximum( 0., strikes_tm1[:,None,:] - xstrikes_t[:,2:,None] ) - 
                np.maximum( 0., strikes_tm1[:,None,:] - xstrikes_t[:,1:-1,None] ) ) /\
            ( xstrikes_t[:,2:,None] - xstrikes_t[:,1:-1,None] )
        LDn = ( np.maximum( 0., strikes_tm1[:,None,:] - xstrikes_t[:,1:-1,None] ) - 
                np.maximum( 0., strikes_tm1[:,None,:] - xstrikes_t[:,:-2,None] ) ) /\
            ( xstrikes_t[:,1:-1,None] - xstrikes_t[:,:-2,None] )
        L = LUp - LDn
        assert L.shape == (T,N,N), f"Internal error: L must have shape {(T,N,N)}, got {L.shape}"       
        assert np.min(L) >= -eps and np.max(L)-1. <= eps, f"Transition operators must be between 0 and 1; found min {np.min(L)} and max {np.max(L)} [tolerance {eps:.4g}]"
        
        wp = np.asarray(wp, dtype=dtype)
        wn = np.asarray(wn, dtype=dtype)
        L  = np.asarray(L, dtype=dtype) 
        return dict(wn=wn, wp=wp, L=L)
    

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

class GridSanos(object):

    def __init__(self, geometry : GridGeometry,
                       bb       : Backbone = BackboneNumpy(),
                       eps      : Eps = Eps(),
                       debug    : bool = False ):

        self.geometry  = geometry
        self.bb        = bb
        self.eps       = eps
        self.debug     = debug

        self.xstrikes  = bb.to_array( self.geometry.xstrikes )  # this is a noop for numpy
        self.xexpiries = bb.to_array( self.geometry.xexpiries ) # this is a noop for numpy
        self.density0  = bb.to_array( self.geometry.density0 )  # this is a noop for numpy

        assert self.xstrikes.shape == (self.T+1,self.N+2), f"Expected xstrikes to have shape {(self.T+1,self.N+2)}, got {self.xstrikes.shape}"
        assert self.xexpiries.shape == (self.T+1,), f"Expected xexpiries to have shape {(self.T+1,)}, got {self.xexpiries.shape}"
        assert self.density0.shape == (self.N,), f"Expected density0 to have shape {(self.N,)}, got {self.density0.shape}"

        self.transp    = self.geometry.transition_operators() # this is a numpy array, but we will convert it to the backbone array when needed in computations
        self.transp    = { k: bb.to_array(v) for k,v in self.transp.items() }

    def verify( self, condition : callable, message : callable | str ):
        """ Verifies a condition and raises an error with the given message if the condition is not met. Only runs if debug mode is enabled. The condition and message can be callables to avoid unnecessary computations when debug mode is disabled. """
        if self.debug:
            verify( condition(), message )

    @property
    def TN(self) -> tuple[int, int]:
        """ Returns the number of expiries and strikes, i.e. use ``T, N = geometry.TN`` to unpack. """
        return self.geometry.TN

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.geometry.T
    
    @property
    def N(self) -> int:
        """ Returns the common number of strikes accross expiries, if that exists, or None otherwise. """
        return self.geometry.N
    
    @property
    def strikes(self) -> np.ndarray:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.xstrikes[1:,1:-1]

    @property
    def expiries(self) -> np.ndarray:
        """ Returns the model expiries excluding the zero expiry """
        return self.xexpiries[1:]
    
    def interpolate_sqrtVars( self, 
                              fwd_vols   : Backbone.Array|float|None,
                              opt_interp : dict[str, Backbone.Array], 
                              opt_mask   : Backbone.Array|None = None ) -> Backbone.Array|None:
        """
        Utility function to compute the square-root variance for SANOS based in fwd_vols.
        Use :meth:`time_interpolation_weights` to compute ``opt_interp``.

        This function is batchable.

        Parameters
        ----------
        fwd_vols : Backbone.Array | float | None
            An array of forward volatilities of ``(..., 1|T)`` or a broadcastable shape. A float or similar is broadcasted accordingly. The function returns ``None`` if ``fwd_vols`` is ``None``.
        opt_interp : dict[str, Backbone.Array]
            The option interpolation weights as returned by :meth:`time_interpolation_weights`, with keys 'lix', 'dweight', 'vweight', 'vextra'.
            Each entry has dimension ``(..., M)`` where M is the number of options.
        bb : Backbone, default ``BackboneNumpy()``
            The backbone to use for computations. Default is ``BackboneNumpy()``.
        debug : bool, default ``False``
            Whether to perform additional checks for debugging purposes. Default is False.

        Returns
        -------
        opt_sqrtVars : Backbone.Array or None
            The interpolated square-root variances for the options, of shape matching ``opt_interp['lix']``. If fwd_vols is None, returns None. 
        """
        if fwd_vols is None:
            return None

        T            = self.T
        opt_lix     = opt_interp['lix']
        opt_vweight = opt_interp['vweight']
        opt_vextra  = opt_interp['vextra']
        M           = opt_lix.shape[-1]
        bb          = self.bb
        head        = opt_lix.shape[:-1]
        hext        = (1,)*len(head)

        if isinstance(fwd_vols, (float, int, np.number)):
            heax     = (1,)*(len(opt_lix.shape)-1)
            fwd_vols = bb.full(heax+(T,), float(fwd_vols), dtype=bb.float64)
        elif len(fwd_vols.shape) == 0:
            fwd_vols = bb.reshape( fwd_vols, (1,)*len(opt_lix.shape) )
        else:           
            verify_inp( len(fwd_vols.shape) <= len(opt_lix.shape), f"fwd_vols must have at most as many dimensions as opt_lix, found {len(fwd_vols.shape)} and {len(opt_lix.shape)}" )
            verify_inp( fwd_vols.shape[-1] in [1,T], lambda : f"'fwd_vols' must have shape ending in 1 or {T}, found {fwd_vols.shape}")
            if len(fwd_vols.shape) < len(opt_lix.shape):
                fwd_vols = bb.reshape( fwd_vols, (1,)*(len(opt_lix.shape)-len(fwd_vols.shape)) + fwd_vols.shape )

        assert len(opt_lix.shape) == len(fwd_vols.shape), f"opt_lix and fwd_vols must have the same number of dimensions, found {len(opt_lix.shape)} and {len(fwd_vols.shape)}"
        assert opt_vweight.shape == opt_lix.shape, f"opt_vweight and fwd_vols must have the same shapes, found {opt_vweight.shape} and {fwd_vols.shape}"
        assert opt_vextra.shape == opt_lix.shape, f"opt_vextra and fwd_vols must have the same shapes, found {opt_vextra.shape} and {fwd_vols.shape}"

        assert self.xexpiries.shape == (self.T+1,), f"Expected xexpiries to have shape {(self.T+1,)}, got {self.xexpiries.shape}"
        assert self.xstrikes.shape == (self.T+1,self.N+2), f"Expected xstrikes to have shape {(self.T+1,self.N+2)}, got {self.xstrikes.shape}"

        target_shape = head + (T,)
        assert bb.are_broadcastable(fwd_vols.shape, target_shape), f"Internal error: fwd_vols must broadcast to {target_shape}, found {fwd_vols.shape}"
        if fwd_vols.shape != target_shape:
            fwd_vols = bb.broadcast_to(fwd_vols, target_shape)

        dts = bb.diff(self.xexpiries)

        if self.geometry.vol_interp == "lin_var":
            r"""
            "sqrtVar" is computed by linear interpolation in variance, and taking the square rtoot: assuming that $t_j<t\leq t_{j+1}$ then:
            $$
                \mathrm{sqrt_var}(t) := \sqrt{ \mathrm{var}(t) } \ : \ \mathrm{var}(t) =  \sigma^2_j \left( t - t_j \right ) + \sum_{t_j<t} \sigma^2_j \left( t_j - t_{j-1} \right) +
            $$
            """
            assert self.geometry.vol_interp == "lin_var", f"Internal error: vol_interp must be 'lin_sqrt_time' or 'lin_var', found '{self.geometry.vol_interp}'"
            dts = bb.reshape(dts, hext + dts.shape)
            totalVar = bb.cumsum((fwd_vols + self.geometry.fwdvol_min) ** 2 * dts, axis=-1)
            assert totalVar.shape[-1] == T, f"Internal error: totalVar must have shape ending in {T}, found {totalVar.shape}"
            self.verify(lambda: bb.all(np.isfinite(totalVar)) & np.all(totalVar >= 0.), "Negative or infinite totalVar")

            prev_var = bb.where(opt_lix == 0, 0., bb.take_along_axis(totalVar, bb.clip(opt_lix-1, min=0), axis=-1))
            opt_sqrtVars = bb.sqrt(opt_vweight * prev_var + (1. - opt_vweight) * bb.take_along_axis(totalVar, opt_lix, axis=-1) + opt_vextra * ((fwd_vols[..., -1:] + self.geometry.fwdvol_min) ** 2))

        else:
            r"""
            Linear in square-root time: assumig that $t_j<t\leq t_{j+1}$ then:
            $$
                \mathrm{sqrt_var}(t) := \sqrt{ \mathrm{var}(t) } \ : \ \mathrm{var}(t) =  \sigma^2_j \left( t - t_j \right ) + \sum_{t_j<t} \sigma^2_j \left( t_j - t_{j-1} \right) 
            $$
            """
            verify_inp( self.geometry.vol_interp == "lin_sqrt_time", f"Internal error: vol_interp must be 'lin_sqrt_time' or 'lin_var', found '{self.geometry.vol_interp}'" )
            dsqrtTs = bb.diff(bb.sqrt(self.xexpiries))
            dsqrtTs = bb.reshape(dsqrtTs, hext + dsqrtTs.shape)
            totalSqrtVar = bb.cumsum((fwd_vols + self.geometry.fwdvol_min) * dsqrtTs, axis=-1)
            assert totalSqrtVar.shape[-1] == T, f"Internal error: totalSqrtVar must have shape ending in {T}, found {totalSqrtVar.shape}"
            self.verify(lambda: bb.all(np.isfinite(totalSqrtVar)) & np.all(totalSqrtVar >= 0.), "Negative or infinite sqrtVars")

            prev_sqrtVar = bb.where(opt_lix == 0, 0., bb.take_along_axis(totalSqrtVar, bb.clip(opt_lix-1, min=0), axis=-1))
            opt_sqrtVars = opt_vweight * prev_sqrtVar + (1. - opt_vweight) * bb.take_along_axis(totalSqrtVar, opt_lix, axis=-1) + opt_vextra * (fwd_vols[..., -1:] + self.geometry.fwdvol_min)

        assert opt_sqrtVars.shape[-1] == M, f"Internal error: opt_sqrtVars must have shape ending in {M}, found {opt_sqrtVars.shape}"
        self.verify( lambda: bb.all( np.isfinite(opt_sqrtVars) ) & np.all( opt_sqrtVars >= 0. ), "Negative or infinite opt_sqrtVars")
        assert len(opt_sqrtVars.shape) == len(opt_lix.shape), f"opt_sqrtVars and opt_lix must have the same number of dimensions, found {len(opt_sqrtVars.shape)} and {len(opt_lix.shape)}"
        return opt_sqrtVars if opt_mask is None else bb.where( opt_mask, opt_sqrtVars, 0. )

    def price(
            self, 
            # market data
            fwd_vols    : Backbone.Array | None,
            densities   : Backbone.Array, 
            #
            opt_strikes : Backbone.Array,
            opt_tte     : Backbone.Array,
            opt_mask    : Backbone.Array | None = None,
            opt_is_call : Backbone.Array | None = None,
            opt_interp  : dict[str|Backbone.Array]|None = None,
        ) -> Backbone.Array:
        """
        SANOS batch option pricing.

        Parameters
        ----------
        fwd_vols : Backbone.Array | None
            The forward volatilities as an array of shape ``(..., 1|T)`` or in a broadcastable shape. Floats are broadcasted accordingly.
            Use ``None`` for non-smooth linear interpolation.
        densities : Backbone.Array
            The model densities for the grid expiries and strikes, of shape ``(..., T, N)``, where ``T`` is the number of non-zero expiries and ``N`` is the number of strikes.
            Densities can be computed from discrete local vol squared using :meth:`densities`.
        opt_strikes : Backbone.Array
            The option strikes, of shape ``(..., M)``, where M is the number of options to price.
        opt_tte : Backbone.Array
            The option times to expiry, of shape ``(..., M)``, where M is the number of options to price.
        opt_mask : Backbone.Array | None, default ``None``
            A boolean array of shape ``(..., M)`` indicating which options are active (True) and which are not (False). If None, all options are active.
        opt_is_call : Backbone.Array | None, default ``None``
            A boolean array of shape ``(..., M)`` indicating which options are calls (True) and which are puts (False). If None, all options are treated as calls.
        opt_interp : dict[str|Backbone.Array]|None, default ``None``
            The option interpolation weights as returned by :meth:`time_interpolation_weights`, each of which has shape  ``(..., M)``.
            If ``None``, these are computed on the fly. However, this is a numpy operation and will be *very costly* torch or jax usage.
            It is strongly advised to calculate these ahead of time.

        Returns
        -------
        Prices : ``Backbone.Array``
            Option prices.
        """
        N        = self.N
        bb       = self.bb
        M        = opt_strikes.shape[-1]
        head     = opt_strikes.shape[:-1]
        hext     = (1,)*len(head)     

        if opt_interp is None:
            if not isinstance(bb, BackboneNumpy):
                warnings.warn(f"{self.__class__.__name__} function 'price' using backbone {bb.__class__.__name__}: no 'opt_interp' was provided. This will be computed on the fly using numpy which will seriously impact performance. Consider computing this ahead of time.")
            opt_interp  = self.geometry.time_interpolation_weights( opt_tte=opt_tte, opt_mask=opt_mask, dtype=self.xexpiries.dtype )
            opt_interp  = { k: bb.to_array(v) for k,v in opt_interp.items() }
        opt_sqrtVars = self.interpolate_sqrtVars( fwd_vols, opt_interp=opt_interp, opt_mask=opt_mask )
        assert opt_sqrtVars is None or opt_sqrtVars.shape[-1] == M, f"Internal error: opt_sqrtVars must have shape ending in {M}, found {opt_sqrtVars.shape}"

        mstrikes     = self.strikes # is bb array
        assert densities.shape[-2:] == mstrikes.shape[-2:], f"Expected densities to have the same last two dimensions as geometry.strikes, found {densities.shape} and {mstrikes.shape}"
        mstrikes     = bb.reshape( mstrikes, hext + mstrikes.shape ) # add leading dims for broadcasting with opt_strikes

        if len(densities.shape) != len(mstrikes.shape):
            densities   = bb.reshape( densities, hext + densities.shape ) # add leading dims for broadcasting with opt_strikes

        def call_price( spots, strikes, sqrtVars, mask ):
            if sqrtVars is None:
                f = bb.relu( spots - strikes ) 
                return f if mask is None else f*mask
            else:
                return bb.bs_price( spot=spots,
                                    strikes=strikes,
                                    sqrtVars=sqrtVars,
                                    mask=mask, 
                                    high_precision=True, debug_verify_bounds=self.debug )

        opt_lix      = opt_interp['lix']
        opt_dweight  = opt_interp['dweight']
        eff_lix      = bb.clip(opt_lix-1, min=0)
        spots        = bb.take_along_axis( mstrikes, eff_lix[...,None], axis=-2 )#
        assert spots.shape == head+(M,N), f"Expected spots to have shape {head+(M,N)}, got {spots.shape}"


        calls_prev   = call_price(  spots=spots,
                                    strikes=opt_strikes[...,:,None], 
                                    sqrtVars=opt_sqrtVars[...,:,None] if not opt_sqrtVars is None else None,
                                    mask=opt_mask[...,:,None] if not opt_mask is None else None
                                ) * opt_dweight[...,:,None]

        assert calls_prev.shape == head+(M,N), f"Expected calls_prev shape {head+(M,N)}, got {calls_prev.shape}"
        dens       = bb.take_along_axis( densities, eff_lix[...,None], axis=-2 )
        assert dens.shape == head+(M,N), f"Expected dens to have shape {head+(M,N)}, got {dens.shape}"
        calls_prev = bb.sum( calls_prev * dens, axis=-1 ) # todo: make more efficient
        assert calls_prev.shape == head+(M,), f"Expected calls_prev to have shape {head+(M,)}, got {calls_prev.shape}"
        eff_lix = None
        spots   = None

        spots        = bb.take_along_axis( mstrikes, opt_lix[...,None], axis=-2 )
        assert spots.shape == head+(M,N), f"Expected spots to have shape {head+(M,N)}, got {spots.shape}"
        calls_next   = call_price(  spots=spots,
                                    strikes=opt_strikes[...,:,None], 
                                    sqrtVars=opt_sqrtVars[...,:,None] if not opt_sqrtVars is None else None ,
                                    mask=opt_mask[...,:,None] if not opt_mask is None else None
                                ) * ( 1.- opt_dweight[...,:,None] )
        assert calls_next.shape == head+(M,N), f"Expected calls_next shape {head+(M,N)}, got {calls_next.shape}"
        dens          = bb.take_along_axis( densities, opt_lix[...,None], axis=-2 )
        assert dens.shape == head+(M,N), f"Expected dens to have shape {head+(M,N)}, got {dens.shape}"
        calls_next    = bb.sum( calls_next * dens, axis=-1 ) # todo: make more efficient
        assert calls_next.shape == head+(M,), f"Expected calls_next to have shape {head+(M,)}, got {calls_next.shape}"

        calls      = calls_prev + calls_next 
        self.verify( lambda : bb.all( np.isfinite(calls) ), lambda : "fit_calls not finite")
        prices     = calls if opt_is_call is None else bb.where( opt_is_call | ~opt_mask, calls, calls + opt_strikes - 1. )
        prices     = prices * opt_mask if opt_mask is not None else prices
        assert prices.shape == head+(M,), f"Expected prices to have shape {head+(M,)}, got {prices.shape}"
        return prices 

    def densities( self, dlvsqs : Backbone.Array ) -> Backbone.Array:
        """
        Compute densities from DLV squared.
        This function is batch compatible.

        Parameters
        ----------
        dlvsqs : array of shape ``(...,T,N-2)``.
            The DLVs squared, i.e. the local variance divided by the time step, at each model time and strike.
            This function will add `geometry.dlv_min**2` to the input DLVs squared to ensure numerical stability.
        
        Returns
        -------
        Densites : Backbone.Array
            Array of shape ``(...,T,N)``, representing the densities at each model time, not including the initial density at time 0 (which can be accessed via ``self.density0``). 
        """
        T,N         = self.TN
        assert dlvsqs.shape[-2:] == (T,N-2), f"Expected dlvsqs to have shape ending in {(T,N-2)}, found {dlvsqs.shape}"

        bb          = self.bb
        head        = dlvsqs.shape[:-2]
        hext        = (1,)*len(head)     
        _text       = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert len(head) <= len(_text), f"Too many batch dimensions: expected at most {len(_text)}, got {len(head)}"
        einhext     = _text[:len(head)] 

        self.verify( lambda : bb.all((self.density0 >= -self.eps.densities) & (self.density0 - 1. <= self.eps.densities)), lambda : f"Density 0 must be between 0 and 1; found min {bb.min(self.density0)} and max {bb.max(self.density0)} [tolerance 'densities' {self.eps.densities}]" )
        self.verify( lambda : bb.all(bb.abs( bb.sum( self.density0, axis=-1 ) - 1. ) < self.eps.densities), lambda : f"Density 0 must have positive mass, found sum {bb.max(bb.abs( bb.sum( self.density0, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

        dts          = bb.diff(self.xexpiries) # in bb array
        mstrikes     = self.strikes # in bb array
        wn           = bb.reshape( self.transp['wn'], hext+self.transp['wn'].shape )
        wp           = bb.reshape( self.transp['wp'], hext+self.transp['wp'].shape )
        L            = bb.reshape( self.transp['L'], hext+self.transp['L'].shape )
        assert wn.shape[-2:] == (T,N-2), f"Expected wn to have shape ending in {(T,N-2)}, found {wn.shape}"
        assert wp.shape[-2:] == (T,N-2), f"Expected wp to have shape ending in {(T,N-2)}, found {wp.shape}"
        assert L.shape[-3:] == (T,N,N), f"Expected L to have shape ending in {(T,N,N)}, found {L.shape}"
    
        densities = []
        min_dlvsq = self.geometry.dlv_min**2

        density0 = bb.reshape(self.density0, hext + (N,))
        density0 = density0 + bb.zeros(head + (N,), dtype=dlvsqs.dtype)

        for t in range(T):
            # interpolate up
            mdensity_tm1 = densities[-1] if t > 0 else density0 
            assert mdensity_tm1.shape[-1] == N, f"Expected mdensity_tm1 to have shape ending in {N}, found {mdensity_tm1.shape}"
            L_t          = L[...,t,:,:]
            mdensity_t   = bb.einsum( f"{einhext}mn, {einhext}n -> {einhext}m", L_t, mdensity_tm1 )  # interpolate density to time t 
            assert mdensity_t.shape[-1] == N, f"Expected mdensity_t after L to have shape ending in {N}, found {mdensity_t.shape}"

            # solve
            dlvsqs_t     = dlvsqs[...,t,:]
            dlvar_t      = (dlvsqs_t+min_dlvsq) * dts[t]  # local variance at time t, shape (...,N-2)
            assert dlvar_t.shape[-1] == N-2, f"Expected dlvar_t shape to end in {N-2}, found {dlvar_t.shape}"

            w_minus      = dlvar_t * wn[...,t,:]
            w_plus       = dlvar_t * wp[...,t,:]
            mdensity_t    = solve_sanos_tridiagonal(bb, w_minus=w_minus, w_plus=w_plus, rhs=mdensity_t ) # keep
            assert mdensity_t.shape[-1] == N, f"Expected mdensity_t to have shape ending in {N}, found {mdensity_t.shape}"
            """
            self.verify( lambda : bb.all((mdensity_t >= -self.eps.densities) & (mdensity_t - 1. <= self.eps.densities)), lambda : f"Interpolated density at {t} must be between 0 and 1; found min {bb.min(mdensity_t)} and max {bb.max(mdensity_t)} [tolerance 'densities' {self.eps.densities }]" )
            self.verify( lambda : bb.all(bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ) <= self.eps.densities), lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            self.verify( lambda : bb.all(bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ) <= self.eps.densities), lambda : f"Interpolated density at {t} must have positive mass, found sum {bb.max(bb.abs( bb.sum( mdensity_t*mstrikes_t, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )
            """
            densities.append( mdensity_t )

        densities = bb.stack( densities, axis=-2 )
        assert densities.shape[-2:] == (T,N), f"Expected densities to have shape ending in {(T,N)}, found {densities.shape[-2:]}"
        return densities
