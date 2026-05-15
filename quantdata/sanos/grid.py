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
from quantdata.sanos.utils import GeometryBase, _time_interpolation_weights, _interpolate_sqrtVars, Eps
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
class GridGeometry(GeometryBase):
    xexpiries  : np.ndarray            #: model expiries in years, starting from 0, of size (T+1,) where T is the number of non-zero expiries. The first ``xexpiry`` must be zero.
    xstrikes   : np.ndarray            #: internal model strikes
    dlv_min    : float = 0.01          #: minimum discrete local volatility (dlv) value.
    dlv_max    : float = 5.0           #: maximum discrete local volatility (dlv) value.
    fwdvol_min : float = 0.0           #: minimum backbone forward volatility value.
    vol_interp : str = "lin_var"       #: Term volatility interpolation method. Must be a 'lin_var' or 'lin_sqrt_time'
    unique_id  : str | None = None     #: optional unique identifier for the geometry, used for caching. If None, a unique identifier will be generated based on the geometry parameters.

    def __post_init__(self):
        verify_inp( self.dlv_min >= 0., lambda : f"'dlv_min' must be non-negative, but found {self.dlv_min:.4g}") 
        verify_inp( self.dlv_max > self.dlv_min, lambda : f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min >= 0., lambda : f"'fwdvol_min' must be non-negative; found {self.fwdvol_min:.4g}" )
        verify_inp( self.vol_interp in {"lin_sqrt_time", "lin_var"}, lambda : f"'vol_interp' must be either 'lin_sqrt_time' or 'lin_var', but found {self.vol_interp}" )

        verify_inp( len(self.xexpiries.shape) == 1 and len(self.xexpiries) > 1, lambda : "'xexpiries' must be non-empty 1D array of at least two expiries, starting in 0" )
        verify_inp( self.xexpiries[0] == 0., lambda : f"First expiry must be zero; found {self.xexpiries[0]}" )
        verify_inp( np.min( np.diff(self.xexpiries) ) > 0., lambda : "'xexpiries' must be strictly increasing" )

        verify_inp( len(self.xstrikes.shape) == 2 and self.xstrikes.shape[0] == len(self.xexpiries), "'strikes' must be a list of 1D arrays or a 2D array with the same number of rows as 'expiries'")
        verify_inp( self.xstrikes.shape[1] >= 5, lambda: f"'strikes' must have at least 5 elements in its second dimension, found shape {self.xstrikes.shape}" )
        verify_inp( np.all( np.diff(self.xstrikes,axis=1) > 0), lambda: f"'strikes' must be strictly increasing in its second dimension, found {np.min(np.diff(self.xstrikes,axis=1)):.4g}")
        verify_inp( np.all( self.xstrikes[:,0] > 0. ), lambda: f"'strikes' must be positive, found {self.xstrikes[:,0]}")
        verify_inp( np.all( self.xstrikes[:,1] < 1.) and np.all( self.xstrikes[:,-2] > 1.), lambda: f"'strikes' in its second dimension must include positive values below and above 1, found boundary values {self.xstrikes[:,:2]} and {self.xstrikes[:,-2:]}" )
        verify_inp( np.all( np.diff( self.xstrikes[:,0] ) <= 0.), lambda: f"The lowest 'strikes' must not increase in time, {self.xstrikes[:,0]:.4g}, {np.diff( self.xstrikes[:,0] ):.4g}")
        verify_inp( np.all( np.diff( self.xstrikes[:,-1] ) >= 0.), lambda: f"The highest 'strikes' must not decrease in time, {self.xstrikes[:,-1]:.4g}, {np.diff( self.xstrikes[:,-1] ):.4g}")

        # Reuse the base-class helper so density0 is defined consistently with other geometries.
        GeometryBase._make_density0(self)

    @property
    def has_grid(self) -> bool:
        """ Returns True if the geometry has a grid, i.e. if the strikes are given as a 2D array with the same number of rows as expiries. Returns False if the strikes are given as a list of 1D arrays. """
        return True

    @property
    def N(self) -> int:
        """ Returns the common number of strikes accross expiries, if that exists, or None otherwise. """
        return self.xstrikes.shape[1]-2

    @property
    def TN(self) -> tuple[int, int]:
        """ Returns the number of expiries and strikes, i.e. use ``T, N = geometry.TN`` to unpack. """
        return self.T, self.N

    @property
    def size(self) -> int:
        return self.T*self.N

    @property
    def strikes(self) -> np.ndarray:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.xstrikes[1:,1:-1]

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density """
        return self._density0   

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
        return _time_interpolation_weights(  vol_interp=self.vol_interp, expiries=self.expiries, opt_tte=opt_tte, opt_mask=opt_mask, dtype=dtype )

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
    
@dataclass(frozen=True)
class GridGeometryGenerator:
    r"""
    Standardized generator for grid generation.

    * Expiries will be generated with exponential growth from :attr:`initial_t` to :attr:`horizon_t`, with a total of :attr:`num_times`
    * Strikes will be generated from normalized strikes. These can be specified as a float for a symmetric range, or tuple for a asymmetric range.#
      The number of normalized strikes is given by :attr:`num_nms`. The normalized stikes are converted to strikes per expiry using :meth:`quantdata.NMS.from_nms`, essentially:
      $$
         x_t = \exp\left( \log(1.1) z \frac{\sqrt{\epsilon+t}}{\sqrt{\epsilon+1}} \right)
      $$ 
    * Once the strikes are generted with above, external strikes will be added up to ``lower_scale`` of the lowest NMS strike, and up to ``upper_scale`` of the highest NMS strike.
    """
    initial_t   : float = 1./DAYS_PER_YEAR             #: Initial model expiry, in years. Default is 1 day.
    horizon_t   : float = 2.                           #: Terminal model expiry, in years. Default is 2 years.
    num_times   : int = 11                             #: Number of time points.
    nms_range   : float | tuple[float,float] = 2.      #: Either a symmetric range, or a tuple indicating a lower and upper NMS. The lower NMS must be negative while the upper NMS must be positive. Default is 2, which corresponds to a symmetric range of (-2,2).
    num_nms     : int = 11                             #: Number of strikes per expiry.
    dlv_min     : float = 0.01                         #: Minimum discrete local volatility (dlv) value. Default is 0.01.
    dlv_max     : float = 5.                           #: Maximum discrete local volatility (dlv) value. Default is 5.0.
    fwdvol_min  : float = 0.01                         #: Minimum forward volatility value. Default is 0.01.
    vol_interp  : str = "lin_var"                      #: Term volatility interpolation method. Must be 'lin_var' or 'lin_sqrt_time'. Default is 'lin_var'.
    lower_scale : float = 0.5                          #: Scaling of the lowest strike compared to the lowest NMS strike. Default is 0.5, which means the lowest strike will be half of the lowest NMS strike.
    upper_scale : float = 1.5                          #: Scaling of the highest strike compared to the highest NMS strike. Default is 1.5, which means the highest strike will be 1.5 times the highest NMS strike.

    def __post_init__(self):
        verify_inp( self.initial_t > 0., f"initial_t must be positive, found {self.initial_t:.4g}" )
        verify_inp( self.horizon_t > self.initial_t, f"horizon_t must be greater than initial_t, found {self.horizon_t:.4g} <= {self.initial_t:.4g}" )
        verify_inp( self.num_times >= 2, f"num_times must be at least 2, found {self.num_times}" )
        verify_inp( isinstance(self.nms_range, (float, tuple)), f"nms_range must be a float or a tuple of two floats, found {type(self.nms_range)}" )
        if isinstance(self.nms_range, tuple):    
            verify_inp( len(self.nms_range) == 2, f"nms_range tuple must have length 2, found {len(self.nms_range)}" )
            verify_inp( self.nms_range[0] < 0. and self.nms_range[1] > 0., f"nms_range must include negative and positive values, found {self.nms_range}" )
        else:
            verify_inp( self.nms_range > 0., f"nms_range must be positive, found {self.nms_range:.4g}" )
        verify_inp( self.num_nms >= 3, f"num_nms must be at least 3, found {self.num_nms}" )

        verify_inp( self.upper_scale > 1., f"upper_scale must be greater than 1, found {self.upper_scale:.4g}" )
        verify_inp( self.dlv_min >= 0., f"dlv_min must be non-negative, found {self.dlv_min:.4g}" )
        verify_inp( self.dlv_max > self.dlv_min, f"dlv_max must be greater than dlv_min, found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min >= 0., f"fwdvol_min must be non-negative, found {self.fwdvol_min:.4g}" )  
        verify_inp( 1. > self.lower_scale > 0., f"lower_scale must be between 0 and 1, found {self.lower_scale:.4g}" ) 
        verify_inp( self.vol_interp in {"lin_sqrt_time", "lin_var"}, f"vol_interp must be either 'lin_sqrt_time' or 'lin_var', found {self.vol_interp}" )   

    def create(self, *, return_inner : bool = False, dtype = np.float64 ) -> GridGeometry|tuple[GridGeometry, np.ndarray]:
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
        nms_lo           = float(-self.nms_range if isinstance(self.nms_range, float) else self.nms_range[0])
        nms_hi           = float(+self.nms_range if isinstance(self.nms_range, float) else self.nms_range[1])

        verify_inp( nms_lo < 0. and nms_hi > 0., f"'nms_lo' must be negative and 'nms_hi' must be positive, found nms_lo {nms_lo:.4g} and nms_hi {nms_hi:.4g}" )    



        xexpiries        = np.zeros( (self.num_times+1,), dtype=dtype )  
        xexpiries[1:]    = np.exp( np.linspace(math.log(self.initial_t), math.log(self.horizon_t), self.num_times) )
        num_xnms         = self.num_nms//6
        num_inms         = self.num_nms - 3*num_xnms

        nms              = np.linspace( nms_lo, nms_hi, num_inms )
        inner_strikes    = NMS.from_nms( nms[None,:], t=xexpiries[:,None] )

        xstrikes         = np.zeros((self.num_times+1,self.num_nms+2), dtype=dtype )
        xstrikes[:,:num_xnms*2+1] = np.linspace(self.lower_scale, 1., 2*num_xnms+2)[None,:-1] * inner_strikes[:,0][:,None] 
        xstrikes[:,-num_xnms-1:] = np.linspace(1., self.upper_scale, num_xnms+2)[None,1:] * inner_strikes[:,-1][:,None] 
        xstrikes[:,2*num_xnms+1:-num_xnms-1] = inner_strikes
        xstrikes[:,-1] *= math.sqrt( self.upper_scale )

        strike_diff = np.diff( xstrikes, axis=1 )
        assert np.min( strike_diff ) > 0., f"Per time step: strikes must increase. Found {np.min(strike_diff):.4g}"
        geometry = GridGeometry(
                xexpiries  = xexpiries,
                xstrikes   = xstrikes,
                dlv_min    = self.dlv_min,
                dlv_max    = self.dlv_max,
                fwdvol_min = self.fwdvol_min ,
                vol_interp = self.vol_interp,
            )
        if return_inner:
            return geometry, inner_strikes
        return geometry

# -------------------------------------------
# Main SANOS class
# -------------------------------------------

class GridSanos(object):

    def __init__(self, geometry : GridGeometry,
                       bb       : Backbone = BackboneNumpy(),
                       eps      : Eps = Eps(),
                       debug    : bool = False ):
        """ Run it """
        verify_inp( isinstance( geometry, GridGeometry ), lambda: f"geometry must be an instance of GridGeometry, found {type(geometry)}" )
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
    def strikes(self) -> Backbone.Array:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.xstrikes[1:,1:-1]

    @property
    def expiries(self) -> Backbone.Array:
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
        return _interpolate_sqrtVars( 
                            bb         = self.bb,
                            xexpiries  = self.xexpiries,
                            vol_interp = self.geometry.vol_interp,
                            fwdvol_min = self.geometry.fwdvol_min,
                            fwd_vols   = fwd_vols,
                            opt_interp = opt_interp, 
                            opt_mask   = opt_mask,
                            debug      = self.debug )

    def price(
            self, 
            # market data
            fwd_vols    : Backbone.Array | None,
            densities   : Backbone.Array, *,
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
            Densities can be computed from discrete local vols squared using :meth:`densities`.
        opt_strikes : Backbone.Array
            The option strikes, of shape ``(..., M)``, where M is the number of options to price.
        opt_tte : Backbone.Array
            The option times to expiry, of shape ``(..., M)``, where M is the number of options to price.
        opt_mask : Backbone.Array | None, default ``None``
            A boolean array of shape ``(..., M)`` indicating which options are active (True) and which are not (False). If None, all options are active.
        opt_is_call : Backbone.Array | None, default ``None``
            A boolean array of shape ``(..., M)`` indicating which options are calls (True) and which are puts (False). If None, all options are treated as calls.
        opt_interp : dict[str|Backbone.Array]|None, default ``None``
            The option interpolation weights as returned by :meth:`GridGeometry.time_interpolation_weights`, each of which has shape  ``(..., M)``.
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
