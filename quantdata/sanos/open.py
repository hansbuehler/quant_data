"""
Open Sanos

SANOS geometry for different number of strikes per expiry.
The open code base for now only supports single sample numpy mode.
"""

from __future__ import annotations

from cdxcore import config

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.sanos.utils import Eps, _interpolate_sqrtVars, _iterative_model_price, _time_interpolation_weights, vec_max_dx, GeometryBase
from quantdata.utils.backbone import Backbone, BackboneNumpy
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
import math as math
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
import warnings as warnings
from collections.abc import Sequence

_npbb = BackboneNumpy()

@dataclass(frozen=True)
class OpenGeometry(GeometryBase):
    """
    SANOS geometry for different number of strikes per expiry.
    The open code base for now only supports single sample numpy mode.
    """
    xexpiries  : np.ndarray            #: model expiries in years, starting from 0, of size (T+1,) where T is the number of non-zero expiries. The first ``xexpiry`` must be zero.
    xexpiries  : np.ndarray            #: model expiries in years, starting from 0, of size (T+1,) where T is the number of non-zero expiries. The first ``xexpiry`` must be zero.
    dlv_min    : float = 0.01          #: minimum discrete local volatility (dlv) value.
    dlv_max    : float = 5.0           #: maximum discrete local volatility (dlv) value.
    fwdvol_min : float = 0.0           #: minimum backbone forward volatility value.
    vol_interp : str = "lin_sqrt_time"  #: Term volatility interpolation method. Must be a 'lin_var' or 'lin_sqrt_time'

    def __post_init__(self):
        verify_inp( self.dlv_min > 0., lambda : f"'dlv_min' must be positive, but found {self.dlv_min:.4g}") 
        verify_inp( self.dlv_max > self.dlv_min, lambda : f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min >= 0., lambda : f"'fwdvol_min' must be non-negative; found {self.fwdvol_min:.4g}" )
        verify_inp( self.vol_interp in {"lin_sqrt_time", "lin_var"}, lambda : f"'vol_interp' must be either 'lin_sqrt_time' or 'lin_var', but found {self.vol_interp}" )

        verify_inp( len(self.xexpiries.shape) == 1 and len(self.xexpiries) > 1, lambda : "'xexpiries' must be non-empty 1D array of at least two expiries, starting in 0" )
        verify_inp( self.xexpiries[0] == 0., lambda : f"First expiry must be zero; found {self.xexpiries[0]}" )
        verify_inp( np.min( np.diff(self.xexpiries) ) > 0., lambda : "'xexpiries' must be strictly increasing" )

        verify_inp( len(self.xstrikes.shape) == len(self.xexpiries), "'strikes' must be a list of 1D arrays or a 2D array with the same number of rows as 'expiries'")
        for t, strikes_t in enumerate(self.strikes):
            verify_inp( len(strikes_t.shape) == 1 and strikes_t.shape[0] >= 5, lambda: f"'strikes' must have at least 5 elements in its second dimension, found shape {strikes_t.shape}" )
            verify_inp( np.all( np.diff(strikes_t) > 0), lambda: f"'strikes' must be strictly increasing in its second dimension, found {np.min(np.diff(strikes_t)):.4g}")
            verify_inp( strikes_t[0] > 0., lambda: f"'strikes' must be positive, found {strikes_t[0]:.4g}" )
            verify_inp( strikes_t[1] < 1. and strikes_t[-2] > 1., lambda: f"'strikes' in its second dimension must include positive values below and above 1, found boundary values {strikes_t[:2]} and {strikes_t[-2:]}" )
            verify_inp( np.all( np.diff( self.xstrikes[:,0] ) <= 0.), lambda: f"The lowest 'strikes' must not increase in time, {self.xstrikes[:,0]:.4g}, {np.diff( self.xstrikes[:,0] ):.4g}")

            if t>0:
                verify_inp( self.xstrikes[t-1,0] >= strikes_t[0], lambda: f"The lowest 'strikes' must not increase in time, found {self.xstrikes[t-1,0]:.4g} and {strikes_t[0]:.4g}" )
                verify_inp( self.xstrikes[t-1,-1] <= strikes_t[-1], lambda: f"The highest 'strikes' must not decrease in time, found {self.xstrikes[t-1,-1]:.4g} and {strikes_t[-1]:.4g}" )

        self._make_density0()

    @property
    def has_grid(self) -> bool:
        """ Returns True if the geometry has a grid, i.e. if the strikes are given as a 2D array with the same number of rows as expiries. Returns False if the strikes are given as a list of 1D arrays. """
        return False

    @property
    def strikes(self) -> list[np.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return [self.xstrikes[t,1:-1] for t in range(1, self.xstrikes.shape[0])]

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density """
        return self._density0   
    
@dataclass(frozen=True)
class OpenGeometryGenerator(GeometryBase):
    """
    Utility class to create an :class:`OpenGeometry` from a list of market expiries and market pure strikes,
    with some heuristics to find the extrapolation strikes and control the density of strikes in the grid.
    This is the code used for the paper https://arxiv.org/pdf/2601.11209.
    """

    max_dx : float = 0.5      # maximum distance between strikes in the grid
    min_dx : float = 0.01     # minimum distance between strikes in the grid
    lower_scale : float = 0.5 # scale factor to apply to the lowest strike of each expiry to find the lowest extrapolation strike, and to the highest strike to find the highest extrapolation strike. The final lowest extrapolation strike is the minimum between the lowest strike times this factor and this factor itself, and similarly for the highest extrapolation strike. For example, with a lowest strike of 0.8 and a lower_scale of 0.5, the lowest extrapolation strike will be min(0.8*0.5, 0.5) = 0.4. With a highest strike of 1.2 and an upper_scale of 1.5, the highest extrapolation strike will be max(1.2*1.5, 1.5) = 1.8.
    upper_scale : float = 1.5 # scale factor to apply to the highest strike of each expiry to find the highest extrapolation strike, as described for lower_scale.

    def __post_init__(self):
        super().__post_init__()

        verify_inp( self.lower_scale > 0., lambda: f"'lower_scale' must be positive, found {self.lower_scale:.4g}" )
        verify_inp( self.upper_scale > 0., lambda: f"'upper_scale' must be positive, found {self.upper_scale:.4g}" )
        verify_inp( self.lower_scale < 1., lambda: f"'lower_scale' must be less than 1, found {self.lower_scale:.4g}" )
        verify_inp( self.upper_scale > 1., lambda: f"'upper_scale' must be greater than 1, found {self.upper_scale:.4g}" )
        verify_inp( self.min_dx > 0., lambda: f"'min_dx' must be positive, found {self.min_dx:.4g}" )
        verify_inp( self.max_dx > 0., lambda: f"'max_dx' must be positive, found {self.max_dx:.4g}" )

    def create_geometry( 
        self,
        expiries     : np.ndarray,
        pure_strikes : list[np.ndarray], *,
        dtype : np.dtype = np.float64            
    ):
        """
        Create the :class:`OpenGeometry` from market expiries and pure strikes,
        with some heuristics to find the extrapolation strikes and control the density of strikes in the grid.
        """
        expiries     = np.asarray( expiries, dtype=dtype )
        pure_strikes = [ np.asarray(strikes, dtype=dtype) for strikes in pure_strikes ]

        verify_inp( len(expiries.shape) == 1, lambda: f"'expiries' must be a 1D array, found array with shape {expiries.shape}" )

        T = len(expiries)

        # Prep
        # ----
        # find extrapolation strikes (heuristic - may need to improve)

        min_k    = dtype(1.)
        max_k    = dtype(1.)
        for t in range(T):
            # read input & check
            # ------------------
            strikes_t = pure_strikes[t]
            verify_inp( len(strikes_t.shape) == 1, lambda: f"'strikes[{t}]' must be a 1D array, found array with shape {strikes_t.shape}" )
            verify_inp( 0. < strikes_t[0] < 1., lambda: f"'strikes[{t}]' first strike must be positive and less than 1, found {strikes_t[0]:.4g}" )
            verify_inp( 1. < strikes_t[-1], lambda: f"'strikes[{t}]' last strike must be greater than 1, found {strikes_t[-1]:.4g}" )
            verify_inp( np.min( np.diff(strikes_t) ) > 0., lambda: f"'strikes[{t}]' must be strictly increasing, found min diff {np.min( np.diff(strikes_t) ):.4g}" )
            min_k       = min( min_k, strikes_t[0] )
            max_k       = max( max_k, strikes_t[-1] )

        min_k = min( self.lower_scale, min_k*self.lower_scale )
        max_k = max( self.upper_scale, max_k*self.upper_scale )

        # prepare output
        # --------------
        xstrikes = []

        for j in range(M):
            strikes_t = pure_strikes[t]
            assert min_k < strikes_t[0] and max_k > strikes_t[-1], ("Internal consistency error?", j, ":", min_k, "<", strikes_t[0], "//", max_k, ">", strikes_t[-1] )    
            xstrikes_t, _ = vec_max_dx( strikes_t, min_k=min_k, max_k=max_k, max_dx=max_dx, min_dx=min_dx )
            xstrikes.append( xstrikes_t )

        xexpiries = np.zeros( (T+1,), dtype=dtype)
        xexpiries[1:] = expiries

        return OpenGeometry(
            xexpiries=xexpiries,
            xstrikes=xstrikes,
            dlv_min=self.dlv_min,
            dlv_max=self.dlv_max,
            fwdvol_min=self.fwdvol_min,
            vol_interp=self.vol_interp
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
        return _time_interpolation_weights( vol_interp=self.vol_interp, expiries=self.expiries, opt_tte=opt_tte, opt_mask=opt_mask, dtype=dtype )
 
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
        xstrikes = [ np.asarray(x_t, dtype=np.float64) for x_t in self.xstrikes ] # we compute the transition operators in higher precision to avoid numerical issues, then cast back to float32 at the end
        T        = len(xstrikes)-1
        # Omega in https://arxiv.org/pdf/2601.11209 in Remark 3.5
        wn       = []
        wp       = []
        for xstrikes_t in xstrikes[1:]:
            strikes_t = xstrikes_t[:,1:-1]
            Nt        = len(strikes_t)
            wx_t      = (strikes_t[1:-1]**2) / (strikes_t[2:] - strikes_t[:-2])
            wn_t      = wx_t / (strikes_t[1:-1] - strikes_t[:-2])
            wp_t      = wx_t / (strikes_t[2:] - strikes_t[1:-1])
            assert wn_t.shape == (Nt-2,), f"Internal error: wn_t must have shape {(Nt-2,)}, found {wn_t.shape}"
            assert wp_t.shape == (Nt-2,), f"Internal error: wp_t must have shape {(Nt-2,)}, found {wp_t.shape}"
            wn.append( np.asarray( wn_t, dtype=dtype ) )
            wp.append( np.asarray( wp_t, dtype=dtype ) )

        # interpolation matrx IL
        # https://arxiv.org/pdf/2601.11209 (22)
        # we compute this in higher precision

        L = []
        for t in range(T):
            xstrikes_t  = xstrikes[t+1]
            strikes_tm1 = xstrikes[t][1:-1]
            Nt = len(xstrikes_t)-2
            Ntmn1 = len(strikes_tm1)

            LUp_t = ( np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[2:,None] ) - 
                      np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[1:-1,None] ) ) /\
                    ( xstrikes_t[2:,None] - xstrikes_t[1:-1,None] )
            LDn_t = ( np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[1:-1,None] ) - 
                      np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[:-2,None] ) ) /\
                    ( xstrikes_t[1:-1,None] - xstrikes_t[:-2,None] )
            L_t = np.asarray( LUp_t - LDn_t, dtype=dtype )
            L.append(L_t)   
            assert L_t.shape == (Ntmn1, Nt), f"Internal error: L must have shape {(Ntmn1, Nt)}, got {L_t.shape}"       
            assert np.min(L_t) >= -eps and np.max(L_t)-1. <= eps, f"Transition operators must be between 0 and 1; found min {np.min(L_t)} and max {np.max(L_t)} [tolerance {eps:.4g}]"
        
        return dict(wn=wn, wp=wp, L=L)

class OpenSanos(object):

    def __init__(self, geometry : OpenGeometry,
                       eps      : Eps = Eps(),
                       debug    : bool = False ):

        self.geometry  = geometry
        self.eps       = eps
        self.debug     = debug
        self.transp    = self.geometry.transition_operators() # this is a numpy array, but we will convert it to the backbone array when needed in computations

    def verify( self, condition : callable, message : callable | str ):
        """ Verifies a condition and raises an error with the given message if the condition is not met. Only runs if debug mode is enabled. The condition and message can be callables to avoid unnecessary computations when debug mode is disabled. """
        if self.debug:
            verify( condition(), message )

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.geometry.T
    
    @property
    def xstrikes(self) -> list[np.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.geometry.xstrikes

    @property
    def xexpiries(self) -> np.ndarray:
        """ Returns the real model expiries excluding the zero expiry and the boundary strikes. """
        return self.geometry.xexpiries

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density """ 
        return self.geometry.density0

    @property
    def strikes(self) -> list[np.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        return self.xstrikes[1:,1:-1]

    @property
    def expiries(self) -> np.ndarray:
        """ Returns the model expiries excluding the zero expiry """
        return self.xexpiries[1:]
    
    def interpolate_sqrtVars( self, 
                              fwd_vols   : np.ndarray|float|None,
                              opt_interp : dict[str, np.ndarray], 
                              opt_mask   : np.ndarray|None = None ) -> np.ndarray|None:
        """
        Utility function to compute the square-root variance for SANOS based in fwd_vols.
        Use :meth:`time_interpolation_weights` to compute ``opt_interp``.

        This function is batchable.

        Parameters
        ----------
        fwd_vols : np.ndarray | float | None
            An array of forward volatilities of ``(..., 1|T)`` or a broadcastable shape. A float or similar is broadcasted accordingly. The function returns ``None`` if ``fwd_vols`` is ``None``.
        opt_interp : dict[str, np.ndarray]
            The option interpolation weights as returned by :meth:`time_interpolation_weights`, with keys 'lix', 'dweight', 'vweight', 'vextra'.
            Each entry has dimension ``(..., M)`` where M is the number of options.
        bb : Backbone, default ``BackboneNumpy()``
            The backbone to use for computations. Default is ``BackboneNumpy()``.
        debug : bool, default ``False``
            Whether to perform additional checks for debugging purposes. Default is False.

        Returns
        -------
        opt_sqrtVars : np.ndarray or None
            The interpolated square-root variances for the options, of shape matching ``opt_interp['lix']``. If fwd_vols is None, returns None. 
        """
        return _interpolate_sqrtVars( 
                            bb         = _npbb ,
                            xexpiries  = self.xexpiries,
                            vol_interp = self.vol_interp,
                            fwdvol_min = self.geometry.fwdvol_min,
                            fwd_vols   = fwd_vols,
                            opt_interp = opt_interp, 
                            opt_mask   = opt_mask,
                            debug      = self.debug )
    def price(
            self, 
            # market data
            fwd_vols    : np.ndarray | None,
            densities   : list[np.ndarray],
            #
            opt_strikes : np.ndarray,
            opt_tte     : np.ndarray,
            opt_mask    : np.ndarray | None = None,
            opt_is_call : np.ndarray | None = None,
            opt_interp  : dict[str|np.ndarray]|None = None,
        ) -> np.ndarray:
        """
        SANOS option pricing.
        
        No batching for now.

        Parameters
        ----------
        fwd_vols : np.ndarray | None
            The forward volatilities as an array of shape ``(1|T,)`` or in a broadcastable shape. Floats are broadcasted accordingly.
            Use ``None`` for non-smooth linear interpolation.
        densities : list[np.ndarray]
            The ``T`` model densities for the grid expiries and strikes.
            Densities can be computed from discrete local vols squared using :meth:`densities`.
        opt_strikes : np.ndarray
            The option strikes, of size ``M``.
        opt_tte : np.ndarray
            The option times to expiry, of size ``M``.
        opt_mask : np.ndarray | None, default ``None``
            A boolean array of size ``M`` indicating which options are active (True) and which are not (False). If None, all options are active.
        opt_is_call : np.ndarray | None, default ``None``
            A boolean array of size ``M`` indicating which options are calls (True) and which are puts (False). If None, all options are treated as calls.
        opt_interp : dict[str|np.ndarray]|None, default ``None``
            The option interpolation weights as returned by :meth:`OpenGeometry.time_interpolation_weights`, each of which has shape  ``(..., M)``.
            If ``None``, these are computed on the fly. However, this is a numpy operation and will be *very costly* torch or jax usage.
            It is strongly advised to calculate these ahead of time.

        Returns
        -------
        Prices : ``np.ndarray``
            Option prices.
        """
        bb       = self.bb
        M        = opt_strikes.shape[-1]
        verify_inp( len(opt_strikes.shape) == 1, lambda: f"'opt_strikes' must be a 1D array of shape (M,), found {opt_strikes.shape}" )

        if opt_interp is None:
            if not isinstance(bb, BackboneNumpy):
                warnings.warn(f"{self.__class__.__name__} function 'price' using backbone {np.__class__.__name__}: no 'opt_interp' was provided. This will be computed on the fly using numpy which will seriously impact performance. Consider computing this ahead of time.")
            opt_interp  = self.geometry.time_interpolation_weights( opt_tte=opt_tte, opt_mask=opt_mask, dtype=self.xexpiries.dtype )
            opt_interp  = { k: [ np.to_array(x) for x in v ] for k,v in opt_interp.items() }
        opt_sqrtVars = self.interpolate_sqrtVars( fwd_vols, opt_interp=opt_interp, opt_mask=opt_mask )
        assert opt_sqrtVars is None or opt_sqrtVars.shape[-1] == M, f"Internal error: opt_sqrtVars must have shape ending in {M}, found {opt_sqrtVars.shape}"

        mstrikes      = self.strikes
        fit_calls     = _iterative_model_price(
            opt_tte     = opt_tte,
            opt_strikes = opt_strikes,
            opt_sqrtVars= opt_sqrtVars,
            opt_lix     = opt_interp['lix'],
            opt_dweight = opt_interp['dweight'],
            strikes     = mstrikes,
            densities   = densities,
            no_warnings = self.no_warnings,
            debug       = self.debug,
            return_array = True
        )

        prices     = fit_calls if opt_is_call is None else np.where( opt_is_call | ~opt_mask, fit_calls, fit_calls + opt_strikes - 1. )
        prices     = prices * opt_mask if opt_mask is not None else prices
        assert prices.shape == (M,), f"Expected prices to have shape {(M,)}, got {prices.shape}"
        return prices 

    def densities( self, dlvsqs : list[np.ndarray] ) -> list[np.ndarray]:
        """
        Compute densities from DLV squared.

        Parameters
        ----------
        dlvsqs : list[np.ndarray]
            The DLVs squared per time step ``T``, i.e. the local variance divided by the time step, at each model time and strike.
            The density for some time ``t`` should have the same dimension as the regula strikes at that time step.
            This function will add `geometry.dlv_min**2` to the input DLVs squared to ensure numerical stability.
        
        Returns
        -------
        Densites : list[np.ndarray]
            List of ``T`` arrays representing the densities at each model time, not including the initial density at time 0 (which can be accessed via ``self.density0``). 
        """
        T           = self.T

        self.verify( lambda : np.all((self.density0 >= -self.eps.densities) & (self.density0 - 1. <= self.eps.densities)), lambda : f"Density 0 must be between 0 and 1; found min {np.min(self.density0)} and max {np.max(self.density0)} [tolerance 'densities' {self.eps.densities}]" )
        self.verify( lambda : np.all(np.abs( np.sum( self.density0, axis=-1 ) - 1. ) < self.eps.densities), lambda : f"Density 0 must have positive mass, found sum {np.max(np.abs( np.sum( self.density0, axis=-1 ) - 1. ))} [tolerance 'densities' {self.eps.densities}]" )

        dts          = np.diff(self.xexpiries) # in numpy array
        wn           = self.transp['wn']
        wp           = self.transp['wp']
        L            = self.transp['L']
    
        densities    = []
        min_dlvsq    = self.geometry.dlv_min**2

        for t in range(T):
            Ntm1          = self.xstrikes[t].shape[0]-2   
            Nt            = self.xstrikes[t+1].shape[0]-2   

            # interpolate up
            mdensity_tm1 = densities[-1] if t > 0 else self.density0 
            assert mdensity_tm1.shape == (Ntm1,_), f"Expected mdensity_tm1 to have shape ending in {N}, found {mdensity_tm1.shape}"
            L_t          = L[t]
            assert L_t.shape == (Ntm1, Nt), f"Expected L_t to have shape {(Ntm1, Nt)}, found {L_t.shape}"

            mdensity_t   = np.einsum( f"mn, n -> m", L_t, mdensity_tm1 )  # interpolate density to time t 
            assert mdensity_t.shape[-1] == Nt, f"Expected mdensity_t after L to have shape ending in {Nt}, found {mdensity_t.shape}"

            # solve
            dlvsqs_t     = dlvsqs[t]
            wn_t         = wn[t]
            wp_t         = wp[t]
            assert dlvsqs_t.shape == (Nt-2,), f"Expected dlvsqs_t to have shape {(Nt-2,)}, found {dlvsqs_t.shape}"
            assert wn_t.shape == (Nt-2,), f"Expected wn_t to have shape {(Nt-2,)}, found {wn_t.shape}"
            assert wp_t.shape == (Nt-2,), f"Expected wp_t to have shape {(Nt-2,)}, found {wp_t.shape}"

            dlvar_t      = (dlvsqs_t+min_dlvsq) * dts[t]  # local variance at time t, shape (...,N-2)

            w_minus      = dlvar_t * wn_t
            w_plus       = dlvar_t * wp_t
            mdensity_t   = solve_sanos_tridiagonal(_npbb, w_minus=w_minus, w_plus=w_plus, rhs=mdensity_t ) # keep
            assert mdensity_t.shape[-1] == Nt, f"Expected mdensity_t to have shape ending in {Nt}, found {mdensity_t.shape}"
            densities.append( mdensity_t )

        return densities
