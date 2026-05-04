"""
SANOS

Model geometry
"""

from dataclasses import dataclass
import math
import numpy as np
from cdxcore.version import version
from cdxcore.err import verify_inp, verify
from aae import expiries
from quantdata.biz_calendar import DAYS_PER_YEAR, NMS
from quantdata.dysanos.simple.sampler import vec_max_dx
from quantdata.utils.backbone import Backbone, BackboneNumpy
from collections.abc import Sequence

@version("1.0")
@dataclass(frozen=True)
class Geometry:
    r"""
    SANOS Model Geometry

    Contains strikes, expiries, interpolation method and bounds on model parameters.
    The SANOS model is built upon $T$ active model expiries $0<t_1<\cdots<t_T$
    and $N_j$ active strikes per expiry. We expand expiries to set $t_0:=0$ and 
    also add boundary strikes around the model. The corresponding extended ``xstrikes``
    have $N_j+2$ strikes per expiry.
    """
    xtimes     : np.ndarray                        #: model expiries in years, starting from 0, of size (T+1,) where T is the number of non-zero expiries. The first ``xexpiry`` must be zero.
    xstrikes   : list[np.ndarray]|np.ndarray       #: model "pure" strikes, including boundary strikes and time-zero strikes. If this is an :class:`numpy.ndarray` then this has dimension (T+1,N+2,N+2)
    dlv_min    : float = 0.01                      #: minimum discrete local volatility (dlv) value.
    dlv_max    : float|None = 5.0                  #: maximum discrete local volatility (dlv) value.
    fwdvol_min : float = 0.01                      #: minimum backbone forward volatility value.
    vol_interp : str = "sqrt_lin_vol"              #: Term volatility interpolation method. Must be a 'sqrt_lin_vol' or 'lin_var'.

    def __post_init__(self):
        verify_inp( self.dlv_min > 0., lambda : f"'dlv_min' must be positive, but found {self.dlv_min:.4g}") 
        verify_inp( self.dlv_max  is None or self.dlv_max > self.dlv_min, lambda : f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min > 0., lambda : f"'fwdvol_min' must be positive; found {self.fwdvol_min:.4g}" )
        verify_inp( self.vol_interp in ["sqrt_lin_vol", "lin_var"], lambda : f"'vol_interp' must be either 'sqrt_lin_vol' or 'lin_var', but found {self.vol_interp}" )

        verify_inp( len(self.xtimes.shape) == 1 and len(self.xtimes) > 1, lambda : "'xtimes' must be non-empty 1D array of at least two expiries, starting in 0" )
        verify_inp( self.xtimes[0] == 0., lambda : f"First expiry must be zero; found {self.xtimes[0]}" )
        verify_inp( np.min( np.diff(self.xtimes) ) > 0., lambda : "'xtimes' must be strictly increasing" )

        if hasattr(self.xstrikes,"shape"):
            verify_inp( len(self.xstrikes.shape) == 2 and self.xstrikes.shape[0] == len(self.xtimes), "'strikes' must be a list of 1D arrays or a 2D array with the same number of rows as 'expiries'")

            verify_inp( self.xstrikes.shape[1] >= 5, lambda: f"'strikes' must have at least 5 elements in its second dimension, found shape {self.xstrikes.shape}" )
            verify_inp( np.all(np.diff(self.xstrikes,axis=1) > 0), lambda: f"'strikes' must be strictly increasing in its second dimension, found {self.xstrikes}" )
            verify_inp( np.all( 0. < self.xstrikes[:,0] ) and np.all( self.xstrikes[:,0] < 1. ) and np.all( self.xstrikes[:,-1] > 1. ), lambda: f"'strikes' in its second dimension must include positive values below and above 1, found boundary values {self.xstrikes[0]:.4g} and {self.xstrikes[-1]:.4g}" )
            verify_inp( np.all( np.diff( self.xstrikes[:,0] ) <= 0.), lambda: f"The lowest 'strikes' must not increase in time, {self.xstrikes[:,0]:.4g}, {np.diff( self.xstrikes[:,0] ):.4g}")
            verify_inp( np.all( np.diff( self.xstrikes[:,-1] ) >= 0.), lambda: f"The highest 'strikes' must not decrease in time, {self.xstrikes[:,-1]:.4g}, {np.diff( self.xstrikes[:,-1] ):.4g}")


        else:
            assert isinstance(self.xstrikes, Sequence), "'strikes' must be a list of 1D arrays or a 2D array"
            verify_inp( isinstance(self.xstrikes, list), "'strikes' must be a list of 1D arrays or a 2D array" )
            verify_inp( len(self.xstrikes) == len(self.xtimes), lambda: f"'strikes' must have the same length as 'expiries', found {len(self.xstrikes)} strikes and {len(self.xtimes)} expiries" )

            for it, xstrikes_t in enumerate(self.xstrikes):
                verify_inp( len(xstrikes_t) >= 5, lambda: f"'xstrikes' for expiry index {it} must have at least 5 elements, found {len(xstrikes_t)}" )
                verify_inp( np.min( np.diff(xstrikes_t) ) > 0., lambda: f"'strikes' for expiry index {it} must be strictly increasing" )
                verify_inp( 0 < xstrikes_t[0] < 1. < xstrikes_t[-1], lambda: f"'strikes' for expiry index {it} must include positive values below and above 1, got {xstrikes_t[0]} and {xstrikes_t[-1]:.4g}" )

                if it>0:
                    strikes_prev = self.xstrikes[it-1]
                    verify_inp( xstrikes_t[0] <= strikes_prev[0], lambda: f"The lowest 'strikes' must not increase in time, got {xstrikes_t[0]:.4g} at expiry index {it} and {strikes_prev[0]} at expiry index {it-1}" )
                    verify_inp( xstrikes_t[-1] >= strikes_prev[-1], lambda: f"The highest 'strikes' must not decrease in time, got {xstrikes_t[-1]:.4g} at expiry index {it} and {strikes_prev[-1]} at expiry index {it-1}" )

        _ = self.density0 # trigger xstrike alignment

    @property
    def grid_mode(self) -> bool:
        """ Whether 'xstrikes' are a numpy matrix or a list of numpy arrays (and, hence, have in homogeneous number of strikes per expiry) """
        return isinstance(self.strikes, np.ndarray)

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.expiries.shape[0]-1

    @property
    def N(self) -> int|None:
        """ Returns the common number of strikes accross expiries, if that exists, or None otherwise. """
        return self.xstrikes.shape[1]-2 if hasattr(self.xstrikes, "shape") else None

    @property
    def strikes(self) -> np.ndarray|list[np.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        if hasattr(self.xstrikes, "shape"):
            return self.xstrikes[1:,1:-1]
        else:
            return [ strikes_t[1:-1] for strikes_t in self.xstrikes[1:] ]

    @property
    def expiries(self) -> np.ndarray:
        """ Returns the model expiries excluding the zero expiry """
        return self.xtimes[1:]

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density, which is a discrete density defined on the first expiry's strikes. This is derived from the first expiry's strikes and must be consistent with them. """
        if not '_density0' in self.__dict__:
            strikes0 = self.xstrikes[0] if self.grid_mode else self.xstrikes[0]             
            N2       = len(strikes0)
            ix_1     = min(N2-2,max(1,np.abs(strikes0 - 1.).argmin()))
            assert strikes0[ix_1-1] < 1. < strikes0[ix_1+1], f"Internal error: strikes should be around 1??? found {strikes0[ix_1-1]} and {strikes0[ix_1+1]}."
            strikes0[ix_1] = 1.
            density0 = np.zeros_like( strikes0[1:-1])
            density0[ix_1-1] = 1.
            self.__dict__['_density0'] = density0
        return self._density0   

    @staticmethod
    def create_grid_nms( *, initial_t : float = 1./DAYS_PER_YEAR,
                            horizon_t : float = 5., 
                            num_times : int = 11,
                            nms_range : float | tuple[float,float] = 1.,
                            num_nms   : int = 11,
                            dlv_min   : float = 0.01,
                            dlv_max   : float|None = 5.,
                            vol_min   : float = 0.01,
                            vol_interp: str = "sqrt_lin_vol",
                            lower_scale : float = 0.2,
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
        vol_interp : str, default ``"sqrt_lin_vol"``
            Volatility interpolation method. Must be 'sqrt_lin_vol' or 'lin_var'. Default is 'sqrt_lin_vol'.

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

        nms_lo  = float(-nms_range if isinstance(nms_range, float) else nms_range[0])
        nms_hi  = float(+nms_range if isinstance(nms_range, float) else nms_range[1])

        xtimes        = np.zeros( (num_times+1,), dtype=dtype )  
        xtimes[1:]    = np.exp( np.linspace(math.log(initial_t), math.log(horizon_t), num_times) )
        nms              = np.linspace( nms_lo, nms_hi, num_nms )
        xstrikes         = np.zeros((num_times+1,num_nms+2), dtype=dtype )
        xstrikes[:,1:-1] = NMS.from_nms( nms[None,:], t=xtimes )

        decay            = np.exp( -xtimes*0.0001 )
        xstrikes[:,0]    = np.min( lower_scale * xstrikes[:,1] * decay )
        xstrikes[:,-1]   = np.max( upper_scale * xstrikes[:,-2] / decay )

        return Geometry(
            xtimes     = xtimes,
            xstrikes   = xstrikes,
            dlv_min    = dlv_min,
            dlv_max    = dlv_max,
            vol_min    = vol_min,
            vol_interp = vol_interp,
        )

    @staticmethod
    def create_from_market(
                    market_expiries : np.ndarray,
                    market_strikes  : list[np.ndarray], *,
                    max_dx : float = 0.5,
                    min_dx : float = 0.05,
                    lower_scale : float = 0.2,
                    upper_scale : float = 1.5,
                    dlv_min    : float = 0.01,                      #: minimum discrete local volatility (dlv) value.
                    dlv_max    : float|None = 5.0,                  #: maximum discrete local volatility (dlv) value.
                    vol_min    : float = 0.01,                      #: minimum backbone volatility value.
                    vol_interp : str = "sqrt_lin_vol",               #: Term volatility interpolation method. Must be a 'sqrt_lin_vol' or 'lin_var'.
                    dtype      : np.dtype = np.float64
            ):
        """
        Creates a "fitting" geometry based on market strikes and expiries.

        This geometry is designed to be used for fitting the model to market data, as it includes strikes that are close to the market strikes and expiries.
        It uses observed market strikes and ensures that they are at most ``max_dx`` apart from each other, while also ensuring that they are at least ``min_dx`` apart.
        The strikes are scaled by ``lower_scale`` and ``upper_scale`` to ensure that the geometry includes strikes below and above the market strikes.

        Boundary strikes are added accordingly.
        """
        verify_inp( len(market_expiries.shape) == 1 and len(market_expiries) > 0, "'market_expiries' must be non-empty 1D array") )
        verify_inp( np.min( np.diff(market_expiries) ) > 0., "'market_expiries' must be strictly increasing" )
        verify_inp( market_expiries[0] > 0., lambda : f"First 'market_expiries' must be positive, found {market_expiries[0]:.4g}" )
        verify_inp( len(market_strikes) == len(market_expiries), lambda:  f"'market_strikes' must have the same length as 'market_expiries', found {len(market_strikes)} strikes and {len(market_expiries)} expiries" )
        verify_inp( max_dx > 0., lambda : f"'max_dx' must be positive, found {max_dx:.4g}" )
        verify_inp( min_dx >= 0., lambda : f"'min_dx' must be non-negative, found {min_dx:.4g}" )
        verify_inp( lower_scale > 0. and lower_scale < 1., lambda : f"'lower_scale' must be in (0,1), found {lower_scale:.4g}" )
        verify_inp( upper_scale > 1., lambda : f"'upper_scale' must be greater than 1, found {upper_scale:.4g}" )

        # heuristic to get min/max strikes
        min_k = 0.4
        max_k = 1.5
        for j, strikes_t in enumerate(market_strikes):
            verify_inp( len(strikes_t) > 0, lambda : f"'market_strikes' for expiry #{j} must be non-empty" )
            verify_inp( np.min( np.diff(strikes_t) ) > 0., lambda : f"'market_strikes' for expiry #{j} must be strictly increasing" )
            verify_inp( 0 < strikes_t[0], lambda : f"'market_strikes' for expiry #{j} must be positive. First strike is {strikes_t[0]:.4g}" )
            min_k = min( min_k, strikes_t[0]*lower_scale )
            max_k = max( max_k, strikes_t[-1]*upper_scale )

        model_strikes = []

        for j, strikes_t in enumerate(market_strikes):
            expiry = market_expiries[j]
            decay  = math.exp( -expiry*0.0001 )
            mstrikes_t, mx_ix_t = vec_max_dx( strikes_t, min_k=min_k*decay, max_k=max_k/decay, max_dx=0.5, min_dx=0.05 )
            model_strikes.append( np.asarray( mstrikes_t, dtype=dtype) )

        return Geometry(
            expiries  = np.asarray( market_expiries, dtype=dtype ),
            strikes   = model_strikes,
            dlv_min   = dlv_min,
            dlv_max   = dlv_max,
            vol_min   = vol_min,
            vol_interp = vol_interp,
        )

    # -------------------------------------
    # Numerical utilities
    # -------------------------------------

    def transition_operators(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        r"""
        Computes the DLV transition operators for the grid geometry, as described in https://arxiv.org/pdf/2601.11209, Remark 3.5 and Section 3.4.
        These do not depend on market data.

        Returns
        -------
        wn : list[np.ndarray]
            The $w^+_t$ transition weights, of shape (T,N-2), where T is the number of non-zero expiries and N is the number of interior strikes per expiry.     
        wp : list[np.ndarray]
            The $w^-_t$ transition weights, of shape (T,N-2), where T is the number of non-zero expiries and N is the number of interior strikes per expiry.
        L : list[np.ndarray]
            The interpolation matrices. If :attr:`xstrikes` is a numpy array, then L is of shape (T,N,N),
            where T is the number of non-zero expiries and N is the number of interior strikes per expiry. If :attr:`xstrikes` is a list of numpy arrays,
            then L is a list of T numpy arrays of shape (Nt,Ntm1), where Nt is the number of interior strikes for expiry t and Ntm1 is the number of interior strikes for expiry t-1.
        """
        dtype =  self.xstrikes.dtype

        if self.grid_mode:
            xstrikes = np.asarray(self.xstrikes, dtype=np.float64) # we compute the transition operators in higher precision to avoid numerical issues, then cast back to float32 at the end
            T, N = xstrikes.shape[0]-1, xstrikes.shape[1]-2
            # Omega in https://arxiv.org/pdf/2601.11209 in Remark 3.5
            wx = (xstrikes[1:,1:-1]**2) / (xstrikes[1:,2:] - xstrikes[1:,:-2])
            wn = wx / (xstrikes[1:,1:-1] - xstrikes[1:,:-2])
            wp = wx / (xstrikes[1:,2:] - xstrikes[1:,1:-1])
            del wx
            assert wn.shape == (T,N-2) and wp.shape == (T,N-2), f"wn and wp must have shape {(T,N-2)}, got {wn.shape} and {wp.shape}"
            assert np.min(wn) >= 0. and np.min(wp) >= 0., f"Transition weights must be non-negative, found min {np.min(wn)} and {np.min(wp)}"

            # interpolation matrx IL
            # https://arxiv.org/pdf/2601.11209 (22)
            # we compute this in higher precision
            xstrikes_t = xstrikes[1:,:]
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
            assert np.min(L) >= 0. and np.max(L) <= 1., f"Transition operators must be between 0 and 1; found min {np.min(L)} and max {np.max(L)}"
            del xstrikes
            
            wp = np.asarray(wp, dtype=dtype)
            wn = np.asarray(wn, dtype=dtype)
            L  = np.asarray(L, dtype=dtype)
            wp = [ wp[t,:,:] for t in range(T) ]
            wn = [ wn[t,:,:] for t in range(T) ]
            L  = [ L[t,:,:] for t in range(T) ]
            return wn, wp, L            

        # list version
        # ------------

        assert isinstance(self.xstrikes, Sequence), "Internal error: xstrikes should be a list of numpy arrays if not a numpy array"
        strikes_tm1 = np.asarray(self.xstrikes[0][1:-1], dtype=np.float64)
        T = len(self.xtimes)-1


        wp = []
        wn = []
        L =  []

        for j in range(1, self.xtimes.shape[0]):
            xstrikes_t  = np.asarray(self.xstrikes[j], dtype=np.float64)

            # Omega in https://arxiv.org/pdf/2601.11209 in Remark 3.5
            wx_t = (xstrikes_t[1:-1]**2) / (xstrikes_t[2:] - xstrikes_t[:-2])
            wn_t = wx_t / (xstrikes_t[1:-1] - xstrikes_t[:-2])
            wp_t = wx_t / (xstrikes_t[2:] - xstrikes_t[1:-1])
            del wx_t
            assert len(wn_t) == N-2 and len(wp_t) == N-2, f"wn and wp must have shape {(N-2,)}, got {wn.shape} and {wp.shape}"
            assert np.min(wn_t) >= 0. and np.min(wp_t) >= 0., f"Transition weights must be non-negative, found min {np.min(wn_t)} and {np.min(wp_t)}"

            wp.append( np.asarray(wp_t, dtype=dtype) )
            wn.append( np.asarray(wn_t, dtype=dtype) )   
            del wp_t, wn_t

            # interpolation matrx IL
            # https://arxiv.org/pdf/2601.11209 (22)
            # we compute this in higher precision

            divup = np.asarray( xstrikes_t[2:,None] - xstrikes_t[1:-1,None] ,dtype=np.float64 )
            divdn = np.asarray( xstrikes_t[1:-1,None] - xstrikes_t[:-2,None] ,dtype=np.float64 )
            assert np.all( divup > 0. ), f"Internal error: divup must be positive, found min {np.min(divup)}"
            assert np.all( divdn > 0. ), f"Internal error: divdn must be positive, found min {np.min(divdn)}"

            Nt         = xstrikes_t.shape[0] - 2
            Ntm1       = strikes_tm1.shape[0] 

            LUp = ( np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[2:,None] ) - 
                    np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[1:-1,None] ) ) /\
                ( xstrikes_t[2:,None] - xstrikes_t[1:-1,None] )
            LDn = ( np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[1:-1,None] ) - 
                    np.maximum( 0., strikes_tm1[None,:] - xstrikes_t[:-2,None] ) ) /\
                ( xstrikes_t[1:-1,None] - xstrikes_t[:-2,None] )
            Lt = LUp - LDn
            assert Lt.shape == (Nt, Ntm1), f"Internal error: L must have shape {(Nt, Ntm1)}, got {Lt.shape}"       
            assert np.min(Lt) >= 0. and np.max(Lt) <= 1., f"Transition operators must be between 0 and 1; found min {np.min(Lt)} and max {np.max(Lt)}"
            
            L.append( np.asarray( Lt.astype, dtype=dtype ) )
            strikes_tm1 = xstrikes_t[:,1:-1]

        return wn, wp, L
    
    def time_interpolation_weights( self,
                                    opt_tte  : np.ndarray, 
                                    opt_mask : np.ndarray | None = None,
                                    dtype    : np.dtype = np.float64 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes option time interpolation weights for variance and densities.
        These do not depend on market data and should be computed ahead of time.

        * ``opt_lix``: per option the lower index into ``self.xtimes`` such that such that ``self.xtimes[opt_lix-1] < opt_tte <= self.expiries[opt_lix]``. Note that ``0<opt_lix<len(self.xtimes)``.
        * ``opt_dweight``: per option the lower interpolation weight for density interpolation.
        * ``opt_vweight``: per option the lower interpolation weight for variance interpolation (can be linear in variance or in sqrt time).
        * ``opt_vextra``: per option extrapolation with final volatility (linear or sqrt)

        Call price BS sqrt-variance is computed as follows in ``sqrt_lin_vol`` interpolation mode:

            * if ``t_i < xtimes[0]`` then ``opt_lix_i = 0`` and 
                ``sqrtVar_i = (1-opt_vweight_i) * sqrtVars[opt_lix_i]``.

            * if ``xtimes[0] <= t_i <= xtimes[-1]`` then ``0 < opt_lix_i <= T-1`` and
                ``sqrtVar_i = opt_vweight_i * sqrtVars[opt_lix_i-1] + (1-opt_vweight_i) * sqrtVars[opt_lix_i]``.

            * if ``t_i > xtimes[-1]`` then we set ``opt_lix_i == T-1`` and
                ``sqrtVar_i = (1 - opt_vweight_i) * sqrtVars[T-1] + opt_vextra_i * vol_extra`` where ``opt_vweight_i = 0``.

        Once set, we also interpolate in model densities.

            * if ``t_i < xtimes[0]`` then ``opt_lix_i = 0`` and
                ``call_i = (1-opt_dweight_i) * call[opt_lix_i]`` with ``opt_dweight_i = 0``.

            * if ``xtimes[0] <= t_i <= xtimes[-1]`` then ``0 < opt_lix_i <= T-1`` and
                ``call_i = opt_dweight_i * call[opt_lix_i-1] + (1-opt_dweight_i) * call[opt_lix_i]``.

            * if ``t_i > xtimes[-1]`` then we set ``opt_lix_i == T-1`` and
                ``call_i = (1 - opt_dweight_i) * call[T-1]`` with ``opt_dweight_i = 0``.

        Note that ``0<opt_lix<len(self.xtimes)``.
        """
        T        = self.T
        shape    = opt_tte.shape
        dtype    = dtype if not dtype is None else opt_tte.xstrikes.dtype

        # we use the mask to set opt_tte to 1; from thereon proceed as usual
        opt_mask = opt_mask if ( not opt_mask is None and not np.all(opt_mask) ) else None
        opt_tte  = np.where( opt_mask, opt_tte, 1. ) if opt_mask is not None else opt_tte 
        opt_tte  = np.asarray(opt_tte, dtype=np.float64)
        assert np.min( opt_tte ) > 0., f"Time to expiry must be positive, found min {np.min(opt_tte)}"
        opt_tte  = opt_tte.flatten()
        M        = len(opt_tte)

        # find indices ixtte such that xtimes[ixtte-1] < opt_tte <= xtimes[ixtte+1]
        ttes     = np.unique( opt_tte )
        sqrtTtes = np.sqrt( ttes )
        times    = np.asarray(self.expiries, dtype=np.float64)
        sqrtT    = np.sqrt( times )
        ixtte    = np.searchsorted( times, ttes, side='left' )
        assert np.min(ixtte) >= 0, f"Internal error - times[0]>0, so searchsorted must return indices of at least 0, found min {np.min(ixtte)}. Please contact authors."
        assert np.max(ixtte) <= T, f"Internal error - searchsorted must return indices of at most len(times), found max {np.max(ixtte)}. Please contact authors."
        assert ixtte.shape == ttes.shape, f"Expected ixtte to have shape {ttes.shape}, got {ixtte.shape}"

        # compute linear and sqrt-linear weightsl
        assert self.vol_interp in {"sqrt_lin_vol","lin_var"}, f"Internal error: vol_interp must be 'sqrt_lin_vol' or 'lin_var', found '{self.vol_interp}'"

        # create weights for ixteel-1. 
        # note that in case of extrapolation ixteel=T-1, hence the weight for ixteel-1 will be 0.
        # dweight is the weight for the previous density call price.
        ixtte_cap   = np.minimum( ixtte, T-1 )
        ixtte_m1    = np.maximum( ixtte-1, 0 )
        prev_times  = np.where( ixtte == 0, 0., times[ixtte_m1] )
        divisor     = np.where( ixtte == T, 1., times[ixtte_cap] - prev_times )
        assert np.min( divisor) > 0., f"Internal error: divisor for time interpolation weights must be positive, found min {np.min(divisor)}. Please contact authors."
        dweight     = np.select( [ixtte==0, ixtte==T],
                                [0.,       0.],
                                ( times[ixtte_cap] - ttes ) / divisor )
        
        if self.vol_interp == "sqrt_lin_vol":
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
        return opt_lix, opt_dweight, opt_vweight, opt_vextra

