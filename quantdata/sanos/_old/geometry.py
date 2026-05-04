"""
Grid Sanos

SANOS itself is a rather generic framework which can accomodate any number of model strikes and expires, potentially depending on market data for example when used in "interpolatnion mode" as in https://arxiv.org/pdf/2601.11209.
Grid SANOS, on the other hand, specifies SANOS on a fixed strike and expiry grid for a machine learned representation of an option surface. 

The framework here represents a fixed grid approach. The core class is :class:`dys.grid_sanos.GridGeometry`, which represents the geometry of the grid.
"""

from __future__ import annotations
from operator import call
import rlcompleter

from torch.distributed import Backend

from dys.biz_calendar import DAYS_PER_YEAR
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
import math as math
from cdxcore.err import verify_inp
from cdxcore.verbose import Context
import warnings as warnings

from dys.grid_sanos.backbone import BackboneNumpy

DEFAULT_TIMES = np.exp( np.linspace(math.log(1./DAYS_PER_YEAR), math.log(5.), 11) ).astype(np.float32) #: Default 'times' for GridSanosConfig, in years. 10 points from 0 to 5 years, with more density near 0.
DEFAULT_NMS = np.linspace(-1.,1,11, dtype=np.float32) #: Default 'strikes' for GridSanosConfig, in NMS:
DEFAULT_NMS_EPS = 1./255./8. #: Small epsilon for NMS computation.

@dataclass
class GridGeometry:
    """
    Geometry for SANOS on a fixed grid.

    SANOS itself is a rather generic framework which can accomodate any number of model strikes and expires, potentially depending on market data for example when used in "interpolatnion mode" as in https://arxiv.org/pdf/2601.11209.
    Grid SANOS, on the other hand, specifies SANOS on a fixed strike and expiry grid for a machine learned representation of an option surface. 

    This class represents the goemetry of such grid.    
    """
    expiries   : np.ndarray = None   #: Times of the grid, in years. Must be strictly increasing and positive.
    strikes    : np.ndarray = None   #: NMS of the grid, in log-moneyness. Must be strictly increasing and include values below and above 0.
    nms_eps    : float = DEFAULT_NMS_EPS #: Small epsilon for NMS computation. Must be positive.
    vol_interp : str = "sqrt_lin_vol" #: Volatility interpolation method. Must be a 'sqrt_lin_vol' or 'lin_var'.

    @property
    def info_label(self) -> str:
        """ String information """
        return f"{len(self.expiries)} expiries from {self.expiries[0]:.4g} to {self.expiries[-1]:.4g} years, and {self.strikes.shape[1]} strikes; "+\
               ("sqrt-Lin Vol BS interpolation." if self.vol_interp=='sqrt_lin_vol' else "lin Var BS interpolation.")

    def __post_init__(self):

        verify_inp( isinstance(self.expiries, np.ndarray), f"'expiries' must be a numpy array, found type {type(self.expiries)}" )
        verify_inp( len(self.expiries.shape) == 1, f"'expiries' must be a 1D array, found shape {self.expiries.shape}" )
        verify_inp( len(self.expiries) > 0, "'expiries' must be non-empty" )
        verify_inp( self.expiries[0] > 0. and ( len(self.expiries) == 1 or np.min(np.diff(self.expiries)) > 0 ), f"'expiries' must be strictly increasing and positive, found {self.expiries}" )
        
        verify_inp( isinstance(self.strikes, np.ndarray), f"'strikes' must be a numpy array, found type {type(self.strikes)}" )

        if len(self.strikes.shape) == 1:
            verify_inp( len(self.strikes) >= 3, f"'strikes' must have at least 3 elements, found shape {self.strikes.shape}" )
            verify_inp( np.all(np.diff(self.strikes) > 0), f"'strikes' must be strictly increasing, found {self.strikes}" )
            verify_inp( 0. < self.strikes[0] < 1. < self.strikes[-1], f"'strikes' must include positive values below and above 1, found boundary values {self.strikes[0]} and {self.strikes[-1]}" )

            strikes_ = np.zeros_like((len(self.expiries),len(self.strikes)), dtype=self.strikes.dtype )
            strikes_[:,:] = self.strikes[None,:]
            self.strikes = strikes_
            del strikes_
        else:
            verify_inp( len(self.strikes.shape) == 2, f"'strikes' must be a vector or a matrix, found shape {self.strikes.shape}" )
            verify_inp( self.strikes.shape[0] == len(self.expiries), f"'strikes' first dimension must match number of expiries, {len(self.expiries)}; found {self.strikes.shape}" )
            verify_inp( self.strikes.shape[1] >= 3, f"'strikes' must have at least 3 elements in its second dimension, found shape {self.strikes.shape}" )
            verify_inp( np.all(np.diff(self.strikes,axis=1) > 0), f"'strikes' must be strictly increasing in its second dimension, found {self.strikes}" )
            verify_inp( np.all( 0. < self.strikes[:,0] ) and np.all( self.strikes[:,0] < 1. ) and np.all( self.strikes[:,-1] > 1. ), f"'strikes' in its second dimension must include positive values below and above 1, found boundary values {self.strikes[0]} and {self.strikes[-1]}" )
            verify_inp( np.all( np.diff( self.strikes[:,0] ) <= 0.), f"The lowest 'strikes' must not increase in time, {self.strikes[:,0]}, {np.diff( self.strikes[:,0] )}")
            verify_inp( np.all( np.diff( self.strikes[:,-1] ) >= 0.), f"The highest 'strikes' must not decrease in time, {self.strikes[:,-1]}, {np.diff( self.strikes[:,-1] )}")

        verify_inp( self.vol_interp in {"sqrt_lin_vol","lin_var"}, f"vol_interp must be 'sqrt_lin_vol' or 'lin_var', found '{self.vol_interp}'" )

        self.expiries = self.expiries.astype(np.float32)

        # dynamic
        self._xstrikes = None

        # strikes and density at time 0
        # insert '1' and set the density accordingly
        t0             = min( self.expiries[0]/2., 1./255./10.)
        N              = self.strikes.shape[1]
        self._strikes0 = (self.strikes[0,:]-1.)*math.exp( -(self.expiries[0]-math.sqrt(t0)) )+1.
        ix_1 = min(N-2,max(1,np.abs(self._strikes0 - 1.).argmin()))
        assert self._strikes0[ix_1-1] < 1. < self._strikes0[ix_1+1], f"Internal error: strikes should be around 1??? found {self._strikes0[ix_1-1]} and {self._strikes0[ix_1+1]}. Contact the authors ... or propose a fix"
        self._density0 = np.zeros_like( self._strikes0 )
        self._strikes0[ix_1] = 1.
        self._density0[ix_1] = 1.

    @property
    def T(self) -> int:
        """ Number of expiries, i.e. use ``T = geometry.T`` to unpack. """
        return len(self.expiries)
    
    @property
    def N(self) -> int:
        """ Number of strikes, i.e. use ``N = geometry.N`` to unpack. """
        return self.strikes.shape[-1]

    @property
    def TN(self) -> tuple[int,int]:
        """ Number of expiries and strikes, i.e. use ``T, N = geometry.TN`` to unpack. """
        return self.strikes.shape

    @property
    def times(self) -> np.ndarray:
        return self.expiries

    @property
    def dts(self) -> np.ndarray:
        """ Time steps of the grid, in years. """
        dts = np.diff( self.expiries, prepend=np.float32(0.) )
        assert np.min(dts) > 0., f"Time steps must be positive, found min {np.min(dts)}"
        return dts
    
    @property
    def dsqrtTs(self) -> np.ndarray:
        dsqrtTs = np.diff( np.sqrt(self.expiries), prepend=np.float32(0.) )
        assert np.min(dsqrtTs) > 0., f"Time steps must be positive, found min {np.min(dsqrtTs)}"
        return dsqrtTs

    @property
    def xtimes(self) -> np.ndarray:
        """ Times of the grid, in years, including leading 0. The returned vector has dimension (T+1,) """
        x = np.zeros( (len(self.expiries)+1,), dtype=self.expiries.dtype )
        x[1:] = self.expiries
        return x
    
    @property
    def strikes0(self) -> np.ndarray:
        """ Model strikes of the grid at time 0, dimension (N,) """
        return self._strikes0
    
    @property
    def density0(self) -> np.ndarray:
        """ Density of the grid at time 0, dimension (N,). If N is odd, this will be a unimodal density centered at the middle strike; if N is even, this will be a bimodal density with modes at the two middle strikes. """
        return self._density0

    @property
    def xstrikes(self) -> np.ndarray:
        """ Extended strikes of the grid, dimension (T+1,N+2) """
        if self._xstrikes is None:
            T, N = self.TN
            strikes     = self.strikes
            assert np.all( strikes[-1,-1] >= strikes ), f"Internal error: strikes must be non-decreasing across time, found final strike {strikes[-1,-1]} and strikes {strikes}"

            self._xstrikes = np.zeros((T+1,N+2), dtype=np.float32)
            self._xstrikes[1:,1:-1]      = strikes
            self._xstrikes[0,1:-1]       = self.strikes0
            self._xstrikes[:,0]          = 0.
            self._xstrikes[:,-1]         = 2.*strikes[-1,-1]
        return self._xstrikes

    @staticmethod
    def create( *, initial_t : float = 1./DAYS_PER_YEAR,
                   horizon_t : float = 5., 
                   num_times : int = 11,
                   nms_range : float | tuple[float,float] = 1.,
                   num_nms   : int = 11,
                   vol_interp: str = "sqrt_lin_vol",
                   nms_eps: float = DEFAULT_NMS_EPS
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
        nms_eps : float, default :attr:`dys.grid_sanos.DEFAULT_NMS_EPS`
            Small epsilon for NMS computation. Default is 1./255./8.

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

        nms_lo = float(-nms_range if isinstance(nms_range, float) else nms_range[0])
        nms_hi = float(+nms_range if isinstance(nms_range, float) else nms_range[1])
        return GridGeometry(
            times  = np.exp( np.linspace(math.log(initial_t), math.log(horizon_t), num_times) ).astype(np.float32),
            nms    = np.linspace( nms_lo, nms_hi, num_nms, dtype=np.float32 ),
            vol_interp = vol_interp,
            nms_eps= nms_eps
        )
    
    def transition_operators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r""" Transition operators for the grid, dimension (T-1,N,N) """
        T, N    = self.TN
        strikes = self.strikes.astype(np.float64) # we compute the transition operators in higher precision to avoid numerical issues, then cast back to float32 at the end

        # Omega in https://arxiv.org/pdf/2601.11209 in Remark 3.5
        wx = (strikes[:,1:-1]**2) / (strikes[:,2:] - strikes[:,:-2])
        wn = wx / (strikes[:,1:-1] - strikes[:,:-2])
        wp = wx / (strikes[:,2:] - strikes[:,1:-1])
        del wx
        assert wn.shape == (T,N-2) and wp.shape == (T,N-2), f"wn and wp must have shape {(T,N-2)}, got {wn.shape} and {wp.shape}"
        assert np.min(wn) >= 0. and np.min(wp) >= 0., f"Transition weights must be non-negative, found min {np.min(wn)} and {np.min(wp)}"

        # interpolation matrx IL
        # https://arxiv.org/pdf/2601.11209 (22)
        # we compute this in higher precision
        xstrikes = self.xstrikes.astype(np.float64)

        xstrikes_t = xstrikes[1:,:]
        xstrikes_tm1 = xstrikes[:-1,:]

        divup = ( xstrikes_t[:,2:,None] - xstrikes_t[:,1:-1,None] ).astype(np.float64)
        divdn = ( xstrikes_t[:,1:-1,None] - xstrikes_t[:, :-2,None] ).astype(np.float64)
        assert np.all( divup > 0. ), f"Internal error: divup must be positive, found min {np.min(divup)}"
        assert np.all( divdn > 0. ), f"Internal error: divdn must be positive, found min {np.min(divdn)}"

        LUp = ( np.maximum( 0., xstrikes_tm1[:,None,1:-1] - xstrikes_t[:,2:,None] ) - 
                np.maximum( 0., xstrikes_tm1[:,None,1:-1] - xstrikes_t[:,1:-1,None] ) ) /\
              ( xstrikes_t[:,2:,None] - xstrikes_t[:,1:-1,None] )
        LDn = ( np.maximum( 0., xstrikes_tm1[:,None,1:-1] - xstrikes_t[:,1:-1,None] ) - 
                np.maximum( 0., xstrikes_tm1[:,None,1:-1] - xstrikes_t[:,:-2,None] ) ) /\
              ( xstrikes_t[:,1:-1,None] - xstrikes_t[:,:-2,None] )
        L = LUp - LDn
        assert L.shape == (T,N,N), f"Internal error: L must have shape {(T,N,N)}, got {L.shape}"       
        assert np.min(L) >= 0. and np.max(L) <= 1., f"Transition operators must be between 0 and 1; found min {np.min(L)} and max {np.max(L)}"

        # DEBUG
        if __debug__:
            for t in range(T):
                E1   = np.sum(L[t,:,:], axis=0 )
                assert np.max( np.abs( E1-1. ) ) < 1e-6, f"Error preserving strikes at time {t}: {np.max( np.abs( E1 - 1. ) )}"
                Kj   = xstrikes[t+1,1:-1]
                Kjm1 = xstrikes[t,1:-1]
                K_1  = Kj @ L[t,:,:]
                assert np.max( np.abs( Kjm1 - K_1 ) ) < 1e-6, f"Error preserving strikes at time {t}: {np.max( np.abs( Kjm1 - K_1 ) )}"

            test_d1     = np.random.normal( size=(N,) )**2
            test_d1     /= np.sum(test_d1)
            strikes_tm1 = self.strikes0
            for t in range(T):
                strikes_t = strikes[t,:]
                test_d2 = L[t,:,:] @ test_d1
                calls2  = np.maximum( strikes_t[None,:] - strikes_t[:,None], 0. ) @ test_d2
                calls1  = np.maximum( strikes_tm1[None,:] - strikes_t[:,None], 0. ) @ test_d1
                assert np.max( np.abs( calls1 - calls2 ) ) < 1e-6, f"Error preserving call prices at time {t}: {np.max( np.abs( calls1 - calls2 ) )}"
                strikes_tm1 = strikes_t

        del xstrikes
        
        wp = wp.astype(np.float32)
        wn = wn.astype(np.float32)
        L = L.astype(np.float32)
        return wn, wp, L
    
    def time_interpolation_weights( self, opt_tte : np.ndarray, opt_mask : np.ndarray | None = None ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes option time interpolation weights for variance and densities.

        * ``opt_lix``: per option the lower index into ``self.xtimes`` such that such that ``self.xtimes[opt_lix-1] < opt_tte <= self.expiries[opt_lix]``. Note that ``0<opt_lix<len(self.xtimes)``.
        * ``opt_dweight``: per option the lower interpolation weight for density interpolation.
        * ``opt_vweight``: per option the lower interpolation weight for variance interpolation (can be linear in variance or in sqrt time).
        * ``opt_vextra``: per option extrapolation with final volatility (linear or sqrt)

        Call price BS sqrt-variance is computed as follows in ``sqrt_lin_vol`` interpolation mode:

            * if ``t_i < xtimes[0]`` then ``opt_lix_i = 0`` and 
                ``sqrtVar_i = (1-opt_vweight_i) * sqrtVars[opt_lix_i]``.

            * if ``times[0] <= t_i <= xtimes[-1]`` then ``0 < opt_lix_i <= T-1`` and
                ``sqrtVar_i = opt_vweight_i * sqrtVars[opt_lix_i-1] + (1-opt_vweight_i) * sqrtVars[opt_lix_i]``.

            * if ``t_i > xtimes[-1]`` then we set ``opt_lix_i == T-1`` and
                ``sqrtVar_i = (1 - opt_vweight_i) * sqrtVars[T-1] + opt_vextra_i * vol_extra`` where ``opt_vweight_i = 0``.

        Once set, we also interpolate in model densities.

            * if ``t_i < xtimes[0]`` then ``opt_lix_i = 0`` and
                ``call_i = (1-opt_dweight_i) * call[opt_lix_i]`` with ``opt_dweight_i = 0``.

            * if ``times[0] <= t_i <= xtimes[-1]`` then ``0 < opt_lix_i <= T-1`` and
                ``call_i = opt_dweight_i * call[opt_lix_i-1] + (1-opt_dweight_i) * call[opt_lix_i]``.

            * if ``t_i > xtimes[-1]`` then we set ``opt_lix_i == T-1`` and
                ``call_i = (1 - opt_dweight_i) * call[T-1]`` with ``opt_dweight_i = 0``.

        Note that ``0<opt_lix<len(self.xtimes)``.
        """
        T        = self.T
        shape    = opt_tte.shape

        # we use the mask to set opt_tte to 1; from thereon proceed as usual
        opt_mask = opt_mask if ( not opt_mask is None and not np.all(opt_mask) ) else None
        opt_tte  = np.where( opt_mask, opt_tte, 1. ) if opt_mask is not None else opt_tte 
        opt_tte  = opt_tte.astype(np.float64)
        assert np.min( opt_tte ) > 0., f"Time to expiry must be positive, found min {np.min(opt_tte)}"
        opt_tte  = opt_tte.flatten()
        M        = len(opt_tte)

        # find indices ixtte such that xtimes[ixtte-1] < opt_tte <= xtimes[ixtte+1]
        ttes     = np.unique( opt_tte )
        sqrtTtes = np.sqrt( ttes )
        times    = self.expiries.astype(np.float64)
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
        opt_dweight = np.zeros((M,), dtype=np.float32 )
        opt_vweight = np.zeros((M,), dtype=np.float32 )
        opt_vextra  = np.zeros((M,), dtype=np.float32 )

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

