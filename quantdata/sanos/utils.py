"""
Basic utilities for SANOS
"""

import numpy as np
from cdxcore.err import verify, verify_inp
from quantdata.utils.backbone import Backbone, BackboneNumpy
from dataclasses import dataclass
import math
import warnings


@dataclass(frozen=True)
class GeometryBase:
    """
    SANOS geometry for different number of strikes per expiry.
    The open code base for now only supports single sample numpy mode.
    """

    @property
    def has_grid(self) -> bool:
        """ Returns True if the geometry has a grid, i.e. if the strikes are given as a 2D array with the same number of rows as expiries. Returns False if the strikes are given as a list of 1D arrays. """
        raise NotImplementedError("has_grid property must be implemented in subclass")

    @property
    def T(self) -> int:
        """ Returns the number if non-zero expiries"""
        return self.xexpiries.shape[0]-1

    @property
    def strikes(self) -> list[np.ndarray]:
        """ Returns the real model strikes excluding the zero expiry and the boundary strikes. """
        raise NotImplementedError("strikes property must be implemented in subclass")

    @property
    def expiries(self) -> np.ndarray:
        """ Returns the model expiries excluding the zero expiry """
        return self.xexpiries[1:]

    @property
    def density0(self) -> np.ndarray:
        """ Returns the initial density """
        raise NotImplementedError("density0 property must be implemented in subclass")
    
    def _make_density0(self):
        """ Make dnsity at time 0 """
        strikes0       = self.xstrikes[0] if not self.has_grid else self.strikes[0,:]
        N1             = len(strikes0)
        assert strikes0[0] < 1. < strikes0[-1], f"Internal error: strikes should include 1, but found {strikes0}."
        ix_1           = min(N1-1,max(0,np.abs(strikes0 - 1.).argmin()))
        strikes0[ix_1] = 1.
        density0       = np.zeros_like( strikes0 )
        density0[ix_1] = 1.
        self.__dict__['_density0'] = density0

@dataclass(frozen=True)
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
        object.__setattr__(self, 'prices', eps)
        object.__setattr__(self, 'theta', eps)
        object.__setattr__(self, 'gamma', eps)
        object.__setattr__(self, 'densities', eps)
        object.__setattr__(self, 'dlvsq', eps)

    @staticmethod
    def create( eps : float ) -> 'Eps':
        """ Create an Eps object with all tolerances set to the same value. """
        verify_inp( eps >= 0., f"Tolerance must be non-negative, found {eps}" )
        return Eps( prices=eps, theta=eps, gamma=eps, densities=eps, dlvsq=eps )

def vec_max_dx( strikes : np.ndarray, *, min_k : float, max_k : float, max_dx : float|None = None, min_dx : float = 0. ) -> tuple:
    """
    Generate model strikes.
    
    This function returns a vector of strikes such that:
        
    * Lowest strike is ``min_k``
    * Highest strike is ``max_k``
    * Strikes are at least ``min_dx`` apart from each other. Strikes which violate this are dropped iteratively from the front.
    * Strikes in the original strike range are at most ``max_dx`` apart from each other. If None, do not add strikes except min_k or max_k. Outside the extreme strikes of ``strikes``` 
      the function adds strikes with ``10 max_dx``.
    
    The function returns a tuple, the first element is the new strike vector.
    The second element represents the index positions of the ``strikes`` in the new vector. An index is -1 if an element was dropped due to ``min_k``.
    The function returns ``slice(1,-1)`` if
    only min_k, max_k were added ie the returned strikes are [min_k, strike[0], ..., strike[-1], max_k].    
    """
    verify_inp( min_k > 0. and min_k < 1. and max_k > 1., "'min_k' must be in (0,1) and 'max_k' must be above 1" )
    verify_inp( min_k < strikes[0] and max_k > strikes[-1], "'min_k' and 'max_k' must be outside 'strikes'")
    verify_inp( max_dx is None or max_dx >= 0., "'max_dx' must be None or non-negative")
    verify_inp( min_dx >= 0., "'min_dx' must be non-negative")

    if max_dx is None:
        if np.max( strikes[1:]-strikes[:-1] ) >= min_dx:
            out = np.empty( (len(strikes)+2,), dtype=strikes.dtype )
            out[1:-1] = strikes
            out[0]    = min_k
            out[-1]   = max_k
            return out, slice(1,-1)
        
        out = [ min_k ]
        six = []
        for x in strikes[1:]:
            dx = x - out[-1]
            if dx <= min_dx:
                six.append(-1)
                continue
            else:
                six.append( len(out ))
                out.append( x )
        return np.asarray( out, dtype=strikes.dtype ), np.asarray( six, dtype=np.int32 )        
            
    # max_dx
    # ------

    outer_max_dx = min(0.25,max_dx*10.) 
    inner_max_dx = max_dx

    left    = strikes[0]-inner_max_dx
    out     = np.linspace( min_k, left, max(2,1+math.floor( (left - min_k)/outer_max_dx )),dtype=strikes.dtype).tolist()\
              if left > min_k else\
              [ min_k ]    
    six     = [ len(out) ]
    out.append( strikes[0] )
    
    for x in strikes[1:]:
        dx = x - out[-1]
        if dx <= min_dx:
            six.append(-1)
            continue
        if dx > inner_max_dx:
            n = max(1,int(dx/inner_max_dx))+2
            r = np.linspace( out[-1], x, n )
            for x in r[1:]:
                out.append(x)
            six.append( len(out)-1 )
        else:
            six.append( len(out ))
            out.append( x )

    right = strikes[-1]+inner_max_dx
    rout  = np.linspace( right, max_k, max(2,1+math.floor( (max_k-right)/outer_max_dx )),dtype=strikes.dtype).tolist()\
            if right < max_k else\
            [ max_k ]
   
    out += rout
    out  = np.array( out, dtype=strikes.dtype )
    six  = np.asarray( six, dtype=np.int32 ) if len(out) > len(strikes)+2 else slice(1,-1)
    assert np.all( out[six] == strikes ), ("Internal error", out[six][:20], strikes[:20] )
    return np.asarray( out, dtype=strikes.dtype ), six     

# --------------------------------------------------------------------------------------------
# Shared utilities for Grid and Open SANOS. No real external use, hence made protected
# --------------------------------------------------------------------------------------------

def _time_interpolation_weights(*, vol_interp : str, 
                                expiries   : np.ndarray,
                                opt_tte    : np.ndarray, 
                                opt_mask   : np.ndarray|None = None,
                                dtype      : np.dtype|None = None ) -> dict[str, np.ndarray]:
    """
    Computes option time interpolation weights for variance and densities.
    These do not depend on market data and should be computed ahead of time.

    * ``lix``: per option the lower index into ``xexpiries`` such that such that ``xexpiries[opt_lix-1] < opt_tte <= expiries[opt_lix]``. Note that ``0<opt_lix<len(xexpiries)``.
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
    T        = expiries.shape[0]
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
    times    = np.asarray(expiries, dtype=np.float64)
    sqrtT    = np.sqrt( times )
    ixtte    = np.searchsorted( times, ttes, side='left' )
    assert np.min(ixtte) >= 0, f"Internal error - times[0]>0, so searchsorted must return indices of at least 0, found min {np.min(ixtte)}. Please contact authors."
    assert np.max(ixtte) <= T, f"Internal error - searchsorted must return indices of at most len(times), found max {np.max(ixtte)}. Please contact authors."
    assert ixtte.shape == ttes.shape, f"Expected ixtte to have shape {ttes.shape}, got {ixtte.shape}"

    # compute linear and sqrt-linear weights
    assert vol_interp in {"lin_var","lin_sqrt_time"}, f"Internal error: vol_interp must be 'lin_var' or 'lin_sqrt_time', found '{vol_interp}'"

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
    
    if vol_interp == "lin_sqrt_time":
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

def _interpolate_sqrtVars(*,bb         : Backbone,
                            xexpiries  : Backbone.Array,
                            vol_interp : str,
                            fwdvol_min : float,
                            fwd_vols   : Backbone.Array|float|None,
                            opt_interp : dict[str, Backbone.Array], 
                            opt_mask   : Backbone.Array|None,
                            debug      : bool ) -> Backbone.Array|None:
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

    T           = len(xexpiries)-1
    opt_lix     = opt_interp['lix']
    opt_vweight = opt_interp['vweight']
    opt_vextra  = opt_interp['vextra']
    M           = opt_lix.shape[-1]
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

    target_shape = head + (T,)
    assert bb.are_broadcastable(fwd_vols.shape, target_shape), f"Internal error: fwd_vols must broadcast to {target_shape}, found {fwd_vols.shape}"
    if fwd_vols.shape != target_shape:
        fwd_vols = bb.broadcast_to(fwd_vols, target_shape)

    dts = bb.diff(xexpiries)

    if vol_interp == "lin_var":
        r"""
        "sqrtVar" is computed by linear interpolation in variance, and taking the square rtoot: assuming that $t_j<t\leq t_{j+1}$ then:
        $$
            \mathrm{sqrt_var}(t) := \sqrt{ \mathrm{var}(t) } \ : \ \mathrm{var}(t) =  \sigma^2_j \left( t - t_j \right ) + \sum_{t_j<t} \sigma^2_j \left( t_j - t_{j-1} \right) +
        $$
        """
        assert vol_interp == "lin_var", f"Internal error: vol_interp must be 'lin_sqrt_time' or 'lin_var', found '{vol_interp}'"
        dts = bb.reshape(dts, hext + dts.shape)
        totalVar = bb.cumsum((fwd_vols + fwdvol_min) ** 2 * dts, axis=-1)
        assert totalVar.shape[-1] == T, f"Internal error: totalVar must have shape ending in {T}, found {totalVar.shape}"
        if debug: verify(bb.all(np.isfinite(totalVar)) & np.all(totalVar >= 0.), "Negative or infinite totalVar")

        prev_var = bb.where(opt_lix == 0, 0., bb.take_along_axis(totalVar, bb.clip(opt_lix-1, min=0), axis=-1))
        opt_sqrtVars = bb.sqrt(opt_vweight * prev_var + (1. - opt_vweight) * bb.take_along_axis(totalVar, opt_lix, axis=-1) + opt_vextra * ((fwd_vols[..., -1:] + fwdvol_min) ** 2))

    else:
        r"""
        Linear in square-root time: assumig that $t_j<t\leq t_{j+1}$ then:
        $$
            \mathrm{sqrt_var}(t) := \sqrt{ \mathrm{var}(t) } \ : \ \mathrm{var}(t) =  \sigma^2_j \left( t - t_j \right ) + \sum_{t_j<t} \sigma^2_j \left( t_j - t_{j-1} \right) 
        $$
        """
        verify_inp( vol_interp == "lin_sqrt_time", f"Internal error: vol_interp must be 'lin_sqrt_time' or 'lin_var', found '{vol_interp}'" )
        dsqrtTs = bb.diff(bb.sqrt(xexpiries))
        dsqrtTs = bb.reshape(dsqrtTs, hext + dsqrtTs.shape)
        totalSqrtVar = bb.cumsum((fwd_vols + fwdvol_min) * dsqrtTs, axis=-1)
        assert totalSqrtVar.shape[-1] == T, f"Internal error: totalSqrtVar must have shape ending in {T}, found {totalSqrtVar.shape}"
        if debug: verify(bb.all(np.isfinite(totalSqrtVar)) & np.all(totalSqrtVar >= 0.), "Negative or infinite sqrtVars")

        prev_sqrtVar = bb.where(opt_lix == 0, 0., bb.take_along_axis(totalSqrtVar, bb.clip(opt_lix-1, min=0), axis=-1))
        opt_sqrtVars = opt_vweight * prev_sqrtVar + (1. - opt_vweight) * bb.take_along_axis(totalSqrtVar, opt_lix, axis=-1) + opt_vextra * (fwd_vols[..., -1:] + fwdvol_min)

    assert opt_sqrtVars.shape[-1] == M, f"Internal error: opt_sqrtVars must have shape ending in {M}, found {opt_sqrtVars.shape}"
    if debug: verify(bb.all(np.isfinite(opt_sqrtVars)) & np.all(opt_sqrtVars >= 0.), "Negative or infinite opt_sqrtVars")
    assert len(opt_sqrtVars.shape) == len(opt_lix.shape), f"opt_sqrtVars and opt_lix must have the same number of dimensions, found {len(opt_sqrtVars.shape)} and {len(opt_lix.shape)}"
    return opt_sqrtVars if opt_mask is None else bb.where( opt_mask, opt_sqrtVars, 0. )


def _iterative_model_price(*,
        opt_tte     : np.ndarray,
        opt_strikes : np.ndarray,
        opt_sqrtVars: np.ndarray|None,
        opt_lix     : np.ndarray,
        opt_dweight : np.ndarray,
        strikes     : list[np.ndarray],
        densities   : list[np.ndarray],
        no_warnings : bool,
        debug       : bool,
        return_array: bool,
        check_finite: bool
    ):
    """
    Prices SANOS option prices by iterating over the unique option expiries and performing the interpolation in strikes and time as described in the documentation of :meth:`time_interpolation_weights`.
    For the torch/batch a more efficient version is implemented in :meth:`grid.GridSanos.price`.
    """

    bbnp = BackboneNumpy()

    def call_prices( *, spots, strikes, sqrtVars ):
        if sqrtVars is None:
            return np.maximum( spots - strikes, strikes.dtype.type(0.) )
        else:
            return bbnp.bs_price(spot=spots,
                                    strikes=strikes,
                                    sqrtVars=sqrtVars,
                                    high_precision = True,
                                    debug_verify_bounds = debug
                            )

    ulixs        = np.unique(opt_lix)
    fitted       = []

    for uix in ulixs:
        uix          = int(uix)
        umask        = opt_lix == uix
        uM           = np.sum(umask)
        tty          = opt_tte[umask][0]
        
        # compute call prices
        eff_lix     = max(uix-1, 0)
        prev_mstrk  = strikes[eff_lix]
        next_mstrk  = strikes[uix]
        upot_strks  = opt_strikes[umask]
        Nt          = len(next_mstrk)  
        Ntm1        = len(prev_mstrk)
        dens_t      = densities[uix]
        dens_tm1    = densities[eff_lix]

        assert dens_t.shape == (Nt,), f"Expected densities[uix] to have shape {(Nt,)}, got {dens_t.shape}"
        assert dens_tm1.shape == (Ntm1,), f"Expected densities[eff_lix] to have shape {(Ntm1,)}, got {dens_tm1.shape}"
 
        if not no_warnings:
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
        assert calls_prev.shape == (uM,Ntm1), f"Expected calls_prev to have shape {(uM,Ntm1)}, got {calls_prev.shape}"
        call_prev   = calls_prev @ dens_tm1
        assert call_prev.shape == (uM,), f"Expected call_prev to have shape {(uM,)}, got {call_prev.shape}"

        calls_next  = call_prices( spots=next_mstrk[None,:],
                                    strikes=upot_strks[:,None], 
                                    sqrtVars=opt_sqrtVars[umask,None] if opt_sqrtVars is not None else None
                                ) * ( 1.- opt_dweight[umask,None] )
        assert calls_next.shape == (uM,Nt), f"Expected calls_next to have shape {(uM,Nt)}, got {calls_next.shape}"
        calls_next  = calls_next @ dens_t
        assert calls_next.shape == (uM,), f"Expected calls_next to have shape {(uM,)}, got {calls_next.shape}"
        fit_calls   = call_prev + calls_next
        if check_finite:
            assert np.all( np.isfinite(fit_calls) ), f"Expected fitted call prices to be finite, found min {np.min(fit_calls)} and max {np.max(fit_calls)}"
        fitted.append( ( fit_calls, umask ) )

    if not return_array:
        return fitted

    fit_calls    = np.zeros_like( opt_strikes )
    for fc, um in fitted:
        fit_calls[um] = fc
    return fit_calls
