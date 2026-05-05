"""
Open Sanos

SANOS geometry for different number of strikes per expiry.
The open code base for now only supports single sample numpy mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings as warnings

import numpy as np

from cdxcore.err import verify, verify_inp

from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.sanos.utils import Eps, _interpolate_sqrtVars, _iterative_model_price, _time_interpolation_weights, vec_max_dx, GeometryBase
from quantdata.utils.backbone import BackboneNumpy

_npbb = BackboneNumpy()

@dataclass(frozen=True)
class OpenGeometry(GeometryBase):
    """
    SANOS geometry for different number of strikes per expiry.
    The open code base for now only supports single sample numpy mode.
    """
    xexpiries  : np.ndarray            #: model expiries in years, starting from 0, of size ``(T+1,)`` where ``T`` is the number of non-zero expiries. The first expiry must be zero.
    xstrikes   : list[np.ndarray] | tuple[np.ndarray, ...]  #: strike vectors including boundary strikes, one vector per model expiry including time 0.
    dlv_min    : float = 0.01          #: minimum discrete local volatility (dlv) value.
    dlv_max    : float = 5.0           #: maximum discrete local volatility (dlv) value.
    fwdvol_min : float = 0.0           #: minimum backbone forward volatility value.
    vol_interp : str = "lin_var"       #: Term volatility interpolation method. Must be a 'lin_var' or 'lin_sqrt_time'

    def __post_init__(self):
        # Normalize inputs first so the remaining checks operate on a consistent layout.
        xexpiries = np.asarray(self.xexpiries)
        xstrikes = tuple(np.asarray(strikes_t, dtype=xexpiries.dtype) for strikes_t in self.xstrikes)

        verify_inp( self.dlv_min >= 0., lambda : f"'dlv_min' must be non-negative, but found {self.dlv_min:.4g}") 
        verify_inp( self.dlv_max > self.dlv_min, lambda : f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}" )
        verify_inp( self.fwdvol_min >= 0., lambda : f"'fwdvol_min' must be non-negative; found {self.fwdvol_min:.4g}" )
        verify_inp( self.vol_interp in {"lin_sqrt_time", "lin_var"}, lambda : f"'vol_interp' must be either 'lin_sqrt_time' or 'lin_var', but found {self.vol_interp}" )

        verify_inp( len(xexpiries.shape) == 1 and len(xexpiries) > 1, lambda : "'xexpiries' must be non-empty 1D array of at least two expiries, starting in 0" )
        verify_inp( xexpiries[0] == 0., lambda : f"First expiry must be zero; found {xexpiries[0]}" )
        verify_inp( np.min( np.diff(xexpiries) ) > 0., lambda : "'xexpiries' must be strictly increasing" )

        verify_inp( len(xstrikes) == len(xexpiries), lambda: f"Expected one strike vector per expiry, found {len(xstrikes)} strike vectors and {len(xexpiries)} expiries" )

        prev_low = None
        prev_high = None
        for t, xstrikes_t in enumerate(xstrikes):
            verify_inp( len(xstrikes_t.shape) == 1 and xstrikes_t.shape[0] >= 5, lambda: f"'xstrikes[{t}]' must have at least 5 points, found shape {xstrikes_t.shape}" )
            verify_inp( np.all( np.diff(xstrikes_t) > 0), lambda: f"'xstrikes[{t}]' must be strictly increasing, found {np.min(np.diff(xstrikes_t)):.4g}" )
            verify_inp( xstrikes_t[0] > 0., lambda: f"'xstrikes[{t}]' must be positive, found {xstrikes_t[0]:.4g}" )
            verify_inp( xstrikes_t[1] < 1. and xstrikes_t[-2] > 1., lambda: f"'xstrikes[{t}]' must include interior values below and above 1, found boundary values {xstrikes_t[:2]} and {xstrikes_t[-2:]}" )

            if prev_low is not None:
                verify_inp( prev_low >= xstrikes_t[0], lambda: f"The lowest 'xstrikes' must not increase in time, found {prev_low:.4g} and {xstrikes_t[0]:.4g}" )
                verify_inp( prev_high <= xstrikes_t[-1], lambda: f"The highest 'xstrikes' must not decrease in time, found {prev_high:.4g} and {xstrikes_t[-1]:.4g}" )

            prev_low = xstrikes_t[0]
            prev_high = xstrikes_t[-1]

        object.__setattr__(self, "xexpiries", xexpiries)
        object.__setattr__(self, "xstrikes", xstrikes)

        # Reuse the base-class helper so density0 is defined consistently with other geometries.
        GeometryBase._make_density0(self)

    @property
    def has_grid(self) -> bool:
        return False

    @property
    def strikes(self) -> list[np.ndarray]:
        """Return interior strikes for non-zero expiries."""
        return [strikes_t[1:-1] for strikes_t in self.xstrikes[1:]]

    @property
    def density0(self) -> np.ndarray:
        """Return the initial density on the interior strike vector at time 0."""
        return self._density0

    def time_interpolation_weights(
        self,
        opt_tte: np.ndarray,
        opt_mask: np.ndarray | None = None,
        dtype: np.dtype | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute option time interpolation weights for the open geometry."""
        return _time_interpolation_weights(vol_interp=self.vol_interp, expiries=self.expiries, opt_tte=opt_tte, opt_mask=opt_mask, dtype=dtype)

    def transition_operators(self, *, eps: float = 1e-8) -> dict[str, list[np.ndarray]]:
        """Compute per-expiry transition operators for the open strike layout."""
        dtype = self.xexpiries.dtype
        xstrikes = [np.asarray(strikes_t, dtype=np.float64) for strikes_t in self.xstrikes]

        wn: list[np.ndarray] = []
        wp: list[np.ndarray] = []
        for xstrikes_t in xstrikes[1:]:
            strikes_t = xstrikes_t[1:-1]
            num_strikes = len(strikes_t)
            wx_t = strikes_t[1:-1] ** 2 / (strikes_t[2:] - strikes_t[:-2])
            wn_t = wx_t / (strikes_t[1:-1] - strikes_t[:-2])
            wp_t = wx_t / (strikes_t[2:] - strikes_t[1:-1])
            assert wn_t.shape == (num_strikes - 2,), f"Internal error: wn_t must have shape {(num_strikes - 2,)}, found {wn_t.shape}"
            assert wp_t.shape == (num_strikes - 2,), f"Internal error: wp_t must have shape {(num_strikes - 2,)}, found {wp_t.shape}"
            wn.append(np.asarray(wn_t, dtype=dtype))
            wp.append(np.asarray(wp_t, dtype=dtype))

        interp: list[np.ndarray] = []
        for t in range(self.T):
            xstrikes_t = xstrikes[t + 1]
            strikes_tm1 = xstrikes[t][1:-1]
            num_strikes = len(xstrikes_t) - 2
            num_prev = len(strikes_tm1)

            lup_t = (
                np.maximum(0.0, strikes_tm1[None, :] - xstrikes_t[2:, None])
                - np.maximum(0.0, strikes_tm1[None, :] - xstrikes_t[1:-1, None])
            ) / (xstrikes_t[2:, None] - xstrikes_t[1:-1, None])
            ldn_t = (
                np.maximum(0.0, strikes_tm1[None, :] - xstrikes_t[1:-1, None])
                - np.maximum(0.0, strikes_tm1[None, :] - xstrikes_t[:-2, None])
            ) / (xstrikes_t[1:-1, None] - xstrikes_t[:-2, None])
            l_t = np.asarray(lup_t - ldn_t, dtype=dtype)
            assert l_t.shape == (num_strikes, num_prev), f"Internal error: L must have shape {(num_strikes, num_prev)}, got {l_t.shape}"
            assert np.min(l_t) >= -eps and np.max(l_t) - 1.0 <= eps, f"Transition operators must be between 0 and 1; found min {np.min(l_t)} and max {np.max(l_t)} [tolerance {eps:.4g}]"
            interp.append(l_t)

        return dict(wn=wn, wp=wp, L=interp)
    
@dataclass(frozen=True)
class OpenGeometryGenerator:
    """
    Utility class to create an :class:`OpenGeometry` from a list of market expiries and market pure strikes,
    with some heuristics to find the extrapolation strikes and control the density of strikes in the grid.
    This is the code used for the paper https://arxiv.org/pdf/2601.11209.
    """

    dlv_min: float = 0.01
    dlv_max: float = 5.0
    fwdvol_min: float = 0.0
    vol_interp: str = "lin_sqrt_time"
    max_dx: float = 0.5
    min_dx: float = 0.01
    max_dx_outer_fac: float = 4.
    lower_scale: float = 0.5
    upper_scale: float = 1.5
    handle_errors : str = "warn"

    def __post_init__(self):
        verify_inp(self.dlv_min >= 0.0, lambda: f"'dlv_min' must be non-negative, but found {self.dlv_min:.4g}")
        verify_inp(self.dlv_max > self.dlv_min, lambda: f"'dlv_max' must be greater than 'dlv_min'; found dlv_max {self.dlv_max:.4g} and dlv_min {self.dlv_min:.4g}")
        verify_inp(self.fwdvol_min >= 0.0, lambda: f"'fwdvol_min' must be non-negative; found {self.fwdvol_min:.4g}")
        verify_inp(self.vol_interp in {"lin_sqrt_time", "lin_var"}, lambda: f"'vol_interp' must be either 'lin_sqrt_time' or 'lin_var', but found {self.vol_interp}")
        verify_inp(self.lower_scale > 0.0, lambda: f"'lower_scale' must be positive, found {self.lower_scale:.4g}")
        verify_inp(self.upper_scale > 1.0, lambda: f"'upper_scale' must be greater than 1, found {self.upper_scale:.4g}")
        verify_inp(self.lower_scale < 1.0, lambda: f"'lower_scale' must be less than 1, found {self.lower_scale:.4g}")
        verify_inp(self.min_dx > 0.0, lambda: f"'min_dx' must be positive, found {self.min_dx:.4g}")
        verify_inp(self.max_dx is None or self.max_dx > 0.0, lambda: f"'max_dx' must be positive, found {self.max_dx:.4g}")
        verify_inp(self.handle_errors in ['ignore', 'warn', 'error'], lambda: f"'handle_errors' must be one of 'ignore', 'warn', or 'error', found {self.handle_errors}")

    def create_geometry( 
        self,
        expiries     : np.ndarray,
        pure_strikes : list[np.ndarray], *,
        dtype : np.dtype = np.float64            
    ):
        """
        Create the :class:`OpenGeometry` from market expiries and pure strikes,
        with some heuristics to find the extrapolation strikes and control the density of strikes in the grid.

        The time-0 strike vector reuses the first fitted strike grid so the initial density
        has a consistent support for the first transition.
        """
        expiries = np.asarray(expiries, dtype=dtype)
        verify_inp(len(expiries.shape) == 1 and len(expiries) > 0, lambda: f"'expiries' must be a non-empty 1D array, found shape {expiries.shape}")
        verify_inp(np.min(expiries) > 0.0, lambda: f"'expiries' must be positive, found min {np.min(expiries):.4g}")
        verify_inp(len(expiries) == 1 or np.min(np.diff(expiries)) > 0.0, lambda: "'expiries' must be strictly increasing")
        verify_inp(len(pure_strikes) == len(expiries), lambda: f"Expected one pure strike vector per expiry, found {len(pure_strikes)} strike vectors and {len(expiries)} expiries")

        T = len(expiries)
        pure_strikes = [np.asarray(strikes_t, dtype=dtype) for strikes_t in pure_strikes]

        min_k = dtype(1.0)
        max_k = dtype(1.0)

        issues = "Found the following market data issues: "
        has_issues = False

        # firsrt pass: check strikes

        for t in range(T):
            strikes_t = pure_strikes[t]
            verify_inp(len(strikes_t.shape) == 1, lambda: f"'pure_strikes[{t}]' must be a 1D array, found shape {strikes_t.shape}")

            verify_inp(0.0 < strikes_t[0], lambda: f"'pure_strikes[{t}]' first strike must be positive, found {strikes_t[0]:.4g}")
            verify_inp(np.min(np.diff(strikes_t)) > 0.0, lambda: f"'pure_strikes[{t}]' must be strictly increasing, found min diff {np.min(np.diff(strikes_t)):.4g}")

            if strikes_t[0] >= 1.:
                msg = f"'pure_strikes[{t}]' first strike must be below 1, found {strikes_t[0]:.4g}"
                if self.handle_errors == "error":
                    raise ValueError(msg)
                issues = issues + msg + "; "
                has_issues = True
                strikes_t = np.insert( strikes_t, 0, 1.-(strikes_t[0]-1.+0.001) )

            if strikes_t[-1] <= 1.:
                msg = f"'pure_strikes[{t}]' last strike must be above 1, found {strikes_t[-1]:.4g}"
                if self.handle_errors == "error":
                    raise ValueError(msg)
                issues = issues + msg + "; "
                has_issues = True
                strikes_t = np.append( strikes_t, 1.+(1.-strikes_t[-1]+0.001) )

            # make strike grid expanding
            if t > 0:
                strikes_tm1 = pure_strikes[t-1]
                if strikes_tm1[0] <= strikes_t[0]:
                    issues += f"'pure_strikes[{t}]' first strike {strikes_t[0]:.4g}  must be below the previous expiry's strike {strikes_tm1[0]:.4g}; "
                    strikes_t = np.insert( strikes_t, 0, strikes_tm1[0]*.99 )
                if strikes_tm1[-1] >= strikes_t[-1]:
                    issues += f"'pure_strikes[{t}]' last strike {strikes_t[-1]:.4g}  must be above the previous expiry's strike {strikes_tm1[-1]:.4g}; "
                    strikes_t = np.append( strikes_t, strikes_tm1[-1]*1.01 )

            assert np.min( np.diff(strikes_t)) > 0., "Internal error: strikes should be strictly increasing after adjustment"
            pure_strikes[t] = strikes_t
            min_k = min(min_k, strikes_t[0])
            max_k = max(max_k, strikes_t[-1])

        if has_issues:
            warnings.warn(issues[:-2] + ". (Strikes have been patched)")

        min_k = min(self.lower_scale, min_k * self.lower_scale)
        max_k = max(self.upper_scale, max_k * self.upper_scale)

        xstrikes = []
        for t, strikes_t in enumerate(pure_strikes):
            verify(min_k < strikes_t[0] and max_k > strikes_t[-1], lambda: f"Internal consistency error for strike vector {t}: expected {min_k:.4g} < {strikes_t[0]:.4g} and {max_k:.4g} > {strikes_t[-1]:.4g}")
            xstrikes_t = vec_max_dx(strikes_t, min_k=min_k, max_k=max_k, max_dx=self.max_dx, min_dx=self.min_dx, outer_fac=self.max_dx_outer_fac)

            if strikes_t[0] <= xstrikes_t[1]:
                xstrikes_t = np.insert( xstrikes_t, 1, np.mean(xstrikes_t[:2]))
            if strikes_t[-1] >= xstrikes_t[-1]:
                xstrikes_t = np.insert( xstrikes_t, -1, np.mean(xstrikes_t[-2:]))

            xstrikes.append(np.asarray(xstrikes_t, dtype=dtype))

        xexpiries = np.zeros((len(expiries) + 1,), dtype=dtype)
        xexpiries[1:] = expiries
        # Reuse the first non-zero strike vector at time 0. The initial density is defined
        # on its interior nodes and represents a point mass at strike 1.
        xstrikes = [np.copy(xstrikes[0])] + xstrikes

        return OpenGeometry(
            xexpiries=xexpiries,
            xstrikes=xstrikes,
            dlv_min=self.dlv_min,
            dlv_max=self.dlv_max,
            fwdvol_min=self.fwdvol_min,
            vol_interp=self.vol_interp,
        )

class OpenSanos(object):
    """Single-sample numpy SANOS pricer for the open geometry."""

    def __init__(
        self,
        geometry: OpenGeometry,
        eps: Eps = Eps(),
        debug: bool = False,
        no_warnings: bool = False,
    ):
        self.geometry = geometry
        self.eps = eps
        self.debug = debug
        self.no_warnings = no_warnings
        self.transp = self.geometry.transition_operators()

    def verify(self, condition: callable, message: callable | str):
        if self.debug:
            verify(condition(), message)

    @property
    def T(self) -> int:
        """Return the number of non-zero model expiries."""
        return self.geometry.T

    @property
    def xstrikes(self) -> tuple[np.ndarray, ...] | list[np.ndarray]:
        """Return the full strike vectors, including boundary strikes."""
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
        """Return interior strikes for non-zero expiries."""
        return self.geometry.strikes

    @property
    def expiries(self) -> np.ndarray:
        """Return non-zero model expiries."""
        return self.geometry.expiries

    def interpolate_sqrtVars(
        self,
        fwd_vols: np.ndarray | float | None,
        opt_interp: dict[str, np.ndarray],
        opt_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Interpolate option square-root variances from forward vols."""
        return _interpolate_sqrtVars(
            bb=_npbb,
            xexpiries=self.xexpiries,
            vol_interp=self.geometry.vol_interp,
            fwdvol_min=self.geometry.fwdvol_min,
            fwd_vols=fwd_vols,
            opt_interp=opt_interp,
            opt_mask=opt_mask,
            debug=self.debug,
        )

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
        M        = opt_strikes.shape[-1]
        verify_inp( len(opt_strikes.shape) == 1, lambda: f"'opt_strikes' must be a 1D array of shape (M,), found {opt_strikes.shape}" )
        verify_inp( opt_tte.shape == opt_strikes.shape, lambda: f"'opt_tte' must match 'opt_strikes', found {opt_tte.shape} and {opt_strikes.shape}" )
        verify_inp( opt_mask is None or opt_mask.shape == opt_strikes.shape, lambda: f"'opt_mask' must match 'opt_strikes', found {opt_mask.shape} and {opt_strikes.shape}" )
        verify_inp( opt_is_call is None or opt_is_call.shape == opt_strikes.shape, lambda: f"'opt_is_call' must match 'opt_strikes', found {opt_is_call.shape} and {opt_strikes.shape}" )
        verify_inp( len(densities) == self.T, lambda: f"Expected one density vector per non-zero expiry, found {len(densities)} densities and {self.T} expiries" )

        if opt_interp is None:
            opt_interp  = self.geometry.time_interpolation_weights( opt_tte=opt_tte, opt_mask=opt_mask, dtype=self.xexpiries.dtype )

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
            return_array = True,
            check_finite = True,
        )

        if opt_is_call is None:
            prices = fit_calls
        elif opt_mask is None:
            prices = np.where( opt_is_call, fit_calls, fit_calls + opt_strikes - 1. )
        else:
            prices = np.where( opt_is_call | ~opt_mask, fit_calls, fit_calls + opt_strikes - 1. )

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
        verify_inp( len(dlvsqs) == T, lambda: f"Expected one DLV vector per non-zero expiry, found {len(dlvsqs)} vectors and {T} expiries" )

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
            assert mdensity_tm1.shape == (Ntm1,), f"Expected mdensity_tm1 to have shape {(Ntm1,)}, found {mdensity_tm1.shape}"
            L_t          = L[t]
            assert L_t.shape == (Nt, Ntm1), f"Expected L_t to have shape {(Nt, Ntm1)}, found {L_t.shape}"

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
