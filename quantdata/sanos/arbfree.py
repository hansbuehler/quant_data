"""
Make arbfree markets
=====================

This module contains :class:`MkArbFree`, a utility class 
has two main modes:

* ``expiry``: fits per expiry an arbitrage-free mid price within the bid-ask spread,
  and drops any expiries for which this is not possible as it implies actionable arbitrage.

  This is a pre-processor for any benchmark which attempts to fit arbitrage-free models
  to otherwise arbitrageable data. It is also a useful tool to detect and handle arbitrage in the data, which is a common issue
  in WIDB data.

* ``surface``: fits a fully arbitrage free surface, on a strike grid which theoretically means
  that a mid-price fit exists if and only if the
  resulting fit is within bid/ask spread. 

  This function can become very slow.

The :meth:`MkArbFree.__call__` function returns an :class:`ArbFreePricer` class which contains the fit to the market and can also compute option
prices for arbitrary strikes and expiries using the fitted model.

*Developer note:*
A really diffult day for testing (likely very wrong data) is ``SPX 2005-11-21``.*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings as warnings

import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from dataclasses import dataclass, field
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import BackboneNumpy
import math

from quantdata.sanos import IMPOSE_FLOOR_ON_DENSITY
from quantdata.sanos.sanos_tools import SanosLPSolver, repair_density_nonnegative, warn_and_fix_density, _compute_market_fit

@version("0.81")
class ArbFreePricer(PrettyObject):
    """
    Arbitrage-Free Option pricer based on DLV/SANOS.

    This class contains the result of :meth:`MkArbFree.__call__`.
    This function call may drop expiries if they have arbitrage which cannot be resolved.
    To map the expiries stored in this class back to the input expiries, use the ``input_index`` field.

    The class attributes are:

    * ``expiries``: array ``(T,)`` of expiries used. This may be fewer expiries than provided as input.

    * ``input_index``: array of size ``(T,)`` which maps each returned expiry to the index of the input expiry it corresponds to. In other words ``output_expiries == [ input_expiries[j] for j in input_index ]``.

    * ``masks``: a list of ``T`` boolean masks by expiry indicating which strikes were considered during the no-arb fit.
        The mask for expory ``j`` has the same length ``Nj`` as the ``strikes[ input_expiries[j] ]``.

        The mask is False for strikes which were dropped because they had zero implied density in the per-expiry fit, and for strikes which were dropped because they had too steep bid-ask slopes at the boundaries.
        The fit minimizes ``max( 0, fit[mask] - ask[mask] ) + max( 0, bid[mask] - fit[mask] )``.

    * ``strikes``: a copy of the ``T`` input strike arrays for the expiries used.

    * ``bids``: a copy of the ``T`` arrays of input bids for the expiries used.

    * ``asks``: a copy of the ``T`` arrays of input asks for the expiries used.

    * ``fits1``: list of ``T`` per-expiry fitted arbitrage-free prices within bid/ask (where masked) for the input ``strikes`` used. The element ``fit1[j]`` is a vector of length ``strikes[j]``.
      Within numerical accuracy we have ``bids[j][mask[j]] <= fit1[j][mask[j]] <= asks[j][mask[j]]``.

    * ``fitsX``: list of ``T`` surface fitted arbitrage-free prices for the input ``strikes``. 
        These prices are globally arbitrage-free across the surface, but they are not necessarily within bid/ask spread; they are the closest fit
        among strikes with ``mask`` True.

        Up to machine precision existence
        of ``fitsX`` within bid/ask spread in the linear case ``smoothness=0`` is equivalent to absence of arbitrage.

        This field is None if ``no_arb_mode`` is "expiry".

    Besides the pure fits, this class also contains the ingredients of the DLV/SANOS model.

    * For the **"expiry"** ``no_arb_mode``
      the function :meth:`MkArbFree.__call__` will generate internal model ``xstrikes`` around the provided market strikes. For those strikes it will
      compute a ``xdensities`` per expiry.
        
      * ``xstrikes``: an extended strike list with two boundary strikes add. Boundary strikes are widening in time; c.f. https://arxiv.org/pdf/2601.11209 remark 2.7.

      * ``xdensities``: list of densities for ``xstrikes``, per expiry. Up to machine precision, each density is positive, sums to one, intergates to one 
         in the sense that `` xstrikes[t]@xdensities[t] == 1``
         and reprices ``fits``.

      * ``run_time_expiry``: the time taken for the per-expiry first, in seconds.

    * For the **"surface"** ``no_arb_mode``
      the function :meth:`MkArbFree.__call__` will generate internal model ``mstrikes`` around the provided market strikes which are a superset of the extended ``xstrike`` grid.
      This surface grid is constructed by adding strikes from the next expiry if they are not too close to existing strikes.
      This is typically a much denser set than the input strikes. Each expiry is guaranteed to have their ``xstrikes`` among the respective model strikes.

      These fields are None if ``no_arb_mode`` is not "surface".

      * ``mstrikes``: the list of internal model strikes per expiry used. 
 
      * ``mdensities```: the list of densities for ``mstrikes``. Up to machine precision, each density is positive, sums to one, intergates to one 
        in the sense that `` mstrikes[t]@mdensities[t] == 1``
        and reprices ``fits``. Moreover, the family of ``mdensities`` is martingale density, each later density evaluaes a convex function 
        higher than the a previous density.
        
      * ``xix_to_mix``: a list of arrays which gives for every ``xstrike[t]`` the position in ``mstrikes[t]``, i.e. ``mstrikes[t][xix_to_mix[t]] == xstrikes[t]``.
        The element ``xix_to_mix[t]`` is ``None`` if no additional strikes were added to expiry ``t``, i.e. the strikes coincide.
  
      * ``run_time_surface``: the time taken for the surface fit, in seconds.
    """

    def __init__(self, smoothness : float|None = None):
        self.smoothness  : float|None = smoothness
        self.expiry_linear: bool = False
        self.min_half_spread : float = 1E-4 
        self.expiries    : np.ndarray = []
        self.variances   : np.ndarray = []
        self.input_index : np.ndarray = []

        self.masks       : list[np.ndarray] = []
        self.strikes     : list[np.ndarray] = []
        self.asks        : list[np.ndarray] = []
        self.bids        : list[np.ndarray] = []
        self.fits1       : list[np.ndarray] = []
        self.fitsX       : list[np.ndarray]|None = None

        self.xstrikes    : list[np.ndarray] = []
        self.xdensities  : list[np.ndarray] = []

        self.xix_to_mix  : list[np.ndarray]|None = None
        self.mstrikes    : list[np.ndarray]|None = None
        self.mdensities  : list[np.ndarray]|None = None

        self.run_time_expiry  : float|None = None
        self.run_time_surface : float|None = None

    _eps_price = 1E-6
    _eps_density = 1E-7

    def price( self, expiries : float|np.ndarray, strikes : float|np.ndarray, *, and_vols: bool = False, mode : str|None = None ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Price options at ``strikes`` with expiries ``expiries``.

        Use ``mode`` to chose which pricing kernel to use: "expiry" uses the per-expiry fit
        which may not be free of arbitrage in time; "surface" uses the full surface
        fit which is free of arbitrage in time. ``None`` will chose the latter
        if available, otherwise the former.

        If ``and_vols`` is ``True`` this function returns call prices and their implied volatilities;
        if ``False``, it will return only call prices.

        Parameters
        ----------
        expiries : float | :class:`np.ndarray`
            The expiry or expiries to price at.
            
            * If ``expiries`` is a float, all strikes are evaluated at that expiry.
            * If ``expiries`` is a 1D array of shape ``(N,)``, then ``strikes`` must also be a 1D array of shape ``(N,)``, and the function will price each strike at their respective expiry.
            * If ``expiries`` is a 2D array of shape ``(M,1)``, then ``strikes`` must be a 2D array of shape ``(1,N)``, and the function will price each strike at each expiry, returning an array of shape ``(M,N)``.

        strikes : float | :class:`np.ndarray`
            The strikes to price at. The shape of this array must be compatible with the shape of ``expiries`` as described above.

        and_vols : bool, default ``False``
            Whether to return implied volatilities in addition to call prices. If ``True``, the function returns a tuple of (call prices, implied volatilities).
            If ``False``, it returns only call prices.

        mode : str | None, default ``None``
            Which pricing kernel to use: "expiry" uses the per-expiry fit which may not be free of arbitrage in time; "surface" uses the full surface fit which is free of arbitrage in time. ``None`` will chose the latter if available, otherwise the former.
        """

        assert isinstance(self.expiries, np.ndarray), "Internal error - forgot to make my 'expiries' a numpy array"
        verify_inp( mode is None or mode in ["surface","expiry"], lambda : f"Expected 'mode' to be one of 'surface', 'expiry' or None, but found {str(mode)}"   )
        per_expiry = mode=="expiry" if not mode is None else ( self.fitsX is None )
        verify_inp( per_expiry or not self.fitsX is None, f"Cannot use 'mode' \"surface\" since no surface fit is available")
        
        strikes = np.asarray(strikes) if not isinstance(strikes, (int,float,np.number)) else np.array([float(strikes)])

        def x_price( expiry, strikes, and_vols ):
            verify_inp( isinstance( expiry, (float, int, np.number)) and expiry>=0, lambda : f"Expected 'expiry' to be a non-negative number, but found {str(expiry)}")

            if self.smoothness is None or expiry<ONE_MINUTE or (self.expiry_linear and per_expiry):
                def call( spots, strikes ):
                    return np.maximum( 0., spots - strikes )
            else:
                vol = self.interpolate_atm_vols( expiries=expiry ) * math.sqrt(self.smoothness)
                def call( spots, strikes ):
                    return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )*spots

            def prices():
                if expiry<ONE_MINUTE:
                    return np.maximum( 0., 1 - strikes )

                ix = np.searchsorted( self.expiries, expiry )
                if ix==0:
                    strikes_0 = self.xstrikes[0] if per_expiry else self.mstrikes[0]
                    p_0       = self.xdensities[0] if per_expiry else self.mdensities[0]
                    return call( spots=strikes_0[None,:], strikes=strikes[:,None] ) @ p_0

                if ix==len(self.expiries):
                    strikes_T = self.xstrikes[-1] if per_expiry else self.mstrikes[-1]
                    p_T       = self.xdensities[-1]  if per_expiry else self.mdensities[-1]
                    return call( spots=strikes_T[None,:], strikes=strikes[:,None] ) @ p_T

                diff_ixm1 = abs( expiry - self.expiries[ix-1] )
                diff_ix   = abs( expiry - self.expiries[ix] )
                diff      = diff_ix if diff_ix < diff_ixm1 else diff_ixm1

                if diff<ONE_MINUTE:
                    ix_mind    = ix if diff_ix < diff_ixm1 else ix-1
                    strikes_ix = self.xstrikes[ix_mind] if per_expiry else self.mstrikes[ix_mind]
                    p_ix       = self.xdensities[ix_mind]  if per_expiry else self.mdensities[ix_mind]
                    return call( spots=strikes_ix[None,:], strikes=strikes[:,None] ) @ p_ix

                strikes_ixm1 = self.xstrikes[ix-1] if per_expiry else self.mstrikes[ix-1]
                p_ixm1       = self.xdensities[ix-1]  if per_expiry else self.mdensities[ix-1]
                c_ixm1       = call( spots=strikes_ixm1[None,:], strikes=strikes[:,None] ) @ p_ixm1

                strikes_ix   = self.xstrikes[ix] if per_expiry else self.mstrikes[ix]
                p_ix         = self.xdensities[ix]  if per_expiry else self.mdensities[ix]
                c_ix         = call( spots=strikes_ix[None,:], strikes=strikes[:,None] ) @ p_ix

                return ((expiry - self.expiries[ix-1])*c_ix + (self.expiries[ix]-expiry)*c_ixm1)/(self.expiries[ix]-self.expiries[ix-1])
            c = prices()
            if not and_vols:
                return c

            v = bs.implied( k=strikes, prices=c, sqrtT=math.sqrt(expiry), eps=self._eps_price )
            return c,v 

        if hasattr(expiries,"shape"):
            if len(expiries.shape)==0:
                expiries = float( expiries.item() )
            elif expiries.shape == (1,):
                expiries = float( expiries[0] )

        if isinstance(expiries, (float, int, np.number)):
            return x_price( expiry=expiries, strikes=strikes, and_vols=and_vols )

        expiries = np.asarray(expiries)
 
        if len(expiries.shape) == 1:
            verify_inp( expiries.shape == strikes.shape, lambda : f"If 'expiries' is a 1D vector, then 'strikes' must have the same shape, but found {expiries.shape} and {strikes.shape}" )

            uttys   = np.unique(expiries)
            results = np.zeros_like(strikes)
            vols    = np.zeros_like(strikes) if and_vols else None
            for utty in uttys:
                mask = expiries==utty
                if not and_vols:
                    results[mask] = x_price( expiry=utty, strikes=strikes[mask], and_vols=False )
                else:
                    results[mask], vols[mask] = x_price( expiry=utty, strikes=strikes[mask], and_vols=True )
            if not and_vols:
                return results
            return results, vols
        
        if strikes.shape == (1,):
            strikes = strikes.reshape(1,1)
        verify_inp( len(expiries.shape) == 2 and expiries.shape[1] == 1, lambda : f"'expiries' can be a vector or a matrix if size '(M,1)'. Found shape {expiries.shape}" )
        M = expiries.shape[0]
        verify_inp( len(strikes.shape) == 2 and strikes.shape[0] in (M,1), lambda : f"If 'expiries' is a matrix of shape '(M,1)', then 'strikes' must be a matrix of shape '(M|1,N)'. Found expiries of shape {expiries.shape} and strikes of shape  {strikes.shape}" )
        N = strikes.shape[1]
        results = np.zeros((M,N),dtype=strikes.dtype)
        vols    = np.zeros((M,N),dtype=strikes.dtype) if and_vols else None
        for ix, tty in enumerate(expiries[:,0]):
            if not and_vols:
                results[ix,:] = x_price( expiry=tty, strikes=strikes[0 if strikes.shape[0] == 1 else ix,:], and_vols=False )
            else:
                results[ix,:], vols[ix,:] = x_price( expiry=tty, strikes=strikes[0 if strikes.shape[0] == 1 else ix,:], and_vols=True )
        if not and_vols:
            return results
        return results, vols

    def interpolate_atm_vols( self, expiries: float|np.ndarray ) -> float|np.ndarray:
        """ Interpolates the (ATM) variances (not smoothed) """
        assert not self.variances is None, "Internal error - variances are missing"
        is_float = not hasattr(expiries,"shape")
        expiries = np.asarray([float(expiries)]) if is_float else np.asarray(expiries)
        expiries = np.clip( expiries, self.expiries[0], self.expiries[-1] )
        vols     = np.sqrt( np.interp( expiries, self.expiries, self.variances ) / expiries )
        assert np.all(np.isfinite(vols)), "Internal error: infinite vol"
        return vols if not is_float else vols[0]

    def compute_market_fit( self, expiries : np.ndarray, strikes : list[np.ndarray], bids : list[np.ndarray], asks : list[np.ndarray], masks : list[np.ndarray]|None = None, *, and_vols : bool = True ) -> PrettyObject:
        """
        Compute standardized fitting error stats

        Returns
        -------
        Errors : :class:`PrettyObject`
            A dictionary-lik object with the following fields:
            * ``fits``: list of fitted prices for the input strikes, per expiry.
            * ``fit_vols``: list of fitted vols for the input strikes, per expiry.
            * ``fit_errs``: list of fitting errors for the input strikes, per expiry, where the error is defined as the distance to the bid/ask spread in units of half-spread, i.e. ``max( 0, fit - ask ) + max( 0, bid - fit) / max( 0.5*(ask-bid), self.min_half_spread )``.
            * ``pcnt_fit``: the percentage of strikes for which the fit is outside the bid/ask spread, i.e. ``sum( fit_errs > 0 ) / len(fit_errs)``.
            * ``median_fit_err``: the median fitting error among strikes for which the fit is outside the bid/ask spread, i.e. ``median( fit_errs[fit_errs>0.] )``.
            * ``mean_fit_err``: the mean fitting error among strikes for which the fit is outside the bid/ask spread, i.e. ``mean( fit_errs[fit_errs>0.] )``.

        """
        return _compute_market_fit( lambda expires, strikes, and_vols : self.price( expiries=expires, strikes=strikes, and_vols=and_vols ),
                                    self.min_half_spread, expiries=expiries, strikes=strikes, bids=bids, asks=asks, masks=masks, and_vols=and_vols )

    @property
    def T(self) -> int:
        return len(self.expiries)

    def _make_numpy(self):
        self.smoothness  = float(self.smoothness) if not self.smoothness is None else None
        self.expiries    = np.asarray(self.expiries)
        self.variances   = np.asarray(self.variances) 
        self.input_index = np.asarray(self.input_index, dtype=np.int32)

        self.masks       = [ np.asarray(x) for x in self.masks ]
        self.strikes     = [ np.asarray(x) for x in self.strikes ]
        self.asks        = [ np.asarray(x) for x in self.asks ]
        self.bids        = [ np.asarray(x) for x in self.bids ]
        self.fits1       = [ np.asarray(x) for x in self.fits1 ]
        self.fitsX       = [ np.asarray(x) for x in self.fitsX ] if not self.fitsX is None else None

        self.xstrikes    = [ np.asarray(x) for x in self.xstrikes ]
        self.xdensities  = [ np.asarray(x) for x in self.xdensities ] 

        self.xix_to_mix  = [ (np.asarray(x,dtype=np.int32) if not x is None else None) for x in self.xix_to_mix ] if not self.xix_to_mix is None else None
        self.mstrikes    = [ np.asarray(x) for x in self.mstrikes ] if not self.mstrikes is None else None
        self.mdensities  = [ np.asarray(x) for x in self.mdensities ] if not self.mdensities is None else None

        self.run_time_expiry = float(self.run_time_expiry)
        self.run_time_surface = float(self.run_time_surface) if not self.run_time_surface is None else None

        T = len(self.expiries) 
        assert self.variances.shape == (T,), f"Internal error: 'variances' must be of shape (T,), but found {self.variances.shape}"
        assert self.input_index.shape == (T,), f"Internal error: 'input_index' must be of shape (T,), but found {self.input_index.shape}"    

        assert len(self.masks) == T, f"Internal error: 'masks' must be a list of length {T}, but found {len(self.masks)}"
        assert len(self.strikes) == T, f"Internal error: 'strikes' must be a list of length {T}, but found {len(self.strikes)}"
        assert len(self.asks) == T, f"Internal error: 'asks' must be a list of length {T}, but found {len(self.asks)}"
        assert len(self.bids) == T, f"Internal error: 'bids' must be a list of length {T}, but found {len(self.bids)}"
        assert len(self.fits1) == T, f"Internal error: 'fits1' must be a list of length {T}, but found {len(self.fits1)}"

        assert len(self.xstrikes) == T, f"Internal error: 'xstrikes' must be a list of length {T}, but found {len(self.xstrikes)}"
        assert len(self.xdensities) == T, f"Internal error: 'xdensities' must be a list of length {T}, but found {len(self.xdensities)}"

        if not self.mstrikes is None:
            assert self.fitsX is None or len(self.fitsX) == T, f"Internal error: 'fitsX' must be a list of length {T}, but found {len(self.fitsX)}"
            assert len(self.mstrikes) == T, f"Internal error: 'mstrikes' must be a list of length {T}, but found {len(self.mstrikes)}"
            assert len(self.mdensities) == T, f"Internal error: 'mdensities' must be a list of length {T}, but found {len(self.mdensities)}"

        for t in range(T):
            mask_t = self.masks[t]
            strikes_t = self.strikes[t]
            asks_t = self.asks[t]
            bids_t = self.bids[t]
            fits1_t = self.fits1[t]
            fitsX_t = self.fitsX[t] if not self.fitsX is None else None

            assert mask_t.shape == strikes_t.shape == asks_t.shape == bids_t.shape == fits1_t.shape, f"Internal error: 'masks[{t}]', 'strikes[{t}]', 'asks[{t}]', 'bids[{t}]' and 'fits1[{t}]' must all have the same shape, but found {mask_t.shape}, {strikes_t.shape}, {asks_t.shape}, {bids_t.shape} and {fits1_t.shape}"

            xstrikes_t = self.xstrikes[t]
            xdensities_t = self.xdensities[t]
            assert xstrikes_t.shape == xdensities_t.shape, f"Internal error: 'xstrikes[{t}]' and 'xdensities[{t}]' must have the same shape, but found {xstrikes_t.shape} and {xdensities_t.shape}"

            if not self.mstrikes is None:
                assert fitsX_t is None or fitsX_t.shape == mask_t.shape, f"Internal error: 'fitsX[{t}]' must have the same shape as 'masks[{t}]', but found {fitsX_t.shape} and {mask_t.shape}"
                mstrikes_t = self.mstrikes[t]
                mdensities_t = self.mdensities[t]
                assert mstrikes_t.shape == mdensities_t.shape, f"Internal error: 'mstrikes[{t}]' and 'mdensities[{t}]' must have the same shape, but found {mstrikes_t.shape} and {mdensities_t.shape}"

@version("0.51", dependencies=[ArbFreePricer])
@dataclass(frozen=True)
class MkArbFree:
    """
    A utility class which generates a "closest" arbitrage free market using the ideas from the paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2642630.

    Given a market with expiries, strikes, bid and asks this functional:
    1) Expands the strike range following https://arxiv.org/pdf/2601.11209 remark 2.7, and drops expiries with too few strikes or with too steep bid-ask slopes at the boundaries.
    2) Fits per expiry a linear no-arbitrage mid price within the bid-ask spread. It drops any expiries for which this is not possible as it implies actionable arbitrage. The function 
       then drops all strikes which have an implied probability density of zero (which means the fitted mid is linear at those strikes).
    3) The function then fits a fully arbitrage free surface, on a strike grid which theoretically means that a mid-price fit exists if and only if the
        resulting fit is within bid/ask spread. The strike grid used is returned as ``mstrikes``.

    See :meth:`MkArbFree.__call__` for more details on the input and output of this functional.
    """
    no_arb_mode             : str = "surface"  # Whether to enforce no-arbirage: "expiry" or "surface".
    smoothness              : float|None = None # Whether to impose SANOS smoothness.
    surface_expiry_linear   : bool = True  # In 'surface' mode, whether to run 'expiry' in linear mode.

    # control over process parameters
    min_half_spread         : float = 1E-4 #: Minimum bid/ask spread. The optimization is weighted by inverse half spread.
    mid_fit_weight          : float = 0.1  #: Weight for fitting to the mid prices of options, where 1 means mid-price fit only is active, while 0 means to fit only a linear penalty for exceeding bid-ask.
    min_atm_vol             : float = 0.05 #: Minimum ATM vol.
    min_dlv                 : float|None = None #: Minimum DLV (difference of log-volatilities between consecutive expiries) to consider an expiry as non-degenerate.
    max_dlv                 : float|None = None  #: Maximum DLV (difference of log-volatilities between consecutive expiries) to consider an expiry as non-degenerate.

    # timeout
    timeout_seconds         : float|None = 60.*5. #: Time limit for the optimization solver, in seconds. This is passed to the CVXPY solver options as "time_limit".

    # micro paramters
    min_strikes_per_expiry  : int = 5      #: (expert use only) Minimum strikes per expiry. Expiries with fewer strikes will be dropped.
    min_density             : float = 1E-4 #: (expert use only) Minimum density value to consider in the optimization. Strikes with lower values will be dropped.
    min_dC_eps              : float = 1E-3 #: (expert use only) Minimum dC/dK slope at the boundaries. Expiries with steeper slopes will be dropped.

    # protected variables do not alter the ID of this field (because they start with '_')
    _debug_verbose_cvxpy  : set[str]|None = None 
    _remove_no_arb_strikes: bool = True
    _debug_super_dense    : bool = False
    _eps_price            : float = 1E-6
    _eps_density          : float = 1E-7
    _eps_dens_1           : float = 1E-4 #: (expert use only) What passes as ``sum(p)==1`` and ``p@K==1`` for the fitted density.
    _eps_no_arb           : float = 0.01 #: % half spread to tighten fit.

    def __post_init__(self):
        verify_inp( self.no_arb_mode in ["expiry", "surface"], lambda : f"'no_arb_mode' must be one of 'expiry' or 'surface', but found '{self.no_arb_mode}'" ) 
        verify_inp( self.min_strikes_per_expiry >= 3, lambda : f"'min_strikes_per_expiry' must be at least 3, but found {self.min_strikes_per_expiry}" )
        verify_inp( self.min_dC_eps > 0., lambda : f"'min_dC_eps' must be positive, but found {self.min_dC_eps}" )
        verify_inp( self.min_half_spread > 0., lambda : f"'min_half_spread' must be positive, but found {self.min_half_spread}" )
        verify_inp( self.min_density > 0., lambda : f"'min_density' must be positive, but found {self.min_density}" )
        verify_inp( self.timeout_seconds is None or self.timeout_seconds >= 0., lambda : f"'timeout_seconds' must be positive or None, but found {self.timeout_seconds}" )
        verify_inp( self.smoothness is None or self.smoothness >= 0., lambda : f"'smoothness' must be positive or None, but found {self.smoothness:.4g}")
        verify_inp( self.mid_fit_weight >= 0. and self.mid_fit_weight <= 1., lambda : f"'mid_fit_weight' must be between 0 and 1, but found {self.mid_fit_weight:.4g}" )
        verify_inp( self.min_atm_vol >= 0., lambda : f"'min_atm_vol' must be non-negative, but found {self.min_atm_vol:.4g}" )

    def __str__(self) -> str:
        """ Very compact but still readable summary """
        surface_expiry_linear = self.surface_expiry_linear and self.no_arb_mode=="surface" and not self.smoothness is None
        return f"{self.no_arb_mode},{f'std={self.smoothness:.2g},' if self.smoothness is not None else ''}{'elin' if surface_expiry_linear else ''}mhs={self.min_half_spread:.3g},"+\
               f"mw={self.mid_fit_weight:.2f},mav={self.min_atm_vol:.2g},"+\
               ( f"midl={self.min_dlv:.3g}," if not self.min_dlv is None else "" ) + ( f"mxdv={self.max_dlv:.3g}," if not self.max_dlv is None else "" ) +\
               f"mik={self.min_strikes_per_expiry},mdn={self.min_density:.3g},dC={self.min_dC_eps:.3g},dps={self._eps_dens_1:.1g}"

    def __unique_hash__( self, unique_hash, debug_trace ) -> str:
        """ Overwrite hash function to ignore non-functional parameters """
        r = asdict(self)
        r.pop("timeout_seconds", None)
        if self.no_arb_mode!="surface" or self.smoothness is None:
            r.pop("surface_expiry_linear", None)
        return unique_hash(r, debug_trace=debug_trace)

    def __call__(self,
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                bids           : list[np.ndarray],
                asks           : list[np.ndarray], *,
                vols           : np.ndarray|float|None = None, # forward volatilities per expiry, used for the strike expansion step. If a float is provided, it is used for all expiries. If None, it is estimated from the data as the median of the implied vols.
                message_context: str|callable[[], str] | None = None,
                verbose        : Context = Context.all ) -> ArbFreePricer:
        """
        Make market data arbitrage-free using ideas from the paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2642630.

        Given a market with expiries, strikes, bid and asks this functional:
        1) Expands the strike range following https://arxiv.org/pdf/2601.11209 remark 2.7, and drops expiries with too few strikes or with too steep bid-ask slopes at the boundaries.
        2) Fits per expiry a linear no-arbitrage mid price within the bid-ask spread. It drops any expiries for which this is not possible as it implies actionable arbitrage. The function 
           then drops all strikes which have an implied probability density of zero (which means the fitted mid is linear at those strikes).
        3) The function then fits a fully arbitrage free surface, on a strike grid which theoretically means that a mid-price fit exists if and only if the
           resulting fit is within bid/ask spread. The strike grid used is returned as ``mstrikes``.

        This function is usually very fast, but it can get stuck when the overall surface is close to arbitrage in areas which have very little
        room to adjust. 

        This function returns a :class:`ArbFreePricer` object which contains the result of the fit, and can be used to price options at arbitrary strikes and expiries using the fitted model.

        Parameters
        ----------
        expiries : :class:``numpy.ndarray`
            Increasing array of expiries (in years).
        strikes : list[:class:``numpy.ndarray`]
            List of strike arrays, one per expiry. Each strike array must be strictly increasing and positive, and must have the same shape as the corresponding bid and ask arrays.
        bids : list[:class:``numpy.ndarray`]
            List of bid price arrays, one per expiry. 
        asks : list[:class:``numpy.ndarray`]
            List of ask price arrays, one per expiry. 
        vols : :class:``numpy.ndarray`` | float | None, default ``None``
            Volatilities per expiry used when ``smoothness`` is not None.
            If left to ``None``, the function will compute an approximate mid-atm volatility from the provided bid and ask prices per expiry.
        message_context : str | callable[[], str] | None, default None
            Optional context to prepend to error and warning messages.
            This can be a string or a callable which returns a string, e.g. a lambda which returns a string with dynamic information such as the current expiry being processed. If None, no context is prepended.
        verbose : :class:`Context`, default ``Context.all``
            Verbose context for logging.
        
        Returns
        -------
        Extended market: ``dict``
           See above.
        """
        def message(with_cols : bool = True) -> str:
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg == "":
                return ""
            msg = msg[0].upper() + msg[1:]
            if not with_cols:
                return msg
            return msg + ": "

        T = len(expiries)

        verify_inp( T > 0, lambda : f"{message()}Expected at least one expiry, but found {T}" )
        verify_inp( len(strikes) == T, lambda : f"{message()}Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(bids) == T, lambda : f"{message()}Expected number of bid arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(asks) == T, lambda : f"{message()}Expected number of ask arrays {T} to match number of expiries {len(expiries)}" )   
        verify_inp( expiries[0] > 0. and (len(expiries)==1 or np.min(np.diff(expiries)) > 0.), lambda : f"{message()}Expiries must be positive and strictly increasing, but found {expiries}" )

        def make_variances():
            if vols is None:
                atm_calls = []
                for t in range(T):
                    ixs = np.argsort( strikes[t] )
                    atm_calls.append( np.interp( 1., strikes[t][ixs], (bids[t][ixs]+asks[t][ixs])/2. ) )
                atm_calls = np.asarray(atm_calls)
                atm_vols  = bs.implied( k=np.ones_like(atm_calls), prices=atm_calls, sqrtT=np.sqrt(expiries), eps=self._eps_price )
                variances = expiries * (np.maximum(atm_vols, self.min_atm_vol)**2)
                dts       = np.diff(expiries, prepend=0.)
                fwd_vars  = np.maximum(0.001**2, np.diff(variances, prepend=0.) / dts ) * dts
                variances = np.cumsum(fwd_vars)
                del atm_calls, atm_vols, dts, fwd_vars
            elif isinstance( vols, np.ndarray ):
                verify_inp( vols.shape == expiries.shape, lambda : f"{message()}If 'vols' is a numpy array, it must have the same shape as 'expiries', but found {vols.shape} and {expiries.shape}" )
                verify_inp( np.all(vols > 0.), lambda : f"{message()}All volatilities must be positive, but found {vols}" )
                variances = expiries * (vols**2) 
                verify_inp( variances[0] > 0. and (len(variances)==1 or np.min(np.diff(variances)) > 0.), lambda : f"{message()}Variances derived from 'vols' must be positive and strictly increasing, but found {variances}" )
            else:
                variances = expiries * (float(vols)**2) 
                verify_inp(vols > 0., lambda : f"{message()}Volatility must be positive, but found {vols}" )
            return variances
        variances = make_variances() 
        del vols

        # generate strike range, fit per-expiry
        # -------------------------------------
        # Add strikes to the left and right which are consistent with the no-arb condition.

        r             = ArbFreePricer(self.smoothness)
        r.min_half_spread = self.min_half_spread
        r.expiries    = []
        r.variances   = [] 
        r.xstrikes    = []
        r.xdensities  = []
        r.masks       = []
        r.strikes     = []
        r.asks        = []
        r.bids        = []
        r.fits1       = []
        tme           = verbose.timer()
        r.expiry_linear = self.surface_expiry_linear and self.no_arb_mode=="surface" and not self.smoothness is None

        for t, expiry in enumerate(expiries):
            def local_message():
                msg = message(False)
                ext = f"expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d"
                if msg == "":
                    msg = ext[0].upper() + ext[1:]
                else:
                    msg += " " + ext
                return msg
            verbose.write(lambda : f"\r{local_message()}: Generating extended strikes... ", end='')

            # strikes
            # -------

            strikes_t = strikes[t]

            if len(strikes_t) < self.min_strikes_per_expiry:
                verbose.write(lambda : f"\r{local_message()}: Skipping expiry since it has fewer than {self.min_strikes_per_expiry} strikes.")
                continue

            xstrikes_t, mask_t, notice = self.make_extended_strikes( strikes_t,
                                                                     bids=bids[t], 
                                                                     asks=asks[t], 
                                                                     prev_lower = r.xstrikes[-1][0] if len(r.xstrikes) > 0 else None,
                                                                     prev_upper = r.xstrikes[-1][-1] if len(r.xstrikes) > 0 else None,
                                                                     message_context = local_message)
            assert not xstrikes_t is None, "Internal error: xstrikes_t is None."
            if not notice is None:
                verbose.write(lambda : notice() + ": ", head=False, end='')
            assert np.all( xstrikes_t[1:-1] == strikes_t[mask_t] ), "Mask error...?"
            verbose.write(lambda : f"using {len(xstrikes_t)} extended strikes based on {len(strikes_t)} market strikes.", head=False)

            # linear fit per expiry
            # ---------------------

            strikes_t  = strikes[t]
            bid_t      = bids[t]
            ask_t      = asks[t]
            verify_inp( bid_t.shape == ask_t.shape == strikes_t.shape, lambda : f"{local_message()}: 'strikes[{t}]', 'bids[{t}]' and 'asks[{t}]' must all have the same shape, but found {strikes_t.shape}, {bid_t.shape} and {ask_t.shape}" )
            verify_inp( np.min( ask_t - bid_t ) > 0., lambda : f"{local_message()}: All options must have positive bid/ask spread, but found minimum spread {np.min(ask_t - bid_t):.4g}" )

            p_t, fit_t = self.expiry_density( 
                                expiry = expiry,
                                variance = variances[t],
                                strikes_t = strikes_t,
                                bid_t = bid_t, 
                                ask_t = ask_t, 
                                mask_t = mask_t, 
                                xstrikes_t = xstrikes_t,
                                force_linear = r.expiry_linear,
                                message_context = local_message,
                                verbose = verbose)
            assert xstrikes_t.shape == p_t.shape, f"Internal error: {local_message()}Shape of xstrikes {xstrikes_t.shape} does not match shape of density {p_t.shape}"

            fit_mask = ( fit_t >= bid_t - self._eps_price ) & ( fit_t <= ask_t + self._eps_price ) & mask_t
            fit_mask_live = fit_mask[mask_t]

            if not np.all(fit_mask_live):
                if np.sum(fit_mask_live) < len(fit_mask_live)//2:
                    verbose.write(lambda : f"\r{local_message()}: skipping expiry since we cannot fit an arbitrage-free mid within bid/ask spread for more than half of the strikes.")
                    continue
    
                if self._remove_no_arb_strikes:
                    assert np.all( xstrikes_t[1:-1] == strikes_t[mask_t] ), "Mask error...?" # only true up here. 
                    xfit_mask    = np.concatenate(([True], fit_mask_live, [True]))
                    xstrikes_R_t = xstrikes_t[xfit_mask]

                    pR_t, fitR_t = self.expiry_density(
                        expiry=expiry,
                        variance=variances[t],
                        strikes_t=strikes_t,
                        bid_t=bid_t,
                        ask_t=ask_t,
                        mask_t=fit_mask,
                        xstrikes_t=xstrikes_R_t,
                        force_linear = r.expiry_linear,
                        message_context=local_message,
                        verbose=verbose,
                    )

                    fitR_mask = ( fitR_t >= bid_t - self._eps_price ) & ( fitR_t <= ask_t + self._eps_price ) & fit_mask
                    fitR_mask_live = fitR_mask[fit_mask]
                    if np.all(fitR_mask_live):
                        verbose.write(lambda : f"\r{local_message()}: fit achived after removing {np.sum(~fit_mask)} of {len(fit_mask)} strikes.")
                        xstrikes_t = xstrikes_R_t
                        p_t        = pR_t
                        fit_t      = fitR_t
                    else:
                        verbose.write(lambda : f"\r{local_message()}: removing {np.sum(~fit_mask)} of {len(fit_mask)} strikes did not achieve fit; keeping all strikes but masking the options out.")
                    del pR_t, fitR_t, fitR_mask, fitR_mask_live

                mask_t = fit_mask
            del fit_mask, fit_mask_live

            m_bid_t      = bids[t][mask_t] 
            m_ask_t      = asks[t][mask_t]
            m_fit_t      = fit_t[mask_t]
            assert np.all( m_fit_t >= m_bid_t - self._eps_price ) and np.all( m_fit_t <= m_ask_t + self._eps_price ), f"Internal error: {local_message()}Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: fitted mid is not within bid/ask spread after attempted fixes. This should not happen."    

            assert xstrikes_t.shape == p_t.shape, f"Internal error: {local_message()}Shape of xstrikes {xstrikes_t.shape} does not match shape of density {p_t.shape}"

            # store result
            r.expiries.append( expiry )
            r.variances.append( variances[t] ) 
            r.input_index.append( t )

            r.xstrikes.append( xstrikes_t )
            r.xdensities.append( p_t )

            r.masks.append( mask_t )
            r.strikes.append( strikes[t] )
            r.bids.append( bids[t] )
            r.asks.append( asks[t] )
            r.fits1.append( fit_t )   # fit is computed for all strikes, also those unmasked

        r.expiries  = np.array(r.expiries, dtype=np.float64)
        r.variances = np.array(r.variances, dtype=np.float64) 
        r.run_time_expiry = tme.seconds

        def expiry_message():
            msg = message(False)
            if msg == "":
                return "Expiry fit"
            else:
                return msg + " expiry fit"
        verbose.write(lambda : f"\r{expiry_message()}: Generated {len(r.expiries)} {'linear' if self.smoothness is None else f'{self.smoothness:.2f}-smooth'} {'arbitrage-free ' if self.no_arb_mode != 'off' else ''}expiries with extended strikes from {T} market expiries. This took {tme}.")

        if self.no_arb_mode != "surface":
            r._make_numpy()
            return r
        
        # backward-dense grid
        # -------------------
        # Carry grid forward as long as the strike range isn't too tight.
        # We are using the xstrike_t grid from which we removed all strikes
        # for which the per-expiry density is zero

        T            = len(r.expiries)
        r.mstrikes   = [None]*T
        r.mdensities = None #set below
        r.xix_to_mix = [None]*T
        r.fitsX      = []

        min_dx       = None
        mguess_p     = [None]*T
        tme          = verbose.timer()

        for t in range(T-1,-1,-1):
            # fix strikes
            # 1) Ensure outside strike range is increasing.
            # 2) Add strikes of next expiry if not too close to existing strikes. Always keep current strikes
            xstrikes_t = r.xstrikes[t]
            mstrikes_t = xstrikes_t
            mp_t       = r.xdensities[t] 
            project = None
            if t<T-1:
                next_mstrikes = r.mstrikes[t+1]
                assert next_mstrikes[0] <= xstrikes_t[0] and next_mstrikes[-1] >= xstrikes_t[-1], f"Internal error: {message()}next_mstrikes[0] {next_mstrikes[0]:.3f} > xstrikes_t[0] {xstrikes_t[0]:.3f} or next_mstrikes[-1] {next_mstrikes[-1]:.3f} < xstrikes_t[-1] {xstrikes_t[-1]:.3f}"

                min_dx   = np.diff(mstrikes_t).min() if not self._debug_super_dense else min( np.diff(mstrikes_t).min(), np.diff(next_mstrikes).min() )
                collect  = []
                project  = []
                mgpcl    = []
                inext    = 0
                add_any  = False

                # skip next strikes until relevant
                while next_mstrikes[inext] <= xstrikes_t[0]+min_dx:
                    inext += 1

                project.append( len(collect) )
                collect = [ xstrikes_t[0] ]
                mgpcl   = [ mp_t[0] ]

                for ithis in range(1, xstrikes_t.shape[0]):
                    k = xstrikes_t[ithis]
                    # add next strikes up to 'k'
                    while next_mstrikes[inext] <= k-min_dx:
                        if next_mstrikes[inext] - collect[-1] >= min_dx:
                            collect.append( next_mstrikes[inext] )
                            mgpcl.append( 0. )
                            add_any = True
                        inext += 1
                    project.append( len(collect) )
                    collect.append( k )
                    mgpcl.append( mp_t[ithis] )
                    # skip next_mstrikes which are too close to k
                    while next_mstrikes[inext] <= k+min_dx:
                        inext += 1
                        if inext >= next_mstrikes.shape[0]:
                            break

                assert len(collect) == len(mgpcl), f"Internal error: {message()}collect length {len(collect)} does not match mgpcl length {len(mgpcl)}"
                if add_any:
                    mstrikes_t = np.array(collect, dtype=xstrikes_t.dtype)
                    mp_t       = np.array(mgpcl, dtype=np.float64)
                    assert np.min( np.diff( mstrikes_t ) ) >= min_dx-1E-6, f"Internal error: {message()}min dx {np.diff(mstrikes_t).min():.6g} < min_dx {min_dx:.6g}"
                    assert mstrikes_t[0] == xstrikes_t[0] and mstrikes_t[-1] == xstrikes_t[-1], f"Internal error: {message()}mstrikes_t[0] {mstrikes_t[0]:.3f} != xstrikes_t[0] {xstrikes_t[0]:.3f} or mstrikes_t[-1] {mstrikes_t[-1]:.3f} != xstrikes_t[-1] {xstrikes_t[-1]:.3f}"
                    project = np.array( project, dtype=np.int32 )
                else:
                    project = None

            assert project is None or np.allclose( xstrikes_t, mstrikes_t[project] ), f"Internal error: {message()}projection error, xstrikes_t {xstrikes_t} should be equal to mstrikes_t[project] {mstrikes_t[project]}"
            r.mstrikes[t]   = mstrikes_t
            r.xix_to_mix[t] = project
            mguess_p[t]     = mp_t

        # solve surface no-arb
        # --------------------

        def surface_message():
            msg = message(False)
            if msg == "":
                return "Surface fit"
            else:
                return msg + " surface fit"

        surface = self.surface_densities(
                expiries = r.expiries,
                variances = r.variances,
                strikes = r.strikes,
                asks = r.asks,
                bids = r.bids,
                masks = r.masks,
                xstrikes = r.mstrikes,
                xguessed_p = mguess_p, 
                hard_fit = False,
                message_context = surface_message,
                verbose = verbose
                )

        r.fitsX      = surface.fits
        r.mdensities = surface.xdensities
        r.run_time_surface = tme.seconds

        # make numpy
        r._make_numpy()
        return r

    def surface_densities( self, *,
                           expiries : np.ndarray,
                           variances : np.ndarray,
                           strikes : list[np.ndarray],
                           asks : list[np.ndarray],
                           bids : list[np.ndarray],
                           masks : list[np.ndarray]|None,
                           xstrikes : list[np.ndarray],
                           xguessed_p : list[np.ndarray]|None = None, 
                           force_linear : bool = False,
                           hard_fit : bool = False,
                           message_context : callable|str|None = None,
                           verbose : Context = Context.quiet
                          ):
        """
        Stand-alone smooth or linear fit to a given surface
        """
        def message(with_cols : bool = True) -> str:
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg == "":
                return ""
            assert msg[-2:] != ": ", f"Internal error: message_context should not end with ': ' found '{msg}'"
            msg = msg[0].upper() + msg[1:]
            if not with_cols:
                return msg
            return msg + ": "
    
        T = len(expiries)
        assert T == len(strikes) == len(xstrikes) == len(asks) == len(bids) == len(variances), f"Internal error: length of expiries {len(expiries)}, strikes {len(strikes)}, xstrikes {len(xstrikes)}, asks {len(asks)}, bids {len(bids)}, and variances {len(variances)} must all be the same."
        assert xguessed_p is None or len(xguessed_p) == T, f"Internal error: length of xguessed_p {len(xguessed_p)} must be the same as length of expiries {T}."
        assert masks is None or len(masks) == T, f"Internal error: length of masks {len(masks)} must be the same as length of expiries {T}."

        with verbose.write_t(f"\r{message()}Finding {'linear' if self.smoothness is None else f'{self.smoothness:.2f}-smooth'} arbitrage-free prices {'between bid/ask ' if hard_fit else ''}across {sum( len(xt) for xt in xstrikes )} strikes... ", end='') as tme:
            variables   = []
            constraints = []
            objective   = 0.
            xdensities  = []
            CXs         = []

            avhs_sum = 0.
            avhs_count = 0
            for t in range(T):
                mask_t   = masks[t] if masks is not None and masks[t] is not None else slice(None,None)
                avhs_sum += np.sum( asks[t][mask_t] - bids[t][mask_t] )/2.
                avhs_count += len(asks[t][mask_t])    
            average_hspread    = avhs_sum / max(1, avhs_count)
            eff_mid_fit_weight = self.mid_fit_weight if not hard_fit else 1.
            del avhs_sum, avhs_count

            for t in range(T):
                expiry     = expiries[t]
                strikes_t  = strikes[t]
                xstrikes_t = xstrikes[t]
                mask_t     = masks[t] if not masks is None else slice(None,None)
                ask_t      = asks[t][mask_t]
                bid_t      = bids[t][mask_t]
                xguess_t   = np.clip( xguessed_p[t], 0., None ) if not xguessed_p is None else None
                assert xguess_t is None or xguess_t.shape == xstrikes_t.shape, f"Internal error: length of xguessed_p[{t}] {len(xguess_t)} must be the same as length of xstrikes[{t}] {len(xstrikes_t)}."

                # density
                q_t         = cp.Variable(len(xstrikes_t), bounds=[0.,1.])#nonneg=True)
                assert q_t.shape == (len(xstrikes_t),), f"Internal error: shape of q_t {q_t.shape} must be the same as length of xstrikes_t {len(xstrikes_t)}." 
                assert xguess_t is None or q_t.shape == xguess_t.shape, f"Internal error: shape of q_t {q_t.shape} must be the same as shape of xguess_t {xguess_t.shape}."
                variables   += [(q_t, xguess_t)]
                constraints += [ cp.sum( q_t ) == 1.,
                                 q_t @ xstrikes_t == 1., # martingale condition
                                 q_t <= 1.
                                ]

                # martingale
                Cm          = np.maximum( 0., xstrikes_t[None,:] - xstrikes_t[1:-1,None] ) @ q_t    # model strikes - do not use "call"
                if t>0:
                    assert xstrikes[t-1].shape == xdensities[t-1].shape, f"Internal error {message()}: shape mismatch between previous and current mstrikes, {xstrikes[t-1].shape} vs {xdensities[t-1].shape}"
                    Cm1         = np.maximum( 0., xstrikes[t-1][None,:] - xstrikes_t[1:-1,None] ) @ xdensities[t-1]
                    constraints += [ Cm >= Cm1 ]
                else:
                    Cm1   = np.maximum( 0., 1. - xstrikes_t[1:-1] )    # model strikes, previous density is dirac at 1

                # dlv constaints
                # dC = 1/2 * dlvsq * x^2 * dt * K^2 / (1/2 (k^+ - k^-)) q
                box         = xstrikes_t[1:-1]**2 * ( expiries[t] - ( expiries[t-1] if t>0 else 0. ) ) / ( xstrikes_t[2:] - xstrikes_t[:-2])
                if not self.min_dlv is None:
                    constraints += [ Cm - Cm1 >= cp.multiply( box * (self.min_dlv**2), q_t[1:-1] ) ]
                if not self.max_dlv is None:
                    constraints += [ Cm - Cm1 <= cp.multiply( box * (self.max_dlv**2), q_t[1:-1] ) ]
                del box, Cm1

                # fit
                def call( spots, strikes ):
                    if force_linear or self.smoothness is None or self.smoothness == 0. or expiry < ONE_MINUTE:
                        return np.maximum( 0., spots - strikes )
                    expiry_ = np.clip( expiry, expiries[0], expiries[-1] )
                    vol     = math.sqrt( self.smoothness * np.interp( expiry_, expiries, variances ) / expiry_ )
                    assert np.isfinite(vol), f"Internal error: {message()}infinite vol"
                    if vol == 0.:
                        return np.maximum( 0., spots - strikes )
                    return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )*spots

                CX           = call( spots=xstrikes_t[None,:], strikes=strikes_t[:,None] )
                fit_mid      = CX[mask_t,:] @ q_t    # market strikes
                half_spread  = (ask_t - bid_t)/2.
                mid_t        = (bid_t + ask_t)/2.
                whalf_spread = np.maximum( half_spread, self.min_half_spread ) / average_hspread

                if hard_fit:
                    constraints += [ fit_mid <= ask_t - half_spread * (1.-self._eps_no_arb),
                                     fit_mid >= bid_t + half_spread * (1.-self._eps_no_arb) ]

                # slack variable version  *should* be better..?
                guess_fit    = CX[mask_t,:] @ xguess_t if not xguess_t is None else None
                if eff_mid_fit_weight < 1.:
                    e_init = np.clip( np.maximum( guess_fit - ask_t, bid_t - guess_fit ) / whalf_spread, 0., None) if guess_fit is not None else None
                    e_box  = cp.Variable(len(bid_t), bounds=[0.,1E6])#nonneg=True)
                    assert e_init is None or e_box.shape == e_init.shape, f"Internal error: shape of e_box {e_box.shape} must be the same as shape of e_init {e_init.shape}."
                    variables += [ (e_box, None) ]

                    constraints += [
                        fit_mid <= ask_t - half_spread * (1.-self._eps_no_arb) + cp.multiply(whalf_spread, e_box),
                        fit_mid >= bid_t + half_spread * (1.-self._eps_no_arb) - cp.multiply(whalf_spread, e_box)
                    ]

                    objective += (1. - eff_mid_fit_weight) * cp.sum(e_box)
                    del e_box

                if eff_mid_fit_weight > 0.:
                    z_init = np.clip( np.max( np.abs( guess_fit - mid_t ) / whalf_spread ), 0., None) if guess_fit is not None else None
                    z_mid  = cp.Variable(bounds=[0.,1E6])#nonneg=True)
                    assert z_init is None or z_mid.shape == z_init.shape, f"Internal error: shape of z_mid {z_mid.shape} must be the same as shape of z_init {z_init.shape}."
                    variables += [ (z_mid, None) ]

                    constraints += [
                        fit_mid - mid_t <= cp.multiply(whalf_spread, z_mid),
                        mid_t - fit_mid <= cp.multiply(whalf_spread, z_mid),
                    ]

                    objective += eff_mid_fit_weight * z_mid
                    del z_mid

                xdensities.append(q_t)
                CXs.append(CX)

            success, status, fell_back = SanosLPSolver(timeout_seconds=self.timeout_seconds,
                                                       _debug_verbose_cvxpy=self._debug_verbose_cvxpy)( 
                                                           objective=objective,
                                                           variables=variables, 
                                                           constraints=constraints, 
                                                           message_context=lambda : message(False), 
                                                           verbose=verbose )
            if fell_back:
                verbose.write(lambda : f"had to fall back to a slower solver... ", head=False, end='' )

            if not success:
                if status in ["infeasible", "infeasible_inaccurate"]:
                    if hard_fit:
                        return None
                    raise RuntimeError(f"{message()}Cannot find no-arbitrage surface mids: problem infeasible??")
                if status in ["unbounded", "unbounded_inaccurate", "infeasible_or_unbounded"]:
                    raise RuntimeError(f"{message()}Cannot find no-arbitrage surface mids: problem unbounded? This should not happen. Please contact the authors with an example.")
                if status == "user_limit":
                    raise RuntimeError(f"{message()}Cannot find no-arbitrage surface mids: reached timeout {self.timeout_seconds} and/or iteration limit.")
                raise RuntimeError(f"{message()}Cannot find no-arbitrage surface mids: solver returned status '{status}'.")

            if status != "optimal":
                verbose.write(lambda : f"solver returned '{status}' -- proceeding anyway... ", head=False, end='' )

            # compute call prices for the original strikes
            fits       = []
            xdensities = [ q.value for q in xdensities ]
            for t in range(T):
                expiry     = expiries[t]
                strikes_t  = strikes[t]
                xstrikes_t = xstrikes[t]
                CX         = CXs[t]
                q_t        = xdensities[t]
                if np.min(q_t) < -self._eps_density or abs( np.sum( q_t )-1. ) > self._eps_dens_1 or abs( q_t@xstrikes_t - 1. ) > self._eps_dens_1:
                    warnings.warn(f"{message()}Warning: no-arbitrage density does not satisfy basic properties after optimization: min {q_t.min():.3e}, sum {q_t.sum():.6f}, integral {q_t@xstrikes_t:.6f} -- attempting fix.")
                    q_t = warn_and_fix_density( xstrikes_t, q_t, eps=self._eps_density, message = lambda : f"{message()}expiry {expiry*DAYS_PER_YEAR:.2f}d: surface density after optimization" )

                # zero out some density
                floor      = max( self.min_density / float(len(q_t)), self._eps_density ) if IMPOSE_FLOOR_ON_DENSITY else 0.
                is_zero    = q_t < floor
                if np.any(is_zero):
                    q_t[is_zero]  = 0.
                q_t           = warn_and_fix_density( xstrikes_t, q_t, eps=self._eps_density, message = lambda : f"{message()} expiry {expiry*DAYS_PER_YEAR:.2f}d: surface density" )
                xdensities[t] = q_t
                fit_t         = CX @ q_t
                assert fit_t.shape == strikes_t.shape, f"Internal error: {message()}Shape of fit_t {fit_t.shape} does not match shape of strikes_t {strikes_t.shape}"
                fits.append( fit_t )
            verbose.write(lambda : f"done. This took {tme}.", head=False )

        # make numpy
        return PrettyObject(
                xdensities = xdensities,
                fits       = fits,
                CXs        = CXs)
    
    def expiry_density(self, *,
                        expiry : float, 
                        variance : float,
                        strikes_t : np.ndarray,
                        bid_t : np.ndarray,
                        ask_t : np.ndarray,
                        mask_t : np.ndarray|None,
                        xstrikes_t : np.ndarray,
                        force_linear : bool = False,
                        message_context : str|callable|None = None, 
                        verbose : Context = Context.all,
                        end : str = "\n") -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a single expiry given market and model strikes.

        This function will first attempt a soft fit. If that soft fit is within bid/ask for all ``strikes_t``
        the function returns. If this fit is not within bid/ask, then the
        function attempts a hard fit in the soft fit starting point. It successful, it returns
        the hard fit. If not successful it returns the soft fit, which is not all within bid/ask.

        Parameters
        -----------
        expiry: float
            Time to expiry in years.
        variance : float | None
            Variance for the expiry, used only if ``self.smoothness`` is not None.
            If ``variance`` is zero or ``None``, the function attempts a linear fit.
        strikes_t : np.ndarray
            Strikes for the expiry. 
        bid_t : np.ndarray
            Bid prices for the strikes.
        ask_t : np.ndarray
            Ask prices for the strikes.
        xstrikes_t : np.ndarray
            Strikes for the density grid. The first and last point should be outside the range of strikes_t, and the rest should include all strikes_t.
        force_linear : boo,l, default ``False``
            If ``True``, the function attempts a linear fit even if ``self.smoothness`` is not None or the variance is positive.
        message_context : str|callable|None, default ``None``
            Optional context for error and debug messages. If a string is provided, it is used as a prefix for all messages.
            If a callable is provided, it is called to generate the message prefix at each call site.
        verbose : Context, default ``Context.all``
            Verbosity context for logging messages.
        end : str, default ``"\n"``
            For ``verbose`` output: how to end the output. Using ``end=''`` will surpess a line feed.

        Returns
        -------
        Fit : ``tuple[np.ndarray, np.ndarray, bool]``
            Returns a tuple with:

            * ``xdensity``: the fitted density on the xstrikes_t grid
            * ``fit``: the fitted call prices on the strikes_t grid
        """
        def message(with_cols : bool = True) -> str:
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg == "":
                return ""
            msg = msg[0].upper() + msg[1:]
            if not with_cols:
                return msg
            return msg + ": "

        assert expiry > 0, f"Internal error {message(False)}: expiry must be positive, but found {expiry}"
        assert variance >= 0, f"Internal error {message(False)}: variance must be non-negative, but found {variance}"
        assert strikes_t.shape == bid_t.shape == ask_t.shape, f"Internal error{message(False)}: strikes_t, bid_t and ask_t must all have the same shape, but found {strikes_t.shape}, {bid_t.shape} and {ask_t.shape}"

        # find fit between bid/ask without hard constraints first
        # -------------------------------------------------------
        tme = verbose.timer()

        # soft fit first
        # --------------

        expiries  = np.array([expiry], dtype=np.float64)
        variances = np.array([variance], dtype=np.float64) if not variance is None else None
        msk_t     = mask_t if not mask_t is None else slice(None,None)

        r = self.surface_densities( 
            expiries = expiries,
            variances = variances,
            strikes = [strikes_t],
            asks = [ask_t],
            bids = [bid_t],
            masks = [mask_t] if not mask_t is None else None,
            xstrikes = [xstrikes_t],
            xguessed_p = None, 
            hard_fit = False,
            force_linear = force_linear,
            message_context = lambda : message(False),
            verbose = Context.quiet
            )
        assert not r is None, "Internal error"
        xdensity = r.xdensities[0]
        fit      = r.fits[0]
        does_fit = (fit > bid_t - self._eps_price) & (fit < ask_t + self._eps_price)
        does_fit_msk = does_fit[msk_t]
        smoothness   = self.smoothness if not force_linear else None
        
        if np.all(does_fit_msk):
            verbose.write(lambda : f"\r{message()}Soft {'linear' if smoothness is None else f'{smoothness:.2f}-smooth'} per-expiry no-arbitrage fit succeeded within bid/ask. This took {tme}.", end=end)
            return xdensity, fit

        # hard fit
        # --------

        r = self.surface_densities( 
            expiries = expiries,
            variances = variances,
            strikes = [strikes_t],
            asks = [ask_t],
            bids = [bid_t],
            masks = [mask_t] if not mask_t is None else None,
            xstrikes = [xstrikes_t],
            xguessed_p = [xdensity],
            hard_fit = True,
            force_linear = force_linear, 
            message_context = lambda : message(False),
            verbose = verbose
            )

        if not r is None:
            fit      = r.fits[0]
            does_fit = (fit > bid_t - self._eps_price) & (fit < ask_t + self._eps_price)
            does_fit_msk = does_fit[msk_t]
            assert np.all( does_fit_msk ), "Internal error: hard fit did not fit within bid/ask??"
            verbose.write(lambda : f"\r{message()}Hard {'linear' if smoothness is None else f'{smoothness:.2f}-smooth'} per-expiry no-arbitrage fit succeeded within bid/ask. This took {tme}.", end=end)
            return r.xdensities[0], r.fits[0]

        verbose.write(lambda : f"\r{message()}Could not hard fit with {'linear' if smoothness is None else f'{smoothness:.2f}-smooth'} per-expiry no-arbitrage -- using masked soft fit instead which is outside bid/ask for {np.sum(~does_fit_msk)} of {len(does_fit_msk)} options. This took {tme}.")
        return xdensity, fit
    
    def make_extended_strikes( self, 
                              strikes: np.ndarray, 
                              bids: np.ndarray, 
                              asks: np.ndarray, *, 
                              prev_lower : float|None = None,
                              prev_upper : float|None = None,                            
                              message_context : str|None = None ) -> tuple[np.ndarray, np.ndarray, str|None]:
        r"""
        Utility function to generate an increasing range of extended strikes such that
        outside points are sufficiently far away, c.f. https://arxiv.org/abs/2601.11209 remark 2.7.

        Essentially, given strikes $K_1,\ldots,K_{n-1}$ and mid-prices $C_1,\ldots,C_n$, we want to find $K_0 < K_1$ and $K_{n+1} > K_n$ 
        with call price values equal to intrinsic value such that the boundary conditions
        $-1 < \frac{C_1 - C_0}{K_1 - K_0}$ and $\frac{C_{n+1} - C_n}{K_{n+1} - K_n} < 0$ are satisfied.

        **Lower strike**

        Let $dC_1  = \frac{C_2 - C_1}{K_2 - K_1}$ be the slope of the line connecting the point $(K_1,C_1)$ to the point $(K_2,C_2)$.
        Then we want to find $K_0$ with $C_0 = 1 - K_0$ such that:
        $$
            -1 <= \frac{ C_1 - C_0 }{ K_1 - K_0 } \leq dC_1 \ .
        $$
        If $dC_1$ is not greater than -1, we drop $K_1$ and proceed. Once $dC_1$ is greater than -1, we can chose any strike below
        $$
            K_0 \leq \frac{1 - C_1 + K_1 dC_1}{1 + dC_1} 
        $$
        which is positive. If $K_0$ is non-positive, we will drop $K_1$ from the strike list and proceed to the next strike.

        The actual strike used is $K'_0 := \max\{ K_0/2, K_0 - 2 ( K_1 - K_0 ) \}$.

        The new lower strike will thenn be the minimum of above calculation and ``prev_lower``, if provided.

        **Upper strike**

        Let $dC_{n-1} := \frac{C_n - C_{n-1}}{K_n - K_{n-1}}$ be the slope of the line connecting the point $(K_{n-1},C_{n-1})$ to the point $(K_n,C_n)$.
        Then we want to find $K_n$ with $C_n = 0$ such that
        $$
            dC_{n-1} \leq \frac{ C_{n+1} - C_n }{ K_{n+1} - K_n } \leq 0 \.
        $$
        If $dC_{n-1} >= 0$ we drop $K_n$. Once $dC_{n-1}$ is negative, we can chose any strike above
        $$
            K_{n+1} \geq K_n - \frac{C_n}{dC_{n-1}} \ .
        $$

        The actual strike used is $K'_{n+1} := K_n + 2 ( K_{n+1} - K_n )$.

        The new upper strike will then be the maximum of above calculation and ``prev_upper``, if provided.

        Parameters
        ----------
        strikes : np.ndarray
            Array of strikes for the expiry, must be strictly increasing and positive.
        bids : np.ndarray   
            Array of bid prices for the expiry, must have the same shape as strikes.
        asks : np.ndarray
            Array of ask prices for the expiry, must have the same shape as strikes and be strictly greater than bids.
        prev_lower : float | None, default ``None``
            Previous lower strike from the previous expiry, if any. The new lower strike will be the minimum of the calculated lower strike and this value, if provided.
        prev_upper : float | None, default ``None``
            Previous upper strike from the previous expiry, if any. The new upper strike will be the maximum of the calculated upper strike and this value, if provided.
        message_context : callable | None, default ``None``
            Optional callable that returns a string to be used as a header in error messages.
            This can be used to provide context such as the expiry time and other relevant information.

        Returns
        -------
        Extended strikes: tuple[np.ndarray, np.ndarray, str|None]
            A tuple containing the extended ``xstrikes``, the corresponding ``mask`` to map from input strikes to extended strikes, and an optional ``notice`` to be printed to the user
            in verbose mode.

            The relation between input strikes, mask, and output "xstrikes" is 
            ```
                xstrikes[1:-1] == strikes[mask]
            ```
        """
        def message(with_cols : bool = True) -> str:
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            if msg == "":
                return ""
            msg = msg[0].upper() + msg[1:]
            if not with_cols:
                return msg
            return msg + ": "

        input_strikes_t = strikes
        input_bid_t     = bids
        input_ask_t     = asks
        notice          = None
        xstrikes_t      = input_strikes_t
        bid_t           = input_bid_t
        ask_t           = input_ask_t
        mid_t           = (input_bid_t + input_ask_t)/2.
        mask_t          = np.ones(input_strikes_t.shape[0], dtype=bool)

        verify_inp( len(xstrikes_t) >= self.min_strikes_per_expiry, f"{message()}At least {self.min_strikes_per_expiry} strikes are required for this expiry, but found {len(xstrikes_t)}.")
        verify_inp( xstrikes_t.shape == bid_t.shape == ask_t.shape, lambda : f"{message()}Expected strikes, bids and asks to have the same shape for expiry, found {xstrikes_t.shape}, {bid_t.shape}, {ask_t.shape}")
        verify_inp( np.all( np.diff(xstrikes_t) > 0 ), lambda : f"{message()}Strikes must be strictly increasing; found minimal difference {np.diff(xstrikes_t).min():.3f}" )
        verify_inp( xstrikes_t[0] > 0., lambda : f"{message()}Lowest strike must be positive; found {xstrikes_t[0]:.3f}" )
        verify_inp( np.all( ask_t - bid_t >0. ), lambda : f"{message()}All ask prices must be greater than bid prices.")

        # extend strike range with boundary strikes
        # -----------------------------------------
        # At K_0 and K_n call prices are intrinsic
        # -1 <= { C_1 - C_0 }/{ K_1 - K_0 } <= dC_1
        # Assuming -1 < dC_1 and noting that K_0 < K_1
        # { 1 - K_0 - C_1) >= dC_1 (K_0 - K_1) =>
        # 1 - C_1 + K_1 dC_1 >= K_0 (1 + dC_1) =>
        # K_0 <= (1 - C_1 + K_1 dC_1)/(1 + dC_1)
        #
        # Assume that this is violated:
        # Buy the 0 strike for C_0 (the ask)

        assert self.min_dC_eps > 0., f"Internal error: min_dC_eps {self.min_dC_eps} must be positive."
        total_input = len(xstrikes_t)
        N  = xstrikes_t.shape[0]

        lower_k = None
        for ileft in range(N-2):
            C_0   = mid_t[ileft]
            C_1   = mid_t[ileft+1]
            K_0   = xstrikes_t[ileft]
            K_1   = xstrikes_t[ileft+1]
            dC_0  = ( C_1 - C_0 )/( K_1 - K_0 )
            if dC_0+1. > self.min_dC_eps:
                lower_k = (1 - C_1 + K_1 * dC_0)/(1 + dC_0)
                if lower_k > 0. and lower_k < K_1:
                    break
                lower_k = None
        if ileft == N-2 or lower_k is None:
            raise RuntimeError(f"{message()}Could not find consistent initial strike for this expiry.")
        upper_k = None
        for iright in range(N-2,ileft-1,-1):
            C_Nm1  = mid_t[iright]
            C_N    = mid_t[iright+1]
            K_Nm1  = xstrikes_t[iright]
            K_N    = xstrikes_t[iright+1]
            dC_Nm1 = ( C_Nm1 - C_N )/( K_Nm1 - K_N )
            if dC_Nm1 < -self.min_dC_eps:
                upper_k = K_Nm1 - C_Nm1/dC_Nm1
                if upper_k > K_Nm1:
                    break
                upper_k = None  
        if iright <= ileft or upper_k is None:
            raise RuntimeError(f"{message()}Could not find consistent terminal strike for this expiry.")

        if ileft > 0 or iright < N-2:
            slc         = slice( ileft, iright+1 )
            xstrikes_t  = xstrikes_t[slc]
            bid_t       = bid_t[slc]
            ask_t       = ask_t[slc]
            mask_t[:]   = False
            mask_t[slc] = True
        assert np.all( xstrikes_t == input_strikes_t[mask_t] ), "Mask error...(2) ?"

        if len(xstrikes_t) < total_input:
           removed_count = total_input - len(xstrikes_t)
           notice = lambda removed_count=removed_count, total_input=total_input: f"removed {removed_count} out of {total_input} strikes to ensure -1 < dC/dK[0] and dC/dK[-1] < 0"

        dK0        = K_1 - lower_k
        lower_k    = max( lower_k/2., K_1 - dK0*2. )
        dkN        = upper_k - K_Nm1
        upper_k    = K_Nm1 + dkN*2.

        if lower_k >= 1.:
            lower_k = 1.-(xstrikes_t[1]-xstrikes_t[0])
        if upper_k <= 1.:
            upper_k = 1.+(xstrikes_t[1]-xstrikes_t[0])

        # previous step?
        lower_k    = min(lower_k, prev_lower) if not prev_lower is None else lower_k
        upper_k    = max(upper_k, prev_upper) if not prev_upper is None else upper_k
        xstrikes_t =  np.concatenate(([lower_k], xstrikes_t, [upper_k]))

        assert np.all( bid_t == input_bid_t[mask_t] ), "Mask error...?"
        assert np.all( ask_t == input_ask_t[mask_t] ), "Mask error...?"
        assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"
        return xstrikes_t, mask_t, notice

    @staticmethod
    def atm_volatilities(
        expiries   : np.ndarray,
        strikes    : list[np.ndarray]|np.ndarray,
        vols       : list[np.ndarray]|np.ndarray,
        fwdvol_min : float = 0.01
        ) -> np.ndarray:
        """
        Computes ATM vols using linear interpolation in volatilities, with a floor of ``fwdvol_min``.

        The forward volatilities will be floored by ``fwdvol_min``.

        Parameters
        ----------
        expiries : :class:`numpy.ndarray`
            1D array of expiries of length ``T``.
        strikes : list[:class:`numpy.ndarray`]|:class:`numpy.ndarray`
            List of ``T`` 1D arrays of strikes or a 2D array of strikes with shape ``(T,N)``.
        vols : list[:class:`numpy.ndarray`]|:class:`numpy.ndarray`
            List of ``T`` 1D arrays of volatilities or a 2D array of volatilities with shape ``(T,N)``.
            For SANOS it is recommended to use ask volatilities.
        fwdvol_min : float, default 0.01
            Minimum forward volatility.

        Returns
        -------
        Vols: :class:`numpy.ndarray`
            1D array of ``T`` volatilities for each expiry, with forward volatility floored by ``fwdvol_min``.
        """

        verify_inp( len(expiries.shape) == 1, lambda: f"'expiries' must be a 1D array, found shape {expiries.shape}" )
        verify_inp( expiries[0] > 0., lambda: f"First expiry must be positive, found {expiries[0]}" )
        verify_inp( np.min( np.diff(expiries)) > 0., lambda: "Expiries must be increasing")
        verify_inp( isinstance( fwdvol_min, (int,float) ) and fwdvol_min >= 0., lambda: f"'fwdvol_min' must be a non-negative number, found {fwdvol_min}" )
        T = expiries.shape[0]

        if hasattr(strikes,"shape"):
            verify_inp( len(strikes.shape) == 2, lambda: f"'strikes' must be a 2D array if it has a shape, found shape {strikes.shape}" )
            verify_inp( strikes.shape[0] == T, lambda: f"First dimension of 'strikes' must match length of 'expiries', found {strikes.shape[0]} and {T}" )
            strikes = [ strikes[t] for t in range(T) ]  
        verify_inp( len(strikes) == T, lambda: f"Expected {T} strike slices, found {len(strikes)}" )
        if hasattr(vols,"shape"):
            verify_inp( len(vols.shape) == 2, lambda: f"'vols' must be a 2D array if it has a shape, found shape {vols.shape}" )
            verify_inp( vols.shape[0] == T, lambda:  f"First dimension of 'vols' must match length of 'expiries', found {vols.shape[0]} and {T}" )
            vols = [ vols[t] for t in range(T) ]  
        verify_inp( len(vols) == T, lambda: f"Expected {T} volatility slices, found {len(vols)}" )

        total_vars = []
        for t in range(T):
            strikes_t = strikes[t] 
            vols_t    = vols[t] 
            if np.min( np.diff(strikes_t) ) < 0.:
                ixs = np.argsort(strikes_t)
                strikes_t = strikes_t[ixs]
                vols_t    = vols_t[ixs]

            vol   = max( np.interp( 1., strikes_t, vols_t ), fwdvol_min )
            var   = vol**2 * expiries[t]

            if t>0:
                var = max( var, total_vars[-1] + fwdvol_min**2 * (expiries[t]-expiries[t-1]) )

            total_vars.append( var )
        return np.sqrt( np.array(total_vars, dtype=np.float64) / expiries.astype(np.float64, copy=False) )
