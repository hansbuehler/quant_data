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
from dataclasses import dataclass, field
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from numpy.testing import verbose
from pyparsing import col
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.utils.backbone import BackboneNumpy
import math

_EPS_PRICE = 1E-6
_EPS_DENSITY = 1E-7

@version("0.2")
class ArbFreePricer(PrettyObject):
    """
    Arbitrage-Free Option pricer based on DLV/SANOS.

    This class contains the result of :meth:`MkArbFree.__call__`, with the following public attributes
    which relate directly to the arbitrage-free fit:

    * ``expiries``: array of expiries. This may be fewer expiries than provided as input.

    * ``input_index``: for each returned expiry, the index of the input expiry which it corresponds to. In other words ``output_expiries == [ input_expiries[i] for i in input_index ]``.

    * ``masks``: a list of boolean masks by expiry, indicating which strikes have been included in the no-arbitrage calculation (True) and which have been dropped (False).              
    
    * ``fits1``: list of per-expiry fitted arbitrage-free prices for the input ``strikes``. This will be within bid/ask wherever ``mask`` is True.
        (This means this mid is not necessarily arbitrage-free across expiries.)

        This field is None if ``no_arb_mode`` is "off".

    * ``fitsX``: list of surface fitted arbitrage-free prices for the input ``strikes``. 
        These prices are globally arbitrage-free across the surface, but they are not necessarily within bid/ask spread; they are the closest fit
        among strikes with ``mask`` True.

        Up to machine precision existence
        of ``fitsX`` within bid/ask spread is equivalent to absence of arbitrage.

        This field is None if ``no_arb_mode`` is "off".

    Besides the pure fits, this class also contains the ingredients of the DLV/SANOS model.

    * For the **"expiry"** ``no_arb_mode``
      the function :meth:`MkArbFree.__call__` will generate internal model ``xstrikes`` around the provided market strikes. For those strikes it will
      compute a ``xdensities`` per expiry.

      These fields are None if ``no_arb_mode`` is "off".
        
      * ``xstrikes``: an extended strike list with two boundary strikes add. Boundary strikes are widening in time; c.f. https://arxiv.org/pdf/2601.11209 remark 2.7.

      * ``xdensities``: list of densities for ``xstrikes``, per expiry. Up to machine precision, each density is positive, sums to one, intergates to one 
         in the sense that `` xstrikes[t]@xdensities[t] == 1``
         and reprices ``fits``.

         This field is None if ``no_arb_mode`` is "off".

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
                    
    """

    def __init__(self, smoothness : float|None = None):
        self.smoothness  = smoothness
        self.xstrikes    = []
        self.expiries    = []
        self.masks       = []
        self.fits1       = None
        self.fitsX       = None
        self.mstrikes    = None
        self.xix_to_mix  = None
        self.xdensities  = None
        self.mdensities  = None
        self.input_index = []
        self.variances   = None

    _ONE_MINUTE = 1./(60.*6.5*DAYS_PER_YEAR)

    def price( self, expiry : float, strikes : np.ndarray, *, and_vols: bool = False, mode : str|None = None ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Price options at ``strikes`` with expiry ``expiry``.

        Use ``mode`` to chose which pricing kernel to use: "expiry" uses the per-expiry fit
        which may not be free of arbitrage in time; "surface" uses the full surface
        fit which is free of arbitrage in time. ``None`` will chose the latter
        if available, otherwise the former.

        If ``and_vols`` is ``True`` this function returns call prices and their implied volatilities;
        if ``False``, it will return only call prices.
        """
        verify_inp( isinstance( expiry, (float, int, np.number)) and expiry>=0, lambda : f"Expected 'expiry' to be a non-negative number, but found {str(expiry)}")
        verify_inp( mode is None or mode in ["surface","expiry"], lambda : f"Expected 'mode' to be one of 'surface', 'expiry' or None, but found {str(mode)}"   )
        assert isinstance(self.expiries, np.ndarray), "Internal error - forgot to make 'expiries' a numpy array"
 
        per_expiry = mode=="expiry" if not mode is None else ( self.fitsX is None )
        verify_inp( per_expiry or not self.fitsX is None, f"Cannot use 'mode' \"surface\" since no surface fit is available")

        if self.smoothness is None or expiry<self._ONE_MINUTE:
            def call( spots, strikes ):
                return np.maximum( 0., spots - strikes )
        else:
            assert not self.variances is None, "Internal error - variances are missing"
            vol = math.sqrt( self.smoothness * np.interp( expiry, self.expiries, self.variances ) / expiry )
            assert np.isfinite(vol), "Internal error: infinite vol"
            def call( spots, strikes ):
                return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )*spots

        def prices():
            if expiry<self._ONE_MINUTE:
                return np.maximum( 0., 1 - strikes )

            ix = np.searchsorted( self.expiries, expiry )
            if ix==0:
                strikes_0 = self.xstrikes[0] if per_expiry else self.mstrikes[0]
                p_0       = self.xdensities[0]  if per_expiry else self.mdensities[0]
                return call( spots=strikes_0[None,:], strikes=strikes[:,None] ) @ p_0

            if ix==len(self.expiries):
                strikes_T = self.xstrikes[-1] if per_expiry else self.mstrikes[-1]
                p_T       = self.xdensities[-1]  if per_expiry else self.mdensities[-1]
                return call( spots=strikes_T[None,:], strikes=strikes[:,None] ) @ p_T

            diff_ixm1 = abs( expiry - self.expiries[ix-1] )
            diff_ix   = abs( expiry - self.expiries[ix] )
            diff      = diff_ix if diff_ix < diff_ixm1 else diff_ixm1

            if diff<self._ONE_MINUTE:
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

        v = bs.implied( k=strikes, prices=c, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )
        return c,v 

    def _make_numpy(self):
        self.xstrikes    = [ np.asarray(x) for x in self.xstrikes ]
        self.expiries    = np.asarray(self.expiries)
        self.input_index = np.asarray(self.input_index, dtype=np.int32)
        self.masks       = [ np.asarray(x) for x in self.masks ]
        self.fits1       = [ np.asarray(x) for x in self.fits1 ] if not self.fits1 is None else None
        self.fitsX       = [ np.asarray(x) for x in self.fitsX ] if not self.fitsX is None else None
        self.mstrikes    = [ np.asarray(x) for x in self.mstrikes ] if not self.mstrikes is None else None
        self.xix_to_mix  = [ (np.asarray(x,dtype=np.int32) if not x is None else None)for x in self.xix_to_mix ] if not self.xix_to_mix is None else None
        self.xdensities  = [ np.asarray(x) for x in self.xdensities ] if not self.xdensities is None else None
        self.mdensities  = [ np.asarray(x) for x in self.mdensities ] if not self.mdensities is None else None
        self.variances   = np.asarray(self.variances) if not self.variances is None else None

@version("0.2", dependencies=[ArbFreePricer])
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
    no_arb_mode           : str = "surface"  # Whether to enforce no-arbirage: "off", "expriy", "surface".
    smoothness            : float|None = None # Whether to impose SANOS smoothness.

    min_strikes_per_expiry: int = 5      #: Minimum strikes per expiry. Expiries with fewer strikes will be dropped.
    min_dC_eps            : float = 1E-3 #: Minimum dC/dK slope at the boundaries. Expiries with steeper slopes will be dropped.

    min_half_spread       : float = 1E-4 #: Minimum bid/ask spread. The optimization is weighted by inverse half spread.
    cvxpy_solver          : str|None = None #:  Solver to use for the convex optimization problem, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options  : dict|None = None #: Options to pass to the CVXPY solver.
    timeout_seconds_expiry: float|None = 20.     #: Time limit for the optimization solver, in seconds. This is passed to the CVXPY solver options as "time_limit". 
    timeout_seconds_surface: float|None = 60.*5. #: Time limit for the optimization solver, in seconds. This is passed to the CVXPY solver options as "time_limit".
    mid_fit_weight        : float = 0.1  #: Weight for fitting to the mid prices of options, where 1 means mid-price fit only is active, while 0 means to fit only a linear penalty for exceeding bid-ask.
    min_density           : float = 1E-4 #: Minimum density value to consider in the optimization. Strikes with lower values will be dropped.
    eps_dens_1            : float = 1E-4 #: What passes as ``sum(p)==1`` and ``p@K==1`` for the fitted density.

    def __post_init__(self):
        verify_inp( self.no_arb_mode in ["off", "expiry", "surface"], lambda : f"'no_arb_mode' must be one of 'off', 'expiry' or 'surface', but found '{self.no_arb_mode}'" ) 
        verify_inp( self.min_strikes_per_expiry >= 3, lambda : f"'min_strikes_per_expiry' must be at least 3, but found {self.min_strikes_per_expiry}" )
        verify_inp( self.min_dC_eps > 0., lambda : f"'min_dC_eps' must be positive, but found {self.min_dC_eps}" )
        verify_inp( self.min_half_spread > 0., lambda : f"'min_half_spread' must be positive, but found {self.min_half_spread}" )
        verify_inp( self.min_density > 0., lambda : f"'min_density' must be positive, but found {self.min_density}" )
        verify_inp( self.timeout_seconds_expiry is None or self.timeout_seconds_expiry >= 0., lambda : f"'timeout_seconds_expiry' must be positive or None, but found {self.timeout_seconds_expiry}" )
        verify_inp( self.timeout_seconds_surface is None or self.timeout_seconds_surface >= 0., lambda : f"'timeout_seconds_surface' must be positive or None, but found {self.timeout_seconds_surface}" )
        verify_inp( self.smoothness is None or self.smoothness >= 0., lambda : f"'smoothness' must be positive or None, but found {self.smoothness:.4g}")

        if self.cvxpy_solver is None:
            verify_inp( self.cvxpy_solver_options is None, lambda : f"If 'cvxpy_solver' is None, then 'cvxpy_solver_options' must also be None" )
            self.__dict__["cvxpy_solver"] = "HIGHS"

            highs_options = {
                "solver": "simplex",
                "presolve": "on",
                "primal_feasibility_tolerance": 1e-6,
                "dual_feasibility_tolerance": 1e-6,
                }
            
            self.__dict__["cvxpy_solver_options"] = { "highs_options":  highs_options, "warm_start": True }
            assert self.cvxpy_solver == "HIGHS"

            supported_solvers = ["HIGHS", "GLOP", "PDLP", "CBC", "CLARABEL", "EXPRESS"]
            if not self.timeout_seconds_expiry is None or not self.timeout_seconds_surface is None:
                verify_inp( self.cvxpy_solver in supported_solvers, lambda : f"'cvxpy_solver' must be one of {supported_solvers} or None, but found '{self.cvxpy_solver}'" )

    def __str__(self) -> str:
        """ Very compact but still readable summary """
        return f"md={self.no_arb_mode[0]},{f'std={self.smoothness:.2g},' if self.smoothness is not None else ''}mink={self.min_strikes_per_expiry},dC={self.min_dC_eps:.3g},mhs={self.min_half_spread:.3g},mdn={self.min_density:.3g},"+\
               f"mw={self.mid_fit_weight:.2f},toe={self.timeout_seconds_expiry if not self.timeout_seconds_expiry is None else '0'},"+\
               f"eps={self.eps_dens_1:.1g}"

    def __unique_hash__( self, unique_hash, debug_trace ) -> str:
        """ Overwrite hash function to ignore non-functional parameters """
        r = asdict(self)
        r.pop("cvxpy_solver", None)
        r.pop("cvxpy_solver_options", None)
        r.pop("timeout_seconds_expiry", None)
        r.pop("timeout_seconds_surface", None)
        return unique_hash(r, debug_trace=debug_trace)

    def get_solver_and_solver_options(self, for_expiry : bool) -> tuple[str, dict]:
        """ Get the CVXPY solver options, including the time limit if set. """

        solver = self.cvxpy_solver

        if solver is None:
            solver  = "HIGHS"
            options = dict( warm_start=True,
                            highs_options={
                            "solver": "simplex",
                            "presolve": "on",
                            "parallel": "on",
                            "threads": 0,
                            "simplex_strategy": 1,   # dual simplex, serial; HiGHS default
                            "kkt_tolerance": 1e-7})
        else:
            options = dict(self.cvxpy_solver_options)        

        timeout_seconds = self.timeout_seconds_expiry if for_expiry else self.timeout_seconds_surface
        if timeout_seconds is not None:
            # https://www.cvxpy.org/tutorial/solvers/index.html
            if self.cvxpy_solver in ["GLOP", "PDLP"]:
                options["time_limit_sec"] = timeout_seconds
            elif self.cvxpy_solver == "CBC":
                options["maximumSeconds"] = timeout_seconds
            elif self.cvxpy_solver == "HIGHS":
                options["time_limit"] = timeout_seconds
            elif self.cvxpy_solver == "CLARABEL":
                options["time_limit"] = timeout_seconds
            elif self.cvxpy_solver == "EXPRESS":
                options["maxtime"] = timeout_seconds
            else:
                raise ValueError(f"Solver '{self.cvxpy_solver}' do not know how to set time limit for this solver. Supported solvers are 'HIGHS', 'GLOP', 'PDLP', 'CBC', 'CLARABEL' and 'EXPRESS'.")
        return solver, options

    def __call__(self,
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                bids           : list[np.ndarray],
                asks           : list[np.ndarray], *,
                vols           : np.ndarray|float = 0.1, # forward volatilities per expiry, used for the strike expansion step. If a float is provided, it is used for all expiries. If None, it is estimated from the data as the median of the implied vols.
                handle_arb     : str = "warn", # how to handle arbitrage situations: "warn", "raise" or "ignore"
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

        This function returns a dictionary with the following data:

        1) The function may drop some expiries. The resulting expiries are returned as:

            * ``expiries``: array of expiries. This may be fewer expiries than provided as input.

            * ``input_index``: for each returned expiry, the index of the input expiry which it corresponds to. In other words ``output_expiries == [ input_expiries[i] for i in input_index ]``.

        2) The function may also drop some strikes. This is reflected as follows:    

            * ``masks``: a list of boolean masks by expiry, indicating which strikes have been included in the no-arbitrage calculation (True) and which have been dropped (False).              

            * ``fits1``: list of per-expiry fitted arbitrage-free prices for the input ``strikes``. This will be within bid/ask wherever ``mask`` is True.
              (This means this mid is not necessarily arbitrage-free across expiries.)

              This field is None if ``no_arb_mode`` is "off".

            * ``fitsX``: list of surface fitted arbitrage-free prices for the input ``strikes``. 
              These prices are globally arbitrage-free across the surface, but they are not necessarily within bid/ask spread; they are the closest fit
              among strikes with ``mask`` True.

              Up to machine precision existence
              of ``fitsX`` within bid/ask spread is equivalent to absence of arbitrage.

              This field is None if ``no_arb_mode`` is "off".

            * ``xstrikes``: an extended strike list with two boundary strikes add. Boundary strikes are widening in time; c.f. https://arxiv.org/pdf/2601.11209 remark 2.7.

            * ``xdensities``: list of densities for ``xstrikes``, per expiry. Up to machine precision, each density is positive, sums to one, intergates to one 
              in the sense that `` xstrikes[t]@xdensities[t] == 1``
              and reprices ``fits``.

              This field is None if ``no_arb_mode`` is "off".

        3) If ``no_arb_mode`` is "surface" then the function will generate an internal ``mstrike`` grid to fit the surface, which is a superset of the extended ``xstrike`` grid.
           This is constructed by adding strikes from the next expiry if they are not too close to existing strikes.
           This is typically a much denser set than the input strikes. Each expiry is guaranteed to have their ``xstrikes`` among the respective model strikes.

            * ``mstrikes``: the list of internal model strikes per expiry used. This field is None if ``no_arb_mode`` is not "surface".

            * ``mdensities```: the list of densities for ``mstrikes``. Up to machine precision, each density is positive, sums to one, intergates to one 
              in the sense that `` mstrikes[t]@mdensities[t] == 1``
              and reprices ``fits``. Moreover, the family of ``mdensities`` is martingale density, each later density evaluaes a convex function 
              higher than the a previous density.
            
              This field is None if ``no_arb_mode`` is not "surface".

            * ``xix_to_mix``: a list of arrays which gives for every ``xstrike[t]`` the position in ``mstrikes[t]``, i.e. ``mstrikes[t][xix_to_mix[t]] == xstrikes[t]``.
              The element ``xix_to_mix[t]`` is ``None`` if no additional strikes were added to expiry ``t``, i.e. the strikes coincide.
                          
              This field is None if ``no_arb_mode`` is not "surface".

        **Comment on arbitrage-free prices**

        If ``C`` are call prices for strikes ``K`` including boundary strikes, i.e. ``C[0]=1-K[0]`` and ``C[-1]=0``, then ``p`` computed as follows is a density which reprices
        all options if and only ``C`` is arbitrage-free::
        
            dCdK_t         = np.diff(C,prepend=1.)/np.diff(K,prepend=0.)
            p              = np.diff(dCdK_t,append=0.)
            assert np.all( p >= -eps )          # positive
            assert abs( np.sum( p )-1. ) <= eps # sums to one
            assert abs( p@K - 1. ) <= floor     # expectation one
            assert abs( C - np.maximum( 0., K[None,:] - K[:,None] ) @ p ) <= eps # reprices all options


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
            Forward volatilities per expiry, used for the strike expansion step. If a float is provided, it is used for all expiries. If None, it is estimated from the data as the median of the implied vols.
        handle_arb : str, default ``"warn"``
            What to do if the function cannot fit an arbitrage-free mid for a given single expiry.
            The options are to ``"ignore"`` it, ``"warn"`` about it, or ``"raise"`` an error.
            If it is ignored or merely warned, then the offending expiry will be dropped from the output.
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
        def msg_header():
            if message_context is None:
                return ""
            if callable(message_context):
                return f"{message_context()}: "
            return f"{message_context}: "

        T = len(expiries)

        verify_inp( len(strikes) == T, lambda : f"{msg_header()}Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(bids) == T, lambda : f"{msg_header()}Expected number of bid arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(asks) == T, lambda : f"{msg_header()}Expected number of ask arrays {T} to match number of expiries {len(expiries)}" )   
        verify_inp( expiries[0] > 0. and (len(expiries)==1 or np.min(np.diff(expiries)) > 0.), lambda : f"{msg_header()}Expiries must be positive and strictly increasing, but found {expiries}" )
        verify_inp( handle_arb in ["warn", "raise", "ignore"], lambda : f"{msg_header()}handle_arb must be one of 'warn', 'raise' or 'ignore', but found '{handle_arb}'" )

        variances = None
        if self.smoothness is not None:
            if isinstance( vols, np.ndarray ):
                verify_inp( vols.shape == expiries.shape, lambda : f"{msg_header()}If 'vols' is a numpy array, it must have the same shape as 'expiries', but found {vols.shape} and {expiries.shape}" )
                verify_inp( np.all(vols > 0.), lambda : f"{msg_header()}All volatilities must be positive, but found {vols}" )
                variances = expiries * (vols**2) 
                verify_inp( variances[0] > 0. and (len(variances)==1 or np.min(np.diff(variances)) > 0.), lambda : f"{msg_header()}Variances derived from 'vols' must be positive and strictly increasing, but found {variances}" )
            else:
                variances = expiries * (float(vols)**2) 
                verify_inp(vols > 0., lambda : f"{msg_header()}Volatility must be positive, but found {vols}" )
        vols = None

        # generate strike range
        # ---------------------
        # Add strikes to the left and right which are consistent with the no-arb condition.

        r             = ArbFreePricer(self.smoothness)
        r.fits1       = None if self.no_arb_mode == "off" else []
        r.xdensities  = None if self.no_arb_mode == "off" else []
        r.variances   = variances
        tme           = verbose.timer()
        used_bids     = []
        used_asks     = []

        for t, expiry in enumerate(expiries):
            verbose.write(lambda : f"\r{msg_header()}Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: generating extended strikes... ", end='')

            if self.smoothness is None:
                vol_t = None
                def call( spots, strikes ):
                    return np.maximum( 0., spots - strikes )
            else:
                vol_t = math.sqrt( self.smoothness * np.interp( expiry, expiries, variances ) / expiry ) if not variances is None else None
                def call( spots, strikes ):
                    return bs.price( k=strikes/spots, vol=vol_t, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )*spots

            input_strikes_t = strikes[t]
            input_bid_t     = bids[t]
            input_ask_t     = asks[t]

            xstrikes_t, mask_t, notice = self.make_extended_strikes( input_strikes_t,
                                                                     input_bid_t, 
                                                                     input_ask_t, 
                                                                     prev_lower = r.xstrikes[-1][0] if len(r.xstrikes) > 0 else None,
                                                                     prev_upper = r.xstrikes[-1][-1] if len(r.xstrikes) > 0 else None,
                                                                     msg_header = lambda : f"{msg_header()}#{t} {expiry*DAYS_PER_YEAR:.1f}d: " )
            if not notice is None:
                verbose.write(notice, head=False, end='')

            if xstrikes_t is None:
                continue

            assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"
            bid_t = input_bid_t[mask_t]
            ask_t = input_ask_t[mask_t]

            if self.no_arb_mode != "off":
                # is it already fine?
                Nt          = len(xstrikes_t)
                floor       = max( self.min_density / float(Nt), 1E-6 )
                active      = None
                
                def market_p():
                    Cx          = np.zeros_like(xstrikes_t, dtype=np.float64)
                    Cx[1:-1]    = ((bid_t + ask_t)/2.).astype(np.float64, copy=False)
                    Cx[0]       = 1. - xstrikes_t[0].astype(np.float64, copy=False)
                    dCdK        = np.diff(Cx,prepend=1.)/np.diff(xstrikes_t,prepend=0.)
                    p           = np.diff(dCdK,append=0.)
                    return p
                p = market_p()

                if np.all(p >= -floor) and abs( np.sum( p )-1. ) <= self.eps_dens_1 and abs( p@xstrikes_t - 1. ) <= self.eps_dens_1 and self.smoothness is None:
                    verbose.write(lambda : f"\r{msg_header()}Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: mid is already arbitrage-free... ", end='')
                else:
                    verbose.write(lambda : f"\r{msg_header()}Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: finding linear no-arbitrage mid... ", end='')
                    active, p = self._expiry_no_arb(xstrikes_t,
                                            bid_t, 
                                            ask_t, 
                                            expiry=expiry, 
                                            guess_p=p,
                                            vol=vol_t,
                                            handle_arb=handle_arb, 
                                            retry=True, 
                                            message_context=msg_header, verbose=verbose)
                    if p is None:
                        continue # dropped expiry
                    assert active is None or (active[0] and active[-1]), "Internal error: boundary strikes should never be dropped since they are fixed and have positive density."

                assert p.shape == xstrikes_t.shape, f"Internal error: density shape {p.shape} does not match strike shape {xstrikes_t.shape} for expiry {expiry*DAYS_PER_YEAR:.1f}d."
                assert active is None or active.shape == xstrikes_t.shape, f"Internal error: active shape {active.shape} does not match strike shape {xstrikes_t.shape} for expiry {expiry*DAYS_PER_YEAR:.1f}d."
                assert np.all( p >= -floor ), f"Internal error: {msg_header()}negative density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.1f}d after removing zero-density strikes."
                assert abs( np.sum( p )-1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}density does not sum to one; sum is {p.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}]."
                assert abs( p@xstrikes_t - 1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}density does not integrate to one; integral is {p@xstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}] ."

                active     = p > floor if active is None else (active & (p > floor))
                active[0]  = True
                active[-1] = True

                if not np.all(active):
                    # do the following even if smoothness is > 0 assuming that the model prices are not affected.
                    Cx             = ( np.maximum( 0., xstrikes_t[None,:] - xstrikes_t[:,None] ) @ p ).astype(np.float64, copy=False)
                    Cx[0]          = 1. - xstrikes_t[0]
                    Cx[-1]         = 0.
                    mask_t[mask_t] = active[1:-1]
                    xstrikes_t     = xstrikes_t[active].astype(np.float64, copy=False)
                    assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"
                    Cx             = Cx[active]
                    dCdK           = np.diff(Cx,prepend=1.)/np.diff(xstrikes_t,prepend=0.)
                    p              = np.diff(dCdK,append=0. )
                    assert np.all( p >= -floor ), f"Internal error: {msg_header()}negative re-computed density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.1f}d after removing zero-density strikes."
                    assert abs( np.sum( p )-1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}re-computed density does not sum to one; sum is {p.sum():.6g} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}]."
                    assert abs( p@xstrikes_t - 1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}re-computed density does not integrate to one; integral is {p@xstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}]."

                # compute true fit values from the density
                fit_t      = call( spots=xstrikes_t[None,:], strikes=input_strikes_t[:,None] ) @ p

                r.xdensities.append( p )
                r.fits1.append( fit_t )   # fit is computed for all strikes, also those unmasked

            assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"
            r.xstrikes.append( xstrikes_t )
            r.expiries.append( expiry )
            r.input_index.append( t )
            r.masks.append( mask_t )

            used_bids.append( input_bid_t )
            used_asks.append( input_ask_t )

        r.expiries = np.array(r.expiries, dtype=np.float32)
        verbose.write(lambda : f"\r{msg_header()}Generated {len(r.expiries)} {'linear' if self.smoothness is None else f'{self.smoothness:.2f}-smooth'} {'arbitrage-free ' if self.no_arb_mode != 'off' else ''}expiries with extended strikes from {T} market expiries. This took {tme}.")

        if self.no_arb_mode != "surface":
            return r
        
        # backward-dense grid
        # -------------------
        # Carry grid forward as long as the strike range isn't too tight.

        T            = len(r.expiries)
        r.mstrikes   = [None]*T
        r.xix_to_mix = [None]*T
        r.fitsX      = []
        min_dx       = None
        mguess_p     = [None]*T

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
                assert next_mstrikes[0] <= xstrikes_t[0] and next_mstrikes[-1] >= xstrikes_t[-1], f"Internal error: {msg_header()}next_mstrikes[0] {next_mstrikes[0]:.3f} > xstrikes_t[0] {xstrikes_t[0]:.3f} or next_mstrikes[-1] {next_mstrikes[-1]:.3f} < xstrikes_t[-1] {xstrikes_t[-1]:.3f}"

                min_dx   = min( np.diff(mstrikes_t).min(), np.diff(next_mstrikes).min() )
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

                assert len(collect) == len(mgpcl), f"Internal error: {msg_header()}collect length {len(collect)} does not match mgpcl length {len(mgpcl)}"
                if add_any:
                    mstrikes_t = np.array(collect, dtype=xstrikes_t.dtype)
                    mp_t       = np.array(mgpcl, dtype=np.float64)
                    assert np.min( np.diff( mstrikes_t ) ) >= min_dx, f"Internal error: {msg_header()}min dx {np.diff(mstrikes_t).min():.3f} < min_dx {min_dx:.3f}"
                    assert mstrikes_t[0] == xstrikes_t[0] and mstrikes_t[-1] == xstrikes_t[-1], f"Internal error: mstrikes_t[0] {mstrikes_t[0]:.3f} != xstrikes_t[0] {xstrikes_t[0]:.3f} or mstrikes_t[-1] {mstrikes_t[-1]:.3f} != xstrikes_t[-1] {xstrikes_t[-1]:.3f}"
                    project = np.array( project, dtype=np.int32 )
                else:
                    project = None

            assert np.allclose( xstrikes_t, mstrikes_t[project] ), f"Internal error: {msg_header()}projection error, xstrikes_t {xstrikes_t} should be equal to mstrikes_t[project] {mstrikes_t[project]}"
            r.mstrikes[t]   = mstrikes_t
            r.xix_to_mix[t] = project
            mguess_p[t]     = mp_t

        # solve surface no-arb
        # --------------------

        r.mdensities = []
        fit_mids     = []
        constraints  = []
        objective    = 0.

        min_dx_text = "" if min_dx is None else f" with a minimum strike distance of {min_dx:.3g}"
        with verbose.write_t(f"{msg_header()}Finding {'linear' if self.smoothness is None else f'{self.smoothness:.2f}-smooth'} arbitrage-free mid prices accross the entire surface of {sum( len(xt) for xt in r.xstrikes )} strikes{min_dx_text}... ", end='') as tme:
            for t in range(T):
                expiry      = r.expiries[t]
                mstrikes_t  = r.mstrikes[t]
                xstrikes_t  = r.xstrikes[t]
                mask_t      = r.masks[t]
                ask_t       = used_asks[t][mask_t]
                bid_t       = used_bids[t][mask_t]
                mp_guess_t  = mguess_p[t]
                assert len(bid_t) == len(ask_t) == xstrikes_t.shape[0]-2, f"Internal error: {msg_header()}bid and ask length {len(bid_t)} does not match number of market strikes {xstrikes_t.shape[0]-2}"

                if self.smoothness is None:
                    def call( spots, strikes ):
                        return np.maximum( 0., spots - strikes )
                else:
                    vol = math.sqrt( self.smoothness * np.interp( expiry, expiries, variances ) / expiry )
                    assert np.isfinite(vol), "Internal error: infinite vol"
                    def call( spots, strikes ):
                        return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )*spots

                q_t         = cp.Variable(len(mstrikes_t), nonneg=True)
                q_t.value   = mp_guess_t 
                fit_mid     = call( spots=mstrikes_t[None,:], strikes=xstrikes_t[1:-1,None] ) @ q_t    # market strikes
                Cm          = np.maximum( 0., mstrikes_t[None,:] - mstrikes_t[1:-1,None] ) @ q_t    # model strikes - do not use "call"

                if t>0:
                    assert r.mstrikes[t-1].shape == r.mdensities[t-1].shape, f"Internal error: {msg_header()}shape mismatch between previous and current mstrikes, {r.mstrikes[t-1].shape} vs {r.mdensities[t-1].shape}"
                    Cm1     = np.maximum( 0., r.mstrikes[t-1][None,:] - mstrikes_t[1:-1,None] ) @ r.mdensities[t-1]
                else:
                    Cm1     = np.maximum( 0., 1 - mstrikes_t[1:-1] )

                constraints += [ cp.sum( q_t ) == 1.,
                                 q_t @ mstrikes_t == 1., # martingale condition
                                 q_t <= 1.,
                                 Cm >= Cm1
                                  ]

                half_spread = np.maximum( (ask_t - bid_t)/2., self.min_half_spread )
                mid         = (bid_t + ask_t)/2.
                excess      = cp.multiply( 1./half_spread, ( cp.maximum( 0., fit_mid - ask_t ) + cp.maximum( 0., bid_t - fit_mid ) ) )
                obj_exc     = cp.sum( (1.-self.mid_fit_weight) * excess ) if self.mid_fit_weight < 1. else 0.
                obj_sqr     = cp.sum( self.mid_fit_weight*cp.max( cp.multiply( 1./half_spread, fit_mid - mid ) ) ) if self.mid_fit_weight > 0. else 0.
                objective   = objective + obj_sqr + obj_exc

                fit_mids.append(fit_mid)
                r.mdensities.append(q_t)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",message=r"Solution may be inaccurate.*",category=UserWarning)
                warnings.filterwarnings("error",message=r"overflow encountered in power",category=RuntimeWarning)
                try:
                    prob        = cp.Problem( cp.Minimize( objective ), constraints )
                    slv, opt    = self.get_solver_and_solver_options(for_expiry=False)
                    prob.solve( solver=slv, **opt )
                except RuntimeWarning as e:
                    warnings.warn(f"{msg_header()}Solver warning: {str(e)}.", category=RuntimeWarning)

            if prob.status != "optimal":
                if prob.status == "infeasible":
                    raise RuntimeError(f"{msg_header()}Cannot find no-arbitrage surface mids: problem infeasible??")
                if prob.status == "user_limit":
                    raise RuntimeError(f"{msg_header()}Cannot find no-arbitrage surface mids: reached timeout {self.timeout_seconds_surface} and/or iteration limit.")
                verbose.write(lambda : f"solver returned '{prob.status}' -- proceeding anyway... ", head=False, end='' )

            # compute call prices for the original strikes
            r.fitsX = []
            r.mdensities = [ q.value for q in r.mdensities ]
            for t in range(T):
                input_t    = r.input_index[t]
                strikes_t  = strikes[input_t]
                mstrikes_t = r.mstrikes[t]
                q_t        = r.mdensities[t]
                expiry     = r.expiries[t]
                mask_t     = r.masks[t]
                project_t  = r.xix_to_mix[t]

                if self.smoothness is None:
                    def call( spots, strikes ):
                        return np.maximum( 0., spots - strikes )
                else:
                    vol = math.sqrt( self.smoothness * np.interp( expiry, expiries, variances ) / expiry )
                    assert np.isfinite(vol), "Internal error: infinite vol"
                    def call( spots, strikes ):
                        return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )*spots

                assert np.min(q_t) >= -floor, f"Internal error: {msg_header()}negative density {q_t[q_t<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.1f}d after optimization."
                assert abs( np.sum( q_t )-1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}density does not sum to one after removing zero-density strikes, sum is {q_t.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}]."
                assert abs( q_t@mstrikes_t - 1. ) <= self.eps_dens_1, f"Internal error: {msg_header()}density does not integrate to one after removing zero-density strikes, integral is {q_t@mstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.1f}d [tolerance {self.eps_dens_1:.3g}]."

                # zero out some density
                is_zero    = q_t < _EPS_DENSITY
                if np.any(is_zero):
                    # set density to zero wherever it is close.
                    q_t[is_zero]    = 0.
                    q_t             = _warn_and_fix_density( mstrikes_t, q_t, _EPS_DENSITY, lambda : f"{msg_header()} expiry {expiry*DAYS_PER_YEAR:.2f}d: surface density", verbose=verbose )
                    r.mdensities[t] = q_t

                # consistency check
                xfit_mids  = fit_mids[t].value  
                fit_t      = call( spots=mstrikes_t[None,:], strikes=strikes_t[:,None] ) @ q_t
                assert np.allclose( fit_t[mask_t], xfit_mids, rtol=0., atol=1E-6), f"Internal error: {msg_header()}fit values computed from density do not match optimization output for expiry {expiry*DAYS_PER_YEAR:.1f}d. Worst fit {np.max(np.abs(fit_t[mask_t] - xfit_mids)):.3e}"

                r.fitsX.append( fit_t )
            verbose.write(lambda : f"done. This took {tme}.", head=False )

        # make numpy
        r._make_numpy()
        return r

    def _expiry_no_arb(self, xstrikes_t,
                             bid_t, 
                             ask_t, 
                             expiry : float, 
                             vol : float,
                             guess_p : np.ndarray, handle_arb : str, retry : bool, *, message_context : str|callable|None = None, verbose : Context = Context.all):
        """
        Iterative way to get some arb-free fit going
        """
        def msg_header():
            if message_context is None:
                return ""
            if callable(message_context):
                return f"{message_context()}: "
            return f"{message_context}: "

        # find fit between bid/ask without hard constraints first
        # -------------------------------------------------------

        if self.smoothness is None:
            assert vol is None, "Internal error - vol should be None"
            def call( spots, strikes ):
                return np.maximum( 0., spots - strikes )
        else:
            assert vol is not None, "Internal error - vol should not be None"
            def call( spots, strikes ):
                return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=_EPS_PRICE )*spots
            
        Nt          = len(xstrikes_t)
        floor       = max( self.min_density / float(Nt), 1E-6 )
        p           = cp.Variable(Nt, nonneg=True)
        p.values    = guess_p
        C           = call( spots=xstrikes_t[None,:], strikes=xstrikes_t[1:-1,None] ) @ p
        intr        = np.maximum(0., 1.-xstrikes_t[1:-1]) 
        assert np.min( ask_t - intr ) > 0., f"Internal error: {msg_header()} 'ask' values must be strictly above intrinsic values for expiry {expiry*DAYS_PER_YEAR:.1f}d, but found minimum difference {np.min(ask_t - intr):.3e}."
        constraints = [ cp.sum(p) == 1., xstrikes_t @ p == 1., # martingale condition
                        p <= 1.,
                        C >= intr, 
                        C <= intr + (ask_t-intr)*10. ]

        half_spread = np.maximum( (ask_t - bid_t)/2., self.min_half_spread )
        error       = cp.abs( cp.multiply( 1./half_spread, (C - (bid_t + ask_t)/2.) ) ) if self.mid_fit_weight > 0. else 0.
        excess      = cp.multiply( 1./half_spread, ( cp.maximum( 0., C - ask_t ) + cp.maximum( 0., bid_t - C ) ) ) if self.mid_fit_weight < 1. else 0.
        objective   = cp.sum( self.mid_fit_weight * error + (1.-self.mid_fit_weight) * excess )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message=r"Solution may be inaccurate.*",category=UserWarning)
            warnings.filterwarnings("error",message=r"overflow encountered in power",category=RuntimeWarning)
            try:
                prob        = cp.Problem( cp.Minimize( objective ), constraints )
                slv, opt    = self.get_solver_and_solver_options(for_expiry=True)
                prob.solve( solver=slv, **opt )
            except RuntimeWarning as e:
                raise RuntimeError(f"{msg_header()}Solver overflow for expiry {expiry*DAYS_PER_YEAR:.2f}d in soft fit mode: {str(e)}.")

        if prob.status in ["infeasible", "infeasible_inaccurate"]:
            raise RuntimeError(f"{msg_header()}Cannot find no-arbitrage mid for expiry {expiry*DAYS_PER_YEAR:.2f}d in soft fit mode: problem reported as infeasible. Please contact the authors with an example.")

        if prob.status in ["user_limit"]:
            msg = f"Cannot find soft no-arbitrage mid for expiry {expiry*DAYS_PER_YEAR:.2f}d: solver hit a timeout after {self.timeout_seconds_expiry:.2f}s."
            if handle_arb == "raise":
                raise RuntimeError(msg_header() + msg)
            verbose.write(f"{msg.lower()} -- removing expiry.", head=False)
            if handle_arb == "warn":
                warnings.warn(msg_header() + msg[:-1] + " -- removing expiry.") # -> maybe we need a proper error here.
            return None, None

        # check whether we succeeded in fitting within bid/ask;
        # if yes -> done

        C      = C.value
        p      = p.value
        mask   = (C >= bid_t-1E-6) | (C <= ask_t+1E-6)

        if np.all(mask):
            return None, _warn_and_fix_density( p=p, x=xstrikes_t, floor=floor, message=lambda : f"{msg_header()} expiry {expiry*DAYS_PER_YEAR:.2f}d: market density", verbose=verbose )

        if np.sum(mask) < 3:
            msg = f"Cannot find arbitrage-free fit within bid/ask even in soft mode for expiry {expiry*DAYS_PER_YEAR:.2f}d."
            if handle_arb == "raise":
                raise RuntimeError(msg_header() + msg)
            verbose.write(f"{msg.lower()} -- removing expiry.", head=False)
            if handle_arb == "warn":
                warnings.warn(msg_header() + msg[:-1] + " -- removing expiry.") # -> maybe we need a proper error here.
            return None, None

        soft_p      = p
        
        # attempt hard fit on all strikes
        # -------------------------------

        p           = cp.Variable(Nt, nonneg=True)
        p.value     = soft_p
        C           = call( spots=xstrikes_t[None,:], strikes=xstrikes_t[1:-1,None] ) @ p
        constraints = [ cp.sum(p) == 1., xstrikes_t @ p == 1., # martingale condition
                        p <= 1.,
                        C <= ask_t,
                        C >= bid_t ]

        half_spread = np.maximum( (ask_t - bid_t)/2., self.min_half_spread )
        objective   = cp.sum( cp.multiply( 1./half_spread, C - (bid_t + ask_t)/2. )**2 )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message=r"Solution may be inaccurate.*",category=UserWarning)
            warnings.filterwarnings("error",message=r"overflow encountered in power",category=RuntimeWarning)
            try:
                prob        = cp.Problem( cp.Minimize( objective ), constraints )
                slv, opt    = self.get_solver_and_solver_options(for_expiry=True)
                prob.solve( solver=slv, **opt )
            except RuntimeWarning as e:
                prob.status = "overflow error"

        if not prob.status in ["infeasible", "infeasible_inaccurate","user_limit","overflow error"]:
            return None, _warn_and_fix_density( p=p.value, x=xstrikes_t, floor=floor, message=lambda : f"{msg_header()} expiry {expiry*DAYS_PER_YEAR:.2f}d: no-arb density", verbose=verbose )

        # Return soft-fit density, masked for the strikes it worked
        # ---------------------------------------------------------

        # shoud actually work out of the box
        mask[0]     = True
        mask[-1]    = True

        if __debug__:
            mstrikes_t  = xstrikes_t[mask]
            mask_t      = ask_t[mask[1:-1]]
            mbid_t      = bid_t[mask[1:-1]]
            mp          = soft_p[mask]
            C           = call( spots=mstrikes_t[None,:], strikes=mstrikes_t[1:-1,None] ) @ mp
            assert np.all( C >= mbid_t-1E-5 ) and np.all( C <= mask_t+1E-5 ), f"Internal error: {msg_header()}soft fit succeeded but constraints violated for expiry {expiry*DAYS_PER_YEAR:.2f}d. Please contact the authors with an example."

        verbose.write(lambda : f"hard fit failed with status '{prob.status}' -- using masked soft fit instead... ", head=False, end='')
        return mask, _warn_and_fix_density( p=p, x=xstrikes_t, floor=floor, message=lambda : f"{msg_header()} expiry {expiry*DAYS_PER_YEAR:.2f}d: soft no-arb density", verbose=verbose )

        """
        verbose.write(lambda : f"hard fit failed with status '{prob.status}' -- trying to fit a subset of strikes... ", head=False, end='')

        mask[0]     = True
        mask[-1]    = True
        mstrikes_t  = xstrikes_t[mask]
        mask_t      = ask_t[mask[1:-1]]
        mbid_t      = bid_t[mask[1:-1]]
        mNt         = len(mstrikes_t)
        mp          = cp.Variable(mNt, nonneg=True)
        mp.value    = soft_p[mask]
        mC          = call(spots=mstrikes_t[None,:], strikes=mstrikes_t[1:-1,None] ) @ mp
        constraints = [ cp.sum(mp) == 1., mstrikes_t @ mp == 1., # martingale condition
                        mp <= 1.,
                        mC <= mask_t,
                        mC >= mbid_t ]

        half_spread = np.maximum( (mask_t - mbid_t)/2., self.min_half_spread )
        objective   = cp.sum( cp.multiply( 1./half_spread, mC - (mbid_t + mask_t)/2. )**2 )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message=r"Solution may be inaccurate.*",category=UserWarning)
            warnings.filterwarnings("error",message=r"overflow encountered in power",category=RuntimeWarning)
            try:
                prob        = cp.Problem( cp.Minimize( objective ), constraints )
                slv, opt    = self.get_solver_and_solver_options(for_expiry=True)
                prob.solve( solver=slv, **opt )
            except RuntimeWarning as e:
                prob.status = "overflow error"

        if prob.status in ["infeasible", "infeasible_inaccurate","user_limit","overflow error"]:
            msg = f"Cannot find no-arbitrage mid for expiry {expiry*DAYS_PER_YEAR:.2f}d, even after removing some offending strikes. Solver returned status '{prob.status}'."
            if handle_arb == "raise":
                raise RuntimeError(msg_header() + msg)
            verbose.write(f"{msg.lower()} -- removing expiry.", head=False)
            if handle_arb == "warn":
                warnings.warn(msg_header() + msg[:-1] + " -- removing expiry.") # -> maybe we need a proper error here.
            return None
        
        p       = np.zeros_like(xstrikes_t)
        p[mask] = mp.value
        C       = call( spots=xstrikes_t[None,:], strikes=xstrikes_t[1:-1,None] ) @ p
        assert np.all( C[mask] >= bid_t[mask]-1E-6 ) and np.all( C[mask] <= ask_t[mask]+1E-6 ), f"Internal error: {msg_header()}hard fit succeeded but constraints violated for expiry {expiry*DAYS_PER_YEAR:.2f}d. Please contact the authors with an example."
                
        # Check that that density is reasonable
        p[mask] = mp.value
        return mask, _warn_and_fix_density( p=p, x=xstrikes_t, floor=floor, message=lambda : f"{msg_header()} expiry {expiry*DAYS_PER_YEAR:.2f}d: soft no-arb density", verbose=verbose )
        """
    def make_extended_strikes( self, 
                              strikes: np.ndarray, 
                              bids: np.ndarray, 
                              asks: np.ndarray, *, 
                              prev_lower : float|None = None,
                              prev_upper : float|None = None,                            
                              msg_header : callable|None = None ) -> tuple[np.ndarray, np.ndarray, str|None]:
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
        msg_header : callable | None, default ``None``
            Optional callable that returns a string to be used as a header in error messages.
            This can be used to provide context such as the expiry time and other relevant information.
            Mostly used for use within :class:`MkArbFree`.

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
        input_strikes_t = strikes
        input_bid_t     = bids
        input_ask_t     = asks
        notice          = None
        xstrikes_t      = input_strikes_t
        bid_t           = input_bid_t
        ask_t           = input_ask_t
        mid_t           = (input_bid_t + input_ask_t)/2.
        mask_t          = np.ones(input_strikes_t.shape[0], dtype=bool)

        if len(xstrikes_t) <= self.min_strikes_per_expiry:
            return None, None, lambda :f"only {len(xstrikes_t)} strikes found. Skipping this expiry."

        verify_inp( xstrikes_t.shape == bid_t.shape == ask_t.shape, lambda : f"{msg_header() if not msg_header is None else ''}Expected strikes, bids and asks to have the same shape for expiry, found {xstrikes_t.shape}, {bid_t.shape}, {ask_t.shape}")
        verify_inp( np.all( np.diff(xstrikes_t) > 0 ), lambda : f"{msg_header() if not msg_header is None else ''}Strikes must be strictly increasing; found minimal difference {np.diff(xstrikes_t).min():.3f}" )
        verify_inp( xstrikes_t[0] > 0., lambda : f"{msg_header() if not msg_header is None else ''}Lowest strike must be positive; found {xstrikes_t[0]:.3f}" )
        verify_inp( np.all( ask_t - bid_t >0. ), lambda : f"{msg_header() if not msg_header is None else ''}All ask prices must be greater than bid prices.")

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
            raise RuntimeError(f"{msg_header() if not msg_header is None else ''}Could not find consistent initial strike for this expiry.")
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
            raise RuntimeError(f"{msg_header() if not msg_header is None else ''}Could not find consistent terminal strike for this expiry.")

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
           notice = lambda removed_count=removed_count, total_input=total_input: f"removed {removed_count} out of {total_input} strikes to ensure -1 < dC/dK[0] and dC/dK[-1] < 0."

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

def _project_simplex(v: np.ndarray, total: float) -> np.ndarray:
    """
    Euclidean projection of v onto { y >= 0, sum(y) = total }.
    """
    verify_inp(total > 0.0, "total must be positive; got %s", total)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u, dtype=np.float64) - total
    js = np.arange(1, v.size + 1, dtype=np.float64)

    ok = u - cssv / js > 0.0
    verify_inp(np.any(ok), "simplex projection failed")

    rho = np.flatnonzero(ok)[-1]
    theta = cssv[rho] / float(rho + 1)

    return np.maximum(v - theta, 0.0)


def repair_density_nonnegative(
    p: np.ndarray,
    x: np.ndarray,
    *,
    target_sum: float = 1.0,
    target_xmoment: float = 1.0,
    floor: float = 0.0,
    tol: float | None = None,
    max_iter: int = 64,
) -> np.ndarray:
    """
    Project p onto:

        y >= floor
        sum(y) = target_sum
        x @ y = target_xmoment

    For floor=0 this gives a nonnegative density.

    This is the L2-closest feasible y to p.
    """

    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    verify_inp(p.ndim == 1, "p must be 1D; got shape %s", p.shape)
    verify_inp(x.ndim == 1, "x must be 1D; got shape %s", x.shape)
    verify_inp(p.shape == x.shape, "p and x must have same shape; got %s and %s", p.shape, x.shape)
    verify_inp(np.all(np.isfinite(p)), "p must be finite")
    verify_inp(np.all(np.isfinite(x)), "x must be finite")
    verify_inp(target_sum > 0.0, "target_sum must be positive; got %s", target_sum)
    verify_inp(floor >= 0.0, "floor must be nonnegative; got %s", floor)

    n = p.size

    # Transform y >= floor into r >= 0 with y = floor + r.
    residual_sum = target_sum - n * floor
    residual_xmoment = target_xmoment - floor * float(np.sum(x, dtype=np.float64))

    verify_inp( residual_sum > 0.0, lambda : f"floor too large: target_sum - n*floor = {residual_sum}" )

    q = p - floor
    target_mean = residual_xmoment / residual_sum

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    verify( xmin <= target_mean <= xmax, lambda : f"no nonnegative solution: target mean {target_mean} is outside [{xmin}, {xmax}]" )

    if tol is None:
        tol = (
            256.0
            * np.finfo(np.float64).eps
            * max(1.0, residual_sum)
            * max(1.0, float(np.max(np.abs(x))))
            * max(1.0, np.sqrt(float(n)))
        )

    z = x - target_mean

    def eval_beta(beta: float):
        # For fixed beta, project q - beta*z onto the simplex.
        r = _project_simplex(q - beta * z, residual_sum)

        # Need z @ r == 0, equivalent to x @ r == residual_xmoment.
        h = float(z @ r)

        active = r > 0.0
        za = z[active]

        # derivative dh/dbeta on the current active set
        if za.size >= 2:
            dz = za - za.mean(dtype=np.float64)
            dh = -float(dz @ dz)
        else:
            dh = 0.0

        return r, h, dh

    # Initial point.
    r, h, dh = eval_beta(0.0)

    if abs(h) <= tol:
        return floor + r

    # h(beta) is monotone decreasing.
    if h > 0.0:
        lo = 0.0
        hi = 1.0

        _, h_hi, _ = eval_beta(hi)
        while h_hi > 0.0:
            hi *= 2.0
            verify(hi < 1e16, lambda : f"failed to bracket beta: hi = {hi}")
            _, h_hi, _ = eval_beta(hi)

    else:
        hi = 0.0
        lo = -1.0

        _, h_lo, _ = eval_beta(lo)
        while h_lo < 0.0:
            lo *= 2.0
            verify(abs(lo) < 1e16, lambda : f"failed to bracket beta: lo = {lo}")
            _, h_lo, _ = eval_beta(lo)

    beta = 0.5 * (lo + hi)

    for _ in range(max_iter):
        r, h, dh = eval_beta(beta)

        if abs(h) <= tol:
            y = floor + r
            return y

        if h > 0.0:
            lo = beta
        else:
            hi = beta

        # Bracketed Newton step.
        if dh < 0.0:
            beta_new = beta - h / dh
        else:
            beta_new = 0.5 * (lo + hi)

        if not (lo < beta_new < hi):
            beta_new = 0.5 * (lo + hi)

        beta = beta_new

    r, h, _ = eval_beta(beta)

    verify( abs(h) <= 10.0 * tol, lambda : f"projection did not converge; residual moment error is {h}" )
    return floor + r

def _warn_and_fix_density( x, p, floor, message : callable, verbose : Context = Context.all  ):
    """ Wrapper around fixing and warning """

    eps = 1E-7
    p   = np.asarray(p, dtype=np.float64)
    x   = np.asarray(x, dtype=np.float64)

    assert np.min(p) > -1E-3, f"{message()}: density has minimum {np.min(p):.6g} < -1E-3, which is too negative to be repaired."
    assert abs(np.sum(p)-1.) < 1E-4, f"{message()}: density sums to {np.sum(p):.6g}, which is too close to zero to be repaired."
    assert abs(p@x-1.) < 1E-4, f"{message()}: density integrates to {p@x:.6g}, which is too close to zero to be repaired."

    if not np.all( p >= -eps ) or abs(np.sum(p)-1.) > eps or abs(p@x-1.) > eps:
        p = repair_density_nonnegative(p=p, x=x)

    assert np.all( p >= -eps ), f"{message()}: negative density {np.min(p):.6g} found after repair."
    assert abs(np.sum(p)-1.) <= eps, f"{message()}: density sums to {np.sum(p):.6g} after repair."
    assert abs(p@x-1.) <= eps, f"{message()}: density sums to {p@x:.6g} after repair."
    return p
