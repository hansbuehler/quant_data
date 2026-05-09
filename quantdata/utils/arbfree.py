
"""
Make arbfree markets
^^^^^^^^^^^^^^^^^^^^

This module contains :class:`MkArbFree`, a utility class which generates a "closest" arbitrage free market using https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2642630.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings as warnings

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from cdxcore.err import verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.pretty import PrettyObject
from pyparsing import col
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.utils.backbone import BackboneNumpy

@version("0.1")
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

    min_strikes_per_expiry: int = 5      #: Minimum strikes per expiry. Expiries with fewer strikes will be dropped.
    min_dC_eps            : float = 1E-3 #: Minimum dC/dK slope at the boundaries. Expiries with steeper slopes will be dropped.

    no_arb_mode           : str = "surface"  # Whether to enforce no-arbirage: "off", "expriy", "surface".
    min_half_spread       : float = 1E-4 #: Minimum bid/ask spread. The optimization is weighted by inverse half spread.
    cvxpy_solver          : str = "CLARABEL" #:  Solver to use for the convex optimization problem, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options  : dict = field(default_factory=dict) #: Options to pass to the CVXPY solver.
    mid_fit_weight        : float = 0.1  #: Weight for fitting to the mid prices of options, where 1 means mid-price fit only is active, while 0 means to fit only a linear penalty for exceeding bid-ask.
    min_density           : float = 1E-4 #: Minimum density value to consider in the optimization. Strikes with lower values will be dropped.
    
    def __post_init__(self):
        verify_inp( self.no_arb_mode in ["off", "expiry", "surface"], lambda : f"no_arb_mode must be one of 'off', 'expiry' or 'surface', but found '{self.no_arb_mode}'" ) 
        verify_inp( self.min_strikes_per_expiry >= 3, lambda : f"min_strikes_per_expiry must be at least 3, but found {self.min_strikes_per_expiry}" )
        verify_inp( self.min_dC_eps > 0., lambda : f"min_dC_eps must be positive, but found {self.min_dC_eps}" )
        verify_inp( self.min_half_spread > 0., lambda : f"min_half_spread must be positive, but found {self.min_half_spread}" )
        verify_inp( self.min_density > 0., lambda : f"min_density must be positive, but found {self.min_density}" )

    def __str__(self) -> str:
        """ Very compact but still readable summary """
        return f"md={self.no_arb_mode[0]},mink={self.min_strikes_per_expiry},dC={self.min_dC_eps:.3g},mhs={self.min_half_spread:.3g},mdn={self.min_density:.3g},mw={self.mid_fit_weight:.2f},solver={self.cvxpy_solver}"

    def __call__(self,
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                bids           : list[np.ndarray],
                asks           : list[np.ndarray], *,
                handle_arb     : str = "warn", # how to handle arbitrage situations: "warn", "raise" or "ignore"
                verbose        : Context = Context.all ) -> dict:
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
        handle_arb : str, default ``"warn"``
            What to do if the function cannot fit an arbitrage-free mid for a given single expiry.
            The options are to ``"ignore"`` it, ``"warn"`` about it, or ``"raise"`` an error.
            If it is ignored or merely warned, then the offending expiry will be dropped from the output.

        verbose : :class:`Context`, default ``Context.all``
            Verbose context for logging.
        
        Returns
        -------
        Extended market: ``dict``
           See above.
        """
        T = len(expiries)

        verify_inp( len(strikes) == T, lambda : f"Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(bids) == T, lambda : f"Expected number of bid arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( handle_arb in ["warn", "raise", "ignore"], lambda : f"handle_arb must be one of 'warn', 'raise' or 'ignore', but found '{handle_arb}'" )

        # generate strike range
        # ---------------------
        # Add strikes to the left and right which are consistent with the no-arb condition.

        issues        = []
        r             = PrettyObject()
        r.xstrikes    = []
        r.strikes     = [] # TODO: remove
        r.bids        = [] # TODO: remove
        r.asks        = [] # TODO: remove
        r.expiries    = []
        r.masks       = []
        r.fits1       = None if self.no_arb_mode == "off" else []
        r.fitsX       = None
        r.mstrikes    = None
        r.xix_to_mix  = None
        r.iix_to_xix  = None
        r.xdensities  = None if self.no_arb_mode == "off" else []
        r.mdensities  = None
        r.input_index = []
        tme        = verbose.timer()

        for t, expiry in enumerate(expiries):
            verbose.write(lambda : f"\rExpiry #{t} {expiry*DAYS_PER_YEAR:.3f}d: generating extended strikes... ", end='')

            input_strikes_t = strikes[t]
            input_bid_t     = bids[t]
            input_ask_t     = asks[t]

            xstrikes_t = input_strikes_t
            bid_t      = input_bid_t
            ask_t      = input_ask_t
            mask_t     = np.ones(input_strikes_t.shape[0], dtype=bool)

            if len(xstrikes_t) <= self.min_strikes_per_expiry:
                verbose.write(lambda :f"only {len(xstrikes_t)} strikes found. Skipping this expiry.", head=False)
                continue

            verify_inp( xstrikes_t.shape == bid_t.shape == ask_t.shape, lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Expected strikes, bids and asks to have the same shape for expiry, found {xstrikes_t.shape}, {bid_t.shape}, {ask_t.shape}")
            verify_inp( np.all( np.diff(xstrikes_t) > 0 ), lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Strikes must be strictly increasing; found minimal difference {np.diff(xstrikes_t).min():.3f}" )
            verify_inp( xstrikes_t[0] > 0., lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Lowest strike must be positive; found {xstrikes_t[0]:.3f}" )
            verify_inp( np.all( ask_t - bid_t >0. ), lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: All ask prices must be greater than bid prices.")

            # extend strike range with boundary strikes
            # -----------------------------------------
            # At K_0 and K_n call prices are intrinsic
            # -1 <= { C_0 - C_1 }/{ K_0 - K_1 } <= dC_1
            # Assuming -1 < dC_1 and noting that K_0 < K_1
            # { 1 - K_0 - C_1) >= dC_1 (K_0 - K_1) =>
            # 1 - C_1 + K_1 dC_1 >= K_0 (1 + dC_1) =>
            # K_0 <= (1 - C_1 + K_1 dC_1)/(1 + dC_1)

            assert self.min_dC_eps > 0., f"Internal error: min_dC_eps {self.min_dC_eps} must be positive."
            N  = xstrikes_t.shape[0]
            ok = False
            for i in range(max(0,(N-1)//2)+1):
                C_1   = ask_t[i]
                K_1   = xstrikes_t[i]

                dC_1 = (C_1 - bid_t[i+1])/(K_1 - xstrikes_t[i+1]) # this must be bigger than -1 (using ask-bid is the steepest inside bid/ask slope)
                if dC_1+1. > self.min_dC_eps:
                    lower_k = (1 - C_1 + K_1 * dC_1)/(1 + dC_1)
                    if lower_k <= 0. or lower_k >= K_1:
                        verbose.write(lambda :f"downside dC/dK is {dC_1:.3g}, which is above -1, but the resulting lower_k {lower_k:.3g} is not between 0 and K_1 {K_1:.3g}. Skipping this strike... ", head=False, end='')
                    else:
                        ok = True
                        break
                else:
                    verbose.write(lambda :f"downside dC/dK is {dC_1:.3g}, which must strictly be above -1. Skipping this strike.. ", head=False, end='')

            if not ok:
                verbose.write(lambda : "could not fix it -- skipping this expiry.", head=False)
                continue

            if i>0:
                mask_t[:i] = False
                xstrikes_t = xstrikes_t[i:]
                bid_t      = bid_t[i:]
                ask_t      = ask_t[i:]
                N          = xstrikes_t.shape[0]
                assert np.all( xstrikes_t == input_strikes_t[mask_t] ), "Mask error...?"

            # K_N
            # dC_{N-2} <= { C_{N-1} - C_N }/{ K_{N-1} - K_N } <= 0
            # Assuming dC_N-1 >= 0 and noting C_N = 0, K_{N-1} < K_N
            # dC_{N-2} ( K_{N-1} - K_N ) >= C_{N-1} =>
            # K_N >= K_{N-1} - C_{N-1}/dC_{N-2}

            for i in range(N-1,2,-1):
                C_Nm1 = ask_t[i]
                K_Nm1 = xstrikes_t[i]
                dC_Nm2 = (C_Nm1 - ask_t[i-1])/(K_Nm1 - xstrikes_t[i-1]) # this must be smaller than 0 (using bid-ask is the steepest inside bid/ask slope)
                if dC_Nm2 < -self.min_dC_eps:
                    upper_k = K_Nm1 - C_Nm1/dC_Nm2
                    if upper_k <= K_Nm1:
                        verbose.write(lambda :f"upside dC/dK is {dC_Nm2:.3g}, which is below 0, but the resulting upper_k {upper_k:.3g} is not bigger than K_Nm1 {K_Nm1:.3g}. Skipping this strike... ", head=False, end='')
                    else:
                        ok = True
                        break
                else:
                    verbose.write(lambda :f"upside dC/dK is {dC_Nm2:.3g}, but must be strictly negative. Skipping this strike.. ", head=False, end='')

            if not ok:
                verbose.write(lambda : "could not fix it -- skipping this expiry.", head=False)
                continue

            if i<N-1:
                off        = N-1-i
                mask_t[-off:] = False
                xstrikes_t = xstrikes_t[:i+1]
                bid_t      = bid_t[:i+1]
                ask_t      = ask_t[:i+1]

                assert np.all( xstrikes_t == input_strikes_t[mask_t] ), "Mask error...?"

            dK0        = K_1 - lower_k
            lower_k    = max( lower_k/2., K_1 - dK0*2. )
            dkN        = upper_k - K_Nm1
            upper_k    = K_Nm1 + dkN*2.

            if lower_k >= 1.:
                lower_k = 1.-(xstrikes_t[1]-xstrikes_t[0])
            if upper_k <= 1.:
                upper_k = 1.+(xstrikes_t[1]-xstrikes_t[0])

            # previous step?
            if t>0:
                lower_k = min(lower_k, r.xstrikes[-1][0])
                upper_k = max(upper_k, r.xstrikes[-1][-1])

            xstrikes_t =  np.concatenate(([lower_k], xstrikes_t, [upper_k]))

            # arb-free force fit
            # ------------------

            assert np.all( bid_t == input_bid_t[mask_t] ), "Mask error...?"
            assert np.all( ask_t == input_ask_t[mask_t] ), "Mask error...?"

            if self.no_arb_mode != "off":
                verbose.write(lambda : f"\rExpiry #{t} {expiry*DAYS_PER_YEAR:.3f}d: finding linear no-arbitrage mid... ", end='')

                Nt          = len(xstrikes_t)
                p           = cp.Variable(Nt, nonneg=True)
                C           = np.maximum( 0., xstrikes_t[None,:] - xstrikes_t[:,None] ) @ p
                constraints = [ cp.sum(p) == 1., xstrikes_t @ p == 1., # martingale condition
                                C[1:-1] <= ask_t,
                                C[1:-1] >= bid_t ]

                half_spread = np.maximum( (ask_t - bid_t)/2., self.min_half_spread )
                mid         = (bid_t + ask_t)/2.
                objective   = cp.sum( cp.multiply( 1./half_spread, C[1:-1] - mid )**2 )

                prob        = cp.Problem( cp.Minimize( objective ), constraints )
                prob.solve( solver=self.cvxpy_solver, **self.cvxpy_solver_options )

                if prob.status != "optimal":
                    if prob.status == "infeasible":
                        msg = f"Cannot find linear no-arbitrage mid for expory {expiry*DAYS_PER_YEAR:4f}d (infeasible problem) -- removing expiry."
                        verbose.write(lambda :f"cannot find linear no-arbitrage mid (infeasible problem) -- removing expiry.", head=False)
                        if handle_arb == "raise":
                            raise RuntimeError(msg)
                        if handle_arb == "warn":
                            warnings.warn(msg) # -> maybe we need a proper error here.
                        continue
                    verbose.write(lambda :f"optimized returned status '{prob.status}' -- proceeding anyway... ", end='', head=False)

                # eliminate calls whose density is zero (e.g. the mid is linear) by flagging them as inactive.
                C          = C.value
                p          = p.value
                floor      = self.min_density / float(Nt)
                verify( np.all(p >= -floor), lambda : f"Internal error: negative density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.3f}d after optimization." )
                assert abs( np.sum( p )-1. ) <= floor, f"Internal error: density does not sum to one after removing zero-density strikes, sum is {p.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."
                assert abs( p@xstrikes_t - 1. ) <= floor, f"Internal error: density does not integrate to one after removing zero-density strikes, integral is {p@xstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."
                active     = p > floor
                active[0]  = True
                active[-1] = True

                # compute true fit values from the density
                fit_t      = np.maximum( 0., xstrikes_t[None,:] - input_strikes_t[:,None] ) @ p
                assert np.allclose( fit_t[mask_t], C[1:-1], rtol=0., atol=1E-6), f"Internal error: fit values computed from density do not match optimization output for expiry {expiry*DAYS_PER_YEAR:.3f}d. Worst fit {np.max(np.abs(fit_t[mask_t] - C)):.3e}"

                dCdK_t         = np.diff(C,prepend=1.)/np.diff(xstrikes_t,prepend=0.)
                p              = np.diff(dCdK_t,append=0. )
                assert np.all( p >= -floor ), f"Internal error: negative density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.3f}d after removing zero-density strikes."
                assert abs( np.sum( p )-1. ) <= floor, f"Internal error: density does not sum to one after removing zero-density strikes, sum is {p.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."
                assert abs( p@xstrikes_t - 1. ) <= floor, f"Internal error: density does not integrate to one after removing zero-density strikes, integral is {p@xstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."

                if not np.all(active):
                    # remove strikes; recompute density (which should be valid)
                    xstrikes_t     = xstrikes_t[active]
                    mask_t[mask_t] = active[1:-1]
                    assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"

                    C              = C[active]
                    dCdK_t         = np.diff(C,prepend=1.)/np.diff(xstrikes_t,prepend=0.)
                    p              = np.diff(dCdK_t,append=0.)
                    assert np.all( p >= -floor ), f"Internal error: negative density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.3f}d after removing zero-density strikes."
                    assert abs( np.sum( p )-1. ) <= floor, f"Internal error: density does not sum to one after removing zero-density strikes, sum is {p.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."
                    assert abs( p@xstrikes_t - 1. ) <= floor, f"Internal error: density does not integrate to one after removing zero-density strikes, integral is {p@xstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."

                r.xdensities.append( p )
                r.fits1.append( fit_t )   # fit is computed for all strikes, also those unmasked

            assert np.all( xstrikes_t[1:-1] == input_strikes_t[mask_t] ), "Mask error...?"
            r.xstrikes.append( xstrikes_t )
            r.strikes.append( input_strikes_t )
            r.bids.append( input_bid_t )
            r.asks.append( input_ask_t )
            r.expiries.append( expiry )
            r.input_index.append( t )
            r.masks.append( mask_t )

        r.expiries = np.array(r.expiries, dtype=np.float32)
        verbose.write(lambda : f"\rGenerated {len(r.expiries)} {'arbitrage-free ' if self.no_arb_mode != 'off' else ''}expiries with extended strikes from {T} market expiries. This took {tme}.")

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

        for t in range(T-1,-1,-1):
            # fix strikes
            # 1) Ensure outside strike range is increasing.
            # 2) Add strikes of next expiry if not too close to existing strikes. Always keep current strikes
            xstrikes_t = r.xstrikes[t]
            mstrikes_t = xstrikes_t
            project = None
            if t<T-1:
                next_mstrikes = r.mstrikes[t+1]
                assert next_mstrikes[0] <= xstrikes_t[0] and next_mstrikes[-1] >= xstrikes_t[-1], f"Internal error: next_mstrikes[0] {next_mstrikes[0]:.3f} > xstrikes_t[0] {xstrikes_t[0]:.3f} or next_mstrikes[-1] {next_mstrikes[-1]:.3f} < xstrikes_t[-1] {xstrikes_t[-1]:.3f}"

                min_dx   = min( np.diff(mstrikes_t).min(), np.diff(next_mstrikes).min() )
                collect  = []
                project  = []
                inext    = 0
                add_any  = False

                # skip next strikes until relevant
                while next_mstrikes[inext] <= xstrikes_t[0]+min_dx:
                    inext += 1

                project.append( len(collect) )
                collect = [ xstrikes_t[0] ]

                for ithis in range(1, xstrikes_t.shape[0]):
                    k = xstrikes_t[ithis]
                    # add next strikes up to 'k'
                    while next_mstrikes[inext] <= k-min_dx:
                        collect.append( next_mstrikes[inext] )
                        add_any = True
                        inext += 1
                    project.append( len(collect) )
                    collect.append( k )
                    # skip next_mstrikes which are too close to k
                    while next_mstrikes[inext] <= k+min_dx:
                        inext += 1
                        if inext >= next_mstrikes.shape[0]:
                            break

                if add_any:
                    mstrikes_t = np.array(collect, dtype=xstrikes_t.dtype)
                    assert np.min( np.diff( mstrikes_t ) ) >= min_dx, f"Internal error: min dx {np.diff(mstrikes_t).min():.3f} < min_dx {min_dx:.3f}"
                    assert mstrikes_t[0] == xstrikes_t[0] and mstrikes_t[-1] == xstrikes_t[-1], f"Internal error: mstrikes_t[0] {mstrikes_t[0]:.3f} != xstrikes_t[0] {xstrikes_t[0]:.3f} or mstrikes_t[-1] {mstrikes_t[-1]:.3f} != xstrikes_t[-1] {xstrikes_t[-1]:.3f}"
                    project = np.array( project, dtype=np.int32 )
                else:
                    project = None

            r.mstrikes[t] = mstrikes_t
            r.xix_to_mix[t] = project

        # solve surface no-arb
        # --------------------

        r.mdensities = []
        fit_mids     = []
        constraints  = []
        objective    = 0.

        min_dx_text = "" if min_dx is None else f" with a minimum strike distance of {min_dx:.3g}"
        with verbose.write_t(f"Finding arbitrage-free mid prices accross the entire surface of {sum( len(xt) for xt in r.xstrikes )} strikes{min_dx_text}... ", end='') as tme:
            for t in range(T):
                mstrikes_t  = r.mstrikes[t]
                xstrikes_t  = r.xstrikes[t]
                mask_t      = r.masks[t]
                ask_t       = r.asks[t][mask_t]
                bid_t       = r.bids[t][mask_t]
                assert len(bid_t) == len(ask_t) == xstrikes_t.shape[0]-2, f"Internal error: bid and ask length {len(bid_t)} does not match number of market strikes {xstrikes_t.shape[0]-2}"

                q_t         = cp.Variable(len(mstrikes_t), nonneg=True)
                fit_mid     = cp.maximum( 0., mstrikes_t[None,:] - xstrikes_t[1:-1,None] ) @ q_t    # market strikes
                Cm          = cp.maximum( 0., mstrikes_t[None,:] - mstrikes_t[1:-1,None] ) @ q_t    # model strikes

                if t>0:
                    assert r.mstrikes[t-1].shape == r.mdensities[t-1].shape, f"Internal error: shape mismatch between previous and current mstrikes, {r.mstrikes[t-1].shape} vs {r.mdensities[t-1].shape}"
                    Cm1     = cp.maximum( 0., r.mstrikes[t-1][None,:] - mstrikes_t[1:-1,None] ) @ r.mdensities[t-1]
                else:
                    Cm1     = np.maximum( 0., 1 - mstrikes_t[1:-1] )

                constraints += [ cp.sum( q_t ) == 1.,
                                 q_t @ mstrikes_t == 1., # martingale condition
                                 Cm >= Cm1 ]

                half_spread = np.maximum( (ask_t - bid_t)/2., self.min_half_spread )
                mid         = (bid_t + ask_t)/2.
                excess      = cp.multiply( 1./half_spread, ( cp.maximum( 0., fit_mid - ask_t ) + cp.maximum( 0., bid_t - fit_mid ) ) )
                objective   = objective + cp.sum( self.mid_fit_weight*cp.multiply( 1./half_spread, fit_mid - mid )**2 + (1.-self.mid_fit_weight) * excess )

                fit_mids.append(fit_mid)
                r.mdensities.append(q_t)

            prob        = cp.Problem( cp.Minimize( objective ), constraints )
            prob.solve( solver=self.cvxpy_solver, **self.cvxpy_solver_options )

            if prob.status != "optimal":
                if prob.status == "infeasible":
                    raise RuntimeError(f"Cannot find no-arbitrage surface mids: problem infeasible??")
                verbose.write(lambda : f"solver returned '{prob.status}' -- proceeding anyway... ", head=False, end='' )

            # compute call prices for the original strikes
            r.fitsX = []
            r.mdensities = [ q.value for q in r.mdensities ]
            for t in range(T):
                strikes_t  = r.strikes[t]
                mstrikes_t = r.mstrikes[t]
                q_t        = r.mdensities[t]

                assert np.min(q_t) >= -floor, f"Internal error: negative density {q_t[q_t<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.3f}d after optimization."
                assert abs( np.sum( q_t )-1. ) <= floor, f"Internal error: density does not sum to one after removing zero-density strikes, sum is {q_t.sum():.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."
                assert abs( q_t@mstrikes_t - 1. ) <= floor, f"Internal error: density does not integrate to one after removing zero-density strikes, integral is {q_t@mstrikes_t:.6f} instead of 1 for expiry {expiry*DAYS_PER_YEAR:.3f}d."

                xfit_mids  = fit_mids[t].value
                mask_t     = r.masks[t]
                fit_t      = np.maximum( 0., mstrikes_t[None,:] - strikes_t[:,None] ) @ q_t
                assert np.allclose( fit_t[mask_t], xfit_mids, rtol=0., atol=1E-6), f"Internal error: fit values computed from density do not match optimization output for expiry {expiry*DAYS_PER_YEAR:.3f}d. Worst fit {np.max(np.abs(fit_t[mask_t] - xfit_mids)):.3e}"
                r.fitsX.append( fit_t )
            verbose.write(lambda : f"done. This took {tme}.", head=False )

        return r

