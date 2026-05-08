
"""
Make arbfree markets.
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
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.utils.backbone import BackboneNumpy

@version("0.1")
@dataclass(frozen=True)
class MkArbFree:
    """
    A utility class which generates a "closest" arbitrage free market.

    Given a market with expiries, strikes, bid and asks this functional:
    1) Expands the strike range following https://arxiv.org/pdf/2601.11209 remark 2.7, and drops expiries with too few strikes or with too steep bid-ask slopes at the boundaries.
    2) Fits per expiry a linear no-arbitrage mid price within the bid-ask spread. It drops any expiries for which this is not possible as it implies actionable arbitrage.
    3) The function then fits a fully arbitrage free surface, on a strike grid which theoretically means that a mid-price fit exists if and only if the
        resulting fit is within bid/ask spread. The strike grid used is returned as ``mstrikes``.
    """

    min_strikes_per_expiry: int = 5      #: Minimum strikes per expiry. Expiries with fewer strikes will be dropped.
    min_dC_eps           : float = 1E-4 #: Minimum dC/dK slope at the boundaries. Expiries with steeper slopes will be dropped.

    no_arb_mode          : str = "surface"  # Whether to enforce no-arbirage: "off", "expriy", "surface".
    min_half_spread      : float = 1E-5 #: Minimum bid/ask spread. The optimization is weighted by inverse half spread.
    cvxpy_solver         : str = "CLARABEL" #:  Solver to use for the convex optimization problem, c.f. `list of cvxpy solvers <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_.
    cvxpy_solver_options : dict = field(default_factory=dict) #: Options to pass to the CVXPY solver.
    mid_fit_weight       : float = 0.1  #: Weight for fitting to the mid prices of options, where 1 means mid-price fit only is active, while 0 means to fit only a linear penalty for exceeding bid-ask.
    min_density          : float = 1E-4 #: Minimum density value to consider in the optimization. Strikes with lower values will be dropped.
    min_dx               : float = 0.00001 #: Minimum distance between strikes in "surface" mode. When adding strikes from the next expiry, only strikes which are at least this far away from existing strikes will be added.
    
    def __post_init__(self):
        verify_inp( self.no_arb_mode in ["off", "expiry", "surface"], lambda : f"no_arb_mode must be one of 'off', 'expiry' or 'surface', but found '{self.no_arb_mode}'" ) 
        verify_inp( self.min_strikes_per_expiry >= 3, lambda : f"min_strikes_per_expiry must be at least 3, but found {self.min_strikes_per_expiry}" )
        verify_inp( self.min_dC_eps > 0., lambda : f"min_dC_eps must be positive, but found {self.min_dC_eps}" )
        verify_inp( self.min_half_spread > 0., lambda : f"min_half_spread must be positive, but found {self.min_half_spread}" )
        verify_inp( self.min_density > 0., lambda : f"min_density must be positive, but found {self.min_density}" )
        verify_inp( self.min_dx > 0., lambda : f"min_dx must be positive, but found {self.min_dx}" )   

    def __call__(self,
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                bids           : list[np.ndarray],
                asks           : list[np.ndarray], *,
                handle_arb     : str = "warn", # how to handle arbitrage situations: "warn", "raise" or "ignore"
                verbose        : Context = Context.all ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Make market data arbitrage-free.

        Given a market with expiries, strikes, bid and asks this function
        1) Expands the strike range following https://arxiv.org/pdf/2601.11209 remark 2.7, and drops expiries with too few strikes or with too steep bid-ask slopes at the boundaries.
        2) Fits per expiry a linear no-arbitrage mid price within the bid-ask spread. It drops any expiries for which this is not possible as it implies actionable arbitrage.
        3) The function then fits a fully arbitrage free surface, on a strike grid which theoretically means that a mid-price fit exists if and only if the
           resulting fit is within bid/ask spread. The strike grid used is returned as ``mstrikes``.

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
        config : :class:`StrikeGeneratorConfig`, default ``StrikeGeneratorConfig()``
            Configuration for the strike generation, including minimum number of strikes per expiry and minimum dC/dK slope at the boundaries.
        verbose : :class:`Context`, default ``Context.all``
            Verbose context for logging.
        
        Returns
        -------
        Exteneded market: dict
            A dictionary with
            * ``expiries``: array of expiries. This may be fewer expiries than provided as input.
            * ``xstrikes``: list of extended strikes.
            * ``bids``: list of bid prices for the extended strikes. There are two less bids than ``xstrikes`` since the model prices boundary strikes at intrinsic.
            * ``asks``: list of ask prices for the extended strikes. There are two less asks than ``xstrikes`` since the model prices boundary strikes at intrinsic.
            * ``mids``: list of mid prices for the extended strikes, from the per-expiry sweep. This is None if ``no_arb_mode`` is "off".
            * ``fit``: list of mid prices for the extended strikes, from the surface fit. This is None if ``no_arb_mode`` is not "surface". The ``fit`` is not
              guaranteed to be witin bid/ask spread.
            * ``mstrikes``: list of strikes for the surface fit. This is None if ``no_arb_mode`` is not "surface".
               The ``mstrikes`` are a superset of the ``xstrikes`` and are constructed by adding strikes from the next expiry if they are not too close to existing strikes.
        """
        T = len(expiries)

        verify_inp( len(strikes) == T, lambda : f"Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(bids) == T, lambda : f"Expected number of bid arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( handle_arb in ["warn", "raise", "ignore"], lambda : f"handle_arb must be one of 'warn', 'raise' or 'ignore', but found '{handle_arb}'" )

        # generate strike range
        # ---------------------

        issues     = []
        r          = PrettyObject()
        r.xstrikes = []
        r.expiries = []
        r.bids     = []
        r.asks     = []
        r.mids     = None if self.no_arb_mode == "off" else []
        r.fits     = None
        r.mstrikes = None
        tme        = verbose.timer()

        for t, expiry in enumerate(expiries):
            verbose.write(lambda : f"\rExpiry #{t} {expiry*DAYS_PER_YEAR:.3f}d: generating extended strikes... ", end='')

            strikes_t  = strikes[t]
            bid_t      = bids[t]
            ask_t      = asks[t]

            if len(strikes_t) <= self.min_strikes_per_expiry:
                verbose.write(lambda :f"only {len(strikes_t)} strikes found. Skipping this expiry.", head=False)
                continue

            verify_inp( strikes_t.shape == bid_t.shape == ask_t.shape, lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Expected strikes, bids and asks to have the same shape for expiry, found {strikes_t.shape}, {bid_t.shape}, {ask_t.shape}")
            verify_inp( np.all( np.diff(strikes_t) > 0 ), lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Strikes must be strictly increasing; found minimal difference {np.diff(strikes_t).min():.3f}" )
            verify_inp( strikes_t[0] > 0., lambda : f"#{t} {expiry*DAYS_PER_YEAR:.3f}d: Lowest strike must be positive; found {strikes_t[0]:.3f}" )
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
            N = strikes_t.shape[0]
            ok = False
            for i in range(max(0,(N-1)//2)+1):
                C_1   = ask_t[i]
                K_1   = strikes_t[i]

                dC_1 = (C_1 - bid_t[i+1])/(K_1 - strikes_t[i+1]) # this must be bigger than -1 (using ask-bid is the steepest inside bid/ask slope)
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
                strikes_t = strikes_t[i:]
                ask_t     = ask_t[i:]
                bid_t     = bid_t[i:]
                N         = strikes_t.shape[0]

            # K_N
            # dC_{N-2} <= { C_{N-1} - C_N }/{ K_{N-1} - K_N } <= 0
            # Assuming dC_N-1 >= 0 and noting C_N = 0, K_{N-1} < K_N
            # dC_{N-2} ( K_{N-1} - K_N ) >= C_{N-1} =>
            # K_N >= K_{N-1} - C_{N-1}/dC_{N-2}

            for i in range(N-1,2,-1):
                C_Nm1 = ask_t[i]
                K_Nm1 = strikes_t[i]
                dC_Nm2 = (C_Nm1 - ask_t[i-1])/(K_Nm1 - strikes_t[i-1]) # this must be smaller than 0 (using bid-ask is the steepest inside bid/ask slope)
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
                strikes_t = strikes_t[:i+1]
                ask_t     = ask_t[:i+1]
                bid_t     = bid_t[:i+1]


            dK0        = K_1 - lower_k
            lower_k    = max( lower_k/2., K_1 - dK0*2. )
            dkN        = upper_k - K_Nm1
            upper_k    = K_Nm1 + dkN*2.

            if lower_k >= 1.:
                lower_k = 1.-(strikes_t[1]-strikes_t[0])
            if upper_k <= 1.:
                upper_k = 1.+(strikes_t[1]-strikes_t[0])

            # previous step?
            if t>0:
                lower_k = min(lower_k, r.xstrikes[-1][0])
                upper_k = max(upper_k, r.xstrikes[-1][-1])

            xstrikes_t =  np.concatenate(([lower_k], strikes_t, [upper_k]))

            # arb-free force fit
            # ------------------

            _use_calls = False

            if self.no_arb_mode != "off":
                verbose.write(lambda : f"\rExpiry #{t} {expiry*DAYS_PER_YEAR:.3f}d: finding linear no-arbitrage mid... ", end='')

                Nt          = len(xstrikes_t)
                if _use_calls:
                    assert False
                    C           = cp.Variable(Nt, nonneg=True)
                    dC          = cp.multiply( C[1:] - C[:-1], 1./( xstrikes_t[1:] - xstrikes_t[:-1] ) )
                    constraints = [ -1 <= dC[0],
                                    dC[:-1] <= dC[1:],
                                    dC[-1] <= 0 ,
                                    C[1:-1] <= ask_t,
                                    C[1:-1] >= bid_t ]
                else:
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
                if _use_calls:
                    C     = C.value
                    dC    = np.diff( C, prepend=1. ) / np.diff(xstrikes_t, prepend=0.)
                    p     = np.diff( dC, prepend=-1. ) / np.diff(xstrikes_t, prepend=0.)
                    C     = C[1:-1]
                else:
                    C     = C.value
                    p     = p.value
                    C     = C[1:-1]
                floor     = self.min_density / float(Nt)
                verify( np.all(p >= -floor), lambda : f"Internal error: negative density {p[p<-floor]} below floor {floor:.3e} found for expiry {expiry*DAYS_PER_YEAR:.3f}d after optimization." )
                active    = p > floor
                active[0] = True
                active[-1] = True
                if not np.all(active):
                    xstrikes_t = xstrikes_t[active]
                    bid_t     = bid_t[active[1:-1]]
                    ask_t     = ask_t[active[1:-1]]
                    fit_t     = C[active[1:-1]]
                else:
                    fit_t     = C

            r.xstrikes.append( xstrikes_t )
            r.bids.append( bid_t )
            r.asks.append( ask_t )
            r.mids.append( fit_t )
            r.expiries.append( expiry )
        r.expiries = np.array(r.expiries, dtype=np.float32)
        verbose.write(lambda : f"\rGenerated {len(r.expiries)} {'arbitrage-free ' if self.no_arb_mode != 'off' else ''}expiries with extended strikes from {T} market expiries. This took {tme}.")

        if self.no_arb_mode != "surface":
            return r
        
        # backward-dense grid
        # -------------------

        T          = len(expiries)
        r.mstrikes = [None]*T
        r.fit      = []

        for t in range(T-1,-1,-1):
            # fix strikes
            # 1) Ensure outside strike range is increasing.
            # 2) Add strikes of next expiry if not too close to existing strikes. Always keep current strikes
            xstrikes_t = r.xstrikes[t]
            mstrikes_t = xstrikes_t
            if t<T-1:
                next_mstrikes = r.mstrikes[t+1]
                assert next_mstrikes[0] <= xstrikes_t[0] and next_mstrikes[-1] >= xstrikes_t[-1], f"Internal error: next_mstrikes[0] {next_mstrikes[0]:.3f} > xstrikes_t[0] {xstrikes_t[0]:.3f} or next_mstrikes[-1] {next_mstrikes[-1]:.3f} < xstrikes_t[-1] {xstrikes_t[-1]:.3f}"

                min_dx   = min( np.diff(mstrikes_t).min(), np.diff(next_mstrikes).min() )
                collect  = []
                inext    = 0
                add_any  = False

                # skip next strikes until relevant
                while next_mstrikes[inext] <= xstrikes_t[0]+min_dx:
                    inext += 1

                collect = [ xstrikes_t[0] ]

                for ithis in range(1, xstrikes_t.shape[0]):
                    k = xstrikes_t[ithis]
                    # add next strikes up to 'k'
                    while next_mstrikes[inext] <= k-min_dx:
                        collect.append( next_mstrikes[inext] )
                        add_any = True
                        inext += 1
                    collect.append( k )
                    # skip next_mstrikes which are too close to k
                    while next_mstrikes[inext] <= k+min_dx:
                        inext += 1
                        if inext >= next_mstrikes.shape[0]:
                            break

                if add_any:
                    mstrikes_t = np.array(collect, dtype=xstrikes_t.dtype)
                    assert np.min( np.diff( mstrikes_t ) ) >= self.min_dx, f"Internal error: min dx {np.diff(mstrikes_t).min():.3f} < min_dx {self.min_dx:.3f}"
                    assert mstrikes_t[0] == xstrikes_t[0] and mstrikes_t[-1] == xstrikes_t[-1], f"Internal error: mstrikes_t[0] {mstrikes_t[0]:.3f} != xstrikes_t[0] {xstrikes_t[0]:.3f} or mstrikes_t[-1] {mstrikes_t[-1]:.3f} != xstrikes_t[-1] {xstrikes_t[-1]:.3f}"

            r.mstrikes[t] = mstrikes_t

        densities   = []
        fit_mids    = []
        constraints = []
        objective   = 0.

        with verbose.write_t("Finding arbitrage-free mid prices within bid/ask accross the entire surface... ", end='') as tme:
            for t in range(T):
                mstrikes_t  = r.mstrikes[t]
                xstrikes_t  = r.xstrikes[t]
                ask_t       = r.asks[t]
                bid_t       = r.bids[t]
                assert len(bid_t) == len(ask_t) == xstrikes_t.shape[0]-2, f"Internal error: bid and ask length {len(bid_t)} does not match number of market strikes {xstrikes_t.shape[0]-2}"

                q_t         = cp.Variable(len(mstrikes_t), nonneg=True)
                fit_mid     = cp.maximum( 0., mstrikes_t[None,:] - xstrikes_t[1:-1,None] ) @ q_t     # market strikes
                Cm          = cp.maximum( 0., mstrikes_t[None,:] - mstrikes_t[1:-1,None] ) @ q_t    # model strikes

                if t>0:
                    assert r.mstrikes[t-1].shape == densities[t-1].shape, f"Internal error: shape mismatch between previous and current mstrikes, {r.mstrikes[t-1].shape} vs {densities[t-1].shape}"
                    Cm1     = cp.maximum( 0., r.mstrikes[t-1][None,:] - mstrikes_t[1:-1,None] ) @ densities[t-1]
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
                densities.append(q_t)

            prob        = cp.Problem( cp.Minimize( objective ), constraints )
            prob.solve( solver=self.cvxpy_solver, **self.cvxpy_solver_options )

            if prob.status != "optimal":
                if prob.status == "infeasible":
                    raise RuntimeError(f"Cannot find no-arbitrage surface mids: problem infeasible??")
                verbose.write(lambda : f"solver returned '{prob.status}' -- proceeding anyway... ", head=False, end='' )

            r.fits = [ fit_mid.value for fit_mid in fit_mids ]
            verbose.write(lambda : f"done. This took {tme}.")

        return r

