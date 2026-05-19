"""
SANOS
=====
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
import math

@version("0.8")
@dataclass(frozen=True)
class SanosPricer:
    """
    Arbitrage-Free Option pricer based on `DLV/SANOS <http://arxiv.org/abs/2211.13227>`_.
    """

    smoothness   : float|None        # SANOS smoothness or None for linear model.
    expiries     : np.ndarray        # Model expiries. Must be increasing and positive.
    variances    : np.ndarray        # Model variances. Must be increasing and positive.
    xstrikes     : list[np.ndarray]  # Extended strike range. Each element must be increasing and positive.
    xdensities   : list[np.ndarray]  # Densities for each ``xstrike``.

    _eps_price   : float = 1E-6

    def __post_init__(self):
        verify_inp( self.expiries[0] > 0., lambda : f"'expiries' must be positive, but found {self.expiries[0]:.4g}" )
        verify_inp( self.variances[0] > 0., lambda : f"'variances' must be positive, but found {self.variances[0]:.4g}" )
        verify_inp( len(self.expiries) == 1 or np.min(np.diff(self.expiries)) > 0., lambda : f"'expiries' must be positive and strictly increasing, but found differences {np.diff(self.expiries).min():.4g}" )
        verify_inp( len(self.variances) == 1 or np.min(np.diff(self.variances)) > 0., lambda : f"'variances' must be positive and strictly increasing, but found differences {np.diff(self.variances).min():.4g}" )
        verify_inp( self.expiries.shape == self.variances.shape, lambda : f"Shape of 'expiries' {self.expiries.shape} must match shape of 'variances' {self.variances.shape}" )

        verify_inp( len(self.xstrikes) == len(self.xdensities) == len(self.expiries), lambda : f"Length of 'xstrikes' and 'xdensities' lists must match length of 'expiries', but found {len(self.xstrikes)}, {len(self.xdensities)} and {len(self.expiries)}" )
        for i, (xstrikes_i, xdensities_i) in enumerate(zip(self.xstrikes, self.xdensities)):
            verify_inp( xstrikes_i.shape == xdensities_i.shape, lambda : f"Shape of 'xstrikes[{i}]' {xstrikes_i.shape} must match shape of 'xdensities[{i}]' {xdensities_i.shape}" )
            verify_inp( xstrikes_i[0] > 0. and np.min( np.diff(xstrikes_i) ) > 0., lambda : f"'xstrikes[{i}]' must be positive and increasing, but found differences {np.diff(xstrikes_i).min():.4g}" )
            verify_inp( np.all(xdensities_i >= 0.), lambda : f"'xdensities[{i}]' must be non-negative, but found {xdensities_i}" )

    @property
    def T(self) -> int:
        return len(self.expiries)

    def __call__( self, expiries: float|np.ndarray, 
                        strikes : np.ndarray, *, 
                        and_vols: bool = False ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Price options at ``strikes`` with expiries ``expiries``.

        If ``and_vols`` is ``True`` this function returns call prices and their implied volatilities;
        if ``False``, it will return only call prices.

        Parameters
        ----------
        expiries : float | :class:`np.ndarray`
            The expiry or expiries to price at.
            
            * If ``expiries`` is a float, all strikes are evaluated at that expiry.
            * If ``expiries`` is a 1D array of shape ``(N,)``, then ``strikes`` must also be a 1D array of shape ``(N,)``, and the function will price each strike at their respective expiry.
            * If ``expiries`` is a 2D array of shape ``(M,1)``, then ``strikes`` must be a 2D array of shape ``(1,N)``, and the function will price each strike at each expiry, returning an array of shape ``(M,N)``.

        strikes : np.ndarray
            The strikes to price at. The shape of this array must be compatible with the shape of ``expiries`` as described above.

        and_vols : bool, default ``False``
            Whether to return implied volatilities in addition to call prices. If ``True``, the function returns a tuple of (call prices, implied volatilities).
            If ``False``, it returns only call prices.
        """
        expiries = np.asarray(expiries,dtype=float)
        strikes  = np.asarray(strikes,dtype=float)
        _0       = strikes.dtype.type(0.)
        _1       = strikes.dtype.type(1.)
        verify_inp( len(strikes.shape) > 0, lambda : "'strikes' must be a non-empty array")

        def x_price( expiry : float, strikes : np.ndarray ):
            verify_inp( isinstance( expiry, (float, int, np.number)) and expiry>=0, lambda : f"Expected 'expiry' to be a non-negative number, but found {str(expiry)}")

            if expiry<ONE_MINUTE:
                return np.maximum( _0, _1 - strikes )

            if self.smoothness is None:
                def call(spots):
                    return np.maximum( _0, spots[None,:] - strikes[:,None] )
            else:
                vol = self.model_vols( expiries=expiry )
                def call(spots):
                    return bs.price( k=strikes[:,None]/spots[None,:], vol=vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )*spots[None,:]

            ix = np.searchsorted( self.expiries, expiry )
            if ix==0:
                strikes_0 = self.xstrikes[0]
                p_0       = self.xdensities[0] 
                return call( spots=strikes_0 ) @ p_0

            if ix==len(self.expiries):
                strikes_T = self.xstrikes[-1]
                p_T       = self.xdensities[-1]
                return call( spots=strikes_T ) @ p_T

            diff_ixm1 = abs( expiry - self.expiries[ix-1] )
            diff_ix   = abs( expiry - self.expiries[ix] )
            diff      = diff_ix if diff_ix < diff_ixm1 else diff_ixm1

            if diff<ONE_MINUTE:
                ix_mind    = ix if diff_ix < diff_ixm1 else ix-1
                strikes_ix = self.xstrikes[ix_mind]
                p_ix       = self.xdensities[ix_mind] 
                return call( spots=strikes_ix ) @ p_ix

            strikes_ixm1 = self.xstrikes[ix-1] 
            p_ixm1       = self.xdensities[ix-1]
            c_ixm1       = call( spots=strikes_ixm1 ) @ p_ixm1

            strikes_ix   = self.xstrikes[ix]
            p_ix         = self.xdensities[ix]
            c_ix         = call( spots=strikes_ix ) @ p_ix

            return ((expiry - self.expiries[ix-1])*c_ix + (self.expiries[ix]-expiry)*c_ixm1)/(self.expiries[ix]-self.expiries[ix-1])

        if len(expiries.shape) == 0:
            prices = x_price( float(expiries), strikes = strikes )
            if not and_vols:
                return prices
            vols   = bs.implied( k=strikes, prices=prices, sqrtT=math.sqrt(float(expiries)), eps=self._eps_price )
            assert prices.shape == vols.shape, f"Shape error {prices.shape} != {vols.shape}"
            return prices, vols

        if expiries.shape == (1,):
            prices = x_price( expiries[0], strikes = strikes )
            if not and_vols:
                return prices
            vols   = bs.implied( k=strikes, prices=prices, sqrtT=math.sqrt(float(expiries[0])), eps=self._eps_price ) 
            assert prices.shape == vols.shape, f"Shape error {prices.shape} != {vols.shape}"
            return prices, vols

        if len(expiries.shape) == 1:
            if strikes.shape == (1,):
                prices = np.zeros_like(expiries, dtype=strikes.dtype)
                for j, expiry in enumerate(expiries):
                    prices[j] = x_price( expiry=expiry, strikes=strikes )
                if not and_vols:
                    return prices
                vols   = bs.implied( k=strikes, prices=prices, sqrtT=np.sqrt(expiries), eps=self._eps_price )
                assert prices.shape == vols.shape, f"Shape error {prices.shape} != {vols.shape}"
                return prices, vols

            verify_inp( strikes.shape == expiries.shape, lambda : f"If 'expiries' is a vector of length M, then 'strikes' must be a vector of length 1 or M, but found {expiries.shape} and {strikes.shape}" )
            uttys   = np.unique(expiries)
            prices  = np.zeros_like(strikes)
            for utty in uttys:
                mask = expiries==utty
                prices[mask] = x_price( expiry=utty, strikes=strikes[mask] )
            if not and_vols:
                return prices
            vols = bs.implied( k=strikes, prices=prices, sqrtT=np.sqrt(expiries), eps=self._eps_price )
            assert prices.shape == vols.shape, f"Shape error {prices.shape} != {vols.shape}"
            return prices, vols
    
        verify_inp( len(expiries.shape) == 2 and expiries.shape[1] == 1, lambda: f"'expiries' can be a vector or a matrix if size '(M,1)'. Found shape {expiries.shape}" )
        M = expiries.shape[0]
        verify_inp( len(strikes.shape) == 2 and strikes.shape[0] in (M,1), lambda: f"If 'expiries' is a matrix of shape '(M,1), then 'strikes' must be a matrix of shape '(M|1,N)'. Found expiries shape {expiries.shape} and strikes {strikes.shape}" )
        N = strikes.shape[1]
        prices = np.zeros((M,N),dtype=strikes.dtype)
        for ix, tty in enumerate(expiries[:,0]):
            prices[ix,:] = x_price( expiry=tty, strikes=strikes[0 if strikes.shape[0] == 1 else ix,:] )
        if not and_vols:
            return prices
        vols = bs.implied( k=strikes, prices=prices, sqrtT=np.sqrt(expiries), eps=self._eps_price )
        assert prices.shape == vols.shape, f"Shape error {prices.shape} != {vols.shape}"
        return prices, vols

    def model_vols( self, expiries: float|np.ndarray ) -> float|np.ndarray:
        """ Interpolates the model variances, scaled by ``smoothness`` """
        assert not self.variances is None, "Internal error - variances are missing"
        is_float = not hasattr(expiries,"shape")
        expiries = np.asarray([float(expiries)]) if is_float else np.asarray(expiries)
        expiries = np.clip( expiries, self.expiries[0], self.expiries[-1] )
        vols     = np.sqrt( self.smoothness * np.interp( expiries, self.expiries, self.variances ) / expiries )
        assert np.all(np.isfinite(vols)), "Internal error: infinite vol"
        return vols if not is_float else vols[0]

    def dlvs( self, *, all_strikes : bool = False, min_dlv : float = 0.01, max_dlv : float = 0.5 ):
        """
        Compute discrete local volatilities.
        """

        verify_inp( 0.<  min_dlv < max_dlv, lambda : f"'min_dlv' must be positive and less than 'max_dlv', found {min_dlv:.4g} and {max_dlv:.4g}" )
        min_dlvsq = float(min_dlv**2)
        max_dlvsq = float(max_dlv**2)

        # dlvs
        # We note that dC_t = 1/2 DLV^2 K^2 dT C''(K) where C''(K) = 1/2 ( k[2:] - K[:-2]) p(k)
        # hence
        #    dC_t = DLV^2 { K^2/( k[2:] - K[:-2]) dt} p(K)
        # We precompute
        #    dlvbox = K^2/( k[2:] - K[:-2]) dt

        T      = len(self.expiries)
        dts    = np.diff( self.expiries, prepend=0. ).astype(np.float64,copy=False)
        dlvsqs = []
        _1     = np.float64(1.)
        _0     = np.float64(0.)
        for t in range(T):
            xstrikes_t   = self.xstrikes[t].astype(np.float64,copy=False)
            xstrikes_tm1 = self.xstrikes[t-1].astype(np.float64,copy=False) if t>0 else None
            p_t          = self.xdensities[t].astype(np.float64,copy=False)
            p_tm1        = self.xdensities[t-1].astype(np.float64,copy=False) if t>0 else None
            p_box        = xstrikes_t[1:-1]**2 / ( xstrikes_t[2:] - xstrikes_t[:-2]) * dts[t] * p_t[1:-1]

            C_t        = np.maximum( _0, xstrikes_t[None,:] - xstrikes_t[1:-1,None] ) @ p_t
            C_tm1      = np.maximum( _0, xstrikes_tm1[None,:] - xstrikes_t[1:-1,None] ) @ p_tm1 if t>0 else np.maximum( _0, 1. - xstrikes_t[1:-1] ) 
            dC_t       = C_t - C_tm1

            assert np.all( np.isfinite(dC_t) ), f"Expected dC_t to be finite, found min {np.min(dC_t)} and max {np.max(dC_t)}"
            assert np.all( np.isfinite(p_box) ), f"Expected p_box to be finite, found min {np.min(p_box)} and max {np.max(p_box)}"
            assert np.min( dC_t ) > -self._eps_price*100., f"Expected dC_t to be non-negative, found min {np.min(dC_t)} [tolerance 'prices' {self._eps_price*100.}]"
            assert np.min( p_box ) > -self._eps_density*100., f"Expected p_box to be non-negative, found min {np.min(p_box)} [tolerance 'densities' {self._eps_density*100.}]"

            dC_zero      = dC_t <= self._eps_price
            too_small    = dC_t <= min_dlvsq * p_box
            too_big      = dC_t >= max_dlvsq * p_box

            divisor      = np.where( too_small|dC_zero|too_big, _1, p_box )  # dC_t can be positive while p_t is zero ... in this case we cannot determine a correct DLV - any will do.
            dlvsqs_t     = np.select( [dC_zero, too_small, too_big ], [min_dlvsq, min_dlvsq, max_dlvsq], default=dC_t / divisor )
            dlvsqs_t     = np.where( np.isfinite(dlvsqs_t), dlvsqs_t, min_dlvsq ) # cover infintie
            dlvsqs_t     = np.clip( dlvsqs_t, min=min_dlvsq, max=max_dlvsq )
            if all_strikes: # makes dlvsq_t the same size as xstrikes_t
                dlvsqs_t = np.concatenate( ([0.], dlvsqs_t, [0.]) )
            dlvsqs.append( dlvsqs_t )
        return dlvsqs

@dataclass(frozen=True)
class SanosLPSolver:
    timeout_seconds      : float = 60*5. # timeout in seconds
    _debug_verbose_cvxpy : set[str] | bool | None = None  # Turn 'verbose' on for cvxpy, optionally scoped by solver mode

    def __call__(self,
             objective: cp.Expression,
             constraints: list[cp.Constraint],
             variables: list[tuple[cp.Variable, np.ndarray]],*,
             message_context : callable|str|None = None,
             for_expiry: bool = False,
             is_fallback : bool = False,
             verbose : Context = Context.quiet ) -> tuple[bool, str, bool]:
        """
        Solve a problem with fall back.

        Returns ``(success, status_str, is_fallback)`` where ``success`` is a
        boolean indicating whether the problem was solved successfully,
        ``status_str`` is the status string returned by the solver, and
        ``is_fallback`` is a boolean indicating whether the fallback solver was used.        
        """

        def message():
            if message_context is None:
                return ""
            if callable(message_context):
                msg = message_context()
            else:
                msg = str(message_context)
            return (msg + ": ") if msg != "" else ""

        # initialize
        for v, val in variables:
            if not val is None:
                assert v.shape == val.shape, f"Internal error: variable shape {v.shape} does not match provided value shape {val.shape}"
                v.value = val
            else:
                v.value = np.zeros(v.shape, dtype=np.float64)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message=r"Solution may be inaccurate.*",category=UserWarning)
            warnings.filterwarnings("error",message=r"overflow encountered in power",category=RuntimeWarning)
            try:
                prob        = cp.Problem( cp.Minimize( objective ), constraints )
                slv, opt    = self._lp_solver_settings( for_expiry=for_expiry, is_fallback=is_fallback )
                prob.solve( solver=slv, **opt )
            except SolverError as e:
                if not is_fallback:
                    warnings.warn(f"{message()}Solver error: {str(e)}. Attempting to solve with fallback solver.")
                    return self( objective=objective, constraints=constraints, variables=variables, message_context=message_context, for_expiry=for_expiry, is_fallback=True, verbose=verbose )
                raise type(e)(f"{message()}Solver error: {str(e)}.") from e
            except RuntimeWarning as e:
                prob.status = "overflow error"
            except Exception as e:
                if not is_fallback:
                    msg = str(e)
                    if 'Cannot unpack invalid solution' in msg:
                        return self( objective=objective, constraints=constraints, variables=variables, message_context=message_context, for_expiry=for_expiry, is_fallback=True, verbose=verbose )
                raise type(e)(f"{message()}Unexpected error: {str(e)}.") from e

            success = prob.status in ["optimal", "optimal_inaccurate"]
        return success, prob.status, is_fallback

    def _lp_solver_settings( self, *, for_expiry : bool, is_fallback : bool ) -> tuple[str, dict]:
        """ Return lp solver settings """
        if isinstance(self._debug_verbose_cvxpy, bool):
            verbose = self._debug_verbose_cvxpy
        elif self._debug_verbose_cvxpy is None:
            verbose = False
        else:
            verbose = ('expiry' if for_expiry else 'surface') in self._debug_verbose_cvxpy

        if not is_fallback:
            solver = "HIGHS"
            options = dict( warm_start=True,
                            highs_options={
                            "solver": "simplex",# if for_expiry else "ipm",
                            "presolve": "on",
                            "threads": 0,
                            "simplex_strategy": 1,   # dual simplex, serial; HiGHS default
                            "kkt_tolerance": 1e-7,
                            "time_limit": self.timeout_seconds
                            },
                            verbose=verbose                 
                            )
        else:
            solver = "CLARABEL"
            options = dict(
                warm_start=True,
                time_limit=self.timeout_seconds,
                tol_gap_abs=1e-5,
                tol_gap_rel=1e-5,
                tol_feas=1e-6,
                tol_infeas_abs=1e-7,
                tol_infeas_rel=1e-7,
                reduced_tol_gap_abs=5e-5,
                reduced_tol_gap_rel=5e-5,
                reduced_tol_feas=5e-5,
                max_threads=0,
                presolve_enable=True,
                equilibrate_enable=True,
                verbose=verbose
            )
        return solver, options

# --------------------------------
# Density repair by ChatGPT
# --------------------------------

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


def warn_and_fix_density( x, p, eps, message : callable ):
    """ Wrapper around fixing and warning """

    p   = np.asarray(p, dtype=np.float64)
    x   = np.asarray(x, dtype=np.float64)

    assert np.min(p) > -1E-3, f"{message()}: density has minimum {np.min(p):.6g} < -1E-3, which is too negative to be repaired."
    assert abs(np.sum(p)-1.) < 1E-2, f"{message()}: density sums to {np.sum(p):.6g}, which is too close to zero to be repaired."
    assert abs(p@x-1.) < 1E-2, f"{message()}: density integrates to {p@x:.6g}, which is too close to zero to be repaired."

    if not np.all( p >= -eps ) or abs(np.sum(p)-1.) > eps or abs(p@x-1.) > eps:
        p = repair_density_nonnegative(p=p, x=x)

    assert np.all( p >= -eps ), f"{message()}: negative density {np.min(p):.6g} found after repair."
    assert abs(np.sum(p)-1.) <= eps, f"{message()}: density sums to {np.sum(p):.6g} after repair."
    assert abs(p@x-1.) <= eps, f"{message()}: density sums to {p@x:.6g} after repair."
    return p
