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
from quantdata.utils.backbone import BackboneNumpy
import math

from quantdata.sanos import IMPOSE_FLOOR_ON_DENSITY
from quantdata.sanos.sanos_tools import SanosPricer, SanosLPSolver, warn_and_fix_density

@version("0.41", dependencies=[SanosPricer])
@dataclass(frozen=True)
class SanosFitter:
    """
    A utility class which generates a "closest" arbitrage-free market using
    the ideas from the paper https://arxiv.org/pdf/2601.11209.
    """
    smoothness           : float|None = None   # Whether to impose SANOS smoothness.
    timeout_seconds      : float|None = 60.*5. #: Time limit for the optimization solver, in seconds. This is passed to the CVXPY solver options as "time_limit".

    min_dlv              : float = 0.01      #: Minimum local volatility to impose in the fit.
    max_dlv              : float = 50.        #: Maximum local volatility to impose in the fit.
    min_density_vol      : float|None = 1E-4 # Minimum local volatility to impose in the fit.
    min_density_cap      : float|None = 1E-5 # Maximum density floor. Only used if :attr:`IMPOSE_FLOOR_ON_DENSITY` is set to True; otherwise 0.

    # protected variables do not alter the ID of this field (because they start with '_')
    _debug_verbose_cvxpy  : set[str]|None = None
    _eps_price            : float = 1E-6
    _eps_density          : float = 1E-7
    _min_density          : float = 1E-5
    _eps_dlv_box          : float = 1E-6
    _dlv_mode             : bool = True
    _raise_on_eps         : bool = True

    def __post_init__(self):
        verify_inp( self.timeout_seconds is None or self.timeout_seconds >= 0., lambda : f"'timeout_seconds' must be positive or None, but found {self.timeout_seconds}" )
        verify_inp( self.smoothness is None or self.smoothness >= 0., lambda : f"'smoothness' must be positive or None, but found {self.smoothness:.4g}")

    def __str__(self) -> str:
        """ Very compact but still readable summary """
        return\
            f"sanos,{f'sth={self.smoothness:.2g},' if self.smoothness is not None else ''}"+\
            f"{',midv='+str(self.min_dlv) if self.min_dlv is not None else ''}{',mxdv='+str(self.max_dlv) if self.max_dlv is not None else ''}{',mindv='+str(self.min_density_vol) if self.min_density_vol is not None else ''}"+\
            ("" if self.min_density_vol is None else f",mdv={self.min_density_vol:.4g}") +\
            ("" if self.min_density_cap is None else f",mdc={self.min_density_cap:.4g}")      

    def __unique_hash__( self, unique_hash, debug_trace ) -> str:
        """ Overwrite hash function to ignore non-functional parameters. Note that unique_hash already ignored protected variables """
        r = asdict(self)
        r.pop("timeout_seconds", None)
        return unique_hash(r, debug_trace=debug_trace)

    def __call__(self,
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                mids           : list[np.ndarray],
                xstrikes       : list[np.ndarray],*,
                weights        : list[np.ndarray]|None = None,
                vols           : np.ndarray|float|None = None, # forward volatilities per expiry, used for the strike expansion step. If a float is provided, it is used for all expiries. If None, it is estimated from the data as the median of the implied vols.
                message_context: str|callable[[], str] | None = None,
                verbose        : Context = Context.all ) -> SanosPricer:
        """
        Fit SANOS to the market data provided following https://arxiv.org/pdf/2601.11209.

        This function performs an L1 closest fit of the SANOS model to the mid prices of the options
        and returns with :class:`SanosPricer` an arbitrage-free surface option pricer.

        This function is usually very fast, but it can get stuck when the overall surface is close to arbitrage in areas which have very little
        room to adjust. 
    
        Parameters
        ----------
        expiries : :class:``numpy.ndarray`
            Increasing array of expiries (in years).
        strikes : list[:class:``numpy.ndarray`]
            List of strike arrays, one per expiry. Each strike array must be strictly increasing and positive.
        mids : list[:class:``numpy.ndarray`]
            List of mid price arrays, one per expiry. 
        xstrikes : list[:class:``numpy.ndarray`]
            List of model strikes for which to calibrate a SANOS density for.
        weights : list[:class:``numpy.ndarray``]|None
            List of weight arrays for the ``mids``, used to weight the fit to the mid prices. If None, all weights are set to 1.
        vols : :class:``numpy.ndarray`` | float | None, default ``None``
            Volatilities per expiry used when ``smoothness`` is not None.
            If left to ``None``, the function will compute an approximate mid-atm volatility from the provided bid and ask prices per expiry.
        message_context : str | callable[[], str] | None, default None
            Optional context to prepend to error and warning messages.
            This can be a string or a callable which returns a string, e.g. a ``lambda`` 
            which returns a string with dynamic information such as the current expiry being processed. If None, no context is prepended.
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

        verify_inp( len(strikes) == T, lambda : f"{message()}Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( len(mids) == T, lambda : f"{message()}Expected number of mids arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( weights is None or len(weights) == T, lambda : f"{message()}Expected number of weights arrays {T} to match number of expiries {len(expiries)}" )
        verify_inp( expiries[0] > 0. and (len(expiries)==1 or np.min(np.diff(expiries)) > 0.), lambda : f"{message()}Expiries must be positive and strictly increasing, but found {expiries}" )

        def make_variances():
            if vols is None:
                atm_calls = []
                for t in range(T):
                    ixs = np.argsort( strikes[t] )
                    atm_calls.append( np.interp( 1., strikes[t][ixs], mids[t][ixs] ) )
                atm_calls = np.asarray(atm_calls)
                atm_vols  = bs.implied( k=np.ones_like(atm_calls), prices=atm_calls, sqrtT=np.sqrt(expiries), eps=self._eps_price )
                variances = expiries * (atm_vols**2)
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

        tme         = verbose.timer()
        variables   = []
        constraints = []
        objective   = 0.
        xdensities  = []
        CXs         = []
        dCts        = []

        for t, expiry in enumerate(expiries):
            def local_message():
                msg = message(False)
                ext = f"expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d"
                if msg == "":
                    msg = ext[0].upper() + ext[1:]
                else:
                    msg += " " + ext
                return msg

            strikes_t  = strikes[t]
            mids_t     = mids[t]
            xstrikes_t = xstrikes[t]
            weights_t  = weights[t] if not weights is None else None

            verify_inp( len(strikes_t) > 1 and strikes_t[0] > 0. and np.min( np.diff(strikes_t) ) > 0., f"{local_message()}: 'strikes' must be positive and increasing.")
            verify_inp( len(xstrikes_t ) > 1 and xstrikes_t[0] > 0. and np.min( np.diff(xstrikes_t) ) > 0., f"{local_message()}: 'xstrikes' must be positive and increasing.")
            verify_inp( strikes_t.shape == mids_t.shape, f"{local_message()}: 'strikes' and 'mids' must have the same shape, found {strikes_t.shape} and {mids_t.shape}")
            if not weights_t is None: verify_inp( strikes_t.shape == weights_t.shape, f"{local_message()}: 'strikes' and 'weights' must have the same shape, found {strikes_t.shape} and {weights_t.shape}")

            if t>0:
                assert strikes_t[0] <= strikes[t-1][0], f"{local_message()}: 'strikes' must be increasing across expiries, but found {strikes[t-1][0]:.4g} and {strikes_t[0]:.4g} for current and previous expiry."
                assert strikes_t[-1] >= strikes[t-1][-1], f"{local_message()}: 'strikes' must be increasing across expiries, but found {strikes[t-1][-1]:.4g} and {strikes_t[-1]:.4g} for current and previous expiry."  

            q_t         = cp.Variable(len(xstrikes_t), bounds=[0.,1.])#nonneg=True)
            constraints += [ cp.sum( q_t ) == 1.,
                            q_t @ xstrikes_t == 1., 
                            q_t <= 1.
                            ]
            if not self.min_density_vol is None:
                bscalls     = bs.price( k=xstrikes_t, vol=self.min_density_vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )
                bsdensity   = np.diff( np.diff( bscalls, prepend=1. ) / np.diff(xstrikes_t, prepend=0.), append=0. )
                bsdensity   = np.clip( bsdensity, a_min=0., a_max=None if self.min_density_cap is None else self.min_density_cap / float(len(xstrikes_t)) )
                constraints += [ q_t >= bsdensity ]
                del bscalls, bsdensity

            # martingale
            Cm          = np.maximum( 0., xstrikes_t[None,:] - xstrikes_t[1:-1,None] ) @ q_t    # internal model prices
            if t>0:
                assert xstrikes[t-1].shape == xdensities[t-1].shape, f"Internal error {message()}: shape mismatch between previous and current mstrikes, {xstrikes[t-1].shape} vs {xdensities[t-1].shape}"
                Cm1         = np.maximum( 0., xstrikes[t-1][None,:] - xstrikes_t[1:-1,None] ) @ xdensities[t-1]
                constraints += [ Cm >= Cm1 ]
            else:
                Cm1   = np.maximum( 0., 1. - xstrikes_t[1:-1] )    # internal model prices, previous density is dirac at 1
            dCt = Cm - Cm1
            dCts.append(dCt)

            # dlv constraints
            # dC = 1/2 * dlvsq * x^2 * dt * K^2 / (1/2 (k^+ - k^-)) q
            box         = xstrikes_t[1:-1]**2 * ( expiries[t] - ( expiries[t-1] if t>0 else 0. ) ) / ( xstrikes_t[2:] - xstrikes_t[:-2])
            constraints += [ dCt >= cp.multiply( box * (self.min_dlv**2), q_t[1:-1] ) ]
            constraints += [ dCt <= cp.multiply( box * (self.max_dlv**2), q_t[1:-1] ) ]
            del box, Cm1, dCt

            # fit
            def call( spots, strikes ):
                if self.smoothness is None or self.smoothness == 0. or expiry < ONE_MINUTE:
                    return np.maximum( 0., spots - strikes )
                vol     = math.sqrt( variances[t] / expiry * self.smoothness ) 
                assert np.isfinite(vol), f"Internal error: {message()}infinite vol"
                return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )*spots

            CX           = call( spots=xstrikes_t[None,:], strikes=strikes_t[:,None] )
            fits_t       = CX @ q_t    # market prices according to the model (with smoothness)
            err_t        = cp.abs(fits_t-mids_t)
            err_t        = err_t if weights_t is None else cp.multiply( weights_t, err_t )
            objective    += cp.sum( err_t )

            xdensities.append(q_t)
            CXs.append(CX)

        with verbose.write_t(lambda : f"Fitting {'linear' if self.smoothness is None else f'{self.smoothness:.2f}-smooth'} SANOS surface... ", end=''):
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
                raise RuntimeError(f"{message()}Cannot find no-arbitrage surface close to mid-points: problem infeasible?? [This should not happen. Please contact the authors with an example.]")
            if status in ["unbounded", "unbounded_inaccurate", "infeasible_or_unbounded"]:
                raise RuntimeError(f"{message()}Cannot find no-arbitrage surface close to mid-points: problem unbounded? [This should not happen. Please contact the authors with an example.]")
            if status == "user_limit":
                raise RuntimeError(f"{message()}Cannot find no-arbitrage surface close to mid-points: reached timeout {self.timeout_seconds} and/or iteration limit.")
            raise RuntimeError(f"{message()}Cannot find no-arbitrage surface close to mid-points: solver returned status '{status}'.")

        if status != "optimal":
            verbose.write(lambda : f"solver returned '{status}' -- proceeding anyway... ", head=False, end='' )

        fits       = []
        xdensities = [ q.value for q in xdensities ]
        for t in range(T):
            expiry     = expiries[t]
            strikes_t  = strikes[t]
            xstrikes_t = xstrikes[t]
            CX         = CXs[t]
            q_t        = xdensities[t]
            if np.min(q_t) < -self._eps_density or abs( np.sum( q_t )-1. ) > self._eps_density or abs( q_t@xstrikes_t - 1. ) > self._eps_density:
                warnings.warn(f"{message()}Warning: no-arbitrage density does not satisfy basic properties after optimization: min {q_t.min():.3e}, sum {q_t.sum():.6f}, integral {q_t@xstrikes_t:.6f} -- attempting fix.")
                q_t = warn_and_fix_density( xstrikes_t, q_t, eps=self._eps_density, message = lambda : f"{message()}expiry {expiry*DAYS_PER_YEAR:.2f}d: surface density after optimization" )

            # debug conditions on dC_t
            box         = xstrikes_t[1:-1]**2 * ( expiries[t] - ( expiries[t-1] if t>0 else 0. ) ) / ( xstrikes_t[2:] - xstrikes_t[:-2])
            q_box       = box * q_t[1:-1]
            dCt         = dCts[t].value
            lower       = dCt - np.maximum((self.min_dlv**2)*q_box, 0.) # must be >= 0
            upper       = (self.max_dlv**2)*q_box - dCt # must be >= 0
            if not np.all(lower >= -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box):
                index  = np.arange((len(lower)))[lower < -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box][:4]
                msg    = f"{message()}expiry {expiry*DAYS_PER_YEAR:.2f}d: dC_t[{index}] {dCt[index]} below min dlv constraint {np.maximum((self.min_dlv**2)*q_box, 0.)[index]} difference {lower[index]} [tolerance {self._eps_dlv_box} -> {(+self._eps_dlv_box + np.abs(dCt)*self._eps_dlv_box)[index]}]"
                warnings.warn(msg)
                assert not self._raise_on_eps or np.all(lower >= -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box), msg

            if not np.all(upper >= -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box):
                index = np.arange((len(upper)))[upper < -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box][:4]
                msg    = f"{message()}expiry {expiry*DAYS_PER_YEAR:.2f}d: dC_t[{index}] {dCt[index]} above max dlv constraint {((self.max_dlv**2)*q_box+1E-6)[index]} difference {upper[index]} [tolerance {self._eps_dlv_box} -> {(self._eps_dlv_box + np.abs(dCt)*self._eps_dlv_box)[index]}]"
                warnings.warn(msg)
                assert not self._raise_on_eps or np.all(upper >= -self._eps_dlv_box - np.abs(dCt)*self._eps_dlv_box), msg

            # zero out some density
            floor      = max( self._min_density / float(len(q_t)), self._eps_density ) if IMPOSE_FLOOR_ON_DENSITY else 0.
            is_zero    = q_t < floor
            if np.any(is_zero):
                q_t[is_zero]  = 0.

            q_t           = warn_and_fix_density( xstrikes_t, q_t, eps=self._eps_density, message = lambda : f"{message()} expiry {expiry*DAYS_PER_YEAR:.2f}d: surface density" )
            xdensities[t] = q_t
            fit_t         = CX @ q_t
            assert fit_t.shape == strikes_t.shape, f"Internal error: {message()}Shape of fit_t {fit_t.shape} does not match shape of strikes_t {strikes_t.shape}"
            fits.append( fit_t )

        verbose.write(lambda : f"done. This took {tme}.", head=False )

        return SanosPricer( smoothness=self.smoothness,
                            expiries=expiries, 
                            variances=variances, 
                            xstrikes=xstrikes, 
                            xdensities=xdensities,
                            min_dlv=self.min_dlv,
                            max_dlv=self.max_dlv,)




