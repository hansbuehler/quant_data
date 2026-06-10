"""
SANOS (Jax)
=============

Fitting the DLVs of SANOS.
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
import scipy
from sympy import true
from quantdata.biz_calendar import DAYS_PER_YEAR, ONE_MINUTE
from quantdata.utils.backbone import Backbone, BackboneNumpy
import math

from quantdata.sanos.sanos_tools import SanosPricer, SanosLPSolver, warn_and_fix_density, make_variances
from quantdata.sanos.sanos import SanosFitter
from quantdata.sanos.tridiag import solve_sanos_tridiagonal
from quantdata.jax.bbjax import BackboneJax

_bbjax =BackboneJax()

import jax
import jax.numpy as jnp

@version("0.1", dependencies=[SanosFitter])
def fit_dlvsqs( self           : SanosFitter,                
                expiries       : np.ndarray, 
                strikes        : list[np.ndarray],
                mids           : list[np.ndarray],
                xstrikes       : list[np.ndarray],*,
                weights        : list[np.ndarray]|None = None,
                vols           : np.ndarray|float|None = None, # forward volatilities per expiry, used for the strike expansion step. If a float is provided, it is used for all expiries. If None, it is estimated from the data as the median of the implied vols.
                init_dlvs      : list[np.ndarray]|None = None, # initial DLVs for the optimization. If None, they are initialized to 0.2.
                message_context: str|callable[[], str] | None = None,
                debug_do_not_optimize : bool = False, # if True, the function will return a pricer with the initial DLVs without optimizing them. This is useful for debugging and testing.
                debug_print_info : bool = False,
                verbose        : Context = Context.all ) -> SanosPricer:
    """
    Fits an implied surface for the given prices and strikes by directly optimizing for the DLVs.    
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
    
    txt_smth = "linear" if self.smoothness is None else f"{self.smoothness:.2f}-smooth"

    T = len(expiries)

    verify_inp( len(strikes) == T, lambda : f"{message()}Expected number of strike arrays {T} to match number of expiries {len(expiries)}" )
    verify_inp( len(mids) == T, lambda : f"{message()}Expected number of mids arrays {T} to match number of expiries {len(expiries)}" )
    verify_inp( weights is None or len(weights) == T, lambda : f"{message()}Expected number of weights arrays {T} to match number of expiries {len(expiries)}" )
    verify_inp( len(xstrikes) == T, lambda : f"{message()}Expected number of xstrike arrays {T} to match number of expiries {len(expiries)}" )
    verify_inp( init_dlvs is None or len(init_dlvs) == T, lambda : f"{message()}Expected number of init_dlvs arrays {T} to match number of expiries {len(expiries)}" )  
    verify_inp( expiries[0] > 0. and (len(expiries)==1 or np.min(np.diff(expiries)) > 0.), lambda : f"{message()}Expiries must be positive and strictly increasing, but found {expiries}" )

    assert not debug_do_not_optimize or not init_dlvs is None, f"{message()}If 'debug_do_not_optimize' is True, 'init_dlvs' must be provided to initialize the DLVs, but found init_dlvs={init_dlvs} and debug_do_not_optimize={debug_do_not_optimize}" 
    variances = make_variances(expiries, strikes, mids, vols=vols, message_context=message_context)
    del vols

    tme         = verbose.timer()
    jdtype      = jnp.float64
    jdevice     = jax.devices("cpu")[0]
    max_dlvsq   = self.max_dlv**2 if not self.max_dlv is None else None
    min_dlvsq   = self.min_dlv**2 

    class JaxObjective:
        def __init__( jself ):
            jself.iw_minus = []
            jself.iw_plus  = []
            jself.L        = []
            jself.CX       = []
            jself.mids     = []
            jself.weights  = None
            jself.xdensity0 = None
            # x
            jself.num_dlvs   = None
            jself.dlv_slices = []

        def verify( jself, condition: bool|Backbone.Array, message : str|callable, exception : Exception = RuntimeError  ):
            def callback( condition ):
                if jnp.all( condition ):
                    return
                raise exception( message() if callable(message) else message )
            jax.debug.callback( callback, condition )
        
        def __call__( jself, rawdlvs : jnp.ndarray, training : bool = True ) -> jnp.ndarray | tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]] :
            q_tm1        = jself.xdensity0
            error        = 0.
            wsum         = 0.

            assert rawdlvs.shape == (jself.num_dlvs,), f"Expected rawdlvs to have shape {(jself.num_dlvs,)}, but found {rawdlvs.shape}"
            rawdlvs      = [ rawdlvs[slice_] for slice_ in jself.dlv_slices ]
            xdensities   = [] if not training else None
            calls        = [] if not training else None

            for t, expiry in enumerate(expiries):
                dlvinp_t    = rawdlvs[t]
                iw_minus_t  = jself.iw_minus[t]
                iw_plus_t   = jself.iw_plus[t]
                L_t         = jself.L[t]
                CX_t        = jself.CX[t]
                mids_t      = jself.mids[t]
                weights_t   = jself.weights[t] if not jself.weights is None else 1.

                # move density forward
                if t>0:
                    q_tm1   = L_t @ q_tm1
                else:
                    q_tm1   = jself.xdensity0

                jself.verify( jnp.min( q_tm1 ) >= -1E-8, lambda : f"{message()}Negative density values found at expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d, before solving DLV equation: min density value is {np.min(q_tm1)}. This may indicate an issue with the strike expansion step. Consider providing a better 'vols' input to improve the strike expansion, or check the input data for inconsistencies {jnp.min( q_tm1 )}." )

                # solve DLV equation
                dlvsqs_t    = ( jax.nn.sigmoid(dlvinp_t) * (max_dlvsq - min_dlvsq) + min_dlvsq ) if not max_dlvsq is None else\
                                ( jnp.exp(dlvinp_t) + min_dlvsq )
                w_minus_t   = dlvsqs_t * iw_minus_t
                w_plus_t    = dlvsqs_t * iw_plus_t
                q_t         = solve_sanos_tridiagonal( w_minus=w_minus_t, w_plus=w_plus_t, rhs=q_tm1, backbone=_bbjax )
                assert q_t.shape == xstrikes[t].shape, f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: shape error in DLV equation solution, expected {xstrikes[t].shape} but found {q_t.shape}"

                # compute market call prices
                C_t          = CX_t @ q_t
                assert C_t.shape == strikes[t].shape, f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: shape of model call prices C_t must match shape of strikes_t, but found {C_t.shape} and {strikes[t].shape}"

                # add errors
                error_t       = weights_t * ( C_t - mids_t )
                error_t       = error_t**2 #jnp.minimum( error_t, 1. ) + 0.5 * jnp.maximum( error_t-1., 0 )**2
                error        += jnp.sum( error_t )
                wsum         += jnp.sum( weights_t )

                if not training:
                    xdensities.append( q_t )
                    calls.append( C_t )

                q_tm1 = q_t 

            error = error / wsum
            if training:
                return error
            return error, xdensities, calls

    jself = JaxObjective()

    # initial density
    # ----------------

    jself.xdensity0   = jax.device_put( np.diff( np.diff( np.maximum( 0., 1. - xstrikes[0] ), prepend=1. ) / np.diff(xstrikes[0], prepend=0.), append=0. ), device=jdevice )
    jself.num_dlvs    = 0
    init_dlvsraw      = None if init_dlvs is None else []

    for t, expiry in enumerate(expiries):
        def local_message():
            msg = message(False)
            ext = f"expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d"
            if msg == "":
                msg = ext[0].upper() + ext[1:]
            else:
                msg += " " + ext
            return msg

        # fit
        def np_call( spots, strikes ):                
            if self.smoothness is None or self.smoothness == 0. or expiry < ONE_MINUTE:
                return np.maximum( 0., spots - strikes )
            expiry_ = np.clip( expiry, expiries[0], expiries[-1] )
            vol     = math.sqrt( self.smoothness * np.interp( expiry_, expiries, variances ) / expiry_ )
            assert np.isfinite(vol), f"Internal error: {message()}infinite vol"
            if vol == 0.:
                return np.maximum( 0., spots - strikes )
            return bs.price( k=strikes/spots, vol=vol, sqrtT=math.sqrt(expiry), eps=self._eps_price )*spots

        strikes_t  = strikes[t]
        mids_t     = mids[t]
        xstrikes_t = xstrikes[t]
        weights_t  = weights[t] if not weights is None else None

        verify_inp( len(strikes_t) > 1 and strikes_t[0] > 0. and jnp.min( jnp.diff(strikes_t) ) > 0., f"{local_message()}: 'strikes' must be positive and increasing.")
        verify_inp( len(xstrikes_t ) > 1 and xstrikes_t[0] > 0. and jnp.min( jnp.diff(xstrikes_t) ) > 0., f"{local_message()}: 'xstrikes' must be positive and increasing.")
        verify_inp( strikes_t.shape == mids_t.shape, f"{local_message()}: 'strikes' and 'mids' must have the same shape, found {strikes_t.shape} and {mids_t.shape}")
        if not weights_t is None: verify_inp( strikes_t.shape == weights_t.shape, f"{local_message()}: 'strikes' and 'weights' must have the same shape, found {strikes_t.shape} and {weights_t.shape}")

        if t>0:
            assert strikes_t[0] <= strikes[t-1][0], f"{local_message()}: 'strikes' must be increasing across expiries, but found {strikes[t-1][0]:.4g} and {strikes_t[0]:.4g} for current and previous expiry."
            assert strikes_t[-1] >= strikes[t-1][-1], f"{local_message()}: 'strikes' must be increasing across expiries, but found {strikes[t-1][-1]:.4g} and {strikes_t[-1]:.4g} for current and previous expiry."  

        Nj               = len(xstrikes_t)-2
        jself.dlv_slices.append( slice(jself.num_dlvs, jself.num_dlvs+Nj) )
        jself.num_dlvs   += Nj

        if not init_dlvs is None:
            init_dlvsqs_t = init_dlvs[t]**2
            assert len(init_dlvsqs_t) == Nj, f"{message()} init_dlvs for expiry #{t} must have length {Nj}, but found {len(init_dlvsqs_t)}"
            if not max_dlvsq is None:
                ddlvsq        = max_dlvsq - min_dlvsq
                init_dlvsqs_t = np.clip( init_dlvsqs_t -min_dlvsq, min=1E-8, max=ddlvsq-1E-8 ) / ddlvsq
                assert np.min( init_dlvsqs_t) > 0. and np.max( init_dlvsqs_t ) < 1. , f"{message()} init_dlvs for expiry #{t} must be between 0 and 1 found min {np.min(init_dlvsqs_t)} and max {np.max(init_dlvsqs_t)}"
                init_dlvsqs_t = np.log( init_dlvsqs_t / ( 1. - init_dlvsqs_t ) )
            else:
                init_dlvsqs_t = np.clip( init_dlvsqs_t, min=min_dlvsq+1E-8 ) - min_dlvsq
                init_dlvsqs_t = np.log( init_dlvsqs_t )
            init_dlvsraw.append( init_dlvsqs_t )

        del Nj

        # compute transition 1
        # --------------------
        dt          = expiry - ( expiries[t-1] if t>0 else 0. )
        box         = xstrikes_t[1:-1]**2 * dt / ( xstrikes_t[2:] - xstrikes_t[:-2])
        w_plus      = box / ( xstrikes_t[2:] - xstrikes_t[1:-1] )
        w_minus     = box / ( xstrikes_t[1:-1] - xstrikes_t[:-2] )
        assert np.min( w_plus ) >= 0., f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: computed 'w_plus' must be non-negative, but found min {np.min(w_plus)}"
        assert np.min( w_minus ) >= 0., f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: computed 'w_minus' must be non-negative, but found min {np.min(w_minus)}"
    
        jself.iw_minus.append( jax.device_put( w_minus, device=jdevice ) )
        jself.iw_plus.append( jax.device_put( w_plus, device=jdevice ) )

        # interpolation matrx L
        # https://arxiv.org/pdf/2601.11209 (22)
        # we compute this in higher precision
        if t>0:
            xstrikes_tm1 = xstrikes[t-1] 

            L = np.maximum( 0., xstrikes_tm1[None,:] - xstrikes_t[:,None] )
            L = np.diff( np.diff( L, axis=0, prepend=1. ) / np.diff( xstrikes_t, prepend=0. )[:,None], axis=0, append=0. )
            assert L.shape == (len(xstrikes_t), len(xstrikes_tm1)), f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: expected shape {(len(xstrikes_t), len(xstrikes_tm1))} but found {L.shape} for interpolation matrix L"
            jself.L.append( jax.device_put( L, device=jdevice ) )
        else:
            jself.L.append( None )

        # objective
        # ---------
        jself.CX.append( jax.device_put( np_call( spots=xstrikes_t[None,:], strikes=strikes_t[:,None] ), device=jdevice ) )
        jself.weights.append( jax.device_put( weights_t, device=jdevice ) ) if not weights_t is None else None
        jself.mids.append( jax.device_put( mids_t, device=jdevice ) )

    # DLVs
    # ----
    dlvsraw  = jnp.full((jself.num_dlvs,), 0., dtype=jdtype, device=jdevice) if init_dlvs is None else\
                jax.device_put( np.concatenate( init_dlvsraw ), device=jdevice )

    _     = jself(dlvsraw)

    # compile
    # -------
    
    with verbose.write_t(lambda : f"{message()}compiling optimizer for {jself.num_dlvs} {txt_smth} DLVs on {jdevice}... ", end='') as tme:
        jself.value_grad     = jax.jit( jax.value_and_grad( lambda x : jself.__call__(x, training=True), has_aux=False ) ).lower(dlvsraw).compile() if not debug_do_not_optimize else None
        jself.value_and_data = jax.jit( lambda x : jself.__call__(x, training=False) ).lower(dlvsraw).compile()
        verbose.write(f"done; this took {tme}.", head=False)

    # compile
    # -------

    if debug_do_not_optimize:
        result_raw = dlvsraw
    
    else:
        with verbose.write_t(lambda : f"{message()}optimizing {jself.num_dlvs} {txt_smth} DLVs on {jdevice}... ", end='') as tme:
            res = scipy.optimize.minimize(
                jself.value_grad,
                dlvsraw,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": 2000,
                    "maxfun": 10000,
                    "maxls": 100,
                    "maxcor": 20,

                    # Make stopping less dominated by tiny relative f improvement.
                    "ftol": 1e-14,

                    # This is the projected-gradient tolerance.
                    "gtol": 1e-8,
                },
            )

            if not res.success :
                raise RuntimeError( f"{message()}Optimizer failed with message: {res.message}" )

            verbose.write(f"done; optimizer returned '{res.message}'. This took {tme}.", head=False)
            if debug_print_info:
                print("success:", res.success)
                print("message:", res.message)
                print("nit:", res.nit)
                print("nfev:", res.nfev)
                print("fun:", res.fun)
                print("|grad|_inf:", np.max(np.abs(res.jac)))
                print("|grad|_2:", np.linalg.norm(res.jac))

            result_raw = res.x

    # evaluate
    # --------

    if debug_print_info:
        rawdlvsqs    = ( jax.nn.sigmoid(result_raw) * (max_dlvsq - min_dlvsq) + min_dlvsq ) if not max_dlvsq is None else\
                        ( jnp.exp(result_raw) + min_dlvsq )
        rawdlvs      = [ np.sqrt(rawdlvsqs[slice_]) for slice_ in jself.dlv_slices ]
        for t, expiry in enumerate(expiries):
            print(f"Expiry #{t} {expiry*DAYS_PER_YEAR:.1f}d: raw DLVs: {rawdlvs[t]}")
    
    err, xdensities, calls = jself.value_and_data( dlvsraw )
    err        = jax.device_get( err )
    xdensities = [ np.clip( jax.device_get( xd ), min=0. ) for xd in xdensities ]
    calls      = [ np.clip( jax.device_get( c ), min=0. ) for c in calls ]
    if debug_print_info:
        print(f"Final root mean squared error: {err:.4g} in price space")

    return SanosPricer( smoothness=self.smoothness,
                        expiries=expiries, 
                        variances=variances, 
                        xstrikes=xstrikes, 
                        xdensities=xdensities,
                        min_dlv=self.min_dlv,
                        max_dlv=self.max_dlv,)




