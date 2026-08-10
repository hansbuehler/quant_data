"""
Deterministic measure change
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_enable_x64", True)        

from cdxcore.dynaplot import figure, m_o_m
from cdxcore.verbose import Context
from cdxcore.pretty import PrettyObject
from cdxcore.util import fmt_big_byte_number
from cdxcore.err import verify_inp
import numpy as np
from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.jax.utility import jax_utility
import jax.numpy as jnp
import scipy
import warnings

def stats( X : np.ndarray, D : np.ndarray|None = None ) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the mean and standard error of a 2D array X along axis=0. """
    assert X.shape[0] > 0, f"Expected X to have at least one row but got {X.shape}"
    assert len(X.shape) == 2 or len(X.shape) == 4, f"Expected X to be a 2D or 4D array but got {X.shape}"
    assert D is None or D.shape == X.shape[:1], f"Expected D to have shape {X.shape[:1]} but got {D.shape}"

    DX   = ( D[:,None]*X if X.ndim == 2 else D[:,None,None,None]*X ) if not D is None else X
    TAU  = X.shape[1]
    T, N = X.shape[2:] if len(X.shape) > 2 else (None, None)
    del D

    r = PrettyObject()
    r.mean = np.mean(DX, axis=0)
    r.err  = np.sqrt(np.mean((DX - r.mean[None,...])**2, axis=0)) / np.sqrt(X.shape[0]) 
    r.dev  = np.where( r.err > 0., r.mean/np.where( r.err>0., r.err, 1.), 0.)

    # if we have more than two original dimensions -> per step stats
    if len(r.mean) > 1:
        mean = r.mean.reshape( r.mean.shape[0], -1 )
        r.mean_mean = np.mean(mean, axis=1)
        r.mean_q10,r.mean_med,r.mean_q90 = np.quantile(mean, [0.1, 0.5, 0.9], axis=1) 
        dev = r.dev.reshape( r.dev.shape[0], -1 )
        r.dev_mean = np.mean(dev, axis=1)
        r.dev_q10,r.dev_med,r.dev_q90 = np.quantile(dev, [0.1, 0.5, 0.9], axis=1) 

    # summary stats accross time
    r.total_q10, r.total_med, r.total_q90 = np.quantile( r.mean, [0.1, 0.5, 0.9] )
    r.tdev_q10, r.tdev_med, r.tdev_q90 = np.quantile( r.dev, [0.1, 0.5, 0.9] )
    r.info = f"{r.tdev_med:.4g} [{r.tdev_q10:.4g}, {r.tdev_q90:.4g}]"  # <- shows the percentiles of mean/err accross all instruments
    return r

class MeasureChange:
    
    def __init__(self, mode="powexp", lmbda=1.0, *, device = "cpu" ):
        r"""
        sup_a: E[ 1 - exp(-adS) ] = sup_a E[ - exp(-\sum_t a_t dS_t) + 1 ] = inf_a E[ exp(-\sum_t a_t dS_t) - 1 ]
        Optimal point:
        t: E[ - exp(-\sum_t a_t dS_t) dS_t ] = 0

        =>
        a* : 0 = E[dS] - a* E[dS^2] => a* = E[dS] / E[dS^2]
        """
        self.mode  = mode
        self.lmbda = lmbda
        self.u, self.deriv, self.u_label = jax_utility(mode, lmbda)
        self.device = jax.devices(device)[0] if isinstance(device, str) else device

        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false", f"Expected XLA_PYTHON_CLIENT_PREALLOCATE to be 'false' but got '{os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}'"

    def _solve_jax(self, 
                   dS, dC, mode : str = "dS", *,
                   verbose : Context = Context.all):
        """
        Find a risk-neutral measure change dQ/dP for the given dX using the utility function specified by self.mode and self.lmbda.
        The measure mean(dQ/dP) = 1
        """
        assert len(dS.shape) == 2, f"Expected dS to be a 2D array but got {dS.shape}"

        assert len(dS.shape) == 2, f"Expected dS to be a 2D array but got {dS.shape}"
        assert len(dC.shape) == 4, f"Expected dC to be a 4D array but got {dC.shape}"
        assert dS.shape == dC.shape[:2], f"Expected dS to have shape {dC.shape[:2]} but got {dS.shape}"

        if mode == "dS":
            dX = dS
        elif mode == "dS,dC":
            dX = np.concatenate( (dS, dC.reshape(dC.shape[0], -1)), axis=1 )
        else:
            assert mode == "dS,dC[0]", f"Expected mode to be 'dS' or 'dS,dC' or 'dS,dC[0]' but got '{mode}'"
            dC0 = dC[:,0,:,:]
            dX = np.concatenate( (dS, dC0.reshape(dC0.shape[0], -1)), axis=1 )

        assert len(dX.shape) == 2, f"Shape error {dX.shape}"
        msg  = lambda : f"\rSolving for {mode} measure change accross {dX.shape[1]} variables using '{self.u_label}'"
        tme = verbose.timer()
        verbose.write(lambda : f"{msg()}: copying dX of shape {dX.shape} and size {fmt_big_byte_number(dX.nbytes)} to {self.device}... ", end='')
        jdX = jax.device_put( dX.astype(np.float64), device=self.device )
        ay0 = jnp.zeros( (dX.shape[1]+1,), device=self.device, dtype=jdX.dtype )
    
        def objective( ay : jnp.array ):
            a = ay[1:]
            y = ay[0]
            X = jnp.sum( a[None,:]*jdX, axis=1)
            return -jnp.mean( self.u(X+y)-y) 
        
        verbose.write(lambda : f"{msg()}: compiling objective; so far this took {tme}... ", end='')
        ff = jax.jit( objective ).lower( ay0 ).compile()
        def f(a):
            a = jax.device_put( a, device=self.device )
            return jax.device_get( ff(a) )

        verbose.write(lambda : f"{msg()}: compiling gradient; so far this took {tme}... ", end='')
        jacc = jax.jit( jax.grad(objective) ).lower( ay0 ).compile()
        def jac(a):
            a = jax.device_put( a, device=self.device )
            return jax.device_get( jacc(a) )

        with verbose.write_t(lambda : f"{msg()}: running scipy optimizer; so far this took {tme}... ", end='') as tme:

            r   = scipy.optimize.minimize(  fun=f, 
                                            x0=ay0, 
                                            jac=jac, 
                                            method="BFGS", options={"disp": False, "gtol": 1e-8, "maxiter": 1000} )

            ok_msg = {"Desired error not necessarily achieved due to precision loss.",
                    "Maximum number of iterations has been exceeded."}
                
            if not r.success:
                if r.message not in ok_msg:
                    raise RuntimeError(f"Solving for measure change using '{self.u_label}': scipy optimizer failed with message: '{r.message}'.")
                warnings.warn(f"Solving for measure change using '{self.u_label}': scipy optimizer failed with message: '{r.message}' -- continuing.")

            ax   = r.x[1:]
            yx   = r.x[0]
            X    = jnp.sum( ax[None,:]*jdX, axis=1)
            dQdP = self.deriv(X+yx)
            assert np.min( dQdP )>=-1E-6, f"Found negative dQdP {np.min( dQdP )}: {dQdP[dQdP<0.]}"
            dQdP = jax.device_get(dQdP)
            mean = np.mean(dQdP)
            if np.max( np.abs(mean)-1. )>1E-5:
                warnings.warn(f"dQdP does not sum up to 1 but {mean}.")

            assert np.max( np.abs(mean)-1. )<1E-2, f"dQdP does not sum up to 1 but {mean}.` yx={yx}, ax={ax}"
            dQdP = dQdP / mean
            assert np.all( np.isfinite(dQdP) ), f"Found non-finite dQdP: {dQdP}"

            mean_dS = np.mean( dQdP[:,None] * dS, axis=0 )
            mean_dC = np.mean( dQdP[:,None] * dC.reshape((dS.shape[0],-1)), axis=0  )
            spot_q05,spot_q95 = np.quantile(mean_dS.reshape((-1,)), [0.05, 0.95])
            opt_q05,opt_q95   = np.quantile(mean_dC.reshape((-1,)), [0.05, 0.95])
            verbose.write(f"found optimal measure. This took {tme}.", head=False)

        verbose.report(1,f"Density sum is {np.mean(dQdP):.4g} and minimum is {np.min(dQdP):.4g}.")
        verbose.report(1,f"Mean quantiles are {spot_q05:.4g} and {spot_q95:.4g} for spot, {opt_q05:.4g} and {opt_q95:.4g} for options.")
        return dQdP

    def _solve_cvx( self, dS, dC, mode : str = "dS", *, patch : bool = False, verbose : Context = Context.all):
        """
        Convex solver
        If ``patch`` is True, only solve spot and front month.
        """
        import cvxpy as cp
        assert len(dS.shape) == 2, f"Expected dS to be a 2D array but got {dS.shape}"
        assert len(dC.shape) == 4, f"Expected dC to be a 4D array but got {dC.shape}"
        assert dS.shape == dC.shape[:2], f"Expected dS to have shape {dC.shape[:2]} but got {dS.shape}"

        if mode == "dS":
            dX = dS
        elif mode == "dS,dC":
            dX = np.concatenate( (dS, dC.reshape(dC.shape[0], -1)), axis=1 )
        else:
            assert mode == "dS,dC[0]", f"Expected mode to be 'dS' or 'dS,dC' or 'dS,dC[0]' but got '{mode}'"
            dC0 = dC[:,0,:,:]
            dX = np.concatenate( (dS, dC0.reshape(dC0.shape[0], -1)), axis=1 )

        assert len(dX.shape) == 2, f"Expected dX to be a 2D array but got {dX.shape}"
        D    = cp.Variable( dX.shape[0], nonneg=True )
        mean = cp.mean( cp.multiply( D[:,None], dX), axis=0 )
        constraints = [
                mean == 0.,
                cp.mean(D) == 1.
        ]
        objective = cp.Minimize( -cp.sum( cp.entr(D) ) )

        with verbose.write_t(f"cvxpy {mode}: Finding optimial measure for {dX.shape[1]} instruments using cvxpy with {dX.shape[0]} samples... ", end='') as tme:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=True, max_iters=2_000)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise RuntimeError(f"Convex solver failed with status '{prob.status}'")
            dQdP              = D.value / np.mean(D.value)
            assert len(dQdP.shape) == 1
            mean_dS           = np.mean( dQdP[:,None] * dS , axis=0 )
            mean_dC           = np.mean( dQdP[:,None] * dC.reshape((dC.shape[0], -1)), axis=0 )
            spot_q05,spot_q95 = np.quantile(mean_dS.reshape((-1,)), [0.05, 0.95])
            opt_q05,opt_q95   = np.quantile(mean_dC.reshape((-1,)), [0.05, 0.95])
            verbose.write(f"found optimal measure with status '{prob.status}'. This took {tme}.", head=False)

        verbose.report(1,f"Density sum is {np.mean(dQdP):.4g} and minimum is {np.min(dQdP):.4g}.")
        verbose.report(1,f"Mean quantiles are {spot_q05:.4g} and {spot_q95:.4g} for spot, {opt_q05:.4g} and {opt_q95:.4g} for options.")
        return dQdP

    def plot(self, label, dS, dC, dQdP, verbose : Context = Context.all ):
        stats_S_0  = stats(dS, None)
        stats_C_0  = stats(dC, None) if not dC is None else None
        stats_S_t  = stats(dS, dQdP)
        stats_C_t  = stats(dC, dQdP) if not dC is None else None
        verbose.report(1,f"{label}: dS stats changed from {stats_S_0.info} to {stats_S_t.info}.")
        verbose.report(1,f"{label}: dC stats changed from {stats_C_0.info} to {stats_C_t.info}.") if not dC is None else None

        with figure(f"Measure change {label}", row_size=4, col_size=6, columns=None ) as fig:
            ax_ret  = fig.add_subplot("Spot expected returns and MC error")
            ax_spot = fig.add_subplot("Spot expected returns / MC error")
            ax_opt  = fig.add_subplot("Options expected returns / MC error\n(ranges indicate percentiles across all instruments)") if not dC is None else None
            ax_D    = fig.add_subplot("Density")
            x_ax    = np.arange(dS.shape[1], dtype=float)

            ax_ret.set_xlabel("Days")
            ax_ret.set_ylabel(r"$E[dS]$")
            ax_ret.plot( x_ax, stats_S_0.mean, "-", color="red", label="Market" )
            ax_ret.plot( x_ax, stats_S_t.mean, "-", color="blue", label="dQ/dP[dS]" )
            ax_ret.fill_between( x_ax, stats_S_0.mean-2*stats_S_0.err, stats_S_0.mean+2*stats_S_0.err, color="red", alpha=0.2, label=r"Market 2$\sigma$" )
            ax_ret.fill_between( x_ax, stats_S_t.mean-2*stats_S_t.err, stats_S_t.mean+2*stats_S_t.err, color="blue", alpha=0.2, label=r"dQ/dP[dS] 2$\sigma$" )
            ax_ret.legend(loc="upper right")

            ax_spot.set_xlabel("Days")
            ax_spot.set_ylabel(r"$E[dS] / \epsilon[dS]$")
            ax_spot.plot( x_ax,  x_ax*0.+2, ":", color="black")
            ax_spot.plot( x_ax, x_ax*0.-2., ":", color="black")
            ax_spot.plot( x_ax, x_ax*0., "--", color="black" )
            ax_spot.plot( x_ax, stats_S_0.dev, "-", color="red", label="Market" )
            ax_spot.plot( x_ax, stats_S_t.dev, "-", color="blue", label="dQ/dP[dS]" )
            mom = m_o_m( stats_S_0.dev, stats_S_t.dev)
            ax_spot.set_ylim( min(-4,mom[0]), max(4, mom[1]) )

            if not dC is None:
                ax_opt.set_xlabel("Days")
                ax_opt.set_ylabel(r"$E[dC] / \epsilon[dC]$")
                ax_opt.plot( x_ax,  x_ax*0.+2, ":", color="black")
                ax_opt.plot( x_ax, x_ax*0.-2., ":", color="black")
                ax_opt.plot( x_ax, x_ax*0., "--", color="black" )
                ax_opt.plot( x_ax, stats_C_0.dev_mean, "-", color="red", label="Market" )
                ax_opt.plot( x_ax, stats_C_t.dev_mean, "-", color="blue", label="dQ/dP[dS]" )
                ax_opt.fill_between( x_ax, stats_C_0.dev_q10, stats_C_0.dev_q90, color="red", alpha=0.2, label="Market q10-q90" )
                ax_opt.fill_between( x_ax, stats_C_t.dev_q10, stats_C_t.dev_q90, color="blue", alpha=0.2, label="dQ/dP[dS] q10-q90" )
                ax_opt.legend(loc="upper right")
                mom = m_o_m( stats_C_0.dev_q10, stats_C_0.dev_q90, stats_C_t.dev_q10, stats_C_t.dev_q90 )
                ax_opt.set_ylim( min(-4,mom[0]), max(4, mom[1]) )

            x = np.arange(1,len(dQdP)+1)
            mean = np.mean(dQdP)
            ax_D.set_title(f"Density\nMean {mean:.4g}, min {np.min(dQdP):.4g}, max {np.max(dQdP):.4g}")
            ax_D.plot( x, dQdP, "-", color="orange", label="dQ/dP" )
            ax_D.plot( x, x*0.+mean, "-", color="blue", label=f"Mean {mean:.4g}")
            ax_spot.legend(loc="upper right")

    def solve( self, 
                dS : np.ndarray, 
                dC : np.ndarray,*,
                mode : str      = "dS,dC[0]",
                method : str    = "jax",
                num_iter : int  = 10,
                plot : bool     = True,
                verbose : Context = Context.all ):
        """ Find a risk-neutral measure change dQ/dP for the given dX using the utility function specified by self.mode and self.lmbda. """
        assert len(dS.shape)==2, f"Expected dS to be a 2D array but got {dS.shape}"
        assert ( dC.shape[:2] == dS.shape and len(dC.shape) == 4 ), f"Expected dC to be None or a 2D array of shape {dS.shape} but got {dC.shape}"
        
        if method == "jax":
            dQdP = self._solve_jax(mode=mode, dS=dS,dC=dC, verbose=verbose)
        else:
            verify_inp( method == "cvxpy", f"'method' must be 'jax' or 'cvxpy'; found {method}")
            dQdP = self._solve_cvx(mode=mode, dS=dS,dC=dC, verbose=verbose)
        if plot:
            self.plot(method + " " + mode, dS, dC, dQdP, verbose=verbose)
        return np.asanyarray(dQdP)

def change(universe,
           method : str = "jax",
           mode : str = "dS",
           plot : bool = True,
           verbose:Context=Context.all):

    spots    = universe["spots"]
    C_t      = universe["C_t"]
    C_tp1    = universe["C_tp1"]
    vols_t   = universe["vols_t"]
    vols_tp1 = universe["vols_tp1"]
    expiries = universe["expiries"]
    strikes_t = universe["strikes_t"]
    strikes_tp1 = universe["strikes_tp1"]
    delta_t = universe["delta_t"]

    S, TAU   = universe["spots"].shape
    T, N     = universe["vols_t"].shape[2:]
    dt       = universe['dt']
    time_d   = np.arange(TAU,dtype=float) * dt * DAYS_PER_YEAR

    dC           = C_tp1 - C_t
    dS           = spots[:,1:] - spots[:,:-1]
    delta_hedged = dC - delta_t * dS[:,:,None,None]

    ms_dC = MeasureChange("entropy")
    dQdP  = ms_dC.solve( dS, dC, plot=plot, method=method, mode=mode, verbose=verbose )

    def test_arb(label, dC):
        max_ret_dC = np.max( dC, axis=0 )
        min_ret_dC = np.min( dC, axis=0 )
        label     = f"{label} ({method} {mode})"

        static_up  = ( max_ret_dC >= min_ret_dC ) & ( min_ret_dC >= 0. )
        static_dn  = ( max_ret_dC <= min_ret_dC ) & ( max_ret_dC <= 0. )
        if np.any(static_up) or np.any(static_dn):
            s = ""
            n_total = dC.shape[1]*dC.shape[2]*dC.shape[3]
            if np.any(static_up):
                s = f"Found {np.sum(static_up)}/{n_total} {label} instruments with only positive returns"
            if np.any(static_dn):
                if s == "":
                    s = f"Found {np.sum(static_dn)}/{n_total} {label} instruments with only negative returns"
                else:
                    s += f" and {np.sum(static_dn)}/{n_total} {label} instruments with only negative returns"

            verbose.write(lambda : f"Warning: {s}. These instruments will be ignored for the measure change optimization.")
        else:
            verbose.write(lambda : f"Found no {label} instruments with only positive or only negative returns. All will be used for the measure change optimization.")

    test_arb("option", dC)
    test_arb("delta-hedged option", delta_hedged)

    return dQdP

