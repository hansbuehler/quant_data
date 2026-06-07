"""
Local extension of quantdata.sanos.fitter.Fitter with second-difference regularizer.

FitterRegularized adds a global second-difference smoothness penalty on the density
  λ · Σ_t Σ_i (d_{i-1,t} - 2·d_{i,t} + d_{i+1,t})²
to the CVXPY objective.  This resolves the null space that arises when the
lowest observed call strike K_lo_obs sits above the lowest anchor: anchors
below K_lo_obs appear in no data equation so CLARABEL may fill them
adversarially without a regularizer.

Usage:
    from fitter_regularized import FitterRegularized
    fitter = FitterRegularized(geometry=geom, no_warnings=True,
                               max_error=None, regularizer_weight=1e-4)

Keep regularizer_weight small (1e-4 to 1e-3).  At λ=1e-4 the regularizer
contributes ~1e-10 of the data-term weight on observed anchors (inv_hspread≈1000),
so bid-ask fit quality is unaffected while the left-tail shape is uniquified.
"""

from __future__ import annotations

import math as math
import warnings as warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.verbose import Context

from quantdata.sanos.fitter import Fitter
from quantdata.sanos.grid import Eps
from quantdata.sanos.utils import _interpolate_sqrtVars, _iterative_model_price
from quantdata.utils.backbone import BackboneNumpy

_npbb = BackboneNumpy()


@dataclass(frozen=True)
class FitterRegularized(Fitter):
    """Fitter with a second-difference smoothness regularizer on density.

    Inherits all fields and behaviour from :class:`quantdata.sanos.fitter.Fitter`.
    Only :meth:`fit_prepared` is overridden to inject the regularizer term.
    """

    regularizer_weight: float = 0.0  #: λ for the regularization penalty
    bid_ask_penalty_weight: float = 0.0  #: huge quadratic penalty for prices outside bid-ask spread
    log_dlv_space: bool = False  #: if True, log-barrier regularizer instead of second-diff
    call_only: bool = False  #: if True, ignore puts in the fit (requires opt_is_calls)
    put_only: bool = False  #: if True, ignore calls in the fit (requires opt_is_calls)

    def __post_init__(self):
        super().__post_init__()
        verify_inp(
            self.regularizer_weight >= 0.0,
            f"regularizer_weight must be non-negative, found {self.regularizer_weight}",
        )
        verify_inp(
            self.bid_ask_penalty_weight >= 0.0,
            f"bid_ask_penalty_weight must be non-negative, found {self.bid_ask_penalty_weight}",
        )
        verify_inp(
            not (self.call_only and self.put_only),
            "call_only and put_only cannot both be True",
        )

    # ------------------------------------------------------------------
    # fit_prepared: full copy of Fitter.fit_prepared with the regularizer
    # block inserted after the data-term objective is assembled.
    # ------------------------------------------------------------------

    def fit_prepared(
        self,
        fwd_vols: np.array | float | None,
        opt_strikes: np.array,
        opt_bid: np.array,
        opt_ask: np.array,
        opt_tte: np.array,
        *,
        opt_is_calls: np.array | None = None,
        opt_mask: np.array | None = None,
        opt_interpolate: dict[str | np.ndarray] | None = None,
        return_only_dlvsqs: bool = True,
        verbose: Context = Context.quiet,
    ):
        T, N = self.TN

        if not fwd_vols is None:
            if hasattr(fwd_vols, "shape"):
                verify_inp(
                    fwd_vols.shape == (T,),
                    lambda: f"Expected fwd_vols shape to be {(T,)}, found {fwd_vols.shape}",
                )
                verify_inp(
                    np.min(fwd_vols) >= 0.0,
                    lambda: (
                        f"Forward volatilities must not be negative, found min {np.min(fwd_vols)}"
                    ),
                )
            elif fwd_vols == 0.0:
                fwd_vols = None
            else:
                verify_inp(fwd_vols > 0.0, lambda: "'fwd_vols' cannot be negative")
                fwd_vols = np.full((T,), fwd_vols, dtype=self.geometry.xexpiries.dtype)

        verify_inp(
            len(opt_tte.shape) == 1,
            f"Expected opt_tte to be a 1D array, found shape {opt_tte.shape} (this function does not support batch dimensions)",
        )
        verify_inp(
            opt_strikes.shape == opt_bid.shape == opt_ask.shape == opt_tte.shape,
            lambda: (
                f"Expected 'opt_strikes', 'opt_bid', 'opt_ask' and 'opt_tte' to have the same shape; found {opt_strikes.shape}, {opt_bid.shape}, {opt_ask.shape}, {opt_tte.shape}"
            ),
        )
        verify_inp(
            opt_is_calls is None or opt_is_calls.shape == opt_tte.shape,
            lambda: (
                f"Expected 'opt_is_calls' to have the same shape as 'opt_tte'; found {opt_is_calls.shape} and {opt_tte.shape}"
            ),
        )
        verify_inp(
            opt_mask is None or opt_mask.shape == opt_tte.shape,
            lambda: (
                f"Expected 'opt_mask' to have the same shape as 'opt_tte'; found {opt_mask.shape} and {opt_tte.shape}"
            ),
        )

        # mask by removing options
        if not opt_mask is None:
            opt_strikes = opt_strikes[opt_mask]
            opt_bid = opt_bid[opt_mask]
            opt_ask = opt_ask[opt_mask]
            opt_tte = opt_tte[opt_mask]
            opt_is_calls = opt_is_calls[opt_mask] if opt_is_calls is not None else None
            opt_interpolate = (
                {k: v[opt_mask] for k, v in opt_interpolate.items()}
                if opt_interpolate is not None
                else None
            )

        # filter by option type (call_only / put_only)
        if (self.call_only or self.put_only) and opt_is_calls is not None:
            type_mask = opt_is_calls if self.call_only else ~opt_is_calls
            # merge into opt_mask so the return-path expansion still works
            if opt_mask is not None:
                full_type_mask = np.zeros(opt_mask.shape, dtype=bool)
                full_type_mask[opt_mask] = type_mask
                opt_mask = full_type_mask
            else:
                opt_mask = type_mask
            opt_strikes = opt_strikes[type_mask]
            opt_bid = opt_bid[type_mask]
            opt_ask = opt_ask[type_mask]
            opt_tte = opt_tte[type_mask]
            opt_is_calls = opt_is_calls[type_mask]
            if opt_interpolate is not None:
                opt_interpolate = {k: v[type_mask] for k, v in opt_interpolate.items()}

        if not self.no_warnings and np.max(opt_tte) > self.geometry.xexpiries[-1]:
            warnings.warn(
                f"Found option with expiry {np.max(opt_tte):.4g} beyond the last model expiry {self.geometry.xexpiries[-1]:.4g}. "
                "While functionaly feasible, this is likely to cause fitting issues. Expand expiry range."
            )

        opt_interpolate = (
            opt_interpolate
            if not opt_interpolate is None
            else self.geometry.time_interpolation_weights(opt_tte, None)
        )
        M = len(opt_strikes)

        # compute sqrt var per option
        if fwd_vols is not None:
            opt_sqrtVars = _interpolate_sqrtVars(
                bb=_npbb,
                xexpiries=self.geometry.xexpiries,
                vol_interp=self.geometry.vol_interp,
                fwdvol_min=self.geometry.fwdvol_min,
                fwd_vols=fwd_vols,
                opt_interp=opt_interpolate,
                opt_mask=opt_mask,
                debug=self.debug,
            )
            verify_inp(
                opt_sqrtVars is None
                or (
                    opt_sqrtVars.shape == (M,)
                    and np.all(np.isfinite(opt_sqrtVars))
                    and np.all(opt_sqrtVars >= 0.0)
                ),
                f"Internal error: opt_sqrtVars must have shape {(M,)} and be non-negative and finite, found shape {opt_sqrtVars.shape} and min {np.min(opt_sqrtVars)} and max {np.max(opt_sqrtVars)}",
            )
        else:
            opt_sqrtVars = None

        half_spread = 0.5 * (opt_ask - opt_bid)
        opt_mid = 0.5 * (opt_ask + opt_bid)
        verify_inp(
            np.min(opt_bid) > -self.eps.prices,
            f"Option bid must be positive, found min {np.min(opt_bid)} [tolerance 'prices' {self.eps.prices}]",
        )
        verify_inp(
            np.min(half_spread) > -self.eps.prices,
            f"Option half spread must be positive, found min {np.min(half_spread)} [tolerance 'prices' {self.eps.prices}]",
        )

        opt_call_mid = (
            opt_mid
            if opt_is_calls is None
            else np.where(opt_is_calls, opt_mid, opt_mid + 1.0 - opt_strikes)
        )
        opt_call_bid = (
            opt_bid
            if opt_is_calls is None
            else np.where(opt_is_calls, opt_bid, opt_bid + 1.0 - opt_strikes)
        )
        opt_call_ask = (
            opt_ask
            if opt_is_calls is None
            else np.where(opt_is_calls, opt_ask, opt_ask + 1.0 - opt_strikes)
        )
        del opt_bid, opt_ask, opt_mid

        # Optimization over densities
        xexpiries = np.asarray(self.geometry.xexpiries, dtype=np.float64)
        dts = np.diff(xexpiries)
        xstrikes = (
            [np.asarray(self.geometry.xstrikes[t, :], dtype=np.float64) for t in range(T + 1)]
            if self.grid_mode
            else [np.asarray(x_t, dtype=np.float64) for x_t in self.geometry.xstrikes]
        )
        mxstrikes = [x_t[1:-1] for x_t in xstrikes]
        density0 = np.asarray(self.geometry.density0, dtype=np.float64)
        Ns = [len(x_t) for x_t in mxstrikes[1:]]

        densities = [cp.Variable((Nt,), nonneg=True) for Nt in Ns]
        conditions = (
            [cp.sum(d_t) == 1.0 for d_t in densities]
            + [d_t <= 1.0 for d_t in densities]
            + [cp.sum(cp.multiply(s_t, d_t)) == 1.0 for s_t, d_t in zip(mxstrikes[1:], densities)]
        )

        _0 = np.float64(0.0)
        _1 = np.float64(1.0)

        assert self.geometry.dlv_max > self.geometry.dlv_min, (
            f"'dlv_max' must exceed 'dlv_min', found {self.geometry.dlv_max} vs {self.geometry.dlv_min}"
        )
        min_dlvsq = np.float64(self.geometry.dlv_min**2)
        max_dlvsq = np.float64(self.geometry.dlv_max**2)
        dlvbox = [x_t[1:-1] ** 2 / (x_t[2:] - x_t[:-2]) * dt for x_t, dt in zip(xstrikes[1:], dts)]
        for dlv in dlvbox:
            dlv[0] = 0.0
            dlv[-1] = 0.0

        dCs = []
        boxs = []

        for t in range(T):
            mx_t = mxstrikes[t + 1]
            mx_tm1 = mxstrikes[t]
            Nt = len(mx_t)
            Ntm1 = len(mx_tm1)
            mcalls_t = np.maximum(mx_t[None, :] - mx_t[:, None], _0)
            assert mcalls_t.shape == (Nt, Nt), f"Expected shape {(Nt, Nt)}, found {mcalls_t.shape}"
            density_t = densities[t]
            mcalls_t = mcalls_t @ density_t
            assert mcalls_t.shape == (Nt,), (
                f"Expected mcalls_t to have shape {(Nt,)}, found {mcalls_t.shape}"
            )

            density_tm1 = densities[t - 1] if t > 0 else density0
            mcalls_tm1 = np.maximum(mx_tm1[None, :] - mx_t[:, None], 0.0)
            assert mcalls_tm1.shape == (Nt, Ntm1), (
                f"Expected shape {(Nt, Ntm1)}, found {mcalls_tm1.shape}"
            )
            mcalls_tm1 = mcalls_tm1 @ density_tm1
            assert mcalls_tm1.shape == (Nt,), (
                f"Expected mcalls_tm1 to have shape {(Nt,)}, found {mcalls_tm1.shape}"
            )

            dC_t = mcalls_t - mcalls_tm1
            box_t = cp.multiply(dlvbox[t], density_t)
            lower_t = min_dlvsq * box_t
            upper_t = max_dlvsq * box_t
            assert lower_t.shape == (Nt,), (
                f"Expected lower_t to have shape {(Nt,)}, found {lower_t.shape}"
            )
            assert upper_t.shape == (Nt,), (
                f"Expected upper_t to have shape {(Nt,)}, found {upper_t.shape}"
            )
            conditions += [dC_t >= lower_t, dC_t <= upper_t]

            dCs.append(dC_t)
            boxs.append(box_t)
            del upper_t, lower_t, box_t, dC_t, mcalls_t, mcalls_tm1

        # computing model prices
        fitted = _iterative_model_price(
            opt_strikes=opt_strikes,
            opt_sqrtVars=opt_sqrtVars,
            opt_lix=opt_interpolate["lix"],
            opt_dweight=opt_interpolate["dweight"],
            strikes=[xstrikes[t + 1][1:-1] for t in range(T)],
            densities=[densities[t] for t in range(T)],
            no_warnings=self.no_warnings,
            debug=self.debug,
            return_array=False,
            check_finite=False,
        )

        # build objective
        half_spread = np.maximum(
            0.5 * (opt_call_ask - opt_call_bid),
            opt_call_ask.dtype.type(self.min_half_spread),
        )
        inv_hspread = 1.0 / half_spread
        objective = 0.0

        for fit_calls, umask in fitted:
            err_sqr = (
                cp.sum(cp.multiply(inv_hspread[umask], fit_calls - opt_call_mid[umask]) ** 2)
                if self.mid_fit_weight > 0.0
                else None
            )
            excess = (
                cp.sum(
                    cp.multiply(
                        inv_hspread[umask],
                        cp.pos(fit_calls - opt_call_ask[umask])
                        + cp.pos(opt_call_bid[umask] - fit_calls),
                    )
                )
                if self.mid_fit_weight < 1.0
                else None
            )

            if self.mid_fit_weight == 0.0:
                objective = objective + excess
            elif self.mid_fit_weight == 1.0:
                objective = objective + err_sqr
            else:
                objective = (
                    objective + self.mid_fit_weight * err_sqr + (1.0 - self.mid_fit_weight) * excess
                )

            if self.bid_ask_penalty_weight > 0.0:
                outside = cp.pos(fit_calls - opt_call_ask[umask]) + cp.pos(
                    opt_call_bid[umask] - fit_calls
                )
                objective = objective + self.bid_ask_penalty_weight * cp.sum_squares(outside)
        del umask, fit_calls, err_sqr, excess

        # ── regularizer ─────────────────────────────────────────────────────
        # log_dlv_space=False (legacy): second-difference smoothness on density.
        #   Fills the null space for anchors below K_lo_obs.  Tiny λ keeps the
        #   data term dominant on observed strikes (ratio ~1e-10 for SPY spreads).
        #
        # log_dlv_space=True: log-barrier  −λ·Σlog(dᵢ).
        #   DCP-compliant (−log is convex).  Penalises small densities
        #   proportionally in log-space: halving d from 0.002→0.001 costs the
        #   same as 0.2→0.1.  Since DLV² ∝ 1/dᵢ this naturally suppresses
        #   extreme DLV² values without distorting the data fit on observed
        #   strikes.  Equivalent to maximum-entropy regularisation.
        if self.regularizer_weight > 0.0:
            for density_t in densities:
                if self.log_dlv_space:
                    objective = objective - self.regularizer_weight * cp.sum(
                        cp.log(density_t + 1e-10)
                    )
                elif density_t.shape[0] >= 3:
                    second_diff = density_t[:-2] - 2 * density_t[1:-1] + density_t[2:]
                    objective = objective + self.regularizer_weight * cp.sum_squares(second_diff)
        # ────────────────────────────────────────────────────────────────────

        assert len(objective.shape) == 0, (
            f"Expected objective to be a scalar, found shape {objective.shape}"
        )

        # optimize
        with verbose.write_t(
            lambda: (
                f"Solving for a consistent margingale density for {len(opt_strikes)} options. Mean half spread is {np.mean(half_spread):.4e}... "
            ),
            end="",
        ) as tme:
            prob = cp.Problem(cp.Minimize(objective), conditions)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Solution may be inaccurate.*", category=UserWarning
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Solution may be inaccurate.", category=UserWarning
                )
                warnings.filterwarnings(
                    "error", message=r"overflow encountered in power", category=RuntimeWarning
                )
                try:
                    prob.solve(solver=self.cvxpy_solver, **self.cvxpy_solver_options)
                except cp.SolverError as e:
                    raise e
                except RuntimeWarning as e:
                    raise OverflowError(
                        f"Solver {self.cvxpy_solver} failed to find a solution: {str(e)}"
                    ) from e

            if prob.status == "infeasible":
                raise RuntimeError("Optimization problem is 'infeasible'.")

            densities = [d.value for d in densities]
            assert all(np.all(np.isfinite(d)) for d in densities), (
                "Infintite fitted densities found"
            )

            fit_calls = np.zeros_like(opt_strikes)
            for fc, um in fitted:
                fit_calls[um] = fc.value
            assert np.all(np.isfinite(fit_calls)), "Infinte fitted calls found"

            objective = objective.value
            excess = inv_hspread * (
                np.clip(fit_calls - opt_call_ask, min=0.0)
                + np.clip(opt_call_bid - fit_calls, min=0.0)
            )

            if np.any(excess > 0.0):
                bad = excess[excess > 0.0]
                max_ = np.max(bad) if len(bad) > 0 else 0.0
                q95, q99, q999 = np.quantile(bad, [0.95, 0.99, 0.999])
                bad_str = f"relative excess error conditional mean {np.mean(bad):.4e}, std {np.std(bad):.4e}, 95th percentile {q95:.4e}, 99th percentile {q99:.4e}, 99.9th percentile {q999:.4e} and max {max_:.4e}"
                del q95, q99, q999, bad
            else:
                bad_str = "zero relative excess error"
                max_ = 0.0

            if not self.max_error is None:
                if max_ > self.max_error:
                    slv_str = (
                        f"successfully returned but"
                        if prob.status == "optimal"
                        else f"return with status '{prob.status}' and"
                    )
                    err_str = (
                        f"Solver {self.cvxpy_solver} {slv_str} failed to find a solution within the specified 'max_error' tolerance. "
                        f"Maximum relative error {max_:.4e} exceeds `max_error` {self.max_error:.4e}. "
                        f"Objective value: {objective:.4e}, {bad_str} "
                        "You can turn this check off by setting 'max_error' to None."
                    )
                    raise RuntimeError(err_str)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                verbose.write(
                    lambda: (
                        f"done; solver {self.cvxpy_solver} returned status '{prob.status}' with {bad_str}. This took {tme}."
                    ),
                    head=False,
                )
            else:
                verbose.write(lambda: f"done with {bad_str}. This took {tme}.", head=False)

            del prob, objective, excess, inv_hspread, half_spread, bad_str, max_

        # computing DLVs
        min_d = [np.min(d) for d in densities]
        max_dff_sum1 = [np.max(np.abs(np.sum(d) - 1.0)) for d in densities]
        max_dff_sum2 = [
            np.max(np.abs(np.sum(densities[t] * mxstrikes[t + 1]) - 1.0)) for t in range(T)
        ]

        assert np.all(np.isfinite(fit_calls)), (
            f"Expected fitted call prices to be finite, found min {np.min(fit_calls)} and max {np.max(fit_calls)}"
        )
        assert min(min_d) > -self.eps.densities, (
            f"Expected densities to be non-negative, found min {min(min_d):.4g} [tolerance 'densities' {self.eps.densities}]"
        )
        assert max(max_dff_sum1) <= self.eps.densities, (
            f"Expected sum of densities to be 1, found {max(max_dff_sum1):.4g} [tolerance 'densities' {self.eps.densities}]"
        )
        assert max(max_dff_sum2) <= self.eps.densities, (
            f"Expected sum of densities*strikes to be 1, found {max(max_dff_sum2):.4g} [tolerance 'densities' {self.eps.densities}]"
        )

        with verbose.write_t(lambda: "Computing DLV squared's... ", end="") as tme:
            dCs = [dC_t.value[1:-1] for dC_t in dCs]
            boxs = [box_t.value[1:-1] for box_t in boxs]
            dlvsqs = []

            for t in range(T):
                dC_t = dCs[t]
                density_box = boxs[t]

                assert np.all(np.isfinite(dC_t)), (
                    f"Expected dC_t to be finite, found min {np.min(dC_t)} and max {np.max(dC_t)}"
                )
                assert np.all(np.isfinite(density_box)), (
                    f"Expected density_box to be finite, found min {np.min(density_box)} and max {np.max(density_box)}"
                )
                assert np.min(dC_t) > -self.eps.prices, (
                    f"Expected dC_t to be non-negative, found min {np.min(dC_t)} [tolerance 'prices' {self.eps.prices}]"
                )
                assert np.min(density_box) > -self.eps.densities, (
                    f"Expected density_box to be non-negative, found min {np.min(density_box)} [tolerance 'densities' {self.eps.densities}]"
                )

                dC_zero = dC_t <= self.eps.dlvsq
                too_small = dC_t <= min_dlvsq * density_box
                too_big = dC_t >= max_dlvsq * density_box

                divisor = np.where(too_small | dC_zero | too_big, _1, density_box)
                dlvsqs_t = np.select(
                    [dC_zero, too_small, too_big],
                    [min_dlvsq, min_dlvsq, max_dlvsq],
                    default=dC_t / divisor,
                )
                dlvsqs_t = np.where(np.isfinite(dlvsqs_t), dlvsqs_t, min_dlvsq)
                dlvsqs_t = np.clip(dlvsqs_t, min=min_dlvsq, max=max_dlvsq) - min_dlvsq
                dlvsqs.append(dlvsqs_t)

            if self.grid_mode:
                dlvsqs = np.stack(dlvsqs, axis=0)
                assert dlvsqs.shape == (T, self.N - 2), (
                    f"Expected dlvsqs to have shape {(T, self.N - 2)}, found {dlvsqs.shape}"
                )
            verbose.write(lambda: f"done; this tool {tme}.", head=False)

        if return_only_dlvsqs:
            return dlvsqs

        fit_prices = (
            np.where(opt_is_calls, fit_calls, fit_calls - 1.0 + opt_strikes)
            if opt_is_calls is not None
            else fit_calls
        )
        assert fit_prices.shape == (M,), (
            f"Expected final fit_prices to have shape {(M,)} but found {fit_prices.shape}"
        )

        if not opt_mask is None:
            fit_prices_ = np.zeros(opt_mask.shape, dtype=fit_prices.dtype)
            fit_prices_[opt_mask] = fit_prices
            fit_prices = fit_prices_
            del fit_prices_

        if self.grid_mode:
            densities = np.stack(densities, axis=0)
            assert densities.shape == (T, self.N), (
                f"Expected final densities to have shape {(T, self.N)}, got {densities.shape}"
            )
        return dlvsqs, fit_prices, densities
