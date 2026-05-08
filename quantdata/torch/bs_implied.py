

from __future__ import annotations

from quantdata.torch import verify
import torch as torch

    def bs_implied_call( 
            self, 
            k: Array,
            prices: Array,
            sqrtT: Array, *,
            price_tol: Array | float = 1e-6,
            vol_min: float = 0.01,
            vol_max: float = 5.0,
            default_vol: float = 0.1,
            mask : Array|None = None,
            max_iters: int = 10,
            eps: float = 1e-8,
            min_vega: float = 1e-12,
            ret_only_vols: bool = True,
            debug : bool = False ) -> tuple[Array|Array]:
            """
            Solve for implied volatilities using a robust bisection and Newton-Raphson hybrid method.
            This function is designed to be run for a large number of potentially unrelated options, for example for a time series of surfaces of options.

            This routine:

            1) Assigns ``max_vol`` to every option whose price is at or above the option price implied by ``vol_max``, and ``vol_min`` to every option whose price is at or below
            the option price implied by ``vol_min``.
            
            2) Initializes the search at ``default_vol`` if it is an array, or otherwise using the `closed-form approximation (20) <https://repub.eur.nl/pub/1472/ERS%202004%20054%20FA.pdf>`__.
            If ``default_vol`` is a float, then it is only used where the approximation fails to yield a value.
            
            3) Run a hybrid bisection and Newton-Raphson method to find the implied volatilities for the remaining options until the maximum number of itersations, ``max_iters``, is reached
            or the method has converged in the sense that::

                    |price(vol) - market_price| < price_tol

            By default the function just returns the implied volatilities, or their last best guesses.
            If ``ret_only_vols`` is set to ``False``,
            then it returns a :class:`cdxcore.pretty.PrettyValueObject` containing the fitted prices, a boolean area indicating issues, and an error statistics object
            in the following fields:

            * ``vols`` contains the fitted volatilities.
            * ``fits`` contains the the prices at the fitted volatilities.
            * ``prices`` contains the input prices at those values.
            * ``failed`` contains a boolean array indicating which fits did not converge.

            * ``max_iters``: the input ``max_iters``.
            * ``iters``: iterations used.
            * ``max_err``: maximum price error.
            * ``l1_err``: average price error.
            * ``l2_err``: quadratic average price error.

            Parameters
            ----------
                k : self.ndarray
                    Strikes.

                prices : self.ndarray
                    Prices in the same shape as ``k``.

                is_call : self.ndarray | bool, default ``True``
                    Boolean or boolean array of shape compatible with ``k``.

                sqrtT : self.ndarray | float, default ``1``
                    Square-root of time as float or array compatible with ``k``.

                price_tol : self.ndarray | float, default ``1E-6``
                    Price tolerance (typically a fraction of spreads) as
                    float or array or array compatible with ``k``.
                    A standard value is ``0.1*spread``.

                vol_min : float, default ``0.01``
                    Minimum volatility.

                vol_max : float, default ``5``
                    Maximum volatility.

                default_vol : self.ndarray | float, default ``0.``
                    Default and initial volatility guess as float or array compatible with ``k``.
                    See description above. Also note that ``default_vol`` will be clipped to the range ``[vol_min, vol_max]``.
                    
                mask : self.ndarray | None, default ``None``
                    A numpy array boolean mask to indicate which options to process.
                    The implied vol of options excluded by ``mask`` will be set to ``default_vol``.

                max_iters : int, default ``100``
                    Maximum iterations. Usually the routine uses very few iterations.

                eps : float, default ``1E-10``
                    Only used to decide whether stirke or sqrtVar are zero.

                min_vega : float, default ``1E-12``
                    Minimum vega for taking an updates step.

                ret_only_vols : bool, default ``True``
                    Return only vols.

                on_exceed_bounds : ``error`` | ``warn`` | ``quiet`` | None, default ``warn``
                    What to do if the input data violate basic option price bounds such as intrinsic from below or
                    unit/strike from above, respectively. 
                    ``None`` is equivalent to ``quiet``.

                verbose : :class:`cdxcore.verbose.Context`, default :attr:`cdxcore.verbose.Context.quiet`
                    For printing progress information.

            Returns
            -------
            results : self.ndarray | PrettyValueObject
                See above.
            """
            shape  = prices.shape
            dtype  = prices.dtype
            device = prices.device

            f0   = self.zeros((), dtype=dtype, device=device)
            f1   = self.ones((), dtype=dtype, device=device)
            fits = self.zeros_like(prices)  # output prices
            vols = self.full_like(prices, float(default_vol))  # output vol
            intr = self.maximum( f0, f1 - k )
            
            logK = self.log( self.where( k>eps, f1, k ) )
            price_min = self.bs_price(k=k, sqrtT=sqrtT, vol=vol_min, logK=logK, eps=eps)
            price_max = self.bs_price(k=k, sqrtT=sqrtT, vol=vol_max, logK=logK, eps=eps)
            assert price_min.shape == prices.shape
            assert price_max.shape == prices.shape

            def IX(v, mask):
                return v[mask] if isinstance(v, self.ndarray) else v

            # boundary cases
            # --------------

            # identify intrinsic
            done = (k <= eps) | (sqrtT <= eps) | (self.abs(intr - prices) <= price_tol)
                
            fits[done] = intr[done]
            vols[done] = IX(default_vol, done)
                                    
            work = ~done
            # identify options priced at or beyond vol_max
            done = (price_max - prices <= price_tol) & work
            fits[done] = price_max[done]
            vols[done] = vol_max
            work &= ~done

            # identify options priced at or beyond vol_min
            ## RS FIX: vol_min wasn't masked with work
            # done           = prices - price_min <= price_tol
            done = (prices - price_min <= price_tol) & work
            fits[done] = price_min[done]
            vols[done] = vol_min
            work &= ~done

            vols_min = self.full_like(vols, vol_min)
            vols_max = self.full_like(vols, vol_max)

            # train
            # -----

            iters = 0
            while True:
                # check convergence
                # -----------------
                fits[work] = self.price( k=k[work], sqrtT=IX(sqrtT, work), vol=vols[work], is_call=IX(is_call, work), logK=IX(logK, work), eps=eps )
                done = (self.abs(fits - prices) < price_tol) & work
                work &= ~done
                if not self.any(work):
                    verbose.report( 2, lambda: f"\rStep {iters}/{max_iters}: implied vol for the remaining {sum(done)} options computed.", end="")
                    break

                if iters >= max_iters:
                    verbose.report(2, lambda: f"\rReached maximum iterations {max_iters} with {sum(work)} options left to do.", end="")
                    break

                # iteration
                # ---------

                iters += 1
                res_max_err = self.max(self.abs(fits[work] - prices[work]) / IX(price_tol, work))
                if tme.interval_test(0.5):
                    verbose.report(2, lambda: f"\rStep {iters}/{max_iters}: implied vol for {sum(done)} options computed; {sum(work)} options with a maximum error of {res_max_err:.4g} left to do.",end="")
                del done, res_max_err

                # adjust min/max
                too_low = (fits < prices) & work
                vols_min[too_low] = vols[too_low]
                too_high = (fits > prices) & work
                vols_max[too_high] = vols[too_high]

                # first order:
                # prices(x) = prices(x0) + prices'(x0) * (x-x0)
                # =>
                # x ~ ( prices(x) - prices(x0) ) / prices'(x0) + x0

                test_vega = self.vega( k=k[work], sqrtT=IX(sqrtT, work), vol=vols[work], logK=IX(logK, work), eps=eps )
                newton_step = (prices[work] - fits[work]) / self.maximum( test_vega, min_vega )
                new_vols = vols[work] + newton_step

                # [27-01-2026 RS] FIX: 
                # --- Newton-Raphson Oscillation Fallback using Bisection --- #
                # Use Newtown-Bisection hybrid step to prevent overshoot.
                # Use bisection when Newton step would jump outside the bracketed interval
                # This prevents oscillation when vega is very small (e.g., deep ITM puts)
                use_bisection = (new_vols < vols_min[work]) | (new_vols > vols_max[work])
                if self.any(use_bisection):
                    bisection_vols = 0.5 * (vols_min[work] + vols_max[work])
                    new_vols = self.where(use_bisection, bisection_vols, new_vols)
                # --------------------------------------------------------- #
                err1 = (vols[work] == vols_min[work]) & (new_vols < vols[work])
                err2 = (vols[work] == vols_max[work]) & (new_vols > vols[work])

                if sum(err1 | err2) > 0:
                    err1 = (
                        f"{sum(err1)} cases where a vega step at the current lower vol would take vol even lower"
                        if sum(err1) > 0
                        else None
                    )
                    err2 = (
                        f"{sum(err2)} cases where a vega step at the current upper vol would take vol even higher"
                        if sum(err2) > 0
                        else None
                    )
                    if err1 is not None and err2 is not None:
                        raise RuntimeError(f"Convergence error: had {err1} and {err2}.")
                    raise RuntimeError(
                        f"Convergence error: had {err1 if err1 is not None else err2}."
                    )

                vols[work] = self.clip(new_vols, vols_min[work], vols_max[work])
                del new_vols, newton_step, test_vega, too_low, too_high, err1, err2

            if ret_only_vols and not verbose.shall_report():
                return vols.reshape(shape)

            iters      = int(iters)
            max_iters  = int(max_iters)
            num_failed = self.sum(work)
            max_err    = float(self.max(self.abs(fits - prices)))
            l1_err     = float(self.mean(self.abs(fits - prices)))
            l2_err     = float(self.std(fits - prices))

            verbose.write(
                lambda: f"\rImplied vol calculation for {total} options finished using {iters}/{max_iters} iterations. "
                + ( f"Failed to converge for {num_failed} options. " if num_failed > 0 else "Converged for all options. " )
                + f"Max error {max_err:.3f}, L1 error {l1_err:.3f}, L2 error {l2_err:.3f}. "
                + f"This took {tme}. "
            )

            vols = vols.reshape(shape)
            if ret_only_vols:
                return vols

            fits   = fits.reshape(shape)
            prices = prices.reshape(shape)
            failed = work.reshape(shape)

            return PrettyValueObject(
                    vols=vols,
                    fits=fits,
                    prices=prices,
                    failed=failed,
                    num_failed=num_failed,
                    max_err=max_err,
                    l1_err=l1_err,
                    l2_err=l2_err,
                )
