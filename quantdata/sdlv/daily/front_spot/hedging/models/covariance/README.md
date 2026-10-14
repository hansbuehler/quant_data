# Independent robust covariance

- **Inputs:** historical public forward-adjusted spot returns and realized
  option/digital returns. The current hedge book is replayed over the rolling
  history at the same initial business DTE, NMS, and option side. Option and
  digital training returns are reconstructed from the released DLV state; no
  SANOS-generated scenarios enter the covariance fit. SANOS marks are used
  only as the common executable market proxy for current prices and realized
  accounting because the release contains no midpoint.
- **Measure:** historical `P`.
- **Price:** none independently; pure hedging uses the common SANOS market mark.
- **Estimator:** one rolling, automatically regularized Tyler covariance over
  spot and the full current relative-option book, with robust marginal scales.
  Each target is then fitted independently by a fixed-weight robust regression;
  its cross-covariance is constructed from the instrument covariance and its
  residual variance, so the joint block is positive definite and a target's
  hedge is invariant to unrelated targets in the benchmark catalog.
- **Expected returns:** sample means are kept separate from Tyler's marginal
  medians. Instrument means are projected onto the covariance eigenportfolios
  and soft-thresholded using Newey--West standard errors at the rolling
  universal threshold `sqrt(2 log p)`. The automatic HAC lag and threshold are
  recomputed in every rolling window rather than fixed from 2022.
- **Objectives:** mean--variance and mean--volatility use the fitted first two
  moments directly. Gaussian entropy is exactly the mean--variance policy at
  the same risk parameter. Power-exponential uses its distinct exact Gaussian
  OCE, profiling the scalar cash translation with stable truncated-normal
  formulas and solving the resulting convex spread-lasso problem; it is not
  relabelled from mean--variance.
- **Hedge:** claim-policy minus zero-claim-policy action with
  transaction-cost-weighted lasso. Mean-variance, mean-volatility, and
  Gaussian power-exponential are supported. The deterministic regression uses
  mean-variance risk aversion 10 and a loose numerical cap of one million. The
  spot penalty contains its one-basis-point entry and liquidation costs; the
  option penalty uses one current half-spread for next-date cash settlement and
  twice the current half-spread as a predictable round-trip proxy otherwise.
- **Mean-volatility boundedness:** the objective is positively homogeneous. A
  covariance-metric distance-to-cost-box calculation checks its recession
  condition before optimization. An unbounded fit is solved under the
  precommitted large numerical cap, retained with `bounded=false`, and
  reported through non-convergence and cap-hit diagnostics. Position-cap hits
  in an otherwise bounded numerical fit remain failed convergence diagnostics.
- **Training:** production configuration and the regression smoke use exactly
  252 valid prior one-session observations. The last training return may end at
  the decision-date close, but no endpoint after that close enters the fit.
- **Current-state isolation:** the fitter receives the predictable current cost
  vector and current relative-coordinate definitions, not the future-bearing
  evaluation return.
- **Current scope:** the full daily relative-option universe and all 20 daily
  digital targets are implemented, together with authenticated restartable
  rolling evaluation across the full eligible date cohort. The full production
  sweep is an external artifact and may still be in progress.
- **Artifacts:** common `hedging-smoke-v2` JSON with explicit claim quantity.

Run on CPU:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.covariance.smoke
```
