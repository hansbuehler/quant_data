# Front-spot implementation notes

These notes record numerical issues encountered while implementing the dynamics
in `quantdata/sdlv/tex/main2.tex`.  They are working notes for the append-only
`Implementation Commets` section requested for the completed paper.

## Normative dynamics

- Use the exact Gaussian information bridge for the front density and spot.
- Evolve positive DLV squared in log coordinates using the risk-neutral drift
  condition and the calibrated fixed-expiry/fixed-strike vol-of-vol surface.
- Reconstruct later densities with the SANOS tridiagonal systems after every
  DLV update.
- Keep the independent 5% Black martingale identical under P and Q.
- Under P use `dW_Q = dW_P + lambda dt`, with the achievable drift obtained by
  projecting the historical drift onto `m_Q + xi lambda`.

## Issues found

1. Uniform four-substep log-Euler simulation is positive and stable, but the
   continuous-time drift condition does not guarantee exact finite-step
   martingality after nonlinear SANOS reconstruction.  A 100,000-path CPU run
   had zero nonfinite values and mass/forward errors below `2e-13`, but seven
   statistically identifiable daily latent-call coordinates exceeded
   `|z| = 5`; the other martingale families had no such exceedance.
2. Empirical standard errors are unreliable for very rare density nodes and
   call exercises.  The audit now keeps every coordinate in the absolute-error
   report, applies z-tests only with at least 20 expected effective
   observations, uses predictable Bernoulli variance for front-density nodes,
   and uses analytic Black-mixture moments for terminal calls.
3. Front-option daily returns also have a known predictable conditional
   variance.  The audit now accumulates this variance rather than estimating it
   from rare realized payoffs.
4. A direct additive Euler update of future densities preserved zero drift but
   left the positive convex-order cone near expiry, producing negative recovered
   DLV squared and subsequent numerical failure.  It was rejected.
5. A simple antithetic density recentering was also rejected: the correction
   could leave the positive convex-order cone and is not the nonlinear branch
   drift solve specified in the paper.
6. An independent 100,000-path seed exposed one extreme path that first became
   nonfinite on day 13.  At an unquoted tail node with density of order
   `1e-24`, relative density volatility reached `8.5e5`; freezing the
   continuous relative DLV drift over a quarter-day produced a log increment
   near `2490`.  This is a discretisation failure, not a failure of the SANOS
   solve or batching.
7. A balanced positive drift split was tested for the same DLV SDE.  It removed
   every nonfinite value in the failure seed, but applying it to all ordinary
   drift increments created statistically significant one-day density and call
   biases in the 100,000-path audit.  It was therefore rejected as the default
   update.
8. The retained update is the paper's frozen log-Euler scheme for every
   relative-drift increment `x = mu dt` with `|x| <= 0.5`.  Only an exceptional
   larger increment is tamed to
   `sign(x) * (0.5 + log(1 + |x| - 0.5))`.  This tail-only safeguard leaves the
   ordinary scheme unchanged, is first-order consistent as the time step tends
   to zero, and keeps the stochastic factor
   `exp(-|xi|^2 dt / 2 + xi dW)` unchanged.  The exact previously failing path
   is finite with mass and forward preserved to machine precision; a
   10,000-path screen had no nonfinite values and no martingale-test statistic
   above five standard errors.
9. Upper-tail Gaussian interval probabilities in the information bridge are
   evaluated by survival-probability differences rather than subtracting two
   CDF values close to one.  At a daily endpoint the expired front is sampled
   directly from the day-start density using the aggregated first-factor
   increment.  This is algebraically the same exact bridge law and removes any
   dependence of settlement on recursive floating-point subtraction.
10. A remaining apparent nine-standard-error front-call failure was reproduced
   in isolation.  The exercise count was 96 against 96.44 expected, but one
   exceptionally large payoff made the predictable-variance Gaussian
   approximation fail: the predictable z-score was 9.18 while the realized
   studentized score was 0.98.  An independent resample from the identical
   conditional densities gave scores of -0.20 and -1.39 respectively.  The
   audit therefore retains the known predictable variance but uses the larger
   of predictable and empirical standard errors for conservative inference.
11. The accepted 100,000-path failure-seed audit uses eight uniform intraday
   substeps.  It has zero nonfinite values and mass and forward errors below
   `2.7e-13`.  Density-to-expiry, option-to-expiry and daily-spot tests have no
   statistic above five standard errors.  Two of 63,810 identifiable daily
   latent-call coordinates remain above five, with the worst mean
   `-5.27e-8`.  This is a genuine, economically tiny finite-step remainder:
   the continuous-time drift condition does not imply exact martingality for
   the nonlinear finite-step SANOS reconstruction.  It is reported rather
   than hidden; exact daily martingality requires the nonlinear branch drift
   correction in `eq:finite-step-drift-correction`.

## Step-grid comparison

The model dynamics remained unchanged while comparing the uniform grid with an
expiry-refined grid that is uniform in squared remaining time.  Matched
4,096-path runs did not show an improvement from expiry refinement, so the
uniform grid remains the production default.  If the final 100,000-path
audit still shows daily-call bias, the next implementation must use the
nonlinear finite-step branch drift correction in
`eq:finite-step-drift-correction`; no ad-hoc density dynamics will be used.
