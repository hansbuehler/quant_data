# Historical SANOS regularized 252-day solver review

## Decision

The production estimator retains 252 uniformly weighted one-session scenarios.
The fitting ridge is increased from $10^{-6}$ to $10^{-4}$ and is applied in
unit-RMS instrument-gain coordinates. A higher-frequency history is explicitly
deferred to a separate model revision.

## Problem identified

A representative daily fit had 252 observations and 133 tradable instruments.
The old exact-cost positive/negative representation exposed 266 nonnegative
optimizer coordinates before the optional cash translation, with strongly
collinear option returns. Several otherwise plausible daily fits missed the
hard $10^{-8}$ residual threshold by a small amount. The resulting failures
were numerical-certification failures, not evidence that the hedging objective
had no finite solution.

The strict weekly replay at origin 2023-01-17 produced 79 `optimal` conic solves
and one `optimal_inaccurate` full-window solve. The latter was the long
down-and-out-put pricing policy, with signed claim quantity
2.095591221231792, 232 scenarios, maximum absolute position
75.93502458030453, objective -0.1577833911975831, and no cap hit. It had been
rejected solely by an exact string comparison with `optimal`.

## Remediation

- Optimize and regularize in unit-RMS instrument-gain coordinates.
- Retain exact transaction costs and the original economic objectives.
- Use the final 252-observation fit as the production fit; the 80/20 fit is a
  selection and validation diagnostic only.
- Accept a stalled daily optimizer only when the independently recomputed KKT
  residual of the original signed-action problem passes the declared
  certificate tolerance.
- Retry an inaccurate primary conic solve cold, then solve the identical problem
  with independent SCS. Accept only an `optimal` independent result whose fitted
  values are finite, whose exact reconstructed objective agrees with the solver
  objective within tolerance, whose primal constraints pass, and which has no
  cap hit.
- Warn whenever the observation-to-action-parameter ratio is below two and
  label the output as a regularized estimate.
- Treat a remaining failure as terminal for that case and retain its numerical
  evidence for quantitative review. Do not silently loosen tolerances.

## Independent quantitative review

The first independent review agreed that the 252-by-133 daily geometry should
have been flagged earlier, but blocked the initial patch because objective
reconstruction alone did not certify optimality and physical-coordinate
snapping broke scale invariance. The remediation adds an independent SCS
confirmation for inaccurate primary solves, a primal-feasibility check,
standardized-coordinate canonicalization, and separates objective error from
the gradient diagnostic. A second review is required before production.

The second review found a deterministic false failure in the daily
mean-volatility certificate: its reconstruction added machine epsilon inside a
zero standard deviation although the SOCP objective did not. The final patch
uses the exact population standard deviation and the admissible zero
subgradient of the norm at zero. A direct symmetric zero-claim regression at
risk parameter 10 reproduces and prevents that failure. The final independent
review passes.

## Regression evidence

The focused daily and weekly Historical-SANOS suite contains explicit tests for
the 252-day default, the stronger ridge, coordinate-scale invariance,
statistically thin books, selection failure followed by a certified final fit,
rejection of an unconfirmed `optimal_inaccurate`, independent SCS recovery, and
standardized weekly scale invariance. All 55 focused tests pass. The complete
hedging test tree passes 418 tests.

The final-gate real-data replays also pass:

- Daily entropy, risk parameter 1, origin 2023-07-20: all 20 targets converge,
  with zero cap hits and zero nonconverged results. The fit has 252 observations
  and 140 action parameters, so its ratio 1.8 is explicitly flagged as thin.
- Frozen-weekly entropy, risk parameter 1, origin 2023-01-17: all 13 targets
  converge, with zero fitted or executed cap hits and zero nonconverged fits.
  The open-loop fit has 232 complete path blocks and 669 action parameters, so
  its ratio 0.347 is explicitly flagged as thin.

## Ridge sensitivity

The same 2023-07-20 daily entropy book was fitted at three predeclared ridge
levels. Selection uses numerical conditioning and action stability only, not the
realized hedge outcome.

| Ridge | Converged targets | Maximum KKT residual | Mean claim-plus-zero iterations | Maximum claim gross exposure | Mean pure-hedge gross exposure |
|---:|---:|---:|---:|---:|---:|
| $10^{-5}$ | 20/20 | $2.23\times10^{-7}$ | 13,687 | 396,102 | 407.2 |
| $10^{-4}$ | 20/20 | $1.63\times10^{-7}$ | 9,732 | 281,227 | 279.5 |
| $10^{-3}$ | 20/20 | $1.13\times10^{-7}$ | 6,400 | 108,239 | 125.4 |

Relative to $10^{-5}$, the mean pure-action $\ell_2$ change is 41.5% at
$10^{-4}$ and 83.0% at $10^{-3}$. We therefore retain $10^{-4}$ as the smallest
candidate that materially reduces iterations and the large cancelling virtual
policies without imposing the much larger action distortion of $10^{-3}$.
