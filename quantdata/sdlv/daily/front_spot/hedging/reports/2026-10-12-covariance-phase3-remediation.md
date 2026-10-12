# Covariance Phase-3 remediation evidence

The independent review identified three blockers: a median used as an expected
return, target-catalog dependence in a joint Tyler fit, and no passing actual
mean-volatility case. The corrected implementation:

- estimates instrument expected returns by sample means and screens observed
  covariance eigenportfolio means using Newey--West standard errors;
- fits Tyler scatter to instruments only and gives each target an independent
  fixed-instrument-weight robust regression block with positive residual
  variance; and
- checks the exact mean-volatility recession condition before optimization and
  emits no tradable action when the problem is unbounded.

On the 252-transition window ending at the 2023-01-17 close, the production
mean-variance smoke over 163 instruments and all 20 digital targets converged
with zero cap hits. Its selected 1D ATM call hedge had gross exposure 705.30,
53 nonzero option actions, and a zero zero-claim policy after the HAC screen.

The full 20-target mean-volatility catalog also passed for every required
value of `kappa`:

| `kappa` | Converged targets | Cap hits | Maximum gross exposure | Wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 20 / 20 | 0 | 253.763 | 21.15 |
| 1.0 | 20 / 20 | 0 | 629.294 | 36.74 |
| 2.0 | 20 / 20 | 0 | 705.850 | 74.59 |
| 3.0 | 20 / 20 | 0 | 740.040 | 63.36 |

All four cases passed the analytic boundedness check. The screened mean lay
inside the spread-cost box, so the recession ratio was zero. The focused CPU
suite passed 61 tests, including skewed mean-versus-median, target-catalog
invariance, positive-definite target blocks, and bounded/unbounded
mean-volatility regressions.
