# Historical-SANOS v1.16 Coordinate Remediation

## Scope

The commit-pinned v1.15 production sweep exposed two daily mean-variance
solutions that OSQP and CLARABEL both labelled optimal but whose independently
recomputed KKT residuals exceeded the precommitted tolerance of $10^{-6}$.
It also exposed a low-risk mean-volatility zero-claim recession direction that
the daily failure artifact initially described only as nonconvergence.

## Unchanged-objective refinement

Version 1.16 retains the exact mean-variance objective, costs, ridge, action
scaling, and position bounds. If both conic/QP solver points fail the independent
certificate, cyclic coordinate descent refines the same scaled box-constrained
lasso QP. For scaled action $z$, it minimizes

$$
\frac12 z^\top H z + q^\top z + \ell^\top |z|,
\qquad -\bar z\leq z\leq\bar z,
$$

where $H$, $q$, $\ell$, and $\bar z$ are algebraically identical to the
original empirical mean-variance problem. Each coordinate uses the exact
soft-thresholded quadratic minimizer followed by the unchanged box projection.
The result is accepted only by the original objective-reconstruction,
feasibility, and KKT certificate.

## Real-data evidence

The following one-origin public-data replays use the fixed 252-transition
window, fixed 2022 target notionals, 1,000-unit executable position cap, and all
20 daily targets:

- Mean-variance $\lambda=0.1$ at 2023-01-24: 20/20 converged, zero cap hits.
  Summary SHA-256:
  `b959345446ad58d9bce5026afa7eb14add78b38d965a60234ff268b423568fee`.
- Mean-variance $\lambda=10$ at 2023-01-25: 20/20 converged, zero cap hits.
  Summary SHA-256:
  `bbc885e5a7ef52d38c1705918bcb0cfef4592d7de3d55927d147c74ca39463f6`.
- Mean-volatility $\lambda=0.1$ at 2023-01-17 remains inadmissible: the common
  zero-claim policy has one certified recession direction. The daily runner now
  classifies this explicitly rather than presenting it as an undifferentiated
  numerical failure. Failure-evidence SHA-256:
  `dea72b7246e0bcaa443b12e860e8a0492a29a0e4c6626733436c7d3aecbf14c9`.

The full hedging test tree passes 433 tests. Syntax compilation, Ruff import
checks, and `git diff --check` pass.

## Independent quantitative review

The reviewer confirmed that the scaled Hessian, linear term, lasso penalty,
box bounds, coordinate updates, and box KKT signs reproduce the unchanged
convex objective. The first review found that the common zero-policy recession
was counted once per target. The corrected diagnostic counts every
claim-specific recession fit plus the single shared zero-policy fit at most
once. The second review returned PASS with no remaining blocker; its focused
suite passed 36 tests.

## Source identities

- `config.py`: `1598ccbd00ffc9fa7d812b8e1eb44512030aded7acbfd44124a41a729baec864`
- `hedger.py`: `ab82877c73da01e247f023cc012f86c63b55f2a397c5cdf45ee5066f5689d8d5`
- `daily.py`: `b57703abfaa6d5331eb87231d350202e79054ef01fe34026110c7f1626730e17`
- `numerical_gate_failure.py`: `7b602ce61134831f074a8c9294608d3d9e99ab38427dc09db858807823ea93a8`
- `test_historical_sanos.py`: `9515d21992632e6b8ceb21025c6f5c3cf9df82c9c94008f1c7e48300e62d880c`
- `test_daily_runner.py`: `21d4d29478b0a3df1d7b020598d447b6c6575c96fdf87fc90945a41c3e425ccf`
