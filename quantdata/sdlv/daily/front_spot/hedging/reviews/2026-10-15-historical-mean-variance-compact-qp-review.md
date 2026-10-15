# Historical weekly mean-variance compact-QP quant review

Date: 2026-10-15

Reviewed source commit: `fe8b12a`

Dataset manifest SHA-256:
`58741552e0f6eea82378a5f26a61e1df04b9d14f3ab166883089812cf903007a`

## Scope

The historical and generated weekly mean-variance policies now solve the exact
open-loop problem as a target-independent compact quadratic program.  Scenario
rows enter through the precomputed mean and covariance geometry; transaction
costs retain their exact turnover epigraph, position bounds retain their original
economic units, and claim-dependent terms are supplied through DPP parameters.

The primary OSQP candidate is not accepted from its status.  The implementation
independently verifies objective reconstruction, primal and dual feasibility,
stationarity, complementarity, Hessian range compatibility, and the scaled
primal-dual gap.  A failed raw certificate may be repaired by an exact active-set
KKT solve and is then subjected to the complete certificate again.  HiGHS is an
independent fallback on the same compact QP, runs single-threaded with a finite
time limit, and its output must pass the same certificate.  Position-cap contact
remains a reported numerical-gate failure rather than an accepted hedge.

## Verification

- The independent expanded-scenario CLARABEL reference agrees with the compact
  objective and fitted value on deterministic regression geometry.
- Genuine OSQP `USER_LIMIT`, rank-deficient zero-ridge, active position-boundary,
  fallback, and failed-certificate cases have dedicated regression tests.
- The focused reviewer suite passed 47 tests.
- The complete historical daily and weekly solver suite passed 78 tests.
- The generated-weekly adapter suite passed all 6 tests, including frozen and
  restarted schedules and target-order reuse.
- A synthetic production-size problem with 4,096 scenarios and 669 parameters
  certified in 0.79 seconds for the first target and 0.25 seconds for the second
  target using the cached DPP geometry.

## Verdict

**Pass.**  No mathematical false-acceptance path was found.  A fresh authenticated
4,096-scenario VolGAN replay is still required as performance and production
readiness evidence; it is not a condition on the source-code correctness verdict.

Independent reviewer: QuantAgent (`historical_failure_quant_review`)
