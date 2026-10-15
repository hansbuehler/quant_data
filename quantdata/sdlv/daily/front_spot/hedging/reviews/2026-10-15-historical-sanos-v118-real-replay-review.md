# Historical-SANOS v1.18 real-replay numerical review

Date: 2026-10-15

Reviewed base commit: `b466984`

Reviewed remediation: `HistoricalSanosConfig` v1.18 in the shared worktree

Real-replay evidence:
`/home/hans/quantdata_data/training/hedging/generated-readiness-b466984-compact-replay/pipeline.log`

## Scope

The 4,096-scenario by 669-parameter VolGAN weekly replay rejected compact
mean-variance candidates despite close primal objective reconstruction.  The
OSQP candidate reported `user_limit`, objective error `2.99769e-09`, primal
violation `6.81828e-05`, and stationarity `7.1788e-07`.  The HiGHS candidate
reported `optimal`, objective error `7.27596e-12`, primal violation
`9.17252e-07`, dual violation `0.00718581`, stationarity `1.07688e-05`, and a
reported gap of `-13788.6`.  Both candidates were correctly rejected by the
independent certificate.

## Defects and remediation

Two numerical defects were identified without changing the optimization
problem.

1. The active-set KKT polish treated every numerically zero turnover row as an
   equality, including rows whose transaction-cost weight was exactly zero.
   Those zero-weight cash-settlement rows are unrestricted by the lasso term;
   imposing zero turnover created false equalities and prevented recovery of a
   valid KKT point.  Version 1.18 activates a zero-turnover equality only when
   its turnover weight is strictly positive.
2. The position bounds were supplied to the solvers in physical coordinates as
   `diag(action_scale) x <= solver_position_cap`.  With action scales reaching
   `1e6`, these rows were poorly equilibrated.  Version 1.18 expresses the
   identical feasible set as `x <= solver_position_cap / action_scale` and
   updates bound-dual recovery, stationarity, complementarity, and the dual
   objective in the same scaled coordinates.

The large negative HiGHS number was not a valid weak-duality gap.  Its proposed
dual violated feasibility by `0.00718581`, principally the turnover-epigraph
condition `alpha + beta = turnover_weight`.  Therefore its computed dual value
was not a lower bound on the primal problem.  The certificate rejected it; no
false acceptance occurred.

## Invariants

The financial and statistical problem is unchanged: scenario rows, claims,
mean-variance risk aversion, transaction costs, turnover and terminal
liquidation, quadratic ridge, physical position caps, selection/final samples,
and certificate tolerances are identical.  The remediation changes only exact
constraint coordinates and removes an equality that the original objective did
not contain.  Physical-unit primal feasibility remains part of the independent
certificate.

## Verification

- `python -m pytest -q tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos_weekly.py -k mean_variance`:
  12 passed.
- `python -m pytest -q tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos_weekly.py`:
  48 passed.
- `python -m pytest -q tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos.py tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos_weekly.py tests/quantdata/sdlv/daily/hedging/test_daily_runner.py`:
  84 passed.
- Independent parent validation of the complete
  `tests/quantdata/sdlv/daily/hedging` tree: 444 passed.
- The new regression covers action scales of `1e6`, zero-cost turnover rows,
  the exact scaled-cap identity, and full primal-dual certification.
- `python -m compileall -q` on the changed source and test modules passed.
- `ruff check --select I` on the changed source and test modules passed.
- `git diff --check` passed.

## Verdict

**PASS.**  Version 1.18 is a mathematically unchanged, strictly certified
numerical remediation of the failures exposed by the real VolGAN replay.  A
fresh authenticated real replay remains required as production-run evidence;
the earlier failed replay is diagnostic evidence, not evidence that the v1.18
run itself has completed.

Independent reviewer: QuantAgent (`historical_failure_quant_review`)
