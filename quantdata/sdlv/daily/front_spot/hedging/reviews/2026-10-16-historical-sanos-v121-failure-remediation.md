# Historical-SANOS v1.20 failure classification and v1.21 remediation review

Date: 2026-10-16
Independent reviewer: QuantAgent (`historical_v112_failure_review`)

## Verdict

**PASS for the four mean-volatility failure classifications; FAIL for the
v1.20 daily mean-variance certificate; PASS for the proposed v1.21 repair.**

- Daily mean-volatility at risk parameter 0.1 and frozen-weekly
  mean-volatility at risk parameters 0.1, 1, and 2 contain genuine empirical
  zero-claim recession directions. They must remain strict production
  failures.
- Daily mean-variance at risk parameter 0.1, origin 2023-01-31, target
  `digital-call-3d-nms+0.50` is a false nonconvergence caused by comparing a
  coordinate-refined action with the stale pre-refinement CVXPY objective.
- The proposed v1.21 patch evaluates the CVXPY objective at the same refined
  action that the independent NumPy reconstruction and KKT certificate assess.
  It changes no objective, cost, cap, scenario population, or tolerance and is
  the narrow correct repair.

## Reviewed identities

- Immutable v1.20 worktree commit:
  `7938ec8d908e93b7de1ac66eb698ccda2af55d79`.
- v1.20 source SHA-256 values:
  - `config.py`: `c353a39dfd1b5abcd128ae293a979e62636049f3c919353d30afe2fb0b67649e`
  - `hedger.py`: `ab82877c73da01e247f023cc012f86c63b55f2a397c5cdf45ee5066f5689d8d5`
  - `weekly_hedger.py`: `8463dab6d16c748937e89fbbc456b077f4825a2c434ec451471c56c4a9d463c4`
  - daily runner: `b57703abfaa6d5331eb87231d350202e79054ef01fe34026110c7f1626730e17`
  - weekly runner: `838b867f136379b9243759585dde0ba9ecb57d447387bf7017c92e8a56156eef`
- Every reviewed production case uses 252 rolling observations, executable cap
  1,000, internal solver cap 1,000,000, quadratic ridge 0.0001, certificate
  tolerance `1e-6`, gradient tolerance `1e-8`, and objective-agreement
  tolerance `1e-7`.

## Classification summary

| Case | First rejected origin | Classification | Decisive evidence |
|---|---:|---|---|
| Daily mean-volatility 0.1 | 2023-01-17 | Genuine recession/unbounded economic objective | Shared zero policy has exact unregularized objective `-182.981`; 20 targets inherit it; zero caps |
| Frozen-weekly mean-volatility 0.1 | 2023-01-17 | Genuine recession/unbounded economic objective plus genuine internal fit caps | Shared zero witness `-3432.74`; each target reports `[long, short, zero, pure]` cap counts `[1,1,1,0]`; 39 fit caps and zero executed caps |
| Frozen-weekly mean-volatility 1 | 2023-01-17 | Genuine recession/unbounded economic objective | Shared zero witness `-0.0467931`; no fit or executed caps |
| Frozen-weekly mean-volatility 2 | 2023-01-18 | Genuine recession/unbounded economic objective | Shared zero witness `-0.0712521`; no fit or executed caps; 2023-01-17 passes |
| Daily mean-variance 0.1 | 2023-01-31 | v1.20 certificate/diagnostic bug | Refined KKT `9.9499257e-7 <= 1e-6`, feasibility zero, but stale objective error `2.88882e-7` exceeds the approximately `2.59135e-7` limit |

## Why the recession diagnoses are conclusive

For a zero claim and zero initial pricing inventory, the unregularized
mean-volatility objective is positively homogeneous in the action:

`phi(a) = -mean(Ga) + lambda * std(Ga) + deterministic_cost(a)`.

Therefore any feasible finite direction with `phi(a) < 0` is a constructive
certificate: `phi(t a) = t phi(a)` tends to negative infinity as `t` grows.
The standardized quadratic ridge makes the numerical surrogate finite; it does
not make the declared unregularized economic objective bounded. An optimizer's
first-order certificate is not required to validate a negative recession
witness, because the witness itself proves unboundedness.

The v1.20 source reconstructs the exact objective, subtracts the declared ridge,
and flags a zero-claim mean-volatility direction below `-1e-6` in
`weekly_hedger.py:1387-1417` and the corresponding daily code. The exact conic
objective includes the unchanged mean, centered-volatility, deterministic-cost,
and ridge terms (`weekly_hedger.py:2239-2268` and `:2385-2414`).

## Weekly evidence and cap semantics

### Risk parameter 0.1

- Production failure:
  `/home/hans/quantdata_data/experiments/hedging/weekly/historical-v120-7938ec8/frozen-weekly-historical-sanos-mean-volatility-0p1/numerical-gate-failure.json`
  has SHA-256
  `6e002a42e040843538c523dcb0b6e206a23f4cbe415d921045eb91d59f59f65c`.
- It records run identity
  `bed34dabd2b5e3d36406cbf88ae97d2a9ef8031303b89dc4f8f351267e08266c`,
  39 fit-cap hits, zero executed-cap hits, 13 nonconverged target policy fits,
  and 13 recession-affected target policy fits.
- Exact pinned diagnostic record:
  `/tmp/quant_data/hedging/historical-v120-review/diagnostic-status-origin-2023-01-17/records/2023-01-17.json`
  has SHA-256
  `8692c5290ffd21025797aa8d124d15e4855dc9683c4002448883d6852cf2b996`.
- Captured component statuses:
  `/tmp/quant_data/hedging/historical-v120-review/diagnostic-status-origin-2023-01-17/captured-statuses.json`
  have SHA-256
  `07a021d6c66a8edf4959bc395a7cfe47a960e2e3a35567d4cfcd94c78fa6656e`.
- All 13 target fits have one frozen policy identity repeated over five decision
  dates. There are 65 decision slots, 13 unique target policy identities, and
  one common market-fit identity. Deduplication by target policy identity gives
  exactly 13 affected fits.
- For every target, long, short, and zero pricing policies each hit one internal
  coordinate cap, while the centered pure hedge hits none. The long and short
  policies are independently converged and non-recessionary; the zero policy is
  the recession component. Each zero solve reports `CLARABEL=optimal`, exact
  objective error `2.27374e-13`, zero constraint violation, and recession value
  `-3432.74`. Maximum executed absolute position over the book is about 4.191,
  far below the executable cap of 1,000.

The aggregate count of 39 is therefore correct under the current schema's
affected-target-policy semantics: three internal component cap hits per each of
13 target fits. It is not 39 executed clips and it is not 39 distinct economic
recession directions. The underlying zero direction is shared, but every target
policy depends on it and is correctly rejected.

### Risk parameter 1

- Production failure SHA-256:
  `181d6595fe81ab5c06f4a3deec7925c500e434101dc7c459e66c0030e6b409ae`.
- Exact diagnostic record SHA-256:
  `4057728c133ed28071e5b7a7333abfa228807807c6a613221b35c73138fd78fc`.
- Captured-status SHA-256:
  `d1e9a504cfc840177a8827bfc71ab85dffaaf607c1f214752b87b2fed1e02cf6`.
- All 13 target fits have converged, non-recessionary long, short, and pure
  policies. Only the common zero policy fails: `CLARABEL=optimal`, exact
  objective error `4.44089e-16`, zero constraint violation, recession value
  `-0.0467931`. There are zero internal and zero executed cap hits.

### Risk parameter 2

- Production failure SHA-256:
  `af9aabba0080b9675538ffa16235fcb0515caf0efacc88ad65233585937b846e`.
- Exact rejected-origin diagnostic record SHA-256:
  `9aebea6fd68773d612b26daaecb8d91b820cfa0765327b928b178ad886f51df5`.
- Captured-status SHA-256:
  `f487e1636b23d0f474254809d7ba56b980ba31d26a8f0ad902ed63624b2a9f5d`.
- Origin 2023-01-17 is a valid recorded pass; the first recession appears on
  2023-01-18. All 13 target fits again have converged long, short, and pure
  policies. The common zero policy reports `CLARABEL=optimal`, exact objective
  error `2.22045e-16`, zero constraint violation, recession value `-0.0712521`,
  and zero cap hits.

The weekly gate logic is internally consistent. `weekly.py:670-701` deduplicates
the five frozen decision references by `policy_fit_identity`, separately counts
executed actions at the 1,000 cap, and returns 39/0/13/13 for the 0.1 case and
0/0/13/13 for the 1 and 2 cases. `weekly.py:633-652` makes recession and
nonconvergence unconditionally fatal; allowing cap hits cannot admit them.

The requested production `records/2023-01-17.json` does not exist for cases
rejected at their first origin. This is intentional fail-before-record behavior:
`weekly.py:403-446` writes `numerical-gate-failure.json`, enforces the gate, and
only writes a normal record if enforcement returns. The scratch records above
were produced by disabling only the raising function after preserving the exact
v1.20 configuration, input hashes, and production run identities. They are
diagnostic evidence, not production records.

## Daily mean-volatility 0.1

- Production failure SHA-256:
  `ed9dd75ebbb43afa9028f93571534855d84683079f5878b2e2e57bae7910020d`.
- Exact pinned diagnostic record:
  `/tmp/quant_data/hedging/historical-v120-review/daily-meanvol-0p1/records/2023-01-17.json`
  has SHA-256
  `53fac098a2e9806c14e4b3f7ace0bcf141b915df7596dd7d84523b5674d80181`
  and the exact production run identity
  `dd5f05ee2ce35898ed36f62365546307d1635a58f5011561d12e5eee7301a445`.
- All 20 claim policies are non-recessionary. The one shared zero policy reports
  `CLARABEL=optimal`, exact objective error `1.42109e-14`, zero constraint
  violation, KKT residual `0.0188601`, and exact unregularized recession value
  `-182.981`.
- The gate correctly counts one shared recession fit and 20 affected target
  results. No internal or executable coordinate reaches its cap. Claim and zero
  virtual gross exposures are about 3.874 million, while pure hedge gross
  exposures are only about 8.3 to 12.3.

The KKT miss means the ridge-surrogate optimizer is not independently certified
as optimal. It does not invalidate the recession classification: the exact
negative value at a finite zero-claim direction is already sufficient to prove
that the positively homogeneous economic objective is unbounded. Treating this
as only a numerical failure would silently discard the stronger economic
certificate.

## Daily mean-variance 0.1 diagnostic bug

- Production failure SHA-256:
  `5cd2499b3eecd4cb4b326dce1bf26221542e3f0d73f8756530d1b75d07f8786f`.
- Exact production-identity diagnostic record:
  `/tmp/quant_data/hedging/historical-v120-review/daily-meanvar-0p1/records/2023-01-31.json`
  has SHA-256
  `a570f0433097be06b408bac5e01a9c5494827563e8bb1563a03934fdb96a48c4`
  and run identity
  `65d1371f412d6ca05608fada807c03e0697d216ca06f2ae89ff255de047d224a`.
- Nineteen targets pass. `digital-call-3d-nms+0.50` alone fails with no cap and
  no recession. Its selection fit passes. Its final refined action has KKT
  `9.949925712003365e-7`, below the unchanged `1e-6` certificate tolerance,
  and zero constraint violation.
- The sole failed condition is objective agreement:
  `2.88882e-7` versus `1e-7 * (1 + abs(-1.591353531...))`, approximately
  `2.59135e-7`.

In v1.20, `hedger.py:892-965` reconstructs the candidate objective but compares
it with `problem.value`. After both external solvers fail the full certificate,
`hedger.py:1002-1028` changes the action through 268 coordinate-refinement
sweeps and calls the same certificate without updating the CVXPY objective.
The refined candidate is therefore compared with a cached objective belonging
to a different pre-refinement action. This is a certificate bookkeeping bug,
not an optimizer or economic failure.

## v1.21 candidate review

The reviewed working-tree patch is on top of v1.20 commit `7938ec8` and has
these SHA-256 values:

- `config.py`: `5b26939fa05375646033acac5fab3f8dd156892c8e3a7d09b25c4356568de73b`
- `hedger.py`: `b642a265b71fb3690906046888f01a5b7c0c3bb1deba2335d95a735a173fdc30`
- daily Historical-SANOS tests:
  `67ed0caa087287baff625d5dd220eb852ae501e67c041f5a58f601b663a0e803`

The patch:

1. adds an explicit reference objective to the daily mean-variance certificate;
2. assigns the refined physical action to the CVXPY scaled-action variable;
3. evaluates `objective.value` at that same action and restores the prior
   variable value;
4. compares this independently evaluated CVXPY expression with the unchanged
   exact NumPy objective; and
5. bumps `HistoricalSanosConfig` from 1.20 to 1.21.

This is mathematically sound. The CVXPY and NumPy implementations evaluate the
same unchanged objective at the same candidate; primal feasibility and KKT are
still checked independently at their original tolerances. No stale value is
accepted and no genuine failure gate is relaxed. The version bump is required
because convergence and serialized identities can change.

The new regression corrupts only cached `problem.value`, forces the independent
coordinate-refinement path, and verifies that same-action objective
reconstruction plus KKT certifies it. The exact real-data v1.21 replay is:

- record:
  `/tmp/quant_data/hedging/diagnostics/v121-daily-mv0p1-2023-01-31/records/2023-01-31.json`,
  SHA-256
  `5479b142d0f7bb63724fc5f23a5766f09adf97115c5944864739dce5e698a2a5`;
- summary:
  `/tmp/quant_data/hedging/diagnostics/v121-daily-mv0p1-2023-01-31/summary.json`,
  SHA-256
  `8b3b90cb58516a88a5a0cbac9edb36e87d15f230d59ae2b97567936a8734b02a`.

It passes all 20 targets with zero caps and zero nonconvergence. The formerly
rejected target retains KKT `9.949925712003365e-7` and now records same-action
objective error `1.33227e-15`.

## Independent checks

- Four focused candidate tests passed: the new same-action objective
  reconstruction regression, coordinate-refinement KKT regression,
  all-solver-failures regression, and daily recession/cap regression.
- Two pinned-v1.20 weekly tests passed: strict runner recession/cap enforcement
  and virtual recession without executable clipping.
- Focused `git diff --check` passed for the three v1.21 candidate files.
- Exact CPU replays reproduced all five production outcomes; no GPU was used.

## Required production treatment

1. Commit and pin the reviewed v1.21 patch before rebuilding matrices.
2. Gracefully stop remaining v1.20 Historical-SANOS production work. The shared
   config version bump changes daily and weekly identities, so further v1.20
   successes cannot be authenticated as v1.21 outputs.
3. Rebuild and execute the full daily and weekly Historical-SANOS matrices at
   the exact v1.21 commit. Do not copy successful v1.20 records into v1.21.
4. Preserve the v1.20 recession artifacts as failure evidence and expect the
   four genuine mean-volatility cells to fail again under unchanged economics.
   Canonicalize them in the failure ledger rather than retrying with relaxed
   tolerances or caps.
5. Do not use `--allow-cap-hits` to admit a recession, increase the internal cap,
   strengthen the ridge until the direction disappears, substitute the pure
   hedge for the pricing policies, or suppress nonconvergence. Any bounded
   numerical result for these four cells requires a separately precommitted
   economic specification and must be a new benchmark case.
