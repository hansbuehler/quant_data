# Hedging PLAN Completion Audit

## Audit rule

This audit maps `quantdata/hedging/PLAN.md` to evidence that exists at the
snapshot time. A service being active, a smoke test passing, or a partial
record prefix does not prove production completion. A benchmark cell is
terminal only when its complete result or authenticated strict-failure ledger
passes the matrix validator.

Snapshot: 16 October 2026, 07:44 Europe/London.

## Requirement matrix

| Scope | Current finding | Authoritative evidence | Evidence still required |
|---|---|---|---|
| Public distributable dataset | **Technical gate complete.** The released artifact contains 891 dates and 2,014,740 model coordinates, reconstructs from an otherwise empty data root, and excludes original WRDS prices, midpoints, and identifiers. Redistribution rights remain a separate legal question. | `2026-10-14-public-dataset-validation.md`; released `manifest.json`; full public-data validator. | None for the technical implementation goal. |
| Protocol and accounting | **Complete implementation gate.** Contract timing, executable SANOS marks, half-spreads, spot costs, free settlement, fixed weekly books, and pricing-versus-hedging decomposition have focused tests and reviews. | `2026-10-12-cpu-vertical-slice.md`; `2026-10-12-weekly-benchmark-engine-review.md`; `2026-10-12-weekly-common-pricing-review.md`. | Final records must continue to pass the same validators. |
| Dedicated model folders and standalone tests | **Implementation complete.** SANOS--LV, covariance, Historical-SANOS, GROMM, VolGAN, and diffusion each have a dedicated directory, README, deterministic CPU smoke entry point, committed fixture checks, and focused pytest module. | `models/*/README.md`; six `models/*/smoke.py` entry points; six `tests/quantdata/sdlv/daily/hedging/models/test_*.py` primary modules. | Rerun all standalone smokes and the complete hedging test tree from the final committed source after production. |
| Target sizing, costs, and caps | **Implementation and pre-outcome gate complete.** Target notionals are selected strictly from 2022 episodes, option entry and liquidation costs follow the declared half-spread rules, spot costs one basis point per trade, cash settlement is free, and the 1,000-unit executable cap is reported separately from internal fitting caps. | Target-sizing artifacts and tests; `2026-10-13-weekly-production-quant-review.md`; accounting reviews; run validators. | Authenticate the same sizing identities and cap diagnostics in every final daily and weekly record. |
| CPU/GPU execution split | **Infrastructure and readiness complete; final execution pending.** CPU preparation, deterministic smokes, managed-GPU training/evaluation, checkpoint recovery, resource checks, and four-minute progress logging are implemented. The accepted generated-readiness matrix covers all 18 model/schedule scopes. | Generated-readiness manifest and independent review; `2026-10-13-evaluation-heartbeat-review.md`; `2026-10-13-concurrent-training-weekly-repin-review.md`; coordinator review SHA-256 `8cd4391583f29b09a59022c78351b93c3821dedcaf5f80b9a16dc56d5081db43`. | Run the reviewed final coordinator after the Historical-SANOS handoff authenticates; retain progress and checkpoint evidence for every material GPU process. |
| Independent quantitative reviews | **All implementation/readiness milestones reviewed; final-results review pending by design.** Protocol, accounting, each model family, generated readiness, solver remediation, production provenance, and report contracts have stored independent reviews. | Review directory inventory; `final-results-review-protocol.md`; current coordinator and readiness review hashes. | Perform the precommitted independent final-results review on the complete matrices, tables, PDF, and manifest. |
| SANOS--LV and covariance baselines | **Implementation complete; reusable native production evidence exists.** Current-notional SANOS--LV and covariance mean--variance cohorts are authenticated. The covariance power-exponential cohorts are also complete. | Phase-2 and Phase-3 reviews; native production authentication reports referenced by the plan. | Compose every reusable result into the final matrices and preserve declared strict failures. |
| Rolling SDLV and GROMM market state | **Complete implementation and range gate.** The current range contains 611 authenticated origins, 576 accepted full physical candidates, and 35 declared front-only fallbacks. | `gromm2-rolling-v4/range-fit-2023-2025-v7.json`; rolling-DLV and physical-selection reviews. | Every final GROMM policy cell must bind this exact range. |
| GROMM policy architecture | **Complete after remediation.** The current factor estimator invalidated the older v1 study. The fresh CPU-only 40-case study evaluated all four candidates and selected `gromm-architecture-2022-h64-d3-w128` under the precommitted one-standard-error rule. | `2026-10-15-gromm-architecture-remediation-review.md`; result SHA-256 `ed68dd71a87465507191f476cf85eea691596751fa122d95ac21c7381abb166b`; launcher SHA-256 `c3e2756b3997fc91e751d774ab32293582abfe3ad0caf014ee0dce6bb639b5b7`. | Bind every final GROMM policy result to the authenticated result and launcher. |
| Historical SANOS | **Implementation complete; strict v1.22 production active.** The daily and weekly implementations use 252 uniformly weighted scenarios, the declared ridge, unchanged economic objectives, and independent numerical certificates. At this snapshot the daily chain has three complete mean--variance cases, four active cases, five queued cases, and one authenticated recession direction; the weekly chain has four active cases, eighteen queued cases, and four authenticated recession directions. These directions are economic boundedness findings, not solver failures. | `2026-10-16-historical-sanos-v122-solver-review.md`; `2026-10-16-historical-v122-weekly-runtime-review.md`; immutable source `8b7ca46c73ddf278ebcb23b7ddcb4eb9b12b8edf`; daily matrix SHA-256 `b99b930c5c76b5ec683ca784fcf19362649127a74565b40477b2dd6534976c5a`; weekly matrix SHA-256 `84824c192c9ef0ac5a275232ed75fab3de89b28f38f4342b16365edd3c1eb2a4`. | Finish both queues and publish the authenticated v1.22 daily and weekly handoff plus failure ledgers. |
| Published generated baselines | **Training, readiness, and independent review complete.** VolGAN, AD--Seq--Vol, and AD--Seq--Vol-FT have authenticated initializers, budgets, and distinct lineages. All eighteen public/rolling daily/frozen/restarted scopes pass; eight weekly scopes carry fitted-policy cap warnings, with zero executed cap hits, nonconvergence, recession directions, or failures. | Immutable source `8b7ca46c73ddf278ebcb23b7ddcb4eb9b12b8edf`; readiness launcher SHA-256 `cd49bc88be0d8ea1a0ca8dbfd108b1a3ae6df0b2e003f97a84a80137704fed28`; manifest SHA-256 `e6a00b1c56af607c50f3b5ab91b492c2f849b45421c8175e402a6c56073c2175`; independent review SHA-256 `a5e66746fd118e6fcb1e24f6f631f8206cf0344a0946c58299cf004451901e12`. | Supply the authenticated manifest and independent-review hashes to the final-production coordinator once the Historical-SANOS handoff is complete. |
| Daily benchmark matrix | **Incomplete.** The canonical grid has 57 cases. Native reusable evidence exists, but Historical-SANOS, GROMM, and generated cells are not all terminal. | Daily matrix builder, validators, active immutable services. | Exactly 57 authenticated terminal outcomes; each successful policy reconstructs to 611 origins and 12,220 target records. |
| Weekly benchmark matrix | **Incomplete.** The canonical grid has 94 cases with distinct frozen and restarted schedules. | Weekly matrix builder, validators, active immutable services. | Exactly 94 authenticated terminal outcomes; each successful policy reconstructs to 528 origins and 6,864 target records. |
| Statistical reporting | **Implementation and pre-outcome protocol complete; final evidence pending.** Daily HAC/bootstrap and weekly Hansen--Hodrick/bootstrap paths, prices, costs, turnover, exposure, sparsity, caps, convergence, and objective appendices are implemented. | `2026-10-14-final-report-contract-rereview.md`; `final-results-review-protocol.md`; report builder tests. | Rebuild from the complete committed matrices and pass the independent results review. |
| Final public report | **Incomplete.** The repository PDF is preliminary. | Preliminary `reports/hedging_benchmark_summary.pdf`. | Regenerate the authenticated PDF and manifest from complete matrices, rerun the full suite, review, commit, and push. |

## Active immutable chain

- `quantdata-hedging-generated-readiness-8b7ca46-v122-full.service` completed
  all 18 generated-model readiness scopes with 4,096 scenarios from immutable
  commit `8b7ca46c73ddf278ebcb23b7ddcb4eb9b12b8edf`. Its launcher SHA-256 is
  `cd49bc88be0d8ea1a0ca8dbfd108b1a3ae6df0b2e003f97a84a80137704fed28`.
- Trust revision r7 published immutable manifest SHA-256
  `e6a00b1c56af607c50f3b5ab91b492c2f849b45421c8175e402a6c56073c2175`.
  Independent review passes at SHA-256
  `a5e66746fd118e6fcb1e24f6f631f8206cf0344a0946c58299cf004451901e12`.
- `quantdata-hedging-historical-v122-daily-v6.service` owns the 13 daily
  Historical-SANOS cases: three complete mean--variance cases, one authenticated
  recession direction, four active cases, and five queued cases at this
  snapshot.
- `quantdata-hedging-historical-v122-weekly-v6.service` owns the 26 weekly
  Historical-SANOS cases: four authenticated recession directions, four active
  cases, and eighteen queued cases at this snapshot. Independent runtime review
  found no demonstrated, provenance-safe benefit from restarting or changing
  the reviewed four-worker queue.
- `quantdata-hedging-historical-v122-handoff-8b7ca46-r2.service` waits for the
  two Historical-SANOS services, validates every terminal case and its failure
  classification, and publishes the hash-bound handoff. Its launcher SHA-256 is
  `ab0a7b6caaeba9d0eddd7f145c812384eb2eed18d8bfac2614d58406d9df0d4a`.
- The reviewed r2 final coordinator launcher has SHA-256
  `981734321acd116b2b7afb7239840558e4f9c670543be843e3b8e9ff95553590`.
  Its independent PASS review has SHA-256
  `8cd4391583f29b09a59022c78351b93c3821dedcaf5f80b9a16dc56d5081db43`.
  It authenticates the accepted r7 readiness manifest and review and remains
  deliberately unlaunched until the exact Historical-SANOS handoff hash can be
  supplied externally.
- No final matrix commitment exists at this snapshot. This is correct: the
  coordinator is forbidden to commit matrices before both prerequisite
  handoffs authenticate.

## Final acceptance checklist

The goal is complete only after all items hold simultaneously.

1. **Satisfied.** The public dataset validates from an otherwise empty data
   root.
2. **Satisfied.** The current rolling GROMM range contains exactly 611
   authenticated origins and passes its physical-selection gate.
3. **Satisfied implementation gate.** The replacement 2022 GROMM architecture
   study is authenticated, and both final matrix builders require its exact
   result and launcher hashes. Items 4 and 5 still require the resulting cells
   to become terminal.
4. Every one of the 57 daily cases has an authenticated terminal outcome;
   every completed case reconstructs to 12,220 records, and every failed case
   has a validated ledger entry rather than an omitted or relaxed result.
5. Every one of the 94 weekly cases has an authenticated terminal outcome;
   every completed case reconstructs to 528 origins and 6,864 records, every
   failed case has a validated ledger entry, and frozen and restarted schedules
   remain distinct.
6. The final report builder authenticates both committed matrix hashes and
   regenerates all tables, appendices, TeX, PDF, and manifest from dated
   records.
7. The complete hedging test suite passes on the final source commit.
8. An independent quantitative results review reconciles headline numbers,
   objectives, costs, caps, pricing decomposition, and overlapping inference.
9. The final PDF and provenance are copied into the repository, committed, and
   pushed without modifying unrelated work.
