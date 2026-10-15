# Hedging PLAN Completion Audit

## Audit rule

This audit maps `quantdata/hedging/PLAN.md` to evidence that exists at the
snapshot time. A service being active, a smoke test passing, or a partial
record prefix does not prove production completion. A benchmark cell is
terminal only when its complete result or authenticated strict-failure ledger
passes the matrix validator.

Snapshot: 15 October 2026, 19:15 Europe/London.

## Requirement matrix

| Scope | Current finding | Authoritative evidence | Evidence still required |
|---|---|---|---|
| Public distributable dataset | **Technical gate complete.** The released artifact contains 891 dates and 2,014,740 model coordinates, reconstructs from an otherwise empty data root, and excludes original WRDS prices, midpoints, and identifiers. Redistribution rights remain a separate legal question. | `2026-10-14-public-dataset-validation.md`; released `manifest.json`; full public-data validator. | None for the technical implementation goal. |
| Protocol and accounting | **Complete implementation gate.** Contract timing, executable SANOS marks, half-spreads, spot costs, free settlement, fixed weekly books, and pricing-versus-hedging decomposition have focused tests and reviews. | `2026-10-12-cpu-vertical-slice.md`; `2026-10-12-weekly-benchmark-engine-review.md`; `2026-10-12-weekly-common-pricing-review.md`. | Final records must continue to pass the same validators. |
| SANOS--LV and covariance baselines | **Implementation complete; reusable native production evidence exists.** Current-notional SANOS--LV and covariance mean--variance cohorts are authenticated. The covariance power-exponential cohorts are also complete. | Phase-2 and Phase-3 reviews; native production authentication reports referenced by the plan. | Compose every reusable result into the final matrices and preserve declared strict failures. |
| Rolling SDLV and GROMM market state | **Complete implementation and range gate.** The current range contains 611 authenticated origins, 576 accepted full physical candidates, and 35 declared front-only fallbacks. | `gromm2-rolling-v4/range-fit-2023-2025-v7.json`; rolling-DLV and physical-selection reviews. | Every final GROMM policy cell must bind this exact range. |
| GROMM policy architecture | **Complete after remediation.** The current factor estimator invalidated the older v1 study. The fresh CPU-only 40-case study evaluated all four candidates and selected `gromm-architecture-2022-h64-d3-w128` under the precommitted one-standard-error rule. | `2026-10-15-gromm-architecture-remediation-review.md`; result SHA-256 `ed68dd71a87465507191f476cf85eea691596751fa122d95ac21c7381abb166b`; launcher SHA-256 `c3e2756b3997fc91e751d774ab32293582abfe3ad0caf014ee0dce6bb639b5b7`. | Bind every final GROMM policy result to the authenticated result and launcher. |
| Historical SANOS | **Implementation complete; strict production active.** Daily v1.13 and weekly v1.14 use 252 uniformly weighted scenarios, the declared ridge, unchanged economic objectives, and independent numerical certificates. Fast terminal failures are retained rather than relaxed. | `2026-10-15-historical-sanos-v113-review.md`; `2026-10-15-historical-sanos-v114-recession-review.md`; matrices with SHA-256 `ac2f6feae8da74baa1dc6b71f463948363b46f0d02f82322955ae107032151f1` and `97dd10954e0982caea8ffdad86aa5aca5aeb9ea43297e303f606d3996517ace8`. | Finish both queues, rerun every cell through canonical queues, and authenticate daily and weekly failure ledgers. |
| Published generated baselines | **Training and quantitative review complete; current-schema readiness pending.** VolGAN, AD--Seq--Vol, and AD--Seq--Vol-FT have authenticated initializers, budgets, and distinct lineages. | Three quant-review artifacts under `generated-readiness-fe7c3a7/evidence`; readiness preflight and process-group shutdown test. | Complete all 18 schema-v3 readiness scopes at 4,096 CUDA scenarios and assemble the three production gates. |
| Daily benchmark matrix | **Incomplete.** The canonical grid has 57 cases. Native reusable evidence exists, but Historical-SANOS, GROMM, and generated cells are not all terminal. | Daily matrix builder, validators, active immutable services. | Exactly 57 authenticated terminal outcomes; each successful policy reconstructs to 611 origins and 12,220 target records. |
| Weekly benchmark matrix | **Incomplete.** The canonical grid has 94 cases with distinct frozen and restarted schedules. | Weekly matrix builder, validators, active immutable services. | Exactly 94 authenticated terminal outcomes; each successful policy reconstructs to 528 origins and 6,864 target records. |
| Statistical reporting | **Implementation and pre-outcome protocol complete; final evidence pending.** Daily HAC/bootstrap and weekly Hansen--Hodrick/bootstrap paths, prices, costs, turnover, exposure, sparsity, caps, convergence, and objective appendices are implemented. | `2026-10-14-final-report-contract-rereview.md`; `final-results-review-protocol.md`; report builder tests. | Rebuild from the complete committed matrices and pass the independent results review. |
| Final public report | **Incomplete.** The repository PDF is preliminary. | Preliminary `reports/hedging_benchmark_summary.pdf`. | Regenerate the authenticated PDF and manifest from complete matrices, rerun the full suite, review, commit, and push. |

## Active immutable chain

- `quantdata-hedging-generated-readiness-fe7c3a7.service` waits for a managed
  GPU, then reruns all 18 generated-model readiness scopes from immutable
  commit `fe7c3a7321345e8ced636c5f75518766b830e452` with 4,096 scenarios.
  Its launcher SHA-256 is
  `c8edb76fdbfa5f45dec274d24e9fc37466400ac381f513b9dd8bfafd6c5363aa`.
- `quantdata-hedging-historical-v113-daily-f4d42c2.service` owns all 13 daily
  Historical-SANOS cases. At this snapshot it has seven strict failures and
  six active cases. Active cases continue to write dated atomic records.
- `quantdata-hedging-historical-v114-weekly-46d9f22.service` owns all 26
  weekly Historical-SANOS cases. At this snapshot it has nine strict failures,
  six active cases, and eleven queued cases. Active cases continue to write
  dated atomic records.
- `quantdata-hedging-failure-canonicalization-v114-46d9f22.service` waits for
  the two Historical-SANOS services to become inactive. It then reruns every
  daily, frozen-weekly, and restarted-weekly case with
  `--continue-after-failure`, validates successful records, and emits
  authenticated daily and weekly failure ledgers plus a hash-bound handoff.
  Dependency service exit codes are diagnostic only; artifact validation is
  authoritative, so expected strict failures cannot deadlock the handoff.
- `quantdata-hedging-full-production-fe7c3a7.service` has passed source,
  architecture, static-input, and review-protocol preflight. It waits for the
  generated readiness and Historical-SANOS handoff before writing the matrix
  commitment or launching final production. Its launcher SHA-256 is
  `d729a52ee477f1db48b1e8eb68cf608eeb185a7fb3cb680f0efaade89891973d`.
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
