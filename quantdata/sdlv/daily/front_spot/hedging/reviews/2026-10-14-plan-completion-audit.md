# Hedging PLAN Completion Audit

## Audit Rule

This audit maps the requirements in `quantdata/hedging/PLAN.md` to evidence
that exists now. A phase is marked complete only when its stated gate and its
production scope are both proved by current artifacts. A running service,
partial checkpoint, smoke test, or plausible downstream path is not completion
evidence.

Snapshot timestamp: 14 October 2026, 22:31 Europe/London. Generated-model
readiness and the final report coordinator run from immutable reviewed commit
`b20dea5bae0ebdce3c698565d74e4de2c729824f`. That successor preserves the
earlier initializer-label remediation and additionally authenticates the
canonical weekly diffusion identities against the runtime adapter identities.
The repository documentation commit does not alter the pinned production
identity.

## Requirement Matrix

| Scope | Current finding | Authoritative evidence | Evidence still required |
|---|---|---|---|
| Public distributable dataset | **Complete technical gate.** The 891-date, 2,014,740-coordinate artifact reconstructs from an empty data root and excludes original WRDS prices, midpoints, and identifiers. Redistribution rights remain a separate legal question. | `2026-10-14-public-dataset-validation.md`; public dataset `manifest.json` and its full validator. | None for the technical implementation goal. |
| Protocol and accounting | **Complete.** Timing, executable SANOS marks, half-spreads, spot costs, free settlement, fixed weekly books, and pricing-versus-hedging decomposition have focused tests and review. | `2026-10-12-cpu-vertical-slice.md`; `2026-10-12-weekly-benchmark-engine-review.md`; `2026-10-12-weekly-common-pricing-review.md`. | Final production records must continue to pass these validators. |
| SANOS--LV reference | **Implementation and daily production complete; both weekly schedules have authenticated terminal strict failures.** Current-notional daily evidence has 611 origins and 12,220 target records. Frozen and restarted weekly execution each terminate at origin `2023-02-08` with the same 26 executed position-cap hits and no fit-cap hit or optimizer nonconvergence. | `2026-10-12-sanos-lv-phase2.md`; `2026-10-12-sanos-lv-weekly-policy-quant-review.md`; authenticated daily production matrix v3 output; combined weekly failure ledger `failure-ledger-65fbf1d.json`, SHA-256 `2dcbece08a86939ead37f17fd03f02ce5505b8a32f4cd562104bc3cc46c13505`. | Preserve both failures in the canonical weekly matrix and final report; do not replace them with relaxed or current-NMS counterfactuals. |
| Independent covariance baseline | **Implementation complete; production incomplete.** Mean--variance, mean--volatility, Gaussian entropy equivalence, and exact Gaussian power OCE pass review. Three power cohorts are queued. | `2026-10-12-covariance-phase3-rereview.md`; `2026-10-12-covariance-dlv-return-correction-review.md`; `2026-10-14-covariance-gaussian-power-review.md`. | Complete and authenticate all ten fitted-policy cases, including the three 611-origin power cohorts and the precommitted low-risk mean--volatility diagnostic. |
| Rolling SDLV estimation | **Complete implementation gate.** Rolling 252-transition estimation, rank 16, aligned factors, eight Greek factors, Q drift, and compatible P drift pass remediation review. | `2026-10-12-rolling-dlv-phase4-remediation-review.md`; `2026-10-12-rolling-dlv-phase4-second-remediation-review.md`. | Dated production artifacts must be present for every GROMM origin. |
| GROMM market model and hedges | **Implementation and current-code rolling range complete; policy production incomplete.** The authenticated range contains all 611 origins, with 576 accepted full physical candidates and 35 declared front-only fallbacks. | `2026-10-12-gromm2-architecture-quant-review.md`; `2026-10-12-gromm-hedging-integration-remediation-review.md`; `2026-10-14-rolling-gromm2-physical-selection-review.md`; `2026-10-14-production-handoff-remediation.md`; current range `gromm2-rolling-v4/range-fit-2023-2025-v7.json`, SHA-256 `8f7e6a63e90dfbfa367a06a220c75ebb4d0c9a0aa5f61ff2a92d8a58858b924e`. | Complete every enabled daily and weekly GROMM policy. |
| GPU production infrastructure | **Mechanical gate, current rolling migration, initializer-label remediation, and weekly canonical/runtime identity remediation complete; downstream production evidence incomplete.** Restart, source pinning, range authentication, progress heartbeat paths, distinct initializer/evaluation identities, and the two weekly diffusion aliases pass review. | `2026-10-13-gromm-production-provenance-review.md`; `2026-10-13-evaluation-heartbeat-review.md`; `2026-10-14-production-handoff-remediation.md`; `2026-10-14-generated-initializer-label-lineage-review.md`; `2026-10-14-weekly-generated-model-identity-review.md`; focused restart tests; authenticated current range above. | Complete the corrected generated-readiness matrix, production gates, and GPU cases. |
| Historical SANOS baseline | **Implementation complete; version-1.11 production active.** Daily and weekly exact-cost solvers and OCE formulations pass review. The final 252-observation fit is authoritative; genuine terminal failures remain failures rather than being replaced by warm-start selection fits. | `2026-10-12-historical-sanos-weekly-quant-review.md`; `2026-10-14-historical-weekly-oce-review.md`; `2026-10-14-historical-exact-cost-production-remediation.md`; immutable version-1.11 matrices at `cd6f440626a59c58196880f47265a3f041962ec4`. | Complete every daily and weekly cell, canonicalize terminal failures, and retain cap-limited recession cases only under their declared diagnostic gate. |
| Published generated baselines | **Implementation, training lineage, initializer-label remediation, weekly identity remediation, and current-source quant review complete; readiness and production incomplete.** VolGAN, AD-Seq-Vol, and AD-Seq-Vol-FT have distinct authenticated lineages and use `hedging-generated-v1` for rolling checkpoint restoration independently of their evaluation labels. Weekly reports retain the canonical diffusion taxonomy while authenticating the adapter's runtime identity. | `2026-10-12-volgan-diffusion-integration-remediation-review.md`; `2026-10-12-generated-rolling-warmstart-quant-review.md`; `2026-10-13-generated-gate-assembler-review.md`; `2026-10-14-weekly-signed-indifference-prices.md`; `2026-10-14-generated-initializer-label-lineage-review.md`; `2026-10-14-weekly-generated-model-identity-review.md`; lineage snapshot SHA-256 `91f04fa3067c2df7d6978abbf083eea2a7d856d0be493a4345f44c894cd5742e`; three quant-review JSON files under `generated-readiness-b20dea5/evidence`. | Complete the corrected 18-scope GPU-readiness matrix, assemble all three production gates, and complete every enabled daily and weekly case. |
| Daily benchmark matrix | **Incomplete.** The canonical specification has 57 cases. Native queues, rolling GROMM, and generated cases are not all terminal. | Versioned daily production matrix builder and validators; active immutable services listed below. | A 57-case matrix in which every specified cell has an authenticated terminal outcome and every completed case reconstructs to the declared 12,220 records per policy. |
| Weekly benchmark matrix | **Incomplete; Historical-SANOS production is active and both SANOS--LV cells have authenticated failures.** The canonical specification has 94 cases. | `2026-10-13-weekly-production-quant-review.md`; native matrix SHA-256 `8d069cadde0c21dd677bbe7ab16cd59e0d3dae9907456608da7fcb4b9bce0b26`; combined SANOS--LV failure ledger above. | A 94-case matrix in which every specified cell has an authenticated terminal outcome and every completed case reconstructs to 528 origins and 6,864 authenticated target records. |
| Statistical reporting | **Python implementation, contract re-review, successor-launcher gate, weekly identity remediation, and pre-outcome final-review protocol complete; final evidence incomplete.** The version-2 pre-outcome commitment binds the dataset, matrices, failure ledgers, fixed commitment hash, and reporting commit. Daily and weekly aggregators retain complete strata, prices, distributions, costs, turnover, exposure, sparsity, caps, and convergence diagnostics. Daily comparisons use 20-target Bartlett-HAC plus moving-block-bootstrap inference; weekly comparisons use 13-target Hansen--Hodrick plus moving-block-bootstrap inference. | `2026-10-14-final-report-contract-rereview.md`; `2026-10-14-weekly-generated-model-identity-review.md`; `final-results-review-protocol.md`; report builder and summary tests; real daily-plus-weekly fragment compile; matrix-commitment adversarial tests; active reviewed `b20dea5` coordinator. | Rebuild all statistics from the complete canonical matrices and pass the fixed independent results-review protocol. |
| Final public report | **Incomplete.** The repository PDF is preliminary and predates full production. | Preliminary `reports/hedging_benchmark_summary.pdf`; active final successor described below. | Authenticated production PDF and manifest from the complete 57-case daily and 94-case weekly matrices, followed by repository refresh and final review. |

## Active Immutable Work

- The legacy affine range is complete for all 611 origins; its authenticated
  fit summary SHA-256 is
  `6cbdcb13913d322d95e1ff644950f638aaf090ea36278553d6eb898b2bce3205`.
- The current GROMM range is complete and authenticated for 611 origins; its
  summary SHA-256 is
  `8f7e6a63e90dfbfa367a06a220c75ebb4d0c9a0aa5f61ff2a92d8a58858b924e`.
- `quantdata-hedging-generated-readiness-b20dea5.service` owns the managed GPU
  and runs the corrected signed-price-aware 18-scope readiness matrix from
  immutable reviewed commit `b20dea5bae0ebdce3c698565d74e4de2c729824f`.
  At this snapshot eight scopes pass, the AD--Seq--Vol restarted-weekly public
  scope is active, and nine scopes remain. Its launcher SHA-256 is
  `8c1ed51e45dc2588ce50294b066e2e22ae3597bbc7fa4a94196c37f515fd009a`.
  The three regenerated quant-review evidence files have SHA-256 values
  `e105355410f9cf546b7c257b85f395e35a592df622c8818d171fab4ab23b3027`,
  `4ded54eb14b96a56670fe15fcf6b237949dc075b0b8624ab3ac2aa895b8796b8`,
  and `03aeded119b57a1269e5489cc7ba2fc1e71ce8fdae12f83427060fb939053901`.
- `quantdata-hedging-historical-v111-daily-cd6f440.service` owns all 13 daily
  Historical-SANOS cells. At this snapshot it has six complete, six strict
  numerical-gate failures, one active, and no queued cells.
- `quantdata-hedging-historical-v111-weekly-cd6f440.service` owns all 26 weekly
  Historical-SANOS cells. At this snapshot two are complete, four have strict
  numerical-gate failures, six are active, and fourteen are queued.
- `quantdata-hedging-failure-canonicalization-8851996-v2.service` waits for both
  version-1.11 matrices and will validate completed cells and canonicalize only
  genuine terminal failures. Its launcher SHA-256 is
  `70bd69a679fd9cef58aba1087c84e49f35f8db796e4cfde8176e17df47252846`.
- `quantdata-hedging-full-production-b20dea5.service` is active from immutable
  reviewed commit `b20dea5bae0ebdce3c698565d74e4de2c729824f` and waits on
  generated readiness and Historical-SANOS canonicalization. Its independently
  reviewed launcher writes the version-2 commitment and immutable SHA-256
  before final production, passed its exact detached-environment preflight, and
  has SHA-256
  `e6de16d5c645f1ae78e3f91b51376ff6496d029d1d5ae8066d8f72cb14207774`.
  It also authenticates the independently approved final-review protocol from
  commit `5fc1bc87f529630bf7b936e4eb77e2ffa6ce7bfe`, SHA-256
  `34ea83410614c3395bb4ccdf561ff95d1acd9ee572f413cfdf2d48bea29a9c7f`,
  before preflight, matrix preparation, queues, and rendering. No version-2
  matrix commitment exists yet. The obsolete `d22ec9a` and `de18586`
  coordinators were stopped while still waiting and launched no production
  case.
- The coordinators use service activity only for capacity scheduling. Their
  authoritative gates reconstruct immutable matrices, production gates, and
  result artifacts, avoiding a race with auto-collected transient units.
- A pre-final regression sweep on the current weekly-identity-remediated source
  passed all 406 SDLV hedging tests and all 58 shared hedging-protocol tests,
  for 464 passing tests in total. The independent review additionally passed
  77 focused tests, Ruff import checks, and `git diff --check`. This remains
  source-validation evidence only; acceptance item 6 also requires the suite on
  the final source after authenticated report artifacts are refreshed.
- This successor-chain audit refresh passed independent provenance review in
  `2026-10-14-plan-completion-audit-successor-review.md`.

## Final Acceptance Checklist

The goal is complete only after all of the following hold simultaneously:

1. **Satisfied.** The public dataset validator passes from an otherwise empty
   data root.
2. **Satisfied.** The rolling GROMM current-code range contains exactly 611
   authenticated origins and passes its physical-selection gate.
3. Every one of the 57 daily cases has an authenticated terminal outcome; every
   completed case reconstructs to exactly 12,220 current-notional records with
   its declared convergence/cap policy, and every failed case is represented by
   a validated failure ledger rather than an omitted or relaxed result.
4. Every one of the 94 weekly cases has an authenticated terminal outcome;
   every completed case reconstructs to exactly 528 origins and 6,864 records,
   every failed case is represented by a validated failure ledger, and frozen
   and restarted schedules remain distinct.
5. The final report builder authenticates both matrix hashes and regenerates
   its tables, appendix, TeX, PDF, and manifest from the dated records.
6. The complete hedging test suite passes on the final source commit.
7. An independent quantitative results review reconciles headline numbers,
   objectives, costs, caps, pricing decomposition, and overlapping inference.
8. The final PDF and its provenance are copied to the repository, committed,
   and pushed without modifying unrelated user work.
