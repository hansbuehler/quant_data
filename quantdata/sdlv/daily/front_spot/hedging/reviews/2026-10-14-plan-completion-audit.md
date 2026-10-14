# Hedging PLAN Completion Audit

## Audit Rule

This audit maps the requirements in `quantdata/hedging/PLAN.md` to evidence
that exists now. A phase is marked complete only when its stated gate and its
production scope are both proved by current artifacts. A running service,
partial checkpoint, smoke test, or plausible downstream path is not completion
evidence.

Snapshot time: 14 October 2026, 06:43 BST. Repository commit:
`661d5752c7813de3d7ab2cbdb8998a0b0ccc47ee`.

## Requirement Matrix

| Scope | Current finding | Authoritative evidence | Evidence still required |
|---|---|---|---|
| Public distributable dataset | **Complete technical gate.** The 891-date, 2,014,740-coordinate artifact reconstructs from an empty data root and excludes original WRDS prices, midpoints, and identifiers. Redistribution rights remain a separate legal question. | `2026-10-14-public-dataset-validation.md`; public dataset `manifest.json` and its full validator. | None for the technical implementation goal. |
| Protocol and accounting | **Complete.** Timing, executable SANOS marks, half-spreads, spot costs, free settlement, fixed weekly books, and pricing-versus-hedging decomposition have focused tests and review. | `2026-10-12-cpu-vertical-slice.md`; `2026-10-12-weekly-benchmark-engine-review.md`; `2026-10-12-weekly-common-pricing-review.md`. | Final production records must continue to pass these validators. |
| SANOS--LV reference | **Implementation and daily production complete; weekly production incomplete.** Current-notional daily evidence has 611 origins and 12,220 target records. | `2026-10-12-sanos-lv-phase2.md`; `2026-10-12-sanos-lv-weekly-policy-quant-review.md`; authenticated daily production matrix v3 output. | Both complete 528-origin weekly schedules in the canonical weekly matrix. |
| Independent covariance baseline | **Implementation complete; production incomplete.** Mean--variance, mean--volatility, Gaussian entropy equivalence, and exact Gaussian power OCE pass review. Three power cohorts are queued. | `2026-10-12-covariance-phase3-rereview.md`; `2026-10-12-covariance-dlv-return-correction-review.md`; `2026-10-14-covariance-gaussian-power-review.md`. | Complete and authenticate all ten fitted-policy cases, including the three 611-origin power cohorts and the precommitted low-risk mean--volatility diagnostic. |
| Rolling SDLV estimation | **Complete implementation gate.** Rolling 252-transition estimation, rank 16, aligned factors, eight Greek factors, Q drift, and compatible P drift pass remediation review. | `2026-10-12-rolling-dlv-phase4-remediation-review.md`; `2026-10-12-rolling-dlv-phase4-second-remediation-review.md`. | Dated production artifacts must be present for every GROMM origin. |
| GROMM market model and hedges | **Implementation complete; legacy rolling evidence complete; current migration incomplete.** The authenticated legacy affine range now has all 611 origins. The corrected migration separately authenticates the range source and the common dated-manifest source before it can claim a GPU. | `2026-10-12-gromm2-architecture-quant-review.md`; `2026-10-12-gromm-hedging-integration-remediation-review.md`; `2026-10-14-rolling-gromm2-physical-selection-review.md`; `2026-10-14-production-handoff-remediation.md`; complete range `gromm2-rolling-v3/range-fit-2023-2025-v5.json`. | Migrate and authenticate all 611 current-code artifacts, then complete every enabled daily and weekly GROMM policy. |
| GPU production infrastructure | **Mechanical gate complete; live production evidence incomplete.** Restart, source pinning, range authentication, and progress heartbeat paths exist. The legacy range completed successfully; two handoff defects were found and remediated before downstream production. | `2026-10-13-gromm-production-provenance-review.md`; `2026-10-13-evaluation-heartbeat-review.md`; `2026-10-14-production-handoff-remediation.md`; focused restart tests. | Successful current rolling migration, generated readiness matrix, production gates, and completed GPU cases. |
| Historical SANOS baseline | **Implementation complete; production incomplete.** Daily and weekly exact-cost solvers and OCE formulations pass review; active daily queues are partial. | `2026-10-12-historical-sanos-weekly-quant-review.md`; `2026-10-14-historical-weekly-oce-review.md`. | Complete every daily objective/parameter and both 528-origin weekly schedules; retain cap-limited recession cases only under their declared diagnostic gate. |
| Published generated baselines | **Implementation and training-lineage review complete; readiness and production incomplete.** VolGAN, AD-Seq-Vol, and AD-Seq-Vol-FT have distinct authenticated lineages. | `2026-10-12-volgan-diffusion-integration-remediation-review.md`; `2026-10-12-generated-rolling-warmstart-quant-review.md`; `2026-10-13-generated-gate-assembler-review.md`; generated training `pipeline-complete.json`. | Complete the replacement 18-scope GPU-readiness matrix, assemble all three production gates, and complete every enabled daily and weekly case. |
| Daily benchmark matrix | **Incomplete.** The canonical specification has 57 cases. Native queues, rolling GROMM, and generated cases are not all complete. | Versioned daily production matrix builder and validators; active immutable services listed below. | A 57-case matrix in which every case is ready, has a complete authenticated output, and reconstructs to the declared 12,220 records per policy. |
| Weekly benchmark matrix | **Incomplete.** The canonical specification has 94 cases and the engine passes mechanical review, but production has not started beyond readiness examples. | `2026-10-13-weekly-production-quant-review.md`; versioned weekly matrix and results validators. | A 94-case matrix in which every case is ready and has 528 origins and 6,864 authenticated target records. |
| Statistical reporting | **Implemented, final evidence incomplete.** Daily and weekly aggregators, overlap-aware inference, prices, drawdowns, and model catalogue are tested. | Report builder and summary tests; synthetic end-to-end PDF compile; matrix-composition adversarial tests. | Rebuild all statistics from the complete canonical matrices and pass an independent results review. |
| Final public report | **Incomplete.** The repository PDF is preliminary and predates full production. | Preliminary `reports/hedging_benchmark_summary.pdf`; active final successor described below. | Authenticated production PDF and manifest from the complete 57-case daily and 94-case weekly matrices, followed by repository refresh and final review. |

## Active Immutable Work

- The legacy affine range is complete for all 611 origins; its authenticated
  fit summary SHA-256 is
  `6cbdcb13913d322d95e1ff644950f638aaf090ea36278553d6eb898b2bce3205`.
- `quantdata-hedging-gromm2-migration-661d575.service` waits to recompute
  current dynamics and physical-selection evidence after authenticating both
  legacy source identities across all 611 manifests.
- `quantdata-hedging-generated-readiness-b434852.service` runs the corrected
  18-scope managed-GPU readiness matrix.
- The corrected covariance and Historical-SANOS daily services continue their
  restartable current-notional cohorts.
- `quantdata-hedging-weekly-native-parallel-a1baa69.service` waits for complete
  native daily prerequisites.
- `quantdata-hedging-covariance-power-656ca65.service` waits for existing
  covariance capacity, then runs the three reviewed Gaussian power cohorts.
- `quantdata-hedging-full-production-a1baa69.service` follows the corrected
  migration/readiness services and produces the original 54-case daily and
  94-case weekly canonical outputs.
- `quantdata-hedging-final-57-656ca65.service` authenticates those outputs plus
  the three power cohorts, composes the required 57-case daily matrix, and
  renders the successor report.

## Final Acceptance Checklist

The goal is complete only after all of the following hold simultaneously:

1. The public dataset validator passes from an otherwise empty data root.
2. The rolling GROMM current-code range contains exactly 611 authenticated
   origins and passes its physical-selection gate.
3. Every one of the 57 daily cases is ready and reconstructs to exactly 12,220
   current-notional records with its declared convergence/cap policy.
4. Every one of the 94 weekly cases is ready and reconstructs to exactly 528
   origins and 6,864 records with frozen and restarted schedules distinct.
5. The final report builder authenticates both matrix hashes and regenerates
   its tables, appendix, TeX, PDF, and manifest from the dated records.
6. The complete hedging test suite passes on the final source commit.
7. An independent quantitative results review reconciles headline numbers,
   objectives, costs, caps, pricing decomposition, and overlapping inference.
8. The final PDF and its provenance are copied to the repository, committed,
   and pushed without modifying unrelated user work.
