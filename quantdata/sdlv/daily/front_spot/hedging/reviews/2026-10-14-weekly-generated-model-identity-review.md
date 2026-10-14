# Weekly generated-model identity review

Date: 2026-10-14

## Scope

This independent review covers the distinction between the canonical production
matrix identities `diffusion-ad-seq-vol` and `diffusion-ad-seq-vol-ft` and the
runtime identities persisted by the weekly adapters, `ad-seq-vol` and
`ad-seq-vol-ft`. It covers weekly evaluation, production completion validation,
strict numerical-gate failure authentication, weekly summary aggregation, and
final report rendering.

## Result

**PASS.** No remaining discrepancy was found.

- The mapping is restricted to the two diffusion matrix identities and is
  idempotent for runtime identities and all other models.
- Weekly run and record identities remain bound to the actual runtime adapter;
  they are not relabelled after evaluation.
- Completion validation independently requires both the run specification and
  the reconstructed aggregate to match the mapped runtime identity.
- Failure-ledger validation maps only the weekly source-matrix identity while
  retaining exact checks on gate model, run identity, measure, schedule,
  dataset, strict-gate status, and failed origin. Daily validation is unchanged.
- Weekly reporting authenticates the runtime identity but exports the canonical
  matrix identity, so the final model catalogue accepts base and fine-tuned
  diffusion policies as distinct models.

## Verification

The reviewer ran 77 focused tests spanning weekly execution, production
validation, failure handling, matrix composition, weekly aggregation, and final
document rendering. All passed. Ruff import-order checks and `git diff --check`
also passed. The primary agent separately ran both complete hedging test trees:
406 SDLV hedging tests and 58 shared hedging-protocol tests passed, for 464 tests
in total.
