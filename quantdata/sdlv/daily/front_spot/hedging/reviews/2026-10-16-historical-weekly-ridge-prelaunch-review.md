# Historical-SANOS weekly ridge pre-launch review

Date: 2026-10-16

Scope: independent quantitative review of the pre-outcome weekly ridge
selection, null-policy fallback, numerical certificates, and daily recession
diagnostics before the 13 production calibration scopes are run.

## Verdict

**GO for production calibration from a clean immutable worktree.**

## Required invariants confirmed

- Each objective/risk-parameter scope selects separately from the frozen ridge
  grid plus the exact zero-position baseline.
- Null may win the one-standard-error rule only after at least one finite-ridge
  candidate passes every numerical gate.
- Candidate regret is `CE(no action) - CE(policy)` and gives equal aggregate
  weight to pricing, shared-zero, and pure-execution strata.
- Both the held-out fit and full-window refit are certified for all 40 policy
  components; the artifact records 80 split/final component gates.
- Internal solver caps and the precommitted economic position cap of 1,000 are
  checked independently. Pure held-out and final policies must remain inside
  the economic cap.
- The raw 252 origins, scenario origins, fit origins, validation suffix, and
  five chronological validation blocks are stored and authenticated.
- Selections and bundles bind the public dataset, target notionals, clean Git
  source commit, schema, complete ridge grid, common calibration design, and
  frozen/restarted weekly schedule family.
- Positive-ridge unregularized mean-volatility recession remains a visible
  warning; a true configured-objective recession remains fatal in both daily
  and weekly numerical gates.

## Validation

The complete hedging test package passed 475 tests before the source-freeze
commit. A second independent review must inspect the exact 13 output artifacts;
the final bundle binds the SHA-256 of that output review rather than this
pre-launch review.
