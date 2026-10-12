# Independent quant review: CPU vertical slice

## Reviewed state

- Review date: 2026-10-12
- Base Git revision: `ed31c4b10ee29eba550b6000b20c5f2cc467513a`
- Source state: uncommitted hedging worktree on `main`
- Evidence supplied: 36 focused tests and both standalone CPU fixture commands
- Reviewer: independent `quant_dataset_review` agent

## Protocol findings

- **Blocked at review:** both smoke fixtures used a 2023 evaluation outcome.
  Disposition: corrected to a 29--30 December 2022 fixture. The covariance
  development smoke uses 200 historical transitions so it remains wholly in
  the initialization period; the production configuration remains 252.
- **Open for the weekly gate:** a stable origin cash-strike identity and later-
  date `K/F` remapping are not implemented. Current daily settlement is correct,
  but this finding blocks the frozen-weekly engine.
- **Non-blocking at review:** the covariance fitter received a complete current
  sample while reading only its spread. Disposition: corrected. Its public API
  now accepts only the current option half-spread, so the future outcome cannot
  enter the fit through that argument.

## Accounting findings

- **Passed:** normalized price units, option settlement, forward-adjusted spot
  gains, long/short signs, option execution sides, and the pricing-versus-
  hedging decomposition were internally consistent.
- **Blocked at review:** the one-day spot hedge paid only its opening one-basis-
  point cost. Disposition: corrected. Evaluation now charges one basis point at
  entry and one at terminal liquidation; the covariance lasso uses a predictable
  two-basis-point per-unit spot penalty.
- **Non-blocking at review:** the result field named `payoff` is an origin-PV
  amount. Disposition: model diagnostics and package documentation now state
  the unit explicitly. Renaming the common result field is deferred to a schema
  version change.

## SANOS-LV findings

- **Passed:** SANOS digital pricing and discounting, homogeneous spot-delta
  bump, and long-claim hedge sign.
- **Corrected edge case:** zero-smoothing call and put digitals previously both
  included an atom at the strike. They now use complementary `>` and `<=`
  conventions, with a regression test.
- **Open Phase-2 scope:** the current model is a static digital-price and spot-
  delta smoke. Eight rolling PCA-factor vegas, all daily targets, rolling daily
  execution, and option-vega actions remain unimplemented.

## Covariance findings

- **Passed:** the long-claim minimum-variance sign and coordinate-descent cost-
  lasso equations in the two-dimensional smoke.
- **Corrected reporting:** the fitted objective now includes one half of the
  robust claim variance instead of reporting the reduced objective without its
  constant term.
- **Blocking Phase 3:** the current shrinkage rule becomes one when dimension
  approaches the 252-observation window, eliminating cross-covariances and
  producing a zero hedge.
- **Blocking Phase 3:** the current history aligns only spot and a rolling ATM
  one-day relative contract, not the complete current fixed-contract universe.
  Mean-volatility and drift-aware variants also remain outstanding.

## Post-fix evidence

- 37 focused hedging tests pass.
- Both standalone smokes pass committed 2022-only tolerance fixtures.
- Import sorting, byte compilation, and `git diff --check` pass.

## Re-review disposition for Phases 0--1

The reviewer subsequently re-ran the corrected slice and gave both the Phase-0
protocol gate and Phase-1 accounting gate a **PASS**. The final evidence at that
gate was 39 focused tests plus both standalone smokes. The remaining exact-cost
refinement was to replace the covariance smoke's flat predictable two-basis-
point spot penalty by the origin-known coefficient
`1bp * [1 + DF*F/S]`; this is non-blocking for the already exact realized
accounting.

The historical findings above describe the state at the original review. The
later SANOS-LV factor-vega implementation and synthetic missing-expiry replay
require the separate Phase-2 model review and are not retroactively approved by
this document.

## Disposition

The corrected CPU daily accounting slice is suitable for continued
development. This review does **not** approve Phase 2 or Phase 3 production
runs. Stable weekly contract replay, the full SANOS-LV factor-vega baseline,
and a non-degenerate high-dimensional covariance estimator require their own
completed implementation and re-review.
