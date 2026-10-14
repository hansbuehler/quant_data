# Historical SANOS Weekly OCE Quant Review

## Verdict

**PASS.** No quantitative or implementation blockers remain in commit
`edf28cc`.

## Scope

- Exact weekly entropy and power-exponential OCE formulations.
- Scenario-dependent spot and option turnover costs.
- Cached wealth geometry versus the common pathwise accounting engine.
- Power-exponential cash translation and conic objective equivalence.
- Numerical convergence, position-cap gates, and cache-version identity.

## Evidence

- Cached terminal wealth matches `weekly_scenario_accounting` path by path to
  absolute and relative tolerance `1e-14`, including nonzero carried positions.
- The entropy cone is the exact log-sum-exp objective of terminal wealth net of
  scenario-dependent costs, plus the declared quadratic fitting ridge.
- The power cone exactly represents both branches of the power-exponential loss
  and its optimized cash translation. Its conic value matches the direct
  objective, and the cash-shift identity holds.
- A CLARABEL result labelled `optimal_inaccurate` is never accepted. The power
  solver makes at most one cold retry at five times the configured tolerance,
  floored at `5e-8`, and still requires `optimal` status with no solver-cap hit.
- `HistoricalSanosConfig` is versioned `1.10`, preventing reuse of artifacts
  produced under the superseded weekly objective semantics.
- The focused weekly suite passes 25 tests. The complete hedging suite passes
  349 tests; compilation, Ruff import checks, and `git diff --check` pass.
- Real-data frozen-weekly first-origin gates cover all 13 targets. Entropy at
  risk parameter 10 and power-exponential at risk parameter 1 both finish with
  zero cap hits and zero nonconverged fits.
