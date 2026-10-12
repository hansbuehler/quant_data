# Covariance DLV-return correction review

## Scope

- released DLV price-array exposure through `PublicMarketDay`
- covariance option and digital returns reconstructed from released DLV state
- common SANOS current execution and realized accounting
- covariance configuration, target-notional cache identity, and focused tests

## Disposition

**PASS -- no remaining blockers.**

The financial correction itself passes review. Historical coordinates retain
their initial business DTE, NMS, and option side; each historical origin maps
that coordinate to a fixed strike and follows it for one session. Origin and
remaining-maturity option values use the released DLV state, while one-day
settlement uses the realized normalized spot. Digital returns use the analogous
DLV probabilities. Discounting and forward/spot normalization are consistent
with discounted origin-PV units.

The current executable hedge universe, marks, spreads, realized settlement, and
unfinished-claim accounting continue to come from the common SANOS market
sample. Altering SANOS historical state leaves the reconstructed training
returns unchanged, while altering DLV state changes both option and digital
returns.

## Resolved cache-version issue

The covariance return geometry now has the explicit semantic identity
`released-dlv-state-v1`. The target-notional artifact schema is version 3,
`TargetNotionalConfig` is version 1.1, and the cache header and path include the
return-geometry identity. Existing pre-correction artifacts therefore cannot be
selected by the corrected implementation. A regression changes the semantic
identity and confirms that the artifact path changes.

## Nonblocking findings

1. Add an explicit regression that current sample prices/accounting remain
   SANOS-based when DLV state is perturbed; the implementation is clear, while
   the current sensitivity test concentrates on historical returns.
2. The final focused CPU suite passed 22 tests. Import-order lint and
   `git diff --check` passed.
