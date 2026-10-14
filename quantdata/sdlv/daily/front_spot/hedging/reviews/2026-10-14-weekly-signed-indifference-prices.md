# Weekly Signed Indifference-Price Remediation

## Verdict

**PASS.** Weekly common market prices and risk-neutral expectations remain
finite and non-negative. Buyer, seller, and continuation indifference values
may be signed, and the common accounting engine now carries a finite signed
model price without changing executable market cashflows or transaction costs.

## Production defect

Generated-model readiness completed the VolGAN daily public-settings gate and
then evaluated ten weekly digital targets. It stopped on the realized-variance
target because `WeeklyPriceRecord` required every price-like field to be
non-negative.

That invariant is too strong. Mean--variance and mean--volatility certainty
equivalents are not monotone: adding a non-negative but sufficiently risky
payoff can reduce the optimized certainty equivalent, producing a negative
buyer indifference value. The continuation midpoint can therefore also be
negative. This does not make the common market price or a risk-neutral
expectation negative.

## Correction

- `WeeklyPriceRecord` requires common market prices and Q expectations to be
  finite and non-negative.
- Buyer, seller, and continuation indifference values are required to be
  finite but may be signed.
- `evaluate_weekly_hedge` retains non-negative common market prices and spot
  costs, but permits a finite signed accounting model price.
- The pricing contribution and pure-hedge error still reconcile exactly to
  economic P&L; executable cashflows continue to use the common market price.

## Evidence

- A regression passes a negative buyer indifference value through the complete
  `run_weekly_model` path and verifies the signed model price, pricing
  contribution, and P&L reconciliation.
- Separate regressions retain rejection of negative common market prices and Q
  expectations.
- The focused weekly benchmark, runner, results, and summary suite passes 44
  tests. Compilation, Ruff import checks, and `git diff --check` pass.
- Independent quant re-review confirms the signed-value semantics, downstream
  accounting compatibility, and absence of conflicting reporting assumptions.

The failed generated-readiness run remains diagnostic. A fresh immutable-source
readiness cohort is required before generated-model production can launch.
