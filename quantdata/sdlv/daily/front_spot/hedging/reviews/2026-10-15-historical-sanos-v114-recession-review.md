# Historical-SANOS v1.14 weekly recession review

## Scope

This review covers the frozen-weekly Historical-SANOS mean-volatility fit at
risk parameter 1 for origin 2023-01-20. The version-1.13 strict run reported 39
internal fit-cap hits, zero executed cap hits, and 13 nonconverged target fits.

## Independent quantitative diagnosis

The failure is an empirical recession direction of the unregularized economic
objective, not a CLARABEL status, scaling, or canonicalization defect. The
shared zero-claim policy gives the decomposition

- mean terminal wealth: 567.144057;
- volatility penalty: 298.081457;
- deterministic transaction cost: 84.271503;
- unregularized homogeneous objective: -184.791097; and
- standardized quadratic ridge: 91.985538.

The ridge therefore makes the numerical surrogate finite at -92.805559, but it
does not remove the negative direction from the positively homogeneous
zero-claim mean-volatility objective. Validation reverses to 80.698335 versus
-92.805559 on the final training window, consistent with speculative overfit in
a statistically thin fit with 232 paths and 755 policy parameters. The centered
pure hedge remains numerically well behaved, with no cap hits and gross position
94.17, so silently substituting it would change the precommitted pricing
objective rather than repair the fitted zero policy.

## Required treatment

Preserve the strict failure. Do not loosen solver tolerances, raise the cap,
strengthen the ridge until the fit appears finite, or replace the pricing policy
by the centered pure hedge. For a zero-claim weekly mean-volatility fit, compute
the exact objective at the certified optimizer, subtract the standardized ridge,
and mark a recession direction when the remainder is below the negative
certificate tolerance. Recession directions must force nonconvergence and take
priority in the machine-readable failure classification.

## Implemented evidence

Historical-SANOS version 1.14 implements that certificate and propagates it
through weekly fit records and strict numerical-gate artifacts. The real-data
replay at 2023-01-20 reproduces all 13 target failures and emits
`mean-volatility-recession-direction` before the accompanying fit-cap and
nonconvergence classifications. Focused model, runner, production-matrix, and
failure-ledger verification passes 100 tests.
