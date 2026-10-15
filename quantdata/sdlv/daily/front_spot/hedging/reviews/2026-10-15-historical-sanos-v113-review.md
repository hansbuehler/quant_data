# Historical SANOS version 1.13 failure review

## Scope

This review classifies the strict version-1.12 production failures without
changing the 252 uniformly weighted scenarios, transaction costs, economic
objectives, risk parameters, position caps, or declared solver tolerances.

## Findings

- Daily mean-volatility at risk parameter 0.1 has a genuine recession
  direction. The fitted zero policy has an unregularized homogeneous objective
  of approximately -182.98. Its roughly 3.87 million gross virtual exposure is
  made finite only by the declared ridge, while its validation loss reverses
  from approximately -91.49 in sample to +888.41. This is an economic
  conditioning failure, not a status-code failure.
- Frozen-weekly mean-volatility at risk parameter 0.1 has the same pathology.
  Each target's long, short, and zero virtual pricing policy reaches the
  one-million internal cap once, accounting for all 39 fit-cap hits. The pure
  executable hedge itself has no cap hit.
- Weekly mean-variance failures were certificate-bookkeeping defects. The raw
  OSQP point was sparsified and the modified exact objective was compared with
  the stale raw solver objective. All inspected solves had status `optimal`,
  zero primal-constraint violation, and objective differences only marginally
  above the inappropriate comparison limit.
- Daily mean-volatility risk parameters 2 and 3 exposed a separate numerical
  issue: the conic objective reconstructed exactly, but active-coordinate KKT
  residuals remained too large for the declared certificate.

## Remediation

Version 1.13 makes four narrow changes.

1. It records a direct recession certificate whenever a zero-claim
   mean-volatility policy exhibits a negative unregularized homogeneous
   objective. The policy remains nonconverged even if the ridge surrogate has a
   finite optimizer.
2. It removes standardized solver noise at inactive daily conic coordinates
   before evaluating the signed-action KKT conditions.
3. If a non-recession daily mean-volatility conic point still misses the KKT
   gate, it receives an exact-cost signed-action refinement of the unchanged
   objective and tolerance.
4. Weekly mean-variance certifies and returns the raw convex solution. A failed
   OSQP certificate triggers a CLARABEL solution of the identical problem; no
   tolerance is widened and no modified point is compared with a stale
   objective.

## Evidence

The focused daily and weekly Historical-SANOS test suite passes 58 tests. On
the public-candidate real-data panel at origin 2023-01-17, daily
mean-volatility risk parameters 2 and 3 each converge for all 20 targets with
zero cap hits. Risk parameter 0.1 remains nonconverged for all 20 targets, with
the zero-policy recession flag set for every target. At weekly origin
2023-01-20, mean-variance risk parameters 1 and 10 each converge for all 13
targets with zero fitted or executed cap hits. These results preserve rather
than suppress the economically genuine failure.

## Independent quantitative review

The independent QuantAgent review classified the low-risk mean-volatility
cases as genuine recession-direction failures and the weekly mean-variance
cases as a raw-versus-canonical certificate bug. It recommended the same
separation implemented above: retain recession cells as explicit failures,
certify raw convex points, and replay only numerically affected cells.
