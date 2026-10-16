# Historical-SANOS v1.22 final solver review

Date: 2026-10-16

## Verdict

**PASS.** The focused v1.22 patch has no correctness or safety blocker. The
turnover-row normalization is algebraically objective-equivalent, its dual
mapping back to the physical certificate is correct, position-cap semantics
are unchanged, and acceptance still requires the independent `1e-6`
certificate. The exact real-data replay of the original failure case certifies
all four policies with substantial tolerance headroom using CLARABEL directly.

The replay log ends with a traceback only because the diagnostic wrapper
deliberately stops after recording the first target. It validates the original
failure case rather than a complete multi-target weekly run.

## Reviewed state

- Base commit: `7420c4788a8b3ae9b4a6a76074cba69fd8c07597`.
- Focused diff SHA-256: `6862d4682ede33cd032a1a8b9b09911b86b29fc51b4677b2e836ac907020538f`.
- `config.py` SHA-256: `cb5260cf111adbcae02d2861551510e6d24882c29ca2081df307b6e0bd2aab5b`.
- `weekly_hedger.py` SHA-256: `69a505ced310d129cedd8f5f0ad58a9e9dc31a0fe81b7ad18542e5c1c8a093a2`.
- Weekly test SHA-256: `b5391f32e045abd30f9668e2d892334f79487b4788f031673779cfe715bc9e12`.
- Replay log SHA-256: `0aec60a295b912818fd4632862997036240c49d04e879b0fc320d7133184bb52`.
- Trace JSON SHA-256: `41169fa54d6bfda8cf57f775f1b1c0b2fbd74b8dd530f2ba6b7058d5d6d249a6`.
- Persisted independent review SHA-256: `9b1c17aee448bbfa1cb84c871d0f11ae21fbf032ccb4cc759fbe1a82ae40d28c`.
- Focused `git diff --check`: clean.

The repository had unrelated dirty files. The independent review covered only
the Historical-SANOS configuration, weekly hedger, weekly tests, and the exact
replay evidence.

## Exact objective equivalence

Let optimizer positions be `x`, physical positions be `D x`, and physical
turnover row `r_i(x) = A_i x - b_i`. The patch defines

`s_i = max(1, max_j |A_ij|, |b_i|)`

and replaces the physical epigraph `t_i >= |r_i|` by a normalized epigraph
`q_i >= |r_i| / s_i`, with objective coefficient `w_i s_i`. Therefore
`(w_i s_i) q_i = w_i t_i` under `t_i = s_i q_i`. The quadratic, linear, and
constant terms are unchanged. The canonicalizer explicitly sets
`q_i = |r_i| / s_i`, including zero-cost rows for which the solver need not
select the minimum epigraph.

The objective regression includes nonzero turnover offsets and compares the
CVXPY objective with the independently constructed physical compact objective
to `1e-12` absolute tolerance.

## Dual mapping and certificate

For normalized positive and negative turnover constraints, the solver duals
are divided by `s_i` before the physical certificate is evaluated. This maps
normalized epigraph stationarity back to the original physical condition and
preserves complementarity. Position-bound rows are not normalized, so their
solver duals remain unchanged in scaled-position coordinates.

- `certificate_tolerance` remains exactly `1e-6`; solver tolerance remains
  `1e-8`.
- Acceptance still requires an admissible solver status, physical-objective
  agreement, and every independent KKT metric at or below `1e-6`.
- The primal certificate checks turnover in physical units and positions
  against the physical `solver_position_cap`.
- Optimizer bounds remain exactly equivalent to the original physical bounds.
- `position_cap` and `solver_position_cap` semantics are unchanged.
- OSQP and HiGHS fallbacks pass through the same independent certificate.

CLARABEL is primary for the compact mean-variance problem, followed by OSQP and
HiGHS. Each returned point is canonicalized and independently certified.
Statuses such as `optimal_inaccurate` or `user_limit` cannot succeed without
passing the independent KKT, gap, physical-primal, and objective checks.

Best-candidate retention is fail-safe: it is used only after all attempts remain
uncertified, preserves `success=False`, and prevents a later catastrophic
fallback from overwriting a materially better diagnostic point.

## Real replay evidence

The replay covers the previously failing `digital-put-5d-atm` target with
4,096 scenarios, 207 contracts, and 669 policy parameters.

- `hedge_converged=true`.
- Long, short, and zero policies each converge in 29 combined selection/full
  solver iterations; pure converges in 24.
- Worst pricing-policy optimality residual: `3.99123086408672e-9`.
- Worst physical constraint violation: `1.1641532182693481e-10`.
- Worst pricing-policy objective error: `4.802132025361061e-9`.
- CLARABEL reports `optimal` for every selection and full fit; no fallback or
  active-set polish is needed.
- Maximum pricing-policy position is `999999.9997675265`, below the physical
  `1,000,000` solver cap.
- Internal pricing policies record 750 cap hits, while the executable pure hedge
  has zero cap hits; speculative pricing positions cancel in the executable
  hedge.

The absolute primal-dual gaps on the large pricing objectives are normalized by
the certificate against objective magnitude. The resulting residual is about
`4e-9`, well below `1e-6`; no tolerance was weakened.

## Tests

The focused mean-variance tests passed: `15 passed, 35 deselected`.
The complete weekly module passed: `50 passed`.

Coverage includes objective equivalence, nonzero offsets, extreme scaling,
zero-cost rows, the real 669-parameter shape, boundary-cap certification,
CLARABEL-to-OSQP fallback, solver-exception fallback, active-set behavior,
rejection of feasible non-KKT points, and best-uncertified-candidate retention.

## Recommendation

The focused v1.22 patch is suitable to commit. The complete frozen and restarted
weekly target suites must still be rerun before the broader generated-readiness
suite can be considered repaired.
