# Historical-SANOS v1.29 out-of-cap review

Date: 2026-10-16  
Verdict: **GO**

## Defect

For low-risk mean-volatility cells, CLARABEL could return a valid unconstrained
optimum outside the model's declared internal solver box. The implementation
only rebuilt the bounded SOCP when CLARABEL reported an unbounded status, so an
otherwise valid out-of-box optimum was rejected and the independent fallback
could collapse to a zero, non-converged result.

## Repair

Historical-SANOS v1.29 now resolves the identical convex objective with the
declared box constraints whenever the unconstrained CLARABEL action is
non-finite or exceeds `solver_position_cap`. The existing adaptive cap expansion
and independent numerical certificates remain unchanged. Solver traces record
the bounded re-solve and iteration accounting includes both CLARABEL solves.

## Real-data replay

The previously failing 2024-01-12 daily origin was replayed under the strict
runner for all 20 targets. All 20 targets converged with:

- maximum KKT residual `9.52e-9`;
- zero internal, fit, executed, or total position-cap hits;
- exact zero cap-stability L2, Linf, and objective errors;
- pure-hedge gross exposure from `2.24` to `7.95`; and
- maximum executable coordinate `0.3067`.

The claim and zero virtual policies each have gross exposure near `3.216e7`,
but their certified difference is the small, stable executable pure hedge. The
positive-ridge unregularized recession warning is diagnostic only: the
ridge-regularized optimization is coercive and all fatal recession flags are
zero.

Evidence record:
`/tmp/quant_data/hedging/historical-v129-strict-20240112/records/2024-01-12.json`
with SHA-256
`e8486900575f66def12d73b452064eb94d17b1edaa14dd35e84fc8d600da39e0`.

## Validation

- Quantitative independent review: **GO**.
- Focused model review: `40 passed`.
- Historical-SANOS and daily-runner suite: `128 passed`.
- Full hedging test tree: `488 passed` with 19 expected warnings.
- `compileall`, Ruff import checks, and `git diff --check`: pass.

Production must run from the exact committed v1.29 source. Any compatibility
transition for the previously calibrated weekly ridge bundle must target that
exact commit and authenticate its own manifest and semantic replay.
