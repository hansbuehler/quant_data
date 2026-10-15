# Historical-SANOS v1.19 real-replay numerical review

Date: 2026-10-16

Reviewed base commit: `0228913`

Reviewed remediation: `HistoricalSanosConfig` v1.19 in the shared worktree

## Scope

The 4,096-scenario by 669-parameter VolGAN frozen-weekly replay exposed two
separate questions: whether the compact mean-variance QP can be solved and
certified in physical units, and whether the resulting historical-P pricing
policies are economically acceptable.  Version 1.19 resolves the numerical
question without changing the objective.  The production cap gate then exposes
a genuine economic/model failure in the pricing policies.

## Numerical remediation

The exact objective remains

\[
  \tfrac12 x^\top Hx+c^\top x
  +\sum_i w_i\lvert A_i x-d_i\rvert
\]

with the same scenarios, claim, mean-variance risk aversion, transaction costs,
terminal liquidation, quadratic ridge, and physical position caps.

The remediation does the following.

1. It retains the physical turnover epigraph.  Algebraically exact row
   normalization was tested and rejected because solver-scale feasibility
   errors mapped back to physical turnover violations of `9.3351` and
   `9.77796`, far above the unchanged `1e-6` physical certificate tolerance.
   Moving the scale into the epigraph weights therefore did not solve the
   numerical problem.
2. It canonicalizes the auxiliary epigraph at every candidate, then applies an
   iterative active-set KKT solve to the unchanged physical QP.  The KKT system
   uses sparse linear algebra and removes only graph-incidence equalities that
   are linearly redundant.  Zero-cost turnover rows are never imposed as false
   equalities.
3. It independently recovers and checks a dual candidate and retains the full
   physical primal, dual-feasibility, stationarity, complementarity,
   primal-dual-gap, and objective-reconstruction gates.  No tolerance is
   relaxed.
4. It uses CLARABEL as an independently bounded compact-QP fallback before
   HiGHS.  `OPTIMAL_INACCURATE` is accepted only when the independent full
   certificate passes.  A `USER_LIMIT` candidate is likewise accepted only
   after the same certificate; otherwise the fallback path continues.
5. It disables OSQP workspace warm starts while retaining the DPP-compiled
   problem.  With warm starts, a reused pricing context returned spot positions
   `[0.01175134, 0.01180331, 0.01176548]`, while a fresh context returned
   `[0.00827462, 0.00827462, 0.00827462]`.  Their exact objectives differed by
   `1.23594e-8`; the reused solve had fallen through to CLARABEL with a
   `6.15527e-8` gap.  Cold OSQP starts remove this target-order dependence, and
   the pre-existing shared-versus-fresh regression now agrees at `1e-8`.

## Real replay evidence

The exact replay is recorded in
`/tmp/quant_data/historical-sanos-v119-physical-replay/replay.log`; its strict
gate artifact is
`/tmp/quant_data/historical-sanos-v119-physical-replay/output/numerical-gate-failure.json`.

The raw OSQP candidates reproduce the original failures:

- 3,277-scenario selection: primal violation `6.07747e-5`, stationarity
  `8.56472e-7`, and gap `-2.77725e-4`;
- 4,096-scenario final fit: primal violation `2.20288e-5`, stationarity
  `1.62010e-6`, and gap `-2.48453e-4`.

The unchanged-objective KKT recovery then certifies:

- 3,277 scenarios: physical primal violation `0`, stationarity `1.28652e-8`,
  and gap `1.87307e-11`;
- 4,096 scenarios: physical primal violation `0`, stationarity
  `3.47454e-13`, and gap `1.80836e-11`.

All thirteen targets completed their numerical solves.  The origin-level gate
then correctly failed with `12,090` internal fit-cap hits, zero executed cap
hits, thirteen nonconverged policy labels, and no recession-direction flags.

## Cap decomposition

The exact first-target trace is recorded in
`/tmp/quant_data/historical-sanos-v119-cap-trace.json` and
`/tmp/quant_data/historical-sanos-v119-cap-trace.log`.

For `digital-put-5d-atm`, each of the long, short, and zero pricing policies has
exactly 310 positions at the internal solver cap of `-1e6`; the pure hedge has
zero cap hits and a maximum absolute position of zero.  Thus each target has
`3 * 310 = 930` internal hits and the full book has `13 * 930 = 12,090`.
There are no spot cap hits and no positive cap hits.  The negative option hits
are:

- step 0: 122 puts -- 17 expiring 2023-01-18, 22 on 2023-01-19, 25 on
  2023-01-20, and 29 each on 2023-01-23 and 2023-01-24;
- step 1: 105 puts -- the same book excluding the expired 2023-01-18 puts;
- step 2: 83 puts -- the same book excluding the expired 2023-01-19 puts;
- steps 3 and 4: no cap hits.

The three pricing solutions have the same capped coordinates, training
objective `-245881.4310373277`, certainty equivalent `303443.3671462088`, zero
physical constraint violation, objective reconstruction error
`7.85803e-10`, and certified optimality `1.64838e-8`.  The pure hedge has zero
positions and zero objective.  The financial signal is therefore a common
historical-P speculative short-put direction that cancels in the
claim-minus-zero hedge.  It is not an active-set sign error.  The quadratic
ridge makes the formal problem coercive, so this is a genuine position-cap
signal rather than mathematical unboundedness, but the implied scale is
economically unacceptable.

The adapter currently folds `cap_hits > 0` into each policy's `converged`
field.  The runner consequently reports both `policy-fit-cap-hit` and
`nonconverged-fit` for the same certified solutions.  A subsequent adapter
change should separate numerical convergence from economic cap acceptability;
the cap gate itself must remain strict.

## Verification

- Focused mean-variance/shared-context tests: 19 passed.
- Complete weekly Historical-SANOS test module: 49 passed.
- Historical daily, weekly, and daily-runner suite: 85 passed.
- `python -m compileall -q` on the changed source and test modules passed.
- `ruff check --select I` on the changed source and test modules passed.
- `git diff --check` passed.

## Verdict

**PASS for the v1.19 numerical remediation.  BLOCKED for VolGAN historical-P
production readiness.**  The exact QP is now recovered and certified without
weakening its objective or certificate.  The remaining failure is the intended
economic cap gate: the model/scenario pricing policies exploit a large common
short-put drift, while the pure executed hedge stays uncapped.

Independent reviewer: QuantAgent (`historical_failure_quant_review`)
