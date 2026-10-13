# Generated weekly runtime review

Date: 2026-10-13

## Verdict

The frozen-weekly VolGAN bottleneck is caused by target-level repetition, not
by a necessary difference in the 13 target models.  For a given origin, all 13
targets currently regenerate the same seeded paths, rebuild and retain the
same fixed-contract scenario geometry, and independently solve the same
zero-claim pricing problem.  Diffusion has the identical structure.

The narrowest exact optimization is to share one immutable generated market
scenario core per fit decision, attach only the target payoff vector for each
claim, and reuse one zero-claim fit.  This changes neither the information set
nor any market, policy, target, seed, price, or action identity.  Target
parallelism should be a second step, after sequential equivalence is proved.

No source code was changed in this review.

## Implementation follow-up

The exact P0 optimization was implemented and independently re-reviewed on
2026-10-13.  The follow-up review passed without blockers: indifference prices
remain the unmodified certainty-equivalent differences; the shared zero policy
is fitted cold on a dedicated geometry; cache identity covers checkpoint,
configuration, seed, dated decision state, and generator live state; and
concurrent reuse of mutable solver geometry is rejected.  Target-order,
cache-invalidation, origin-release, and both-generator regressions pass.

Validation at review time comprised 27 focused tests, 97 weekly tests, and 305
tests in the complete hedging suite, together with successful syntax, Ruff
import-order, and diff checks.

## Current call graph and duplicated work

`weekly_book_function` builds one episode and then calls `run_weekly_model`
separately for every one of the 13 claims.  The existing market-fit cache is
correctly shared across targets, because `WeeklyFitInstruction.market_fit_identity`
does not contain the target.  The policy identity intentionally does contain
the target, so all 13 policy fits remain distinct.

For each generated-model policy fit, both `VolganWeeklyAdapter.fit_policy` and
`DiffusionWeeklyAdapter.fit_policy` currently perform the following work:

1. sample the remaining path recursively with the same path count and seed;
2. call `build_generated_weekly_scenarios` to reprice the same frozen option
   book along those paths;
3. change only the terminal claim payoff and target label;
4. call `fit_weekly_open_loop_hedge`.

The path sampler is target independent.  In
`build_generated_weekly_scenarios`, spot paths, option marks, spreads, option
gains, alive/settlement masks, and spot gains are also target independent.
Only `terminal_claim_values_pv` and `target_label` depend on the claim.

`fit_weekly_open_loop_hedge` solves four policies: long claim, short claim,
zero claim, and the centered-gain pure hedge.  Each policy is solved first on
the training split and then on all scenarios.  The zero policy always uses
zero initial holdings and a zero signed claim.  It is therefore identical for
all targets, including when claim quantities differ.  For restarted weekly,
the execution pure hedge depends on each target's previous holdings, but the
pricing zero policy still does not.

### Exact work counts

For one frozen-weekly origin, the current implementation performs:

- 13 identical generated-path calls;
- 13 identical fixed-book repricings;
- 104 optimizer solves: 13 targets times four policies times two fits.

After sequential reuse, it should perform:

- one generated-path call;
- one fixed-book repricing;
- 80 optimizer solves: one zero policy times two fits, plus 13 targets times
  three target-specific policies times two fits.

For one restarted-weekly origin, the corresponding counts fall from 65 to
five path calls and book repricings, and from 520 to 400 optimizer solves.
There must be one shared core per decision date; future decision states must
not be prepared at the episode origin.

## Prioritized implementation design

### P0: Split generated market geometry from target payoff

Introduce an immutable `GeneratedWeeklyScenarioCore` produced once from
`GeneratedWeeklyPaths` and `WeeklyDecisionState`.  It should own:

- the current-plus-future inception-forward paths;
- the full inception-to-terminal target paths, or equivalently the immutable
  realized prefix plus future suffixes;
- spot present values and step gains;
- option step gains and half-spreads;
- option alive and settlement masks;
- contract IDs, fit date, model-fit identity, measure, and seed;
- the terminal discount and normalization scale needed to value a claim.

Then attach one claim by evaluating the existing payoff function path by path
and returning the existing `WeeklyScenarioBatch`.  The large core arrays must
be referenced, not copied.  Keep `build_generated_weekly_scenarios` as a thin
compatibility wrapper around “build core, attach claim” so existing callers
and unit fixtures continue to work.

This split must not vectorize or rewrite payoff definitions in the first
change.  Calling the existing claim payoff on the shared full paths preserves
the current barrier, realized-variance, and digital semantics exactly.

### P0: Make sharing explicit and origin scoped

Add an explicit runner-owned generated-work cache whose lifetime is exactly
one weekly origin.  Do not use module globals and do not leave large arrays in
the adapter or the eight-entry cross-origin market-fit cache.

The cache should distinguish at least:

- market-fit identity and immutable model checkpoint fingerprint;
- episode origin and decision index;
- remaining step count;
- frozen contract IDs;
- scenario count and seed;
- measure and policy configuration identity.

For frozen weekly there is one entry.  For restarted weekly there are five,
created only when their decision is reached.  The runner may continue to
process targets in its current order initially; the generated adapters should
obtain the shared paths/core from this explicit cache and attach their own
claim vector.

The same seed is already reset for every target, so the intended current
construction uses common random numbers.  Reusing the first generated array
therefore preserves rather than changes the sampling design.  Before rollout,
the production device must pass a repeat-sampling equality test for both
generators.

### P0: Reuse only the zero-claim pricing fit

Refactor `fit_weekly_open_loop_hedge` so a validated, previously fitted
zero-policy result can be supplied.  If absent, retain the present behavior.
The reusable zero object must be bound to the exact scenario core, hedge
configuration, scenario row order, and train/validation split.

Reuse is valid because price geometry always starts from zero spot and option
holdings and the signed claim is exactly zero.  It remains valid for restarted
weekly even though each target's pure execution policy starts from different
previous holdings.  Do not reuse any of the following:

- long-claim or short-claim price policies;
- the pure hedge;
- target payoff vectors;
- target-specific convergence and cap accounting other than referencing the
  same zero-policy result in each target's aggregate diagnostics.

Each target must retain its existing `HistoricalWeeklyHedgeFit`, buyer/seller
price calculation, convergence flag, cap count, and target-specific
`policy_fit_identity`.  Shared numerical work is not a shared policy identity.

### P1: Add bounded target parallelism only after equivalence

Once sampling, book repricing, and the zero fit have completed, the remaining
13 target jobs are independent for frozen weekly.  A bounded thread pool is
the narrowest safe approach because the target jobs can read the same NumPy
arrays without copying and no CUDA operation remains in the worker section.

Requirements:

- make shared arrays immutable before dispatch;
- compute the zero fit once before dispatch;
- give every target its own scenario wrapper, optimizer objects, result slot,
  and exception boundary;
- collect results in the canonical target catalogue order, never completion
  order;
- prevent BLAS/OpenMP oversubscription and start with two to four workers;
- never fork a process after Torch has initialized CUDA;
- never let concurrent tasks write a common checkpoint or cache entry.

For restarted weekly, target parallelism requires decision-major orchestration:
at each realized decision, build that decision's core, fit all target policies
using their target-specific previous positions, record their actions, and only
then advance.  Implementing this is wider than the P0 reuse and should wait
until the frozen-weekly benchmark shows that it is necessary.

If thread-level solver reproducibility cannot be demonstrated, keep the
target solves sequential.  The P0 reuse is exact and captures the dominant
avoidable generated-path and fixed-book work without concurrency risk.

### P2: Prepare repeated optimizer matrices only if still necessary

The open-loop geometry repeatedly materializes the same instrument-gain and
turnover matrices for the training and full-sample fits.  These matrices can
later be prepared once per core and shared read-only.  Do not initially reuse
a mutable CVXPY problem or warm-start one target from another: that can make
the numerical result depend on target order.  This is a secondary optimization
after the path/core/zero changes are timed.

## Correctness constraints

1. **No look-ahead.** A shared core may use only the current
   `WeeklyDecisionState`, its realized path prefix, the dated market fit, and
   generated future suffixes.  Restarted decision `d+1` cannot be prepared
   while processing decision `d`.
2. **Fixed contract universe.** The core must retain the episode-inception
   contract IDs and the existing expiry/alive/settlement handling unchanged.
3. **Path-dependent claims.** Barrier and realized-variance payoffs must use
   the full realized-prefix-plus-generated-suffix path, not merely the
   remaining suffix.
4. **Pricing semantics.** Buyer and seller indifference prices remain the
   long-minus-zero and zero-minus-short certainty equivalents.  The executable
   hedge remains the centered-instrument-gain pure policy; it is not obtained
   by subtracting the zero policy's positions.
5. **Initial holdings.** Shared pricing fits start flat exactly as today.
   Target-specific pure fits retain their actual previous target holdings.
6. **Randomness.** Preserve scenario order, path count, seed, generator state,
   precision, and device.  Do not concatenate targets into the generator batch
   or consume additional random numbers.
7. **Identities and audit.** Market-fit identities, target policy-fit
   identities, target ordering, model checkpoint hashes, recorded prices,
   convergence flags, and cap counts remain unchanged.  The source commit is
   the provenance for the implementation optimization.
8. **Failure behavior.** A failure in one target must fail the complete origin
   as it does now; never persist a partially completed 13-target book.
9. **Memory lifetime.** The shared core is released after the origin is
   serialized.  Retaining it in the cross-origin model cache would replace the
   runtime problem with avoidable RAM growth.

## Expected speedup

Let `G` be one recursive generated-path call plus fixed-book repricing and let
`S` be the average cost of one optimizer solve.  Ignoring the cheap target
payoff attachment, frozen-weekly runtime changes approximately from

`13 G + 104 S`

to

`G + 80 S`.

Sequential reuse therefore gives a limiting speedup of 13 times when generated
scenario construction dominates, and about 1.3 times when optimization alone
dominates.  With four bounded target workers, the target-specific solver part
falls from 78 serial solves to roughly four batches of six solves, in addition
to the two shared zero solves.  Real speedup will be lower because the solves
are not equal-cost and CPU memory bandwidth is shared.

The nested path-by-contract spread construction in
`build_generated_weekly_scenarios` makes a large scenario-construction share
plausible.  Stage timers around sampling, fixed-book construction, payoff
attachment, zero fitting, and each target's three remaining policies should be
added before quoting a production estimate.  These timers and the existing
heartbeat must remain lightweight and flushed.

The same split also reduces the retained large scenario arrays from 13 copies
per frozen origin to one shared copy plus 13 terminal payoff vectors.

## Required tests

### Numerical equivalence

- For VolGAN and diffusion, fixed seed and fixed fit: two legacy sampling calls
  must agree on the production device; one shared call must match them.
- For every one of the 13 claims, legacy and shared-core scenario batches must
  agree exactly on all common arrays and on the terminal payoff vector.
- Cover a restarted state with a non-empty realized prefix, including already
  breached and not-yet-breached barrier paths.
- For all four objectives, compare legacy and reused fits for prices, action
  matrices, certainty equivalents, train/validation losses, convergence,
  gradients, iterations, and cap hits.  Require bitwise equality where the
  same serial solver call is retained and otherwise use a precommitted tight
  floating tolerance.
- Assert every existing market and policy fit identity and artifact fingerprint
  is unchanged.

### Work-count and cache-boundary tests

- Frozen 13-target book: one market fit, one path sample, one core build, one
  zero-policy fit, and 13 each of long, short, and pure fits.
- Restarted 13-target book: five market fits, five path samples, five core
  builds, five zero-policy fits, and 65 each of long, short, and pure fits.
- Changing any cache-key field must miss; permuting only target order must not
  rebuild the core or zero fit.
- Different episode origins or frozen contract IDs must never share a core.
- Shared arrays must share memory across target wrappers and must reject
  mutation.
- The origin-scoped cache must be empty or collectible after the book returns.

### No-look-ahead and concurrency tests

- Retain the hostile as-of panel tests and add a restarted shared-cache test
  whose panel raises if any later decision state is touched early.
- Run the parallel target book repeatedly and compare it with the sequential
  shared implementation, including canonical output order and exception
  propagation.
- Instrument concurrent execution to prove one zero fit and no simultaneous
  writes to a shared result or checkpoint.
- Run with one and multiple target workers while forcing numerical library
  thread counts to one; results must meet the same equivalence gate.

## Blockers and implementation sequence

The current code has three engineering blockers to reuse: target payoff and
market geometry are coupled in `WeeklyScenarioBatch`; the weekly adapter
protocol has no explicit origin-scoped work object; and the zero fit is hidden
inside `fit_weekly_open_loop_hedge`.  There is no quantitative-model blocker.

Recommended sequence:

1. add stage timings and reproduce one frozen VolGAN origin;
2. split generated scenario core from target attachment;
3. add the explicit one-origin cache and sequential reuse in both adapters;
4. expose and validate reusable zero-policy fitting;
5. pass exact-equivalence, call-count, no-look-ahead, and memory tests;
6. benchmark the same readiness origin;
7. add bounded frozen-target parallelism only if optimization remains material;
8. consider decision-major restarted parallelism and prepared optimizer matrices
   only after the frozen production path is stable.

The resource manager daemon was unavailable during this static review, so
`gpuctl resources --json` used its local fallback; RAM and monitored disks were
reported `OK`.  No GPU or production job was started or modified.
