# Independent review: public model-training layer

## Gate recommendation

**FAIL.** The public-only data boundary, dated training windows, train-only
normalization, GROMM scenario semantics, four hedge objectives, historical
SANOS optimizer, and diffusion static-arbitrage penalty are materially sound.
The present layer is not yet a valid Phase-5/Phase-9 production gate, however:
the VolGAN and diffusion entry points regenerate the final training target
rather than forecast from the as-of state; neither published generator is yet
connected to an option hedge or the common smoke contract; and neural
checkpoints are not sufficiently identified for safe exact restart.

## Reviewed state

- Review date: 2026-10-12
- Base Git revision: `cb6adb9bcfac5b1dcbdc62ad26ffa1f4cb6eead2`
- Source state: uncommitted public hedging/model worktree on `main`
- Scope: `data/generative_surfaces.py`, `models/gromm`,
  `models/historical_sanos`, `models/volgan`, `models/diffusion`, focused tests,
  standalone CPU commands, and the public-only runtime guard
- Plan gates: Phase 5 and Phase 9 in `quantdata/hedging/PLAN.md`
- Reviewer: independent `phase4_quant_review` agent

## Blockers

### 1. VolGAN and diffusion do not forecast from the as-of state

Both standalone commands pass the final *training example's input* to the
generator:

- VolGAN uses `fit.data.conditions[-1]` in `models/volgan/smoke.py`.
- Diffusion uses `fit.data.sequences[-1, :-1]` in
  `models/diffusion/smoke.py`.

For the real public 2023-01-17 fit, the last VolGAN condition is dated
2023-01-13 and its target is 2023-01-17. The last diffusion sequence starts on
2022-12-14, has target 2023-01-17, and its supplied 21-day context likewise
ends on 2023-01-13. Thus both smokes generate an in-window reconstruction of
2023-01-17, not the next transition conditional on information available on
2023-01-17.

Add explicit builders for the live VolGAN condition and live 21-day diffusion
context ending at the as-of state, with dated assertions proving that the
generated transition starts at the decision date and ends at the next eligible
session. Tests should reject a context whose last observation predates the
declared as-of date.

### 2. The published generators are not hedging baselines yet

The VolGAN and diffusion `hedger.py` files return only a generated forward
return and next constant-maturity log-IV grid. They do not:

- reprice the fixed-contract hedge book and digital claim from each transition;
- solve a cost-lasso or any agreed hedge objective;
- apply common market execution/accounting;
- emit actions, costs, pricing difference, pure hedge error, cap diagnostics,
  or unhedged/hedged P&L; or
- compare against a committed tolerance fixture.

Their smoke schemas are `volgan-training-smoke-v1` and
`diffusion-training-smoke-v1`, not the required `hedging-smoke-v1`, and they do
not hedge the ATM one-day digital required by `PLAN.md`. The historical SANOS
smoke does perform the complete hedge and emits the common schema, but it also
lacks a committed fixture/check. Consequently none of the three Phase-9
standalone gates is complete as specified.

There is also no retained comparison against the public VolGAN/diffusion
implementations showing that the stated public settings have first been
reproduced. Shape-and-finiteness tests are not evidence for that Phase-9
requirement.

### 3. Checkpoint identity and restart semantics are unsafe

VolGAN and diffusion checkpoint names include model geometry and only the
first/last dates plus row count. Their compatibility checks omit the training
configuration, data-value fingerprint, split/normalization identity, precision,
and seed. Diffusion fine-tuning additionally omits the base-checkpoint identity.
A changed validation fraction, learning-rate schedule, dataset contents, seed,
or fine-tuning base can therefore silently load a stale checkpoint; loading the
state dictionary also overwrites the newly computed normalization buffers.

GROMM checkpoint identity similarly omits the scenario-data fingerprint and
`GrommTrainConfig`. More importantly, after early stopping it loads the best
policy but saves that policy beside the optimizer state from the final training
epoch. Continuing training is not an exact continuation of either trajectory.
A previously converged checkpoint is also continued unless `just_load=True`.

No model has a forced-interruption versus uninterrupted equivalence test. These
issues block the exact/near-exact restart and audit claims required before
material GPU production.

### 4. The Phase-5 selection claim is ahead of the implementation

The GROMM code exposes four candidate geometries and a correct generic
cheapest-within-one-standard-error selector. There is no dated 2022 selection
runner, frozen selected-architecture artifact, or rolling adapter that loads
that frozen choice. The README currently states that the four architectures
*are trained on 2022 scenarios* and frozen, which was not evidenced in this
review.

The committed GROMM smoke is a useful deterministic interface gate, but it
exercises only a Q mean-variance fit. It does not gate P generation or the
mean-volatility, entropy, and power-exponential paths required by Phase 5.

## Passed quantitative and timing checks

- Runtime public adapters require `PublicHedgingPanel`, validate the released
  schema, and the runtime import scan found no private-source loader. The fixed
  grids are reconstructed solely from released SANOS states and normalized
  public returns.
- All rolling histories end at or before their explicit as-of date. VolGAN and
  diffusion use chronological newest-block validation; their means/scales are
  fitted from the training prefix only. Historical SANOS uses 252 prior
  one-session transitions, whose last endpoint may equal the decision-date
  close but never follows it.
- VolGAN correctly annualizes the forward log-return target for training and
  divides by `sqrt(252)` at sampling. Its surface target is a one-session
  log-IV increment. Diffusion keeps return and log-IV channels separate and
  normalizes them using training sequences only.
- GROMM generates distinct Q/P paths from the rolling rank-16 SDLV calibration,
  uses eight substeps, preserves the independent 5% Black background, rejects
  an absent/unaccepted P drift, uses antithetic pairs, and starts training gains
  from its own Q-consistent marks. Common realized scoring remains at SANOS
  market marks.
- Mean-variance, mean-volatility, entropy, and power-exponential signs and
  gradients are correct in the reviewed synthetic tests. Claim-minus-zero is
  implemented for GROMM and historical SANOS as specified.
- The diffusion call-grid penalty correctly prices normalized calls and tests
  calendar, call-spread, and non-uniform-strike butterfly inequalities. An
  independent check gave penalty `2.91e-32` for a flat 20% surface,
  `5.30e-5` for a forced calendar violation, and `86.57` for an alternating
  strike surface.
- The real GROMM Q smoke passed its committed fixture exactly: 128 paths,
  rank 16, eight substeps, converged claim/zero fits, no cap hits, spot action
  `-15.9705990`, model digital price `0.5354354120`, and common market price
  `0.5419907144`.
- The real historical SANOS smoke completed with the common schema, 252
  training returns, 163 instruments, both optimizers converged, and no cap
  hits. Its training origins end 2023-01-13 and the final return endpoint is
  the 2023-01-17 decision close.

## Non-blocking findings

1. Diffusion validation sequences overlap the training boundary heavily. On
   the real window, the final training target is 2022-10-04, while the first
   validation target is 2022-10-05 and that validation sequence starts
   2022-09-06. Moreover, validation averages denoising error over all 22 days,
   so the genuinely forecast target contributes only one twenty-second of the
   basic loss. Report a target-day validation metric, preferably alongside the
   current denoising diagnostic.
2. Diffusion inpainting reuses one fixed context-noise tensor at every reverse
   time. Each overwritten context has the correct marginal `q(x_t|x_0)`, but
   the sequence across diffusion times is not a forward Markov path. Reconcile
   this choice with the referenced algorithm and add a public-implementation
   sampling fixture.
3. `converged=True` for VolGAN and diffusion means only that the configured
   epoch count completed. The reduced smokes therefore report convergence at
   40 and 3 epochs respectively, without an optimization or distributional
   acceptance criterion. Rename this diagnostic or add a precommitted gate.
4. Diffusion fine-tuning penalizes predicted training surfaces, but no test or
   smoke measures static-arbitrage violation rates in generated samples before
   and after fine-tuning. Add that comparison; the implementation is
   arbitrage-penalized, not guaranteed arbitrage-free.
5. Historical SANOS mean-variance optimization uses exact absolute costs plus
   a declared quadratic ridge, while the reported loss uses a smoothed absolute
   cost and omits the ridge. For power-exponential hedging, the reported
   differenced-action objective reuses the claim-policy cash translation.
   These affect diagnostics, not the fitted actions, but should be made
   internally consistent before headline reporting.
6. The fixed-grid builder records which IV nodes were interior but the
   generator losses weight repaired flat-tail nodes identically. This is a
   defensible declared target convention, but retain the mask in diagnostics
   so model quality is not dominated or overstated by imputed tails.
7. GROMM training uses predictable round-trip cost proxies, while realized
   benchmark accounting uses endpoint spreads and terminal spot for closing
   costs. This is acceptable for a predictable training objective but should
   remain explicit in model comparisons.

## Verification performed

- Focused CPU suite: `23 passed` in `9.47s` covering public-grid construction,
  the public-only import/instance guard, GROMM policy/objectives/checkpoint
  load, historical SANOS objectives, VolGAN, and diffusion.
- Fresh real-date CPU GROMM smoke: passed committed tolerance fixture.
- Fresh real-date CPU historical SANOS smoke: passed functional/common-schema
  checks; no fixture exists.
- Fresh real-date CPU VolGAN smoke: 146 examples, 40 epochs, 32 finite samples,
  validation loss `1.11319`; this was a training smoke, not a hedge.
- Fresh real-date CPU diffusion smoke: 146 sequences, 3 epochs, 8 finite
  samples, validation loss `1.14630`; this was a training smoke, not a hedge.
- Independent date audit proved that the generator smoke contexts end on
  2023-01-13 while their final training target/as-of date is 2023-01-17.
- `gpuctl resources --json` could not reach the daemon and used its local
  fallback; every reported RAM/disk check was `OK`. No GPU was used.
- No implementation file was modified by this review.
