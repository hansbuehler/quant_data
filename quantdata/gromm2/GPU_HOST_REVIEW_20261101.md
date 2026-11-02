# GPU and host pipeline review — 1 November 2026

## Implemented follow-up

The primary agent subsequently checkpointed the conditional trainer at epoch
218, one committed minibatch into the next epoch. All 100,000 paths, Adam
states, RNG states and the remaining observation permutation were retained.
The pause was for the identified performance defect, not a telemetry timeout.

`IndependentPortfolioStep` now retains its computation graph, inputs, status
and gradient buffers when only learning rates change, and records only a
replacement optimizer graph. Snapshot/restore preserves parameter and Adam
tensors in place. GPU regression coverage exercises LR changes, eager tails,
compiled evaluation, RNG preservation and non-finite rejection. Local audit
logs now set the stage before training and partition epoch wall time into
training, graph capture, population evaluations and checkpoint saving.

A bounded float32 probe used the actual 15,625-by-614 state and
15,625-by-276 return geometry. The first ordered probe measured the LR-change
update at 2.537 seconds for archived code versus 0.01153 seconds for the fix.
A warmed-cache repeat measured 0.06852 versus 0.00893 seconds and required
exact equality of every parameter and Adam-state value. Ordinary warm updates
remained approximately 0.0023–0.0028 seconds. These are isolated measurements,
not a whole-audit speedup claim; cache state strongly affects recapture cost.

Focused CPU checks passed 33 tests, with seven GPU-only skips. The broader
allocated-GPU run passed 42 tests and failed one existing unseeded recurrent
compiled/eager precision test. Its tolerance was not weakened. A controlled
archived/current/eager comparison at prespecified seeds 0 and 20261101 found
exact archived/current parameters, gradients and Adam states at every update.
Seed 0 failed the original eager-parameter tolerance under both implementations;
seed 20261101 passed under both. This is retained evidence of a pre-existing
seed-sensitive precision issue, not a clean full-suite result.

Full evidence and immutable pre-resume checkpoint:
`/home/hans/quantdata_data/experiments/gromm2/noarb-pca16-20261031-v1/lr-recapture-recovery-20261101-v1/`.
Allocated production preflight additionally passed six active-tower tests,
26 runtime tests and the local-objective phase-parity test. The host then
rebooted before the trainer resumed; a new boot was confirmed and the same
retained checkpoint and paths were verified before restoring the service.

No extra host population cache was enabled: each training fold already fits
in VRAM. Bounded asynchronous startup/frozen-evaluation staging remains a
separate optimization. Normalization still precedes the trainer heartbeat,
and mid-epoch checkpoint timing is not separately partitioned; the remaining
review findings below should not be read as fully resolved.

After reboot the trainer was directly observed waiting in `p9_client_rpc`
while reloading the `/mnt/d` population: 35,840 of 50,000 paths at 660 seconds.
This is input waiting, not a measured optimizer/GPU-computation rate. The
manager then reported approximately 90 GiB free host RAM. An optional exact
raw cache helper (`audit_host_cache.py`) and 18 passing CPU tests were added
to support this case. It uses manager-controlled full/partial admission,
bounded reads, pressure-triggered release and HDF5 fallback. It is not yet
integrated into the active trainer; integration is reserved for a safe
process boundary so current loaded-source provenance and checkpoints remain
unchanged.

Live resumed production subsequently completed epochs 219–283. The 62 ordinary
warm epochs had median wall time 1.01545 seconds and maximum 1.04592 seconds.
LR changes at epochs 226 and 277 recorded optimizer-only recaptures of
0.00593 and 0.00487 seconds, with no computation-graph rebuild. Their complete
epochs took 1.02260 and 1.02811 seconds. This confirms that LR changes no
longer create the observed long pauses in this measured production window;
it does not remove startup or frozen-population I/O.

## Raw-cache integration

The next-case loader was staged outside Git under the external experiment's
`cache-integration-20261101-v1/` directory and installed only after the first
case's conditional trainer exited. Astra's static review found no remaining normalization, fold-isolation,
source-binding or checkpoint issue after correcting read-only NumPy staging
to use pinned `torch.empty` followed by an exact native-dtype copy. An already
requested stop is checked before allocating resident tensors and again before
each upload chunk.

Thirty-three installed CPU regressions pass, including launcher gating, with
two allocated-GPU checks deferred. Coverage
includes a nonzero predictable-drift positive control: cached and uncached
training produce identical losses, model parameters, Adam state, RNG state and
frozen holdout scores. Tests also verify raw input bits, masks, normalization,
exception cleanup and stop-before-allocation behavior. These are correctness
checks, not an end-to-end speedup measurement.

The prospective case launcher rechecks its exact source fingerprint after
path generation, before conditional training. It enables the cache only when
the installed loader, helper and tests match an explicit activation manifest;
changed sources require a new allocated-GPU preflight, including pinned-upload
parity. The activation manifest now records the safe completed-process boundary;
allocated-GPU parity subsequently passed before material cached training.
The 58 allocated preflight checks include the new pinned-upload parity test.
The first production raw cache retained 44.5 GB in 116.84 seconds with healthy
manager headroom; no end-to-end speedup is claimed from this loading time alone.
All saved populations, original Q weights, scientific objectives,
optimizer batch sizes and statistical selection rules remain unchanged.

The first production fold completed 300 epochs and its frozen 50,000-path
holdout evaluation. Both diagnostics selected the zero-action baseline from
internal training-fold validation. The resulting zero scores are no detected
strategy in this fitted class, not a proof of conditional martingality. The
second fold subsequently also completed 300 epochs, selecting epoch 115 for
both diagnostics. The complete 100,000-path, 125-day legacy-grid eight-step
case has frozen cross-fitted statistics of -1.3072 (entropy) and -1.1789
(squared). Daily unconditional 99th-percentile and maximum absolute statistics
are 2.7597 and 3.8976: 241 pointwise-99% exceedances, but no Bonferroni-familywise
99% exceedance. Option-to-expiry maximum absolute statistic is 1.5139.
This is not a proof of martingality; the separate rare-tail investigation
documents substantial standard-error sensitivity to one antithetic pair.
The repaired-grid eight-step audit has also completed. Its daily screen rejects
with maximum |z| 16.0496 and 77 familywise-99% exceedances. Both conditional
folds selected zero actions after 300 epochs, making their zero-score/zero-SE
statistics uninformative rather than evidence against the unconditional
failure. Both daily-step audits have now also completed, each with 100,000
paths over 125 days. Their daily maximum |z| values are 32.0408 (legacy) and
493.2043 (repaired). The repaired daily-step entropy/squared conditional
statistics are 51.2060 and 49.3660, decisively detecting predictable drift.
All four cases reconcile against their raw pair moments and frozen scores;
the dispatcher exited successfully and its allocation was released. Full
numerical findings, timing evidence and completion verification live in the
external experiment directory. Computational completion is not a claim that
the no-arbitrage tests passed.

## Scope and evidence

Requested Astra-high review of the canonical combined-book and conditional
audit trainers. This is a deliberately retained source review, not generated
training data. The review did not interrupt a job, run a competing GPU
benchmark, change scientific code, or modify a checkpoint.

Sources reviewed: `local_audit_training.py`, `conditional_audit.py`,
`market_training_runtime.py`, `market_batching.py`, `combined_hedging.py`,
`market_hedging.py`, the relevant population loop in `portfolio_hedging.py`,
the existing `GPU_TRAINING_REVIEW.md`, and focused regression tests.
Line references below identify the source at inspection, before prospective
runtime changes. Evidence is local source, retained JSON training history,
and the persistent job log; no inference of GPU activity from VRAM allocation.

Current workload: `quant-data-gromm2-ordered-full-audits-20261101-v1.service`,
allocation `547d08f09b9a`, conditional audit of 100,000 paths with 125 days,
614 state features and 276 returns. The previously reviewed combined DH run
has finished its required 100 epochs; its disk-backed population pipeline is
not the present conditional trainer's steady-state pipeline.

## Implemented after the review

The owner checkpointed the conditional trainer at epoch 218 plus one committed
minibatch, archived its sources and checkpoint, and resumed the same population
and optimizer/RNG state after allocated-GPU checks. No paths, objective, model,
normalization, batch size, Q weights or statistical selection rule changed.
The recovery record is in the external experiment's
`lr-recapture-recovery-20261101-v1/` directory.

`IndependentPortfolioStep` now retains its computation graph and input/status/
gradient storage when only Adam learning rates change. It refreshes only the
optimizer graph, restoring parameters, moments, counters and RNG in place.
Full recapture remains necessary for other signature changes. The conditional
trainer now sets its stage before blocking training work and reports separate
training, capture, population-evaluation and end-of-epoch checkpoint timings.
Normalization has explicit start/end events; its in-loop heartbeat and the
mid-epoch checkpoint timing remain follow-up work, not completed fixes.

At the actual 15,625-by-614 input and 276-return geometry, an ordered isolated
probe measured an LR-change update at 2.5373 seconds before versus 0.01153
seconds after. A warm-cache repeat measured 0.06852 versus 0.00893 seconds.
The repeat required exact parameter and Adam-state equality, with zero maximum
absolute difference for both. These are bounded update measurements, not an
overall audit speedup or proof that every second of the historical pauses was
capture time. Startup and compiler-cache conditioning materially affect them.

Focused CPU checks passed 33 tests, with seven GPU-only skips. The allocated
GPU suite passed 42 tests and exposed one existing unseeded recurrent
compiled/eager tolerance failure. A controlled archived/current/eager probe
at seeds 0 and 20261101 found exact old/current parameters, gradients and Adam
states across all updates. Seed 0 reproduced the eager-tolerance failure in
both implementations; the other seed passed in both. Tolerances were not
relaxed. All LR-recapture, resume, finite-update, tail and local-action GPU
tests passed; the existing recurrent precision sensitivity remains recorded.

No additional whole-population host cache was enabled. The current optimizer
already reads a GPU-resident fold. A bounded pinned-reader or admitted single-
fold raw cache remains an I/O optimization to measure separately, not a remedy
for unnecessary computation-graph rebuilds.

Evidence roots:

- `/home/hans/quantdata_data/experiments/gromm2/noarb-pca16-20261031-v1/audit-dispatcher.log`
- `/home/hans/quantdata_data/experiments/gromm2/noarb-pca16-20261031-v1/full-audits/grid-minus3-plus2-substeps8/conditional/`
- `/mnt/d/quant_data/gromm2/noarb-pca16-20261031-v1/grid-minus3-plus2-substeps8/paths.h5`

Resource-manager checks during inspection were all OK: approximately
47.6 GiB available host RAM of 94.0 GiB, and 177.4 GiB available on the
monitored root filesystem. Inside the inspection sandbox `gpuctl resources`
reported its documented local measurement fallback, with valid OK checks.
No independent RAM/disk usage estimator was substituted for the manager.

## Principal finding: LR changes rebuild too much

**High priority:** `market_training_runtime.py:237–246` includes the numeric
learning rate in the same signature as input geometry and active policies.
Any LR change invokes the complete capture path, discarding both graphs
(`:143–147`), cloning inputs and parameter/optimizer snapshots, warming all
forward/backward work three times (`:167–170`), and recording both computation
and optimizer graphs (`:186–194`). The forward/backward computation does not
depend on Adam's learning rate. Rebuilding it for an LR-only change is avoidable.

This mechanism aligns precisely with the largest mid-phase delays in the
current log. Replaying the configured `ReduceLROnPlateau` selection logic
against the retained internal-validation history places both policies' LR
reductions after epochs 72 and 123. The following epochs are the outliers:

| Fold 0 event | Time since preceding epoch event | Interpretation |
| --- | ---: | --- |
| First epoch | 131.23 s since the trainer timer began | Includes resident upload and initial compile/capture; excludes normalization |
| Epoch 73, affine | 161.61 s | First epoch after both LRs change |
| Epoch 101, full network | 80.52 s | First deep-network phase, new runtime and graphs |
| Epoch 124, full network | 74.60 s | First epoch after both LRs change again |

At the epoch-174 inspection, the 99 affine epoch intervals after epoch 1
had median 0.946 s and total 257.61 s. The 73 full-network intervals after
the phase-transition epoch had median 1.322 s and total 174.30 s. Epoch 174
was logged at elapsed 643.66 s. These complete intervals include intervening
checkpoint writes and evaluation, not merely an optimizer throughput window.

The source proves that complete recapture happens at those LR changes.
The timing alignment makes recapture the leading explanation for these
outliers, but current logs do **not** partition the delay into compiler work,
capture, checkpointing, driver waits, or host scheduling. Do not claim the
entire 161.61/74.60 seconds is measured capture time or a guaranteed saving.

**Minimal prospective optimization:** retain the computation graph, static
inputs, status output and gradient buffers for an LR-only signature change;
recapture only the optimizer graph. Preserve independent Adam optimizers and
the existing pre-commit finite-status decision. Snapshot and restore model
and Adam tensors in place, correctly order capture against prior queued
optimizer replay, preserve RNG, and publish the new graph/signature only
after success. Keep full recapture for geometry, active-policy, phase and
other relevant optimizer changes. Device-resident LR scalars are an
alternative, but change optimizer/checkpoint typing and are a less surgical
first change.

## Progress currently misattributes the long pauses

**High priority:** `local_audit_training.py:346–349` updates the training stage
only after `step(...)` returns. At an epoch boundary the preceding stage is
`internal-training-fold-evaluation` (`:355`), and no new stage is set before
permutation, capture or the next update. Consequently the heartbeats during
epochs 73 and 124 retained the previous evaluation label. They do not prove
that population evaluation caused those pauses.

Normalization (`:242–243`) runs before the heartbeat thread and its elapsed
timer start (`:286–299`). Its cost is omitted from the reported first-epoch
elapsed time. The upload stage also remains stale during initial capture.
Checkpoint saving (`:301–314`, `:377`) has no distinct stage or duration.

Set the stage before entering expensive work; report stage elapsed time,
effective batch sizes, full and optimizer-only capture counts/durations,
training/evaluation/checkpoint durations and complete epoch duration. Use
one-time/low-frequency instrumentation, not extra GPU synchronizations every
batch. Start progress and resource monitoring before normalization. Include
work-unit counts and meaningful ETA for upload and evaluation. Capture ETA
may legitimately remain unknown.

## Host memory helps startup and frozen evaluation, not resident epochs

Each 50,000-path fold occupies 20.722 GiB in float32 features plus returns.
`_resident_fold` allocates those arrays once on the GPU (`:176–177`). All
optimizer batches and both internal population evaluations subsequently
index those resident tensors (`:198–211`, `:319–359`). There are no HDF5 reads
inside those epochs. More host RAM cannot remove their LR-triggered recaptures.

There are nevertheless concrete host-pipeline inefficiencies outside epochs:

1. `_normalization` reads features and returns from HDF5 (`:143–144`), masks
   them and converts selected values to float64. `_resident_fold` rereads
   the same fold (`:182–183`). This repeats HDF5 decompression and user-space
   materialization even if the operating system has cached the physical bytes.
   The source files use LZF-compressed 128-path chunks
   (`path_audit.py:283–287`); this is not the combined trainer's contiguous
   memmap format.
2. Upload allocates fresh pinned copies for every 512 paths (`:186–187`),
   then uses blocking copies (`:188`, `:190`) on the serial read/upload loop.
   Pinning alone provides no producer/transfer overlap. A chunk is about
   217 MiB; two reusable raw pinned slots are about 435 MiB total.
3. Frozen evaluation rereads training and holdout folds (`:416–421`), uses
   serial tensor transfers, and copies each policy result to CPU separately
   (`:423–434`). It does not reuse the resident training-fold tensors and
   does not use the bounded market prefetch iterator.

Observed upload heartbeats were 25,600/50,000 paths at 60 seconds and all
50,000 at 120 seconds. The second label can include initial compilation
because progress is stale, so this is not a precise standalone upload timer.
The `/mnt/d` location is a plausible extra I/O cost, not proof that storage
latency dominates: no isolated storage/decompression/transfer benchmark was
run during the active job.

**Minimal host improvement:** one bounded reader and two reusable pinned
buffers, with asynchronous copies and events protecting reuse. Align reads
with HDF5 chunks; retain the existing normalization reduction order and
float64 accumulation. HDF5 reader concurrency should not be assumed to scale.
Pack frozen diagnostics for both policies before one CPU transfer, preserving
all per-path scores, original return units and antithetic-pair inference.

**Optional larger host cache:** manager-admitted exact-dtype raw arrays for
one fold could avoid its second HDF5 pass and support later frozen training
evaluation if their lifetime is designed explicitly. A single fold needs
20.722 GiB; both folds need 41.444 GiB. With the observed approximately
47.6 GiB available, blindly allocating both would violate the manager's
projected headroom. Use its reported limits, periodic rechecks, release on
WARN, and checkpoint/stop behavior on STOP. Do not pin an entire population,
change precision, or copy datasets into Git. A cache must be opt-in and
measured; retaining an unused duplicate of already resident GPU data has no
steady-state benefit.

## What is already correct, and what not to optimize blindly

- Training batch selection chooses 15,625 observations, giving exactly 360
  updates over 5,625,000 training transitions. Evaluation chooses 31,250,
  giving 180 training and 20 internal-validation batches. There is no tail
  shape at this geometry, so repeated tail recapture does not explain the
  current conditional gaps. Changing optimizer batch size changes the
  optimizer trajectory and is not a semantics-free performance fix.
- `market_training_runtime.py:254–264` has one packed CPU safety decision
  before any independent optimizer commits. Removing that synchronization
  would weaken the finite-input/loss/gradient transaction contract. The
  conditional population loop already keeps accumulators on device and
  transfers only its final two losses (`local_audit_training.py:211`).
- Training uses compiled policy forwards inside manual CUDA graphs. The
  conditional population loop does not compile its entire gather/objective/
  reduction transaction; that is a possible later improvement, but it is not
  established as the cause of the measured outliers.
- The completed combined trainer already has exact-dtype raw host caching,
  shared input assembly, bounded pinned slots, producer threads and an async
  transfer stream (`market_batching.py:109–209`, `:217–270`, `:294–387`). Its
  documented mapped-validation bottleneck is genuine prior evidence, not a
  diagnosis of this GPU-resident conditional run. Its recurrent policy
  dependence (`combined_hedging.py:87–121`) is real and must not be removed to
  manufacture utilization.
- Low instantaneous utilization or power does not quantify useful throughput
  or energy efficiency. Conversely, fast warm epochs do not excuse long idle
  intervals. Compare full wall time and, when reliable telemetry is available,
  integrated energy for identical useful work. Driver telemetry failure alone
  is not a reason to restart a healthy allocated training process.

## Validation and measurement plan

1. Apply prospective timing instrumentation first. Retain the active process
   and its checkpoint; do not rerun completed estimates solely to retrofit
   performance changes. Keep source fingerprints in later launch preflights.
2. For optimizer-only recapture, test that compute graph identity, static
   inputs, gradient pointers and full capture count stay unchanged after LR
   cuts. Compare parameters, clipped gradients, Adam moments and steps,
   losses, RNG, scheduler state and restart behavior against eager updates.
   Include compiled population evaluation between updates, an eager tail,
   repeated independent LR changes, active-policy changes and non-finite
   injection before commit. Test the production shape/precision under an
   explicit allocation when it will not compete with the active job.
3. Instrument several complete epochs around an LR reduction and the phase
   boundary. Separate optimizer-only capture from full capture and compilation.
   Report the distribution and total, not just a selected fast median. The
   two LR-associated outlier intervals total 236.21 seconds in this snapshot;
   they identify substantial opportunity, not a measured achievable speedup.
4. Benchmark host staging separately on bounded, identical HDF5 chunks:
   read/decompression, CPU checks/normalization, pin-copy, transfer, and total.
   Use the same warm/cold-cache conditions and do not hold a large cache during
   the no-cache control. Compare complete fold startup and frozen evaluation.
5. For any host-path change, verify exact sample/fold order, normalization mask,
   native input bits and eligibility, unchanged float64 reduction convention,
   forward values/gradients, partial chunks, exception cleanup and buffer/stream
   lifetime. Preserve all observations and independent holdout semantics.

No new scientific tests or GPU performance benchmark were run for this review.
The timing table and LR-reduction alignment were computed with bounded local
JSON-history inspection. Speedup remains to be validated after a focused fix.
