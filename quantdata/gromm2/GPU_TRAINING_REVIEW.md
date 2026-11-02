# GPU training review — 28 October 2026

## Scope and outcome

Reviewed the independent zero/short/long power-exponential hedges, input
assembly, recurrent policy, optimizer, full-population valuation, checkpoint
restart and price monitor. The review used Astra at high reasoning effort,
independent correctness inspection and isolated allocated-GPU measurements.
Production was checkpointed before benchmarking; its weights and path files
were not changed by the benchmarks.

The model, original Q weights, unit-L1 costs, three independent networks,
training batch of 256 and all one million paths in each split are unchanged.
The last training batch has 64 actual observations. Validation continues to
select checkpoints, learning rates and stopping independently per policy.

## Measured bottleneck and replacement

The original resident-data batch dispatched 3,108 device kernels over a
261.93 ms profiled interval, although their summed execution time was only
9.216 ms. Host dispatch was the principal measured waste. Profiling perturbs
timing, so the unprofiled comparison is the throughput evidence:

| Three policies, 256 paths | Legacy compiled loop | Captured transaction |
|---|---:|---:|
| Median resident batch | 145.63 ms | 9.10 ms |
| Mean resident batch, 24 measurements | 144.45 ms | 13.34 ms |
| Per-batch host safety decisions | 6 | 1 |

The captured mean includes four initially slower measurements; the following
20 were approximately 8.4–9.6 ms. This is a resident-computation comparison,
not a promise that disk-backed training improves by the same factor.

The replacement executes two CUDA graphs: all independent forward/backward
calculations and gradient clipping, then independent optimizer commits.
One packed finite-status read separates them. Non-finite inputs, losses or
gradient norms prevent every optimizer commit. No device assertion poisons
the CUDA context, and invalid batches are never silently discarded.

The recurrent dependence on previously chosen positions is genuine and
remains unchanged; it is captured rather than removed or replaced by a
different LSTM model.

## Streaming and evaluation

- Gather shared raw market inputs once. Preserve NumPy float64 logarithms
  and strike calculations before the existing float32 feature conversion.
- Keep two bounded pinned staging slots, an ordered producer queue and
  asynchronous transfer stream. One or two preparation workers are
  configurable without affecting model identity or sample order.
- A shared capture lock excludes producer CUDA calls during explicit or
  compiler-managed graph capture. Events protect pinned-buffer reuse;
  device tensors record their consumer stream.
- Compile packed population diagnostics and transfer one packed result per
  evaluation batch instead of twelve small results. Keep every observation
  and the original global OCE calculations, including the partial batch.
- Reuse checkpointed normalization on restart instead of rescanning the
  million-path training set. Report separate evaluation counters and recent
  training-window throughput, not elapsed-time rates polluted by compilation.

Materialized features, prices and payoff occupy about 69.96 GiB per split,
139.92 GiB for both. Even the compact raw inputs are about 39.28 GiB per
split. They do not fit the approximately 32 GiB GPU. Full-data VRAM loading
is therefore not an available optimization; bounded staging is intentional.

The integrated input-plus-training benchmark reached approximately
10,048–12,018 paths/second on warmed inputs, but only 2,535–3,874 paths/second
in earlier trials as the disk cache warmed. These measurements do not establish
a universal winner between one and two preparation workers. Both are bounded;
the default uses two to overlap preparation and disk-read latency.

## Correctness and restart findings

Fixed the old partial-minibatch checkpoint failure: if an unexpected
exception interrupts an optimizer commit, the preceding complete checkpoint
is retained rather than saving partially advanced policies with an old
sample cursor. Numerical validation failures occur before any commit.

Capture warmup is rolled back in place, including parameters, Adam moments,
step counters and RNG. Changes of batch shape, active-policy set or optimizer
hyperparameters trigger recapture. Optimizers remain separate. Capturable
Adam uses device step tensors; checkpoint names and values are preserved.

A 24-update real-checkpoint comparison, including a learning-rate change and
the 64-path tail, found maximum absolute differences of 5.96e-8 in parameters,
1.45e-7 in clipped gradients, 1.56e-8 in Adam state and 2.48e-9 in loss. This
is ordinary floating-point agreement, not bitwise equivalence to legacy Adam.

Also corrected CPU-to-GPU restart when no saved CUDA RNG exists, respected
minimum epochs in the independent trainer, and required matching epoch and
independently selected best epochs before old final quotes can override the
live price monitor.

No blocking economic, objective or causality defect was found in this
training review. This does not certify finite-step simulation martingality;
the previously recorded discretization defects remain outside this change.

## Reviewable code and evidence

- [Training transaction](market_training_runtime.py)
- [Shared input assembly and pinned streaming](market_batching.py)
- [Training, restart and packed evaluation](portfolio_hedging.py)
- [Live selected-checkpoint prices](hedging_price_monitor.py)
- [Legacy/graph-mode profiling](benchmark_market_training.py)
- [Integrated streaming/evaluation benchmark](benchmark_training_pipeline.py)
- [Atomic update, non-finite injection and CUDA restart tests](../../tests/test_market_training_runtime.py)
- [Input bit-equivalence, ordering and stream-lifetime tests](../../tests/quantdata/gromm2/test_market_batching.py)
- [Population and checkpoint regressions](../../tests/quantdata/gromm2/test_portfolio_hedging.py)

Persistent benchmark traces, numerical comparisons and reports live outside
Git under:

`/home/hans/quantdata_data/experiments/gromm2/training-performance-20261028-v1/`

Subdirectories identify `baseline-default`, `reduce-overhead`,
`runtime-captured-v1` and `integrated-pipeline-v1`. The production pause
record is `paired-real-books-20261028-v1/cohort-unit-l1-20261028-v1/performance_pause.json`
under the same external GROMM2 experiment root. Its epoch-2 checkpoint has
`next_batch=1,000,000`: restart must finish epoch-3 population evaluation,
not replay epoch-3 training observations.

## Fresh production launch

After validation, the user requested a fresh start rather than resuming the
three-epoch attempt. The new service is
`quant-data-gromm2-cohorts-graph-fresh-20261028-v1.service`. Its unique external
control/output root is:

`/home/hans/quantdata_data/experiments/gromm2/paired-real-books-20261028-v1/cohort-graph-fresh-20261028-v1/`

It reuses the original one-million-path training and validation files but
starts new parameters, optimizer states and schedulers; the old checkpoint
is preserved. The source-data and model specification are unchanged. A
separate price monitor follows only the fresh checkpoint directory.

Validation completed before launch: 121 focused CPU tests passed (nine
GPU-only skips), and 42 tests passed under an explicit GPU allocation,
including exact captured save/restart agreement, input lifetime/ordering
checks and non-finite rejection. Compiled evaluation was checked at both
4,096 paths and the actual 576-path final population batch.

## Combined-book host-memory review, 2026-11-01

An independent Astra-high review identifies two remaining pipeline problems.
Random training and full-population evaluation compete for the operating
system's file cache. Separately, the single captured-graph slot discarded
the 256-path graph for each 64-path final batch, then captured it again in
the next epoch. Peak optimizer-window throughput therefore overstated
whole-epoch performance and did not explain long GPU-idle intervals.

The optional `--host-cache-training` mode copies only the compact raw
training arrays into ordinary, exact-dtype host memory: 42,168,000,000 bytes
(39.272 GiB). It does not cache expanded feature/price tensors or the
independent validation population. The existing bounded pinned staging
buffers still handle asynchronous transfers. Admission and ongoing cleanup
use resource-manager thresholds; WARN releases the cache, while a known
resource-stop error releases it without delaying the checkpoint deadline
for another probe. Source files, path weights, sample order, normalization,
model identity, optimizer state and the financial objective are unchanged.

Rare partial batches now execute the original policies eagerly while
retaining the nominal full-batch graph and its gradient buffers. This also
applies when resuming directly at the final partial batch. No observations
are padded, dropped or given different objective weights. Graph capture,
producer wait and optimizer-window time are logged separately. Evaluate
the improvement using complete epochs, not GPU utilization or peak rates.

CPU resident-data probes measured roughly 429 paths/second for the same
three policy updates with four threads. Observed GPU optimizer windows
ranged from approximately 1,500 to 22,000 paths/second, but these are not a
controlled whole-epoch speedup measurement. Allocated-GPU numerical
regressions and post-resume epoch timings must establish the benefit of
the new pipeline before any stronger performance claim.

### Graph lifetime correction and recovery

The retained full-batch graph exposed a previously hidden workspace-lifetime
defect. Warmup allocated cuBLAS workspaces outside its graph pool; compiled
population evaluation clears those caches. Replaying the old training graph
after evaluation could consequently access freed storage. The previous
per-epoch recapture had hidden this interaction.

An isolated reproduction used the actual 584-feature, 101-control checkpoint:
256-path captured training, a 64-path eager tail, compiled population batches
of 4,096 and 576, checkpoint serialization, then retained-graph training.
The old capture convention failed with CUDA illegal memory access. Clearing
the cuBLAS cache immediately before and after recording makes workspaces
graph-owned; the corrected sequence passed two cycles without recapture.
Maximum absolute differences against independent eager updates were
8.20e-8 for parameters, 3.58e-7 for gradients, 3.54e-8 for Adam state and
4.32e-9 for loss. RNG states were unchanged. Regression tests now cover this
training/evaluation boundary, not just consecutive training updates.

The epoch-60 checkpoint contained one committed post-evaluation update before
the following update failed. Although its tensors were finite, that is not
enough to certify the committed update. It and its reports were archived;
recovery uses the immutable epoch-59, offset-299264 snapshot, preserving all
optimizer, scheduler, RNG, sample-order and validation-selection state.
Discarded work does not count toward the minimum 100 completed epochs.

The host cache can now admit a subset of exact raw fields when all 39.272 GiB
would exceed the manager's projected headroom. Largest fields are considered
first; every cumulative admission uses the same manager thresholds. Remaining
arrays stay mapped. Cached names and bytes are reported, and WARN releases
only owned copies. For example, retaining every field except option prices
uses 26.129 GiB. No cache is admitted during an existing resource warning.

The bounded ABBA benchmark also had a biased recommendation: mapped trials
competed with the anonymous cache for RAM, unlike the no-cache controls.
Recommendations now require superiority over the unpressured before/after
controls. The production cache remains an explicitly labelled whole-epoch
trial, not a speedup inferred from peak windows or memory-pressured trials.

Exact-model failure/success reports and the recovery record are retained in
the external `noarb-pca16-20261031-v1/graph-lifetime-*` directories and the
combined run's `resume-host-cache-20261101-v1/graph-failure-recovery-20261101-v1/`.

Final focused validation: 142 CPU tests passed (13 allocated-GPU skips);
the clean allocated-GPU preflight passed 72 tests. Source fingerprints are
checked before launching the continuation. Import sorting, syntax and
focused whitespace checks also pass.

Two complete warm production epochs then finished in 200.13 and 209.12 seconds.
Each includes one million optimizer observations and full one-million-path
training and validation evaluations for all three policies. The original
full-batch graph remained at capture count one across both epochs, without
non-finite updates. Recent optimizer windows process approximately
20,000–23,000 paths/second; these are not whole-epoch rates.

Population evaluation remains substantial: the cached training population
took 32.93–36.55 seconds, versus 97.77–115.11 seconds for the independent,
disk-streamed validation population. Both populations remain complete.
These absolute measurements establish the current throughput, not a controlled
speedup against previous runs with different cache and startup conditions.
The external continuation's `whole_epoch_timings.json` retains the intervals.

### Independent live-pipeline review

A fresh Astra-high review on 2026-11-01 confirms that the remaining measured
bottleneck is validation input, not repeated graph capture. At epoch 76,
cached training evaluation took 28.47 seconds and mapped validation took
98.32 seconds; the next cached training evaluation took 24.58 seconds.
The latest 17 complete epoch intervals have a median of 240.43 seconds,
with a range of 179.63–345.60 seconds. The earlier two-epoch snapshot is
not a representative distribution or a speedup estimate.

Both populations require 39.272 GiB of compact raw arrays. Each expands to
72.621 GiB of model features and prices during evaluation; two bounded
4,096-path pinned slots occupy about 609 MiB. The training population is
already cached in ordinary host RAM. Another full validation cache is not
admissible with the observed manager headroom of roughly 19–24 GiB.
The trainer also has about 2.04 GiB of swapped anonymous pages; that does
not establish current swap activity, but argues against blindly reserving
more memory. Keep resource-manager admission and release checks, and do
not pin whole populations or change data precision to fit them.

The next controlled experiment should compare one versus two validation
preparation workers at the same 4,096-path batch size, followed by bounded
sequential raw-chunk read-ahead if needed. Keep the training cache, all
observations, original Q weights and optimizer/RNG state unchanged. Judge
the result by complete epoch timings, including both population evaluations.
Cache repartitioning is an unproven experiment because sacrificing randomly
accessed training data could cost more than faster sequential validation.

Add explicit preparation, transfer/computation wait, CPU-statistics and
checkpoint durations before attributing idle periods more precisely.
Current update timings include outstanding transfer dependencies and the
required atomic finite-status check, not just GPU computation. A delayed
driver sample reported 86% utilization and 201.85 W, so neither a quiet
instantaneous display nor allocated VRAM alone describes whole-run activity.

CPU probes reject two tempting distractions: individual million-path OCE
translation solves took 0.204–0.217 seconds on synthetic data, while exact
zlib caching projects 29.01 GiB and only about 250–343 MiB/s decompression
for the large fields. Neither is supported as the primary optimization.
All six running-source fingerprints still match the allocated-GPU preflight.
No production code, checkpoint or live service was changed by this review.
