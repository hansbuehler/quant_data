# Gromm v2 4,3

For the current forward-start **deep-hedging review and paired real-contract
experiment**, see [DH_TEST.md](../../DH_TEST.md), especially Sections 9-10.
That experiment explicitly uses the selected earlier `(-3,2)` AR(1) model,
not the newer `(-4,3)` rolling calibration described below. Its runner is
`paired_experiment.py`: shared latent paths for the 5%/zero Black controls,
the static/locked-cohort book and the spot/PCA-basket/daily-ATM-call book,
with independent zero/short/long policies. No synthetic latent asset is traded.

The current calibration uses the consolidated source market books and saved
DYSANOS smooth surface parameters. Each observed contract retains its own
strike, expiry, call/put identity, and observed half-spread. Its fitting band is
the DYSANOS fair value plus/minus that half-spread; market midpoints are not the
calibration targets. Original bid–ask coverage is reported separately.

Use the **existing trained SANOS parameter sets**. Do not regenerate, refit,
or replace them for GROMM2. Pin their exact paths with `--source-inventory`;
do not choose newly generated alternatives by scanning for the longest
surface. Read the saved `af_pricer` and evaluate it at the required market
strikes and expiries. Beyond its final saved node, use the SANOS pricer's
native rule: retain the terminal latent density and its terminal kernel
volatility. This is model-based terminal pricing, not additional observed
calibration evidence. Record the source-node horizon and use of that rule
in each GROMM fit report. A 125-day tower does not require refitting SANOS
to create new 125-day source nodes.

Only the GROMM DLV representation is fitted. The saved SANOS source
parameters remain read-only throughout calibration and simulation.
If an individual historical cache file is absent, a small JSON reference
may point to that date's already fitted row in the existing parameter
dataset. The reference records dataset-file hashes; it does not copy or
re-estimate SANOS parameters.

The tower has 200 interior strikes in NMS `(-4,3)`, two exterior boundaries,
daily expiries through 25 business days, and five-day intervals through 125
business days. The Black background volatility is fixed at 5%. Observations
outside NMS `(-3.5,2.5)` receive 10% fitting weight. Both AM and PM contracts
are retained, except the existing 0.2-business-day exclusion.

## Two stages

1. Fit each observation date independently on the grid constructed from that
   date's information. Save quote-level errors, original-market coverage,
   density invariants and coverage across dates. Check all dates before
   starting the rolling estimation.
2. For each eligible origin `t`, hold its spot-relative tower grid fixed and
   refit every available observation in the preceding 252-business-day window
   to its own market contracts and DYSANOS fair values. An interpolated own-date
   state may initialize the optimizer, but every historical state is refitted
   on the `t` grid. Align each one-session pair to fixed cash strikes and fixed
   expiries before estimating the coordinatewise AR(1) of log DLV-squared.

The date `t` is excluded from AR(1) training. Missing observations do not extend
the historical window; pairs crossing more than one business session are
excluded. Report interpolation boundary clipping and unidentified AR slopes.
Each origin has its own fitted history and AR(1); an AR(1) estimated from a
single sequence of changing own-date grids is not substituted for it.

The current run estimates only **29 August 2025**, using its preceding
252-business-day source window. Pass `--origin-date 2025-08-29` to avoid
calibrating unrelated origins. Historical states are fitted chronologically
on this origin's fixed grid, initializing from the previous fitted date's
weights. Only the first required history state starts without prior weights.
Completed estimates are retained rather than rerun solely for warm-starting.

## Retained results

The [26 October 2026 progress snapshot](results/gromm_v2_4_3_2025_08_29_progress_2026_10_26.json)
records the completed origin-date fit and historical calibration reports,
with source hashes and the external run directory. It is an incomplete-run
snapshot, not a completed AR(1) estimate or a passed no-arbitrage audit.

## Execution

Use the project interpreter, `QUANTDATA_ROOT` and `PYTHONPATH`. The one-date
float64 fit runs substantially faster on CPU than on the shared GPU. The
rolling runner uses up to four CPU workers and writes checkpoints and flushed
progress to the uniquely identified external run directory.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.rolling_calibration \
  --source-inventory /path/to/pinned/saved-sanos-inventory.json \
  --run-directory /path/to/external/unique/run \
  --origin-date 2025-08-29 --workers 4
```

`--inventory-only` checks saved parameter availability and records which
dates need native terminal pricing. It never rebuilds the source fit.
The legacy `--source-root` discovery mode retains its strict source-node
coverage check; the current task uses the pinned-parameter mode instead.

The source inventory and both calibration stages are restartable. A completed
date must have at least 90% coverage and valid density mass/forward moments to
unlock rolling AR(1). Weak fits receive continuation where the optimizer has
reached its epoch ceiling; unresolved dates remain visible in the summary and
require review. Core and outer-wing metrics remain separate in each date's
report.

## Rolling unconditional audit

For every completed rolling origin, simulate 100,000 one-business-day `Q`
transitions with eight Brownian substeps using the production front bridge
and risk-neutral DLV drift. The fitted historical innovations supply the
shared `P`/`Q` diffusion; the estimated physical AR(1) drift is not used in
the `Q` transition. The first factor is fitted jointly to front-distribution
Gaussian spot scores, with 15 residual PCA factors.

Test each fixed contract's next-day gain divided by inception spot, without
return centering or empirical measure preconditioning. The reduced book has
11 NMS levels `(-4,-3.5,-2.5,-1.5,-0.5,0,0.5,1,1.5,2.5,3)` at each five-day
expiry through 125 business days, plus the one-day front expiry. Use puts
below ATM and calls otherwise. Including spot, this gives 287 coordinates.

Streaming moments and sample standard errors treat each complete antithetic
pair as one independent observation. Report coordinate means, ordinary `z`,
pointwise 99% confidence intervals and a familywise 99% Bonferroni screen.
Separately report the economic-floor score `mean / (1e-8 + SE)`; this is not
a Gaussian `z` statistic. Density invariants and drift-taming frequency are
saved alongside the statistics.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.rolling_audit \
  --calibration-root /path/to/external/unique/run/calibration \
  --watch --origin 2025-08-29 --backend cpu --paths 100000 --substeps 8
```

This task is **CPU-only**, including path generation, pricing and audit;
do not claim a GPU. Simulation and pricing are batched, JIT-compiled JAX
operations, with compiled scans rather than Python loops over paths.
The watcher waits for the requested origin and never requests a GPU in CPU
mode. It saves chunk-level restart state and per-date statistics under
`rolling/YYYY-MM-DD/unconditional_q/`, with an aggregate JSON summary in the
calibration root. No conditional policy is trained for these daily audits.

## Forward-start straddle hedging

`forward_hedging.py` generates 100,000 training and 100,000 independently
seeded validation paths over twenty business days, with eight Brownian
substeps per day. The claim pays `abs(S_20 / S_10 - 1)` in inception-spot
cash units. The primary Monte Carlo price and its antithetic-pair standard
error use the training sample; the independent validation price is reported
separately. These are simulations of the model's discretized Q dynamics,
without empirical return recentering or measure adjustment.

The fixed hedge book contains spot and one inception-OTM side per strike:
puts below one, calls at or above one. Expiries are 5, 10, 15 and 20 business
days, each with strikes `0.98, 0.99, 1, 1.01, 1.02`; the 20-day expiry also
has `0.95` and `1.05`. There are 22 options plus spot. Options cash-settle
once and cannot be traded at or after expiry. There are no transaction
costs. Daily marks, claim payoffs and P&L retain float64 precision.

The observable features contain time, spot, all fixed option prices, live
instrument masks, and the day-10 fixing once it is known. Additional latent
inputs are the live front density, sixteen spatial projections of current
log-DLV levels, maturity-level log-DLV means, cumulative past factor
Brownian increments and the independent Black background level. This is a
compact latent feature set, not an assertion that sixteen projections alone
are the complete Markov state. No future shock or future fixing is exposed.

Each action is an affine head plus a deep head, both applied to the current
features, previous holdings and LSTM hidden output. Only after choosing the
new holdings does the LSTM consume current features and old/new holdings
to update the next day's memory. The default numerical cap is ten units per
instrument, with a smooth tanh cap and a separate penalty beyond eight raw
units. Cap frequencies, the unpenalized economic objective and the numerical
penalty are reported separately; the penalty is not an economic trading cost.

For each risk aversion `lambda=1,10`, fit an empty-liability policy and then
a short-one-claim policy. The power-exponential utility is
`u(x)=(1-exp(-lambda*x))/lambda` for nonnegative `x`, and
`u(x)=x-lambda*x*x/2` otherwise. The training objective is its OCE,
`sup_m {m + E[u(P&L-m)]}`, less the numerical action penalty. Normalization,
checkpoint selection, learning-rate reviews and convergence checks use only
the 100,000 training paths. All four policies are frozen before opening the
separate validation sample. Final statistics use complete antithetic pairs;
the zero-liability control must not be interpreted as proving arbitrage
absence beyond its information set, strategy class and numerical scheme.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.forward_hedging \
  --calibration-root /path/to/quality-gated/repaired/calibration \
  --run-directory /path/to/external/unique/forward-straddle-run \
  --origin 2025-08-29 --paths 100000 --batch-size 256 --substeps 8 --wait
```

The rolling-model route waits for the repaired model's complete
`ar1.npz` and quality-passing report. It does not silently substitute an older
model or waive failed historical fits. HDF5 files checkpoint complete path
batches, and policy checkpoints retain optimizer, scheduler, RNG,
normalization, epoch and minibatch state. Maximum epochs are a safety
ceiling, not a convergence certificate. The default backend remains CPU.

The separately labelled saved-model route uses the earlier `(-3,2)`
affine-200 history and its fitted sixteen-factor AR(1) law. This was explicitly
selected for the first hedging experiment while the `(-4,3)` rolling repair
continues. It is not a result for the repaired rolling model. At each day's
start, the saved AR(1) intercept and slope predict log-squared factor scores;
the bounded, calibrated scales multiply the DLV loading columns, not the
Gaussian shocks or the front bridge. The realized daily factor scores update
the AR(1) state only after that day's transition. Physical drift coefficients
are not used under Q. Loading surfaces are interpolated in remaining maturity
each day. The AR(1) state and predictable scales are included in policy inputs.

```bash
JAX_PLATFORMS=cpu /home/hans/vqdata/bin/python \
  -m quantdata.gromm2.forward_hedging \
  --legacy-history /path/to/affine-200-history.h5 \
  --legacy-estimate /path/to/affine-200-models-rank16-final.npz \
  --legacy-factor-law /path/to/factor-laws.npz \
  --run-directory /path/to/external/unique/legacy-ar1-straddle-run \
  --origin 2025-08-29 --paths 100000 --batch-size 256 --substeps 8
```

Each scenario report includes unconditional screens of raw daily gains,
spot-scaled daily gains and inception-to-settlement option gains, using
independent antithetic pairs. Expired contracts are excluded, and settlement
is counted once. These pointwise screens diagnose finite-step bias; neither
the continuous-time Q drift nor a fitted covariance law guarantees that the
daily numerical scheme is exactly martingale.

Following the explicit revocation of the CPU-only instruction, the same saved
AR(1) experiment also supports `--backend gpu`. Its forward sampler uses JAX
CUDA and its hedge training uses `TrainConfig(device="cuda")`. Both require
allocation-provided `CUDA_VISIBLE_DEVICES`, refuse silent CPU fallback, and
perform a JAX float64/Torch gradient preflight after allocation. GPU RNG state
is included in policy restart files. The financial model and 5% background
are unchanged by selecting a different compute device.

`gpu_forward_queue.py` is a standard-library-only waiting parent. It retries
only the GPU manager's explicit busy responses and does not import or probe
CUDA while waiting. A uniquely identified GPU run directory keeps CPU and GPU
scenario files separate. Once allocated, it asks the previous owned CPU worker
to checkpoint and exit, then runs the GPU command through that allocation.
Unrelated GPU owners are never interrupted. Queue state and progress are saved
in `gpu_queue_status.json`; ordinary completion releases the claim. A resource
STOP halts the owned child within 60 seconds and keeps the claim pending
explicit allocation-specific release approval.

The optional `--linear-solver pcr` uses parallel cyclic reduction for the same
float64 symmetric tridiagonal systems; the default remains Thomas. A saved
full-path check uses identical Gaussian draws for CPU/Thomas and GPU/PCR, with
twenty days and eight substeps: option-price differences are below `1.5e-15`,
payoffs agree, and density conservation errors remain below `7e-14`. On the
allocated RTX 5090, a warmed 256-path batch takes approximately 3.2 seconds,
compared with roughly thirteen minutes for the sequential GPU solver. This is
a numerical implementation change, not a model, volatility or time-step change.
The faster production run has its own directory and code/config identity; the
earlier completed prefix remains separate. Provisional price and unconditional
diagnostic snapshots are written after the first batch and every 4,096 paths.

After an explicit user-requested stop, `hedging_policy --device cuda
--freeze-empty-risk-aversions 1` loads the existing lambda-one empty policy's
best checkpoint read-only and starts its payoff hedge from those weights.
This does not claim that the stopped policy converged. The lambda-ten schedule
remains unchanged, and validation stays closed until every policy is frozen.
Use a separate run directory with links to completed scenario files and a copy
of the stopped policy checkpoint to preserve the original run.
`plot_hedging_training --policy-directory ... --output ...png` plots completed
training epochs and any recorded validation curve, separates the numerical
penalty, and saves its source snapshot without reopening scenario files.

## Joint options-only, cost-stabilized experiment

`hedging_measure.py` independently fits strictly positive, minimum-relative-
entropy path weights for the existing 100,000 training and 100,000 validation
paths. Both members of an antithetic pair receive the same weight. All 290
live option/day cells enter twice, as raw price increments and increments
divided by pre-trade spot: 580 constraints. Settlement enters once; expired
contracts and spot trading do not enter the constraints. Standard-deviation
floors and covariance whitening only condition the solver. Every original
mean is checked afterwards; infeasible constraints are not silently recentered.
Reports retain effective sample size, weight concentration and residual means.

`joint_hedging.py` defaults to the original Q samples with equal path weights;
it does not read entropy-weight files unless `--preconditioned` is explicitly
requested. Optimization batches default to 256 paths, with learning rate
`0.00025`; population loss evaluation is still over every path. The retired
weighted run remains separate and cannot be silently resumed as unweighted.

The joint policy trades the 22 options but not spot. Spot remains an observed
state and the transaction-cost numeraire. Entry and subsequent live-contract
turnover cost `0.00001 * spot * abs(change_in_holdings)`; cash settlement and
resetting expired holdings are free. The ten-unit cap and the separate
numerical soft-cap penalty remain, with their effects reported separately.

The empty and exotic policies have separate affine-plus-deep action heads,
positions, LSTM states and OCE translations, but share the LSTM parameters.
They minimize the sum of their weighted power-exponential risk objectives at
each requested risk aversion (default `lambda=1,10`; use
`--risk-aversions 100` for the completed high-risk-aversion experiment).
The exotic is included as an unpriced short liability.
Uniformly sampled minibatches use global `N * path_weight` multipliers, never
batch-normalized importance weights. Feature normalization uses training only.

Both full-population losses are recorded every epoch. Zero trading is included
as an explicit epoch-zero validation candidate, with its OCE translation fitted
on training only. Validation joint loss selects the checkpoint, halves the learning rate after 25 stalled epochs, and
stops training after 50 without meaningful improvement. Curves refresh every
five epochs. There is deliberately no third test sample. The reported price
is `DH(Z)-DH(0)`, where `DH` is negative optimized OCE for the selected policy,
including actual turnover costs but excluding the numerical action penalty.
These are jointly restricted-policy, validation-selected estimates, not a
claim of globally optimal individual hedges or an untouched arbitrage test.

```bash
python -m quantdata.gromm2.joint_hedging --run-directory /external/unique/run --device cuda
```

For the optional empirical entropy-weighted variant, first run
`hedging_measure --run-directory ...`, then explicitly add `--preconditioned`
to the joint fitting command.

Launch CUDA only inside a `gpuctl` allocation. Each run has its own output
directory and links to completed source scenarios; no paths need regenerating.
Checkpoints preserve both policy heads, shared recurrence, optimizer, scheduler,
minibatch position, RNG states and the validation-selected best weights.

`hedging_uncertainty.py` evaluates the saved best checkpoint without training.
It computes optimized-OCE influence-function errors using complete antithetic
pairs, preserving covariance in `DH(Z)-DH(0)` and its difference from the Q
expectation. These are plug-in Monte Carlo errors for frozen policies, not
uncertainty from training or validation-based checkpoint selection.
The deliberately retained
`results/forward_straddle_lambda100_100k_2026_10_28.json` records the completed
original-Q run and its provenance. Training stops at epoch 63, restoring epoch
13; all subsequent overfitted checkpoints are excluded from the quoted price.

The next requested experiment uses one million training and one million
independent validation paths, with separate network weights for zero,
short-claim and long-claim policies. It first adds five options at day 10,
expiring at day 20 with fixed strikes
`S_10 * [0.98, 0.99, 1, 1.01, 1.02]` and the original ten-day book's OTM sides,
then repeats without these added contracts on the same paths. This is a new
experiment specification, not a completed result or a shared-backbone fit.

## Million-path forward-start comparison

`million_forward_experiment.py` supervises this experiment using the saved
legacy AR(1) dynamics and eight Brownian substeps per day. Each Black-smoothing
case has 1,000,000 training and 1,000,000 independent validation paths. The
5% and zero-smoothing cases use common random numbers; no historical DLV or
factor-law parameters are refitted. Setting smoothing to zero also reprices
the initial model book, so it is an ablation, not a calibration to the same
inception option surface. Both sample sets and the earlier 100,000-path runs
are retained separately.

Five ordinary cash-strike options are introduced at day 10 with expiry day 20:
`K = S_10 * [0.98, 0.99, 1, 1.01, 1.02]`. They are puts below relative ATM and
calls at/above ATM, matching the initial ten-day book. Their strikes stay
fixed after listing. They are not forward-start instruments tradable at
inception, and no holdings may earn their day-10 birth price. Cash settlement
is free; entering and changing positions incurs the same 0.1 bp spot-notional
cost as the original options. The target remains `abs(S_20/S_10-1)`.

`portfolio_hedging.py` first uses this extended book and then restricts trading
to the original book, on identical paths and with identical causal price and
latent-state information. Each of the zero, short-payoff and long-payoff
policies has independent LSTM/affine/deep weights, optimizer, scheduler and
validation-selected checkpoint. No neural parameters or empirical path
weights are shared or fitted across policies. They train concurrently on the
same minibatch schedule, with 256-path optimization batches and independent
validation-plateau stopping. Optional `--compile` accelerates their Torch
forward/backward graphs without changing the financial sampler.

The seller ask is `DH(H)-DH(0)` and the buyer bid is `DH(0)-DH(-H)`.
`quotes.json` reports each optimized risk, Q expectation, price and their
antithetic-pair Monte Carlo errors. The expanded bracket is
`[bid - 3 SE(bid), ask + 3 SE(ask)]`; the Q expectation also has a three-error
band. Differences use paired influence scores, preserving covariance.
These bounds exclude fitting and validation-selection uncertainty; they
are not a simultaneous confidence guarantee or certified global optima.

`hedging_data.py` maps completed contiguous HDF5 tensors read-only and creates
only minibatch-sized augmented views. `scenario_statistics.py` merges pair
moments once per simulated batch, avoiding repeated reads of growing path
prefixes. Raw paths, checkpoints, progress logs and loss plots stay in the
unique external run directory. Interrupts save a complete simulation batch
or complete three-policy optimizer batch; resumed stages validate identities.
