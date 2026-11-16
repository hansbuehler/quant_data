# Hedging Benchmark Implementation Plan

## Status

### Current execution and objective audit — 2026-11-14

This section supersedes historical queue descriptions below. The report's
completed 44-date DYSANOS v2-labelled STRT-16 results use the earlier nine-expiry
implementation, not the standalone 41-expiry density-mode DYSANOS v2. Do not
attribute those simulated forecast and stability metrics to the standalone model.
Earlier results, including DYSANOS v1 and GROMM, are blue.

The actual standalone V2 density-mode fit and ex-post straddle projection cover
all 44 dates. Their matched-date extracted results are under the external run
`standalone-v2-paper-metrics-20261114`: 103,686 same-day quotes, 85.8911% inside
spread, and straddle squared-change span capture of 94.0955% with spot plus
eight factors and 96.7522% with spot plus sixteen. Span capture fits coefficients
to each realized change; it is not a predicted variance or trading performance.
Matched one-day forecast covariance and tail metrics have now been calculated;
long-path stability and static-arbitrage tests remain outstanding.

The standalone V2 one-day forecast evaluation completed 44/44 dates on CPU under
`standalone-v2-forecast-nms-20261114`. It uses the shared NMS convention, the
standalone Gaussian factor/spot law and a deterministic conversion from its
dividend-adjusted spot return to the common one-day-forward numeraire. Earlier
preliminary results omitted the NMS offset and are not comparable. V2 source
surfaces use smoothness 0.1 while the common realized reference uses 0.25.
The conditional predicted variance divided by squared forecast error is 0.3025;
the pooled covariance Frobenius error is 0.4399. These are forecast metrics,
not the ex-post straddle-span measure above.

The code-agent batching review found independent GROMM view fits suitable for
larger chunks when the prior weight is zero. Benchmark 16, 32 and 64 on identical
inputs, reporting prices, gradients, time per view and peak GPU memory; each
chunk-size change needs its own checkpoint identity. DYSANOS minibatches each
perform an Adam step, so changing their size changes the optimization path.
First benchmark fewer GPU synchronizations and less frequent checkpointing while
preserving fail-fast checks, progress and restart semantics. No larger chunk is
adopted before a numerical and throughput comparison.

GPU allocation `e4190cbd46d7` belongs to this thread,
`01a12f09-4975-7431-a499-7c321ab35fa5`, “Hedging — DYSANOS and GROMM”.
`hedging-paper-gpu-completion-20261114.service` runs independent 44-origin
DYSANOS v1 and GROMM2 fits and releases the allocation after GPU work.
One CPU producer prepares GROMM quote views concurrently. Only completely
prepared origins enter GPU training. GPU-dependent GROMM P/Q forecasts precede
release; other statistics can run on CPU.

DYSANOS v1 uses a fresh run, `paper-dysanos1-rolling-20261114`, with observed
midpoint centering and the joint affine-200/five-session VAR. GROMM2 uses
`repository-gromm2-paper-20261114`, with the paper quote objective and explicit
repository geometry/dependence extensions. Neither loads retired-objective
weights or preceding-origin fits. Details are in `OBJECTIVES.md`.

Progress and exact errors are external under
`paper-completion-20261114/logs/completion.log` and `progress.json`.
Fit completion is distinct from forecast validity and no-arbitrage tests.
The retired intermediate v1 daily-embedding failure is retained without
eigenvalue repairs. V2's variance/MSE is 0.5084; its guarded static tests
have zero violations, invalid IVs or IVs above 300%, with maximum IV 38.16%.
This is not an unguarded non-explosion or dynamic-martingale proof.

### Takeover — 2026-11-13

**Latest instruction supersedes the frozen-standalone proposal below:** use
this project's original `quantdata` research engines, not the failed standalone
model repositories. Original-code run directory:
`/home/hans/quantdata_data/quant_data/hedge2/runs/original-rolling-20261113T1500/`.
Entry points are `model_audit.legacy_queue` (paper DYSANOS v1) and
`model_audit.legacy_v2` (direct STRT-16). All 44 April--May origins are fresh,
independent 250-session windows including the decision date. Two omitted
historical states are recreated from the identical saved smoothness-0.1
market fits, without extending or shortening a window. The source files and
original comparison reference remain unchanged. Recovery provenance is in
`RECOVERED_INPUTS.json`.

V1 uses the current paper's front/term-eta, 18-tenor, 3,620-control chart;
PCA initializes a market-refined affine-200 and full five-session joint VAR.
V2 uses the retained nine-tenor direct chart, STRT-16 and daily VAR/tail law,
with the complete spot-correlated native loading preserved. Both define P
only. DYSANOS v1 and v2 run on CPU. The original v1 wrapper hard-coded a CUDA
device, although neither the paper model nor its trainer requires a GPU.
The CPU continuation preserves completed native/affine stages, same-origin
joint optimizer/scheduler state, precision, objective, and convergence audit;
device-specific numerical/RNG trajectories need not be bitwise identical.
GROMM alone retains the managed-GPU queue after DYSANOS v1 completes.
Original GROMM v1 uses `sdlv/daily/train.py` and `front_spot`, not `gromm2`;
its causal-window integration follows the DYSANOS work. A queued fit is not
a completed or statistically validated model.

Execution checkpoint: v2 has 44/44 independent fitted and one-day-scored
origins with 1,000 scenarios each and no unusable IV paths. Its conditional
variance/forecast-MSE ratio is 0.5084 under the **original bounded decoder**.
Coordinate guards activate in almost every path; these are not unguarded
diffusion results or a static-arbitrage certificate. Per-tenor path rates,
grid-coordinate rates and origin-probability masses on clipped nodes are
saved in each result. No forward-grid scale clips were observed. The
Apr19 tail optimizer warning `ABNORMAL: ` is retained; independent projected
gradients at two finite-difference scales are below 1e-6, verifying local
accuracy without altering parameters. Model-specific result provenance is
separate from the common reference outcomes.

V1's first native and affine market fits are complete. Joint epoch 300 missed
the original trailing-window convergence gate (0.00479997 versus 0.002), so its
saved model/optimizer/scheduler/RNG state is continued in 100-epoch audit blocks,
preserving the original schedule horizon and accuracy thresholds. Completed
surface stages are not retrained. On November 14 the GPU worker was stopped
and its valid epoch-365 checkpoint backed up with its checksum and CUDA RNG
state. The queue now continues on CPU, without claiming or waiting for a GPU.
GROMM v1's first independent input window has been validated through the
original daily25 loader (250 dates, 853,514 separate market books); production
market training follows v1, with no preceding-origin weights. Other quote,
recursive stability and static-arbitrage diagnostics remain pending.
GROMM has also passed synthetic CPU checks of its full native-dimension
probability-PCA/P-Q drift estimation and both one-day bridge/repricing paths;
those synthetic outputs are confined to `checks/`, never production results.

Superseded preparation record (not an active production implementation):

The user requests local execution of DYSANOS v1-P and v2-P, followed by
GROMM v1-P/Q. GROMM v2 is not in this takeover queue. Use the same 44
April--May 2024 origins and independent rolling fits. The sibling DYSANOS
repository was deliberately cleared; do not restore or modify it. Runtime
source is copied from this project's reviewed frozen snapshot, with exact
source hashes and an explicit operational change to use the equivalent
compiled SANOS band solve rather than Python-unrolled elimination in v1.
Validate forward values and adjoint directional derivatives, then select
date batch size by measured full-geometry quote-gradient throughput.

Inputs are actual market bid/ask from the unchanged licensed smoothness-0.1
cache, not fitted-price targets. V2 daily quote-shape fits are deterministic,
independent single-date computations and may be memoized; each window's
PCA, AR and tail law is fitted afresh. No prior-window parameters initialize
any fit. Preserve numerical warnings and fit-quality gates; a saved model
alone is not an accepted fit or a successful forecast audit.

Entry point: `quantdata.hedging.model_audit.takeover`. Run artifacts:
`/home/hans/quantdata_data/quant_data/hedge2/runs/takeover-dysanos-gromm1-20261113-142518/`.

### Corrected 44-origin rolling comparison — 2026-11-13

The current production comparison uses exactly the existing 44 April--May
2024 origins and unchanged observable reference, outcomes, frozen diagnostic
PCA and statistical definitions. Retain only DYSANOS v1-P (the actual
market-refined affine-200, jointly refined five-session VAR), DYSANOS v2-P
(direct STRT-16), GROMM v1-P/Q (probability-weighted innovation PCA), and
GROMM v2-P/Q. DYSANOS has no invented Q construction.

**Independent fits at every origin; no warm starts for any model**, as
explicitly requested on 2026-11-13. No previous rolling date's fitted weights,
all-history checkpoint or later-origin fit initializes these windows. A saved
checkpoint from the same origin remains a resumable calculation, not a rolling
warm start. DYSANOS uses 250-session windows under the standalone specification;
each model's exact dated inputs are retained with its checkpoint.

The GPU queue is fitting GROMM and will dispatch ready DYSANOS origins inside
its allocation-owned process tree. DYSANOS v2 daily quote-shape calibration
runs separately on CPU. DYSANOS v1's initial 128-date GPU batch exhausted VRAM
before its first update; retain the exact error and retry with batch size 16,
without reducing the observations, model dimensions or training stages.
Fit completion is not statistical validation. Evaluate all six cases on the
same reference before updating comparisons; then run the requested
no-arbitrage tests for all models. Historical pilots below remain controls,
not substitutes for the rolling results.

Production artifacts and logs:
`/home/hans/quantdata_data/quant_data/hedge2/runs/rolling-six-cases-44origins-20261112-v1/`.

### Model correction execution — 2026-11-12

Code and focused mathematical tests now implement the actual 200-state paper
adapter (`model_audit/paper_v1.py`) and the standalone GROMM1 probability-PCA
control (`model_audit/gromm_probability.py`). The authenticated paper checkpoint
has a positive daily covariance: minimum eigenvalue 0.0009287881, no projection;
five daily steps reproduce its fitted weekly covariance to numerical precision.
Its one-origin, 64-path decoder pilot passed. The 44-origin April--May evaluation
completed with 512 antithetic paths per origin: predicted variance trace
0.494684, forecast MSE 1.684396, ratio 0.293686. The available checkpoint fits all 891
dates, so this is explicitly **in-sample parameter evaluation**, not a corrected
causal rolling out-of-sample backtest.

The GROMM1 probability-weighted refit completed on its saved 2025-08-29 origin,
using the unchanged 247 historical pairs, calibrated representation and
independent-Y likelihood. Weighted-metric innovation retention increases from
90.9079% to 91.8520%; native-coordinate retention changes from 90.8956% to
89.3852%. Compatible P/Q drifts are finite and recomputed. These retention
numbers are not decoded-IV variance or forecast-MSE results. A matched 512-path
P/Q decoded-IV comparison uses `model_audit/gromm_probability_forecast.py`.
The validated weighted inference archive is in `gromm1-pweight-v2/`; the initial
export omitted its new latent-initialization schema marker and is not the
retained inference result. The corrected writer verifies a round trip through
the unmodified production loader before declaring success. The matched decoded
comparison is in `gromm1-pweight-forecast-v2/`. This single-origin experiment is
not the older April--May GROMM2 cohort. The decoded comparison completed all
four baseline/weighted P/Q cells, 512 paths each, with no invalid paths. Q
log-IV variance trace changes from 1.481412 to 1.521679 (+2.72%); P changes
from 2.230006 to 2.758307 (+23.69%). Probability weighting alone therefore
does not reduce decoded variance on this pilot. No forecast-MSE ratio is
claimed because a matched next-session outcome is not included in this
single-origin control. These are descriptive Monte Carlo comparisons, not
new no-arbitrage certificates or a multi-origin calibration result.

Jobs, logs, protocols and numeric outputs are isolated below
`/home/hans/quantdata_data/quant_data/hedge2/runs/model-corrections-20261112-v1/`.
Source changes stay on `hedging-benchmark` in
`/home/hans/python/quant_data-hedging`; the user's main checkout is untouched.

### Execution order and GROMM weighting control — 2026-11-12

First evaluate the actual paper affine-200 v1 and the agreed new SANOS v2;
do not run their static-arbitrage tests, or GROMM static tests, ahead of that
variance/model-identity work. Independent FuNVol, VolGAN and AD-Seq-Vol/FT
static tests may proceed in the background on the requested 44 origins,
1,000 paths and shared 1/5/10/125/259-session snapshots.

The current standalone GROMM1 does select log-DLV innovation directions by
PCA, followed by an independent-Y-compatible latent dependence fit and a
residual-factor truncation. Its temporal observation weights are not node
probability weights. A proposed controlled probability-weighted PCA ablation
must preserve the calibrated representation and factor budget, unscale the
loadings back to native coordinates, retain the independent-Y observation
law, and recompute the compatible P/Q drift fields. It must not simply
attenuate volatility or change the pricing/data convention. Probability
weighting need not reduce decoded variance.

The GROMM-P/Q ratios already reported here came from the retained older
hedging GROMM2 artifacts, not the new standalone GROMM1. Do not transfer
those results or labels between the implementations. The v1/v2 SANOS
correction remains the first priority.

### Paper-model identity correction — 2026-11-12

The current audit row labelled DYSANOS v1 is an eight-PC, separately fitted
one-session PCA/VAR surrogate, **not the paper's market-calibrated affine-200
model with jointly refined five-session VAR dynamics**. The eight decoded-IV
PCs in the paper are descriptive invariant-distribution diagnostics, not a
replacement for its 200-dimensional state process. Do not attribute the
surrogate's weak variance calibration to the paper model, or present its
weighted-PCA ablations as a controlled replacement for the paper calibration.

The mismatch is confirmed from the paper, training code and checkpoint tensor
dimensions. The correct paper-model variance/forecast-MSE ratio has not yet
been computed here. Hold affected paper-v1 comparisons and reruns until the
adapter is corrected; preserve existing arrays as explicitly labelled
surrogate controls. A full-history paper checkpoint also cannot supply a
causal rolling out-of-sample result for dates inside its fitting history.
See `quantdata/hedging/MODEL_IDENTITY_REVIEW.md` for evidence and the existing
ratios. This correction supersedes the model-identity interpretation of the
historical entries below; it does not retrospectively change their numbers.

### SANOS v2 — agreed representation and dynamics

SANOS v2 uses the observed market ATM-volatility curve to determine grid and
backbone scale through the existing SANOS construction. Smoothness is an
explicit input, default **0.1**, in the core SANOS convention; it is not an
independently learned front/later-term backbone multiplier. Fit DLVs against
the unchanged externally defined quote population, without allowing a shrinking
grid to remove difficult quotes from the fitting or reporting population.

ATM remains **stochastic**: estimate one-business-day ATM and DLV-shape changes
with 16 surface factors, default STRT metric and Tail Diffusion driver, retaining
spot/Y dependence. Gaussian diffusion remains an alternative. In default JOINT
mode, ATM and DLV-shape coordinates share a factor basis. Optional SEPARATE mode
allocates the same total factor budget between ATM and residual DLV-shape
blocks, retaining regression cross-loadings and a joint innovation covariance.
In both cases ATM shocks move the grid and backbone through the fixed mapping;
there are no additional free smoothing factors. The separate-mode rank allocation
has not yet been agreed.

Requested exports are 201 relative strikes including exactly one ATM strike 1,
NMS [-4,3], at business days 1--20 and 25,30,...,125 (41 maturities). This differs
from the standalone package's existing 45-tenor calendar and the nine-tenor
legacy audit. The implementation request and findings are in
`/home/hans/python/openqdata_models_dysanos/HANDOVER.md`.

This is the agreed **new specification**, not an implemented or validated model.
The earlier DYSANOS v1/v2 labels identify retained affine-200/eight-PC and direct
STRT/sixteen-PC experiments; their results must not be relabelled as SANOS v2.
The historical results below remain evidence for the design, not new-v2 results.

### Model-dynamics audit follow-up — 2026-11-12

The 44 April–May 2024 origins remain the common comparison window, with
252 causal one-session transitions per fit and a fixed 16-factor innovation
budget. The retained daily-state DYSANOS-labelled audit fits projections and
VAR separately; it is not a jointly trained JAX affine/dynamics model.

The explicit ATM-derived grid experiment is complete: three grid-volatility
factors plus thirteen residual DLV-shape factors. Its spot-orthogonal forecast
RMSE improves from 0.1195 to 0.1162, but conditional predicted variance divided
by realized forecast-error variance remains low, improving from 0.506 to 0.525.
The grid factors describe the ATM-derived forward-volatility curve, not pure
decoded ATM-IV shocks.

The completed first controlled comparison learns the affine level projection using
training-window mean grid probability masses before fitting factor changes.
It uses rank 200 and retains an unweighted affine-200 control. Separately learn
16 change directions using local option/density sensitivities: ITM premium
returns, straddle changes per initial spot, straddle changes divided by
initial spot and square-root maturity, and absolute probability-mass changes.
The latter are not relative returns divided by tiny probabilities. Reprice
on each origin's fixed spot, cash strikes, tenors and probability grid, with
spot dependence retained separately. Fit ordinary and spot-orthogonal versions.

All twelve projection/spot variants have completed all 44 origins: 528 scored
origin-model cells, without invalid forecast surfaces. The ordinary weighted
affine-200 control has forecast RMSE 0.1242 versus 0.1742 unweighted. The ITM
premium-return metric reduces ordinary RMSE to 0.1139; raw and square-root-time
scaled straddles give 0.1157 and 0.1156, and probability changes give 0.1231.
These are local-sensitivity projections, not nonlinear return-loss calibration.
Predicted variance divided by realized forecast-error variance remains about
0.45–0.53 for the weighted/observable alternatives. All fitted unconstrained
VAR spectral radii are below 0.957; no stability rescaling was needed. Native
decoder clipping persists and is retained rather than silently dropped.

The full ordinary/orthogonal comparison, checkpoints, 304 checked input
hashes, Jacobian half-step checks and validation ledger are stored under
`$QUANTDATA_ROOT/quant_data/hedge2/runs/model-dynamics-audit-20261112-projection-metrics-v3/`.
The earlier narrow-strike v1 pilot is superseded by this full-native-strike
comparison and is not a reported model result. Revision v3 also preserves the
entire native spot/surface regression loading: the earlier reduced-coordinate
simulation had dropped its component outside the affine span. This affects
the orthogonal forecast rows, not the ordinary rows or learned projection
coefficients. A regression test explicitly checks that missing contribution.

### Fixed-instrument return variance explained — 2026-11-12

The log-IV forecast RMSE above is not the fraction of option-return variance
represented by factors. A separate evaluation holds each origin's cash
strikes and terminal expiries fixed for one business day, with actual
normalized spot movement, remaining maturity reduced by one day and exact
settlement of 1D instruments. Both endpoint surfaces come from the same saved
0.1 decoded daily source, removing the previous 0.1-versus-0.25 reference
mismatch for this attribution. Straddle gains are divided by initial spot
and square-root initial maturity. Factor bases and encoders are fitted
through the origin only; next-day realized native changes supply factor
values, with no hindsight least-squares coefficients.

Report one minus centered residual-return variance divided by centered
realized-return variance, across the 44 origins and all nine-by-200 source
coordinates. Also report the same fraction after removing exact frozen-u
spot/time repricing. The latter avoids hiding surface-factor losses behind
spot, whose frozen-surface repricing alone explains 98.90% of total variance.
Separate all-node, central NMS (-2,2), surviving-expiry and tenor-level
summaries are retained, including every prefix from zero to sixteen factors.

Before the sixteen-factor change compression, unweighted affine-200 retains
98.02% of surface-related return variance and probability-weighted
affine-200 retains 99.97%. After sixteen change factors, the ordinary
weighted-native and weighted-affine models retain 96.49%; the separate-grid
model retains 95.81%. ITM premium-return factors retain 99.55%, unscaled
straddles 99.51%, square-root-time-scaled straddles 99.55%, and absolute
probability-change factors 96.60%. The unweighted-affine/Euclidean-change
pipeline retains only 25.30%. Orthogonal return-based variants are similar.
These are reconstruction results, not evidence that the simulated factor
innovation distribution has the correct forecast covariance.

Results, per-origin arrays, all input checksums and pipeline-stage scores:
`$QUANTDATA_ROOT/quant_data/hedge2/runs/model-dynamics-audit-20261112-fixed-straddles-v3/`.
Twenty focused regression and mathematical tests pass. Numerical guard clips
are retained explicitly; finite decoded prices do not establish a good fit.

### Fitting effort and the direct-straddle baseline — 2026-11-12

A single-thread CPU timing study uses three representative rolling origins,
one warm-up and four measured repetitions per origin. With saved daily states
resident in memory, direct probability-weighted PCA plus VAR takes a median
0.0160 seconds per fit; adding weighted affine-200 takes 0.0266 seconds, or
1.66 times as much. Ordinary PCA takes 0.0182 seconds, the same practical
cost class. The four standalone return/density-metric fits each take about
0.672–0.675 seconds, or 42 times the weighted-native fit, including the
native-span decomposition and both Jacobian step sizes. Those four metrics
can share that expensive preparation. Original surface fitting, data loading,
checkpointing and forecast evaluation are excluded. The raw repeated timings
are in `model-dynamics-audit-20261112-effort-v1/TIMINGS.json` beneath the run root.

On the 44-origin return-reconstruction test, direct weighted PCA and weighted
affine-200 followed by weighted-metric PCA retain 96.4883% and 96.4903% of
surface-related straddle-return variance with sixteen factors. With eight,
the corresponding fractions are 95.3714% and 95.3737%. This does not justify
the extra affine level-compression stage for this saved-state dynamics
experiment; it is not a conclusion about jointly calibrated DYSANOS market
representations. These are causal out-of-sample reconstruction results using
realized factor values, not in-sample fits or forecasts.

The next variance-forecast control is direct PCA of scaled fixed-instrument
straddle returns, without DLV/native coordinates. Build a price-only panel
using the saved 0.1 SANOS market interpolation, fixed strikes/terminal dates
inside each one-day pair and the same causal 252-transition windows. Report
total returns and the residual after frozen-surface spot/time repricing
separately. Start with a rolling PCA covariance forecast and a full empirical
covariance control; this baseline does not require a VAR on changing option
contracts. Keep panel-construction cost separate from PCA fitting effort.

For forecast calibration, report forecast covariance trace divided by realized
forecast-error squared norm, using forecast means estimated through the origin
only. Also report pooled forecast variance versus centered realized variance.
Ratios can exceed 100%; they are calibration ratios, not variance-explained
scores. This direct observable-return baseline has not yet been run.

Use the existing decoder, paired historical spot/factor law and common
49-coordinate floating log-IV forecast tests. Keep weights and projections
causal; retain Jacobian half-step convergence diagnostics, clipping, fit
coverage and per-origin checkpoints. No ex-post factor-span score substitutes
for predictive covariance calibration. The source-state smoothness 0.1 versus
the common reference smoothness 0.25 remains explicitly disclosed. New
recursive-path and quote tests follow only after the one-day comparison.

### Rolling 1D stage — 2026-10-17

The first rolling launch stopped because the 1D target selection received
quantities for the full 20-target family. This was corrected and verified on
one real origin. The second launch produced 60 covariance origins and 12
Historical-SANOS origins before the latter stopped at 2023-02-02: a
$1.8\times10^{-18}$ zero-claim action was treated as an active coordinate by
the nonsmooth KKT certificate. A roundoff-scale zero test now certifies that
origin; the exact solver warning is retained in the diagnostic run. The
Historical-SANOS entropy, risk-1 rolling cell restarted in an isolated run.
Neither the failed launch nor the 13-date replay is a full rolling P&L result.

The active scope is now rolling one-business-day digital targets only, over
2023–2025, using 252 historical transitions available through each origin.
Do not launch 3D claims or weekly/restarted-weekly experiments in this stage.
The earlier status entries below are historical records, not launch instructions.

Compare all four objectives and the agreed risk grids across non-LV models,
with demeaning and feasible exact MEMM. Historical empirical MEMM infeasibility
is an explicit outcome, never replaced silently by approximate weights.
Generative books use at least max(10,000, 10 times instrument count) scenarios.
Use the public-candidate dataset exclusively, common target sizes and costs,
and retain dated model/scenario identities. Resolve model compatibility and
regularization discrepancies before interpreting cross-model comparisons.

Launch ready CPU cells first, with bounded concurrency; launch generated cells
only with compatible artifacts and managed GPU allocations. Save each completed
origin and monitor convergence, exact warnings, zero-claim positions, gross and
maximum exposure, cap hits, costs, and MEMM residual/ESS/weight concentration.
Investigate problematic cells without silently relaxing numerical gates or
selecting settings using realized future returns.

Report rolling out-of-sample net P&L, tracking errors, drawdowns and the four
objective values by model/objective/risk/mean reduction/target, with coverage
and failed or unavailable dates explicit. The three diagnostic dates establish
portfolio reasonableness only; they are not standalone performance evidence.

This document is the implementation plan for the hedging benchmark built around
the daily front-spot SDLV/GROMM model. It records the agreed experimental design
and implementation order.

**Current live state (2026-10-24).** The distributable public-candidate dataset
is the default benchmark input. The temporary prohibition on using original
source data for regeneration has been lifted; any such use must retain separate
provenance and must not put original records into released artifacts or Git.
The dataset's technical release gate is complete: the
artifact contains 891 dates and 2,014,740 hedge coordinates, reconstructs from
an otherwise empty data root, and contains no original WRDS prices, midpoints,
or identifiers. Redistribution approval remains a separate legal question.

The current immutable source is commit
`8b7ca46c73ddf278ebcb23b7ddcb4eb9b12b8edf`. Historical-SANOS version 1.22
retains the precommitted 252-transition window and $10^{-4}$ ridge, distinguishes
certified numerical solutions from genuine recession directions, and applies
independent final-fit certificates without relaxing the economic objective.
Independent quantitative review is recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-16-historical-sanos-v122-solver-review.md`.
The active CPU-only daily and weekly matrices have SHA-256 values
`b99b930c5c76b5ec683ca784fcf19362649127a74565b40477b2dd6534976c5a`
and
`84824c192c9ef0ac5a275232ed75fab3de89b28f38f4342b16365edd3c1eb2a4`.
Their launchers run four cases each, emit four-minute progress, and currently
report no unexpected numerical failures. All three daily mean--variance cases
are complete. One daily and four weekly mean-volatility cells have authenticated
recession directions; these are economic non-existence results, not solver
failures. Independent runtime review recommends leaving the active four-worker
queues unchanged and is recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-16-historical-v122-weekly-runtime-review.md`.
The reviewed handoff waiter will authenticate all terminal cases and their
failure ledgers before publishing `handoff.json`.

Service `quantdata-hedging-generated-readiness-8b7ca46-v122-full.service`
completed the exact three-model by six-scope readiness matrix with 4,096 CUDA
scenarios. All eighteen scopes pass their numerical gates. Eight weekly scopes
retain fitted-policy cap warnings, but there are no executed cap hits,
nonconverged fits, recession directions, or failed scopes. The immutable
completion manifest has SHA-256
`e6a00b1c56af607c50f3b5ab91b492c2f849b45421c8175e402a6c56073c2175`.
Independent review passes at SHA-256
`a5e66746fd118e6fcb1e24f6f631f8206cf0344a0946c58299cf004451901e12`
after reconstructing all eighteen scopes, the exact 308-artifact inventory,
and every referenced checkpoint and result hash.

The r2 final-production coordinator is frozen at SHA-256
`981734321acd116b2b7afb7239840558e4f9c670543be843e3b8e9ff95553590`.
Independent review passes at SHA-256
`8cd4391583f29b09a59022c78351b93c3821dedcaf5f80b9a16dc56d5081db43`:
all prerequisite authentication, exact 57/94 population, source-immutability,
resource, process-group, signal, and report-provenance tests pass. This revision
pins the accepted r7 generated-readiness trust root and uses isolated r2 output
roots without changing any production gate. The coordinator remains correctly
fail-closed until the final Historical-SANOS handoff exists and its exact hash
is supplied externally; the generated-readiness manifest and review are already
published and authenticated. No final matrix commitment or production result
has been written yet.

**Superseded state (2026-10-15).** The earlier generated-model readiness
passes all 18 scopes (three models times six configurations) at immutable commit
`b20dea5bae0ebdce3c698565d74e4de2c729824f`, but its weekly evaluation and
record artifacts use schema v2 and are not admissible under the current schema
v3 source. Independent quantitative review rejected a split-commit executor.
Service `quantdata-hedging-generated-readiness-fe7c3a7.service` therefore
reauthenticates the unchanged model/checkpoint lineage and reruns all 18 scopes
from immutable commit `fe7c3a7321345e8ced636c5f75518766b830e452` with 4,096
CUDA scenarios, strict cap/convergence/recession gates, and four-minute progress
logging. Its detached preflight and 72 focused generated-weekly, gate, matrix,
and executor tests pass; the service currently waits for a managed GPU. The
installed readiness launcher SHA-256 is
`c8edb76fdbfa5f45dec274d24e9fc37466400ac381f513b9dd8bfafd6c5363aa`.
The current factor estimator invalidated the earlier 2022 GROMM architecture
study because `RollingDlvFactorConfig` changed materially from version 1.1 to
1.2. A fresh CPU-only study from immutable commit `fe7c3a7` and the public
dataset completed all four prescribed candidates. Every fit converged with
zero cap hits; the one-standard-error rule selects
`gromm-architecture-2022-h64-d3-w128`. The authenticated result SHA-256 is
`ed68dd71a87465507191f476cf85eea691596751fa122d95ac21c7381abb166b`.
Independent pinned-worktree matrix builds report all 27 daily and 54 weekly
GROMM cases ready. Quantitative review is recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-15-gromm-architecture-remediation-review.md`
(SHA-256
`687f9b4ee61eb101318fb20e20364ee8af2d3cd9f6836d6e26ababd57342e96e`).
The first full v1.15 Historical-SANOS sweep then exposed two additional daily
mean-variance points whose OSQP and CLARABEL solutions missed the unchanged
$10^{-6}$ KKT gate, plus a genuine low-risk mean-volatility zero-policy
recession direction. Version 1.16 adds an exact cyclic coordinate refinement of
the same scaled box-constrained lasso QP only after both primary certificates
fail. Public-data replays now pass all 20 targets at the two affected
mean-variance origins with zero cap hits and nonconvergence. Daily failure
evidence separately counts claim-policy recession fits and the one shared
zero-policy fit, so mean-volatility $\lambda=0.1$ is classified as an
inadmissible objective setting rather than a generic solver failure.
The complete hedging test tree passes 433 tests, and independent quantitative
review passes after correcting the shared-fit count. Evidence is recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-15-historical-sanos-v116-coordinate-remediation.md`.
Historical-SANOS version 1.16 is reviewed, passes the complete 433-test hedging
suite, and is committed at
`1b68b34d8c6ad0e83b65ee26c99dfa0f94c22481`. The superseded v1.15 service
chain stopped cleanly before replacement. CPU-only services
`quantdata-hedging-historical-v116-daily-1b68b34.service` and
`quantdata-hedging-historical-v116-weekly-1b68b34.service` now run all 13 daily
and 26 weekly cases from the clean pinned worktree. Their daily and weekly
matrix SHA-256 values are
`0d250de389ca5f2eac450a9adbcc3516440760588595ebed00790366ade88acd` and
`fa522ced64a389090dda7610370a0f98bcbf34f64e832f50997d4f3a76770ef2`;
their launcher SHA-256 values are
`82c7c110a961bee18b0758fbcd12ef3ffbe84f017857060858b19141a512237d` and
`09714ca7f8950836c4d63bfc631640140bcba693b94af7d27b09c3597c721c54`.
Service `quantdata-hedging-failure-canonicalization-v116-1b68b34.service`
waits for those queues and will write authenticated failure ledgers plus the
hash-bound v1.16 handoff; its launcher SHA-256 is
`79d4996e6e471d8090c836212d3130e997575b09e9a3afe1c15c89c19e3ec6e1`.
Coordinator `quantdata-hedging-full-production-v116-1b68b34.service` passed its
detached preflight and waits for that handoff plus the unchanged generated-model
readiness service before composing the final 57 daily and 94 weekly cases. Its
launcher SHA-256 is
`404894e63a1be7329b4d917ad65fe7c98668c5d79d74710f8109a631953aed5b`.
The three exact-Gaussian covariance power-exponential cases are complete. A
conservative storage cleanup
removed approximately 315 MiB of superseded target-notional runs, early GROMM
smoke artifacts, old readiness revisions, and reproducible compiler/temp files.
The 51 GiB legacy GROMM tree remains because current authenticated migration
manifests bind it explicitly as their source lineage.

**Current authenticated production chain (2026-10-14).** The `d22ec9a`
generated-readiness run halted after numerically completing the first weekly
AD--Seq--Vol scope because its validator compared the canonical matrix identity
`diffusion-ad-seq-vol` literally with the weekly adapter identity
`ad-seq-vol`. No production matrix commitment had been written. Immutable
commit `b20dea5bae0ebdce3c698565d74e4de2c729824f` now maps only the two
diffusion matrix identities to their weekly runtime identities during
authentication, while reports continue to export the canonical taxonomy. Both
complete hedging test trees pass, with 464 tests in total. Independent review is
recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-weekly-generated-model-identity-review.md`
(SHA-256
`c8c8e0c9864da6fbddf4be2d21c9c00e2a1632191a63f47e3b47f73ffc20a5f9`).
Service `quantdata-hedging-generated-readiness-b20dea5.service` owns the managed
GPU and reauthenticates every one-origin readiness scope. Its seeded run and
checkpoint files are byte-identical to the prior tree and remain inadmissible
until the new validator accepts them. Its reviewed launcher SHA-256 is
`8c1ed51e45dc2588ce50294b066e2e22ae3597bbc7fa4a94196c37f515fd009a`.
Service `quantdata-hedging-full-production-b20dea5.service` waits for that
readiness evidence and the remaining native prerequisites before it may compose
and commit the daily and weekly matrices. Its reviewed launcher SHA-256 is
`e6de16d5c645f1ae78e3f91b51376ff6496d029d1d5ae8066d8f72cb14207774`.
The superseded coordinator was stopped before commitment.

**Current Historical-SANOS numerical design (2026-10-15).** Version `1.14`
retains the precommitted 252 uniformly weighted daily scenarios and increases
the fitting ridge to $10^{-4}$ in unit-RMS instrument-return coordinates. It
does not introduce a two-year or exponentially weighted daily window. Books
with fewer than two observations per fitted action parameter are explicitly
flagged as statistically thin and interpreted only as regularized estimates.
Daily exact-cost fits use independently recomputed signed-action KKT residuals;
mean-volatility receives an unchanged-objective exact refinement when the conic
point misses that certificate. A negative homogeneous zero-claim
mean-volatility objective is recorded as a recession certificate and cannot be
accepted merely because the numerical ridge makes the surrogate finite. The
same test is applied to the weekly zero-claim pricing policy and is propagated
through resumable records into a dedicated strict-failure classification.
Weekly mean-variance certifies the raw OSQP point and retries a failed
certificate with CLARABEL. Other weekly conic fits reconstruct the exact
objective and check primal feasibility; if the primary solver remains
`optimal_inaccurate`, an independent SCS solve must report `optimal` before
acceptance. Selection fits remain diagnostics, while the certified full-window
fit controls production acceptance. Any remaining real failure stops that case,
emits its numerical evidence, and requires independent quantitative review
rather than a silent tolerance change. Higher-frequency histories are deferred
to a separate follow-on model revision.

**Current production correction (2026-10-14).** Historical-SANOS version
`1.11` makes the final 252-observation fit, rather than its 80% warm-start
selection fit, authoritative. This invalidates the Historical-SANOS `1.10`
records and supersedes the older Historical-SANOS service references later in
this status history. Fresh commit-pinned daily and weekly matrices now run from
`cd6f440626a59c58196880f47265a3f041962ec4`; their SHA-256 values are
`68a8edb5cbcafaae91abbb056ae48fc2a9818d02e5407487bd02169287beb0c0` and
`e465f1594b5904b2c9c1a7beaa363b5df9c6dd062ada4ab8d880ea12f1dee5c0`.
Services `quantdata-hedging-historical-v111-daily-cd6f440.service` and
`quantdata-hedging-historical-v111-weekly-cd6f440.service` own all 13 daily and
26 weekly Historical-SANOS cells, respectively. A strict entropy-risk-`1`
replay at `2023-06-13` passes all 20 targets with zero cap hits and zero
nonconvergence. Entropy risk `0.1` and one entropy-risk-`10` target retain their
genuine final-fit failures and must not be relabelled as passing production.
Service `quantdata-hedging-failure-canonicalization-8851996-v2.service` waits for
both matrices, then validates completed cases and replays only incomplete or
failed daily/frozen/restarted cells under the structured gate-artifact runner.
It emits terminal queue evidence and separate authenticated daily and weekly
failure ledgers. Its immutable launcher SHA-256 is
`70bd69a679fd9cef58aba1087c84e49f35f8db796e4cfde8176e17df47252846`.
Both the frozen- and restarted-weekly SANOS-LV references retain a strict
failure at origin
`2023-02-08`: all 13 targets load the same two one-day calls to the 1,000-unit
execution cap after those frozen strikes drift to approximately `+3.46` and
`+3.68` current NMS. The replay has zero fit-cap hits, so this is an executable
conditioning/economic-universe failure rather than an optimizer-fit failure.
Production reporting now treats this outcome explicitly rather than making the
completion criterion impossible or relaxing the precommitted gate. Immediately
before a strict numerical gate raises, the runner atomically writes a canonical
`hedging-numerical-gate-failure-v1` artifact containing the immutable run
identity, failed origin, exact gate diagnostics, and derived classification.
The queue pins the path and SHA-256 of the artifact generated by that exact
failed subprocess attempt; it does not accept an unchanged artifact left by an
earlier attempt. A fail-closed `hedging-production-failure-ledger-v2` then
authenticates the terminal `complete-with-failures` queue, normalized source
command, strict `run.json`, dataset manifest, failed record, and pinned gate
artifact. The report excludes authenticated failed cells from performance
estimates, displays them in a dedicated production-outcome table, and rejects a
stale failure if any valid completed result is available. Matrix composition
therefore preserves mixed successful/failed model grids while preferring a
strict completion over an earlier failure regardless of source order. Focused
tests and independent quantitative review pass. Both strict replays are
terminal `complete-with-failures` and their queues pin the freshly generated
gate artifacts. The combined canonical ledger is
`/home/hans/quantdata_data/experiments/hedging/weekly/native-b37a643/failure-ledger-65fbf1d.json`,
SHA-256
`2dcbece08a86939ead37f17fd03f02ce5505b8a32f4cd562104bc3cc46c13505`.
The real replay also verifies the explicit reporting-to-operational measure
mapping from `Q-price/historical-transition` to the runner's `Q`; an unrelated
P/Q mismatch remains rejected by regression test.
A current-NMS counterfactual at the failed origin has zero cap hits, but it is
not production evidence because changing eligibility after episode inception
would change the precommitted frozen-book strategy.

The distributable public-candidate data layer and CPU accounting vertical slice
are implemented and passed independent quant review. The SANOS-LV Phase-2
implementation has rolling 252-transition fixed-contract DLV factors, eight
factor vegas, the complete 20-target daily digital family, stable next-date
synthetic contract replay, minimum-predictable-option-cost factor hedges, and a
residual spot-delta hedge. Its independent Phase-2 review and 160-target
multi-date CPU pilot passed. The independent covariance baseline now uses an
instrument-only shrinkage Tyler geometry, target-specific robust cross-moments,
sample-mean/HAC drift screening, exact mean-volatility recession checks,
Gaussian entropy equivalence, and an exact Gaussian power-exponential OCE;
its Phase-3 implementation and independent re-reviews passed. The rolling
rank-16 SDLV calibration, factor alignment, eight Greek factors, analytical Q
drift, and compatible P-drift estimation passed the Phase-4 remediation review.
The Phase-8 weekly accounting engine now also passes its independent mechanical
review. Common SANOS/DLV weekly claim prices, exact rolling fit identities,
precommitted as-of information windows, and Hansen--Hodrick plus moving-block
overlap inference passed their independent re-review. A resumable daily runner
now pins dataset and model identities, enforces the complete 20-target book and
numerical gates, and writes one atomic record per origin. Earlier SANOS-LV and
historical-SANOS runs each completed all 611 eligible origins and 12,220 target
records. Those records used superseded target-notional identities and remain
diagnostic rather than being relabelled as current production evidence. The
clean current-notional SANOS-LV rerun is complete: all 611 origins and 12,220
records pass the current record-by-record validator, with zero cap hits and zero
nonconvergence, and deterministic reconstruction reproduces the pinned
production-results SHA-256. The three corrected robust-covariance
mean--variance cases are also complete and authenticated on the same current
target-notional cohort, with zero cap hits and zero nonconvergence. The corrected
Historical-SANOS mean--variance case at risk parameter $0.1$ is likewise complete
and authenticated for all 611 origins and 12,220 records, with zero cap hits and
zero nonconvergence. The remaining robust-covariance and Historical-SANOS sweeps
continue from immutable worktrees. The remaining Historical-SANOS launcher keeps
four CPU cases active with a work-conserving queue: when any case finishes, the
next precommitted objective starts immediately rather than waiting for the other
three cases in its original batch. Existing atomic origin records are reused,
while the model code, production matrix, case order, and four-worker limit remain
pinned. The legacy `0b8d20a` launcher is restricted to its two unfinished
mean--variance cases at risk parameters $1$ and $10$; the already authenticated
$0.1$ result remains unchanged, and that service no longer launches corrected
mean--volatility, entropy, or power-exponential cases owned by the newer matrix.
The persisted launcher SHA-256 values are
`86ef24fe3770ebd8f35e5dc3b9e85ad499803a5dabda68dc67439c671a8c9db8` and
`223d5132eaa9f480062b6b23cbdc5598b69af9f65215136d18effaeb39ce5eaf`,
respectively.
Other daily production and full weekly production remain outstanding. The six
Historical-SANOS entropy and power-exponential cases from the older `edf28cc`
cohort remain failed diagnostics: their exact-cost optimizer could reject a
valid solution after the split-variable solver stalled. The corrected
implementation at immutable commit
`cff579c38c489e315b624d055d4336e80554cd58` uses the original-action KKT
residual as the acceptance test and one canonical signed restart without
changing the objective, costs, bounds, or tolerance. Independent review passed,
and service `quantdata-hedging-historical-exact-cost-cff579c-v2.service` is
producing the six fresh cases concurrently. Its concurrency-only restart reused
97 atomic origin records; the launcher SHA-256 is
`425a95f3a7da39e890b637c9aae0938e42e1feb15ed65b6b213f8e029a87523d`.
The three exact-Gaussian covariance
power-exponential cases run independently under service
`quantdata-hedging-covariance-power-656ca65.service`.

A unified weekly runner supports frozen and restarted schedules for SANOS-LV,
historical SANOS, GROMM, VolGAN, and both diffusion variants. It requires an
authenticated strictly-2022 sizing preflight, validates every public dataset
member, holds a single-writer lock, records exact fitted-artifact fingerprints,
and rejects incomplete five-decision records. Generated weekly evaluation shares
one immutable target-independent path and fixed-book geometry per decision state
and one cold zero-claim pricing fit across targets. Historical weekly evaluation
prices the fixed option book once per decision state and attaches each target
payoff to the cached paths. Target-specific convex problems are rebuilt rather
than carried across claims, and the reusable pure-hedge geometry receives the
current claim payoff explicitly. Weekly entropy and power-exponential policies
use exact conic epigraphs for scenario-dependent turnover costs plus the declared
fitting ridge, which is excluded from reported certainty equivalents. Real
first-origin frozen-weekly gates cover all 13 targets: entropy at risk parameter
10 and power-exponential at risk parameter 1 both pass with zero cap hits and
zero nonconverged fits. Exact cached wealth matches the common pathwise
accounting engine to $10^{-14}$, and independent quant review passes.

The fresh 94-case native weekly matrix has 28 ready native cases, four generated
gate waiters, and 62 explicitly blocked cases. It was generated from immutable
commit `b37a643ea8e5ef11b7557e37c035cf650a729e95` and has SHA-256
`8d069cadde0c21dd677bbe7ab16cd59e0d3dae9907456608da7fcb4b9bce0b26`.
The two independent SANOS--LV weekly cases now run ahead of the daily recovery
queues. The 26 Historical-SANOS weekly cases use an atomic per-case pool whose
low-priority concurrency expands only after the protected GPU-readiness and
daily CPU services finish. Their persisted launcher SHA-256 values are
`361b4376fe8257cc0a608bca348bcccc2a1004f728b9b66e764989ed984ae676` and
`056574fcc73f22a69e93e1751b4aaeac84188575af0779d41f4a185a228e00e9`.
Service `quantdata-hedging-weekly-native-b37a643-v5.service` waits for those
two checkpointed pools and the active daily recovery queues, then authenticates
all 28 native cases through the original matrix prefixes. Its launcher SHA-256
is `992a61911968e5246f277b99fe9354865e423538b20bddb5a3dff18a68c0ea6c`.
That v5 coordinator is now superseded: the frozen case is represented by the
authenticated failure ledger above, the restarted case continues from its
16-origin prefix under service
`quantdata-hedging-sanos-lv-restarted-8851996.service`, and the Historical-SANOS
cases are owned by the version-1.11 weekly matrix.

The legacy rolling GROMM range and its current-code migration are complete and
authenticated for all 611 origins. The migration separately authenticated the
range-level and dated-manifest source identities, accepted 576 full physical
candidates, and used the declared front-only fallback at 35 origins. Its range
summary `gromm2-rolling-v4/range-fit-2023-2025-v7.json` has SHA-256
`8f7e6a63e90dfbfa367a06a220c75ebb4d0c9a0aa5f61ff2a92d8a58858b924e`.
Generated-model readiness now owns the managed GPU and runs from immutable commit
`d22ec9a0893313363fb503f8f53f5528159b1549`. It retains the earlier CPU-first
checkpoint restoration and signed buyer, seller, and continuation indifference
values, and additionally binds every rolling scope to the authenticated
initializer label rather than its operational run label. Independent quant
review passed the source correction, all generated matrix/CLI paths, both model
test suites, and the detached launchers. The current formal review artifacts
for VolGAN, AD-Seq-Vol, and AD-Seq-Vol-FT have SHA-256 values
`8b344877a1755596eb883bacd0fc9d12c42dcee24e75b245ea0928aa565f0546`,
`4014b863a6970892ae24ecce679fd291c5b22239e4c0bacf8b3256823949c6fc`,
and `405fde6cba65c74e272a4ef51891e9b272adda9a6bc6018bd10e6525e9f5edb5`,
respectively. Service `quantdata-hedging-generated-readiness-d22ec9a.service`
is executing the fresh 18-scope readiness matrix.

The corrected full-production service
`quantdata-hedging-full-production-d22ec9a.service` runs from the same immutable
commit and waits for generated readiness, the version-1.11 Historical-SANOS
matrices and failure canonicalization, and the remaining authenticated native
sources. Its launcher SHA-256 is
`b6c7c9534cf73b4c03bce1563e1190a700de41f2f6b9cbe360c516df8ca03dff`.
The coordinators use service activity only for capacity scheduling; because
transient units may be collected immediately after exit, completion is accepted
only by reconstructing the required matrices, gates, and result artifacts.
The daily composition prioritizes the corrected exact-cost and covariance-power
matrices before older sources, so a failed legacy case cannot shadow its
authenticated successor. The stale `a1baa69` weekly and full coordinators and
their report-only successors were stopped before they launched production work.
Newly ready GPU cases run through `gpuctl`, with four-minute progress and
resource checks; the remaining daily GROMM Q-expectation plus eight-factor-vega
case starts only after native weekly production releases CPU capacity. The last
durable complete-suite record has 349 passing tests; later focused suites pass,
and a final complete-suite rerun remains required after production.
The daily production-queue validator
now authenticates the immutable run and completion summary, recomputes the true
eligible-origin cohort from the public panel, reparses every dated result record,
and rebuilds every aggregate statistic before requiring exact agreement with the
persisted report. The weekly production validator now applies the same rule: it
reconstructs the overlapping-origin cohort from the validated public panel,
reparses every dated weekly record, recomputes exposures, diagnostics,
inference, and risk slices, and requires exact persisted-aggregate equality.
Integrated tests reject both a forged eligible-origin summary and a tampered
record. The report-input builder now also accepts one or more immutable daily
production matrices, discovers every completed case by case ID and command
output, reconstructs each aggregate from its run and dated records, rejects
conflicting duplicate cases, and can require that every ready matrix case is
complete. Its real current-cohort smoke automatically authenticated the SANOS-LV
reference, three robust-covariance policies, and the completed Historical-SANOS
policy and generated all four objective tables plus target appendices. The
weekly report layer also reconstructs inception market, accounting,
risk-neutral, buyer-indifference, and seller-indifference unit prices directly
from authenticated dated records. It rejects any disagreement with realized
accounting and renders cross-origin target-level price means without filling
unavailable Q or indifference quantities. Focused report and authentication
suites pass; a final complete-suite rerun remains required after production. The
strictly-2022 target-sizing exercise now
fixes each target to one
maturity-matched ATM spot-risk unit; all 20 calibration hedges converged with
no cap hit. Its atomic derived artifact is keyed by the public dataset manifest
and complete sizing configuration, preventing repeated production runs from
silently using different quantities. The same runner now supports VolGAN and
AD-Seq-Vol with full
training/scenario configuration identities. Abbreviated one-origin CPU/GPU
integration runs passed all 20 target records with no cap hit; they are
orchestration checks, not benchmark estimates. The completed-run results layer
now validates immutable run/record identities and reports unhedged, economic,
pure-hedging, and pricing losses overall, by year, by 2022-fixed target-maturity
ATM-volatility regime, and by year/regime intersection. It also recomputes
eligibility from the immutable run specification, revalidates every record,
and reports normalized entry turnover/exposure, cap rates, and explicit HAC
metadata; independent quant review passed. The first full SANOS-LV run exposed
a small-coefficient linear-program equality-scaling failure. Row scaling fixes
the numerical cause without relaxing the precommitted residual tolerance; the
clean versioned 2023--2025 CPU rerun is complete.

The benchmark should ultimately provide a public, model-neutral methodology for
comparing hedging systems. The first objective is the paper and a reproducible
internal benchmark. A public extension API follows after the paper methodology
and results are stable.

### Runtime data boundary

- The distributable `hedging-public-candidate-v1` dataset is the default market
  data input to model fits, training jobs, scenarios, valuation, and benchmark
  runs. For the current regeneration, original source data may be used to
  rebuild it; alternate direct-source experiments require explicit provenance
  and separate result identities.
- Runtime model and training code remains independent of the private dataset
  exporter and raw-data loaders; the offline exporter reconstructs the candidate.
- Stored SANOS states are the reference market interpolation whenever a price,
  target mark, or fixed contract is required away from a released instrument
  coordinate. Stored DLV states remain model inputs and are not substituted for
  the common SANOS market mark.
- The private-source exporter remains an offline construction tool only. It is
  outside the runtime dependency graph and is not invoked by training.

## Objectives

The benchmark must:

1. Evaluate hedging performance strictly out of sample from 2023 through the
   final eligible retained date.
2. Compare daily, frozen-weekly, and restarted-weekly hedging protocols.
3. Separate pricing differences from pure hedging performance.
4. Use identical market data, contracts, costs, claims, accounting, and metrics
   across all models.
5. Make every model easy to locate, understand, modify, and test independently.
6. Provide deterministic CPU smoke tests before any material GPU work.
7. Subject the protocol, accounting, models, and final results to independent
   quant-agent review.

## Agreed Experimental Design

### Estimation and evaluation dates

- Use 2022 for model initialization and architecture validation.
- Use a rolling 252-business-day estimation window, rolled daily.
- The current public-candidate panel ends on 2025-08-29. Evaluate strictly out
  of sample from 2023 through the final eligible origin recorded in its
  manifest. Extend through year-end 2025 only after a later dataset version
  actually contains those observations.
- Do not use future market observations for fitting, normalization, architecture
  selection, early stopping, checkpoint selection, or target sizing.
- Start a weekly episode on every eligible business day. Weekly episodes may
  overlap; inference must account for that dependence.

### Hedging schedules

#### Daily

- Fit using information available through date `t`.
- Hedge from `t` to `t+1`.
- A one-day claim settles at `t+1`.
- A longer claim is marked at `t+1` using the independently fitted market
  SANOS-LV density on that date.

#### Frozen weekly

- Fit the model and policy at episode origin `t0`.
- Rebalance daily for five business days.
- Keep model parameters and policy weights fixed throughout the episode.
- The policy may react to the observed state each day.
- Freeze synthetic inception contracts identified by expiry date and cash
  strike at `t0`.
- Allow those synthetic contracts to expire, but do not introduce contracts
  whose expiry date and strike were absent from the inception universe.
- On later dates, recover each frozen contract's current `K/F` from the common
  normalized spot/forward path and interpolate its half-spread cross-sectionally.

#### Restarted weekly

- Use the same five-business-day objective and episode origins as frozen weekly.
- Refit the model and recompute the policy each day using information available
  at that date.

### Normalization and market accounting

- Use `K/F`, model prices `C/(DF*F)`, half-spreads `h/S`, per-expiry `F/S` and
  `DF`, and forward-adjusted spot returns. This removes the deterministic carry
  represented by the previous-date forward curve without discarding the cash
  discount factor needed for accounting.
- Maintain one normalization and numeraire throughout each episode.
- Maintain three explicit spot paths without mixing their roles. All weekly
  targets settle on the inception-forward-normalized path
  `X[n]=S(t_n)/F(t_0,t_n)`. ATM target strike and barriers are expressed
  relative to `X[0]=1`; non-ATM digital strikes are the inception NMS-derived
  `K/F`. The actual cumulative `S[n]/S[0]` path is used only to recover fixed
  hedge-contract `K/F`, marks, settlements, and traded spot notional. The
  compounded rolling one-session forward-adjusted path is retained only as a
  carry-basis diagnostic: settling on it would make the claim depend on future
  carry curves unknown at inception. All accounting paths are converted to the
  same discounted episode-origin PV unit.
- Charge one basis point per transaction on traded normalized spot notional.
- Use the SANOS fitted model price as the common market mark. Execute a buy at
  the mark plus the published `h/S` and a sale at the mark minus `h/S`, after
  converting both to the same spot-one unit.
- Charge no transaction cost when a cash-settled index option expires.
- Use the SANOS-LV interpolated model mark as the common claim price. It is not
  represented as, or claimed to be, an observed midpoint.
- Report each model's buyer and seller indifference prices separately.
- Report risk-neutral expectations only for models that genuinely provide a
  risk-neutral measure.

For a short claim with market price `p_market`, model seller price
`p_model_sell`, hedge gain `G`, transaction costs `C`, and payoff `Y`, use

```text
economic_pnl
    = p_market + G - C - Y
    = (p_market - p_model_sell)
      + (p_model_sell + G - C - Y).
```

The first term is the pricing contribution. The second term is the pure model
hedge error. Use the corresponding buyer indifference price and long-claim
hedge for a long claim.

### Target claims

ATM always means normalized strike `K=1`, equivalently NMS zero. Only non-ATM
strikes are constructed from NMS, using the inception expiry-specific ATM
volatility. All target strikes are fixed at inception.

#### Daily targets

For both one-day and three-day maturities:

- ATM digital call at `K=1`.
- ATM digital put at `K=1`.
- OTM put digitals at NMS `-1`, `-0.75`, `-0.5`, and `-0.25`.
- OTM call digitals at NMS `0.25`, `0.5`, `0.75`, and `1`.

Each digital pays one unit of normalized cash.

#### Weekly targets

- The same digital family at one-week maturity.
- Five-day realized variance:

```text
(1/5) * sum_{n=1}^5 ((S[n] - S[n-1]) / S[n-1])^2.
```

- ATM down-and-out put with an end-of-day barrier at 95% of inception normalized
  spot/forward.
- ATM up-and-out call with an end-of-day barrier at 105% of inception normalized
  spot/forward.
- Observe barriers only at business-day closes.
- Do not smooth target payoffs.

### Hedge instruments

- Include normalized spot.
- Include eligible option expiries through five business days.
- If no exact five-day expiry exists, also include the first expiry after five
  business days.
- Include options strictly within `(-2, 2)` NMS, computed using the current
  expiry-specific ATM volatility.
- Treat `(-2, 2)` as a dynamic hedge-eligibility rule, not as a storage filter.
  The public candidate retains the superset `[-2.5, 2]`, tags the paper's exact
  `[-2.5, 1.5]` calibration population, and separately tags the contemporaneous
  `(-2, 2)` hedge screen.
- Exclude AM-settled contracts from the primary hedge universe. The public
  close-to-close path does not contain the special opening settlement level
  needed to settle them exactly; the retained flag supports this exclusion.
- Exclude exterior DLV boundary strikes.
- Use one liquid OTM instrument per strike: puts below ATM and calls above ATM.
- Avoid redundant put-call-parity copies.
- For weekly experiments, determine eligibility at `t0` and freeze the
  synthetic expiry-date/cash-strike universe for the episode. Exact listed
  contract replay is an internal-data extension, not a public benchmark
  capability.

## Package Architecture

The repository style guide prohibits one research project from depending on
another. The new SDLV study must therefore not import implementation directly
from `quantdata.dysanos.flex`. Genuinely reusable functionality should be moved
behind a neutral package while compatibility wrappers preserve existing
DySANOS experiments.

Planned layout:

```text
quantdata/hedging/
    PLAN.md
    contracts.py
    accounting.py
    objectives.py
    metrics.py
    inference.py
    protocols.py

quantdata/sdlv/daily/front_spot/hedging/
    README.md
    benchmark/
        specification.py
        episodes.py
        accounting.py
        metrics.py
        inference.py
    data/
        market_panel.py
        contracts.py
        rolling_windows.py
        fixtures.py
    claims/
        digitals.py
        variance.py
        barriers.py
    models/
        sanos_lv/
        gromm/
        covariance/
        historical_sanos/
        volgan/
        diffusion/
    runners/
        run_daily.py
        run_frozen_weekly.py
        run_restarted_weekly.py
    reviews/
    reports/

tests/quantdata/hedging/
tests/quantdata/sdlv/daily/hedging/
```

Each model directory must contain:

```text
README.md
config.py
adapter.py
hedger.py
smoke.py
```

The model README must state:

- Input data and fitted quantities.
- Whether the model represents `P`, `Q`, or neither.
- Its price interpretation.
- Its scenario-generation mechanism.
- Its hedge objective and supported risk aversions.
- Its trainable components and validation rule.
- Its expected artifacts and standalone test command.

## Common Interfaces

The benchmark should depend on small, explicit interfaces rather than model
internals.

### Payoff

- Define the claim at episode inception.
- Evaluate settlement from a normalized path.
- Mark an unfinished claim using a supplied market snapshot.

### Market model

- Fit from an explicitly dated training window.
- Produce state variables available at a decision time.
- Simulate under a named measure when supported.
- Price a claim when supported.
- Serialize sufficient state for exact restart and audit.

### Hedger

- Fit only from the supplied training or generated scenarios.
- Return a predictable action from the current state and existing positions.
- Report objective value, convergence, caps, and numerical diagnostics.

### Numerical stabilization and solver failures

- Use scale normalization, stable parameterizations, explicit finite-value
  checks, warm starts, and objective-appropriate regularization before treating
  a numerical issue as a model failure.
- Do not equate a solver status label with mathematical failure. Accept a
  numerically good-enough result only when an independent, predeclared
  residual, feasibility, KKT, and objective-reconstruction gate certifies the
  solution at the benchmark tolerance; record both the raw status and the
  independent certificate.
- Never silently relax a gate or omit a failed origin. If the independent
  certificate fails after the documented stabilization and fallback attempts,
  stop the affected case, retain the complete diagnostics, request QuantAgent
  review, and document an alternative numerical approach before rerunning it.

### Backtest

- Construct the episode universe.
- Apply actions using the common execution and cost model.
- Settle expiring instruments.
- Mark the target and remaining positions.
- Emit a model-independent episode record.

## Model Implementations

### SANOS-LV reference

- Fit the market SANOS surface.
- Price in local volatility.
- Compute spot delta.
- Compute DLV vegas with the first eight contemporaneous rolling PCA factor
  bumps.
- Use central factor differences in log DLV-squared coordinates.

### GROMM

- Estimate 16 rolling DLV innovation factors.
- Align rolling factor bases using permutation and sign matching, with
  Procrustes rotation inside near-degenerate eigenspaces.
- Use all 16 factors in the stochastic SDLV model.
- Use the first eight factors for comparable DLV-vega hedges.
- Use eight Brownian substeps per business day.
- Preserve the independent 5% Black background under both `P` and `Q`.
- Estimate the compatible physical drift and risk-neutral drift separately.
- Support mean-variance, mean-volatility, entropy, and power-exponential
  deep-hedging objectives under both `P` and `Q`.
- Fit claim, short-claim, and zero-claim policies separately for indifference
  pricing. Retain the claim-minus-zero action as a diagnostic. Because absolute
  turnover is nonlinear, train the executable pure hedge directly in net-action
  coordinates on conditional Q-shadow gains. At every simulated state, reuse
  the same Brownian shock under the unshifted Q transition and measure gains
  from the model mark. This preserves local hedge shocks while removing both
  state-dependent P drift and common-mark/model-mark basis positions, without
  estimating a conditional mean from validation paths. Under `Q`, the
  zero-claim policy should be zero in population.
- Also provide conventional risk-neutral expectation plus eight-factor DLV-vega
  hedging.

The path-dependent policy should use an LSTM followed by the established
affine-plus-deep action architecture. Test four architecture candidates on 2022
generated training and validation sets. Fix the architecture thereafter and
retrain weights rolling. Choose the cheapest candidate whose validation loss is
within one standard error of the best candidate.

### Independent covariance baseline

- Use raw historical normalized option and spot returns.
- Do not use SANOS-generated scenarios.
- Estimate a rolling robust high-dimensional covariance matrix.
- Prefer a shrinkage Tyler estimator that does not require a parameter fixed
  permanently from 2022.
- Apply half-spread-weighted cost lasso derived from the common execution rule.
- Support mean-variance and mean-volatility directly.
- Under a Gaussian assumption, support entropy through its mean-variance
  equivalence and power-exponential objectives where numerically appropriate.

### Historical statistical SANOS baseline

- Use historically fitted SANOS surfaces to price the same contracts and target
  claims.
- Construct historical normalized return scenarios.
- Apply the same transaction-cost and risk-objective implementation as the
  other methods.

### Published generative baselines

Implement the following only after the native benchmark passes:

- VolGAN and the cost-lasso hedging protocol from the public implementation.
- AD-Seq-Vol and arbitrage-aware AD-Seq-Vol-FT from the published diffusion
  implementation.

For each external baseline:

1. Reproduce the public settings first.
2. Record any deviations required for the local data schema.
3. Retrain under the common rolling benchmark protocol.
4. Report both public-setting and rolling-trained results.

## Risk Objectives

Support:

1. Mean-variance.
2. Mean-volatility.
3. Entropy OCE.
4. Power-exponential OCE.

For mean-volatility use:

```text
rho_kappa(L) = E[L] + kappa * SD(L)
```

with `kappa` in `{0.1, 1, 2, 3}`.

For Gaussian loss with standard deviation `sigma`, entropy and mean-variance
have risk-premium equivalence

```text
lambda = 2 * kappa / sigma.
```

Risk parameters and losses should be expressed in a documented episode risk
unit. The confirmed reference is inception ATM spot risk

```text
sigma_ref(t, H) = sigma_atm(t) * sqrt(H / 252).
```

## Target Notional and Position Caps

Target notional materially affects the relevance of option half-spreads. The
preliminary sizing experiment uses 2022 only:

1. Normalize each target by its 2022 unhedged standard deviation.
2. Use the independent robust covariance hedge with mean-volatility `kappa=1`.
3. Measure transaction cost relative to the target's unhedged risk and gross
   option exposure.
4. Set each target quantity so its scaled unhedged standard deviation equals
   one maturity-matched ATM spot-risk unit.
5. Freeze those target quantities for all out-of-sample tests.

The 242-transition initialization run through 2022-12-30 converged for all 20
targets with no cap hit. Median, 90th-percentile, and maximum transaction cost
were respectively 1.69%, 3.78%, and 7.54% of the reference risk. A common
rescaling cannot select a more attractive size here: under proportional costs
and the homogeneous mean-volatility objective the cost-to-risk fraction is
scale invariant except at a zero-action or cap boundary. The dated calibration
report records the target quantities and diagnostics.

An internal-data extension may additionally report display-size utilization,
but it cannot determine the primary public benchmark notional because display
size is deliberately absent from the distributable dataset.

Use the following cap regimes:

- A loose numerical cap of 1,000 normalized action units, applied to spot and
  option coordinates. Cap hits and nonconvergence fail the numerical gate by
  default. A mean-volatility case whose unconstrained SOCP certifies a recession
  direction precommits the relaxed diagnostic gate instead: the capped solve is
  retained and explicitly flagged as an economically unbounded direction, not
  relabelled as a converged finite hedge.
- An internal-only display-size cap sensitivity when source size data are
  available.

## CPU and GPU Work Split

Every implementation must have a deterministic CPU execution path suitable for
smoke testing. GPU work starts only after the CPU invariants and standalone
model test pass.

| Work | CPU responsibility | GPU responsibility |
| --- | --- | --- |
| Market loading and contract matching | Primary | None |
| Rolling-window construction | Primary | None |
| Claim, settlement, and accounting tests | Primary | None |
| Rolling PCA and factor alignment | Primary | Optional acceleration |
| Robust covariance estimation | Primary | None initially |
| SANOS-LV pricing and Greek smoke tests | Primary | Optional batching |
| SDLV/GROMM simulation | Small deterministic pilots | Production paths |
| Rolling DLV fits | Smoke and fallback | Production fitting |
| Deep-hedging training | Small architecture tests | Production training |
| VolGAN and diffusion training | Interface smoke tests | Production training |
| Metrics, inference, and reports | Primary | None |

CPU tasks should materialize immutable, versioned inputs before GPU launch.
GPU jobs should perform only the fitting, path generation, or policy training
that materially benefits from GPU execution.

Every material GPU job must:

- Run through `gpuctl`.
- Check `gpuctl resources --json` before launch.
- Use the explicit project interpreter, working directory, `QUANTDATA_ROOT`, and
  `PYTHONPATH` in detached execution.
- Validate imports and data-root discovery in the detached environment before
  starting material work.
- Checkpoint all model, optimizer, scheduler, scaler, and RNG state required for
  a near-exact restart.
- Emit flushed progress at least every five minutes.
- Report elapsed time, completed and total work units, and ETA when meaningful.
- Avoid retaining unnecessary large intermediate arrays.

## Standalone Model Tests

Every model must have an independently runnable CPU smoke test:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.gromm.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.covariance.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.volgan.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.diffusion.smoke
```

Each smoke test must:

- Use a fixed small date range and deterministic seed.
- Run on CPU without hidden GPU initialization.
- Hedge one ATM one-day digital.
- Complete quickly enough for routine developer use.
- Produce the common compact JSON result schema.
- Report unhedged P&L, hedged P&L, costs, actions, pricing difference, pure
  hedge error, convergence, and cap diagnostics.
- Compare key outputs with a committed tolerance fixture.

Keep one pytest module per model:

```text
tests/quantdata/sdlv/daily/hedging/models/test_sanos_lv.py
tests/quantdata/sdlv/daily/hedging/models/test_gromm.py
tests/quantdata/sdlv/daily/hedging/models/test_covariance.py
tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos.py
tests/quantdata/sdlv/daily/hedging/models/test_volgan.py
tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py
```

GPU integration tests must be separately marked and must not be required for a
normal CPU test run.

## Quant-Agent Review Gates

Use a separate quant-review agent at each material milestone. Store reviews in
`quantdata/sdlv/daily/front_spot/hedging/reviews/` with the reviewed commit,
configuration identity, evidence, findings, and disposition.

### Protocol review

Review:

- Information timing and rolling windows.
- Contract identity and weekly universe freezing.
- Normalization and numeraire consistency.
- Claim definitions and settlement.
- Look-ahead protections.

### Accounting review

Review:

- Buyer and seller signs.
- Initial cash and common market price.
- Indifference-price decomposition.
- Bid/ask execution and spot costs.
- Expiry settlement and remaining-position marks.

### Model review

Perform a separate review for every model directory before its production run.
Review:

- Measure interpretation.
- Calibration inputs and train/validation separation.
- Price semantics.
- Simulation and hedge objective.
- Risk scaling and transaction-cost implementation.
- Standalone smoke-test evidence.

### GPU-readiness review

Review:

- CPU/GPU numerical agreement on a fixed fixture.
- Reproducibility and RNG handling.
- Checkpoint recovery.
- Batch-size invariance within tolerance.
- Memory scaling and progress logging.

### Results review

Review:

- Statistical inference and effective independent units.
- HAC or block-bootstrap treatment of overlapping weekly episodes.
- Pairwise comparison construction.
- Cap and transaction-cost effects.
- Data leakage and model-selection leakage.
- Reproducibility of all headline tables.

Unresolved correctness, arbitrage, look-ahead, measure, or accounting findings
block the next production stage. Performance and presentation comments may
remain open if documented.

## Statistical Reporting

For every model, payoff, maturity, year, and volatility regime, report:

- Mean and confidence interval.
- Median.
- Sample standard deviation.
- RMSE and MAE.
- 95% and 99% loss quantiles.
- Expected shortfall.
- Mean-variance, mean-volatility, entropy, and power-exponential scores.
- Pricing contribution, pure hedge error, and total economic P&L.
- Turnover and transaction costs.
- Sparsity, gross exposure, and net exposure.
- Numerical-cap hit rates and, only for internal-data extensions, display-size
  cap hit rates.

Use complete episodes as sampling units. Use paired episode-level differences
for model comparisons. Use HAC or block-bootstrap inference for overlapping
weekly episodes.

## Implementation Phases and Gates

### Phase 0: specification and schemas (complete)

Implement the versioned benchmark configuration, common artifact schema,
contract identity, and episode timing definitions.

The 891-date public-candidate artifact passes checksum, leakage-control, schema,
and full SANOS/DLV reconstruction validation from an otherwise empty data root;
durable evidence is in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-public-dataset-validation.md`.
Formal redistribution rights remain separate from this technical gate.

Gate: protocol quant review passes.

### Phase 1: accounting vertical slice (complete)

Implement market snapshots, one ATM digital, spot plus one option, SANOS-mark
plus/minus-half-spread execution, settlement, and pricing/hedging decomposition.

Run a deterministic CPU test over a few dates.

Gate: accounting quant review and accounting unit tests pass.

### Phase 2: SANOS-LV daily baseline (complete)

Implement rolling market marks, delta, eight PCA-factor vegas, all daily target
claims, and the common reporting record.

Gate: standalone SANOS-LV smoke test passes.

### Phase 3: independent covariance baseline (complete)

Implement same-contract historical returns, robust covariance, cost lasso,
mean-variance, mean-volatility, Gaussian entropy equivalence, and the exact
Gaussian power-exponential OCE.

Gate: standalone covariance smoke test and model review pass.

The production implementation subsequently exposed a distinct numerical issue:
the 1,000-unit executable position cap was also bounding the claim and
zero-claim virtual policies before their subtraction. Those virtual policies
now use a separate 1,000,000-unit internal solver cap, while only the executable
claim-minus-zero hedge uses the 1,000-unit benchmark cap. Internal and
executable cap hits are reported separately. They fail the strict numerical
gate; a precommitted mean-volatility recession diagnostic may retain them only
as an explicitly nonconverged, cap-limited direction.
An independent exact-date replay and a public-data 20-target smoke passed with
no cap hit or nonconvergence. Results produced under the shared-cap implementation
are retained only as failed diagnostics; the clean production sweep is running
from a pinned source worktree.

### Phase 4: rolling SDLV calibration (complete)

Implement the rolling 252-day innovation fit, 16-factor PCA alignment, eight
Greek factors, compatible `P` drift, and `Q` drift.

Gate: CPU deterministic calibration test and factor-alignment tests pass.

### Phase 5: GROMM daily benchmark

Implement GROMM scenario generation and the four hedge objectives under `Q`,
then under `P`. Use the currently specified `quantdata/gromm2` market model:
the 200-strike JAX implementation with AR(1) DLV dynamics. Its greater-than-80%
market fit is accepted for this benchmark iteration. Treat `quantdata/gromm2`
as a read-only architecture reference and do not modify it from the hedging
project.

The market-model architecture is therefore fixed rather than selected inside
the hedging benchmark. Dated rolling P/Q artifacts and exact benchmark-book
output are separate from the four affine/deep/LSTM hedge-policy candidates;
only the hedge-policy architecture is selected on the 2022 initialization set.

The dated market-artifact contract, CPU preparation stage, daily P/Q
benchmark-book output, and weekly recurrent-policy interface are implemented.
A production artifact contains exactly 252 transitions in the declared origin's
information set, fixes the 200-strike geometry from that origin, hashes all
inputs, and rejects full-panel or future-dated histories. When a released
origin follows a missing intermediate market date, the fit retains the latest
252 exact released one-session pairs and appends the current origin state for
the geometry and DLV level; it never substitutes a multi-session return for a
one-session transition. A real first-eligible 2023-01-17
fit completed from the public dataset using 252 exact released one-session
pairs supported by 257 states. It places 84.90% of observations inside spread,
and the rank-16 innovation basis explains 95.15% of variance. The compatible-P
candidate is accepted only when its linked holdout improvement is at least
0.5%, its Newey--West gain statistic is at least 1.96, and its daily risk-price
norm is at most 3. The density-weighted projection fraction is retained as a
descriptive fit statistic rather than used as an admissibility gate. If the
candidate is rejected, the precommitted physical null retains the full-window
front-factor risk price and sets the other 15 DLV-factor prices to zero. This
front-only fallback is accepted only when finite and within the same norm gate;
it is labelled explicitly and is not reported as an accepted full DLV
risk-price fit. Its authenticated four-path Q replay produces finite returns
over 163 instruments. The loader bounds
float32 density-mass roundoff by stored source precision, normalizes accepted
residuals in float64, and logs the correction. The five-session sampler recomputes
the compatible P risk price pathwise and prices the frozen contract book
without retaining full density histories. Every artifact is bound to the exact
public manifest. A compatible-P candidate must pass the precommitted holdout
improvement, Newey--West gain-statistic, and daily risk-price-norm gates; the
projection fraction remains descriptive. Generated books start at the common
SANOS executable marks. Signed policies provide prices and diagnostics; the
directly trained pure policy uses conditional Q-shadow gains, exact net
turnover, free cash settlement, and terminal liquidation. Managed-GPU affine fitting and 2022
hedge-architecture selection are complete for the first rolling origin. Full
rolling Q production and any future-origin P acceptance remain outstanding. A
range orchestrator now authenticates and reuses each dated preparation or fit,
records every manifest hash atomically, prevents concurrent writers, and emits
a persistent heartbeat at most every five minutes. Borderline affine fits keep
the fixed 80% inside-spread gate and continue from their authenticated 2,000
epoch checkpoint to at most 4,000 and 8,000 epochs; completed epochs, retry
count, and the actual fit configuration are retained in the dated artifact.
Its real first-origin resume smoke passed. CPU preparation, the legacy
managed-GPU fit, and the current-code migration are complete for all 611
eligible origins, including the first sparse-release origin after an
authenticated restart. The authenticated current range contains 576 accepted
full physical candidates and 35 declared front-only fallbacks. Generated-model
readiness now owns the managed GPU.

The range evidence reports full-candidate, front-only-fallback, and unavailable
physical-origin counts separately. A versioned migration path authenticates the
completed legacy range and every dated manifest, hard-links only its immutable
source, prepared-book, and accepted affine-history files into a distinct target
root, and recomputes all dynamics and physical-selection evidence under the
current code. Legacy physical decisions and legacy dynamics are never reused.
Independent review of these selection and authentication semantics passes in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-rolling-gromm2-physical-selection-review.md`;
it does not
stand in for the still-incomplete rolling-production evidence.

Gate: GROMM CPU smoke test, model review, and GPU-readiness review pass.

### Phase 6: GPU production infrastructure

Implement restartable rolling fit, path-generation, and deep-hedging jobs with
the required resource checks, checkpointing, and progress logs.

The rolling GROMM2 range runner provides authenticated per-origin restart, an
atomic range state, a single-writer lock, and four-minute progress heartbeats.
A completed range summary binds the dataset, configuration, evaluation dates,
artifact root, and every dated manifest hash. The production matrix enables Q
cases only from this complete evidence; it enables P cases only when every
origin has either an accepted full candidate or an accepted, explicitly labelled
front-only null fallback. The current-code migration is complete; live
managed-GPU forced-restart evidence for the downstream generated workflows
remains outstanding. A deterministic
mid-range failure test now verifies that the failed state identifies the exact
origin, preserves the authenticated completed prefix, and reuses that prefix on
restart before completing the missing origins.

The queue executor can retain a numerically rejected parameter case and
continue an otherwise independent sweep. Such a queue is explicitly marked
`complete-with-failures`; the exact failed attempt is bound to its runner-emitted
gate artifact, and the failed case remains failed and cannot satisfy a
model-results gate. The ledger and composition authenticators reject command
relaxations, incomplete queues, forged or stale artifacts, and manual failure
classifications. Independent quantitative review reports no remaining
production-integrity blocker.

Gate: CPU/GPU fixture agreement and forced-restart recovery test pass.

### Phase 7: full daily evaluation

Run all daily targets and native baselines from 2023 through the dataset's last
eligible origin. Resolve target notional and cap conventions using 2022 before
starting this stage.

The production results schema and all four agreed ex-post risk functionals are
implemented. A completed run is rejected if any eligible origin, target,
dataset identity, run identity, convergence gate, or cap count is missing or
inconsistent. The failed SANOS-LV `production-v1` is retained as a numerical
diagnostic; the row-scaled `production-v2` run completed all 611 eligible
origins and wrote 12,220 target records, but its superseded target-notional
identity prevents its use as current production evidence. The clean
current-notional SANOS-LV production run is now authenticated under run identity
`c67b549ca9eeb404be9e70c6fe2ba22c827d8d4ec75115e403a0bb2e54666a68`.
Its 611 origins and 12,220 records reconstruct deterministically to results
SHA-256 `9e4252f60540d6830eacda10b451677320760404e4599a5caa4e346367262b2d`,
with zero cap hits and zero nonconvergence. The summary report excludes the
superseded historical-SANOS cohort rather than comparing it with this run. The
corrected Historical-SANOS mean--variance case at risk parameter $0.1$ has
completed all 611 origins and 12,220 records under the current target-notional
identity, with zero cap hits and zero nonconvergence; its results SHA-256 is
`ad5850d4ba33c180cdac776623e49725260f46dd88ee6f7fd70b73b89319261e`.
The remaining Historical-SANOS and covariance sweeps remain in progress. The
legacy `edf28ccdb52ccac1788e20a51ce226220a5f74f3` queue now owns only the
precommitted low-risk mean--volatility recession diagnostic. Its failed entropy
and power-exponential cells remain failed and cannot satisfy a production gate.
Fresh exact-cost entropy and power-exponential cases run from independently
reviewed commit `cff579c38c489e315b624d055d4336e80554cd58`; the original-action
KKT residual, rather than the split-variable solver status, is authoritative.
The covariance mean--variance cases at risk
parameters $0.1$, $1$, and $10$ have completed all 611 origins and 12,220
records under the current target-notional identity, with zero cap hits and zero
nonconvergence. Gaussian entropy is represented exactly by these same
mean--variance policies at equal risk parameter. A distinct exact-Gaussian
power-exponential covariance fit is implemented and passes independent
quantitative review in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-covariance-gaussian-power-review.md`.
Its three risk-parameter cohorts are running from immutable commit `656ca65`
under `quantdata-hedging-covariance-power-656ca65.service`; the persisted launcher
SHA-256 is
`f359b2de7b710bd76652bb5c27fe0089b6740bab3fba9fecf8ca91b9cd46fdc2`.
The versioned daily production matrix contains 57
unique cases: 24 runnable native CPU cases, two
staged published-setting GPU cases awaiting the generated-model GPU-readiness
gate, and 31 explicitly blocked rolling-generated or GROMM cases. No blocked
case is silently emitted as a launch command.

Gate: interim results quant review passes.

### Phase 8: weekly engines (mechanical gate complete)

Implement frozen-weekly and restarted-weekly episodes, fixed synthetic
expiry-date/cash-strike books, daily rebalancing, variance, and barrier claims.

Gate: timing, expiry, and overlapping-episode tests pass.

The common five-session engine, target catalogue, frozen cash-strike replay,
daily rebalancing, cash settlement, first-later-expiry liquidation, and
overlapping-origin enumeration pass focused tests and independent quant review.
The settlement convention has subsequently been hardened to use the inception
forward curve represented by the SANOS and DLV states; the rolling-forward
alternative remains a diagnostic. Frozen/restarted fit identities, exact
rolling-window information facades, final SANOS-LV common-price review, and
overlap-aware results inference now pass their focused gates. GROMM, SANOS-LV,
historical SANOS, VolGAN, and diffusion weekly adapters are implemented at
CPU-smoke level; production execution remains outstanding. The adapter supplies
the predictable realized path prefix and prior positions to a restarted fit,
so path-dependent prices cannot be computed from an information-incomplete
fit call; frozen fits still occur only once at episode inception.

The model-neutral resumable evaluator writes one atomic 13-target record per
origin. The fixed sizing handoff uses 221 complete 2022 episodes, a five-day
ATM spot-risk standard deviation of `0.029423402816172756`, and an authenticated
artifact SHA-256 of
`9494bbe23699126b5137b5ab3b6d53070002e2ba8ee60d92b0327ff747fcf9e1`.
A real one-origin frozen SANOS-LV public-data run and its no-recompute restart
check pass. The corresponding restarted one-origin integration run also passes;
its no-recompute restart check validates the exact five dated refits. These are
orchestration checks, not benchmark estimates. The native SANOS-LV and
historical-SANOS weekly cases are materialized and deferred behind the daily
CPU production queues so they do not compete for the currently saturated host.
Before accepting a completed production cell, the queue fully validates the
released dataset, recomputes the true overlapping-origin cohort, reparses all
weekly records, rebuilds every reported statistic, and requires exact aggregate
equality; stored self-hashes alone are not treated as completion evidence.

### Phase 9: historical and published baselines

Add historical SANOS, VolGAN, and diffusion baselines. Reproduce public settings
before rolling retraining.

Historical SANOS now has a model-specific daily implementation; its
mean-volatility cost-lasso is solved as an exact CLARABEL SOCP rather than by an
unstable smooth line search. The solver first attempts the economically
unconstrained problem. If CLARABEL certifies a recession direction, it resolves
the same SOCP with the separate numerical solver cap solely to expose and record
the direction. Such a result is retained only when the immutable production
case precommits the relaxed mean-volatility diagnostic gate; the report exposes
its cap-hit and nonconvergence counts rather than presenting it as a finite
optimum. VolGAN and
diffusion have exact within-fit restart checkpoints and a fixed-source
rolling-update path. One immutable checkpoint
trained strictly before evaluation initializes every rolling origin. Its file
hash and training end date are checkpointed, source dates must be strictly
earlier than current training dates, and current-window normalization is
analytically rebased without changing the source model's economic mapping.
Update epochs are selected from precommitted candidates on 2022-only validation
origins with the one-standard-error rule. The independent generated rolling
warm-start quant review passes with no correctness blockers. The production
runs remain outstanding; the corrected managed-GPU readiness matrix is now
regenerating the VRAM, timing, batch-invariance, and restart evidence required
before production is admitted. The production matrix
pins one immutable initializer hash across every rolling origin, while each
fit verifies the warm-start and fine-tuning-base hashes before and after load.
The 2022 budget study uses a separate same-geometry source checkpoint ending
before its first 126-observation validation window; the resulting budget is
then paired with the complete end-2022 production initializer. This separation
is authenticated explicitly and avoids the impossible requirement that an
end-2022 checkpoint predate its own 2022 validation origins.
Because the released 2022 panel contains 247 dates, this is explicitly a proxy
for the 252-session production window. The precommitted candidate grids are
`1,10,100,1000` for VolGAN and `1,3,10,30` for AD-Seq-Vol and AD-Seq-Vol-FT.
Diffusion base checkpoints additionally carry a shared semantic-geometry
identity, preventing same-shaped but economically different transformer or
surface configurations from loading silently.

The first full-geometry managed-GPU readiness origin now passes. CPU preflight
pins and authenticates the target-notional artifact before allocation; the
2023-01-17 AD-Seq-Vol gate then produced all 20 target results from 4,096
scenarios in 158.3 seconds with zero cap hits or nonconvergence and 2,636 MiB
observed VRAM. An interrupted epoch checkpoint resumed successfully after the
restart loader was corrected to restore CPU RNG state on CPU. The detailed
evidence is recorded in
`reports/2026-10-12-generated-gpu-readiness.md`. Production-scale generated
runs and batch-invariance evidence remain outstanding. The authenticated 2022
initializer and epoch-budget pipeline has now completed for all three generated
baselines. The one-standard-error selections are 1,000 rolling epochs for
VolGAN, 30 for AD-Seq-Vol, and one for AD-Seq-Vol-FT; their complete end-2022
production initializers and distinct selection-source lineages pass the strict
evidence loader. The authenticated legacy and current-code rolling GROMM ranges
are complete for all 611 eligible origins. The migrated range passes full
authentication with 576 accepted physical candidates and 35 declared
front-only fallbacks; generated readiness now owns the managed GPU. A
production-gate assembler now reconstructs exact public-versus-rolling scope
bindings, refuses stale or aliased evidence, writes under an output lock through
a unique same-directory temporary file, and re-authenticates the completed gate
before publishing it. The initial reviewed 18-scope managed-GPU readiness
matrix ran from immutable commit
`31b57e617dc6f57f12625515160131a8cf845e36`; VolGAN daily-public readiness
passed, but its first frozen-weekly case exposed two independent issues. The
checkpoint restore path was corrected at `b434852`, and the weekly price schema
was corrected at `b37a643ea8e5ef11b7557e37c035cf650a729e95` to permit finite
signed buyer, seller, and continuation indifference values. Common market and
genuine-Q prices remain finite and non-negative. This distinction is required
for non-monotone mean--variance and mean--volatility objectives; a negative
indifference value is not by itself a numerical failure. The weekly
mean--variance quadratic program uses OSQP, while mean--volatility retains
CLARABEL for its conic problem. On a synthetic 4,096-path geometry matching the
first production origin's 207 contracts and five decisions, the shared zero
policy plus first complete target fit took 45.1 seconds and converged without a
cap hit. A solver-reference test matches the CLARABEL optimum, and weekly runs
report target-level progress inside each origin.

Independent re-review at `b37a643` passed 161 focused tests, reconstructed all
six scope bindings for each generated model, and authenticated all three
production-gate round trips. A later rolling-readiness attempt exposed the
initializer-label/run-label alias described in the current status. The source
fix at `d22ec9a0893313363fb503f8f53f5528159b1549` passed an additional
independent quantitative review and 67 focused tests. Service
`quantdata-hedging-generated-readiness-d22ec9a.service` emits a persistent
four-minute heartbeat, acquires its GPU only through `gpuctl`, and is producing
the corrected readiness evidence required before generated production cases can
launch.

A real 2023-01-17 frozen Historical-SANOS origin then completed all 13 targets
in 336.3 seconds with zero cap hits and zero nonconverged fits; its immediate
restart authenticated and reused the dated record. Single-origin readiness runs
now skip statistical result aggregation, whose loss and inference slices require
at least two origins, while complete production runs continue to aggregate
normally.

The 28 ready native weekly cases are repinned to
`b37a643ea8e5ef11b7557e37c035cf650a729e95` and are queued behind the
authenticated Historical-SANOS daily CPU artifacts they require. Their fresh
matrix still contains all 94 specified cells: 28 ready native cases, four
generated cases waiting on production gates, and 62 explicitly blocked cases.

Gate: one model-specific review and standalone smoke test per baseline pass.

### Phase 10: final benchmark and paper artifacts

Run the complete benchmark, generate versioned tables and figures, perform the
final results review, and reconcile every reported number with its source
artifact.

The daily and weekly report paths are matrix-driven and authenticated rather
than manually transcribed. The public CLI accepts only the public dataset,
immutable production matrices, and their version-2 pre-outcome commitment; it
rebuilds the metrics and LaTeX fragments and writes a manifest binding dataset,
matrix, failure-ledger, run, aggregation, report, and reporting-commit hashes.
The commitment records the exact dataset and matrix paths and hashes before any
generated or GROMM production outcome exists. Its SHA-256 is fixed at creation,
validated before aggregation and after rendering, and supplied explicitly to
the report builder. Final rendering requires the canonical 57-cell daily and
94-cell weekly grids to have authenticated terminal outcomes, so a post-outcome
matrix cannot suppress a policy by relabelling it blocked.
Weekly ingestion requires the complete
528-origin, 6,864-record population, verifies objective labels against the
immutable run snapshot, keeps frozen and restarted schedules distinct, and
generates aggregate objective tables and target-level appendix tables for all
four ex-post objective families at every precommitted parameter. Objective
functionals are evaluated in unscaled inception-spot-normalized loss units and
only the resulting scalar is converted to basis points for display. Final
weekly comparisons use equal-weighted 13-target episode books and report
candidate-minus-SANOS paired differences with Hansen--Hodrick and moving-block
bootstrap uncertainty. The model catalogue follows first appearance in the
authenticated daily grid, followed by weekly-only models, and uses the same
display names as the result tables. A standalone final-document builder now
renders all daily and weekly objective grids without selecting policies on
2023--2025
outcomes, target-level appendices, model references, and immutable provenance
directly from those authenticated inputs; its synthetic end-to-end PDF compile
passes. Daily ingestion likewise retains complete overall, year,
volatility-regime, and year-by-regime distributions, all price decompositions,
costs, turnover, exposure, sparsity, cap and convergence diagnostics. Its
equal-weighted 20-target book comparisons use five-lag Bartlett HAC inference
and a 2,000-sample moving-block bootstrap. The rendered daily and weekly
objective and target appendices are multipage tables rather than selected
headline rows.
A matrix-composition utility at immutable commit
`a1baa69605514ad9a0aba007ab8844f5fed29824` now replaces only selected native
cases in a fresh canonical matrix with the first complete authenticated source
case. Before accepting a source it loads the supplied public panel once, applies
the production executor's full authentication path, reconstructs the eligible
origin cohort and every dated record and aggregate, binds the actual dataset,
requires the canonical target book and common target quantities, and enforces
each source run's immutable numerical-gate policy. Strict cases reject any cap
hit or nonconvergence; precommitted mean-volatility recession diagnostics are
retained with their counts. It records the dataset, target-notional, and
target-cohort identities together with every source-matrix hash. Adversarial
tests reject a missing record, forged aggregate, wrong dataset, and wrong target
cohort; a real 12,220-record SANOS--LV composition smoke and focused composition
tests pass, and independent quant re-review closes the authentication blocker.
The pre-production complete-suite record on the remediated report source has
458 passing tests; the suite is rerun after every final source change and will
be rerun again after production. This permits the final
57-cell daily and 94-cell weekly matrices to reuse completed native evidence
without copying it or rerunning it. The reviewed detached final-production
service `quantdata-hedging-full-production-de18586-v2.service` is active from
immutable reporting commit `de185864a9eb414ca431b92321b56c41062fd651`. Its
exact detached-environment preflight passed and its installed launcher has
SHA-256 `9ca7c53ecf1d843ae6e1aa7c124665b85384737c4352607d01601006b935c51c`.
It creates and persists the version-2 commitment and fixed SHA-256 before any
final generated or GROMM production case, revalidates them on restart and
immediately before reporting, and rechecks its clean pinned worktree before the
final render. It also fail-closes unless the independently approved final-review
protocol from commit `5fc1bc87f529630bf7b936e4eb77e2ffa6ce7bfe` has SHA-256
`34ea83410614c3395bb4ccdf561ff95d1acd9ee572f413cfdf2d48bea29a9c7f`;
that identity is logged before preflight, matrix preparation, queues, and final
rendering. It retains the existing ordered daily sources, authenticated
weekly native source, and combined SANOS--LV failure ledger, then runs only
missing GROMM and generated cases. The obsolete `d22ec9a` coordinator and the
unbound first `de18586` waiter were stopped while still waiting and launched no
production case.

Weekly objective tables now
include equal-weight episode-book and worst-target maximum drawdown, computed
from chronological overlapping episode-origin P\&L and explicitly distinguished
from a non-overlapping self-financing-account drawdown. Fitted-policy parameters
and ex-post risk-score parameters have distinct headings and the correct
$\kappa$ or $\lambda$ symbol. The target appendix separately reports authenticated
inception market, accounting, genuine-Q, buyer-indifference, and
seller-indifference price means and rejects disagreement between stored price
records and realized accounting. Final completed-case figures, production-data
PDF regeneration, the final complete-suite rerun, and the final results review
remain outstanding.
The independent Python contract re-review is recorded in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-final-report-contract-rereview.md`.
The independent post-production acceptance procedure is fixed before outcomes
in `quantdata/sdlv/daily/front_spot/hedging/reviews/final-results-review-protocol.md`;
it requires record-level population, objective, price, cost, inference,
rendering, provenance, and public-data-boundary reconciliation.
The requirement-by-requirement acceptance state and the evidence still needed
for completion are tracked in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-15-plan-completion-audit.md`.

## Minimum Test Gates

Before any large experiment, verify:

- No look-ahead in windows, marks, normalization, or model selection.
- Stable contract identity across daily marks.
- Frozen weekly universes contain no newly listed contracts.
- Correct mark-plus/minus-half-spread side for every trade direction.
- Zero transaction cost at cash settlement.
- One-basis-point spot cost applied to traded notional only.
- Exact buyer/seller P&L sign symmetry where expected.
- Pricing plus hedging decomposition reconciles to economic P&L.
- Density mass and forward invariants.
- Finite positive DLV-squared values.
- Distinct and correctly labelled `P` and `Q` outputs.
- Train/validation/test separation.
- Deterministic fixture reproducibility.
- Checkpoint restart equivalence within numerical tolerance.
- CPU/GPU agreement on the same small fixture.

## Initial Deliverable

The first implementation deliverable is a CPU-only vertical slice containing:

1. The common contract and accounting APIs.
2. One ATM one-day digital claim.
3. One SANOS-LV market price and hedge.
4. One independent covariance hedge.
5. Common pricing-versus-hedging P&L decomposition.
6. Standalone smoke commands for both models.
7. Protocol and accounting quant-agent reviews.

This exercises the full benchmark path before introducing large simulations,
deep hedging, or GPU execution.

## Resolved Production Decisions

The four GROMM hedge-policy candidates are fixed as
`(LSTM hidden size, deep depth, deep width)` equal to `(16,1,32)`, `(32,2,64)`,
`(64,2,128)`, and `(64,3,128)`, all with one recurrent layer, SiLU activation,
and the common affine action. The authenticated 2022-only selection study
evaluated all four over 40 measure/target cases and applied the precommitted
cheapest-within-one-standard-error rule.  It selected `(64,3,128)` with
validation loss `0.08181613770898842`; the one-standard-error threshold was
`0.09032001233718524`.
The selection artifact has SHA-256
`ed68dd71a87465507191f476cf85eea691596751fa122d95ac21c7381abb166b` and is
stored outside the source tree at
`/home/hans/quantdata_data/training/hedging/gromm/architecture-2022-v2-fe7c3a7/result.json`.

The episode risk unit, target-notional rule, 1,000-unit numerical position cap,
common spot-cap convention, statistical PCA alignment tolerance, and
inception-forward barrier reference are also resolved in the implementation
and their dated review or calibration reports. No benchmark-design decision
remains open before production; incomplete production cases remain gated only
on their required authenticated model evidence.
# SANOS-derived rolling Greek comparison

After the daily SANOS-derived 9-by-200 state build and v2 dataset validation,
fit two distinct eight-factor PCA models on the last 252 exact one-session
changes in the daily raw state, using only transitions ending on or before the
hedge decision date. This is a SANOS factor estimate, not the substantially
more expensive rolling DYSANOS AR(1) fit. SANOS Greek, SANOS parametric,
SANOS statistical, and SANOS normal-statistical are four hedge methods applied
to the same SANOS market model. The ordinary Greek variant applies PCA to centered
state changes and retains their empirical correlation with normalized spot
returns. The spot-orthogonal variant first projects each state change onto
the normalized spot return and applies PCA to the residual. Its fitted
spot-correlated state loading remains a separate, explicitly reported state
shock rather than being discarded. Both variants use the same dates, state
coordinates, target/hedge contracts, costs, eight-factor rank, and one-standard-
deviation factor bumps.

For both variants, the spot *partial* Greek moves spot while holding the raw
state fixed. Factor Greeks subsequently reprice each fixed-cash-strike
contract after a state bump. For the orthogonal variant, distinguish the
fixed-state spot partial from the additional price exposure to the fitted
spot-correlated state loading; only their sum is the exposure to the
orthogonalized joint spot shock. Do not silently call that sum the spot
partial. No hedge result from the withdrawn Greek or parametric runs is a
substitute for this matched rolling comparison.
