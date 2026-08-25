# FLEX Phase 3: weekly physical-measure model and Torch simulator

> **Shared-strikes implementation update (2026-08-20).** The original
> 140-state Strong results below remain milestone history. The active branch is
> now the shared-strikes 2,828/200-state program described in the final section
> of this file. It implements, rather than merely defers, exact-week normal and
> generic target-measure kernels. Results are filled from immutable reports as
> the persistent pipeline completes.

## Status

Phase 3 implements the base historical-measure model (P). It does not fit or
parameterize a change of measure. The later risk-neutral/HJM exercise may alter
the exact one-week conditional law directly, but that is research outside the
current calibrated model.

This phase is structural research toward that later risk-neutral-measure
problem. Market fit and temporal validation quality are reported diagnostics,
not production acceptance thresholds. The hard gates are genuine optimizer
convergence, finite and non-collapsed states and innovations, stable SANOS
dynamics, preserved pricing invariants, and reproducible simulation. A weak but
non-degenerate empirical result is therefore retained and described honestly.

The static surface representation is the Phase 2 **Strong** triangular model:

```text
(16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5)
```

This gives 140 state coordinates and a frozen affine triangular map to the
2,828 bounded native SANOS parameters. Two state histories are distinguished:

- **production history:** a market-to-Strong encoder trained only through 2024
  and frozen for the 2025 test;
- **direct oracle:** the independently optimized Phase 2 per-date Strong
  states. This is useful for testing the dynamics capacity but is not a live
  market-to-state result. Its coordinate system was fitted on all Phase 2
  dates, including 2025, so its OOS statistics are explicitly labelled oracle.

The production encoder is a fixed Gaussian-RBF set summary over expiry and NMS,
followed by a two-hidden-layer width-512 SiLU network. Training first matches
the train-only Strong states and then fine-tunes the real equal-date quote
objective through the frozen Strong SANOS decoder. Its completed checkpoint
stores all encoded states, dates, market fingerprints, and `market.py`
reference spots with SHA-256 proofs.

## Flat native-state encoder controls

The active diagnostic branch tests whether validation loss arises before the
hierarchical state construction.  It uses the same fixed 14-by-11
expiry/NMS Gaussian-RBF quote summary and two width-512 SiLU hidden layers, but
maps the market to native SANOS raw coordinates without the Strong triangular
decoder:

- the full control has 2,828 explicit surface coordinates and an identity
  standardized-native output map;
- the compressed control has 140 explicit surface coordinates and one affine
  140-to-2,828 native decoder.  Its initial gauge is the exact training-only
  PCA of standardized native raw states, after which both encoder and affine
  decoder are fine-tuned jointly through the quote objective.

Both controls first learn the independently calibrated daily native states on
2022--2024, must pass an explicit convergence gate, and then minimize the real
equal-date bid/ask-plus-weak-fit objective.  The untouched 2025 dates are used
only for temporal validation.  This is same-set amortized calibration: all
eligible books for a date enter the set summary and the same books are priced.
It is not masked-quote valuation.

In this branch, a raw DLV coordinate represents
`0.01^2 + exp(z)`.  Pricing mechanically caps it at `3^2`; a separate
half-squared softplus barrier with log width `0.10` and log standard deviation
`0.35` supplies the restoring gradient at and above the finite cap coordinate.
Forward volatilities and forward mus retain their sigmoid bounds.  The retained
Strong checkpoint keeps its historical sigmoid-DLV gauge and is never silently
reinterpreted.

Both controls converged.  On the frozen 2025 same-set calibration task their
results are:

| representation | date objective | all inside bid/ask | weekly inside bid/ask | positive excess mean | positive excess median |
|---|---:|---:|---:|---:|---:|
| direct per-day native ceiling | 1.59667 | 69.1676% | 83.1603% | 4.89178 | 1.48363 |
| flat native 2,828 | 7.11224 | 28.2805% | 26.1971% | 9.30596 | 6.16143 |
| flat native 140 | 8.15621 | 26.2271% | 24.4106% | 10.4545 | 7.09186 |
| global market-to-Strong 140 | 8.75539 | 25.1493% | 23.4330% | 10.9964 | 7.25179 |
| hierarchical market-to-Strong 140 | 11.5404 | 23.7739% | 20.1537% | 13.9197 | 8.60876 |

The 140-dimensional affine control retains `99.6769%` of standardized native
training variance.  After quote fine-tuning its train objective is `1.08729`,
only `1.68%` above the full control's `1.06931`.  On 2025, compression costs
2.0533 all-option containment points and 1.7866 weekly points relative to the
full control, while still beating the global Strong encoder.  This makes the
dominant problem the temporal generalization of market-to-state inference,
not the hierarchical decoder alone.

The capped-unbounded gauge also exposes an OOS extrapolation warning.  Training
has no mechanically capped DLV nodes in either flat control, and the direct
2025 ceiling has none.  The full encoder sends 369 of 453,600 validation nodes
above the cap (`0.08135%`), while flat 140 sends 1,074 (`0.23677%`).  Their mean
validation surface cap penalties are `6.275` and `71.255`, respectively, with
highly concentrated maxima.  This is evidence of market-encoder extrapolation,
not successful use of the upper DLV range; those diagnostics must accompany
any future dynamics fit.

The production encoder-state dynamics now use the accepted **DYSANOS/grid
hierarchy**. Both the reduced 118-dimensional and full 140-dimensional fits
pass the matrix-log, transient-power, invariant-law, decoded-saturation, and
Torch path-generation gates. Forecast means beat persistence at 1W, 2W, and
4W. Their Gaussian log scores remain slightly worse than the unconditional
control, which is retained as an honest research limitation rather than a
stability failure.

Every FLEX command-line entry point runs inside a shared durable failure
boundary. Uncaught main-process exceptions, catchable termination signals,
uncaught worker-thread exceptions, and unraisable exceptions are written as
atomic, fsynced JSON records beside the requested report output before the
original nonzero exit is preserved. A temporary `faulthandler` log remains on
disk only when native code or the interpreter terminates too abruptly to reach
the Python exception handler. Independent systemd execution, persistent
stdout/stderr, checkpoint phases, and MonitorWorker supervision cover external
`SIGKILL`, OOM, and host failures that no in-process `try/except` can catch.

## Data and split

The source is the filtered SPX market pipeline in `quantdata/dysanos/market.py`
and `quantdata/dysanos/flex/market.py`.

| Split | Dates | Books | Use |
|---|---:|---:|---|
| 2022-01-03 to 2024-12-31 | 712 | 2,784,378 | Strong encoder and (P)-dynamics training |
| 2025-01-02 to 2025-08-29 | 162 | 708,404 | untouched temporal validation |

No 2026 observation is loaded. Quote objectives first average within each date
and then across dates; a date with more listed options therefore receives no
extra weight. Dynamics use exact endpoints separated by five NYSE sessions,
giving 671 overlapping training transitions and 155, 150, and 140 validation
pairs at 1W, 2W, and 4W.

## State PCA

Write the Strong state by expiry as


\[
x_t=(x_t^1,\ldots,x_t^{14}).
\]

Each block has its own training-only PCA. Full mode retains every one of the
140 coordinates. Reduced mode selects the smallest block rank that explains at
least 99.5% of that block's level variance. On the production market-encoder
history the selected ranks are

```text
(8, 13, 13, 12, 12, 10, 10, 10, 5, 5, 5, 5, 5, 5), total 118.
```

PCA coordinates are centered on the 2022--2024 training mean and standardized
by their population standard deviations before fitting dynamics. The daily
affine regression estimates its own equilibrium; the weekly transition and
affine shift preserve that same equilibrium. Standardization makes the
stability controls substantially less dependent on the arbitrary affine gauge
of the Strong latent state.

## Hierarchical weekly dynamics

For expiry block (j), the fitted adjacent-session equation is

\[
z_{t+1}^j=c_j+z_t^jB_{jj}
 +\mathbf 1_{j>1}z_t^{j-1}B_{j-1,j}+e_{t+1}^j,
\]

with no preceding-expiry term for (j=1). This is lag-only: it deliberately
excludes the contemporaneous regressor (z_{t+1}^{j-1}). Assemble the complete
block-upper-bidiagonal one-session transition (B), require its principal
matrix logarithm to be real, stable, and numerically reliable, and define

\[
K_d=-\log B,
\qquad A_{1W}=\exp(-5K_d)=B^5.
\]

No coefficient scaling, symmetrization, eigenvalue clipping, or silent fallback
is applied. The logarithm round trip is a hard gate. So are the maximum
standardized transition-power norm and the ratio of fitted invariant variance
to empirical training variance. The accepted production fits are:

| Variant | dim | spectral radius | max standardized (\lVert A^n\rVert_2) | invariant/training variance |
|---|---:|---:|---:|---:|
| reduced | 118 | 0.82869 | 1.25708 | 0.94824 |
| full | 140 | 0.86185 | 1.34275 | 0.95280 |

The corresponding longest front-expiry half-lives are 3.69 and 4.66 weeks.
This is consistent with the observed front forward-volatility and DLV series;
the former 138-week result was an artifact of symmetrizing and clipping the old
transition, not an empirical estimate of volatility persistence.

The fitted weekly Gaussian innovation covariance is unrestricted across the
surface state. This preserves material cross-expiry residual covariance without
making a contemporaneous regressor endogenous. Spot is not stationary. Its
weekly log return is

\[
\Delta s_{t+1}=\mu_s+\beta^\top u^1_{t+1}
                 +\sigma_\perp z_{t+1},
\]

where (u^1) is the whitened front-expiry innovation and (z) is an independent
normal. No later-expiry shock enters spot directly. On the production encoder
history the front innovation explains 55.3% and 54.0% of weekly spot-return
variance in reduced and full mode respectively.

The exact 1W transition and innovation factor, and composed 2W/4W moments and
factors, are precomputed once. The Torch step contains only matrix operations
and Gaussian draws. Spot remains a unit-root output while the SANOS surface is
stationary.

### Resolution of the skeptical review

The skeptical review correctly rejected the former projected hierarchy despite
its stable eigenvalues. The replacement addresses the immediate objections by
standardizing the fitted coordinates, removing contemporaneous dependence and
all post-fit coefficient projections, retaining full residual covariance, and
making transient-power and invariant-law checks hard failures. The exact
comparison and rejection history are preserved in
`PHASE_3_DYNAMICS_ESTIMATOR_COMPARISON.md`.

This hierarchy remains HJM/Musiela-inspired rather than an HJM theorem: expiry
ordering is built into the conditional mean, but future no-arbitrage conditions
must be expressed in decoded prices or roll moments when the sampling measure
is changed.

## Torch simulation and option returns

`TorchWeeklyDynamics` and `WeeklySanosSimulator` keep state generation, Strong
decoding, SANOS pricing, and option returns in Torch. Gradients propagate from
returns through simulated shocks, PCA reconstruction, the Strong decoder, and
SANOS prices.

The primary `normal` spot mode is the calibrated (P) model. `implied` mode
uses the pre-shock front-week SANOS inverse CDF to transform the correlated
Gaussian spot coordinate. This exactly reproduces the front-week SANOS return
law, but is only a structural test mode in Phase 3; it is not labelled as the
historical-(P) calibration or a completed risk-neutral model.

The tradable universe uses fixed inception log-strikes for each weekly expiry.
It keeps one ITM side by default, supports OTM mode, and always includes one
zero-strike call per expiry as the delta hedge. Returns support:

- `period`: one-week value change of the same fixed-cash-strike contract;
- `expiry`: entry value to cash settlement, using the required (2M-1)
  internal weekly steps when (M) entry dates and (M) expiries are requested.

Expiry settlement has no option exit half-spread. Zero-strike calls bypass the
option spread model and use their own spot cost.

## Spread costs

Spread attribution is trained on 2022--2024 and tested on the same untouched
2025 interval, with equal total weight per date. The target is normalized-price
half-spread, not volatility spread.

| Model | 2025 MAE | 2025 RMSE | 2025 (R^2) |
|---|---:|---:|---:|
| linear | 0.0002355 | 0.0003378 | 0.1399 |
| histogram gradient boosting | 0.0001972 | 0.0002881 | 0.3745 |
| Torch feed-forward, eligible books | 0.0001364 | 0.0002577 | 0.4997 |

The new simulation model is a two-hidden-layer width-64 SiLU network. It uses
vega, square-root TTE, NMS, log strike, signed delta, and log dollar option
price, predicts log normalized half-spread, and stays in Torch during path
generation. Training excludes approximate dollar premiums below $0.25, which
retains 99.960% of training and 99.984% of validation books. Inference applies
an explicit half-tick floor: $0.025 below $3 premium and $0.05 otherwise. The
dollar premium approximation is normalized AF price times spot, avoiding rates
and forwards. These filter/tick choices are ad hoc and visible in the report.

For 2025 ITM books, the feed-forward median absolute error in proportional
half-spread is `0.00444`, versus `0.00555` for the tree benchmark. Its predicted
median proportional half-spread is `0.01306` against actual `0.00848`, so the
remaining conservative upward bias is material despite the better aggregate
fit. During simulation the SANOS strike-total-volatility curve supplies the
differentiable ATM-volatility proxy for NMS and Black Greeks. The predicted
half-spread is divided by live option price to form proportional cost.

## Code and artifacts

Core modules:

- `market_encoder.py`: fixed-RBF market-to-Strong model and certified history;
- `failure_reporting.py`: atomic process, signal, thread, and native-crash
  diagnostics shared by every executable FLEX entry point;
- `weekly_dynamics.py`: PCA, hierarchical fit, matrix-exponential gate,
  serialization, and Torch transitions;
- `weekly_simulation.py`: normal/implied paths, SANOS decoding, option returns,
  costs, and future HJM residual helper;
- `weekly_validation.py`: exact 1W/2W/4W likelihood and baseline statistics;
- `weekly_simulation_validation.py`: real-checkpoint GPU/path diagnostics;
- `spread_cost.py`: date-balanced linear, tree, and Torch feed-forward spread
  attribution with explicit tick policy;
- `PHASE_3_DYNAMICS_ESTIMATOR_COMPARISON.md`: estimator comparison, rejection
  rationale for the superseded hierarchy, and accepted production diagnostics;
- `run_phase3_encoder.py`, `run_weekly_phase3.py`, and
  `run_spread_benchmark.py` and `run_spread_ffw.py`: experiment entry points.

The completed production report is
`reports/phase3-weekly/phase3-weekly-encoder.json`; its reduced and full
checkpoints are under
`/media/storage/quantdata/training/flex-phase3/weekly-dynamics-grid-v1/`.

## Ad-hoc choices and open research issues

- Strong's expiry allocation remains hand-designed from Phase 2 quote tests.
- RBF anchor counts/bandwidths and the width-512 encoder are ad hoc.
- The 99.5% PCA threshold, exact five-session horizon, transition-power cap,
  invariant/training variance bounds, and zero-ridge main fit are ad hoc.
- Static latent coordinates have an arbitrary within-block gauge; dynamics
  need a whitened or price/Jacobian-weighted canonical metric.
- Gaussian innovations do not fit the observed joint tails.
- The differentiable spread model uses an ATM-volatility proxy rather than a
  live implied-vol inversion; its price filter, tick schedule, network, and
  log-Huber objective are ad hoc.
- Compiling the complete path/SANOS decoder remains a performance follow-up;
  only compatible inner numerical kernels currently compile.
- Implied-spot mode fixes the front marginal only. Enforcing weekly martingale
  roll restrictions for later expiries, and parameterizing a direct measure
  change at the one-week matrix-exponential step, are explicitly deferred.

These limitations are acceptable for the present structural-research goal as
long as the process remains finite, stationary in SANOS state, non-collapsed,
and reproducibly calibrated under (P).

## Active shared-strikes program

### State and decoder contract

For expiry `j` and market date `t`, the notation is now fixed:

- `x[t,j]` contains the constrained SANOS parameters: 200 DLV-squared values,
  one forward volatility, and one forward backbone multiplier;
- `u[t,j]` is the corresponding 202-dimensional unconstrained raw vector;
- `h[t,j]` is the compressed real-support state used by dynamics.

Full mode sets `h=u`, with 14×202 = 2,828 coordinates and an identity decoder.
The compressed controls use 200 total coordinates. Direct-200 has one global
affine map. Hierarchical-200 uses the expiry-triangular allocation

```text
(23, 27, 25, 22, 19, 14, 14, 14, 7, 7, 7, 7, 7, 7).
```

Every expiry uses the same relative strike support within a date. The support
merges 100 front-expiry and 100 last-expiry NMS-implied candidates, applies the
minimum-spacing rule, and floats between dates. This is the active
`shared strikes` geometry, distinct from the retained `nms strikes` control.

The independent daily native calibration completed on 891 usable dates and
3,569,180 options. Across the complete surface, 73.4281% of quotes were inside
bid/ask. Conditional positive excess had mean 3.99035 and median 1.35209
half-spreads. The date-balanced objective was 1.13941. Short-end diagnostics
remain much worse: inside percentages were 25.8848%, 24.3783%, and 33.2227%
for 0DTE, 1DTE, and 2DTE. These figures are the representation ceiling before
market-to-state generalization, not an encoder result.

### Physical weekly law

Each accepted state history is split at 2024-12-31; 2025 is untouched for
estimation. Within every expiry block, the code evaluates both all-component
PCA (`pca-all`) and the smallest PCA retaining at least 99.5% of training
level variance (`pca-995`). `pca-all` is a coordinate rotation, not additional
compression of `h`.

In standardized PCA coordinates, the DYSANOS/grid estimator is

```text
z[t+1,j] = a[j] + z[t,j] B[j,j]
                     + 1{j>1} z[t,j-1] B[j-1,j] + e[t+1,j].
```

The direct-200 special case has one block and therefore estimates a global
full matrix. A reliable real stable `K=-log(B)` is required; no clipping,
diagonalization, or silent estimator fallback is permitted. The exact weekly
transition `exp(-5K)` and exact five-session innovation covariance are
precomputed outside the simulator. Spot remains a unit-root output. Historical
likelihood and exact 1W/2W/4W forecast diagnostics are reported only under P.

### Exact-week target laws

The Q simulator always draws the gross spot return from the current front SANOS
inverse CDF. Thus spot and the expiring front-week option share the exact
decoded martingale law. The target kernel changes only the assembled one-week
surface transition; it never attaches likelihood weights to pre-generated P
paths.

The reference joint weekly covariance is factored in the ordering
`(spot rank, expiry-1 state, ..., expiry-M state)` by a lower Cholesky factor.
This makes the stochastic construction expiry-causal. For hierarchical mode,
line `r` produces the next state for remaining expiry `r` and is conditioned on
the current prefix through expiry `r+1`, realized implied spot, and already
drawn lower lines. The longest new boundary line stays at its regularized P
proposal because the fixed surface has no current `(M+1)`-week option to
identify it. Direct mode fits one joint 200-dimensional kernel.

The `normal` candidate is conditionally Gaussian line by line. It learns a
context-dependent mean and positive diagonal scale plus a nonsingular local
lower-triangular factor. Because spot is deliberately non-Gaussian and later
hierarchical lines condition on earlier realized lines, the complete joint law
need not be multivariate Gaussian; `normal` means conditional-normal in this
construction.

The `generic` candidate implements the requested blend in standardized-noise
coordinates:

```text
F_i(z) = shift_i + scale_i z_i + amplitude_i tanh(z_i) + L_i z_<i
z_i^Q  = (1-g_i) z_i + g_i F_i(z).
```

Here `scale_i>0`, `|amplitude_i| < 0.9 scale_i`, `g_i` is sigmoid-bounded, and
`L` is strictly lower triangular. Consequently every diagonal Jacobian entry
is strictly positive and the map is onto real space. This supplies a
triangular full-support bijection, starts exactly at the P identity, and gives
an explicit equivalence diagnostic.

Both candidates minimize fixed-cash-strike weekly roll moments

```text
E_Q[R[t+1] c[t+1,j-1](k/R[t+1]) - c[t,j](k) | state at t] = 0.
```

Hierarchical lines are trained and frozen front-to-back. A batch averages 128
paths separately for each of 16 equally weighted date origins before squaring
the conditional moment. Training uses 11 log-moneyness strikes from -0.30 to
0.30 and a `1e-4` penalty toward P. The objective is compiled by default.
Every line must improve a fixed Monte Carlo audit (or meet the absolute
tolerance), is checkpointed every 25 epochs, and is then scored on 2025 current
states. Finished kernels must also pass a 256-path, 520-week finite/non-explosion
test; the generic kernel reports its minimum triangular Jacobian diagonal.

These network widths, strike count/range, 200 epochs per line, Monte Carlo
counts, regularization, convergence threshold, and long-horizon gate are
explicitly ad hoc research choices.

### Active code and autonomous execution

- `run_shared_strikes_full_encoder.py`: full 2,828-state validation control;
- `run_shared_strikes_compressed_encoders.py`: clean direct/hierarchical 200
  decoders and market encoders;
- `run_shared_strikes_weekly_dynamics.py`: P fits and 2025 likelihood tests;
- `weekly_measure.py`: normal and generic exact-week kernels;
- `weekly_measure_training.py`: sequential conditional-moment training;
- `run_shared_strikes_weekly_q.py`: Q experiment and long-run validation;
- `monitor_services.py`: independent heartbeat/resource watchdogs.

All long jobs run as checkpointed user `systemd` services with user lingering
enabled. GPU work is restricted to physical GPU 0. Market LP/preprocessing
processes are centrally capped at four. A watchdog restarts a failed or stale
stage at most twice, then writes a serious error and stops its watched project
jobs. Staleness requires both an old journal heartbeat and no increase in the
worker cgroup's cumulative CPU time, so a long silent matrix calculation is not
mistaken for a dead process. The watchdog services themselves restart on
failure. RAM and the persistent/temporary filesystems configured through
`quantdata.env.locations` are monitored continuously.
User-manager lingering was verified while the login state was closing, so the
chain does not depend on this Codex chat or an attached shell remaining alive.

### 2026-08-20 shared-strike encoder and capacity checkpoint

The full 2,828-state market encoder completed on 891 dates. Its 2022--2024
training fit put 73.26% of all quotes and 86.55% of weekly-expiry quotes inside
bid/ask. On the untouched 2025 dates those figures fell to 26.28% and 24.57%,
whereas independent direct calibration of the same 2025 surfaces achieved
70.87% and 85.41%. The weak market-to-state validation therefore exists even
without state compression or a hierarchical decoder; hierarchy cannot by
itself explain the encoder generalization gap.

The first affine PCA-200 capacity run failed explicitly. Its analytic
initialization had objective 2.9297, but the ad-hoc `2e-2` market fine-tuning
rate destroyed it and converged to 59.4953. No fallback checkpoint was
promoted. The recovery code corrects the capacity objective to average within
each date and then equally across dates. The immutable native checkpoint
retains its original quote-weighted objective only for exact certification;
compressed-state training and comparisons use the equal-date objective.

Before the clean 200-state train/validation run, four 50-epoch direct-decoder
pilots were started from the identical analytic PCA state on the 729 training
dates. Constant `1e-4` reached a held re-evaluation objective of 1.32190 and
61.83% bid/ask containment. Cosine `3e-4 -> 3e-5` reached 1.19535 and 67.79%.
Adding ten warm-up epochs reached 1.18949 and 67.78%, with a maximum DLV-cap
penalty of only `7.01e-5`. A `1e-3` peak was rejected: its training objective
rose to 4.5726 as the warm-up reached the peak. The production decoder schedule
is therefore ten warm-up epochs followed by cosine decay from `3e-4` to a
`3e-5` floor. These are training-split decoder diagnostics; only the subsequent
market encoder produces a 2025 validation result.

### 2026-08-22 jointly learned Direct-200 dynamics

The first joint experiment tests whether the affine Direct decoder and the
physical dynamics should be identified together rather than sequentially.  It
uses one unconstrained full-support state `h[t]` in `R^200` and

```text
u[t] = b + W h[t]
x[t] = SANOS_constraints(u[t]).
```

The initial `(h,W,b)` is an exact whitening/re-gauging of the accepted
train-only Direct-200 model: its maximum change in any decoded raw SANOS
coordinate is `2.84e-14`.  The jointly learned weekly physical law is

```text
h[t+1] = m + (h[t]-m) exp(-K) + epsilon[t+1]
K = L L' + 0.001 I + (R-R')/2
epsilon ~ Normal(0, C C').
```

Thus `K` is full and non-diagonal, while its symmetric part is strictly
positive.  `C` is a full learned Cholesky factor.  Training alternates one
pass of equal-date quote minibatches with one exact-week Gaussian-NLL step over
all 703 training transitions.  The loss adds the DLV-cap barrier, ramps the
dynamics weight to `0.10` over 50 epochs, and uses a `0.10` factor-gauge
penalty that keeps `h` near zero mean and identity covariance.  State, decoder,
and dynamics peak learning rates are respectively `1e-4`, `3e-5`, and `3e-4`,
with 20 warm-up epochs and cosine decay to one tenth of each peak.  All these
weights and schedules, as well as the dissipative form of `K`, are ad hoc.

The 300-epoch fixed-weight convergence audit passed.  Joint learning preserved
and marginally improved the complete 2022--2024 static fit:

| fit | date objective | inside bid/ask | positive excess mean | positive excess median |
|---|---:|---:|---:|---:|
| sequential Direct-200 warm start | 1.05381 | 73.8084% | 3.75214 | 1.28545 |
| jointly learned Direct-200 | 1.04816 | 74.0066% | 3.76064 | 1.29333 |
| 2025 native-state least-squares projection | 6.94410 | 28.8348% | 9.31700 | 6.41428 |

The 2025 row is an oracle projection of independently calibrated native raw
states onto the learned affine manifold.  It is neither a market encoder nor a
quote-optimal per-day recalibration.  Its poor fit nevertheless shows that the
training affine manifold is not temporally portable under simple raw-state
least squares; joint dynamics training did not solve the existing validation
generalization problem.

The fitted physical transition is genuinely stationary: its weekly spectral
radius is `0.96377`, its slowest generator rate is `0.03691` per week, its
largest transition-power norm is `0.98797` at one week, and the norm decays to
`2.33e-17` after 1,040 weeks.  On the projected 2025 states:

| horizon | joint model RMSE | persistence RMSE | unconditional RMSE | Mahalanobis / dim |
|---|---:|---:|---:|---:|
| 1W | 1.20225 | 1.43533 | 1.17252 | 2.17261 |
| 2W | 1.20158 | 1.47331 | 1.17463 | 1.48267 |
| 4W | 1.18221 | 1.45490 | 1.18245 | 1.28003 |

The joint model clearly beats persistence, but is slightly worse than the
unconditional-mean forecast at 1W and 2W and only ties it at 4W.  The excessive
one-week Mahalanobis statistic also indicates under-dispersed OOS innovations.
This is a validation rather than optimization failure: on 2022--2024 the same
1W/2W/4W model RMSEs are `0.88131`, `0.95090`, and `0.97653`, all below the
corresponding unconditional RMSEs `1.00136`, `0.99936`, and `0.99526`; training
Mahalanobis statistics per dimension are close to one (`0.864`, `0.891`, and
`0.875`).

The experiment therefore succeeds as a stable, jointly trainable Direct
baseline, not yet as a strong out-of-sample physical generator.  The next
research controls are a quote-optimal 2025 frozen-decoder projection and a
hierarchical joint decoder/dynamics fit, before fitting any target-measure
kernel.

### Joint Hierarchical-200 control

The corresponding hierarchical experiment retains the Strong allocation

```text
(23,27,25,22,19,14,14,14,7,7,7,7,7,7).
```

Write its ordered full-support blocks as `h^1[t],...,h^14[t]`.  Raw SANOS block
`u^j[t]` is affine in only the prefix `h^1[t],...,h^j[t]`; applying the usual
DLV/forward-vol/mu constraints gives `x^j[t]`.  The accepted sequential
Hierarchical-200 checkpoint is re-gauged exactly by whitening each expiry
block separately.  This keeps the decoder triangular and preserves genuine
cross-expiry covariance instead of forcing the complete 200-vector covariance
to identity.

The joint weekly P law uses row-vector convention

```text
h[t+1] = m + (h[t]-m) exp(-K) + epsilon[t+1].
```

`K` is block upper triangular.  Each diagonal block is
`L_j L_j' + 0.001 I + (R_j-R_j')/2`; blocks above the diagonal are unrestricted
and blocks below it are exactly zero.  Hence all generator eigenvalues remain
in the stable half-plane, while next-expiry block `j` can depend on current
blocks `1,...,j` but never on later expiries.  `exp(-K)` has the same block
causality.  A full lower Cholesky innovation factor supplies the ordered
stochastic construction: shock block `j` may use shocks `1,...,j`.

Training alternates the same equal-date quote objective and exact-week
Gaussian likelihood as Direct-200.  Its gauge penalty standardizes means and
population covariances within each expiry block only.  The learning rates,
loss weights, Strong allocation, Gaussian innovation law, and stable-generator
parameterization remain ad-hoc research choices.

For 2025, raw-state least squares is recorded only as an initializer.  The
actual static capacity test freezes the trained decoder and P dynamics, then
learns one independent `h[t]` for every untouched 2025 date directly against
bid/ask excess plus the small AF-fit term.  This is a quote-optimal capacity
ceiling, not a deployable market encoder.  Dynamics diagnostics use those
quote-fitted states and report 1W/2W/4W errors against persistence and the
training unconditional mean, long-run transition norms, global and per-expiry
innovation calibration, and exact block-causality residuals.

The first 2025 state-only schedule completed all 600 epochs but did not pass
the strict plateau gate: its final quote objective was `1.717357`, its DLV-cap
penalty was `0.00107469`, and its last-50-epoch relative improvement was
`0.003306`, above the configured `0.002` threshold.  This checkpoint remains
`training`; it is not relabelled as converged and is never retried with the
same optimizer command.  An explicit continuation validates its schema,
model/source hashes, market fingerprints, completed epoch count, finite
history, state dimensions, and exact non-plateau failure signature.  It then
starts a fresh SparseAdam optimizer and cosine scheduler in a separately
versioned checkpoint.  The continuation runs 400 epochs from `1e-4` to
`1e-5`, without warm-up, and retains the same strict 50-epoch/0.2% convergence
gate.  The final report and completion proof are written only if that new
phase converges.

### 2026-08-23 quote-optimal 2025 states and hierarchical result

The earlier `28.8348%` Direct-200 validation containment was not a valid test
of the decoder's static capacity.  It was obtained by least-squares projection
of independently calibrated raw SANOS coordinates.  Raw-coordinate distance
is not the quote objective and gives equal importance to coordinates that can
have very different effects on option prices.  The corrected experiment keeps
the train-through-2024 decoder and dynamics frozen and learns one independent
`h[t] in R^200` for every 2025 market date by minimizing the same equal-date
bid/ask-excess objective used in calibration.  There is no temporal loss,
market encoder, or use of 2025 during decoder training in this capacity test.

| frozen decoder; independently learned 2025 `h[t]` | date objective | inside bid/ask | positive excess mean | positive excess median | surface excess mean |
|---|---:|---:|---:|---:|---:|
| native per-date SANOS ceiling | 1.37889 | 70.8692% | 4.41621 | 1.53551 | 1.28648 |
| Direct-200, raw-coordinate least squares | 6.94410 | 28.8348% | 9.31700 | 6.41428 | 6.63046 |
| Direct-200, quote-optimal `h[t]` | 1.45152 | 65.8973% | 3.97079 | 1.32440 | 1.35415 |
| Hierarchical-200, raw-coordinate least squares | 9.01629 | 27.4622% | -- | -- | -- |
| Hierarchical-200, quote-optimal `h[t]` | 1.64223 | 63.8836% | 4.43494 | 1.48851 | 1.60174 |

Direct-200 therefore retains `92.98%` of native validation containment.  The
hierarchical decoder retains `96.94%` of Direct-200 containment and `90.14%`
of native containment.  The hierarchy costs `2.01` containment percentage
points relative to Direct-200, so the triangular restriction is material but
not destructive.  Both learned-state optimizations passed their configured
convergence audits.  Hierarchical validation required the explicit fresh-
optimizer continuation described above; its last-50-epoch relative change was
`0.000653`, below the `0.002` threshold.

The short end remains intrinsically difficult.  Direct-200 containment is
`22.65%`, `19.72%`, and `32.85%` for 0DTE, 1DTE, and 2DTE respectively;
Hierarchical-200 obtains `22.82%`, `18.80%`, and `30.86%`.  These are close to
the independently calibrated native ceiling, so this particular weakness is
not evidence that 200-dimensional compression or hierarchy is failing.  It is
primarily a geometry/interpolation and quote-quality issue at expiries before
the first weekly model expiry.

The same frozen train-through-2024 P transitions were then evaluated on the
correct quote-optimal 2025 states:

| model and horizon | model RMSE | persistence RMSE | training-unconditional RMSE | Mahalanobis / dim |
|---|---:|---:|---:|---:|
| Direct-200 1W | 1.20384 | 1.44029 | 1.17720 | 2.18104 |
| Direct-200 2W | 1.20618 | 1.48048 | 1.17932 | 1.49344 |
| Direct-200 4W | 1.18662 | 1.46432 | 1.18668 | 1.28974 |
| Hierarchical-200 1W | 1.06585 | 1.33454 | 1.05165 | 1.40418 |
| Hierarchical-200 2W | 1.05770 | 1.37089 | 1.05322 | 1.08013 |
| Hierarchical-200 4W | 1.06156 | 1.36259 | 1.06196 | 1.00914 |

The Direct law beats persistence but is worse than an unconditional forecast
at 1W and 2W; learning the correct 2025 `h[t]` therefore confirms rather than
removes its weak OOS dynamics.  The hierarchical law is substantially better:
it beats persistence at every horizon, is close to unconditional at 1W/2W,
and narrowly beats it at 4W.  Its front-expiry 1W RMSE is `0.60109` versus
`0.67095` unconditional.  Its principal remaining calibration problem is
under-dispersion in later-expiry one-week innovations, whose per-expiry
whitened standard deviation rises toward roughly `1.65` at the long end.

Both processes are long-run stable.  Direct-200 has weekly spectral radius
`0.96377`, slowest generator rate `0.03691` per week (half-life `18.78` weeks),
and transition-power norm `2.33e-17` at 1,040 weeks.  Hierarchical-200 has
spectral radius `0.90282`, slowest rate `0.10224` (half-life `6.78` weeks),
transition-power norm `1.08e-46` at 1,040 weeks, and exactly zero forbidden
later-to-earlier generator and transition blocks.  The corrected capacity
result makes Hierarchical-200 the stronger current P-generator candidate.  It
does not yet constitute a market encoder or a Q-measure construction.
