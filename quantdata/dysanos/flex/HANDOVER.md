# FLEX research handover

Last updated: 2026-09-09 on Sommer WSL.

Historical snapshot, not live process status. Current DYSANOS work is indexed
in [MODELS.md](../MODELS.md) and [REVIEW.md](../REVIEW.md). The figures and
restart notes below retain their original experimental scope.

## Executive status

FLEX has two distinct workstreams that must not be conflated:

1. **Historical-law (`P`) surface dynamics.** The all-history weekly and
   daily-front/weekly-tail affine-200 models are complete. Both have stable
   fitted weekly AR(1) dynamics and presentation-ready market-fit, PCA, and
   invariant-distribution diagnostics.
2. **Existence and learning of a risk-neutral law (`Q`).** This work deliberately
   does not assume that `Q` is a reweighted or locally adjusted version of the
   fitted historical law. It asks whether a sufficiently flexible conditional
   transport can directly satisfy the finite collection of one-week option-roll
   moment restrictions. Fixed-state and six-state experiments succeeded. A
   single joint law over all 891 historical states has not yet passed independent
   validation and confirmation simultaneously.

The latest all-state `Q` run is paused. Its checkpoint reports `status=training`
because it was stopped between audits, but no worker is currently expected to
be running. Do not interpret that field as evidence of a live process. Preserve
the retained checkpoints until the next objective is agreed.

All persistent artifacts are under `/home/hans/quantdata_data`. Generated W&B
logs and obsolete failure
snapshots were cleaned from the source repository. The completed `P` models and
paused `Q` checkpoints were not deleted.

## Fixed-grid four-week front-loaded model

The requested four-expiry affine hierarchy is complete on 891 dates and
2,028,470 option books.  It uses explicit 5D, 10D, 15D, and 20D nodes, one
representative fixed relative-strike grid, and the state allocation
`(100, 50, 25, 25)`.

The first static run exposed a real implementation defect: conditional PCA was
performed in float32, whose numerical-rank tolerance discarded small but
nonzero front components 81--100.  The resulting checkpoint had 20 exact zero
front factors and correctly failed the dynamics whitening gate.  Conditional
PCA now performs its decomposition and rank decision in float64 before casting
the exact factorization to the model dtype.  The rejected rank-80 checkpoint and
report remain under `checkpoints/rejected-fits` and `reports/rejected-fits`.

The corrected static checkpoint has full block ranks `(100, 50, 25, 25)`.  Its
minimum block singular scales are `0.00108048`, `0.00484786`, `0.0130996`, and
`0.0186231`.  Both analytic-PCA and market convergence audits pass.  Overall
conditional-PCA explained variance is `99.998498%`; the per-expiry conditional
fractions are `99.999998%`, `99.988596%`, `99.754763%`, and `99.683204%`.

The 300-epoch stationary expiry-causal exponential AR(1) also passes its
trailing convergence gate.  The weekly spectral radius is `0.852897`, the
minimum real generator eigenvalue is `0.159114`, and both forbidden
strict-lower generator and transition blocks are exactly zero.  It uses 861
exact one-week pairs.  Quote fine-tuning improves the fixed-grid hierarchy from
`53.8048%` to `54.8860%` inside bid/ask, with mean excess `2.29803`, conditional
positive mean `5.09383`, and conditional positive median `1.99442` half-spreads.

This hierarchy does not improve quote fit over the non-hierarchical weekly
Direct-200 reference.  The latter reports `73.1560%` inside bid/ask, mean excess
`1.09137`, conditional mean `4.06558`, and conditional median `1.40734`, although
its aggregate universe contains 3,566,742 books through 70 business days and is
therefore not a like-for-like aggregate comparison.  On the identical 1D and
2D books, the hierarchy is lower by `3.95` and `4.27` inside-spread percentage
points, respectively.

A generated-surface audit drew 16 independent fitted-P innovations from every
historical state, for 14,256 one-week paths.  The float64 structural replay has
finite implied volatilities, density mass/mean errors below `7.5e-14`, and call
bound, strike-monotonicity, strike-convexity, and calendar errors at machine
precision.  The implied-volatility median is `17.22%`, the 99th percentile is
`53.80%`, and the maximum is `115.79%`.  Float32 decoding has worst-case density
roundoff around `3.8e-5`; it is not a structural arbitrage failure.

Artifacts:

- Static checkpoint:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/checkpoints/triangular-four-week-front-loaded-200-200.pt`
  (`d63ef6f27f5f92696df33f1dab9feb1eb11679f068231bc329bed11fe7b459bb`).
- P checkpoint:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/joint-hierarchical-front-loaded/checkpoints/joint-hierarchical-front-loaded-200.pt`
  (`6b2c485a7a8ecc8b936c5bfb6c22cb5b90e3f9bd07efd1a99aca894af37574fd`).
- P report:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/joint-hierarchical-front-loaded/reports/joint-hierarchical-front-loaded-200.json`.
- Float64 generated-surface audit:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/joint-hierarchical-front-loaded/reports/joint-hierarchical-front-loaded-200-generated-surface-audit-float64.json`.

## Completed all-history `P` models

Both models use 891 dates from 2022 through 2025-08-29, exclude the 0.2-day/AM
population, and use an affine SANOS decoder

`u_t = b + W h_t`, with `h_t in R^200`.

There is no additional nonlinear network on top of the affine decoder.

| Model | Expiry nodes (business days) | Option books | Native inside bid/ask | Affine-200 inside bid/ask |
| --- | --- | ---: | ---: | ---: |
| Weekly | 5, 10, ..., 70 | 3,566,742 | 73.2911% | 72.9744% |
| Daily-front + weekly | 1, 2, 3, 4, 5, 10, ..., 70 | 3,566,258 | 81.8431% | 80.0141% |

The weekly model extrapolates poorly before 5D. Adding explicit 1--4D nodes is
the main improvement; the 200-state affine compression is a much smaller fit
cost. The fitted dynamics are:

| Model | `rho(A)` | Slowest half-life | Invariant covariance effective rank |
| --- | ---: | ---: | ---: |
| Weekly | 0.959735 | 16.87 weeks | 193.7/200 |
| Daily-front + weekly | 0.953502 | 14.56 weeks | 193.7/200 |

Gaussian innovations are too thin-tailed relative to market state returns.
Replaying independently sampled historical one-week innovations is exact at 1W
by construction and is materially closer than the Gaussian law at 2W and 3W,
but remains lighter-tailed than the market.

Artifacts:

- Weekly:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/integer-weekly-no-am-all-history-v1`
- Daily-front + weekly:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1`
- Presentation handover: [`paper/FLEX_PRESENTATION.md`](../../../paper/FLEX_PRESENTATION.md)

## Independent `Q` question

### Exact-density DLV construction (2026-09-09)

A separate, non-learned construction now gives an exactly consistent finite
spot/surface law on the four-expiry fixed grid.  At each weekly transition it:

1. forms 16 symmetric four-factor DLV density branches whose probability-
   weighted density is the current density exactly;
2. samples the expiring discrete front node and an independent 10% Black
   increment;
3. conditions every surviving density on the revealed node through the SANOS
   transition; and
4. divides the discrete support by the revealed node, thereby expressing the
   same conditional cash law on the next floating relative grid.  A new far
   expiry is appended behind the original surviving contracts so the operation
   can be repeated four times.

The Black factor is genuinely 10%: eta is held deterministically at 0.40 and
the forward coordinates are solved so `forward_vol * eta` produces
`0.10 * sqrt(T)`.  Spot and DLVs are the only random quantities.  Every week
uses fresh adjacent antithetic pairs for all three sources: opposite DLV
cubature branches, inverse-CDF front-node ranks `u, 1-u`, and Black normals
`Z, -Z`.

The 21-state audit used 10,000 paths per state.  Across weeks 1--4, its maximum
coordinate absolute z-scores were `2.400, 3.203, 2.871, 2.968`; the
covariance-aware Fisher-combined state-test p-values were `0.871, 0.672,
0.852, 0.522`.  Exact option and density tower residuals were `1.11e-16` and
`2.78e-17`, and conditional mass/mean errors stayed below `7.3e-14`.

The heterogeneous audit has 5,000 independently shocked origin pairs (10,000
paths), explicitly includes all 891 historical states at least once, and fills
the remaining origins randomly with replacement.  An independent-seed
confirmation had unconditional maximum absolute z-scores `1.575, 1.970,
1.878, 1.040` and p-values `0.702, 0.414, 0.067, 0.317`.  Its state-feature-
weighted p-values were `0.027, 0.759, 0.251, 0.0089`; those high-dimensional
hard-spectral-cutoff p-values remain visibly less well calibrated than the
coordinate and deterministic tower diagnostics and must not be hidden.

The density-weighted realized four-factor covariance differs from the
empirical input covariance by at most `1.87%` in confirmation (RMS `0.55%`).
The raw unweighted comparison is worse because it assigns equal importance to
zero-density tail coordinates, at which DLV is unidentified.  Both versions
are retained in the reports.

For 10,000 heterogeneous paths the new method ran 40,000 path-week steps in
`8.84s` (`4,527/s`), including the `7.36s` construction of 5,000 distinct
initial kernels.  This is about 23 times faster than the earlier finite-h
Newton transition (`195/s`) but about 5.7 times slower than the local SDE
transition (`25,649/s`).  Unlike the earlier SDE/raw-coordinate-regrid test,
whose fixed-cash z-scores grew from 35 to 48, the new re-grid is a conditional
density transformation and preserves cash prices path by path.

The latest shared `2x2048` all-891 learned transport stopped after 100 sweeps
with development-monitor worst adjusted z `16.138` and without independent
validation or confirmation.  That comparison is not perfectly like-for-like:
the transport targeted 11 one-week instruments, whereas the DLV audit carries
all 44 initial expiry/strike contracts through their lives.  The qualitative
difference is nevertheless clear: the DLV construction enforces the full
density tower algebraically instead of estimating a finite mean penalty through
noisy gradient descent.

Artifacts:

- 21-state report:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/multistep-price-preserving-regrid-dlv-v1/report.json`
- All-state heterogeneous primary and confirmation:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/heterogeneous-multistep-price-preserving-regrid-dlv-v1`
- Core implementations: `fixed_grid_dlv.py`, `floating_grid_dlv.py`,
  `run_multistep_regrid_dlv_simulator.py`, and
  `run_heterogeneous_regrid_dlv_simulator.py`.

This establishes a risk-neutral law for the modeled finite discrete surface;
it is not a uniqueness result and does not claim a continuous-strike diffusion
between weekly dates.

#### All-state four-cycle and 100,000-path audits (2026-09-09)

The certified static checkpoint contains **891**, not 981, unique historical
states.  The new all-state runner records that discrepancy explicitly and uses
every one of the 891 certified states.  Paths are allocated nearly uniformly by
initial state, with the two members of every antithetic pair sharing the same
origin.

The long-horizon audit used 10,000 paths (5,000 independent antithetic pairs),
five or six pairs per initial state, and repeated the price-preserving weekly
re-grid for 16 weeks.  It found no accumulating floating-point instability:

- the all-state exact density and option tower residuals were `6.49e-15` and
  `1.67e-16`;
- across all 16 rolls the largest decoder-density residual was `2.35e-13` and
  conditional mass/mean errors stayed below `6.6e-14`;
- DLVsq stayed strictly within its configured `(0.0001, 9.0)` bounds, and the
  largest per-batch fallback-coordinate fraction reached `0.553%` at week 16;
- the spot-from-origin maximum absolute z was `2.126` over the 16 horizons;
  the rolling four-expiry option maximum absolute z was `2.417`, and every
  covariance-aware rolling-option p-value was at least `0.0839`.

The run did expose a small, real finite-grid effect rather than numerical
roundoff.  A path that reveals an outer discrete support node is absorbing for
the DLV component.  The fraction of fully collapsed front densities grew from
`0.01%` after week 1 to `0.48%` after week 16.  At week 16 the median front
density still had `23.44` effective nodes, but its first-percentile effective
node count was `1.15`, and the 99th-percentile largest atom was `93.22%`.
Thus the bulk remains nondegenerate over four cycles, while a small boundary
tail accumulates as expected on a fixed finite support.

The high-power four-week audit used 100,000 paths (50,000 independent pairs),
56 or 57 pairs per state.  For weeks 1--4, the original fixed-cash contracts
plus spot had maximum coordinate absolute z-scores `2.213, 1.636, 2.207,
1.871`.  The corresponding covariance-aware joint zero-mean statistics
`J = N m' Sigma^+ m` were `8.912, 5.224, 7.190, 4.282`, using retained
effective degrees of freedom `8, 7, 6, 6`; their omnibus p-values were `0.350,
0.633, 0.304, 0.639`.  The maximum absolute option means were `0.697, 0.941,
1.475, 1.488` bps of initial spot.  These are ordinary Monte Carlo
fluctuations around the exact state-by-state density tower, not fitted or
validation-set results.  Report z-scores with this joint `J`, effective
degrees of freedom, and p-value: coordinate z-scores are marginal diagnostics,
whereas `J` accounts for their covariance and tests the joint zero-mean null.
Per-state z maxima are retained but have only 56--57 independent observations
per state and thousands of simultaneous comparisons, so they are not a more
reliable rejection criterion than the exact per-state tower plus the aggregate
audit.

The 16-week run processed 160,000 path-week steps in `13.81s` (`11,587/s`).
The 100,000-path run processed 400,000 steps in `20.68s` (`19,346/s`) and used
about 23.9 GiB of the 32 GiB GPU at 96% utilization.  Artifacts are under:

- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/all-state-repeated-regrid-dlv-v1/16-week-10000`;
- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/all-state-repeated-regrid-dlv-v1/4-week-100000`.

The canonical runner is `run_all_state_regrid_dlv_audit.py`.  The 16-week
directory includes four complete selected-path trajectories used for the
four-cycle animation.

#### Floating 4W AR(1) boundary correction (2026-09-09)

The repeated-roll implementation above originally appended the newly issued
4W DLV slice by copying the surviving 3W slice.  This preserved the finite
density tower but was a degenerate floating-tenor boundary law: after three
rolls, almost all cross-expiry DLV variation descended from the original 4W
slice.  Across the four displayed paths, the RMS dispersion of
`sqrt(DLVsq)` across expiries fell to roughly `0.0002--0.0012`.

The canonical runner now fits a **last-expiry-only AR(1)**.  It takes the 200
raw `log(DLVsq - min DLVsq)` coordinates of the historical 4W slice, estimates
a training-only 25-factor PCA, and fits one unrestricted weekly VAR(1) to the
scores through 2024-12-31.  The model uses 703 exact weekly training pairs;
25 factors explain `99.9912%` of last-expiry curve variation and the fitted
weekly spectral radius is `0.908915`.  On 155 untouched 2025 pairs, raw-curve
forecast RMSE is `0.5902`, versus `0.6526` for persistence, a `9.56%`
improvement.  Gaussian innovations are antithetic.  Because the Gaussian law
is unbounded while DLV is not, each generated curve is restricted to the
coordinatewise training-period raw envelope before being mapped into the
strict physical DLV bounds.

At each weekly transition, old fixed-expiry 2W--4W contracts still determine
the new 1W--3W densities through the exact Q density tower.  The AR(1)
independently supplies only the newly issued floating 4W DLV slice; SANOS then
constructs its density as a static martingale extension of the new 3W density.
No old 5W expiry is introduced or assumed.

The 10,000-path, all-891-state, 16-week audit passes all numerical checks.
Median cross-expiry `sqrt(DLVsq)` dispersion is `0.0459` after week 1 and
`0.0569` after week 16, so the term structure no longer collapses.  The four
selected paths retain week-16 dispersions `0.0675, 0.1065, 0.0305, 0.0387`.
The exact decoder, density-mass, and density-mean residuals remain below
`3.1e-13`.  The mean raw-grid extrapolation fraction grows from `1.71%` to
`9.93%` over 16 weeks as floating supports move; the corresponding AR forecast
training-envelope clip fraction grows from `0.48%` to `1.00%`.  No coordinate
reaches the final physical DLV cap.

The independent 100,000-path four-week confirmation gives the following
original fixed-cash-book-plus-spot zero-drift results:

| Week | worst `abs(z)` | covariance-aware `J` | retained df | p-value |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.377 | 12.185 | 8 | 0.143 |
| 2 | 2.312 | 8.321 | 7 | 0.305 |
| 3 | 1.084 | 4.340 | 6 | 0.631 |
| 4 | 1.267 | 2.753 | 6 | 0.839 |

Thus the joint zero-mean null is not rejected at any of the four high-power
horizons.  The AR-boundary reports are under:

- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/all-state-repeated-regrid-dlv-ar1-boundary-v1/16-week-10000`;
- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/all-state-repeated-regrid-dlv-ar1-boundary-v1/4-week-100000`.

#### Fixed-expiry versus re-grid martingale diagnostic (2026-09-09)

The survivor re-grid is theoretically martingale-preserving.  Let `S_t=s`,
let `x_j` be the inherited cash-support multipliers, let `i` be the revealed
front node, and let `Y` be the independent mean-one front Black factor.  The
next spot and relative support are

`S_{t+h} = s x_i Y`, and `x'_j = x_j / x_i`.

Consequently `S_{t+h} x'_j = s Y x_j` path by path.  A fixed-cash option with
strike `K` therefore has exactly the same conditional value before and after
the coordinate division.  Averaging the residual Black factor uses the Black
semigroup; averaging the conditional DLV transition gives the selected surface
branch; and averaging the symmetric surface branches gives the starting
density.  Thus, for every surviving fixed cash strike and expiry,

`E_t[C_{t+h}(K,T)] = C_t(K,T)`.

This proof requires fixed cash strikes and option types within each weekly
step, a branch-invariant front density, unit-mass/martingale conditional
transition rows, the correct residual Black variance, and no interpolation of
surviving densities.  The implementation satisfies these conditions.  The
last-expiry AR(1) changes only the newly issued 4W contract and leaves all three
survivors unchanged.  QuantAgent independently reviewed and agreed with this
derivation.

The runner now decomposes each weekly option change, using identical random
draws, into (a) a fixed-strike/fixed-expiry surface-branch change and (b) the
spot-conditioning/re-grid change.  It also values every survivor directly on
the inherited cash grid and after relative re-gridding.  In the matched
100,000-path, four-step run these two valuations agree pathwise to at most
`6.7e-16`, and their z/J/p statistics are identical:

| Week | fixed-expiry branch p | direct/re-grid p | antithetic full p | max direct/re-grid difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.843 | 0.123 | 0.713 | 0.0 |
| 2 | 0.939 | 0.654 | 0.269 | 4.4e-16 |
| 3 | 0.512 | 0.429 | 0.585 | 2.3e-16 |
| 4 | 0.790 | 0.307 | 0.642 | 6.7e-16 |

This exonerates the re-grid as the source of the occasional low p-values.
The evidence instead points to ordinary multiple testing plus an unstable
asymptotic Wald calibration for highly collinear option moments.  The test
retains only 7--9 of 44 correlation modes using a hard `0.01` eigenvalue cutoff
estimated from the same sample.  The AR 16-week minimum `p=0.003854` has
Bonferroni-adjusted p `0.0617`.  Independently, the exact fixed-grid one-step
audit produced minimum state p `0.002871` across 21 states even though its
option tower is exact to `1.1e-16`; its Bonferroni-adjusted p is `0.0603` and
Fisher-combined p is `0.195`.  Moreover, the earlier week-2 rolling p moved
from `0.0179` with 10,000 paths to `0.945` with 100,000 paths.

The matched report is:

- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/fixed-expiry-vs-regrid-diagnostic-v1/4-week-100000-seed-20260912/report.json`.

The smaller calibration report additionally records p-value sensitivity at
correlation cutoffs `0.005, 0.01, 0.02, 0.05` and two independent
covariance/mean half-sample orientations:

- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/fixed-expiry-vs-regrid-diagnostic-v1/4-week-10000-calibration-seed-20260912/report.json`.

#### X-only matched fixed-support/re-grid audit (2026-09-10)

The all-state runner now supports a genuinely X-only mode by setting
`bs_backbone_annualized_volatility=0`.  The independent Black component is
removed exactly at the pricing and path layer; changing the otherwise generated
Black normals then has no effect on spot or options.  The native DLV density
branches are unchanged.  Zero backbone is now a valid floating-surface state,
and zero-volatility option pricing takes the intrinsic-value limit explicitly.

The fp64 run used all 891 certified states, 100,000 paths (50,000 independent
antithetic pairs), four one-week steps, 11 strikes per expiry, and seed
`20260918`.  It is a matched fixed-support/floating-coordinate decomposition of
one X-only process, not two independently specified dynamics.  The fixed cash
support is held through each weekly transition; after the front node is
revealed, division by that node restores a unit-mean relative density for the
next transition.  A nontrivial process cannot instead keep the same relative
grid after conditioning without violating that mean constraint.

The still-surviving original time-zero cash contracts plus spot gave:

| Week | worst `abs(z)` | maximum option mean, bp spot | `J` | df | p-value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.796 | 0.497 | 10.980 | 7 | 0.139 |
| 2 | 2.143 | 0.863 | 7.228 | 7 | 0.406 |
| 3 | 1.371 | 0.696 | 5.858 | 6 | 0.439 |
| 4 | 2.057 | 1.245 | 5.506 | 5 | 0.357 |

The freshly struck rolling four-expiry books gave pair-mean p-values `0.139,
0.128, 0.0208, 0.546` and worst coordinate absolute z-scores `1.796, 1.545,
1.317, 1.789`.  Week 3 rejects at an unadjusted 5% level but not under the
pre-specified four-look Bonferroni cutoff `0.0125`.  It has no coordinate above
`abs(z)=2` and its largest mean is `0.363` bp of spot.  Its p-value changes to
`0.0429, 0.0879, 0.295` at correlation cutoffs `0.005, 0.02, 0.05`, while the
two independent covariance/mean splits give `0.214` and `0.338`.  This is a
clear covariance-mode-calibration warning rather than evidence of re-grid
drift.

The fixed-strike/fixed-expiry DLV branch component alone has p-values `0.366,
0.538, 0.911, 0.945`; its largest option means by week are only `0.0046,
0.0056, 0.0013, 0.0050` bp.  Direct inherited-cash and relative re-grid prices
agree pathwise to at most `2.22e-16` in all four transition decompositions, and
their z/J/p statistics are identical.  The first-roll price residual is
`3.33e-16`; the exact initial option and density towers close to `1.67e-16` and
`6.49e-15`; and the largest later decoder residual is `1.77e-13`.

Numerical stability passed.  Boundary-node incidence is `0.003%, 0.003%,
0.003%, 0.006%` across the four weeks, too small to explain the p-values.  At an
outer node, X is absorbing and interior DLVs are price-unidentified; the
positive-density drift formula no longer applies there, and the implementation
fallback is an economically invisible numerical convention.  QuantAgent
independently reviewed the X-only construction, exact re-grid identity,
boundary interpretation, and covariance-test wording.

Artifact:

- `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/x-only-fixed-vs-regrid-dlv-v1/4-week-100000-seed-20260918/report.json`.

At an observed state `(s_0, h_0)`, the front-week spot distribution is sampled
first. For each sampled future spot `s_t`, a law generates the next front
surface state `h_t^1`. For each of 11 in-the-money call/put instruments, the
one-week roll return is

`R_j = next option value_j - current option value_j`.

The target is a conditional law satisfying

`E_Q[R_j | s_0, h_0] = 0`, for all 11 instruments,

up to independently measured Monte Carlo error. This is an existence problem
for *a* risk-neutral conditional law over the modeled finite set of instruments.
It is not a claim of uniqueness, equivalence to the fitted `P` law, or a complete
continuous-strike martingale measure.

The most flexible successful state-local law is a spot-conditional triangular
transport. Let `z_s` be the standard-normal transform of the sampled spot's
probability rank under the front-week mixture. Given the current front state,
`z_s`, and independent Gaussian noise `epsilon`, the network generates

`h_t^1 = T_theta(h_0^1, z_s, epsilon)`.

The triangular monotone transform has full continuous support and no artificial
discrete branch. Its context network has depth 3 and width 256. A single-state
law has its own parameters; the joint model shares one state-conditional
transport across historical states.

## Control variate and reported errors

The spot return is a denoiser, not part of the risk-neutral constraint. For each
historical state and each instrument, each dataset separately estimates

`beta_j = Cov(R_j, S_t / S_0 - 1) / Var(S_t / S_0 - 1)`.

The reported adjusted sample mean is

`mu_hat_j = mean(R_j) - beta_j mean(S_t / S_0 - 1)`.

`beta` and the spot mean are recomputed independently in training, validation,
and confirmation. Their gradients are detached. The raw return mean and raw
variance remain mandatory diagnostics. For the earlier pass/fail criterion,
the reporting scale was

`max(SE(R_j), abs(beta_j) SE(S_t / S_0 - 1))`.

Therefore the noise-scaled statistic is

`z_j = mu_hat_j / max(SE(R_j), abs(beta_j) SE(spot return))`.

This adjustment must not be described as changing the payoff, changing the
sampled spot, or constructing a density. It only removes correlated Monte Carlo
noise from the estimator.

## `Q` experiment history and measured results

### 1. One fixed state and 11 instruments

The fixed-state vector law converged and passed independent validation using
24,000 paths. The independent raw worst absolute mean/MC-error statistic was
`1.436`, below the target of 2. This established that the finite 11-instrument
moment problem is feasible at at least one historical state.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-fixed-state-exact-pooled-v1`
- Checkpoint SHA-256:
  `77df36bc60ba5bd5e67b706f43466e52ac4de3144e6d0ae53ffd068fd58838f7`

### 2. Five additional independent states

Five difficult states were fitted independently with the generic triangular
transport. All five completed and passed their independent validation and
confirmation gates, each using 24,000 fit, validation, and confirmation paths.
Together with the original state, this supplied six state-local teachers.

Artifact:

- `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-independent-five-v1`

### 3. One joint transport over six states

A single shared conditional transport was initialized from the six state-local
teachers and trained jointly. It passed:

| Dataset | Worst absolute noise-scaled adjusted mean | Worst-10% mean |
| --- | ---: | ---: |
| Independent validation | 1.568 | 1.130 |
| Independent confirmation | 1.261 | 0.680 |

This is the clearest positive evidence that the state-local solutions can be
represented by one network rather than requiring unrelated parameters for each
state.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-joint-six-v1`
- Checkpoint SHA-256:
  `e1a0c898f2595a136095e47fa937e293d6add3ed6e784fd54acce6a132571aa9`

### 4. One joint generic transport over all 891 states

The six-state network was distilled into an affine state initialization and
extended to all historical states. After 571 sweeps it was close but failed the
agreed simultaneous gate:

| Dataset | Worst absolute noise-scaled adjusted mean | Worst-10% mean | Gate |
| --- | ---: | ---: | --- |
| Independent validation | 1.977 | 0.810 | pass |
| Independent confirmation | 2.245 | 1.057 | fail |

The independent confirmation failure is small in aggregate but decisive for
the stated worst-constraint target. This result must be retained as a near miss,
not reported as an all-state risk-neutral solution.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/generic-joint-all-history-affine-init-v1`
- Checkpoint SHA-256:
  `bf7d0a91d071b360da884b100e1456732e6d301be9f86c256c541e9924039122`

An earlier all-state common-random vector audit also failed: its independent raw
worst statistic was `4.104`, while the raw worst-10% mean was `1.962`. The broad
tail was acceptable, but one or a few states remained problematic.

### 5. IID path redesign and smooth price-normalized objective

The next experiment corrected the sampling design so that every path contains
one genuinely paired spot draw and one surface-state draw. Each sweep uses
10,000 fresh paths split into ten 1,000-path cuts. Fixed validation and
confirmation datasets each use 100,000 paths split into ten non-overlapping
10,000-path blocks.

For block `b`, state `s`, and option `j`, let

`q_bsj = mu_hat_bsj / current_option_price_sj`.

The smooth objective is

`L = LSE_tau((mean_b q_bsj)^2) + 0.25 sum_r w_r LSE_tau(q_(r),sj^2)`,

where block losses are ordered from worst to best and
`w_r` is proportional to `0.5^r`. The first completed run used `tau=0.05`.
Training paths were fully refreshed each sweep. Validation and confirmation
also report exact ten-block sign-flip tests and their mutual consistency.

The loss converged numerically, but the martingale diagnostics did not:

| Dataset | Worst noise-scaled adjusted mean | Worst-10% mean |
| --- | ---: | ---: |
| Validation | 30.742 | 3.011 |
| Confirmation | 37.036 | 3.528 |

The run completed 1,465 sweeps and achieved a best validation loss of
`3.168e-6`. This is evidence that the smooth price-normalized loss can become
small while still missing the desired simultaneous statistical statement. It
is an objective-alignment/generalization problem, not evidence of arbitrage.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-all-history-iid10k-smooth-price-v2`
- Best-validation checkpoint SHA-256:
  `c0b273fbc9dca108399097e191db0758217c5788866a195fcdacea7829663caa`

A sharper tail-focused continuation used `tau=0.0015`. It was paused after 273
sweeps. Its latest training worst/worst-10% statistics were `9.072 / 1.336`, but
fixed validation remained `29.717 / 3.032`. The discrepancy confirms that more
epochs alone are not the current answer.

Artifact:

- Root:
  `/home/hans/quantdata_data/training/flex-shared-strikes/q-existence/vector-all-history-iid10k-smooth-price-tail-v3`
- Best-validation checkpoint SHA-256:
  `eeabb22c4aee28a9b67f0abec0b65820af89e94e52364da9cf508aae455f8bbe`

## What has and has not been established

Established:

- The finite 11-instrument conditional moment problem is solvable for multiple
  individual historical states.
- One shared generic network solves the six tested states.
- A joint all-history network can get close to the two-MC-error target.
- The generic transport does not need to be derived from `P`.
- One-to-one spot/surface path sampling and independent large audits are now
  implemented.

Not established:

- No all-891-state network has passed both validation and confirmation.
- No continuous-strike or all-expiry risk-neutral measure has been proved.
- The learned law has not yet passed equivalence, density-floor, innovation-rank,
  long-horizon, or sequential-expiry gates.
- A low price-normalized smooth loss is not itself a martingale validation.

A finite collection of option moments usually leaves many stochastic degrees of
freedom, so non-uniqueness is expected. Non-existence can still arise if the
requested moments lie outside the convex hull generated by the allowed support,
if the transport is too restricted, if positivity/integrability constraints
exclude the needed law, or if the option quotes themselves imply incompatible
conditional restrictions. The current near-miss does not diagnose any of these
as a genuine arbitrage. The successful state-local fits argue against a simple
pointwise feasibility obstruction; the present failure is at the shared-network
and validation level.

## Recommended next `Q` step

Do not resume the paused tail run unchanged. First choose a training objective
that is aligned with the final simultaneous statistical gate.

Recommended sequence:

1. Start from the retained all-history generic-transport checkpoint rather than
   from the statewise affine-vector law.
2. Keep the genuine one-spot/one-surface draw, 10,000 fresh training paths, and
   fixed independent 100,000-path validation and confirmation sets.
3. Keep raw return means, raw variances, adjusted means, and worst/worst-10%
   statistics in every report.
4. Optimize a smooth maximum of squared means using a detached but statistically
   meaningful scale. Compare price normalization against a stabilized estimate
   of `max(SE(R_j), abs(beta_j) SE(spot))`; do not silently change the reporting
   criterion when changing the loss.
5. Use gradient accumulation or larger effective training samples to reduce
   sweep noise before increasing model width.
6. Require validation and confirmation to agree blockwise. Stop only when both
   independently pass the same predeclared gate.
7. If a small set of states remains, inspect its option-payoff Jacobian and the
   singular values of the moment map. That distinguishes insufficient network
   sensitivity from a potentially incompatible moment system.

## Source map and verification

Key source files:

- `run_state_generic_q_existence.py`: state-local generic triangular transport.
- `run_all_state_joint_generic_q_existence.py`: one generic transport over many
  historical states.
- `run_all_state_vector_q_existence.py`: statewise affine-vector law and control
  variate/statistical reporting.
- `run_all_state_vector_smooth_q_existence.py`: IID all-state smooth-objective
  trainer with fixed validation and confirmation blocks.
- `smooth_q_objective.py`: price-normalized LSE loss, robust block aggregation,
  sign-flip test, and validation/confirmation consistency statistic.
- `run_all_history_joint_direct.py`: all-history affine-200 `P` models.

Focused checks at the latest source commit:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_flex_direct_state.py \
  tests/test_flex_joint_direct.py \
  tests/quantdata/dysanos/flex/test_smooth_q_objective.py
```

Result: `60 passed`.

Ruff import checks and Python compilation checks passed for the changed FLEX
modules. The presentation package contains 50 slides, opens with `python-pptx`,
and passes `ZipFile.testzip()`.

## Repository and operational cautions

- Repository: `/home/hans/python/quant_data`
- Branch: `main`
- Completed implementation/presentation commit: `c1e123b`
- No push was performed.
- Two unrelated modified notebooks remain outside the FLEX commits; preserve
  them as user work.
- `paper/2026 09 02 SANOS ICCF.pre-codex-review.pptx` is an intentionally
  retained local backup and is not tracked.
- Never delete the completed `P` roots or the retained `Q` checkpoints without a
  new explicit decision about which research branches are abandoned.
