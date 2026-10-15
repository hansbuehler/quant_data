# Tangent Lévy model: project plan

## Objective and scope

Implement, validate, and benchmark the Tangent Lévy framework described in
Carmona, Ma, and Nadtochiy, *Simulation of Implied Volatility Surfaces via
Tangent Lévy Models*. The implementation belongs entirely under
`quantdata/dysanos/tangent_levy/` and must produce reproducible Monte Carlo
paths of normalized SPX option surfaces.

The historical scope is fixed:

- ticker: `SPX`;
- requested range: `2022-01-01` through `2025-08-30`, inclusive;
- actual exchange sessions: all business dates returned by the repository
  calendar that are present in the shared grid panel. The grid definition
  excludes half-days and ends on Friday `2025-08-29` because `2025-08-30` is a
  Saturday;
- source surface: the existing cache-only `dysanos/grid` SPX panel whose cache
  definition spans `2020-01-01` through `2025-08-30`;
- baseline grid: 100 normalized-moneyness strikes at maturities
  `2/255`, `5/255`, `10/255`, `20/255`, `40/255`, `0.5`, and `1.0` years;
- smoothing: the grid default `smoothness=0.25`.

This is an implementation and empirical-validation project. Reproducing every
numerical choice in the 2015 preprint is not itself the goal. Any deliberate
departure from the paper must have a faithful reference mode, a documented
reason, and an ablation result.

## Implementation status

The double-exponential core, cache-only grid-market adapter, constrained
maturity reconstruction, constrained PCA/HJM dynamics, seeded risk-neutral
simulator, optional constant physical-measure drift, daily evaluation metrics,
versioned history cache, CLI, and diagnostic plotting command are implemented.
The discrete extension remains behind its stated go/no-go gate.

A full cache-only static calibration and evaluation was completed on
2026-08-25 for the requested 2022--2025 period. Of 918 exchange-calendar dates,
910 were present in the shared grid panel. All 910 available dates calibrated,
and all 6,370 maturity fits (seven per date) reported optimizer success. The
eight sessions without a cache observation were `2022-11-25`, `2023-07-03`,
`2023-11-24`, `2024-07-03`, `2024-11-29`, `2024-12-24`, `2025-01-08`, and
`2025-07-03`. The run used the default six-start, 300-iteration static-fit
configuration and took 322 seconds for calibration plus daily evaluation on
the current Linux host. No raw data were queried or downloaded.

The full audit confirms a sharp distinction between static approximation and
Tangent Lévy consistency. Daily fits are moderately close to the cached smooth
SANOS prices, particularly at short maturities, but the nonnegative
cumulative-density reconstruction gate failed on **all 910 dates**. Production
dynamics and simulation therefore continue to fail fast. The CLI exposes
`--allow-failed-reconstruction` only for explicitly labeled diagnostic runs.
Accordingly, no locked-test, historical-prediction, or discrete-model claim has
been made.

The shared grid artifact does not retain raw quote intervals, exercise-style
metadata, or the upstream option-cap diagnostic. Those quantities must be
reported as unavailable, never reconstructed from the grid. Executable-quote
metrics therefore require a separately cached compatible raw-quote panel and
are outside the primary cache-only run.

### Full-panel static-fit findings

Daily price statistics aggregate the seven maturities and 100 cached strikes
per maturity. Normalized price errors are also shown in basis points of the
forward, where one basis point is `1e-4` in normalized price. The grid-weighted
RMSE is
`sqrt(mean((weight * (TangentLevyCall - grid_mid))**2))`; it is an objective
unit, not a bid/ask-spread unit.

| Daily statistic | Median | 90th percentile | 95th percentile | 99th percentile | Worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized-price RMSE | 12.54 bp | 21.54 bp | 24.65 bp | 29.95 bp | 39.11 bp |
| normalized-price MAE | 7.99 bp | 13.45 bp | 16.86 bp | 21.23 bp | 29.45 bp |
| maximum pointwise price error | 54.50 bp | 112.11 bp | 125.76 bp | 149.72 bp | 179.90 bp |
| implied-volatility RMSE | 1.11 vol points | 2.68 vol points | 3.88 vol points | 4.91 vol points | 5.57 vol points |
| grid-weighted RMSE | 0.00432 | 0.00734 | 0.00852 | 0.01001 | 0.01384 |

Price metrics used all 700 active grid points on each available day. Implied
volatility inversion used between 678 and 700 points; it excluded only active
points whose fitted or target price lay outside numerical no-arbitrage bounds.
The larger IV tail reflects the ill-conditioning of IV inversion in extreme
wings as well as model error.

Fit quality deteriorates materially at the long end:

| Cached maturity | Median grid-weighted RMSE | 95th percentile | Median maximum pointwise price error |
| --- | ---: | ---: | ---: |
| 2 days | 0.00140 | 0.00746 | 6.44 bp |
| 5 days | 0.00183 | 0.00460 | 9.51 bp |
| 10 days | 0.00287 | 0.00705 | 16.72 bp |
| 20 days | 0.00135 | 0.00471 | 7.67 bp |
| 40 days | 0.00247 | 0.00756 | 14.13 bp |
| 0.5 years | 0.00478 | 0.01125 | 28.93 bp |
| 1 year | 0.00814 | 0.01523 | 54.21 bp |

The five worst daily price-RMSE dates were `2025-04-08` (39.11 bp),
`2025-04-07` (35.97 bp), `2022-03-08` (35.39 bp), `2022-03-03` (32.27 bp), and
`2022-01-26` (31.89 bp). The concentration around stressed periods is evidence
of state-dependent model misspecification and should be included in any later
volatility-regime analysis.

For the reconstruction gate, the relative `L1` modification required to turn
the independently fitted average Lévy densities into a nonnegative forward
density had minimum, median, 95th-percentile, and maximum values of 0.313,
0.616, 0.847, and 1.061, respectively. Even the best day exceeded the configured
0.25 tolerance. The corresponding grid-weighted repricing RMSE had median
0.00374, 95th percentile 0.01909, and maximum 0.03576, all below its configured
2.0 tolerance. Thus the zero pass rate is driven by a large cross-maturity
density projection, not optimizer failure or the repricing threshold.

The current empirical conclusion is therefore limited but clear: the
double-exponential family is a useful daily static approximation to the smooth
SPX grid, especially at short maturities, but the independently calibrated
maturities do not form a valid Tangent Lévy code-book under the implemented
nonnegative-forward-density requirement. Static RMSE alone must not be used to
justify dynamics, simulation, or the discrete extension.

### 29 August 2025 stochastic Q audit

A separate v2 experiment initializes the Tangent Lévy codebook on 29 August
2025, estimates 16 target-local factors from 698 valid one-session historical
changes through 31 December 2024, and applies the restricted-universe drift
tests from `quantdata/sdlv/tex/main2.tex`. The target fit is poor: 34.07% of
independent-maturity prices and 5.55% of daily-grid reconstructed prices lie
inside bid/ask, and all 726 retained historical reconstructions fail the gate.

The 100,000-path stochastic Q construction fails all requested diagnostics.
The unconditional screen has maximum absolute z-score 40.63, with 134 of 570
coordinates above five. The out-of-fold squared and entropy strategy tests
have t-statistics 4.400 and 7.078, respectively. The predictable positivity
time change is active on 99.12% of path-substeps and has mean scale 0.0545,
showing that the fitted raw diffusion is nearly frozen in simulation. The
canonical numerical artifacts are under
`/home/hans/quantdata_data/training/dysanos/tangent-levy/v2/spx-2025-08-29-historical-diffusion-v2`;
the committed TeX/PDF report records the full scope and limitations.

## Primary references

- [Requested arXiv preprint, version 1, 24 March 2015](https://arxiv.org/abs/1504.00334)
- [Published SIAM Journal version, 2017](https://doi.org/10.1137/15M1015510)
- [Tangent Lévy market models: consistency theory](https://doi.org/10.1007/s00780-011-0158-8)
- Repository market contract: [`../MARKET.md`](../MARKET.md)

The requested arXiv version is the initial implementation specification. The
published paper changed the empirical comparison and appears to narrow the
model presentation. Before freezing formulas, compare every implemented
equation and algorithm against the published version where it is available;
record the source and equation number in code comments and tests.

## Summary of the paper

### Code-book and consistency idea

At each observation time `t`, the call-price surface is encoded by the current
spot `S_t` and a nonnegative, time-inhomogeneous Lévy jump density
`kappa_t(T, x)`. For fixed `t`, an additive exponential Lévy process with this
density is *tangent* to the market when it reproduces the market prices of all
European calls. This turns an arbitrage-free option surface into a code-book on
a linear function space.

The code-book evolves as

```text
d kappa_t(T, x) = alpha_t(T, x) dt
                  + sum_n beta_t^n(T, x) dB_t^n.
```

The volatility fields `beta` are the free statistical inputs. A convolutional
HJM-like drift restriction determines `alpha`; the spot jump compensator is the
short end `kappa_t(t, x)`; moment/symmetry constraints are imposed on `beta`;
and `kappa` must remain nonnegative. Subject to the paper's regularity and true
martingale assumptions, these conditions make the option-price dynamics
consistent with conditional risk-neutral valuation rather than merely keeping
each simulated surface statically arbitrage-free.

The maturity coordinate is changed to Musiela form,
`hat_kappa_t(tau, x) = kappa_t(t + tau, x)`, before estimating daily dynamics.
The paper estimates the covariance of daily `hat_kappa` changes, applies PCA,
sets `beta_n = sqrt(eigenvalue_n) * eigenmode_n`, derives the HJM drift, and
uses an Euler scheme with a state-dependent scalar boundary factor intended to
keep the density nonnegative.

### Double-exponential Tangent Lévy model

The parametric variant fits, separately at each maturity, a Kou-style
asymmetric double-exponential average Lévy density. Continuity at zero and a
deterministic symmetry-index curve reduce the daily fit to two free parameters
per maturity. A maturity interpolation/differentiation step converts the fitted
average density `eta_t(tau, x)` into the forward density `kappa_t(T, x)`.
Normalized call prices can be evaluated from the characteristic function or
the series formula in the paper.

This variant is the first implementation target because it is low dimensional,
faster, and reported by the paper to be more stable across samples.

### Discrete Tangent Lévy model

The nonparametric variant restricts log jumps to a finite regular grid and fits
nonnegative atomic intensities. The HJM drift becomes a finite convolution.
Call time values follow a finite ODE system that can be evaluated with matrix
exponentials. The paper starts with 301 jump locations at spacing `0.005` but
ties them into 24 intensity groups, adds maturity and jump-direction smoothing,
adds a barrier away from zero, and imposes a deterministic symmetry index.

This variant is more expressive but its static inverse problem is nonconvex,
computationally expensive, and noisy. It is a second-stage implementation,
entered only after the double-exponential pipeline and evaluation harness are
working.

### Paper's empirical test

The preprint fits legacy SPX calls with less than one year to expiry, estimates
code-book dynamics from one calendar year, and tests an eight-day
minimum-variance portfolio problem on subsequent data. It reports realized
portfolio deviation, predicted deviation, and oscillation of portfolio
quantities. Those metrics will be reproduced as a secondary benchmark, but the
primary evaluation in this project is daily surface fit and predictive
statistical fidelity.

## Concerns and required safeguards

| Concern | Consequence | Required response |
| --- | --- | --- |
| The arXiv v1 and final 2017 publication are not identical. | A literal preprint implementation may not represent the authors' final method. | Maintain a formula manifest containing source version, section, and equation for every pricing, drift, and simulation formula. Review the final version before declaring paper fidelity. |
| The authors explicitly note that an exponential Lévy model cannot fit every arbitrage-free multi-strike surface. | Even the smoothed SANOS target may lie outside the Tangent Lévy model class. | Treat calibration error as a model-capacity result, never force an exact fit, and report failures and residual structure. |
| Static inversion is ill-conditioned; the paper identifies numerical instability as its most serious limitation. | Daily parameter noise can dominate PCA and make apparently rich dynamics spurious. | Use transformed parameters, deterministic multi-starts, warm starts that do not change the selected optimum, regularization selected only on training/validation data, condition diagnostics, and parameter-stability metrics. |
| The discrete model is particularly unstable and expensive. | A better in-sample price fit can produce worse dynamics. | Gate discrete-model work on the parametric model and evaluation harness; compare fit improvement against out-of-sample score degradation and runtime. |
| The paper estimates diffusion covariance from historical changes but imposes a risk-neutral HJM drift. | The resulting process is naturally a `Q` model; historical `P` means and risk premia are not identified. | Evaluate covariance and centered innovations under the `Q` implementation. Label historical mean-drift comparisons as descriptive unless a separate physical-measure risk-premium layer is added. |
| Raw PCA modes and interpolation need not preserve the moment/symmetry constraints. | The numerically estimated `beta` can violate the theorem even if the analytical formula is correct. | Project observations and factor modes onto the discrete constraint nullspace, recompute explained variance after projection, and test every mode numerically. |
| The continuous-time positivity argument does not make a one-day explicit Euler step positivity preserving; Gaussian shocks are unbounded. | Silent clipping can break the HJM drift and bias tails. | Use adaptive substeps or step rejection/refinement, record boundary-scaling and rejection rates, and fail a path that cannot be advanced without violating tolerance. Do not silently clip `kappa`. |
| A single near-zero grid cell drives the paper's global boundary scale. | One irrelevant wing cell may freeze the entire surface. | Reproduce the scalar scheme in reference mode, then test a documented alternative such as adaptive stepping on a support-trimmed grid. Report the time spent boundary-limited. |
| The paper assumes stationary daily changes in Musiela coordinates and constant factor loadings. | The 2022--2025 sample contains distinct volatility regimes and volatility clustering. | Test stationarity, rolling covariance, factor stability, autocorrelation, and squared-innovation autocorrelation. Add a rolling/regime extension only after the constant-factor baseline. |
| Both paper variants are pure jump and omit a continuous martingale component. | Very short-dated ATM SPX behavior may be poorly represented, especially in the modern 0DTE market. | Report the cached two-day maturity separately from maturities of five days and longer. Add a diffusion extension only as a versioned follow-up. |
| The SANOS source is a smoothed, arbitrage-free estimate rather than an untouched quote surface. | Fit quality partly measures agreement between two model classes and can hide quote noise or smoothing bias. | Fit the Tangent Lévy model to the cached SANOS grid as requested. Quote-interval and smoothing-sensitivity studies require separately available, pre-registered cache artifacts; they must not trigger raw-data generation during the primary run. |
| The paper's empirical evidence uses two old samples, 500 scenarios, overlapping horizons, unconstrained covariance inversion, and a small baseline set. | Its reported superiority should not be treated as an expected project result. | Use rolling-origin tests, shrinkage/regularization for portfolio covariance, confidence intervals that account for overlapping windows, and persistence/direct-surface baselines. |
| Dependence between spot returns and code-book innovations is not determined by daily option marginals alone. | Surface-only simulations can match option changes while missing the SPX leverage effect. | Estimate and report historical spot/factor dependence separately. Do not add a coupling to the risk-neutral simulator until its compatibility with the consistency conditions is established. |

The `P`/`Q` distinction is an evaluation boundary, not merely a report label.
The HJM-consistent simulator is evaluated for risk-neutral consistency and for
instantaneous covariance or centered short-horizon innovation fidelity. Proper
scores against subsequently realized markets require a separately estimated
physical-measure drift or market-price-of-risk layer. Treating `P = Q` is an
explicit benchmark assumption and cannot support a claim of historical
predictive accuracy by itself.

## Market-data contract

### Generation route

Create `quantdata/dysanos/tangent_levy/markets.py` as the only historical-data
entry point for this project. The primary run must reconstruct the same
`GridMarketDefinition` used by `dysanos/grid` and load it strictly cache-only:

```python
from datetime import date

from quantdata.biz_calendar import DateRange
from quantdata.dysanos.grid.torch_market import GridConfig, GridMarketDefinition
from quantdata.pool import PoolConfig

definition = GridMarketDefinition(
    ticker="SPX",
    date_range=DateRange(
        start=date(2020, 1, 1),
        end=date(2025, 8, 30),
        exclude_half_days=True,
    ),
    grid_config=GridConfig(
        grid_num_strikes=100,
        smoothness=0.25,
    ),
)
panel = definition.load(
    pool_config=PoolConfig.off(),
    override_cache_mode="cacheonly",
)
```

The `DateRange`, `RawDayFilter`, `ExpiryFilter`, `OptionsFilter`, `MkArbFree`,
and `GridConfig` inputs are immutable versioned fields. The apparently wider
2020 cache definition is intentional: it identifies the artifact already on
disk; the adapter then selects the requested 2022--2025 observations in memory.
Operational worker count and verbosity remain outside the logical config.

`override_cache_mode="cacheonly"` is mandatory. If that exact artifact is
absent, the run must fail with a cache miss. It must not invoke
`smooth_interpolated_market()`, WRDS, or any other raw-data fallback. The grid
cache contains 1,412 dates overall and 910 dates in the requested 2022--2025
window, from `2022-01-03` through `2025-08-29`.

Use `mids` as the daily smooth SANOS targets, `xstrikes[..., 1:-1]` as their
matching strikes, and the stored positive `weights` as objective multipliers.
The cache's seven expiries and 100 strikes are already fixed across days, so no
second surface interpolation or coverage-selected market grid is needed.
Empty or excluded sessions are recorded, not silently replaced.

### Mapping to paper coordinates

| Project value | Paper value | Mapping |
| --- | --- | --- |
| `tau` | `T - t` | Use the cached grid fractions, whose maturity basis is 255 days; one simulated exchange-session step remains `1 / 252`. |
| `k` | forward-relative strike | `k = K / F`. |
| `x` | log moneyness | `x = log(k)`. |
| `c` | zero-carry normalized call | `c = C / (DF * F)`, already represented by the market contract. |
| normalized intrinsic | call intrinsic | `max(1 - k, 0)`. |
| normalized time value | fitted target | `v = c - max(1 - k, 0)`. |
| `panel["mids"]` | arbitrage-free target call | Primary daily fitting target. |
| `panel["weights"]` | grid loss multiplier | `1 / (mid_vega / sqrt(tau))` where the upstream vega floor is active, otherwise zero. |

Fit where the cached multiplier is finite and positive, and report zero-weight
points as inactive. The artifact does not contain bid/ask intervals, raw option
identities, AM/PM flags, or exercise style. The adapter represents these as
unavailable and must not fabricate them. Any later executable-interval study
must load a separately versioned compatible artifact and join it explicitly.

### Common grid and coverage pilot

The market panel is already fixed at seven maturities and 100 NMS strikes. The
dynamics maturity grid is those same seven nodes prefixed by `tau=0`; this
avoids a second maturity interpolation. The jump-size grid used to represent
Lévy densities is a separate object and must not be confused with the cached
market-strike grid. Store both grids in the experiment manifest.

Maintain two distinct evaluation alignments. Surface-dynamics diagnostics use
fixed Musiela `(tau, x)` coordinates. The paper's portfolio reproduction holds
actual contract expiry and strike identities fixed, so remaining maturity
falls over the forecast horizon. Results from these alignments are not
interchangeable.

For each requested date, report cache availability, retained maturities,
active/zero-weight grid points, and any reason for exclusion. Early-close and
raw-option-cap fields are unavailable in this artifact; report them as such.

### Time split and leakage rules

- development training: `2022-01-01` through `2023-12-31`;
- validation and hyperparameter selection: `2024-01-01` through `2024-12-31`;
- locked test: `2025-01-01` through `2025-08-30`;
- final reported test run: refit dynamic parameters through `2024-12-31`, then
  use the expanding or fixed-window rolling-origin scheme selected on the
  validation set, without looking ahead.

`run_experiment.py` therefore defaults `--dynamics-end` to `2024-12-31` even
though it loads and reports the complete static panel through `2025-08-30`.

Daily static calibration may use that day's surface because it defines the
current code-book. Symmetry curves, regularization strengths, grids, PCA
dimension, covariance shrinkage, and simulator choices may use only information
available before the forecast origin. Preserve failed days in the chronology;
never compress time and call nonconsecutive observations one-day changes.

## Proposed implementation

### Module layout

```text
quantdata/dysanos/tangent_levy/
    __init__.py
    config.py             immutable, versioned model/data/fit/simulation configs
    markets.py            SPX market generation, reconstruction, fixed-grid data
    coordinates.py        tau, k, x, price, time-value transformations
    double_exponential.py density, characteristic function, pricing kernel
    pricing.py            generic numerical-density characteristic/Fourier pricer
    discrete.py           atomic density and matrix-ODE pricing kernel
    static_fit.py         daily calibration and maturity reconstruction
    constraints.py        moment/symmetry projections and HJM drift
    dynamics.py           Musiela changes, covariance estimation, constrained PCA
    physical_measure.py   optional P drift/market-price-of-risk estimation
    simulator.py          seeded path generation and positivity control
    metrics.py            fit, distributional, arbitrage, and timing metrics
    evaluation.py         daily SANOS and executable-interval scoring
    experiment.py         train/validation/test orchestration and manifests
    plotting.py           reproducible diagnostic figures
    plot_results.py       cached-artifact plotting command
    run_experiment.py     command-line entry point
    TANGENT_LEVY.md       this plan and final contract
tests/
    test_tangent_levy_*.py
```

All public configurations should be frozen keyword-only dataclasses with
`@version`, `label`, `group`, and `user_version` where model artifacts are
stored. Expensive market generation, static fits, and derived code-books should
use versioned repository caches keyed by every result-affecting configuration.
Do not cache Torch models with the general-purpose cache.

### Phase 0: equation manifest and numerical fixtures

1. Transcribe the parameter domains, average-density relation, symmetry/moment
   constraints, continuous and discrete HJM drifts, and simulation update.
2. Record the preprint and final-publication equation numbers beside each
   implementation.
3. Build independent numerical fixtures:
   - characteristic exponent has the martingale condition at `u = -i`;
   - call prices agree between the paper's double-exponential series formula,
     the generic numerical-`eta` Fourier formula from equation (2.36), and a
     high-sample Monte Carlo check;
   - the discrete matrix-ODE price agrees with direct compound-Poisson Monte
     Carlo on a tiny grid;
   - finite-difference HJM drift agrees with the discrete convolution limit.

Exit criterion: pricing and drift kernels pass in float64 before calibration
code is written.

### Phase 1: historical market panel

1. Reconstruct the exact `dysanos/grid` definition in `markets.py` and load it
   with `override_cache_mode="cacheonly"`.
2. Adapt the cached fixed-grid `mids`, interior `xstrikes`, and `weights`
   directly. Do not refit an `ArbFreePricer` and do not invent bid/ask data.
3. Store spot returns and metadata without using them as calibration targets.
4. Produce a data audit for every requested session and explicitly identify
   diagnostics absent from the artifact.

The implemented market grid uses the exact cached maturities `2/255`, `5/255`,
`10/255`, `20/255`, `40/255`, `0.5`, and `1.0`; the dynamics grid prefixes
zero. Each maturity has 100 normalized-moneyness strikes.

Exit criterion: all retained target calls satisfy normalized bounds and static
arbitrage tolerances; date gaps and exclusions are explicit.

### Phase 2: double-exponential static fit

1. Implement the pure-jump double-exponential characteristic exponent and a
   vectorized Fourier pricer. Keep the series formula only as a reference test.
2. Reproduce the paper's parameter reduction exactly. On the initial training
   day, fit the three-variable reference calibration and infer `p` from the
   continuity condition. After freezing the deterministic symmetry curve
   `Xi(tau)`, fit only `lambda` and `lambda_1`; derive `lambda_2` and `p` from
   equations (2.18) and (2.16). Enforce `lambda > 0`, the full domain in
   equation (2.19), including its upper bound on `lambda_1` when `Xi > 0`, and
   verify the derived `lambda_2 > 0` and `0 < p < 1`.
3. Implement calibration of normalized call time values with the cached grid
   multipliers. The objective is the mean square of
   `weight * (TL_call - grid_mid)`. A separately cached raw-quote ablation may
   use inverse half-spreads, but it is not part of the primary run.
4. Fit the deterministic symmetry-index curve using training data only. Report
   the literal first-training-day estimate and a robust training-window estimate
   as an ablation; choose on validation data.
5. Reconstruct forward `kappa` from average `eta`. The reference mode uses the
   paper's maturity interpolation. The production candidate instead fits
   cumulative intensity `A(tau, x) = tau * eta(tau, x)` with a constrained
   spline satisfying `A(0, x) = 0` and nonnegative maturity derivative. Require
   the reconstructed `kappa` to preserve every calibrated maturity integral,
   repricing, and the deterministic exponential-moment relation within stated
   numerical tolerances.

The reconstruction gate uses both the relative `L1` change in cumulative
intensity after the nonnegative projection and the RMS change in fitted prices,
measured in the same cached grid-weight units. A material-floor pointwise
maximum and the maximum normalized-price change are also reported for
localization, but negligible tail cells do not define pass/fail.
6. Use deterministic multi-start calibration and previous-day parameters only
   as starts. Save all objective values so warm starts cannot conceal a worse
   local optimum.

Exit criterion: daily fits are reproducible, finite, and produce nonnegative
`kappa`; residuals and parameter instability are fully reported.

Status: **not met**. Although every available daily and maturity calibration
completed, zero of 910 dates passed the nonnegative forward-density
reconstruction gate. Phases 3--6 may be exercised only as explicitly labeled
diagnostics until the model or reconstruction specification is revised and
this gate is rerun.

### Phase 3: constrained dynamics

1. Evaluate every daily `hat_kappa` on the frozen Musiela grid.
2. Construct changes only across consecutive exchange sessions and scale them
   consistently by `dt = 1 / 252`.
3. Estimate one-session centered covariance `Sigma_delta` with a documented
   shrinkage estimator. If `(lambda_delta_n, eigenmode_n)` is one of its
   eigenpairs, set
   `beta_n = sqrt(lambda_delta_n / dt) * eigenmode_n`; equivalently, diagonalize
   the annualized covariance `Sigma_delta / dt`. Verify that the factors'
   one-step covariance reconstructs `Sigma_delta`. Project changes and PCA
   modes into the moment/symmetry constraint nullspace.

The production covariance estimate additionally freezes jump cells whose
minimum training density is at or below `1e-4`, or below five empirical
one-session standard deviations. This support-trimmed PCA avoids estimating
Gaussian volatility on the positivity boundary; the mask is stored in
`DynamicsFit`, and both factors and the finite-support HJM drift are projected
on its active coordinates. Both thresholds are versioned training choices and
must be included in sensitivity reporting.

At least 60 consecutive session changes are required before fitting dynamics;
shorter samples are useful only for market/static/cache smoke tests.
4. Select factor count using training explained variance and validation
   one-step covariance reconstruction or centered-innovation diagnostics;
   report the full spectrum rather than hard-code the paper's three factors.
   Multi-day proper forecast scores may select this choice only after a
   physical-measure layer exists.
5. Compute `alpha` from the HJM drift restriction and the Musiela maturity
   derivative. Keep NumPy and any Torch implementation formula-for-formula
   aligned.
6. Test constant full-sample factors first. Rolling or regime-dependent factors
   are a separate versioned extension.

Exit criterion: every mode satisfies the discrete constraints, drift residuals
are below numerical tolerance, and no validation/test observations influenced
the selected basis.

### Phase 4: risk-neutral simulator

1. Implement deterministic seeded Brownian draws. Surface/IV simulation does
   not require a spot path; add compound-Poisson spot jumps from the short-end
   compensator only for joint spot/surface diagnostics and the paper's
   calls-plus-underlying portfolio experiment.
2. Advance `hat_kappa` with reference Euler and adaptive-substep variants.
3. Reject or refine steps that cross the positivity boundary; report every
   intervention. Do not silently clamp states.

The implementation applies the paper's scalar boundary factor with
`epsilon=1e-6`. The cumulative-density reconstruction imposes the same lower
barrier as a constraint before differentiation; this prevents a projected
initial cell at exactly zero from freezing every factor under equation (2.34).
Any path that subsequently reaches `gamma=0` is reported as frozen.
4. Convert simulated `kappa` to average density, then price with the generic
   numerical-density characteristic-exponent/Fourier formula in equation
   (2.36). A simulated state is generally no longer double exponential, so the
   static double-exponential pricing shortcut is invalid off the calibration
   manifold. Convert the resulting normalized calls to IV on requested
   horizons.
5. Verify at each step:
   - finite nonnegative density;
   - exponential moment and martingale identities;
   - normalized call bounds, strike monotonicity/convexity, and calendar order;
   - in joint spot/surface mode, cross-path discounted fixed-contract option
     martingale residuals within Monte Carlo error.

Initial horizons are 1, 5, 8, and 20 exchange sessions. The eight-day horizon
matches the paper; the others reveal horizon sensitivity.

### Phase 5: optional physical-measure forecast layer

1. Keep the HJM `Q` drift unchanged in the risk-neutral simulator.
2. If historical forecast claims are in scope, estimate a low-dimensional,
   regularized market price of factor risk using training data, giving a
   separately versioned physical drift of the form
   `alpha_P = alpha_Q + sum_n beta_n * theta_n`, subject to the chosen Girsanov
   sign convention. Select the information window and shrinkage on validation
   data only.
3. Recheck positivity and static-arbitrage numerics under `P`; a physical drift
   does not remove the requirement that every path encode a valid current
   surface.
4. Use proper forecast scores against realized markets only for this `P` model.
   Report `P = Q` as a labeled benchmark, not as a physical forecast model.

Go/no-go criterion: omit this layer if the available sample cannot identify a
stable risk premium. Without it, restrict claims to `Q` consistency, static
fit, instantaneous covariance, and descriptive centered-innovation fidelity.

### Phase 6: discrete-model extension

1. Implement the paper's regular jump grid and grouped intensity basis.
2. Implement matrix-exponential time-value pricing and the discrete symmetry
   and HJM constraints. Enforce the paper's jump-support restriction and
   project factor modes onto both discrete nullspaces: zero total mass change
   and zero exponential-moment change.
3. Fit nonnegative intensities with separately configurable maturity smoothing,
   jump-direction smoothing, and boundary barrier.
4. Repeat the same dynamics, simulation, and evaluation pipeline without
   changing metric definitions.

Go/no-go criterion: proceed only if the double-exponential model, market panel,
and evaluation report are stable. Retain the discrete model only if its
out-of-sample distributional improvement justifies its runtime and parameter
noise; a lower daily price RMSE alone is insufficient.

## Metrics

All aggregates must be reported by date, maturity bucket, log-moneyness region,
calendar year, and volatility regime, with counts and block-bootstrap confidence
intervals. Never report only a grand average.

### Data and operational metrics

- requested, available, empty, excluded, calibrated, and failed date counts;
- active and zero-weight cached grid points by date/maturity;
- static-fit wall time, iterations, restarts, selected start, and failure cause;
- dynamic-fit and simulation time, paths per second, peak memory, and cache hit;
- boundary-scale distribution, rejected/refined step count, and failed-path rate.

### Outright daily fit

Let `c_j^TL` be the fitted Tangent Lévy call, `m_j` the cached smooth SANOS
`mid`, and `w_j` the cached grid multiplier. Upstream, the grid defines

```text
w_j = 1 / (mid_vega_j / sqrt(tau_j))
```

where the upstream vega threshold is met, and zero otherwise. Use finite,
positive weights as the primary mask and report inactive points separately.

1. **SANOS price error**

   ```text
   delta_c_j = c_j^TL - m_j
   grid_weighted_RMSE = sqrt(mean_j((w_j * delta_c_j)^2))
   ```

   Also report unweighted RMSE, MAE, median absolute error, 95th/99th absolute
   error, and maximum absolute error.

2. **Executable-interval error**, optional secondary analysis

   The shared grid cache has no bid/ask intervals, so interval scores are
   `null` in the primary report. If a separately cached compatible quote panel
   is explicitly joined, use the repository half-spread convention and report
   both SANOS and Tangent Lévy interval errors. Never infer spreads from `w_j`.

3. **Implied-volatility error**: RMSE, MAE, median, 95th percentile, and maximum
   absolute error in volatility points, after checking price bounds before
   inversion.

4. **Time-value error**: absolute and relative error in
   `v = c - max(1-k, 0)`, with a configured time-value floor. Relative errors
   below the floor are excluded and counted explicitly.

5. **Residual structure**: mean signed error and autocorrelation by maturity,
   moneyness, call delta, and date. Plot residual surfaces so cancellation in
   RMSE cannot hide systematic skew or term-structure misspecification.

6. **Fit robustness**: parameter distributions, day-to-day parameter changes,
   cross-start objective dispersion, Hessian/Jacobian condition estimates,
   symmetry residual, minimum `kappa`, and maturity reconstruction error.

7. **No-arbitrage numerics**: maximum violations of call bounds, strike
   monotonicity, strike convexity, calendar monotonicity, nonnegative `kappa`,
   the exponential moment constraint, and repricing identities.

### Statistical properties versus the market panel

Define two vectors on the frozen grid:

- `Y_t^kappa`: the calibrated Musiela code-book, which is a latent,
  inversion-dependent model quantity rather than a direct observation;
- `Y_t^surface`: normalized call or total-implied-variance values from the
  smooth SANOS surface.

Primary statistical fidelity is measured on `Y_t^surface`. Code-book results
are conditional diagnostics and must include sensitivity to calibration starts,
regularization, and market smoothness. Surface comparisons use fixed Musiela
coordinates and prefer normalized calls or total implied variance over raw IV
where maturity changes make raw IV unstable.

Keep two estimands separate:

- For the `Q` simulator, compare the instantaneous/one-session centered
  covariance and factor structure with historical centered innovations, and
  test risk-neutral martingale consistency internally. Multi-day comparisons
  with realized markets are descriptive `P = Q` diagnostics only.
- For an implemented `P` layer, transform every ensemble path at each rolling
  origin and horizon to the same vector, compare the conditional distribution
  with the one subsequently realized market vector, and aggregate paired
  forecast metrics across origins. Do not treat the Monte Carlo paths from a
  single origin as independent historical observations.

1. **First and second moments**: forecast mean error of changes, marginal
   variance ratios of demeaned innovations, relative Frobenius covariance
   error, correlation-matrix error, and covariance condition number. For the
   `Q` comparison, remove historical conditional-mean estimates using only the
   training information set.

2. **Factor structure**: eigenvalue-spectrum error, cumulative explained
   variance, principal angles between empirical and simulated leading
   eigenspaces, and stability of loadings across rolling windows.

3. **Marginal distributions**: errors in standard deviation, skewness,
   kurtosis, 1/5/50/95/99% quantiles, tail exceedance rates, and one-dimensional
   Wasserstein distance for ATM level, downside skew, term slope, and leading
   factor scores. These are physical-distribution metrics only for a `P` layer;
   for `Q` paths label them descriptive.

4. **Dependence and dynamics**: cross-maturity and cross-moneyness correlation,
   autocorrelation of factor changes and squared changes at lags 1, 5, and 20,
   and the historical correlation of spot returns with ATM, skew, and factor
   changes. Treat spot dependence as diagnostic until a consistent coupling is
   implemented.

5. **Proper multivariate forecast scores for the `P` layer**:
   - energy score for each origin/horizon, using the unbiased finite-ensemble
     estimator with 5,000 paths;
   - variogram score of order `0.5` on immediately adjacent maturity and
     moneyness pairs, with normalized inverse-grid-distance weights;
   - marginal 50%, 80%, and 95% interval coverage and average interval width;
   - randomized-rank/PIT histograms for scalar surface features, using a fixed
     seed to break ties.

   Scale vector components by a training-only robust scale, apply frozen
   quadrature surface weights that sum to one, and use the same ensemble size
   and random-number policy for every model. All proper scores are lower is
   better. Confirm Monte Carlo convergence on a prespecified subset with 10,000
   paths. Report the paired score difference `model - baseline` and its
   block-bootstrap confidence interval, not two unpaired score averages.

   For weighted norm `||.||_W`, observation `y`, and ensemble members `X_m`,
   implement the energy score as

   ```text
   ES = mean_m ||X_m - y||_W
        - sum_(m != l) ||X_m - X_l||_W / (2 M (M - 1)).
   ```

   Implement the variogram score as

   ```text
   VS = sum_(i,j) w_ij (
          abs(y_i-y_j)^0.5
          - mean_m abs(X_mi-X_mj)^0.5
        )^2.
   ```

6. **Tail and regime behavior**: scores by calendar year, initial SANOS ATM-vol
   quartile, positive/negative SPX return, and large observed surface move.
   Report the number of observations in every cell.

7. **Simulation validity**: fraction of paths with any static-arbitrage,
   moment, positivity, or numerical violation; maximum violation magnitude;
   option martingale residual divided by its Monte Carlo standard error.

8. **Baselines**:
   - unchanged-surface persistence;
   - resampled historical changes;
   - unconstrained Gaussian PCA of direct SANOS surface changes;
   - double-exponential Tangent Lévy;
   - discrete Tangent Lévy when available.

   A SABR benchmark and the paper's direct-IV simulation are follow-up
   comparisons, not prerequisites for validating the Tangent Lévy code.

9. **Paper portfolio metric, secondary**: reproduce the eight-day portfolios
   with three, four, and five near-ATM strikes. Report predicted deviation `P`,
   realized deviation `Q`, `abs(P-Q)`, quantity-oscillation `K`, gross and net
   exposure, covariance shrinkage, and sensitivity to position constraints.
   Hold contract strike and expiry identities fixed over the horizon. Use
   block-bootstrap intervals because forecast windows overlap, with an initial
   20-session block length and prespecified 40- and 60-session sensitivity
   checks. Name these statistics `P_portfolio` and `Q_portfolio` in code so they
   cannot be confused with the probability measures.

## Acceptance gates

Thresholds that depend on empirical scale must be frozen after the training
pilot and before reading locked-test results. The non-negotiable gates are:

1. Formula fixtures and float64 numerical invariants pass deterministically.
2. Every result is tied to exact data/model/software versions and a seed.
3. No train/validation/test leakage is present in grids, symmetry curves,
   regularization, factor selection, or covariance estimation.
4. At least 99% of available baseline-period market days produce a finite
   static calibration, or every remaining failure is explained and included in
   the reported failure rate.
5. Every daily code-book admitted to dynamics passes the nonnegative
   forward-density reconstruction gate without exceeding either its density or
   repricing tolerance. The current baseline fails this gate on all 910 dates.
6. No accepted simulated path silently clips a density or hides a numerical or
   no-arbitrage violation.
7. Locked-test reporting includes all metrics and all origins, not only days on
   which the model performs well.
8. Only an implemented physical-measure layer may be described as historically
   predictive. It must beat unchanged-surface persistence on a prespecified
   5- or 8-day proper multivariate score: the upper endpoint of the 95% block-
   bootstrap confidence interval for paired `model - persistence` scores must
   be below zero. A `Q`-only implementation makes no predictive claim and is
   not failed merely because this conditional claim gate is inapplicable.
9. The discrete extension must improve a prespecified validation score after
   accounting for runtime and failed fits; in-sample price fit alone is not an
   acceptance criterion.

## Testing and verification plan

- Unit-test coordinate and normalization identities against `MARKET.md`.
- Test pricing kernels against independent formulas and Monte Carlo references.
- Test the generic numerical-density pricer against the double-exponential
  shortcut before using it on simulated nonparametric states.
- Test analytical/automatic gradients against finite differences where used.
- Test transformed parameter domains and all failure messages.
- Test average-to-forward density reconstruction on known synthetic surfaces.
- Test `A(0, x) = 0`, maturity-integral preservation, and repricing after every
  density reconstruction method.
- Test continuous and discrete constraint projections and HJM drift residuals.
- Test annualization by reconstructing the observed one-session covariance from
  the fitted `beta` factors.
- Test a synthetic recovery problem with known double-exponential parameters.
- Test irregular market maturities, missing days, dropped pricer expiries,
  short expiries, sparse wings, and non-finite optimizer output.
- Test deterministic replay across cold and warm starts and after checkpoint
  restoration.
- Test forward values and gradients if Torch is introduced.
- Run focused tests first, then all `test_tangent_levy_*.py`, syntax compilation,
  Ruff import checks, and relevant existing Dysanos market tests.

## Deliverables

1. Versioned historical market-panel manifest and coverage report.
2. Double-exponential pricer, daily calibration, and fitted code-book cache.
3. Constrained dynamic-factor model and seeded risk-neutral simulator.
4. Optional physical-measure layer, required only for historical forecast
   claims.
5. Static-fit, statistical-fidelity, no-arbitrage, and operational reports.
6. Paper portfolio-metric reproduction as a secondary experiment.
7. Discrete-model implementation and comparison, conditional on the go/no-go
   gate.
8. A standalone plotting/report command that loads artifacts without training.
9. Updated documentation describing exact formulas, deliberate departures,
   known limitations, and whether each acceptance gate passed.

## Recommended execution order

Proceed in this order: equation fixtures, market panel, double-exponential
pricing, daily static fit, code-book reconstruction, constrained dynamics,
generic simulated-state pricing, risk-neutral simulation, optional physical-
measure estimation, locked evaluation, and only then the discrete extension.
This order tests the paper's central consistency claim with the least ill-
conditioned model before investing in the substantially harder nonparametric
inverse problem.
