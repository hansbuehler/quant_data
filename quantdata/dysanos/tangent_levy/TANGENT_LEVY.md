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
  calendar, including early-close sessions, and ending on Friday `2025-08-29`
  because `2025-08-30` is a Saturday;
- source surface: `smooth_interpolated_market()` from
  `quantdata/dysanos/market.py`;
- baseline horizon: expiries from five business days through one year. Shorter
  expiries, including 0DTE, form a separate stress test rather than part of the
  first fit.

This is an implementation and empirical-validation project. Reproducing every
numerical choice in the 2015 preprint is not itself the goal. Any deliberate
departure from the paper must have a faithful reference mode, a documented
reason, and an ablation result.

## Implementation status

The double-exponential core, canonical market adapter, constrained maturity
reconstruction, constrained PCA/HJM dynamics, seeded risk-neutral simulator,
optional constant physical-measure drift, daily evaluation metrics, versioned
history cache, CLI, and diagnostic plotting command are implemented. The
discrete extension remains behind its stated go/no-go gate.

A real-data smoke test on SPX sessions 2024-03-05 through 2024-03-07 produced
all three requested market artifacts and 128 successful maturity fits. The
median maturity objective was 0.55 half-spreads and the largest normalized
price error was 0.00490. On 2024-03-05 alone, the aggregate normalized-price
RMSE was 0.000117, IV RMSE was 0.000443, and all 43 optimizers converged.

The same smoke test did **not** pass the nonnegative cumulative-density
reconstruction gate. On the production 101-point jump grid, relative `L1`
projections were 0.469, 0.475, and 0.517 against the configured 0.25 threshold;
repricing changes were 6.95, 7.48, and 8.11 half-spreads RMS, with maximum
normalized-price changes between 0.00254 and 0.00446. This is a material
model-consistency warning, not a tail-division artifact. Production dynamics
and simulation therefore fail fast when any required date misses the gate.
The CLI exposes `--allow-failed-reconstruction` only for explicitly labeled
diagnostic runs. Accordingly, no locked-test, historical-prediction, or
discrete-model claim has been made. The full 2022--2025 training audit must
establish whether the failure is pervasive before the later phases proceed.

The 1,000-record market cap bound on all five sampled dates spanning
2022--2025. The canonical SANOS solve completed in roughly five seconds per
sampled day; an attempted effectively uncapped solve was not operationally
viable. Higher-cap sensitivity therefore remains conditional on its solver
feasibility pilot.

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
| Both paper variants are pure jump and omit a continuous martingale component. | Very short-dated ATM SPX behavior may be poorly represented, especially in the modern 0DTE market. | Exclude expiries below five business days from the first acceptance gate; report them separately. Add a diffusion extension only as a versioned follow-up. |
| The SANOS source is a smoothed, arbitrage-free estimate rather than an untouched quote surface. | Fit quality partly measures agreement between two model classes and can hide quote noise or smoothing bias. | Fit the Tangent Lévy model to SANOS prices as requested, but score it both against the SANOS target and against the original normalized bid/ask intervals retained in the market payload. Run a smoothing-parameter sensitivity analysis. |
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
entry point for this project. It must enumerate exchange sessions and call the
canonical producer:

```python
from datetime import date

from quantdata.biz_calendar import DateRange
from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.widb.options_filter import OptionsFilter

date_range = DateRange(
    start=date(2022, 1, 1),
    end=date(2025, 8, 30),
    exclude_half_days=False,
)
dates = date_range.get_business_days()
payload = smooth_interpolated_market(
    "SPX",
    market_date,
    smoothness=0.25,
    options_filter=OptionsFilter(
        normalize=True,
        max_num_options=1_000,
        max_business_days_to_expiry=252,
    ),
)
```

The exact `DateRange`, `RawDayFilter`, `ExpiryFilter`, `OptionsFilter`, input
`MkArbFree`, and smoothing configurations must be immutable versioned fields in
a `TangentLevyMarketConfig`; the example above is the initial default, not an
untracked convention. Operational worker count and verbosity stay outside the
logical configuration. The 1,000-option cap is the repository's operational
baseline for the global SANOS solve; the data audit must report whenever it
binds, and higher-cap sensitivity runs must first pass a solver-feasibility
pilot.

Use `smoothness=0.25` for the primary panel. Pre-register `0.05` and `1.0` as
sensitivity panels; all values obey `market.py`'s strict `smoothness > 0.01`
contract. Select no smoothing value after observing locked-test performance.

`markets.py` must not refit or independently smooth raw quotes. It reconstructs
the serialized `ArbFreePricer` and evaluates the existing smooth surface at the
project grid. Empty days are recorded, not silently replaced.

### Mapping to paper coordinates

| Project value | Paper value | Mapping |
| --- | --- | --- |
| `tau` | `T - t` | Use the year fraction returned by the market file; one simulation day is `1 / 252`. |
| `k` | forward-relative strike | `k = K / F`. |
| `x` | log moneyness | `x = log(k)`. |
| `c` | zero-carry normalized call | `c = C / (DF * F)`, already represented by the market contract. |
| normalized intrinsic | call intrinsic | `max(1 - k, 0)`. |
| normalized time value | fitted target | `v = c - max(1 - k, 0)`. |
| `af_pricer.price(tau, k)` | arbitrage-free target call | Primary daily fitting target. |
| `market.bids`, `market.asks` | observed call interval | Secondary executable-interval score at original strikes. |

At original strikes, use `market.af_call_mids` (equivalently
`fit_result.fits`) as the SANOS target and `market.model_masks` to identify
strikes retained by the final pricer. Report excluded and inactive observations
instead of dropping them from counts. `market.bids` and `market.asks` are
synthetic pure-call intervals; keep them distinct from executable real
`call_bids`/`call_asks` and `put_bids`/`put_asks`, whose inclusion is defined by
`call_active` and `put_active`.

Fit only European exercise observations. Preserve AM/PM metadata for
diagnostics. Use all retained SPX expiries in the data extract, then select the
configured Tangent Lévy maturity domain without relying on rounded expiry
identities.

### Common grid and coverage pilot

The dynamics require a fixed `(tau, x)` grid across dates. Do not choose it from
the locked test sample. Run a training-only coverage pilot and freeze:

- maturity nodes between 5 business days and 1 year, with extra resolution
  below 60 days;
- a log-forward-moneyness interval supported by at least 95% of training days;
- a dense scoring grid and a smaller dynamics grid;
- extrapolation rules, which should normally reject a point rather than hide a
  large extrapolation behind flat values.

The initial implementation candidate is 10 maturity nodes and 21 scoring
moneyness nodes. This is not the discrete model's jump grid, which has its own
paper-reference specification. The chosen grids and coverage statistics are
stored in the experiment manifest.

Maintain two distinct evaluation alignments. Surface-dynamics diagnostics use
fixed Musiela `(tau, x)` coordinates. The paper's portfolio reproduction holds
actual contract expiry and strike identities fixed, so remaining maturity
falls over the forecast horizon. Results from these alignments are not
interchangeable.

For each requested date, report whether it is a regular or early-close session,
whether the market payload exists, number of input and retained expiries,
number of quotes, final-pricer fit statistics, common-grid coverage, and any
reason for exclusion.

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

1. Implement the exact date/config route in `markets.py`.
2. Reconstruct each `ArbFreePricer` and create both:
   - a fixed-grid panel of smooth call prices and implied volatilities;
   - ragged original-strike SANOS prices plus normalized bid/ask intervals.
3. Store spot returns and metadata without using them as calibration targets.
4. Produce a data audit for every requested session and freeze the common grid
   from the training sample.

The implemented reference grid uses maturities of 0, 5, 10, 20, 40, 60, 90,
126, and 189 business days. The market fit still retains expiries through 252
business days. A five-date coverage smoke test spanning 2022--2025 found the
longest retained expiry between 239 and 249 business days, so 252 days is not a
valid fixed daily endpoint; the nine-month endpoint avoids extrapolation. The
full training audit must confirm this choice before the locked evaluation.

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
3. Implement paper-reference calibration of normalized call time values with
   inverse-half-spread-squared weights at original strikes.
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
measured in original half-spreads. A material-floor pointwise maximum and the
maximum normalized-price change are also reported for localization, but
negligible tail cells do not define pass/fail.
6. Use deterministic multi-start calibration and previous-day parameters only
   as starts. Save all objective values so warm starts cannot conceal a worse
   local optimum.

Exit criterion: daily fits are reproducible, finite, and produce nonnegative
`kappa`; residuals and parameter instability are fully reported.

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
- common-grid coverage and number of original quotes by date/maturity;
- static-fit wall time, iterations, restarts, selected start, and failure cause;
- dynamic-fit and simulation time, paths per second, peak memory, and cache hit;
- boundary-scale distribution, rejected/refined step count, and failed-path rate.

### Outright daily fit

Let `c_j^TL` be the fitted Tangent Lévy call, `c_j^AF` the smooth SANOS call,
`b_j` and `a_j` the original normalized pure-call bid and ask, and

```text
h_j = max(0.5 * (a_j - b_j), h_floor).
```

Use the final-pricer `model_masks` for the primary original-strike aggregate and
report all inactive/excluded points separately. Compute the spread-weighted
metrics only at original market strikes. On the fixed dense grid, where no
observed half-spread exists, use unweighted errors and a frozen quadrature
weighting over `(tau, x)`; do not invent or interpolate spreads.

1. **SANOS price error**

   ```text
   delta_c_j = c_j^TL - c_j^AF
   weighted_RMSE = sqrt(sum_j delta_c_j^2 / h_j^2 / sum_j 1 / h_j^2)
   ```

   Also report unweighted RMSE, MAE, median absolute error, 95th/99th absolute
   error, and maximum absolute error.

2. **Executable-interval error**, using the repository convention

   ```text
   e_j = (max(0, c_j^TL - a_j) + max(0, b_j - c_j^TL)) / h_j.
   ```

   Report fraction inside spread, fraction outside spread, median/mean positive
   `e_j`, total median/mean `e_j`, and its 95th/99th percentiles. Report the same
   figures for SANOS so the smoothing error and Tangent Lévy representation
   error are not conflated. This metric uses synthetic pure-call intervals.
   Separately price real calls and puts, by normalized parity where necessary,
   and score their executable intervals under `call_active` and `put_active`.

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
5. No accepted simulated path silently clips a density or hides a numerical or
   no-arbitrage violation.
6. Locked-test reporting includes all metrics and all origins, not only days on
   which the model performs well.
7. Only an implemented physical-measure layer may be described as historically
   predictive. It must beat unchanged-surface persistence on a prespecified
   5- or 8-day proper multivariate score: the upper endpoint of the 95% block-
   bootstrap confidence interval for paired `model - persistence` scores must
   be below zero. A `Q`-only implementation makes no predictive claim and is
   not failed merely because this conditional claim gate is inapplicable.
8. The discrete extension must improve a prespecified validation score after
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
