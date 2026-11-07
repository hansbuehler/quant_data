# DYSANOS rolling-model review

Scope: the current 25-day affine-200/daily-AR(1) calibration, not the historical
weekly FLEX or risk-neutral experiments. [MODELS.md](MODELS.md) owns the model
settings and run status. Full independent review and forward path validation
remain pending; calibration alone is not a dynamics-quality result.

## Equations and conventions

Use row-vector factors, matching `v11/rolling_daily.py`:

\[
u_t=b+h_t B,\qquad
h_{t+1}=m+(h_t-m)A+\varepsilon_{t+1},\qquad
\varepsilon_{t+1}\sim N(0,LL').
\]

There are 200 factors and 2,816 native coordinates: 14 expiry blocks of
200 DLV squares plus one forward volatility, followed by two backbone-ratio
coordinates. The 200 inner strikes include ATM; two boundary support nodes
are additional. Market eligibility is filtered to NMS `[-4,3]` **at each
market expiry**, not by the extremes of the wider shared pricing support.

The stable transition is learned directly at one business session:

\[
A=\exp\!\left[-\left(DD'+\tfrac12(R-R')+0.001I\right)\right].
\]

`D` and `L` are lower triangular. `L` has diagonal
`1e-4 + softplus(diag(q_raw))`; this ensures positive definiteness, not an
eigenvalue floor on `LL'`. Dates spanning a missing session are excluded
from the daily likelihood. Forecast initialization across such a gap uses
the actual session count.

For date weights `w_t`, the jointly minimized objective is

\[
\mathcal L
=\frac{\sum_t w_t(q_t+C_t)}{\sum_t w_t}
 +0.1\,\ell_{\rm AR}+0.1\,\mathcal G,
\]

where `C_t` is the configured DLV-cap penalty and

\[
q_t=\frac1{n_t}\sum_j
\frac{(p_{tj}-\mathrm{ask}_{tj})^+
 +(\mathrm{bid}_{tj}-p_{tj})^+
 +0.05|p_{tj}-\mathrm{mid}_{tj}|}
 {\max((\mathrm{ask}_{tj}-\mathrm{bid}_{tj})/2,10^{-12})}.
\]

`ell_AR` is the Gaussian residual negative log-likelihood per factor, averaged
with endpoint-date weights over valid pairs. `G` penalizes the squared
weighted factor mean and deviation of the weighted factor covariance from
identity. Thus `b`, `B`, every `h_t`, the stable generator, `m` and `L` are
all learned against this combined objective; this is **not** PCA followed
by a frozen-state AR regression.

The native DLV square is `min_dlvsq + exp(min(raw, log_cap))`. Pricing is
hard-capped, but a softplus-quadratic excess penalty retains a restoring
gradient above the cap. Forward volatilities and backbone ratios use bounded
sigmoid transforms. The strike total volatility is intentionally
`V_j = sum_{i<=j} forward_vol_i * sqrt(delta_T_i)`, not the square root of
integrated forward variance. The backbone analogue includes the ratio
`eta_i`; its total volatility, and the discrete densities, are interpolated
linearly between expiry nodes. Pricing time is in years; AR time is in
business sessions.

## Implementation and data flow

1. `v11/licensed_market.py` reads observed licensed bid/ask prices, normalized
   once by discounted forward. Saved smoothness-0.1 SANOS surfaces are
   certificate/hash checked and reused, never refitted by this runner.
2. `v11/dlv_initialization.py` recovers a zero-backbone DLV initializer on the
   model grid, reporting floors, caps and repricing error. PCA is only an
   initializer; `jax_affine_lbfgs.py` subsequently fits market quotes.
3. `v11/rolling_daily.py` initializes daily dynamics from that price-fitted
   affine state. Every roll retains overlapping parameters/device buffers,
   transfers one new quote date, filters its factor under the previous model,
   then jointly recalibrates the full window using L-BFGS-B and GPU gradients.
4. Each origin writes its model, per-date fit CSV and report. The manifest
   records source identity, dates, geometry and missing sessions. A separate
   `daily_fit_report.py` produces the compact daily comparison CSV without
   interrupting fitting. All run outputs are external to Git.

The current initializer uses Gaussian AR residual covariance shrinkage only
for its starting value; the subsequent fit learns the full Cholesky factor.
Rolling runs warm-start weights. Resuming restores the latest checkpoint,
but resets L-BFGS curvature; it is not a bitwise optimizer continuation.

## Reading the diagnostics

- **Today's fit:** that day's quotes after the full joint recalibration.
  The separately recorded `forward_state_fit` uses today's quotes to fit only
  its factor under yesterday's decoder; it is filtering, not an unobserved
  price forecast.
- **Previous-day comparison:** weighted mean and inverse-ECDF weighted median
  of the other 249 daily fit percentages. Each date receives its decay weight
  once, not an additional quote-count weight.
- **Window fit:** the logged aggregate containment percentage is quote-count
  weighted, without decay weights. It differs from both the equally
  date-normalized training loss and the decay-weighted previous-day panel.
  The 10th/90th percentile bands are unweighted quantiles across window dates.
- **Grid error:** normalized call-price error against SANOS on the fitted
  pricing grid; this is separate from market bid/ask containment. The rolling
  summary's grid error is for the current date.
- **Runtime:** elapsed daily update including new-day preparation, filtering,
  joint fit and diagnostics. Initial whole-panel loading precedes this timer.

The rolling fit may stop when market containment is at least 90% and, over
100 accepted iterations, improves by less than 0.1 percentage point while
quote-plus-cap loss drops by less than 1%. This is a market-fit plateau,
not proof of AR convergence. The configured iteration ceilings still apply.

## Verification and review priorities

CPU checks cover JAX/Torch price parity, the independent grid pricer, date
weights, daily AR timing and masked gaps, price-preserving re-gauging,
one-day device-buffer replacement, reporter weighting and legacy checkpoint
resume behavior. Run without claiming another GPU:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python -m pytest -q tests/test_dysanos_daily_rolling.py \
  tests/test_dysanos_daily_fit_report.py tests/test_dysanos_jax_affine200.py \
  tests/test_dysanos_rolling_affine_benchmark.py
```

Use the project's explicit interpreter for detached execution. Reviewers
should prioritize:

1. Full 200-dimensional transition/covariance estimation from at most 249
   daily pairs: identification, gauge sensitivity and held-out calibration.
2. Optimizer/plateau behavior: overall window fit can hide weak current-day
   fit, and state filtering can reach its iteration cap without convergence.
3. Joint spot/surface dependence and decoded ATM leverage, volatility-of-vol,
   VIX-type distributions and tail/path statistics. The current rolling runner
   does not estimate a joint spot covariance or demonstrate those properties.
4. Independent between-node linear-volatility pricing checks, cap-activity
   diagnostics, full-gradient checks, device-capacity growth and interrupted
   rolling-resume tests beyond the focused invariants already covered.

A checkpoint sensitivity audit found that rescaling `h`, `m` and `L` by
0.1 and `B` by 10 preserved native states and physical innovation loadings
to numerical precision, but reduced the combined objective from 0.419 to
0.156. The likelihood and soft gauge therefore leave a scale-identification
issue: a lower loss alone need not mean better prices or physical dynamics.
The audit is `cpu-audits/latent-rescaling-2023-07-05.json` within the current
external run directory identified in [MODELS.md](MODELS.md). The running
objective has not been changed.

Static no-arbitrage of the SANOS decoder does not establish that the generated
`P` option-price paths are martingales. No `Q` claim is made here.
