# Historical statistical SANOS

- **Inputs:** 252 trailing one-session public forward-adjusted spot returns and
  historical option/digital returns reconstructed from the released SANOS
  states at the current book's initial DTE, NMS, and side. Raw source data are
  neither read nor required.
- **Measure:** historical `P`.
- **Price:** the released SANOS mark is the common market price; this baseline
  does not assert an independent risk-neutral price.
- **Scenarios:** the empirical rolling return rows themselves, without a
  parametric generator.
- **Objectives:** mean-variance, mean-volatility, entropy OCE, and the benchmark
  power-exponential OCE, all with predictable spread costs. If the homogeneous
  mean-volatility objective fails its recession condition at low risk aversion,
  the fit is retained under the precommitted large numerical cap and reported
  with non-convergence and cap-hit diagnostics rather than treated as a finite
  unconstrained optimum.
- **Hedge:** independently train fresh-book long, short, and zero policies for
  internally consistent indifference prices. The executed hedge is a separate
  drift-neutral policy fitted with centered instrument gains and the actual
  carried position, so its turnover cost matches common realized accounting.
- **Training:** deterministic full-batch optimization. Mean-variance uses exact
  transaction-cost lasso coordinate descent with a small declared quadratic
  ridge and a KKT stopping rule; entropy and power-exponential use an exact
  positive/negative action split with analytic gradients and a KKT gate, while
  mean-volatility uses an exact conic program. The entropy and
  power-exponential gate is the independently recomputed KKT residual of the
  original signed-action problem, not the optimizer's line-search status; a
  stalled split solve receives one restart from its canonical signed action
  without changing the objective or tolerance. The oldest 80% is fitted and
  the newest 20% is held out for a validation diagnostic; after validation the
  policy is refitted on all 252 observations. Selection-fit convergence and
  cap diagnostics are retained, but only the final full-window fit determines
  whether the executable policy converged. No future return is used.
- **Artifacts:** common `hedging-smoke-v2` JSON with explicit claim quantity. The model has no neural
  checkpoint because its rolling optimizer is deterministic and inexpensive.
- **Daily book:** one shared 252-transition return panel supplies all 20 agreed
  1D/3D digital targets. The zero-claim policy is fitted once per date and
  shared across targets; each target retains an independent claim policy,
  claim-minus-zero hedge, and common execution accounting.
- **Weekly book:** complete historical remaining-horizon blocks reprice the
  frozen option universe at the current relative DTE, NMS, and side. Frozen
  weekly uses one five-row open-loop policy; restarted weekly refits the
  remaining rows from the realized claim prefix and carried positions. Future
  claim paths preserve the fixed inception-forward normalization while using
  the current known forward curve. Weekly entropy and power-exponential OCEs
  use exact conic epigraphs for scenario-dependent turnover costs. Their
  declared quadratic ridge makes the over-parameterized empirical problem
  finite; the ridge is a fitting regularizer and is excluded from reported
  certainty equivalents. Scaled action coordinates improve conditioning while
  CLARABEL's optimal certificate supplies the convergence gate. A solve labelled
  `optimal_inaccurate` is never accepted: power-exponential fitting makes one
  cold retry at five times the configured tolerance, floored at
  $5\times 10^{-8}$, and still requires an `optimal` certificate. Virtual pricing
  policies use the declared
  solver cap before the pure hedge is clipped to the executable cap. Weekly
  records preserve internal fitted-policy convergence and cap counts once per
  unique fit identity, separately from cap hits in the five realized action
  books. Planned executable clipping in an unused restarted-policy suffix
  still marks that fit nonconverged without double-counting a realized cap.

Run on CPU:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.smoke

python -m \
  quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_smoke
```
