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
  power-exponential OCE, all with predictable spread costs. A negative
  homogeneous zero-claim mean-volatility objective is a direct recession
  certificate: the unregularized problem is unbounded along that direction.
  Such a cell remains explicitly nonconverged even when the declared ridge
  makes its numerical surrogate finite; cap hits and the recession flag remain
  separate diagnostics.
- **Hedge:** independently train fresh-book long, short, and zero policies for
  internally consistent indifference prices. The executed hedge is a separate
  drift-neutral policy fitted with centered instrument gains and the actual
  carried position, so its turnover cost matches common realized accounting.
- **Training:** deterministic full-batch optimization on 252 uniformly weighted
  daily scenarios. Mean-variance uses exact transaction-cost lasso coordinate
  descent with a declared quadratic ridge of $10^{-4}$ in unit-RMS instrument
  coordinates and a KKT stopping rule; entropy and power-exponential use an exact
  positive/negative action split with analytic gradients and a KKT gate, while
  mean-volatility uses an exact conic program followed, only when its
  independently recomputed KKT residual misses tolerance, by an exact-cost
  signed-action refinement of the unchanged objective. The entropy and
  power-exponential gate is the independently recomputed KKT residual of the
  original signed-action problem, not the optimizer's line-search status; a
  stalled split solve receives one restart from its canonical signed action
  without changing the objective or tolerance. Away from the declared scale
  floors and position caps, the common coordinate scaling and ridge make the
  result invariant to a change of instrument-return units.
  The oldest 80% is fitted and
  the newest 20% is held out for a validation diagnostic; after validation the
  policy is refitted on all 252 observations. Selection-fit convergence and
  cap diagnostics are retained, but only the final full-window fit determines
  whether the executable policy converged. The implementation warns when there
  are fewer than two observations per action parameter and labels that result as
  a regularized estimate rather than an unrestricted empirical optimum. No
  future return is used. Higher-frequency histories are a separate future model
  revision; they are not mixed into this daily estimator.
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
  finite; the same $10^{-4}$ unit-RMS ridge is a fitting regularizer and is
  excluded from reported
  certainty equivalents. Scaled action coordinates improve conditioning while
  solver certificate, an exact objective reconstruction, and an explicit primal
  constraint check supply the convergence gate. Weekly mean-variance certifies
  the raw point associated with the solver-reported objective and retries a
  failed OSQP certificate with CLARABEL; it never compares a sparsified point
  against the stale raw objective. A primary conic solve labelled
  `optimal_inaccurate` receives a cold retry and is then solved independently by
  SCS. A result labelled `optimal` or `optimal_inaccurate` is accepted only if
  all values are finite, the exact reconstructed objective agrees with the
  solver objective, the primal constraints pass the declared tolerance, and no
  solver or executable cap is hit. Thus `optimal_inaccurate` is treated as a
  status requiring an explicit certificate rather than as either automatic
  success or automatic failure. Selection-fit failure alone does not reject a
  separately certified full-window fit. Virtual pricing
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
