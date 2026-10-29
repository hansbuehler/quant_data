# Historical statistical SANOS

- **Inputs:** 252 trailing one-session public forward-adjusted spot returns and
  historical option/digital returns reconstructed from the released SANOS
  states at the current book's initial DTE, NMS, and side. Raw source data are
  neither read nor required.
- **Return-source provenance:** daily runs with model-config version 1.31 or
  later explicitly reprice historical options and digitals from SANOS states.
  Earlier version-1.30 daily runs inadvertently used the DLV replay shared
  with the covariance baseline and must be labelled legacy DLV-replay results,
  not SANOS-statistical results. Weekly SANOS repricing is separate.
- **Measure:** historical `P`.
- **Price:** the released SANOS mark is the common market price; this baseline
  does not assert an independent risk-neutral price.
- **Scenarios:** the empirical rolling return rows themselves, without a
  parametric generator.
- **Objectives:** mean-variance, mean-volatility, entropy OCE, and the benchmark
  power-exponential OCE, all with predictable spread costs. An optional
  entropy-only raw-action ridge adds
  $\epsilon\|a\|_2^2/2$ outside the log moment, equivalently inside each
  scenario's wealth. It is distinct from the default unit-RMS-coordinate
  ridge, which should be set to zero when the raw-action objective alone is
  requested. A negative
  homogeneous zero-claim mean-volatility objective is a direct recession
  certificate: the unregularized problem is unbounded along that direction.
  With zero ridge that certificate remains fatal. With positive ridge the
  configured estimator is instead coercive and strongly convex: an
  independently certified, cap-free optimum is accepted while the ridge-zero
  recession certificate remains visible as a nonfatal
  `unregularized_recession_direction` warning. Cap hits, certificate failures,
  and configured-objective recession flags remain separate fatal diagnostics.
- **Hedge:** independently train fresh-book claim and zero policies for the
  daily claim-minus-zero hedge. Weekly buyer, seller, and zero policies supply
  internally consistent indifference prices, while the weekly executed hedge
  is a separate drift-neutral policy fitted with centered instrument gains and
  the actual carried position so its turnover cost matches common realized
  accounting.
- **Training:** deterministic full-batch optimization on 252 uniformly weighted
  daily scenarios. Mean-variance uses exact transaction-cost lasso coordinate
  descent with a declared quadratic ridge of $10^{-4}$ in unit-RMS instrument
  coordinates and a KKT stopping rule. A refined candidate is certified against
  the conic objective evaluated at that same candidate, never against a stale
  pre-refinement solver objective. Entropy and power-exponential use an exact
  positive/negative action split with analytic gradients and a KKT gate, while
  mean-volatility uses an exact conic program followed, only when its
  independently recomputed KKT residual misses tolerance, by an exact-cost
  signed-action refinement of the unchanged objective. The entropy and
  power-exponential gate is the independently recomputed KKT residual of the
  original signed-action problem, not the optimizer's line-search status; a
  stalled split solve receives one restart from its canonical signed action
  without changing the objective or tolerance. When the daily numerical bound
  binds a finite ridge-regularized solution, the solver expands it by a factor
  of ten and accepts the result only after a second, higher-bound solve returns
  the same cap-free policy, objective, and pure hedge within the declared
  tolerances. This safeguard changes neither the objective nor the executable
  position cap. Away from the declared scale floors and position caps, the
  common coordinate scaling and ridge make the result invariant to a change of
  instrument-return units.
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
  finite and is excluded from reported certainty equivalents. Before any
  production outcome, each objective/risk-parameter scope selects its own ridge
  from the frozen grid $10^{-4},\ldots,10^3$ plus the exact zero-position
  baseline. Candidate regret relative to zero is evaluated on five untouched
  chronological blocks, with equal aggregate weight on pricing policies, the
  shared zero policy, and pure execution policies. The one-standard-error rule
  chooses the strongest regularization, including the exact null policy, that
  remains statistically tied with the best feasible candidate, but the null
  policy cannot mask a study in which every finite-ridge fit failed. Both the
  held-out fit and full-window refit must pass independent certificates, and
  pure execution policies must remain strictly inside the precommitted
  economic position cap of 1,000 in both fits. The complete
  date blocks, objective, risk parameter, source commit, and selected policy
  are authenticated in the resulting artifact. This single pre-outcome
  schedule-family calibration is declared for both frozen-weekly and
  restarted-weekly runs; it is never reused for daily fits. Scaled action
  coordinates improve conditioning while
  solver certificate, an exact objective reconstruction, and an explicit primal
  constraint check supply the convergence gate. Weekly mean-variance normalizes
  each turnover epigraph row without changing its physical objective, solves
  first with CLARABEL, and retries an uncertified point independently with OSQP
  and HiGHS. Every candidate is judged by the same unscaled physical certificate;
  the implementation never compares a sparsified point against a stale solver
  objective. A primary conic solve labelled
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

pytest -q tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos.py \
  tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos_weekly.py
```

Reference for the released surface representation: Buehler, Horvath, Kratsios,
Limmer, and Saqur,
[SANOS: Smooth strictly arbitrage-free non-parametric option surfaces](https://arxiv.org/abs/2601.11209).
