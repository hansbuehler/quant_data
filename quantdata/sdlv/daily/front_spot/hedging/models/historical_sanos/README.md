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
  power-exponential OCE, all with predictable spread costs.
- **Hedge:** independently train fresh-book long, short, and zero policies for
  internally consistent indifference prices. The executed hedge is a separate
  drift-neutral policy fitted with centered instrument gains and the actual
  carried position, so its turnover cost matches common realized accounting.
- **Training:** deterministic full-batch optimization. Mean-variance uses exact
  transaction-cost lasso coordinate descent with a small declared quadratic
  ridge and a KKT stopping rule; the other objectives use analytic-gradient
  L-BFGS-B. The oldest 80% is fitted and the newest 20% is held out for a
  validation diagnostic; after validation the policy is refitted on all 252
  observations. No future return is used.
- **Artifacts:** common `hedging-smoke-v2` JSON with explicit claim quantity. The model has no neural
  checkpoint because its rolling optimizer is deterministic and inexpensive.
- **Daily book:** one shared 252-transition return panel supplies all 20 agreed
  1D/3D digital targets. Each target retains an independent claim-minus-zero
  policy fit and common execution accounting.
- **Weekly book:** complete historical remaining-horizon blocks reprice the
  frozen option universe at the current relative DTE, NMS, and side. Frozen
  weekly uses one five-row open-loop policy; restarted weekly refits the
  remaining rows from the realized claim prefix and carried positions. Future
  claim paths preserve the fixed inception-forward normalization while using
  the current known forward curve.

Run on CPU:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.smoke

python -m \
  quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.weekly_smoke
```
