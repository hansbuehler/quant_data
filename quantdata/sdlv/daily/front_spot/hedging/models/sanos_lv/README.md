# SANOS-LV reference

- **Inputs:** public SANOS and DLV states/geometries, SANOS common market marks,
  normalized half-spreads, forward/discount curves, and forward-adjusted spot
  transitions.
- **Measure:** SANOS common marks and DLV local-vol prices/Greeks are
  risk-neutral; hedge evaluation uses the subsequent historical transition.
- **Price:** the common market price is the SANOS digital mark. The model price
  and hedge Greeks come from the corresponding public DLV local-vol geometry.
- **Scenarios and fitted quantities:** this reference does not fit a stochastic
  scenario generator or a trainable hedge policy. It fits the rolling DLV
  factor basis below, then obtains prices and Greeks from the local-volatility
  repricer on the released geometry.
- **Factors:** estimate one front-spot-projected DLV loading plus seven
  orthogonal residual principal components from the trailing 252 exact
  one-session transitions. Reported factor coverage is the fraction of
  smoothed DLV innovation energy explained. Each pair holds cash strike and calendar expiry
  fixed, then aligns the endpoint log-DLV-squared surface through both dates'
  `F/S` curves. The 25-day synthetic DLV tail holds the last released `F/S`
  value flat when it extends beyond the final listed expiry; the fit reports
  that coordinate share.
- **Hedge:** central one-daily-standard-deviation bumps in log-DLV-squared
  coordinates. A linear program chooses the exact factor-vega hedge with
  minimum predictable option transaction cost; normalized spot then neutralizes
  the remaining delta. The 1D target has zero future-DLV factor vega, while 3D
  targets use at most eight hedge options.
- **Weekly policy:** reprice the fixed inception claim conditionally from the
  immutable realized path prefix and the current DLV state. Frozen weekly uses
  the origin rolling factor basis at all five decisions; restarted weekly fits
  and aligns a new dated basis. The fixed inception forward curve maps every
  future current-forward-normalized state back to the claim's settlement basis.
- **Targets:** ATM call/put and the agreed OTM NMS grid at 1D and 3D, for 20
  targets in total.
- **Costs:** one basis point at spot entry and liquidation; options pay their
  opening half-spread, plus the discounted closing half-spread when unexpired.
  Cash settlement is free. Hedge optimization uses the current opening
  half-spread for next-date settlement and twice the current half-spread as a
  predictable round-trip proxy for contracts that remain alive; realized
  accounting uses the actual next-date synthetic closing spread.
- **Fixture:** the committed rolling-factor smoke starts on 17 January 2023,
  the first date with 252 retained exact-session transitions, and evaluates the
  next untouched market transition. It checks both the one-day delta-only path
  and a three-day target with the full eight-factor option hedge.
- **Validation:** factor estimation and hedge construction are deterministic;
  no evaluation transition selects parameters. The smoke checks the next
  untouched transition against a committed fixture, while the focused tests
  cover pricing, factors, Greeks, and weekly policies.
- **Terminal value:** for a one-day target the common `payoff` field is cash
  settlement. For a three-day target it is the discounted independently fitted
  next-date SANOS market value of the unfinished claim.
- **Artifacts:** common `hedging-smoke-v2` JSON with explicit claim quantity.

Run on CPU:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.weekly_policy_smoke
pytest -q tests/quantdata/sdlv/daily/hedging/models/test_sanos_lv.py \
  tests/quantdata/sdlv/daily/hedging/models/test_sanos_lv_greeks.py
```

Programmatic full-book entry point:

```python
run_sanos_lv_daily_book(panel, origin_date=origin_date)
```

Reference: Buehler, Horvath, Kratsios, Limmer, and Saqur,
[SANOS: Smooth strictly arbitrage-free non-parametric option surfaces](https://arxiv.org/abs/2601.11209).
