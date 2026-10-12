# Weekly benchmark and common-pricing review

## Verdict

**PASS.** I found no blocking mathematical, timing, or accounting defect in the
corrected weekly benchmark/common-pricing gate. The implementation consistently
settles claims on the inception-forward-normalized path, uses direct SANOS
prices for European digitals, uses the inception SANOS-DLV model only where
path dependence requires it, and keeps all realized hedge cashflows in
discounted episode-origin PV units.

This is a gate for common weekly claims, reference prices, and mechanical
accounting. It is not approval of model-specific weekly policies or
overlap-aware performance inference, which remain later benchmark work.

## Findings

### Settlement and information sets

- `benchmark/weekly.py:315` fixes each digital strike from the origin five-day
  ATM volatility and origin NMS geometry. All 13 claims use
  `WeeklyPathBasis.INCEPTION_FORWARD`.
- `benchmark/weekly.py:446` constructs three distinct paths. Target settlement
  uses `S(t_i)/F(t_0,t_i)`; actual `S(t_i)/S(t_0)` is reserved for fixed
  cash-strike hedge replay and traded spot notional; the compounded rolling
  forward-adjusted path is diagnostic only. This is the correct correction:
  using rolling future curves would make the target itself depend on
  information unavailable at inception.
- `benchmark/weekly.py:434` records only the 252 origins strictly before the
  decision date as training data. Restarted schedules request a new trailing
  window at each permitted refit date; frozen schedules refit only at origin.
  Future market days are used only to realize the episode, mark the already
  frozen contracts, and settle them.

### Common and model prices

- `benchmark/weekly.py:330` prices European digitals directly from the origin
  SANOS density at their fixed `K/F(t_0,T)` and multiplies by the origin
  discount factor.
- `models/sanos_lv/weekly_pricing.py:72` preserves those direct SANOS prices as
  the common market marks. It fills only realized variance and the barriers
  with SANOS-DLV reference prices because their path dependence is not
  identified by a terminal SANOS density alone.
- `models/sanos_lv/weekly_pricing.py:102` also reports a full set of DLV model
  prices, including DLV digitals. The standalone smoke makes the distinction
  explicit as `common_prices` versus `dlv_model_prices`; it does not silently
  substitute DLV digitals for common SANOS marks.

### DLV transitions and background

- `quantdata/hedging/data/pricing.py:377` builds each transition column by
  applying the same implicit DLV step to a basis density. Consequently
  `T @ q` advances a column density. The checks correctly sum over the new-state
  index and verify `1' T = 1'` and `K' T = K'`.
- On the public 2023-01-17 state, the independently reproduced maximum mass and
  martingale errors are `1.3323e-14` and `1.3101e-14`; all transition entries
  are non-negative. Repeated application preserves mass and forward mean to the
  same numerical scale.
- `models/sanos_lv/weekly_pricing.py:224` computes each simple-return second
  moment exactly under the discrete DLV transition and independent mean-one 5%
  lognormal background:
  `E[(K_i/K_j B-1)^2]=(K_i/K_j)^2 exp(sigma^2 dt)-2K_i/K_j+1`.
  Averaging the five increments therefore matches the benchmark's exact
  unannualized realized-variance definition within the discrete DLV model.
- `models/sanos_lv/weekly_pricing.py:249` applies the DLV transition, Gaussian
  convolution, mean-one lognormal drift, and end-of-day barrier kill in the
  right order. The uniform log grid recombines, explicitly rejects tail-mass
  loss above `1e-12`, and converges materially with refinement.

### Accounting

- Fixed contracts retain inception expiry and cash strike. They are marked on
  later surfaces at the fixed contract geometry, settle for cash without an
  expiry transaction cost, and cannot be reopened after expiry.
- Spot and option gains, opening/rebalancing costs, expiry settlements, and
  terminal liquidation are all converted to episode-origin PV before
  aggregation. The common market price/model price decomposition reconciles
  exactly to economic P&L.

## Independent evidence

- Focused suite:
  `22 passed` for the weekly benchmark, SANOS-LV weekly pricing, and public
  dataset pricing tests.
- Public-candidate standalone smoke on 2023-01-17 completed successfully with
  13 finite non-negative common prices and 13 DLV model prices. The ATM direct
  SANOS call and put digitals sum to the five-day discount factor.
- Independently reproduced 2023-01-17 prices include realized variance
  `8.48291268e-05`, down-and-out put `0.0067339621`, and up-and-out call
  `0.0076204355` at the 2,049-point production grid.
- The 2,049-point default differs from 4,097 points by less than `1.2e-7` on
  2023-01-17. On the independently checked adverse date 2025-04-07, the maximum
  difference is `1.50e-6` (0.029% relative), improved from `1.05e-5` at the old
  1,025-point default. The updated default is therefore supported by the
  available convergence evidence.

## Non-blocking qualifications

1. Barrier pricing remains a converged numerical integration, not a closed-form
   result. Production artifacts should retain grid size and convergence
   diagnostics, especially for dates with mass close to a barrier.
2. `WeeklySanosLvPrices` intentionally contains DLV digital model prices while
   `build_common_weekly_claims` intentionally retains direct SANOS digital
   market marks. Downstream tables should preserve the same explicit naming so
   the two price concepts cannot be confused.
3. The common-pricing smoke is a strong deterministic gate, but benchmark
   inference over 528 overlapping origins still needs the planned HAC or block
   bootstrap treatment.

## Resource note

`gpuctl resources --json` could not reach the daemon and used its local
fallback. All reported RAM and disk checks were `OK`; this review used CPU only.
