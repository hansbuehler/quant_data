# Independent quant review: SANOS-LV Phase 2

## Reviewed state

- Review date: 2026-10-12
- Base Git revision: `ed31c4b10ee29eba550b6000b20c5f2cc467513a`
- Source state: uncommitted hedging worktree on `main`
- Scope: public market panel, rolling DLV factors, daily digital benchmark,
  SANOS-LV configuration/Greeks/adapter/smoke, DLV pricing, and focused tests
- Reviewer: independent `quant_dataset_review` agent

## Disposition

The Phase-2 SANOS-LV development gate **PASSES**. The implementation provides
rolling market and DLV state, eight daily DLV-factor exposures, all 20 agreed
daily digital targets, fixed-contract option replay, delta/factor hedges, common
accounting, and a passing standalone fixture. This is approval of the Phase-2
CPU baseline, not approval of a full 2023--2025 production backtest.

## Information timing

- The rolling fit takes the last 252 gap-free origins strictly before the
  decision date. A pair's endpoint may equal the decision date, so the fit uses
  information through the close at which the hedge is formed, but never an
  endpoint after that date.
- On the first eligible decision date, 2023-01-17, the fitted transitions begin
  on 2022-01-03 and end on 2023-01-17. The 2023-01-18 transition is used only
  to evaluate the frozen action.
- Target strikes, hedge eligibility, SANOS marks, DLV prices/Greeks, and
  predictable cost weights all come from the decision-date state. Endpoint
  state and spreads enter only the subsequent realized mark and transaction
  cost.

The committed smoke fixture therefore contains an out-of-sample realized
transition, but that transition does not enter fitting or action selection. It
must remain a regression fixture rather than a source of architecture,
parameter, or target-size choices.

## Fixed-contract DLV factors

- The alignment is quantitatively correct. Origin slab `h+1` and endpoint slab
  `h` share a calendar expiry. For an origin coordinate `K/F_t(T)`, the endpoint
  query is

  ```text
  K/F_{t+1}(T)
      = [K/F_t(T)] [F_t(T)/S_t]
        / [S_{t+1}/S_t] / [F_{t+1}(T)/S_{t+1}].
  ```

  The observed gross spot return is reconstructed as the stored one-session
  `F_t(t+1)/S_t` times `S_{t+1}/F_t(t+1)`.
- `F/S` is log-linearly interpolated from zero DTE inside the released curve.
  For synthetic DLV nodes beyond its last expiry, the implementation explicitly
  holds the terminal `F/S` level flat. Endpoint strike queries outside the DLV
  interior are clipped by constant endpoint interpolation, and both conventions
  are reported.
- The coordinatewise AR residual is smoothed before factor extraction. Rank
  eight means one DLV loading projected on the standardized front-spot Gaussian
  score plus seven orthogonal residual principal components. The reported
  explained fraction is relative to the smoothed residual energy.
- Factor bumps are central `+/-1` daily-score shocks in `log(DLV squared)`, then
  converted back to the public raw log-excess state with the configured DLV
  bounds. The front one-day DLV slab is deliberately unchanged. Consequently a
  one-day target has zero future-DLV factor vega; its remaining first-order risk
  is spot delta.

For the first eligible fit, rank eight explains 77.6590% of smoothed innovation
energy. The mean fixed-strike clipping fraction is 2.5981%, and 8.4904% of
forward-curve queries use the documented flat-tail convention. These are not
gate failures, but they are material diagnostics that must accompany results
and should receive sensitivity checks before headline production runs.

## Prices, Greeks, and hedge construction

- SANOS supplies the common market target mark and executable option marks.
  DLV supplies the model target price, option prices used for Greeks, spot
  deltas, and factor vegas. The reported pricing contribution is therefore the
  DLV-minus-SANOS price difference for the long target.
- The DLV digital kernel is the strike derivative of the DLV Black-mixture call
  kernel: calls use `N(d2)` and puts use `N(-d2)`. In the atomic limit it uses
  complementary call `>` and put `<=` conventions. Regression tests verify the
  strike derivative and call-put complement.
- A fixed-cash digital delta changes `K/F` by division by the relative spot
  bump while leaving its cash payoff unchanged. A fixed-cash vanilla delta also
  scales its forward price by `1 +/- bump`. The implemented central differences
  follow those homogeneous formulas.
- The linear program first neutralizes all retained DLV-factor vegas using the
  least predictable option-cost portfolio. Expiring options receive one current
  half-spread and unexpired options receive twice the current half-spread as a
  predictable round-trip proxy. Normalized spot then removes the residual
  delta.
- Factor and uncapped delta residuals are checked against tolerance. Any option
  or spot position at the configured numerical cap is counted and makes the
  result non-converged. An infeasible capped factor system raises instead of
  being silently reported as a hedge.

## Targets and accounting

- The target catalog contains 20 unique claims: ATM put/call plus four OTM puts
  and four OTM calls at both one- and three-business-day maturity.
- One-day claims settle at the next close. Three-day claims are marked after one
  session from the independently fitted endpoint SANOS state with fixed cash
  strike and two business days remaining.
- Expiring hedge options settle for cash without an exit spread. Unexpired
  fixed contracts are repriced at the endpoint, discounted back one session,
  and pay the endpoint interpolated closing half-spread. Spot pays one basis
  point at entry and at its discounted terminal liquidation value.
- Market price, DLV model price, hedge gain, costs, pricing contribution, pure
  hedge error, and economic P&L remain in origin-PV spot-one units.

For three-day targets the common result field named `payoff` contains the
next-date discounted market value rather than final settlement. The arithmetic
is correct, but a later schema version should rename it to `terminal_value` or
document this distinction in every aggregate report.

## Validation evidence

- 48 focused hedging tests pass.
- The standalone SANOS-LV public-candidate fixture passes.
- The complete 20-target book on 2023-01-17 completes on CPU with 20/20
  converged records, zero cap hits, no more than eight nonzero option positions
  per target, maximum factor residual `8.65e-18`, maximum absolute delta
  residual zero, and maximum absolute action 148.46.

## Non-blocking follow-ups

1. Add a committed three-day/full-factor fixture. The current standalone
   one-day ATM fixture has zero future-DLV factor vega, so it does not by itself
   regression-protect the principal Phase-2 LP path.
2. Run a multi-date CPU pilot before production to measure LP feasibility, cap
   hits, missing-contract replay, clipping, flat-tail use, turnover, and action
   tails across market regimes.
3. Label rank eight precisely as one spot-projected loading plus seven
   orthogonal PCs, and label its explained variance as smoothed-residual
   coverage rather than generic DLV variance coverage.
4. Treat twice the current half-spread as a predictable optimization proxy;
   realized accounting already uses the actual endpoint synthetic spread.
