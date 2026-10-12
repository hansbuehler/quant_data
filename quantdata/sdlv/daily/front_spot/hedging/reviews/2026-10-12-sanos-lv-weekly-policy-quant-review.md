# SANOS-LV weekly policy quantitative review

**Date:** 2026-10-12
**Final verdict:** **PASS** — both original blockers are resolved. No blocker
remains in the requested SANOS-LV weekly policy scope.

## Scope

Reviewed:

- `benchmark/weekly.py`
- `benchmark/weekly_runner.py`
- `models/sanos_lv/weekly_conditional.py`
- `models/sanos_lv/weekly_adapter.py`
- the weekly policy, weekly pricing, Greek, and weekly benchmark tests
- the directly used SANOS-LV weekly pricing and Greek routines

No implementation files were modified.

## Resolved blockers

### 1. Spot bumps now preserve the realized prefix

The original implementation changed the latest observed close during the spot
bump. That could resurrect an already knocked-out barrier and revalue accrued
realized variance.

The corrected implementation passes the realized prefix unchanged to every
conditional payoff (`weekly_conditional.py:147-166`). The future scale builder
uses the bumped live spot only for unobserved future dates and explicitly sets
its time-zero scale to the immutable observed endpoint
(`weekly_conditional.py:363-388`). Consequently:

- the barrier history test always sees the actual observed closes;
- accrued realized returns are independent of the hedge bump;
- the first remaining realized return uses the fixed latest observation as
  denominator and the bumped first future close as numerator; and
- subsequent future-return ratios use adjacent simulated suffix values.

The exact prior failure now returns zero price and zero delta for latest
observed down-barrier hits at `0.9498`, `0.94995`, and `0.94999`. The new test
also verifies that changing the accrued RV prefix changes the conditional
price but not its current spot delta or factor vegas.

### 2. The exact factor hedge now minimizes weighted turnover

`solve_delta_factor_hedge` now accepts the carried live option positions and
introduces non-negative turnover variables satisfying

```text
u_i >= q_i - q_previous_i,
u_i >= q_previous_i - q_i.
```

It minimizes `sum_i weight_i u_i` subject to exact factor-vega neutrality and
position bounds (`greeks.py:213-325`). The weekly adapter passes the previous
positions for the currently alive contracts (`weekly_adapter.py:174-183`).
Expired contracts are absent from the optimization and remain zero in the
full frozen-book action.

The regression has two identical-vega instruments, a cheap instrument from
zero, and an already carried expensive instrument. The corrected LP retains
the carried instrument because doing so has zero turnover, proving that the
objective is no longer absolute position size.

The one-spread weight for next-session cash settlement and two-spread
predictable round-trip proxy for a surviving contract remain internally
consistent. Realized accounting separately charges actual position changes,
free cash settlement, and terminal liquidation.

## Correct components retained

- The target path is consistently `S(t_i)/F(t_0,t_i)`. Current conditional
  scales use the observed `S_t/S_0`, the current forward curve, and the fixed
  inception forward curve; common accounting remains in discounted
  episode-origin units.
- Conditional digitals use the remaining DLV terminal distribution at the
  fixed inception strike. On the real 2023-01-17 episode at decision two, the
  ATM call-plus-put price equalled discounted unit cash to `1.5e-15`; summed
  delta was `6.9e-13` and maximum summed factor vega was `1.1e-14`.
- The realized-variance future expectation uses the exact independent 5%
  lognormal second moment. Barrier propagation uses the same DLV transition
  orientation and background process as the accepted inception pricer.
- Target and hedge factor vegas are both per-current-spot values under the
  same one-daily-standard-deviation log-DLV-squared bump. The omitted current
  spot multiplier is common and cancels in the factor-neutrality equation;
  spot deltas are in origin-normalized spot units.
- Rolling factor fits use only transitions ending on or before the dated
  as-of state. Frozen weekly reuses one origin fit while actions react to the
  current observed state; restarted weekly uses five distinct dated fits.
- Policy identity includes schedule, model fit, target label, claim quantity,
  policy configuration, episode origin, and frozen contract identities.
  Expired contracts are excluded from repricing/action and cannot be reopened;
  cash settlement and terminal liquidation are handled by the common engine.

## Non-blocking qualifications

- The two-spread surviving-contract weight is a predictable round-trip proxy,
  not a solution of a multistage transaction-cost control problem. That is an
  explicit baseline choice; the LP now solves its stated myopic weighted-
  turnover problem exactly.
- The benchmark target catalogue makes `target_label + episode_origin`
  sufficient to recover each canonical claim. A more general public cache API
  should hash the complete numerical claim definition and path basis so a
  custom claim cannot reuse a canonical-label cache entry.
- Cache hits are keyed by the expected hash but are not revalidated against
  the identities stored inside a caller-supplied cached object. The normal
  in-process cache is safe; explicit hit-time checks would harden external
  cache injection.
- Weekly policy output still discards factor residual, delta residual, and
  cap-hit diagnostics. Production reporting should retain them.

## Evidence

- Focused CPU suite: **39 passed** in 6.19 seconds.
- Direct latest-fixing barrier reproduction: all three previously problematic
  observed closes now return price `0` and delta `0`.
- Real frozen-weekly public replay from 2023-01-17 completed with one reused
  fit, finite actions, 207 frozen contracts, transaction cost
  `0.0436534214`, and economic P&L `0.1578042784` for the ATM put digital.
- Real restarted-weekly replay completed with five distinct fit identities,
  finite actions, transaction cost `0.0435715338`, and economic P&L
  `0.1155734361` for the same target.
