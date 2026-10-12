# GROMM Hedging Integration Remediation Review

**Date:** 2026-10-12
**Current verdict:** **PASS** after final re-review on 2026-10-13
**Initial remediation verdict:** **FAIL**
**Prior review:** `2026-10-12-gromm-hedging-integration-quant-review.md`
**Scope:** independent re-review of the four requested remediations. No
implementation files were modified.

## Remaining blockers

### 1. The directly trained executable policy is not a pure hedge

The nonlinear-turnover mismatch is mechanically removed: `hedger.py:137`--
`hedger.py:145` trains a fourth policy, `gromm_pure_hedge_actions` executes that
policy directly, and the weekly adapter carries its own net position across
restarted fits (`weekly_adapter.py:197`--`weekly_adapter.py:224` and
`weekly_adapter.py:305`--`weekly_adapter.py:328`). Its objective therefore sees
the same net turnover, free settlement, and terminal liquidation that the
benchmark charges.

However, `pure` is trained on the same unmodified `GrommTrainingData` as the
long-claim policy. In particular its instrument gains retain their P drift and,
under Q, any difference between the common SANOS executable marks and GROMM2
conditional expectations. It is therefore an independently initialized second
claim policy, not a speculation-free pure hedge. This contradicts the project's
reason for separating claim and zero books: remove positions that would be held
without the claim. It also makes the issue material precisely after fixing the
entry marks, because a nonzero SANOS-versus-GROMM price basis is now an
optimizable expected return.

The executable training law must remove the zero-claim/speculative component
while retaining exact net-action costs. A direct drift-neutral construction,
such as the centered-instrument-gain geometry already used by historical SANOS,
would satisfy that requirement. Tests currently establish only that the fourth
checkpoint exists and returns a finite capped action; they do not test absence
of speculative demand.

### 2. Initial weekly feature equality still assumes the fitted and released DLV states are identical

The common-mark portion is fixed: generated time-zero option marks now come from
`state.option_marks_pv` in `gromm2_weekly.py:106`, matching the live feature.
The committed equality regression passes because its synthetic fitted GROMM2
state and released market day use identical DLV arrays.

The code paths still differ for the DLV feature coordinates. Generated time zero
uses `market_state.dlvsq` in `gromm2_weekly.py:183`--`gromm2_weekly.py:195`,
whereas the live feature uses `state.market_day.dlvsq` in
`gromm2_weekly.py:463`--`gromm2_weekly.py:485`. The affine GROMM2 fit is not
contractually equal to the released DLV state; indeed the production gate only
requires a percentage of prices to lie inside spread.

An adversarial repetition of the existing fixture with the fitted DLV state
changed by 1% produced four different initial feature coordinates (indices
18--21), with maximum absolute difference `0.009950330853168321`. Initial
generated and live conditioning must select the same state by construction, and
the regression must use deliberately distinct fitted and released arrays.

## Resolved findings

### Common executable marks and price reconstruction

- Daily generated option gains now start from released SANOS marks via
  `instrument_entry_prices` (`gromm2_daily.py:281`--`gromm2_daily.py:284` and
  `gromm2_daily.py:381`--`gromm2_daily.py:383`).
- Daily claim returns are consistently centered on the common market claim mark,
  and the diagnostics add that same mark back to signed indifference increments
  (`gromm2_daily.py:416`; `adapter.py:317`--`adapter.py:324`). The separate
  GROMM2 Q expectation remains the reported model price.
- Weekly option books start from `state.option_marks_pv`; settlement and future
  GROMM2 repricing remain in episode-origin PV units.

### Physical-drift and source-identity gates

- Rolling artifacts now record and require the exact runtime public-manifest
  hash. Both daily and weekly adapters supply `panel.dataset_manifest_sha256`.
- Loading and fitting enforce minimum physical projection and maximum daily
  market-price-of-risk norm gates (`rolling_gromm2.py:240`--
  `rolling_gromm2.py:246`, `rolling_gromm2.py:319`--`rolling_gromm2.py:325`, and
  `rolling_gromm2.py:478`--`rolling_gromm2.py:513`).
- The daily scenario regression now uses nonzero loadings and a nonzero physical
  risk price, verifies finite P output, and verifies P differs from Q.

## Verification

The six focused GROMM CPU modules pass **28 tests** in 19.64 seconds. Ruff passes
on the GROMM implementation and focused tests, and `git diff --check` passes for
the reviewed implementation/tests/PLAN paths.

The remediations close the original entry-mark, nonlinear mechanical-cost,
physical-gate, and dataset-identity defects. Production acceptance remains
blocked only by the two semantic issues above.

## Final Re-review — 2026-10-13

**Verdict:** **FAIL**

The second feature blocker is resolved. Generated time-zero weekly features now
use the released `state.market_day.dlvsq` (`gromm2_weekly.py:183`--
`gromm2_weekly.py:195`), exactly as the live feature does. The regression now
deliberately changes the fitted GROMM2 DLV state by 1% and still obtains equality
at the initial decision.

The executable policy now receives per-step, per-instrument gains centered
across all simulated paths (`hedger.py:236`--`hedger.py:243`). This removes a
constant unconditional drift, but does **not** complete the pure-hedge fix for a
state-dependent LSTM policy:

1. Unconditional centering leaves conditional drift. For example, with a
   predictable state `X` taking values `+1` and `-1`, return `R=X` has zero
   unconditional mean after the implemented centering, while action `a(X)=X`
   earns `E[aR]=1`. The weekly P risk price is explicitly recomputed from the
   path's floating DLV state and previous factor score, so this is the relevant
   case rather than a hypothetical edge case. A direct check through
   `_centered_pure_hedging_data` gives unconditional return mean `0.0` and
   predictable-action gain mean `1.0`.
2. Centering is performed before `fit_gromm_policy` creates its antithetic
   train/validation split (`training.py:255`--`training.py:259`). Consequently
   validation returns enter the estimated centering vector and the validation
   set is not untouched. This biases validation loss, early stopping, and the
   one-standard-error architecture comparison.

The pure policy requires a predictable conditional-drift removal fitted only on
the training paths and then frozen on validation paths, or an equivalent coupled
claim-versus-zero formulation whose executed net-action cost is evaluated
exactly. A regression that adds only a constant unconditional drift does not
exercise either failure mode.

Final focused verification: the six GROMM CPU modules pass **29 tests** in 19.87
seconds; Ruff and `git diff --check` pass. No implementation file was modified by
this review.

## Conditional Q-Shadow Re-review — 2026-10-13

**Final verdict:** **PASS**

The prior unconditional-centering blocker is resolved. Statistical centering has
been removed entirely. The executable pure policy now receives model-defined
conditional Q-shadow gains on the same scenario geometry as the pricing policy:

- daily generation retains the unshifted antithetic factor normals, advances a Q
  state with the same background shock, and measures pure option gains from the
  current GROMM model prices (`gromm2_daily.py:322`--`gromm2_daily.py:375` and
  `gromm2_daily.py:386`--`gromm2_daily.py:422`);
- at every weekly step, `q_result` is advanced from the **current realized P
  state** with the same Brownian and background shocks but without the physical
  risk-price shift (`gromm2_weekly.py:241`--`gromm2_weekly.py:289`); and
- weekly Q-shadow gains are measured against that P state's current GROMM model
  marks, after which only the P state is carried to the next decision
  (`gromm2_weekly.py:316`--`gromm2_weekly.py:401`). Thus this is a sequence of
  conditional one-step Q shadows, not a separate full Q path.

Claims, policy features, spreads, transaction costs, settlement masks, terminal
liquidation costs, and carried net positions remain those of the requested P or
Q scenario. `fit_gromm_claim_minus_zero` validates that pure and pricing data
share those arrays and differ only in the permitted instrument-gain channel
(`hedger.py:238`--`hedger.py:267`). No conditional-mean estimator is fit and no
validation observation enters a centering statistic, so the previous holdout
leakage is also eliminated.

The daily nonzero-risk-price regression verifies that actual P and Q returns
differ while their Q-shadow returns agree. The weekly regression verifies that P
and Q option paths differ and that the first conditional Q-shadow step agrees.
The released-DLV adversarial initial-feature regression remains in place.

Final focused CPU verification passes **28 tests** in 22.83 seconds. Ruff and
`git diff --check` pass. As a non-blocking strengthening, add a later-step weekly
test that independently reconstructs the Q shadow from a realized P state; the
current first-step test plus direct code inspection are sufficient for this
gate.
