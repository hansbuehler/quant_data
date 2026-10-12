# Independent quant review: robust covariance Phase 3

## Reviewed state

- Review date: 2026-10-12
- Base Git revision: `ed31c4b10ee29eba550b6000b20c5f2cc467513a`
- Source state: uncommitted hedging worktree on `main`
- Scope: common robust objectives and NMS conversion, public market replay,
  covariance return construction/configuration/solver/adapter/smoke, fixtures,
  and focused tests
- Reviewer: independent `quant_dataset_review` agent

## Disposition

The Phase-3 robust covariance gate **FAILS**. Information timing, normalized
return accounting, NMS reconstruction, cost signs, mean-variance algebra, and
cap diagnostics are sound. Three quantitative defects prevent approval:

1. the quantity optimized as an expected return is a componentwise median, not
   an estimator of the mean required by either advertised objective;
2. a target's hedge materially depends on which unrelated benchmark targets
   are included in the joint Tyler fit; and
3. the implemented mean-volatility baseline has no passing actual-data smoke at
   the required `kappa=1` setting and returns a degenerate capped policy on the
   reviewed date.

## Blocking findings

### 1. The mean and its screen are not estimates of expected return

`regularized_tyler_covariance` returns the componentwise sample median as
`estimate.center`. `fit_covariance_book_actions` then uses that vector as
`instrument_center`, labels `claim_center + instrument_center @ action` as
`expected_profit`, and inserts it into the mean-variance and mean-volatility
objectives. For skewed option and digital returns, the median is not the
expectation.

This is material in the actual 252-transition panel used by the fixture:

```text
norm(sample instrument mean)                 0.00197036
norm(componentwise instrument median)        0.01766456
norm(mean minus median)                       0.01779398
ATM 1D put target sample mean                 0.09455636
ATM 1D put target median                      0.50649268
maximum target absolute mean/median gap       0.41193632
```

The drift screen is not calibrated to its center either. It projects the
componentwise median into covariance eigenvectors but assigns standard errors
`sqrt(eigenvalue / n)`, the iid sample-mean formula. Even under a Gaussian law,
the asymptotic standard error of a scalar median is larger by
`sqrt(pi / 2)`; after a non-rotation-equivariant componentwise median, Tyler
scatter, and daily serial dependence, the implemented quantity is not a valid
Gaussian `z` score. The fixture retains 85 of 163 directions and its zero-claim
policy has gross exposure 10,269.9, so this is not a harmless naming issue.

**Required correction:** estimate the expectation separately from the robust
scatter. Use a mean-targeting estimator such as a sample mean with robust/HAC
standard errors, a Huber mean, or median-of-means. Apply the screen to observed
eigenportfolio returns themselves, with a standard error appropriate to that
mean estimator and time dependence. Keep the marginal median/MAD only as
robust centering/scaling inputs to the Tyler shape if desired. Regression tests
must distinguish a skewed distribution's mean from its median.

### 2. The hedge depends on unrelated targets

One Tyler estimate is fitted to `[all instruments, all 20 targets]`. Tyler row
weights and the automatic shrinkage both depend on the full dimension and every
row's full vector. Consequently, the instrument covariance and the
instrument/claim cross-covariance for one claim change when another target is
added or removed. A claim-specific hedge should not depend on the arbitrary
benchmark target catalog.

On the fixture's actual 252-transition panel, fitting the ATM 1D put alone
rather than beside the other 19 targets changes:

```text
shrinkage, 20 targets / 1 target              0.420690 / 0.394231
action L2 norm, 20 targets / 1 target         157.949 / 385.779
L2 norm of action difference                  326.750
difference relative to 20-target action       206.87%
reported robust residual SD                   0.092852 / 0.243288
```

**Required correction:** make each target fit invariant to the presence of
other targets. The simplest valid implementation is a separate robust joint
fit of `[X, Y_j]` for each target. A more efficient implementation may estimate
the instrument scatter once and estimate target cross-scatter with fixed
instrument-derived row weights, provided it preserves a positive-semidefinite
joint block and is tested for target-catalog invariance.

### 3. Mean-volatility is diagnosed as unbounded but is not a usable baseline

The implemented objective and IRLS stationarity are mathematically consistent:

```text
-E[Y + X a] + kappa * SD(Y + X a) + c'|a|.
```

Its recession function is homogeneous. It is bounded only when, for every
direction `a`,

```text
mu' a <= kappa * sqrt(a' Sigma a) + c'|a|.
```

The implementation correctly treats cap hits as non-convergence rather than as
a valid finite optimum. The synthetic unit test exercises that behavior.
However, an actual-data run for the ATM 1D put with 252 observations,
`kappa=1`, and one target produces:

```text
converged                                      false
claim plus zero-policy cap hits                326
zero-policy gross exposure                     163,000,000
final claim-minus-zero gross exposure          0
claim-policy coordinate iterations             97,617
zero-policy coordinate iterations              133,546
elapsed CPU time                               87.49 seconds
```

Both direct policies hit all 163 coordinate caps and cancel to a zero reported
pure hedge. The failure flag is correct, but this does not satisfy the Phase-3
requirement to support mean-volatility, and the default standalone smoke covers
only mean-variance.

**Required correction:** repair the mean estimator first, add an explicit
recession/boundedness check, and do not emit a tradable differenced action when
either direct policy is unbounded. Add bounded and unbounded actual-objective
fixtures, including the required `kappa` values. A passing actual-data
mean-volatility smoke is required before this gate can pass.

## Passed quantitative checks

### Information timing and coordinates

- The rolling panel uses 252 eligible origins strictly before the decision
  date. The last historical return may end at the decision-date close, but no
  later endpoint enters fitting.
- Current quotes contribute only predictable coordinate geometry and current
  cost weights. The evaluation transition is not passed to the fitter.
- Every current option is replayed at the same initial business DTE, NMS, and
  option side. The historical fixed cash strike is reconstructed from that
  date's interpolated expiry ATM volatility and the exact stored
  `book_selection_epsilon_atm_vol`.
- Daily target construction uses the same stored epsilon convention. This
  removes the prior mismatch between public-book NMS and target/replay NMS.

### Units and accounting

- Spot is the discounted forward-adjusted one-session gain in origin spot-one
  units.
- Vanilla origin values use `DF * (F/S) * C/(DF F)`. Expiring options settle
  with the one-session discount factor; unfinished fixed-cash-strike contracts
  are remapped to endpoint `K/F`, marked, scaled by endpoint spot/forward, and
  discounted back to the origin.
- Digital cash values correctly omit the spot scaling used by homogeneous
  vanilla prices. Calls use `>` and puts use `<=` at settlement.
- Mean-variance signs are correct for a long-claim residual `Y + X a`, and the
  reported objective contains the full claim variance term.
- The claim-minus-zero action is implemented and the final action is evaluated
  with its own risk and transaction-cost objective.
- Realized accounting charges one basis point on spot entry and liquidation,
  opening option half-spreads, no expiry spread, and actual endpoint spreads on
  unfinished options. Optimization deliberately uses the predictable proxies
  of two basis points for spot and twice current half-spread for unfinished
  options; this approximation is clearly reported and is non-blocking.

### Robust scatter and numerical diagnostics

- The regularized Tyler shape remains positive definite in the reviewed
  high-dimensional case and retains cross-covariances.
- The automatic shrinkage is recomputed from each rolling sample rather than
  fixed from 2022.
- Mean-volatility cap hits and direct-policy non-convergence propagate to the
  final result; they are not silently labelled successful.

## Validation evidence

- 54 focused tests under `tests/quantdata/hedging` and
  `tests/quantdata/sdlv/daily/hedging` pass.
- The default mean-variance standalone fixture passes in 24.45 seconds on CPU
  with peak RSS about 1.21 GiB.
- The fixture uses 252 transitions from 2022-01-03 through an endpoint of
  2023-01-17, then evaluates 2023-01-17 to 2023-01-18 strictly out of sample.
- The separate actual-data mean-volatility diagnostic above completes in 87.49
  seconds and correctly returns a failed result.

## Non-blocking follow-ups

1. Update the model README statement that the fitter receives only a current
   option half-spread. The full-book fitter receives a predictable cost vector,
   current relative coordinates, and the historical panel; it still receives
   no future-bearing evaluation return.
2. Update the plan status, which still says the full same-contract universe and
   Phase-3 objectives are open. They are implemented but have not passed this
   quantitative gate.
3. Add a fixture that regression-protects all action coordinates or robust
   aggregate fingerprints. The current fixture records only three actions from
   a dense 106-option ATM-call hedge.
4. Report sparsity, gross exposure, direct-policy exposure, and cancellation
   ratios in production diagnostics. These are important stability indicators
   for claim-minus-zero policies.
