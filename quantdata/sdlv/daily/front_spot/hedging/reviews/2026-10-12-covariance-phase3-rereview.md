# Independent quant re-review: robust covariance Phase 3

## Reviewed state

- Review date: 2026-10-12
- Base Git revision: `ed31c4b10ee29eba550b6000b20c5f2cc467513a`
- Source state: corrected uncommitted hedging worktree on `main`
- Prior review: `2026-10-12-covariance-phase3.md`
- Scope: common robust objectives, covariance return construction,
  configuration/solver/adapter/smoke, fixtures, diagnostics, documentation, and
  focused tests
- Reviewer: independent `quant_dataset_review` agent

## Disposition

The corrected Phase-3 robust covariance gate **PASSES**. The three prior
blockers are resolved:

1. expected returns are estimated from sample means and kept separate from the
   Tyler marginal center;
2. every target block and hedge is invariant to unrelated targets while the
   constructed joint scatter is positive definite; and
3. mean-volatility boundedness is checked from the exact recession condition
   before optimization, with no tradable action emitted when it fails.

Both the mean-variance and `kappa=1` mean-volatility full 20-target books on
2023-01-17 converge without cap hits.

## Mean estimation and screening

- The instrument Tyler fit now uses instruments only. Its componentwise median
  remains a robust location used for scatter construction, not an estimate of
  expected return.
- The raw drift estimate is the coordinatewise sample mean. The claim mean is
  likewise the sample mean of that claim's historical return.
- Instrument returns are projected into the robust covariance eigenbasis.
  Newey--West standard errors are calculated directly from the resulting
  eigenportfolio return histories with a Bartlett kernel, then the projected
  sample means are soft-thresholded at `sqrt(2 log p)` unless configured
  otherwise.
- The automatic lag rule is recomputed in each window. For 252 observations it
  selects four lags.

On the actual first out-of-sample window, the raw instrument sample-mean norm is
`0.00197036`, rather than the old componentwise-median norm `0.01766456`. The
screen has threshold `3.191786`, maximum absolute mean statistic `3.855979`,
and retains 3 of 163 eigenportfolio directions. After spread costs, the fitted
zero-claim policy is exactly zero in both reviewed objectives.

A skewed synthetic regression explicitly verifies that a distribution with
zero median and nonzero mean uses the mean. This closes the prior semantic and
statistical mismatch.

## Target blocks and catalog invariance

The instrument scatter and its Tyler row geometry are fixed before any target
is processed. For each target, a separate fixed-weight robust regression gives
coefficient `beta`; the implementation sets

```text
cross covariance = Sigma * beta,
target variance  = beta' Sigma beta + residual variance,
```

where the robust residual variance is strictly positive. The joint block's
Schur complement is therefore that residual variance, so positive definiteness
follows from positive definiteness of `Sigma`.

The independent actual-panel check over all 20 targets found:

```text
minimum Schur complement                       3.22996e-4
minimum joint-block eigenvalue                  1.72821e-9
```

For the actual ATM one-day call, fitting it alone or beside the other 19
targets gives exactly identical shrinkage, objective, and action; the maximum
absolute action difference is zero. The synthetic catalog-invariance
regression independently verifies the same property. The earlier 206.9%
catalog-induced action change is eliminated.

## Mean-volatility recession condition

For direct-policy action `a`, screened instrument mean `mu`, robust scatter
`Sigma`, and linear transaction-cost vector `c`, the recession function is

```text
-mu' a + kappa * sqrt(a' Sigma a) + c'|a|.
```

The objective is bounded exactly when `mu` belongs to the sum of the cost box
`[-c,c]` and the covariance dual ellipsoid of radius `kappa`. Equivalently, the
implementation computes

```text
minimize over z in [-c,c]
    sqrt((mu-z)' Sigma^{-1} (mu-z))
```

and compares that distance with `kappa`. This is the correct dual-norm
condition. A two-coordinate analytical check with diagonal covariance gives
the exact expected ratio `0.75`.

If the condition fails, both direct policies are marked unbounded, their
actions remain zero placeholders, the pure action is `None`, and the benchmark
adapter raises rather than emitting a tradable result. The synthetic unbounded
regression verifies that optimization is not attempted and no cap-dependent
hedge is returned.

For the actual 2023-01-17 book at `kappa=1`, the screened mean lies inside the
transaction-cost box, so the recession ratio is zero. All 20 targets are
bounded and converge with zero cap hits. The largest claim-policy iteration
count is 8,563, maximum pure gross exposure is 629.294, and maximum absolute
action is 88.7794. The reviewed full-book run completes in 40.35 seconds on a
warm CPU process; an independent cold invocation took 70.55 seconds.

Supplemental retained evidence also covers required `kappa` values
`0.1, 1, 2, 3`: all 20 targets are bounded and converged without cap hits for
each value.

## Mean-variance real-book result

The committed standalone fixture passes. An independent full-book run on
2023-01-17 reports:

```text
targets                                         20
all converged                                   true
total cap hits                                  0
maximum claim-policy iterations                 3,785
maximum pure gross exposure                     707.816
maximum absolute action                         88.2670
retained mean directions                        3
```

The standalone cold run takes 41.09 seconds on CPU with peak RSS about 1.20
GiB. Its ATM one-day call has no zero-policy exposure, 53 nonzero option
actions, pure gross exposure 705.299, and maximum absolute action 56.3515.

## Other passed checks

- The 252 historical transitions remain strictly prior to the evaluation
  return. The final training endpoint may equal the decision-date close but
  never follows it.
- Same-DTE, same-NMS, same-side replay continues to use each historical date's
  ATM term structure and exact stored NMS epsilon.
- Spot, option, and cash-digital returns remain in discounted origin-PV
  spot-one units with correct fixed-cash-strike remapping and settlement.
- Mean-variance and mean-volatility signs, full claim-risk term, linear-cost
  penalty, claim-minus-zero policy, and cap propagation remain correct.
- Diagnostics now expose the HAC lag, maximum mean statistic, boundedness,
  recession ratio, claim/zero/pure gross exposures, maximum action, sparsity,
  and cancellation ratio.
- Configuration version is bumped and the model README accurately describes
  mean semantics, target-block construction, cost proxies, and boundedness.

## Validation evidence

- 61 focused tests under `tests/quantdata/hedging` and
  `tests/quantdata/sdlv/daily/hedging` pass.
- The committed mean-variance standalone fixture passes.
- Independent full-book mean-variance and `kappa=1` mean-volatility runs both
  pass for all 20 targets with zero cap hits.
- Actual target-catalog invariance and positive-definite block checks pass.

## Non-blocking follow-ups

1. Add a committed mean-volatility CLI/fixture, ideally a compact summary over
   all 20 targets and required `kappa` values. The present committed standalone
   fixture exercises mean-variance; mean-volatility is covered by unit tests
   and retained review evidence rather than a standard fixture command.
2. Make the HAC autocovariance explicitly aware of breaks between retained
   one-session transitions. In the reviewed window the valid contiguous pair
   counts for lags 1--4 are `247, 242, 237, 232`; omitting cross-gap products
   leaves the retained-direction count at 3 and changes the maximum statistic
   only from `3.855979` to `3.860422`, so this does not block the gate.
3. Update the implementation-plan status paragraph, which still says the full
   Phase-3 universe and objectives remain open.
4. Expand the committed smoke fingerprint beyond three option coordinates.
   Aggregate exposure diagnostics now protect much more of the action, but a
   broader deterministic action fingerprint would detect more localized
   regressions.
