# FLEX Phase 4: front-implied spot and the first weekly Q line

## Scope

This first target-measure experiment deliberately stops after two maturities.
It uses the accepted jointly trained Hierarchical-200 decoder and weekly P law,
not the earlier sequential decoder/PCA pipeline.  The decoder, SANOS geometry,
P transition, and historic states are frozen.  The experiment:

1. verifies that sampling spot from the current front SANOS inverse CDF
   reproduces current one-week calls; and
2. learns only the first conditional state kernel under Q so that current
   two-week fixed-cash-strike calls equal expected rolled one-week calls.

No later expiry line and no generic/non-Gaussian measure change is fitted in
this experiment.  Training states end in 2024.  The independently quote-fitted
2025 `h_t` states are used only for validation.  No 2026 data is read.

## Timing and martingale equations

Let `h_t=(h_t^1,...,h_t^14)` be the unconstrained, full-real-support Strong-200
state and let `C_t^j(k)` be a normalized SANOS call with `j=1` denoting the
front weekly expiry.  Rates and forwards remain absent.  The current front
surface supplies the Q spot rank:

```text
z_s ~ Normal(0,1)
U   = Phi(z_s)
R   = S[t+1]/S[t] = F^{-1}_{front,t}(U).
```

The inverse CDF is the complete SANOS front distribution: the discrete DLV
martingale mixture convolved with its mean-one lognormal backbone.  Therefore

```text
C_t^1(k) = E_Q[(R-k)^+].
```

The next restriction uses a fixed cash strike.  In spot-normalized units the
same option has next-week normalized strike `k/R`, hence

```text
C_t^2(k) = E_Q[R C_{t+1}^1(k/R) | h_t].
```

The final new state entering the rolled front price is `h_{t+1}^1`, not
`h_{t+1}^2`.  Because the decoder is triangular, the current 2W target depends
on current `(h_t^1,h_t^2)`, while the rolled 1W price depends only on next
`h_{t+1}^1` and realized spot.

## Normal Q1 kernel

The joint Hierarchical-200 P checkpoint contains a 200-dimensional stationary
surface law but no separate spot equation.  Before constructing Q, regress the
realized weekly log-spot return under P on the first block of whitened weekly
surface innovations:

```text
Delta log S = drift_P + beta_P' z_P^1 + sigma_perp z_perp.
```

This fit uses only dated states and spot returns through 2024.  It determines
the complete Gaussian spot/surface covariance.  A Cholesky factor with spot
ordered first then supplies a standard-normal **correlated spot rank** and the
conditional surface innovation.  Implied sampling applies `Phi` and the front
SANOS inverse CDF to that P-fitted spot rank.  It therefore preserves the
learned P Gaussian copula while replacing the spot marginal.  Q1 never alters
the spot rank itself.

Q1 changes only the first 23-dimensional ordered conditional surface-normal
block:

```text
context = (h_t^1, h_t^2, log R)
z_Q^1   = mean_theta(context)
          + diag(exp(log_scale_theta(context))) L_theta z_P^1.
```

`L_theta` is lower triangular with a strictly positive exponential diagonal.
Consequently the conditional Gaussian has full support and remains equivalent
to the P proposal.  All remaining standard-normal blocks are unchanged.  The
complete frozen P innovation Cholesky then maps the ordered normals into the
next 200-state vector.  This also propagates the Q1 shock into later state
blocks without allowing later shocks to affect `h_{t+1}^1`.

The regularizer is the analytic conditional Gaussian
`KL(Q1 || P1)` in these ordered standard-normal coordinates, per first-block
coordinate.  It is zero at initialization.

## Live Monte Carlo objective

Each optimizer update chooses origin dates with equal probability.  For every
origin it generates fresh Torch normals in exact `z,-z` antithetic pairs,
constructs spot and the Q state inside the Torch graph, decodes the next front
surface through SANOS, and differentiates the squared conditional moment:

```text
mean_paths[R C_{t+1}^1(k/R) - C_t^2(k)].
```

The loss is the mean squared conditional moment across origin dates and 11
fixed normalized strikes with log-moneyness from `-0.30` to `0.30`, plus a
small conditional KL penalty.  Paths are never cached or pregenerated.  A
separate fixed-seed antithetic audit estimates Monte Carlo standard errors;
the training batches always receive new draws.

The network width/depth, strike set, learning rate, KL weight, path counts,
and epoch count are ad-hoc feasibility choices and must not be treated as a
final Q specification.

## Results

The immutable completion report will populate this section after both the
training convergence gate and full 2025 antithetic audit pass.
