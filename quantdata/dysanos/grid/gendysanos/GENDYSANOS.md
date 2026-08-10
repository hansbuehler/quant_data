# Generative DYSANOS and PCA-IV benchmark

## Objective

Generate coherent multi-time option markets from both the trained ML-SANOS
decoder and a competitive implied-volatility PCA decoder, and test them for
detectable dynamic arbitrage. Each generator has three ordered parts:

1. fit the implied-volatility parameters
   $h_t=(h_t^1,\ldots,h_t^{n_h})$ as an autonomous stationary process;
2. fit log-spot $s_t=\log S_t$ conditionally on the fitted surface process;
3. importance sample spot, without changing the physical law, so the simulated
   spot range adequately covers every option strike and return boundary.

The implementation may continue to store the extended state in the existing
order $xstate=(s,h)$. The symbol $h$ never includes spot.

SANOS guarantees that every decoded cross-section is smooth and free of static
arbitrage. This does not by itself guarantee absence of dynamic arbitrage
between generated trading times. The PCA-IV decoder has no corresponding
structural guarantee.

## Competing generators

Use the following generator axis:

$$
g\in\{\mathrm{DYSANOS},\mathrm{PCA\mbox{-}IV}\}.
$$

The DYSANOS case uses the trained ML-SANOS decoder in
`quantdata/dysanos/grid`. The PCA-IV case uses the regular-grid implied-
volatility PCA market simulator specified in
`quantdata/dysanos/implied/IMPLIED.md`: construct the PCA from
`market["mid_vols"]`, model its retained log-implied-volatility scores as the
surface state $h$, fit spot only after the autonomous surface process, and
decode with the documented unrepaired total-variance interpolation. Do not
project, reject, or repair PCA-IV surfaces before measuring their arbitrage
properties.

Both generators must use the same market dates, NMS grid, contract definitions,
trading-time construction, return definitions, cost rules, importance-sampling
coverage criteria, and no-arbitrage diagnostics. Use matched random seeds and
matched experiment configurations where their state dimensions permit it.
Their fitted laws and decoded paths are different and therefore must not be
treated as pathwise identical samples.

It is a pre-specified empirical hypothesis that PCA-IV produces more static
violations and more dynamic-arbitrage candidates than DYSANOS. It is not an
assumption in candidate classification and must not alter tolerances,
hyperparameters, stopping rules, or reporting.

## Operating perimeter

1. Select the DYSANOS or PCA-IV decoder explicitly; do not infer it from array
   shape or cache contents.
2. Use only Gaussian surface innovations: $bandwidth=1$.
3. Use the surface-first fit and the exact matrix-exponential transition:
   $surface\_first=True$ and $use\_matrix\_exp=True$.
4. Simulate one coherent daily state and spot path for each Monte Carlo
   realization. Never draw a separate spot by option, strike, expiry, or
   trading time.
5. Generate paths in blocks. The one-million-path experiment must not create
   a second full in-memory copy or copy the full universe to a GPU.
6. Include the complete configuration and cache version in every cache key.
   Change the version whenever return timing, cost, delta, or importance
   sampling semantics change.
7. Fail immediately on unexpected NaN or infinity in states, prices, deltas,
   returns, costs, likelihoods, or diagnostics.

Keep one shared coherent-path, return, cost, importance-sampling, and diagnostic
implementation around the two fitted surface decoders. Remove legacy generator
and measure-change branches rather than preserving them as active alternatives.

## Surface process

First fit $h$ without current or lagged spot. In column-vector notation,

$$
dh_t=\kappa(m-h_t)\,dt+\Sigma\,dW_t^h,
$$

where $W^h\in\mathbb R^{n_\alpha}$. Fit the affine drift to the historical
one-day transitions and apply PCA to the drift residuals.

Require every eigenvalue of $\kappa$ to have strictly positive real part.
Then $h$ has invariant mean and covariance

$$
h_\infty\sim N(m,P),
\qquad
\kappa P+P\kappa^\top=\Sigma\Sigma^\top.
$$

For a step of length $\Delta$, simulate

$$
h_{t+\Delta}=A_\Delta h_t+(I-A_\Delta)m+\eta_\Delta,
\qquad
\eta_\Delta\sim N(0,Q_\Delta),
$$

with

$$
\begin{aligned}
A_\Delta&=e^{-\kappa\Delta},\\
Q_\Delta&=\int_0^\Delta
 e^{-\kappa u}\Sigma\Sigma^\top
 e^{-\kappa^\top u}\,du.
\end{aligned}
$$

Calculate these quantities using the stable matrix-exponential implementation.
Do not replace the reported model by an Euler transition.

Report the retained factor count, all eigenvalues of $\kappa$, invariant
mean and covariance checks, and one-day residual covariance reconstruction.
The factor count is a driver configuration, not a separate market definition.

## Spot conditional on the surface

Only after the surface model is fixed, fit

$$
ds_t=\mu\,dt+\beta^\top dW_t^h+\varsigma\,dW_t^s,
$$

where $W^s$ is a scalar Brownian motion independent of $W^h$. Regress the
log-spot innovation on the retained surface-factor innovations and fit the
orthogonal residual variance. This step must not change the fitted drift or
diffusion of $h$.

The implemented baseline uses constant $\mu$, $\beta$, and $\varsigma$.
Record the spot drift, total spot variance
$\|\beta\|^2+\varsigma^2$, surface-explained spot-innovation variance, and
orthogonal residual variance. Verify the reconstructed joint transition
covariance against the empirical residual covariance.

For weekly markets, simulate all intervening daily states and spots. Weekly
trading times are a subsample of the daily path, not independently generated
five-day observations.

## Trading times and fixed contracts

Index Monte Carlo realizations by $r$, trading times by $m$, expiries by $j$,
and strikes by $i$. Generate ten trading timestamps

$$
t_m=t_0+m q,\qquad m=0,\ldots,9,
$$

where $q=1$ business day for the daily market and $q=5$ business days for the
weekly market. Decode the complete option grid at all ten timestamps.

At $t_m$, option $(j,i)$ has time-to-expiry $T_j$, relative strike
$K_{r,m,j,i}$, calendar expiry

$$
\tau_{m,j}=t_m+T_j,
$$

and fixed cash strike

$$
k_{r,m,j,i}=S_{r,t_m}K_{r,m,j,i}.
$$

Once the trade is defined, keep $(\tau_{m,j},k_{r,m,j,i})$ fixed. A later
option value must be evaluated at its remaining time-to-expiry and its new
relative strike $k_{r,m,j,i}/S_{r,u}$.

Simulate the daily path through

$$
t_9+\max_j T_j.
$$

This supplies the ten trading surfaces, all intermediate expiry spots, and
all terminal option payoffs from one path.

## Four base markets

The four base markets are the Cartesian product

$$
\{\mathrm{daily},\mathrm{weekly}\}
\times
\{\mathrm{expiry},\mathrm{incremental}\}.
$$

Daily and weekly refer only to the spacing of trading times. Expiry and
incremental specify the return horizon.

### Expiry returns

An option entered at any of the ten trading times is held to its own expiry:

$$
dC^{\mathrm{expiry}}_{r,m,j,i}
=
\left(S_{r,\tau_{m,j}}-k_{r,m,j,i}\right)^+
-C_{r,t_m}(\tau_{m,j},k_{r,m,j,i}).
$$

The matching spot return is expiry-specific:

$$
dS^{\mathrm{expiry}}_{r,m,j}
=S_{r,\tau_{m,j}}-S_{r,t_m}.
$$

There are ten entry cohorts. A spot return with one option expiry must never
be substituted for another option expiry.

### Incremental returns

Incremental returns run between consecutive trading times, so ten trading
timestamps give nine return intervals. Set $u_m=t_{m+1}$ for
$m=0,\ldots,8$. The endpoint option value is

$$
V_{r,m+1,j,i}
=
\begin{cases}
C_{r,u_m}(\tau_{m,j},k_{r,m,j,i}),
    & u_m<\tau_{m,j},\\
\left(S_{r,\tau_{m,j}}-k_{r,m,j,i}\right)^+,
    & u_m\geq\tau_{m,j}.
\end{cases}
$$

Thus

$$
dC^{\mathrm{incremental}}_{r,m,j,i}
=V_{r,m+1,j,i}
-C_{r,t_m}(\tau_{m,j},k_{r,m,j,i}),
$$

and the spot return is always measured between trading times:

$$
dS^{\mathrm{incremental}}_{r,m}
=S_{r,t_{m+1}}-S_{r,t_m}.
$$

If a weekly interval contains an option expiry, compute the payoff using spot
at the exact daily expiry index. With discount factors equal to one, that cash
payoff is carried unchanged to the next trading time. Close the option's delta
hedge at the same expiry. The separately traded spot instrument remains open
for the full trading-time interval.

Store the option settlement and delta-hedge endpoints explicitly. They are
both the exact expiry spot for an expired option and the next trading-time
spot for an option which remains alive.

## Raw and delta-hedged variants

For each base market, generate both raw and delta-hedged option returns. Let
$\Delta_{r,m,j,i}$ be the cash-call delta at entry.

For raw options,

$$
dH^{\mathrm{raw}}=dC.
$$

For delta-hedged options,

$$
dH^{\Delta}
=dC-\Delta\,dS^{\mathrm{option}},
$$

where

$$
dS^{\mathrm{option}}
=
\begin{cases}
dS^{\mathrm{expiry}}_{r,m,j},
    & \text{expiry market},\\
dS^{\mathrm{expiry}}_{r,m,j},
    & \text{incremental market and }\tau_{m,j}\leq t_{m+1},\\
dS^{\mathrm{incremental}}_{r,m},
    & \text{incremental market and }\tau_{m,j}>t_{m+1}.
\end{cases}
$$

Retain the raw spot returns as separate instruments where required. Do not
replace the option-specific expiry hedge by the next-trading-time spot return
in an expiry market.

## Transaction costs

Run every raw and delta-hedged variant with

$$
\gamma\in\{0,\ 10^{-4},\ 10^{-3}\},
$$

corresponding to zero, one basis point, and ten basis points.

For one unit of a raw option, the gross cost notional is

$$
G^{\mathrm{raw}}_C=|C_{r,t_m}|.
$$

For one unit of a raw spot trade,

$$
G_S=|S_{r,t_m}|.
$$

For one unit of a delta-hedged option,

$$
G^\Delta_C
=|C_{r,t_m}|+|\Delta_{r,m,j,i}S_{r,t_m}|.
$$

The option and hedge legs are grossed and must never be netted. For a signed
position $a$, the net one-trade P&L is

$$
X(a)=a\,dH-\gamma|a|G.
$$

The long and short unit P&Ls used in diagnostics are therefore

$$
X^+=dH-\gamma G,
\qquad
X^-=-dH-\gamma G.
$$

Costs are charged when a new cohort trade is initiated. Do not net costs
across different cash strikes, calendar expiries, trading times, option legs,
or embedded delta hedges. If a separate experiment explicitly models
liquidation or rebalancing of the same contract, add the corresponding
turnover cost and label that experiment separately.

For each generator, the required matrix has

$$
2\times2\times2\times3=24
$$

cases: two trading frequencies, two return horizons, raw or delta hedged, and
three cost levels. Across DYSANOS and PCA-IV there are 48 reported cases.

Expose these dimensions directly in the versioned experiment configuration:

- $generator\in\{\mathrm{dysanos},\mathrm{pca\_iv}\}$;
- $trading\_frequency\in\{\mathrm{daily},\mathrm{weekly}\}$;
- $return\_horizon\in\{\mathrm{expiry},\mathrm{incremental}\}$;
- $delta\_hedged\in\{\mathrm{False},\mathrm{True}\}$; and
- $cost\_rate\in\{0,10^{-4},10^{-3}\}$.

Also store $num\_trading\_times=10$. Do not encode these choices through
legacy combined mode names or infer them from array shapes. Include all four
market flags and the generator in cache and report identities.

## Spot importance sampling

Importance sampling changes the proposal, not the fitted physical law. Use a
mixture

$$
q=\alpha p+\frac{1-\alpha}{M}\sum_{\ell=1}^M q_\ell,
$$

where $p$ is the fitted path law. Each $q_\ell$ broadens a selected future
log-spot coordinate while preserving the surface path and the exact Gaussian
conditional law of the remaining spot coordinates. Store

$$
w_r=\frac{p(X_r)}{q(X_r)}
$$

for every complete path.

Choose the target coordinates and proposal widths from the contracts actually
generated. For each option in the sampling range, require at least 1,000
proposal observations on both sides of its own cash strike at the relevant
expiry. Check this per trading time, expiry, and strike rather than only
against a global minimum or maximum strike.

For a delta-hedged expiry return with $0<\Delta<1$, also check both sides of
its P&L break-even levels

$$
L=S-C/\Delta,
\qquad
U=(k+C-\Delta S)/(1-\Delta).
$$

Require at least 1,000 proposal observations in both delta-hedged P&L sign
regions. For incremental returns, strike coverage is necessary but not
sufficient because the next surface also changes; apply the return-sign tests
below after repricing.

The likelihood ratio does not depend on whether the reported return is raw or
delta hedged: after a proposal $q$ is fixed, the same path weight $p/q$ applies
to every measurable payoff on that path. Proposal *design* does depend on the
return definition. Delta hedging usually removes much of the central
first-order spot variation, but an expiry delta-hedged call has negative
central P&L and positive convex tails. Smaller central variance is therefore
not a reason to narrow the spot proposal.

For every generated contract, form the union of:

- its cash strike;
- every raw long and short cost-adjusted root of $X^+=0$ or $X^-=0$;
- every delta-hedged long and short cost-adjusted root of $X^+=0$ or $X^-=0$;
  and
- any additional endpoint sign boundary found by repricing an incremental
  return.

Calibrate one common proposal per generator, path count, and trading frequency
to cover this union for both return horizons, both hedge variants, and all
three cost rates. If a piecewise branch has no finite root, record that fact
rather than inventing a boundary. Reuse the resulting paths and weights across
raw versus delta-hedged returns and across costs. This matched design makes the
comparison less noisy and prevents a supposedly easier delta-hedged case from
receiving poorer tail coverage.

All probabilities, moments, conditional statistics, and strategy objectives
under the physical law use $w_r$. Query states for conditional diagnostics
must come from the direct-$p$ mixture component. Report

- $E_q[w]$ and its path-level standard error;
- $\min w$, $\max w$, and quantiles of $w$;
- effective sample size
  $(\sum_r w_r)^2/\sum_r w_r^2$;
- mixture-component counts;
- direct-$p$ versus weighted-$q$ means and variances of log spot at every
  target time; and
- direct-$p$ versus weighted-$q$ realized log-variance moments.

Do not launch a reported arbitrage test if likelihood normalization, moment
recovery, or tail coverage fails.

## Numerical tolerance

Keep three concepts separate:

1. $c_{r,t,\omega}=\gamma G_{r,t,\omega}$ is the economic transaction
   cost;
2. $\eta_{r,t,\omega}$ is a numerical floating-point error band; and
3. Monte Carlo sampling error is measured by confidence intervals.

Transaction cost is part of the payoff, not part of the statistical or
numerical tolerance. For a pre-cost instrument return $R_{r,t,\omega}$ define

$$
X^+_{r,t,\omega}=R_{r,t,\omega}-c_{r,t,\omega},
\qquad
X^-_{r,t,\omega}=-R_{r,t,\omega}-c_{r,t,\omega}.
$$

The same numerical band applies to the long and short directions because
they use the same primitive arithmetic.

Construct the band from unreduced operands, not from the possibly
cancellation-dominated final return. For an expiry option use

$$
A^C_{r,t,\omega}
=|S_{r,\tau_\omega}|+|k_{r,t,\omega}|+|C_{r,t,\omega}|.
$$

For an unexpired incremental option use

$$
A^C_{r,t,\omega}
=|C_{r,t+1,\omega}|+|C_{r,t,\omega}|.
$$

If the incremental option expires before the next trading time, use the
expiry scale above. For a delta-hedged option add the unreduced hedge scale

$$
A^\Delta_{r,t,\omega}
=A^C_{r,t,\omega}
+|\Delta_{r,t,\omega}|
 \left(|S_{r,\mathrm{hedge\ end}}|+|S_{r,t}|\right).
$$

For spot use

$$
A^S_{r,t}=|S_{r,\mathrm{end}}|+|S_{r,t}|.
$$

Finally add the magnitude of the computed cost to the relevant scale because
the subtraction of $c$ is another floating-point operation. With
$A^{net}=A+c$, define

$$
\eta_{r,t,\omega}
=32\,\epsilon_{32}\max\{A^{net}_{r,t,\omega},
                         \operatorname{tiny}_{32}\},
\qquad
\epsilon_{32}=\operatorname{finfo}(\mathrm{float32}).\mathrm{eps}.
$$

Thus transaction cost affects the test primarily by moving the economic P&L
from $\pm R$ to $\pm R-c$. It affects $\eta$ only through the order
$\epsilon_{32}c$ rounding error of subtracting that cost. It is never added
to $\eta$ as an economic buffer.

Do not impose an arbitrary cash floor such as one dollar: that would make the
tolerance economically material for cheap options. A net P&L is positive if
it exceeds $\eta$, negative if it is below $-\eta$, and numerically neutral
otherwise.

For a basket $a$, define

$$
X_{r,t}(a)
=a^\top R_{r,t}-|a|^\top c_{r,t}
$$

and use the propagated band

$$
\eta_{r,t}(a)
=\sum_\omega |a_\omega|\eta_{r,t,\omega}
+32\epsilon_{32}
 \left(
   \sum_\omega|a_\omega R_{r,t,\omega}|
   +\sum_\omega|a_\omega|c_{r,t,\omega}
 \right).
$$

Recompute every reported candidate from the original path in float64 and
require it to survive the corresponding float64 band. If the decoder itself
has a measured numerical pricing error beyond arithmetic roundoff, report
that error separately and add a validated bound; do not hide it inside an
empirically chosen $\eta$.

## Per-trading-time arbitrage definition

Let $\mathcal F_t$ contain the generated history available at trading time
$t$, and let $z_t=(s_t,h_t)$. At each trading time test the complete
cross-section of spot and option returns jointly. An admissible action

$$
a_t=a_t(z_t)
$$

is $\mathcal F_t$-measurable and satisfies $\|a_t(z_t)\|_1\leq1$. Its net
return is

$$
X_{r,t}(a_t)
=a_t(z_{r,t})^\top R_{r,t}
-|a_t(z_{r,t})|^\top c_{r,t}.
$$

A per-trading-time conditional arbitrage exists if some admissible $a_t$ has

$$
X_t(a_t)\geq0\quad\mathbb P\text{-a.s.},
\qquad
\mathbb P\!\left(X_t(a_t)>0\right)>0.
$$

The action may be zero outside a state region. Consequently an arbitrage that
exists only for a positive-probability subset of states is included in this
definition. The $L^1$ constraint fixes scale but does not remove a scalable
arbitrage direction.

### Relation to the sum over trading times

An unconditional test at each $t$ is not sufficient for the sum. For example,
a return can have both signs unconditionally while its sign is known from
$z_t$; choosing the position from $z_t$ then creates a conditional arbitrage.

For a standard incremental discrete-time market, exact conditional
no-arbitrage of the full one-period cross-section at every history implies
no dynamic arbitrage for the self-financing sum

$$
\Pi_r=\sum_tX_{r,t}(a_t).
$$

This is the usual local-to-global result and requires the conditional
statement for every predictable portfolio, not merely an unconditional sign
test for every instrument.

Expiry returns overlap in calendar time and are not consecutive wealth
increments. Absence of arbitrage in every expiry cohort separately therefore
does not, without additional assumptions, prove absence of arbitrage in
combinations of overlapping cohorts. The primary reported result for all four
markets is accordingly:

> no per-trading-time conditional arbitrage detected within the tested
> strategy class and statistical resolution.

A test of $\sum_tX_t$ may be added as a separate stronger experiment, but it
must not be inferred from the unconditional screens below.

Finite Monte Carlo cannot verify an almost-sure condition for every state and
every predictable portfolio. Never describe a passed empirical test as a
mathematical proof of no arbitrage.

## Per-trading-time no-arbitrage diagnostics

Run every diagnostic separately for all 48 cases.

Before testing strategies, verify every decoded surface against the static
call-price bounds, strike monotonicity, strike convexity, and calendar
conditions at the numerical tolerance above. A DYSANOS violation is a decoder
or numerical failure and invalidates that run. A PCA-IV violation is a measured
benchmark outcome: report it, classify the generated market as not free of
static arbitrage at that resolution, and continue the dynamic diagnostics
without repairing the surface.

For PCA-IV report two resolutions separately. The dense common $7\times201$
mesh tests all static conditions. The actual state-dependent $7\times20$
traded grid tests call monotonicity, the vertical-spread slope bound, and
irregular-grid strike convexity, with weights recomputed for every surface.
In both cases the signed violation $v$ is compared with its positive numerical
band. The sum of absolute arithmetic operands defines only the roundoff scale;
negative $v$ never violates a condition, and violations are not summed across
portfolios. Replay completed PCA-IV matched caches with
`quantdata.dysanos.grid.gendysanos.run_traded_strike_static`.

Also verify fixed cash strikes, fixed calendar expiries, one common spot path,
exact return endpoints, and that every action uses only information available
at its trading time. A failure of these path, timing, or information checks
invalidates either generator's dynamic test.

### Reference and audit split

Split independent paths 80:20 into a reference set and an untouched audit
set, stratified by importance-mixture component. Keep every trading time from
one path in the same split. Fit state normalization, conditional estimators,
candidate regions, baskets, action functions, thresholds, and hyperparameters
on the reference set only. Select conditional-estimator hyperparameters by
cross-validation or cross-fitting inside the reference set. Use the audit set
once for candidate certification and estimator calibration.

A holdout is not needed for deterministic identities or for a statistic and
direction fixed before looking at the simulation. Use all paths for finiteness,
surface-arbitrage checks, timing and accounting identities, importance-weight
normalization, direct-$p$ versus weighted moment recovery, and pre-specified
unconditional summaries with confidence intervals. A holdout remains necessary
whenever the data choose a direction, basket, state region, action function,
threshold, or conditional-estimator configuration. Reporting the best of many
full-sample directions without an audit would retain selection bias even though
no density is being learned.

For importance-sampled universes, retain weights and mixture labels through
the split. Use direct-$p$ paths for unweighted query states and exact binomial
incidence intervals; use all proposal paths with weights for physical-law
moments. At 100,000 and one million total paths, the nominal audit sizes are
20,000 and 200,000 paths respectively, before restricting to a mixture
component or conditional state region. If a rare audit region is too small,
report it as unresolved; do not move audit paths into the reference fit.

### Unconditional tests: necessary screens

The unconditional tests suppress the state and test only positions that are
constant across paths at a given trading time. They are necessary screens
because constant positions are a subset of the conditional strategy class.
Passing them does not rule out an action whose direction depends on $(s_t,h_t)$.

Run the following separately for every trading time. Do not pool times before
testing.

#### Individual directions

For every instrument $\omega$, evaluate the unit long and short net P&Ls

$$
X^+_{r,t,\omega}=R_{r,t,\omega}-c_{r,t,\omega},
\qquad
X^-_{r,t,\omega}=-R_{r,t,\omega}-c_{r,t,\omega}.
$$

For each direction report:

- proposal loss, neutral, and gain counts using its observation-specific
  $\eta_{r,t,\omega}$;
- likelihood-weighted physical loss and gain probabilities;
- direct-$p$ loss and gain counts;
- mean, population standard deviation, path-level standard error, and
  mean divided by standard error; and
- the one-sided confidence bounds for loss and gain probabilities.

A direction is a finite-sample candidate when it has no observed loss below
$-\eta$ and at least one gain above $\eta$. It is not certified until it
passes the independent audit and confidence-bound requirements below.
Transaction costs enter through $X^\pm$ before sign classification. Sampling
uncertainty never enters $\eta$.

For options also report payoff/no-payoff counts, strike coverage, and, when
delta hedged, both pre-cost P&L sign regions. These are support diagnostics;
the arbitrage classification always uses the net-of-cost $X^\pm$.

#### Constant baskets

Individual signs do not detect combinations of instruments. At every trading
time also solve the state-independent feasibility/search problem

$$
\max_{\|a_t\|_1\leq1}
\widehat E\!\left[X_t(a_t)\right]
\quad\text{subject to}\quad
X_{r,t}(a_t)\geq0
\quad\text{on reference paths}.
$$

With proportional costs this is a finite-dimensional convex
piecewise-linear feasibility problem after splitting
$a_t=a_t^+-a_t^-$ with $a_t^\pm\geq0$. Use a linear or convex optimizer and
then evaluate the frozen basket on audit paths.

The reference optimizer enforces the economic non-loss condition without a
numerical relaxation. The independently replayed audit classifies losses and
gains with the propagated $\eta_{r,t}(a_t)$. Do not put $\eta$ into the split
linear-program constraints: for sufficiently cheap instruments that makes the
effective cost negative and allows simultaneous positive $a_t^+$ and $a_t^-$
to manufacture tolerance without a net position.

Report the fitted action, binding paths, minimum P&L, gain count, weighted
mean, audit confidence bounds, and optimizer feasibility tolerance.
Recompute a candidate independently in float64. Passing the individual and
constant-basket tests rules out only state-independent arbitrage in the tested
sample; it remains only a necessary condition for conditional no-arbitrage.

Pooling instruments or trading times is permitted only as a summary after all
time-resolved tests have been reported.

### Conditional tests

Run the conditional test separately at each trading time $t$. Its null object
is the conditional law

$$
\mathcal L(R_t,c_t\mid z_t=z),
\qquad z_t=(s_t,h_t),
$$

not the pooled law across all times. A conditional arbitrage candidate is a
state region $A_t$ of positive probability and an action $a_t(z)$ such that

$$
\mathbb P\!\left(
X_t(a_t)\geq0\mid z_t\in A_t
\right)=1,
\qquad
\mathbb P\!\left(
X_t(a_t)>0\mid z_t\in A_t
\right)>0.
$$

The action is set to zero outside $A_t$.

#### Local conditional support

Estimate the conditional distribution nonparametrically with nearest
neighbours in normalized $(s,h)$:

1. fit the state center and scale on reference paths only;
2. keep trading times separate;
3. use 1,024 neighbours at 100,000 paths and 4,096 neighbours at one million
   paths, all from distinct paths, so the neighbour count grows while its
   fraction of the sample shrinks;
4. for reference queries, exclude every observation from the query path;
5. use likelihood weights for importance-sampled reference paths;
6. use query states from the direct-$p$ component;
7. report the radius to the furthest neighbour and local effective sample
   size; and
8. report ordered loss and gain tails at ranks 1, 2, 10, 20, and 50.

For each query state and each instrument direction, calculate local estimates

$$
\widehat p^-_t(z)
=\frac{\sum_{r\in\mathcal N_t(z)}
 w_r\,1_{\{X_{r,t}<-\eta_{r,t}\}}}
 {\sum_{r\in\mathcal N_t(z)}w_r},
$$

$$
\widehat p^+_t(z)
=\frac{\sum_{r\in\mathcal N_t(z)}
 w_r\,1_{\{X_{r,t}>\eta_{r,t}\}}}
 {\sum_{r\in\mathcal N_t(z)}w_r}.
$$

A reference screen marks a direction only when no neighbour has a loss beyond
its own numerical band and at least one neighbour has a gain beyond its band.
Record the query state, direction, neighbour radius, raw counts, weighted
probabilities, local effective sample size, and upper confidence bound for
the unobserved loss probability. A zero local loss count is not a zero
conditional loss probability.

For audit, define the candidate region before looking at audit returns. The
default region associated with query $z$ is the closed normalized
state ball

$$
A_t(z)=
\left\{z':\|z'-z\|_2\leq\rho_t(z)\right\},
$$

where $\rho_t(z)$ is the reference distance to the final configured neighbour.
Treat
overlapping balls as separate hypotheses unless a deterministic reference-only
merging rule is specified.

Also report

$$
\widehat E[X_t\mid z],\qquad
\widehat{\operatorname{Std}}(X_t\mid z),\qquad
\frac{|\widehat E[X_t\mid z]|}
     {\widehat{\operatorname{Std}}(X_t\mid z)}.
$$

Ratios above two or three identify states requiring inspection. They do not
establish arbitrage because a positive mean can coexist with losses.

#### Local conditional baskets

Single-instrument tails remain only a necessary test and must never gate a
basket test. At every query state, independently of every individual-direction
result, solve the basket feasibility problem over all spot and option
instruments in its reference neighbourhood. Define

$$
L_t(z;a)=\min_{r\in\mathcal N_t(z)}X_{r,t}(a)
$$

and the all-basket gain metric

$$
G_t(z)=\max_{\|a_t\|_1\leq1}
\widehat E_{\mathbb P}\!\left[X_t(a_t)\mid z_t\approx z\right]
$$

subject to

$$
L_t(z;a_t)\geq0.
$$

The zero action makes the problem feasible and gives $G_t(z)\geq0$. A nonzero
reference candidate requires $G_t(z)>0$, $L_t(z;a_t)\geq0$ at the optimizer
feasibility tolerance, and at least one reference gain beyond its propagated
numerical band. Every unit long or short instrument direction is a feasible
basket, so a correct all-basket search cannot miss an individual candidate.
Store the fitted action, $L_t$, $G_t$, gain count, binding count, and its state
region. Do not refit it on audit.

On the frozen audit region report the minimum and maximum economic P\&L, loss,
neutral, and gain counts, direct-$p$ loss and gain probability bounds,
importance-weighted gain probability and mean with adjusted lower bounds,
regional effective sample size, and the propagated numerical-band replay. A
basket survives only if it has no audit loss beyond the band and both its gain
probability and mean have positive adjusted lower bounds under the direct and
importance-weighted calculations.

#### State-dependent static-spread positive control

For raw expiry returns, run an exact positive control independently of the
nonparametric conditional estimator. At every state and trading time enumerate
all call-monotonicity, vertical-spread-slope, and irregular-grid convexity
portfolios on the traded $7\times20$ grid. Include transaction costs, normalize
each action to $L^1$ norm one, and select the portfolio with the largest
state-known guaranteed terminal P\&L beyond the arithmetic band. Set the action
to zero if no portfolio qualifies.

This is a predictable state-dependent action: its strikes, weights, prices,
and costs are known at entry, while its terminal payoff inequality holds for
every future spot. It does not use realized returns to select the trade and
therefore requires no reference/audit split. Replay the terminal P\&L in
float64 and report, separately by trading time, the active-state probability,
spread type, minimum guaranteed and realized P\&L, numerical-band loss and gain
counts, direct-$p$ incidence bound, weighted mean and error, and payoff-identity
residual. The generic basket search is invalid if this positive control finds
a traded-grid violation which its reported conditional strategy classes omit.

As a complementary smoother search, fit a joint action function
$a_t(s,h)$ on reference paths at that trading time by first minimizing

$$
\widehat E_{\mathbb P}
\left[(-X_t(a_t)-\eta_t(a_t))^+\right]
$$

and then maximizing $\widehat E_{\mathbb P}[X_t(a_t)]$ subject to negligible
downside. Enforce $\|a_t(s,h)\|_1\leq1$. This is a direct per-trading-time
arbitrage search; it does not combine returns across times.

#### Statewise multi-expiry static-arbitrage LP test

At a sampled current state, fix one trading time and combine all seven
expiries in one static portfolio. Expiry~$j$ contains its matched spot leg and
20 calls. Let $a_{tj}\in\mathbb R^{21}$ and impose the single global budget

$$
\sum_{j=1}^7\|a_{tj}\|_1\leq1.
$$

Carry every expiry cashflow to a common value date; the current experiments
use zero interest. For possible expiry spots $y=(y_1,\ldots,y_7)$, define

$$
\Pi_t(y;a_t)
=\sum_{j=1}^7
\left\{a_{tj}^{\mathsf T}\Phi_{tj}(y_j)
-|a_{tj}|^{\mathsf T}c_{tj}\right\}.
$$

The fitted diffusion has full support on $(0,\infty)^7$, so

$$
\inf_{y\in[0,\infty)^7}\Pi_t(y;a_t)
=\sum_{j=1}^7\inf_{y_j\geq0}
\left\{a_{tj}^{\mathsf T}\Phi_{tj}(y_j)
-|a_{tj}|^{\mathsf T}c_{tj}\right\}.
$$

Each summand is piecewise linear. Introduce one lower-margin variable per
expiry and impose its constraints at zero, at all 20 cash strikes, and on the
slope above the largest strike. Use separate nonnegative long and short
variables. The first LP maximizes the summed complete-support lower bound.

A static arbitrage may have a zero rather than a strictly positive lower
bound. Therefore solve a second LP subject to exact non-loss and maximize a
fixed positive linear functional of the payoff values at every strike kink
and at one upper-tail point. A candidate must have a recomputed lower bound no
smaller than its propagated numerical band with negative sign and a positive
payoff witness exceeding that band. This detects a nonzero, everywhere
nonnegative payoff without using realized future returns.

Use ten disjoint batches of 1,000 paths. Solve one LP for every selected
$(t,\omega)$, jointly across expiries. For each batch report the number and
fraction of candidates, the direct-$p$ count, the self-normalized physical
fraction, and the median complete-support lower bound. Across batches report
the mean and empirical 10th and 90th percentiles of each statistic. Batching
measures the variation across independent path groups; it is not a
training/validation split.

For every candidate, estimate its conditional physical P&L using a ball in
the full state $(s,h)$. Normalize coordinates using direct-$p$ state moments,
and let the ball contain the 1,024 nearest states from all simulated paths,
excluding the query path. The direct-$p$ estimate is

$$
\widehat m_p(s,h)
=\frac{\sum_{r\in p\cap\mathcal N_t(s,h)}X_r}
       {|p\cap\mathcal N_t(s,h)|},
$$

and the importance-sampled estimate is

$$
\widehat m_q(s,h)
=\frac{\sum_{r\in\mathcal N_t(s,h)}w_rX_r}
       {\sum_{r\in\mathcal N_t(s,h)}w_r},
\qquad w_r=\frac{p_r}{q_r}.
$$

Evaluate the frozen action on every joint collection of expiry spots in the
ball before applying these weights. Report the ball radius, direct-$p$ count,
effective sample size, mean, path-level standard error, and 10th, 50th, and
90th P&L percentiles for both estimators. A conditional mean more than three
standard errors above zero is called statistically positive. Report this
diagnostic in addition to the complete-support certificate; do not use it to
select the action.

#### Local conditional-portfolio test

Use the same ten disjoint path batches in a ten-fold cross-fit. In fold~$b$,
the 1,000 paths in batch~$b$ supply query and validation states, while the
other 9,000 paths supply training states. At each query state and trading time,
normalize $(s,h)$ using the training paths and define the ball by the 1,024th
nearest training path. Freeze this radius, and use every held-out path inside
it except the query path for validation.

Using the query state's entry prices, strikes, deltas, and costs, fit one
portfolio jointly across all expiries. On the training neighbours solve

$$
\max_{\sum_j\|a_{tj}\|_1\leq1}
\widehat E_p[X_t(a_t)\mid\mathcal N_t(s,h)]
$$

subject to non-loss on every training-neighbour joint return. A training
candidate has no loss beyond its observation-specific numerical band, at
least one gain, and a training mean more than three path-level standard errors
above zero. Evaluate the frozen action on the held-out paths inside the frozen
ball. Report the complete chain: training candidate, validation population,
validation loss and gain counts, direct-$p$ and importance-weighted means and
errors, and complete-support lower-bound replay. A validation failure does not
delete a training finding; classify it explicitly as not confirmed. Because
the returns select the local portfolio, only the cross-fitted result supports
a statement about the generated law.

The implementation may add the neighbour non-loss constraints iteratively to
avoid repeatedly passing a dense 1,024-row matrix to the LP solver. After each
solve, replay the action on every training neighbour and add every material
violator before classification. This is only constraint generation: it must
produce the same maximum-mean solution as the LP containing all neighbour
constraints at once.

#### Expiry-separated learned searches

Fit a separate action network for every trading-time/expiry pair. Each cell
contains its matched spot leg and 20 calls. Its action and net return are

$$
a_{t,j}(s_t,h_t)\in\mathbb R^{21},
\qquad
X_{t,j}=a_{t,j}^\top R_{t,j}-|a_{t,j}|^\top c_{t,j}.
$$

The 70 networks may be evaluated and optimized together, but they share no
parameters and no gradients. No objective combines trading times or expiries.
Run the following utilities:

1. power-exponential utility;
2. exponential utility; and
3. mean--variance, $E[X]-\gamma\operatorname{Var}(X)/2$.

Do not use a hyperbolic tangent, radial map, or action cap. Precondition each
instrument output by its training return RMS:

$$
a_{t,j,\ell}(s,h)
=\frac{v_{t,j,\ell}(s,h)}
       {\gamma\,\operatorname{RMS}_{\mathbb P}(R_{t,j,\ell})}.
$$

This is an unconstrained strategy, not an arbitrage certificate with an
$L^1$ budget. The preconditioner permits large portfolios without saturating
an output nonlinearity. Use constant-learning-rate AdamW, no gradient
clipping, compiled training, and explicit finite-value checks.

For power-exponential and exponential utility define, separately on training
and validation,

$$
D_{t,j}=\frac{u'(X_{t,j})}{E_{\mathbb P}[u'(X_{t,j})]}.
$$

For mean--variance the analogous marginal is
$1-\gamma(X_{t,j}-E_{\mathbb P}[X_{t,j}])$; it is a density only if it remains
nonnegative. Report $E_{\mathbb P}[D]$, its negative fraction, return moments,
the transaction-cost-adjusted first-order residual

$$
E_{\mathbb P}\!left[
D_{t,j}\{R_{t,j}-\operatorname{sign}(a_{t,j})c_{t,j}\}
\right],
$$

its path-level MC error, and parameter-gradient norms on training and
validation. Also report the effective sample size of the combined
importance-and-density weights. Treat moment statistics from a cell with
effective sample size below 20 as unresolved; a self-normalized standard
error can otherwise become spuriously small when one tail path carries nearly
all density mass.

With the preconditioner above and homogeneous proportional costs, changing
$\gamma$ from $1$ through $10,100,1000,10000$ scales actions, P&L, objectives,
and their MC errors by $1/\gamma$. The marginal utility, normalized density,
and moment $z$ statistics are unchanged. Verify this identity numerically;
separate refits at each risk aversion are redundant.

Promote a learned method from 100,000 to one million paths only if it
converges, detects the PCA-IV positive control, and remains finite and
well-conditioned for both generators. Failure of this gate is a failed search
method, not evidence of absence of arbitrage.

#### Arb tracing

Use **arb tracing** to separate the conditional portfolio direction from its
notional. At each trading time estimate the physical state mean and covariance
on training paths, including the likelihood weights of the spot proposal:

$$
m_t=E_{\mathbb P}[x_t],\qquad
C_t=Q_t\Lambda_tQ_t^\top,\qquad
\xi_t=(x_t-m_t)Q_t\Lambda_t^{-1/2}.
$$

Fit these quantities on training paths only and use the same transformation
on validation. Apply an explicit relative eigenvalue floor and report the
resulting factor means and covariance errors. The transition is Gaussian
conditional on its starting state, but the reported universes sample their
starting states from the historical state set. Their marginal state law is
therefore a Gaussian mixture rather than one Gaussian. The transformation
makes its physical covariance the identity; it does not make the factors
independent. Do not whiten under the importance proposal.

For each trading time fit an independent network which trades all seven
expiries jointly. Preserve state-dependent activity with

$$
d_t(x_t)=\frac{v_t(\xi_t)}{1+\|v_t(\xi_t)\|_1},
\qquad \|d_t(x_t)\|_1<1.
$$

The network learns only this direction at a moderate probe notional, initially
$H_0=2$. Keep the economic notional outside the network. After fitting, freeze
the direction and evaluate on a logarithmic notional grid

$$
H\longmapsto
E_{\mathbb P}\!\left[u\!\left(X_t(a_t^{(0)}+Hd_t)\right)\right],
\qquad
X_t(a)=a^\top R_t-|a|^\top c_t.
$$

Always value the combined position $a_t^{(0)}+Hd_t$ before applying absolute
transaction costs. In particular, do not add the separately costed P\&Ls of
the base and residual positions. Use stable log-sum-exp valuation for entropy.
Report the derivative with respect to $H$, its path-level MC error, the
marginal-density effective sample size, and all NaN or infinity checks.

If training utility has a finite maximum, freeze that position as the new base
and fit another residual direction. This continuation decision uses training
only; validation classifies the already frozen portfolio and must not select
the next residual. If utility continues increasing, audit the direction before
the marginal derivative becomes statistically or numerically unresolved.

For each frozen scale report the physical mean and its MC error, the 1\% and
5\% P\&L quantiles, and

$$
\widehat p^-_t
=\widehat P_{\mathbb P}\!\left[X_t<-\epsilon_{\mathrm{num},t}\right]
$$

on both training and validation. Call a result a near-arbitrage candidate only
when both sets have statistically positive mean and

$$
\widehat p^-_t
\leq \operatorname{MCSE}(\widehat p^-_t)+\epsilon_p,
\qquad \epsilon_p=N^{-1},
$$

under both the likelihood-weighted proposal estimate and the direct-$\mathbb
P$ subsample. A portfolio whose utility has a finite optimum and whose loss
probability materially exceeds this band is a high-quality finite-risk trade,
not an arbitrage. Record it separately because such trades explain why a
single utility optimum can hide a lower-return arbitrage direction.

#### Independent conditional audit

Reference neighbours identify the candidate direction, basket, and state
region $A_t$. Certification uses actual audit outcomes:

1. freeze the direction or basket and $A_t$;
2. select audit paths whose $z_{r,t}$ lies in $A_t$;
3. evaluate the frozen $X_{r,t}$ on those paths;
4. report direct-$p$ loss, neutral, and gain counts;
5. report importance-weighted probabilities and means using all proposal
   paths in the region; and
6. apply the confidence and multiplicity rules below.

If the audit region contains too few direct-$p$ paths or has inadequate
weighted effective sample size, label the candidate unresolved rather than
arbitrage or no-arbitrage. Merely evaluating an audit query with the reference
neighbour distribution is an out-of-sample state check, not an independent
return-distribution audit.

Use the audit set also to check that the conditional estimator itself works,
independently of candidate selection. At every audit query, estimate the
conditional mean, loss probability, and gain probability using reference
neighbours only. With bins fixed before reading audit outcomes, report
reliability tables comparing predicted with realized direct-$p$ loss and gain
frequencies, their likelihood-weighted analogues, conditional-mean residuals,
and empirical coverage of the reported conditional intervals. Keep trading
times separate. Poor calibration invalidates claims based on conditional
support even when a selected candidate happens to pass its audit.

### Sampling uncertainty

The independent statistical unit is a complete path. Because the primary
tests are time-resolved, each path contributes one outcome to a test at time
$t$. Never treat path-time or path-time-instrument observations as
independent.

For an unconditional weighted mean at time $t$, report

$$
\widehat m_t=\frac1N\sum_{r=1}^Nw_rX_{r,t},
\qquad
\operatorname{se}(\widehat m_t)
=\frac{\operatorname{Std}_r(w_rX_{r,t})}{\sqrt N}.
$$

For a conditional region $A_t$, use the self-normalized estimate

$$
\widehat m_t(A_t)
=
\frac{\sum_rw_r1_{\{z_{r,t}\in A_t\}}X_{r,t}}
     {\sum_rw_r1_{\{z_{r,t}\in A_t\}}},
$$

and obtain its interval by a complete-path bootstrap. Report the number of
direct-$p$ paths in the region and the regional effective sample size

$$
N_{\mathrm{eff}}(A_t)
=
\frac{\left(\sum_rw_r1_{\{z_{r,t}\in A_t\}}\right)^2}
     {\sum_rw_r^2 1_{\{z_{r,t}\in A_t\}}}.
$$

Use a path bootstrap whenever weights are materially dispersed. Never replace
$N$ by the number of instruments or trading-time observations.

For direct-$p$ audit paths, report one-sided exact
Clopper--Pearson intervals for loss and gain probabilities. If $L$ losses and
$G$ gains are observed among $n$ independent paths, use

$$
\overline p_{\mathrm{loss}}
=B^{-1}_{1-\alpha}(L+1,n-L),
\qquad
\underline p_{\mathrm{gain}}
=B^{-1}_{\alpha}(G,n-G+1),
$$

with the usual boundary conventions, where $B^{-1}_q(a,b)$ is the beta
quantile. If no loss is observed, the one-sided confidence-$1-\alpha$ upper
bound simplifies to

$$
1-\alpha^{1/n},
$$

approximately $-\log(\alpha)/n$. At 95% confidence this is approximately
$3/n$, not zero. A candidate with zero observed losses must be described
using this bound.

For importance-weighted incidence estimates, use a path bootstrap and confirm
the conclusion on the direct-$p$ component. Correct confidence levels for the
number of instruments, query states, strategies, and 48 market cases using a
familywise procedure such as Holm. Report both pointwise and adjusted
results.

### Candidate certification

Call a frozen direction or basket an empirical per-trading-time arbitrage
candidate only if, on independent audit paths at its specified time and
state region:

1. no P&L is below its propagated negative numerical band;
2. the adjusted lower confidence bound for
   $\mathbb P(X_t>\eta_t)$ is positive;
3. the adjusted lower confidence bound for
   $\mathbb E_{\mathbb P}[X_t]$ is positive;
4. the direct-$p$ and importance-weighted conclusions agree;
5. the importance likelihood and tail-coverage checks pass; and
6. a float64 replay reproduces the result.

Even then, report the upper confidence bound on loss probability and identify
the exact trading time, state region, and strategy class. If a condition
fails, report the candidate as rejected rather than silently omitting it.

## Required runs and reports

The canonical executable is
`quantdata.dysanos.grid.gendysanos.run_required_experiments`. It generates one
matched path cache per generator, frequency, and path count; derives both
return horizons, both hedge conventions, and all three cost rates from that
cache; keeps trading-time diagnostics separate; and enforces the functional
gate before launching a one-million-path stage. The older
`run_terminal_arbitrage_tests` command is retired because it pooled trading
times and did not implement the matched case matrix.

Run a 1,000-path smoke test whenever a distinct generator, timing, return, or
hedge code path has not yet been exercised. Smoke tests are mechanical and do
not replace any reported 100,000-path experiment.

Execute the reported experiments in the following order. At each path count,
run the four generator-frequency combinations concurrently on four GPUs:

| GPU | Generator | Trading frequency |
|---:|:---|:---|
| 0 | DYSANOS | daily |
| 1 | DYSANOS | weekly |
| 2 | PCA-IV | daily |
| 3 | PCA-IV | weekly |

1. Run 100,000 paths with expiry returns, delta hedging, and one-basis-point
   cost. Pass the functional gate below, then run the same four cases with one
   million paths and check convergence.
2. Repeat the 100,000-to-one-million sequence with incremental returns, delta
   hedging, and one-basis-point cost.
3. Repeat both return horizons with raw, unhedged returns and one-basis-point
   cost, again completing and checking 100,000 paths before one million.
4. For every completed generator, frequency, return-horizon, and hedge case,
   add zero- and ten-basis-point costs. Reuse the same generated paths,
   importance weights, prices, raw returns, and deltas; changing cost must not
   trigger path simulation.

This produces all 48 cases. Within a fixed generator, frequency, and path
count, also reuse a sufficiently long coherent path wherever daily subsampling
or return reconstruction makes that exact reuse possible. Never reuse a
DYSANOS likelihood ratio for PCA-IV or conversely.

### Functional gate

Do not proceed from a 100,000-path stage to its one-million-path stage until:

- every state, price, return, delta, cost, and weight is finite;
- exact timing tests pass for daily, weekly, expiry, and incremental returns;
- fixed contracts, raw returns, delta hedges, and gross cost identities replay
  in float64;
- PCA-IV and DYSANOS use the intended decoders and neither silently repairs a
  PCA-IV surface;
- every option meets the applicable strike and cost-adjusted sign-boundary
  coverage targets;
- importance weights normalize and recover direct-$p$ spot and return moments;
- reference and audit paths are disjoint and have the intended mixture shares;
- conditional-neighbour radii and effective sample sizes are numerically
  meaningful;
- any unexpected extreme price, delta, P&L, likelihood, static violation, or
  candidate has been replayed and explained; and
- CPU and GPU memory remain within the configured limits.

### One-million-path convergence

Compare estimates, probabilities, and candidate rates, not raw candidate
counts. The one-million-path result passes the convergence assessment only if:

- $E_q[w]$ remains statistically consistent with one;
- effective sample size grows approximately in proportion to path count and
  its fraction of total paths does not collapse;
- direct-$p$ and weighted-$q$ physical moments remain consistent, with standard
  errors decreasing at the expected path-level rate;
- pre-specified probability and mean estimates are compatible across path
  counts after their confidence intervals are considered;
- conditional neighbourhood radii shrink while local effective sample sizes
  rise under the specified neighbour schedule;
- audit candidate classifications either stabilize or changes are explained by
  the smaller numerical or statistical resolution; and
- static-violation and dynamic-candidate rates stabilize for each generator.

Ordinary path-level standard errors should fall by approximately
$\sqrt{10}$. Extreme maxima and minima need not converge, and an increased raw
number of rare candidates is not by itself deterioration when the sample is
ten times larger.

For every case, write a machine-readable report containing:

- the complete versioned configuration;
- path and trading-time geometry;
- return, hedge, and cost definitions;
- numerical-tolerance formula and precision;
- importance and effective-sample-size diagnostics;
- unconditional and conditional support diagnostics;
- trained strategy class and convergence diagnostics;
- path-level confidence intervals and multiplicity correction;
- all rejected and surviving candidates; and
- peak CPU and GPU memory.

The aggregate report must compare DYSANOS with PCA-IV for each of the four base
markets and show separately the effect of delta hedging and each cost level.
Report static violations separately from dynamic candidates. Its conclusion
must use the finite-sample language specified above.

## Tests

Add focused tests for:

1. stationary surface fitting without spot leakage;
2. exact matrix-exponential mean and covariance;
3. conditional spot covariance reconstruction;
4. ten daily and ten weekly trading timestamps;
5. daily intermediate states in weekly markets;
6. fixed cash strikes and calendar expiries;
7. exact-expiry payoff when expiry lies inside a weekly interval;
8. nine incremental intervals from ten trading timestamps;
9. expiry-specific spot returns;
10. incremental trading-time spot returns;
11. raw and delta-hedged return identities;
12. gross option-plus-delta cost without netting;
13. all three cost rates;
14. importance likelihood normalization and physical-moment recovery;
15. per-option strike and break-even coverage;
16. long and short transaction costs applied before numerical classification;
17. numerical bands from unreduced arithmetic scales without a cash floor;
18. propagated basket numerical bands;
19. state-independent basket feasibility at each trading time;
20. path-level 80:20 reference/audit separation stratified by mixture;
21. time-separated, leave-one-path-out conditional neighbours;
22. local weighted loss/gain probabilities and effective sample size;
23. frozen candidate regions evaluated on actual audit outcomes;
24. held-out calibration of conditional means and tail probabilities;
25. path-level weighted standard errors and regional bootstrap intervals;
26. Clopper--Pearson zero-loss bounds;
27. multiplicity-adjusted candidate classification;
28. float64 replay of flagged candidates;
29. NaN and infinity rejection at every stage;
30. identical contract, return, cost, and diagnostic interfaces for DYSANOS
    and PCA-IV;
31. PCA-IV decoding without static-arbitrage repair or sample rejection;
32. generator-specific cache and likelihood isolation;
33. reuse of paths and weights across hedge and cost variants;
34. a basket arbitrage is found when every individual instrument has both
    return signs, proving that individual screens do not gate basket search;
35. every qualifying traded-grid static spread has nonnegative raw expiry P\&L
    under its state-dependent action; and
36. the exact one-expiry payoff-cone program detects a deliberately
    misordered call surface, including after converting between raw and
    delta-hedged representations with the matched spot leg;
37. the payoff-cone tail-slope constraint rejects a portfolio with unbounded
    terminal loss;
38. radial \(L^1\) normalization retains an inward gradient outside the unit
    ball;
39. power-exponential OCE importance weighting leaves the deterministic
    offset term unweighted; and
40. convergence checks from 100,000 to one million paths;
41. physical likelihood-weighted state whitening produces zero factor means
    and identity covariance separately at every trading time;
42. arb-direction networks share no parameters or gradients across trading
    times while jointly trading all expiries within a time;
43. the smooth direction map retains a nonzero gradient at large raw norms;
44. base-plus-residual transaction costs remain exact when an instrument
    position crosses zero;
45. analytic entropy and power-exponential notional derivatives agree with
    finite differences; and
46. synthetic nonnegative and positive-mean risky portfolios are classified
    respectively as near-arbitrage candidates and finite-risk trades.
