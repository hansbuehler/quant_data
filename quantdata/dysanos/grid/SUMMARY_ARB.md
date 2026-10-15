# DYSANOS arbitrage work: summary and handoff

This document consolidates the DYSANOS arbitrage discussion and experiments as
of 2026-08-10. It is intended as a basis for the next DYSANOS version.
gendysanos/GENDYSANOS.md remains the executable specification.

## Current conclusion

DYSANOS guarantees that each decoded SANOS surface is smooth and free of
static arbitrage. This does not imply that the generated price process is free
of dynamic arbitrage.

The current numerical evidence is:

- The statewise complete-support to-expiry LP detects no DYSANOS candidate in
  the reported daily or weekly experiments at 0, 1, or 10 basis points. The
  same test detects the known PCA-IV positive control.
- The local period-return test finds candidates for both generators and more
  candidates for DYSANOS in every reported case.
- Of the two strongest DYSANOS weekly period candidates at 10 basis points, one
  had losses in one million fresh conditional continuations and was rejected.
  The other had no observed loss and was not falsified at that sampling
  resolution.
- OCE and learned-utility searches did not produce a reliable
  state-dependent martingale density or a general conditional-arbitrage
  certificate.

A zero count is a scoped non-detection. It is not proof that DYSANOS is free of
dynamic arbitrage.

## Notation

- $h_t=(h_t^1,\ldots,h_t^{n_h})$ is the implied-volatility surface state.
- $s_t=\log S_t$ is log spot.
- $x_t=(s_t,h_t)$ is the complete observable state.
- $t$ is a trading time, $\omega$ a Monte Carlo path, and
  $\ell=1,\ldots,L$ an instrument.
- $T_t^\ell$ and $K_t^\ell$ are floating time-to-expiry and spot-relative
  strike.
- $\tau_t^\ell=t+T_t^\ell$ and $k_t^\ell=S_tK_t^\ell$ are the fixed calendar
  expiry and cash strike of the contract opened at $t$.

The implementation may continue storing xstate in the order $(s,h)$. The
symbol $h$ must not include spot.

## Canonical market generator

### Surface first

Fit the autonomous surface process without spot:

\[
 dh_t=\kappa(m-h_t)\,dt+\Sigma\,dW_t^h.
\]

The eigenvalues of $\kappa$ must have positive real part. Reported simulation
uses the exact matrix-exponential transition, not Euler. The invariant
surface distribution exists under this stability condition.

After fixing the surface model, fit spot conditionally:

\[
 ds_t=\mu\,dt+\beta^\top dW_t^h+\varsigma\,dW_t^s,
\]

where $W^s$ is independent of $W^h$. Fitting spot must not change the fitted
drift or diffusion of $h$.

The final arbitrage benchmark uses Gaussian innovations with bandwidth equal
to one. Earlier historical-alpha sampling experiments are not canonical.

### One coherent path

Each Monte Carlo realization contains one coherent daily path through every
trading time and every option expiry. Never draw spot separately by strike,
expiry, option, or trading time.

Weekly trading times are a five-day subsample of the daily path. Intermediate
daily states are still simulated so short-dated options settle at their exact
expiry.

At entry, freeze $(\tau_t^\ell,k_t^\ell)$. A later value reprices the same
contract at remaining maturity and relative strike $k_t^\ell/S_u$.

### Returns

For a call opened at $t$, the to-expiry raw return is

\[
 R_{t,\ell}^{\mathrm{expiry}}
 =(S_{\tau_t^\ell}-k_t^\ell)^+
  -C_t(\tau_t^\ell,k_t^\ell).
\]

For the next trading time $u$, the period return is

\[
 R_{t,\ell}^{\mathrm{period}}
 =V_u(\tau_t^\ell,k_t^\ell)
  -C_t(\tau_t^\ell,k_t^\ell),
\]

where $V_u$ is the mark-to-market value when $u<\tau_t^\ell$ and the exact
expiry payoff if the option expires before $u$.

Ten trading timestamps produce ten to-expiry cohorts and nine period returns.
Daily and weekly refer to trading-time spacing, not independent payoff draws.

### Delta hedging and costs

A preconstructed entry-delta-hedged option return is

\[
 R_{t,\ell}^{\Delta}
 =R_{t,\ell}^{C}
  -\Delta_{t,\ell}(S_{e(t,\ell)}-S_t),
\]

where $e(t,\ell)$ is the option expiry if it occurs before the next trading
time and otherwise is the next trading time. The embedded delta hedge closes
at the exact option expiry. A separately traded stock instrument follows its
own stated return horizon.

For one unit, the gross entry cost notionals are

\[
 G_C^{\mathrm{raw}}=|C_t|,
 \qquad
 G_C^\Delta=|C_t|+|\Delta_tS_t|,
 \qquad
 G_S=|S_t|.
\]

The option and stock legs are not netted. At cost rate
$\gamma\in\{0,10^{-4},10^{-3}\}$,

\[
 \pi_t(a_t)=a_t^\top R_t-\gamma|a_t|^\top G_t.
\]

Transaction cost is economic. Numerical tolerance absorbs floating-point
roundoff. Monte Carlo uncertainty is statistical. These are three distinct
quantities.

For a primitive return, let $A^{\rm net}$ be the sum of the absolute unreduced
price, payoff, hedge, and cost operands. The float32 classification band is

\[
 \eta
 =32\epsilon_{32}
  \max\{A^{\rm net},\operatorname{tiny}_{32}\},
 \qquad
 \epsilon_{32}=\operatorname{finfo}(\mathrm{float32}).\mathrm{eps}.
\]

Basket bands propagate the primitive bands with absolute action weights and
include the final dot-product roundoff scale. Replay every retained candidate
from primitive values in float64.

### Spot importance sampling

Plain physical sampling initially produced apparent arbitrage in deep OTM
options because too few paths reached their strikes. The corrected proposal
widens spot while preserving physical expectations through complete-path
likelihood weights:

\[
 q(\omega)=0.1p(\omega)+0.9\bar q(\omega),
 \qquad
 w(\omega)=\frac{p(\omega)}{q(\omega)}.
\]

The paper runs used a spot standard-deviation multiplier of eight and targeted
at least 1,000 observations on both sides of relevant strike and return-sign
boundaries.

Every run must verify:

- $E_q[w]\simeq1$;
- likelihood-weight effective sample size;
- agreement of direct-$P$ and importance-weighted physical moments; and
- strike and P&L-boundary coverage.

Importance sampling improves rare-loss discovery. It does not alter the
economic classification, which remains under $P$.

## Corrections that invalidate older reports

An old result is non-canonical unless its cache version confirms:

- one coherent spot and surface path;
- fixed cash strikes and calendar expiries after entry;
- exact expiry payoff rather than an independently drawn terminal spot;
- consistent intermediate daily states for weekly trading;
- exact-expiry closure of an option and its embedded delta hedge;
- gross option-plus-stock delta cost without netting;
- generator-specific complete-path likelihood weights;
- Monte Carlo errors computed across independent paths, not path-time cells;
  and
- the corrected distinction between to-expiry and period returns.

The pooled run_terminal_arbitrage_tests workflow and experiments using
separately sampled expiry payoffs are retired.

## Arbitrage definition and logical scope

At trading time $t$, the conditional action is predictable:

\[
 a_t=a_t(s_t,h_t),
 \qquad
 \|a_t\|_1\leq1.
\]

It is an arbitrage in the tested one-period market if

\[
 \pi_t(a_t)\geq0\quad P\text{-a.s.},
 \qquad
 P\{\pi_t(a_t)>0\}>0.
\]

The $L^1$ constraint fixes scale. It does not remove a scalable arbitrage
direction. The action may be zero outside a positive-probability state region.

For consecutive period returns, exact conditional no-arbitrage of the complete
cross-section at every history implies no dynamic arbitrage for a
self-financing sum over trading times. Unconditional tests and
instrument-by-instrument tests are insufficient.

To-expiry cohorts overlap in calendar time. Passing each cohort separately
does not prove no-arbitrage for arbitrary overlapping strategies.

Finite Monte Carlo cannot verify an almost-sure condition over every state and
predictable portfolio. Supported wording is:

> No conditional arbitrage candidate was detected within the stated strategy
> class, sampled state regions, and numerical resolution.

## Diagnostic hierarchy

### Cross-sectional surface audit

For every decoded trading-time surface, check call bounds, strike
monotonicity, vertical-spread slope, strike convexity, and calendar
monotonicity.

DYSANOS should pass by construction; a failure is a numerical or implementation
error. PCA-IV is deliberately not repaired and exhibits violations on the
dense $7\times201$ grid and the traded $7\times20$ grid.

This is a static surface test. It does not test returns between trading times.

### Unconditional screens

Instrument sign imbalance, long/short means, and state-independent baskets are
sampler and accounting controls. They are only necessary conditions.

Individual instruments must never gate portfolio search. Every individual
return may have both signs while a basket is an arbitrage. A direction may
also change with $x_t$ and disappear in an unconditional average.

### Conditional tail and mean estimates

Nearest-neighbour estimates of positive and negative conditional tails were
used to locate suspicious instruments and initialize strategies. Reduced
per-instrument state and full raw state $(s_t,h_t)$ were explored.

These estimates are not certificates. A positive conditional mean can coexist
with rare losses.

### Statewise multi-expiry complete-support LP

At a fixed state and trading time, solve jointly over all strikes and
expiries. Split the portfolio into non-negative long and short notionals and
impose $\|a_t\|_1\leq1$.

For to-expiry calls, payoff is piecewise affine in terminal spot. Non-loss over
complete terminal support is enforced at strike breakpoints and through the
lower- and upper-tail slope conditions. Require strict gain somewhere.

This is the strongest current to-expiry test. Direct-$P$ and
importance-weighted nearest-neighbour means measure the physical incidence of
the fixed certificate; they do not replace its support-wide non-loss
condition.

### Local conditional-portfolio period test

At a selected state, fit one portfolio jointly over calls and tradable spot
coordinates using a local training neighbourhood. Freeze state normalization,
centre, radius, and action. Evaluate the frozen action on disjoint validation
paths inside the same ball.

Classification requires no material validation loss, positive gain
probability, and positive mean under direct-$P$ and importance-weighted
estimators, with multiplicity adjustment.

This remains an empirical local test. Its occurrence rate is the fraction of
selected state-time centres retained by the procedure, not the physical
probability of arbitrage.

### Exact-state conditional replay

Freeze a candidate state and action and simulate fresh physical continuations
from that state. Report minimum P&L, losses beyond tolerance, mean and MC
error, and a one-sided confidence bound on loss probability.

This is the most useful current numerical discriminator between sampling
artifacts and persistent conditional candidates. It does not prove a
support-wide inequality.

## Headline results

### To-expiry complete-support LP

Each PCA-IV entry below is
complete-support candidate occurrence / occurrence significant under both
physical estimators. The denominator is 100,000 audited state-time pairs.
DYSANOS was $0/0$ in every row at both universe sizes.

| Interval | Cost | PCA-IV 100k | PCA-IV 1m | DYSANOS 100k and 1m |
|---|---:|---:|---:|---:|
| Daily | 0 bp | 5.273% / 1.721% | 5.097% / 1.564% | 0% / 0% |
| Daily | 1 bp | 0.950% / 0.331% | 0.844% / 0.258% | 0% / 0% |
| Daily | 10 bp | 0.042% / 0.037% | 0.043% / 0.029% | 0% / 0% |
| Weekly | 0 bp | 5.525% / 2.043% | 5.537% / 1.970% | 0% / 0% |
| Weekly | 1 bp | 0.923% / 0.415% | 0.940% / 0.375% | 0% / 0% |
| Weekly | 10 bp | 0.059% / 0.057% | 0.059% / 0.050% | 0% / 0% |

The PCA-IV rates are stable from 100,000 to one million paths. This is the
important positive control. The DYSANOS zeros are consistent with the SANOS
terminal payoff cone and show no detected candidate in this test. They are not
a general dynamic no-arbitrage result.

### Period-return local portfolios

Each row uses 9,216 selected state-time centres. Unresolved means that the
validation ball lacked the prescribed direct-$P$ count or weighted effective
sample size.

| Paths | Interval | Cost | PCA-IV candidate | PCA-IV unresolved | DYSANOS candidate | DYSANOS unresolved |
|---:|---|---:|---:|---:|---:|---:|
| 100k | Daily | 0 bp | 0.803% | 10.569% | 7.107% | 9.429% |
| 100k | Daily | 1 bp | 0.315% | 10.569% | 5.534% | 9.429% |
| 100k | Daily | 10 bp | 0.054% | 1.421% | 2.365% | 9.429% |
| 100k | Weekly | 0 bp | 0.130% | 7.975% | 2.268% | 7.227% |
| 100k | Weekly | 1 bp | 0.076% | 7.975% | 1.736% | 7.227% |
| 100k | Weekly | 10 bp | 0.022% | 5.968% | 0.738% | 7.140% |
| 1m | Daily | 0 bp | 1.584% | 0% | 12.500% | 0% |
| 1m | Daily | 1 bp | 0.564% | 0% | 9.082% | 0% |
| 1m | Daily | 10 bp | 0.141% | 0% | 3.266% | 0% |
| 1m | Weekly | 0 bp | 0.423% | 0% | 2.365% | 0% |
| 1m | Weekly | 1 bp | 0.358% | 0% | 1.758% | 0% |
| 1m | Weekly | 10 bp | 0.184% | 0% | 0.412% | 0% |

The larger universe resolves the sparse regions but does not remove the period
findings. DYSANOS has the larger retained occurrence in every row.

One million fresh continuations from the two strongest DYSANOS weekly raw
states at 10 basis points gave:

| Rank | Loss | Beyond band | Minimum P&L | Mean P&L | MC error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000% | 0.0000% | $2.963\times10^{-4}$ | $4.683\times10^{-3}$ | $1.019\times10^{-6}$ |
| 2 | 0.0752% | 0.0746% | $-1.478\times10^{-3}$ | $5.821\times10^{-3}$ | $2.175\times10^{-6}$ |

Rank 2 is rejected. Rank 1 was not falsified. Zero losses give a one-sided 95%
upper bound of approximately $2.996\times10^{-6}$ on its conditional loss
probability. This is strong numerical evidence, not proof that the probability
is zero.

### Scope of retained reports

gendysanos/reports/paper-source contains raw-coordinate to-expiry and period
runs. The optimizer can select separately tradable spot coordinates and can
therefore construct a hedge. This is not the same experiment as replacing
every option coordinate by a preconstructed entry-delta-hedged return.

The retained archive is not the full raw-versus-preconstructed-delta matrix in
GENDYSANOS.md. The next version must either run that matrix with exact-expiry
hedge closure or describe the current tables as joint raw-call-plus-spot
portfolio tests.

## Direct and utility-based searches

### V2.0

V2.0 used a state-dependent raw network output $v(x_t)$ and

\[
 a(x_t)=\frac{v(x_t)}{\max\{1,\|v(x_t)\|_1\}},
 \qquad
 \|a(x_t)\|_1\leq1.
\]

The loss maximized mean P&L and penalized average downside, tail downside, and

\[
 \max(0,\|v\|_1-1)^2.
\]

The last term supplies an inward gradient when the raw output is outside the
unit ball. Original-state, reduced-state, masked, and unmasked variants were
tested. V2.0 was useful for locating directions, but its sample downside
penalties were not complete-support constraints. It did not produce a robust
certificate.

### Original OCE measure change

The original work optimized OCE objectives over conditional strategies and
formed densities from marginal utility. Variants included conditional and
unconditional cost, per-instrument caps, multiple measure-change stages,
path-level densities, masks, entropy, power-exponential utility, and CVaR.

The main failure modes were:

- flat action maps or saturated utility derivatives;
- marginal density concentrated on very few adverse paths;
- apparently small moments after effective sample size collapsed;
- deterministic strategies dominated by sample-mean noise;
- excellent finite-risk trades preferred to lower-return arbitrage
  directions; and
- unstable transfer from training to validation.

When evaluating

\[
 D=\frac{u'(a^\top R)}{E[u'(a^\top R)]}
\]

on validation data, recompute the denominator on validation. The training
normalization constant need not transfer.

For the simplified no-offset objective

\[
 \sup_a E_P[u(a^\top R)],
\]

an interior optimum satisfies

\[
 E_P[u'(a^\top R)R]=0.
\]

This identity is a useful optimizer check. It does not solve rare-tail
concentration, sample uncertainty, transaction-cost kinks, or incomplete
optimization of the conditional strategy class.

### Saturation

There were two distinct effects:

1. A tanh or hard action cap can have little or no derivative near or outside
   its boundary.
2. Even with stable manual gradients, exponential-type marginal utility can
   concentrate on a handful of worst paths and lose effective sample size.

A radial map retains an inward derivative and makes $a=0$ regular, but its
derivative still shrinks at large raw norms. Increasing a cap changes the
scale; it does not ensure that a rare adverse path is present or that the
optimal direction remains the same.

### Arb tracing and Case C

Arb tracing separated direction from notional:

1. whiten $x_t$ under physical likelihood weights;
2. learn a bounded direction at moderate notional;
3. freeze it and line-search utility over external notional $H$; and
4. if utility has a finite optimum, freeze the position and learn a residual
   direction.

Whitening improved conditioning but not identification. A true arbitrage
remains non-loss-making as notional increases. A good risky trade eventually
pays a tail penalty. However, exponential utility and its derivative also
approach an asymptote along a true arbitrage, so a small derivative at large
$H$ is not evidence against arbitrage.

Case C fixed utility risk aversion, increased external notional $n$, used
separate non-negative long and short variables, and evaluated analytic
gradients. This removed avoidable numerical underflow but not density
concentration, boundary effects, or changing optimal directions. Warm starts
can retain a direction that is no longer optimal. Cold starts and independent
directions by notional are safer.

Feed-forward, learned-partition, and learned-kernel policies were compared on
opposite 50:50 folds. None became a reliable certificate. Bucketed
worst-expectation objectives are useful stress tests, but their result depends
on bucket allocation and is not CVaR certification.

Mean-variance is unsuitable as a primary arbitrage detector: a true arbitrage
can have variable positive payoff, so a variance penalty can create a finite
optimum.

For

\[
 u_p(x)=-(1+\lambda)\max(-x,0)^p,
\]

$p=2$ is $C^1$ but not $C^2$ at zero. It focuses on downside, but a finite
sample with no observed loss still produces a flat or misleading objective.

### Empirical CVaR density search

For

\[
 u_\lambda(x)=(1+\lambda)\min(x,0),
\]

the OCE dual density satisfies

\[
 E_P[D]=1,
 \qquad
 0\leq D\leq1+\lambda.
\]

For deterministic $t_0$ portfolios on 100,000 weekly to-expiry paths, exact
finite-sample LPs found cap 10 infeasible and cap 12.5 feasible for both PCA-IV
and DYSANOS. The feasible retained-tail fraction was 8%, and finite-sample
density-weighted return errors were near machine precision.

At state-dependent $t_6$, a hard 64-cell partition produced very wide cap
ranges and target-density effective sample sizes near 100. A richer radial
feed-forward action remained positive to $\lambda=8\times10^6$ for PCA-IV and
$10^7$ for DYSANOS. The corresponding tail effective sample sizes were only
about 5.6 and 6.2, while the selected sample portfolios still had negative
minima near $-0.777$ and $-0.801$.

The latest feed-forward search used all 100,000 fixed paths for fitting and
replay. It was therefore an in-sample diagnostic. No reliable minimum
state-dependent $\lambda$ and no state-dependent martingale density were
found.

## Main lessons

1. A positive conditional mean is not arbitrage.
2. Training/validation is necessary but cannot resolve losses rarer than the
   validation sample.
3. Importance sampling is essential for OTM and rare-loss coverage.
4. Individual-option tests cannot gate joint portfolio searches.
5. Utility optimization ranks trades; it does not directly impose the
   support-wide non-loss condition.
6. Analytic gradients solve numerical underflow, not statistical
   concentration.
7. State whitening helps optimization but does not certify an action.
8. Transaction-cost monotonicity applies to a frozen action. Counts from
   separately refitted strategies need not be monotone.
9. A sample arbitrage may be only a missing-tail observation.
10. Optimizer failure, zero gradients, or zero non-convex objective cannot be
    interpreted as absence of arbitrage.

## Priorities for the next version

### Resolve the strongest period candidate

For every leading frozen state/action:

1. express the exact next-state transition in independent Gaussian shock
   coordinates;
2. minimize P&L using multi-start differentiation and widened proposals;
3. reweight probability statements back to $P$;
4. replay discovered minima from primitive values in float64;
5. run at least one million independent conditional continuations; and
6. report binomial loss-probability bounds.

If feasible, combine interval or global optimization on expanding shock boxes
with a separate Gaussian tail bound. A negative interior P&L point has positive
probability under a continuous transition. Random sampling alone cannot prove
that no negative region exists.

### Complete the matched experiment matrix

For DYSANOS and PCA-IV, run 100,000 and one million paths for:

- daily and weekly trading;
- to-expiry and period returns;
- raw options with separately tradable spot;
- preconstructed entry-delta-hedged options with exact-expiry hedge closure;
  and
- costs of 0, 1, and 10 basis points.

Reuse the same path cache across hedge transformations and costs. Replay each
frozen strategy through the cost hierarchy. Require every detector to find the
PCA-IV positive control before interpreting a DYSANOS zero.

### Connect tests to a transition condition

For a fixed state and period, no-arbitrage of the traded vector is connected
to the current price vector lying in the relative interior of the convex hull
of conditional next values, with transaction costs handled through the
appropriate solvency geometry.

Investigate whether the period candidates arise from:

- low-rank conditional surface innovations with num_alpha equal to 5;
- decoder geometry restricting conditional next-price support;
- the fitted drift placing current prices close to a conditional convex-hull
  boundary; or
- an endpoint, hedge, cost, or accounting defect.

Compare num_alpha equal to 5 and num_alpha equal to None under matched
simulation. If a genuine violation remains, constrain or repair the transition
law rather than the already static-arbitrage-free SANOS cross-section.

### Keep OCE secondary

Do not return OCE to the main paper until it:

- detects PCA-IV reliably;
- retains adequate tail effective sample size;
- reproduces across folds and fresh conditional continuations;
- has analytic gradients matching finite differences;
- remains finite without gradient clipping; and
- agrees under direct-$P$ and importance-weighted estimators.

## Evidence and implementation map

- gendysanos/GENDYSANOS.md: current generator and diagnostic specification.
- ../tex/arbitrage.tex: current paper discussion and headline tables.
- gendysanos/reports/paper-source: reports underlying the paper tables.
- gendysanos/multi_expiry_arbitrage.py: statewise multi-expiry LP.
- gendysanos/conditional_static_arbitrage.py: local conditional portfolio test.
- gendysanos/candidate_replay.py: frozen-candidate replay.
- gendysanos/arb_tracing.py: direction/notional continuation.
- gendysanos/case_c.py: fixed-risk-aversion notional experiments.
- gendysanos/run_implied_cvar.py: latest empirical CVaR network search.
- rndysanos/RNDYSANOS.md: historical OCE specification and tests.

## Wording for the next paper

Use:

> The reported test detected no candidate in the stated strategy class and at
> the stated numerical resolution.

Do not use:

> DYSANOS is free of dynamic arbitrage.

It remains correct to say:

> Every decoded SANOS surface is free of static arbitrage by construction.

Always report the strategy class, path count, return horizon, trading interval,
hedge semantics, cost, importance law, and loss-probability resolution.
