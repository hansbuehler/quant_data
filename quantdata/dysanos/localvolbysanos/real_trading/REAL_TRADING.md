# Real trading

## Objective

Use the estimated physical dynamics of the local-volatility/forward-volatility
state to choose an optimized-certainty-equivalent (OCE) maximizing portfolio of
actual listed options. Every eligible instrument must have an observed bid and
ask at the decision time. The initial trading horizon is one week.

The portfolio optimization is deterministic conditional on the current state.
At each decision time, construct one fixed conditional scenario set and run
three specifications named `normalised`, `actual`, and `ITM`. Within each
specification solve six problems: power-exponential and entropy OCEs, each at
risk aversion `1`, `10`, and `100`. This is a per-state optimization, not a
learned trading policy or a stochastic optimizer.

This subtask is distinct from estimating bid/ask widths for hypothetical
options. Actual entry quotes determine the executable entry prices. The current
experiment exits survivors at the scenario-model midpoint; a future spread
model could instead impose conservative scenario-horizon liquidation costs.

The project may use shared production code such as `quantdata.dysanos` and
`quantdata.sanos`, but must not import from `aae` or another research project.

## Does the proposal make sense?

Yes. It is a coherent one-period stochastic portfolio problem:

1. observe the current arbitrage-free market and actual listed option quotes;
2. encode the current DLV/forward-volatility surface as a low-dimensional state;
3. draw the joint one-week distribution of the future surface and underlying;
4. reprice the same listed contracts, settling any that expire within the week;
5. compute scenario P&L using executable entry prices and conservative exit
   conventions; and
6. maximize an OCE subject to capital, Greek, margin, concentration, and
   liquidity constraints.

The idea's main missing link is indeed the **joint physical dynamics of spot
returns and surface changes**. There are also four additional issues that must
be resolved before the optimization is meaningful:

- physical versus risk-neutral probabilities;
- arbitrage and internal consistency of generated future surfaces;
- treatment of contracts expiring inside the horizon; and
- look-ahead-free estimation and backtesting.

## Physical scenarios versus risk-neutral repricing

The current option surface and its SANOS densities describe risk-neutral
pricing at a point in time. The AR model is intended to describe how the
observed market state changes through calendar time under the physical measure.
These roles should remain separate:

- **P dynamics:** scenario probabilities for future spot and surface states;
- **Q repricing:** arbitrage-free option values conditional on each future
  scenario state.

An OCE is an expected-utility/risk functional, so its scenario weights should
normally be physical probabilities. Sampling terminal spot directly from the
current option-implied density produces a risk-neutral distribution. That is
useful for pricing and stress testing, but it is not an estimate of physical
expected returns and should not silently become the OCE probability measure.

A risk-neutral density could later be used as a prior or ambiguity-set anchor,
then tilted toward historical physical returns. That is a robust-model variant,
not the recommended first implementation.

## State representation

The existing annual panel contains separate PCA/AR descriptions of log DLV
squared and log forward volatility. Both are derived from the same option
surface, so drawing them independently can create mutually inconsistent
states.

The scenario model should designate one authoritative repricing state:

- **DLV-first:** simulate log-DLV factors, reconstruct positive DLVs, and use
  SANOS to generate an arbitrage-free future call surface. Forward-volatility
  factors are observations or diagnostics, not a second independent price
  source; or
- **forward-volatility-first:** simulate an arbitrage-safe implied/forward-
  volatility parameterization and derive DLVs from the resulting calls.

DLV-first is the natural continuation of this project. Modeling `log(DLV^2)`
ensures positive local variance, but positivity alone is not the entire
arbitrage check. Every decoded scenario must still pass density normalization,
martingale, strike-convexity, calendar, and input-price-recovery diagnostics.

If both observed states materially improve forecasting, stack their PCA scores
and estimate one joint transition, but still decode prices from only one
authoritative representation.

## Recommended joint spot/surface model

Do not sample an AR surface shock and an independent spot return and then add an
ad-hoc correlation. Estimate their joint innovations directly.

Let `z_d` be the standardized PCA state on trading day `d`. A practical first
daily model is

```text
z_(d+1) = A z_d + epsilon_z,(d+1)

r_(d+1) = alpha
          + beta' z_d
          + gamma' epsilon_z,(d+1)
          + epsilon_S,(d+1),
```

where `r` is the signed log return of the SPX/forward over the day. The
`gamma` term captures contemporaneous leverage and spot/surface-shock
correlation. Estimate the complete residual vector
`[epsilon_z, epsilon_S]`, not only marginal variances.

For a Gaussian benchmark, the same relation can be written in continuous time:

```text
dz = -K z dt + L_z dW_z
dlog(S) = mu(z) dt + b' dW_z + sigma_perp dW_perp.
```

The existing matrix-exponential AR supplies the conditional surface mean
`exp(-K dt) z`; it still needs an innovation covariance. The joint model adds
the covariance between those innovations and spot returns.

Given an estimated joint Gaussian one-week distribution, drawing a surface
endpoint first and then spot is mathematically valid using

```text
E[R | Z] = mu_R + Sigma_RZ Sigma_ZZ^(-1) (Z - mu_Z),

Var[R | Z] = Sigma_RR - Sigma_RZ Sigma_ZZ^(-1) Sigma_ZR.
```

This answers the conditional-spot question directly. In implementation,
however, daily joint simulation is preferable because options can expire before
the one-week endpoint.

Gaussian innovations are only a baseline. Equity return/surface shocks are
heavy-tailed and asymmetric. The next benchmark should resample historical
joint residual blocks, optionally conditioned on the current volatility regime.
This preserves observed leverage and tail co-movement without estimating a
large parametric copula from a small sample. A multivariate Student-t innovation
model is a useful intermediate comparison.

## One-week scenario generation

For decision date `t` and horizon `H` of five trading sessions:

1. initialize the physical factor state at `z_t` and spot/forward at their
   observed values;
2. draw five daily joint spot/surface innovations, respecting holidays and
   actual calendar gaps;
3. update the state using the matrix exponential for each gap;
4. decode and validate an arbitrage-free surface at each required day;
5. update the forward-relative strike of every fixed absolute contract strike;
6. settle contracts on their actual expiration day; and
7. price surviving contracts at the horizon with their remaining TTE.

Endpoint-only sampling is acceptable only when every traded contract survives
past the horizon and no path-dependent constraint is present.

## Options expiring before the horizon

An option that expires before one week cannot be filled by interpolating its
option value to the horizon. For a European cash-settled option with expiry
`tau <= H`, scenario P&L must use

```text
call settlement = discount_to_horizon * max(S_tau - K, 0)
put settlement  = discount_to_horizon * max(K - S_tau, 0).
```

The settlement cash is carried from the option's expiry to the trading horizon.
This requires the simulated spot at the intermediate expiry, which is why daily
paths are the clean solution. If only a weekly endpoint model is available, an
explicit conditional bridge for spot is required; using `S_H` in place of
`S_tau` is inconsistent.

The initial scope should remain European SPX options. American exercise and
physical delivery require an exercise policy and stock/funding account and are
outside the first experiment.

## Actual-contract data requirement

The current balanced state panel stores forward-relative strikes but not all
contract metadata needed for real trading. The trading universe must preserve:

- contract identifier;
- option type;
- absolute strike;
- exact expiration date and settlement convention;
- observed bid, ask, sizes, and quote timestamp;
- contract multiplier;
- current forward and discount factor for its expiry; and
- the mapping from the listed contract to the normalized AF surface.

At a future scenario date the contract's absolute strike is unchanged, but its
forward-relative strike is

```text
scenario_K_over_F = absolute_strike / scenario_forward.
```

Repricing the old `K/F` would accidentally turn a fixed listed contract into a
floating-strike option.

The present state model only reaches 120 business days. The trading universe
must initially exclude contracts whose remaining maturity is outside the
scenario surface support, or the state grid must be extended before claiming
to trade all listed expiries.

## Normalised, actual, and ITM runs

Use current expiry-specific forward moneyness:

```text
ITM contracts: calls with K < F_t(T), puts with K > F_t(T)
OTM contracts: calls with K > F_t(T), puts with K < F_t(T).
```

The three specifications are:

- `ITM`: trade the ITM contract and divide by its own executable price;
- `actual`: trade the OTM contract and divide by its own executable price; and
- `normalised`: trade the same OTM contract as `actual`, but divide by the
  executable price of its paired ITM contract at the same strike and expiry.

Both option universes contain calls and puts. Classify a contract once at the
decision time; do not change universe membership when it crosses the scenario
forward later. Exclude exact-ATM contracts initially. Apply the same dates,
scenarios, utilities, risk aversions, quote filters, capital unit, and risk
constraints in all runs.

Use gross-cost dollar coordinates. Here **cost means the divider used to turn
an option return into a dollar position**; it does not replace the option's
actual executable entry price.

Use the following side-specific dividers:

```text
run / contract                  long divider       short divider
ITM / ITM option                own ask            own bid
actual / OTM option             own ask            own bid
normalised / OTM call, K > F    ITM put ask        ITM put bid
normalised / OTM put, K < F     ITM call ask       ITM call bid.
```

This follows, at theoretical mids, from

```text
OTM call = ITM put  + D(F - K),  K > F
OTM put  = ITM call - D(F - K),  K < F.
```

Let `a_i` and `b_i` be the actual ask and bid of the option being traded,
`g_i^long` and `g_i^short` its dividers from the table, `M_i` its contract
multiplier, and `G` the cash growth factor to the scenario horizon. Define
nonnegative long gross-cost dollars `l_i` and short gross-cost dollars `s_i`.
They correspond to

```text
long contracts  = l_i / (M_i g_i^long)
short contracts = s_i / (M_i g_i^short).
```

If `V_(i,m)^sell` is the scenario bid or carried settlement received by a long,
and `V_(i,m)^buy` is the scenario ask or carried settlement paid by a short,
then

```text
P&L_m(l, s) = sum_i [
    l_i * (V_(i,m)^sell - G a_i) / g_i^long
  + s_i * (G b_i - V_(i,m)^buy) / g_i^short
].
```

Thus the divider is the assumed gross cost, while actual OTM bid/ask still
determines the executable P&L. A paired ITM and OTM option now have the same
gross-dollar scale and contract count for a given side. This removes the cheap
OTM denominator as the source of artificial contract leverage, but does not
remove tail-model or short-option risk. Both paired quotes must pass quality
filters, and put-call-parity residuals should be reported. Contract-count,
margin, Greek, and worst-scenario-loss limits remain mandatory.

## Entry and exit prices

Observed entry quotes make the initial cash flow executable:

- long positions enter at the actual ask;
- short positions enter at the actual bid.

At the scenario horizon, a model mid is sufficient for a mark-to-model
objective but not for a liquidation objective. A conservative liquidation
objective should value:

- a long position at a scenario bid; and
- a short position at a scenario ask.

Because future quotes are unknown, choose and label one of:

1. reuse the current half-spread in normalized or volatility units;
2. predict the scenario half-spread from contract and scenario state;
3. use a fixed upper-quantile half-spread stress; or
4. optimize mark-to-model but report a separate liquidation stress.

Using actual entry quotes does not eliminate the future-spread problem unless
every option is held to settlement.

## OCE optimization

At decision time `t`, let `x_t` contain the observed market state and let the
conditional scenario engine produce fixed outcomes and physical weights
`{scenario_m(x_t), p_m(x_t)}`. For normalized scenario P&L `X_m(l,s; x_t)`
from long and short dollar allocations and concave utility `u_gamma`, the
weighted sample OCE is

```text
OCE_(u,gamma)(X(l,s); x_t)
    = max_eta { eta + sum_m p_m(x_t) u_gamma(X_m(l,s; x_t) - eta) }.
```

For every current state and each universe, solve the following six
deterministic problems:

```text
utility       in {power-exponential, entropy}
risk aversion in {1, 10, 100}

((l*,s*)_(t,u,gamma), eta*_(t,u,gamma))
    = argmax_(l,s,eta) OCE_(u,gamma)(X(l,s); x_t)
subject to trading, capital, and risk constraints.
```

Use the repository's existing utility convention:

```text
entropy: u_gamma(x) = (1 - exp(-gamma x)) / gamma

power-exponential:
    u_gamma(x) = (1 - exp(-gamma x)) / gamma,  x >= 0
                 x - gamma x^2 / 2,            x < 0.
```

Both have `u(0) = 0`, `u'(0) = 1`, and local absolute risk aversion
`-u''(0) / u'(0) = gamma`. Thus `gamma` has inverse-P&L units. Apply it to a
dimensionless P&L,

```text
X_m(l,s; x_t) = net_P&L_m(l,s; x_t) / B,
```

where `B` is one fixed, ex-ante portfolio capital/risk unit. The same `B` must
be used across dates, universes, and both utilities. The reported strategy is
in dollars while the utility sees return per dollar of fixed risk budget.
Applying `1`, `10`, or `100` directly to raw dollar P&L would not have a stable
economic interpretation.

The conditional law remains probabilistic, but the optimization must see a
frozen finite scenario problem. Use predeclared quadrature/quasi-Monte-Carlo
nodes, or fixed innovation draws and weights, and reuse the identical scenarios
for all six utility configurations. This makes results repeatable and ensures
that differences across utilities and risk aversion are not Monte Carlo noise.

With fixed scenario prices, linear portfolio P&L, concave utility, and convex
constraints, this is a tractable convex optimization after choosing consistent
sign conventions. Actual bid/ask entry costs are piecewise linear in position
sign and remain compatible with a convex formulation. Integer contract counts
or nonlinear market impact turn the final problem into a mixed or nonlinear
problem and should be deferred until the continuous solution is understood.

At minimum impose:

- gross and net premium/cash limits;
- maximum long and short contracts per option;
- scenario loss or margin limits;
- aggregate delta, gamma, and vega limits;
- expiry and strike concentration limits;
- quote-quality and liquidity filters; and
- a no-trade alternative.

Without these constraints, a finite scenario set and small surface-model errors
can make the OCE choose extreme short-option or wing positions.

Report the no-trade OCE and incremental OCE for every solve. Do not compare raw
OCE levels across the two utility families as if their global shapes were the
same; compare their selected portfolios, scenario P&Ls, realized outcomes, and
improvement over no trade. Other OCE families are outside the initial scope.

## Look-ahead and sample size

The existing 2025 PCA and AR estimates use the full 163-date panel. They are
diagnostic estimates and cannot be used unchanged in a historical trading
backtest. For each decision date `t`:

- the NMS support and state definition must be fixed ex ante;
- PCA loadings, means, AR parameters, and residual distributions must use only
  dates available before `t`; and
- model selection and risk-aversion calibration must not use the test period.

The current common finite support was selected using the complete annual
sample, which is also a small form of look-ahead. A real-trading panel needs a
predeclared support or a rolling rule that never inspects future dates.

Eight months and 163 observations are too few for reliable tail-sensitive OCE
optimization with a joint multi-factor model. The first serious backtest should
use several years of daily markets. The 2025 panel remains suitable for a
pipeline prototype and numerical checks.

## Minimum viable experiment

1. Build matched forward-ITM and forward-OTM call/put pairs inside the modeled
   maturity/NMS support, then form the `normalised`, `actual`, and `ITM` runs.
2. Use one authoritative three-to-five-factor log-DLV state.
3. Fit the daily joint linear model for DLV factors and SPX log returns in a
   rolling or expanding window.
4. Estimate joint innovation covariance and retain the empirical residuals.
5. Generate five-day paths with both Gaussian innovations and joint residual
   block bootstrap.
6. Reconstruct arbitrage-free SANOS surfaces along every path.
7. Settle early-expiring options on the simulated daily spot path and reprice
   survivors at the horizon.
8. Use actual bid/ask for entry and a declared conservative exit convention.
9. Express long and short positions in gross-cost dollars using the divider
   specified by the run.
10. Freeze the conditional scenarios and, for each universe, solve
    power-exponential and entropy OCEs at risk aversion `1`, `10`, and `100`,
    with one declared P&L normalization and strict Greek, premium,
    scenario-loss, and position constraints.
11. Compare the chosen portfolio with no-trade, delta-hedged, and simple
    volatility-spread benchmarks on untouched future dates.

## Diagnostics

Before optimizing trades, validate:

- one-day and five-day factor forecast errors;
- spot/factor innovation correlations and leverage signs;
- return and surface-shock tail dependence;
- generated versus realized weekly factor and return distributions;
- static and calendar arbitrage for every decoded surface;
- repricing of actual contracts at observed future markets;
- settlement treatment for every pre-horizon expiry;
- scenario convergence of OCE and optimal positions; and
- sensitivity to exit spreads, risk aversion, number of factors, and residual
  distribution.

## Relevant published work

- [Cont and da Fonseca (2002), *Dynamics of Implied Volatility
  Surfaces*](https://doi.org/10.1088/1469-7688/2/1/304) show that index implied-
  volatility surfaces are driven by a small number of factors and study their
  correlation with the underlying. This directly supports a joint low-rank
  spot/surface model.

- [Goncalves and Guidolin (2006), *Predictable Dynamics in the S&P 500 Index
  Options Implied Volatility Surface*](https://doi.org/10.1086/500686) model
  surface coefficients dynamically and find predictability, but also find that
  trading profits disappear when transaction costs are increased and broad
  surface segments are traded. This is a strong warning to use actual bid/ask
  prices and a no-trade alternative.

- [Ben-Tal and Teboulle (2007), *An Old-New Concept of Convex Risk Measures:
  The Optimized Certainty Equivalent*](https://doi.org/10.1111/j.1467-9965.2007.00311.x)
  develop the OCE as a flexible family connecting utility and convex risk
  measures and provide the mathematical basis for the proposed objective.

- [Cohen, Reisinger, and Wang (2023), *Simulation of Arbitrage-Free Implied
  Volatility Surfaces*](https://doi.org/10.1080/1350486X.2023.2277960) focus on
  dynamic surface scenarios that retain static-arbitrage constraints. This is
  relevant to the decode-and-project stage.

- [Vuletic and Cont (2025), *VolGAN: A Generative Model for Arbitrage-Free
  Implied Volatility Surfaces*](https://doi.org/10.1080/1350486X.2025.2471317)
  jointly simulate arbitrage-free SPX implied-volatility surfaces and the
  underlying. It is a useful advanced benchmark after the transparent joint
  AR/residual-bootstrap model is established.

## Implemented 2025 diagnostic

`experiment.py` implements the first deterministic prototype. Recreate it from
the repository root with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.experiment \
  --scenario-paths 256 \
  --output-dir quantdata/dysanos/localvolbysanos/real_trading/results/2025
```

The run uses:

- all 163 available market dates from 2025-01-02 through 2025-08-29;
- the existing five-factor full-sample log-forward-volatility PCA and
  matrix-exponential AR;
- a joint Gaussian surface/forward-return innovation model;
- 256 fixed scrambled-Sobol paths over five trading sessions;
- up to eight expiries and two strikes per expiry wing;
- a $1 million risk-budget unit, 10% per-action cap, 100% gross cap, and a
  conservative 25% sum-of-standalone-worst-loss cap; and
- observed ask entry for longs and bid entry for shorts, followed by a
  scenario-model midpoint exit.

Every date supplied 30--32 matched strike/expiry pairs. All 2,934 solves
(`163 dates * 3 runs * 2 utilities * 3 risk aversions`) converged, with no
constraint violations and no simultaneous long/short position in one option.

Mean **model-scenario** one-week returns per $1 million risk-budget unit were:

| Utility | Risk aversion | ITM | actual | normalised |
|---|---:|---:|---:|---:|
| entropy | 1 | 19.05% | 51.13% | 24.32% |
| entropy | 10 | 8.29% | 11.17% | 9.71% |
| entropy | 100 | 1.59% | 2.24% | 2.07% |
| power-exp | 1 | 19.14% | 52.16% | 24.43% |
| power-exp | 10 | 10.21% | 14.27% | 11.96% |
| power-exp | 100 | 2.15% | 3.03% | 2.78% |

The `actual` divider exposes the cheap-OTM leverage directly; `normalised`
substantially reduces it. More importantly, the conservative loss constraint
binds in 2,921 of 2,934 solves. The displayed returns are therefore largely
constraint-driven conditional model outputs, not realized returns or evidence
of profitability.

This run is deliberately labeled a diagnostic, not a backtest. It estimates
PCA and dynamics on the complete 2025 panel, approximates forward returns with
SPX spot returns, ignores the discount-factor ratio over the week, decodes
future prices through the forward-volatility surface rather than a final
DLV/SANOS arbitrage projection, and uses overlapping one-week outcomes. A
trading conclusion requires a walk-forward fit and out-of-sample scenario
validation.

## Observed midpoint comparison

Recreate the realized comparison after the optimization run with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.realized \
  --results-dir quantdata/dysanos/localvolbysanos/real_trading/results/2025 \
  --output-dir quantdata/dysanos/localvolbysanos/real_trading/results/2025
```

The comparison tracks the exact absolute strike and stable AM/PM expiry ID.
It pays the observed ask for a long entry or receives the observed bid for a
short entry. A surviving contract exits at its observed bid/ask midpoint five
panel sessions later. A PM-settled contract expiring inside the horizon uses
intrinsic value at that date's SPX `spot_ref`; funding is ignored in both the
prediction and realized calculation.

Of 17,659 evaluated nonzero positions, 17,253 had a usable exit (97.7%). A
portfolio is reported only when all its positions were observed, leaving 2,568
of 2,844 comparable portfolios (90.3%). The 406 missing exits comprise 134
expiry-date spot observations absent from the panel and 272 future contracts
without the exact expiry/strike quote.

The table reports `entropy / power-exp` means over complete portfolios. Returns
are fractions of the $1 million risk-budget unit.

| Universe | Risk aversion | Predicted mean | Observed mean | Observed median | Correlation |
|---|---:|---:|---:|---:|---:|
| ITM | 1 | 18.80% / 18.90% | 6.11% / 6.16% | 0.12% / 0.22% | 0.03 / 0.03 |
| ITM | 10 | 8.26% / 10.11% | 1.50% / 1.71% | -0.55% / -0.26% | 0.47 / 0.34 |
| ITM | 100 | 1.67% / 2.26% | 0.57% / 0.84% | -0.05% / 0.00% | 0.96 / 0.94 |
| actual | 1 | 50.56% / 51.59% | 3.66% / 3.70% | -8.13% / -8.25% | -0.20 / -0.19 |
| actual | 10 | 11.21% / 14.26% | 1.96% / 2.24% | -1.19% / -1.23% | 0.44 / 0.31 |
| actual | 100 | 2.33% / 3.22% | 0.83% / 1.32% | -0.03% / -0.05% | 0.95 / 0.95 |
| normalised | 1 | 24.26% / 24.49% | 6.13% / 6.28% | -1.17% / -1.00% | -0.06 / -0.06 |
| normalised | 10 | 9.76% / 11.70% | 1.62% / 1.44% | -0.84% / -0.68% | 0.40 / 0.24 |
| normalised | 100 | 2.15% / 2.94% | 0.71% / 1.10% | 0.02% / 0.02% | 0.94 / 0.93 |

The model overpredicts mean returns in every configuration. At risk aversion
1, forecast/realized correlation is near zero or negative and the `actual`
OTM run has an observed median loss around 8%. Risk aversion 100 has high raw
return correlation, but all forecasts are positive and position sizes vary by
date, so this alone is not evidence of directional predictability. The median
observed return remains approximately zero. These are in-sample diagnostics
with overlapping horizons and must not be read as backtest performance.

## Spot and convexity attribution

Recreate the return attribution with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.attribution
```

For each of the 18 complete-portfolio series, the script joins the observed
five-session SPX `spot_ref` return and estimates

```text
R_portfolio = alpha + beta R_SPX + residual
```

with four-lag Bartlett/Newey-West standard errors for the overlapping weekly
returns. It also estimates a diagnostic quadratic regression on `R_SPX` and
`R_SPX^2`. The latter separates a directional/delta-like component from a
large-move/convexity component. It is still a statistical attribution, not a
Greek P&L explain.

The directional regression explains only 0--24% of return variance, and no
directional beta has an absolute HAC t-statistic above 1.9. Adding squared spot
return raises explained variance to 31--68%; every squared-return coefficient
has a HAC t-statistic between 2.75 and 7.38. The mean quadratic attribution,
reported as `entropy / power-exp`, is:

| Universe | Risk aversion | Mean return | Linear spot | Convexity | Spot-neutral residual |
|---|---:|---:|---:|---:|---:|
| ITM | 1 | 6.11% / 6.16% | 0.38% / 0.38% | 6.04% / 6.01% | -0.31% / -0.23% |
| ITM | 10 | 1.50% / 1.71% | 0.46% / 0.50% | 2.52% / 2.77% | -1.48% / -1.55% |
| ITM | 100 | 0.57% / 0.84% | 0.02% / 0.06% | 1.31% / 1.84% | -0.76% / -1.06% |
| actual | 1 | 3.66% / 3.70% | 0.40% / 0.40% | 9.63% / 9.71% | -6.37% / -6.41% |
| actual | 10 | 1.96% / 2.24% | -0.27% / 0.56% | 5.49% / 3.50% | -3.27% / -1.82% |
| actual | 100 | 0.83% / 1.32% | 0.07% / 0.14% | 1.80% / 2.67% | -1.04% / -1.49% |
| normalised | 1 | 6.13% / 6.28% | 0.26% / 0.23% | 8.45% / 8.67% | -2.58% / -2.62% |
| normalised | 10 | 1.62% / 1.44% | -0.21% / 0.99% | 4.73% / 2.97% | -2.90% / -2.52% |
| normalised | 100 | 0.71% / 1.10% | -0.07% / 0.08% | 1.58% / 2.20% | -0.80% / -1.19% |

The unconditional spot regression is incomplete because each optimized
portfolio has a different entry delta. The script therefore also computes the
mid-implied entry forward delta using the actual optimized quantity and defines

```text
entry-delta spot P&L = net entry forward delta * realized SPX return.
```

Net entry cost uses executable prices: long asks are positive premium debits
and short bids are negative premium credits. Gross cost ignores that sign. A
net delta of `1` means that a 1% relative forward move produces approximately
1% of the $1 million risk-budget unit, before gamma and volatility effects.

| Universe | Risk aversion | Net delta | Net cost | Gross cost | Entry-delta P&L | Residual P&L |
|---|---:|---:|---:|---:|---:|---:|
| ITM | 1 | 0.89 / 0.82 | 23.05% / 23.02% | 25.78% / 25.75% | 9.42% / 9.52% | -3.30% / -3.36% |
| ITM | 10 | 0.45 / 0.60 | 18.01% / 18.02% | 22.81% / 22.91% | 2.55% / 2.74% | -1.04% / -1.03% |
| ITM | 100 | 0.17 / 0.26 | 2.98% / 3.10% | 14.83% / 14.84% | 0.79% / 1.17% | -0.22% / -0.33% |
| actual | 1 | 5.87 / 6.17 | 23.07% / 23.06% | 26.03% / 26.03% | 10.71% / 10.82% | -7.05% / -7.12% |
| actual | 10 | 0.68 / 0.48 | 11.35% / 12.84% | 18.75% / 19.75% | 3.22% / 3.82% | -1.26% / -1.58% |
| actual | 100 | 0.24 / 0.39 | 2.98% / 2.59% | 12.10% / 12.36% | 1.15% / 1.77% | -0.33% / -0.44% |
| normalised | 1 | 1.56 / 1.57 | 22.95% / 22.92% | 25.89% / 25.84% | 9.62% / 9.76% | -3.49% / -3.48% |
| normalised | 10 | 0.61 / 0.67 | 12.06% / 13.49% | 18.47% / 19.69% | 2.94% / 3.43% | -1.32% / -1.98% |
| normalised | 100 | 0.22 / 0.37 | 3.14% / 3.04% | 11.86% / 11.84% | 1.06% / 1.53% | -0.34% / -0.43% |

The entry-delta component has 55--96%
correlation with realized returns and explains 30--92% of their variance. It is
larger than total mean P&L in all 18 series; the remaining gamma, volatility,
theta, spread, and interaction component is negative on average. Thus the
portfolios are materially long, time-varying market delta. The bare-return
regression understated this because it treated every date as having the same
exposure.

To distinguish a delta-sizing signal from fixed long-SPX exposure, decompose

```text
delta_t * R_SPX
  = mean(delta) * R_SPX
  + (delta_t - mean(delta)) * R_SPX.
```

The first term is a fixed-size long-SPX benchmark; the second is the incremental
in-sample delta-timing return. A second fixed-SPX benchmark is scaled to the
same population standard deviation as the complete realized option strategy.
Funding and SPX trading costs are ignored.

| Universe | Risk aversion | Dynamic delta P&L | Fixed-average-delta P&L | Timing P&L | HAC timing t | Full strategy / risk-matched SPX |
|---|---:|---:|---:|---:|---:|---:|
| ITM | 1 | 9.42% / 9.52% | 0.31% / 0.29% | 9.11% / 9.23% | 3.11 / 3.11 | 6.11% / 6.16% / 3.03% / 3.05% |
| ITM | 10 | 2.55% / 2.74% | 0.20% / 0.27% | 2.34% / 2.47% | 2.16 / 2.21 | 1.50% / 1.71% / 1.69% / 2.04% |
| ITM | 100 | 0.79% / 1.17% | 0.10% / 0.16% | 0.69% / 1.01% | 1.37 / 1.43 | 0.57% / 0.84% / 0.86% / 1.20% |
| actual | 1 | 10.71% / 10.82% | 2.79% / 2.93% | 7.92% / 7.89% | 2.53 / 2.47 | 3.66% / 3.70% / 6.67% / 6.80% |
| actual | 10 | 3.22% / 3.82% | 0.43% / 0.25% | 2.79% / 3.56% | 2.09 / 2.39 | 1.96% / 2.24% / 3.06% / 2.88% |
| actual | 100 | 1.15% / 1.77% | 0.17% / 0.27% | 0.99% / 1.50% | 1.36 / 1.41 | 0.83% / 1.32% / 1.35% / 1.92% |
| normalised | 1 | 9.62% / 9.76% | 0.83% / 0.81% | 8.79% / 8.95% | 3.04 / 3.08 | 6.13% / 6.28% / 5.87% / 5.73% |
| normalised | 10 | 2.94% / 3.43% | 0.44% / 0.45% | 2.50% / 2.98% | 2.07 / 2.17 | 1.62% / 1.44% / 3.09% / 2.90% |
| normalised | 100 | 1.06% / 1.53% | 0.17% / 0.26% | 0.89% / 1.27% | 1.46 / 1.46 | 0.71% / 1.10% / 1.22% / 1.67% |

The timing component is positive and overlap-HAC significant in-sample for
risk aversion 1 and 10, but not 100. That is evidence that time-varying delta
contains more than a constant long-market position inside this fitted sample.
It is not evidence of a real tradable signal: the PCA, transition, and joint
spot/surface model were estimated on the complete 2025 sample, and 18 related
strategies were inspected. HAC errors address overlapping returns but not
look-ahead or selection bias.

For total strategy performance, only the risk-aversion-1 ITM strategies beat
the volatility-matched fixed-SPX benchmark clearly; the normalised versions
are only slightly ahead. The `actual` OTM strategy and every risk-aversion-10
or 100 strategy underperform fixed SPX on this mean/standard-deviation basis.
Thus the current result is largely long-delta exposure plus a promising but
unverified in-sample timing effect, not established alpha.

## Option-implied spot-only strategy

The spot-only test removes option P&L entirely while retaining the option
optimizer as a signal generator. Recreate it with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.spot_only
```

For every decision date and optimized option portfolio:

1. compute each option position's mid-implied entry forward delta;
2. replace all options by their signed SPX delta and net them into one entry
   spot trade;
3. assign each delta slice to its option's expiry when that expiry is inside
   the holding week, otherwise to the five-session horizon;
4. close each slice at the corresponding SPX `spot_ref`; and
5. do not rebalance between entry and the assigned close date.

The calculation has no option trades, spot spread, commissions, funding,
borrow, or market impact. Of 2,844 cohorts, 2,790 have every required SPX
observation. The raw implied-delta results are `entropy / power-exp`:

| Universe | Risk aversion | Mean | Median | Standard deviation | Win rate | Mean absolute spot notional |
|---|---:|---:|---:|---:|---:|---:|
| ITM | 1 | 7.62% / 7.67% | 4.74% / 4.93% | 16.80% / 16.99% | 71.6% / 71.6% | 8.53 / 8.72 |
| ITM | 10 | 2.51% / 2.84% | 0.61% / 0.86% | 8.43% / 9.04% | 74.2% / 70.3% | 1.54 / 1.77 |
| ITM | 100 | 0.69% / 1.02% | 0.09% / 0.11% | 4.09% / 5.48% | 68.4% / 69.0% | 0.25 / 0.38 |
| actual | 1 | 8.53% / 8.65% | 5.49% / 5.51% | 18.78% / 19.02% | 68.4% / 69.0% | 9.63 / 9.89 |
| actual | 10 | 2.89% / 3.21% | 0.74% / 0.86% | 10.19% / 10.98% | 71.6% / 72.3% | 1.62 / 1.94 |
| actual | 100 | 0.88% / 1.37% | 0.07% / 0.10% | 5.31% / 7.31% | 68.4% / 68.4% | 0.31 / 0.50 |
| normalised | 1 | 7.91% / 8.02% | 4.51% / 4.54% | 17.63% / 17.87% | 71.6% / 71.6% | 8.54 / 8.70 |
| normalised | 10 | 2.60% / 2.94% | 0.72% / 0.80% | 8.78% / 9.71% | 71.6% / 71.0% | 1.52 / 1.76 |
| normalised | 100 | 0.77% / 1.19% | 0.10% / 0.12% | 4.33% / 5.99% | 69.0% / 70.3% | 0.29 / 0.47 |

Notional is per $1 million risk-budget unit. The risk-aversion-1 signals use
8.5--9.9 times mean absolute SPX notional, with 95th-percentile scheduled
notional around 17--20 times; these raw returns require leverage and are not
cash-capital comparable.

To remove leverage magnitude, divide every cohort's P&L by its total absolute
scheduled SPX delta. The resulting unit-gross weekly signal has:

| Statistic across all 18 specifications | Range |
|---|---:|
| Mean return | 0.62% to 0.99% |
| Median return | 0.48% to 0.62% |
| Standard deviation | 1.81% to 2.21% |
| Win rate | 68.4% to 74.2% |
| Four-lag HAC mean t-statistic | 2.87 to 3.75 |

Unlike the option strategy, removing the best return decile leaves a positive
mean in every raw spot-only specification. The spot-only mean also exceeds the
mean of a volatility-matched fixed long-SPX position in all 18 fitted-sample
runs and exceeds the corresponding realized option return on their common
dates. This is substantially cleaner evidence for a delta-sizing signal than
the option P&L.

It remains an in-sample diagnostic, not a backtest. The option optimizer that
generates the delta was fitted using the complete 2025 panel. The next required
test is the identical spot-only construction with expanding-window PCA,
dynamics, and residual estimates, plus a predeclared notional cap and realistic
funding or futures implementation.

## Delta-constrained OCE and continuous spot sizing

The leverage-safe variant applies the spot-notional limit inside the OCE
optimization rather than normalizing the resulting trade afterward. For action
weights `a` and signed entry forward-delta coefficients `delta`, it retains the
existing gross-allocation, position, and conservative-loss constraints and adds

```text
-1 <= delta @ a <= 1
```

where one unit is the configured $1 million risk budget. Each delta coefficient
uses the option's mid-implied entry forward delta, its long/short sign, and the
same side-specific cost divider used by the scenario-return column. Recreate
the constrained optimization and its one-day spot evaluation with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.experiment \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m \
  --max-abs-net-delta-fraction 1

/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.realized \
  --results-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m

/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.spot_only \
  --positions-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m/positions.parquet \
  --results-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m/oce_results.parquet \
  --realized-comparison-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m/realized_comparison.parquet \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_delta_cap_1m/spot_1d \
  --horizon-sessions 1 --hac-lags 0
```

The reported continuous-strategy return is P&L divided by the full $1 million
capital, not by absolute deployed notional. Thus an optimized $250,000 spot
position contributes one quarter of the corresponding signed SPX return.

For the representative `normalised / entropy / risk aversion 10` strategy, the
mean absolute position is $780,749, the median absolute position is $1 million,
and the cap binds on 59.5% of dates. The strategy is long on 57.7% and short on
42.3% of optimization dates. Its 162 realized one-session trades have:

| Statistic | Constrained continuous size | Post-normalized sign only |
|---|---:|---:|
| Mean daily return | 0.248% | 0.264% |
| Daily standard deviation | 1.208% | 1.340% |
| Win rate | 58.0% | 58.0% |
| Annualized Sharpe | 3.25 | 3.13 |
| Mean daily P&L | $2,476 | $2,638 |
| Fixed-notional sample P&L | $401,174 | $427,435 |

The cap therefore reduces mean P&L and volatility, with a modest Sharpe
improvement for this representative specification. It is not a uniform
improvement: only 5 of the 18 inspected specifications improve on their
sign-only Sharpe. Across all 18, the continuous one-day annualized Sharpe is
1.90--3.45. In particular, the risk-aversion-100 models usually shrink exposure
substantially and lose Sharpe. The sizing signal is therefore weaker and less
stable than the directional signal.

All 163 dates optimize successfully. The delta recomputed independently from
the persisted positions agrees with the optimizer's stored net delta to within
`9e-15`, and no stored portfolio breaches the configured entry cap. These are
still full-sample-fitted results with midpoint SPX execution and no costs,
funding, or market impact.

## Sequential $1 million exposure and retained P&L

The sequential variant carries the actual spot holding instead of treating
each daily signal as a separate cohort. Let `n[t-1]` be the signed spot
notional as a fraction of $1 million after yesterday's rebalance. At today's
observed spot, first calculate

```text
carried_before_cap = n[t-1] * spot[t] / spot[t-1]
daily_pnl          = n[t-1] * (spot[t] / spot[t-1] - 1)
carried            = clip(carried_before_cap, -1, 1)
```

`daily_pnl` is transferred to the retained-P&L ledger and is not reinvested.
The active spot book therefore remains subject to the fixed $1 million cap
regardless of accumulated profit or loss. The difference between
`carried_before_cap` and `carried` is the explicit cap-reduction trade.

For sign-only, today's final target is exactly +$1 million or -$1 million. In
the absence of transaction costs, this sequential accounting has exactly the
same daily P&L as the earlier one-day sign-return calculation; it now makes the
position and retained-P&L accounts explicit.

For continuous sizing, the clipped carried spot is a fixed, nontradable
scenario endowment. For simulated one-week spot return `r_spot`, option-action
return matrix `R`, signed action delta `delta`, and nonnegative virtual action
weights `a`, solve

```text
maximize OCE(carried * r_spot + R @ a)
subject to -1 <= carried + delta @ a <= 1
```

The virtual options remain a signal generator and are not executed. The new
actual spot target is `carried + delta @ a`, and the actual trade is the
difference between that target and `carried_before_cap`. Thus previous spot
cannot be optimized away directly; the OCE can only choose an incremental
option-implied delta inside the remaining capacity.

Recreate the stateful run and comparison with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.experiment \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_sequential_cap_1m \
  --carry-spot-position --max-abs-net-delta-fraction 1

/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.sequential \
  --continuous-results-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_sequential_cap_1m/oce_results.parquet \
  --sign-comparison-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025/one_day/spot_only_comparison.parquet \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_sequential_cap_1m
```

For `normalised / entropy / risk aversion 10`, the 162 realized daily returns
are:

| Statistic | Sequential continuous OCE | Sequential sign-only |
|---|---:|---:|
| Retained P&L | $406,306 | $427,435 |
| Mean daily P&L | $2,508 | $2,638 |
| Daily P&L standard deviation | $11,898 | $13,402 |
| Annualized Sharpe | 3.35 | 3.13 |
| Maximum drawdown on $1 million plus retained P&L | -4.61% | -5.81% |
| Mean absolute active spot | $777,627 | $1,000,000 |
| Mean absolute daily trade | $332,276 | $395,270 |
| Cap-binding fraction after rebalance | 59.5% | 100% |
| Number of market-move cap reductions | 59 | 92 |
| Terminal active spot | -$1,000,000 | -$1,000,000 |

The continuous version earns less retained P&L but reduces volatility,
drawdown, and turnover enough to improve the representative Sharpe by 0.22.
The mechanical cap-reduction trades transfer a net $251,315 to retained cash
in the continuous run and $306,466 in sign-only; total retained P&L also
contains gains and losses realized when the next target changes within the cap.

Across all 18 specifications, sequential continuous Sharpes are 1.94--3.48 and
7 of 18 improve on sign-only. The risk-aversion-100 sizing remains materially
weaker than its sign signal. This reinforces the earlier conclusion: direction
looks stronger than continuous signal magnitude. All 2,934 stateful OCE solves
succeed, and every post-rebalance target respects the $1 million cap.

## Independent daily loss diagnostics

The independent one-day sign strategy is also analyzed as a daily panel. The
return regression outcome is the signed next-session SPX return, and the
classification outcome is one when that return is negative. The requested
variables are first fitted one at a time and then jointly:

- next-session SPX return squared;
- front-week ATM implied variance scaled to one day, `ATM vol^2 / 252`;
- today's SPX return; and
- today's SPX return squared.

The first variable is only known after entry. It is an ex-post explanation of
whether the strategy made more money in large realized moves, not a predictor
that can be used to trade. The other three are available at the signal date.
OLS coefficients use standardized predictors and HC1 robust errors. Logistic
loss classifiers are reported both in-sample and with an expanding window
after the first 60 dates; the expanding fits are L2 regularized.

For `normalised / entropy / risk aversion 10`, there are 161 observations and
the loss rate is 42.2%. The individual regressions are:

| Individual variable | P&L coefficient per 1 SD | Robust t | OLS R2 | Loss-logit coefficient | Expanding AUC |
|---|---:|---:|---:|---:|---:|
| Next-day return squared | +0.905% | 9.26 | 45.4% | -0.574 | 0.406 |
| ATM one-day variance | +0.413% | 1.24 | 9.5% | -0.015 | 0.375 |
| Today's return | -0.358% | -2.30 | 7.1% | +0.137 | 0.464 |
| Today's return squared | -0.122% | -0.62 | 0.8% | +0.143 | 0.371 |

The next-day squared-return P&L coefficient is positive in all 18 strategy
specifications and significant at 5% in 16. Thus the signal earns more during
large next-day moves. Its loss-logit coefficient is negative in all 18, but is
significant in none: large moves appear somewhat more likely to be caught in
the correct direction, but the evidence is not precise enough to claim a
correct-sign indicator.

Among the variables available at entry, ATM variance has a positive P&L
coefficient in all 18 but is significant in none. Today's return has a negative
P&L coefficient in all 18 and is significant in 15, suggesting short-horizon
reversal in the signal's realized payoff. Yet none of the three ex-ante
univariate loss classifiers beats the expanding rolling-frequency baseline on
Brier score in any of the 18 specifications. For the representative joint
ex-ante classifier, expanding AUC is 0.409 and Brier score is 0.252 versus
0.243 for the baseline. These variables therefore explain some P&L variation
but do not yet provide a usable filter for whether the sign will be right.

Recreate the diagnostics with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.loss_diagnostics
```

Natural next variables should remain split by timing. Candidate predictors
available at entry are the AR-model forecast of next-day variance; signal
strength such as absolute optimal delta or predicted-return-to-scenario-risk;
sign agreement across utility/universe specifications; ATM term-structure
slope; DLV/PCA state and innovations known at entry; implied minus trailing
realized variance; 1/5/20-day momentum and drawdown; and quote-width or market
fit quality. Natural ex-post attribution variables are realized variance
surprise, next-day ATM/term-slope changes, next-day DLV/PCA shocks, overnight
versus intraday spot moves, and realized liquidity changes. The latter group
must never enter a tradable classifier.

## Daily CVaR objectives

Two lower-tail objectives extend the independent daily OCE experiment. The
labels follow the requested best-tail-exclusion convention rather than the
other common CVaR convention:

```text
cvar10 = discard the best 10% and maximize the mean of the worst 90%
cvar50 = discard the best 50% and maximize the mean of the worst 50%
```

For lower-tail mass `q` and scenario return `X`, the optimized certainty
equivalent is

```text
max_eta  eta - E[max(eta - X, 0)] / q.
```

The implementation solves the finite-scenario problem as an exact linear
program, including the existing gross-allocation, position, conservative-loss,
and ±$1 million delta constraints. Scenario paths still span one week. A new
portfolio is optimized independently each date, and its option-implied delta
is tested over the next observed SPX session at midpoint with no costs.

Recreate the two objectives and one-day spot test with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.experiment \
  --utilities cvar10 cvar50 \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_daily

/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.spot_only \
  --positions-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_daily/positions.parquet \
  --results-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_daily/oce_results.parquet \
  --realized-comparison-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025/realized_comparison.parquet \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_daily/one_day \
  --horizon-sessions 1 --hac-lags 0
```

All 978 optimizations across 163 dates, three universes, and two objectives
succeed. There are 162 next-session observations. The continuous delta results
are:

| Universe | Objective | Mean abs delta | Mean daily return | Daily std | Annualized Sharpe |
|---|---|---:|---:|---:|---:|
| ITM | Worst 90% | $928k | 0.256% | 1.311% | 3.10 |
| ITM | Worst 50% | $362k | 0.164% | 1.055% | 2.47 |
| Actual | Worst 90% | $967k | 0.216% | 1.338% | 2.56 |
| Actual | Worst 50% | $480k | 0.155% | 1.151% | 2.14 |
| Normalised | Worst 90% | $936k | 0.273% | 1.320% | 3.28 |
| Normalised | Worst 50% | $448k | 0.175% | 1.143% | 2.43 |

For normalised worst-90%, replacing the continuous size with a $1 million
long/short position gives a 61.1% win rate, 0.285% mean daily return, and 3.39
annualized Sharpe. Normalised worst-50% gives 55.6%, 0.243%, and 2.87. For a
matched comparison, the earlier capped normalised entropy/gamma-10 run gives
58.0%, 0.264%, and 3.13 for sign-only, and 0.248% mean with 3.25 Sharpe for
continuous sizing. Worst-90% is therefore modestly better in this full-sample
diagnostic, while worst-50% is more conservative but has lower realized
one-day performance.

The right-tail caveat remains: removing the best realized decile makes mean
continuous return negative for all six CVaR series. These are full-sample model
diagnostics, not look-ahead-free estimates.

### Stateful $1 million CVaR carry

The two CVaR objectives also use the sequential accounting described above.
Yesterday's spot target is marked at today's observed SPX level, excess market-
move exposure outside ±$1 million is closed into retained cash, and the clipped
position enters today's CVaR optimization as a fixed nontradable endowment.
Only incremental option-implied delta is selectable. Daily P&L is retained and
not reinvested.

Recreate the stateful run and sign comparison with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.experiment \
  --utilities cvar10 cvar50 \
  --carry-spot-position --max-abs-net-delta-fraction 1 \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_sequential_cap_1m

/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.sequential \
  --continuous-results-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_sequential_cap_1m/oce_results.parquet \
  --sign-comparison-file \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_daily/one_day/spot_only_comparison.parquet \
  --output-dir \
  quantdata/dysanos/localvolbysanos/real_trading/results/2025_cvar_sequential_cap_1m
```

For the normalised universe:

| Objective | Strategy | Retained P&L | Annualized Sharpe | Max drawdown | Mean abs spot | Mean daily trade |
|---|---|---:|---:|---:|---:|---:|
| Worst 90% | Continuous carry OCE | $431,762 | 3.22 | -5.73% | $933k | $359k |
| Worst 90% | Sign-only | $461,787 | 3.39 | -5.04% | $1,000k | $395k |
| Worst 50% | Continuous carry OCE | $296,106 | 2.49 | -4.67% | $454k | $287k |
| Worst 50% | Sign-only | $394,285 | 2.87 | -5.85% | $1,000k | $445k |

Worst-90% continuous carry binds the cap on 87.1% of dates and has 81 market-
move cap clips. Worst-50% binds on 35.0% and has 36 clips. Thus worst-50%
achieves the intended exposure, turnover, and drawdown reduction, but at a
large cost in retained P&L and Sharpe. For worst-90%, post-normalizing to the
sign is better on P&L, Sharpe, and drawdown; continuous sizing only reduces
turnover modestly.

All 978 stateful optimizations succeed. `ITM / cvar50` has two independent
dates with exactly zero delta. In the sequential sign comparator, zero means
no new directional instruction, so the preceding ±$1 million sign is retained;
all other series have no zero signals.

## Entry-time tail-risk overlays

The capped independent one-day entropy, power-exponential, and worst-90% CVaR
returns are tested against five entry-time exposure signals after an initial
60-day training window:

- front-week ATM volatility;
- absolute same-date spot return;
- the OCE scenario return standard deviation;
- the magnitude of the OCE scenario 5% downside; and
- agreement between the seven entropy, power-exponential, and CVaR signs.

For each positive risk measure `x`, the chronological overlay is

```text
exposure[t] = min(1, median(x before t) / x[t]).
```

Sign agreement is used directly as exposure. Every overlay is compared with a
constant position whose exposure equals that overlay's realized mean exposure.
This controls for the trivial tail reduction obtained merely by investing less.
Tail risk is the empirical mean of the worst 10% of next-session returns,
including fractional boundary mass.

Recreate the analysis with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.tail_risk_diagnostics
```

ATM volatility targeting is the only consistent tail reducer. It lowers worst-
decile expected shortfall in all 21 series: median reductions are 20.8% for
entropy, 22.1% for power-exponential, and 20.8% for worst-90% CVaR. It also
improves standard-deviation Sharpe in all 21. For the normalised representative
strategies over the 101 evaluation dates:

| Objective | Mean exposure | Overlay mean | Constant mean | Overlay ES10 | Constant ES10 | ES reduction | Overlay/constant Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy, gamma 10 | 90.3% | 0.206% | 0.291% | -0.880% | -1.105% | 20.3% | 4.62 / 3.83 |
| Power-exp, gamma 10 | 90.3% | 0.203% | 0.281% | -0.980% | -1.271% | 22.9% | 4.43 / 3.62 |
| Worst 90% CVaR | 90.3% | 0.240% | 0.312% | -1.149% | -1.451% | 20.8% | 4.83 / 3.89 |

Maximum drawdowns also fall from 3.18% to 2.39%, 3.84% to 3.04%, and 4.42%
to 3.36%, respectively. The reduction is not free: mean return falls by about
7--8 basis points per evaluation day relative to matched constant exposure,
and mean return divided by absolute ES10 worsens in all 21 series. ATM targeting
therefore helps if the objective is an explicit tail or drawdown cap, but it
does not improve return per unit of expected-shortfall risk in this sample.

Five-day moving-block bootstrap intervals for representative ES reduction are
approximately -3% to +33% for entropy, -3% to +36% for power-exponential, and
-3% to +33% for CVaR. Between 91% and 92% of bootstrap samples show a positive
reduction, but every 95% interval includes zero. The result is suggestive rather
than statistically decisive.

Absolute today's return gives only a 3--4% median tail reduction and worsens
Sharpe and mean-to-tail efficiency in every objective family. OCE scenario
standard deviation, scenario downside, and cross-objective agreement do not
reduce realized ES reliably. The earlier expanding loss-probability classifier
also fails its Brier benchmark, so it is not a credible tail filter. Realized
next-day return squared remains attribution-only and cannot enter an overlay.

The 21 variants share the same market dates and are not independent evidence.
Moreover, overlay timing is chronological but the underlying OCE state model
is still fitted on the full sample.

### $1 million raw and ATM-adjusted investment paths

The normalised entropy/gamma-10, power-exponential/gamma-10, and worst-90%
CVaR signals are compared on a common $1 million scale. `Continuous` uses the
optimizer's signed delta fraction. `Digital` uses its sign as an exact ±$1
million position. In both cases P&L is retained rather than reinvested. The
ATM-adjusted variants multiply the target by the expanding-median ATM exposure
rule above. After the 60-date warm-up, the common comparison has 101 next-day
returns from 3 April through 28 August 2025.

| Objective | Strategy | Terminal P&L | ES10 | Sharpe | Max drawdown | Mean abs notional |
|---|---|---:|---:|---:|---:|---:|
| Entropy 10 | Continuous | $324,941 | -$12,236 | 3.83 | -3.48% | $758k |
| Entropy 10 | Continuous ATM | $208,202 | -$8,805 | 4.62 | -2.39% | $667k |
| Entropy 10 | Digital | $313,258 | -$16,611 | 3.45 | -4.83% | $1,000k |
| Entropy 10 | Digital ATM | $206,219 | -$12,099 | 3.99 | -3.36% | $903k |
| Power-exp 10 | Continuous | $314,314 | -$14,071 | 3.62 | -4.20% | $786k |
| Power-exp 10 | Continuous ATM | $205,335 | -$9,796 | 4.43 | -3.04% | $693k |
| Power-exp 10 | Digital | $324,968 | -$16,611 | 3.58 | -4.83% | $1,000k |
| Power-exp 10 | Digital ATM | $217,928 | -$12,099 | 4.23 | -3.36% | $903k |
| Worst 90% CVaR | Continuous | $349,214 | -$16,069 | 3.89 | -4.83% | $946k |
| Worst 90% CVaR | Continuous ATM | $242,888 | -$11,491 | 4.83 | -3.36% | $851k |
| Worst 90% CVaR | Digital | $356,986 | -$16,069 | 3.96 | -4.83% | $1,000k |
| Worst 90% CVaR | Digital ATM | $249,764 | -$11,494 | 4.90 | -3.36% | $903k |

Recreate the paths, summary, and graph with:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
  /home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.investment_comparison
```

The raw worst-90% CVaR digital signal has the largest terminal P&L. ATM
adjustment lowers ES10 and drawdown and raises Sharpe for all six strategies,
but produces materially less retained P&L.

### ATM-level logistic loss classifier

The ad-hoc inverse-ATM exposure rule is replaced by the requested two-feature
logistic specification:

```text
P(loss[t+1]) = logistic(
    a * ATM[t] + b * median(ATM[t-5:t-1]) + c
).
```

The lagged median contains exactly the five preceding available market dates
and excludes the decision date. Both features are standardized using each
expanding training window. The L2-regularized logistic model (`C=1`) is refit
daily after 60 feature-complete training observations. Its benchmark is the
expanding loss frequency known before the prediction date. The 97 strictly
chronological evaluation observations run from 9 April through 28 August
2025. Continuous and digital strategies have identical loss labels because
their positions always have the same sign, so their classification metrics are
identical within an objective.

| Objective | Loss rate | Logistic AUC | Logistic Brier | Baseline Brier | Brier change | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Entropy 10 | 38.14% | 0.331 | 0.2490 | 0.2437 | +0.0053 | 0.499 |
| Power-exp 10 | 38.14% | 0.314 | 0.2492 | 0.2437 | +0.0054 | 0.499 |
| Worst 90% CVaR | 32.99% | 0.402 | 0.2395 | 0.2324 | +0.0071 | 0.485 |

Lower Brier is better, so every Brier change is adverse. All AUCs are below
0.5, and balanced accuracy is no better than chance. The two ATM levels do not
classify next-session strategy losses out of sample in this period. A plain
unregularized logistic sensitivity gives the same conclusion: AUCs are
0.331, 0.313, and 0.402 and all Brier scores remain worse than their baselines.

The final L2 fit on all 157 feature-complete observations has the following
raw decimal-volatility coefficients. These full-panel coefficients describe
the last fitted model; the chronological predictions above use the coefficient
values available on each prediction date.

| Objective | `a`: current ATM | `b`: past-five median | `c`: intercept |
|---|---:|---:|---:|
| Entropy 10 | -0.1532 | -0.5124 | -0.1834 |
| Power-exp 10 | -0.3501 | -0.1068 | -0.2182 |
| Worst 90% CVaR | +0.2925 | -0.2551 | -0.4599 |

Using the classifier as a binary $1 million entry filter gives the following
trading comparison on those same 97 out-of-sample dates. The original signed
±$1 million position is taken when predicted loss probability is below 50%; a
prediction at or above 50% holds cash. P&L is retained and not reinvested.

| Objective | Days traded | Terminal P&L: always / filter | Daily std: always / filter | ES10: always / filter | Sharpe: always / filter | Max DD: always / filter | Filter mean notional |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy 10 | 89 / 97 | $176.4k / $187.8k | $9.28k / $7.01k | -$16.09k / -$9.66k | 3.11 / 4.39 | -5.49% / -2.46% | $918k |
| Power-exp 10 | 89 / 97 | $188.1k / $199.5k | $9.26k / $6.97k | -$16.09k / -$9.61k | 3.32 / 4.68 | -5.49% / -2.46% | $918k |
| Worst 90% CVaR | 95 / 97 | $220.1k / $200.7k | $9.18k / $9.05k | -$15.51k / -$15.51k | 3.92 / 3.63 | -5.49% / -7.44% | $979k |

The entropy and power-exp filters omit eight dates whose aggregate signed P&L
is -$11,441, so their terminal P&L and tail statistics improve. The CVaR filter
omits only two dates, both winners with aggregate P&L +$19,419, and therefore
worsens P&L, Sharpe, and drawdown. This does not rescue the classifier as a
loss model: its Brier scores and AUCs remain adverse, and the favorable result
for two objectives is driven by the magnitudes of just eight omitted returns.

### ATM-level outright P&L prediction and loss cutoffs

The binary loss target is replaced with the requested expanding OLS forecast
of next-session signed P&L on a $1 million position:

```text
predicted P&L[t+1]
    = a[t] * ATM[t]
    + b[t] * median(ATM[t-5:t-1])
    + c[t].
```

Each coefficient vector is fitted only to earlier observations. The benchmark
forecast is the expanding mean of prior P&L. On the same 97 evaluation dates,
the level forecast does not beat that simple benchmark:

| Objective | Forecast/actual correlation | Forecast RMSE | Baseline RMSE | OOS R2 vs baseline |
|---|---:|---:|---:|---:|
| Entropy 10 | -0.036 | $9,722 | $9,410 | -6.7% |
| Power-exp 10 | -0.038 | $9,710 | $9,378 | -7.2% |
| Worst 90% CVaR | -0.120 | $9,912 | $9,295 | -13.7% |

Nevertheless, the forecast is tested as a do-not-trade indicator at five fixed
dollar loss levels chosen to span the observed forecast range, without using
realized P&L to optimize the cutoffs. A position is skipped when predicted P&L
is below the threshold and is otherwise the original signed ±$1 million trade.

| Objective | Loss cutoff | Traded | Terminal P&L | ES10 | Sharpe | Max DD | Skipped realized P&L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy 10 | Always | 97 | $176.4k | -$16.09k | 3.11 | -5.49% | $0.0k |
| Entropy 10 | -$5,000 | 95 | $200.5k | -$14.45k | 3.67 | -3.46% | -$24.1k |
| Entropy 10 | -$2,000 | 94 | $199.5k | -$14.45k | 3.65 | -3.46% | -$23.1k |
| Entropy 10 | -$1,000 | 85 | $151.5k | -$14.45k | 2.85 | -4.89% | +$24.9k |
| Entropy 10 | -$500 | 84 | $146.3k | -$14.45k | 2.76 | -4.89% | +$30.1k |
| Entropy 10 | $0 | 74 | $144.8k | -$14.45k | 2.74 | -4.89% | +$31.6k |
| Power-exp 10 | Always | 97 | $188.1k | -$16.09k | 3.32 | -5.49% | $0.0k |
| Power-exp 10 | -$5,000 | 95 | $212.2k | -$14.45k | 3.90 | -3.46% | -$24.1k |
| Power-exp 10 | -$2,000 | 94 | $211.2k | -$14.45k | 3.88 | -3.46% | -$23.1k |
| Power-exp 10 | -$1,000 | 85 | $163.2k | -$14.45k | 3.08 | -4.89% | +$24.9k |
| Power-exp 10 | -$500 | 84 | $158.0k | -$14.45k | 2.98 | -4.89% | +$30.1k |
| Power-exp 10 | $0 | 75 | $162.9k | -$14.45k | 3.09 | -4.89% | +$25.2k |
| Worst 90% CVaR | Always | 97 | $220.1k | -$15.51k | 3.92 | -5.49% | $0.0k |
| Worst 90% CVaR | -$5,000 | 97 | $220.1k | -$15.51k | 3.92 | -5.49% | $0.0k |
| Worst 90% CVaR | -$2,000 | 95 | $244.3k | -$13.82k | 4.53 | -3.46% | -$24.1k |
| Worst 90% CVaR | -$1,000 | 92 | $228.0k | -$13.82k | 4.24 | -3.46% | -$7.8k |
| Worst 90% CVaR | -$500 | 87 | $209.2k | -$13.82k | 3.90 | -3.46% | +$10.9k |
| Worst 90% CVaR | $0 | 78 | $204.3k | -$13.82k | 3.83 | -3.46% | +$15.8k |

The -$2,000 cutoff is the only tested common level that improves terminal P&L,
ES10, Sharpe, and drawdown for all three objectives. It skips only three dates
for entropy/power-exp and two for CVaR. The result is not monotone: moving the
cutoff toward zero rapidly removes net-profitable dates. Combined with the
negative forecast correlations and negative out-of-sample R2, this is evidence
of a small number of favorable tail flags, not a reliable P&L forecast.

Recreate the predictions and cutoff sweep with:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
  /home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.atm_pnl_predictor
```

Recreate the panel, expanding predictions, coefficients, and summary with:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
  /home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.atm_loss_classifier
```

### Delta attribution by strike region, option type, and side

The one-day option-to-delta positions are split without fitting any predictive
model. The eight mutually exclusive categories are

```text
{long, short} x {K/F < 1, K/F >= 1} x {call, put}.
```

`Long` and `short` refer to the original option position, not the sign of its
delta. `Below ATM` means forward-relative strike `K/F < 1`; `above ATM` uses
`K/F >= 1` for an exhaustive boundary convention. Each category contains the
signed forward-delta contribution after option side, optimized allocation, and
the universe-specific divider. Its realized attribution is the existing delta
contribution multiplied by the applicable close-date SPX return.

The complete panel has 23,825 option positions, 3,888 dated portfolios, and
31,104 rows: exactly eight bucket rows per portfolio, including explicit
zeros. Summing the eight buckets reconstructs portfolio net delta with maximum
absolute error `4.44e-15` and spot-only return with error `1.11e-16`.

The canonical eight-category view uses the option universe in which each
instrument is available: below-ATM calls and above-ATM puts come from `ITM`;
below-ATM puts and above-ATM calls come from `normalised`. These are separately
optimized source portfolios and are never added together. The `actual` OTM
attribution remains available in the complete panel as a robustness view.

Because call/put identity is synthetic, the economic attribution is also
collapsed to `long/short x below/above ATM`. The tables below are deliberately
separate by source. Each cell is `mean signed delta / cumulative delta P&L in
$000` per $1 million option risk budget over all 162 dates. Long plus short and
above plus below each reconstruct the source total independently.

#### ITM source

| Objective | Long | Short | Above ATM | Below ATM | Total P&L |
|---|---:|---:|---:|---:|---:|
| Entropy 10 | +0.061 / +$484.5k | +0.333 / +$883.6k | -4.839 / +$704.0k | +5.234 / +$664.2k | +$1,368.1k |
| Power-exp 10 | +0.177 / +$542.1k | +0.355 / +$947.9k | -5.454 / +$771.5k | +5.986 / +$718.6k | +$1,490.0k |
| Worst 90% CVaR | -0.030 / +$86.2k | +0.123 / +$328.0k | -8.758 / -$417.8k | +8.851 / +$832.0k | +$414.2k |

#### Normalised source

| Objective | Long | Short | Above ATM | Below ATM | Total P&L |
|---|---:|---:|---:|---:|---:|
| Entropy 10 | -0.062 / +$491.2k | +0.577 / +$898.9k | +3.802 / +$711.2k | -3.287 / +$678.9k | +$1,390.2k |
| Power-exp 10 | +0.009 / +$474.8k | +0.533 / +$1,025.2k | +4.878 / +$631.9k | -4.336 / +$868.1k | +$1,500.0k |
| Worst 90% CVaR | -0.018 / +$163.1k | +0.164 / +$278.6k | +9.507 / +$657.0k | -9.360 / -$215.3k | +$441.7k |

#### Actual source

| Objective | Long | Short | Above ATM | Below ATM | Total P&L |
|---|---:|---:|---:|---:|---:|
| Entropy 10 | -0.001 / +$465.0k | +0.612 / +$1,094.1k | +3.763 / +$718.6k | -3.153 / +$840.5k | +$1,559.1k |
| Power-exp 10 | -0.114 / +$532.4k | +0.572 / +$1,156.5k | +4.682 / +$721.1k | -4.225 / +$967.8k | +$1,689.0k |
| Worst 90% CVaR | +0.286 / +$79.3k | +0.156 / +$270.5k | +13.486 / +$900.0k | -13.044 / -$550.2k | +$349.8k |

The long/short distinction is consistent: short-option buckets contribute more
cumulative P&L than long-option buckets for all nine source/objective pairs,
despite much smaller delta exposure. The above/below distinction is not
consistent. Entropy and power-exp are relatively balanced or mildly favor
below ATM, while CVaR flips by source: below ATM dominates `ITM`, but above ATM
dominates `normalised` and `actual`.

These figures remain accounting attribution only. Positive cumulative P&L does
not establish predictive power because the buckets inherit the option
optimizer's date-varying allocations and source-universe choices.

Recreate the position annotation, complete bucket panel, reconciliation, and
eight-category attribution with:

```bash
/home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.delta_buckets
```

### Does surface reversion generate the spot signal?

The hypothesis has three distinct empirical implications:

1. the matrix-exponential AR should forecast the future surface better than
   leaving today's surface unchanged;
2. its predictable surface state/reversion should forecast spot returns; and
3. realized surface innovations should co-move with contemporaneous spot
   returns.

These are tested with expanding windows starting after 60 dates. At every
origin, the five-factor PCA, continuous-time AR, and spot equation are refitted
using only information through that origin. The evaluation therefore has 103
one-session forecasts and 99 five-session forecasts. Five-session uncertainty
uses Newey-West loss-difference statistics with four lags.

The surface forecast itself does **not** beat persistence out of sample:

| State | Horizon | AR RMSE | No-change RMSE | MSE skill | Direction correct |
|---|---:|---:|---:|---:|---:|
| Log forward vol | 1 session | 0.1255 | 0.1200 | -9.4% | 54.4% |
| Log forward vol | 5 sessions | 0.2156 | 0.2054 | -10.1% | 59.6% |
| Log DLV squared | 1 session | 0.3497 | 0.3277 | -13.8% | 64.1% |
| Log DLV squared | 5 sessions | 0.4982 | 0.4799 | -7.8% | 79.8% |

About 97% of expanding AR fits have mean-reverting eigenvalues, and their
in-sample factor R2 is about 0.47. That stability is not enough: the predicted
changes have mildly useful direction but worse magnitude error than simply
holding the current surface fixed. Front-week ATM predicted-versus-realized
change correlations are effectively zero (`0.009` at one session and `0.005`
at five).

The feasible spot forecast is also weak:

| Horizon | Forecast | RMSE | MSE skill vs expanding mean | Correlation | Sign correct |
|---|---|---:|---:|---:|---:|
| 1 session | AR state | 1.569% | -6.6% | +0.031 | 60.2% |
| 5 sessions | AR reverting state | 3.029% | -14.0% | -0.076 | 69.7% |
| 5 sessions | Frozen state | 3.425% | -45.6% | +0.069 | 63.6% |

At five sessions, allowing the state to revert reduces spot-forecast MSE by
21.7% relative to freezing today's state; the HAC t-statistic of the loss
improvement is `1.68`. This is weak-to-moderate evidence that reversion is a
useful regularizer over a week. It is not an absolute spot-return signal,
because even the reverting forecast is 14.0% worse than the expanding-mean
benchmark. Direct predicted ATM reversion has correlation `-0.034` with the
subsequent five-session spot return.

The strong finding is instead the contemporaneous leverage channel. Apply the
spot/surface coefficients estimated before each origin to the subsequently
realized surface shock. This non-tradable explanatory component has correlation
`0.927` with the one-session spot residual and reduces its MSE by 86.0%; over
five sessions the corresponding figures are `0.896` and 80.3%. Thus the joint
scenario model captures very strong spot/surface-shock co-movement, but the
shock is learned only when spot is moving and is not known at trade entry.

Existing OCE deltas have only a descriptive link because those optimizations
used the earlier full-sample dynamics. At risk aversion 10, one-session delta
correlations with the chronological spot forecast range from `0.14` to `0.33`;
at five sessions they range from `-0.05` to `0.10`. Their high sign agreement
is not decisive: the predicted spot return is positive on 69%/74% of dates and
the option-derived delta is positive on roughly 79--88% of dates.

The current conclusion is therefore: the data support a strong leverage
correlation between *realized* surface shocks and spot, and weakly support AR
decay rather than a frozen state over one week. They do not yet support the
full tradable chain “forecastable surface reversion predicts spot.” The next
decisive portfolio test is a chronological OCE rerun with identical scenarios
under (a) the fitted AR and (b) a frozen-state generator `G=0`.

Recreate the expanding fits, tables, joined signal diagnostics, and graph with:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
  /home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.surface_reversion_attribution
```

### Controlled OCE reversion ablation

The decisive portfolio test was run on 99 chronological origins after the
initial 60-date training window. At every origin, PCA, the matrix-exponential
AR, the joint spot equation, and the innovation covariance are re-estimated
using only data available through that origin. Three representative objectives
are solved in all `ITM`, `actual`, and `normalised` universes:

- entropy with risk aversion 10;
- power-exponential with risk aversion 10; and
- expected return over the worst 90% of scenarios.

Each problem is solved twice with common random numbers. The controlled
`frozen_state` ablation sets only the AR generator to `G=0`. PCA, factor scales,
today's PCA residual, spot coefficients, innovation covariance, Sobol nodes,
listed instruments, executable entry bid/ask, and all portfolio constraints
remain identical. This is an ablation of deterministic surface drift, not a
separately re-estimated alternative statistical model.

The fitted reversion materially changes the portfolio. For the normalised
universe:

| Objective | Mean absolute delta change | Delta correlation | Sign agreement | Allocation overlap |
|---|---:|---:|---:|---:|
| Entropy 10 | 0.329 | 0.680 | 79.8% | 44.7% |
| Power-exp 10 | 0.337 | 0.660 | 79.8% | 49.1% |
| Worst 90% CVaR | 0.340 | 0.660 | 82.8% | 59.7% |

Thus `G=0` is not a cosmetic change: reversion flips the delta sign on roughly
17--20% of dates and replaces a large portion of the option allocation.

Cross-evaluation confirms that each optimizer adapts to its own scenario
environment. Under the fitted-AR scenarios, the mean normalised-universe OCE of
the fitted portfolio versus the frozen portfolio is `0.0533` versus `0.0440`
for entropy, `0.0613` versus `0.0502` for power-exp, and `0.1814` versus `0.1574`
for CVaR. Under frozen scenarios, the ordering reverses.

The clean realized test holds either the entry net delta or only its sign in
SPX spot through the five-session horizon. It therefore contains no option
P&L, bid/ask, vega, gamma, or expiry-slice effects. Dollar figures use a $1
million capital unit and sum the 99 overlapping five-session decisions.
Annualized Sharpe scales by `sqrt(252/5)`; the paired difference t-statistic is
Newey-West with four lags.

#### Normalised-universe five-session entry signal

| Objective | Strategy | Fitted AR P&L / Sharpe | Frozen P&L / Sharpe | AR minus frozen P&L | HAC t |
|---|---|---:|---:|---:|---:|
| Entropy 10 | Continuous entry delta | +$767.4k / 2.31 | +$534.5k / 1.59 | +$232.9k | 1.38 |
| Entropy 10 | Digital entry sign | +$894.1k / 2.52 | +$453.6k / 1.22 | +$440.5k | 1.93 |
| Power-exp 10 | Continuous entry delta | +$807.3k / 2.42 | +$483.3k / 1.38 | +$324.0k | 1.95 |
| Power-exp 10 | Digital entry sign | +$889.1k / 2.50 | +$476.2k / 1.28 | +$412.8k | 1.78 |
| Worst 90% CVaR | Continuous entry delta | +$565.1k / 1.59 | +$342.3k / 0.92 | +$222.7k | 1.13 |
| Worst 90% CVaR | Digital entry sign | +$582.6k / 1.58 | +$340.9k / 0.91 | +$241.8k | 1.25 |

All six fitted-AR variants outperform their frozen counterpart. More broadly,
the fitted-minus-frozen entry-signal P&L is positive for all 18 combinations of
three universes, three objectives, and continuous/digital sizing. Statistical
strength varies: it is clearest for normalised power-exp continuous sizing and
normalised entropy digital sizing, and strongest in several `actual`-universe
digital comparisons.

This is not just greater long exposure. In the normalised universe, the mean
fitted-minus-frozen delta is `+0.003` for entropy, `-0.005` for power-exp, and
`-0.049` for CVaR. Nevertheless, the delta difference has correlations of
`0.168`, `0.224`, and `0.148`, respectively, with the subsequent five-session
spot return. Reversion is changing the timing of exposure in a favorable way.

The option-expiry-sliced convention gives a less uniform continuous result but
the same directional digital evidence. Fitted-minus-frozen digital P&L in the
normalised universe is +$510.0k for entropy, +$568.9k for power-exp, and
+$465.8k for CVaR, with HAC t-statistics `1.98`, `2.39`, and `1.80`.

The controlled ablation therefore supports the narrower claim that modeled
surface reversion contributes materially to the OCE portfolio's spot timing,
even though a direct linear AR-state forecast of spot did not beat the
historical-mean benchmark. It is still a small, single-2025 sample with
overlapping returns and multiple related specifications. Validation on later
or earlier years, with model selection frozen in advance, is required before
calling this a robust trading effect.

Two five-session expiry-sliced observations require a 2025-07-03 close, which
is absent from the saved spot panel. They are excluded without imputation and
recorded in `missing_realized_delta_returns.parquet`. The structural OCE and
direct entry-to-horizon comparisons retain all 99 origins.

Recreate the full chronological refits and 1,782 OCE optimizations with:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
  /home/hans/vqdata/bin/python \
  -m quantdata.dysanos.localvolbysanos.real_trading.reversion_oce_counterfactual
```

For the option strategy, the positive mean is carried by a right tail. Win
rates are only 38--51.5%,
the best 10% of observations contribute 54--81% of all positive P&L, and the
mean after removing the best decile is negative in all 18 series. For the
`actual`, risk-aversion-1 entropy run specifically, the win rate is 38%, the
average win is 37.5%, the average loss is -17.1%, and removing the top decile
changes the mean from +3.66% to -5.84%.

The spot-neutral residual must not be called pure volatility P&L. It also
contains higher-order spot effects, theta, entry spread, changing position
sizes, and spot/volatility interactions. A genuine volatility attribution must
next regress or explain this residual with realized forward-volatility/DLV
factor changes and option Greeks.

Outputs:

- `results/2025/oce_results.parquet`: one row per optimization;
- `results/2025/positions.parquet`: nonzero dollar allocations;
- `results/2025/universes.parquet`: eligible pair/action counts;
- `results/2025/summary.parquet`: the 18 run/utility/risk aggregates; and
- `results/2025/report.json`: configuration, methodology, and failure audit;
- `results/2025/realized_positions.parquet`: exact-contract entry/exit audit;
- `results/2025/realized_comparison.parquet`: predicted and observed portfolio
  returns;
- `results/2025/realized_summary.parquet`: comparison statistics; and
- `results/2025/realized_report.json`: coverage and missing-contract audit;
- `results/2025/spot_attribution_observations.parquet`: observation-level spot,
  convexity, and residual components;
- `results/2025/spot_attribution_summary.parquet`: the 18 regression and
  tail-shape summaries; and
- `results/2025/spot_attribution_report.json`: attribution conventions and
  complete results;
- `results/2025/spot_only_positions.parquet`: option-position deltas and their
  assigned SPX close dates;
- `results/2025/spot_only_close_slices.parquet`: net SPX delta by portfolio and
  close date;
- `results/2025/spot_only_comparison.parquet`: spot-only, option, and benchmark
  returns;
- `results/2025/spot_only_summary.parquet`: all 18 spot-only summaries;
- `results/2025/spot_only_report.json`: strategy conventions and results;
- `results/2025_sequential_cap_1m/oce_results.parquet`: stateful OCE targets,
  carried spot, cap trades, and retained P&L by date;
- `results/2025_sequential_cap_1m/sign_only_sequential_ledger.parquet`:
  sequential ±$1 million holdings and retained P&L;
- `results/2025_sequential_cap_1m/sequential_spot_summary.parquet`: comparable
  18-series sign-only and continuous performance; and
- `results/2025_sequential_cap_1m/sequential_spot_report.json`: accounting
  conventions and complete summary;
- `results/2025/loss_diagnostics/daily_loss_observations.parquet`: dated
  strategy returns, loss flags, and diagnostic variables;
- `results/2025/loss_diagnostics/daily_loss_coefficients.parquet`: individual
  and joint OLS/logit coefficients with uncertainty estimates;
- `results/2025/loss_diagnostics/daily_loss_summary.parquet`: fit and
  classification metrics, including expanding-window comparisons;
- `results/2025/loss_diagnostics/daily_loss_indicator_bins.parquet`: quintile
  loss rates and P&L by requested variable; and
- `results/2025/loss_diagnostics/daily_loss_report.json`: timing conventions
  and the complete diagnostic summary;
- `results/2025_cvar_daily/oce_results.parquet` and `positions.parquet`: daily
  worst-90% and worst-50% solutions; and
- `results/2025_cvar_daily/one_day/spot_only_summary.parquet`: next-session
  continuous and sign-only CVaR signal performance;
- `results/2025_cvar_sequential_cap_1m/oce_results.parquet`: stateful CVaR
  targets and retained-P&L accounting; and
- `results/2025_cvar_sequential_cap_1m/sequential_spot_summary.parquet`:
  continuous carry and sequential sign-only CVaR comparisons;
- `results/2025/tail_risk_diagnostics/tail_overlay_results.parquet`: matched-
  exposure overlay and constant tail metrics;
- `results/2025/tail_risk_diagnostics/tail_overlay_observations.parquet`:
  date-level exposures and returns; and
- `results/2025/tail_risk_diagnostics/tail_overlay_bootstrap.parquet`: five-day
  block-bootstrap uncertainty for representative ATM overlays;
- `results/2025/investment_comparison/investment_paths.parquet`: date-level $1
  million raw and ATM-adjusted investment paths;
- `results/2025/investment_comparison/investment_summary.parquet`: terminal
  P&L, ES10, Sharpe, drawdown, and notional comparison; and
- `results/2025/investment_comparison/investment_comparison.png`: reproducible
  cumulative retained-P&L graph;
- `results/2025/atm_loss_classifier/atm_loss_panel.parquet`: six raw strategy
  return series with current ATM, lagged-five ATM median, and loss flags;
- `results/2025/atm_loss_classifier/atm_loss_predictions.parquet`: dated
  expanding logistic and baseline loss probabilities;
- `results/2025/atm_loss_classifier/atm_loss_summary.parquet`: out-of-sample
  classification metrics; and
- `results/2025/atm_loss_classifier/atm_loss_coefficients.parquet`: raw and
  standardized final logistic coefficients;
- `results/2025/atm_loss_classifier/atm_classifier_trading_paths.parquet`:
  always-on and classifier-filtered $1 million date-level ledgers; and
- `results/2025/atm_loss_classifier/atm_classifier_trading_summary.parquet`:
  terminal P&L, volatility, ES10, Sharpe, drawdown, and notional statistics;
- `results/2025/atm_pnl_predictor/atm_pnl_predictions.parquet`: dated
  expanding P&L forecasts, benchmark forecasts, and fitted coefficients;
- `results/2025/atm_pnl_predictor/atm_pnl_prediction_summary.parquet`:
  out-of-sample forecast errors and correlations;
- `results/2025/atm_pnl_predictor/atm_pnl_filter_paths.parquet`: date-level
  always-on and threshold-filtered $1 million paths; and
- `results/2025/atm_pnl_predictor/atm_pnl_filter_summary.parquet`: trading
  statistics for every tested predicted-loss cutoff;
- `results/2025/delta_buckets/delta_bucket_positions.parquet`: every option
  position annotated by side, strike region, option type, and signed delta;
- `results/2025/delta_buckets/delta_bucket_panel.parquet`: eight explicit
  bucket rows for every dated portfolio;
- `results/2025/delta_buckets/delta_bucket_reconciliation.parquet`: portfolio
  net-delta and realized-return reconstruction errors;
- `results/2025/delta_buckets/eight_category_delta_attribution.parquet`: the
  date-level canonical eight-category attribution with source universe; and
- `results/2025/delta_buckets/eight_category_delta_attribution_summary.parquet`:
  mean delta and cumulative P&L attribution for the three objectives;
- `results/2025/delta_buckets/source_characteristic_delta_attribution.parquet`:
  date-level `long/short x below/above ATM` attribution kept separate by source;
- `results/2025/delta_buckets/source_characteristic_delta_attribution_summary.parquet`:
  complete cell and marginal characteristic attribution; and
- `results/2025/delta_buckets/{itm,normalised,actual}_characteristic_delta_attribution.parquet`:
  separate source tables requested for direct inspection;
- `results/2025/surface_reversion_attribution/surface_reversion_observations.parquet`:
  chronological surface and spot forecasts at every origin;
- `results/2025/surface_reversion_attribution/surface_forecast_summary.parquet`:
  AR-versus-no-change forecast skill and directional diagnostics;
- `results/2025/surface_reversion_attribution/spot_channel_summary.parquet`:
  feasible state forecasts and non-tradable realized-shock explanations;
- `results/2025/surface_reversion_attribution/reversion_counterfactual.parquet`:
  reverting-state versus frozen-state spot forecast comparison;
- `results/2025/surface_reversion_attribution/oce_delta_signal_links.parquet`:
  descriptive links to existing OCE deltas; and
- `results/2025/surface_reversion_attribution/surface_reversion_spot.png`:
  four-panel forecast and realized-shock diagnostic;
- `results/2025/reversion_oce_counterfactual/oce_results.parquet` and
  `positions.parquet`: chronological fitted/frozen solutions and allocations;
- `results/2025/reversion_oce_counterfactual/portfolio_comparisons.parquet`:
  paired date-level net delta and allocation differences;
- `results/2025/reversion_oce_counterfactual/portfolio_comparison_summary.parquet`:
  allocation overlap, delta correlation, and sign agreement;
- `results/2025/reversion_oce_counterfactual/scenario_cross_evaluation.parquet`:
  every fitted/frozen portfolio evaluated under both scenario environments;
- `results/2025/reversion_oce_counterfactual/delta_difference_timing.parquet`:
  exposure-difference bias and subsequent-spot correlations;
- `results/2025/reversion_oce_counterfactual/realized_delta_returns.parquet`:
  direct entry and expiry-sliced continuous/digital returns;
- `results/2025/reversion_oce_counterfactual/realized_delta_summary.parquet`:
  P&L, Sharpe, HAC inference, ES10, drawdown, and win rates;
- `results/2025/reversion_oce_counterfactual/realized_delta_difference_summary.parquet`:
  paired fitted-minus-frozen P&L tests; and
- `results/2025/reversion_oce_counterfactual/net_delta_counterfactual.png`:
  normalised-universe fitted versus frozen net delta scatter plots.
