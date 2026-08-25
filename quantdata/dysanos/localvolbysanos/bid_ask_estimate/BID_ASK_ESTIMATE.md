# Bid/ask estimate

## Objective

Estimate the quoted bid/ask half-spread of an arbitrary option as a conservative
crossing-cost assumption, conditional on its contract location and the
contemporaneous market state. The initial core features proposed for this
project are delta, vega, and time to expiry (TTE).

The first application is the filtered European SPX market returned by
`quantdata.dysanos.localvolbysanos.markets`. All option prices, strikes, and Greeks must use the same
forward-normalized convention as the parent `localvolbysanos` project. The
subproject must not import from another `aae.ideas` project.

## First decision: what is being estimated?

Quoted spread and execution cost are different targets and should not be mixed.

For this project, the primary cost target is now fixed as the **quoted
half-spread**. Paying the full half-spread assumes that an order crosses the
prevailing market without receiving price improvement. This is intentionally
conservative relative to typical effective execution costs.

### Quoted half-spread

For normalized bid `b` and ask `a`, define

```text
quoted_half_spread = (a - b) / 2.
```

This is the correct target when the objective is to construct plausible bid and
ask bounds around a model fair value, stress marks, fill missing quotes, or
apply a conservative crossing-cost assumption to a strategy backtest. The base
model will predict the quoted half-spread, not the effective spread.

### Effective half-spread

For a trade price `p`, prevailing midpoint `m`, and trade direction `q` equal to
`+1` for a buyer-initiated trade and `-1` for a seller-initiated trade, define

```text
effective_half_spread = q * (p - m).
```

This is the better target for execution-cost estimation and backtesting, but it
requires synchronized trades, quotes, and a defensible trade-direction rule.
Muravyev and Pearson show that option execution timing makes conventional
spread-based trading-cost estimates materially too high: the overall average
effective spread is about one-quarter smaller, while traders who time execution
have effective spreads below 40% of conventional measures. Therefore this
project should begin with quoted half-spreads but treat an execution model as a
separate second phase, not interpret quoted half-spread as realized slippage.

Effective-spread evidence is retained as a diagnostic showing that the selected
half-spread convention is conservative. It does not change the primary target.

## Signal-conditioned adverse selection

A quoted half-spread model alone cannot express the possibility that a strong
signal causes the market to move before the order fills. Decision-to-fill cost
has two distinct components. For trade direction `q`, equal to `+1` for a buy
and `-1` for a sell,

```text
q * (fill_price - decision_mid)
    = q * (fill_price - fill_mid)
    + q * (fill_mid - decision_mid).

      crossing/effective cost
    + delay and adverse-selection cost
```

The first-phase conservative model sets the crossing component to the predicted
quoted half-spread. A later signal-aware execution overlay should estimate the
second component from information available at the decision time:

```text
conservative_cost
    = predicted_quoted_half_spread
    + max(0, predicted_adverse_move_during_latency)
    + explicit_fees_and_size_impact.
```

The adverse-move term should depend on signed signal strength, expected holding
horizon, execution latency, order size, underlying volatility, and current
depth/liquidity. It must be estimated using genuinely out-of-sample signals;
realized future returns cannot be inserted as a cost feature because that would
leak the answer into the backtest.

Signal correctness is not literally an if-and-only-if condition for high cost.
A good slow-moving signal can execute without moving the quote, while an
uninformed order can cross a wide market. The economically relevant condition
is whether the signal contains short-horizon information that other traders or
market makers incorporate during the strategy's execution latency. The model
should test this empirically by comparing signed decision-to-fill midpoint
moves across ex-ante signal-strength buckets.

## Why delta, vega, and TTE are not enough

Delta, vega, and TTE identify much of an option's location on a volatility
surface, but they do not identify the cost or risk of supplying liquidity.
Options with similar Greeks can have very different spreads because of:

- the underlying or futures bid/ask spread and depth;
- the minimum option tick and price level;
- gamma and the cost of discrete delta rebalancing;
- implied and realized volatility, jumps, and volatility-of-volatility;
- trading volume, open interest, quote size, and stale-quote age;
- time of day, event risk, and the overall volatility/liquidity regime;
- exchange competition and market structure; and
- dealer inventory and one-sided customer demand.

Delta, vega, and TTE are also mechanically dependent through the option pricing
model. A flexible regression can mistake this collinearity for a stable
liquidity relation. They are a useful baseline feature set, not a complete
economic model.

## Recommended target units

The primary target should be the forward-normalized price half-spread used by
`quantdata.dysanos.localvolbysanos.markets`. It can be converted back to currency units by multiplying
by the relevant forward and contract multiplier.

Two secondary targets should be reported where numerically valid:

```text
relative_half_spread = quoted_half_spread / max(model_price, price_floor)
volatility_half_spread ~= quoted_half_spread / max(vega, vega_floor).
```

Neither should replace the price target. Relative spreads explode for cheap
wing options and volatility spreads explode when vega is small. They are useful
diagnostic coordinates with explicit floors.

## Economic decomposition

A better starting point than an unrestricted regression is a nonnegative
hedging-cost decomposition:

```text
half_spread_price
    = tick_or_fixed_cost
    + delta_hedge_cost
    + volatility_hedge_cost
    + discrete_rehedging_risk
    + liquidity_and_inventory_residual,

delta_hedge_cost       ~ abs(delta) * underlying_half_spread,
volatility_hedge_cost  ~ vega * volatility_risk_price,
discrete_rehedging_risk~ abs(gamma) * forward^2 * variance * hedge_interval.
```

This is a modeling template rather than an asserted exact pricing identity. It
encodes the main mechanisms in the published literature: underlying hedge
liquidity, volatility uncertainty, and discrete rebalancing risk. A statistical
residual model can then learn persistent deviations caused by market activity,
inventory, and quote competition.

## Proposed feature sets

### Core baseline

- forward delta, with a consistent call/put convention;
- absolute or OTM delta for call/put-symmetric diagnostics;
- vega in forward-normalized price units;
- `TTE`, `sqrt(TTE)`, and `log(TTE)`;
- call/put indicator if both original quote types are retained; and
- AF model price, so the tick and relative-price effects can be learned without
  using the observed midpoint as a hidden proxy for the target.

### Preferred structural additions

- gamma or dollar gamma;
- AF implied volatility and local skew;
- underlying/futures half-spread and depth;
- realized volatility and short-horizon underlying return volatility;
- minimum tick and a tick-binding indicator;
- quote size, volume, open interest, and quote age;
- time-of-day and minutes to close;
- date-level volatility/liquidity state, such as ATM volatility or VIX; and
- event indicators where known without look-ahead.

If the desired prediction is for a hypothetical option with no live quote,
features derived from its observed midpoint, volume, or quote size are
unavailable and must not be required by the primary model. Models using those
features should be labeled as live-quote refinements.

## Recommended model ladder

All models must predict a positive half-spread and respect the option tick.

### 0. Bucketed benchmark

Estimate the median half-spread in bins of absolute delta and `sqrt(TTE)`, with
vega or price sub-bins. This is transparent, difficult to overfit, and provides
the benchmark that every more complex model must beat out of sample.

### 1. Delta-vega-TTE regression

Fit a robust model for `log(half_spread)` using splines and interactions rather
than a raw linear polynomial:

```text
s(abs(delta)) + s(log(vega)) + s(sqrt(TTE))
    + tensor(abs(delta), sqrt(TTE)).
```

A tensor-product generalized additive model (GAM) is preferred because a
smooth spread surface is easy to inspect and interpolate. Date or regime
effects should be partially pooled rather than represented by thousands of
unregularized dummies.

### 2. Tick-aware hurdle model

Quoted option spreads often have a point mass at the minimum feasible tick.
Use two stages:

1. estimate the probability that the spread is tick-bound; and
2. conditional on exceeding the tick floor, estimate the positive excess
   spread on a log or Gamma scale.

This is usually better specified than forcing one continuous regression to fit
both the tick mass and the wide-spread tail.

### 3. Quantile model

Estimate the median and upper conditional quantiles, not only the conditional
mean. A 50th-percentile estimate gives a typical quote; a calibrated 90th or
95th percentile gives a conservative transaction-cost or risk bound. Quantile
coverage should be checked separately by delta and TTE bucket.

### 4. Gradient-boosted-tree benchmark

Use a histogram gradient-boosted tree model on the expanded feature set,
including interactions and missing-value handling. It will likely improve raw
prediction accuracy over a simple linear model, but the smooth GAM and
structural decomposition remain necessary interpretability benchmarks. The
tree model must use date-blocked validation and must not consume unavailable or
future information.

### 5. Hierarchical surface model

For multiple underlyings, use a common delta-by-`sqrt(TTE)` surface with
underlying-, date-, and expiry-level random effects or low-rank factors. This
shares information with sparse contracts without assuming that SPX, single
stocks, and other products have the same absolute liquidity level.

## Recommended first implementation

The first experiment should compare four models on the same 2025 SPX sample:

1. delta-by-`sqrt(TTE)` bucket median;
2. spline/GAM using only delta, vega, and TTE;
3. tick-aware GAM with model price, gamma, ATM volatility, and date state; and
4. histogram gradient boosting with the same expanded features.

Each model should output median and 90th-percentile normalized half-spreads.
The structural/tick-aware GAM is the recommended default if its out-of-time
error is close to the tree model; the tree model is preferred only when its
gain is material and stable across regimes.

For strategy costing, the default estimate will be the predicted quoted
half-spread. The 90th-percentile prediction is an additional stress estimate,
not the default crossing assumption. Signal-conditioned delay cost belongs to
a separate overlay and will only be added when decision and fill timestamps and
out-of-sample signal values are available.

No observed bid, ask, spread, or quote-derived midpoint may appear in the
feature set used to predict a missing quote. The AF surface price is permitted
because it is the fair-value input available for any requested option.

## Dataset construction

The initial dataset should be built directly from
`quantdata.dysanos.localvolbysanos.markets.load_market_day`, not from another ideas project. For every
retained quote, store at least:

- market date and expiry;
- forward-relative strike `K/F`;
- normalized bid, ask, midpoint, and quoted half-spread;
- AF surface price and implied volatility;
- forward delta, gamma, and vega computed under one documented convention;
- TTE in years and days;
- front-week and expiry ATM volatility; and
- data-quality flags, including tick-bound, zero bid, locked/crossed, and
  stale quote if available.

The exact Greek units must be written into the schema. In particular, record
whether vega is price change per unit volatility or per volatility point and
whether delta is spot, forward, or premium-adjusted. The preferred convention
for this project is forward delta and vega per unit volatility, both on the
normalized price scale.

## Validation design

Random quote-level splitting is invalid because neighboring strikes, expiries,
and repeated dates are strongly dependent. Use chronological date blocks:

- training dates first;
- validation dates next for model and hyperparameter selection; and
- a final untouched test period.

All quotes from one date must remain in the same split. If multiple underlyings
are added, also report leave-one-underlying-out performance.

Primary metrics:

- median absolute error in normalized price half-spread;
- mean absolute error, with and without the widest 1% of observations;
- absolute error in tick units;
- median absolute error in volatility points where vega is reliable;
- 50th/90th-percentile calibration and empirical coverage; and
- errors by delta, TTE, price, and volatility-regime bucket.

Compare every model with the bucketed median and with a date-level constant.
Report performance on tick-bound and non-tick-bound quotes separately.

## Quoted-spread surface constraints

The output must satisfy:

- finite and nonnegative half-spread;
- at least the applicable half-tick after conversion to quoted price units;
- no use of future market information;
- stable extrapolation or an explicit out-of-support flag; and
- deterministic conversion between normalized, currency, relative, and
  volatility-spread units.

Unlike option prices, bid/ask widths do not obey strike convexity or calendar
arbitrage identities. Smoothness across delta and expiry is desirable, but it
must not be mislabeled as a no-arbitrage constraint.

## Published work and implications

- [Jameson and Wilhelm (1992), *Market Making in the Options Markets and the
  Costs of Discrete Hedge Rebalancing*](https://doi.org/10.1111/j.1540-6261.1992.tb04409.x)
  find that discrete hedge-rebalancing risk and uncertainty about underlying
  volatility explain economically significant portions of option spreads.
  This supports adding gamma and volatility-risk features.

- [George and Longstaff (1993), *Bid-Ask Spreads and Trading Activity in the
  S&P 100 Index Options Market*](https://www.anderson.ucla.edu/documents/areas/fac/finance/bid-ask-spreads.pdf)
  relate cross-sectional spreads to market-making costs and trading activity
  and show substitution effects between calls and puts. This argues for price,
  activity, and paired-option information beyond Greeks.

- [Cho and Engle (1999), *Modeling the Impacts of Market Activity on Bid-Ask
  Spreads in the Option Market*](https://www.nber.org/papers/w7331)
  develop derivative hedge theory and find option spreads positively related
  to underlying-market spreads. This makes underlying liquidity a central
  feature, not an optional refinement.

- [Mayhew (2002), *Competition, Market Structure, and Bid-Ask Spreads in Stock
  Option Markets*](https://doi.org/10.1111/1540-6261.00447)
  shows that multiple listing and market structure materially affect quoted
  and effective spreads. A model intended for multiple venues or underlyings
  needs venue/competition controls or hierarchical effects.

- [Cao and Wei (2010), *Option Market Liquidity: Commonality and Other
  Characteristics*](https://doi.org/10.1016/j.finmar.2009.09.004)
  document commonality in option liquidity even after controlling for
  underlying liquidity and volatility. This motivates date-level or market-
  liquidity factors.

- [Christoffersen, Goyenko, Jacobs, and Karoui (2018), *Illiquidity Premia in
  the Equity Options Market*](https://doi.org/10.1093/rfs/hhx113)
  use intraday effective spreads and connect option illiquidity to risky market-
  maker inventory. This supports a later effective-spread and inventory phase.

- [Muravyev and Pearson (2020), *Options Trading Costs Are Lower than You
  Think*](https://doi.org/10.1093/rfs/hhaa010)
  show that quoted or conventionally measured spreads can substantially
  overstate realized execution costs because traders time their executions.
  This is the main reason to keep quote-width and execution-cost models
  separate.

- [Hendershott, Khan, and Riordan (2026), *Option Auctions*](https://doi.org/10.1093/rfs/hhaf043)
  explicitly identify delta times the underlying quoted spread as the initial
  hedge cost faced by an option market maker and distinguish quoted from
  effective spreads and price improvement. This supports the structural
  delta-hedge-cost term.

A recent working paper, [Götz, Riordan, Schuster, and Uhrig-Homburg,
*Measuring Option Liquidity*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239635),
proposes an elasticity-adjusted spread. It is useful as a diagnostic
normalization, especially when comparing contracts with very different price
elasticities, but it is not a substitute for predicting the price half-spread
needed to construct bid and ask bounds.

## Milestones

1. Use quoted half-spread as the fixed conservative base target and freeze the
   Greek/unit conventions.
2. Build and audit the quote-level SPX dataset.
3. Plot the empirical half-spread surface by delta and `sqrt(TTE)`.
4. Fit the bucket, GAM, tick-aware, and boosted-tree benchmarks.
5. Perform strictly chronological validation and quantile calibration.
6. Select the simplest model with stable out-of-time performance.
7. Add signal-conditioned decision-to-fill delay and adverse-selection costs
   only when synchronized decisions, trades, quotes, and out-of-sample signals
   are available.

## Current status

The subproject has been created and quoted half-spread is fixed as the
conservative base cost target. The feature hierarchy, model ladder, validation
protocol, signal-conditioned execution overlay, and literature basis are
specified. Dataset and model code have not yet been implemented; Greek units
and quote timing still need to be fixed before committing to a schema.
