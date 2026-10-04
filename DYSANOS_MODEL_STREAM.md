# DYSANOS main model stream

Last updated: 2026-09-21. This is a compact map of the principal model line,
from observed option books to scenario generation, hedging, and the separate
risk-neutral-law research. It is a navigation document rather than a new model
specification; the linked project notes and stored reports remain authoritative.

## One-page view

```text
SPX option books
    |
    v
SANOS cross-sectional surface
    |  discrete densities and DLVsq; static arbitrage excluded by construction
    v
DYSANOS affine state h_t in R^200
    |  u_t = b + W h_t; decode h_t back into a SANOS surface
    v
Stable AR(1) historical law P
    |  Gaussian innovations for analytics; historical replay for realistic tails
    v
Spot and option scenarios
    |------------------------------|
    v                              v
one-step statistical hedge         multi-step Deep Hedge
(mean-variance or OCE)              (state-dependent daily actions)
    |                              |
    +---------------+--------------+
                    v
             buyer/seller prices,
             hedge P&L and policies

Separate Q stream:
P-to-Q utility tilts / transports  <  native price-preserving DLV construction
       (research diagnostics)         (successful finite-grid martingale law)
```

The most important separation is between:

1. **The fitted historical law `P`**, which describes observed surface moves and
   is used to generate realistic hedging scenarios; and
2. **A risk-neutral law `Q`**, under which discounted spot and the modeled option
   prices must be martingales. A good `P` model is not automatically a `Q` model.

## 1. Cross-sectional model: SANOS

At one observation date, SANOS represents an option surface on an expiry/strike
grid through discrete terminal densities and discrete-local-variance-squared
(`DLVsq`) transitions. Positivity, unit mass, unit forward mean, strike
monotonicity/convexity, and calendar consistency are structural rather than
soft training penalties.

The statistical model works in unconstrained native coordinates `u`; the SANOS
map converts these coordinates into valid positive DLV and forward parameters,
then into densities and option prices. Thus every decoded state is a statically
arbitrage-free surface, subject only to floating-point error.

## 2. Main fitted DYSANOS model

The presentation model is the **daily-front plus weekly-tail Direct-200 model**.

- History: 891 SPX dates from 2022 through 2025-08-29.
- Exclusion: next-day AM/0.2-day population.
- Model expiries: `1, 2, 3, 4, 5, 10, 15, ..., 70` business days (18 nodes).
- Strike geometry: a shared 200-node grid, formed as the union of two 100-node
  normalized-strike grids.
- Native coordinates: DLV/forward coordinates on that fixed geometry.
- Compressed state: `h_t in R^200`.
- Decoder: `u_t = b + W h_t`, with no nonlinear network after the affine map.

The 200 state coordinates are **not** implied-volatility principal components.
The IV PCs shown in the presentation are computed afterwards by pushing the
fitted invariant state covariance through the decoder.

### Cross-sectional fit

| Geometry/model | Books | Inside bid/ask |
| --- | ---: | ---: |
| Weekly native (`5, 10, ..., 70D`) | 3,566,742 | 73.2911% |
| Weekly affine-200 | 3,566,742 | 72.9744% |
| Daily-front + weekly native | 3,566,258 | 81.8431% |
| Daily-front + weekly affine-200 | 3,566,258 | 80.0141% |

Adding explicit 1D--4D nodes is the main short-end improvement. Compression to
200 affine states is a smaller source of fit loss.

## 3. Historical dynamics under `P`

The compressed state follows a stable affine AR(1), equivalently parameterized
through a continuous generator:

```text
h_(t+Delta) = h_bar + (h_t - h_bar) A_Delta + epsilon_(t,Delta).
```

For the daily-front plus weekly-tail model:

- weekly transition spectral radius: `0.953502`;
- slowest fitted half-life: `14.56` weeks;
- invariant covariance effective rank: `193.7 / 200`;
- exact one-week historical innovations: 861.

Gaussian innovations are much too thin-tailed. Independent replay of empirical
one-week innovations is exact at one week by construction and remains closer to
market tails at two and three weeks, although it still understates them.

For the five-session presentation experiment, the exact weekly Gaussian law is
factored into five equal daily transitions so that composing the daily laws
recovers the fitted weekly transition and covariance.

## 4. Scenario generation

Given current state `h_t`:

1. draw the next state innovation and advance the AR state;
2. decode the future SANOS surface;
3. draw spot from the front-expiry density;
4. value every surviving fixed-cash option at its remaining maturity and cash
   strike; and
5. record spot, option values, state features, and any path flags required by a
   path-dependent claim.

Presentation paths use the Direct-200 checkpoint and the 2025-03-18 state.
The standard run contains 131,072 paths: 98,304 training and 32,768 validation.

The historical-law simulation is a hedging/scenario model. It does not assert
that the simulated option prices are martingales under `P`.

## 5. Hedging and pricing objectives

### Static/statistical hedge

A static action is selected at time zero from the initially listed instruments.
The action may include an initial spot/forward position. Instruments that expire
before the target date settle into cash; no new rolling instruments enter unless
the experiment explicitly says so.

The main objectives are:

- mean-variance; and
- optimized certainty equivalents (OCE), principally entropy and the more
  numerically robust power-exponential utility.

For utility `u`, the cash OCE is written as

```math
U(X)=\sup_{\eta}\left\{\eta+\mathbb E[u(X-\eta)]\right\}.
```

Buyer and seller values subtract the optimized zero-claim opportunity. This is
essential when `P` contains statistical risk premia.

### Deep hedge

The dynamic policy chooses target positions at each trading date. In the 5D
presentation experiment it observes the current 200-dimensional DYSANOS state,
log spot/forward, and current inventory. Separate two-layer SiLU networks are
used at each day. It can retrade surviving options and spot daily; expired
instruments are masked out. Proportional option half-spreads and a small trade
ridge are charged. Spot half-spread was zero in the slide-42 experiment.

Deep hedging should improve most when the claim is path-dependent or when the
future state contains information that a time-zero portfolio cannot use.

## 6. Current presentation hedging conclusions

### Five-day narrow ATM call spread

The slide-42 comparison already gives both methods spot access:

- **Static:** one initial delta `Delta_0`, held to day 5.
- **Deep:** a state-dependent spot target rebalanced at days `0, ..., 4`.

Under the fixed OCE-derived measures, static and deep prices are almost equal:

| Objective | Static buyer/seller | Deep buyer/seller |
| --- | ---: | ---: |
| Mean-variance | `-0.1642 / 1.2597` | `-0.1645 / 1.2598` |
| Entropy OCE | `0.4456 / 0.6362` | `0.4455 / 0.6353` |
| Power-exp OCE | `0.4465 / 0.6358` | `0.4466 / 0.6341` |

This does **not** mean spot was useless. A separate `P`-law ablation shows that
adding spot sharply reduced the buyer/seller width:

| Method/objective | No spot | With spot |
| --- | ---: | ---: |
| Static mean-variance | 2.7779 | 1.5420 |
| Static entropy | 0.5484 | 0.1904 |
| Static power-exp | 0.4938 | 0.1869 |
| Deep mean-variance | 2.1313 | 1.2469 |
| Deep entropy | 0.4944 | 0.1816 |
| Deep power-exp | 0.3541 | 0.1321 |

The slide-42 result says that, after both methods receive spot and the same rich
set of 1D, 2D, and 5D options, *additional daily rebalancing* contributes little
for this nearly terminal-only claim under the derived measure. One target leg
is also almost statically spanned by the near-ATM 5D put plus spot through
put-call parity. The remaining last-day gamma cannot be eliminated by a spot
trade made one day before expiry.

### Path-dependent example

The barrier/knockout experiments show the intended use case. Once the future
hedge depends on an observed barrier state, Deep Hedging can adapt after the
observation while a time-zero static portfolio cannot. The dynamic interval is
materially narrower in the best presentation example.

## 7. The separate risk-neutral `Q` stream

### Learned measure changes and transports

Three learned approaches were investigated:

1. minimize option-drift moments or a covariance-aware quadratic statistic;
2. learn a generic conditional transport for the next surface state; and
3. use OCE marginal utility as a positive likelihood ratio.

For an optimized zero-claim OCE strategy, the candidate weight is proportional
to marginal utility:

```math
\frac{dQ}{dP}\propto u'(X^*-\eta^*).
```

At a true interior optimum, the weighted means of returns available to the
strategy class are zero (or inside transaction-cost bands). In practice:

- fixed-state and small multi-state transports solved finite moment problems;
- one shared transport over all 891 states did not pass independent audits;
- OCE tilts often had very small effective sample size and utility saturation;
- changing option-return weights cannot recover state-drift components outside
  the innovation/likelihood-ratio span; and
- a small adversary gradient can reflect saturation or parameterization, not a
  globally valid martingale measure.

These methods remain useful diagnostics, but are not the certified all-state
`Q` construction.

### Native price-preserving DLV construction

The successful `Q` construction works directly with the finite SANOS density
tower rather than learning zero drift from Monte Carlo means. At each weekly
step it:

1. creates symmetric DLV density branches whose weighted density equals the
   current density exactly;
2. samples the expiring front node (with antithetic ranks);
3. conditions every surviving density on that node;
4. re-expresses the same conditional cash law on the new relative grid; and
5. appends a new far expiry using the fitted floating 4W boundary law.

For fixed cash strikes, the re-grid is a coordinate change. If the revealed
front node is `x_i`, then `S_(t+h)=S_t x_i` and the inherited support becomes
`x'_j=x_j/x_i`, so `S_(t+h)x'_j=S_t x_j` path by path. Therefore re-gridding
does not change surviving cash option values.

The all-state X-only audit used all 891 initial states, 100,000 paths, four
weekly steps, and 11 strikes per expiry. The still-surviving original contracts
plus spot had joint zero-mean p-values `0.139, 0.406, 0.439, 0.357`; worst
coordinate absolute z-scores were `1.796, 2.143, 1.371, 2.057`. Direct inherited
cash and re-grid valuations agreed pathwise to at most `2.22e-16`.

This establishes a martingale law for the modeled finite discrete surface. It
is not a uniqueness result and does not claim a continuous-strike diffusion
between weekly dates.

## 8. Statistical audit convention

For return vector `R`, sample mean `m`, covariance `Sigma`, and `N` paths, the
covariance-aware joint statistic is

```math
J=N m^\top\Sigma^+m.
```

Small p-values reject the joint zero-mean hypothesis; large p-values mean that
the observed means are compatible with Monte Carlo noise. Report together:

- maximum coordinate `|z|`;
- maximum mean in economically meaningful units (for example bp of spot);
- `J`, retained effective degrees of freedom, and p-value;
- covariance regularization/eigenvalue cutoff; and
- independent path count after antithetic pairing.

Near-collinear option books make the covariance rank and cutoff material. A
single low p-value is not sufficient evidence of arbitrage when many horizons,
states, or cutoff choices were inspected.

## 9. Canonical locations

- Persistent data, checkpoints, and numerical artifacts:
  `/home/hans/quantdata_data`
- Weekly `P` model:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/integer-weekly-no-am-all-history-v1`
- Daily-front + weekly Direct-200 `P` model:
  `/home/hans/quantdata_data/training/flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1`
- Native four-week `Q` audits:
  `/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence`
- Presentation hedging artifacts:
  `/home/hans/quantdata_data/experiments/dysanos-presentation-hedging/direct200-2025-03-18`
- Current presentation baseline (not modified by this note):
  `/mnt/c/Users/hans/OneDrive/2026 09 22 CBOE - codex v2.pptx`

## 10. Detailed references

- [`quantdata/dysanos/flex/HANDOVER.md`](quantdata/dysanos/flex/HANDOVER.md):
  full `P` and `Q` experiment history and artifacts.
- [`paper/FLEX_PRESENTATION.md`](paper/FLEX_PRESENTATION.md): affine-200 fit,
  AR(1), tail, PCA, and presentation results.
- [`MODEL_BASICS.md`](MODEL_BASICS.md): OCE and power-exponential conventions.
- [`quantdata/dysanos/flex/POWEREXP_UTILITY_ADVERSARY.md`](quantdata/dysanos/flex/POWEREXP_UTILITY_ADVERSARY.md):
  bounded-action utility adversary specification.
- [`quantdata/dysanos/localvolbysanos/LOCALVOLBYSANOS.md`](quantdata/dysanos/localvolbysanos/LOCALVOLBYSANOS.md):
  earlier direct DLV inversion/PCA/AR research stream.
- [`quantdata/dysanos/localvolbysanos/results/fixed_contract_hedge_2025-03-18/key_results/16_fixed_contract_bid_ask_summary.md`](quantdata/dysanos/localvolbysanos/results/fixed_contract_hedge_2025-03-18/key_results/16_fixed_contract_bid_ask_summary.md):
  fixed-contract static/deep hedge summary.

## Current interpretation

The main usable model is the Direct-200 DYSANOS historical simulator feeding
statistical and deep hedging. Its static-arbitrage properties are structural;
its time-series law is a fitted physical law. For a finite-grid martingale law,
the native density-tower construction is currently stronger than learned
measure changes. OCE remains valuable both as a pricing/hedging objective and
as an adversarial diagnostic, but its implied density must always be audited for
effective sample size, saturation, and all-instrument zero drift.
