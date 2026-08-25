# Phase 3 cost model: brief research note

## Question

The trading simulator needs a realistic option half-spread model that remains
differentiable inside a Torch path-generation and optimal-trading loop. A
price-based half-spread is more natural here than a volatility spread because
it is the cash execution penalty charged by the return calculation.

## Data and target

The source is every positive-time call and put book retained by
`flex.market.load_market_day`, including intermediate market expiries. The
target is normalized price half-spread

\[
y=(\mathrm{ask}-\mathrm{bid})/2.
\]

Training uses 712 dates from 2022-01-03 through 2024-12-31. The untouched test
uses 162 dates from 2025-01-02 through 2025-08-29. Each date has equal total
weight regardless of its option count. No 2026 data is read.

## Selected model

The selected simulation model is a two-hidden-layer width-64 SiLU network. Its
features are Black vega, square-root TTE, NMS, log strike, signed call/put
delta, and log approximate dollar option price. Approximate dollar price is
normalized AF fitted price times spot, so rates and forwards are not model
inputs.

The network predicts log normalized half-spread using a date-weighted Huber
loss. Books below $0.25 approximate premium are excluded before fitting. This
retains 99.960% of training books and 99.984% of test books.

At inference, the predicted half-spread is floored at half the configured SPX
tick:

\[
\widehat y_{\mathrm{exec}}
=\max\left(\widehat y_{\mathrm{network}},
\frac{0.5\,\mathrm{tick}(V)}{S}\right),
\]

where the assumed tick is $0.05 below $3 premium and $0.10 otherwise. The
simulator then uses proportional cost

\[
c=\widehat y_{\mathrm{exec}}/\max(V,p_{\min}).
\]

The network, tick floor, and division remain native Torch operations. Gradients
propagate through simulated state-dependent prices and Greeks; zero-strike
calls continue to use the separately configured spot cost.

## Frozen 2025 results

| model | date-balanced MAE | date-balanced RMSE | date-balanced R2 |
| --- | ---: | ---: | ---: |
| linear Torch baseline | 0.00023552 | 0.00033783 | 0.13989 |
| histogram gradient booster | 0.00019720 | 0.00028810 | 0.37446 |
| selected Torch network | **0.00013639** | **0.00025765** | **0.49975** |

For the default ITM universe, the network's median absolute proportional-cost
error is `0.00444`, versus `0.00555` for the tree. Its median predicted
half-spread/price is `0.01306` against actual `0.00848`, so it remains
conservatively biased upward despite the better aggregate errors.

## Interpretation and limitations

The network is a satisfactory Phase 3 research cost model: it materially
improves temporal OOS fit, is small, reloads in about 0.35 ms, and introduces no
CPU transfer into simulation. It is not an execution guarantee. The $0.25
filter, tick schedule, dollar-price approximation, architecture, log-Huber
loss, and optimizer schedule are ad hoc. The simulated SANOS surface also uses
its strike-total-volatility curve as the differentiable volatility proxy for
Greeks rather than repeating an implied-volatility inversion.

Artifacts are `reports/phase3-spread/feed_forward.pt` and
`reports/phase3-spread/phase3-spread-ffw.json`.

The earlier linear and histogram-gradient-boosting fitted objects were retired
after this comparison. Their measured train/2025 benchmark results above are
retained as history, but they are no longer accepted simulation checkpoints.
