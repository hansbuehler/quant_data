# Weekly common-pricing evidence

## Settlement convention

Every weekly target settles on the inception-forward-normalized path

`X_i = S(t_i) / F(t_0,t_i)`.

This is the path represented by the inception SANOS and DLV geometries. The
actual `S(t_i)/S(t_0)` path remains the basis for replaying fixed cash-strike
hedge contracts and computing spot notional. The compounded rolling one-day
forward-adjusted path remains stored as a diagnostic only.

The distinction is material. For an episode define

`R = product_i [F(t_i,t_{i+1})/S(t_i)] / [F(t_0,t_5)/S(t_0)]`.

Then `X_5 / X_5^rolling = R`. Across all 528 eligible weekly origins from
2023-01-17 through 2025-08-22, `R` ranged from 0.99679764 to 1.00330660. The
root-mean-square absolute log difference was 10.89 basis points and the maximum
was 33.01 basis points.

An ex-post counterfactual repriced each of the ten digital strikes on the
origin SANOS density after applying this rolling-carry ratio. Across 5,280
origin/target pairs, the absolute probability change had mean 0.01573, median
0.01230, 95th percentile 0.04187, 99th percentile 0.06341, and maximum 0.12270.
The rejected rolling convention therefore was not a harmless approximation:
it made settlement depend on future carry curves and could differ materially
from the inception market mark.

## Common prices

- European digitals use the direct inception SANOS interpolation at their
  fixed inception-NMS strike.
- Five-day realized variance and the two end-of-day barrier options use the
  SANOS-DLV reference model because the terminal SANOS density alone cannot
  price their path dependence.
- Every price is a discounted inception-PV amount per unit inception spot. No
  price is described as an observed midpoint.

The DLV transition matrices preserve probability and the normalized forward.
On 2023-01-17 the maximum transition mass error was `1.33e-14` and the maximum
martingale error was `1.31e-14`.

## Background integration

The independent 5% Black background is integrated on a conservative,
recombining uniform log grid. Gaussian convolution and a conservative drift
translation are applied between daily DLV transitions. This replaces a tensor
Gauss--Hermite tree whose discontinuous barrier indicators oscillated with
quadrature order and reached approximately 10 GiB peak RSS at order 15.

For 2023-01-17 the barrier-price convergence was:

| Grid points | Down-and-out put | Up-and-out call | Wall seconds |
| ---: | ---: | ---: | ---: |
| 257 | 0.0067353142 | 0.0076224166 | 0.112 |
| 513 | 0.0067329438 | 0.0076211554 | 0.175 |
| 1,025 | 0.0067338322 | 0.0076205879 | 0.352 |
| 2,049 | 0.0067339621 | 0.0076204355 | 0.897 |
| 4,097 | 0.0067340810 | 0.0076203353 | 2.999 |

On this date the 2,049-point default differs from the 4,097-point check by less
than `1.2e-7` in absolute price. A separate adverse convergence check on
2025-04-07 found a `1.50e-6` maximum absolute difference (0.029% relative) at
2,049 points, versus `1.05e-5` (about 0.2% relative) at 1,025 points. This is
why 2,049 rather than 1,025 is the production default. The 4,097-point process
used about 1.3 GiB peak RSS, including loading the full public panel.

The same date produced a five-day realized-variance price of
`8.48291268e-05`. All 13 common weekly prices were finite and non-negative;
the ATM call and put digital prices summed to the five-day discount factor, and
each barrier price lay below its corresponding vanilla price.

An eight-date check spanning 2023-01-17 through 2025-08-22 also produced all 13
finite non-negative prices on every date. Per-date pricing took 0.35--0.38
seconds after panel load. Across those dates the maximum transition mass and
martingale errors were `3.59e-14` and `3.57e-14`, respectively. The complete
common hedging test suite passed 135 tests after the settlement and integration
changes.
