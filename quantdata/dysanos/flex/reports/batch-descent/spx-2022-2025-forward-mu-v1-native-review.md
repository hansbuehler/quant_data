# FLEX native-state completion review

Status: **native_complete**.

Checkpoint schema: `2`; phase: `complete`.

Checkpoint SHA-256: `e3a6952c901b48c749ff88e1b4dd89a400d3df9733d061fdcf2e47e3a1ef02d8`.

Raw-state SHA-256: `8b8c423cb8be1f6c3a7d3f4bdb22698da28ae64b3cead19223a18de7b2160a75`.

## Training convergence

| Epochs | First objective | Last objective | Best objective | Best epoch | Last-25 relative improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 25.275384492934286 | 1.3391087677387252 | 1.3391076225198137 | 299 | 0.00015979178969470023 |

## Native-state bounds

All `874 x 2828` raw states finite: **True**.

| Coordinate | Minimum | Median | Maximum | Near lower | Near upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| dlvsq | 0.00042299659 | 0.04674051 | 4.0408149 | 0.0266836% | 0% |
| forward_vol | 0.010000898 | 0.048170961 | 0.54944545 | 0.0572082% | 0% |
| forward_mu | 0.027412461 | 0.25173563 | 0.49216294 | 0% | 0% |

Near-bound fractions use sigmoid probabilities within `0.0001` of zero or one.

## Market reconstruction

| Bucket | Options | Zero excess | Conditional positive mean | Conditional positive median | Surface mean | Surface median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3492782 | 71.4044% | 4.36899 | 1.32237 | 1.24934 | 0 |
| 0DTE | 2430 | 26.5844% | 13.7216 | 11.7925 | 10.0738 | 6.38454 |
| 1DTE | 77236 | 20.3001% | 17.939 | 10.6678 | 14.2973 | 5.87089 |
| 2DTE | 102320 | 29.7977% | 12.9598 | 8.50368 | 9.09807 | 3.29836 |

Whole-surface error against `market.py` `fit`:

| Measure | Half-spreads | Normalized price |
| --- | ---: | ---: |
| Mean absolute error | 1.7964 | 9.12326e-05 |
| Median absolute error | 0.439314 | 3.66646e-05 |
| Root mean square error | — | 0.000222382 |
| Signed bias | — | -1.93926e-05 |

Mean training objective: `1.3391582`.
