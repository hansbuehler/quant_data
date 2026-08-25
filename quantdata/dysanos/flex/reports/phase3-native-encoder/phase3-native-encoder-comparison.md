# Phase 3 native-encoder comparison

Frozen 2025 same-set amortized-calibration results:

| model | date objective | all inside | weekly inside | positive mean | positive median |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat_native_140 | 8.15621 | 26.2271% | 24.4106% | 10.4545 | 7.09186 |
| flat_native_2828 | 7.11224 | 28.2805% | 26.1971% | 9.30596 | 6.16143 |
| direct_native_ceiling | 1.59667 | 69.1676% | 83.1603% | 4.89178 | 1.48363 |
| strong_global_140 | 8.75539 | 25.1493% | 23.4330% | 10.9964 | 7.25179 |
| strong_hierarchical_140 | 11.5404 | 23.7739% | 20.1537% | 13.9197 | 8.60876 |

The Strong baselines use the earlier sigmoid-DLV parameterization; the flat-native rows use the new capped-unbounded DLV² gauge.
