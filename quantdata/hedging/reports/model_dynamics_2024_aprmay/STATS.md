# April–May 2024 retained-model audit

Audit of 44 origins; existing fits unchanged. 22 missing May FT fits added by the original causal rolling recipe; no hedge reruns.

The common observable PCA basis is frozen before April. Native rolling PCs are not concatenated.

| Model | Surface days | Conditional trace ratio | Pooled trace ratio | Relative covariance error |
|---|---:|---:|---:|---:|
| DYSANOS ordinary (16 factors) | 44 | 0.0798449 | 0.165531 | 0.820685 |
| DYSANOS spot-orthogonal (16 factors) | 44 | 0.154517 | 0.214446 | 0.775426 |
| FuNVol | 44 | 0.396839 | 0.494708 | 0.57998 |
| VolGAN PCA8 | 44 | 0.000250929 | 0.969687 | 0.282744 |
| AD-Seq-Vol | 44 | 0.374178 | 1.54238 | 0.786157 |
| AD-Seq-Vol-FT | 44 | 0.301754 | 1.45454 | 0.697106 |
| GROMM-P | 44 | 0.796015 | 1.26814 | 0.333905 |
| GROMM-Q | 44 | 0.763 | 1.2051 | 0.280175 |
| DYSANOS ordinary (16 factors) (tail) | 44 | 0.0797109 | 0.165529 | 0.820325 |
| DYSANOS spot-orthogonal (16 factors) (tail) | 44 | 0.154183 | 0.214164 | 0.775577 |

Full definitions, quote scores, tail diagnostics, PCA/ATM persistence and model-choice appendices are in `MODEL_DYNAMICS_REPORT.pdf`.

Inference retention: 88 exact EMA exports validated; 88 full restart checkpoints retired (4.430 GiB gross). Compact archive: 0.756 GiB. Historical contexts, scenarios and other model files remain retained.

Coverage limits: base diffusion has 43 retained surface forecasts plus one regenerated saved-checkpoint forecast; fine-tuned diffusion has 44 scored forecasts and 44 trained dates. Every recursive test uses 32 paths over 115 sessions. See coverage.csv for actual completed dates. All 1%/5% tail inference is exploratory with 44 or fewer realized dates.

Traffic lights: pass black, warn orange, fail red. Except for the AR stationarity boundary and conventional exact-binomial p-value cutoffs, thresholds are explicit retrospective engineering tolerances, not confidence bounds or arbitrage proofs. See TRAFFIC_LIGHT_RULES.json and the report's threshold table.

DYSANOS uses our own legacy nine-tenor fits to 0.1-SANOS, not the unimplemented corrected 13-tenor specification. Its floored/unidentified coordinates and numerical guards are disclosed, not validated away.

VolGAN's pooled variance must not be interpreted as conditional dispersion.

Stationarity scope: the 16 floating-volatility PCA–AR(1) coordinates only. Spot-level stationarity is not required or tested. Gaussian stock-return shocks contribute to the factor innovation covariance without changing the AR stability criterion.

| DYSANOS backbone | Stable fits | Minimum AR radius | Maximum AR radius | Maximum modal half-life (business sessions) |
|---|---:|---:|---:|---:|
| DYSANOS ordinary (16 factors) | 44/44 | 0.565029 | 0.734507 | 2.246426 |
| DYSANOS spot-orthogonal (16 factors) | 44/44 | 0.573515 | 0.729851 | 2.201064 |
