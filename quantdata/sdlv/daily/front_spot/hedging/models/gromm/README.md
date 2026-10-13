# GROMM hedge model

This directory contains the native generative risk-neutral option-market model.

The market model is the read-only `quantdata/gromm2` architecture fixed by the
benchmark: 200 interior strikes, rank-200 affine JAX market fit, 16-factor
AR(1) log-DLV-squared dynamics, and eight Brownian substeps per business day.
The hedging wrapper never uses the existing full-panel GROMM2 artifacts for a
historical origin. Each production artifact contains exactly 252 transitions,
ends on its declared as-of date, records file hashes and the public-manifest
identity, and is rejected if any state date lies in the future or its compatible
physical drift explains less than 50% of the density-weighted target-drift
energy or has a fitted daily risk-price norm above 3.

- **Inputs:** only the distributable public candidate's dated DLV states,
  SANOS reference marks, forward/discount curves, spreads, and normalized spot
  returns.
- **Measures:** stochastic SDLV scenarios under both `Q` and the accepted
  compatible rolling `P` drift.
- **Prices:** the model exposes its `Q` expectation; common benchmark marks and
  off-grid interpolation always use the stored SANOS state. Long, short, and
  zero policies are fitted independently, so buyer and seller indifference
  prices do not rely on sign symmetry.
- **Scenarios:** rank-16 rolling DLV shocks, independent 5% Black background,
  antithetic paths, and eight Brownian substeps per business day.
- **Hedges:** LSTM plus affine-and-deep actions for mean-variance,
  mean-volatility, entropy, and power-exponential objectives. Signed claim,
  short-claim, and zero-claim policies determine indifference prices and the
  claim-minus-zero diagnostic. The executable pure hedge is trained directly in
  net-position coordinates on conditional Q-shadow gains. Each simulated P
  state is advanced once more with the same Brownian shock and no market-price-
  of-risk shift, with gains measured from its GROMM model mark. This removes
  state-dependent P drift and the common-mark/model-mark basis without fitting a
  conditional mean on held-out paths. It is needed because the absolute
  turnover cost of a policy difference is not the difference of the two
  policies' costs.
- **Conventional comparator:** a separate result uses the exact current GROMM
  `Q` expectation for daily digitals, Monte Carlo GROMM `Q` expectation for
  weekly claims, and central common-random-number repricing of that same GROMM
  expectation for spot delta and the first eight rolling DLV-factor vegas.
  Hedge-option vegas use the identical contemporaneous DLV bumps. It does not
  train a deep policy and is reported separately from every
  objective-dependent GROMM hedge.
- **Weekly paths:** the dated market state is rolled recursively for five
  sessions. The compatible P risk price is recomputed from each path's
  floating log-DLV state and previous factor score; Q factors remain unshifted.
  Frozen policies replay one fitted LSTM over observed states, while restarted
  policies use only the first action of each dated refit.
- **Costs:** generated books start from the common SANOS executable marks.
  Recurrent pure-hedge training charges actual net-position turnover, resets
  cash-settled options without an exit spread, and liquidates only positions
  surviving the terminal close.
- **Selection:** four architectures are trained on 2022 scenarios; the cheapest
  candidate within one validation standard error of the best is frozen for
  rolling retraining. The public panel starts on 2022-01-03, so the strictly
  2022 selection origin uses the last 200 eligible transitions through
  2022-12-30; production fits use the agreed 252-transition rolling window
  from the first eligible 2023 origin. The selection population contains all
  20 daily digital targets under both `Q` and accepted compatible `P`, with
  independent antithetic scenario blocks per case.
- **Artifacts:** versioned restartable Torch checkpoints containing policy,
  optimizer, RNG, validation, and best-state data.

Standalone CPU test:

```bash
JAX_PLATFORMS=cpu python -m \
  quantdata.sdlv.daily.front_spot.hedging.models.gromm.smoke
```

Rolling market preparation is CPU-only and uses only the distributable public
dataset. The arrays named `mids` inside GROMM2 are reconstructed SANOS reference
prices, not unavailable WRDS market midpoints.

```bash
JAX_PLATFORMS=cpu python -m \
  quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2 \
  --stage prepare \
  --dataset "$QUANTDATA_ROOT/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1" \
  --artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling" \
  --as-of-date 2023-01-10
```

The affine stage must run under a managed GPU allocation. The GROMM2 fitter
checkpoints every ten epochs and emits flushed epoch, elapsed-time, and ETA
progress.

```bash
gpuctl run --gpus 1 -- env \
  QUANTDATA_ROOT="$QUANTDATA_ROOT" \
  PYTHONPATH=/home/hans/python/quant_data \
  python -m quantdata.sdlv.daily.front_spot.hedging.runners.fit_rolling_gromm2 \
  --stage fit \
  --dataset "$QUANTDATA_ROOT/datasets/hedging/spx-2022-2025-hedging-public-candidate-v1" \
  --artifact-root "$QUANTDATA_ROOT/training/hedging/gromm2-rolling" \
  --as-of-date 2023-01-10
```

Architecture selection:

```bash
JAX_PLATFORMS=cpu python -m \
  quantdata.sdlv.daily.front_spot.hedging.models.gromm.select_architecture \
  --checkpoint-directory /tmp/quant_data/gromm-architecture/checkpoints \
  --output /tmp/quant_data/gromm-architecture/result.json
```
