# VolGAN

- **Inputs:** SANOS states converted to an annualized Black-IV grid, plus
  forward-adjusted spot returns. The published-grid benchmark uses nine
  expiries and eleven fixed forward-moneyness strikes; the short-end adaptation
  uses 1, 2, 3, 4, 5, 7, and 10 business days and nineteen fixed NMS
  coordinates from -3.5 to 3.5. Their actual strikes float with that day's
  expiry-specific SANOS ATM volatility. The conditioning
  vector contains the previous two annualized returns, trailing 21-session
  realized volatility, and the current log-IV surface.
- **Measure:** historical `P`; the generator is not a risk-neutral model.
- **Price:** generated states define scenario values. Observed and off-grid
  benchmark marks continue to use the released SANOS interpolation.
- **Hedge:** daily tests reprice fixed contracts in generated next-day states.
  Weekly tests recursively update every path's surface, last two returns, and
  trailing 21-session realized volatility, then fit a multistep open-loop
  policy with exact expiry masks, carried positions, and terminal liquidation.
  Long, short, zero, and drift-neutral pure-hedge policies are distinct.
- **Generator:** the public two-layer Softplus architecture with noise dimension
  32 and hidden dimension 16. The discriminator uses one Softplus hidden layer.
- **Optional eight-PC adaptation:** `--volgan-pca-rank 8` retains the spot-return
  output but replaces the 133 log-IV levels in the condition and 133 log-IV
  increments in the target with separate eight-dimensional PCA scores. Both
  bases are fitted only on the chronological training block; validation and
  live states use those frozen bases. Rolling fits warm-started from the 2022
  initializer retain its frozen PCA bases; only latent normalization and
  network weights are updated. This is **not** a rolling PCA. Generated
  increments are decoded to the
  full surface before smoothness penalties, scenario repricing, or recursive
  sampling. This is a paper-inspired dimensionality reduction, **not** the
  published VolGAN architecture. The original full-surface model and its
  checkpoint identity remain unchanged. The initializer manifest records
  training/validation PCA coverage and a conditional-dispersion audit:
  within-state generated spot/IV shocks against out-of-sample residuals from
  a train-only ridge predictor, including spot–first-IV-PC correlation.
  The current checkpoint is the final epoch; `best_validation_loss` is tracked
  for diagnosis but does not identify a separately saved best checkpoint.
- **Training:** public optimizer and parameter defaults are RMSProp, 10,000
  epochs, batch size 100, learning rate `1e-4` for both networks, and 25
  gradient-matching epochs for strike/maturity smoothness penalties. The newest
  20% of exact-session examples is untouched validation data. Normalization is
  fitted on training rows only.
- **Rolling protocol:** `fit_rolling_volgan` creates a dated, SHA-256-checked
  checkpoint chain without generating hedge scenarios. Each fit starts from
  the immediately preceding fitted date; dates lacking an exact one-session
  VolGAN target carry the last checkpoint unchanged and receive a dated gap
  record. `run_daily --rolling-warm-start` applies the same rule to scenario
  hedges. A separate 2022-only one-standard-error selection on the short-end
  PCA8 geometry chose 300 update epochs per date; the 10,000-epoch 2022
  initializer and 10,000 **generated paths per hedge date** are different
  quantities. Selection uses 200 observations because the released 2022 panel
  contains fewer than 252 dates; production fits use 252. This window mismatch
  is an explicit limitation, not hidden tuning on 2023–2024 outcomes.
- **Calibration caution:** the 2022 PCA8 initializer remains conditionally
  under-dispersed in spot and IV shocks on held-out states. Its rolling hedges
  are exploratory and should not be interpreted as a calibrated competitor
  result until that diagnostic improves.
- **Artifacts:** version-checked restartable Torch checkpoints containing both
  networks, both optimizers, CPU/CUDA RNG state, progress, gradient-matching
  state, and the train-only normalization buffers. Changed validation splits
  or optimizers are incompatible; other changed training controls require an
  explicit warm-start opt-in and additional epochs.
- **Deviation from the public code:** the gradient-matching phase currently
  continues from its updated weights rather than restarting from the initial
  weights; the 25 epochs are included in the 10,000 total. The model also uses
  train-only standardization, unlike the paper's unnormalized inputs. These
  remain methodology differences even when optimizer and parameter settings
  match. Surfaces are forward-moneyness SANOS
  reconstructions rather than the paper's OptionMetrics surfaces. The original
  repository has no stated license, so this is a
  project-native implementation of the published architecture rather than
  copied source.
- **Short-end limitation:** simulated spot and IV do not include stochastic
  forward-curve or discount-factor moves. Such moves affect the next-day marks
  of fixed-strike digitals and must be measured separately in the hedge audit.

The CPU smoke trains on the public panel through 2023-01-17, hedges the common
ATM one-day digital, emits `hedging-smoke-v2`, and checks a committed fixture:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.volgan.smoke
pytest -q tests/quantdata/sdlv/daily/hedging/models/test_volgan.py
```

`VolganWeeklyAdapter` implements both frozen-weekly and restarted-weekly fit
schedules. VolGAN remains a `P` model and therefore reports buyer/seller
indifference prices, not a risk-neutral expectation.

Reference: Vuletić and Cont,
[VolGAN: A generative model for arbitrage-free implied volatility surfaces](https://doi.org/10.1080/1350486X.2025.2471317).
