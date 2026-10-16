# VolGAN

- **Inputs:** released SANOS states converted to a fixed 9-by-11 annualized
  Black-IV grid, plus released forward-adjusted spot returns. The conditioning
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
- **Training:** public defaults are 10,000 epochs, batch size 100, learning rate
  `1e-4`, and gradient-matched strike/maturity smoothness penalties. The newest
  20% of exact-session examples is untouched validation data. Normalization is
  fitted on training rows only.
- **Artifacts:** version-checked restartable Torch checkpoints containing both
  networks, both optimizers, CPU/CUDA RNG state, progress, gradient-matching
  state, and the train-only normalization buffers. Changed validation splits
  are incompatible; other changed training controls require an explicit
  warm-start opt-in and additional epochs.
- **Deviation from the public code:** surfaces are forward-moneyness SANOS
  reconstructions because proprietary OptionMetrics surfaces are unavailable
  and prohibited. The original repository has no stated license, so this is a
  project-native implementation of the published architecture rather than
  copied source.

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
