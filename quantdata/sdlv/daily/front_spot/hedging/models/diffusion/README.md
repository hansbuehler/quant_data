# Adapted sequential volatility diffusion (Renyuan Xu coauthored baseline)

This directory implements the AD-Seq-Vol and AD-Seq-Vol-FT baselines from
Han, Zhang, Torres, Acero, and Renyuan Xu, *Diffusion Models for Dynamic
Volatility Surface Generation and Data-Driven Hedging* (2026). The benchmark
and reports retain the published model names and stable internal identities;
"Renyuan" is not a separate duplicate model.

- **Inputs:** 22-day exact-session sequences reconstructed from SANOS states.
  Each day has a log-IV channel and the normalized forward log return broadcast
  over a common surface grid. The published-grid benchmark uses nine expiries
  and eleven fixed forward-moneyness strikes; the short-end adaptation uses
  1, 2, 3, 4, 5, 7, and 10 business days and nineteen fixed NMS
  coordinates from -3.5 to 3.5. Their actual strikes float with that day's
  expiry-specific SANOS ATM volatility. The first 21 days
  condition the generated day 22.
- **Measure:** historical `P`; neither diffusion variant is risk-neutral.
- **Price:** generated states define scenario values. Observed and off-grid
  benchmark marks use the released SANOS interpolation.
- **Hedge:** daily tests reprice fixed contracts in generated next-day states.
  Weekly tests sample autoregressively, appending each generated surface and
  broadcast return to its path's 21-state context. The multistep policy uses
  exact expiry masks, carried positions, and terminal liquidation, with
  separate long, short, zero, and drift-neutral pure-hedge fits.
- **AD-Seq-Vol:** causal transformer, width 256, eight layers, four heads,
  feed-forward multiplier two, dropout 0.1, and a 200-step cosine DDPM trained
  to predict the clean sequence.
- **AD-Seq-Vol-FT:** loads the EMA base checkpoint, freezes it, and trains a
  zero-initialized rank-four output adapter with differentiable calendar,
  call-spread, and butterfly penalties.
- **Training:** published settings use batch size 256, learning rate `7e-5`,
  weight decay 0.01, 1,600 warm-up steps, cosine decay, 3,000 epochs, EMA 0.995,
  and CUDA mixed precision. The newest 20% of sequences is held out; channel
  normalization is fitted on training sequences only. Rolling fits retain the
  initial checkpoint's normalization because the DDPM noise law is defined in
  normalized coordinates. Rebasing normalization can preserve a deterministic
  denoiser map while changing the generated distribution.
- **Artifacts:** version-checked restartable checkpoints store model, EMA,
  optimizer, scheduler, scaler, CPU/CUDA RNG states, epoch, global step, and
  validation history. Dated rolling checkpoints record the source-fixed
  normalization policy and reject older rebased checkpoints. Changed validation
  splits are incompatible; other
  changed training controls require an explicit warm-start opt-in and
  additional epochs.
  Retained daily fits also write `generated-scenarios/YYYY-MM-DD.npz` below
  their external checkpoint directory. Each artifact contains the sampled
  forward log returns and next-day log-IV surfaces, together with origin and
  forecast dates, dataset/config/dated-checkpoint hashes, source-checkpoint
  hashes, and sampling seed. The daily record authenticates the artifact SHA,
  allowing later hedge objectives to reuse exactly the same paths.
- **Data adaptation:** the published price/IV files are not used in this
  benchmark. The surface is reconstructed from the SANOS state; this changes
  the training distribution relative to the published study. The generated
  spot and IV process omits stochastic forward-curve and discount-factor moves,
  which can materially affect next-day fixed-strike digital marks.

The CPU smoke trains on the public panel through 2023-01-17, hedges the common
ATM one-day digital, emits `hedging-smoke-v2`, and checks a committed fixture:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.diffusion.smoke
pytest -q tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py
```

`DiffusionWeeklyAdapter` implements both frozen-weekly and restarted-weekly
schedules for AD-Seq-Vol and AD-Seq-Vol-FT. Both remain `P` models and therefore
do not report a risk-neutral expectation.

Reference: Han, Zhang, Torres, Acero, and Xu,
[Diffusion models for dynamic volatility surface generation and data-driven hedging](https://arxiv.org/abs/2609.13402).
