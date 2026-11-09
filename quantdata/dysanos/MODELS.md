# DYSANOS models

This is the single current model/status summary. Implementation and mathematical
review details are in [REVIEW.md](REVIEW.md); operational rules are in
[AGENTS.md](AGENTS.md). Historical reports retain their original definitions.

## Current dynamics convention

**DYSANOS always uses daily AR(1) for new model estimation.** The transition
is one business session, estimated directly from daily observations. It is
not a five-session AR(1) converted to a daily law. Weekly models remain
historical comparisons only; the weekly construction did not improve the
model and gave poorer spot/ATM-volatility correlations.

## Current rolling experiment

- Purpose: market calibration and realistic physical (`P`) dynamics; this
  run does not construct or establish a risk-neutral (`Q`) law.
- Full affine-200 and stable daily AR(1) recalibration every business day.
- A rolling 250-date window, with up to 249 one-session transition pairs;
  exclude pairs crossing missing source sessions.
- Warm-start all trainable parameters from the previous day's fit; retain
  overlapping fitted states and estimate the new day's factor state.
- Retain the window's data in VRAM, dropping the oldest date and transferring
  only the new date.
- Date weights are `w(t-k) = 4^(-k/249)`, for `k=0,...,249`: 100% today and
  25% on the oldest date. Normalize weighted losses by their weight sum;
  transition pairs use the endpoint-date weight.
- The new model has business-day expiries
  `1,2,3,4,5,6,7,8,9,10,12,15,20,25`, NMS `[-4,3]`, and linear interpolation
  of total backbone volatility in time; 200 strikes and 200 affine factors.
- Fit to licensed market bid/ask quotes. Reuse historical SANOS surfaces
  with smoothness 0.1 for initialization and reference grid prices, without
  refitting those source surfaces or replacing market targets by fitted prices.
- Preserve the previous 70-business-day model and its results.

The implementation is `v11/rolling_daily.py`. It runs the entire sweep in
one allocated GPU process, saves a distinct checkpoint and fit report for
each origin, and writes `progress.json` and `daily-rolling-fit.png` in its
external run directory. `COMPLETE.json` is written only after the last
available date has finished. The current source panel has 661 calibration
origins from 30 December 2022 through 29 August 2025, the last available
2025 source date. All 661 origins completed on 9 November 2026; the producer
and reporting services have exited and the GPU allocation is released.
Its run label is `daily-ar1-25d-full-sweep-20261107-v1`, under
`experiments/dysanos/v11/` in the configured external data root.

`v11/daily_fit_report.py` maintains `daily-fit-report.csv` alongside the
sweep. Each row reports that origin's own-day fit, the decay-weighted mean
and median fit of its previous 249 dates under the same fitted model, and
the daily update's runtime in seconds and minutes. Today is excluded from
the historical comparison; the median is the inverse weighted empirical
CDF at 50%.

The completed sweep's own-day bid/ask containment is:

| Statistic | Fit |
| --- | --- |
| Median across all 661 origins | 90.95% |
| Median across the final 50 origins | 85.01% |
| Final origin, 29 August 2025 | 84.93% |
| Origins reaching at least 90% | 377 / 661 |

The final origin's previous-249-date decay-weighted mean and median are
88.72% and 88.51%. The producer loop took 36.36 hours; the median daily
update after initialization took 78 seconds. There were 518 market-plateau
stops and 143 stops at the 1,000-iteration ceiling, not 661 convergence
certificates. Recent fit is materially weaker than the full-sweep median.

The external run's `review/fit-review.csv` consolidates fit, grid error,
conditional excess, convergence and runtime for every origin.
`review/completion-audit.json` records the final coverage and consistency
checks. Older 70-day checkpoints were preserved; no path-validation or
risk-neutrality conclusion follows from completion of this sweep.

Fit percentages are market-price calibration diagnostics, not AR forecast
or simulated-path validation. Per-origin reports retain convergence flags,
the 10th/90th percentiles of daily window fit, and grid-price errors against
the source SANOS surfaces. A fit plateau can end recalibration before all
optimizer tolerances are met.

Missing source sessions are recorded in the run manifest. Transitions
crossing a missing session are excluded from the one-session likelihood;
they are never treated as one-day returns.

The earlier November 2026 pilot fits used a five-session AR transition and
must not be relabelled as daily AR(1) results. The older `v11/jax_joint_ar.py`
is a historical pilot, not the daily rolling runner.

## Historical comparators

These are earlier experiments, not additional models being trained by the
current sweep. Their original studies used 252-session windows or pre-2025
fits, not the new weighted 250-date calibration.

| Model | Disposition and measured limitation |
| --- | --- |
| Paper affine-200 with weekly dynamics | Preserved baseline, including older 70-day results. New estimation uses daily transitions instead. |
| Daily joint Direct-200, Gaussian | Earlier selected dynamics benchmark; improved 2025 predictive calibration relative to the weekly law. Distinct from the current rolling fit. |
| Daily/weekly Tail Diffusion | Retained tail comparators. Heavier innovation-radius tails, but the stationary-scale daily version was over-dispersed on the 2025 holdout. |
| LSTM and selective-SSM laws | Retained references. Conditional NLL improved; no material improvement was established for targeted multistep path statistics. |
| Direct daily-spot MLP; regularised/rolling DLV-backbone spot-loading overlays | Retired. Local regression improvements did not establish better generated leverage term structure. |
| DLV-only diagnostic | Retired. Established that a fixed Black backbone alone does not explain the leverage shortfall. |

The original post-paper plan compared fixed versus per-origin grids before
AR and tail estimation. Earlier leverage research requires joint daily
spot/surface innovation fitting, decoded tenor-by-tenor leverage validation
and gauge alignment before concatenating latent histories from different
origins. These remain dynamics-review requirements, not properties already
demonstrated by the new rolling calibration. Retired means no further runs,
not permission to delete retained checkpoints.
