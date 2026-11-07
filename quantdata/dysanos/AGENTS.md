# DYSANOS model conventions

## Daily AR(1)

All new DYSANOS dynamics estimation uses a **one-business-session AR(1)**,
fitted directly from consecutive daily observations:

\[
h_{t+1}=m+A_1(h_t-m)+\varepsilon_{t+1}.
\]

This is a model convention, independent of how often parameters are
recalibrated. Daily recalibration must not silently reuse a five-session
transition. A complete 250-session window supplies 249 daily transition pairs;
exclude pairs crossing a missing source session from the daily likelihood.
Do not estimate a weekly AR(1) and convert it to daily dynamics.

The daily convention supersedes the paper's weekly estimation for new work:
the weekly construction did not provide a useful improvement and produced
poorer spot/ATM-volatility correlations. Fit spot dependence using the same
daily observations.

Keep historical weekly checkpoints, reports and paper results as explicitly
labelled historical comparisons. Do not rerun or alter them merely to apply
this convention. New weekly experiments require an explicit user request.

See [MODELS.md](MODELS.md) for the current rolling-model specification.
