# Local volatility by SANOS

## Project objective

This project recovers discrete local-volatility (DLV) surfaces from the smooth,
arbitrage-free SPX option surface produced by `quantdata.dysanos.localvolbysanos.markets`. It then
turns those surfaces into a balanced time-series panel and studies their first
statistical representation:

1. construct a numerically stable forward-relative strike grid;
2. invert arbitrage-free call prices into SANOS densities and DLVs;
3. verify that the recovered DLVs reproduce the input option prices;
4. measure the arbitrage-free surface's fit to observed bid/ask quotes;
5. build a fixed-shape 2025 panel in effective-NMS coordinates;
6. compute spatial PCA factors for log DLV squared and log forward volatility;
7. estimate a matrix-exponential AR(1) model for each factor state; and
8. generate the diagnostic graphs and 3D surface video.

The current work is an experiment, not a production local-volatility model.
The complete numerical findings are recorded in [RESULTS.md](RESULTS.md).

## Project isolation

The code in this directory may use general production packages such as
`quantdata.dysanos` and `quantdata.sanos`, but it must not import from `aae` or
another research project. Functionality that becomes genuinely reusable should be
moved to a general non-project package rather than creating an ideas-to-ideas
dependency.

## Market input

`quantdata.dysanos.localvolbysanos.markets.load_market_day` is the only market loader used by this
project. It returns:

- filtered forward-normalized SPX option quotes;
- expiries and forward-relative strikes `K/F`;
- bid and ask prices in normalized call-price units; and
- an `ArbFreePricer` that prices arbitrary strikes and expiries on the fitted
  arbitrage-free surface.

The experiments use surface smoothness `0.25`. The AF pricer's `surface` mode
is used for DLV inputs so the calls are arbitrage-free jointly across strike and
expiry.

## Coordinates and model grid

The model expiries are

```text
{2, 5, 10, 15, 20, 30, 40, 50, 70, 120} / DAYS_PER_YEAR.
```

All strikes stored or modeled here are dimensionless forward-relative strikes
`K/F`; spot strikes are never used as statistical coordinates.

The initial grid starts from 100 NMS points over `[-3, 2]`. The points are
mapped to log strike at the 5-day and maximum model expiries using the
respective ATM volatilities from the AF pricer. The 5-day grid is retained and
the longer grid is merged after rejecting candidates that are too close to an
existing log strike. The ATM point is forced to exactly `K/F = 1`.

For the original nested-grid experiment, each expiry keeps reference strikes
between the strike bounds implied by that expiry's own ATM volatility and the
requested NMS endpoints. SANOS boundary strikes are added at 50% of the lowest
and 150% of the highest inner strike.

## Full-reference-grid inversion

Calendar differences are cleanest when the same forward-relative strikes are
present at every expiry. The annual calculation therefore evaluates densities,
call increments, and DLVs on the complete common reference grid at every model
expiry. This removes interpolation across expiries and reduces negative call
increments to numerical noise.

The direct inversion is:

1. price calls on the common strike grid with the AF surface;
2. recover each discrete terminal density from strike-convex call prices;
3. difference calls between consecutive model expiries on identical strikes;
4. apply the discrete Dupire/SANOS flux identity to infer DLV squared; and
5. run those DLVs forward from the time-zero point mass and compare the
   reconstructed calls with every AF input price.

A second price-fitting SANOS linear program is deliberately not used. With
loose DLV bounds, many densities can reproduce the same smooth call prices and
the LP can select a non-identified solution with DLVs at the upper bound.

Deep short-expiry wings can have density and calendar flux at machine precision.
DLVs at those locations are undefined or poorly identified even if input prices
are recovered accurately. The raw panel retains this information rather than
silently presenting the wing values as reliable local volatilities.

## Balanced statistical panel

The full common grid is required for the inversion, but a statistical sample
also needs the same number of observations at each date and expiry. The annual
runner therefore makes a second, balanced representation after the exact
full-grid calculation.

For each date, the 5-day ATM volatility scales the NMS transformation at every
expiry. Because the model expiries and the 100 NMS coordinates are fixed, every
sample uses the same effective relative-strike positions. The actual `K/F`
values may move with front-week volatility, which is intentional. Projection
onto this panel is only across strike within an already computed expiry; there
is no interpolation across expiry or calendar time.

The complete finite support shared by all 2025 samples is
`[-1.40902, 0.93935]` in effective NMS. The resulting tensor has shape
`[date, expiry, strike] = [163, 10, 100]` for both:

- `log_dlvsq = log(DLV^2)`; and
- `log_forward_volatility`.

The logarithms reduce the effect of scale changes and make positive states
unconstrained for the statistical model.

## Bid/ask fitting diagnostics

`fit_diagnostics.py` evaluates the AF surface on every retained market quote.
For fitted normalized price `p`, the reported spread error is

```text
distance(p, [bid, ask]) / max((ask - bid) / 2, 1e-4).
```

It is exactly zero when the price is inside bid/ask. Results are stored at the
quote, expiry, date, and month levels. The graph shows the daily percentage
inside bid/ask and the conditional mean and median errors among outside-spread
quotes.

The large 2025-02-14 conditional-mean spike is a long-expiry surface effect.
The independent per-expiry fits put all quotes inside bid/ask, whereas the
cross-expiry surface adjustment moves a cluster of narrow-spread 220--273 day
quotes outside. It is not a front-end or broad per-expiry fit failure.

## PCA surface factors

`surface_dynamics.py` centers every one of the 1,000 spatial grid points over
time and applies an SVD to the flattened `10 x 100` surface. PCA is performed
separately for log DLV squared and log forward volatility. Component signs are
fixed deterministically by requiring the loading with the largest absolute
value to be positive.

The reusable outputs contain:

- explained and cumulative variance for every available component;
- the first five loading surfaces;
- the first five daily factor scores; and
- 3D plots of the first three loading surfaces.

The 3D axes are effective NMS strike, `sqrt(expiry_years)`, and PCA loading.

## Matrix-exponential AR(1)

The AR model is estimated in PCA factor space, because a free `1000 x 1000`
transition matrix cannot be identified from 163 dates. The primary model uses
five factors. Each score is divided by its sample standard deviation during
estimation so that the nonlinear least-squares objective does not let the first
factor dominate.

For each state the conditional mean is

```text
z(t + dt) = exp(G * dt) z(t) + error,
```

where `dt` uses the actual calendar gap in years. Zero factor score corresponds
to the sample mean surface. Spot is not included: the two models contain only
log DLV squared or log forward volatility.

Mean reversion is checked from the fitted generator `G`. The estimate is mean
reverting when every eigenvalue of `G` has negative real part, equivalently when
the matrix exponential has spectral radius below one for a positive horizon.
The report contains the generator, one-calendar-day transition, eigenvalues,
mode half-lives, spectral radius, optimizer status, and in-sample R-squared.
The check is repeated for PCA ranks one through ten to show whether the result
depends on the selected rank.

This is a first in-sample benchmark. It does not yet contain a train/test split,
parameter uncertainty, regularization, or an innovation covariance model.

## Code layout

- `experiment.py`: grid construction, direct DLV inversion, forward price
  recovery, two-day experiments, and DLV curve plot.
- `year.py`: available-date annual runner, raw full-grid output, balanced panel,
  daily diagnostics, and annual ATM-state graph.
- `fit_diagnostics.py`: quote-level bid/ask errors and time-series graph.
- `surface_dynamics.py`: PCA, rank checks, matrix-exponential AR(1), PCA 3D
  graph, and MP4 surface animation.
- `bid_ask_estimate/`: isolated subproject for quoted option half-spread and,
  later, effective execution-cost estimation. Its design and literature review
  are in `bid_ask_estimate/BID_ASK_ESTIMATE.md`.
- `real_trading/`: scenario-based OCE optimization over actual listed options,
  including joint spot/surface dynamics, pre-horizon expiry treatment, and the
  implemented 2025 `normalised`/`actual`/`ITM` diagnostic. Its design,
  reproduction command, results, and limitations are in
  `real_trading/REAL_TRADING.md`.
- `RESULTS.md`: detailed empirical results and limitations.
- `results/`: generated Parquet, JSON, PNG, and MP4 artifacts.

## Key results

- The two-day direct inversion has median DLV near 13% and does not exhibit the
  LP-induced upper-bound explosion.
- A full common strike grid removes material negative calendar call increments
  and accurately recovers the AF input prices on the pilot dates.
- The available 2025 panel contains 163 dates from 2025-01-02 through
  2025-08-29, ten expiries, and 100 effective-NMS strikes per expiry.
- The annual sample exposes poorly identified deep-wing DLV spikes when density
  is tiny. The maximum balanced-panel DLV is 21.85.
- Input-price recovery passes the `1e-8` tolerance on 158 of 163 dates. Two
  dates have material maximum normalized errors near `2.62e-3` and require
  further investigation.
- Across 162,699 retained quotes, 96.91% of AF surface prices are inside
  bid/ask. The all-quote mean and median errors are 0.0450 and zero
  half-spreads.
- Five PCA factors explain 91.90% of log-DLV-squared variation and 99.68% of
  log-forward-volatility variation.
- Both five-factor matrix-exponential AR estimates are mean reverting. Their
  slowest fitted half-lives are 19.23 and 21.92 calendar days. Every tested rank
  from one through ten is also mean reverting.

## Recreating the results

Run commands from the repository root with the `vqdata` Python environment.
The examples below use its absolute interpreter path on Linux.

### Prerequisites

The market-data steps require access to the data used by
`quantdata.dysanos.localvolbysanos.markets`. Cached dates run without a new download; uncached dates
may require configured WRDS or other market-data credentials.

Graphs require Matplotlib. MP4 creation requires either a system `ffmpeg`
executable or the small Python wrapper that provides one:

```bash
/home/hans/vqdata/bin/python -m pip install imageio-ffmpeg
```

`surface_dynamics.py` first looks for system `ffmpeg` and then for
`imageio_ffmpeg`.

### 1. Original two-day nested-grid experiment

```bash
/home/hans/vqdata/bin/python -m quantdata.dysanos.localvolbysanos.experiment
```

This writes `results/two_day_summary.json` and `results/two_day_dlvs.png`.

### 2. Two-day full-reference-grid ablation

```bash
/home/hans/vqdata/bin/python -c "from pathlib import Path; from quantdata.dysanos.localvolbysanos.experiment import ExperimentConfig, run_experiment; run_experiment(config=ExperimentConfig(full_reference_grid=True), output_directory=Path('quantdata/dysanos/localvolbysanos/results/full_reference_grid'))"
```

This also performs the forward input-price recovery check.

The separate front-week-scaled-grid pilot can be reproduced with:

```bash
/home/hans/vqdata/bin/python -c "from pathlib import Path; from quantdata.dysanos.localvolbysanos.experiment import ExperimentConfig, run_experiment; run_experiment(config=ExperimentConfig(front_week_vol_scaled_nms=True), output_directory=Path('quantdata/dysanos/localvolbysanos/results/front_week_scaled_nms'))"
```

This grid remains forward-relative: its stored strikes are `K/F`. Front-week
volatility scales the common NMS coordinate; it does not turn the strikes into
spot-relative levels.

### 3. Build the annual raw and balanced panels

```bash
/home/hans/vqdata/bin/python -m quantdata.dysanos.localvolbysanos.year
```

This creates or replaces the following files under
`results/2025_full_reference_grid/`:

- `raw_full_grid.parquet`;
- `balanced_panel.parquet`;
- `daily_summary.parquet`;
- `report.json`; and
- `year_state.png`.

The annual run is the expensive step because it reloads and inverts every
eligible market date. The runner uses the current repository data-validity
cutoff, so a future run may include more dates than the recorded 163-date
sample if more 2025 data become available.

### 4. Recompute market-fit errors and their graph

```bash
/home/hans/vqdata/bin/python -m quantdata.dysanos.localvolbysanos.fit_diagnostics
```

The exact dates are read from `balanced_panel.parquet`. This regenerates the
quote, expiry, daily, and monthly Parquet files, `market_fit_report.json`, and
`market_fit_time.png`.

### 5. Recompute PCA, AR estimates, 3D plots, and video

```bash
/home/hans/vqdata/bin/python -m quantdata.dysanos.localvolbysanos.surface_dynamics
```

This reads only `balanced_panel.parquet` and regenerates:

- `pca_summary.parquet`;
- `pca_loadings.parquet`;
- `pca_scores.parquet`;
- `mean_reversion_rank_check.parquet`;
- `surface_dynamics_report.json`;
- `pca_factors_3d.png`; and
- `surface_history_3d.mp4`.

The video contains all 163 available sessions at 12 frames per second. It shows
the log-DLV-squared and log-forward-volatility surfaces side by side with fixed
vertical scales, effective NMS on the strike axis, and `sqrt(T)` on the expiry
axis.

To recompute the statistical files and PCA graph without rendering MP4, run:

```bash
/home/hans/vqdata/bin/python -c "from quantdata.dysanos.localvolbysanos.surface_dynamics import run_surface_dynamics; run_surface_dynamics(render_video=False)"
```

### 6. Verification

The following checks cover the project modules changed by this experiment:

```bash
ruff check quantdata/dysanos/localvolbysanos
/home/hans/vqdata/bin/python -m py_compile \
  quantdata/dysanos/localvolbysanos/experiment.py \
  quantdata/dysanos/localvolbysanos/year.py \
  quantdata/dysanos/localvolbysanos/fit_diagnostics.py \
  quantdata/dysanos/localvolbysanos/surface_dynamics.py
```

The commands intentionally overwrite artifacts with the same names. Preserve a
copy first if results from different data cutoffs or configurations need to be
compared.
