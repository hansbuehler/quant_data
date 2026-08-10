# Direct implied-volatility PCA benchmark

Throughout this document,
$h_t=(h_t^1,\ldots,h_t^{n_h})$ denotes only the implied-volatility PCA state
and $s_t=\log S_t$ denotes log-spot. The implementation may store the combined
array as $xstate=(s,h)$, but spot is never a component of $h$.

## Objective

Implement a deliberately simple generative benchmark for DYSANOS:

1. load the same regular SPX normalized-moneyness "nms" grid used to train ML-SANOS;
2. represent log-implied-volatility surface levels by PCA;
3. fit an autonomous AR(1) model to the PCA surface state, without spot;
4. add log-spot conditionally on the fitted surface innovations;
5. simulate coherent spot and option paths with 100,000 and 1,000,000 independent paths; and
6. run the same static, unconditional, conditional-tail, and importance-sampling arbitrage diagnostics used for the SANOS generator.

The purpose is to determine whether the statically arbitrage-free SANOS decoder materially improves arbitrage behaviour relative to a direct linear PCA decoder. This is called the **PCA-IV-AR(1) benchmark**. It is not an implementation of Cont and Vuletić's scenario-reweighting method: do not penalize, reject, project, or reweight paths according to a static-arbitrage score.

Static no-arbitrage and dynamic no-arbitrage must remain distinct. The experiment may show that sampled fixed-contract returns do not contain detected one-sided opportunities. It does not prove absence of arbitrary dynamic arbitrage.

## Operating perimeter

1. Follow the repository-wide `STYLE_GUIDE.md`.
2. Put the implementation and generated comparison reports under `quantdata/dysanos/implied/`. Put tests under `tests/`.
3. Reuse read-only utilities from `quantdata/dysanos/grid/`, especially the surface-first AR fit, exact Gaussian transition, terminal-universe timing, importance sampler, and arbitrage diagnostics. Do not create subtly different definitions of returns, deltas, costs, roundoff, or path weights.
4. If a generic helper must be made public or factored out of `quantdata/dysanos/grid/`, make the smallest possible change and preserve the SANOS behaviour with regression tests.
5. Cache the fitted PCA model and generated universes with versioned configurations. Never cache a result solely by path count or filename.
6. Use streaming/path blocks for the million-path experiment. Do not materialize a second full copy of the universe, and do not copy the full universe to a GPU.

## Scope of the comparison

`GridMarketDefinition.load()` does not return untouched raw quotes. It first fits a smooth statically arbitrage-free SANOS surface to each date and evaluates that surface on the regular grid. Therefore this experiment answers:

> Does the statically arbitrage-free SANOS **generative decoder** improve on a direct PCA decoder when both are trained from the same cleaned and SANOS-interpolated historical grid?

It does not answer whether SANOS preprocessing improves on a raw OptionMetrics interpolation. Record this qualification in every aggregate comparison report.

Do not use `market["fit_vols"]` as the PCA input. Use `market["mid_vols"]`, matching the implied-volatility observations supplied to ML-SANOS. In the existing ML-SANOS path, `mid_vols` enters the encoder features and the loss fits the corresponding `market["mids"]` prices. `fit_vols` is instead the output of the separate, per-date small-SANOS refit performed while constructing the grid; it is not the ML-SANOS training target. The PCA benchmark must therefore fit its surface representation to `mid_vols`, not to `fit_vols`.

## Experiment configuration

Create a versioned frozen configuration, for example `ImpliedPCAExperiment`, with at least:

- `num_samples`: `100_000` or `1_000_000`;
- `grid_strikes=100`: training-grid points per expiry, matching the ML-SANOS data extraction;
- `num_traded_nms=20`: traded strike points per expiry, matching the SANOS terminal universes;
- `num_surface_state=20`: number of PCA level scores retained as the surface state;
- `num_alpha=5` or `None`: five surface innovation factors or all 20;
- `use_matrix_exp=True`;
- `bandwidth=1.0`: Gaussian innovations only;
- `trading_frequency`: `daily` or `weekly`;
- `return_horizon`: `expiry` or `incremental`;
- `num_trading_times=10`: ten entry cohorts for expiry returns and nine
  consecutive intervals for incremental returns;
- `dt=1/DAYS_PER_YEAR`;
- `seed`, `block_size`, and the spot-importance-sampling configuration.

Use the same logical seed, initial-date indices, action times, expiry offsets, traded normalized-moneyness points, and diagnostic configuration in the PCA and SANOS runs wherever possible. Record every configuration in the output JSON.

## Market data

Use exactly this loading route:

```python
print("** Starting to load market data...")
grid_config = GridConfig(grid_num_strikes=experiment.grid_strikes)
grid_market_def = GridMarketDefinition(
    ticker="SPX",
    date_range=DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30)),
    grid_config=grid_config,
)
market = grid_market_def.load(
    pool_config=PoolConfig(num_workers=4, threading=False),
    override_cache_mode="cacheonly",
    verbose=Context.all,
)
print(f"nms /done - found {len(market['dates'])} dates.")
```

Use these fields:

| field | meaning | expected geometry |
|---|---|---|
| `dates` | historical observation dates | `(D,)` |
| `time_y` | business time in years | `(D,)` |
| `mid_vols` | PCA target implied volatilities | `(D,T,N_grid)` |
| `nms` | normalized-moneyness coordinates | `(D,T,N_grid)` |
| `expiries` | time-to-expiry grid | `(D,T)` |
| `xstrikes` | spot-normalized strikes, including two boundaries | `(D,T,N_grid+2)` |
| `spot_ref` | cash spot | `(D,)` |
| `spot_ret` | historical spot return supplied by the pipeline | `(D,)` |

Validate the actual arrays rather than relying only on the table. Require finite, strictly positive implied volatilities and spots; increasing expiries; and a common `nms` grid across dates. Use `time_y` rather than assuming all adjacent observations are exactly one calendar day apart.

## PCA surface representation

Let

$$
    y_d:=\operatorname{vec}(\log \sigma_d)\in\mathbb R^{TN_{grid}}.
$$

Use log implied volatility so the direct decoder is positive without clipping. Fit ordinary centered, unweighted PCA to the **levels** $y_d$:

$$
    \bar y=\frac1D\sum_dy_d,
    \qquad
    h_d=(y_d-\bar y)B',
$$

where the rows of $B\in\mathbb R^{n_h\times TN_{grid}}$ are orthonormal and $n_h=20$. Decode by

$$
    \widehat\sigma(h)
      =\exp\left(\operatorname{unvec}(\bar y+hB)\right).
$$

Do not standardize each grid coordinate before PCA and do not vega-weight the primary benchmark. Do not perform PCA on daily changes: daily-change PCA is a diagnostic, whereas $h_d$ is the surface-level state. Resolve PCA signs deterministically for reproducible plots.

Report:

- explained and cumulative variance of the first 20 components;
- reconstruction RMSE and maximum error in IV and log-IV;
- errors by expiry and normalized-moneyness region; and
- the first five IV eigensurfaces on the same $7\times100$ grid used for SANOS diagnostics.

The primary benchmark is exactly 20 level components. Any alternative dimension or direct-IV rather than log-IV sensitivity must be labelled as a separate experiment and must not replace it.

## Surface-first AR(1) and spot

Fit the surface score process before using spot. In column-vector notation,

$$
    dh_t=\kappa(m-h_t)dt+\Sigma\,dW_t^h.
$$

Fit the affine drift to the observed irregular business-time transitions. Apply innovation PCA after the drift fit, retaining either five factors or all 20 according to `num_alpha`. Use the existing surface-first fitting logic in `quantdata/dysanos/grid/torch_simulator.py`; do not regress the surface drift on current or lagged spot.

Only after the surface process is fixed, fit

$$
    ds_t=\mu dt+\beta' dW_t^h+\varsigma dW_t^s,
    \qquad s_t=\log S_t,
$$

where $W^s$ is independent of $W^h$. Include the orthogonal spot residual even when all 20 surface factors are retained.

Use the exact affine Gaussian transition for every simulation step:

$$
h_{t+\Delta}=A_\Delta h_t+(I-A_\Delta)m+\epsilon_\Delta,
\qquad
\epsilon_\Delta\sim N(0,Q_\Delta),
$$

where

$$
\begin{aligned}
    A_\Delta&=e^{-\kappa\Delta},\\
    Q_\Delta&=\int_0^\Delta e^{-\kappa s}
       \Sigma\Sigma'e^{-\kappa's}\,ds.
\end{aligned}
$$

Pre-compute these matrices by distinct step size. Do not use an Euler approximation in the reported experiments. Verify that every eigenvalue of $\kappa$ has positive real part. Report the fitted invariant Gaussian surface distribution $N(m,P)$, where $\kappa P+P\kappa'=\Sigma\Sigma'$, and compare it with historical PCA states using the same mean, covariance, sliced-Wasserstein, and coordinate-KS metrics as the SANOS diagnostics.

Initialize reported path universes from sampled historical surface states and their matching historical spots, exactly as in the matched SANOS run. Use invariant initialization only for the separate invariant-distribution diagnostic.

## Continuous implied-volatility decoder

The PCA output is a regular normalized-moneyness grid, not yet a price function for arbitrary fixed contracts. Implement one documented continuous interpolation used consistently by repricing and static-arbitrage diagnostics:

1. decode positive IVs on the $(\hat T_j,z_i)$ grid;
2. convert to total variance $w(\hat T_j,z_i)=\widehat\sigma^2\hat T_j$;
3. add the anchor $w(0,z)=0$;
4. interpolate total variance linearly in $T$ and $z$;
5. use flat extrapolation in $z$ and report how often it is used; and
6. convert between $z$ and relative strike $K$ with the repository's `NormalizedStrikes` convention and the interpolated ATM volatility.

This interpolation is part of the PCA benchmark and must be versioned. Do not repair its output by projecting prices or volatilities onto an arbitrage-free set. Do not clip calendar slopes, butterflies, prices, or PCA scores. Positive IV follows from the log transform; all other violations must remain visible to the diagnostics.

Price normalized calls with the repository Black--Scholes implementation and zero rates/dividends. Check intrinsic and upper bounds before implied-volatility inversion.

For a fixed cash contract $(\tau,k)$ at observation time $t$,

$$
    T_t=\tau-t,
    \qquad K_t=k/S_t,
    \qquad
    C_t(\tau,k)=S_t c_t(T_t,K_t).
$$

At $t+\Delta$, reprice the **same** contract at $(T_t-\Delta,k/S_{t+\Delta})$. Never compare the same moving normalized-moneyness grid coordinate across dates and call it a fixed-option return.

Compute cash delta from the complete fixed-surface price map

$$
    \Delta_t(\tau,k)
      =\partial_S\left[S c_t(T_t,k/S)\right]_{S=S_t}.
$$

This includes the strike dependence of interpolated IV. Implement it by a stable central derivative or automatic differentiation and test it against the analytic Black--Scholes delta for a flat surface. Do not use the Black--Scholes delta while silently ignoring the smile derivative.

## Coherent path universes

Build the same four markets as the current SANOS terminal experiment:

- `one-step`: one action at day 0;
- `one-step-weekly`: one action at day 0 with the spot instrument measured to day 5;
- `path`: nine daily actions;
- `weekly`: nine actions at daily indices $0,5,\ldots,40$.

Use expiry offsets approximately equal to 2, 5, 10, 20, 40, 126, and 252 business days. Each path must be simulated daily through the farthest expiry required by its last action.

At action date $t$, choose the 20 traded normalized-moneyness locations as a deterministic subset of the 100-point training grid. Define each option's cash strike and calendar expiry at entry. Every option expiry on one Monte Carlo sample must use the same simulated spot path; never draw a separate terminal spot by expiry or strike.

For Monte Carlo realization $r$, an expiry payoff is

$$
    dC_{r,t,j,i}
      =(S_{r,t+e_j}-k_{r,t,j,i})^+-C_{r,t,j,i}.
$$

The expiry-matched delta-hedged return and gross trade notional are

$$
\begin{aligned}
    dH_{r,t,j,i}
      &=dC_{r,t,j,i}
        -\Delta_{r,t,j,i}(S_{r,t+e_j}-S_{r,t}),\\
    G_{r,t,j,i}
      &=|C_{r,t,j,i}|+|\Delta_{r,t,j,i}S_{r,t}|.
\end{aligned}
$$

Also implement the matched one-day repricing return as a secondary diagnostic. Clearly distinguish it from the terminal-payoff return in filenames and reports.

Generate universes in blocks and update diagnostics online. If a diagnostic requires neighbour search, persist only the compact state, return, roundoff, weight, and instrument descriptors needed by that diagnostic.

## Spot-tail importance sampling

Use the same conditional spot proposal as the surface-first SANOS generator:

$$
    q=\alpha p+\frac{1-\alpha}{M}\sum_{m=1}^M q_m,
$$

with default spot standard-deviation multiplier 8, base probability $\alpha=0.1$, and three target action/expiry steps. Conditional on the complete surface path, widen only the scalar orthogonal spot residual and reconstruct the spot path by the exact Gaussian bridge. Preserve the complete simulated surface path.

Every physical-measure statistic must use the complete path likelihood $p/q$ exactly once. Report:

- $E_q[p/q]$;
- minimum and maximum log weight;
- effective sample size and fraction;
- proposal-component counts;
- direct-$p$ versus weighted-proposal means and variances of target log-spots; and
- direct-$p$ versus weighted-proposal mean and variance of realized log variance.

Use direct-$p$ mixture paths as query states and likelihood-weighted proposal paths as neighbour/reference observations, matching the SANOS diagnostic.

## Static surface-arbitrage diagnostics

This diagnostic is specific to the PCA decoder and must be run before return-based tests. Evaluate every generated surface at every action date without repair.

On a fixed dense relative-strike mesh common to adjacent expiries, compute normalized call prices $c(T,K)$. Test:

1. bounds: $(1-K)^+\le c(T,K)\le1$;
2. call monotonicity and vertical-spread slope bounds;
3. convexity in $K$ using correct irregular-grid butterfly weights; and
4. calendar monotonicity $c(T_{j+1},K)\ge c(T_j,K)$ at the same $K$.

Measure violations in initial normalized cash value, not only derivative units. Propagate an arithmetic roundoff band from the actual call prices and portfolio weights. Report sensitivities at roundoff multipliers `0, 1, 8, 32, 128`, with 32 as the primary float32 convention.

For each constraint and in aggregate report:

- violating surfaces and fraction;
- paths with at least one violating surface and fraction;
- number of violating grid portfolios;
- mean, 95%, 99%, and maximum violation conditional on violation;
- expiry and strike/NMS location of the worst violations; and
- the same statistics on the historical PCA reconstructions before AR simulation.

SANOS should be evaluated by the same numerical checker as a control. Its decoded surface should pass up to the propagated arithmetic band; otherwise the checker is wrong or the queried extrapolation is outside the guaranteed SANOS domain.

## Return-based arbitrage diagnostics

When PCA-IV is used as the competitive generator, the market matrix, return
semantics, costs, importance sampling, 80:20 reference/audit split, staged run
order, and no-arbitrage diagnostics in
`quantdata/dysanos/grid/gendysanos/GENDYSANOS.md` are authoritative. This
document remains authoritative for the PCA fit and decoder. Do not create a
weaker PCA-only test or retain a conflicting legacy split.

For every model, factor count, sample count, and market mode run:

### 1. Unconditional sign and tail coverage

- Count positive, negative, and neutral expiry-matched delta-hedged returns per option using the float32 roundoff band.
- Count paths with at least one positive and at least one negative return per option.
- Report minority sign probabilities under the physical likelihood weights.
- Report instruments with zero observations on either side and the corresponding 95% binomial upper probability.
- Report raw spot crossings, option payoff counts, and both break-even sides of the delta-hedged payoff.
- Rank the worst 20 instruments by minority probability and normalized imbalance.

### 2. Conditional local tails

Use $(s_t,h_t)$ as the raw state. Do not include the full IV grid, strikes, or constant expiries in the state: they are deterministic functions of the PCA state and instrument identifier.

Use the current SANOS settings unless a matched report records otherwise:

- 1,024 nearest neighbours at 100,000 paths and 4,096 at one million paths;
- up to 8,192 candidate neighbours per instrument;
- tail ranks 1, 2, 10, 20, and 50;
- signal thresholds 2 and 3;
- 1,024 query states per reference and audit split;
- 20-instrument valuation blocks;
- leave-one-path-out reference queries; and
- an 80:20 independent-path reference/audit split, stratified by importance
  mixture component.

Report local positive/negative probabilities, positive/negative parts, quantiles, lower CVaR, neighbour radii, local effective sample sizes, one-sided candidate states, and realized adverse outcomes for any inferred single-instrument policy. Progress output must show current long-only, short-only, and both-direction candidate counts.

### 3. Cost-adjusted local directions

When the matched SANOS report includes transaction costs, reuse exactly its cost definition and calibration sample size. At minimum include the one-basis-point proportional cost based on

$$
    |C|+|\Delta S|,
$$

not the net option-plus-hedge value. Test long and short directions separately.

### 4. Numerical integrity

Fail immediately on non-finite states, IVs, prices, deltas, returns, likelihoods, or costs. Count interpolation wing hits. Confirm array shapes and the independent-path interpretation of every standard error.

These diagnostics search for simple unconditional and state-conditional one-instrument arbitrage. They are not a complete multi-instrument dynamic-arbitrage test.

## Required experiment matrix

Run the following PCA-IV-AR(1) matrix:

| surface innovations | paths | modes |
|---|---:|---|
| `num_alpha=5` | 100,000 | all four |
| `num_alpha=5` | 1,000,000 | all four |
| `num_alpha=None` | 100,000 | all four |
| `num_alpha=None` | 1,000,000 | all four |

Run the 1,000-path smoke version of every distinct code path first. The 100,000-path run is the stage gate: do not launch one million paths until its reports are complete, finite, and internally consistent.

Compare against configuration-matched SANOS reports. The current principal references are under:

- `quantdata/dysanos/grid/gendysanos/reports/terminal-importance-arbitrage-tests-autonomous-alpha5-1000000/`;
- `quantdata/dysanos/grid/gendysanos/reports/terminal-importance-arbitrage-tests-autonomous-alphaNone-1000000/`; and
- `quantdata/dysanos/tex/generated/surface_first_diagnostics.json`.

Do not compare with an existing SANOS report merely because its sample count matches. Verify factor count, surface-first flag, path mode, return horizon, strikes, expiry offsets, importance proposal, state definition, neighbour count, costs, roundoff, and seed. Generate a new matched SANOS control if any material field differs.

## Material-improvement report

Write one aggregate JSON report and a concise Markdown summary. For each matched run place PCA and SANOS columns side by side. The primary comparison metrics are:

1. fraction of generated surfaces and paths with cross-sectional static violations;
2. maximum and upper-quantile normalized static-arbitrage value;
3. number of unconditional one-sided instruments;
4. minimum physical minority-sign probability;
5. reference and audit one-sided local candidate counts;
6. adverse-outcome counts for inferred local policies;
7. local effective sample sizes and neighbour radii;
8. importance-sampling ESS and moment-control errors; and
9. surface and weekly-path fidelity diagnostics, to exclude an apparently safe model that merely collapses its variance.

Use path-clustered uncertainty or a bootstrap over independent paths for differences in probabilistic metrics. Report absolute differences, ratios where the denominator is stable, and confidence intervals. Do not use $\sqrt{S\,TAU}$ as if path-time observations were independent.

Interpretation:

- SANOS materially improves **static arbitrage** if the PCA model has violations materially beyond arithmetic noise while the matched SANOS decoder does not.
- SANOS materially improves the sampled one-instrument diagnostics if PCA has reproducible one-sided candidates or materially smaller minority-tail probabilities in both reference and audit, with comparable support and effective sample size.
- If both models pass the return diagnostics, conclude only that no material difference was detected for those tests. Do not convert failure to reject into a proof of dynamic no-arbitrage.
- Separate benefits of the decoder from benefits of the state driver. Both models use the same surface-first AR(1) construction; the intended difference is the PCA versus SANOS surface map.

## Deliverables

At minimum provide:

- a versioned PCA fit/decoder module;
- a surface-first exact-Gaussian simulator with conditional spot;
- a coherent fixed-contract universe builder supporting all four modes;
- streaming static and return-based diagnostic runners;
- a command-line orchestrator for the required matrix;
- one JSON report per run plus an aggregate PCA-versus-SANOS report;
- plots for PCA reconstruction, first five eigensurfaces, invariant/weekly distances, static violations, minority-tail comparisons, and local candidate counts; and
- focused tests under `tests/`.

Suggested report root:

```text
quantdata/dysanos/implied/reports/pca-vs-sanos/
    smoke/
    alpha5/100000/
    alpha5/1000000/
    alphaNone/100000/
    alphaNone/1000000/
    pca-vs-sanos.json
    pca-vs-sanos.md
```

## Tests and stage gates

Add focused tests for:

1. PCA centring, orthonormality, deterministic signs, score geometry, and reconstruction;
2. positivity and finiteness of the log-IV decoder without clipping;
3. recovery of known synthetic PCA factors;
4. surface-first autonomy: simulated surfaces must not depend on current spot;
5. exact-transition means and covariances versus analytic or high-accuracy controls;
6. invariant-distribution moments;
7. fixed cash strike and calendar expiry across repricing dates;
8. one common spot path across every strike and expiry;
9. full-smile delta versus finite differences, including a flat-vol analytic control;
10. a known arbitrage-free Black--Scholes surface passing the static checker;
11. deliberately perturbed calendar, vertical, and butterfly surfaces failing the correct constraint;
12. streaming diagnostics agreeing with an in-memory small-array calculation;
13. importance likelihood normalization and direct-$p$ moment recovery;
14. 80:20 reference/audit path separation and leave-one-path-out neighbours; and
15. 1,000-path CPU generation plus the available CUDA diagnostic path.

Run, in order:

```bash
python -m py_compile quantdata/dysanos/implied/*.py
pytest -q tests/test_dysanos_implied_*.py
ruff check --select I quantdata/dysanos/implied tests/test_dysanos_implied_*.py
```

Then run the 1,000-path smoke matrix, followed by 100,000 paths. Inspect all non-finite, static-violation, tail-coverage, weight, ESS, and memory fields before proceeding to one million paths.
