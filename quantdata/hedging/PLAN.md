# Hedging Benchmark Implementation Plan

## Status

This document is the implementation plan for the hedging benchmark built around
the daily front-spot SDLV/GROMM model. It records the agreed experimental design
and implementation order.

The distributable public-candidate data layer and CPU accounting vertical slice
are implemented and passed independent quant review. The SANOS-LV Phase-2
implementation has rolling 252-transition fixed-contract DLV factors, eight
factor vegas, the complete 20-target daily digital family, stable next-date
synthetic contract replay, minimum-predictable-option-cost factor hedges, and a
residual spot-delta hedge. Its independent Phase-2 review and 160-target
multi-date CPU pilot passed. The independent covariance baseline now uses an
instrument-only shrinkage Tyler geometry, target-specific robust cross-moments,
sample-mean/HAC drift screening, and exact mean-volatility recession checks;
its Phase-3 implementation and independent re-review passed. The rolling
rank-16 SDLV calibration, factor alignment, eight Greek factors, analytical Q
drift, and compatible P-drift projection passed the Phase-4 remediation review.
The Phase-8 weekly accounting engine now also passes its independent mechanical
review. Common SANOS/DLV weekly claim prices, exact rolling fit identities,
precommitted as-of information windows, and Hansen--Hodrick plus moving-block
overlap inference passed their independent re-review. A resumable daily runner
now pins dataset and model identities, enforces the complete 20-target book and
numerical gates, and writes one atomic record per origin. SANOS-LV and
historical-SANOS each completed all 611 eligible origins and 12,220 target
records; robust-covariance production is in progress. Other daily production
and full weekly production remain outstanding. A unified weekly runner now
supports frozen and restarted schedules for SANOS-LV, historical SANOS,
GROMM, VolGAN, and both diffusion variants. It requires an authenticated
strictly-2022 sizing preflight, validates every public dataset member, holds a
single-writer lock, records exact fitted-artifact fingerprints, and rejects
incomplete five-decision records. The
strictly-2022 target-sizing exercise now fixes each target to one
maturity-matched ATM spot-risk unit; all 20 calibration hedges converged with
no cap hit. Its atomic derived artifact is keyed by the public dataset manifest
and complete sizing configuration, preventing repeated production runs from
silently using different quantities. The same runner now supports VolGAN and
AD-Seq-Vol with full
training/scenario configuration identities. Abbreviated one-origin CPU/GPU
integration runs passed all 20 target records with no cap hit; they are
orchestration checks, not benchmark estimates. The completed-run results layer
now validates immutable run/record identities and reports unhedged, economic,
pure-hedging, and pricing losses overall, by year, by 2022-fixed target-maturity
ATM-volatility regime, and by year/regime intersection. It also recomputes
eligibility from the immutable run specification, revalidates every record,
and reports normalized entry turnover/exposure, cap rates, and explicit HAC
metadata; independent quant review passed. The first full SANOS-LV run exposed
a small-coefficient linear-program equality-scaling failure. Row scaling fixes
the numerical cause without relaxing the precommitted residual tolerance; the
clean versioned 2023--2025 CPU rerun is complete.

The benchmark should ultimately provide a public, model-neutral methodology for
comparing hedging systems. The first objective is the paper and a reproducible
internal benchmark. A public extension API follows after the paper methodology
and results are stable.

### Runtime data boundary

- The distributable `hedging-public-candidate-v1` dataset is the sole market
  data input to every model fit, training job, scenario generator, valuation,
  and benchmark run.
- Runtime model and training code must not import the private dataset exporter,
  WRDS/WIDB loaders, DySANOS research pipelines, or legacy raw SDLV arrays.
- Stored SANOS states are the reference market interpolation whenever a price,
  target mark, or fixed contract is required away from a released instrument
  coordinate. Stored DLV states remain model inputs and are not substituted for
  the common SANOS market mark.
- The private-source exporter remains an offline construction tool only. It is
  outside the runtime dependency graph and is not invoked by training.

## Objectives

The benchmark must:

1. Evaluate hedging performance strictly out of sample from 2023 through the
   final eligible retained date.
2. Compare daily, frozen-weekly, and restarted-weekly hedging protocols.
3. Separate pricing differences from pure hedging performance.
4. Use identical market data, contracts, costs, claims, accounting, and metrics
   across all models.
5. Make every model easy to locate, understand, modify, and test independently.
6. Provide deterministic CPU smoke tests before any material GPU work.
7. Subject the protocol, accounting, models, and final results to independent
   quant-agent review.

## Agreed Experimental Design

### Estimation and evaluation dates

- Use 2022 for model initialization and architecture validation.
- Use a rolling 252-business-day estimation window, rolled daily.
- The current public-candidate panel ends on 2025-08-29. Evaluate strictly out
  of sample from 2023 through the final eligible origin recorded in its
  manifest. Extend through year-end 2025 only after a later dataset version
  actually contains those observations.
- Do not use future market observations for fitting, normalization, architecture
  selection, early stopping, checkpoint selection, or target sizing.
- Start a weekly episode on every eligible business day. Weekly episodes may
  overlap; inference must account for that dependence.

### Hedging schedules

#### Daily

- Fit using information available through date `t`.
- Hedge from `t` to `t+1`.
- A one-day claim settles at `t+1`.
- A longer claim is marked at `t+1` using the independently fitted market
  SANOS-LV density on that date.

#### Frozen weekly

- Fit the model and policy at episode origin `t0`.
- Rebalance daily for five business days.
- Keep model parameters and policy weights fixed throughout the episode.
- The policy may react to the observed state each day.
- Freeze synthetic inception contracts identified by expiry date and cash
  strike at `t0`.
- Allow those synthetic contracts to expire, but do not introduce contracts
  whose expiry date and strike were absent from the inception universe.
- On later dates, recover each frozen contract's current `K/F` from the common
  normalized spot/forward path and interpolate its half-spread cross-sectionally.

#### Restarted weekly

- Use the same five-business-day objective and episode origins as frozen weekly.
- Refit the model and recompute the policy each day using information available
  at that date.

### Normalization and market accounting

- Use `K/F`, model prices `C/(DF*F)`, half-spreads `h/S`, per-expiry `F/S` and
  `DF`, and forward-adjusted spot returns. This removes the deterministic carry
  represented by the previous-date forward curve without discarding the cash
  discount factor needed for accounting.
- Maintain one normalization and numeraire throughout each episode.
- Maintain three explicit spot paths without mixing their roles. All weekly
  targets settle on the inception-forward-normalized path
  `X[n]=S(t_n)/F(t_0,t_n)`. ATM target strike and barriers are expressed
  relative to `X[0]=1`; non-ATM digital strikes are the inception NMS-derived
  `K/F`. The actual cumulative `S[n]/S[0]` path is used only to recover fixed
  hedge-contract `K/F`, marks, settlements, and traded spot notional. The
  compounded rolling one-session forward-adjusted path is retained only as a
  carry-basis diagnostic: settling on it would make the claim depend on future
  carry curves unknown at inception. All accounting paths are converted to the
  same discounted episode-origin PV unit.
- Charge one basis point per transaction on traded normalized spot notional.
- Use the SANOS fitted model price as the common market mark. Execute a buy at
  the mark plus the published `h/S` and a sale at the mark minus `h/S`, after
  converting both to the same spot-one unit.
- Charge no transaction cost when a cash-settled index option expires.
- Use the SANOS-LV interpolated model mark as the common claim price. It is not
  represented as, or claimed to be, an observed midpoint.
- Report each model's buyer and seller indifference prices separately.
- Report risk-neutral expectations only for models that genuinely provide a
  risk-neutral measure.

For a short claim with market price `p_market`, model seller price
`p_model_sell`, hedge gain `G`, transaction costs `C`, and payoff `Y`, use

```text
economic_pnl
    = p_market + G - C - Y
    = (p_market - p_model_sell)
      + (p_model_sell + G - C - Y).
```

The first term is the pricing contribution. The second term is the pure model
hedge error. Use the corresponding buyer indifference price and long-claim
hedge for a long claim.

### Target claims

ATM always means normalized strike `K=1`, equivalently NMS zero. Only non-ATM
strikes are constructed from NMS, using the inception expiry-specific ATM
volatility. All target strikes are fixed at inception.

#### Daily targets

For both one-day and three-day maturities:

- ATM digital call at `K=1`.
- ATM digital put at `K=1`.
- OTM put digitals at NMS `-1`, `-0.75`, `-0.5`, and `-0.25`.
- OTM call digitals at NMS `0.25`, `0.5`, `0.75`, and `1`.

Each digital pays one unit of normalized cash.

#### Weekly targets

- The same digital family at one-week maturity.
- Five-day realized variance:

```text
(1/5) * sum_{n=1}^5 ((S[n] - S[n-1]) / S[n-1])^2.
```

- ATM down-and-out put with an end-of-day barrier at 95% of inception normalized
  spot/forward.
- ATM up-and-out call with an end-of-day barrier at 105% of inception normalized
  spot/forward.
- Observe barriers only at business-day closes.
- Do not smooth target payoffs.

### Hedge instruments

- Include normalized spot.
- Include eligible option expiries through five business days.
- If no exact five-day expiry exists, also include the first expiry after five
  business days.
- Include options strictly within `(-2, 2)` NMS, computed using the current
  expiry-specific ATM volatility.
- Treat `(-2, 2)` as a dynamic hedge-eligibility rule, not as a storage filter.
  The public candidate retains the superset `[-2.5, 2]`, tags the paper's exact
  `[-2.5, 1.5]` calibration population, and separately tags the contemporaneous
  `(-2, 2)` hedge screen.
- Exclude AM-settled contracts from the primary hedge universe. The public
  close-to-close path does not contain the special opening settlement level
  needed to settle them exactly; the retained flag supports this exclusion.
- Exclude exterior DLV boundary strikes.
- Use one liquid OTM instrument per strike: puts below ATM and calls above ATM.
- Avoid redundant put-call-parity copies.
- For weekly experiments, determine eligibility at `t0` and freeze the
  synthetic expiry-date/cash-strike universe for the episode. Exact listed
  contract replay is an internal-data extension, not a public benchmark
  capability.

## Package Architecture

The repository style guide prohibits one research project from depending on
another. The new SDLV study must therefore not import implementation directly
from `quantdata.dysanos.flex`. Genuinely reusable functionality should be moved
behind a neutral package while compatibility wrappers preserve existing
DySANOS experiments.

Planned layout:

```text
quantdata/hedging/
    contracts.py
    accounting.py
    objectives.py
    metrics.py
    inference.py
    protocols.py

quantdata/sdlv/daily/front_spot/hedging/
    PLAN.md
    README.md
    benchmark/
        specification.py
        episodes.py
        accounting.py
        metrics.py
        inference.py
    data/
        market_panel.py
        contracts.py
        rolling_windows.py
        fixtures.py
    claims/
        digitals.py
        variance.py
        barriers.py
    models/
        sanos_lv/
        gromm/
        covariance/
        historical_sanos/
        volgan/
        diffusion/
    runners/
        run_daily.py
        run_frozen_weekly.py
        run_restarted_weekly.py
    reviews/
    reports/

tests/quantdata/hedging/
tests/quantdata/sdlv/daily/hedging/
```

Each model directory must contain:

```text
README.md
config.py
adapter.py
hedger.py
smoke.py
```

The model README must state:

- Input data and fitted quantities.
- Whether the model represents `P`, `Q`, or neither.
- Its price interpretation.
- Its scenario-generation mechanism.
- Its hedge objective and supported risk aversions.
- Its trainable components and validation rule.
- Its expected artifacts and standalone test command.

## Common Interfaces

The benchmark should depend on small, explicit interfaces rather than model
internals.

### Payoff

- Define the claim at episode inception.
- Evaluate settlement from a normalized path.
- Mark an unfinished claim using a supplied market snapshot.

### Market model

- Fit from an explicitly dated training window.
- Produce state variables available at a decision time.
- Simulate under a named measure when supported.
- Price a claim when supported.
- Serialize sufficient state for exact restart and audit.

### Hedger

- Fit only from the supplied training or generated scenarios.
- Return a predictable action from the current state and existing positions.
- Report objective value, convergence, caps, and numerical diagnostics.

### Backtest

- Construct the episode universe.
- Apply actions using the common execution and cost model.
- Settle expiring instruments.
- Mark the target and remaining positions.
- Emit a model-independent episode record.

## Model Implementations

### SANOS-LV reference

- Fit the market SANOS surface.
- Price in local volatility.
- Compute spot delta.
- Compute DLV vegas with the first eight contemporaneous rolling PCA factor
  bumps.
- Use central factor differences in log DLV-squared coordinates.

### GROMM

- Estimate 16 rolling DLV innovation factors.
- Align rolling factor bases using permutation and sign matching, with
  Procrustes rotation inside near-degenerate eigenspaces.
- Use all 16 factors in the stochastic SDLV model.
- Use the first eight factors for comparable DLV-vega hedges.
- Use eight Brownian substeps per business day.
- Preserve the independent 5% Black background under both `P` and `Q`.
- Estimate the compatible physical drift and risk-neutral drift separately.
- Support mean-variance, mean-volatility, entropy, and power-exponential
  deep-hedging objectives under both `P` and `Q`.
- Fit claim, short-claim, and zero-claim policies separately for indifference
  pricing. Retain the claim-minus-zero action as a diagnostic. Because absolute
  turnover is nonlinear, train the executable pure hedge directly in net-action
  coordinates on conditional Q-shadow gains. At every simulated state, reuse
  the same Brownian shock under the unshifted Q transition and measure gains
  from the model mark. This preserves local hedge shocks while removing both
  state-dependent P drift and common-mark/model-mark basis positions, without
  estimating a conditional mean from validation paths. Under `Q`, the
  zero-claim policy should be zero in population.
- Also provide conventional risk-neutral expectation plus eight-factor DLV-vega
  hedging.

The path-dependent policy should use an LSTM followed by the established
affine-plus-deep action architecture. Test four architecture candidates on 2022
generated training and validation sets. Fix the architecture thereafter and
retrain weights rolling. Choose the cheapest candidate whose validation loss is
within one standard error of the best candidate.

### Independent covariance baseline

- Use raw historical normalized option and spot returns.
- Do not use SANOS-generated scenarios.
- Estimate a rolling robust high-dimensional covariance matrix.
- Prefer a shrinkage Tyler estimator that does not require a parameter fixed
  permanently from 2022.
- Apply half-spread-weighted cost lasso derived from the common execution rule.
- Support mean-variance and mean-volatility directly.
- Under a Gaussian assumption, support entropy through its mean-variance
  equivalence and power-exponential objectives where numerically appropriate.

### Historical statistical SANOS baseline

- Use historically fitted SANOS surfaces to price the same contracts and target
  claims.
- Construct historical normalized return scenarios.
- Apply the same transaction-cost and risk-objective implementation as the
  other methods.

### Published generative baselines

Implement the following only after the native benchmark passes:

- VolGAN and the cost-lasso hedging protocol from the public implementation.
- AD-Seq-Vol and arbitrage-aware AD-Seq-Vol-FT from the published diffusion
  implementation.

For each external baseline:

1. Reproduce the public settings first.
2. Record any deviations required for the local data schema.
3. Retrain under the common rolling benchmark protocol.
4. Report both public-setting and rolling-trained results.

## Risk Objectives

Support:

1. Mean-variance.
2. Mean-volatility.
3. Entropy OCE.
4. Power-exponential OCE.

For mean-volatility use:

```text
rho_kappa(L) = E[L] + kappa * SD(L)
```

with `kappa` in `{0.1, 1, 2, 3}`.

For Gaussian loss with standard deviation `sigma`, entropy and mean-variance
have risk-premium equivalence

```text
lambda = 2 * kappa / sigma.
```

Risk parameters and losses should be expressed in a documented episode risk
unit. The confirmed reference is inception ATM spot risk

```text
sigma_ref(t, H) = sigma_atm(t) * sqrt(H / 252).
```

## Target Notional and Position Caps

Target notional materially affects the relevance of option half-spreads. The
preliminary sizing experiment uses 2022 only:

1. Normalize each target by its 2022 unhedged standard deviation.
2. Use the independent robust covariance hedge with mean-volatility `kappa=1`.
3. Measure transaction cost relative to the target's unhedged risk and gross
   option exposure.
4. Set each target quantity so its scaled unhedged standard deviation equals
   one maturity-matched ATM spot-risk unit.
5. Freeze those target quantities for all out-of-sample tests.

The 242-transition initialization run through 2022-12-30 converged for all 20
targets with no cap hit. Median, 90th-percentile, and maximum transaction cost
were respectively 1.69%, 3.78%, and 7.54% of the reference risk. A common
rescaling cannot select a more attractive size here: under proportional costs
and the homogeneous mean-volatility objective the cost-to-risk fraction is
scale invariant except at a zero-action or cap boundary. The dated calibration
report records the target quantities and diagnostics.

An internal-data extension may additionally report display-size utilization,
but it cannot determine the primary public benchmark notional because display
size is deliberately absent from the distributable dataset.

Use the following cap regimes:

- A loose numerical cap of 1,000 normalized action units, applied to spot and
  option coordinates, with every cap hit treated as a failed numerical gate.
- An internal-only display-size cap sensitivity when source size data are
  available.

## CPU and GPU Work Split

Every implementation must have a deterministic CPU execution path suitable for
smoke testing. GPU work starts only after the CPU invariants and standalone
model test pass.

| Work | CPU responsibility | GPU responsibility |
| --- | --- | --- |
| Market loading and contract matching | Primary | None |
| Rolling-window construction | Primary | None |
| Claim, settlement, and accounting tests | Primary | None |
| Rolling PCA and factor alignment | Primary | Optional acceleration |
| Robust covariance estimation | Primary | None initially |
| SANOS-LV pricing and Greek smoke tests | Primary | Optional batching |
| SDLV/GROMM simulation | Small deterministic pilots | Production paths |
| Rolling DLV fits | Smoke and fallback | Production fitting |
| Deep-hedging training | Small architecture tests | Production training |
| VolGAN and diffusion training | Interface smoke tests | Production training |
| Metrics, inference, and reports | Primary | None |

CPU tasks should materialize immutable, versioned inputs before GPU launch.
GPU jobs should perform only the fitting, path generation, or policy training
that materially benefits from GPU execution.

Every material GPU job must:

- Run through `gpuctl`.
- Check `gpuctl resources --json` before launch.
- Use the explicit project interpreter, working directory, `QUANTDATA_ROOT`, and
  `PYTHONPATH` in detached execution.
- Validate imports and data-root discovery in the detached environment before
  starting material work.
- Checkpoint all model, optimizer, scheduler, scaler, and RNG state required for
  a near-exact restart.
- Emit flushed progress at least every five minutes.
- Report elapsed time, completed and total work units, and ETA when meaningful.
- Avoid retaining unnecessary large intermediate arrays.

## Standalone Model Tests

Every model must have an independently runnable CPU smoke test:

```bash
python -m quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.gromm.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.covariance.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.volgan.smoke
python -m quantdata.sdlv.daily.front_spot.hedging.models.diffusion.smoke
```

Each smoke test must:

- Use a fixed small date range and deterministic seed.
- Run on CPU without hidden GPU initialization.
- Hedge one ATM one-day digital.
- Complete quickly enough for routine developer use.
- Produce the common compact JSON result schema.
- Report unhedged P&L, hedged P&L, costs, actions, pricing difference, pure
  hedge error, convergence, and cap diagnostics.
- Compare key outputs with a committed tolerance fixture.

Keep one pytest module per model:

```text
tests/quantdata/sdlv/daily/hedging/models/test_sanos_lv.py
tests/quantdata/sdlv/daily/hedging/models/test_gromm.py
tests/quantdata/sdlv/daily/hedging/models/test_covariance.py
tests/quantdata/sdlv/daily/hedging/models/test_historical_sanos.py
tests/quantdata/sdlv/daily/hedging/models/test_volgan.py
tests/quantdata/sdlv/daily/hedging/models/test_diffusion.py
```

GPU integration tests must be separately marked and must not be required for a
normal CPU test run.

## Quant-Agent Review Gates

Use a separate quant-review agent at each material milestone. Store reviews in
`quantdata/sdlv/daily/front_spot/hedging/reviews/` with the reviewed commit,
configuration identity, evidence, findings, and disposition.

### Protocol review

Review:

- Information timing and rolling windows.
- Contract identity and weekly universe freezing.
- Normalization and numeraire consistency.
- Claim definitions and settlement.
- Look-ahead protections.

### Accounting review

Review:

- Buyer and seller signs.
- Initial cash and common market price.
- Indifference-price decomposition.
- Bid/ask execution and spot costs.
- Expiry settlement and remaining-position marks.

### Model review

Perform a separate review for every model directory before its production run.
Review:

- Measure interpretation.
- Calibration inputs and train/validation separation.
- Price semantics.
- Simulation and hedge objective.
- Risk scaling and transaction-cost implementation.
- Standalone smoke-test evidence.

### GPU-readiness review

Review:

- CPU/GPU numerical agreement on a fixed fixture.
- Reproducibility and RNG handling.
- Checkpoint recovery.
- Batch-size invariance within tolerance.
- Memory scaling and progress logging.

### Results review

Review:

- Statistical inference and effective independent units.
- HAC or block-bootstrap treatment of overlapping weekly episodes.
- Pairwise comparison construction.
- Cap and transaction-cost effects.
- Data leakage and model-selection leakage.
- Reproducibility of all headline tables.

Unresolved correctness, arbitrage, look-ahead, measure, or accounting findings
block the next production stage. Performance and presentation comments may
remain open if documented.

## Statistical Reporting

For every model, payoff, maturity, year, and volatility regime, report:

- Mean and confidence interval.
- Median.
- Sample standard deviation.
- RMSE and MAE.
- 95% and 99% loss quantiles.
- Expected shortfall.
- Mean-variance, mean-volatility, entropy, and power-exponential scores.
- Pricing contribution, pure hedge error, and total economic P&L.
- Turnover and transaction costs.
- Sparsity, gross exposure, and net exposure.
- Numerical-cap hit rates and, only for internal-data extensions, display-size
  cap hit rates.

Use complete episodes as sampling units. Use paired episode-level differences
for model comparisons. Use HAC or block-bootstrap inference for overlapping
weekly episodes.

## Implementation Phases and Gates

### Phase 0: specification and schemas (complete)

Implement the versioned benchmark configuration, common artifact schema,
contract identity, and episode timing definitions.

Gate: protocol quant review passes.

### Phase 1: accounting vertical slice (complete)

Implement market snapshots, one ATM digital, spot plus one option, SANOS-mark
plus/minus-half-spread execution, settlement, and pricing/hedging decomposition.

Run a deterministic CPU test over a few dates.

Gate: accounting quant review and accounting unit tests pass.

### Phase 2: SANOS-LV daily baseline (complete)

Implement rolling market marks, delta, eight PCA-factor vegas, all daily target
claims, and the common reporting record.

Gate: standalone SANOS-LV smoke test passes.

### Phase 3: independent covariance baseline (complete)

Implement same-contract historical returns, robust covariance, cost lasso,
mean-variance, and mean-volatility.

Gate: standalone covariance smoke test and model review pass.

### Phase 4: rolling SDLV calibration (complete)

Implement the rolling 252-day innovation fit, 16-factor PCA alignment, eight
Greek factors, compatible `P` drift, and `Q` drift.

Gate: CPU deterministic calibration test and factor-alignment tests pass.

### Phase 5: GROMM daily benchmark

Implement GROMM scenario generation and the four hedge objectives under `Q`,
then under `P`. Use the currently specified `quantdata/gromm2` market model:
the 200-strike JAX implementation with AR(1) DLV dynamics. Its greater-than-80%
market fit is accepted for this benchmark iteration. Treat `quantdata/gromm2`
as a read-only architecture reference and do not modify it from the hedging
project.

The market-model architecture is therefore fixed rather than selected inside
the hedging benchmark. Dated rolling P/Q artifacts and exact benchmark-book
output are separate from the four affine/deep/LSTM hedge-policy candidates;
only the hedge-policy architecture is selected on the 2022 initialization set.

The dated market-artifact contract, CPU preparation stage, daily P/Q
benchmark-book output, and weekly recurrent-policy interface are implemented.
A production artifact contains exactly 252 transitions in the declared origin's
information set, fixes the 200-strike geometry from that origin, hashes all
inputs, and rejects full-panel or future-dated histories. When a released
origin follows a missing intermediate market date, the fit retains the latest
252 exact released one-session pairs and appends the current origin state for
the geometry and DLV level; it never substitutes a multi-session return for a
one-session transition. A real first-eligible 2023-01-17
fit completed from the public dataset using 252 exact released one-session
pairs supported by 257 states. It places 84.90% of observations inside spread,
and the rank-16 innovation basis explains 95.15% of variance. The compatible-P
estimator fails the unchanged projection and holdout-improvement gates, so this
artifact is Q-only. Its authenticated four-path Q replay produces finite
returns over 163 instruments; P generation is rejected. The loader bounds
float32 density-mass roundoff by stored source precision, normalizes accepted
residuals in float64, and logs the correction. The five-session sampler recomputes
the compatible P risk price pathwise and prices the frozen contract book
without retaining full density histories. Every artifact is bound to the exact
public manifest and the compatible P drift must pass projection and daily
risk-price-norm gates. Generated books start at the common SANOS executable
marks. Signed policies provide prices and diagnostics; the directly trained
pure policy uses conditional Q-shadow gains, exact net turnover, free cash
settlement, and terminal liquidation. Managed-GPU affine fitting and 2022
hedge-architecture selection are complete for the first rolling origin. Full
rolling Q production and any future-origin P acceptance remain outstanding. A
range orchestrator now authenticates and reuses each dated preparation or fit,
records every manifest hash atomically, prevents concurrent writers, and emits
a persistent heartbeat at most every five minutes. Its real first-origin
resume smoke passed. The 611-origin CPU preparation service is running and has
passed the first sparse-release origin after an authenticated restart.

Gate: GROMM CPU smoke test, model review, and GPU-readiness review pass.

### Phase 6: GPU production infrastructure

Implement restartable rolling fit, path-generation, and deep-hedging jobs with
the required resource checks, checkpointing, and progress logs.

The rolling GROMM2 range runner provides authenticated per-origin restart, an
atomic range state, a single-writer lock, and four-minute progress heartbeats.
A completed range summary binds the dataset, configuration, evaluation dates,
artifact root, and every dated manifest hash. The production matrix enables Q
cases only from this complete evidence; it enables P cases only when every
origin also passes the physical-drift gate. Full managed-GPU range fitting and
forced-restart recovery evidence remain outstanding.

Gate: CPU/GPU fixture agreement and forced-restart recovery test pass.

### Phase 7: full daily evaluation

Run all daily targets and native baselines from 2023 through the dataset's last
eligible origin. Resolve target notional and cap conventions using 2022 before
starting this stage.

The production results schema and all four agreed ex-post risk functionals are
implemented. A completed run is rejected if any eligible origin, target,
dataset identity, run identity, convergence gate, or cap count is missing or
inconsistent. The failed SANOS-LV `production-v1` is retained as a numerical
diagnostic; the row-scaled `production-v2` run completed all 611 eligible
origins and wrote 12,220 target records. Historical SANOS also completed all
611 origins and 12,220 records; covariance production is resumable and remains
in progress. The versioned daily
production matrix contains 53 unique cases: 21 runnable native CPU cases, two
staged published-setting GPU cases awaiting the generated-model GPU-readiness
gate, and 30 explicitly blocked rolling-generated or GROMM cases. No blocked
case is silently emitted as a launch command.

Gate: interim results quant review passes.

### Phase 8: weekly engines (mechanical gate complete)

Implement frozen-weekly and restarted-weekly episodes, fixed synthetic
expiry-date/cash-strike books, daily rebalancing, variance, and barrier claims.

Gate: timing, expiry, and overlapping-episode tests pass.

The common five-session engine, target catalogue, frozen cash-strike replay,
daily rebalancing, cash settlement, first-later-expiry liquidation, and
overlapping-origin enumeration pass focused tests and independent quant review.
The settlement convention has subsequently been hardened to use the inception
forward curve represented by the SANOS and DLV states; the rolling-forward
alternative remains a diagnostic. Frozen/restarted fit identities, exact
rolling-window information facades, final SANOS-LV common-price review, and
overlap-aware results inference now pass their focused gates. GROMM, SANOS-LV,
historical SANOS, VolGAN, and diffusion weekly adapters are implemented at
CPU-smoke level; production execution remains outstanding. The adapter supplies
the predictable realized path prefix and prior positions to a restarted fit,
so path-dependent prices cannot be computed from an information-incomplete
fit call; frozen fits still occur only once at episode inception.

The model-neutral resumable evaluator writes one atomic 13-target record per
origin. The fixed sizing handoff uses 221 complete 2022 episodes, a five-day
ATM spot-risk standard deviation of `0.029423402816172756`, and an authenticated
artifact SHA-256 of
`9494bbe23699126b5137b5ab3b6d53070002e2ba8ee60d92b0327ff747fcf9e1`.
A real one-origin frozen SANOS-LV public-data run and its no-recompute restart
check pass. The corresponding restarted one-origin integration run also passes;
its no-recompute restart check validates the exact five dated refits. These are
orchestration checks, not benchmark estimates.

### Phase 9: historical and published baselines

Add historical SANOS, VolGAN, and diffusion baselines. Reproduce public settings
before rolling retraining.

Historical SANOS now has a model-specific daily implementation; its
mean-volatility cost-lasso is solved as an exact CLARABEL SOCP rather than by an
unstable smooth line search. VolGAN and diffusion have exact within-fit restart
checkpoints and a fixed-source rolling-update path. One immutable checkpoint
trained strictly before evaluation initializes every rolling origin. Its file
hash and training end date are checkpointed, source dates must be strictly
earlier than current training dates, and current-window normalization is
analytically rebased without changing the source model's economic mapping.
Update epochs are selected from precommitted candidates on 2022-only validation
origins with the one-standard-error rule. The independent generated rolling
warm-start quant review passes with no correctness blockers. The production
runs and managed-GPU VRAM, timing, batch-invariance, and restart evidence remain
outstanding. The production matrix
pins one immutable initializer hash across every rolling origin, while each
fit verifies the warm-start and fine-tuning-base hashes before and after load.
Diffusion base checkpoints additionally carry a shared semantic-geometry
identity, preventing same-shaped but economically different transformer or
surface configurations from loading silently.

The first full-geometry managed-GPU readiness origin now passes. CPU preflight
pins and authenticates the target-notional artifact before allocation; the
2023-01-17 AD-Seq-Vol gate then produced all 20 target results from 4,096
scenarios in 158.3 seconds with zero cap hits or nonconvergence and 2,636 MiB
observed VRAM. An interrupted epoch checkpoint resumed successfully after the
restart loader was corrected to restore CPU RNG state on CPU. The detailed
evidence is recorded in
`reports/2026-10-12-generated-gpu-readiness.md`. Production-scale generated
runs and batch-invariance evidence remain outstanding.

Gate: one model-specific review and standalone smoke test per baseline pass.

### Phase 10: final benchmark and paper artifacts

Run the complete benchmark, generate versioned tables and figures, perform the
final results review, and reconcile every reported number with its source
artifact.

## Minimum Test Gates

Before any large experiment, verify:

- No look-ahead in windows, marks, normalization, or model selection.
- Stable contract identity across daily marks.
- Frozen weekly universes contain no newly listed contracts.
- Correct mark-plus/minus-half-spread side for every trade direction.
- Zero transaction cost at cash settlement.
- One-basis-point spot cost applied to traded notional only.
- Exact buyer/seller P&L sign symmetry where expected.
- Pricing plus hedging decomposition reconciles to economic P&L.
- Density mass and forward invariants.
- Finite positive DLV-squared values.
- Distinct and correctly labelled `P` and `Q` outputs.
- Train/validation/test separation.
- Deterministic fixture reproducibility.
- Checkpoint restart equivalence within numerical tolerance.
- CPU/GPU agreement on the same small fixture.

## Initial Deliverable

The first implementation deliverable is a CPU-only vertical slice containing:

1. The common contract and accounting APIs.
2. One ATM one-day digital claim.
3. One SANOS-LV market price and hedge.
4. One independent covariance hedge.
5. Common pricing-versus-hedging P&L decomposition.
6. Standalone smoke commands for both models.
7. Protocol and accounting quant-agent reviews.

This exercises the full benchmark path before introducing large simulations,
deep hedging, or GPU execution.

## Open Decisions Before Production Runs

The following do not block the CPU vertical slice but must be resolved before
the full out-of-sample evaluation:

1. Fix the four deep-hedging architecture candidates after reconciling the
   hedging design with `quantdata/gromm2`.

The episode risk unit, target-notional rule, 1,000-unit numerical position cap,
common spot-cap convention, statistical PCA alignment tolerance, and
inception-forward barrier reference are resolved in the implementation and
their dated review or calibration reports.
