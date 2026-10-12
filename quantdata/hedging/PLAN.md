# Hedging Benchmark Implementation Plan

## Status

This document is the implementation plan for the hedging benchmark built around
the daily front-spot SDLV/GROMM model. It records the agreed experimental design
and implementation order. No model implementation is implied by this document.

The benchmark should ultimately provide a public, model-neutral methodology for
comparing hedging systems. The first objective is the paper and a reproducible
internal benchmark. A public extension API follows after the paper methodology
and results are stable.

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
  The public candidate retains the union `[-2.5, 2]`, tags the paper's exact
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
- Compute the hedge of the claim as the claim policy minus the zero-claim
  policy. Under `Q`, the zero-claim policy should be zero in population and is
  retained as a diagnostic and numerical control.
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
unit. The proposed reference is inception ATM spot risk

```text
sigma_ref(t, H) = sigma_atm(t) * sqrt(H / 252),
```

but this convention must be confirmed before production evaluation.

## Target Notional and Position Caps

Target notional materially affects the relevance of option half-spreads. Run a
preliminary sizing experiment using 2022 only:

1. Normalize each target by its 2022 unhedged standard deviation.
2. Use the independent robust covariance hedge with mean-volatility `kappa=1`.
3. Measure transaction cost relative to the target's unhedged risk and gross
   option exposure.
4. Select a common risk size that creates meaningful but non-dominating costs.
5. Freeze target notionals for all tests after the 2022 sizing exercise.

An internal-data extension may additionally report display-size utilization,
but it cannot determine the primary public benchmark notional because display
size is deliberately absent from the distributable dataset.

Run two cap regimes:

- A loose numerical cap, with every cap hit treated as a diagnostic failure or
  warning.
- An internal-only display-size cap sensitivity when source size data are
  available.

The spot cap convention remains open because spot display size is unavailable.

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

### Phase 0: specification and schemas

Implement the versioned benchmark configuration, common artifact schema,
contract identity, and episode timing definitions.

Gate: protocol quant review passes.

### Phase 1: accounting vertical slice

Implement market snapshots, one ATM digital, spot plus one option, SANOS-mark
plus/minus-half-spread execution, settlement, and pricing/hedging decomposition.

Run a deterministic CPU test over a few dates.

Gate: accounting quant review and accounting unit tests pass.

### Phase 2: SANOS-LV daily baseline

Implement rolling market marks, delta, eight PCA-factor vegas, all daily target
claims, and the common reporting record.

Gate: standalone SANOS-LV smoke test passes.

### Phase 3: independent covariance baseline

Implement same-contract historical returns, robust covariance, cost lasso,
mean-variance, and mean-volatility.

Gate: standalone covariance smoke test and model review pass.

### Phase 4: rolling SDLV calibration

Implement the rolling 252-day innovation fit, 16-factor PCA alignment, eight
Greek factors, compatible `P` drift, and `Q` drift.

Gate: CPU deterministic calibration test and factor-alignment tests pass.

### Phase 5: GROMM daily benchmark

Implement GROMM scenario generation and the four hedge objectives under `Q`,
then under `P`. Implement the four 2022 architecture candidates and freeze the
selection rule.

Gate: GROMM CPU smoke test, model review, and GPU-readiness review pass.

### Phase 6: GPU production infrastructure

Implement restartable rolling fit, path-generation, and deep-hedging jobs with
the required resource checks, checkpointing, and progress logs.

Gate: CPU/GPU fixture agreement and forced-restart recovery test pass.

### Phase 7: full daily evaluation

Run all daily targets and native baselines from 2023 through the dataset's last
eligible origin. Resolve target notional and cap conventions using 2022 before
starting this stage.

Gate: interim results quant review passes.

### Phase 8: weekly engines

Implement frozen-weekly and restarted-weekly episodes, fixed synthetic
expiry-date/cash-strike books, daily rebalancing, variance, and barrier claims.

Gate: timing, expiry, and overlapping-episode tests pass.

### Phase 9: historical and published baselines

Add historical SANOS, VolGAN, and diffusion baselines. Reproduce public settings
before rolling retraining.

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

1. Confirm the episode risk unit based on inception ATM spot volatility.
2. Confirm the proposed target-notional utilization rule.
3. Define the large numerical position cap.
4. Define the spot-cap convention; display-size caps remain internal-only.
5. Fix the four deep-hedging architecture candidates.
6. Fix the precise eigenvalue tolerance for rolling PCA subspace alignment.
7. Confirm the barrier reference level convention as normalized inception
   spot/forward.
