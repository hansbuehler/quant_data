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
sample-mean/HAC drift screening, exact mean-volatility recession checks,
Gaussian entropy equivalence, and an exact Gaussian power-exponential OCE;
its Phase-3 implementation and independent re-reviews passed. The rolling
rank-16 SDLV calibration, factor alignment, eight Greek factors, analytical Q
drift, and compatible P-drift estimation passed the Phase-4 remediation review.
The Phase-8 weekly accounting engine now also passes its independent mechanical
review. Common SANOS/DLV weekly claim prices, exact rolling fit identities,
precommitted as-of information windows, and Hansen--Hodrick plus moving-block
overlap inference passed their independent re-review. A resumable daily runner
now pins dataset and model identities, enforces the complete 20-target book and
numerical gates, and writes one atomic record per origin. Earlier SANOS-LV and
historical-SANOS runs each completed all 611 eligible origins and 12,220 target
records. Those records used superseded target-notional identities and remain
diagnostic rather than being relabelled as current production evidence. The
clean current-notional SANOS-LV rerun is complete: all 611 origins and 12,220
records pass the current record-by-record validator, with zero cap hits and zero
nonconvergence, and deterministic reconstruction reproduces the pinned
production-results SHA-256. The three corrected robust-covariance
mean--variance cases are also complete and authenticated on the same current
target-notional cohort, with zero cap hits and zero nonconvergence. The corrected
Historical-SANOS mean--variance case at risk parameter $0.1$ is likewise complete
and authenticated for all 611 origins and 12,220 records, with zero cap hits and
zero nonconvergence. The remaining robust-covariance and Historical-SANOS sweeps
continue from immutable worktrees. The remaining Historical-SANOS launcher keeps
four CPU cases active with a work-conserving queue: when any case finishes, the
next precommitted objective starts immediately rather than waiting for the other
three cases in its original batch. Existing atomic origin records are reused,
while the model code, production matrix, case order, and four-worker limit remain
pinned. The legacy `0b8d20a` launcher is restricted to its two unfinished
mean--variance cases at risk parameters $1$ and $10$; the already authenticated
$0.1$ result remains unchanged, and that service no longer launches corrected
mean--volatility, entropy, or power-exponential cases owned by the newer matrix.
The persisted launcher SHA-256 values are
`86ef24fe3770ebd8f35e5dc3b9e85ad499803a5dabda68dc67439c671a8c9db8` and
`223d5132eaa9f480062b6b23cbdc5598b69af9f65215136d18effaeb39ce5eaf`,
respectively.
Other daily production and full weekly production remain outstanding. The six
Historical-SANOS entropy and power-exponential cases from the older `edf28cc`
cohort remain failed diagnostics: their exact-cost optimizer could reject a
valid solution after the split-variable solver stalled. The corrected
implementation at immutable commit
`cff579c38c489e315b624d055d4336e80554cd58` uses the original-action KKT
residual as the acceptance test and one canonical signed restart without
changing the objective, costs, bounds, or tolerance. Independent review passed,
and service `quantdata-hedging-historical-exact-cost-cff579c-v2.service` is
producing the six fresh cases concurrently. Its concurrency-only restart reused
97 atomic origin records; the launcher SHA-256 is
`425a95f3a7da39e890b637c9aae0938e42e1feb15ed65b6b213f8e029a87523d`.
The three exact-Gaussian covariance
power-exponential cases run independently under service
`quantdata-hedging-covariance-power-656ca65.service`.

A unified weekly runner supports frozen and restarted schedules for SANOS-LV,
historical SANOS, GROMM, VolGAN, and both diffusion variants. It requires an
authenticated strictly-2022 sizing preflight, validates every public dataset
member, holds a single-writer lock, records exact fitted-artifact fingerprints,
and rejects incomplete five-decision records. Generated weekly evaluation shares
one immutable target-independent path and fixed-book geometry per decision state
and one cold zero-claim pricing fit across targets. Historical weekly evaluation
prices the fixed option book once per decision state and attaches each target
payoff to the cached paths. Target-specific convex problems are rebuilt rather
than carried across claims, and the reusable pure-hedge geometry receives the
current claim payoff explicitly. Weekly entropy and power-exponential policies
use exact conic epigraphs for scenario-dependent turnover costs plus the declared
fitting ridge, which is excluded from reported certainty equivalents. Real
first-origin frozen-weekly gates cover all 13 targets: entropy at risk parameter
10 and power-exponential at risk parameter 1 both pass with zero cap hits and
zero nonconverged fits. Exact cached wealth matches the common pathwise
accounting engine to $10^{-14}$, and independent quant review passes.

The fresh 94-case native weekly matrix has 28 ready native cases, four generated
gate waiters, and 62 explicitly blocked cases. It was generated from immutable
commit `b37a643ea8e5ef11b7557e37c035cf650a729e95` and has SHA-256
`8d069cadde0c21dd677bbe7ab16cd59e0d3dae9907456608da7fcb4b9bce0b26`.
The two independent SANOS--LV weekly cases now run ahead of the daily recovery
queues. The 26 Historical-SANOS weekly cases use an atomic per-case pool whose
low-priority concurrency expands only after the protected GPU-readiness and
daily CPU services finish. Their persisted launcher SHA-256 values are
`361b4376fe8257cc0a608bca348bcccc2a1004f728b9b66e764989ed984ae676` and
`056574fcc73f22a69e93e1751b4aaeac84188575af0779d41f4a185a228e00e9`.
Service `quantdata-hedging-weekly-native-b37a643-v5.service` waits for those
two checkpointed pools and the active daily recovery queues, then authenticates
all 28 native cases through the original matrix prefixes. Its launcher SHA-256
is `992a61911968e5246f277b99fe9354865e423538b20bddb5a3dff18a68c0ea6c`.

The legacy rolling GROMM range and its current-code migration are complete and
authenticated for all 611 origins. The migration separately authenticated the
range-level and dated-manifest source identities, accepted 576 full physical
candidates, and used the declared front-only fallback at 35 origins. Its range
summary `gromm2-rolling-v4/range-fit-2023-2025-v7.json` has SHA-256
`8f7e6a63e90dfbfa367a06a220c75ebb4d0c9a0aa5f61ff2a92d8a58858b924e`.
Generated-model readiness now owns the managed GPU and runs from immutable commit
`b37a643ea8e5ef11b7557e37c035cf650a729e95`, which includes CPU-first
checkpoint restoration and permits finite signed buyer, seller, and continuation
indifference values while keeping common market and genuine-Q prices
non-negative. Independent quant review passed all 18 generated workflows and
all six daily/frozen/restarted public/rolling scope bindings for each model. The
review artifact SHA-256 values are
`0c8df8ce78645b278fb8587a90c9d03b4ed61ca56d4c60ab7848510e94506a87`,
`2333db2cd8894b2949bfddc98d41bdc4dfea9985ac0f421f6ba72e9a042f23b1`,
and `062e7268f6c25461e15da9ea61ef33fef138767630c55111736c1a5652332c41`
for VolGAN, AD-Seq-Vol, and AD-Seq-Vol-FT, respectively. Service
`quantdata-hedging-generated-readiness-b37a643.service` is executing the fresh
signed-price-aware 18-scope readiness matrix.

The corrected full-production service
`quantdata-hedging-full-production-b37a643-v5.service` runs from the same immutable
commit and waits for generated readiness, native weekly production,
Historical-SANOS exact-cost recovery, and covariance power-exponential
recovery. Its launcher SHA-256 is
`ba8971871b011d2ca1ff24674ffaa3a94c8c90227d8f902df050c7dd095a049a`.
The coordinators use service activity only for capacity scheduling; because
transient units may be collected immediately after exit, completion is accepted
only by reconstructing the required matrices, gates, and result artifacts.
The daily composition prioritizes the corrected exact-cost and covariance-power
matrices before older sources, so a failed legacy case cannot shadow its
authenticated successor. The stale `a1baa69` weekly and full coordinators and
their report-only successors were stopped before they launched production work.
Newly ready GPU cases run through `gpuctl`, with four-minute progress and
resource checks; the remaining daily GROMM Q-expectation plus eight-factor-vega
case starts only after native weekly production releases CPU capacity. The last
durable complete-suite record has 349 passing tests; later focused suites pass,
and a final complete-suite rerun remains required after production.
The daily production-queue validator
now authenticates the immutable run and completion summary, recomputes the true
eligible-origin cohort from the public panel, reparses every dated result record,
and rebuilds every aggregate statistic before requiring exact agreement with the
persisted report. The weekly production validator now applies the same rule: it
reconstructs the overlapping-origin cohort from the validated public panel,
reparses every dated weekly record, recomputes exposures, diagnostics,
inference, and risk slices, and requires exact persisted-aggregate equality.
Integrated tests reject both a forged eligible-origin summary and a tampered
record. The report-input builder now also accepts one or more immutable daily
production matrices, discovers every completed case by case ID and command
output, reconstructs each aggregate from its run and dated records, rejects
conflicting duplicate cases, and can require that every ready matrix case is
complete. Its real current-cohort smoke automatically authenticated the SANOS-LV
reference, three robust-covariance policies, and the completed Historical-SANOS
policy and generated all four objective tables plus target appendices. The
weekly report layer also reconstructs inception market, accounting,
risk-neutral, buyer-indifference, and seller-indifference unit prices directly
from authenticated dated records. It rejects any disagreement with realized
accounting and renders cross-origin target-level price means without filling
unavailable Q or indifference quantities. Focused report and authentication
suites pass; a final complete-suite rerun remains required after production. The
strictly-2022 target-sizing exercise now
fixes each target to one
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
    PLAN.md
    contracts.py
    accounting.py
    objectives.py
    metrics.py
    inference.py
    protocols.py

quantdata/sdlv/daily/front_spot/hedging/
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
  option coordinates. Cap hits and nonconvergence fail the numerical gate by
  default. A mean-volatility case whose unconstrained SOCP certifies a recession
  direction precommits the relaxed diagnostic gate instead: the capped solve is
  retained and explicitly flagged as an economically unbounded direction, not
  relabelled as a converged finite hedge.
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

The 891-date public-candidate artifact passes checksum, leakage-control, schema,
and full SANOS/DLV reconstruction validation from an otherwise empty data root;
durable evidence is in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-public-dataset-validation.md`.
Formal redistribution rights remain separate from this technical gate.

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
mean-variance, mean-volatility, Gaussian entropy equivalence, and the exact
Gaussian power-exponential OCE.

Gate: standalone covariance smoke test and model review pass.

The production implementation subsequently exposed a distinct numerical issue:
the 1,000-unit executable position cap was also bounding the claim and
zero-claim virtual policies before their subtraction. Those virtual policies
now use a separate 1,000,000-unit internal solver cap, while only the executable
claim-minus-zero hedge uses the 1,000-unit benchmark cap. Internal and
executable cap hits are reported separately. They fail the strict numerical
gate; a precommitted mean-volatility recession diagnostic may retain them only
as an explicitly nonconverged, cap-limited direction.
An independent exact-date replay and a public-data 20-target smoke passed with
no cap hit or nonconvergence. Results produced under the shared-cap implementation
are retained only as failed diagnostics; the clean production sweep is running
from a pinned source worktree.

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
candidate is accepted only when its linked holdout improvement is at least
0.5%, its Newey--West gain statistic is at least 1.96, and its daily risk-price
norm is at most 3. The density-weighted projection fraction is retained as a
descriptive fit statistic rather than used as an admissibility gate. If the
candidate is rejected, the precommitted physical null retains the full-window
front-factor risk price and sets the other 15 DLV-factor prices to zero. This
front-only fallback is accepted only when finite and within the same norm gate;
it is labelled explicitly and is not reported as an accepted full DLV
risk-price fit. Its authenticated four-path Q replay produces finite returns
over 163 instruments. The loader bounds
float32 density-mass roundoff by stored source precision, normalizes accepted
residuals in float64, and logs the correction. The five-session sampler recomputes
the compatible P risk price pathwise and prices the frozen contract book
without retaining full density histories. Every artifact is bound to the exact
public manifest. A compatible-P candidate must pass the precommitted holdout
improvement, Newey--West gain-statistic, and daily risk-price-norm gates; the
projection fraction remains descriptive. Generated books start at the common
SANOS executable marks. Signed policies provide prices and diagnostics; the
directly trained pure policy uses conditional Q-shadow gains, exact net
turnover, free cash settlement, and terminal liquidation. Managed-GPU affine fitting and 2022
hedge-architecture selection are complete for the first rolling origin. Full
rolling Q production and any future-origin P acceptance remain outstanding. A
range orchestrator now authenticates and reuses each dated preparation or fit,
records every manifest hash atomically, prevents concurrent writers, and emits
a persistent heartbeat at most every five minutes. Borderline affine fits keep
the fixed 80% inside-spread gate and continue from their authenticated 2,000
epoch checkpoint to at most 4,000 and 8,000 epochs; completed epochs, retry
count, and the actual fit configuration are retained in the dated artifact.
Its real first-origin resume smoke passed. CPU preparation, the legacy
managed-GPU fit, and the current-code migration are complete for all 611
eligible origins, including the first sparse-release origin after an
authenticated restart. The authenticated current range contains 576 accepted
full physical candidates and 35 declared front-only fallbacks. Generated-model
readiness now owns the managed GPU.

The range evidence reports full-candidate, front-only-fallback, and unavailable
physical-origin counts separately. A versioned migration path authenticates the
completed legacy range and every dated manifest, hard-links only its immutable
source, prepared-book, and accepted affine-history files into a distinct target
root, and recomputes all dynamics and physical-selection evidence under the
current code. Legacy physical decisions and legacy dynamics are never reused.
Independent review of these selection and authentication semantics passes in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-rolling-gromm2-physical-selection-review.md`;
it does not
stand in for the still-incomplete rolling-production evidence.

Gate: GROMM CPU smoke test, model review, and GPU-readiness review pass.

### Phase 6: GPU production infrastructure

Implement restartable rolling fit, path-generation, and deep-hedging jobs with
the required resource checks, checkpointing, and progress logs.

The rolling GROMM2 range runner provides authenticated per-origin restart, an
atomic range state, a single-writer lock, and four-minute progress heartbeats.
A completed range summary binds the dataset, configuration, evaluation dates,
artifact root, and every dated manifest hash. The production matrix enables Q
cases only from this complete evidence; it enables P cases only when every
origin has either an accepted full candidate or an accepted, explicitly labelled
front-only null fallback. The current-code migration is complete; live
managed-GPU forced-restart evidence for the downstream generated workflows
remains outstanding. A deterministic
mid-range failure test now verifies that the failed state identifies the exact
origin, preserves the authenticated completed prefix, and reuses that prefix on
restart before completing the missing origins.

The queue executor can retain a numerically rejected parameter case and
continue an otherwise independent sweep. Such a queue is explicitly marked
`complete-with-failures`; the failed case remains failed and cannot satisfy a
model-results gate.

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
origins and wrote 12,220 target records, but its superseded target-notional
identity prevents its use as current production evidence. The clean
current-notional SANOS-LV production run is now authenticated under run identity
`c67b549ca9eeb404be9e70c6fe2ba22c827d8d4ec75115e403a0bb2e54666a68`.
Its 611 origins and 12,220 records reconstruct deterministically to results
SHA-256 `9e4252f60540d6830eacda10b451677320760404e4599a5caa4e346367262b2d`,
with zero cap hits and zero nonconvergence. The summary report excludes the
superseded historical-SANOS cohort rather than comparing it with this run. The
corrected Historical-SANOS mean--variance case at risk parameter $0.1$ has
completed all 611 origins and 12,220 records under the current target-notional
identity, with zero cap hits and zero nonconvergence; its results SHA-256 is
`ad5850d4ba33c180cdac776623e49725260f46dd88ee6f7fd70b73b89319261e`.
The remaining Historical-SANOS and covariance sweeps remain in progress. The
legacy `edf28ccdb52ccac1788e20a51ce226220a5f74f3` queue now owns only the
precommitted low-risk mean--volatility recession diagnostic. Its failed entropy
and power-exponential cells remain failed and cannot satisfy a production gate.
Fresh exact-cost entropy and power-exponential cases run from independently
reviewed commit `cff579c38c489e315b624d055d4336e80554cd58`; the original-action
KKT residual, rather than the split-variable solver status, is authoritative.
The covariance mean--variance cases at risk
parameters $0.1$, $1$, and $10$ have completed all 611 origins and 12,220
records under the current target-notional identity, with zero cap hits and zero
nonconvergence. Gaussian entropy is represented exactly by these same
mean--variance policies at equal risk parameter. A distinct exact-Gaussian
power-exponential covariance fit is implemented and passes independent
quantitative review in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-covariance-gaussian-power-review.md`.
Its three risk-parameter cohorts are running from immutable commit `656ca65`
under `quantdata-hedging-covariance-power-656ca65.service`; the persisted launcher
SHA-256 is
`f359b2de7b710bd76652bb5c27fe0089b6740bab3fba9fecf8ca91b9cd46fdc2`.
The versioned daily production matrix contains 57
unique cases: 24 runnable native CPU cases, two
staged published-setting GPU cases awaiting the generated-model GPU-readiness
gate, and 31 explicitly blocked rolling-generated or GROMM cases. No blocked
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
orchestration checks, not benchmark estimates. The native SANOS-LV and
historical-SANOS weekly cases are materialized and deferred behind the daily
CPU production queues so they do not compete for the currently saturated host.
Before accepting a completed production cell, the queue fully validates the
released dataset, recomputes the true overlapping-origin cohort, reparses all
weekly records, rebuilds every reported statistic, and requires exact aggregate
equality; stored self-hashes alone are not treated as completion evidence.

### Phase 9: historical and published baselines

Add historical SANOS, VolGAN, and diffusion baselines. Reproduce public settings
before rolling retraining.

Historical SANOS now has a model-specific daily implementation; its
mean-volatility cost-lasso is solved as an exact CLARABEL SOCP rather than by an
unstable smooth line search. The solver first attempts the economically
unconstrained problem. If CLARABEL certifies a recession direction, it resolves
the same SOCP with the separate numerical solver cap solely to expose and record
the direction. Such a result is retained only when the immutable production
case precommits the relaxed mean-volatility diagnostic gate; the report exposes
its cap-hit and nonconvergence counts rather than presenting it as a finite
optimum. VolGAN and
diffusion have exact within-fit restart checkpoints and a fixed-source
rolling-update path. One immutable checkpoint
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
The 2022 budget study uses a separate same-geometry source checkpoint ending
before its first 126-observation validation window; the resulting budget is
then paired with the complete end-2022 production initializer. This separation
is authenticated explicitly and avoids the impossible requirement that an
end-2022 checkpoint predate its own 2022 validation origins.
Because the released 2022 panel contains 247 dates, this is explicitly a proxy
for the 252-session production window. The precommitted candidate grids are
`1,10,100,1000` for VolGAN and `1,3,10,30` for AD-Seq-Vol and AD-Seq-Vol-FT.
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
runs and batch-invariance evidence remain outstanding. The authenticated 2022
initializer and epoch-budget pipeline has now completed for all three generated
baselines. The one-standard-error selections are 1,000 rolling epochs for
VolGAN, 30 for AD-Seq-Vol, and one for AD-Seq-Vol-FT; their complete end-2022
production initializers and distinct selection-source lineages pass the strict
evidence loader. The authenticated legacy and current-code rolling GROMM ranges
are complete for all 611 eligible origins. The migrated range passes full
authentication with 576 accepted physical candidates and 35 declared
front-only fallbacks; generated readiness now owns the managed GPU. A
production-gate assembler now reconstructs exact public-versus-rolling scope
bindings, refuses stale or aliased evidence, writes under an output lock through
a unique same-directory temporary file, and re-authenticates the completed gate
before publishing it. The initial reviewed 18-scope managed-GPU readiness
matrix ran from immutable commit
`31b57e617dc6f57f12625515160131a8cf845e36`; VolGAN daily-public readiness
passed, but its first frozen-weekly case exposed two independent issues. The
checkpoint restore path was corrected at `b434852`, and the weekly price schema
was corrected at `b37a643ea8e5ef11b7557e37c035cf650a729e95` to permit finite
signed buyer, seller, and continuation indifference values. Common market and
genuine-Q prices remain finite and non-negative. This distinction is required
for non-monotone mean--variance and mean--volatility objectives; a negative
indifference value is not by itself a numerical failure. The weekly
mean--variance quadratic program uses OSQP, while mean--volatility retains
CLARABEL for its conic problem. On a synthetic 4,096-path geometry matching the
first production origin's 207 contracts and five decisions, the shared zero
policy plus first complete target fit took 45.1 seconds and converged without a
cap hit. A solver-reference test matches the CLARABEL optimum, and weekly runs
report target-level progress inside each origin.

Independent re-review at `b37a643` passed 161 focused tests, reconstructed all
six scope bindings for each generated model, and authenticated all three
production-gate round trips. Service
`quantdata-hedging-generated-readiness-b37a643.service` emits a persistent
four-minute heartbeat and acquires its GPU only through `gpuctl`. A fresh
managed-GPU readiness execution remains required before generated production
cases can launch.

A real 2023-01-17 frozen Historical-SANOS origin then completed all 13 targets
in 336.3 seconds with zero cap hits and zero nonconverged fits; its immediate
restart authenticated and reused the dated record. Single-origin readiness runs
now skip statistical result aggregation, whose loss and inference slices require
at least two origins, while complete production runs continue to aggregate
normally.

The 28 ready native weekly cases are repinned to
`b37a643ea8e5ef11b7557e37c035cf650a729e95` and are queued behind the
authenticated Historical-SANOS daily CPU artifacts they require. Their fresh
matrix still contains all 94 specified cells: 28 ready native cases, four
generated cases waiting on production gates, and 62 explicitly blocked cases.

Gate: one model-specific review and standalone smoke test per baseline pass.

### Phase 10: final benchmark and paper artifacts

Run the complete benchmark, generate versioned tables and figures, perform the
final results review, and reconcile every reported number with its source
artifact.

The daily and weekly report paths are matrix-driven and authenticated rather
than manually transcribed. The public CLI accepts only the public dataset and
immutable production matrices; it rebuilds the metrics and LaTeX fragments and
writes a manifest binding dataset, matrix, run, aggregation, and report hashes.
Precommitted matrix digests are explicit trust anchors. Final rendering requires
the canonical 57-cell daily and 94-cell weekly grids to be fully ready and
complete, so a post-outcome matrix cannot suppress a policy by relabelling it
blocked.
Weekly ingestion requires the complete
528-origin, 6,864-record population, verifies objective labels against the
immutable run snapshot, keeps frozen and restarted schedules distinct, and
generates aggregate objective tables and target-level appendix tables for all
four ex-post objective families at every precommitted parameter. Objective
functionals are evaluated in unscaled inception-spot-normalized loss units and
only the resulting scalar is converted to basis points for display. Final
weekly comparisons use equal-weighted 13-target episode books and report
candidate-minus-SANOS paired differences with Hansen--Hodrick and moving-block
bootstrap uncertainty. The model catalogue follows first appearance in the
authenticated daily grid, followed by weekly-only models, and uses the same
display names as the result tables. A standalone final-document builder now
renders all daily and weekly objective grids without selecting policies on
2023--2025
outcomes, target-level appendices, model references, and immutable provenance
directly from those authenticated inputs; its synthetic end-to-end PDF compile
passes.
A matrix-composition utility at immutable commit
`a1baa69605514ad9a0aba007ab8844f5fed29824` now replaces only selected native
cases in a fresh canonical matrix with the first complete authenticated source
case. Before accepting a source it loads the supplied public panel once, applies
the production executor's full authentication path, reconstructs the eligible
origin cohort and every dated record and aggregate, binds the actual dataset,
requires the canonical target book and common target quantities, and enforces
each source run's immutable numerical-gate policy. Strict cases reject any cap
hit or nonconvergence; precommitted mean-volatility recession diagnostics are
retained with their counts. It records the dataset, target-notional, and
target-cohort identities together with every source-matrix hash. Adversarial
tests reject a missing record, forged aggregate, wrong dataset, and wrong target
cohort; a real 12,220-record SANOS--LV composition smoke and focused composition
tests pass, and independent quant re-review closes the authentication blocker.
The last durable complete-suite record has 349 passing tests; the final full
suite will be rerun after production. This permits the final
57-cell daily and 94-cell weekly matrices to reuse completed native evidence
without copying it or rerunning it. The detached final-production service
`quantdata-hedging-full-production-b37a643-v5.service` passed its exact-environment
preflight and is waiting on generated-model GPU readiness, the corrected native
weekly matrix, Historical-SANOS exact-cost recovery, and covariance
power-exponential recovery. Once those prerequisites succeed, it
composes fresh canonical matrices under dedicated `production-matrix-b37a643`
roots, runs only the missing GROMM and generated-model cases, and rebuilds the
authenticated PDF. Its ordered daily source list gives first priority to the
corrected Historical-SANOS exact-cost matrix and covariance power-exponential
matrix, followed by the covariance mean--volatility recovery and remaining
authenticated native sources. The weekly source is the fresh `b37a643` native
matrix. Its read-only persisted launcher has SHA-256
`ba8971871b011d2ca1ff24674ffaa3a94c8c90227d8f902df050c7dd095a049a`.
The superseded `a1baa69` coordinator and its report-only/final-57 successors
were stopped while still waiting and did not launch production cases.

Weekly objective tables now
include equal-weight episode-book and worst-target maximum drawdown, computed
from chronological overlapping episode-origin P\&L and explicitly distinguished
from a non-overlapping self-financing-account drawdown. Fitted-policy parameters
and ex-post risk-score parameters have distinct headings and the correct
$\kappa$ or $\lambda$ symbol. The target appendix separately reports authenticated
inception market, accounting, genuine-Q, buyer-indifference, and
seller-indifference price means and rejects disagreement between stored price
records and realized accounting. Final completed-case figures, production-data
PDF regeneration, the final complete-suite rerun, and the final results review
remain outstanding.
The requirement-by-requirement acceptance state and the evidence still needed
for completion are tracked in
`quantdata/sdlv/daily/front_spot/hedging/reviews/2026-10-14-plan-completion-audit.md`.

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

## Resolved Production Decisions

The four GROMM hedge-policy candidates are fixed as
`(LSTM hidden size, deep depth, deep width)` equal to `(16,1,32)`, `(32,2,64)`,
`(64,2,128)`, and `(64,3,128)`, all with one recurrent layer, SiLU activation,
and the common affine action. The authenticated 2022-only selection study
evaluated all four over 40 measure/target cases and applied the precommitted
cheapest-within-one-standard-error rule.  It selected `(64,3,128)` with
validation loss `0.0798512`; the one-standard-error threshold was `0.0875964`.
The selection artifact has SHA-256
`3170f666a9bd750d91fe45143c87c5daf79c6f8b6ee57014f887f5b29420e3db` and is
stored outside the source tree at
`/home/hans/quantdata_data/training/hedging/gromm/architecture-2022-v1/result-v2.json`.

The episode risk unit, target-notional rule, 1,000-unit numerical position cap,
common spot-cap convention, statistical PCA alignment tolerance, and
inception-forward barrier reference are also resolved in the implementation
and their dated review or calibration reports. No benchmark-design decision
remains open before production; incomplete production cases remain gated only
on their required authenticated model evidence.
