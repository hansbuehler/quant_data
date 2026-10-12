# Weekly model-policy architecture review

## Scope and verdict

This review covers the common five-session engine and the existing daily model
adapters under `quantdata/sdlv/daily/front_spot/hedging`. It is a design review;
no implementation file was changed.

**Verdict: the common weekly episode, accounting, fit-timing, and inference
shells are suitable foundations, but production weekly policies are blocked.**
No model implements `WeeklyModelAdapter`, and the daily adapters cannot be
called from the weekly runner without changing their financial semantics. The
minimum correct implementation is a shared weekly scenario/policy layer plus
thin model-specific market-fit and scenario adapters.

The earlier GROMM2 reconciliation remains current after a read-only inspection
of `quantdata/gromm2`: its expanding-support Q simulator is useful architecture,
but it still does not expose a rolling dated P/Q benchmark artifact. It must not
be imported directly or treated as production-ready by the hedging package.

## What is already correct

- `benchmark/weekly.py` freezes the inception expiry-date/cash-strike book,
  preserves a fixed instrument dimension, applies an alive mask, settles cash
  options without an exit spread, and liquidates surviving contracts at the
  fifth close.
- The realized target uses the inception-forward-normalized path, while fixed
  hedge contracts and traded spot use the actual episode-origin spot path. The
  rolling-forward path remains diagnostic only.
- `WeeklyFitInstruction` gives one fit identity for frozen weekly and five
  dated identities for restarted weekly. Its rolling training origins are
  strictly before the fit date.
- `AsOfWeeklyMarketPanel` prevents a model fit from reading a future market
  state or transition. `WeeklyDecisionState` exposes only the realized path
  prefix, current marks/spreads/alive mask, and prior positions.
- `run_weekly_model` evaluates five predictable action rows and converts a
  current-date price per current spot into episode-origin PV correctly.
- Common inception prices exist for all 13 claims. Direct SANOS prices are used
  for digitals; SANOS-DLV is used only for the path-dependent common prices.
- Overlap-aware Hansen--Hodrick and block-bootstrap inference has already
  passed review.

## Blocking interface issues

### 1. The as-of view is not accepted by the reusable fitting code

`require_public_hedging_panel` requires an instance of the concrete
`PublicHedgingPanel`. The rolling factor, calibration, VolGAN, and diffusion
builders call that guard. `AsOfWeeklyMarketPanel` is intentionally a restricted
facade and therefore cannot be passed to them.

Unwrapping the facade inside an adapter would defeat the no-lookahead boundary.
Introduce a provenance-bearing read-only panel protocol accepted by the data
builders, add `next_one_session_date` to the as-of facade, and keep the concrete
unrestricted panel inaccessible to model code. An equally safe alternative is
for the runner to materialize immutable, dated fit tensors from the facade and
pass only those tensors to adapters.

### 2. Model fitting and policy fitting are conflated

The current `WeeklyModelAdapter.fit` is target- and frozen-book-specific. That
would repeat the same rolling factor fit or generator training for every target
and every overlapping episode. The production interface needs two identities:

1. a dated **market fit**, shared across targets and books when its inputs are
   identical; and
2. a **policy fit**, keyed by market-fit hash, target, claim sign, objective,
   risk parameter, episode origin, frozen contract IDs, and, for a restarted
   path-dependent claim, the realized prefix.

The separation is primarily a feasibility requirement, but it also makes the
audit trail unambiguous.

### 3. The price record is too small

`WeeklyFittedPolicy` exposes one non-negative `model_price_per_fit_spot`. That
cannot represent all quantities required by the benchmark:

- common SANOS/SANOS-LV market price;
- optional risk-neutral expectation, only for a genuine Q model;
- buyer and seller indifference prices;
- a restarted conditional continuation price at later decisions; and
- the signed claim policy to which the action belongs.

Replace the scalar conceptually with a price record. Store buyer and seller
prices separately and allow a Q expectation to be absent. The episode P&L must
use the inception price for the traded claim; later conditional prices are
diagnostics and state variables, not retrospective replacements for the
inception price. For a long backtest use the buyer price in the pure-hedge
decomposition; for a short backtest use the seller price.

For a translation-invariant objective with certainty equivalent `CE`, use

```text
p_buy  = CE(+Y + G_claim - C_claim) - CE(G_zero - C_zero)
p_sell = CE(G_zero - C_zero) - CE(-Y + G_short - C_short).
```

The long, short, and zero policies must be optimized separately when costs or
the risk functional prevent sign symmetry. “Claim minus zero” means subtracting
the zero-claim action from the action for the same signed claim; it does not
mean assuming the short action is the negative of the long action.

### 4. The daily objectives have the wrong weekly turnover semantics

The current daily objectives charge a one-period proxy proportional to
`|a|`. A weekly policy must charge predictable turnover,

```text
sum_n spread_n * |a_n - a_{n-1}|,
```

reset an expiring option position to zero without a closing spread, and charge
terminal liquidation only for surviving positions. The GROMM weekly training
loss and every empirical/generated weekly optimizer must use exactly the same
settlement and liquidation convention as `evaluate_weekly_hedge`.

### 5. A one-step target return is not a weekly path-dependent claim

The daily return builders and `build_generated_daily_scenarios` support one
transition and a European target mark. They cannot represent accumulated
realized variance, an already-breached barrier, or a fixed five-day digital
after part of the inception-forward path is observed. A restarted fit must
combine the immutable realized prefix with simulated or historical future
suffixes before evaluating the claim.

The weekly scenario object should contain, at minimum:

- pathwise target-normalized spot levels for all remaining closes;
- per-step spot and fixed-contract gains in fit-date or episode-origin PV units;
- fixed contract IDs, expiry dates, active masks, and settlement masks;
- per-step predictable spread costs and terminal liquidation costs;
- terminal signed claim payoff after attaching the realized prefix; and
- model-fit, policy-fit, measure, seed, and data-window identities.

### 6. The inception target basis must be explicit in every adapter

At a restarted decision, the remaining claim is still the inception claim. Its
strike, barriers, realized-variance prefix, and normalization do not reset.
The adapter therefore needs the inception forward curve or an equivalent fixed
conversion for all remaining dates. The current state supplies the realized
inception-forward prefix, but this dependency should be explicit in the fit
input rather than recovered incidentally from a rolling panel window.

## Schedule semantics

### Frozen weekly

- Fit the market model and the claim/zero policy once at episode origin.
- Keep all fitted parameters, normalization statistics, factor bases, policy
  weights, training samples, and random-seed conventions fixed for five days.
- Re-evaluate that same policy on each observed state prefix. Fixed analytic
  rules may solve their state-dependent action at evaluation time; this is not
  a refit if the rule and fitted inputs are unchanged.
- Reconstruct an LSTM state by replaying the observed prefix, or carry and
  audit the hidden state. Do not pass a later observation through a freshly
  normalized or retrained policy.
- Retain the inception contract dimension and force dead coordinates to zero.

### Restarted weekly

- At each decision date, refit the market model on the trailing window ending
  at that date and refit the policy for the remaining horizon.
- Attach the realized inception-target prefix before computing path-dependent
  future payoffs. Do not redefine the target from the current date.
- Use the previous realized positions in the first turnover cost of the new
  optimization.
- Use only the first action of the new remaining-horizon policy, then repeat on
  the next decision date.
- A warm start may come from a fixed pre-evaluation initializer or a strictly
  earlier checkpoint, but its hash and end date must be recorded. Validation,
  normalization, and checkpoint selection remain bounded by the decision date.

## Model-specific minimum designs

### SANOS-LV reference

**Reuse:** public DLV/SANOS states, common weekly prices, rolling factor fit,
fixed-contract repricing, DLV option Greeks, the factor-cost linear program,
and weekly accounting.

**Add:** a conditional remaining-claim pricer and central-difference Greek
engine that accepts the realized prefix. For digitals, transform the fixed
inception strike into the current model coordinate. For realized variance,
carry the accumulated squared-return sum and add the conditional remaining
sum. For barriers, carry a knocked-out flag and return zero value and zero
Greeks after breach. Compute Greeks for the currently alive frozen contracts,
not a newly selected option book.

For frozen weekly, hold the origin rolling PCA basis and Greek bump definition
fixed while applying the deterministic hedge rule to the observed market
state. For restarted weekly, re-estimate and align the rolling basis on each
decision date. In both schedules solve for the first eight factor vegas and
spot delta with the current predictable turnover costs.

The existing weekly pricer is valid for inception Q prices but not for
conditional prices or Greeks. The daily SANOS-LV adapter is invalid because it
reselects the hedge universe and prices only a 1D/3D European target.

If the benchmark requires literal buyer/seller indifference prices from this
reference, a risk objective and SANOS-LV path distribution must be specified.
Otherwise report the frictionless Q expectation as such; do not call it a
transaction-cost indifference price.

### Independent covariance

**Reuse:** the robust Tyler estimator, mean screening, mean-variance and
mean-volatility primitives, and cost-lasso diagnostics.

**Add:** historical remaining-horizon blocks constructed entirely from public
spot and option-price returns. At episode origin, reproduce the frozen book at
the same inception DTE/NMS/right coordinates on each historical start and
follow each synthetic fixed-strike contract through up to five transitions.
Construct the corresponding target payoff from each complete normalized path.
The last endpoint of every training block must be no later than the fit date.

There is also a model-definition blocker in the current daily implementation:
`models/covariance/returns.py` reconstructs its history with SANOS prices, while
the plan distinguishes independent covariance from the historical-SANOS
baseline. The distributable instrument rows contain separate DLV price arrays,
but `PublicMarketDay` currently exposes only SANOS prices. Either expose and use
the released DLV price geometry for covariance estimation, leaving SANOS only
as the common execution mark, or rename the method as a SANOS-covariance
variant. It cannot be called independent of SANOS in its present form.

The minimum frozen policy can be an open-loop five-row position schedule fitted
at inception; “may react” does not require state dependence. The minimum
restarted policy refits an open-loop remaining-horizon schedule each day after
attaching the observed prefix, and trades only its first row. This remains an
independent covariance baseline because it uses only empirical return blocks
and their robust first two moments.

The weekly optimizer must work on flattened step/instrument exposures while
preserving active masks and turnover costs. The present one-step
`build_historical_return_panel` and coordinate-descent action are not valid
weekly substitutes. Entropy and power-exponential may be reported only through
the predeclared Gaussian equivalence; the direct covariance model supports
mean-variance and mean-volatility natively.

### Historical statistical SANOS

**Reuse:** public SANOS repricing, the four empirical risk functionals,
claim-minus-zero separation, and deterministic optimization diagnostics.

**Add:** the same complete historical remaining-horizon blocks as above, but
retain their full empirical joint distribution rather than reducing it to a
robust covariance. A fully consecutive window of 252 one-day transitions
provides 248 overlapping five-day blocks; retained-date gaps reduce that count
and must never be bridged. Shorter restarted horizons provide more complete
blocks. Every block must finish by the fit date.

Fit signed claim and zero policies to cumulative episode wealth with actual
position changes, expiry resets, and terminal liquidation. The frozen minimum
is an inception-fitted open-loop five-row policy. The restarted minimum refits
the remaining rows daily using the realized prefix and previous position.

The current static `fit_historical_sanos_hedge` objective and its daily return
panel are useful numerical kernels but do not implement this dynamic wealth or
cost accounting.

### GROMM P and Q

**Reuse:** the four objective kernels, affine-plus-deep/LSTM policy family,
claim-minus-zero checkpoint separation, rolling 16-factor calibration concepts,
and first-eight-factor conventional Greek hedge. Reuse numerical ideas from
GROMM2 only through a stable shared kernel or immutable artifact contract.

**Add:** a dated rolling GROMM2-compatible market artifact with an exact front
bridge, expanding supports, density-localized 16-factor dynamics, and both P
and Q transitions. It must be built only from the fit window and price the
exact frozen benchmark contracts and all weekly targets. The current GROMM2
history/estimate artifacts are full-panel/end-date constructions and its path
runner is Q-only, so this is a hard gate.

Generate complete five-day scenario paths at inception. Policy features should
include the current model state, remaining horizon, target descriptors,
inception-normalized spot prefix, accumulated variance or barrier state,
alive-mask information, current fixed-contract marks/spreads, and prior
positions. Train claim, short-claim, and zero policies on cumulative wealth
with turnover and liquidation costs.

For frozen weekly, train once and replay the same network on the realized
prefix. For restarted weekly, recalibrate and retrain a remaining-horizon
policy daily. Q runs report both the risk-neutral expectation and objective-
specific buyer/seller indifference prices; P runs report indifference prices
only. The conventional Q-expectation plus eight-factor DLV-vega hedge remains a
separate GROMM result, not a deep-policy price.

The existing one-day `generate_gromm_daily_scenarios`, one-step feature tensor,
and `gromm_hedging_objective` are not weekly-valid. The policy network itself is
structurally reusable once trained on genuine path-step tensors.

### VolGAN

**Reuse:** dated rolling training, immutable warm-source checks, normalization
rebasing, one-step conditional sampler, and fixed-grid SANOS surface encoding.

**Add:** recursive five-day sampling. After each generated transition, update
the current surface, the last two returns, and the trailing 21-day realized-vol
condition before drawing the next transition. Reprice only the frozen
cash-strike contracts, apply their expiry masks, and evaluate the weekly target
on the full inception-normalized path.

The published minimum-variance cost-lasso can fit an open-loop path policy. Its
centered objective has a zero-claim optimum of zero in population; still compute
and record the zero-policy diagnostic before differencing. Frozen weekly uses
the one origin-fitted generator and one origin-fitted policy. Restarted weekly
refits the generator and remaining-horizon policy daily, conditional on the
realized state and prefix.

VolGAN is a P model. It does not supply a risk-neutral expectation. The current
daily generated-scenario repricer and one-step lasso are not reusable as a
weekly policy.

### AD-Seq-Vol and AD-Seq-Vol-FT

**Reuse:** dated rolling training, fixed initializer/base hashes, exact
normalization rebasing, EMA sampling, and the 21-state live context.

**Add:** autoregressive five-day sampling. Append each generated return/surface
state to the 21-state context, drop the oldest state, and sample the next day.
Reprice the fixed inception contracts and weekly target along the resulting
path. Use the same open-loop published cost-lasso and explicit zero-policy
diagnostic as VolGAN.

Frozen weekly keeps the fitted diffusion/EMA model and policy fixed for the
episode. Restarted weekly refits on each decision date and rebuilds the
remaining path distribution from the observed context and target prefix. Both
diffusion variants are P models and do not report Q expectations.

The current sampler deliberately returns only day 22 and the common generated
scenario builder handles only one step, so neither is yet a weekly engine.

## Reusable versus invalid daily components

| Component | Weekly status |
| --- | --- |
| Public market states, fixed-contract repricing, common claims, weekly accounting | Reuse unchanged |
| Weekly fit identities and realized-prefix decision state | Reuse after the price/provenance extensions above |
| Rolling DLV factors and calibration math | Reuse behind the as-of panel boundary |
| SANOS-LV common inception prices | Reuse; not a continuation pricer |
| SANOS-LV daily target Greeks/book selection | Do not reuse directly |
| Tyler covariance and risk solvers | Reuse as kernels inside a multistep turnover-aware optimizer |
| Daily historical relative-return panel | Do not reuse as the weekly sample |
| Historical SANOS four loss formulas | Reuse after changing wealth and cost construction |
| GROMM LSTM affine-plus-deep network | Reuse for genuine multistep features |
| GROMM one-step scenarios/objective | Do not reuse directly |
| VolGAN/diffusion fit, checkpoint, and one-step conditional samplers | Reuse and roll recursively |
| One-step generated repricing and lasso | Do not reuse as a weekly policy |

## Implementation sequence

1. **Harden interfaces.** Add the provenance-preserving as-of public-panel
   protocol, split market-fit and policy-fit identities/caches, and replace the
   scalar model price with an inception/continuation price record. Add tests
   that deliberately attempt future access through every model data builder.
2. **Implement common weekly scenario tensors and objectives.** Centralize
   fixed-contract repricing, alive/settlement masks, prefix-plus-suffix target
   evaluation, position-change costs, terminal liquidation, signed claim/
   zero policies, and buyer/seller price extraction. Prove numerical agreement
   between the training wealth and `evaluate_weekly_hedge` on deterministic
   paths.
3. **Implement SANOS-LV first.** Add conditional path-dependent prices and
   Greeks, then verify frozen factor-basis reuse and restarted refits on a small
   public episode.
4. **Implement historical SANOS.** Build no-lookahead five-day empirical
   blocks and the multistep four-objective optimizer. This supplies the first
   complete claim-minus-zero weekly policy and tests the common tensor layer.
5. **Implement covariance from the same blocks.** Reduce to robust first two
   moments and support native mean-variance/mean-volatility plus explicitly
   labelled Gaussian-equivalent objectives.
6. **Implement VolGAN and diffusion recursion.** Use their already hardened
   rolling fits and checkpoint paths, validate one-step CPU/GPU agreement, then
   test five-step recursive state updates, fixed-contract repricing, and
   restart recovery.
7. **Gate GROMM2 integration.** Produce rolling dated P/Q market artifacts,
   exact benchmark-book pricing, martingale/physical-drift audits, and bounded
   GPU memory. Only then train the multistep GROMM claim/short/zero policies and
   the conventional Q Greek hedge.
8. **Run production and inference.** Cache market fits across targets and
   overlapping episodes, retain hashes and progress logs, reject cap or
   convergence failures, and apply the existing overlap-aware inference to
   completed episode records.

## Required acceptance tests

- Future market dates, future transition endpoints, and future normalization
  rows are inaccessible in every adapter and generated-model builder.
- Frozen schedules have one market-fit hash and one policy-fit hash per signed
  claim; restarted schedules have five dated hashes.
- A frozen neural policy produces the same action from replayed prefixes before
  and after serialization; no normalization is refitted inside the episode.
- The contract IDs and action dimension remain fixed, dead coordinates are
  exactly zero, and no newly listed option enters.
- Barrier breach and realized-variance accumulation are determined solely by
  the observed prefix plus each simulated suffix.
- Training pathwise wealth, transaction costs, expiry resets, and terminal
  liquidation agree with common accounting on deterministic fixtures.
- Long, short, and zero policy identities are distinct; buyer/seller price
  signs and Q-expectation labels are correct.
- Every historical training block ends no later than the fit date.
- VolGAN and diffusion recursive contexts advance exactly one retained session
  per step.
- GROMM Q paths pass conditional/unconditional martingale audits; GROMM P paths
  pass the precommitted physical-drift gates before hedge training.
