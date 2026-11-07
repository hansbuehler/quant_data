# Hedge2 — specification, implementation status and review

Updated 2026-11-07. This document supersedes the duplicated Markdown progress
diaries. It distinguishes implemented code, numerical evidence and unfinished
rolling experiments. Implementation comments and diagnoses below are AI-agent
findings; noted QuantAgent reviews are not independent empirical validation.

## 1. Scope and timing

- Use saved smoothness-0.1 SANOS interpolators and licensed observed European
  midpoints. Do not refit the upstream interpolator or substitute fitted
  prices for missing observed quotes.
- Initialize from 2022; evaluate eligible 2023–2025 origins. The available
  archive ends on **2025-08-29**; September–December 2025 is missing.
- Each close uses information through that close and the latest **252 exact
  one-session transitions**. Same-close fitting and execution is a disclosed
  latency idealization. It is not an expanding-history estimator.
- Fit a new model daily and recompute remaining-horizon hedges from carried
  inventory. Start overlapping episodes every eligible session. Simulations
  at today's decision freeze today's estimated dynamics; they cannot consult
  tomorrow's historical fitted model.
- Keep inception contracts, cash strikes, barrier and episode numeraire fixed.
  Allow contracts to expire but add no newly listed expiry during an episode.
  Group statistical uncertainty by market date and account for overlap.
- Alpha forecasting is out of scope. Remove forecast instrument means in the
  model-consistent way, retaining raw gains and removed means. **Never demean
  realized cash P&L or claim gains to improve reported hedge performance.**

## 2. Data, daily state and rolling dynamics

### Source and direct conversion

The saved source archive is
`/home/hans/quantdata_data/quant_data/qdata/runs/spx-private-public-20261101T135819Z-819d9b59`.
Each date's pricer is `days/<date>/models/all-0.1/model.h5`. Source files,
licensed carry and observed quotes retain separate hashes and meanings.

The first representation is **smoothness-zero DLV**, not a fitted smoothed
SANOS state or rolling DYSANOS joint-ML model. Use tenors **1, 2, 3, 4, 5,
10, 15, 20, 25 business days**, with 200 interior strikes shared across
expiries on each date. `shared_inner_strikes` merges front/final cumulative
forward-volatility NMS geometries over **(-4,3)**, retaining ATM and minimum
spacing. It is not a separate uniform NMS grid at each expiry. Geometry floats
between dates; strike-homogeneous here means common across that date's expiries,
not necessarily equally spaced in strike. Zero backbone has no eta coordinate.

Price the grid from the stored SANOS pricer, then recover DLV squared using
backward implicit Dupire, not an LP or iterative price fit:

\[
 \Sigma_{j,i}
 =\frac{2(C_{j,i}-C_{j-1,i})}
 {\Delta_j K_i^2\,\Gamma_{j,i}},
\]

where `Gamma` is the nonuniform-grid second difference at the later expiry.
The previous calls already share the grid; no inter-expiry regridding is needed.
See equation (20) and section 4.3.2 of
[Discrete Local Volatility for Large Time Steps](https://quantitative-research.de/dl/Discrete%20Local%20Volatility%20-%20Extensive.pdf).

Store source/calculation/decoded calls, support, forward variances, identification
masks and the expiry-major log-DLV-squared/log-forward-volatility state (1,809
coordinates). DLV conversion has no fitting weights. A future smoothed extension
may use the requested 10% wing weight outside (-3.5,2.5); it is not implemented.

Numerical safeguards must remain visible:

- Local-volatility bounds 0.001–1000 protect arithmetic; values above 10 have a
  separate extreme flag. Preserve raw values and every fallback/floor/cap count.
- Certified source mass/first-moment roundoff and a calendar-prefix envelope
  are corrected only within the source's 1e-5 tolerance. Decoded calls must
  reconstruct the **untouched source** within 1e-6 in forward-normalized units;
  corrections do not receive separate error allowances.
- Record first quoted tenor and short-end extrapolations. A valid call fit does
  not guarantee accurate digitals or stable latent PCA dynamics.

### PCA and AR(1)

Fit eight PCs to floating-state changes. Compare ordinary and spot-orthogonal
variants separately. The orthogonal variant removes the historical physical
spot-return loading before PCA, and restores its **full state-space loading**
once in simulation and parametric spot exposure, including components outside
the retained PCA span. This is not alone a spot-variance calibration.

The current AR(1) is ridge linear regression on factor levels, followed by a
spectral-radius restriction to 0.999; it is not a 252-day neural/L-BFGS fit.
Carry factor ordering/signs from the preceding fitted date, record the
initialization date and reject incompatible geometry rather than cold-starting
silently. PCA uses sample covariance; the uniform empirical innovation law uses
population covariance. Save both constrained/unconstrained persistence,
stationary mean, explained fractions, residuals and raw/spot-conditioned errors.

Three legacy spot returns mixed AM/PM carry. `observations` now shares the
PM-only `historical.licensed_curve` used in cash accounting. Old artifacts are
retained, not overwritten. Corrected preparation contains 910 observations,
901 exact score pairs and 657 fitted dates per PCA variant; all 1,314 windows
align with 252 preceding scores. Ordinary Greek bases are unchanged; 51
orthogonal bases change. Array equality does not replace a cash/risk audit.

## 3. Claims and common market accounting

NMS levels are **-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1**. ATM means
cash strike equal to inception spot, not an ATM-forward convention. Other
strikes use `NMS.from_nms` with the inception tenor-specific ATM volatility;
do not replace its implemented convention with an undocumented approximation.

| Family | Definition | Targets |
|---|---|---|
| 1D/3D/5D digitals | Put for negative NMS, call for positive; both put then call at ATM | 10 per tenor |
| 3D observation / 5D KO | Inception barrier at spot, checked only on day 3; call/down-out below-ATM strike, otherwise put/up-out; strike uses 2D NMS | 9 |
| 3D reset / 5D ratio straddle | `abs(S_5/S_3-K)`, strike ratio from 2D NMS | 9 |

Total: **48 targets**, each long and short. Digital call pays on `S>K`; put on
`S<=K`. KO barrier equality knocks out. A knocked-out hedge stops on day 3;
it must not incur artificial further trading or exit charges.

The hedge book uses ITM observed Europeans within **(-2,2)** expiry-ATM NMS,
with puts/calls chosen to avoid parity duplication. The 1D book uses expiries
up to 5D or the first later expiry if 5D is absent. Multi-day digitals use
co-terminal Europeans; exotics retain fixed inception 3D/5D contracts.
Scenario methods include co-terminal synthetic zero-strike calls explicitly.
**Simple Greek matching excludes these forwards**; spot remains available.

- Observed European mids and half-spreads determine actual P&L/cost. Missing
  quotes stay missing, never zero or model-filled observations.
- Source pricing uses parity forward `F` and `df*F`; financed spot gains use
  the separate reference forward `F_ref`: `D_1*(S_next-F_ref)`.
  European gains are `D_1*V_next-V_now`.
- Spot cost is zero. Options pay **gamma=1 times entry/rebalance half-spread**.
  Settlement has no exit trade. Report hypothetical exit cost separately.
- Carry all cash flows in one inception numeraire. Actual ITM hedge positions
  are current-premium dollar allocations, capped at **$10 per instrument**.
  They convert back to contract holdings for cash P&L and turnover. Target
  digitals remain unit cash payoffs, not premium-normalized target returns.
- The separate experimental factor-basket limits are $10 per instrument and
  $100 gross per basket; the $100 limit is **not** a final-hedge gross limit.
- Common digital marks come from the stored smooth SANOS European curve;
  they are interpolated market marks, not observed digital quotes. Do not
  silently replace them by recovered atomic-DLV prices in one method only.

Complete-book evaluation cohorts contain 467 one-day origins, 319 3D-digital,
201 5D-digital and 167 each KO/ratio origins. These are ex-post complete-case
subsets of causally selected books, not the entire historical population.
The upstream licensed collector filters tiny bids; missing marks and excluded
origins must remain explicit. Black comparisons may have additional bracketing
coverage restrictions and must use a separately matched subset.

## 4. Methods and objectives

| Method | Risk/hedge construction | Main limitation |
|---|---|---|
| Unhedged | Common mark and realized payoff, no trade | Coverage-matched reference |
| BS delta + vega | Bracket target strike with two European vegas; offset added delta in spot | Frozen book may not bracket after reset |
| DLV Greek | Spot plus eight finite PCA shocks; re-solve under Greek/cost constraints | Atomic digital exposure depends on bump size |
| DLV parametric | Same exposures with full spot/factor covariance; Gaussian P&L approximation | Local daily risk, not terminal optimal control |
| DLV statistical | Reprice generated joint spot/surface scenarios under sample objective | Corrected front law not yet selected/rolled |
| SANOS historical | Replay exact fixed contracts over 252 saved SANOS one-day pairs | Historical risk baseline, not today's exact conditional law |

Greek cost budgets are **100%, 80%, 60%, 40%, 20%, 0%** of the full Greek
hedge's cost. Re-optimize at each budget; never scale a saved position vector.
Reduced budgets minimize a diagonal-risk-weighted Greek residual, not the full
covariance parametric objective. Box or mandatory-turnover infeasibility must
be proven and reported separately from numerical failure, with executed prefixes
preserved. Current no-forward Greek populations request 1,321 family/origin
episodes and 154,512 portfolios per PCA variant.

For the statistical/parametric methods, optimize both sides separately:

- Mean–variance risk `-E[W]+lambda*Var(W)/2`, lambda **1,10,100,1000**;
  lambda=infinity means **pure variance**, exactly zero cost coefficient in
  the training risk. Actual entry costs are still charged in realized P&L.
- Power-exponential OCE at lambda **1,10,100,1000**. Utility is
  `(1-exp(-lambda*x))/lambda` for nonnegative wealth and
  `x-lambda*x*x/2` below zero; optimize its cash shift. This is not the
  squared-exponential or square-root utility discussed elsewhere.
- Wealth is in fixed initial-notional units, with **no hidden /10,000** in
  lambda. Future turnover costs enter pathwise terminal wealth inside the risk.

Historical SANOS replay maps the current contract to each historical start,
then keeps its cash strike fixed across that pair while maturity ages one day.
Instrument gains use each historical entry premium before demeaning; target
cash gains retain their own units and varying historical start marks. Keep raw
gains, means, dates, mapped strikes, queried carry and source hashes. The
current historical production scope is 1D digitals only. European marginals
alone do not define historical KO/reset scenarios.

For multi-day **statistical** hedging, optimize final hedge error through
expiry with nonanticipative future actions, not only tomorrow's mark change.
The shared-state terminal policy, differentiable cash accounting and finite-tree
reference exist. A locally optimized one-day controller is not an equivalent
replacement. Neural training stability is not a global optimum certificate.

Black exotics use one coherent variance curve: forward variance from 3D to 5D
for the ratio straddle, and conditional surviving vanilla value integrated over
the day-3 distribution for KO. Survival probability times unconditional vanilla
price is incorrect. Hedge nonzero 0–3D and 3–5D variance buckets separately.
Before ratio reset its pure-BS spot delta is zero; afterward keep reset spot
fixed when differentiating. Offset European hedge delta with spot.

## 5. Front-density correction — implemented candidates, unresolved law

The intended one-session spot marginal is the **current recovered front DLV
density**: draw genuine `Z~N(0,1)` and apply its inverse CDF to `Phi(Z)`.
`front_spot` uses tail-stable normal thresholds and separate parity/financing
forwards. The surface beta multiplies **physical centered spot return**, not Z.

Dependence training uses preceding smooth-SANOS scores
`Phi^-1(F_previous(S_next/F_previous_forward))`. Scores are not bootstrapped or
empirically normalized to replace the new standard-normal spot driver. Atomic
DLV CDF scores remain separate diagnostics, not continuous uniform PITs.
Across 901 transitions, raw smooth scores have mean 0.00456 and population
standard deviation 0.99095; zero are tail-clipped and 31 flag short-expiry
extrapolation. These summaries do not establish conditional calibration.

`front_joint` implements two **unselected** candidate joint laws:

1. `empirical_rank`: couple the new Gaussian rank to empirical factor-residual
   ranks. Ties share their combined interval, sampling tied rows uniformly;
   do not replace tied residuals by averages. Exact finite branch intersections
   retain front and empirical marginals, including tested tiny tails.
2. `joint_gaussian`: estimate score/factor covariance, Gaussianize innovations
   jointly, then transform spot through the discrete front CDF. It preserves
   fitted factor mean/covariance but changes higher moments and tail dependence.

`front_scenarios` and `objective_run` wire both candidates explicitly through
`--front-scores-path` and `--front-surface-law`. Parametric covariance uses the
chosen physical-return/factor covariance with correct factor units; treating
resulting P&L as Gaussian remains an approximation. Snapshot/config identities
bind source code, scores, dependence, common marks, geometry, seed and inputs.
Defaults retain the legacy law for honest checkpoint compatibility, not as an
endorsement of raw-spot production.

Generate at least `max(10000,10*instruments)` draws per forecast. Smaller counts
require diagnostic mode. Empirical compression merges only identical front-atom
and residual-row pairs with their actual frequencies. Gaussian paths stay
continuous. More Monte Carlo draws do not create additional historical data.
Instrument gains are frequency-demeaned; claim gains are not. This frontend
is next-day marked risk, **not** a multi-day terminal solution.

The completed diagnostic covers three dates, both PCA variants and both laws:
12 cases and 120,000 draws. Independent selected-row cash repricing error is
at most **1.91e-14** in spot units; target error is zero. This checks financial
implementation, not hedge performance or conditional martingality. Many latent
coordinates reach lower/upper decode guards. Counts are saved, not suppressed;
clipping changes the latent dynamics and must be assessed before production.

## 6. Status and retained evidence

### Running snapshot at publication, 2026-11-07

| Work | Verified snapshot | Status |
|---|---|---|
| Ordinary no-forward Greek | 337/1,321 episodes; 37,490 complete portfolios, 2,050 unavailable | CPU running |
| Corrected orthogonal no-forward Greek | 142/1,321 episodes; 15,926 complete portfolios, 778 unavailable | CPU running |
| SANOS historical 1D tensor | Two active date partitions, third queued; atomic accepted cells retained | CPU running |
| Front marginal/scores/dynamics | 910 observations, 901 pairs, 657 fits per variant | Preparation complete |
| Candidate front joint one-day laws | 12 x 10,000-draw cash-repricing diagnostic | Complete, diagnostic only |
| Corrected generated hedge tensor | No candidate joint law selected | Not run |
| Corrected terminal simulation/policy | New front law not wired through terminal paths | Incomplete |

These are timestamped progress counts, not completion promises. Read the
external run's `progress.json` and process status for newer values. This project
holds no GPU at the snapshot; another project's allocation is not ours to use.

All run names below are relative to
`/home/hans/quantdata_data/quant_data/hedge2/runs`:

| Evidence/run | Meaning |
|---|---|
| `branch-publish-validation-20261107-re2aEN` | Isolated `hedging-benchmark` source-stable validation: **756 CPU tests passed**, two unallocated-GPU skips; import sorting/syntax pass; two original warnings retained |
| `front-calibrated-preparation-20261107-yiz7DS` | Corrected PM-only observations, smooth/atomic scores, both daily rolling variants; all 1,314 score alignments; max front-mean/parity error 8.08e-16 |
| `front-joint-market-validation-20261107-wXieTD` | Three-date candidate-law cash/target replay; frozen source, arrays, guard counts; no optimizer or OOS claim |
| `greek-spot-options-book-20261107-WiGACn/ordinary` | Current no-forward ordinary Greek population |
| `greek-corrected-carry-orthogonal-20261107-q4tggI` | Current carry-corrected orthogonal Greek population |
| `historical-only-continuation-20261107-nj4u38` | Two-worker continuation of original historical atomic cells |
| `one-day-itm-full-roll-20261106T2200` | Historical tensor inputs/results; stopped raw-spot generated baselines remain separately labelled |
| `greek-budget-tangent-recovery-20261107-xSVnag` | Original-input wider-arithmetic tangent certificate, same saved candidate and unchanged acceptance criteria |
| `greek-cost-bound-recovery-20261107-gowJbw` | Original-input minimum-cost bound repair and independent prefix replay |
| `terminal-coordinate-actualgate-20261107-fLeoJe` | Copy-only terminal numerical-coordinate value/gradient/restart checks on the legacy law, not corrected rolling results |
| `one-day-all-methods-matched-20261107-ftMRmh` | Interim matched report, legacy generated law; not a corrected full-population comparison |

The publication run also archives the three **unmodified pre-consolidation
Markdown files and SHA256SUMS** in `markdown-before-consolidation`. Those
chronological diaries contain superseded/live-at-the-time statements; use this
document for current decisions. No licensed scenario arrays, model checkpoints,
logs or raw records are included in the source commit.

## 7. Code and quant review pack

### Recommended reading and standalone tests

| Review area | Start here | Focused tests |
|---|---|---|
| Source, carry, reconstruction | `source`, `cash`, `dlv`, `state`, `pricing` | `test_source`, `test_cash`, `test_dlv`, `test_state`, `test_pricing` |
| Causal factors and AR(1) | `dynamics`, `rolling`, `observations` | `test_dynamics`, `test_rolling`, `test_observations` |
| Front law and dependence | `front_spot`, `front_scores`, `front_joint`, `front_scenarios` | Corresponding `test_front_*` modules |
| Greek budgets and feasibility | `greek_hedge`, `greek_budget`, `premium`, `greek_frontier` | `test_greek_*`, `test_premium` |
| Objectives and certification | `objectives`, `optimization`, `dual`, `parametric` | `test_objectives`, `test_optimization`, `test_dual`, `test_parametric` |
| Historical and actual cash replay | `historical`, `audit_daily`, `audit_episode` | `test_historical`, `test_audit_*` |
| Terminal control and restart | `terminal*`, `snapshot_run` | `test_terminal_*`, `test_snapshot_run` |

Test names are module stems in `tests/quantdata/hedge2`; use the matching
`test_*.py` file. The full isolated-branch test log preserves the literal
initialization warning and intentional overflow-stress warning. GPU tests skip
explicitly without an allocation; no GPU validation is implied by CPU success.

### Quant review questions — open unless stated otherwise

1. Choose and justify the spot/surface coupling. Marginal front calibration
   alone does not fix surface tail dependence or option conditional drift.
2. Quantify decode guard activity by coordinate and scenario weight; assess
   its contribution to PCA, forecast risk, prices and digital exposures.
3. Verify sampled one-day mean removal versus exact branch integration, and
   implement predictable conditional correction throughout terminal paths.
4. Check atomic digital strike-side conventions and spot/PCA bump sensitivity.
   Call reconstruction is insufficient evidence for stable digital Greeks.
5. Preserve PM-only carry and physical-beta/factor-scale units across every
   simulation, parametric covariance and realized-cash adapter.
6. Review complete-case selection, missed later quotes and Black bracketing
   exclusions before interpreting matched OOS rankings.
7. Confirm each multi-day statistical decision optimizes final net hedge error
   from actual inventory, with no future observations in the shared-state policy.
8. Distinguish a numerical convergence certificate, a training-stable neural
   candidate, a no-arbitrage diagnostic and economic hedge performance.

### Code review checks

- Validate units, shape/dtype/device, finite probabilities and guarded division;
  verify independent scalar/cash calculations rather than self-repricing only.
- Check hashes and local dependency closure in immutable snapshots. Never load
  old checkpoints under changed economics, geometry, scores or source code.
- Preserve exact warning text. `OPTIMAL_INACCURATE` or “Solution may be
  inaccurate.” is not alone a rejection: assess independent original-unit
  primal/dual gaps, feasibility and finite values. A real unresolved failure
  remains visible, with no silent portfolio substitution or cap relaxation.
- Numerical cost/tangent certificates use original-input wider arithmetic and
  explicit error allowances; fall back conservatively where wider precision
  is unavailable. Do not describe these as rigorous interval proofs.
- Terminal checkpoints retain physical weights, optimizer/scheduler/RNG state,
  coordinate mappings and iteration counts. Input-coordinate whitening and
  shared-history acceleration must preserve values, gradients and exact restart.
- The legacy terminal pilot reached its 100,000-update limit without acceptance.
  A copy-only CPU coordinate trial is not a promoted policy. No holdout evidence
  or corrected terminal rolling result has been accepted from either trial.

Prior read-only QuantAgent reviews covered the front marginal/candidate-joint
algebra, financial cash conventions and several numerical certificate repairs.
They do not certify the newest runner integration, resolve the questions above,
or establish out-of-sample model adequacy. Review the full current diff.

## 8. Next implementation and execution order

1. Review/select the front coupling and assess guards before launching a new
   generated hedge population. Retain both candidate implementations as
   explicitly labelled experiments until that decision is made.
2. Integrate the selected law and conditional instrument correction into terminal
   simulation, risk covariance, cache/checkpoint identities and independent
   settlement/mean/cash tests. Do not silently recycle raw-spot paths.
3. Finish current Greek/historical CPU populations with independent cash/risk
   and unavailable-prefix audits. Preserve accepted predecessors unchanged.
4. Run the corrected complete one-day tensor on a small dated pilot, then full
   cohort; subsequently complete remaining-horizon statistical policy training
   and rolling execution. Use generated validation only for training decisions;
   untouched evaluation paths/market outcomes cannot choose stopping or settings.
5. Report all requested model/objective/risk/side/target cells on matched dates:
   costed mean/median P&L, std, MAE, downside/tail risk, entry and hypothetical
   exit costs, cap-hit frequency, numerical status and missing coverage. Show
   pricing-versus-hedging contributions separately; use date-block uncertainty.

CPU handles conversion, PCA/AR, pricing pilots, convex objectives and audits.
Use bounded parallel independent workers without breaking causal rolling
initialization. Optional terminal neural acceleration requires measured benefit
and an explicit `gpuctl` allocation. Check `gpuctl resources --json` before
material launches and every five minutes during background inspection; follow
host WARN/STOP actions. Every long GPU job must flush completed/total work,
elapsed time and meaningful ETA at least every five minutes.

Detached jobs use `/home/hans/vqdata/bin/python`, an explicit working directory,
`QUANTDATA_ROOT` and `PYTHONPATH`; validate imports and data discovery in that
same environment first. Every run owns a unique external directory. Estimate
ETA from measured work units and report blocked calibration/training separately,
not as a promised finish time. Commit only source/tests/specifications and
deliberately retained summaries to `hedging-benchmark`, never to `main`.
