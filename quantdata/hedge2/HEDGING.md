# Hedging benchmark: agreed specification

Updated: 2026-11-09.

This is the single review document for hedge2: agreed specification,
implementation concerns, and proposed validation sequence. It consolidates
the former README, plan, review notes, and handoff. Historical numerical
claims are not carried forward as current evidence. Their complete original
text is preserved in the external archive listed in section 9.

**Execution remains paused.** The user stopped the hedging work because the
implementation mixed up hedge estimation, valuation, and optimization
horizons. The latest instruction is **consolidate for review; do not code
yet**. No implementation changes or rolling runs are authorized by this
document. Earlier hedge results are superseded, not validated evidence for
this specification.

## 1. Purpose and scope

Compare daily hedging methods on the same targets, market observations,
tradable instruments, costs, and dates. The benchmark concerns hedging, not
forecasting speculative alpha or choosing different entry prices for each
method.

Use 2022--2025 history, with 2022 providing initialization and subsequent
eligible dates providing rolling out-of-sample evaluation. Use the preceding
252 eligible one-session transitions for each daily dynamics estimate. A
fresh as-of-t model uses information available through close t, including
that day's fitted surface, and hedges t to t+1. Trading immediately at that
same close is a latency idealization that must be disclosed.

## 2. Separate hedge estimation from valuation

### Common source for both hedge representations

For **both DLV and SANOS hedging models**, use the already fitted historical
**smoothness-0.1 SANOS market interpolation**. Do not refit the original
market quotes or invent another interpolation algorithm.

The licensed source archive is
`/home/hans/quantdata_data/quant_data/qdata/runs/spx-private-public-20261101T135819Z-819d9b59/`.
Its saved fits are under `days/<date>/models/all-0.1/model.h5`; retain
source certificates and checksums with every derived daily state. The
presence of earlier dates in that archive does not extend the agreed
2022--2025 estimation period.

For every date, construct the requested hedging representation:

- Business-day expiries: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 25**.
- **200 strikes**, spanning NMS **(-4, 3)**.
- Use ATM volatility to construct the relative strike grid, with the
  established SANOS/DYSANOS NMS geometry. The grid floats with the market
  date; it is not one permanently fixed cash-strike grid.
- Preserve the strike-homogeneous geometry required by the backward
  implicit Dupire construction. Save the actual relative strikes, ATM-vol
  scale, expiry times, and NMS mapping rather than leaving them implicit.
- Fit the DLV and SANOS hedge representations to prices from that same
  0.1 source. Source-interpolation smoothness and hedge-representation
  smoothness are different concepts and must be recorded separately.
- Compute PCA and rolling dynamics from the resulting daily states.

This replaces the old nine-expiry hedge grid and the handoff's interim
instruction to build DLV **hedging** states from smoothness-0 SANOS.
The current requested hedge grid has **13 expiry rows**, not nine.

### Market prices and common target marks

**Premiums are market premiums**, not the hedge model's preferred prices.
For quoted hedge instruments use observed mids from the licensed source
cache. Keep observed mids, fitted interpolation prices, and hypothetical
scenario prices distinguishable in stored data and reports.

Target valuation is a separate service shared by every hedge method.
The handoff designates the native smoothness-0 SANOS DLV in LV mode for
unquoted claim valuation. Its native geometry must not be replaced by the
new hedge-estimation grid. An unquoted claim's common reference mark is
not an observed market quote and must not be described as one.

The optimizer needs joint daily target and hedge gains, sensitivities, and
costs. It does **not** need a separate model entry premium to substitute
for the market premium. Hypothetical revaluation is needed to estimate
scenario gains and Greeks, not to change the price at which trading occurs.

### DLV validity

Unexpected zero, nonpositive, undefined, or unidentified DLV coordinates
must be investigated against the already fitted smooth 0.1 source.
They are not an accepted explanation for an unusable log-state vector.

Use the existing SANOS fitter and canonical DLV conversion/reconstruction
logic. Check source prices, expiry/strike conventions, boundary treatment,
precision, and reconstructed prices. Preserve raw diagnostics; do not
silently floor, cap, replace, or drop problematic coordinates to make PCA
run. Any unresolved fitting or reconstruction failure must be reported.

## 3. Rolling risk factors and hedge methods

Estimate **eight PCA factors** on the floating hedge-model state, including
the coordinates representing the front density. Compare ordinary PCA with
PCA orthogonal to spot as distinct variants. Do not confuse floating-state
changes with returns of a fixed traded contract.

Estimate a mean-reverting AR(1) on the reduced state, using the causal
252-transition window. Save loadings, orientation, explained variance,
AR coefficients, mean-reversion constraints, residuals, spot dependence,
and training dates. Align factors and compatible trainable weights with
the previous date without using future observations.

| Representation | Greek | Parametric | Statistical |
|---|---|---|---|
| DLV fitted to 0.1-SANOS prices | Spot and PCA price sensitivities | The same sensitivities weighted by spot/factor risk covariance | Joint repriced daily gains under the fitted dynamics |
| SANOS fitted to 0.1-SANOS prices | Spot and PCA price sensitivities | The same sensitivities weighted by spot/factor risk covariance | Joint repriced daily gains under the fitted dynamics |

- A Greek is a partial price derivative. A spot bump holds the remaining
  state fixed; a PCA bump changes the specified surface coordinates.
  Front-density exposures must not be set to zero merely because earlier
  code shocked only future DLV intervals.
- Greek matching uses today's price-sensitivity matrix. It is **not** a
  historical regression seeking dollar portfolios that reproduce latent
  PCA scores; the latter produced ill-conditioned cancelling positions.
  Tradable factor portfolios need not have zero cross-factor sensitivities
  or be orthogonal. Their joint sensitivity matrix must be recorded.
- Parametric hedging uses estimated factor dynamics to approximate joint
  P&L risk, with costs. Statistical hedging uses joint return scenarios
  directly, retaining nonlinear repricing.
- The additional historical-replay benchmark uses the exact target and
  hedge contracts' one-day historical gains from the 0.1 SANOS pricer.
- Unhedged and BS delta-plus-vega are baselines, not competing generators.
  BS vega uses the two surrounding eligible options, weighted by distance
  to the target strike and scaled to its vega; spot neutralizes the added
  option delta. Use the specified forward-volatility buckets for exotics.

For generated scenarios, the intended one-day spot marginal comes from
the current front-expiry distribution, coupled to surface shocks. Raw
historical spot resampling must not silently replace this marginal.
Spot-orthogonal PCA does not by itself calibrate the spot distribution.
Drift estimation for speculative alpha is out of scope; any drift removal
must be explicit and consistent with the chosen scenario model, without
silently deleting the estimated surface mean reversion.

## 4. Optimize daily marked P&L, including for multi-day claims

**We optimize daily, mark to market daily, and carry inventory.** A five-day
claim does not turn this experiment into a five-day neural control problem.
At every hedge date use that date's freshly estimated model, even inside
an episode that began several days earlier.

Let Z_t be the signed target position's common mark, H_t the vector of
hedge marks, a the holdings chosen for the next session, and a_prev the
surviving carried holdings. In consistent financed/normalized units, the
one-day gain being optimized is

\[
G_{t+1}(a)
= (Z_{t+1}-Z_t)
  +a^\top(H_{t+1}-H_t)
  -c_t(a-a_{\mathrm{prev}}).
\]

Include settlements and other contractual cash flows in the corresponding
gains. Holdings multiply instrument gains; **changes in holdings** determine
trading cost. Reconcile the daily accounting independently with realized
market observations, not with an in-sample simulated holdout.

Greek hedging matches exposures; parametric hedging approximates the risk
of this gain using sensitivities and factor statistics; statistical hedging
uses its scenario distribution. This follows the one-period distinction in
Hans Buehler's [Learning to Trade I: Greek, Parameter and Statistical
Hedging](http://hans.buehler.london/dl/Learning%20to%20Trade%20I%20-%20Greek%20Parameter%20and%20Statistical%20Hedging.pdf),
particularly the one-day setup and gains definition on slides 29--32 and
the historical-revaluation construction on slide 46.

## 5. Targets, instruments, and costs

- Hedge the agreed 1D, 3D, and 5D digitals, on both long and short sides.
  ATM is strike 1/NMS 0. Other inception strikes use NMS
  -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, and 1.
- The KO target has a single ATM barrier observation on day 3 and an
  underlying option expiring on day 5. Its strike uses the 2D NMS scale;
  an ITM call is down-and-out and the corresponding put is up-and-out.
- The forward-start target is the day-3 to day-5 ratio straddle
  abs(S_5/S_3 - K), using the same NMS set on the 2D scale.
- Start overlapping episodes every eligible business day after the
  estimation warm-up, not once every five days. Keep inception strikes,
  contractual reset/barrier rules, and each episode's accounting fixed.
- Hedge with daily spot and the agreed fixed co-terminal European
  contracts. The hedge-option eligibility screen is (-2, 2) NMS; do not
  confuse this tradable-book screen with the (-4, 3) estimation grid.
- Scenario methods retain the requested co-terminal zero-strike
  calls/forwards. Do not add them to simple Greek matching.
- **Spot trades at mid: zero spot cost.** Options trade at mid plus
  **gamma = 1 times the observed half-spread on turnover** in all methods.
  Preserve long/short side and position-normalization conventions.
- Carry positions between days. Expiry settles at intrinsic without an
  exit charge. At the episode's end or KO, mark surviving positions at mid
  with no forced exit cost. Report hypothetical exit cost separately.
- The requested premium-exposure limits were $10 per instrument and
  $100 per PCA-replicating portfolio. Do not revive the old arbitrary
  1,000-contract cap or conflate factor-portfolio budgets with all hedge
  constraints. Record the exact units and scope of every applicable cap.

## 6. Objectives, numerical standards, and reporting

The hedge2 comparison requested mean--variance at lambda = 1, 10, 100,
1000, and pure minimum variance; and power-exponential objectives at
lambda = 1, 10, 100, 1000. Optimize each side and configuration separately.
Greek frontiers re-solve at 100%, 80%, 60%, 40%, 20%, and 0% of the
full-Greek cost; scaling previously solved positions is not re-optimization.

Treat these daily portfolio problems as convex numerical problems to be
solved carefully. Use sound scaling, conditioning, stable objective
evaluation, and independent feasibility/optimality checks. Do not hide
failure by changing the economics, relaxing limits, or silently clipping
positions. Preserve exact solver warnings, including `Solution may be
inaccurate.`; that warning alone is not a reason to reject a usable solve.
Report genuine unresolved failures explicitly and seek quant review rather
than quietly dropping dates or inventing a fallback hedge.

Report every requested model/objective/aversion/side on matched dates:
realized costed mean and median P&L, standard deviation, MAE, tail risk,
trading cost, hypothetical exit cost, cap-hit frequency, and failure counts.
Keep training diagnostics separate from out-of-sample results. Account for
overlapping episodes in inference. No favorable-date or configuration
selection based on realized performance.

No new numerical results are asserted here. Before any authorized restart,
reconcile implementation and tests with these decisions, distinguish
valuation-state provenance from hedge-state provenance, and use new
uniquely identified external run directories rather than resuming
superseded financial specifications.

## 7. Implementation concerns to review

These are specific findings from the earlier handoff, code inspection, and
AI-agent quant reviews, not new approved modeling choices. Recheck each
against the active source before declaring it fixed.

| Area | Required correction or verification |
|---|---|
| Hedge-state source and geometry | `dlv.py` currently defaults to smoothness 0 and nine expiries; `state.py` rejects 0.1-derived states. Both conflict with section 2. |
| Common claim valuation | Claim-pricing paths use `DailyState.decode().pricer`, which is a regridded hedge state. Keep native reference valuation independent of that state and of hedge-model selection. |
| Canonical conversion | Use the existing SANOS fitter/conversion and verify native reconstruction. Do not use regularization, log floors, or caps to conceal invalid recovered states. |
| Front spot law | Episode, objective, snapshot, and resume paths must use the same explicitly identified front marginal and spot/surface coupling. No silent raw-spot fallback. |
| Optimization horizon | Replace the conflicting remaining-horizon neural-policy interpretation with the agreed daily convex MTM optimization. A multi-day target still requires daily repricing and decisions. |
| Resume/configuration identity | Earlier snapshot/resume paths omitted option side, forwards, premium-cap settings, or front-law configuration. Incompatible financial specifications must not share results. |
| Parametric evaluator | The handoff identifies inconsistent cached versus fresh return-tuple lengths in `parametric.py`. Verify and test both paths. |
| Failure coverage | Earlier aggregators omitted failed or unavailable episodes. Retain the requested population, explain every unresolved case, and do not invent replacement P&L. |
| Numerical warnings | Save exact warning text and assess the actual candidate. A solver warning is neither proof of failure nor proof of convergence. |
| Cash and financing | Independently verify carried inventory, turnover, settlements, forward/spot normalization, AM/PM expiry conventions, and zero forced exit costs. |

Additional review safeguards:

- Historical Gaussian scores are diagnostic observations, not automatically
  standard-normal simulation drivers. Empirical rank-copula and jointly
  Gaussian surface couplings are distinct model choices.
- Demeaning instrument gains alone does not establish a dynamically
  risk-neutral joint model. Do not equate a cash-accounting test with
  distributional calibration or absence of arbitrage.
- Large cancelling positions can arise from redundant spot/forward legs or
  ill-conditioned sensitivities. Inspect net exposures and conditioning;
  distinguish a genuine cap/budget infeasibility from a numerical failure.
- Per-basket premium budgets do not automatically bound the combined final
  hedge. Check constraints on the holdings that are actually executed.
- Historical test counts, cash residuals, saved checkpoints, and reported
  progress belong to their recorded code/data specifications. They do not
  validate the newly clarified experiment.

## 8. Proposed implementation and validation sequence

**For review only; not started by this consolidation.**

1. **Separate data roles.** Identify market mids/half-spreads, the stored
   0.1 hedge-calibration source, and the common native reference valuation.
   Add provenance and tests preventing accidental substitution between them.
2. **Validate daily representations.** Build both requested 13-by-200 hedge
   representations on a few dated calm/stress examples. Check canonical
   price reconstruction, positive finite state coordinates, source-grid
   mapping, mass/mean, and all boundary/extrapolation diagnostics. Investigate
   any failure before PCA or an all-date conversion.
3. **Validate rolling estimation.** Fit eight-factor PCA/AR(1) models with
   exactly 252 causal one-day transitions, ordinary and spot-orthogonal
   variants. Verify factor orientation, front-coordinate coverage,
   mean reversion, spot coupling, and previous-date initialization.
4. **Validate daily hedgers.** Give each of the six adapters standalone
   synthetic and small market-date tests. Check target/hedge joint gains,
   Greeks, costs, objective scaling, feasibility, and independent numerical
   optimality checks. Include cache and snapshot/resume tests.
5. **Validate actual episodes.** Hand-audit one example of each target and
   side, including a multi-day position carried through daily refits,
   barrier/reset events, expiry, and the final mid mark. Compare all methods
   on identical observed target cash flows and dates.
6. **Only then run rolling evaluation.** Start with a matched pilot and
   report all failures. Expand to the full eligible population only after
   the corrected pilot is accepted. Do not publish an ETA based on obsolete
   remaining-horizon training or native-grid assumptions.

CPU is the default for source loading, grid conversion, PCA/AR estimation,
and these small convex optimization tests. Use a managed GPU only for a
measured bottleneck; no GPU allocation is needed for this documentation
review. Keep each method's source and standalone tests identifiable.

Generated data, scenarios, logs, and checkpoints belong in uniquely named
runs beneath `/home/hans/quantdata_data/quant_data/hedge2/runs/`, not in Git.
Future material work must follow the host resource checks, detached-process
environment, checkpointing, and progress-logging rules. The source branch
for publishing this project is `hedging-benchmark`, not `main`; this
documentation consolidation does not commit, push, or switch branches.

## 9. Consolidated-document provenance

The four overlapping documents were removed from the active hedge2
directory: `README.md`, `PLAN.md`, `REVIEW_NOTES.md`, and
`CODEX_HANDOFF.md`. This file is now the only hedge2 Markdown review entry.
Unrelated projects' hedging documents are outside this consolidation.

Their originals, together with this file's pre-consolidation version, are
preserved at:

`/home/hans/quantdata_data/quant_data/hedge2/runs/documentation-consolidation-20261108-JeJFT6/pre-consolidation-markdown.tar.gz`

The adjacent `original-sha256.txt` records each original file's SHA-256.
The archive retains the historical run identifiers, numerical evidence,
AI-agent review commentary, and superseded instructions without presenting
them as current results. No source code, tests, data, or checkpoints were
changed or deleted by this consolidation.

## 10. Cleanup record (2026-11-09)

Removed 65,071 disposable files in 165 project-owned test/cache/runtime
directories, accounting for 2.878 GiB of allocated files. Removed another
18 empty legacy package directories after deleting their stale bytecode.
No active project file handles were found in the selected directories.
Temporary log files were archived before deletion.

The deletion pass verified 339 non-bytecode source/document files unchanged
by SHA-256. This subsequent cleanup note is the only repository content
edit. Licensed data, fitted states, model checkpoints, reusable scenarios,
audit evidence, explicitly retained large simulation arrays, and the
`hedging-benchmark` worktree remain intact. No hedge implementation or
training run was started.

The exact deleted paths, reasons, inventory, and preservation checks are in
`/home/hans/quantdata_data/quant_data/hedge2/runs/cleanup-20261109-rY2buO/summary.json`
and the adjacent `deleted.json` and `plan.json`.
