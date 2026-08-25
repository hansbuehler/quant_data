# FLEX: fixed-weekly SANOS market model

FLEX is a new Torch-native SANOS code path under `quantdata/dysanos/flex`.
It may use `quantdata/dysanos/market.py` and the common SANOS mathematics, but
it does not depend on the old `quantdata/dysanos/grid` model implementation.

This file is the authoritative specification. Earlier FLEX experiments using
NMS5, 49 strikes, a `(-5,2)` range, or one model-density node per observed
market expiry are provisional and are not evidence for this project.

## Canonical state notation

For expiry `j` on trading date `t`, let `N_x=N+1+1=202` and use:

- `x_t[j] in X subset R^N_x` for the constrained SANOS parameters: 200
  DLV-squared values, one forward/backbone-volatility coordinate, and one
  forward-`mu` coordinate;
- `u_t[j] in R^N_x` for the unconstrained full-support coordinates before the
  exponential/sigmoid parameter transforms, so `x_t[j]=Transform(u_t[j])`;
  and
- `h_t[j] in R^N_h[j]` for the compressed state on which weekly AR(1) dynamics
  and later measure changes are built.

A hierarchical decoder is expiry triangular,

```text
u_t[j] = Decoder_j(h_t[1], ..., h_t[j]).
```

A **direct** decoder is not constrained to have this triangular structure. In
**full** mode, `N_h[j]=N_x` and the decoder is the identity, so `h_t[j]=u_t[j]`.
The words `full`, `direct`, and `hierarchical` must be used only with these
meanings. PCA/standardized coordinates used internally to fit an AR(1) are
coordinate representations of `h`; they are not additional model states.

## Fixed geometry

Use the canonical `quantdata.biz_calendar.NMS`, whose epsilon is approximately
one trading hour. Do not use `NMS5`.

- NMS range: `(-3, 2)`.
- Inner strikes per model expiry: `N = 200`, including exactly one ATM node.
- Model expiries: `M = 14` weekly nodes.
- First model expiry:

  ```text
  T[0] = (5 - 0.2) / DAYS_PER_YEAR = 4.8 / DAYS_PER_YEAR
  ```

- Later model expiries:

  ```text
  T[j] = T[j-1] + 5 / DAYS_PER_YEAR,  j = 1,...,13
  ```

Thus the last node is `69.8 / DAYS_PER_YEAR`, just beyond three calendar
months. Time zero is not a learned density node.

For one day, bounded forward volatilities `sigma[j]` define the learned total
volatility coordinates

```text
V[j] = sum(sigma[l] * sqrt(T[l] - T[l-1]), l <= j),  T[-1] := 0,
0.01 <= sigma[j] <= 1.
```

This cumulative-square-root-time definition is intentional; it must not be
silently replaced by square-root accumulated forward variance. Let
`sigma_epsilon = V[0] / sqrt(T[0])`. For canonical NMS epsilon `epsilon`, the
inner normalized strikes are

```text
K[j,i] = exp(z[i] * sqrt(
    (sigma_epsilon^2 * epsilon + V[j]^2) / (epsilon + 1)
)),
```

where the 200 fixed `z[i]` span `[-3,2]` and contain exactly one zero. Every
model expiry has two additional density-support strikes:

```text
K_outer_left[j]  = 0.50 * K[j,0]
K_outer_right[j] = 1.50 * K[j,-1].
```

No strikes from another expiry are inserted into this fixed grid. This original
geometry is named **NMS strikes** in comparative experiments and remains a live
variant.

### Shared strikes: live parallel variant

The **shared strikes** variant uses the live forward-`mu`,
capped-unbounded-DLV model; it is distinct from the retired scalar-`eta`
experiment. For each market date, compute `N/2=100` canonical-NMS candidate
relative strikes using the learned strike-total volatility at `T[0]` and
another 100 at `T[M-1]`. Merge and sort their log-relative strikes, retain
exactly one ATM node, and impose a minimum adjacent log-strike distance of 5%
of the mean unspaced gap. The resulting 200 inner relative strikes, plus the
usual 50% and 150% boundary strikes, are used at all 14 model expiries for that
date.

Thus the support is homogeneous by expiry within a date but floats between
dates: another date's learned front and final strike-total volatilities usually
generate another support. Market eligibility remains the per-market-expiry
canonical-NMS `[-3,2]` filter. NMS strikes and shared strikes have the same
number of DLV, forward-volatility, and forward-`mu` coordinates.

There is an important distinction between raw-block and price hierarchy in
this geometry. The current affine mask makes raw block `u[j]` depend only on
`h[1],...,h[j]`, but the common support uses the terminal cumulative strike
volatility. Consequently a later `h` block can move the support and therefore
change an earlier-expiry option price. The current shared-strikes model is
block-triangular in raw coordinates but is **not an exactly price-triangular
decoder**. This is small empirically but structurally decisive for a sequential
target-measure proof: a later line can undo an earlier martingale constraint.

The planned exact factorization introduces a root geometry state `g` before
the expiry blocks,

```text
C[j] = Decoder_j(g, h[1], ..., h[j]).
```

`g` owns the common grid scales and any revealed-node grid recentering. It is
sampled with the first line and then held fixed while later lines are solved.
Acceptance requires an automatic Jacobian test that `d C[j] / d h[k]` is zero
for every `k>j` with `g` fixed. Until that refactor passes, the exact native
SANOS martingale roll is the P-free raw-surface benchmark, projection into the
current shared-NMS decoder is explicitly approximate, and an exact sequential
Q claim is not permitted. A direct decoder may instead use one joint Q kernel
with every expiry restriction enforced simultaneously.

The shared-strikes capacity experiment has four ordered stages:

1. an exact density/slack LP with per-day density/DLV coordinates;
2. direct Torch calibration of all 2,828 unconstrained `u` coordinates per day;
3. an affine PCA-style 200-state decoder into the native SANOS state; and
4. an expiry-causal affine decoder with the scaled Strong allocation
   `(23,27,25,22,19,14,14,14,7,7,7,7,7,7)`.

Stages 1 and 2 first compare NMS and shared strikes on 2024-03-05 and
2025-08-29. Stages 3 and 4 proceed only after that geometry audit, use the
shared-strikes full `u` states as their supervised target, and then fine-tune on
the same bid/ask-plus-weak-fit objective. The 200-state allocation is an ad-hoc
proportional scaling of the selected 140-state Strong allocation and must be
reported as such rather than treated as a structural optimum.

The next shared-strikes gate is a **full native market encoder**. Its unordered
quote input is summarized on the established 14-by-11 Gaussian expiry/NMS RBF
anchors and passed through a two-hidden-layer width-512 SiLU network. The
network outputs all 2,828 standardized unconstrained `u` coordinates directly;
this is full mode and its `h -> u` decoder is the identity. Fit
the quote normalizers, native-state supervision, and quote-objective
fine-tuning using dates no later than 2024-12-31. Dates from 2025-01-02 through
2025-08-29 are frozen temporal validation. The direct per-date 2025 native
states are an oracle geometry ceiling and state-error diagnostic only and must
not enter fitting or normalization. Give every date total objective weight one
regardless of its number of eligible quotes.

If that encoder is converged and non-degenerate, compare a flat full-capacity
state construction with an expiry-causal hierarchical full-capacity
construction, followed by their corresponding learned state decoders. Each
accepted state history then receives its own weekly physical-measure dynamics
fit. The base surface generator is the stable matrix-exponential Gaussian
AR(1) already specified for Phase 3: estimate the complete non-diagonal
mean-reversion generator, then precompute the exact one-week transition,
affine shift, innovation covariance, and covariance factor outside the Torch
simulation loop. The shared-strikes simulator defaults to the front-week SANOS
inverse-CDF spot mode; ordinary Gaussian spot sampling remains an explicit
diagnostic mode.

Acceptance requires comparison of fitted mean-reversion speeds with empirical
weekly decay, transition likelihood and covariance diagnostics, and bounded
long-horizon simulation tests. Do not repair instability by clipping simulated
states. After the historical-`P` Gaussian reference passes these gates, the
one-week transition interface may be extended first to an alternative Gaussian
law that can change both conditional mean and covariance, and then to a generic
learnable generator. Both changes operate on the assembled exact weekly law,
not on daily/PCA increments, and generate directly without path weights. Their
learning objective must be identified explicitly before they are called a
measure change.

### Sequential target-measure roll construction

The target-measure construction follows the weekly maturity roll. Let
`R[t+1]=S[t+1]/S[t]`. Sampling `R[t+1]` from the current front-week SANOS
distribution enforces

```text
c_t[1](k) = E_Q[(R[t+1] - k)^+ | F_t].
```

For the same cash strike `K=k*S[t]`, the current two-week option must satisfy

```text
c_t[2](k) = E_Q[
    R[t+1] * c_{t+1}[1](k/R[t+1])
    | F_t
].
```

The factor `R[t+1]` converts the next normalized option price back to current
spot units, while `k/R[t+1]=K/S[t+1]` keeps the cash strike fixed. In cash
prices, with the project's zero-rate convention, this is the undiscounted
martingale equality between the current two-week call and the rolled
front-week call. A discount factor must be added only if rates are restored.

There is a one-expiry shift in the sequential kernel. To keep that shift
unambiguous, write `h_hat[t+1,j->j-1]` for the compressed state generated by
transition line `j`: `j` identifies its originating current maturity, while
`j-1` identifies the remaining tenor at `t+1`. It is subsequently stored as
`h[t+1,j-1]`; it is not the newly created `j`-week block at `t+1`.

First fit the hierarchical AR(1) under `P`. At target-measure simulation time,
use its conditional draw as the reference proposal, but sample `R[t+1]`
through the current front SANOS inverse CDF. Then choose a candidate-`Q`
kernel for `h_hat[t+1,2->1]`, conditional on the realized lower/common factor,
`R[t+1]`, and the relevant current state including `h[t,2]`, so that the
two-week identity above holds. Decode this draw through
`h -> u -> x -> SANOS` and store it as `h[t+1,1]`. Thus the changed state is
the state used to price the rolled front-week option.

Continue along the roll cone: current `j`-week options constrain the target
kernel for `h_hat[t+1,j->j-1]`, conditional on the lower realized roll lines,
spot, and the relevant current prefix. Later draws must not feed back into an
earlier decoded expiry. The newly created longest-tenor block has no
one-week-longer current option inside the fixed `M`-expiry surface to identify
it; retain a regularized reference law unless a longer current maturity is
included explicitly.

This iterative calibration requires a hierarchical decoder. If the decoder is
direct, a later `h` component can change an option moment matched at an earlier
expiry. Then learn one joint target kernel for `(R[t+1],h[t+1])` and impose all
expiry roll identities simultaneously. Although that joint kernel can be
factorized into conditional densities for sampling, the components must be
fitted together and do not constitute the iterative construction.

#### Dimension and feasibility

Counting option constraints against stochastic state dimension is a useful
first diagnostic for a restricted local measure change, but is not a general
existence criterion. For a mean-only Gaussian adjustment with `d` independent
controls, local exact matching of `n` genuinely independent option moments
generically requires `n <= d` and a rank-`n` moment Jacobian. Allowing a full
Gaussian covariance adds up to `d*(d+1)/2` covariance controls, before applying
the hierarchical sparsity restrictions. Neither count guarantees feasibility:
the target prices must still lie in the attainable image and the sensitivity
matrix must have sufficient rank.

Conversely, `n > d` does not make a generic measure change impossible. A
conditional distribution or invertible transport of even low-dimensional
noise has functional degrees of freedom and can satisfy many finite moment
conditions. Prices across strikes are also strongly linked by monotonicity,
convexity, and the common terminal distribution, so quote count overstates the
number of independent restrictions. For every expiry layer report:

- the number of strike moments and kernel parameters;
- singular values and effective rank of the option-moment Jacobian;
- the irreducible roll-residual after optimization;
- covariance rank and conditioning for the normal kernel; and
- positivity/equivalence diagnostics for the generic transport.

Use the normal kernel as the controlled finite-dimensional feasibility test.
Escalate to the generic triangular transport when the normal kernel has a
material residual; do not infer impossibility solely from `number of
options > stochastic dimension`.

#### Research gates

Proceed in this order:

1. **Full validation control.** Set `N_h[j]=N_x` and use the identity decoder
   `h[j]=u[j]`. Train the market-to-state map through 2024 and report its 2025
   quote validation. This need not be perfect for structural research, but it
   must be non-degenerate and establishes whether compression is being blamed
   for an encoder/generalization failure.
2. **Compressed decoder comparison.** With `sum_j N_h[j]=200`, compare a direct
   decoder with an expiry-triangular decoder against the full control, first in
   unconstrained-state reconstruction and then in quote space. For the
   hierarchical candidate allocate more coordinates to early expiries, using
   the prior Strong experiments as initialization rather than treating that
   allocation as fixed truth.
3. **Weekly hierarchical dynamics and target kernels.** If the hierarchical
   decoder retains acceptable validation performance, fit its fixed-tenor
   weekly AR(1) under `P`. Sample spot from the current front implied
   distribution, then fit both a normal target kernel that may change the
   exact weekly conditional mean and covariance and a generalized conditional
   transport. Enforce the fixed-cash-strike roll identities expiry by expiry.

For gate 2, distinguish the representation ceiling from live validation.
Optimizing a fresh `h` for each held-out surface while freezing the decoder is
an oracle capacity diagnostic, not out-of-sample inference. The actual 2025
validation freezes both a train-through-2024 decoder and its market-to-`h`
encoder. Report both so decoder capacity and market-encoder generalization are
not conflated.

The surface buckets have floating calendar expiries but fixed weekly tenors.
The instruments in each roll restriction retain their cash strike and calendar
expiry: a current two-week option is compared with the same contract priced as
a front-week option one trading step later. Satisfying the conditional
martingale equations for the finite traded grid establishes dynamic
no-arbitrage for that instrument set under the fitted one-week kernel, subject
to integrability and kernel-equivalence checks. It does not by itself prove
the restriction for every strike in the continuum.

There are no observed `Q` transition samples. Fit the target kernels with
state-conditional option-martingale losses, using current-state instruments or
state bins rather than only unconditional average residuals, and regularize
them toward the fitted `P` kernel. Score historical likelihood only for `P`;
score `Q` by roll residuals, static SANOS validity, support/equivalence,
integrability, and long-horizon simulation stability.

## Market observations

`market.py` supplies every active, filtered, normalized European call and put
book separately, including ITM options. Each observation contains:

- observed expiry and normalized strike;
- call/put side;
- real bid and ask;
- the smooth pre-SANOS `fit` price returned by `market.py`.

Keep market expiries from 0DTE through `T[M-1]`. The upstream integer business
day filter may load through 70DTE, but FLEX applies the exact floating horizon
after loading, so an expiry beyond `69.8 / DAYS_PER_YEAR` is not eligible.

For every observed market expiry `t`, reconstruct the `market.py` arbitrage-free
pricer with `ArbFreePricer.from_dict()` and obtain its ATM annual volatility.
Let `A(t) = ATMVol(t) * sqrt(t)` be its ATM total volatility and use the ATM
annual volatility at `T[0]` for the NMS epsilon term. Retain only observed
strikes between the canonical-NMS strikes at `z=-3` and `z=2` computed from
that ATM curve. Apply this eligibility test per expiry before either call or
put books enter an objective.

Rates, discount factors, spot levels, and forward levels do not enter FLEX
after the normalization already performed by `market.py`.

## SANOS pricing between expiries

The density variables exist only at the 14 weekly model expiries. Market
expiries never become additional model nodes.

At a market expiry between two model nodes, use the Dysanos SANOS convention:

1. linearly interpolate the two density-conditional call prices in expiry;
2. linearly interpolate squared backbone total volatility in expiry;
3. evaluate both conditional calls with that same interpolated backbone
   variance before blending them.

For a market expiry `0 <= t < T[0]`, use the Dysanos left-extrapolation rule.
With `alpha=t/T[0]`, interpolate the first model density from the deterministic
time-zero density on the first support, and use backbone variance
`alpha * backbone_total_variance[0]`. At 0DTE this reduces to intrinsic value.

Market observations later than the last model expiry are excluded rather than
right-extrapolated.

## Prior scalar-eta parameters and backbone

For each trading day the native raw state `y_prime[t]` has exactly

```text
N*M + M + 1 = 200*14 + 14 + 1 = 2,815
```

values:

- `N*M` squared DLVs, one per inner strike and weekly model expiry, bounded by
  `0.01^2 <= DLVsq <= 3^2`;
- `M` forward volatilities used in the specified cumulative formula for
  `V[j]`, bounded by `0.01 <= sigma <= 1`;
- one day-level `eta`, shared across all expiries and bounded by
  `0.01 <= eta <= 0.5`.

The Black backbone total volatility at model expiry `j` is

```text
backbone_total_vol[j] = eta * V[j].
```

There is no separate Y term structure and no SSVI variant in this project
definition.

## Forward-mu expiry-state benchmark

The next benchmark replaces the single surface-wide `eta` with one learned
`mu[j]` on every forward-volatility interval. It retains the same 14 weekly
model expiries, 200 inner NMS strikes per expiry, quote universe, SANOS
recursion, and market objective.

Write the constrained SANOS state as 14 expiry-major blocks

```text
x[j] = (DLVsq[j,1:200], sigma[j], mu[j]).
```

Its unconstrained pre-transform block `u[j]` has `N+2=202` real coordinates,
and the full state has

```text
M*(N+2) = 14*202 = 2,828
```

coordinates per market date. All blocks are calibrated jointly against the
complete eligible surface; the block layout is logical and does not impose a
sequential optimizer.

The bounded strike forward volatility is

```text
sigma[j] = min_vol + sigmoid(x_vol[j]) * (max_vol - min_vol),
```

with `min_vol=0.01` and `max_vol=1`. The NMS strike grid uses cumulative total
volatility

```text
A[j] = sum(l<=j, sqrt(T[l]-T[l-1]) * sigma[l]).
```

The Black backbone instead uses

```text
B[j] = sum(l<=j, sqrt(T[l]-T[l-1]) * mu[l] * sigma[l]),
```

where every `mu[l]` is independently bounded in `[0.01,0.5]`. Positivity makes
both cumulative curves increasing. The separate names `A` and `B` avoid using
one symbol for the strike and backbone total-volatility curves.

Fit and compare three models on the identical direct checkpoint and quotes:

1. **Full 2,828:** one independent `u=h` state per date with identity decoder.
2. **Global 140:** an unrestricted rank-140 affine decoder from one daily
   surface state `h` to all 2,828 `u` coordinates.
3. **Triangular uniform 140:** ten state coordinates per expiry, with
   `u[j]` depending only on `h[1],...,h[j]`.

The global and triangular models use the same affine layer, daily state budget,
supervised raw-state target, quote objective, and fine-tuning schedule. The
triangular layer applies an exact block mask

```text
u[j] = b[j] + sum(l<=j, A[j,l] h[l]).
```

Global 140 versus Direct measures the compression cost. Triangular Uniform
versus Global measures the combined maturity-causal and equal-allocation cost.

Initialize Global 140 by ordinary PCA. Initialize Triangular Uniform 140 by
sequential conditional PCA: regress each `u[j]` on prior state blocks and take
PCA directions of the remaining residual for `h[j]`.

These states organize maturity dependence but do not yet impose time dynamics.
The later weekly physical-measure transition should model roll-down explicitly
and generate the new curve front-to-back conditional on the preceding curve.

## Objective and reporting

For option `i`, define its observed half-spread and excess by

```text
half_spread[i] = (ask[i] - bid[i]) / 2
excess[i] = (
    max(price[i] - ask[i], 0) + max(bid[i] - price[i], 0)
) / half_spread[i].
```

The fitting objective is linear excess plus a small normalized absolute pull
to the `market.py` fit:

```text
sum(excess[i] + lambda_fit * abs(price[i] - fit[i]) / h[i]).
```

The same quote weighting and objective semantics apply to the native and
compressed Torch stages. `lambda_fit` is small and explicit in reports. No LP
fit or LP-derived state is used by the capacity benchmark.

Call and put books remain separate observations with their own bid, ask, and
`fit`. For execution speed, a shared `(expiry, strike)` location is priced once
as a call and gathered back to both books before put-call parity and the
quote-wise loss are applied. This is an exact pricing deduplication, not a
merge or reweighting of market observations.

Every stage reports, over all eligible options:

- number of options;
- percentage with exactly zero excess, i.e. inside bid/ask;
- mean excess conditional on positive excess;
- median excess conditional on positive excess;
- mean and median excess over the whole quote surface, including the zero
  excess of quotes inside bid/ask.

If no observation has positive excess, the conditional mean and median are
reported as zero. Report the same fields separately for 0DTE, 1DTE, and 2DTE
options to expose any left-extrapolation problem. A bucket with no eligible
options is explicitly identified by a zero option count.

Every stage also reports whole-surface error against the smooth `market.py`
`fit` over all eligible options. Absolute error is summarized by its mean and
median in observed half-spreads, and by MAE, median absolute error, RMSE, and
signed bias in normalized option-price units.

## Two sequential capacity stages

The earlier LP benchmark remains useful as a pricing diagnostic, but its low
containment is not a stop condition and it is not part of these stages.

### 1. Native Torch batch descent

Give every market day an independent raw `y_prime[t]` vector with 2,815
values. Apply sigmoid bounds directly:

- raw DLV coordinates to `[0.01^2, 3^2]`;
- raw forward-vol coordinates to `[0.01, 1]`;
- raw eta to `[0.01, 0.5]`.

This has the same economic SANOS state and exact geometry that the LP fit
recovers, but not the LP's auxiliary linear density/slack variables. Torch
stores only the `N*M` DLVsq values, `M` forward vols, and one eta. The SANOS
recursion reconstructs the densities and option prices inside the loss.

Decode the bounded values with the Torch-native SANOS recursion and optimize
all daily vectors directly against their eligible quotes with batched gradient
descent. There is no lower-dimensional state and no feed-forward surface
decoder in this stage. Initialize the forward-vol curve and the strikewise DLV
level from the smooth market ATM curve, but do not use an LP solution.

Report the aggregate and 0DTE/1DTE/2DTE statistics plus whole-surface error.
The earlier LP containment ratio is reported nowhere and does not stop
training. The benchmark stops after writing this report; compression is not
started automatically.

In-progress checkpoints record the exact training phase, completed epoch,
optimizer and scheduler states, NumPy and Torch random-generator states, and
the accumulated history. Market-phase checkpoints retain only the pre-training
evaluation score needed to detect objective regression; they do not retain a
fallback model state. A restart therefore resumes the interrupted native,
supervised-compression, or market-fine-tuning phase without acquiring a silent
recovery path.
Checkpoints also bind the model configuration version and the ordered market
fingerprints; a different geometry, model identity, or quote universe is
rejected.

Every iterative phase must pass an explicit convergence gate before its output
can seed the next phase or be marked complete. By default, the final objective
must be within `0.1%` of both the objective 25 epochs earlier and the best
recorded objective. Market fine-tuning must additionally avoid any eager
objective regression relative to its pre-training candidate. Failure of
either condition leaves the checkpoint incomplete, writes a `*_failed` report,
and raises `TrainingConvergenceError`. No earlier state is substituted.

Exact continuation also requires the saved objective, epoch schedule, batch
size, optimizer settings, clipping, seed, precision, device, and
compiled-versus-eager SANOS solver choice to match. The explicit
`--warm-start` option is the distinct weights-only operation for changed
settings; it discards optimizer, scheduler, RNG, epoch, and history instead of
silently mixing configurations.

The resume signature also carries an explicit training-algorithm version.
Changes to batching, quote deduplication, or diagnostic reduction order reject
older in-progress checkpoints even when their user-facing train configuration
is unchanged.

Completed or legacy checkpoints cannot be retrained implicitly: callers must
choose a permitted read-only path or the explicit weights-only warm start. The
in-memory native fit also retains its original ordered market fingerprints and
dtype, so the reusable compression API rejects a same-length but different
market-day tuple or a dtype-changing target cast.

The production run started before this checkpoint schema was introduced and
writes a legacy schema-1 native checkpoint. It remains valid as a weights-only
warm start, but it cannot restore its optimizer, scheduler, RNG, or epoch. A
read-only load never rewrites or silently upgrades that artifact. Once that
run writes its `native_complete` report, the compression gate binds the
schema-1 file to a durable digest proof and loads it with `just_load`; no
migration or warm-start retraining is required for the compression handoff.
The same certification can be performed before review, without loading market
data or starting compression, using
`python -m quantdata.dysanos.flex.run_production_gate
--certify-native-only` with the original label, date range, output directory,
and checkpoint directory.
The durable record includes convergence over the saved epoch history plus
finite-state, coordinate-range, and sigmoid-bound saturation diagnostics for
DLVsq, forward volatility, and eta. These diagnostics do not reprice or alter
the checkpoint. Certification also writes `<label>-native-review.md`, which
combines these diagnostics, immutable checkpoint hashes, and all required
market-fit tables for independent review.

Native-mode `just_load` is not allowed to promote a legacy schema-1 snapshot,
because that format has no completion phase. Only the pre-existing
`native_complete` report written by the legacy process may be bound through
`--certify-native-only`. Report and checkpoint mutations hold the same logical
file lock for the complete run, and an existing durable certificate is removed
only by an explicit weights-only warm start.

### 2. Compressed-state decoding

After native-state training, fit a simple shared feed-forward map

```text
h[t] -> u[t]
```

for separate compressed-state dimensions 20, 100, 200, and 1,000. Keep the decoder
architecture and its 1,000-coordinate input width controlled across the
comparison: only the first requested number of coordinates is learned and the
remainder is fixed at zero. Warm-start by regressing to the fitted native raw
states without SANOS pricing, then fine-tune through the Torch SANOS objective
against the real market books.

As a separate controlled alternative, `--pca-decoder` replaces the nonlinear
feed-forward map by one affine layer

```text
u[t] = mean_u + loadings @ h[t].
```

An exact SVD of the centered full `u` states initializes `mean_u`, the first
`state_size` principal-component loadings, and the corresponding daily scores.
Each loading has a deterministic sign anchor. The PCA initialization is then
evaluated without changing any parameter before the affine layer and daily
coordinates are jointly fine-tuned through the same quote objective as the
nonlinear decoder. Thus the initial reconstruction is the optimal Euclidean
rank-`state_size` affine approximation, while the final subspace is allowed to
rotate toward low-variance directions that matter disproportionately for
option prices. The checkpoint records both the PCA initialization diagnostics
and the final eager market evaluation.

If State-200 retains more than a two-percent relative containment loss against
the native fit, also run a full-capacity State-2,815 decoder control. This
control uses 2,815 input coordinates and hidden width 2,815, so the shared
feed-forward decoder contains no 1,000-coordinate dimensional bottleneck. The
existing State-20/100/200/1,000 configurations and checkpoints retain their
controlled 1,000-coordinate input and hidden width. Comparing State-2,815 with
the full `u` state separates a dimensional-compression loss from a shared
decoder or optimization loss.

This is a separate invocation which requires the completed native checkpoint.
The runner requires both the checkpoint and a matching durable native-complete
report; an interim checkpoint written during native optimization cannot start
compression. The completion record contains the checkpoint SHA-256, native
state digest, model identity, and ordered-market fingerprint digest. It first
loads and evaluates that exact checkpoint without updating it. Therefore the
native representation used as the compression target is exactly the one whose
fit was reviewed after stage 1. Every latent checkpoint also stores that native
target identity and refuses to resume against a different native state tensor.

A restarted four-size compression run loads every already-complete latent
checkpoint read-only, resumes the first interrupted supervised or market phase
from its exact saved state, and then continues with the remaining dimensions.
An explicit `--warm-start` remains the only way to retrain a completed size.
Compression reloads only the sorted `market_dates` represented by the native
checkpoint—not every originally requested date. This both avoids retrying
known unavailable dates and prevents a formerly missing date from silently
changing the native supervision universe.

Each completed latent report records the checkpoint SHA-256, model identity,
latent-state tensor shape and SHA-256, ordered-market fingerprint digest, and
the native-state shape and digest used for supervision. This makes the reported
fit traceable to one immutable compressed-state artifact rather than only to
its training-history values.

The production CLI uses 16 market days per optimizer batch by default. This
keeps the shared decoder's dense updates efficient on GPU while preserving the
quote-weighted objective; larger batches must be benchmarked on GPU 0 before
changing the experiment configuration.

The production compression run compiles the differentiable tridiagonal SANOS
solve for optimizer steps. The actual final candidate and every reported price,
objective, containment statistic, and whole-surface error are recomputed with
the eager kernel. A worse or invalid candidate is a hard failure; there is no
fallback selection. This keeps reports exactly comparable with the eager
native benchmark while retaining the compiled training throughput. Switching
this flag deliberately invalidates exact optimizer-state continuation and
therefore requires an explicit weights-only warm start.

Keep the three production learning rates distinct. The native 2,815-state fit
used `0.02`; direct supervised reconstruction of the native raw states uses
`0.002`; and latent quote-objective fine-tuning uses `0.0002`. The final value
was selected from controlled state-20 pilots (`0.02`, `0.002`, and `0.0002`)
before the four-size comparison and is then held fixed for 20, 100, 200, and
1,000 states. The larger two rates moved the quote objective far above the
supervised starting point and are failures, whereas `0.0002` improved it
within five epochs.

Report the same aggregate and short-expiry fit statistics for each dimension.
This stage measures empirical compression of the fitted daily surfaces; it is
not yet a market-quote encoder and uses no validation holdout.

Treat this comparison as operational decoder capacity, not proof of an
intrinsic manifold dimension. The jointly learned daily coordinates have a
non-identifiable gauge: rotations, rescalings, and more general coordinate
changes can be absorbed by the shared decoder. Before using them as targets
for an encoder or a `P`-dynamics model, fix a reproducible gauge (at minimum
centering/whitening and sign/orientation anchoring) and assess weekly temporal
continuity. Those later constraints must not be retrofitted into this static
no-holdout capacity comparison.

## Measured capacity result

The production universe contains 874 market days and 3,492,782 eligible option
books. The completed eager repricing report is
[`reports/batch-descent/spx-2022-2025-native-v1.json`](reports/batch-descent/spx-2022-2025-native-v1.json).
The independent final audit is
[`reports/batch-descent/spx-2022-2025-native-v1-capacity-review.md`](reports/batch-descent/spx-2022-2025-native-v1-capacity-review.md).

| State | Objective | Inside bid/ask | Conditional positive mean | Conditional positive median | Surface mean | Fit MAE (half-spreads) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct 2,815 | 1.39764 | 70.6082% | 4.43982 | 1.29868 | 1.30495 | 1.85399 |
| Decoder 20 | 2.62038 | 46.3870% | 4.59399 | 2.44079 | 2.46298 | 3.14813 |
| Decoder 100 | 1.89647 | 52.1306% | 3.70715 | 1.81671 | 1.77459 | 2.43745 |
| Decoder 1,000 | 1.31155 | 63.0552% | 3.30455 | 1.18977 | 1.22086 | 1.81388 |
| Decoder 1,500 | 1.22589 | 66.9407% | 3.44946 | 1.17440 | 1.14037 | 1.71054 |
| Decoder 2,000 | 1.20399 | 67.9612% | 3.49501 | 1.17826 | 1.11976 | 1.68458 |
| Decoder 2,815 | 1.20633 | 67.6573% | 3.46832 | 1.16286 | 1.12175 | 1.69167 |

The no-bottleneck State-2,815 decoder improves the stated objective by 13.69%,
mean excess by 14.04%, and half-spread fit MAE by 8.75% relative to the direct
fit. This proves that the shared decoder can express a lower-loss solution when
given the full daily state dimension. It also shows that the original direct
batch-descent result is not a global optimization ceiling.

Containment gives a different conclusion. State-2,815 retains 95.82% of native
containment, a 4.18% relative loss, and therefore misses the specified 98%
retention gate. The continuous excess objective penalizes violation magnitude,
not the count of quotes with any positive excess. State-2,815 consequently
produces more small interval violations while reducing their mean magnitude.
It demonstrates objective capacity, but does not certify containment capacity
under the current optimization objective.

Short-expiry containment is:

| State | 0DTE | 1DTE | 2DTE |
| --- | ---: | ---: | ---: |
| Direct 2,815 | 26.9136% | 19.4534% | 29.1957% |
| Decoder 20 | 25.6790% | 23.0980% | 30.0166% |
| Decoder 100 | 25.2675% | 23.6625% | 31.9439% |
| Decoder 1,000 | 25.6790% | 23.9823% | 32.5665% |
| Decoder 1,500 | 25.5967% | 24.2167% | 32.8245% |
| Decoder 2,000 | 25.3909% | 24.4420% | 33.0727% |
| Decoder 2,815 | 25.4321% | 24.4795% | 33.0923% |

State-200 was stopped at its atomic epoch-60 checkpoint at the user's request;
it has no completed eager result and is excluded from the table. The subsequent
controlled search establishes a failed State-1,000 and passing States 1,500 and
2,000 when comparable containment is defined as no more than a two-percentage-
point loss relative to the State-2,815 decoder. State-1,500 loses 0.7166 points
and retains 98.94% of State-2,815 containment; State-1,000 loses 4.6021 points
and retains 93.20%. The tested passing boundary is therefore bracketed between
1,000 and 1,500; 1,500 is the smallest completed passing state tested so far.

All completed latent checkpoints are schema 2, phase `complete`, finite, and
bound to the same ordered 874-day market fingerprint digest and native raw
state digest. The controlled search is recorded in
[`reports/batch-descent/spx-2022-2025-latent-capacity-sweep.md`](reports/batch-descent/spx-2022-2025-latent-capacity-sweep.md),
including checkpoint hashes for States 1,000, 1,500, 2,000, and 2,815.

### Affine PCA decoder result

The simpler affine experiment uses the same 874 days, 3,492,782 option books,
objective, and 300 market epochs. Its complete eager report is
[`reports/batch-descent/spx-2022-2025-native-v1.md`](reports/batch-descent/spx-2022-2025-native-v1.md).

| Affine state | Raw variance retained | Initial objective | Initial inside bid/ask | Final objective | Final inside bid/ask | Conditional positive mean | Conditional positive median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 97.3053% | 4.07583 | 38.2820% | 1.72061 | 57.2481% | 3.76321 | 1.49433 |
| 100 | 99.2525% | 3.44384 | 41.3974% | 1.37684 | 66.3241% | 3.81213 | 1.22354 |

Rank 100 is the smallest affine state tested that satisfies the earlier
comparison against the full State-2,815 nonlinear decoder: its containment is
1.3332 percentage points lower and retains 98.03% of that decoder's 67.6573%.
It uses 284,315 shared affine parameters plus 100 daily coordinates, rather
than the full decoder's multi-layer 2,815-wide map. Rank 50 loses 10.4092
containment points against the full nonlinear decoder, so the tested boundary
is bracketed between 50 and 100 coordinates.

The rank-100 affine model's whole-surface normalized-price RMSE is `0.00022060`
and its half-spread MAE is `1.86144`. Its short-expiry containment is 26.2963%
for 0DTE, 21.1430% for 1DTE, and 30.4545% for 2DTE. These numbers remain poor
in absolute terms, but are comparable to or better than the corresponding
full-decoder values except at 0DTE.

The large gap between the untouched PCA and fine-tuned results is material.
Ordinary PCA preserves raw-state variance, not option-price sensitivity; small
raw-logit directions can matter greatly after the bounded transforms and SANOS
solve. The affine architecture is therefore useful because it supplies a
well-conditioned reproducible gauge, but quote-objective fine-tuning remains
essential. The next market-to-state or dynamics model should target the final
100-dimensional daily coordinates, not the untouched PCA scores.

## Later work

Only after these two static capacity stages should FLEX add the market-to-state
map and weekly physical-measure dynamics. The later model will use native
Torch path generation, five-business-day trading steps, and an explicit
choice of sampling measure. None of those components is implemented or
benchmarked by the static capacity code above.

The dynamic model is to be learned under the historical measure `P`. At each
five-business-day trading step it will roll a fixed set of weekly expiries,
simulate the lower state inside the Torch graph, decode that state through the
same SANOS surface, and expose the corresponding weekly option prices to a
trading loss. The market-to-state map should first learn the directly fitted
daily states and then be fine-tuned through the quote objective. Its input is
an unordered set of live option books; use a controlled set-invariant radial
basis or transformer-style architecture rather than a slow generic Deep Set.

Bid/ask intervals remain observation uncertainty during state inference. The
smooth pre-SANOS `fit` remains the weak central target inside those intervals;
raw book midpoints do not replace it.

Execution costs require a separate positive half-spread model. Model the
realized half-spread primarily in normalized option-price units, conditional
on NMS/moneyness, expiry, call/put side, model delta and vega, and available
volume/open-interest fields. This is more natural than making implied-volatility
spread the target: trading P&L pays a price spread directly, while converting a
volatility spread back to price is unstable when vega is small, especially for
short-dated and deep ITM/OTM options. A relative-price or log-spread transform
may stabilize scale, but it needs a positive price floor near zero; volatility
half-spread remains a useful diagnostic or feature rather than the execution
cost itself.

The SANOS density decoded from option prices is a risk-neutral conditional law,
even when the historical state dynamics are learned under `P`. Drawing the next
spot directly from the front-week SANOS density would therefore use that law as
a proposal, not by itself produce a physical spot transition. The dynamic stage
must make the sampling measure explicit—for example through a learned physical
tilt with likelihood weights—so the same Torch simulator can later revisit the
risk-neutral discussion by changing measure. Risk-neutral versus physical spot
sampling is deliberately not resolved by the two static stages above.

## Research history and artifact retention

FLEX remains in early structural iteration. Any number of active candidates
may run when they answer useful, distinct research questions. Do not terminate
or delete a candidate merely because it appears redundant or out of scope:
present the reason and ask the user first. Once the user discards a model or
idea, delete its checkpoints, transient logs, intermediate failure files,
generated diagnostics, and implementation branches that no active model needs.
Filtered market data and shared source-market caches are never part of this
cleanup. Near a strong model, retain useful close variants for controlled
comparison rather than cleaning them up automatically.

Every guarded FLEX activity measures Linux `MemAvailable`, which includes
immediately reclaimable page cache. It emits a warning below 200 GiB and
terminates at or below 100 GiB, recording the memory-triggered termination in
the activity's failure directory. Literal `MemFree` is not used because it
would classify useful reclaimable cache as unavailable memory. These checks run
at startup and every 30 seconds thereafter.

Do not preserve large obsolete artifacts merely for reproducibility. Instead,
retain a compact milestone report in this history, including the specification,
headline train/OOS results, rejection or selection reason, and hashes of any
selected artifact that remains. The detailed Phase 2 history is in
`PHASE_2_SUMMARY.md`; Phase 3 implementation and estimator decisions are in
`PHASE_3_WEEKLY_CODE.md` and `PHASE_3_DYNAMICS_ESTIMATOR_COMPARISON.md`.

Current milestones are:

- The native 2,828-state calibration established the static SANOS fit ceiling.
- The selected Strong triangular 140-state decoder established the by-expiry
  capacity allocation; superseded compression, fixed-mu, and the obsolete
  scalar-eta shared-strike checkpoints have been deleted while their measured
  conclusions remain in the Phase 2 report. The forward-mu shared-strikes
  geometry above is a separate live experiment.
- The DYSANOS/grid lag-only matrix-exponential hierarchy replaced the rejected
  contemporaneous and exact-weekly-log dynamics estimators. Their rationale and
  comparison statistics remain in the dynamics-estimator report.
- The retired Global-140 surface-decoder control reached 25.98% all-option and
  24.11% weekly 2025 containment. That was only 0.83 and 0.67 percentage points
  above the Strong decoder with its global market encoder, so changing the
  surface-decoder topology did not resolve OOS deterioration. Its large
  checkpoints and experiment branch were removed after this conclusion.
- The global market-to-Strong encoder is the incumbent OOS control. The
  expiry-causal hierarchical market-to-Strong encoder remains a live variant;
  both checkpoints and implementations are retained.
- A capped-unbounded native-DLV branch now represents raw DLV² as
  `0.01^2 + exp(z)`, caps it mechanically at `3^2` for pricing, and uses a
  separate smooth log-coordinate cap penalty.  Legacy Strong checkpoints keep
  their original sigmoid gauge.  On frozen 2025, the flat market-to-native
  2,828 and flat affine-140 controls reached 28.28%/26.20% and 26.23%/24.41%
  all/weekly containment, respectively, versus a 69.17%/83.16% independently
  calibrated direct-state ceiling.  The small compression loss and large
  ceiling gap identify market-to-state temporal generalization as the dominant
  problem.  Both encoders also extrapolate above the DLV pricing cap OOS
  despite zero training cap contacts; this is a live diagnostic concern.
- The linear and histogram-gradient-boosting spread models remain historical
  attribution benchmarks only. Their fitted objects were removed; the
  differentiable Torch feed-forward spread model is the simulation candidate.
