# FLEX Phase 2: fixed-weekly SANOS state and capacity result

## Status and scope

Phase 2 establishes a static, differentiable map from a 140-dimensional daily
surface state to a complete arbitrage-free SANOS option surface. The selected
configuration is the **Strong triangular 140-state model**, called
`triangular-stronger` by the code's benchmark selector and checkpoint stem;
its JSON report variant is `triangular_stronger_140`.

The selected expiry-state allocation is

```text
(16, 19, 17, 15, 13, 10, 10, 10, 5, 5, 5, 5, 5, 5).
```

This is a total of 140 state coordinates. The state block introduced at expiry
`j` can affect native parameter blocks `j,j+1,...,14`, but cannot affect an
earlier expiry. The final model is an affine decoder followed by sigmoid bounds
and the Torch SANOS recursion; it has no hidden decoder layers.

**Verified implementation fact:** Phase 2 does not contain a market-to-state
encoder, a time-series model, a train/validation split, or path simulation.
There is one freely learned 140-vector for every calibration date. The Phase 2
result is therefore an in-sample representation/capacity result, not a dynamic
forecasting result.

**Interpretation:** the Strong model is a suitable surface representation to
carry into the dynamics phase. Its fitted daily coordinates are not yet a
statistically identified state process: affine rotations and rescalings can be
absorbed between the daily coordinates and decoder weights.

## How the project arrived here

The progression supported by the current code and retained reports was:

1. The geometry was fixed at 14 weekly expiries and 200 canonical-NMS strikes
   per expiry. Provisional geometries using NMS5, 49 strikes, or a `(-5,2)`
   range were discarded.
2. Each date was first calibrated directly with all native SANOS coordinates
   free. This established the surface capacity before attempting compression.
3. The original surface-wide backbone multiplier was replaced by one
   multiplier per forward-volatility interval. This produced 14 logical
   expiry blocks of 202 values, or 2,828 native coordinates per date.
4. A global affine rank-140 model showed that a compact shared affine decoder
   could retain almost all of the useful quote fit.
5. A maturity-causal triangular decoder with ten coordinates at every expiry
   was materially worse. Its finalized continuation reached 68.4331% quote
   containment, compared with 70.5474% for Global 140.
6. Nearest-model-expiry quote diagnostics showed that the triangular shortfall
   was concentrated at the front of the surface. Uniform-minus-Global
   containment gaps, in percentage points, were approximately

   ```text
   expiry index:  0      1      2      3      4      5      6      7
   gap:          -2.99  -4.11  -3.47  -2.23  -1.71  -0.78  -0.45  -0.13

   expiry index:  8      9     10     11     12     13
   gap:          +0.37  +0.96  +1.22  +0.86  +0.57  +0.39
   ```

7. Raw-state PCA and reduced-rank diagnostics were useful as rejection tests
   but did not produce a reliable allocation. For example, idealized
   allocations `(20,108,1,...,1)` and `(30,98,1,...,1)` retained 99.25% and
   99.59% of decoded Global raw-state variation, yet achieved only 41.40% and
   43.48% quote containment when repriced. This demonstrated that Euclidean
   raw-logit variance is not option-price sensitivity.
8. Two hand-designed reallocations were then trained with the real quote
   objective:

   - Conservative:
     `(14,15,14,13,12,10,10,10,7,7,7,7,7,7)`;
   - Stronger:
     `(16,19,17,15,13,10,10,10,5,5,5,5,5,5)`.

   Stronger recovered essentially all of the Global-140 fit while preserving
   the triangular maturity structure and was selected for subsequent work.

The allocations in step 8 are **ad-hoc empirical choices**. They were motivated
by the quote diagnostics and not obtained from a validated discrete allocation
optimizer. In particular, their labels describe relative front-loading, not a
statistical confidence statement.

## Market data and calibration universe

The completed experiment uses SPX market dates from `2022-01-03` through
`2025-08-29`:

- 874 market days;
- 3,492,782 eligible call and put books;
- every retained date and quote is used in calibration;
- no validation holdout;
- no LP solution or LP-derived state is used for initialization or training.

The source pipeline is `quantdata/dysanos/market.py`:

1. `smooth_interpolated_market()` filters and normalizes one option market.
2. It fits a large smooth, statically arbitrage-free reference surface with
   `MkArbFree(smoothness=0.25, surface_expiry_linear=True)`.
3. FLEX reconstructs that reference with `ArbFreePricer.from_dict()` to obtain
   ATM volatilities and uses the already returned `af_call_mids` and
   parity-derived `af_put_mids` as the weak central `fit` targets.
4. Real call and put books remain separate observations. ITM and OTM options
   are both eligible. An observation must be active, European, have a finite AF
   reference, lie inside the exact expiry horizon, and lie inside its
   expiry-specific canonical-NMS strike range.

The upstream call and put prices are divided by discounted forward `DF*F`, and
strikes are divided by `F`. FLEX therefore receives dimensionless prices and
forward-relative strikes. Interest rates, discount factors, cash spot, and
forward levels do not subsequently enter the FLEX model. Put prices are
recovered from normalized call prices by

```text
P(t,K) = C(t,K) - 1 + K.
```

Calls and puts at one `(expiry,strike)` location are priced once as a call and
then restored as separate quote observations through parity. This is an exact
pricing optimization; it does not merge their bid/ask books or change their
objective weights.

### Eligibility geometry for market quotes

Only market expiries `0 <= t <= T[14]` are retained. At every market expiry,
the serialized AF reference supplies an interpolated ATM annual volatility.
FLEX converts the configured NMS endpoints `(-3,2)` to strikes using that ATM
curve and retains only market strikes in the resulting interval. The upstream
loader uses a deliberately non-binding NMS5 filter; the canonical-NMS test is
applied inside FLEX.

The 100,000-option upstream cap is intended to be non-binding. Market
smoothness `0.25`, the active-quote filters, and the underlying upstream WIDB
filters remain part of the data definition.

## Fixed model geometry

Let `DAYS_PER_YEAR` be the repository business-year convention. The model has
`M=14` expiry nodes:

```text
T[1] = 4.8 / DAYS_PER_YEAR,
T[j] = T[j-1] + 5 / DAYS_PER_YEAR,  j=2,...,14.
```

Thus the model business DTEs are

```text
4.8, 9.8, 14.8, ..., 69.8.
```

The first `4.8` rather than `5.0` represents the configured 20% overnight
count. There is no learned density node at time zero.

Each expiry has `N=200` inner NMS coordinates spanning `[-3,2]` with exactly
one zero. The current construction puts 120 coordinates below zero, the ATM
coordinate, and 79 above zero. It does not import strikes from neighboring
expiries.

Every expiry also has two boundary support points:

```text
K[j,0]   = 0.50 * K_inner[j,1],
K[j,201] = 1.50 * K_inner[j,200].
```

The density support therefore has 202 points although there are 200 learned
DLV-squared coordinates.

## Native state and bounded parameterization

The forward-mu native raw state is expiry-major. For expiry `j`, define

```text
D[j] = (a[j,1], ..., a[j,200], v[j], m[j]) in R^202.
```

With the logistic function `s(x)=1/(1+exp(-x))`, the bounded coordinates are

```text
DLVsq[j,i] = 0.01^2 + s(a[j,i]) * (3^2 - 0.01^2),
sigma[j]   = 0.01   + s(v[j])   * (1.00 - 0.01),
mu[j]      = 0.01   + s(m[j])   * (0.50 - 0.01).
```

There are therefore

```text
14 * (200 + 1 + 1) = 2,828
```

native raw coordinates per date.

The learned strike total-volatility curve is

```text
A[j] = sum(sigma[l] * sqrt(T[l]-T[l-1]), l=1,...,j),  T[0]:=0.
```

The learned Black-backbone total-volatility curve is

```text
B[j] = sum(mu[l] * sigma[l] * sqrt(T[l]-T[l-1]), l=1,...,j).
```

This cumulative square-root-time convention is exactly what the code uses. It
is not the usual square root of accumulated forward variance and must not be
silently reinterpreted as such.

## Strike construction

Let `epsilon=NMS.EPS`, let

```text
sigma_epsilon = A[1] / sqrt(T[1]),
q[j] = sqrt((sigma_epsilon^2 * epsilon + A[j]^2) / (1 + epsilon)),
```

and let `z[i]` denote the 200 fixed NMS coordinates. Inner forward-relative
strikes are

```text
K_inner[j,i] = exp(q[j] * z[i]).
```

The grid therefore changes with the date's learned forward-volatility state.
The selected model uses this per-expiry **NMS strikes** construction. The
experimental shared-strikes geometry is not the selected Phase 2 model.

## Torch SANOS density recursion

For one expiry, write the full support as
`K[0] < K[1] < ... < K[N] < K[N+1]`, with learned DLV values on inner indices
`1,...,N`. Before the first model step, the code represents the deterministic
time-zero law on the first support by taking discrete second differences of
the intrinsic call curve `(1-K)^+`.

At later steps, the preceding density is first represented on the current
support. If the preceding support and masses are `(X[r],p[r])`, the transferred
call curve is

```text
C_before(K[i]) = sum(p[r] * max(X[r]-K[i],0), r),
```

and discrete call slopes recover the transferred martingale masses
`p_before[i]`.

For model interval `j`, define for inner support index `i=1,...,N`

```text
box[i] = K[i]^2 * (T[j]-T[j-1]) / (K[i+1]-K[i-1]),
w_plus[i]  = DLVsq[j,i] * box[i] / (K[i+1]-K[i]),
w_minus[i] = DLVsq[j,i] * box[i] / (K[i]-K[i-1]).
```

The new masses `p_new` solve the tridiagonal SANOS system

```text
p_new[0] - w_minus[1] p_new[1] = p_before[0],

(1+w_minus[1]+w_plus[1]) p_new[1]
    - w_minus[2] p_new[2] = p_before[1],

-w_plus[i-1] p_new[i-1]
    + (1+w_minus[i]+w_plus[i]) p_new[i]
    - w_minus[i+1] p_new[i+1] = p_before[i],
    i=2,...,N-1,

-w_plus[N-1] p_new[N-1]
    + (1+w_minus[N]+w_plus[N]) p_new[N] = p_before[N],

-w_plus[N] p_new[N] + p_new[N+1] = p_before[N+1].
```

The implementation uses a differentiable Thomas solve rather than a dense
matrix. Debug validation checks finite, non-negative density, unit mass, and
unit forward-normalized mean.

## Option-price decoder and market-expiry interpolation

For support state `X`, strike `K`, and Black total volatility `b`, let
`BlackCall(X,K,b)` be the zero-carry Black call, with intrinsic value used when
`b=0`. A model-expiry call is the density mixture of these conditional calls.

For a market expiry `t` between model nodes `T[j]` and `T[j+1]`, let

```text
alpha = (t-T[j]) / (T[j+1]-T[j]),
b(t)^2 = (1-alpha) B[j]^2 + alpha B[j+1]^2.
```

The same interpolated backbone variance is used for both density mixtures:

```text
C(t,K) = (1-alpha)
           * sum(p[j,r]   * BlackCall(K[j,r],  K, b(t)), r)
       + alpha
           * sum(p[j+1,r] * BlackCall(K[j+1,r],K, b(t)), r).
```

For `0 <= t < T[1]`, set `alpha=t/T[1]`, use the deterministic time-zero
density as the lower density, the first learned density as the upper density,
and `b(t)^2=alpha*B[1]^2`. This converges to intrinsic value as `t` approaches
zero. Using the first learned density directly would instead imply a jump at
the origin.

Market expiries beyond `T[14]` are excluded rather than extrapolated.

## Direct and compressed architectures

### Direct 2,828

For each date `t`, `NativeStateSanosModel` stores an independent sparse
embedding row

```text
y[t] in R^2828.
```

There is no neural decoder. Sigmoid bounds transform `y[t]` directly into the
native coordinates, after which the SANOS decoder prices the surface.

Initialization comes from the smooth market ATM curve:

- forward volatilities are the differences of ATM total volatilities divided
  by `sqrt(delta T)` and then clipped to their bounds;
- every DLV-squared value at an expiry starts at that expiry's ATM annual
  variance;
- every forward `mu` starts at `0.20`.

### Global 140

Global 140 stores one independent daily state `h[t] in R^140` and one shared
affine decoder

```text
y[t] = b + W h[t],
W in R^(2828 x 140),  b in R^2828.
```

There are no hidden layers or activation functions in this map. Ordinary PCA
of the direct raw states initializes `b`, `W`, and `h`. The sign of every PCA
component is anchored by its largest absolute loading.

### Triangular 140

Partition the daily state into expiry blocks

```text
h[t] = (h[1,t], ..., h[14,t]),
h[j,t] in R^s[j].
```

The raw native block for expiry `j` is

```text
D[j,t] = b[j] + sum(W[j,l] h[l,t], l=1,...,j).
```

Equivalently, the affine weight matrix is block lower-triangular. The mask is
applied explicitly on every decoder call. Each of the 2,828 output coordinates
has its own bias. For Strong,

```text
s = (16,19,17,15,13,10,10,10,5,5,5,5,5,5).
```

The first state block can affect every expiry; the last can affect only the
last expiry.

For Strong, the cumulative input widths seen by native blocks 1 through 14
sum to 1,315. The effective mask therefore admits `202*1,315=265,630` affine
weights plus 2,828 output biases. The module stores a full `2,828x140` weight
tensor, but the remaining entries are multiplied by zero and receive no
quote-loss gradient. The separate per-date state table contains
`874*140=122,360` fitted values.

Triangular initialization is sequential conditional PCA. At expiry `j`, the
native block is centered, regressed on all previously constructed state
scores, and the requested leading singular modes of the residual become the
new state block `h[j]`. The fitted regression coefficients populate the links
from earlier states; the new residual loadings populate the diagonal block.
Component signs are anchored as in the global PCA.

The PCA initialization is analytic. Although reports retain a generic
`supervised_train_config` with 300 epochs and learning rate `0.002`, those
supervised epochs are **not run** for Global or Triangular affine PCA models.
Their supervised convergence record is `analytic_pca_solution` at epoch zero.

After initialization, both daily state embeddings and all allowed affine
weights and biases are fine-tuned jointly through SANOS against the real quote
objective. Consequently the final Strong states are no longer orthogonal PCA
scores, and raw explained variance no longer characterizes their final role.

## Objective and reporting

For real quote `i`, define

```text
half_spread[i] = (ask[i]-bid[i]) / 2,

excess[i] = (
    max(price[i]-ask[i],0) + max(bid[i]-price[i],0)
) / half_spread[i],

central[i] = abs(price[i]-market_fit[i]) / half_spread[i].
```

The trained scalar objective is the equally quote-weighted mean

```text
L = mean(excess[i] + 0.05 * central[i], i=1,...,Q).
```

Bid/ask is therefore the primary uncertainty interval. The smooth pre-SANOS
`market.py` fit is a weak central pull, not a replacement for the real book and
not the raw book midpoint. FLEX uses the observed positive half-spread without
the AF pricer's separate minimum-half-spread floor.

Reported containment is the percentage of quotes with exactly zero excess.
For quotes with positive excess, the report gives their conditional mean and
median; it also reports mean and median across the complete surface including
zeros. Whole-surface errors against `market.py` `fit` are reported both in
half-spreads and normalized-price units.

Because the objective includes the central term, its value is not the same as
the reported surface mean excess. It is also possible to reduce objective and
mean excess while moving a larger number of quotes only slightly outside their
spreads.

## Training and convergence protocol

The completed native calibration used:

- 300 epochs;
- learning rate `0.02`;
- 16 dates per batch;
- sparse Adam for daily native rows;
- zero weight decay;
- gradient-norm cap `10`;
- cosine learning-rate decay to approximately zero;
- seed `1729`, float32, and a compiled SANOS tridiagonal solve inside the
  optimizer steps on CUDA;
- eager SANOS repricing for final reported metrics.

The Strong and Conservative triangular fits used:

- analytic conditional-PCA initialization;
- 600 market-objective epochs from a cold start;
- learning rate `0.0002`;
- 16 dates per batch;
- sparse Adam for daily 140-state rows and dense Adam for the shared affine
  decoder;
- zero weight decay, gradient-norm cap `10`, seed `1729`, and float32;
- one 600-epoch cosine schedule;
- a compiled SANOS tridiagonal solve inside the optimizer steps and eager
  final repricing.

An iterative result is accepted only if its final objective is within 0.1% of
both the value 25 epochs earlier and the best recorded epoch. Fine-tuning must
also not regress relative to its eager pre-training objective. Failure is a
hard error: the code does not silently substitute a prior or fallback model.

The completed Strong convergence audit reports a last-25-epoch relative
improvement of `2.726e-05`, a final gap to the best recorded epoch of
`7.043e-06`, and best epoch 589 of 600.

## Final capacity comparison

All rows below use the same 874 dates and 3,492,782 quotes. The Uniform row is
the finalized second-schedule continuation, not the earlier 63.4394% result.

| Model | Objective | Inside bid/ask | Positive mean excess | Positive median excess | Surface mean excess | Fit MAE (half-spreads) | Fit RMSE (normalized price) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct 2,828 | 1.339158 | 71.4044% | 4.36899 | 1.32237 | 1.24934 | 1.79640 | 0.000222382 |
| Global 140 | **1.251221** | 70.5474% | **3.95646** | **1.27946** | **1.16528** | **1.71877** | **0.000208677** |
| Triangular Uniform 140 | 1.447079 | 68.4331% | 4.28082 | 1.47553 | 1.35132 | 1.91515 | 0.000226456 |
| Triangular Conservative 140 | 1.318310 | 70.1150% | 4.11256 | 1.40240 | 1.22904 | 1.78542 | 0.000217026 |
| **Triangular Strong 140** | **1.260246** | **70.9779%** | **4.04483** | **1.35446** | **1.17390** | **1.72699** | **0.000210353** |

Strong improves containment by 2.545 percentage points over finalized Uniform
and by 0.863 points over Conservative. It is 0.430 points above Global on
containment, while its objective is 0.72% higher than Global. It is only 0.427
points below Direct containment, despite compressing 2,828 raw coordinates to
140 daily state coordinates.

Global and Strong both have lower objective and mean excess than Direct. This
does not prove that a compressed model has intrinsically more capacity than a
free direct calibration. It shows that the direct optimization is not a global
minimum and that shared cross-date structure can guide the optimizer to a
different solution.

### Initialization is not the final result

The analytic initialization statistics reinforce the difference between raw
variance and quote relevance:

| Model | Raw-state variance explained | Initial objective | Initial containment | Final objective | Final containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global 140 | 99.6495% | 2.67248 | 46.6518% | 1.25122 | 70.5474% |
| Uniform 140 | 99.2500% | 2.00555 | 59.5939% | 1.44708 | 68.4331% |
| Conservative 140 | 99.1168% | 1.32876 | 69.9030% | 1.31831 | 70.1150% |
| Strong 140 | 98.8018% | 3.43357 | 43.3095% | 1.26025 | 70.9779% |

Strong has the weakest raw conditional-PCA reconstruction and the worst
triangular PCA quote initialization in this table, yet the best fine-tuned
triangular quote result. Any future expiry-state allocation method should use
quote-aware sensitivity or a dynamics-aware criterion rather than raw-state
PCA variance alone.

## Short-end fitting issue

The first learned density is at 4.8 business days. All 0DTE, 1DTE, and 2DTE
quotes are therefore priced by the time-zero-to-first-density extrapolation
rather than by bracketing learned density nodes.

Strong's short-end statistics are:

| Bucket | Quotes | Inside bid/ask | Positive mean excess | Positive median excess | Surface mean excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0DTE | 2,430 | 25.8848% | 13.2011 | 11.7118 | 9.7840 |
| 1DTE | 77,236 | 22.7264% | 15.7325 | 8.9768 | 12.1571 |
| 2DTE | 102,320 | 31.0106% | 11.4926 | 7.3429 | 7.9287 |

For comparison, Direct 2,828 achieves 26.5844%, 20.3001%, and 29.7977%
containment in those buckets. The short-end problem is therefore already
present with all native parameters free; it is not primarily caused by the
140-state triangular compression. Strong improves 1DTE and 2DTE relative to
Direct but does not make the absolute fit good.

**Interpretation:** the concentration of large errors before the first model
node points to a geometry/interpolation limitation. Adding short model nodes,
changing the left-end dynamics, or weighting the short end separately are
plausible remedies. This evidence does not by itself distinguish those causes
from short-dated quote microstructure, narrower spreads, settlement timing, or
the equal-per-quote loss. The `0DTE`/`1DTE`/`2DTE` labels use the floor of the
repository's fractional business-DTE clock, so AM settlement fractions are
included in their nominal day buckets.

## Ad-hoc choices and current limitations

The following are design or hyperparameter choices, not conclusions learned
from an out-of-sample model-selection exercise:

- NMS range `(-3,2)`, 200 inner strikes, boundary multipliers `0.5` and `1.5`;
- 14 model expiries, first expiry 4.8 business days, and five-day spacing;
- DLV-squared bounds `[0.01^2,3^2]`, forward-vol bounds `[0.01,1]`, and
  forward-mu bounds `[0.01,0.5]`;
- the cumulative `sum(sigma*sqrt(delta T))` total-volatility convention;
- market reference smoothness `0.25` and the non-binding 100,000-option cap;
- central-fit weight `0.05` and equal weight per quote;
- direct initialization with ATM annual variance and `mu=0.20`;
- total compressed dimension 140;
- the Strong per-expiry allocation itself;
- Adam/SparseAdam, learning rates, batch sizes, gradient cap, cosine schedule,
  epoch counts, float32 precision, and seed;
- the 25-epoch/0.1% convergence rule;
- no validation holdout in the static capacity experiment.

The optimizer schedules are not perfectly controlled across every table row.
Global used one 300-epoch cosine schedule. Finalized Uniform used a 300-epoch
fit followed by an explicit weights-only 300-epoch continuation that reset the
optimizer, scheduler, RNG, epoch, and history. Conservative and Strong each
used one 600-epoch cosine schedule. The final table is an accurate comparison
of completed artifacts, but small differences cannot be attributed solely to
architecture or allocation.

The learned native coordinates are mostly away from their sigmoid extremes,
but not entirely. The direct-state audit found 0.0267% of DLV probabilities
near the lower bound and 0.0572% of forward-vol probabilities near the lower
bound, using a probability threshold of `1e-4`; none were near their upper
bounds, and no forward-mu probabilities were near either bound. This does not
establish that the bounds are economically optimal.

Finally, the smooth AF `fit` is itself a model-derived reference. Bid/ask is
the uncertainty interval, while `fit` only resolves locations weakly through
the 0.05 central term. Phase 2 does not claim that its risk-neutral SANOS
density is a physical spot-transition law.

## Verified artifacts

The principal sources used for this summary are:

- `config.py`, `market.py`, `model.py`, `sanos.py`, `training.py`, and
  `run_production_gate.py` in this directory;
- `tests/test_flex_direct_state.py`, `tests/test_flex_forward_mu.py`, and
  `tests/test_flex_state_allocation.py`;
- `reports/batch-descent/spx-2022-2025-forward-mu-v1.json` for Direct and
  Global;
- `reports/batch-descent/continuations/`
  `spx-2022-2025-forward-mu-v1-triangular-schedule2/`
  `spx-2022-2025-forward-mu-v1.json` for finalized Uniform;
- `reports/batch-descent/rebalanced/`
  `spx-2022-2025-forward-mu-v1/`
  `spx-2022-2025-forward-mu-v1.json` for Conservative;
- `reports/batch-descent/rebalanced-stronger/`
  `spx-2022-2025-forward-mu-v1/`
  `spx-2022-2025-forward-mu-v1.json` for Strong;
- `/media/storage/quantdata/training/flex-fixed-weekly/`
  `spx-2022-2025-forward-mu-v1-stronger-parallel/`
  `triangular-stronger-140.pt` for the selected Strong checkpoint;
- the retained state-allocation and nearest-grid quote diagnostics under
  `reports/batch-descent/`.

Only the selected `spx-2022-2025-forward-mu-v1-stronger-parallel` native and
Strong checkpoints are retained. The superseded fixed-mu, shared-strikes,
latent-capacity-sweep, Uniform, Conservative, and Global checkpoint directories
were deleted after selection; their compact reports remain as research
provenance. Filtered market caches were not touched.

The completed Strong checkpoint is bound to 874 ordered market fingerprints.
Its reported latent-state shape is `[874,140]`; the report records checkpoint
and state SHA-256 digests so subsequent work can verify that it uses the exact
selected artifact. The selected checkpoint's SHA-256 is
`367fe85e67b0a09719c74d72dca5ab5fbfb16564e3c5828b13e482a3cfbb638a`,
which matches the Strong report's checkpoint proof.
