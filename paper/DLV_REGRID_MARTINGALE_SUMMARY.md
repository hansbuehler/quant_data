# X-only finite-grid DLV martingale simulation and weekly re-gridding

**Research summary — revised 11 September 2026**

## Executive conclusion

The density-first finite-step construction gives an exact risk-neutral law for
the modeled finite DLV market. On a fixed cash support, every branch is a valid
SANOS surface and the probability-weighted branch density equals the starting
density. After the front spot node is revealed, dividing the inherited support
by that node is an exact change of coordinates. It therefore preserves every
surviving fixed-cash option price, path by path, before Monte Carlo averaging.

The implementation does **not** implode at a finite-grid boundary. If spot
selects the lowest or highest support node, however, the conditional DLV law
collapses to that node and becomes absorbing. At that point the DLV drift
condition is vacuous: DLV coordinates may move numerically, but they no longer
change the conditional density or option prices. This is a finite-support
modeling limitation, not a numerical failure.

The new four-week audit is fully **X-only**: its Black backbone is exactly zero.
It used all **891** certified historical states, 100,000 paths (50,000
independent antithetic pairs), 11 strikes per expiry, four weekly steps, fp64,
and a 10% DLV log-volatility scale. The still-surviving original fixed-cash
book plus spot had covariance-aware p-values 0.139, 0.406, 0.439, and 0.357.
Direct fixed-cash-grid and floating re-grid valuations agreed pathwise to at
most `2.22e-16` in the transition decomposition and `3.33e-16` in the initial
re-grid check.

## 1. Notation and probability law

The construction uses two different node indices and two different layers of
future density. They must not be conflated. All formulas below condition on one
fixed current state; its state/batch index is suppressed.

| Symbol or index | Meaning |
| --- | --- |
| `F_t`, `P_t`, `E_t` | Current information and risk-neutral probability/expectation conditional on that information. |
| `t`, `h` | Current calendar time and one calendar step; here `h = 1W`. |
| `T_m = t + m h` | The `m`-th option expiry. The expiry index is `m = 1,...,M`; here `M = 4`. |
| `S_t = s` | Current spot of the X-only process. The independent Black factor is disabled, so `Y = 1` and `S = X` in this experiment. |
| `N_x` | Number of nodes in the finite spot support. |
| `j = 0,...,N_x-1` | Terminal support-node index. `x_j` is a dimensionless spot multiplier and `s x_j` is its cash spot value. |
| `i = 0,...,N_x-1` | Realized **front-expiry** node index. At `T_1 = t+h`, drawing `i` sets `S_(t+h) = s x_i`. Although `i` and `j` range over the same grid, `i` labels the observed 1W node whereas `j` labels a possible node at a surviving later expiry. |
| `q_m(j)` | Current risk-neutral probability mass of node `j` at expiry `T_m`, conditional on current information `F_t`. It obeys `sum_j q_m(j)=1` and `sum_j q_m(j)x_j=1`. “Density” in this report means this discrete probability mass. |
| `f = 1,...,F` | DLV shock-factor index. The audit uses `F = 4` factors. |
| `DLV`, `DLVsq` | Discrete local volatility and its square. DLVsq is the local-variance coordinate used by the SANOS recursion on each interior support node and expiry. |
| `b = 1,...,2^F` | Cubature-branch index. Each branch is one of the `2^F` sign vectors; the audit therefore has 16 branches. |
| `B`, `I` | Random branch and random front-node index, respectively; their realized values are written `b` and `i`. |
| `epsilon_b` | The `F`-vector of `-1/+1` Rademacher signs assigned to branch `b`. |
| `pi_b` | Risk-neutral probability of branch `b`: `pi_b = P(B=b | F_t)`. For the equally weighted Rademacher cubature, `pi_b = 2^(-F) = 1/16`. Here `pi` denotes a probability weight, not the circle constant. |
| `Gamma_m` | State-dependent `N_x x F` diffusion-loading matrix of the discrete probability vector at expiry `m`, obtained from the specified DLVsq loadings. Its suppressed state index is understood. Its front-expiry value is zero: `Gamma_1 = 0`. |
| `q_m^b(j)` | Branch-conditioned expiry-`m` mass **before** the front node is observed. It is an auxiliary branch marginal, not the final pathwise conditional surface. |
| `r_ij^(b,m)` | Probability of later node `j` at `T_m`, conditional on branch `b` and on having observed front node `i` at `T_1`. The row `r_i.^(b,m)` is the actual pathwise survivor density after both random outcomes are known. |
| `x_j'^(i)` | Re-gridded relative node `x_j/x_i` after front node `i` is observed. |
| `K`, `C_t^m(K)` | Fixed cash strike and time-`t` price of the call expiring at `T_m`. |
| `(a)^+`, `E_t` | `max(a,0)` and risk-neutral expectation conditional on `F_t`. |

Let `B` denote the random branch and `I` the random front-node index. The
implementation draws them independently conditional on the current state:

`P_t(B=b, I=i) = pi_b q_1(i)`.

Thus `q_m^b` is the expiry density after the surface shock `b` but before
conditioning on `I=i`; the row `r_i.^(b,m)` is the density carried on an actual
path after both `b` and `i` are known. The front density is frozen across
branches because `Gamma_1=0`, so `q_1^b=q_1`. This particular exact law does
not introduce a contemporaneous correlation between the raw surface-branch
shock and the front-node draw. The rolled surface nevertheless depends on spot
through conditioning and re-gridding.

## 2. What is constructed

Consider only the discrete DLV factor `X` first. At time `t`, let the current
spot be `s`, the fixed cash-support multipliers be

`x_0 < x_1 < ... < x_(N_x-1)`,

and let `q_m(j)` be the risk-neutral density of the `m`-th fixed expiry on that
support. Each density has unit mass and unit mean. The DLVsq coordinates are the
local-variance representation of the density tower through the discrete SANOS
recursion.

For one calendar week, the branch masses are

`q_m^b(j) = q_m(j) + sqrt(h) [Gamma_m epsilon_b]_j`.

The cubature weights obey `sum_b pi_b=1`, `sum_b pi_b epsilon_b=0`, and
`sum_b pi_b epsilon_b epsilon_b' = I_F`, where `I_F` is the `F x F` identity
matrix. Consequently `sum_b pi_b q_m^b(j)=q_m(j)` for every expiry `m` and node
`j`. The implementation then:

1. Compute the diffusion of each density implied by the chosen DLVsq
   diffusion loadings and the fixed-contract DLV drift restriction.
2. Use the symmetric Rademacher cubature above to form probability-mass
   branches. Symmetry gives the exact finite-step branch mean.
3. Recover the DLVsq coordinates of each branch algebraically from its density
   using the discrete SANOS call recursion. This makes every retained branch a
   valid decoded surface; it avoids Euler weak drift and a nonlinear finite-step
   fit.
4. Sample `B=b` with probability `pi_b` and, independently, `I=i` with
   probability `q_1(i)`. Propagate the point mass at front node `i` through the
   selected branch to obtain the pathwise rows `r_ij^(b,m)` for `m=2,...,M`.
5. Set the next spot to `S_(t+h) = s x_i`. The experiment summarized here sets
   the independent Black factor identically to one; no `Y` random variable is
   simulated or priced.

The construction is admissible while branch masses remain nonnegative and
the recovered DLVsq values remain inside their configured open bounds. The
implementation checks mass, mean, positivity, decoder consistency, and DLV
bounds at every roll. A materially negative branch is rejected; a negative
value no larger than the configured round-off tolerance is clamped to zero and
the resulting branch-mean residual is reported.

## 3. Why weekly re-gridding preserves option prices

For a surviving expiry `m=2,...,M`, branch `b`, and revealed front node `i`,
the conditional rows satisfy

`sum_j r_ij^(b,m) x_j = x_i`,

`sum_i q_1(i) r_ij^(b,m) = q_m^b(j)`, and

`sum_b pi_b q_m^b(j) = q_m(j)`.

The first identity says that the conditional X process remains a martingale
from `T_1` to `T_m`. The second averages pathwise rows over the possible front
nodes to recover the branch marginal. The third averages branch marginals to
recover the time-`t` density.

After the spot realization, define the floating relative support

`x_j'^(i) = x_j / x_i`.

Then `S_(t+h) x_j'^(i) = s x_j` path by path. A surviving fixed-cash call with
strike `K` and expiry `T_m` therefore has the same conditional value on the
inherited cash grid and on the re-gridded relative grid:

`C_(t+h)^(b,i,m)(K) = sum_j r_ij^(b,m) (s x_j - K)^+`.

Averaging over the explicitly defined joint law gives

`E_t[C_(t+h)^m(K)]`

`= sum_b pi_b sum_i q_1(i) sum_j r_ij^(b,m) (s x_j-K)^+`

`= sum_j [sum_b pi_b q_m^b(j)] (s x_j-K)^+`

`= sum_j q_m(j) (s x_j-K)^+ = C_t^m(K)`.

Put prices follow from parity. In fact, the density tower preserves the price
of any payoff represented on the finite support. No interpolation of survivor
densities is involved. The newly issued floating 4W slice is separate: there
is no pre-existing 5W contract whose value it must inherit.

## 4. Floating 1W--4W roll

At the end of each week, old fixed-expiry 2W--4W densities become the new
floating 1W--3W densities after conditioning and re-gridding. A separately
estimated last-expiry model supplies only the new 4W DLVsq curve; SANOS then
constructs its density as a static martingale extension of the surviving 3W
density.

The current boundary model is a training-only 25-factor principal-component
analysis (PCA) followed by an unrestricted weekly vector autoregression of
order one, VAR(1), in factor scores. It used 703 weekly training pairs
through 2024-12-31 and 155 untouched 2025 validation pairs. The factors explain
99.9912% of the last-expiry curve variation, the fitted spectral radius is
0.9089, and validation RMSE is 0.5902 versus 0.6526 for persistence, a 9.56%
improvement. This AR model is the only fitted component in the construction;
the martingale transition itself is analytical, not trained.

## 5. Martingale tests and decision rule

Let `a=1,...,n` index independent antithetic pairs and let `k=1,...,d` index
the audited return coordinates (spot and option contracts). `R_ak` is the mean
return of the two antithetic members in pair `a` for coordinate `k`; `Rbar_k`
is its average over pairs; and `sigmahat_k` is its sample standard deviation
with denominator `n-1`. Then

`z_k = sqrt(n) Rbar_k / sigmahat_k`.

Let `Rbar` be the `d`-vector of coordinate means and `Sigmahat` the `d x d`
sample covariance matrix of the pair observations. `Sigmahat^+` is its
truncated Moore--Penrose inverse: the code first converts to correlation space
and discards correlation eigenmodes at or below 0.01. The joint
covariance-aware Wald statistic is

`J = n Rbar' Sigmahat^+ Rbar`.

If `nu` modes remain, `nu` is the reported degrees of freedom and the reference
approximation under zero drift is `J ~ chi-square(nu)`. The p-value is the
upper-tail probability

`p = P(chi-square(nu) >= observed J | all d population means are zero)`.

A small p-value is evidence against zero drift; a large p-value means the
observed mean vector is compatible with Monte Carlo noise. The per-test cutoff
is 0.05. Across four weekly looks, a conservative Bonferroni cutoff is 0.0125;
across 16 looks it is 0.003125. A high p-value is not proof of equivalence, and
the hard eigenvalue cutoff makes the asymptotic p-value sensitive when option
returns are highly collinear. Exact density-tower residuals and matched direct
versus re-grid prices are therefore the primary structural checks.

## 6. Four-week, 100,000-path X-only result

The high-power run used seed 20260918 and all 891 certified states.
Each state received 56 or 57 independent antithetic pairs. The audited vector
contains spot plus the still-surviving original fixed-cash contracts; there are
11 fixed call/put contracts per surviving expiry. Here `bp` means one basis
point, `10^(-4)` of the stated reference spot.

| Week | Worst abs(z) | Max option mean (bp of initial spot) | J | df | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.796 | 0.497 | 10.980 | 7 | 0.139 |
| 2 | 2.143 | 0.863 | 7.228 | 7 | 0.406 |
| 3 | 1.371 | 0.696 | 5.858 | 6 | 0.439 |
| 4 | 2.057 | 1.245 | 5.506 | 5 | 0.357 |

All four joint tests fail to reject at the 5% criterion. A coordinate slightly
above two is not itself a joint rejection, especially among many correlated
coordinates.

The deterministic all-state checks were much tighter than Monte Carlo error:

- maximum density-tower residual: `6.49e-15`;
- maximum option-tower residual: `1.67e-16`;
- maximum spot-mean residual: `2.60e-14`;
- maximum decoder residual over the four weekly rolls: `1.77e-13`;
- maximum first re-grid cash-price residual: `3.33e-16`.

The run took 169.8 seconds, or 2,355 path-week steps per second, with detailed
all-state diagnostics enabled.

## 7. Matched fixed-expiry versus re-grid diagnostic

A single matched 100,000-path X-only run used identical random draws to
decompose the weekly option change into the fixed-strike/fixed-expiry surface
branch and the spot-conditioning/re-grid component. It also priced every
survivor directly on the inherited cash support and after re-gridding.
This is a matched fixed-support/floating-coordinate decomposition of one X-only
DLV process, not a comparison of two independently specified dynamics.
There is no separate nontrivial four-step process that keeps the same relative
grid after spot conditioning: once `x_i` is revealed, the inherited conditional
density has mean `x_i`, so dividing support by `x_i` is what restores unit mean
for the next step. “Fixed grid” therefore means fixed cash support within each
weekly transition; “floating grid” is its exact pathwise coordinate expression
between transitions.

| Week | Fixed-expiry branch p | Full transition, first-member p (direct = re-grid) | Full antithetic p | Max direct/re-grid difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.366 | 0.684 | 0.139 | 0.0 |
| 2 | 0.538 | 0.163 | 0.128 | `2.22e-16` |
| 3 | 0.911 | 0.414 | 0.0208 | `2.22e-16` |
| 4 | 0.945 | 0.0808 | 0.546 | `2.22e-16` |

The fixed-expiry and direct/re-grid columns use one independently distributed
first member from each pair; the full column uses antithetic pair means. The
direct and re-grid return matrices have identical z/J/p statistics to numerical
precision. The week-3 pair-mean test is a nominal per-test rejection at 5%, but
not at the pre-specified four-look Bonferroni cutoff of 0.0125. Because the
direct and re-grid path matrices are identical, this result cannot be caused by
the coordinate change.

The week-3 p-value is visibly covariance-mode sensitive: it is 0.0429, 0.0208,
0.0879, and 0.295 at correlation-eigenvalue cutoffs 0.005, 0.01, 0.02, and
0.05. Using one half-sample to estimate covariance and the other to estimate the
mean gives p-values 0.214 and 0.338. No week-3 coordinate has `abs(z) > 2` and
the maximum economic mean is only 0.363 bp of spot. These checks support the
interpretation that the low nominal p comes from sample-selected covariance
modes rather than an economically material drift.

The pair-mean rolling-book statistics were:

| Week | Worst abs(z) | Max option mean (bp of pre-step spot) | J | df | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.796 | 0.497 | 10.985 | 7 | 0.139 |
| 2 | 1.545 | 0.455 | 9.916 | 6 | 0.128 |
| 3 | 1.317 | 0.363 | 16.522 | 7 | 0.0208 |
| 4 | 1.789 | 0.711 | 6.913 | 8 | 0.546 |

The fixed-expiry surface-branch component alone had p-values 0.366, 0.538,
0.911, and 0.945. Its maximum absolute option means were only 0.0046, 0.0056,
0.0013, and 0.0050 bp of spot. The extra sampling variation in the full
transition comes from selecting the front X node and conditioning survivor
densities, not from the DLV branch mean or the re-grid.

## 8. Four-week stability and boundary behavior

The four-week X-only run remained numerically stable:

- maximum decoder-density residual: `1.77e-13`;
- conditional mass and mean residuals remained below `1.0e-13`;
- largest per-batch fallback-coordinate fraction: 0.0038%;
- no DLVsq coordinate reached the final physical cap;
- median cross-expiry dispersion of `sqrt(DLVsq)` rose from 0.0459 at week 1
  to 0.0540 at week 4, so the term structure did not collapse.

The finite boundary was selected on 0.003%, 0.003%, 0.003%, and 0.006% of paths
over weeks 1--4. Median effective front-density nodes were 34.81, 37.80, 35.69,
and 38.75; the week-4 first percentile was 7.44 nodes. The week-4 p99 largest
density atom was 27.45%. Here p01 and p99 mean the first and 99th percentiles.
Boundary incidence is therefore real but too small to
explain the four-week p-value behavior.

If the lowest node `x_0` is selected, the martingale condition requires

`sum_j r_0j^(b,m) x_j = x_0`, with `x_j >= x_0`.

Hence `r_00^(b,m) = 1`: the conditional distribution is a point mass and every
surviving X-only option is intrinsic. Interior DLVsq coordinates become
price-unidentified. The positive-density drift formula, which divides density
diffusion by `q`, ceases to apply at zero-density interior nodes; the
implementation's fallback continuation is a numerical convention invisible to
the absorbed X law. The same argument applies at the upper support node.

Avoiding this absorbing artifact requires a genuine tail continuation, dynamic
support expansion before the boundary is hit, or a boundary atom that retains
an internal conditional tail distribution. Re-gridding alone cannot invent
missing support beyond an extreme node while preserving the entire old option
surface exactly.

## 9. Scope and current assessment

What is established:

- an exact one-step martingale density tower for every certified initial state;
- pathwise price-preserving re-gridding for every surviving fixed-cash option;
- stable repeated weekly simulation through four X-only steps in the tested
  regime;
- no 5% joint zero-drift rejection for the surviving original fixed-cash book;
- one nominal 5% rolling-book rejection that does not survive the pre-specified
  four-look correction and cannot be attributed to re-gridding;
- a clear diagnosis of the absorbing finite-support boundary.

What is not established:

- uniqueness of the risk-neutral law;
- a nonzero contemporaneous correlation between the surface-branch shock `b`
  and the front-node innovation `i`;
- a continuous-strike or unbounded-support martingale measure;
- exact chi-square calibration of the truncated-eigenmode Wald statistic;
- a martingale constraint linking the newly issued 4W option to a pre-existing
  5W contract;
- absence of economically material boundary bias beyond the four X-only weeks
  tested here.

The strongest conclusion is therefore: the approach is internally
martingale-consistent on its modeled finite support, re-gridding is exact for
survivors, and the X-only implementation remains stable over four weekly rolls;
the next modeling issue is tail/support treatment rather than drift fitting.

## Reproducibility pointers

Canonical implementation:

- `quantdata/dysanos/flex/lvsq_drift.py`
- `quantdata/dysanos/flex/fixed_grid_dlv.py`
- `quantdata/dysanos/flex/floating_grid_dlv.py`
- `quantdata/dysanos/flex/last_expiry_ar1.py`
- `quantdata/dysanos/flex/run_all_state_regrid_dlv_audit.py`

Primary reports (large artifacts, intentionally outside Git):

- `.../x-only-fixed-vs-regrid-dlv-v1/4-week-100000-seed-20260918/report.json`

QuantAgent independently reviewed the martingale tower, re-grid proof, boundary
argument, and statistical interpretation before this summary was finalized.
