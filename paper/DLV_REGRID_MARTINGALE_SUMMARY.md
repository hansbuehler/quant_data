# X-only finite-grid DLV martingale simulation and weekly re-gridding

**Research summary — 10 September 2026**

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

## 1. What is constructed

Consider only the discrete DLV factor `X` first. At time `t`, let the current
spot be `s`, the fixed cash-support multipliers be

`x_0 < x_1 < ... < x_M`,

and let `q_m(j)` be the risk-neutral density of the `m`-th fixed expiry on that
support. Each density has unit mass and unit mean. The DLVsq coordinates are the
local-variance representation of the density tower through the discrete SANOS
recursion.

For one calendar week:

1. Compute the diffusion of each density implied by the chosen DLVsq
   diffusion loadings and the fixed-contract DLV drift restriction.
2. Use symmetric Rademacher cubature to form density branches
   `q_m^b = q_m + sqrt(h) B_m epsilon_b`. Symmetry gives the exact finite-step
   identity `sum_b pi_b q_m^b = q_m`.
3. Recover the DLVsq coordinates of each branch algebraically from its density
   using the discrete SANOS call recursion. This makes every retained branch a
   valid decoded surface; it avoids Euler weak drift and a nonlinear finite-step
   fit.
4. Sample a surface branch `b`, then a front node `i` from the branch-invariant
   front density. Propagate the one-hot front density through the selected
   branch to obtain conditional transition rows `r_ij^(b,m)`.
5. Set the next spot to `S_(t+1) = s x_i`. The experiment summarized here sets
   the independent Black factor identically to one; no `Y` random variable is
   simulated or priced.

The construction is admissible while branch densities remain nonnegative and
the recovered DLVsq values remain inside their configured open bounds. The
implementation checks mass, mean, positivity, decoder consistency, and DLV
bounds at every roll.

## 2. Why weekly re-gridding preserves option prices

For branch `b`, expiry `m`, and revealed front node `i`, the conditional rows
satisfy

`sum_j r_ij^(b,m) x_j = x_i`,

`sum_i q_1(i) r_ij^(b,m) = q_m^b(j)`, and

`sum_b pi_b q_m^b(j) = q_m(j)`.

After the spot realization, define the floating relative support

`x'_j = x_j / x_i`.

Then `S_(t+1) x'_j = s x_j` path by path. A surviving fixed-cash call with
strike `K` therefore has the same conditional value on the inherited cash grid
and on the re-gridded relative grid:

`C_(t+1)^(b,i)(K) = sum_j r_ij^(b,m) (s x_j - K)^+`.

Averaging first over `i` and then over `b` gives

`E_t[C_(t+1)(K)] = sum_j q_m(j) (s x_j - K)^+ = C_t(K)`.

Put prices follow from parity. In fact, the density tower preserves the price
of any payoff represented on the finite support. No interpolation of survivor
densities is involved. The newly issued floating 4W slice is separate: there
is no pre-existing 5W contract whose value it must inherit.

## 3. Floating 1W--4W roll

At the end of each week, old fixed-expiry 2W--4W densities become the new
floating 1W--3W densities after conditioning and re-gridding. A separately
estimated last-expiry model supplies only the new 4W DLVsq curve; SANOS then
constructs its density as a static martingale extension of the surviving 3W
density.

The current boundary model is a training-only 25-factor PCA followed by an
unrestricted weekly VAR(1) in factor scores. It used 703 weekly training pairs
through 2024-12-31 and 155 untouched 2025 validation pairs. The factors explain
99.9912% of the last-expiry curve variation, the fitted spectral radius is
0.9089, and validation RMSE is 0.5902 versus 0.6526 for persistence, a 9.56%
improvement. This AR model is the only fitted component in the construction;
the martingale transition itself is analytical, not trained.

## 4. Martingale tests and decision rule

Antithetic paths are combined into independent pair means before inference.
For return moment `k` and `n` independent pair means,

`z_k = sqrt(n) mean(R_k) / sample_sd(R_k)`.

The joint covariance-aware statistic is

`J = n mean(R)' Sigma^+ mean(R)`.

`Sigma^+` is formed in correlation space after discarding sample correlation
eigenmodes at or below 0.01. If `r` modes remain, the reported approximation is
`J ~ chi-square(r)` under zero drift, and

`p = P(chi-square(r) >= observed J)`.

A small p-value is evidence against zero drift; a large p-value means the
observed mean vector is compatible with Monte Carlo noise. The per-test cutoff
is 0.05. Across four weekly looks, a conservative Bonferroni cutoff is 0.0125;
across 16 looks it is 0.003125. A high p-value is not proof of equivalence, and
the hard eigenvalue cutoff makes the asymptotic p-value sensitive when option
returns are highly collinear. Exact density-tower residuals and matched direct
versus re-grid prices are therefore the primary structural checks.

## 5. Four-week, 100,000-path X-only result

The high-power run used seed 20260918 and all 891 certified states.
Each state received 56 or 57 independent antithetic pairs. The audited vector
contains spot plus the still-surviving original fixed-cash contracts; there are
11 fixed call/put contracts per surviving expiry.

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

## 6. Matched fixed-expiry versus re-grid diagnostic

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

## 7. Four-week stability and boundary behavior

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
density atom was 27.45%. Boundary incidence is therefore real but too small to
explain the four-week p-value behavior.

If the lowest node `x_0` is selected, the martingale condition requires

`sum_j r_0j x_j = x_0`, with `x_j >= x_0`.

Hence `r_00 = 1`: the conditional distribution is a point mass and every
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

## 8. Scope and current assessment

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
