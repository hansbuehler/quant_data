# Fixed-grid risk-neutral experiments

**Analytical DLVsq drift Monte Carlo versus power-exponential marginal-utility measure change**  
**Research status — 14 September 2026**

## Executive conclusion

Two fixed-grid, X-only approaches were tested on the same initial SANOS state,
the same four-week shrinking option tower, and the same 2,000 one-step option
return coordinates.

1. **Analytical DLVsq-drift Monte Carlo.** The DLVsq diffusion uses the
   theoretical fixed-contract drift and a ten-substep stochastic Heun
   integrator. On 100,000 paths it did not reject zero drift for the 2,000
   one-step mark-to-market returns: the dependence-aware joint p-value was
   0.2746. Its worst coordinate had `abs(t)=3.334`, below the simultaneous 5%
   cutoff. This is positive evidence for the finite-step implementation, but
   the transition remains a numerical approximation to an infinitesimal drift
   equation.
2. **Marginal-utility measure change.** Physical paths use the same DLV
   covariance but zero raw log-DLV drift. A positive empirical density is
   obtained from normalized power-exponential marginal utility and refined by
   repeated adversarial root corrections. The retained `Q4` weights have
   effective sample size 69,272. Under `Q4`, the 2,000 raw one-step option
   returns have joint p-value 1.000, worst `abs(t)=2.047`, and RMS t-score
   0.441. A fresh matched fixed-feature adversary finds utility
   `3.91e-6`; its frozen-strategy mean-P&L t-score is 0.776.

The second result is not a general learned risk-neutral model. Its nonlinear
features were frozen at width four, it was fitted and tested on the same fixed
100,000-path sample by design, and it supplies empirical path weights rather
than a reusable transition kernel. When the nonlinear layers were learned,
fresh same-architecture adversaries continued to find large utilities around
0.5--0.7. The analytical construction is therefore the stronger current
model result; the favorable frozen-feature measure change is a useful
finite-sample root-finding demonstration, not a broad no-arbitrage certificate.

This report compares the analytical SDE experiment with the empirical
measure-change experiment. The separate density-first, exact finite-support
construction is documented in `DLV_REGRID_MARTINGALE_SUMMARY.pdf` and should
not be confused with the ten-substep analytical SDE baseline here.

## 1. Common experiment

Both experiments use:

- the fixed relative-strike SANOS decoder with 200 interior strikes;
- source state index 44, dated 8 March 2022;
- an X-only four-week shrinking tower and no re-gridding;
- fixed cash strikes within every weekly transition;
- 100,000 paths arranged into 50,000 antithetic pairs;
- 50 antithetic pairs per randomized-stratification block, giving 1,000
  independent block observations;
- four trade weeks with 800, 600, 400, and 200 live one-step option returns;
- all 2,000 return coordinates in the final joint test;
- rank-25 innovations estimated from 703 historical weekly observations
  through 31 December 2024; and
- 10,000 shared-sign multiplier-bootstrap replications in the retained final
  tests.

The 2,000 returns are one-step price returns of the same fixed-cash-strike,
fixed-expiry claims:

\[
R_t^{j,i}=\frac{P_{t+1}^{j,i}}{P_t^{j,i}}-1,
\]

where the endpoint value of a front-expiry option is its payoff. At each trade
date an ITM call or put is selected pathwise so that the denominator is not a
near-zero OTM price.

The path sets are independently generated runs rather than a matched-random
comparison. Their dimensions, initial state, option grid, covariance estimate,
and test protocol agree; their dynamics and probability measures deliberately
differ.

| Property | Analytical DLVsq drift | Utility measure change |
| --- | --- | --- |
| Starting dynamics | Risk-neutral DLVsq drift | Zero raw log-DLV drift (`P`) |
| DLV covariance | Fixed-expiry AR residual covariance, rank 25 | Same estimator and rank |
| Weekly integration | 10-substep stochastic Heun | One exact raw innovation before reweighting |
| Resulting object | Simulated transition law | Positive weights on fixed paths |
| States in this comparison | One: state 44 | One: state 44 |
| Paths / independent blocks | 100,000 / 1,000 | 100,000 / 1,000 |
| Option-return coordinates | 2,000 | 2,000 |
| Precision | fp64 | fp64 simulation; fp32 policy tensors |

## 2. Analytical DLVsq-drift Monte Carlo

Let `v` denote DLVsq, `v_min` its strict lower bound, and

\[
z=\log(v-v_{\min}).
\]

The fitted loading matrix supplies the diffusion of `z`. The drift is computed
from the theoretical condition that each fixed-contract option price has zero
instantaneous drift. In density coordinates the implementation transports the
previous-expiry density diffusion through the discrete SANOS operator and
uses the relative density diffusion to calculate the physical DLVsq drift
correction. The Itô correction `-0.5 ||sigma||^2` is included in log
excess-DLV coordinates.

Each week is integrated with ten stochastic-Heun substeps. The same Gaussian
increment is used in the predictor and corrector, appropriate because the
diffusion is additive in `z`. After the DLV endpoint is constructed, spot is
drawn from the front density, the expired slab is removed, and the surviving
conditional densities are propagated on the unchanged cash grid.

This approach uses the analytical infinitesimal drift but is not an exact
finite-one-week martingale scheme. Its validity is consequently assessed by
Monte Carlo zero-mean tests as well as density mass and forward checks.

### Quantities tested

Five families were audited:

1. **One-step mark-to-market ITM option returns:** the primary 2,000-coordinate
   comparison with the learned measure.
2. **Terminal ITM option returns:** each option opened during the tower is held
   to its cash expiry.
3. **Terminal option price changes:** the same terminal claims without division
   by their initial prices.
4. **Integrated DLV drift control:** prices after integrating the DLV endpoint
   but before revealing spot. Eight hundred coordinates were numerically
   unresolved because their block variance was effectively zero; 1,200 active
   coordinates were tested.
5. **Weekly spot returns:** four coordinates.

### Results

| Audit | Active | max `abs(t)` | RMS t | max/RMS combined p | 5% decision |
| --- | ---: | ---: | ---: | ---: | --- |
| One-step option returns | 2,000 | 3.334 | 1.109 | 0.2746 | Do not reject |
| Terminal option returns | 2,000 | 3.334 | 1.044 | 0.3210 | Do not reject |
| Terminal price changes | 2,000 | 2.934 | 1.156 | 0.3500 | Do not reject |
| Integrated DLV control | 1,200 | 1.950 | 0.878 | 1.0000 | Do not reject |
| Weekly spot returns | 4 | 2.431 | 1.561 | 0.0876 | Do not reject |

For the primary one-step option-return audit, the simultaneous max-t bootstrap
p-value was 0.1373 and the RMS-t p-value was 0.2465. The Bonferroni familywise
p-value was 1.000 and its 5% max-t cutoff was 4.235. Thus individual
coordinates above two standard errors did not imply joint rejection.

Numerical diagnostics were:

- maximum density mass residual: `9.33e-14`;
- maximum conditional-forward residual: `8.93e-14`;
- minimum density: exactly zero, reflecting point masses or finite-support
  boundary behavior rather than a negative density; and
- simulation time: 72.07 seconds, or 1,387.6 paths per second.

## 3. Statistical definitions and decision rule

The raw paths are not treated as 100,000 independent observations. Adjacent
paths form antithetic pairs, and 50 pairs form one randomized-stratification
block. For coordinate `k` and block `b`, let `X_bk` be the mean contribution
of the 100 paths in that block. With `B=1,000`,

\[
t_k=\frac{\sqrt B\,\bar X_k}{s_k},
\qquad
\bar X_k=\frac1B\sum_b X_{bk},
\]

where `s_k` is the sample standard deviation of the block means.

Two simultaneous statistics are used:

\[
T_{\max}=\max_k |t_k|,
\qquad
T_{\mathrm{RMS}}=\sqrt{\frac1d\sum_k t_k^2}.
\]

A Rademacher shared-sign multiplier bootstrap is applied to centered block
vectors. The same sign multiplies every coordinate in a block, retaining the
observed cross-option dependence. If `p_max` and `p_RMS` are the two bootstrap
p-values, the reported decision value is

\[
p_{\mathrm{combined}}=\min\{1,2\min(p_{\max},p_{\mathrm{RMS}})\}.
\]

Zero drift is rejected at 5% when `p_combined < 0.05`. A large p-value means
that the observed sample mean is compatible with zero drift under this test;
it is not proof that the population drift is exactly zero. A two-sided
Bonferroni max-coordinate test is also reported as a conservative check.

Under empirical weights `w_n` normalized to sample mean one, the learned-Q
block contribution is

\[
X_{bk}^{Q}=\frac1{100}\sum_{n\in b}w_nR_{nk}.
\]

The same block t-statistics and shared-sign bootstrap are then applied. This
tests the weighted sample moments while respecting the original independent
block structure.

## 4. Marginal-utility measure change

### Physical path law

The physical test paths use the same initial state, fixed cash grid, spot draw,
and rank-25 DLV innovation covariance. Their raw log-DLV dynamics intentionally
omit the analytical risk-neutral drift:

\[
\log(v_{t+1}-v_{\min})-\log(v_t-v_{\min})
=\sqrt h\,\xi_t\varepsilon_t.
\]

This path law contains a very strong statistical arbitrage signal. The
recorded nearly uniform initialization, whose density ESS fraction was
numerically 1.000, had worst option `abs(t)` about 80.63 and RMS t about 16.77;
the 250-sample bootstrap rejected with its minimum resolvable combined
p-value, 0.00797. This diagnostic is from a tiny-random-action initialization,
not a separately serialized exact-uniform audit, but its weights were
numerically uniform.

### Utility and density

For terminal trading P&L `pi`, the implemented direct utility is

\[
u(\pi)=
\begin{cases}
1-e^{-\pi},&\pi>0,\\
\pi-\pi^2/2,&\pi\leq0.
\end{cases}
\]

Its strictly positive marginal is normalized into an empirical density:

\[
w_n^{\mathrm{new}}
=\frac{w_n^{\mathrm{base}}u'(\pi_n)}
       {N^{-1}\sum_m w_m^{\mathrm{base}}u'(\pi_m)}.
\]

Successive stages apply this formula multiplicatively. The base weights are
detached, so a later adversary cannot modify an earlier fitted measure.

**Terminology.** This run is often called the “OCE measure change” in the
project discussion because normalized marginal utility is also the OCE density
mechanism. The code actually optimized direct power-exponential utility. It did
not optimize an OCE translation `y`, so it is more precise to call this the
power-exponential marginal-utility measure change.

### Conditioning and strategy instruments

The raw option returns are highly collinear. Once under the original uniform
physical sample, each option return is beta-adjusted against its same-week spot
return,

\[
U_t=R_t-\beta_t S_t,
\qquad
\beta_t=\frac{\operatorname{Cov}_P(R_t,S_t)}
                    {\operatorname{Var}_P(S_t)}.
\]

The unconditional residual covariance is diagonalized. Eigenvalues below
`lambda_max/1000` are lifted to that floor by fixed-seed antithetic Gaussian
noise and the result is whitened. Standardized spot is retained as a separate
factor. Beta, covariance, eigenvectors, and noise scale are detached from the
optimization. The final acceptance test always uses the original noise-free
spot and option returns.

The policy observes the square root of the live density tower. In the retained
experiment two nonlinear hidden layers of width four were initialized once and
frozen. Only the final linear maps were trained:

\[
\phi_t(H_t)=\operatorname{SiLU}
\left(W_{2,t}\operatorname{SiLU}(W_{1,t}H_t+b_{1,t})+b_{2,t}\right),
\qquad
a_t(H_t)=B_t\phi_t(H_t)+c_t.
\]

This gives 10,020 trainable final-layer parameters out of 18,196 total model
parameters. The same feature seed and exact same policy class are used by the
measure builder and every fresh adversary. The notional radius was 65,536;
the maximum realized action norm was only 8.08 for the retained measure, so
the cap was inactive. Full-path direct utility was optimized with L-BFGS.

## 5. Measure-change results

The retained measure is `Q4`, the output of the fourth marginal-utility tilt.
Stage 5 is a fresh same-class adversary against `Q4`; its own tilted `Q5` is not
called the retained measure because it has not received another independent
same-class challenge.

| Stage | Base measure challenged | Maximum utility found | Resulting-weight ESS |
| --- | --- | ---: | ---: |
| 1 | Uniform physical paths | 0.204156 | 69,290 |
| 2 | Q1 | `9.65e-5` | 69,283 |
| 3 | Q2 | `1.92e-5` | 69,263 |
| 4 | Q3 | `7.55e-6` | 69,272 (`Q4`) |
| 5 certificate | Q4 | `3.91e-6` | 69,260 (`Q5`, not certified) |

### Raw martingale audit under Q4

| Return family | Coordinates | max `abs(t)` | RMS t | max p | RMS p | combined p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All one-step options | 2,000 | 2.047 | 0.441 | 0.9640 | 0.9998 | 1.0000 |
| Weekly spot | 4 | `8.02e-7` | `4.34e-7` | 1.0000 | 1.0000 | 1.0000 |

The option results by trade week were:

| Trade week | Coordinates | max `abs(t)` | RMS t | max absolute weighted mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 800 | 2.047 | 0.625 | 0.004403 |
| 1 | 600 | 0.842 | 0.295 | 0.001769 |
| 2 | 400 | 1.238 | 0.236 | 0.000890 |
| 3 | 200 | 0.416 | 0.095 | 0.000592 |

The transformed factor-space audit also had combined p-value 1.000, worst
`abs(t)=1.21e-6`, and RMS t `3.91e-7`. The zero-strategy direct-objective
gradient RMS under `Q4`, measured at the beginning of the Stage-5 challenge,
was `1.69e-6`, versus `5.38e-3` at the physical starting measure: a ratio of
`3.15e-4`.

The Stage-5 frozen adversarial strategy had:

- mean utility `3.91e-6`, block standard error `1.01e-5`, and descriptive
  t-score 0.387;
- mean P&L `7.83e-6`, block standard error `1.01e-5`, and descriptive t-score
  0.776;
- P&L standard deviation under `Q4` of 0.002806; and
- maximum action norm 5.60, far below the radius 65,536.

These t-scores treat the final strategy as frozen. Because it was selected on
the same fixed path set, they are descriptive rather than an independent
out-of-sample significance test. Same-sample fitting was intentional: the
experiment was posed as finite-sample root finding.

### Learned-feature control

Freezing the hidden features makes the action linear in the learned parameters
and the utility problem concave while the action cap is inactive. This is why
the same-class certificate became easy. It is a restricted result.

With two learned hidden layers of width 16, the first physical-measure strategy
found utility 0.7216. Fresh same-architecture adversaries after successive
measure changes still found utilities 0.5714 and 0.5922. A larger learned model
also found material residual utility. Therefore the learned-feature version did
not establish a risk-neutral measure, even though unconditional raw option
means could be made statistically small.

## 6. Direct comparison

| Primary one-step audit | Analytical DLVsq drift | Retained empirical Q4 |
| --- | ---: | ---: |
| Option coordinates | 2,000 | 2,000 |
| max `abs(t)` | 3.334 | 2.047 |
| RMS t | 1.109 | 0.441 |
| dependence-aware combined p | 0.2746 | 1.0000 |
| 5% zero-drift decision | Do not reject | Do not reject |
| Weekly-spot combined p | 0.0876 | 1.0000 |
| Effective sample size | 100,000 simulated paths | 69,272 weighted paths |
| Density mass residual | `9.33e-14` | `8.59e-14` in physical path construction |
| Conditional-forward residual | `8.93e-14` | `8.30e-14` in physical path construction |

The empirical Q4 statistics are numerically smaller, but this is not evidence
that Q4 is the better general risk-neutral model. Q4 was optimized against the
same finite sample and a restricted adversary, whereas the analytical DLVsq
drift was specified before the Monte Carlo draw. The analytical test therefore
has the stronger interpretation despite its larger z-scores.

The analytical run also tests terminal option returns and terminal price
changes. The measure-change report tests one-step same-claim returns only, so no
terminal learned-Q result should be inferred.

Timing is indicative rather than a controlled benchmark. The analytical
100,000-path simulation took 72.1 seconds. Physical path generation for the
measure-change experiment took about 4.5 seconds, but the four construction
tilts required approximately 222 seconds of solver time and the Stage-5
certificate another 56 seconds. Separate processes regenerated and re-audited
the deterministic path sample at every stage.

## 7. What has and has not been established

### Established for the analytical DLVsq experiment

- The four-week fixed-grid implementation remained numerically stable.
- The analytical drift plus ten-step Heun discretization did not reject the
  2,000 one-step or terminal option martingale restrictions at 5%.
- Spot also did not reject at 5%, although its joint p-value of 0.0876 was much
  closer to the cutoff.

### Established for the empirical Q4 experiment

- Positive finite-sample weights with ESS 69.3% were found.
- Spot and all 2,000 raw option-return means passed the dependence-aware test on
  the fitted sample.
- A fresh adversary using the identical frozen-feature class found no
  statistically material P&L on the block diagnostic.

### Not established

- Q4 is not a reusable state-conditional transition kernel.
- Q4 has not been established for all 891 historical states.
- The learned hidden-feature strategy class still detects substantial residual
  opportunity; hence general conditional no-arbitrage has not been shown.
- Neither large p-values nor a small fitted adversary constitute proof of exact
  population zero drift.
- This comparison is not matched-path and does not measure distance between the
  two risk-neutral laws.
- The analytical SDE does not become an exact finite-step construction merely
  because the Monte Carlo test fails to reject.

The current ranking is therefore:

1. the separate density-first construction is exact on its finite support;
2. the analytical DLVsq SDE is a credible approximate fixed-grid risk-neutral
   simulator for this experiment; and
3. the empirical Q4 weights solve a restricted fixed-sample problem, while the
   general learned measure remains unresolved.

## 8. Reproducibility and software tests

Primary analytical artifact:

```text
/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/
  dlv-analytic-baselines/shrinking-tower-state0044-100k-v2/report.json
```

Retained empirical measure and fresh certificate:

```text
/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/
  direct-utility/empirical-density-state0044-fixed-features-w4-v1/
    stage4-certificate.pt
    stage4-certificate.json
    stage5-certificate.json
```

Canonical source files:

- `quantdata/dysanos/flex/fixed_grid_dlv_attempt2_triangular.py`
- `quantdata/dysanos/flex/run_fixed_grid_dlv_shrinking_baseline.py`
- `quantdata/dysanos/flex/direct_utility_frontier.py`
- `quantdata/dysanos/flex/run_direct_utility_frontier.py`
- `quantdata/dysanos/flex/fixed_expiry_ar1.py`

Before the research code was committed, the combined focused regression suite
ran:

```text
python -m pytest -q -p no:cacheprovider \
  tests/quantdata/dysanos/flex/test_atm_q_existence.py \
  tests/quantdata/dysanos/flex/test_lvsq_drift.py \
  tests/quantdata/dysanos/flex/test_direct_utility_frontier.py \
  tests/quantdata/dysanos/flex/test_fixed_expiry_ar1.py \
  tests/quantdata/dysanos/flex/test_full_state_q_generator.py \
  tests/quantdata/dysanos/flex/test_zero_drift.py
```

Result: **88 passed**. Python compilation, Ruff import-order checks, shell
syntax, and Git diff checks also passed. The implementation and tests were
committed as `7b2aad7`.

