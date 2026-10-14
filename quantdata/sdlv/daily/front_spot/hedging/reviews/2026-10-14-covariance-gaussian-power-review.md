# Gaussian Power-Exponential Covariance Review

## Verdict

**PASS for the Phase-4 quantitative gate at commit
`00d2424899f9dead3e8ae9dfdad8e49d100f48fa`.** The centered-Gaussian
power-exponential OCE, cash-translation convention, radial solver, lasso KKT
gate, entropy equivalence, and production-matrix coverage are mathematically
and mechanically consistent. No blocking issue was found.

This review covers implementation readiness. The three new rolling production
cases have not yet produced the full 611-origin result cohort.

## Scope

The review covered:

- `models/covariance/config.py` and `models/covariance/hedger.py`;
- `models/covariance/README.md`;
- the covariance branch of `runners/run_daily.py`;
- covariance cases in `runners/production_matrix.py`;
- the daily report matrix-size update;
- `tests/quantdata/sdlv/daily/hedging/models/test_covariance.py`; and
- `tests/quantdata/sdlv/daily/hedging/test_production_matrix.py`.

## Mathematical Check

Let pre-cost terminal wealth for one virtual policy be

```text
Y(a) = C + a'R,
m(a) = E[Y(a)],
s(a)^2 = Var[Y(a)]
       = v_C + 2 gamma'a + a'Sigma a.
```

The daily spread proxy is deterministic at the decision state,
`k(a)=p'|a|`. Cash invariance therefore puts this cost outside the stochastic
risk premium exactly; it is not an approximation.

The benchmark power-exponential utility is

```text
u_lambda(x) = (1-exp(-lambda*x))/lambda,  x >= 0,
              x-lambda*x^2/2,             x < 0.
```

For centered Gaussian wealth `s Z`, the implementation computes

```text
g_lambda(s) = -sup_b { b + E[u_lambda(s Z-b)] },  Z ~ N(0,1),
J(a) = -m(a) + k(a) + g_lambda(s(a)).
```

Writing the residual mean as `r=-b`, the first-order condition is

```text
E[u_lambda'(N(r,s^2))] - 1 = 0.
```

The code's exponential positive-part term, negative probability, and truncated
negative first moment reproduce this equation exactly. Its returned translation
is `b=-r`, matching the utility-side convention used by the scenario objectives.
The expected utility uses the correct truncated first and second normal moments,
and `centered_risk = r-E[u]` equals minus the optimized OCE.

The envelope derivative is

```text
g_lambda'(s) = -E[u_lambda'(s Z-b*) Z].
```

The implemented density, negative-tail, and exponentially tilted positive-tail
terms reproduce this derivative. Consequently the smooth action gradient is

```text
-mu_R + (g_lambda'(s)/s) * (Sigma a + gamma).
```

The radial iteration therefore correctly reuses the quadratic lasso solver with
effective variance aversion `g_lambda'(s)/s`. Its zero-volatility limit is
`lambda`, consistent with
`g_lambda(s)=lambda*s^2/2+o(s^2)`.

## Convexity and Numerical Gate

The negative OCE is convex, cash invariant, and nondecreasing in centered
Gaussian dispersion. Since `s(a)` is the norm induced by the positive joint
claim/instrument covariance block, `g_lambda(s(a))` is convex. Adding the linear
mean and spread-weighted absolute-value penalty preserves convexity.

Each radial step solves the correct lasso stationarity equation. Convergence is
accepted only after the action and radial coefficient stabilize and the final
lasso KKT residual, evaluated with the updated coefficient, is at most ten times
the configured action tolerance. Bound-coordinate KKT signs are correct. Solver
and executable cap diagnostics retain the existing claim-policy-minus-zero-policy
semantics.

Claim and zero policies are optimized separately. Their cash translations are
nuisance OCE variables and are correctly not subtracted as positions. The
executable action remains the capped difference of the two virtual actions, and
its objective is recomputed with the executable spread lasso.

## Entropy and Production Coverage

For Gaussian loss and deterministic costs,

```text
rho_entropy(L) = E[L] + lambda*Var(L)/2.
```

Thus the covariance entropy policy is exactly the mean-variance policy at the
same `lambda`; a second numerical fit would be duplicate computation. The
README and PLAN state this equivalence explicitly rather than relabelling a
different parameterization.

Power-exponential is not mean-variance away from the small-risk limit, so the
production matrix correctly adds three distinct robust-covariance cases at
`lambda` equal to `0.1`, `1`, and `10`. The matrix version and canonical report
count are bumped from 54 to 57. The CLI accepts the new objective through the
versioned covariance configuration.

## Independent Evidence

Focused repository tests:

```text
python -m pytest -q -p no:cacheprovider \
  tests/quantdata/sdlv/daily/hedging/models/test_covariance.py \
  tests/quantdata/sdlv/daily/hedging/test_production_matrix.py
```

Result: **38 passed**.

Report integration tests:

```text
python -m pytest -q -p no:cacheprovider \
  tests/quantdata/sdlv/daily/hedging/test_summary_document.py \
  tests/quantdata/sdlv/daily/hedging/test_summary_metrics.py
```

Result: **17 passed**.

An independent CPU calculation additionally checked:

- the analytical centered risk and translation against direct normal
  quadrature and scalar OCE maximization over a grid of risk parameters and
  standard deviations;
- the analytical slope against numerical differentiation;
- monotonicity, finiteness, and nonnegative slope on an independent 338-point
  grid with `lambda` from `1e-3` through `1e3` and standard deviation from zero
  through `1e2`; and
- the radial lasso solution against direct bounded scalar minimization in
  nonzero-mean, nonzero-cross-covariance, and nonzero-spread examples.

All comparisons passed. Direct action objective differences were below
`1e-9` in the independent scalar checks.

## Non-Blocking Qualifications

- The exactness claim is conditional on the covariance baseline's declared
  joint-Gaussian approximation and deterministic one-day spread proxy. It does
  not extend to random or path-dependent transaction costs.
- The committed integration test checks one direct quadrature point. Broader
  extreme-scale and nonzero-spread radial checks were performed independently
  for this review but would be useful permanent regression coverage.
- Full production acceptance still requires all three new cases to complete
  the authenticated 611-origin cohort with the existing convergence and cap
  gates.
