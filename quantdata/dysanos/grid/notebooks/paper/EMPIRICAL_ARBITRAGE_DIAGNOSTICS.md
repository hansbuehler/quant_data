# Empirical arbitrage diagnostics

## Objective

We have simulated returns

\[
dC^\ell_t(\omega),\qquad
\omega=1,\ldots,\Omega,\quad
t=0,\ldots,T-1,\quad
\ell=0,\ldots,L.
\]

The array convention is:

```python
dC.shape == (num_paths, num_steps, num_instruments)
dC[omega, t, ell] == dC_t^ell(omega)
```

Instrument 0 is usually spot; the remaining instruments are calls.

The basic LP uses deterministic time/instrument positions

\[
a_t^\ell,
\]

common to all paths. There is no path index on `a`, so the LP does not use future information or choose `sign(dC)` path by path.

The terminal net payoff is

\[
G_\omega(a)
=
\sum_{t,\ell} a_t^\ell dC_t^\ell(\omega)
-
\sum_{t,\ell}|a_t^\ell|\,k_{\omega,t,\ell}.
\]

The per-unit absolute-position cost is

\[
k_{\omega,t,\ell}
=
\text{prop\_cost}\,|X_{\omega,t,\ell}|
+
\delta\,
\frac{\operatorname{std}_\omega(dC^\ell_t)}{\sqrt{\Omega}}.
\]

`X` represents the economic scale/cost of entering a position. It may be scalar or broadcastable to the shape of `dC`, for example `(T,L)` or `(Omega,T,L)`.

`delta` penalises each position by a multiple of the Monte Carlo standard error of the corresponding return mean, to avoid exploiting sampling noise.

## Main LP

The first-stage LP solves

\[
\max_a \frac1\Omega\sum_\omega G_\omega(a)
\]

subject to

\[
G_\omega(a)\ge -\varepsilon
\quad\forall\omega,
\qquad
|a_t^\ell|\le A_{\max}.
\]

Because costs use `abs(a)`, the implementation uses

\[
a=a^+-a^-,\qquad a^+,a^-\ge0.
\]

A positive optimal mean together with non-negative pathwise payoff is an empirical arbitrage certificate for this finite path set and strategy class.

## Sparse follow-up

The first LP often returns a dense corner solution. A second LP finds a minimum-L1 certificate:

\[
\min_a \|a\|_1
\]

subject to pathwise non-negativity and a required positive mean payoff. The code runs this at 50%, 25%, and 10% of the first-stage mean. Due to homogeneity, these commonly identify the same ray at different scales.

The module also includes:

- iterative reweighted L1;
- position-contribution ranking;
- deletion importance;
- greedy pruning;
- binding-path identification;
- time-step payoff summaries;
- ray-comparison diagnostics;
- JSON, CSV, and NumPy output.

## Requested experiments

Run:

1. `delta=0`, `prop_cost=0`
2. `delta=1`, `prop_cost=0`
3. `delta=0`, `prop_cost=0.0001` (1 bp)

Python:

```python
from empirical_arbitrage import run_standard_experiments

results = run_standard_experiments(
    dC,
    X,
    output_dir="empirical_arbitrage_results",
)
```

CLI input:

```python
np.savez("input.npz", dC=dC, X=X)
```

Then:

```bash
python empirical_arbitrage.py input.npz --output-dir results
```

If the NPZ has no `X`, the proportional entry-cost component is zero.

## Important interpretation

This is an empirical finite-support problem. A certificate on the supplied paths does not automatically prove population arbitrage. Validate fixed positions on fresh independently simulated paths.

The global LP tests cumulative terminal payoff:

\[
\sum_t a_t^\top dC_t.
\]

A per-time LP tests a stronger one-step pathwise condition and can miss cumulative cross-time cancellation.

## Files

- `empirical_arbitrage.py`: implementation.
- `EMPIRICAL_ARBITRAGE_DIAGNOSTICS.md`: this document.

## Suggested Codex tasks

1. Add tests using small synthetic markets:
   - deterministic arbitrage;
   - symmetric no-arbitrage returns;
   - arbitrage removed by proportional cost;
   - arbitrage removed by MC-error cost.
2. Add an optional per-time LP diagnostic.
3. Add mapping from flat instrument index to `(expiry, strike)` metadata.
4. Add plotting:
   - position heatmaps;
   - payoff histograms;
   - support by time;
   - binding-path contribution charts.
5. Benchmark SciPy/HiGHS on about 40,000 paths, 9 time steps, and 141 instruments.
6. Consider direct `highspy` integration for explicit thread control and sparse matrices.
