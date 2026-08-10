# PCA-IV-AR(1) benchmark

This package implements the direct implied-volatility PCA benchmark specified
in [`IMPLIED.md`](IMPLIED.md). It fits centered, unweighted PCA to
the levels of `log(market["mid_vols"])`, fits the existing autonomous
surface-first AR(1), adds conditional log-spot, and prices fixed contracts from
an un-repaired linear total-variance interpolation.

Notation follows the autonomous model: $h=(h^1,\ldots,h^{n_h})$ contains only
the implied-volatility PCA parameters and $s=\log S$ is log-spot. The stored
extended array remains $xstate=(s,h)$.

Run one experiment with:

```bash
python -m quantdata.dysanos.implied.run_experiment \
  --num-samples 100000 --trading-frequency daily \
  --return-horizon expiry --num-alpha all
```

This package-level command reports PCA fit and decoder diagnostics. The
canonical matched DYSANOS/PCA-IV arbitrage matrix is run from
[`GENDYSANOS.md`](../grid/gendysanos/GENDYSANOS.md) with:

```bash
python -m quantdata.dysanos.grid.gendysanos.run_required_experiments
```

That command uses one coherent path cache for both return horizons, both hedge
conventions, and all transaction-cost levels. Do not use the older PCA-only
comparison command as a substitute for the required 48-case report.

For a package-level CPU smoke diagnostic only, use:

```bash
python -m quantdata.dysanos.implied.run_experiment \
  --num-samples 1000 --trading-frequency weekly \
  --return-horizon incremental --num-alpha all --device cpu
```
This includes the package's fixed-contract repricing diagnostic. `--direct-p`
disables spot-tail importance sampling. `--skip-local-tail` leaves the compact
neighbour-search store cached without launching the exact local-tail valuation.
`--device auto` uses the repository CUDA device lock, while `--device cpu`
provides the required CPU smoke path. For 1,000-path one-step runs the local
neighbour count is safely capped at the 499 leave-one-out training references;
the 100,000- and 1,000,000-path runs use the configured 1,024 neighbours.

Reports and eigensurface figures are written below `implied/reports/` by
default. Simulated universes are cached in independently readable chunks keyed
by the complete versioned experiment configuration; million-path runs do not
materialize a second full universe or copy raw surface grids to a GPU.

The historical grid is already cleaned and interpolated by a statically
arbitrage-free SANOS fit. Consequently this benchmark compares generative
decoders on the same processed `mid_vols` input; it is not a raw OptionMetrics
interpolation comparison and it is not a scenario-reweighting method.

The checked-in `reports/pca-vs-sanos/smoke/` artifacts cover both innovation
factor counts and all four expiry-return modes, plus the secondary one-day
fixed-contract code path. Its aggregate comparison is deliberately marked
incomplete because no fully configuration-matched SANOS reports were supplied.
