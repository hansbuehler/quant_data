# Local Vol by SANOS — migration status

Snapshot prepared: 2026-08-25 UTC

## Post-snapshot location update

Later on 2026-08-25, `/home/hans/Python3/quant_data` became inaccessible, while the
canonical checkout reappeared at `/home/hans/python/quant_data`. The project and all
five transfer artifacts are now together in its correct directory,
`quantdata/dysanos/localvolbysanos/`. The unrelated `/home/hans/python/aae` repository is
not the project destination. The archived snapshot remains based on the commit
recorded below.

## Critical transfer warning

The relocation from `aae` to `quantdata.dysanos.localvolbysanos` is complete in the
working tree but is not yet committed. Until the move is committed, a fresh clone
will not reproduce it. Transfer either the complete current project directory or
the accompanying project archive in addition to a compatible `quant_data` clone.

The project archive has been rebuilt after the relocation and therefore restores
the new `quantdata/dysanos/localvolbysanos/` layout directly. The separate market
cache archive is unchanged.

## Project state

The implementation, documentation, figures, video, Parquet panels, JSON reports,
and 2025 trading outputs are present. The project archive contains about 50 MB of
uncompressed project material; transient `__pycache__` files are excluded.

Primary documentation:

- `quantdata/dysanos/localvolbysanos/LOCALVOLBYSANOS.md`: project definition and reproduction
- `quantdata/dysanos/localvolbysanos/RESULTS.md`: DLV grid, recovery, fitting, PCA, and dynamics results
- `quantdata/dysanos/localvolbysanos/real_trading/REAL_TRADING.md`: OCE, realized trading, delta attribution, diagnostics, and reversion counterfactuals
- `quantdata/dysanos/localvolbysanos/bid_ask_estimate/BID_ASK_ESTIMATE.md`: superseded bid/ask-estimation subproject notes

The implementation is self-contained and does not import from `aae`. Its market
loader is now project-owned at `quantdata/dysanos/localvolbysanos/markets.py`.

## Core experiment specification

- Model expiries: `{2, 5, 10, 15, 20, 30, 40, 50, 70, 120} / DAYS_PER_YEAR`.
- Strikes are forward-relative.
- The reference log-strike grid merges grids scaled from `T_M = 1Y` and `5/255`,
  uses the relevant ATM volatility, removes near-duplicates, and contains 1.
- The later full-reference-grid experiment uses the same reference strike grid at
  every expiry so each sample has the same number and interpretation of DLVs.
- DLV prices recover the input surface without the earlier cross-expiry interpolation
  concern; the saved diagnostics check numerical stability and market-price recovery.
- The 2025 panel contains 163 usable market dates from 2025-01-02 through 2025-08-29.
  2025-01-08 is rejected as a bad market and 2025-07-03 is unavailable.

## Latest completed analysis: reversion contribution to the OCE delta signal

Implemented in:

- `quantdata/dysanos/localvolbysanos/real_trading/surface_reversion_attribution.py`
- `quantdata/dysanos/localvolbysanos/real_trading/reversion_oce_counterfactual.py`

The controlled ablation compares the fitted AR surface generator against a frozen
surface generator (`G = 0`). Both arms use identical PCA state representation,
factor exposure coefficients, innovation covariance, Sobol scenario nodes, listed
option universe, bid/ask entry prices, objective, and constraints. Thus the measured
difference isolates the modeled conditional surface drift rather than scenario noise
or a different feasible portfolio.

Run audit:

- 99 chronological forecast origins after 60 training dates
- 1,782 OCE optimizations
- 0 optimization failures
- Three universes: `actual`, `normalised`, and `ITM`
- Three objectives: entropy, power, and CVaR 10%
- Continuous and digital $1,000,000 delta implementations
- Exit at mid; actual listed-option bid/ask is used at entry where applicable

For the normalised universe, enabling fitted reversion changes absolute net delta by
roughly 0.33 of the $1m cap, with AR/frozen delta correlations around 0.66–0.68,
sign agreement around 80–83%, and allocation overlap around 45–60%.

Five-session direct-entry, overlapping realized spot P&L ($1m delta cap):

| Objective | Sizing | Fitted AR | Frozen | AR − frozen | Paired t |
|---|---:|---:|---:|---:|---:|
| Entropy | continuous | $767.4k | $534.5k | $232.9k | 1.38 |
| Entropy | digital | $894.1k | $453.6k | $440.5k | 1.93 |
| Power | continuous | $807.3k | $483.3k | $324.0k | 1.95 |
| Power | digital | $889.1k | $476.2k | $412.8k | 1.78 |
| CVaR 10% | continuous | $565.1k | $342.3k | $222.7k | 1.13 |
| CVaR 10% | digital | $582.6k | $340.9k | $241.8k | 1.25 |

All 18 direct continuous/digital comparisons across the three universes and three
objectives have positive fitted-AR-minus-frozen realized P&L. Two expiry-sliced
observations require the missing 2025-07-03 spot; they are explicitly audited and
not imputed. Direct-entry comparisons retain all 99 forecast origins.

Interpretation: the controlled result supports the narrow claim that modeled surface
reversion contributes to OCE spot timing in this 2025 sample. It is not yet robust
out-of-sample evidence: returns overlap, several specifications were inspected, and
the sample covers only part of one year. Earlier direct surface diagnostics also
show that the AR grid forecast did not beat persistence and that predictable spot
state was weak, even though realized contemporaneous surface shocks were strongly
related to spot.

Latest report:

`quantdata/dysanos/localvolbysanos/real_trading/results/2025/reversion_oce_counterfactual/reversion_oce_counterfactual_report.json`

## Verification at snapshot time

Focused tests were rerun from the surviving read-only tree with bytecode and pytest
cache writes disabled:

```bash
PYTHONPATH=/home/hans/Python3/quant_data \
PYTHONDONTWRITEBYTECODE=1 \
/home/hans/vqdata/bin/python -m pytest -q -p no:cacheprovider \
  /home/hans/Python3/quant_data/tests/test_localvolbysanos_real_trading.py
```

Result: `32 passed, 1 warning in 2.80s`.

The latest report was also read successfully and records 99 completed dates, zero
failures, and 36 missing expiry-sliced portfolio horizons (18 portfolios for each of
the two affected dates).

## Software snapshot

- Python 3.13.7
- NumPy 2.4.6
- pandas 2.2.3
- SciPy 1.17.1
- PyArrow 24.0.0
- scikit-learn 1.9.0
- Matplotlib 3.11.0
- Existing virtual environment: `/home/hans/vqdata`

The base repository's `pyproject.toml` and its normal editable installation provide
the remaining project dependencies, including `cdxcore`.

## Data required to reproduce

Current environment variables:

```text
QUANTDATA_RAW_DATA=/mnt/aae/quantdata
QUANTDATA_PROCESSING_DATA=/media/storage/quantdata/processing
QUANTDATA_TRAINING_DATA=/media/storage/quantdata/training
QUANTDATA_EXPERIMENTS=/media/storage/quantdata/experiments
QUANTDATA_ROOT=/media/storage/quantdata
```

The exact smooth SPX market cache used by the project contains 392 HDF5 files,
about 255 MB, spanning 2022-01-03 through 2025-08-29:

```text
dysanos/market/bid=0.011, nat,eth=10,bth=4,mps=1e-06,m d872d290/
  sm_0.25 6b930b59/SPX/
```

It is supplied as a separate archive. Restore that relative hierarchy under the new
machine's `QUANTDATA_TRAINING_DATA` directory.

The exact smooth-market cache plus the project bundle is enough to rerun the surface
and latest counterfactual pipeline. Rebuilding the actual listed-option panels from
scratch (`include_contract_details=True` and `realized.py`) may also require the WIDB
processing/raw option store. All currently generated listed-option and realized
outputs are already included in the project archive and remain available for
inspection and downstream analyses without that larger raw store.

## Restore procedure on the new box

1. Clone the full `quant_data` repository and check out commit
   `f2832033d9446d994069e86fbf88e99d9eae6150` (or deliberately port onto a newer
   compatible commit).
2. Extract `localvolbysanos_project_20260825.tar.gz` at the repository root. It
   restores `quantdata/dysanos/localvolbysanos/` and the focused test directly in
   the relocated namespace. The manifest and style-guide patch are included inside
   the project directory.
3. Apply `STYLE_GUIDE.localvolbysanos.patch` from the repository root, unless that
   isolation rule already exists.
4. Extract `localvolbysanos_spx_market_cache_d872d290_20260825.tar.gz` underneath
   the new `QUANTDATA_TRAINING_DATA` directory.
5. Create/install the repository's standard Python environment (editable install),
   or replicate `/home/hans/vqdata` using the repository's dependency metadata.
6. Set the `QUANTDATA_*` environment variables for the new storage layout and use
   `QUANTDATA_CACHE_MODE=cacheonly` for an exact cached-input rerun.
7. Run the focused test command below from the repository root.

```bash
python -m pytest -q tests/test_localvolbysanos_real_trading.py
```

Reproduce the latest controlled analysis:

```bash
MPLCONFIGDIR=/tmp/aae_localvol_matplotlib \
QUANTDATA_CACHE_MODE=cacheonly \
python -m quantdata.dysanos.localvolbysanos.real_trading.reversion_oce_counterfactual
```

The other reproduction commands and output locations are documented in the three
primary project Markdown files listed above.

## Recommended next action

Freeze the current choices before extending the sample: expiry grid, strike grid,
PCA rank, AR fitting window, objective parameters, universes, $1m cap, entry/exit
rules, and the fitted-vs-frozen comparison. Then run the identical controlled
counterfactual on another year. That is the cleanest next test of whether the
positive reversion contribution is persistent rather than a 2025 selection effect.
