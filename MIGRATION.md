# openqai monorepo migration

Three independent distributions sharing one PEP 420 namespace `openqai`:

| Distribution        | Directory     | Package root                          | Depends on            |
|---------------------|---------------|---------------------------------------|-----------------------|
| `openqai-core`      | `core/`       | `core/src/openqai/core/`              | —                     |
| `openqai-data`      | `data/`       | `data/src/openqai/data/`              | core                  |
| `openqai-benchmarks`| `benchmarks/` | `benchmarks/src/openqai/benchmarks/`  | core, data            |

Dependencies flow one direction only: `core <- data <- benchmarks`.

**Rule that keeps the namespace working:** none of the three `src/openqai/`
directories may contain an `__init__.py`. Only the leaf packages (`openqai/core/`,
`openqai/data/`, `openqai/benchmarks/`) have one. Add `__init__.py` at the
`openqai/` level and the sibling distributions vanish at import time.

Install all three editable for development:

    uv pip install -e ./core -e ./data -e ./benchmarks

---

## File moves

### -> openqai.core  (from quantdata/)

| From                         | To                                       | Notes                                        |
|------------------------------|------------------------------------------|----------------------------------------------|
| `__init__.py` (Environment)  | `core/.../core/env.py`                   | 308L: Environment, Locations, `env` singleton, cvxpy patch, warning setup |
| `pool.py`                    | `core/.../core/pool.py`                  |                                              |
| `biz_calendar.py`            | `core/.../core/calendar.py`              | **minus `NormalizedStrikes`**                |
| `biz_calendar.py::NormalizedStrikes` | `core/.../core/state.py`         | surface state x_t — split out                |
| `utils/pl.py`                | `core/.../core/polars.py`                | **rename** — it's polars, not P&L            |
| `utils/pd.py`                | `core/.../core/pandas.py`                | rename for symmetry                          |
| `utils/testing.py`           | `core/.../core/testing.py`               |                                              |
| `utils/cvxpy.py`             | `core/.../core/cvxpy.py`                 |                                              |
| `utils/backbone.py`          | `core/.../core/backbone/__init__.py`     | Backbone, BackboneNumpy                      |
| `torch/bbtorch.py`           | `core/.../core/backbone/torch.py`        |                                              |
| `jax/bbjax.py`               | `core/.../core/backbone/jax.py`          |                                              |
| `torch/__init__.py`          | `core/.../core/nn/torch.py`              | BaseEnvironment, BaseModel, helpers          |
| `jax/__init__.py`            | `core/.../core/nn/jax.py`                | copy-paste twin of the above — see TODO      |
| `torch/ffw.py`               | `core/.../core/nn/ffw.py`                |                                              |
| `jax/utility.py`             | `core/.../core/risk.py`                  | entropic u(x)=1-e^-x — the only risk measure |

### -> openqai.data  (from quantdata/)

| From             | To                                  |
|------------------|-------------------------------------|
| `widb/`          | `data/.../data/widb/`               |
| `massive/`       | `data/.../data/massive/`            |
| `yahoo/`         | `data/.../data/yahoo/`              |
| `fred/`          | `data/.../data/fred/`               |
| `tickers/`       | `data/.../data/tickers/`            |

### -> openqai.benchmarks  (from quantdata/)

| From                       | To                                          |
|----------------------------|---------------------------------------------|
| `sanos/`                   | `benchmarks/.../benchmarks/sanos/`          |
| `sanos/widb_sanos_day.py`  | stays in `sanos/` (this file is the adapter)|
| `ncde/`                    | `benchmarks/.../benchmarks/ncde/`           |
| `volsim/`                  | `benchmarks/.../benchmarks/volsim/`         |
| `volume_curves/`           | **UNDECIDED** — parked, see below           |

New empty packages (docstring only): `benchmarks/.../benchmarks/deephedging/`,
`deepbellman/`, `memm/`.

---

## Import rewrites

- `from quantdata import env, SubDir, Context, version`  (19 sites)
  -> `from openqai.core.env import env, SubDir, Context, version`
  Consider re-exporting these four from `openqai/core/__init__.py` so the sites
  become `from openqai.core import ...`.
- `from quantdata.biz_calendar import ...` -> `from openqai.core.calendar import ...`
  (but `NormalizedStrikes` -> `from openqai.core.state import NormalizedStrikes`)
- `from quantdata.utils.pl import ...` -> `from openqai.core.polars import ...`
- `from quantdata.utils.pd import ...` -> `from openqai.core.pandas import ...`
- `from quantdata.pool import ...` -> `from openqai.core.pool import ...`
- `from quantdata.torch import ...` -> `from openqai.core.nn.torch import ...`
- `from quantdata.jax import ...` -> `from openqai.core.nn.jax import ...`
- `from quantdata.<adapter> import ...` -> `from openqai.data.<adapter> import ...`
- `from quantdata.sanos... / .ncde... / .volsim...` -> `from openqai.benchmarks...`

PyCharm: rename `quantdata` -> the new locations with **Refactor > Move** /
**Rename** so references update. It will NOT touch the 397 imports across 44
notebooks — those are string edits (find/replace `quantdata.` per subtree).

---

## Must fix during the move — cross-distribution cycles

These are not style issues. They make the distributions un-installable or
circular once split.

1. **`widb/arbfree.py` (data) imports `sanos.arbfree.MkArbFree` (benchmarks).**
   This is `data -> benchmarks`, the wrong direction, and becomes a circular
   dependency between two published packages. Must invert: either lift the
   arb-free primitive `MkArbFree` down into `core`, or move whatever in
   `widb/arbfree.py` needs it up into `benchmarks/sanos`.
   (`sanos/widb_sanos_day.py` importing `widb` is the allowed direction —
   benchmarks -> data — and is fine.)

2. **`massive <-> tickers` cycle (both in data).** `tickers/__init__.py` imports
   `MassiveClient`; `massive/expiries.py` + `massive/tickers/{data,filtered_history}.py`
   import `_locations_tickers` and `EXPIRY` back out of `tickers`. Intra-data so
   it won't block install, but resolve it: `_locations_tickers` / `EXPIRY` are
   location + enum config — push them down to `core`, cycle dissolves.

---

## Undeclared dependencies (now declared — do not lose)

Previously worked only via transitive installs; explicit now:
- `sympy` (benchmarks) — `sanos/sanos_jax.py`
- `prettytable`, `requests`, `urllib3` (data) — massive, widb, fred

## Dropped / parked dependencies
- Removed from any distribution: `wandb`, `orbax`, `scikit-learn`, `ipynbname`,
  `highspy` — imported by no .py file. Parked under
  `openqai-benchmarks[training]` extra pending confirmation with Hans.

## TODO for follow-up (not this PR)
- `core/nn/torch.py` and `core/nn/jax.py` are copy-paste twins (identical
  `MetricTacker` typo, identical stray `from pyexpat import model`). Merge
  `BaseEnvironment`/`BaseModel` onto the `Backbone` abstraction. Decide with
  Hans whether jax stays at all. Track as a GitHub issue.
- `jax/bbjax.py:9` imports `BaseEnvironment` from `quantdata.torch` — a bug;
  the jax backend should not require torch. Fix when merging the twins.

---

## Tests

Split `tests/` to sit beside each distribution. Mapping by actual imports:

### core/tests/
- test_env.py, test_biz_calendar.py, test_cvxpy_patch.py
- test_pd.py, test_pl.py
- test_backbone.py, test_ffw.py, test_torch_helpers.py, test_trainer_checkpoint.py

### data/tests/
- test_fred_rates.py
- test_massive_options_contracts.py, test_massive_processing.py, test_trade_conds.py
- test_tickers.py, test_ticker_calendar.py
- fixtures: stored_data/raw/massive, stored_data/raw/widb, stored_data/processing/widb

### benchmarks/tests/
- test_arbfree.py, test_arbfree_density_projection.py, test_sanos_utils.py
- test_ncde_voluvol_generate.py
- fixture: stored_data/training/grid_sanos_four_samples.h5

### conftest.py
`tests/conftest.py` creates the stored_data dirs and sets `QUANTDATA_TEST_MODE=1`.
Both data and benchmarks tests need it. Options: duplicate a small conftest into
each, or keep one at the workspace root. The `QUANTDATA_*` env var names live in
`core/env.py` — rename to `OPENQAI_*` there and in every test in the same pass,
or leave as-is for now (they still work). test_env.py hard-codes the names.
