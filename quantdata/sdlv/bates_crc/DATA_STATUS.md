# Data status: dy2's SPX dataset and checkpoints

## 1. What is, and is not, present in this environment

Checked directly, not assumed:

- No `WRDS_USERNAME`, `MASSIVE_TOKEN`, `QUANTDATA_ROOT`,
  `QUANTDATA_RAW_DATA`/`QUANTDATA_PROCESSING_DATA`/`QUANTDATA_TRAINING_DATA`/
  `QUANTDATA_EXPERIMENTS` environment variables are set (`env | grep -i
  "quantdata|wrds|massive"` returns nothing).
- No `~/.pgpass` file (the WRDS Option Metrics connector's own required
  credential file, per `README.md`'s setup instructions).
- No `.env` file anywhere in this checkout.
- The only locally usable data is `tests/stored_data/` — a small, sparse
  fixture set (a handful of scattered dates per identifier: 2005-11-21,
  2010-07-07, 2011-07-07, 2024-03-05, 2024-12-26, 2025-01-03 for
  secids `108105`/`101594`, plus a synthetic `TEST` ticker with
  `2025-06-03`). **Not** a continuous multi-year weekly series — confirmed
  directly by listing `tests/stored_data/raw/widb/options/`.
- **No trained checkpoint** for the weekly dynamics or joint decoder model
  exists anywhere in this checkout (only `flex/reports/phase3-spread/
  feed_forward.pt`, an unrelated spread-cost regression — see
  `REPOSITORY_MAP.md` Section 8).
- `flex/reports/batch-descent/spx-2022-2025-forward-mu-v1*.{md,json}`
  contains **summary metrics only** (aggregated tables: `3,492,782` total
  options in its "all" bucket — close to, not identical to, `dy2.pdf`'s
  reported "3.57 million quotes", likely a different filtered snapshot of
  the same underlying run), **not** raw quotes and **not** a loadable model
  state. Confirmed by reading the file directly.

**Conclusion: the original `dy2.pdf` SPX quote dataset and its trained
checkpoint are not available in this environment.**

## 2. What is needed to obtain them (the exact missing-data request)

The codebase is already built, specifically and only, for two authorised
sources (`README.md` Section "Data Access"):

1. **WRDS Option Metrics** (the source `dy2.pdf` almost certainly uses,
   given the SPX/index-option focus and the authors' academic affiliation):
   requires (a) a WRDS account with the Option Metrics dataset entitlement,
   (b) two-factor authentication enabled, (c) a `~/.pgpass` file per the
   WRDS Python instructions linked in `README.md`, and (d)
   `WRDS_USERNAME` set to that account's username.
2. **Massive.com**: requires an account and `MASSIVE_TOKEN`.

**Exact request**: to reproduce `dy2.pdf`'s reported experiment, or run any
new comparison against real data, this environment needs either (1) or (2)
above, plus a populated `QUANTDATA_ROOT` (or the four granular
`QUANTDATA_*_DATA` variables) pointing at read/write storage for the raw/
processed/training/experiment caches. None of this can be substituted by
fabricating quotes or by reconstructing bid/ask from an implied-volatility
curve — both explicitly disallowed by the assignment and avoided here.

Absent that, the **next best already-authorised option** is to locate and
mount an existing populated `QUANTDATA_ROOT` (e.g. a shared drive or prior
machine the user has already configured, per `AGENTS.md`'s own reference to
a OneDrive-hosted working tree) rather than re-fetching from WRDS/Massive
from scratch — this should be checked by the user before any fresh
(re-)download is attempted, since WRDS downloads are rate/volume-limited
and `DATA-MANAGEMENT.md` explicitly warns that a full re-pipelining event
"can be very tedious."

## 3. The single-snapshot vs. historical-dynamics distinction (recorded
explicitly, per the assignment's own instruction)

If real data becomes available, a **single day's quote snapshot** (e.g. one
`smooth_interpolated_market(ticker, d)` call) permits only a **static
calibration comparison**: does each candidate decoder (dy2's fitted state,
coherent DLV x Black, DLV x ordinary Bates, DLV x extended-Bates+CRC) fit
that one day's book within bid/ask, at what smoothing parameter, with what
DLV positivity margin. It does **not** permit any claim about historical
dynamics, transition likelihood, or the martingale/HJM residual tests
central to Sections 2-3 of the prior CRC-extension task — those require a
genuine **chronological sequence** of dates (the weekly-stride requirement
in `WeeklyDynamicsConfig`), which in turn requires real multi-date access,
not a single pull. This distinction is recorded here so that a future single
successful data pull is not mistakenly treated as sufficient for the full
comparison programme.

## 4. Import adapter (built here, usable once real data is supplied)

`quantdata/sdlv/bates_crc/quote_import.py` defines a **schema-validated**
import path for real option-quote data that does *not* originate from the
WRDS/Massive connectors (e.g. a CSV export the user already has authorised
access to, from WRDS or elsewhere). It:

- requires literal `bid`/`ask` columns (rejects any file that only supplies
  a single mid or implied-volatility column — no bid/ask reconstruction is
  performed, matching the assignment's explicit prohibition);
- requires `underlying`, `strike`, `expiry`, `valuation_date`, and either
  an explicit discount factor/forward pair or a deterministic zero-rate/
  dividend-yield pair from which one can be built (never silently defaults
  to zero carry for real data — zero carry is a modelling normalisation
  applied *after* import, per `MODEL_CONVENTIONS.md` Section 2, not an
  assumption about the raw quotes);
- converts into the same "pure" (`C/(DF*F)`, `K/F`) units as
  `quantdata.widb.day`'s own schema, so that downstream code
  (`smooth_interpolated_market`-style fitting, the FLEX adapters in
  `comparison_interface.py`) does not need to know whether a given day came
  from WRDS or this adapter;
- raises with an explicit, actionable message (not a silent `NaN` fill) on
  any missing required field, and separately reports which rows were
  dropped and why (a `QuoteImportReport`), mirroring the exclusion-reporting
  standard already used by `filter_options_day`/`OptionsFilter` elsewhere
  in this repository.

It has been tested only against synthetic, explicitly-labelled-synthetic
fixtures (`tests/test_bates_crc_quote_import.py`) — it has **not** been run
against any real data, since none is available here.

## 5. What continues without real data (per the assignment's own instruction
not to stop)

The mathematical implementation and synthetic validation work
(`MODEL_CONVENTIONS.md`, the comparison interface in
`comparison_interface.py`, and the prior task's `section2_pricing_identity.py`
/`section5_crc_sequence.py` machinery) all proceed on synthetic-but-
structurally-faithful data, exactly as the prior CRC-extension task already
established as the correct honest labelling convention in this environment.
