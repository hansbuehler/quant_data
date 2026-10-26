# Data provenance, date-selection rule, and the model-selection/held-out split

## 1. What data this study uses

Real WRDS OptionMetrics quotes, cached locally as WIDB `.pdq` fixtures under
`tests/stored_data/raw/widb/options/{secid}/{date}.pdq` (this checkout's only
locally-authorised real option data; see `../DATA_STATUS.md` for the full
search that established no other real source is reachable in this
environment — no `WRDS_USERNAME`/`MASSIVE_TOKEN`/`QUANTDATA_ROOT`, no
`~/.pgpass`). Two secids are present:

| secid | ticker | dates available |
|---|---|---|
| 108105 | SPX | 2005-11-21, 2010-07-07, 2011-07-07, 2024-03-05, 2024-12-26, 2025-01-03 |
| 101594 | AAPL | same six dates |

This is **not** `dy2.pdf`'s full multi-year WRDS series (that requires live
WRDS/Massive access, unavailable here) — it is a small, non-continuous,
6-date fixture set. Confirmed genuine (not synthetic): each `.pdq` loads via
`quantdata.dysanos.market.smooth_interpolated_market`, under
`QUANTDATA_TEST_MODE=generate_tests` (compute-and-persist against the local
fixture store only, zero network calls — **not** `QUANTDATA_TEST_MODE=1`,
which hard-codes `cache_mode="cacheonly"` and cannot compute the derived
per-day fit even from fully-cached raw inputs), into real bid/ask books
(587–24,412 contracts, 6–56 expiries per day; spot levels match the
historical record for each date, e.g. SPX 1254.85 on 2005-11-21 and 5078.65
on 2024-03-05).

**Every model in this study is fit to and evaluated against this same
loaded day-book** (`day_loader.load_day`): same cleaning
(`filter_options_day`'s own arbitrage/liquidity filters, already applied
inside `smooth_interpolated_market`), same "pure" carry treatment
(`C/(DF·F)`, `K/F`, discount factor and forward taken from the WIDB day's
own put-call-parity-implied `df`/`fwd` fields — never a separately-assumed
zero/dividend curve), same evaluation contract set (Section 3 below).

## 2. Date-selection rule (documented, applied before any calibration)

**Rule: use every date present in the authorised local fixture cache — no
cherry-picking.** This is the least-biased selection rule available (it
cannot be gamed after the fact, since it was fixed by what the environment
happens to contain, not by which dates look favourable), and it happens to
span a useful range of regimes without any manual curation:

| Date | Regime (documented from public record, not fit to make a point) |
|---|---|
| 2005-11-21 | Pre-crisis, low realised/implied vol, thin book (587 SPX contracts) |
| 2010-07-07 | Post-2008-crisis, still-elevated vol, European sovereign stress building |
| 2011-07-07 | Weeks before the US debt-ceiling/downgrade shock (Aug 2011) — elevated tail pricing |
| 2024-03-05 | Calm, dense modern book (22,022 SPX contracts, 54 expiries) |
| 2024-12-26 | Holiday-thin trading, low liquidity, modern book |
| 2025-01-03 | Recent, dense, first trading days of a new year |

## 3. Model-selection vs. held-out split (fixed *before* any calibration was run)

- **Selection dates** (used only to choose: number of chain anchors,
  parameter bounds, reserve-budget sweep grid, solver initial values and
  multi-start count, evaluation-grid tenors/moneyness): **2005-11-21,
  2010-07-07**.
- **Held-out evaluation dates** (final reported comparison numbers; no
  design choice made after seeing these numbers): **2011-07-07,
  2024-03-05, 2024-12-26, 2025-01-03**.

This split, and every hyperparameter choice it produced, is committed in
`config.py` before the held-out dates' calibrations were run.

## 4. Tickers

**SPX is the primary underlying** (index, matches `dy2.pdf`'s own focus,
and is the only one with genuinely dense modern books). **AAPL is a
secondary robustness check only** (single-name, smaller books) — run for
the held-out dates but not part of the primary reported comparison table,
to bound compute; reported separately in an appendix table.

## 5. Common evaluation contract set

For each day, a **fixed target grid** of `(tenor, moneyness)` pairs —
tenors `{1w, 1m, 2m, 3m, 6m, 1y}` in year-fractions, moneyness `K/F ∈
{0.85, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15}` — is
mapped onto the **nearest actually-available real quote**: nearest
expiry in the day's own `market.expiries` to each target tenor (deduplicated
if two tenors map to the same nearest expiry), then within that expiry the
real strike nearest in log-moneyness to each target moneyness, accepted
only if within `0.03` log-moneyness and the quote is finite with
`ask > bid ≥ 0`. **Every model is evaluated on exactly the same matched set
for a given day** — this is what "same evaluation contracts for every
model" means operationally here. Target grid points with no acceptable
match (e.g. very short-dated books on 2005-11-21 lacking a 1y tenor, or thin
moneyness wings on holiday-thin days) are recorded as **exclusions**, not
silently dropped — see each day's `evaluation_contracts.json` `excluded`
list.

Each model may additionally report **its own coverage/exclusions** beyond
this common set (e.g. dy2's `model_masks` may drop a contract the common
set retained because it had zero implied density in dy2's own arbitrage
filter; the DLV/Bates closed-form pricers never drop a contract). Both the
common-coverage numbers and each model's own additional exclusions are
reported side by side in the final report, per the task's explicit
instruction not to conflate the two.

## 6. Reproducibility

Running any script in `real_data_study/` requires
`QUANTDATA_TEST_MODE=generate_tests` (not `1`) — see Section 1. This will
create/refresh small derived-cache `.pdq` files under
`tests/stored_data/{processing,training}/`; consistent with this
repository's existing convention (similar derived caches from an earlier
task are already tracked in git), these are committed alongside the code
for reproducibility and to avoid repeating ~20s cvxpy solves.
