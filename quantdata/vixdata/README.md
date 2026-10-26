# `quantdata.vixdata`

## Dataset contract

**Dataset:** official Cboe daily VIX OHLC history.  It is used as an
independent market-level benchmark for 30-calendar-day S&P 500 implied
variance, including the DYSANOS VIX comparison.  It is not an option-quote
dataset and must not be used as a substitute for the WRDS option surface.

**Canonical location:** resolve the location through
`quantdata.env.locations("vixdata", fmt_raw=SubDir.POLARS_PARQUET).raw`; do
not hard-code a data-root path.  On the current production configuration the
two cache artifacts are:

| Artifact | Production path |
| --- | --- |
| Data | `/mnt/nas/quantdata/vixdata/cboe/vix_eod.prq` |
| Provenance manifest | `/mnt/nas/quantdata/vixdata/cboe/vix_eod_metadata.json` |

The manifest records the source URL, UTC retrieval time, observation count,
date range, and units.  The data and manifest are deliberately external to
Git; this README is the repository-tracked contract other agents should use.

**Schema:** `date` (`Date`), `open`, `high`, `low`, and `close` (`Float64`).
Levels are annualized implied-volatility percentage points.  `close` is the
official end-of-day series used by current DYSANOS diagnostics.

**Source and refresh:** Cboe's public
[`VIX_History.csv`](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv).
Use `quantdata.vixdata.refresh_cboe_vix()` to refresh and
`load_cboe_vix(start, end)` to read the inclusive date interval.  The loader
validates the source schema, positive levels, and unique dates.  Some early
Cboe rows have anomalous intraday high/low fields; valid closing levels are
retained.
