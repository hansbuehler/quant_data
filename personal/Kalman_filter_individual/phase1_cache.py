"""
Phase 1 cache for the single-strike (individual) IV Kalman filter.

Runs the expensive, hyperparameter-independent part of the pipeline once —
fetching quotes + matched trades per option, estimating the implied forward, and
computing every implied vol in one batched cdxcore.bs pass (build_iv_frames) —
and serialises the result to disk.  iv_kf_sweep.py then loads the cache and runs
only the cheap per-ticker IV Kalman (Phase 3) for every (batch_seconds, q_scale)
combination, so the slow data load is paid exactly once.

Usage
-----
    python phase1_cache.py [--date 2024-01-03] [--out phase1_cache_2024-01-03.pkl]
"""

import argparse
import os
import pickle
import sys
from datetime import date, datetime

import polars as pl

project_root = "/Users/elie/Documents/GitHub/quant_data/"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Match the environment the Kalman_filter_individual notebooks run under.
os.environ.setdefault("QUANTDATA_RAW_DATA", "!/quantdata/raw")
os.environ.setdefault("QUANTDATA_TRAINING_DATA", "!/quantdata/training")
os.environ.setdefault("QUANTDATA_PROCESSING_DATA", "!/quantdata/processing")
os.environ.setdefault("QUANTDATA_CACHE_MODE", "on")
os.environ.setdefault("QUANTDATA_TEST_MODE", "0")

import personal.shared_functions.kalman_filters as kalman_filters
from quantdata.massive.client import MassiveClient

# ── Phase 1 config ──────────────────────────────────────────────────────────
TICKER = "SPY"
TRADE_DATE = date(2024, 1, 3)
MIN_TRADES = 100
SIMPLE = False  # estimate the implied forward via put-call parity (notebook default)
DISCOUNT_FACTOR_CLIP = None  # matches sanos_dlv_intraday / notebook (no clip)


def _tickers_path(trade_date: date) -> str:
    """Prefer the date-specific ticker file, fall back to the generic one."""
    dated = os.path.join(project_root, "personal", f"spy_options_tickers_{trade_date}.csv")
    generic = os.path.join(project_root, "personal", "spy_options_tickers.csv")
    return dated if os.path.exists(dated) else generic


def main(trade_date: date, out_path: str) -> None:
    df_tickers = pl.read_csv(_tickers_path(trade_date))
    client = MassiveClient()
    df_spy = client.fetch_trades(TICKER, trade_date)

    print(f"Building IV frames for {trade_date} ({df_tickers.height} tickers) ...")
    all_quotes_iv, all_trades_iv, ticker_meta = kalman_filters.build_iv_frames(
        df_tickers=df_tickers,
        trade_date=trade_date,
        df_spy=df_spy,
        min_trades=MIN_TRADES,
        simple=SIMPLE,
        discount_factor_clip=DISCOUNT_FACTOR_CLIP,
    )

    # datetime expiries are pickled as-is; keep meta as a plain list of tuples.
    cache = dict(
        all_quotes_iv=all_quotes_iv,
        all_trades_iv=all_trades_iv,
        ticker_meta=[(float(k), bool(c), e) for (k, c, e) in ticker_meta],
        phase1_config=dict(
            ticker=TICKER,
            trade_date=str(trade_date),
            min_trades=MIN_TRADES,
            simple=SIMPLE,
            discount_factor_clip=DISCOUNT_FACTOR_CLIP,
        ),
    )

    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_q = all_quotes_iv.height
    n_t = all_trades_iv.height
    print(
        f"Done. {len(cache['ticker_meta'])} contracts, "
        f"{n_q:,} quote rows / {n_t:,} matched-trade rows → {out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(TRADE_DATE), help="trade date YYYY-MM-DD")
    parser.add_argument(
        "--out", default=None, help="output pickle (default: phase1_cache_{date}.pkl)"
    )
    args = parser.parse_args()
    td = datetime.strptime(args.date, "%Y-%m-%d").date()
    out = args.out or f"phase1_cache_{td}.pkl"
    main(td, out)
