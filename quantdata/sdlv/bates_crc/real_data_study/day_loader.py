r"""Section A: shared data loading, cleaning, carry treatment, and the
common evaluation contract set used by EVERY model in this comparison.

load_day() (the only function that touches real data) requires
QUANTDATA_TEST_MODE=generate_tests (not "1") -- see DATA_PROVENANCE.md
Section 1/6 for why. The dataclasses in this module (DayBook,
EvaluationContract, EvaluationSet) and build_evaluation_set() have no such
requirement and are usable under plain QUANTDATA_TEST_MODE=1 (e.g. to build
synthetic evaluation sets for calibrator unit tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import numpy as np

from quantdata.dysanos.market import smooth_interpolated_market
from quantdata.sanos.arbfree import ArbFreePricer
from quantdata.sdlv.bates_crc.real_data_study import config as cfg


@dataclass(frozen=True)
class DayBook:
    """One day's real, cleaned, pure-unit ('pure' = C/(DF*F), K/F) option
    book, plus dy2's own fitted pricer for that day."""

    ticker: str
    d: date
    spot_ref: float
    fwd_over_spot: float  # market.fwd/spot_ref is not separately exposed; forward is implicit in pure units (F=1)
    expiries: np.ndarray  # (T,) tte_y, ascending
    strikes: list  # T arrays of pure strike K/F
    bids: list  # T arrays of pure call-equivalent bid, C/(DF*F)
    asks: list  # T arrays of pure call-equivalent ask
    model_masks: list  # T boolean arrays: dy2's own post-arbfree-fit usability mask
    dy2_af_pricer: ArbFreePricer
    dy2_fit_result: object
    n_raw_contracts: int


@dataclass(frozen=True)
class EvaluationContract:
    expiry_years: float
    target_tenor_years: float
    target_moneyness: float
    strike: float  # pure K/F
    bid: float
    ask: float
    dy2_model_mask: bool
    log_moneyness_error: float


@dataclass(frozen=True)
class EvaluationSet:
    ticker: str
    d: date
    contracts: list  # list[EvaluationContract]
    excluded: list  # list[dict] -- target grid points with no acceptable match


def load_day(ticker: str, d: date, smoothness: float = cfg.DY2_SMOOTHNESS) -> DayBook:
    if os.environ.get("QUANTDATA_TEST_MODE") != "generate_tests":
        raise RuntimeError(
            "load_day requires QUANTDATA_TEST_MODE=generate_tests (not '1') -- "
            "'1' hard-codes cache_mode='cacheonly' and cannot compute the derived "
            "per-day SANOS fit even from fully-cached raw inputs. See DATA_PROVENANCE.md."
        )
    raw = smooth_interpolated_market(ticker, d, smoothness=smoothness)
    if not raw:
        raise ValueError(f"smooth_interpolated_market returned no data for {ticker} {d}")
    market = raw["market"]
    af_pricer = ArbFreePricer.from_dict(raw["af_pricer"])
    n_raw = int(sum(len(s) for s in market["strikes"]))
    return DayBook(
        ticker=ticker, d=d, spot_ref=float(raw["spot_ref"]), fwd_over_spot=1.0,
        expiries=np.asarray(market["expiries"], dtype=float),
        strikes=[np.asarray(s, dtype=float) for s in market["strikes"]],
        bids=[np.asarray(b, dtype=float) for b in market["bids"]],
        asks=[np.asarray(a, dtype=float) for a in market["asks"]],
        model_masks=[np.asarray(m, dtype=bool) for m in market["model_masks"]],
        dy2_af_pricer=af_pricer, dy2_fit_result=raw["fit_result"], n_raw_contracts=n_raw,
    )


def build_evaluation_set(book: DayBook, *, target_tenors: tuple = cfg.TARGET_TENORS_YEARS,
                          target_moneyness: tuple = cfg.TARGET_MONEYNESS,
                          log_moneyness_tol: float = cfg.LOG_MONEYNESS_TOLERANCE) -> EvaluationSet:
    contracts: list[EvaluationContract] = []
    excluded: list[dict] = []

    used_expiry_idx: set[int] = set()
    tenor_to_expiry_idx: dict[float, int] = {}
    for tenor in target_tenors:
        idx = int(np.argmin(np.abs(book.expiries - tenor)))
        tenor_to_expiry_idx[tenor] = idx

    for tenor, idx in tenor_to_expiry_idx.items():
        expiry_years = float(book.expiries[idx])
        strikes = book.strikes[idx]
        bids = book.bids[idx]
        asks = book.asks[idx]
        masks = book.model_masks[idx]
        if strikes.size == 0:
            for m in target_moneyness:
                excluded.append(dict(target_tenor=tenor, target_moneyness=m, reason="no quotes at nearest expiry"))
            continue
        log_k = np.log(strikes)
        sane = np.isfinite(bids) & np.isfinite(asks) & (asks > bids) & (bids >= 0.0)
        seen_strike_idx: set[int] = set()

        for m in target_moneyness:
            target_log_k = np.log(m)
            diffs = np.abs(log_k - target_log_k)
            diffs = np.where(sane, diffs, np.inf)
            j = int(np.argmin(diffs))
            if not np.isfinite(diffs[j]) or diffs[j] > log_moneyness_tol:
                excluded.append(dict(target_tenor=tenor, target_moneyness=m,
                                      matched_expiry_years=expiry_years,
                                      reason=f"nearest sane quote {diffs[j]:.4f} log-moneyness away (tol={log_moneyness_tol})"))
                continue
            if j in seen_strike_idx:
                excluded.append(dict(target_tenor=tenor, target_moneyness=m,
                                      matched_expiry_years=expiry_years,
                                      reason="duplicate: same real quote already matched to a nearer target moneyness"))
                continue
            seen_strike_idx.add(j)
            contracts.append(EvaluationContract(
                expiry_years=expiry_years, target_tenor_years=tenor, target_moneyness=m,
                strike=float(strikes[j]), bid=float(bids[j]), ask=float(asks[j]),
                dy2_model_mask=bool(masks[j]) if masks.size else False,
                log_moneyness_error=float(diffs[j]),
            ))
            used_expiry_idx.add(idx)

    return EvaluationSet(ticker=book.ticker, d=book.d, contracts=contracts, excluded=excluded)
