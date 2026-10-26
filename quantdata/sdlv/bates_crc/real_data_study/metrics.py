r"""Section C metrics: bid-ask coverage, spread-unit distance (reusing this
codebase's own documented `min_half_spread=1e-4` floor convention from
`quantdata/sanos/sanos_tools.py::_compute_market_fit`), price/IV errors by
maturity/moneyness bucket, and runtime/memory. One function, reused
identically for every model, so numbers are directly comparable.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
from cdxcore.bs import bs

from quantdata.sdlv.bates_crc.real_data_study.day_loader import EvaluationContract

MIN_HALF_SPREAD = 1.0e-4  # matches quantdata/sanos/arbfree.py ArbFreePricer.min_half_spread exactly


@dataclass(frozen=True)
class PerContractResult:
    expiry_years: float
    strike: float
    bid: float
    ask: float
    price: float
    fit_err: float  # spread-unit hinge error, see MIN_HALF_SPREAD
    inside_bid_ask: bool
    market_mid_iv: float | None
    model_iv: float | None
    iv_error: float | None  # model_iv - market_mid_iv, None if either inversion failed


@dataclass(frozen=True)
class FitMetrics:
    model_name: str
    n_contracts: int
    pcnt_inside_bid_ask: float
    median_fit_err: float  # among contracts outside bid/ask; 0 if none
    mean_fit_err: float
    median_total_fit_err: float  # among ALL contracts, including zeros
    mean_iv_error_bps: float | None
    median_abs_iv_error_bps: float | None
    n_iv_inversions_failed: int
    runtime_seconds: float
    peak_memory_bytes: int
    per_contract: list  # list[PerContractResult]
    by_bucket: dict  # {(tenor_bucket_label, moneyness_bucket_label): FitMetrics-like summary dict}


def _iv_or_none(k: np.ndarray, prices: np.ndarray, sqrtT: np.ndarray) -> np.ndarray:
    result = bs.implied(k=k, prices=prices, is_call=True, sqrtT=sqrtT, ret_only_vols=False,
                         on_exceed_bounds="quiet")
    vols = np.asarray(result.vols, dtype=float)
    failed = np.asarray(result.failed, dtype=bool)
    vols = np.where(failed, np.nan, vols)
    return vols


def _tenor_bucket(tau: float) -> str:
    if tau <= 45 / 365.25:
        return "short (<=1.5m)"
    if tau <= 180 / 365.25:
        return "medium (1.5-6m)"
    return "long (>6m)"


def _moneyness_bucket(k: float) -> str:
    if k < 0.95:
        return "OTM put wing (K/F<0.95)"
    if k > 1.05:
        return "OTM call wing (K/F>1.05)"
    return "near ATM (0.95-1.05)"


def evaluate_model(model_name: str, price_fn, contracts: list) -> FitMetrics:
    """`price_fn(strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray`."""
    strikes = np.array([c.strike for c in contracts])
    expiries = np.array([c.expiry_years for c in contracts])
    bids = np.array([c.bid for c in contracts])
    asks = np.array([c.ask for c in contracts])
    mids = 0.5 * (bids + asks)

    tracemalloc.start()
    t0 = time.perf_counter()
    prices = price_fn(strikes, expiries)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    half_spread = np.maximum(0.5 * (asks - bids), MIN_HALF_SPREAD)
    fit_err = (np.maximum(0.0, prices - asks) + np.maximum(0.0, bids - prices)) / half_spread
    inside = fit_err <= 0.0

    sqrtT = np.sqrt(expiries)
    market_iv = _iv_or_none(strikes, mids, sqrtT)
    model_iv = _iv_or_none(strikes, prices, sqrtT)
    iv_error = model_iv - market_iv

    per_contract = []
    for i, c in enumerate(contracts):
        me = market_iv[i] if np.isfinite(market_iv[i]) else None
        mo = model_iv[i] if np.isfinite(model_iv[i]) else None
        ive = float(iv_error[i]) if np.isfinite(iv_error[i]) else None
        per_contract.append(PerContractResult(
            expiry_years=c.expiry_years, strike=c.strike, bid=c.bid, ask=c.ask, price=float(prices[i]),
            fit_err=float(fit_err[i]), inside_bid_ask=bool(inside[i]),
            market_mid_iv=me, model_iv=mo, iv_error=ive,
        ))

    bad = fit_err > 0.0
    by_bucket: dict = {}
    for tenor_label in {"short (<=1.5m)", "medium (1.5-6m)", "long (>6m)"}:
        for money_label in {"OTM put wing (K/F<0.95)", "near ATM (0.95-1.05)", "OTM call wing (K/F>1.05)"}:
            sel = np.array([
                _tenor_bucket(c.expiry_years) == tenor_label and _moneyness_bucket(c.strike) == money_label
                for c in contracts
            ])
            if not np.any(sel):
                continue
            sel_iv = iv_error[sel]
            sel_iv = sel_iv[np.isfinite(sel_iv)]
            by_bucket[f"{tenor_label} / {money_label}"] = dict(
                n=int(sel.sum()),
                pcnt_inside_bid_ask=float(inside[sel].mean()),
                median_fit_err=float(np.median(fit_err[sel][bad[sel]])) if np.any(bad[sel]) else 0.0,
                mean_abs_iv_error_bps=float(np.mean(np.abs(sel_iv)) * 1e4) if sel_iv.size else None,
            )

    finite_iv_err = iv_error[np.isfinite(iv_error)]
    return FitMetrics(
        model_name=model_name, n_contracts=len(contracts),
        pcnt_inside_bid_ask=float(inside.mean()) if len(contracts) else float("nan"),
        median_fit_err=float(np.median(fit_err[bad])) if np.any(bad) else 0.0,
        mean_fit_err=float(np.mean(fit_err[bad])) if np.any(bad) else 0.0,
        median_total_fit_err=float(np.median(fit_err)) if len(contracts) else float("nan"),
        mean_iv_error_bps=float(np.mean(finite_iv_err) * 1e4) if finite_iv_err.size else None,
        median_abs_iv_error_bps=float(np.median(np.abs(finite_iv_err)) * 1e4) if finite_iv_err.size else None,
        n_iv_inversions_failed=int(np.sum(~np.isfinite(model_iv)) + np.sum(~np.isfinite(market_iv))),
        runtime_seconds=elapsed, peak_memory_bytes=int(peak),
        per_contract=per_contract, by_bucket=by_bucket,
    )
