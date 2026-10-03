r"""The independently-fitted SANOS snapshot baseline: a per-expiry,
UNSMOOTHED arbitrage-free fit (MkArbFree(no_arb_mode="expiry",
smoothness=None)), run directly on the SAME cleaned book dy2's own fit
consumed. This isolates representation capacity from dy2's cross-expiry
SANOS-smoothness penalty and from the DLV-chain family's compression
(finite anchors + a scalar transition rate): if this snapshot fits far
better than dy2 or the chain models, the gap is attributable to smoothing/
compression, not to a lack of achievable arbitrage-free representations.
"""

from __future__ import annotations

import time

import numpy as np
from cdxcore.verbose import Context

from quantdata.sanos.arbfree import ArbFreePricer, MkArbFree

from quantdata.dysanos.bates_crc.real_data_study.day_loader import DayBook


def fit_sanos_snapshot(book: DayBook) -> tuple[ArbFreePricer, float]:
    """Returns (pricer, wall_clock_seconds)."""
    fitter = MkArbFree(no_arb_mode="expiry", smoothness=None)
    t0 = time.perf_counter()
    pricer = fitter(expiries=book.expiries, strikes=book.strikes, bids=book.bids, asks=book.asks,
                     verbose=Context.quiet)
    elapsed = time.perf_counter() - t0
    return pricer, elapsed


def price_sanos_snapshot(pricer: ArbFreePricer, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    prices = pricer.price(expiries=expiries, strikes=strikes, and_vols=False, mode="expiry")
    return np.asarray(prices, dtype=float)
