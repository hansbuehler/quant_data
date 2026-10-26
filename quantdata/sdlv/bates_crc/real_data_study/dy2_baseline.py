r"""Model 1: the repository's dy2 static decoder.

`day_loader.load_day` already runs `smooth_interpolated_market` (dy2's own
production per-day pipeline: filter -> MkArbFree(smoothness=0.25,
no_arb_mode="surface") -> ArbFreePricer), so this module is a thin wrapper
around `DayBook.dy2_af_pricer` exposing the same `.price(strikes,
expiries)` interface used by every other model here.
"""

from __future__ import annotations

import numpy as np

from quantdata.sdlv.bates_crc.real_data_study.day_loader import DayBook


def price_dy2(book: DayBook, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    prices = book.dy2_af_pricer.price(expiries=expiries, strikes=strikes, and_vols=False, mode="surface")
    return np.asarray(prices, dtype=float)
