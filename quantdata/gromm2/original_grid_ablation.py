"""Build the original GROMM quote book around public SANOS midpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quantdata.gromm2.market import load_public_sanos_days, public_sanos_prices

FloatArray = NDArray[np.float64]


def reconstruct_quote_expiries(
    *,
    model_times_y: FloatArray,
    lower: NDArray[np.integer],
    upper: NDArray[np.integer],
    alpha: FloatArray,
    before_first: NDArray[np.bool_],
) -> FloatArray:
    """Recover the exact quote expiries encoded by the original interpolation."""
    times = np.asarray(model_times_y, dtype=np.float64)
    lower_indices = np.asarray(lower, dtype=np.intp)
    upper_indices = np.asarray(upper, dtype=np.intp)
    weights = np.asarray(alpha, dtype=np.float64)
    before = np.asarray(before_first, dtype=bool)
    if not (
        lower_indices.shape
        == upper_indices.shape
        == weights.shape
        == before.shape
    ):
        raise ValueError("quote interpolation arrays must have identical shapes")
    previous = np.where(before, 0.0, times[lower_indices])
    expiries = previous + weights * (times[upper_indices] - previous)
    if np.any(~np.isfinite(expiries)) or np.any(expiries < 0.0):
        raise ValueError("reconstructed quote expiries are invalid")
    return expiries


def center_bands(
    *,
    synthetic_midpoints: FloatArray,
    market_bids: FloatArray,
    market_asks: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Center the original market widths on synthetic SANOS midpoints."""
    midpoints = np.asarray(synthetic_midpoints, dtype=np.float64)
    bids = np.asarray(market_bids, dtype=np.float64)
    asks = np.asarray(market_asks, dtype=np.float64)
    half_spreads = 0.5 * (asks - bids)
    if (
        midpoints.shape != bids.shape
        or bids.shape != asks.shape
        or np.any(~np.isfinite(midpoints))
        or np.any(~np.isfinite(half_spreads))
        or np.any(half_spreads <= 0.0)
    ):
        raise ValueError("invalid midpoint or market-spread arrays")
    return midpoints - half_spreads, midpoints + half_spreads


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_original_grid_sanos_book(
    *,
    market_book: Path,
    public_dataset: Path,
    output: Path,
) -> dict[str, object]:
    """Replace original market centers by SANOS prices, preserving widths."""
    with np.load(market_book, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    if str(payload["schema"].item()) != "daily25-jax-market-v1":
        raise ValueError(f"unexpected original market-book schema in {market_book}")
    dates = payload["dates"].astype("U10")
    days = {
        day.trading_date.isoformat(): day
        for day in load_public_sanos_days(public_dataset)
    }
    if any(date not in days for date in dates):
        raise ValueError("public SANOS history does not cover every market-book date")

    quote_expiries = reconstruct_quote_expiries(
        model_times_y=payload["model_times_y"],
        lower=payload["lower"],
        upper=payload["upper"],
        alpha=payload["alpha"],
        before_first=payload["before_first"],
    )
    synthetic_mids = np.asarray(payload["mids"], dtype=np.float64).copy()
    mask = np.asarray(payload["mask"], dtype=bool)
    for index, date in enumerate(dates):
        active = mask[index]
        calls = public_sanos_prices(
            days[date],
            expiries_y=quote_expiries[index, active],
            strikes=np.asarray(payload["strikes"][index, active], dtype=np.float64),
        )
        synthetic_mids[index, active] = np.where(
            payload["is_call"][index, active],
            calls,
            calls - 1.0 + payload["strikes"][index, active],
        )

    active_mids = synthetic_mids[mask]
    active_bids, active_asks = center_bands(
        synthetic_midpoints=active_mids,
        market_bids=payload["bids"][mask],
        market_asks=payload["asks"][mask],
    )
    payload["mids"] = synthetic_mids.astype(np.float32)
    payload["bids"] = np.asarray(payload["bids"], dtype=np.float32).copy()
    payload["asks"] = np.asarray(payload["asks"], dtype=np.float32).copy()
    payload["bids"][mask] = active_bids.astype(np.float32)
    payload["asks"][mask] = active_asks.astype(np.float32)
    _atomic_npz(output, payload)

    report: dict[str, object] = {
        "schema": "gromm2-original-grid-sanos-midpoints-v1",
        "source_market_book": str(market_book.resolve()),
        "source_public_dataset": str(public_dataset.resolve()),
        "output": str(output.resolve()),
        "date_count": int(len(dates)),
        "quote_count": int(np.count_nonzero(mask)),
        "construction": (
            "original GROMM quote locations and market half-spreads centered "
            "on published SANOS prices"
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-book", type=Path, required=True)
    parser.add_argument("--public-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_original_grid_sanos_book(
        market_book=arguments.market_book,
        public_dataset=arguments.public_dataset,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
