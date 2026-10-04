"""Bid/ask reconstruction metrics for FLEX capacity benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ExcessStatistics:
    """Containment and excess in observed half-spread units."""

    num_options: int
    num_zero: int
    percent_zero: float
    num_positive: int
    positive_mean: float
    positive_median: float
    surface_mean: float
    surface_median: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class ExpiryExcessStatistics:
    """All-option statistics plus the required 0DTE, 1DTE, and 2DTE cuts."""

    all: ExcessStatistics
    dte_0: ExcessStatistics
    dte_1: ExcessStatistics
    dte_2: ExcessStatistics

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Return a JSON-safe representation with stable bucket names."""
        return {
            "all": self.all.to_dict(),
            "0DTE": self.dte_0.to_dict(),
            "1DTE": self.dte_1.to_dict(),
            "2DTE": self.dte_2.to_dict(),
        }


@dataclass(frozen=True)
class SurfaceFitStatistics:
    """Whole-surface error against the smooth market fit."""

    num_options: int
    half_spread_mae: float
    half_spread_median_absolute_error: float
    normalized_price_mae: float
    normalized_price_median_absolute_error: float
    normalized_price_rmse: float
    normalized_price_bias: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe representation."""
        return asdict(self)


def normalized_excess(
    fit: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
) -> np.ndarray:
    """Compute exact distance outside bid/ask in observed half-spreads."""
    fit = np.asarray(fit, dtype=np.float64)
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    if fit.shape != bid.shape or bid.shape != ask.shape:
        raise ValueError(
            f"'fit', 'bid', and 'ask' shapes must match, found {fit.shape}, "
            f"{bid.shape}, and {ask.shape}"
        )
    half_spread = 0.5 * (ask - bid)
    if not np.all(np.isfinite(fit) & np.isfinite(bid) & np.isfinite(ask)):
        raise ValueError("'fit', 'bid', and 'ask' must be finite")
    if not np.all(half_spread > 0.0):
        raise ValueError("Every observed half-spread must be strictly positive")
    return (
        np.maximum(fit - ask, 0.0) + np.maximum(bid - fit, 0.0)
    ) / half_spread


def surface_fit_statistics(
    model_fit: np.ndarray,
    market_fit: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
) -> SurfaceFitStatistics:
    """Measure errors against ``market.py`` fit over every eligible quote."""
    model_fit = np.asarray(model_fit, dtype=np.float64).reshape(-1)
    market_fit = np.asarray(market_fit, dtype=np.float64).reshape(-1)
    bid = np.asarray(bid, dtype=np.float64).reshape(-1)
    ask = np.asarray(ask, dtype=np.float64).reshape(-1)
    if not (
        model_fit.shape == market_fit.shape == bid.shape == ask.shape
    ):
        raise ValueError(
            "'model_fit', 'market_fit', 'bid', and 'ask' shapes must match, "
            f"found {model_fit.shape}, {market_fit.shape}, {bid.shape}, and "
            f"{ask.shape}"
        )
    half_spread = 0.5 * (ask - bid)
    if not np.all(
        np.isfinite(model_fit)
        & np.isfinite(market_fit)
        & np.isfinite(bid)
        & np.isfinite(ask)
    ):
        raise ValueError("Whole-surface fit inputs must be finite")
    if not np.all(half_spread > 0.0):
        raise ValueError("Every observed half-spread must be strictly positive")

    error = model_fit - market_fit
    absolute_error = np.abs(error)
    absolute_half_spreads = absolute_error / half_spread
    num_options = int(len(error))
    if num_options == 0:
        return SurfaceFitStatistics(
            num_options=0,
            half_spread_mae=0.0,
            half_spread_median_absolute_error=0.0,
            normalized_price_mae=0.0,
            normalized_price_median_absolute_error=0.0,
            normalized_price_rmse=0.0,
            normalized_price_bias=0.0,
        )
    return SurfaceFitStatistics(
        num_options=num_options,
        half_spread_mae=float(np.mean(absolute_half_spreads)),
        half_spread_median_absolute_error=float(
            np.median(absolute_half_spreads)
        ),
        normalized_price_mae=float(np.mean(absolute_error)),
        normalized_price_median_absolute_error=float(np.median(absolute_error)),
        normalized_price_rmse=float(np.sqrt(np.mean(error**2))),
        normalized_price_bias=float(np.mean(error)),
    )


def excess_statistics(excess: np.ndarray) -> ExcessStatistics:
    """Summarize containment and excess conditional on positive excess."""
    excess = np.asarray(excess, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(excess) & (excess >= 0.0)):
        raise ValueError("'excess' must contain finite non-negative values")
    positive = excess[excess > 0.0]
    num_options = int(len(excess))
    num_positive = int(len(positive))
    num_zero = num_options - num_positive
    return ExcessStatistics(
        num_options=num_options,
        num_zero=num_zero,
        percent_zero=(
            100.0 * num_zero / num_options if num_options > 0 else 0.0
        ),
        num_positive=num_positive,
        positive_mean=float(np.mean(positive)) if num_positive > 0 else 0.0,
        positive_median=float(np.median(positive)) if num_positive > 0 else 0.0,
        surface_mean=float(np.mean(excess)) if num_options > 0 else 0.0,
        surface_median=float(np.median(excess)) if num_options > 0 else 0.0,
    )


def expiry_excess_statistics(
    fit: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    business_dtes: np.ndarray,
    *,
    dte_tolerance: float = 1.0e-4,
) -> ExpiryExcessStatistics:
    """Compute the required aggregate and short-expiry report buckets."""
    business_dtes = np.asarray(business_dtes, dtype=np.float64).reshape(-1)
    excess = normalized_excess(fit, bid, ask).reshape(-1)
    if business_dtes.shape != excess.shape:
        raise ValueError(
            "'business_dtes' must have one value per option, found "
            f"{business_dtes.shape} and {excess.shape}"
        )
    if not np.all(np.isfinite(business_dtes) & (business_dtes >= 0.0)):
        raise ValueError("'business_dtes' must be finite and non-negative")

    # AM expiries carry the repository's fractional-day clock (for example
    # 0.2, 1.2, and 2.2 business days).  The market convention calls these
    # 0DTE, 1DTE, and 2DTE, so bucket by the completed integer day count.
    nominal_dtes = np.floor(business_dtes + dte_tolerance).astype(np.int64)

    def bucket(dte: int) -> ExcessStatistics:
        mask = nominal_dtes == dte
        return excess_statistics(excess[mask])

    return ExpiryExcessStatistics(
        all=excess_statistics(excess),
        dte_0=bucket(0),
        dte_1=bucket(1),
        dte_2=bucket(2),
    )


def nominal_expiry_excess_statistics(
    fit: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    business_dtes: np.ndarray,
    *,
    dte_tolerance: float = 1.0e-4,
) -> tuple[dict[str, int | float], ...]:
    """Summarize every represented nominal market DTE.

    AM expiries use the repository's fractional-day convention, so the
    nominal expiry is the completed integer business-day count, consistently
    with :func:`expiry_excess_statistics`. Empty maturities are omitted.
    """
    business_dtes = np.asarray(business_dtes, dtype=np.float64).reshape(-1)
    excess = normalized_excess(fit, bid, ask).reshape(-1)
    if business_dtes.shape != excess.shape:
        raise ValueError(
            "'business_dtes' must have one value per option, found "
            f"{business_dtes.shape} and {excess.shape}"
        )
    if not np.all(np.isfinite(business_dtes) & (business_dtes >= 0.0)):
        raise ValueError("'business_dtes' must be finite and non-negative")

    nominal_dtes = np.floor(business_dtes + dte_tolerance).astype(np.int64)
    rows: list[dict[str, int | float]] = []
    for nominal_dte in np.unique(nominal_dtes):
        statistics = excess_statistics(excess[nominal_dtes == nominal_dte])
        rows.append(
            {
                "nominal_market_dte": int(nominal_dte),
                "num_quotes": statistics.num_options,
                "percent_inside_bid_ask": statistics.percent_zero,
                "mean_excess": statistics.surface_mean,
                "conditional_mean_excess": statistics.positive_mean,
                "conditional_median_excess": statistics.positive_median,
            }
        )
    return tuple(rows)
