"""Fixed-notional performance and paired calendar-block sampling uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lower_tail_mean(values: np.ndarray, *, fraction: float = .05) -> float:
    """Empirical lower-tail mean, fractionally weighting the boundary observation."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all() or not 0 < fraction <= 1:
        raise ValueError("Tail means need a finite sample and a fraction in (0,1]")
    ordered = np.sort(data)
    count = fraction * len(data)
    whole = int(np.floor(count))
    numerator = float(ordered[:whole].sum())
    if whole < len(data):
        numerator += (count - whole) * float(ordered[whole])
    return numerator / count


def summarize_products(frame: pd.DataFrame, *, group_columns: tuple[str, ...],
                       date_column: str = "date") -> pd.DataFrame:
    """Compute across-date sample statistics separately for every target/configuration.

    P&L is net of actual entry/rebalance costs and excludes formal exit
    costs. These are not premium returns and are not annualized. Aggregate
    product statistics only afterward; do not pool strikes as independent.
    """
    if frame.empty or frame.duplicated([*group_columns, date_column]).any():
        raise ValueError("Performance needs unique target/configuration/date observations")
    rows = []
    for key, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        pnl = group["costed_pnl"].to_numpy(dtype=float)
        costs = group["entry_cost"].to_numpy(dtype=float)
        if len(pnl) < 2 or not np.isfinite(pnl).all() or not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("Performance requires at least two finite dates and nonnegative costs")
        std = float(pnl.std(ddof=1))
        row = dict(zip(group_columns, key))
        row.update(dates=len(pnl), mean_pnl=float(pnl.mean()), median_pnl=float(np.median(pnl)),
            std_pnl=std, mean_absolute_pnl=float(np.abs(pnl).mean()),
            mean_loss=float(np.maximum(-pnl, 0.).mean()),
            downside_rms=float(np.sqrt(np.mean(np.minimum(pnl, 0.)**2))),
            lower_5pct_mean_pnl=lower_tail_mean(pnl), p10_pnl=float(np.quantile(pnl, .1)),
            p90_pnl=float(np.quantile(pnl, .9)), mean_entry_cost=float(costs.mean()),
            cost_over_std=float(costs.mean() / std) if std > 0 else np.nan,
            mean_over_std=float(pnl.mean() / std) if std > 0 else np.nan)
        if "cap_hits" in group:
            row["cap_fraction"] = float(np.mean(group["cap_hits"].to_numpy() > 0))
        elif "cap_days" in group:
            row["cap_fraction"] = float(np.mean(group["cap_days"].to_numpy() > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def calendar_block_weights(*, session_indices: np.ndarray, count: int = 2000,
                           block_length: int = 20, seed: int = 1) -> np.ndarray:
    """Circular moving blocks on the full business-session grid, including quote gaps.

    One sampled date weight is shared by all products and strategies. Thus
    cross-strike dependence and overlapping multi-day episodes are retained.
    Missing observed dates are not compressed into adjacent business days.
    """
    indices = np.asarray(session_indices)
    if (indices.ndim != 1 or len(indices) < 4 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(np.diff(indices) <= 0) or count < 100 or block_length < 1):
        raise ValueError("Block weights need ordered integer sessions, four dates, >=100 draws and positive blocks")
    shifted = indices - indices[0]
    length = int(shifted[-1]) + 1
    blocks = int(np.ceil(length / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, length, size=(count, blocks))
    sampled = ((starts[:, :, None] + np.arange(block_length)) % length).reshape(count, -1)[:, :length]
    weights = np.asarray([np.bincount(row, minlength=length)[shifted] for row in sampled], dtype=float)
    valid = weights.sum(axis=1) > 1.
    if valid.sum() < max(100, .9 * count):
        raise ValueError("Too few informative calendar-block draws for this sparse observed sample")
    return weights[valid]


def paired_risk_interval(*, candidate: np.ndarray, reference: np.ndarray,
                         session_indices: np.ndarray, count: int = 2000,
                         block_length: int = 20, seed: int = 1) -> dict[str, float | int]:
    """Unadjusted 95% paired intervals for average per-product std and net-mean differences."""
    candidate, reference = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if (candidate.ndim != 2 or reference.shape != candidate.shape
            or candidate.shape[0] != len(session_indices) or candidate.shape[1] == 0
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("Paired risk arrays must be finite and aligned as dates by products")
    weights = calendar_block_weights(session_indices=session_indices, count=count,
                                     block_length=block_length, seed=seed)
    totals = weights.sum(axis=1)

    def bootstrap_std(matrix: np.ndarray) -> np.ndarray:
        centered = matrix - matrix.mean(axis=0)
        means = (weights @ centered) / totals[:, None]
        second = weights @ centered**2
        variance = (second - totals[:, None] * means**2) / (totals[:, None] - 1.)
        return np.sqrt(np.maximum(variance, 0.)).mean(axis=1)

    differences = bootstrap_std(candidate) - bootstrap_std(reference)
    mean_differences = (weights @ (candidate - reference)).mean(axis=1) / totals
    lower, upper = np.quantile(differences, [.025, .975])
    mean_lower, mean_upper = np.quantile(mean_differences, [.025, .975])
    return {"std_difference": float(np.mean(candidate.std(axis=0, ddof=1) - reference.std(axis=0, ddof=1))),
        "std_difference_ci95_lower": float(lower), "std_difference_ci95_upper": float(upper),
        "std_difference_bootstrap_se": float(differences.std(ddof=1)),
        "mean_difference": float(np.mean(candidate - reference)),
        "mean_difference_ci95_lower": float(mean_lower), "mean_difference_ci95_upper": float(mean_upper),
        "bootstrap_draws": len(weights), "calendar_block_sessions": block_length}
