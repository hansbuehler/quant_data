"""Bounded-memory, antithetic-pair summaries for large forward samples."""

from __future__ import annotations

import numpy as np

from quantdata.gromm2.rolling_audit import combine_moments, paired_samples


class PairMoments:
    """Merge independent pair observations with stable centered moments."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = np.asarray(0.0)
        self.squares = np.asarray(0.0)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("pair summaries require finite observations")
        pairs = paired_samples(values)
        self.count, self.mean, self.squares = combine_moments(self.count, self.mean, self.squares, pairs)

    @property
    def standard_error(self) -> np.ndarray:
        if self.count < 2:
            raise ValueError("at least two independent pairs are required")
        return np.sqrt(np.maximum(self.squares, 0.0) / ((self.count - 1) * self.count))

    def screen(self, included: np.ndarray) -> dict[str, object]:
        error = self.standard_error
        statistic = np.divide(self.mean, error, out=np.zeros_like(self.mean), where=error > 0.0)
        absolute = np.abs(statistic[included])
        coordinates = np.argwhere(included)
        worst = np.argsort(absolute)[-10:][::-1]
        return {"independent_pairs": self.count, "coordinate_count": len(absolute),
            "maximum_absolute_statistic": float(np.max(absolute)),
            "percentile_99_absolute_statistic": float(np.quantile(absolute, 0.99)),
            "count_above_1_96": int(np.sum(absolute > 1.96)), "count_above_2_576": int(np.sum(absolute > 2.576)),
            "nonzero_deterministic_coordinates": int(np.sum(included & (error == 0.0) & (self.mean != 0.0))),
            "worst_coordinates": [{"index": coordinates[index].tolist(),
                "mean": float(self.mean[tuple(coordinates[index])]),
                "standard_error": float(error[tuple(coordinates[index])]),
                "statistic": float(statistic[tuple(coordinates[index])])} for index in worst]}

    def scalar_summary(self) -> dict[str, float | int]:
        mean, error = float(self.mean), float(self.standard_error)
        return {"mean": mean, "standard_error": error, "ci95_lower": mean - 1.96 * error,
            "ci95_upper": mean + 1.96 * error, "three_se_lower": mean - 3.0 * error,
            "three_se_upper": mean + 3.0 * error, "paths": 2 * self.count, "independent_pairs": self.count}


class ScenarioStatistics:
    """Update diagnostics once per batch rather than rereading growing prefixes."""

    def __init__(self, *, book: dict[str, np.ndarray], days: int) -> None:
        self.book, self.days = book, days
        self.payoff, self.raw, self.scaled, self.expiry = (PairMoments() for _ in range(4))
        self.minimum, self.maximum = float("inf"), float("-inf")
        self.terminal_minimum, self.terminal_maximum = float("inf"), float("-inf")
        self.below, self.above, self.paths = 0, 0, 0

    def update(self, prices: np.ndarray, payoff: np.ndarray) -> None:
        self.payoff.update(payoff)
        increments = np.diff(prices, axis=1)
        self.raw.update(increments)
        self.scaled.update(increments / prices[:, :-1, :1])
        contracts = np.flatnonzero(self.book["option_expiry_days"] <= self.days)
        self.expiry.update(prices[:, self.book["option_expiry_days"][contracts], contracts + 1] - prices[:, 0, contracts + 1])
        spots = prices[:, :, 0]
        self.minimum, self.maximum = min(self.minimum, float(spots.min())), max(self.maximum, float(spots.max()))
        self.terminal_minimum = min(self.terminal_minimum, float(spots[:, -1].min()))
        self.terminal_maximum = max(self.terminal_maximum, float(spots[:, -1].max()))
        self.below += int(np.any(spots < 0.95, axis=1).sum())
        self.above += int(np.any(spots > 1.05, axis=1).sum())
        self.paths += len(prices)

    def audit(self) -> dict[str, object]:
        valid = np.column_stack((np.ones(self.days, dtype=bool),
            np.arange(1, self.days + 1)[:, None] <= self.book["option_expiry_days"][None]))
        contracts = self.book["option_expiry_days"] <= self.days
        return {"interpretation": "Pointwise unconditional Gaussian screens, not a simultaneous test or proof of conditional martingality. Daily indices are zero-based [day, instrument], spot instrument zero. Expiry indices follow the saved option book. Expired positions are excluded; cash settlement is included once.",
            "raw_daily_gains": self.raw.screen(valid), "spot_scaled_daily_gains": self.scaled.screen(valid),
            "option_to_expiry_gains": self.expiry.screen(np.ones(int(contracts.sum()), dtype=bool))}

    def exploration(self) -> dict[str, float]:
        return {"fraction_ever_below_0_95": self.below / self.paths, "fraction_ever_above_1_05": self.above / self.paths,
            "minimum": self.minimum, "maximum": self.maximum}
