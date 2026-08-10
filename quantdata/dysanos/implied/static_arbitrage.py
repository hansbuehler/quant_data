"""Streaming static-arbitrage checks for direct PCA decoded call surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    FLOAT32_UNIT_ROUNDOFF,
)

_CONSTRAINTS = (
    "lower_bound",
    "upper_bound",
    "call_monotonicity",
    "vertical_spread_slope",
    "strike_convexity",
    "calendar_monotonicity",
)
_STRIKE_DIRECTION_CONSTRAINTS = (
    "call_monotonicity",
    "vertical_spread_slope",
    "strike_convexity",
)


@dataclass(frozen=True, kw_only=True)
class _ConstraintValues:
    name: str
    violation: np.ndarray
    arithmetic_scale: np.ndarray
    expiry_index: np.ndarray
    strike_index: np.ndarray


@dataclass(frozen=True, kw_only=True)
class _TradedStrikeConstraintValues:
    name: str
    violation: np.ndarray
    arithmetic_scale: np.ndarray
    strike_offset: int


@dataclass
class _ViolationSummary:
    reservoir_size: int
    rng: np.random.Generator
    portfolios: int = 0
    magnitude_sum: float = 0.0
    maximum: float = 0.0
    worst_location: dict[str, Any] | None = None
    surface_count: int = 0
    path_count: int = 0
    weighted_portfolio_mass: float = 0.0
    weighted_magnitude_sum: float = 0.0
    weighted_surface_mass: float = 0.0
    weighted_path_mass: float = 0.0
    weight_square_surface_rate_sum: float = 0.0
    weight_square_surface_rate_square_sum: float = 0.0
    weight_square_path_indicator_sum: float = 0.0
    _reservoir_values: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    _reservoir_priorities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def update_values(
        self, values: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        weights = (
            np.ones(values.size, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        assert weights.shape == values.shape
        self.portfolios += values.size
        self.magnitude_sum += float(np.sum(values, dtype=np.float64))
        self.weighted_portfolio_mass += float(np.sum(weights))
        self.weighted_magnitude_sum += float(np.dot(weights, values))
        priorities = np.full(values.size, np.inf, dtype=np.float64)
        positive_weight = weights > 0.0
        uniforms = np.maximum(
            self.rng.random(np.count_nonzero(positive_weight)),
            np.finfo(float).tiny,
        )
        # log(-log(U)/w) preserves priority ordering without overflowing when
        # the likelihood ratio is representable but extremely small.
        priorities[positive_weight] = np.log(-np.log(uniforms)) - np.log(
            weights[positive_weight]
        )
        combined_values = np.concatenate((self._reservoir_values, values))
        combined_priorities = np.concatenate(
            (self._reservoir_priorities, priorities)
        )
        if combined_values.size > self.reservoir_size:
            keep = np.argpartition(
                combined_priorities, self.reservoir_size - 1
            )[: self.reservoir_size]
            combined_values = combined_values[keep]
            combined_priorities = combined_priorities[keep]
        self._reservoir_values = combined_values
        self._reservoir_priorities = combined_priorities

    def report(
        self,
        *,
        surfaces: int,
        paths: int,
        weighted_surfaces: float,
        weighted_paths: float,
        path_weight_square_sum: float,
    ) -> dict[str, Any]:
        values = self._reservoir_values
        weighted_surface_fraction = (
            self.weighted_surface_mass / weighted_surfaces
            if weighted_surfaces
            else 0.0
        )
        weighted_path_fraction = (
            self.weighted_path_mass / weighted_paths
            if weighted_paths
            else 0.0
        )
        weighted_mean = (
            self.weighted_magnitude_sum / self.weighted_portfolio_mass
            if self.weighted_portfolio_mass
            else 0.0
        )
        surface_error = math.sqrt(
            max(
                self.weight_square_surface_rate_square_sum
                - 2.0
                * weighted_surface_fraction
                * self.weight_square_surface_rate_sum
                + weighted_surface_fraction**2 * path_weight_square_sum,
                0.0,
            )
        ) / weighted_paths
        path_error = math.sqrt(
            max(
                self.weight_square_path_indicator_sum
                * (1.0 - 2.0 * weighted_path_fraction)
                + weighted_path_fraction**2 * path_weight_square_sum,
                0.0,
            )
        ) / weighted_paths
        return {
            "violating_surfaces": self.surface_count,
            "violating_surface_fraction": weighted_surface_fraction,
            "proposal_violating_surface_fraction": (
                self.surface_count / surfaces if surfaces else 0.0
            ),
            "paths_with_violation": self.path_count,
            "path_fraction": weighted_path_fraction,
            "proposal_path_fraction": self.path_count / paths if paths else 0.0,
            "physical_weighted_violating_surface_fraction": weighted_surface_fraction,
            "physical_weighted_violating_surface_fraction_path_clustered_error": (
                surface_error
            ),
            "physical_weighted_path_fraction": weighted_path_fraction,
            "physical_weighted_path_fraction_standard_error": path_error,
            "violating_grid_portfolios": self.portfolios,
            "conditional_magnitude": {
                "mean": weighted_mean,
                "proposal_mean": (
                    self.magnitude_sum / self.portfolios
                    if self.portfolios
                    else 0.0
                ),
                "physical_weighted_mean": weighted_mean,
                "q95": float(np.quantile(values, 0.95)) if values.size else 0.0,
                "q99": float(np.quantile(values, 0.99)) if values.size else 0.0,
                "maximum": self.maximum,
                "quantile_observations": int(values.size),
                "quantile_measure": (
                    "physical P via likelihood-weighted priority reservoir"
                ),
                "quantiles_are_reservoir_estimates": (
                    self.portfolios > values.size
                ),
            },
            "worst_location": self.worst_location,
        }


def _validate_surface_inputs(
    prices: np.ndarray,
    strikes: np.ndarray,
    expiries: np.ndarray,
    nms: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    prices = np.asarray(prices)
    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    verify_inp(
        prices.ndim in (3, 4),
        lambda: f"call prices must have shape (S,T,K) or (S,A,T,K), got {prices.shape}",
    )
    if prices.ndim == 3:
        prices = prices[:, None]
        if nms is not None and np.asarray(nms).ndim == 3:
            nms = np.asarray(nms)[:, None]
    S, A, T, K = prices.shape
    verify_inp(
        min(S, A, T) > 0 and K >= 3,
        lambda: f"static checker requires positive S/A/T and at least three strikes, got {prices.shape}",
    )
    verify_inp(
        strikes.shape == (K,)
        and np.all(np.isfinite(strikes))
        and np.all(strikes > 0.0)
        and np.all(np.diff(strikes) > 0.0),
        lambda: f"strikes must be a finite positive increasing {(K,)} vector, got {strikes}",
    )
    verify_inp(
        expiries.shape == (T,)
        and np.all(np.isfinite(expiries))
        and np.all(expiries > 0.0)
        and np.all(np.diff(expiries) > 0.0),
        lambda: f"expiries must be finite, positive, and increasing, got {expiries}",
    )
    verify_inp(
        np.issubdtype(prices.dtype, np.floating) and np.all(np.isfinite(prices)),
        "call prices must be finite floating data",
    )
    if nms is not None:
        nms = np.asarray(nms, dtype=np.float64)
        verify_inp(
            nms.shape == prices.shape and np.all(np.isfinite(nms)),
            lambda: f"NMS locations must match prices {prices.shape}, got {nms.shape}",
        )
    return prices.astype(np.float64, copy=False), nms


def _constraint_values(
    prices: np.ndarray,
    strikes: np.ndarray,
) -> tuple[_ConstraintValues, ...]:
    """Construct violation value and actual arithmetic scale per portfolio."""

    S, A, T, K = prices.shape
    intrinsic = np.maximum(1.0 - strikes, 0.0)[None, None, None, :]
    lower = intrinsic - prices
    upper = prices - 1.0
    bound_expiry = np.broadcast_to(
        np.arange(T)[None, None, :, None], (S, A, T, K)
    )
    bound_strike = np.broadcast_to(
        np.arange(K)[None, None, None, :], (S, A, T, K)
    )

    left = prices[..., :-1]
    right = prices[..., 1:]
    strike_width = np.diff(strikes)[None, None, None, :]
    pair_expiry = np.broadcast_to(
        np.arange(T)[None, None, :, None], left.shape
    )
    pair_strike = np.broadcast_to(
        np.arange(K - 1)[None, None, None, :], left.shape
    )

    dx_left = strikes[1:-1] - strikes[:-2]
    dx_right = strikes[2:] - strikes[1:-1]
    endpoint_left_weight = dx_right / (dx_left + dx_right)
    endpoint_right_weight = dx_left / (dx_left + dx_right)
    convex_reference = (
        endpoint_left_weight[None, None, None, :] * prices[..., :-2]
        + endpoint_right_weight[None, None, None, :] * prices[..., 2:]
    )
    convex = prices[..., 1:-1] - convex_reference
    convex_scale = (
        np.abs(prices[..., 1:-1])
        + endpoint_left_weight[None, None, None, :] * np.abs(prices[..., :-2])
        + endpoint_right_weight[None, None, None, :] * np.abs(prices[..., 2:])
    )
    convex_expiry = np.broadcast_to(
        np.arange(T)[None, None, :, None], convex.shape
    )
    convex_strike = np.broadcast_to(
        np.arange(1, K - 1)[None, None, None, :], convex.shape
    )

    calendar = prices[..., :-1, :] - prices[..., 1:, :]
    calendar_expiry = np.broadcast_to(
        np.arange(T - 1)[None, None, :, None], calendar.shape
    )
    calendar_strike = np.broadcast_to(
        np.arange(K)[None, None, None, :], calendar.shape
    )
    return (
        _ConstraintValues(
            name="lower_bound",
            violation=lower,
            arithmetic_scale=np.abs(prices) + np.abs(intrinsic),
            expiry_index=bound_expiry,
            strike_index=bound_strike,
        ),
        _ConstraintValues(
            name="upper_bound",
            violation=upper,
            arithmetic_scale=np.abs(prices) + 1.0,
            expiry_index=bound_expiry,
            strike_index=bound_strike,
        ),
        _ConstraintValues(
            name="call_monotonicity",
            violation=right - left,
            arithmetic_scale=np.abs(right) + np.abs(left),
            expiry_index=pair_expiry,
            strike_index=pair_strike,
        ),
        _ConstraintValues(
            name="vertical_spread_slope",
            violation=left - right - strike_width,
            arithmetic_scale=np.abs(left) + np.abs(right) + strike_width,
            expiry_index=pair_expiry,
            strike_index=pair_strike,
        ),
        _ConstraintValues(
            name="strike_convexity",
            violation=convex,
            arithmetic_scale=convex_scale,
            expiry_index=convex_expiry,
            strike_index=convex_strike,
        ),
        _ConstraintValues(
            name="calendar_monotonicity",
            violation=calendar,
            arithmetic_scale=np.abs(prices[..., :-1, :])
            + np.abs(prices[..., 1:, :]),
            expiry_index=calendar_expiry,
            strike_index=calendar_strike,
        ),
    )


def _traded_strike_constraint_values(
    prices: np.ndarray,
    strikes: np.ndarray,
) -> tuple[_TradedStrikeConstraintValues, ...]:
    """Return strike-direction violations on surface-specific strike grids."""

    left = prices[..., :-1]
    right = prices[..., 1:]
    strike_width = strikes[..., 1:] - strikes[..., :-1]

    dx_left = strikes[..., 1:-1] - strikes[..., :-2]
    dx_right = strikes[..., 2:] - strikes[..., 1:-1]
    total_width = dx_left + dx_right
    left_weight = dx_right / total_width
    right_weight = dx_left / total_width
    convex_reference = (
        left_weight * prices[..., :-2] + right_weight * prices[..., 2:]
    )
    return (
        _TradedStrikeConstraintValues(
            name="call_monotonicity",
            violation=right - left,
            arithmetic_scale=np.abs(right) + np.abs(left),
            strike_offset=0,
        ),
        _TradedStrikeConstraintValues(
            name="vertical_spread_slope",
            violation=left - right - strike_width,
            arithmetic_scale=np.abs(left) + np.abs(right) + strike_width,
            strike_offset=0,
        ),
        _TradedStrikeConstraintValues(
            name="strike_convexity",
            violation=prices[..., 1:-1] - convex_reference,
            arithmetic_scale=(
                np.abs(prices[..., 1:-1])
                + left_weight * np.abs(prices[..., :-2])
                + right_weight * np.abs(prices[..., 2:])
            ),
            strike_offset=1,
        ),
    )


def _validate_traded_strike_inputs(
    prices: np.ndarray,
    strikes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prices = np.asarray(prices)
    strikes = np.asarray(strikes)
    verify_inp(
        prices.ndim == 4 and prices.shape == strikes.shape,
        lambda: (
            "traded calls and strikes must share shape (S,A,T,K), got "
            f"{prices.shape}/{strikes.shape}"
        ),
    )
    verify_inp(
        min(prices.shape[:3]) > 0 and prices.shape[-1] >= 3,
        lambda: (
            "traded-strike checker requires positive S/A/T and at least three "
            f"strikes, got {prices.shape}"
        ),
    )
    verify_inp(
        np.issubdtype(prices.dtype, np.floating)
        and np.issubdtype(strikes.dtype, np.floating)
        and np.all(np.isfinite(prices))
        and np.all(np.isfinite(strikes)),
        "traded calls and strikes must be finite floating arrays",
    )
    verify_inp(
        np.all(strikes > 0.0) and np.all(np.diff(strikes, axis=-1) > 0.0),
        "traded relative strikes must be strictly positive and increasing",
    )
    return (
        prices.astype(np.float64, copy=False),
        strikes.astype(np.float64, copy=False),
    )


class StaticArbitrageAccumulator:
    """Update exact counts and bounded-memory violation quantiles by blocks."""

    def __init__(
        self,
        *,
        strikes: np.ndarray,
        expiries: np.ndarray,
        roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128),
        reservoir_size: int = 200_000,
        seed: int = 19_731,
    ) -> None:
        strikes = np.asarray(strikes, dtype=np.float64)
        expiries = np.asarray(expiries, dtype=np.float64)
        verify_inp(
            strikes.ndim == expiries.ndim == 1,
            "static-check strike and expiry grids must be one-dimensional",
        )
        verify_inp(
            tuple(sorted(set(roundoff_multipliers))) == roundoff_multipliers
            and roundoff_multipliers[0] == 0,
            lambda: f"invalid roundoff multipliers {roundoff_multipliers}",
        )
        verify_inp(
            isinstance(reservoir_size, int) and reservoir_size > 0,
            lambda: f"'reservoir_size' must be positive, got {reservoir_size!r}",
        )
        self.strikes = strikes
        self.expiries = expiries
        self.roundoff_multipliers = roundoff_multipliers
        root_rng = np.random.default_rng(seed)
        self._summaries = {
            multiplier: {
                name: _ViolationSummary(
                    reservoir_size=reservoir_size,
                    rng=np.random.default_rng(root_rng.integers(0, 2**63)),
                )
                for name in (*_CONSTRAINTS, "aggregate")
            }
            for multiplier in roundoff_multipliers
        }
        self.paths = 0
        self.surfaces = 0
        self.weighted_paths = 0.0
        self.weighted_surfaces = 0.0
        self.path_weight_square_sum = 0.0

    def _location(
        self,
        constraint: _ConstraintValues,
        flat_index: int,
        nms: np.ndarray | None,
    ) -> dict[str, Any]:
        index = np.unravel_index(flat_index, constraint.violation.shape)
        path, action = int(index[0]), int(index[1])
        expiry_index = int(constraint.expiry_index[index])
        strike_index = int(constraint.strike_index[index])
        result: dict[str, Any] = {
            "path_in_block": path,
            "action_index": action,
            "expiry_index": expiry_index,
            "expiry": float(self.expiries[expiry_index]),
            "strike_index": strike_index,
            "relative_strike": float(self.strikes[strike_index]),
        }
        if nms is not None:
            result["normalized_moneyness"] = float(
                nms[path, action, expiry_index, strike_index]
            )
        return result

    def update(
        self,
        prices: np.ndarray,
        *,
        normalized_moneyness: np.ndarray | None = None,
        log_path_weights: np.ndarray | None = None,
    ) -> None:
        prices, normalized_moneyness = _validate_surface_inputs(
            prices,
            self.strikes,
            self.expiries,
            normalized_moneyness,
        )
        constraints = _constraint_values(prices, self.strikes)
        block_paths, block_actions = prices.shape[:2]
        if log_path_weights is None:
            path_weights = np.ones(block_paths, dtype=np.float64)
        else:
            log_path_weights = np.asarray(log_path_weights, dtype=np.float64)
            verify_inp(
                log_path_weights.shape == (block_paths,)
                and np.all(np.isfinite(log_path_weights)),
                lambda: (
                    f"static path log weights must have shape {(block_paths,)}, "
                    f"got {log_path_weights.shape}"
                ),
            )
            path_weights = np.exp(log_path_weights)
            verify_inp(
                np.all(np.isfinite(path_weights))
                and np.all(path_weights >= 0.0)
                and np.any(path_weights > 0.0),
                "static path likelihood weights must be finite, nonnegative, and nonzero",
            )
        self.paths += block_paths
        self.surfaces += block_paths * block_actions
        self.weighted_paths += float(np.sum(path_weights))
        self.weighted_surfaces += float(np.sum(path_weights)) * block_actions
        self.path_weight_square_sum += float(np.sum(np.square(path_weights)))

        for multiplier in self.roundoff_multipliers:
            summaries = self._summaries[multiplier]
            aggregate_surface = np.zeros(
                (block_paths, block_actions), dtype=bool
            )
            aggregate_path = np.zeros(block_paths, dtype=bool)
            for constraint in constraints:
                band = (
                    multiplier
                    * FLOAT32_UNIT_ROUNDOFF
                    * np.maximum(1.0, constraint.arithmetic_scale)
                )
                mask = constraint.violation > band
                surface_mask = np.any(mask.reshape(block_paths, block_actions, -1), axis=2)
                path_mask = np.any(surface_mask, axis=1)
                summary = summaries[constraint.name]
                summary.surface_count += int(np.count_nonzero(surface_mask))
                summary.path_count += int(np.count_nonzero(path_mask))
                summary.weighted_surface_mass += float(
                    np.sum(path_weights[:, None] * surface_mask)
                )
                summary.weighted_path_mass += float(
                    np.sum(path_weights * path_mask)
                )
                weight_square = np.square(path_weights)
                surface_rate = np.mean(surface_mask, axis=1)
                summary.weight_square_surface_rate_sum += float(
                    np.dot(weight_square, surface_rate)
                )
                summary.weight_square_surface_rate_square_sum += float(
                    np.dot(weight_square, np.square(surface_rate))
                )
                summary.weight_square_path_indicator_sum += float(
                    np.dot(weight_square, path_mask)
                )
                values = constraint.violation[mask]
                portfolio_weights = np.broadcast_to(
                    path_weights.reshape(
                        block_paths,
                        *([1] * (constraint.violation.ndim - 1)),
                    ),
                    constraint.violation.shape,
                )[mask]
                summary.update_values(values, portfolio_weights)
                if values.size:
                    masked = np.where(mask, constraint.violation, -np.inf)
                    flat_index = int(np.argmax(masked))
                    maximum = float(masked.reshape(-1)[flat_index])
                    if maximum > summary.maximum:
                        summary.maximum = maximum
                        summary.worst_location = self._location(
                            constraint,
                            flat_index,
                            normalized_moneyness,
                        ) | {"constraint": constraint.name}
                aggregate_surface |= surface_mask
                aggregate_path |= path_mask

                aggregate = summaries["aggregate"]
                aggregate.update_values(values, portfolio_weights)
                if values.size and float(np.max(values)) > aggregate.maximum:
                    flat_index = int(
                        np.argmax(np.where(mask, constraint.violation, -np.inf))
                    )
                    aggregate.maximum = float(np.max(values))
                    aggregate.worst_location = self._location(
                        constraint,
                        flat_index,
                        normalized_moneyness,
                    ) | {"constraint": constraint.name}
            aggregate = summaries["aggregate"]
            aggregate.surface_count += int(np.count_nonzero(aggregate_surface))
            aggregate.path_count += int(np.count_nonzero(aggregate_path))
            aggregate.weighted_surface_mass += float(
                np.sum(path_weights[:, None] * aggregate_surface)
            )
            aggregate.weighted_path_mass += float(
                np.sum(path_weights * aggregate_path)
            )
            weight_square = np.square(path_weights)
            aggregate_surface_rate = np.mean(aggregate_surface, axis=1)
            aggregate.weight_square_surface_rate_sum += float(
                np.dot(weight_square, aggregate_surface_rate)
            )
            aggregate.weight_square_surface_rate_square_sum += float(
                np.dot(weight_square, np.square(aggregate_surface_rate))
            )
            aggregate.weight_square_path_indicator_sum += float(
                np.dot(weight_square, aggregate_path)
            )

    def report(self) -> dict[str, Any]:
        by_multiplier = {
            str(multiplier): {
                name: summary.report(
                    surfaces=self.surfaces,
                    paths=self.paths,
                    weighted_surfaces=self.weighted_surfaces,
                    weighted_paths=self.weighted_paths,
                    path_weight_square_sum=self.path_weight_square_sum,
                )
                for name, summary in summaries.items()
            }
            for multiplier, summaries in self._summaries.items()
        }
        return {
            "paths": self.paths,
            "surfaces": self.surfaces,
            "measure": (
                "original physical P; complete path p/q applied exactly once"
            ),
            "mean_path_likelihood": (
                self.weighted_paths / self.paths if self.paths else 0.0
            ),
            "roundoff": {
                "unit_roundoff": FLOAT32_UNIT_ROUNDOFF,
                "multipliers": list(self.roundoff_multipliers),
                "primary_multiplier": 32,
                "band": (
                    "multiplier * float32_unit_roundoff * "
                    "max(1, sum(abs(portfolio_weight * call_price)))"
                ),
            },
            "by_roundoff_multiplier": by_multiplier,
            "primary": by_multiplier["32"] if "32" in by_multiplier else None,
        }


class TradedStrikeStaticArbitrageAccumulator:
    """Stream strike-direction checks on surface-specific traded strikes."""

    def __init__(
        self,
        *,
        roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128),
        reservoir_size: int = 200_000,
        seed: int = 29_741,
    ) -> None:
        verify_inp(
            tuple(sorted(set(roundoff_multipliers))) == roundoff_multipliers
            and roundoff_multipliers[0] == 0,
            lambda: f"invalid roundoff multipliers {roundoff_multipliers}",
        )
        verify_inp(
            isinstance(reservoir_size, int) and reservoir_size > 0,
            lambda: f"'reservoir_size' must be positive, got {reservoir_size!r}",
        )
        self.roundoff_multipliers = roundoff_multipliers
        root_rng = np.random.default_rng(seed)
        names = (*_STRIKE_DIRECTION_CONSTRAINTS, "aggregate")
        self._summaries = {
            multiplier: {
                name: _ViolationSummary(
                    reservoir_size=reservoir_size,
                    rng=np.random.default_rng(root_rng.integers(0, 2**63)),
                )
                for name in names
            }
            for multiplier in roundoff_multipliers
        }
        self._tested = {
            multiplier: {name: 0 for name in names}
            for multiplier in roundoff_multipliers
        }
        self._weighted_tested = {
            multiplier: {name: 0.0 for name in names}
            for multiplier in roundoff_multipliers
        }
        self._surface_count_by_action: dict[
            int, dict[str, np.ndarray]
        ] = {}
        self._surface_mass_by_action: dict[
            int, dict[str, np.ndarray]
        ] = {}
        self._expiry_slice_count: dict[int, dict[str, np.ndarray]] = {}
        self._expiry_slice_mass: dict[int, dict[str, np.ndarray]] = {}
        self._geometry: tuple[int, int, int] | None = None
        self.paths = 0
        self.surfaces = 0
        self.weighted_paths = 0.0
        self.weighted_surfaces = 0.0
        self.path_weight_square_sum = 0.0

    def _initialize_geometry(self, *, actions: int, expiries: int, strikes: int) -> None:
        geometry = actions, expiries, strikes
        if self._geometry is not None:
            verify_inp(
                geometry == self._geometry,
                lambda: (
                    f"traded-strike geometry changed from {self._geometry} "
                    f"to {geometry}"
                ),
            )
            return
        self._geometry = geometry
        names = (*_STRIKE_DIRECTION_CONSTRAINTS, "aggregate")
        for multiplier in self.roundoff_multipliers:
            self._surface_count_by_action[multiplier] = {
                name: np.zeros(actions, dtype=np.int64) for name in names
            }
            self._surface_mass_by_action[multiplier] = {
                name: np.zeros(actions, dtype=np.float64) for name in names
            }
            self._expiry_slice_count[multiplier] = {
                name: np.zeros(expiries, dtype=np.int64) for name in names
            }
            self._expiry_slice_mass[multiplier] = {
                name: np.zeros(expiries, dtype=np.float64) for name in names
            }

    @staticmethod
    def _worst_location(
        constraint: _TradedStrikeConstraintValues,
        *,
        flat_index: int,
        strikes: np.ndarray,
        prices: np.ndarray,
        band: np.ndarray,
        path_offset: int,
    ) -> dict[str, Any]:
        path, action, expiry, portfolio = (
            int(value)
            for value in np.unravel_index(
                flat_index, constraint.violation.shape
            )
        )
        center = portfolio + constraint.strike_offset
        left = center - 1 if constraint.name == "strike_convexity" else center
        right = center + 1 if constraint.name == "strike_convexity" else center + 1
        index = path, action, expiry
        return {
            "path_index": path_offset + path,
            "action_index": action,
            "expiry_index": expiry,
            "strike_index": center,
            "relative_strikes": [
                float(value) for value in strikes[index][left : right + 1]
            ],
            "normalized_calls": [
                float(value) for value in prices[index][left : right + 1]
            ],
            "violation": float(constraint.violation.reshape(-1)[flat_index]),
            "roundoff_band": float(band.reshape(-1)[flat_index]),
            "constraint": constraint.name,
        }

    def update(
        self,
        prices: np.ndarray,
        *,
        strikes: np.ndarray,
        log_path_weights: np.ndarray | None = None,
        path_offset: int = 0,
    ) -> None:
        prices, strikes = _validate_traded_strike_inputs(prices, strikes)
        verify_inp(
            isinstance(path_offset, int) and path_offset >= 0,
            lambda: f"'path_offset' must be nonnegative, got {path_offset!r}",
        )
        constraints = _traded_strike_constraint_values(prices, strikes)
        block_paths, actions, expiries, strike_count = prices.shape
        self._initialize_geometry(
            actions=actions, expiries=expiries, strikes=strike_count
        )
        if log_path_weights is None:
            path_weights = np.ones(block_paths, dtype=np.float64)
        else:
            log_path_weights = np.asarray(log_path_weights, dtype=np.float64)
            verify_inp(
                log_path_weights.shape == (block_paths,)
                and np.all(np.isfinite(log_path_weights)),
                lambda: (
                    "traded-strike log weights must have shape "
                    f"{(block_paths,)}, got {log_path_weights.shape}"
                ),
            )
            path_weights = np.exp(log_path_weights)
            verify_inp(
                np.all(np.isfinite(path_weights))
                and np.all(path_weights >= 0.0)
                and np.any(path_weights > 0.0),
                "traded-strike likelihood weights must be finite, nonnegative, and nonzero",
            )
        weight_sum = float(np.sum(path_weights))
        weight_square = np.square(path_weights)
        self.paths += block_paths
        self.surfaces += block_paths * actions
        self.weighted_paths += weight_sum
        self.weighted_surfaces += weight_sum * actions
        self.path_weight_square_sum += float(np.sum(weight_square))

        for multiplier in self.roundoff_multipliers:
            summaries = self._summaries[multiplier]
            aggregate_surface = np.zeros((block_paths, actions), dtype=bool)
            aggregate_path = np.zeros(block_paths, dtype=bool)
            aggregate_expiry = np.zeros(
                (block_paths, actions, expiries), dtype=bool
            )
            for constraint in constraints:
                band = (
                    multiplier
                    * FLOAT32_UNIT_ROUNDOFF
                    * np.maximum(1.0, constraint.arithmetic_scale)
                )
                mask = constraint.violation > band
                expiry_mask = np.any(mask, axis=-1)
                surface_mask = np.any(expiry_mask, axis=-1)
                path_mask = np.any(surface_mask, axis=-1)
                summary = summaries[constraint.name]
                summary.surface_count += int(np.count_nonzero(surface_mask))
                summary.path_count += int(np.count_nonzero(path_mask))
                summary.weighted_surface_mass += float(
                    np.sum(path_weights[:, None] * surface_mask)
                )
                summary.weighted_path_mass += float(
                    np.sum(path_weights * path_mask)
                )
                surface_rate = np.mean(surface_mask, axis=1)
                summary.weight_square_surface_rate_sum += float(
                    np.dot(weight_square, surface_rate)
                )
                summary.weight_square_surface_rate_square_sum += float(
                    np.dot(weight_square, np.square(surface_rate))
                )
                summary.weight_square_path_indicator_sum += float(
                    np.dot(weight_square, path_mask)
                )
                portfolio_weights = np.broadcast_to(
                    path_weights.reshape(block_paths, 1, 1, 1), mask.shape
                )[mask]
                values = constraint.violation[mask]
                summary.update_values(values, portfolio_weights)
                tested = int(mask.size)
                weighted_tested = weight_sum * int(np.prod(mask.shape[1:]))
                self._tested[multiplier][constraint.name] += tested
                self._weighted_tested[multiplier][constraint.name] += (
                    weighted_tested
                )
                self._surface_count_by_action[multiplier][constraint.name] += (
                    np.count_nonzero(surface_mask, axis=0)
                )
                self._surface_mass_by_action[multiplier][constraint.name] += (
                    np.sum(path_weights[:, None] * surface_mask, axis=0)
                )
                self._expiry_slice_count[multiplier][constraint.name] += (
                    np.count_nonzero(expiry_mask, axis=(0, 1))
                )
                self._expiry_slice_mass[multiplier][constraint.name] += np.sum(
                    path_weights[:, None, None] * expiry_mask, axis=(0, 1)
                )
                if values.size:
                    masked = np.where(mask, constraint.violation, -np.inf)
                    flat_index = int(np.argmax(masked))
                    maximum = float(masked.reshape(-1)[flat_index])
                    if maximum > summary.maximum:
                        summary.maximum = maximum
                        summary.worst_location = self._worst_location(
                            constraint,
                            flat_index=flat_index,
                            strikes=strikes,
                            prices=prices,
                            band=band,
                            path_offset=path_offset,
                        )
                    aggregate = summaries["aggregate"]
                    aggregate.update_values(values, portfolio_weights)
                    if maximum > aggregate.maximum:
                        aggregate.maximum = maximum
                        aggregate.worst_location = self._worst_location(
                            constraint,
                            flat_index=flat_index,
                            strikes=strikes,
                            prices=prices,
                            band=band,
                            path_offset=path_offset,
                        )
                aggregate_surface |= surface_mask
                aggregate_path |= path_mask
                aggregate_expiry |= expiry_mask

            aggregate = summaries["aggregate"]
            aggregate.surface_count += int(np.count_nonzero(aggregate_surface))
            aggregate.path_count += int(np.count_nonzero(aggregate_path))
            aggregate.weighted_surface_mass += float(
                np.sum(path_weights[:, None] * aggregate_surface)
            )
            aggregate.weighted_path_mass += float(
                np.sum(path_weights * aggregate_path)
            )
            aggregate_surface_rate = np.mean(aggregate_surface, axis=1)
            aggregate.weight_square_surface_rate_sum += float(
                np.dot(weight_square, aggregate_surface_rate)
            )
            aggregate.weight_square_surface_rate_square_sum += float(
                np.dot(weight_square, np.square(aggregate_surface_rate))
            )
            aggregate.weight_square_path_indicator_sum += float(
                np.dot(weight_square, aggregate_path)
            )
            self._tested[multiplier]["aggregate"] += sum(
                int(constraint.violation.size) for constraint in constraints
            )
            self._weighted_tested[multiplier]["aggregate"] += sum(
                weight_sum * int(np.prod(constraint.violation.shape[1:]))
                for constraint in constraints
            )
            self._surface_count_by_action[multiplier]["aggregate"] += (
                np.count_nonzero(aggregate_surface, axis=0)
            )
            self._surface_mass_by_action[multiplier]["aggregate"] += np.sum(
                path_weights[:, None] * aggregate_surface, axis=0
            )
            self._expiry_slice_count[multiplier]["aggregate"] += (
                np.count_nonzero(aggregate_expiry, axis=(0, 1))
            )
            self._expiry_slice_mass[multiplier]["aggregate"] += np.sum(
                path_weights[:, None, None] * aggregate_expiry, axis=(0, 1)
            )

    def report(self) -> dict[str, Any]:
        verify_inp(self._geometry is not None, "no traded-strike blocks were added")
        actions, expiries, strikes = self._geometry
        by_multiplier: dict[str, dict[str, Any]] = {}
        for multiplier, summaries in self._summaries.items():
            multiplier_report: dict[str, Any] = {}
            for name, summary in summaries.items():
                item = summary.report(
                    surfaces=self.surfaces,
                    paths=self.paths,
                    weighted_surfaces=self.weighted_surfaces,
                    weighted_paths=self.weighted_paths,
                    path_weight_square_sum=self.path_weight_square_sum,
                )
                tested = self._tested[multiplier][name]
                weighted_tested = self._weighted_tested[multiplier][name]
                item.update(
                    {
                        "tested_strike_portfolios": tested,
                        "proposal_violating_portfolio_fraction": (
                            summary.portfolios / tested if tested else 0.0
                        ),
                        "physical_weighted_violating_portfolio_fraction": (
                            summary.weighted_portfolio_mass / weighted_tested
                            if weighted_tested
                            else 0.0
                        ),
                        "proposal_violating_surface_fraction_by_trading_time": (
                            self._surface_count_by_action[multiplier][name]
                            / self.paths
                        ).tolist(),
                        "physical_weighted_violating_surface_fraction_by_trading_time": (
                            self._surface_mass_by_action[multiplier][name]
                            / self.weighted_paths
                        ).tolist(),
                        "proposal_violating_expiry_slice_fraction": (
                            self._expiry_slice_count[multiplier][name]
                            / (self.paths * actions)
                        ).tolist(),
                        "physical_weighted_violating_expiry_slice_fraction": (
                            self._expiry_slice_mass[multiplier][name]
                            / (self.weighted_paths * actions)
                        ).tolist(),
                    }
                )
                multiplier_report[name] = item
            by_multiplier[str(multiplier)] = multiplier_report
        return {
            "paths": self.paths,
            "surfaces": self.surfaces,
            "geometry": {
                "trading_times": actions,
                "expiries": expiries,
                "traded_strikes_per_expiry": strikes,
            },
            "scope": (
                "strike direction only on the actual state-dependent traded "
                "grid; bounds and calendar conditions excluded"
            ),
            "measure": (
                "original physical P; complete path p/q applied exactly once"
            ),
            "mean_path_likelihood": self.weighted_paths / self.paths,
            "roundoff": {
                "unit_roundoff": FLOAT32_UNIT_ROUNDOFF,
                "multipliers": list(self.roundoff_multipliers),
                "primary_multiplier": 32,
                "band": (
                    "multiplier * float32_unit_roundoff * "
                    "max(1, arithmetic scale of the tested portfolio)"
                ),
            },
            "by_roundoff_multiplier": by_multiplier,
            "primary": by_multiplier["32"] if "32" in by_multiplier else None,
        }


def check_traded_strike_static_arbitrage(
    prices: np.ndarray,
    *,
    strikes: np.ndarray,
    log_path_weights: np.ndarray | None = None,
    roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128),
) -> dict[str, Any]:
    """Check monotonicity, slope, and convexity on actual traded strikes."""

    prices_array = np.asarray(prices)
    reservoir_size = max(1, 4 * prices_array.size)
    accumulator = TradedStrikeStaticArbitrageAccumulator(
        roundoff_multipliers=roundoff_multipliers,
        reservoir_size=reservoir_size,
    )
    accumulator.update(
        prices_array,
        strikes=strikes,
        log_path_weights=log_path_weights,
    )
    return accumulator.report()


def check_static_arbitrage(
    prices: np.ndarray,
    *,
    strikes: np.ndarray,
    expiries: np.ndarray,
    normalized_moneyness: np.ndarray | None = None,
    roundoff_multipliers: tuple[int, ...] = (0, 1, 8, 32, 128),
) -> dict[str, Any]:
    """Run all static checks and return one exact small-sample report."""

    prices_array = np.asarray(prices)
    reservoir_size = max(1, 8 * prices_array.size)
    accumulator = StaticArbitrageAccumulator(
        strikes=strikes,
        expiries=expiries,
        roundoff_multipliers=roundoff_multipliers,
        reservoir_size=reservoir_size,
    )
    accumulator.update(
        prices_array, normalized_moneyness=normalized_moneyness
    )
    return accumulator.report()


def check_sanos_static_control(
    prices: np.ndarray,
    *,
    strikes: np.ndarray,
    expiries: np.ndarray,
    normalized_moneyness: np.ndarray | None = None,
    require_primary_pass: bool = True,
) -> dict[str, Any]:
    """Evaluate SANOS calls with exactly the same numerical checker."""

    report = check_static_arbitrage(
        prices,
        strikes=strikes,
        expiries=expiries,
        normalized_moneyness=normalized_moneyness,
    )
    primary_violations = report["primary"]["aggregate"][
        "violating_grid_portfolios"
    ]
    if require_primary_pass and primary_violations:
        raise RuntimeError(
            "SANOS control failed the shared static checker at 32x float32 "
            "roundoff. The checker is inconsistent or the query lies outside "
            f"the guaranteed SANOS domain ({primary_violations} violations)."
        )
    report["control"] = "statically arbitrage-free SANOS decoder"
    return report


__all__ = [
    "StaticArbitrageAccumulator",
    "TradedStrikeStaticArbitrageAccumulator",
    "check_sanos_static_control",
    "check_static_arbitrage",
    "check_traded_strike_static_arbitrage",
]
