"""State-dependent traded-spread positive control for dynamic-arbitrage tests."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from cdxcore.err import verify_inp
from scipy.stats import beta, norm

from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    FLOAT32_UNIT_ROUNDOFF,
)

from .required_runs import DerivedCaseArrays

_ROUNDOFF_MULTIPLIER = 32.0
_EPSILON = _ROUNDOFF_MULTIPLIER * FLOAT32_UNIT_ROUNDOFF
_KINDS = ("call_monotonicity", "vertical_spread_slope", "strike_convexity")


class StateDependentStaticSpreadAccumulator:
    """Replay predictable traded-grid spread arbitrages without fitting.

    At each path state and trading time the policy selects the strike-direction
    portfolio with the largest strictly positive guaranteed margin after costs
    and a float32 arithmetic band. The action is zero otherwise. All candidate
    spreads have L1 norm two before normalization.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        verify_inp(0.0 < alpha < 1.0, lambda: f"invalid alpha {alpha}")
        self.alpha = alpha
        self.paths = 0
        self.times: int | None = None
        self.direct_paths = 0
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.active_count: np.ndarray | None = None
        self.gain_count: np.ndarray | None = None
        self.loss_count: np.ndarray | None = None
        self.direct_active_count: np.ndarray | None = None
        self.direct_gain_count: np.ndarray | None = None
        self.direct_loss_count: np.ndarray | None = None
        self.weighted_active_sum: np.ndarray | None = None
        self.weighted_gain_sum: np.ndarray | None = None
        self.weighted_loss_sum: np.ndarray | None = None
        self.weighted_pnl_sum: np.ndarray | None = None
        self.weight_square_pnl_sum: np.ndarray | None = None
        self.weight_square_pnl_square_sum: np.ndarray | None = None
        self.direct_pnl_sum: np.ndarray | None = None
        self.direct_pnl_square_sum: np.ndarray | None = None
        self.minimum_pnl: np.ndarray | None = None
        self.minimum_active_pnl: np.ndarray | None = None
        self.minimum_active_guarantee: np.ndarray | None = None
        self.minimum_identity_replay: np.ndarray | None = None
        self.kind_count: np.ndarray | None = None

    def _initialize(self, times: int) -> None:
        if self.times is not None:
            verify_inp(self.times == times, "static-spread trading times changed")
            return
        self.times = times
        integer = lambda: np.zeros(times, dtype=np.int64)
        floating = lambda: np.zeros(times, dtype=np.float64)
        self.active_count = integer()
        self.gain_count = integer()
        self.loss_count = integer()
        self.direct_active_count = integer()
        self.direct_gain_count = integer()
        self.direct_loss_count = integer()
        self.weighted_active_sum = floating()
        self.weighted_gain_sum = floating()
        self.weighted_loss_sum = floating()
        self.weighted_pnl_sum = floating()
        self.weight_square_pnl_sum = floating()
        self.weight_square_pnl_square_sum = floating()
        self.direct_pnl_sum = floating()
        self.direct_pnl_square_sum = floating()
        self.minimum_pnl = np.full(times, math.inf, dtype=np.float64)
        self.minimum_active_pnl = np.full(times, math.inf, dtype=np.float64)
        self.minimum_active_guarantee = np.full(times, math.inf, dtype=np.float64)
        self.minimum_identity_replay = np.full(times, math.inf, dtype=np.float64)
        self.kind_count = np.zeros((times, len(_KINDS)), dtype=np.int64)

    @staticmethod
    def _take_best(values: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
        flat = values.reshape(*values.shape[:2], -1)
        return np.take_along_axis(flat, flat_index[..., None], axis=2)[..., 0]

    def update(
        self,
        universe: Mapping[str, np.ndarray],
        case: DerivedCaseArrays,
    ) -> None:
        """Update from one expiry/raw block of the matched universe."""

        verify_inp(
            case.return_horizon == "expiry" and case.hedge == "raw",
            "static-spread strategy requires raw expiry returns",
        )
        calls = np.asarray(universe["C"], dtype=np.float64)
        option_return = np.asarray(universe["dC"], dtype=np.float64)
        spot = np.asarray(universe["spot"], dtype=np.float64)
        relative_strike = np.exp(
            np.asarray(universe["log_strikes"], dtype=np.float64)
        )
        verify_inp(
            calls.shape == option_return.shape == relative_strike.shape
            and calls.ndim == 4
            and spot.shape == calls.shape[:2],
            "static-spread inputs have inconsistent geometry",
        )
        block_paths, times, expiries, strikes = calls.shape
        verify_inp(strikes >= 3, "static-spread strategy requires three strikes")
        self._initialize(times)
        assert self.active_count is not None
        spot_instruments = expiries
        option_cost = case.costs[..., spot_instruments:].reshape(calls.shape).astype(
            np.float64, copy=False
        )
        option_band = case.numerical_band[
            ..., spot_instruments:
        ].reshape(calls.shape).astype(np.float64, copy=False)
        cash_strike = spot[..., None, None] * relative_strike

        best_score = np.full((block_paths, times), -math.inf, dtype=np.float64)
        best_pnl = np.zeros((block_paths, times), dtype=np.float64)
        best_dynamic_band = np.zeros_like(best_pnl)
        best_guarantee = np.zeros_like(best_pnl)
        best_kind = np.full((block_paths, times), -1, dtype=np.int8)

        def consider(
            *,
            kind: int,
            violation: np.ndarray,
            gross_cost: np.ndarray,
            arithmetic_scale: np.ndarray,
            pnl: np.ndarray,
            dynamic_band: np.ndarray,
        ) -> None:
            state_band = _EPSILON * np.maximum(
                spot[..., None, None], arithmetic_scale + gross_cost
            )
            guarantee = 0.5 * (violation - gross_cost)
            score = guarantee - 0.5 * state_band
            flat_index = np.argmax(score.reshape(block_paths, times, -1), axis=2)
            kind_score = self._take_best(score, flat_index)
            replace = kind_score > best_score
            best_score[replace] = kind_score[replace]
            best_pnl[replace] = self._take_best(pnl, flat_index)[replace]
            best_dynamic_band[replace] = self._take_best(
                dynamic_band, flat_index
            )[replace]
            best_guarantee[replace] = self._take_best(
                guarantee, flat_index
            )[replace]
            best_kind[replace] = kind

        lower_call = calls[..., :-1]
        upper_call = calls[..., 1:]
        lower_return = option_return[..., :-1]
        upper_return = option_return[..., 1:]
        lower_cost = option_cost[..., :-1]
        upper_cost = option_cost[..., 1:]
        adjacent_cost = lower_cost + upper_cost
        adjacent_dynamic_band = 0.5 * (
            option_band[..., :-1]
            + option_band[..., 1:]
            + _EPSILON
            * (
                np.abs(lower_return)
                + np.abs(upper_return)
                + adjacent_cost
            )
        )
        consider(
            kind=0,
            violation=upper_call - lower_call,
            gross_cost=adjacent_cost,
            arithmetic_scale=np.abs(lower_call) + np.abs(upper_call),
            pnl=0.5 * (lower_return - upper_return - adjacent_cost),
            dynamic_band=adjacent_dynamic_band,
        )
        strike_width = cash_strike[..., 1:] - cash_strike[..., :-1]
        consider(
            kind=1,
            violation=lower_call - upper_call - strike_width,
            gross_cost=adjacent_cost,
            arithmetic_scale=(
                np.abs(lower_call) + np.abs(upper_call) + np.abs(strike_width)
            ),
            pnl=0.5 * (-lower_return + upper_return - adjacent_cost),
            dynamic_band=adjacent_dynamic_band,
        )

        left_width = cash_strike[..., 1:-1] - cash_strike[..., :-2]
        right_width = cash_strike[..., 2:] - cash_strike[..., 1:-1]
        total_width = left_width + right_width
        verify_inp(
            np.all(left_width > 0.0) and np.all(right_width > 0.0),
            "traded cash strikes must be strictly increasing",
        )
        lambda_left = right_width / total_width
        lambda_right = left_width / total_width
        convex_cost = (
            lambda_left * option_cost[..., :-2]
            + option_cost[..., 1:-1]
            + lambda_right * option_cost[..., 2:]
        )
        convex_dynamic_band = 0.5 * (
            lambda_left * option_band[..., :-2]
            + option_band[..., 1:-1]
            + lambda_right * option_band[..., 2:]
            + _EPSILON
            * (
                lambda_left * np.abs(option_return[..., :-2])
                + np.abs(option_return[..., 1:-1])
                + lambda_right * np.abs(option_return[..., 2:])
                + convex_cost
            )
        )
        consider(
            kind=2,
            violation=(
                calls[..., 1:-1]
                - lambda_left * calls[..., :-2]
                - lambda_right * calls[..., 2:]
            ),
            gross_cost=convex_cost,
            arithmetic_scale=(
                np.abs(calls[..., 1:-1])
                + lambda_left * np.abs(calls[..., :-2])
                + lambda_right * np.abs(calls[..., 2:])
            ),
            pnl=0.5
            * (
                lambda_left * option_return[..., :-2]
                - option_return[..., 1:-1]
                + lambda_right * option_return[..., 2:]
                - convex_cost
            ),
            dynamic_band=convex_dynamic_band,
        )

        active = best_score > 0.0
        pnl = np.where(active, best_pnl, 0.0)
        dynamic_band = np.where(active, best_dynamic_band, 0.0)
        guarantee = np.where(active, best_guarantee, 0.0)
        loss = pnl < -dynamic_band
        gain = pnl > dynamic_band
        log_weight = np.asarray(
            universe.get("log_path_weight", np.zeros(block_paths)),
            dtype=np.float64,
        )
        weights = np.exp(log_weight)
        component = np.asarray(
            universe.get("importance_component", np.zeros(block_paths)),
            dtype=np.int16,
        )
        direct = component == 0
        weight = weights[:, None]
        weight_square = np.square(weight)

        self.paths += block_paths
        self.direct_paths += int(np.count_nonzero(direct))
        self.weight_sum += float(np.sum(weights))
        self.weight_square_sum += float(np.sum(np.square(weights)))
        self.active_count += np.sum(active, axis=0)
        self.gain_count += np.sum(gain, axis=0)
        self.loss_count += np.sum(loss, axis=0)
        self.direct_active_count += np.sum(active[direct], axis=0)
        self.direct_gain_count += np.sum(gain[direct], axis=0)
        self.direct_loss_count += np.sum(loss[direct], axis=0)
        self.weighted_active_sum += np.sum(weight * active, axis=0)
        self.weighted_gain_sum += np.sum(weight * gain, axis=0)
        self.weighted_loss_sum += np.sum(weight * loss, axis=0)
        self.weighted_pnl_sum += np.sum(weight * pnl, axis=0)
        self.weight_square_pnl_sum += np.sum(weight_square * pnl, axis=0)
        self.weight_square_pnl_square_sum += np.sum(
            np.square(weight * pnl), axis=0
        )
        self.direct_pnl_sum += np.sum(pnl[direct], axis=0)
        self.direct_pnl_square_sum += np.sum(np.square(pnl[direct]), axis=0)
        self.minimum_pnl = np.minimum(self.minimum_pnl, np.min(pnl, axis=0))
        self.minimum_active_pnl = np.minimum(
            self.minimum_active_pnl,
            np.min(np.where(active, pnl, math.inf), axis=0),
        )
        active_guarantee = np.where(active, guarantee, math.inf)
        self.minimum_active_guarantee = np.minimum(
            self.minimum_active_guarantee,
            np.min(active_guarantee, axis=0),
        )
        self.minimum_identity_replay = np.minimum(
            self.minimum_identity_replay,
            np.min(
                np.where(active, pnl - guarantee + dynamic_band, math.inf),
                axis=0,
            ),
        )
        for kind in range(len(_KINDS)):
            self.kind_count[:, kind] += np.sum(active & (best_kind == kind), axis=0)

    @staticmethod
    def _gain_lower(gains: int, paths: int, alpha: float) -> float:
        return float(beta.ppf(alpha, gains, paths - gains + 1)) if gains else 0.0

    def report(self) -> dict[str, Any]:
        """Return per-time incidence, P&L, and pathwise-identity metrics."""

        verify_inp(self.paths > 0 and self.times is not None, "no spread paths")
        assert self.active_count is not None
        adjusted_alpha = self.alpha / self.times
        z = float(norm.ppf(1.0 - adjusted_alpha))
        reports: list[dict[str, Any]] = []
        certified_times: list[int] = []
        for time_index in range(self.times):
            weighted_mean = self.weighted_pnl_sum[time_index] / self.weight_sum
            influence_square = (
                self.weight_square_pnl_square_sum[time_index]
                - 2.0
                * weighted_mean
                * self.weight_square_pnl_sum[time_index]
                + weighted_mean**2 * self.weight_square_sum
            )
            weighted_error = math.sqrt(max(influence_square, 0.0)) / self.weight_sum
            direct_mean = (
                self.direct_pnl_sum[time_index] / self.direct_paths
                if self.direct_paths
                else 0.0
            )
            direct_variance = max(
                self.direct_pnl_square_sum[time_index] / self.direct_paths
                - direct_mean**2,
                0.0,
            )
            direct_error = math.sqrt(direct_variance / self.direct_paths)
            gain_lower = self._gain_lower(
                int(self.direct_gain_count[time_index]),
                self.direct_paths,
                adjusted_alpha,
            )
            active_lower = self._gain_lower(
                int(self.direct_active_count[time_index]),
                self.direct_paths,
                adjusted_alpha,
            )
            conditions = {
                "no_economic_loss": self.minimum_pnl[time_index] >= 0.0,
                "no_loss_beyond_numerical_band": self.loss_count[time_index] == 0,
                "strictly_positive_state_known_guarantee": (
                    self.minimum_active_guarantee[time_index] > 0.0
                ),
                "pathwise_payoff_identity_replayed": (
                    self.minimum_identity_replay[time_index] >= 0.0
                ),
                "adjusted_direct_P_active_probability_lower_positive": (
                    active_lower > 0.0
                ),
                "physical_weighted_active_probability_positive": (
                    self.weighted_active_sum[time_index] > 0.0
                ),
            }
            additional_mean_conditions = {
                "adjusted_direct_P_mean_lower_positive": (
                    direct_mean - z * direct_error > 0.0
                ),
                "adjusted_weighted_P_mean_lower_positive": (
                    weighted_mean - z * weighted_error > 0.0
                ),
            }
            certified = bool(all(conditions.values()))
            if certified:
                certified_times.append(time_index)
            reports.append(
                {
                    "trading_time_index": time_index,
                    "active_paths": int(self.active_count[time_index]),
                    "active_path_fraction": float(
                        self.active_count[time_index] / self.paths
                    ),
                    "physical_weighted_active_probability": float(
                        self.weighted_active_sum[time_index] / self.weight_sum
                    ),
                    "gain_count": int(self.gain_count[time_index]),
                    "loss_count": int(self.loss_count[time_index]),
                    "direct_P_paths": self.direct_paths,
                    "direct_P_active_count": int(
                        self.direct_active_count[time_index]
                    ),
                    "adjusted_direct_P_active_probability_lower": active_lower,
                    "direct_P_gain_count": int(self.direct_gain_count[time_index]),
                    "direct_P_loss_count": int(self.direct_loss_count[time_index]),
                    "adjusted_direct_P_gain_probability_lower": gain_lower,
                    "physical_weighted_gain_probability": float(
                        self.weighted_gain_sum[time_index] / self.weight_sum
                    ),
                    "physical_weighted_loss_probability": float(
                        self.weighted_loss_sum[time_index] / self.weight_sum
                    ),
                    "direct_P_mean": direct_mean,
                    "direct_P_mean_standard_error": direct_error,
                    "adjusted_direct_P_mean_lower": direct_mean - z * direct_error,
                    "physical_weighted_mean": weighted_mean,
                    "physical_weighted_mean_standard_error": weighted_error,
                    "adjusted_physical_weighted_mean_lower": (
                        weighted_mean - z * weighted_error
                    ),
                    "minimum_economic_pnl": float(self.minimum_pnl[time_index]),
                    "minimum_active_economic_pnl": float(
                        self.minimum_active_pnl[time_index]
                    ),
                    "minimum_active_guaranteed_pnl": float(
                        self.minimum_active_guarantee[time_index]
                    ),
                    "minimum_payoff_identity_pnl_plus_band": float(
                        self.minimum_identity_replay[time_index]
                    ),
                    "active_count_by_spread": {
                        name: int(self.kind_count[time_index, kind])
                        for kind, name in enumerate(_KINDS)
                    },
                    "conditions": conditions,
                    "additional_mean_conditions": additional_mean_conditions,
                    "certified": certified,
                }
            )
        return {
            "applicability": "raw option returns held to expiry",
            "strategy": (
                "predictable state-dependent maximizer over every traded-grid "
                "call-monotonicity, vertical-spread, and convexity portfolio; "
                "zero outside the positive guaranteed-margin region"
            ),
            "uses_future_outcomes_to_select_action": False,
            "requires_reference_audit_split": False,
            "l1_norm_when_active": 1.0,
            "roundoff_multiplier": _ROUNDOFF_MULTIPLIER,
            "paths": self.paths,
            "direct_P_paths": self.direct_paths,
            "mean_path_likelihood": self.weight_sum / self.paths,
            "multiplicity": {
                "method": "Bonferroni over trading times",
                "family_size": self.times,
                "adjusted_alpha": adjusted_alpha,
            },
            "times": reports,
            "certified_trading_times": certified_times,
            "has_pathwise_dynamic_arbitrage": bool(certified_times),
        }


__all__ = ["StateDependentStaticSpreadAccumulator"]
