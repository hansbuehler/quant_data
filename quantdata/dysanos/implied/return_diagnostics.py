"""Streaming matched SANOS return diagnostics for PCA universe blocks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from cdxcore.err import verify_inp

from quantdata.dysanos.grid.gendysanos.run_terminal_arbitrage_tests import (
    _returns_and_roundoff,
)
from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    SignAccumulator,
    delta_hedged_option_cost_notionals,
    net_return_roundoff_band,
    rank_no_arbitrage_candidates,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: _json_value(item) for name, item in value.items()}
    return value


def _zero_success_upper_probability(observations: int) -> float:
    """Exact one-sided 95% binomial upper bound after zero successes."""

    return 1.0 - math.pow(0.05, 1.0 / observations) if observations else 1.0


class ReturnDiagnosticsAccumulator:
    """Accumulate physical-measure signs, coverage, costs, and weights online."""

    def __init__(
        self,
        *,
        expiries: int,
        strikes: int,
        cost_proportion: float = 1.0e-4,
    ) -> None:
        verify_inp(
            isinstance(expiries, int)
            and isinstance(strikes, int)
            and expiries > 0
            and strikes > 0,
            lambda: f"invalid return geometry {expiries}/{strikes}",
        )
        verify_inp(
            math.isfinite(cost_proportion) and cost_proportion >= 0.0,
            lambda: f"invalid proportional cost {cost_proportion}",
        )
        self.expiries = expiries
        self.strikes = strikes
        self.option_instruments = expiries * strikes
        self.instruments = 1 + self.option_instruments
        self.cost_proportion = cost_proportion
        self.signs = SignAccumulator(instruments=self.instruments)
        self.cost_spot_long = SignAccumulator(instruments=1)
        self.cost_spot_short = SignAccumulator(instruments=1)
        self.cost_long = SignAccumulator(instruments=self.option_instruments)
        self.cost_short = SignAccumulator(instruments=self.option_instruments)
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.weighted_total = np.zeros(self.instruments, dtype=np.float64)
        self.weighted_square_total = np.zeros(self.instruments, dtype=np.float64)
        self.weight_square_path_mean_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.weight_square_path_mean_square_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.weighted_positive = np.zeros(self.instruments, dtype=np.float64)
        self.weighted_negative = np.zeros(self.instruments, dtype=np.float64)
        self.weight_square_path_positive_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.weight_square_path_positive_square_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.weight_square_path_negative_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.weight_square_path_negative_square_total = np.zeros(
            self.instruments, dtype=np.float64
        )
        self.paths_with_both_signs = np.zeros(self.instruments, dtype=np.int64)
        self.spot_crossing_count = np.zeros(
            self.option_instruments, dtype=np.int64
        )
        self.spot_non_crossing_count = np.zeros(
            self.option_instruments, dtype=np.int64
        )
        self.weighted_spot_crossing = np.zeros(
            self.option_instruments, dtype=np.float64
        )
        self.weighted_spot_non_crossing = np.zeros(
            self.option_instruments, dtype=np.float64
        )
        self.return_horizon: str | None = None
        self.minimum_log_weight = math.inf
        self.maximum_log_weight = -math.inf
        self.cost_weighted_positive = np.zeros(
            (2, self.instruments), dtype=np.float64
        )
        self.cost_weighted_negative = np.zeros_like(
            self.cost_weighted_positive
        )
        self.paths = 0
        self.steps: int | None = None

    def update(self, universe: dict[str, np.ndarray]) -> None:
        returns, roundoff = _returns_and_roundoff(universe)
        S, steps, instruments = returns.shape
        verify_inp(
            instruments == self.instruments,
            lambda: f"return instrument count changed to {instruments}",
        )
        if self.steps is None:
            self.steps = steps
        verify_inp(
            self.steps == steps,
            lambda: f"return action count changed from {self.steps} to {steps}",
        )
        self.signs.update(returns, roundoff)
        positive = returns > roundoff
        negative = returns < -roundoff
        self.paths_with_both_signs += (
            positive.any(axis=1) & negative.any(axis=1)
        ).sum(axis=0, dtype=np.int64)

        log_weight = np.asarray(
            universe.get("log_path_weight", np.zeros(S)), dtype=np.float64
        )
        verify_inp(
            log_weight.shape == (S,) and np.all(np.isfinite(log_weight)),
            lambda: f"invalid path likelihoods {log_weight.shape}",
        )
        weight = np.exp(log_weight)
        verify_inp(
            np.all(np.isfinite(weight))
            and np.all(weight >= 0.0)
            and np.any(weight > 0.0),
            "physical path likelihoods must be finite, nonnegative, and nonzero",
        )
        self.minimum_log_weight = min(
            self.minimum_log_weight, float(np.min(log_weight))
        )
        self.maximum_log_weight = max(
            self.maximum_log_weight, float(np.max(log_weight))
        )
        self.weight_sum += float(np.sum(weight))
        self.weight_square_sum += float(np.sum(np.square(weight)))
        values = returns.astype(np.float64, copy=False)
        mass_weight = np.broadcast_to(weight[:, None, None], values.shape)
        self.weighted_total += np.sum(mass_weight * values, axis=(0, 1))
        self.weighted_square_total += np.sum(
            mass_weight * np.square(values), axis=(0, 1)
        )
        self.weighted_positive += np.sum(
            mass_weight * (returns > roundoff), axis=(0, 1)
        )
        self.weighted_negative += np.sum(
            mass_weight * (returns < -roundoff), axis=(0, 1)
        )

        # Standard errors use independent paths as clusters. The expression
        # retained here is enough to evaluate the self-normalized importance
        # estimate sqrt(sum(w^2 (x-mu)^2))/sum(w) after the stream completes.
        weight_square = np.square(weight)[:, None]
        path_mean = np.mean(values, axis=1)
        path_positive = np.mean(positive, axis=1)
        path_negative = np.mean(negative, axis=1)
        self.weight_square_path_mean_total += np.sum(
            weight_square * path_mean, axis=0
        )
        self.weight_square_path_mean_square_total += np.sum(
            weight_square * np.square(path_mean), axis=0
        )
        self.weight_square_path_positive_total += np.sum(
            weight_square * path_positive, axis=0
        )
        self.weight_square_path_positive_square_total += np.sum(
            weight_square * np.square(path_positive), axis=0
        )
        self.weight_square_path_negative_total += np.sum(
            weight_square * path_negative, axis=0
        )
        self.weight_square_path_negative_square_total += np.sum(
            weight_square * np.square(path_negative), axis=0
        )

        horizon = "expiry" if "expiry_dS" in universe else "one-day"
        if self.return_horizon is None:
            self.return_horizon = horizon
        verify_inp(
            self.return_horizon == horizon,
            lambda: (
                f"option return horizon changed from {self.return_horizon} "
                f"to {horizon}"
            ),
        )
        option_dS = np.asarray(
            universe["expiry_dS" if horizon == "expiry" else "option_dS"],
            dtype=np.float32,
        )
        entry_spot = np.asarray(universe["spot"], dtype=np.float32)
        relative_strikes = np.exp(
            np.asarray(universe["log_strikes"], dtype=np.float32)
        )
        crossed = (
            entry_spot[..., None, None] + option_dS[..., None]
            > entry_spot[..., None, None] * relative_strikes
        ).reshape(S, steps, self.option_instruments)
        self.spot_crossing_count += crossed.sum(axis=(0, 1), dtype=np.int64)
        self.spot_non_crossing_count += (~crossed).sum(
            axis=(0, 1), dtype=np.int64
        )
        option_weight = np.broadcast_to(
            weight[:, None, None], (S, steps, self.option_instruments)
        )
        self.weighted_spot_crossing += np.sum(
            option_weight * crossed, axis=(0, 1)
        )
        self.weighted_spot_non_crossing += np.sum(
            option_weight * (~crossed), axis=(0, 1)
        )

        option_returns = returns[..., 1:]
        spot_cost = self.cost_proportion * np.abs(entry_spot)[..., None]
        spot_returns = returns[..., :1]
        spot_long = spot_returns - spot_cost
        spot_short = -spot_returns - spot_cost
        spot_long_roundoff = net_return_roundoff_band(spot_returns, spot_cost)
        spot_short_roundoff = net_return_roundoff_band(-spot_returns, spot_cost)
        self.cost_spot_long.update(spot_long, spot_long_roundoff)
        self.cost_spot_short.update(spot_short, spot_short_roundoff)
        gross_notional = delta_hedged_option_cost_notionals(
            C=np.asarray(universe["C"], dtype=np.float32),
            delta=np.asarray(universe["delta"], dtype=np.float32),
            spot=np.asarray(universe["spot"], dtype=np.float32),
        )
        cost = self.cost_proportion * gross_notional
        long = option_returns - cost
        short = -option_returns - cost
        long_roundoff = net_return_roundoff_band(option_returns, cost)
        short_roundoff = net_return_roundoff_band(-option_returns, cost)
        self.cost_long.update(long, long_roundoff)
        self.cost_short.update(short, short_roundoff)
        path_weight = weight[:, None, None]
        self.cost_weighted_positive[0, 0] += np.sum(
            path_weight * (spot_long > spot_long_roundoff)
        )
        self.cost_weighted_negative[0, 0] += np.sum(
            path_weight * (spot_long < -spot_long_roundoff)
        )
        self.cost_weighted_positive[1, 0] += np.sum(
            path_weight * (spot_short > spot_short_roundoff)
        )
        self.cost_weighted_negative[1, 0] += np.sum(
            path_weight * (spot_short < -spot_short_roundoff)
        )
        self.cost_weighted_positive[0, 1:] += np.sum(
            option_weight * (long > long_roundoff), axis=(0, 1)
        )
        self.cost_weighted_negative[0, 1:] += np.sum(
            option_weight * (long < -long_roundoff), axis=(0, 1)
        )
        self.cost_weighted_positive[1, 1:] += np.sum(
            option_weight * (short > short_roundoff), axis=(0, 1)
        )
        self.cost_weighted_negative[1, 1:] += np.sum(
            option_weight * (short < -short_roundoff), axis=(0, 1)
        )
        self.paths += S

    def report(self) -> dict[str, Any]:
        verify_inp(self.paths > 0 and self.steps is not None, "no returns accumulated")
        arrays = self.signs.arrays()
        observations = self.paths * self.steps
        weighted_mass = self.weight_sum * self.steps
        weighted_mean = self.weighted_total / weighted_mass
        weighted_variance = np.maximum(
            self.weighted_square_total / weighted_mass
            - np.square(weighted_mean),
            0.0,
        )
        weighted_positive = self.weighted_positive / weighted_mass
        weighted_negative = self.weighted_negative / weighted_mass
        weighted_minority = np.minimum(weighted_positive, weighted_negative)
        weighted_mean_error = np.sqrt(
            np.maximum(
                self.weight_square_path_mean_square_total
                - 2.0
                * weighted_mean
                * self.weight_square_path_mean_total
                + np.square(weighted_mean) * self.weight_square_sum,
                0.0,
            )
        ) / self.weight_sum
        weighted_positive_error = np.sqrt(
            np.maximum(
                self.weight_square_path_positive_square_total
                - 2.0
                * weighted_positive
                * self.weight_square_path_positive_total
                + np.square(weighted_positive) * self.weight_square_sum,
                0.0,
            )
        ) / self.weight_sum
        weighted_negative_error = np.sqrt(
            np.maximum(
                self.weight_square_path_negative_square_total
                - 2.0
                * weighted_negative
                * self.weight_square_path_negative_total
                + np.square(weighted_negative) * self.weight_square_sum,
                0.0,
            )
        ) / self.weight_sum
        imbalance = np.divide(
            np.abs(weighted_positive - weighted_negative),
            weighted_positive + weighted_negative,
            out=np.zeros_like(weighted_positive),
            where=weighted_positive + weighted_negative > 0.0,
        )
        option_positive_paths = np.asarray(arrays["positive_paths"])[1:]
        option_negative_paths = np.asarray(arrays["negative_paths"])[1:]
        missing_positive = np.flatnonzero(option_positive_paths == 0)
        missing_negative = np.flatnonzero(option_negative_paths == 0)
        rank = np.argsort(weighted_minority[1:], kind="stable")[:20]
        raw_imbalance = np.asarray(arrays["imbalance"])
        raw_rank = np.argsort(-raw_imbalance, kind="stable")[:20]

        def instrument(option: int) -> dict[str, Any]:
            index = option + 1
            return {
                "instrument": index,
                "expiry_index": option // self.strikes,
                "strike_index": option % self.strikes,
                "weighted_minority_probability": float(
                    weighted_minority[index]
                ),
                "weighted_normalized_imbalance": float(imbalance[index]),
                "weighted_positive_probability": float(
                    weighted_positive[index]
                ),
                "weighted_negative_probability": float(
                    weighted_negative[index]
                ),
                "positive_paths": int(option_positive_paths[option]),
                "negative_paths": int(option_negative_paths[option]),
            }

        def raw_instrument(index: int) -> dict[str, Any]:
            result: dict[str, Any] = {
                "instrument": index,
                "kind": "spot" if index == 0 else "delta_hedged_option",
                "imbalance": int(raw_imbalance[index]),
                "imbalance_fraction_all": float(
                    np.asarray(arrays["imbalance_fraction_all"])[index]
                ),
            }
            if index > 0:
                option = index - 1
                result.update(
                    expiry_index=option // self.strikes,
                    strike_index=option % self.strikes,
                )
            return result

        crossing_probability = self.weighted_spot_crossing / weighted_mass
        non_crossing_probability = (
            self.weighted_spot_non_crossing / weighted_mass
        )
        assert self.return_horizon is not None
        return {
            "return_semantics": (
                "spot next-action return plus fixed-contract delta-hedged option "
                "return; each option uses its matched expiry or explicitly "
                "labelled one-day repricing horizon"
            ),
            "measure": (
                "original physical P; proposal paths carry the complete path "
                "likelihood p/q exactly once"
            ),
            "paths": self.paths,
            "steps": self.steps,
            "observations_per_instrument": observations,
            "standard_error_convention": (
                "population moments over action dates with independent Monte "
                "Carlo paths as clusters; path_clustered_mean_error uses one "
                "within-path action-date mean per independent path"
            ),
            "path_weights": {
                "mean_p_over_q": self.weight_sum / self.paths,
                "minimum_log_weight": self.minimum_log_weight,
                "maximum_log_weight": self.maximum_log_weight,
                "effective_sample_size": (
                    self.weight_sum**2 / self.weight_square_sum
                ),
                "effective_sample_fraction": (
                    self.weight_sum**2
                    / self.weight_square_sum
                    / self.paths
                ),
            },
            "per_instrument": _json_value(
                arrays
                | {
                    "weighted_mean": weighted_mean,
                    "weighted_std_population": np.sqrt(weighted_variance),
                    "weighted_path_clustered_mean_error": weighted_mean_error,
                    "weighted_positive_probability": weighted_positive,
                    "weighted_path_clustered_positive_probability_error": (
                        weighted_positive_error
                    ),
                    "weighted_negative_probability": weighted_negative,
                    "weighted_path_clustered_negative_probability_error": (
                        weighted_negative_error
                    ),
                    "weighted_minority_probability": weighted_minority,
                    "weighted_normalized_imbalance": imbalance,
                    "paths_with_at_least_one_positive_and_negative": (
                        self.paths_with_both_signs
                    ),
                }
            ),
            "zero_observed_option_sides": {
                "without_positive_path": missing_positive.astype(int).tolist(),
                "without_negative_path": missing_negative.astype(int).tolist(),
                "zero_count_95_percent_binomial_upper_probability": (
                    _zero_success_upper_probability(self.paths)
                ),
            },
            "payoff_and_break_even_coverage": {
                "option_return_horizon": self.return_horizon,
                "expiry_payoff_observed": self.return_horizon == "expiry",
                "proposal_positive_payoff_count": (
                    self.spot_crossing_count.tolist()
                    if self.return_horizon == "expiry"
                    else None
                ),
                "proposal_zero_payoff_count": (
                    self.spot_non_crossing_count.tolist()
                    if self.return_horizon == "expiry"
                    else None
                ),
                "proposal_spot_crossing_count": (
                    self.spot_crossing_count.tolist()
                ),
                "proposal_spot_non_crossing_count": (
                    self.spot_non_crossing_count.tolist()
                ),
                "weighted_P_spot_crossing_probability": (
                    crossing_probability.tolist()
                ),
                "weighted_P_spot_non_crossing_probability": (
                    non_crossing_probability.tolist()
                ),
                "options_with_both_spot_sides": int(
                    np.count_nonzero(
                        (self.spot_crossing_count > 0)
                        & (self.spot_non_crossing_count > 0)
                    )
                ),
                "options_with_both_delta_hedged_break_even_sides": int(
                    np.count_nonzero(
                        (option_positive_paths > 0) & (option_negative_paths > 0)
                    )
                ),
                "raw_spot_crossing_definition": (
                    "same-path option-horizon spot strictly above the fixed "
                    "cash strike; at expiry this is equivalent to a strictly "
                    "positive call payoff"
                ),
            },
            "worst_20_by_minority_probability": [
                instrument(int(option)) for option in rank
            ],
            "worst_20_by_raw_imbalance": [
                raw_instrument(int(index)) for index in raw_rank
            ],
            "transaction_costs": {
                "proportion": self.cost_proportion,
                "notional": (
                    "spot: abs(S); delta-hedged option: "
                    "abs(C)+abs(delta*S), without netting"
                ),
                "long_and_short_tested_separately": True,
                "spot_long": _json_value(self.cost_spot_long.arrays()),
                "spot_short": _json_value(self.cost_spot_short.arrays()),
                "option_long": _json_value(self.cost_long.arrays()),
                "option_short": _json_value(self.cost_short.arrays()),
                "physical_weighted_positive_probability_by_direction": (
                    self.cost_weighted_positive / weighted_mass
                ).tolist(),
                "physical_weighted_negative_probability_by_direction": (
                    self.cost_weighted_negative / weighted_mass
                ).tolist(),
                "worst_20_directions": rank_no_arbitrage_candidates(
                    self.cost_long,
                    self.cost_short,
                    strikes=self.strikes,
                    count=20,
                ),
            },
        }


__all__ = ["ReturnDiagnosticsAccumulator"]
