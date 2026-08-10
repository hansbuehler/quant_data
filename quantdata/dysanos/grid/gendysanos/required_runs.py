"""Canonical case construction and statistical checks for required runs.

This module contains the generator-independent part of the experiment.  A
generator supplies one coherent path block using the common universe schema;
the routines below derive every hedge and transaction-cost case without
resimulating that block.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np
from cdxcore.err import verify_inp
from scipy.optimize import linprog
from scipy.stats import beta, norm

from quantdata.dysanos.grid.rndysanos.unconditional_option_test import (
    DEFAULT_ROUNDOFF_MULTIPLIER,
    FLOAT32_UNIT_ROUNDOFF,
)

Generator = Literal["dysanos", "pca-iv"]
TradingFrequency = Literal["daily", "weekly"]
ReturnHorizon = Literal["expiry", "incremental"]
Hedge = Literal["raw", "delta"]

REPORTED_PATH_COUNTS = (100_000, 1_000_000)
REPORTED_COST_RATES = (0.0, 1.0e-4, 1.0e-3)


def positive_relative_weights(log_weights: np.ndarray) -> np.ndarray:
    """Exponentiate log weights stably for self-normalized calculations.

    Likelihood ratios can be arbitrarily close to zero. Subtracting the
    largest log weight preserves every self-normalized statistic; flooring at
    the smallest positive float64 value only represents values which would
    otherwise underflow to zero and has no measurable effect on the result.
    """

    values = np.asarray(log_weights, dtype=np.float64)
    verify_inp(values.size > 0, "relative likelihood weights cannot be empty")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("log likelihood weights contain NaN or infinity")
    shifted = values - np.max(values)
    weights = np.exp(np.maximum(shifted, math.log(np.finfo(np.float64).tiny)))
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise FloatingPointError("relative likelihood weights are not finite-positive")
    return weights
GPU_ASSIGNMENT = {
    ("dysanos", "daily"): 0,
    ("dysanos", "weekly"): 1,
    ("pca-iv", "daily"): 2,
    ("pca-iv", "weekly"): 3,
}


@dataclass(frozen=True, kw_only=True)
class RequiredCase:
    """One of the 48 reported cases at a fixed Monte Carlo path count."""

    generator: Generator
    trading_frequency: TradingFrequency
    return_horizon: ReturnHorizon
    hedge: Hedge
    cost_rate: float
    path_count: int
    num_trading_times: int = 10
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        verify_inp(
            self.generator in {"dysanos", "pca-iv"},
            lambda: f"unsupported generator {self.generator!r}",
        )
        verify_inp(
            self.trading_frequency in {"daily", "weekly"},
            lambda: f"unsupported trading frequency {self.trading_frequency!r}",
        )
        verify_inp(
            self.return_horizon in {"expiry", "incremental"},
            lambda: f"unsupported return horizon {self.return_horizon!r}",
        )
        verify_inp(
            self.hedge in {"raw", "delta"},
            lambda: f"unsupported hedge convention {self.hedge!r}",
        )
        verify_inp(
            self.cost_rate in REPORTED_COST_RATES,
            lambda: f"unsupported reported cost rate {self.cost_rate!r}",
        )
        verify_inp(
            isinstance(self.path_count, int) and self.path_count > 1,
            lambda: f"path count must exceed one, got {self.path_count!r}",
        )
        verify_inp(
            self.num_trading_times == 10,
            lambda: (
                "reported markets require exactly ten trading timestamps, got "
                f"{self.num_trading_times}"
            ),
        )

    @property
    def return_times(self) -> int:
        """Number of time-resolved return cross-sections."""

        return 10 if self.return_horizon == "expiry" else 9

    @property
    def gpu(self) -> int:
        """Required GPU assignment for this generator-frequency pair."""

        return GPU_ASSIGNMENT[(self.generator, self.trading_frequency)]

    @property
    def base_key(self) -> tuple[str, str, int]:
        """Identity of the path sample shared by all twelve derived cases."""

        return self.generator, self.trading_frequency, self.path_count


def required_cases(path_count: int) -> tuple[RequiredCase, ...]:
    """Return the complete 48-case matrix for one reported path count."""

    return tuple(
        RequiredCase(
            generator=generator,
            trading_frequency=frequency,
            return_horizon=horizon,
            hedge=hedge,
            cost_rate=cost,
            path_count=path_count,
        )
        for generator in ("dysanos", "pca-iv")
        for frequency in ("daily", "weekly")
        for horizon in ("expiry", "incremental")
        for hedge in ("delta", "raw")
        for cost in REPORTED_COST_RATES
    )


@dataclass(frozen=True, kw_only=True)
class DerivedCaseArrays:
    """Pre-cost returns, costs, and observation-specific numerical bands."""

    returns: np.ndarray
    costs: np.ndarray
    numerical_band: np.ndarray
    instrument_kind: tuple[str, ...]
    expiry_index: np.ndarray
    strike_index: np.ndarray
    return_horizon: ReturnHorizon
    hedge: Hedge
    cost_rate: float

    def directional(self, direction: Literal["long", "short"]) -> np.ndarray:
        """Return net P&L for a unit long or short direction."""

        verify_inp(
            direction in {"long", "short"},
            lambda: f"unknown direction {direction!r}",
        )
        signed = self.returns if direction == "long" else -self.returns
        return signed - self.costs


class TimeSeparatedUnconditionalAccumulator:
    """Stream unit-direction diagnostics without pooling trading times."""

    def __init__(self, *, alpha: float = 0.05) -> None:
        verify_inp(0.0 < alpha < 1.0, lambda: f"invalid alpha {alpha}")
        self.alpha = alpha
        self.paths = 0
        self.direct_paths = 0
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self._shape: tuple[int, int] | None = None

    def update(
        self,
        case: DerivedCaseArrays,
        *,
        log_path_weights: np.ndarray | None = None,
        importance_components: np.ndarray | None = None,
    ) -> None:
        """Accumulate one independent path block."""

        S, times, instruments = case.returns.shape
        shape = times, instruments
        if self._shape is None:
            self._shape = shape
            count_shape = (2, times, instruments)
            self.loss = np.zeros(count_shape, dtype=np.int64)
            self.gain = np.zeros(count_shape, dtype=np.int64)
            self.neutral = np.zeros(count_shape, dtype=np.int64)
            self.direct_loss = np.zeros(count_shape, dtype=np.int64)
            self.direct_gain = np.zeros(count_shape, dtype=np.int64)
            self.weighted_loss = np.zeros(count_shape, dtype=np.float64)
            self.weighted_gain = np.zeros(count_shape, dtype=np.float64)
            self.weighted_loss_square = np.zeros(count_shape, dtype=np.float64)
            self.weighted_gain_square = np.zeros(count_shape, dtype=np.float64)
            self.weighted_value_sum = np.zeros(count_shape, dtype=np.float64)
            self.weighted_value_square_sum = np.zeros(
                count_shape, dtype=np.float64
            )
            self.direct_value_sum = np.zeros(count_shape, dtype=np.float64)
            self.direct_value_square_sum = np.zeros(
                count_shape, dtype=np.float64
            )
            self.instrument_kind = case.instrument_kind
            self.expiry_index = case.expiry_index.copy()
            self.strike_index = case.strike_index.copy()
        verify_inp(self._shape == shape, "unconditional case geometry changed")
        verify_inp(
            self.instrument_kind == case.instrument_kind
            and np.array_equal(self.expiry_index, case.expiry_index)
            and np.array_equal(self.strike_index, case.strike_index),
            "unconditional instrument descriptors changed",
        )
        log_weights = (
            np.zeros(S, dtype=np.float64)
            if log_path_weights is None
            else np.asarray(log_path_weights, dtype=np.float64)
        )
        components = (
            np.zeros(S, dtype=np.int16)
            if importance_components is None
            else np.asarray(importance_components)
        )
        verify_inp(
            log_weights.shape == components.shape == (S,)
            and np.all(np.isfinite(log_weights)),
            "streaming path likelihood/component arrays have invalid geometry",
        )
        weights = np.exp(log_weights)
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("streaming path weights contain NaN or infinity")
        direct = components == 0
        weight = weights[:, None, None]
        for direction, sign in enumerate((1.0, -1.0)):
            value = sign * case.returns - case.costs
            loss = value < -case.numerical_band
            gain = value > case.numerical_band
            self.loss[direction] += loss.sum(axis=0, dtype=np.int64)
            self.gain[direction] += gain.sum(axis=0, dtype=np.int64)
            self.neutral[direction] += (~loss & ~gain).sum(
                axis=0, dtype=np.int64
            )
            self.direct_loss[direction] += loss[direct].sum(
                axis=0, dtype=np.int64
            )
            self.direct_gain[direction] += gain[direct].sum(
                axis=0, dtype=np.int64
            )
            weighted_value = weight * value.astype(np.float64, copy=False)
            self.weighted_loss[direction] += np.sum(weight * loss, axis=0)
            self.weighted_gain[direction] += np.sum(weight * gain, axis=0)
            self.weighted_loss_square[direction] += np.sum(
                np.square(weight * loss), axis=0
            )
            self.weighted_gain_square[direction] += np.sum(
                np.square(weight * gain), axis=0
            )
            self.weighted_value_sum[direction] += np.sum(
                weighted_value, axis=0
            )
            self.weighted_value_square_sum[direction] += np.sum(
                np.square(weighted_value), axis=0
            )
            self.direct_value_sum[direction] += np.sum(
                value[direct].astype(np.float64, copy=False), axis=0
            )
            self.direct_value_square_sum[direction] += np.sum(
                np.square(value[direct].astype(np.float64, copy=False)), axis=0
            )
        self.paths += S
        self.direct_paths += int(np.count_nonzero(direct))
        self.weight_sum += float(weights.sum())
        self.weight_square_sum += float(np.square(weights).sum())

    def report(self) -> dict[str, Any]:
        """Return complete time-resolved physical-law diagnostics."""

        verify_inp(
            self.paths > 0 and self._shape is not None,
            "cannot report an empty unconditional accumulator",
        )
        verify_inp(
            self.direct_paths > 0,
            "unconditional report has no direct-P paths",
        )
        times, instruments = self._shape
        mean = self.weighted_value_sum / self.paths
        variance = np.maximum(
            self.weighted_value_square_sum / self.paths - np.square(mean),
            0.0,
        )
        standard_error = np.sqrt(variance / self.paths)
        weighted_loss = self.weighted_loss / self.paths
        weighted_gain = self.weighted_gain / self.paths
        weighted_loss_error = np.sqrt(
            np.maximum(
                self.weighted_loss_square / self.paths - np.square(weighted_loss),
                0.0,
            )
            / self.paths
        )
        weighted_gain_error = np.sqrt(
            np.maximum(
                self.weighted_gain_square / self.paths - np.square(weighted_gain),
                0.0,
            )
            / self.paths
        )
        one_sided_95_z = float(norm.ppf(0.95))
        direct_mean = self.direct_value_sum / self.direct_paths
        direct_variance = np.maximum(
            self.direct_value_square_sum / self.direct_paths
            - np.square(direct_mean),
            0.0,
        )
        direct_standard_error = np.sqrt(direct_variance / self.direct_paths)
        reports: list[dict[str, Any]] = []
        for time_index in range(times):
            directions: dict[str, list[dict[str, Any]]] = {}
            for direction, name in enumerate(("long", "short")):
                items: list[dict[str, Any]] = []
                for instrument in range(instruments):
                    gain_lower, _ = clopper_pearson_one_sided(
                        int(self.direct_gain[direction, time_index, instrument]),
                        self.direct_paths,
                        alpha=self.alpha,
                    )
                    _, loss_upper = clopper_pearson_one_sided(
                        int(self.direct_loss[direction, time_index, instrument]),
                        self.direct_paths,
                        alpha=self.alpha,
                    )
                    error = standard_error[direction, time_index, instrument]
                    value_mean = mean[direction, time_index, instrument]
                    items.append(
                        {
                            "instrument": instrument,
                            "kind": self.instrument_kind[instrument],
                            "expiry_index": int(self.expiry_index[instrument]),
                            "strike_index": int(self.strike_index[instrument]),
                            "proposal_loss_count": int(
                                self.loss[direction, time_index, instrument]
                            ),
                            "proposal_neutral_count": int(
                                self.neutral[direction, time_index, instrument]
                            ),
                            "proposal_gain_count": int(
                                self.gain[direction, time_index, instrument]
                            ),
                            "weighted_P_loss_probability": float(
                                weighted_loss[direction, time_index, instrument]
                            ),
                            "weighted_P_loss_probability_standard_error": float(
                                weighted_loss_error[
                                    direction, time_index, instrument
                                ]
                            ),
                            "weighted_P_loss_probability_upper_95": float(
                                min(
                                    1.0,
                                    weighted_loss[
                                        direction, time_index, instrument
                                    ]
                                    + one_sided_95_z
                                    * weighted_loss_error[
                                        direction, time_index, instrument
                                    ],
                                )
                            ),
                            "weighted_P_gain_probability": float(
                                weighted_gain[direction, time_index, instrument]
                            ),
                            "weighted_P_gain_probability_standard_error": float(
                                weighted_gain_error[
                                    direction, time_index, instrument
                                ]
                            ),
                            "weighted_P_gain_probability_lower_95": float(
                                max(
                                    0.0,
                                    weighted_gain[
                                        direction, time_index, instrument
                                    ]
                                    - one_sided_95_z
                                    * weighted_gain_error[
                                        direction, time_index, instrument
                                    ],
                                )
                            ),
                            "weighted_P_mean": float(value_mean),
                            "weighted_P_path_standard_error": float(error),
                            "weighted_P_mean_lower_95": float(
                                value_mean - one_sided_95_z * error
                            ),
                            "weighted_P_mean_upper_95": float(
                                value_mean + one_sided_95_z * error
                            ),
                            "weighted_P_mean_over_standard_error": (
                                float(value_mean / error)
                                if error > 0.0
                                else math.copysign(math.inf, value_mean)
                                if value_mean != 0.0
                                else 0.0
                            ),
                            "direct_P_loss_count": int(
                                self.direct_loss[direction, time_index, instrument]
                            ),
                            "direct_P_gain_count": int(
                                self.direct_gain[direction, time_index, instrument]
                            ),
                            "direct_P_mean": float(
                                direct_mean[direction, time_index, instrument]
                            ),
                            "direct_P_std_population": float(
                                math.sqrt(
                                    direct_variance[
                                        direction, time_index, instrument
                                    ]
                                )
                            ),
                            "direct_P_path_standard_error": float(
                                direct_standard_error[
                                    direction, time_index, instrument
                                ]
                            ),
                            "direct_P_mean_lower_95": float(
                                direct_mean[direction, time_index, instrument]
                                - one_sided_95_z
                                * direct_standard_error[
                                    direction, time_index, instrument
                                ]
                            ),
                            "direct_P_mean_upper_95": float(
                                direct_mean[direction, time_index, instrument]
                                + one_sided_95_z
                                * direct_standard_error[
                                    direction, time_index, instrument
                                ]
                            ),
                            "direct_P_loss_probability_upper": loss_upper,
                            "direct_P_gain_probability_lower": gain_lower,
                        }
                    )
                directions[name] = items
            reports.append(
                {"trading_time_index": time_index, "directions": directions}
            )
        return {
            "definition": (
                "unit directions separated by trading time; physical estimates "
                "use N^-1 sum_r (p/q)_r f_r"
            ),
            "paths": self.paths,
            "direct_P_paths": self.direct_paths,
            "trading_times": times,
            "instruments": instruments,
            "path_weights": {
                "mean_p_over_q": self.weight_sum / self.paths,
                "effective_sample_size": (
                    self.weight_sum**2 / self.weight_square_sum
                ),
            },
            "times": reports,
        }


class OptionCoverageAccumulator:
    """Stream strike, payoff, and pre-cost sign support by trading time."""

    def __init__(self) -> None:
        self.paths = 0
        self._shape: tuple[int, int, int] | None = None

    def update(
        self,
        universe: Mapping[str, np.ndarray],
        *,
        log_path_weights: np.ndarray | None = None,
    ) -> None:
        """Accumulate one coherent path block for both hedge definitions."""

        dC = np.asarray(universe["dC"], dtype=np.float32)
        S, times, expiries, strikes = dC.shape
        shape = times, expiries, strikes
        if self._shape is None:
            self._shape = shape
            count_shape = (times, expiries * strikes)
            for name in (
                "crossing",
                "noncrossing",
                "raw_positive",
                "raw_negative",
                "delta_positive",
                "delta_negative",
            ):
                setattr(self, name, np.zeros(count_shape, dtype=np.int64))
                setattr(
                    self,
                    f"weighted_{name}",
                    np.zeros(count_shape, dtype=np.float64),
                )
        verify_inp(self._shape == shape, "option coverage geometry changed")
        option_dS = np.asarray(
            universe["expiry_dS" if "expiry_dS" in universe else "option_dS"],
            dtype=np.float32,
        )
        settlement_dS = np.asarray(
            universe.get("option_settlement_dS", option_dS), dtype=np.float32
        )
        spot = np.asarray(universe["spot"], dtype=np.float32)
        strike = np.exp(np.asarray(universe["log_strikes"], dtype=np.float32)) * spot[
            ..., None, None
        ]
        endpoint_spot = spot[..., None] + settlement_dS
        crossing = (endpoint_spot[..., None] > strike).reshape(S, times, -1)
        raw = derive_case_arrays(universe, hedge="raw", cost_rate=0.0)
        delta = derive_case_arrays(universe, hedge="delta", cost_rate=0.0)
        spot_instruments = sum(kind == "spot" for kind in raw.instrument_kind)
        verify_inp(
            spot_instruments == expiries,
            lambda: (
                f"expected one spot instrument per expiry, got "
                f"{spot_instruments}/{expiries}"
            ),
        )
        raw_return = raw.returns[..., spot_instruments:]
        raw_band = raw.numerical_band[..., spot_instruments:]
        delta_return = delta.returns[..., spot_instruments:]
        delta_band = delta.numerical_band[..., spot_instruments:]
        events = {
            "crossing": crossing,
            "noncrossing": ~crossing,
            "raw_positive": raw_return > raw_band,
            "raw_negative": raw_return < -raw_band,
            "delta_positive": delta_return > delta_band,
            "delta_negative": delta_return < -delta_band,
        }
        weights = np.exp(
            np.zeros(S, dtype=np.float64)
            if log_path_weights is None
            else np.asarray(log_path_weights, dtype=np.float64)
        )
        verify_inp(
            weights.shape == (S,) and np.all(np.isfinite(weights)),
            "option coverage weights have invalid geometry",
        )
        for name, event in events.items():
            getattr(self, name)[:] += event.sum(axis=0, dtype=np.int64)
            getattr(self, f"weighted_{name}")[:] += np.sum(
                weights[:, None, None] * event, axis=0
            )
        self.paths += S

    def report(self, *, target_count: int = 1_000) -> dict[str, Any]:
        """Return complete time-resolved coverage counts and probabilities."""

        verify_inp(self.paths > 0 and self._shape is not None, "no coverage accumulated")
        times, expiries, strikes = self._shape
        reports = []
        for time_index in range(times):
            reports.append(
                {
                    "trading_time_index": time_index,
                    **{
                        name: getattr(self, name)[time_index].tolist()
                        for name in (
                            "crossing",
                            "noncrossing",
                            "raw_positive",
                            "raw_negative",
                            "delta_positive",
                            "delta_negative",
                        )
                    },
                    **{
                        f"weighted_P_{name}_probability": (
                            getattr(self, f"weighted_{name}")[time_index]
                            / self.paths
                        ).tolist()
                        for name in (
                            "crossing",
                            "noncrossing",
                            "raw_positive",
                            "raw_negative",
                            "delta_positive",
                            "delta_negative",
                        )
                    },
                }
            )
        minima = {
            name: int(np.min(getattr(self, name)))
            for name in (
                "crossing",
                "noncrossing",
                "raw_positive",
                "raw_negative",
                "delta_positive",
                "delta_negative",
            )
        }
        return {
            "paths": self.paths,
            "trading_times": times,
            "expiries": expiries,
            "strikes": strikes,
            "target_proposal_count_per_side": target_count,
            "minimum_counts": minima,
            "all_strike_and_precost_sign_sides_meet_target": bool(
                all(value >= target_count for value in minima.values())
            ),
            "times": reports,
        }


def _option_endpoint_scale(
    universe: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return unreduced option arithmetic scale and matched spot change."""

    dC = np.asarray(universe["dC"], dtype=np.float32)
    C = np.asarray(universe["C"], dtype=np.float32)
    spot = np.asarray(universe["spot"], dtype=np.float32)
    log_strikes = np.asarray(universe["log_strikes"], dtype=np.float32)
    option_dS_name = "expiry_dS" if "expiry_dS" in universe else "option_dS"
    option_dS = np.asarray(universe[option_dS_name], dtype=np.float32)
    settlement_dS = np.asarray(
        universe.get("option_settlement_dS", option_dS), dtype=np.float32
    )
    # An option and its entry delta hedge close together.  For an incremental
    # interval containing expiry, the matched hedge endpoint is therefore the
    # exact settlement spot rather than the later trading-time spot.
    if "expiry_dS" not in universe:
        option_dS = settlement_dS
    verify_inp(
        dC.shape == C.shape == log_strikes.shape,
        lambda: (
            "option return, premium, and strike shapes differ: "
            f"{dC.shape}/{C.shape}/{log_strikes.shape}"
        ),
    )
    verify_inp(
        option_dS.shape == dC.shape[:3],
        lambda: f"matched option spot return has invalid shape {option_dS.shape}",
    )
    verify_inp(
        settlement_dS.shape == dC.shape[:3],
        lambda: (
            "matched option settlement spot return has invalid shape "
            f"{settlement_dS.shape}"
        ),
    )
    cash_strike = np.exp(log_strikes) * spot[..., None, None]
    endpoint_spot = spot[..., None] + settlement_dS
    expiry_scale = (
        np.abs(endpoint_spot[..., None]) + np.abs(cash_strike) + np.abs(C)
    )
    marked_scale = np.abs(C + dC) + np.abs(C)
    expired = np.asarray(
        universe.get(
            "option_expired",
            np.ones((1, 1, dC.shape[2]), dtype=bool),
        ),
        dtype=bool,
    )
    verify_inp(
        expired.shape == (1, 1, dC.shape[2]),
        lambda: f"option expiry marker has invalid shape {expired.shape}",
    )
    scale = np.where(expired[..., None], expiry_scale, marked_scale)
    return scale, option_dS


def derive_case_arrays(
    universe: Mapping[str, np.ndarray],
    *,
    hedge: Hedge,
    cost_rate: float,
    roundoff_multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER,
) -> DerivedCaseArrays:
    """Derive one case from a shared coherent path block.

    Expiry and incremental cases contain one spot instrument for every option
    expiry. For incremental returns the spot leg closes with its expiry group:
    at exact expiry when it occurs inside the interval and at the next trading
    time otherwise. Option costs are ``q*|C|`` when raw and
    ``q*(|C|+|delta*S|)`` when delta hedged. No option/spot notional is netted.
    """

    verify_inp(hedge in {"raw", "delta"}, lambda: f"unknown hedge {hedge!r}")
    verify_inp(
        cost_rate in REPORTED_COST_RATES,
        lambda: f"unsupported reported cost rate {cost_rate!r}",
    )
    verify_inp(
        math.isfinite(roundoff_multiplier) and roundoff_multiplier > 0.0,
        lambda: f"invalid roundoff multiplier {roundoff_multiplier!r}",
    )
    dC = np.asarray(universe["dC"], dtype=np.float32)
    C = np.asarray(universe["C"], dtype=np.float32)
    delta = np.asarray(universe["delta"], dtype=np.float32)
    spot = np.asarray(universe["spot"], dtype=np.float32)
    verify_inp(
        dC.ndim == 4 and C.shape == delta.shape == dC.shape,
        lambda: f"option arrays must share shape (S,m,T,N), got {dC.shape}",
    )
    S, times, expiries, strikes = dC.shape
    verify_inp(
        spot.shape == (S, times),
        lambda: f"entry spot has invalid shape {spot.shape}",
    )
    option_scale, option_dS = _option_endpoint_scale(universe)
    option_return = dC.copy()
    option_cost_notional = np.abs(C)
    if hedge == "delta":
        option_return -= delta * option_dS[..., None]
        option_cost_notional = option_cost_notional + np.abs(
            delta * spot[..., None, None]
        )
        endpoint_spot = spot[..., None] + option_dS
        option_scale = option_scale + np.abs(delta) * (
            np.abs(endpoint_spot[..., None])
            + np.abs(spot[..., None, None])
        )
    option_cost = np.float32(cost_rate) * option_cost_notional

    horizon: ReturnHorizon = (
        "expiry" if "expiry_dS" in universe else "incremental"
    )
    spot_return = option_dS
    spot_entry = np.broadcast_to(spot[..., None], spot_return.shape)
    spot_kind = ("spot",) * expiries
    spot_expiry = np.arange(expiries, dtype=np.int32)
    spot_endpoint = spot_entry + spot_return
    spot_cost = np.float32(cost_rate) * np.abs(spot_entry)
    spot_scale = np.abs(spot_endpoint) + np.abs(spot_entry)

    option_return = option_return.reshape(S, times, expiries * strikes)
    option_cost = option_cost.reshape(S, times, expiries * strikes)
    option_scale = option_scale.reshape(S, times, expiries * strikes)
    returns = np.concatenate((spot_return, option_return), axis=-1)
    costs = np.concatenate((spot_cost, option_cost), axis=-1)
    unreduced_scale = np.concatenate((spot_scale, option_scale), axis=-1)
    numerical_band = (
        roundoff_multiplier
        * FLOAT32_UNIT_ROUNDOFF
        * np.maximum(
            unreduced_scale + np.abs(costs),
            np.finfo(np.float32).tiny,
        )
    )
    option_expiry = np.repeat(np.arange(expiries, dtype=np.int32), strikes)
    option_strike = np.tile(np.arange(strikes, dtype=np.int32), expiries)
    result = DerivedCaseArrays(
        returns=returns.astype(np.float32, copy=False),
        costs=costs.astype(np.float32, copy=False),
        numerical_band=numerical_band.astype(np.float32, copy=False),
        instrument_kind=spot_kind + ("option",) * (expiries * strikes),
        expiry_index=np.concatenate((spot_expiry, option_expiry)),
        strike_index=np.concatenate(
        (
            np.full(len(spot_kind), -1, dtype=np.int32),
            option_strike,
        )),
        return_horizon=horizon,
        hedge=hedge,
        cost_rate=cost_rate,
    )
    expected = (S, times, len(result.instrument_kind))
    verify_inp(
        result.returns.shape
        == result.costs.shape
        == result.numerical_band.shape
        == expected,
        lambda: f"derived case geometry is inconsistent with {expected}",
    )
    for name, value in (
        ("returns", result.returns),
        ("costs", result.costs),
        ("numerical_band", result.numerical_band),
    ):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"derived case {name} contain NaN or infinity")
    verify_inp(np.all(result.costs >= 0.0), "transaction costs must be nonnegative")
    verify_inp(
        np.all(result.numerical_band > 0.0),
        "numerical bands must be strictly positive",
    )
    return result


def propagated_basket_band(
    action: np.ndarray,
    returns: np.ndarray,
    costs: np.ndarray,
    instrument_band: np.ndarray,
    *,
    roundoff_multiplier: float = DEFAULT_ROUNDOFF_MULTIPLIER,
) -> np.ndarray:
    """Propagate instrument bands to a basket without a cash floor."""

    action = np.asarray(action, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    instrument_band = np.asarray(instrument_band, dtype=np.float64)
    verify_inp(
        action.ndim == 1
        and returns.shape == costs.shape == instrument_band.shape
        and returns.shape[-1] == action.size,
        "basket action and case arrays have incompatible shapes",
    )
    absolute_action = np.abs(action)
    return np.einsum("...j,j->...", instrument_band, absolute_action) + (
        roundoff_multiplier
        * FLOAT32_UNIT_ROUNDOFF
        * (
            np.einsum("...j,j->...", np.abs(returns), absolute_action)
            + np.einsum("...j,j->...", costs, absolute_action)
        )
    )


def float64_replay_report(
    universe: Mapping[str, np.ndarray],
    case: DerivedCaseArrays,
) -> dict[str, Any]:
    """Replay return and gross-cost identities from primitive arrays in float64."""

    dC = np.asarray(universe["dC"], dtype=np.float64)
    C = np.asarray(universe["C"], dtype=np.float64)
    delta = np.asarray(universe["delta"], dtype=np.float64)
    spot = np.asarray(universe["spot"], dtype=np.float64)
    option_dS = np.asarray(
        universe[
            "expiry_dS"
            if "expiry_dS" in universe
            else (
                "option_settlement_dS"
                if "option_settlement_dS" in universe
                else "option_dS"
            )
        ],
        dtype=np.float64,
    )
    option_return = dC - delta * option_dS[..., None] if case.hedge == "delta" else dC
    option_cost_notional = np.abs(C)
    if case.hedge == "delta":
        option_cost_notional += np.abs(delta * spot[..., None, None])
    option_cost = case.cost_rate * option_cost_notional
    spot_return = option_dS
    spot_entry = np.broadcast_to(spot[..., None], spot_return.shape)
    replay_return = np.concatenate(
        (
            spot_return,
            option_return.reshape(*option_return.shape[:2], -1),
        ),
        axis=-1,
    )
    replay_cost = np.concatenate(
        (
            case.cost_rate * np.abs(spot_entry),
            option_cost.reshape(*option_cost.shape[:2], -1),
        ),
        axis=-1,
    )
    return_error = np.abs(replay_return - case.returns.astype(np.float64))
    cost_error = np.abs(replay_cost - case.costs.astype(np.float64))
    band = case.numerical_band.astype(np.float64)
    expiry_error = np.empty(0, dtype=np.float64)
    expiry_band = np.empty(0, dtype=np.float64)
    option_band = band[
        ..., len(case.instrument_kind) - dC.shape[2] * dC.shape[3] :
    ].reshape(dC.shape)
    if case.return_horizon == "expiry":
        cash_strike = (
            np.exp(np.asarray(universe["log_strikes"], dtype=np.float64))
            * spot[..., None, None]
        )
        payoff = np.maximum(
            spot[..., None, None] + option_dS[..., None] - cash_strike,
            0.0,
        )
        expiry_error = np.abs((payoff - C) - dC).reshape(
            *dC.shape[:2], -1
        )
        expiry_band = option_band.reshape(*dC.shape[:2], -1)
    else:
        expired = np.asarray(universe["option_expired"], dtype=bool)
        if np.any(expired):
            settlement_dS = np.asarray(
                universe["option_settlement_dS"], dtype=np.float64
            )
            cash_strike = (
                np.exp(np.asarray(universe["log_strikes"], dtype=np.float64))
                * spot[..., None, None]
            )
            payoff = np.maximum(
                spot[..., None, None]
                + settlement_dS[..., None]
                - cash_strike,
                0.0,
            )
            expired_mask = np.broadcast_to(expired[..., None], dC.shape)
            expiry_error = np.abs((payoff - C) - dC)[expired_mask]
            expiry_band = option_band[expired_mask]
    passed = bool(
        np.all(return_error <= band)
        and np.all(cost_error <= band)
        and (
            expiry_error.size == 0
            or np.all(expiry_error <= expiry_band)
        )
    )
    return {
        "passed": passed,
        "precision": "float64 replay from stored float32 primitive operands",
        "maximum_absolute_return_identity_error": float(np.max(return_error)),
        "maximum_absolute_cost_identity_error": float(np.max(cost_error)),
        "maximum_absolute_expiry_payoff_identity_error": (
            float(np.max(expiry_error)) if expiry_error.size else None
        ),
        "all_errors_within_observation_specific_float32_band": passed,
    }


@dataclass(frozen=True, kw_only=True)
class PathSplit:
    """Disjoint proposal-mixture-stratified reference and audit paths."""

    reference: np.ndarray
    audit: np.ndarray
    component_counts: dict[int, dict[str, int]]
    audit_fraction: float
    seed: int


def stratified_path_split(
    components: np.ndarray,
    *,
    audit_fraction: float = 0.2,
    seed: int,
) -> PathSplit:
    """Split complete paths 80:20 while preserving mixture proportions."""

    components = np.asarray(components)
    verify_inp(
        components.ndim == 1 and components.size >= 2,
        lambda: f"mixture components must contain at least two paths, got {components.shape}",
    )
    verify_inp(
        0.0 < audit_fraction < 1.0,
        lambda: f"audit fraction must lie in (0,1), got {audit_fraction}",
    )
    labels, counts = np.unique(components, return_counts=True)
    target = int(round(audit_fraction * components.size))
    exact = audit_fraction * counts
    audit_counts = np.floor(exact).astype(np.int64)
    remainder = target - int(audit_counts.sum())
    order = np.argsort(-(exact - audit_counts), kind="stable")
    audit_counts[order[:remainder]] += 1
    rng = np.random.default_rng(seed)
    reference_parts: list[np.ndarray] = []
    audit_parts: list[np.ndarray] = []
    report: dict[int, dict[str, int]] = {}
    for label, count, audit_count in zip(labels, counts, audit_counts, strict=True):
        indices = np.flatnonzero(components == label)
        rng.shuffle(indices)
        selected = indices[:audit_count]
        retained = indices[audit_count:]
        audit_parts.append(selected)
        reference_parts.append(retained)
        report[int(label)] = {
            "total": int(count),
            "reference": int(retained.size),
            "audit": int(selected.size),
        }
    reference = np.sort(np.concatenate(reference_parts)).astype(np.int64)
    audit = np.sort(np.concatenate(audit_parts)).astype(np.int64)
    verify_inp(
        reference.size + audit.size == components.size
        and audit.size == target
        and np.intersect1d(reference, audit).size == 0,
        "stratified split is not a disjoint exhaustive partition",
    )
    return PathSplit(
        reference=reference,
        audit=audit,
        component_counts=report,
        audit_fraction=audit_fraction,
        seed=seed,
    )


def clopper_pearson_one_sided(
    successes: int,
    observations: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return exact one-sided lower and upper binomial confidence bounds."""

    verify_inp(
        0 <= successes <= observations and observations > 0,
        lambda: f"invalid binomial counts {successes}/{observations}",
    )
    verify_inp(0.0 < alpha < 1.0, lambda: f"invalid alpha {alpha}")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha, successes, observations - successes + 1))
    )
    upper = (
        1.0
        if successes == observations
        else float(beta.ppf(1.0 - alpha, successes + 1, observations - successes))
    )
    return lower, upper


def _weighted_moment(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Return ``N^-1 sum(w*x)``, its path error, and effective size."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    verify_inp(
        values.shape == weights.shape and values.ndim == 1,
        "weighted moment requires equal one-dimensional arrays",
    )
    mass = float(weights.sum())
    square_mass = float(np.square(weights).sum())
    verify_inp(
        math.isfinite(mass) and mass > 0.0 and square_mass > 0.0,
        "weighted moment has zero or non-finite mass",
    )
    weighted_value = weights * values
    mean = float(np.mean(weighted_value))
    error = float(np.std(weighted_value, ddof=0) / math.sqrt(values.size))
    return {
        "mean": mean,
        "standard_error": error,
        "effective_sample_size": mass * mass / square_mass,
    }


def time_separated_unconditional_report(
    case: DerivedCaseArrays,
    *,
    log_path_weights: np.ndarray | None = None,
    importance_components: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Report unit long/short screens separately at every trading time."""

    S, times, instruments = case.returns.shape
    log_weights = (
        np.zeros(S, dtype=np.float64)
        if log_path_weights is None
        else np.asarray(log_path_weights, dtype=np.float64)
    )
    components = (
        np.zeros(S, dtype=np.int16)
        if importance_components is None
        else np.asarray(importance_components)
    )
    verify_inp(
        log_weights.shape == components.shape == (S,)
        and np.all(np.isfinite(log_weights)),
        "path likelihood/component arrays have invalid geometry",
    )
    weights = np.exp(log_weights)
    verify_inp(np.all(np.isfinite(weights)), "path weights contain NaN or infinity")
    direct = components == 0
    verify_inp(np.any(direct), "no direct-P paths are available for incidence checks")
    time_reports: list[dict[str, Any]] = []
    for time_index in range(times):
        direction_reports: dict[str, list[dict[str, Any]]] = {}
        for direction in ("long", "short"):
            values = case.directional(direction)[:, time_index]
            band = case.numerical_band[:, time_index]
            loss = values < -band
            gain = values > band
            items: list[dict[str, Any]] = []
            for instrument in range(instruments):
                weighted_value = _weighted_moment(values[:, instrument], weights)
                weighted_loss = float(
                    np.dot(weights, loss[:, instrument]) / S
                )
                weighted_gain = float(
                    np.dot(weights, gain[:, instrument]) / S
                )
                direct_loss = int(np.count_nonzero(loss[direct, instrument]))
                direct_gain = int(np.count_nonzero(gain[direct, instrument]))
                direct_count = int(np.count_nonzero(direct))
                _, loss_upper = clopper_pearson_one_sided(
                    direct_loss, direct_count, alpha=alpha
                )
                gain_lower, _ = clopper_pearson_one_sided(
                    direct_gain, direct_count, alpha=alpha
                )
                items.append(
                    {
                        "instrument": instrument,
                        "kind": case.instrument_kind[instrument],
                        "expiry_index": int(case.expiry_index[instrument]),
                        "strike_index": int(case.strike_index[instrument]),
                        "proposal_loss_count": int(np.count_nonzero(loss[:, instrument])),
                        "proposal_neutral_count": int(
                            S
                            - np.count_nonzero(loss[:, instrument])
                            - np.count_nonzero(gain[:, instrument])
                        ),
                        "proposal_gain_count": int(np.count_nonzero(gain[:, instrument])),
                        "weighted_P_loss_probability": weighted_loss,
                        "weighted_P_gain_probability": weighted_gain,
                        "weighted_P_mean": weighted_value["mean"],
                        "weighted_P_path_standard_error": weighted_value[
                            "standard_error"
                        ],
                        "weighted_P_mean_over_standard_error": (
                            weighted_value["mean"] / weighted_value["standard_error"]
                            if weighted_value["standard_error"] > 0.0
                            else math.copysign(math.inf, weighted_value["mean"])
                            if weighted_value["mean"] != 0.0
                            else 0.0
                        ),
                        "direct_P_paths": direct_count,
                        "direct_P_loss_count": direct_loss,
                        "direct_P_gain_count": direct_gain,
                        "direct_P_loss_probability_upper": loss_upper,
                        "direct_P_gain_probability_lower": gain_lower,
                    }
                )
            direction_reports[direction] = items
        time_reports.append(
            {"trading_time_index": time_index, "directions": direction_reports}
        )
    return {
        "definition": (
            "pre-specified unit directions, reported separately by trading time; "
            "transaction cost is subtracted before comparison with eta"
        ),
        "paths": S,
        "trading_times": times,
        "instruments": instruments,
        "direct_P_paths": int(np.count_nonzero(direct)),
        "path_weight_mean": float(weights.mean()),
        "path_weight_effective_sample_size": float(
            np.square(weights.sum()) / np.square(weights).sum()
        ),
        "times": time_reports,
    }


def solve_constant_basket(
    case: DerivedCaseArrays,
    *,
    time_index: int,
    reference_paths: np.ndarray,
    log_path_weights: np.ndarray | None = None,
    feasibility_tolerance: float = 1.0e-10,
    constraints_per_iteration: int = 128,
    maximum_iterations: int = 50,
) -> dict[str, Any]:
    """Solve the constant-basket reference problem by constraint generation."""

    paths = np.asarray(reference_paths, dtype=np.int64)
    verify_inp(
        paths.ndim == 1 and paths.size > 0,
        "constant-basket solver requires nonempty reference paths",
    )
    verify_inp(
        0 <= time_index < case.returns.shape[1],
        lambda: f"invalid trading time {time_index}",
    )
    returns = case.returns[paths, time_index].astype(np.float64, copy=False)
    costs = case.costs[paths, time_index].astype(np.float64, copy=False)
    bands = case.numerical_band[paths, time_index].astype(np.float64, copy=False)
    S, instruments = returns.shape
    weights = (
        np.ones(S, dtype=np.float64)
        if log_path_weights is None
        else positive_relative_weights(
            np.asarray(log_path_weights, dtype=np.float64)[paths]
        )
    )
    weights /= weights.sum()
    mean_return = weights @ returns
    mean_cost = weights @ costs
    zero_cost = not np.any(costs)
    if zero_cost:
        # Represent the signed action directly and bound its absolute value by
        # an auxiliary vector. This is exactly the zero-cost problem and
        # removes the a_plus/a_minus null directions which otherwise make the
        # LP highly degenerate.
        objective = np.concatenate((-mean_return, np.zeros(instruments)))
        l1 = np.concatenate(
            (np.zeros((1, instruments)), np.ones((1, instruments))), axis=1
        )
        identity = np.eye(instruments, dtype=np.float64)
        absolute_envelope = np.concatenate(
            (
                np.concatenate((identity, -identity), axis=1),
                np.concatenate((-identity, -identity), axis=1),
            ),
            axis=0,
        )
        bounds: tuple[float | None, float | None] | list[tuple[float | None, float | None]] = [
            (None, None)
        ] * instruments + [(0.0, None)] * instruments
    else:
        objective = -np.concatenate(
            (mean_return - mean_cost, -mean_return - mean_cost)
        )
        l1 = np.ones((1, 2 * instruments), dtype=np.float64)
        absolute_envelope = np.empty((0, 2 * instruments), dtype=np.float64)
        bounds = (0.0, None)
    selected = np.empty(0, dtype=np.int64)
    result = None
    action = np.zeros(instruments, dtype=np.float64)
    full_constraint_fallback = False
    highs_feasibility_tolerance = max(feasibility_tolerance, 1.0e-10)

    def result_action(optimized: Any) -> np.ndarray:
        return (
            optimized.x[:instruments]
            if zero_cost
            else optimized.x[:instruments] - optimized.x[instruments:]
        )

    def optimize(constraint_paths: np.ndarray) -> Any:
        if constraint_paths.size:
            R = returns[constraint_paths]
            C = costs[constraint_paths]
            # Enforce the economic non-loss condition -X <= 0. The numerical
            # band is used only when classifying and auditing the resulting
            # net action. Including it here can make the effective cost
            # negative and permits simultaneous positive a_plus and a_minus
            # to create an artificial tolerance without a net position.
            A = (
                np.concatenate((-R, np.zeros_like(R)), axis=1)
                if zero_cost
                else np.concatenate((-R + C, R + C), axis=1)
            )
            A_ub = np.concatenate((l1, absolute_envelope, A), axis=0)
            b_ub = np.concatenate(
                (
                    np.ones(1),
                    np.zeros(absolute_envelope.shape[0] + constraint_paths.size),
                )
            )
        else:
            A_ub = np.concatenate((l1, absolute_envelope), axis=0)
            b_ub = np.concatenate(
                (np.ones(1), np.zeros(absolute_envelope.shape[0]))
            )
        optimized = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": highs_feasibility_tolerance,
                "dual_feasibility_tolerance": highs_feasibility_tolerance,
                "ipm_optimality_tolerance": highs_feasibility_tolerance,
            },
        )
        if not optimized.success:
            raise RuntimeError(
                f"constant-basket optimizer failed: {optimized.message}"
            )
        return optimized

    for iteration in range(maximum_iterations):
        result = optimize(selected)
        action = result_action(result)
        pnl = returns @ action - costs @ np.abs(action)
        propagated = propagated_basket_band(action, returns, costs, bands)
        violation = -pnl
        violating = np.flatnonzero(violation > feasibility_tolerance)
        if violating.size == 0:
            break
        worst = violating[
            np.argsort(-violation[violating], kind="stable")[:constraints_per_iteration]
        ]
        expanded = np.unique(np.concatenate((selected, worst)))
        if expanded.size == selected.size:
            full_constraint_fallback = True
            selected = np.arange(S, dtype=np.int64)
            result = optimize(selected)
            action = result_action(result)
            pnl = returns @ action - costs @ np.abs(action)
            propagated = propagated_basket_band(action, returns, costs, bands)
            if np.any(-pnl > feasibility_tolerance):
                raise RuntimeError(
                    "constant-basket full-constraint solution is infeasible at "
                    "the requested numerical tolerance"
                )
            break
        selected = expanded
    else:
        full_constraint_fallback = True
        selected = np.arange(S, dtype=np.int64)
        result = optimize(selected)
        action = result_action(result)
        pnl = returns @ action - costs @ np.abs(action)
        propagated = propagated_basket_band(action, returns, costs, bands)
        if np.any(-pnl > feasibility_tolerance):
            raise RuntimeError(
                "constant-basket full-constraint solution is infeasible at the "
                "requested numerical tolerance"
            )
        iteration = maximum_iterations
    assert result is not None
    pnl = returns @ action - costs @ np.abs(action)
    propagated = propagated_basket_band(action, returns, costs, bands)
    return {
        "time_index": time_index,
        "action": action,
        "l1_norm": float(np.abs(action).sum()),
        "objective_mean": float(np.dot(weights, pnl)),
        "minimum_economic_pnl": float(np.min(pnl)),
        "maximum_economic_pnl": float(np.max(pnl)),
        "minimum_pnl_minus_negative_band": float(np.min(pnl + propagated)),
        "gain_count": int(np.count_nonzero(pnl > propagated)),
        "gain_fraction": float(np.mean(pnl > propagated)),
        "binding_path_count": int(
            np.count_nonzero(np.abs(pnl + propagated) <= feasibility_tolerance)
        ),
        "constraints_retained": int(selected.size),
        "iterations": iteration + 1,
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "feasibility_tolerance": feasibility_tolerance,
        "reference_constraint": "economic P&L X >= 0",
        "highs_feasibility_tolerance": highs_feasibility_tolerance,
        "full_constraint_fallback": full_constraint_fallback,
        "zero_cost_signed_action_formulation": (
            "free-signed-action-v2" if zero_cost else False
        ),
    }


def certify_unconditional_directions(
    case: DerivedCaseArrays,
    *,
    split: PathSplit,
    log_path_weights: np.ndarray,
    importance_components: np.ndarray,
    family_size: int = 131_472,
    alpha: float = 0.05,
    bootstrap_repetitions: int = 500,
    seed: int = 1,
    prerequisite_checks_passed: bool,
) -> dict[str, Any]:
    """Screen on reference paths and certify frozen unit directions on audit.

    The state region is the complete state space, so this is the independent
    audit for unconditional individual directions. Conditional regions and
    fitted baskets require their own frozen-region audit.
    """

    verify_inp(family_size > 0, "candidate family size must be positive")
    verify_inp(
        bootstrap_repetitions > 0,
        "candidate bootstrap repetitions must be positive",
    )
    S, times, instruments = case.returns.shape
    log_weights = np.asarray(log_path_weights, dtype=np.float64)
    components = np.asarray(importance_components)
    verify_inp(
        log_weights.shape == components.shape == (S,),
        "candidate likelihood arrays have invalid geometry",
    )
    reference_values = np.stack(
        (
            case.returns[split.reference] - case.costs[split.reference],
            -case.returns[split.reference] - case.costs[split.reference],
        ),
        axis=0,
    )
    reference_band = case.numerical_band[split.reference][None]
    reference_loss = np.sum(reference_values < -reference_band, axis=1)
    reference_gain = np.sum(reference_values > reference_band, axis=1)
    screened = np.argwhere((reference_loss == 0) & (reference_gain > 0))
    adjusted_alpha = alpha / family_size
    z_adjusted = float(norm.ppf(1.0 - adjusted_alpha))
    audit_weights = positive_relative_weights(log_weights[split.audit])
    audit_direct = components[split.audit] == 0
    direct_count = int(np.count_nonzero(audit_direct))
    verify_inp(direct_count > 1, "candidate audit requires two direct-P paths")
    rng = np.random.default_rng(seed)
    rejected: list[dict[str, Any]] = []
    surviving: list[dict[str, Any]] = []
    for direction, time_index, instrument in screened:
        sign = 1.0 if direction == 0 else -1.0
        audit_value = (
            sign * case.returns[split.audit, time_index, instrument]
            - case.costs[split.audit, time_index, instrument]
        ).astype(np.float64, copy=False)
        audit_band = case.numerical_band[
            split.audit, time_index, instrument
        ].astype(np.float64, copy=False)
        loss = audit_value < -audit_band
        gain = audit_value > audit_band
        direct_value = audit_value[audit_direct]
        direct_loss = int(np.count_nonzero(loss[audit_direct]))
        direct_gain = int(np.count_nonzero(gain[audit_direct]))
        direct_mean = float(np.mean(direct_value))
        direct_error = float(np.std(direct_value, ddof=0) / math.sqrt(direct_count))
        direct_mean_lower = direct_mean - z_adjusted * direct_error
        direct_gain_lower, _ = clopper_pearson_one_sided(
            direct_gain, direct_count, alpha=adjusted_alpha
        )
        _, direct_loss_upper = clopper_pearson_one_sided(
            direct_loss, direct_count, alpha=adjusted_alpha
        )

        weighted_mass = float(audit_weights.sum())
        weighted_mean = float(np.dot(audit_weights, audit_value) / weighted_mass)
        weighted_gain = float(np.dot(audit_weights, gain) / weighted_mass)
        weighted_loss = float(np.dot(audit_weights, loss) / weighted_mass)
        weighted_mean_error = float(
            np.sqrt(
                np.dot(
                    np.square(audit_weights),
                    np.square(audit_value - weighted_mean),
                )
            )
            / weighted_mass
        )
        weighted_gain_error = float(
            np.sqrt(
                np.dot(
                    np.square(audit_weights),
                    np.square(gain.astype(np.float64) - weighted_gain),
                )
            )
            / weighted_mass
        )
        weighted_mean_lower = weighted_mean - z_adjusted * weighted_mean_error
        weighted_gain_lower = weighted_gain - z_adjusted * weighted_gain_error

        bootstrap_mean = np.empty(bootstrap_repetitions, dtype=np.float64)
        bootstrap_gain = np.empty(bootstrap_repetitions, dtype=np.float64)
        for repetition in range(bootstrap_repetitions):
            multiplicity = rng.poisson(1.0, size=split.audit.size)
            bootstrap_weight = audit_weights * multiplicity
            mass = float(bootstrap_weight.sum())
            if mass <= 0.0:
                bootstrap_mean[repetition] = weighted_mean
                bootstrap_gain[repetition] = weighted_gain
            else:
                bootstrap_mean[repetition] = float(
                    np.dot(bootstrap_weight, audit_value) / mass
                )
                bootstrap_gain[repetition] = float(
                    np.dot(bootstrap_weight, gain) / mass
                )
        pointwise_bootstrap_mean_lower = float(
            np.quantile(bootstrap_mean, alpha, method="linear")
        )
        pointwise_bootstrap_gain_lower = float(
            np.quantile(bootstrap_gain, alpha, method="linear")
        )
        conditions = {
            "no_audit_loss_beyond_band": int(np.count_nonzero(loss)) == 0,
            "adjusted_direct_P_gain_lower_positive": direct_gain_lower > 0.0,
            "adjusted_direct_P_mean_lower_positive": direct_mean_lower > 0.0,
            "adjusted_weighted_P_gain_lower_positive": weighted_gain_lower > 0.0,
            "adjusted_weighted_P_mean_lower_positive": weighted_mean_lower > 0.0,
            "direct_and_weighted_conclusions_agree": (
                direct_gain_lower > 0.0
                and direct_mean_lower > 0.0
                and weighted_gain_lower > 0.0
                and weighted_mean_lower > 0.0
            ),
            "importance_coverage_and_replay_pass": prerequisite_checks_passed,
        }
        item = {
            "trading_time_index": int(time_index),
            "instrument": int(instrument),
            "kind": case.instrument_kind[instrument],
            "expiry_index": int(case.expiry_index[instrument]),
            "strike_index": int(case.strike_index[instrument]),
            "direction": "long" if direction == 0 else "short",
            "state_region": "complete state space",
            "strategy_class": "unconditional unit instrument direction",
            "reference_loss_count": int(reference_loss[direction, time_index, instrument]),
            "reference_gain_count": int(reference_gain[direction, time_index, instrument]),
            "audit_paths": int(split.audit.size),
            "audit_loss_count": int(np.count_nonzero(loss)),
            "audit_gain_count": int(np.count_nonzero(gain)),
            "direct_P_paths": direct_count,
            "direct_P_loss_count": direct_loss,
            "direct_P_gain_count": direct_gain,
            "adjusted_direct_P_loss_probability_upper": direct_loss_upper,
            "adjusted_direct_P_gain_probability_lower": direct_gain_lower,
            "direct_P_mean": direct_mean,
            "adjusted_direct_P_mean_lower": direct_mean_lower,
            "weighted_P_loss_probability": weighted_loss,
            "weighted_P_gain_probability": weighted_gain,
            "adjusted_weighted_P_gain_probability_lower": weighted_gain_lower,
            "weighted_P_mean": weighted_mean,
            "adjusted_weighted_P_mean_lower": weighted_mean_lower,
            "pointwise_bootstrap_weighted_P_gain_lower": (
                pointwise_bootstrap_gain_lower
            ),
            "pointwise_bootstrap_weighted_P_mean_lower": (
                pointwise_bootstrap_mean_lower
            ),
            "regional_effective_sample_size": float(
                weighted_mass**2 / np.square(audit_weights).sum()
            ),
            "conditions": conditions,
            "certified": bool(all(conditions.values())),
        }
        (surviving if item["certified"] else rejected).append(item)
    return {
        "strategy_class": "unconditional unit instrument directions",
        "state_region": "complete state space",
        "reference_screened_candidates": int(screened.shape[0]),
        "multiplicity": {
            "method": "Bonferroni familywise correction",
            "family_size": family_size,
            "pointwise_alpha": alpha,
            "adjusted_alpha": adjusted_alpha,
        },
        "weighted_interval": {
            "adjusted": "normal path-level influence interval",
            "pointwise": (
                f"Poisson complete-path bootstrap, {bootstrap_repetitions} repetitions"
            ),
        },
        "surviving": surviving,
        "rejected": rejected,
    }


def audit_constant_basket(
    solution: Mapping[str, Any],
    case: DerivedCaseArrays,
    *,
    split: PathSplit,
    log_path_weights: np.ndarray,
    importance_components: np.ndarray,
    family_size: int = 480,
    alpha: float = 0.05,
    prerequisite_checks_passed: bool,
) -> dict[str, Any]:
    """Evaluate one frozen reference-fitted constant basket on audit paths."""

    time_index = int(solution["time_index"])
    action = np.asarray(solution["action"], dtype=np.float64)
    verify_inp(
        action.shape == (case.returns.shape[-1],),
        "constant basket action has invalid geometry",
    )
    if np.abs(action).sum() <= 1.0e-14 or solution["gain_count"] == 0:
        return {
            **dict(solution),
            "action": action,
            "audit_status": "no nonzero reference candidate",
            "certified": False,
        }
    returns = case.returns[split.audit, time_index].astype(np.float64, copy=False)
    costs = case.costs[split.audit, time_index].astype(np.float64, copy=False)
    bands = case.numerical_band[split.audit, time_index].astype(
        np.float64, copy=False
    )
    pnl = returns @ action - costs @ np.abs(action)
    band = propagated_basket_band(action, returns, costs, bands)
    loss = pnl < -band
    gain = pnl > band
    components = np.asarray(importance_components)[split.audit]
    direct = components == 0
    direct_count = int(np.count_nonzero(direct))
    verify_inp(direct_count > 1, "constant basket audit lacks direct-P paths")
    direct_loss = int(np.count_nonzero(loss[direct]))
    direct_gain = int(np.count_nonzero(gain[direct]))
    adjusted_alpha = alpha / family_size
    gain_lower, _ = clopper_pearson_one_sided(
        direct_gain, direct_count, alpha=adjusted_alpha
    )
    _, loss_upper = clopper_pearson_one_sided(
        direct_loss, direct_count, alpha=adjusted_alpha
    )
    direct_mean = float(np.mean(pnl[direct]))
    direct_error = float(np.std(pnl[direct], ddof=0) / math.sqrt(direct_count))
    z = float(norm.ppf(1.0 - adjusted_alpha))
    direct_mean_lower = direct_mean - z * direct_error
    weights = positive_relative_weights(
        np.asarray(log_path_weights, dtype=np.float64)[split.audit]
    )
    mass = float(weights.sum())
    weighted_mean = float(np.dot(weights, pnl) / mass)
    weighted_error = float(
        np.sqrt(np.dot(np.square(weights), np.square(pnl - weighted_mean))) / mass
    )
    weighted_mean_lower = weighted_mean - z * weighted_error
    conditions = {
        "no_audit_loss_beyond_band": int(np.count_nonzero(loss)) == 0,
        "adjusted_direct_P_gain_lower_positive": gain_lower > 0.0,
        "adjusted_direct_P_mean_lower_positive": direct_mean_lower > 0.0,
        "adjusted_weighted_P_mean_lower_positive": weighted_mean_lower > 0.0,
        "importance_coverage_and_replay_pass": prerequisite_checks_passed,
    }
    return {
        **dict(solution),
        "action": action,
        "audit_status": "evaluated frozen reference basket",
        "audit_paths": int(split.audit.size),
        "audit_loss_count": int(np.count_nonzero(loss)),
        "audit_gain_count": int(np.count_nonzero(gain)),
        "direct_P_paths": direct_count,
        "direct_P_loss_count": direct_loss,
        "direct_P_gain_count": direct_gain,
        "adjusted_direct_P_loss_probability_upper": loss_upper,
        "adjusted_direct_P_gain_probability_lower": gain_lower,
        "direct_P_mean": direct_mean,
        "adjusted_direct_P_mean_lower": direct_mean_lower,
        "weighted_P_mean": weighted_mean,
        "adjusted_weighted_P_mean_lower": weighted_mean_lower,
        "regional_effective_sample_size": float(mass**2 / np.square(weights).sum()),
        "multiplicity": {
            "method": "Bonferroni familywise correction",
            "family_size": family_size,
            "adjusted_alpha": adjusted_alpha,
        },
        "conditions": conditions,
        "certified": bool(all(conditions.values())),
    }


def convergence_report(
    small: Mapping[str, float],
    large: Mapping[str, float],
) -> dict[str, Any]:
    """Check the required importance and path-error convergence invariants."""

    required = {
        "paths",
        "weight_mean",
        "weight_mean_standard_error",
        "effective_sample_size",
        "mean_standard_error",
        "conditional_radius",
        "conditional_effective_sample_size",
    }
    missing = required - small.keys() | required - large.keys()
    verify_inp(not missing, lambda: f"convergence inputs miss {sorted(missing)}")
    path_ratio = float(large["paths"]) / float(small["paths"])
    expected_error_ratio = math.sqrt(path_ratio)
    checks = {
        "path_count_increased": path_ratio > 1.0,
        "weight_mean_consistent_with_one_small": abs(small["weight_mean"] - 1.0)
        <= 3.0 * small["weight_mean_standard_error"],
        "weight_mean_consistent_with_one_large": abs(large["weight_mean"] - 1.0)
        <= 3.0 * large["weight_mean_standard_error"],
        "effective_sample_size_increased": large["effective_sample_size"]
        > small["effective_sample_size"],
        "effective_sample_fraction_not_collapsed": (
            large["effective_sample_size"] / large["paths"]
            >= 0.8 * small["effective_sample_size"] / small["paths"]
        ),
        "mean_standard_error_decreased": large["mean_standard_error"]
        < small["mean_standard_error"],
        "conditional_radius_shrank": large["conditional_radius"]
        < small["conditional_radius"],
        "conditional_effective_sample_size_increased": (
            large["conditional_effective_sample_size"]
            > small["conditional_effective_sample_size"]
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "path_count_ratio": path_ratio,
        "expected_standard_error_reduction": expected_error_ratio,
        "observed_mean_standard_error_reduction": (
            small["mean_standard_error"] / large["mean_standard_error"]
        ),
    }


_FUNCTIONAL_GATE_CHECKS = (
    "finite_arrays",
    "timing_identities",
    "float64_accounting_replay",
    "decoder_identity",
    "strike_and_sign_boundary_coverage",
    "importance_normalization_and_moment_recovery",
    "stratified_disjoint_split",
    "conditional_radius_and_ess",
    "extremes_replayed",
    "memory_within_limits",
)


def functional_gate_report(checks: Mapping[str, bool | Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the mandatory 100,000-to-one-million functional gate.

    A missing check fails closed. A detailed check mapping must contain a
    boolean ``passed`` field, allowing reports to retain the underlying
    measurements without weakening the gate.
    """

    normalized: dict[str, dict[str, Any]] = {}
    for name in _FUNCTIONAL_GATE_CHECKS:
        value = checks.get(name, False)
        if isinstance(value, Mapping):
            detail = dict(value)
            passed = detail.get("passed") is True
        else:
            passed = value is True
            detail = {}
        normalized[name] = {"passed": passed, **detail}
    blockers = [name for name, value in normalized.items() if not value["passed"]]
    return {
        "passed": not blockers,
        "checks": normalized,
        "blockers": blockers,
        "action": (
            "one-million stage authorized"
            if not blockers
            else "one-million stage blocked"
        ),
    }


def case_configuration(case: RequiredCase) -> dict[str, Any]:
    """Return a JSON-safe complete case configuration."""

    return asdict(case)


__all__ = [
    "DerivedCaseArrays",
    "GPU_ASSIGNMENT",
    "OptionCoverageAccumulator",
    "PathSplit",
    "REPORTED_COST_RATES",
    "REPORTED_PATH_COUNTS",
    "RequiredCase",
    "TimeSeparatedUnconditionalAccumulator",
    "case_configuration",
    "audit_constant_basket",
    "certify_unconditional_directions",
    "clopper_pearson_one_sided",
    "convergence_report",
    "derive_case_arrays",
    "functional_gate_report",
    "float64_replay_report",
    "propagated_basket_band",
    "positive_relative_weights",
    "required_cases",
    "solve_constant_basket",
    "stratified_path_split",
    "time_separated_unconditional_report",
]
