"""Common multistep weekly scenario accounting and risk functionals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyClaimDefinition,
    WeeklyEpisodePath,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class WeeklyRiskObjective(StrEnum):
    """Risk objectives shared by empirical and generated weekly policies."""

    MEAN_VARIANCE = "mean-variance"
    MEAN_VOLATILITY = "mean-volatility"
    ENTROPY = "entropy"
    POWER_EXPONENTIAL = "power-exponential"


@dataclass(frozen=True, kw_only=True)
class WeeklyScenarioBatch:
    """Complete remaining-horizon paths in episode-origin present-value units."""

    inception_forward_paths: FloatArray
    spot_values_pv: FloatArray
    spot_step_gains: FloatArray
    option_step_gains: FloatArray
    option_half_spreads_pv: FloatArray
    option_alive: BoolArray
    option_settles_next: BoolArray
    terminal_claim_values_pv: FloatArray
    contract_ids: tuple[str, ...]
    target_label: str
    fit_as_of_date: date
    model_fit_identity: str
    measure: str
    seed: int | None

    def __post_init__(self) -> None:
        scenario_count, path_length = self.inception_forward_paths.shape
        step_count = path_length - 1
        contract_count = len(self.contract_ids)
        if scenario_count == 0 or step_count <= 0 or contract_count == 0:
            raise ValueError("weekly scenarios require paths, steps, and contracts")
        expected_shapes = (
            (self.spot_values_pv, (scenario_count, step_count + 1)),
            (self.spot_step_gains, (scenario_count, step_count)),
            (
                self.option_step_gains,
                (scenario_count, step_count, contract_count),
            ),
            (
                self.option_half_spreads_pv,
                (scenario_count, step_count + 1, contract_count),
            ),
            (self.option_alive, (step_count + 1, contract_count)),
            (self.option_settles_next, (step_count, contract_count)),
            (self.terminal_claim_values_pv, (scenario_count,)),
        )
        if any(values.shape != shape for values, shape in expected_shapes):
            raise ValueError("weekly scenario arrays do not align")
        float_arrays = (
            self.inception_forward_paths,
            self.spot_values_pv,
            self.spot_step_gains,
            self.option_step_gains,
            self.option_half_spreads_pv,
            self.terminal_claim_values_pv,
        )
        if any(not np.all(np.isfinite(values)) for values in float_arrays):
            raise ValueError("weekly scenario arrays must be finite")
        if (
            np.any(self.inception_forward_paths <= 0.0)
            or np.any(self.spot_values_pv <= 0.0)
            or np.any(self.option_half_spreads_pv < 0.0)
            or np.any(self.terminal_claim_values_pv < 0.0)
        ):
            raise ValueError("weekly scenarios contain invalid levels or prices")
        if len(set(self.contract_ids)) != contract_count:
            raise ValueError("weekly scenario contract identities must be unique")
        if not self.target_label.strip() or not self.model_fit_identity.strip():
            raise ValueError("weekly scenario identities must be non-empty")
        if not self.measure.strip():
            raise ValueError("weekly scenario measure must be non-empty")
        if np.any(self.option_settles_next & ~self.option_alive[:-1]):
            raise ValueError("only live options can settle on the next step")
        if np.any(self.option_alive[1:] & ~self.option_alive[:-1]):
            raise ValueError("expired weekly options cannot become live again")

    @property
    def scenario_count(self) -> int:
        return len(self.inception_forward_paths)

    @property
    def step_count(self) -> int:
        return self.inception_forward_paths.shape[1] - 1

    @property
    def contract_count(self) -> int:
        return len(self.contract_ids)


@dataclass(frozen=True, kw_only=True)
class WeeklyScenarioAccounting:
    """Pathwise claim, gains, turnover costs, and terminal wealth."""

    signed_claim_values: FloatArray
    hedge_gains: FloatArray
    transaction_costs: FloatArray
    terminal_wealth: FloatArray
    step_hedge_gains: FloatArray
    step_transaction_costs: FloatArray

    def __post_init__(self) -> None:
        scenario_count = len(self.terminal_wealth)
        if any(
            values.shape != (scenario_count,)
            for values in (
                self.signed_claim_values,
                self.hedge_gains,
                self.transaction_costs,
            )
        ):
            raise ValueError("weekly accounting vectors do not align")
        if self.step_hedge_gains.shape[0] != scenario_count:
            raise ValueError("weekly step gains do not align")
        if self.step_transaction_costs.shape != (
            scenario_count,
            self.step_hedge_gains.shape[1] + 1,
        ):
            raise ValueError("weekly step costs must include terminal liquidation")
        arrays = (
            self.signed_claim_values,
            self.hedge_gains,
            self.transaction_costs,
            self.terminal_wealth,
            self.step_hedge_gains,
            self.step_transaction_costs,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("weekly accounting must be finite")
        if np.any(self.transaction_costs < 0.0) or np.any(
            self.step_transaction_costs < 0.0
        ):
            raise ValueError("weekly transaction costs must be non-negative")


@dataclass(frozen=True, kw_only=True)
class WeeklyIndifferencePrices:
    """Buyer and seller prices from separately optimized signed policies."""

    buyer: float
    seller: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.buyer) or not math.isfinite(self.seller):
            raise ValueError("weekly indifference prices must be finite")


def weekly_scenario_batch_from_episode(
    episode: WeeklyEpisodePath,
    *,
    claim: WeeklyClaimDefinition,
    model_fit_identity: str = "deterministic-realized-episode",
    measure: str = "realized",
) -> WeeklyScenarioBatch:
    """Represent one realized episode in the common pathwise tensor format."""
    return WeeklyScenarioBatch(
        inception_forward_paths=(
            episode.inception_forward_normalized_spot_path[None, :].copy()
        ),
        spot_values_pv=(
            episode.discount_to_origin * episode.spot_over_episode_origin
        )[None, :],
        spot_step_gains=episode.spot_step_gains[None, :].copy(),
        option_step_gains=episode.option_step_gains[None, :, :].copy(),
        option_half_spreads_pv=episode.option_half_spreads_pv[None, :, :].copy(),
        option_alive=episode.option_alive.copy(),
        option_settles_next=episode.option_settles_next.copy(),
        terminal_claim_values_pv=np.asarray(
            [episode.discount_to_origin[-1] * claim.payoff(episode)],
            dtype=np.float64,
        ),
        contract_ids=tuple(contract.contract_id for contract in episode.contracts),
        target_label=claim.target.label,
        fit_as_of_date=episode.spec.origin_date,
        model_fit_identity=model_fit_identity,
        measure=measure,
        seed=None,
    )


def attach_weekly_future_suffixes(
    realized_inception_forward_prefix: NDArray[np.floating],
    future_suffixes: NDArray[np.floating],
) -> FloatArray:
    """Attach simulated future closes to one immutable realized target prefix."""
    prefix = np.asarray(realized_inception_forward_prefix, dtype=np.float64)
    suffixes = np.asarray(future_suffixes, dtype=np.float64)
    if prefix.ndim != 1 or len(prefix) == 0 or not np.all(np.isfinite(prefix)):
        raise ValueError("weekly realized prefix must be a non-empty finite vector")
    if suffixes.ndim != 2 or len(suffixes) == 0 or not np.all(np.isfinite(suffixes)):
        raise ValueError("weekly future suffixes must be a non-empty finite matrix")
    if np.any(prefix <= 0.0) or np.any(suffixes <= 0.0):
        raise ValueError("weekly target paths must remain positive")
    return np.concatenate(
        (np.broadcast_to(prefix, (len(suffixes), len(prefix))), suffixes),
        axis=1,
    )


def weekly_target_values_from_suffixes(
    claim: WeeklyClaimDefinition,
    *,
    realized_path_prefix: NDArray[np.floating],
    future_path_suffixes: NDArray[np.floating],
    terminal_discount_to_episode_origin: float,
) -> FloatArray:
    """Evaluate the fixed inception claim after attaching realized history."""
    if (
        not math.isfinite(terminal_discount_to_episode_origin)
        or terminal_discount_to_episode_origin <= 0.0
    ):
        raise ValueError("weekly terminal discount must be finite-positive")
    paths = attach_weekly_future_suffixes(
        realized_path_prefix,
        future_path_suffixes,
    )
    return np.asarray(
        [
            terminal_discount_to_episode_origin * claim.claim.payoff(path)
            for path in paths
        ],
        dtype=np.float64,
    )


def weekly_scenario_accounting(
    scenarios: WeeklyScenarioBatch,
    *,
    spot_positions: NDArray[np.floating],
    option_positions: NDArray[np.floating],
    signed_claim_quantity: float,
    spot_cost_bps: float = 0.0,
    initial_spot_position: float = 0.0,
    initial_option_positions: NDArray[np.floating] | None = None,
) -> WeeklyScenarioAccounting:
    """Apply predictable positions with turnover costs and no exit charge."""
    if not math.isfinite(signed_claim_quantity):
        raise ValueError("signed claim quantity must be finite")
    if not math.isfinite(spot_cost_bps) or spot_cost_bps < 0.0:
        raise ValueError("spot cost must be finite and non-negative")
    if not math.isfinite(initial_spot_position):
        raise ValueError("initial spot position must be finite")
    spot = _broadcast_positions(
        spot_positions,
        scenario_count=scenarios.scenario_count,
        expected_tail=(scenarios.step_count,),
        name="spot_positions",
    )
    options = _broadcast_positions(
        option_positions,
        scenario_count=scenarios.scenario_count,
        expected_tail=(scenarios.step_count, scenarios.contract_count),
        name="option_positions",
    )
    dead = np.broadcast_to(
        ~scenarios.option_alive[:-1],
        (scenarios.scenario_count, scenarios.step_count, scenarios.contract_count),
    )
    if np.any(np.abs(options[dead]) > 1.0e-14):
        raise ValueError("weekly scenarios cannot trade expired contracts")

    step_gains = np.empty(
        (scenarios.scenario_count, scenarios.step_count),
        dtype=np.float64,
    )
    step_costs = np.empty(
        (scenarios.scenario_count, scenarios.step_count + 1),
        dtype=np.float64,
    )
    previous_spot = np.full(
        scenarios.scenario_count,
        initial_spot_position,
        dtype=np.float64,
    )
    if initial_option_positions is None:
        initial_options = np.zeros(scenarios.contract_count, dtype=np.float64)
    else:
        initial_options = np.asarray(initial_option_positions, dtype=np.float64)
        if initial_options.shape != (scenarios.contract_count,) or not np.all(
            np.isfinite(initial_options)
        ):
            raise ValueError("initial option positions must align and be finite")
        if np.any(np.abs(initial_options[~scenarios.option_alive[0]]) > 1.0e-14):
            raise ValueError("initial option positions cannot retain expired contracts")
    previous_options = np.broadcast_to(
        initial_options,
        (scenarios.scenario_count, scenarios.contract_count),
    ).copy()
    for step in range(scenarios.step_count):
        previous_options = np.where(
            scenarios.option_alive[step][None, :],
            previous_options,
            0.0,
        )
        spot_change = spot[:, step] - previous_spot
        option_change = options[:, step] - previous_options
        step_costs[:, step] = (
            np.abs(spot_change)
            * scenarios.spot_values_pv[:, step]
            * spot_cost_bps
            * 1.0e-4
            + np.sum(
                np.abs(option_change)
                * scenarios.option_half_spreads_pv[:, step],
                axis=1,
            )
        )
        step_gains[:, step] = (
            spot[:, step] * scenarios.spot_step_gains[:, step]
            + np.sum(
                options[:, step] * scenarios.option_step_gains[:, step],
                axis=1,
            )
        )
        previous_spot = spot[:, step]
        previous_options = np.where(
            scenarios.option_settles_next[step][None, :],
            0.0,
            options[:, step],
        )
    step_costs[:, -1] = 0.0
    hedge_gains = np.sum(step_gains, axis=1)
    transaction_costs = np.sum(step_costs, axis=1)
    signed_claim_values = (
        signed_claim_quantity * scenarios.terminal_claim_values_pv
    )
    terminal_wealth = signed_claim_values + hedge_gains - transaction_costs
    return WeeklyScenarioAccounting(
        signed_claim_values=signed_claim_values,
        hedge_gains=hedge_gains,
        transaction_costs=transaction_costs,
        terminal_wealth=terminal_wealth,
        step_hedge_gains=step_gains,
        step_transaction_costs=step_costs,
    )


def weekly_certainty_equivalent(
    wealth: NDArray[np.floating],
    *,
    objective: WeeklyRiskObjective,
    risk_parameter: float,
) -> float:
    """Return the translation-equivariant value for one terminal wealth sample."""
    values = np.asarray(wealth, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("weekly wealth must be a non-empty finite vector")
    if not math.isfinite(risk_parameter) or risk_parameter <= 0.0:
        raise ValueError("weekly risk parameter must be finite-positive")
    mean = float(np.mean(values))
    centered = values - mean
    if objective is WeeklyRiskObjective.MEAN_VARIANCE:
        return mean - 0.5 * risk_parameter * float(np.mean(centered**2))
    if objective is WeeklyRiskObjective.MEAN_VOLATILITY:
        return mean - risk_parameter * float(np.sqrt(np.mean(centered**2)))
    if objective is WeeklyRiskObjective.ENTROPY:
        return float(
            -(logsumexp(-risk_parameter * values) - math.log(len(values)))
            / risk_parameter
        )
    if objective is WeeklyRiskObjective.POWER_EXPONENTIAL:
        translation = _power_exponential_translation(
            values,
            risk_parameter=risk_parameter,
        )
        residual = values - translation
        utility = np.where(
            residual >= 0.0,
            -np.expm1(-risk_parameter * np.maximum(residual, 0.0))
            / risk_parameter,
            residual - 0.5 * risk_parameter * residual**2,
        )
        return float(translation + np.mean(utility))
    raise ValueError(f"unsupported weekly objective {objective!r}")


def weekly_indifference_prices(
    *,
    long_claim_wealth: NDArray[np.floating],
    short_claim_wealth: NDArray[np.floating],
    zero_claim_wealth: NDArray[np.floating],
    objective: WeeklyRiskObjective,
    risk_parameter: float,
) -> WeeklyIndifferencePrices:
    """Extract buyer and seller prices from separately optimized policies."""
    zero = weekly_certainty_equivalent(
        zero_claim_wealth,
        objective=objective,
        risk_parameter=risk_parameter,
    )
    long = weekly_certainty_equivalent(
        long_claim_wealth,
        objective=objective,
        risk_parameter=risk_parameter,
    )
    short = weekly_certainty_equivalent(
        short_claim_wealth,
        objective=objective,
        risk_parameter=risk_parameter,
    )
    return WeeklyIndifferencePrices(
        buyer=long - zero,
        seller=zero - short,
    )


def _broadcast_positions(
    values: NDArray[np.floating],
    *,
    scenario_count: int,
    expected_tail: tuple[int, ...],
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == expected_tail:
        array = np.broadcast_to(array, (scenario_count, *expected_tail))
    if array.shape != (scenario_count, *expected_tail) or not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} must have shape {expected_tail} or "
            f"{(scenario_count, *expected_tail)}"
        )
    return np.asarray(array, dtype=np.float64)


def _power_exponential_translation(
    wealth: FloatArray,
    *,
    risk_parameter: float,
) -> float:
    def first_order_condition(translation: float) -> float:
        residual = wealth - translation
        marginal = np.where(
            residual >= 0.0,
            np.exp(-risk_parameter * np.maximum(residual, 0.0)),
            1.0 - risk_parameter * residual,
        )
        return float(np.mean(marginal) - 1.0)

    scale = max(float(np.std(wealth)), 1.0 / risk_parameter, 1.0e-8)
    lower = float(np.min(wealth) - 8.0 * scale)
    upper = float(np.max(wealth) + 8.0 * scale)
    while first_order_condition(lower) >= 0.0:
        lower -= 2.0 * scale
    while first_order_condition(upper) <= 0.0:
        upper += 2.0 * scale
    return float(brentq(first_order_condition, lower, upper, xtol=1.0e-12))
