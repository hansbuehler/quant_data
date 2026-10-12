"""Model-neutral enforcement of frozen and restarted weekly policy fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quantdata.hedging.contracts import OptionContract
from quantdata.sdlv.daily.front_spot.hedging.benchmark.weekly import (
    WeeklyClaimDefinition,
    WeeklyEpisodePath,
    WeeklyFitInstruction,
    WeeklyHedgeResult,
    WeeklyMarketPanel,
    evaluate_weekly_hedge,
    weekly_fit_instructions,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_view import (
    PublicHedgingPanelView,
)

FloatArray = NDArray[np.float64]


class AsOfWeeklyMarketPanel(PublicHedgingPanelView):
    """Read-only facade restricted to one precommitted rolling fit window."""

    def __init__(
        self,
        panel: WeeklyMarketPanel,
        *,
        as_of_date: date,
        training_dates: tuple[date, ...],
    ) -> None:
        if as_of_date not in panel.dates:
            raise ValueError(f"weekly fit as-of date {as_of_date} is unavailable")
        if (
            not training_dates
            or tuple(sorted(set(training_dates))) != training_dates
            or training_dates[-1] >= as_of_date
        ):
            raise ValueError(
                "weekly fit training dates must be unique, increasing, and "
                "strictly before the as-of date"
            )
        self._panel = panel
        self._as_of_date = as_of_date
        self.geometry = dict(panel.geometry)
        self.dataset_manifest_sha256 = getattr(
            panel,
            "dataset_manifest_sha256",
            "synthetic-panel-without-manifest",
        )
        self.training_dates = training_dates
        self._permitted_market_dates = set((*training_dates, as_of_date))
        self.dates = (*training_dates, as_of_date)
        observations = getattr(panel, "observations", None)
        if observations is not None:
            self.observations = observations.filter(
                pl.col("date").is_in(list(self.dates))
            )

    def market_day(self, trading_date: date) -> PublicMarketDay:
        """Return a dated state only when it is within the fit information set."""
        self._require_permitted_market_date(trading_date)
        return self._panel.market_day(trading_date)

    def one_day_transition(self, origin_date: date) -> OneDayTransition:
        """Return a transition only for a precommitted training origin."""
        if origin_date not in self.training_dates:
            raise ValueError(
                f"{origin_date} is outside the weekly fit training window"
            )
        transition = self._panel.one_day_transition(origin_date)
        if transition.next_date > self._as_of_date:
            raise RuntimeError("weekly fit transition crosses its as-of date")
        self._permitted_market_dates.add(transition.next_date)
        return transition

    def prior_one_day_origins(
        self,
        origin_date: date,
        *,
        count: int,
    ) -> tuple[date, ...]:
        """Return only the requested suffix of the precommitted training window."""
        self._require_permitted_market_date(origin_date)
        eligible = tuple(value for value in self.training_dates if value < origin_date)
        if count <= 0 or count > len(eligible):
            raise ValueError(
                f"weekly fit contains {len(eligible)} training origins before "
                f"{origin_date}, but {count} were requested"
            )
        return eligible[-count:]

    def next_one_session_date(self, origin_date: date) -> date:
        """Return the scheduled next session without exposing its market state."""
        self._require_permitted_market_date(origin_date)
        next_date = self._panel.next_one_session_date(origin_date)
        if origin_date != self._as_of_date and next_date > self._as_of_date:
            raise RuntimeError("weekly fit next session crosses its as-of date")
        return next_date

    def _require_permitted_market_date(self, trading_date: date) -> None:
        if trading_date > self._as_of_date:
            raise ValueError(
                f"{trading_date} is after weekly fit cutoff {self._as_of_date}"
            )
        if trading_date not in self._permitted_market_dates:
            raise ValueError(
                f"{trading_date} is outside the weekly fit information window"
            )


@dataclass(frozen=True, kw_only=True)
class WeeklyPriceRecord:
    """Distinct market, risk-neutral, and indifference-price quantities."""

    common_market_price_per_fit_spot: float | None
    q_expectation_per_fit_spot: float | None
    buyer_indifference_per_fit_spot: float | None
    seller_indifference_per_fit_spot: float | None
    continuation_price_per_fit_spot: float | None

    def __post_init__(self) -> None:
        values = (
            self.common_market_price_per_fit_spot,
            self.q_expectation_per_fit_spot,
            self.buyer_indifference_per_fit_spot,
            self.seller_indifference_per_fit_spot,
            self.continuation_price_per_fit_spot,
        )
        if any(
            value is not None and (not math.isfinite(value) or value < 0.0)
            for value in values
        ):
            raise ValueError("weekly price quantities must be finite and non-negative")
        if (
            self.q_expectation_per_fit_spot is None
            and self.buyer_indifference_per_fit_spot is None
        ):
            raise ValueError(
                "weekly prices require a Q expectation or buyer indifference price"
            )

    @property
    def long_inception_price_per_fit_spot(self) -> float:
        """Return the price used for a long-claim pricing/hedging decomposition."""
        if self.buyer_indifference_per_fit_spot is not None:
            return self.buyer_indifference_per_fit_spot
        assert self.q_expectation_per_fit_spot is not None
        return self.q_expectation_per_fit_spot


class WeeklyMarketFit(Protocol):
    """Target-independent dated market-model fit."""

    market_fit_identity: str


class WeeklyFittedPolicy(Protocol):
    """Target-specific weekly policy and its distinct price record."""

    market_fit_identity: str
    policy_fit_identity: str
    prices: WeeklyPriceRecord


class WeeklyModelAdapter(Protocol):
    """Model-specific fit and predictable-action interface."""

    model: str
    measure: str
    model_config_identity: str
    policy_config_identity: str

    def fit_market(
        self,
        panel: AsOfWeeklyMarketPanel,
        *,
        instruction: WeeklyFitInstruction,
    ) -> WeeklyMarketFit: ...

    def fit_policy(
        self,
        market_fit: WeeklyMarketFit,
        *,
        instruction: WeeklyFitInstruction,
        claim: WeeklyClaimDefinition,
        contracts: tuple[OptionContract, ...],
        state: WeeklyDecisionState,
    ) -> WeeklyFittedPolicy: ...

    def action(
        self,
        fit: WeeklyFittedPolicy,
        *,
        state: WeeklyDecisionState,
    ) -> WeeklyPolicyAction: ...


@dataclass(frozen=True, kw_only=True)
class WeeklyDecisionState:
    """Information exposed to a policy at one realized hedge decision."""

    instruction: WeeklyFitInstruction
    decision_index: int
    next_session_date: date
    market_day: PublicMarketDay
    contracts: tuple[OptionContract, ...]
    spot_over_episode_origin: float
    inception_forward_over_episode_spot: FloatArray
    inception_forward_path: FloatArray
    rolling_forward_path: FloatArray
    discount_to_origin: float
    option_marks_pv: FloatArray
    option_half_spreads_pv: FloatArray
    option_alive: NDArray[np.bool_]
    previous_spot_position: float
    previous_option_positions: FloatArray

    def __post_init__(self) -> None:
        contract_count = len(self.contracts)
        if self.decision_index < 0:
            raise ValueError("decision_index must be non-negative")
        if self.market_day.trading_date != self.instruction.decision_date:
            raise ValueError("weekly decision state date does not match its instruction")
        if self.next_session_date <= self.market_day.trading_date:
            raise ValueError("weekly next session must follow the decision date")
        if self.inception_forward_over_episode_spot.shape != (6,):
            raise ValueError("weekly inception forward curve must cover days zero to five")
        if self.inception_forward_path.shape != (self.decision_index + 1,):
            raise ValueError("inception-forward history includes unavailable dates")
        if self.rolling_forward_path.shape != (self.decision_index + 1,):
            raise ValueError("rolling-forward history includes unavailable dates")
        vector_shape = (contract_count,)
        if any(
            values.shape != vector_shape
            for values in (
                self.option_marks_pv,
                self.option_half_spreads_pv,
                self.option_alive,
                self.previous_option_positions,
            )
        ):
            raise ValueError("weekly decision option vectors do not align")
        scalar_values = (
            self.spot_over_episode_origin,
            self.discount_to_origin,
            self.previous_spot_position,
        )
        if any(not math.isfinite(value) for value in scalar_values):
            raise ValueError("weekly decision scalars must be finite")
        float_arrays = (
            self.inception_forward_over_episode_spot,
            self.inception_forward_path,
            self.rolling_forward_path,
            self.option_marks_pv,
            self.option_half_spreads_pv,
            self.previous_option_positions,
        )
        if any(not np.all(np.isfinite(values)) for values in float_arrays):
            raise ValueError("weekly decision arrays must be finite")
        if (
            self.spot_over_episode_origin <= 0.0
            or self.discount_to_origin <= 0.0
            or np.any(self.inception_forward_over_episode_spot <= 0.0)
            or np.any(self.inception_forward_path <= 0.0)
            or np.any(self.rolling_forward_path <= 0.0)
            or np.any(self.option_marks_pv < 0.0)
            or np.any(self.option_half_spreads_pv < 0.0)
        ):
            raise ValueError("weekly decision state contains invalid market values")


@dataclass(frozen=True, kw_only=True)
class WeeklyPolicyAction:
    """Predictable holdings selected for one weekly holding interval."""

    spot_position: float
    option_positions: FloatArray

    def __post_init__(self) -> None:
        if not math.isfinite(self.spot_position):
            raise ValueError("weekly spot position must be finite")
        if self.option_positions.ndim != 1 or not np.all(
            np.isfinite(self.option_positions)
        ):
            raise ValueError("weekly option positions must be a finite vector")


@dataclass(frozen=True, kw_only=True)
class WeeklyModelRun:
    """One model's actions, fit audit trail, and common accounting result."""

    result: WeeklyHedgeResult
    market_fit_identities: tuple[str, ...]
    policy_fit_identities: tuple[str, ...]
    fit_as_of_dates: tuple[str, ...]
    fit_price_records: tuple[WeeklyPriceRecord, ...]
    fit_long_prices_origin_pv: FloatArray
    spot_positions: FloatArray
    option_positions: FloatArray

    def __post_init__(self) -> None:
        decision_count = len(self.policy_fit_identities)
        if (
            len(self.market_fit_identities) != decision_count
            or len(self.fit_as_of_dates) != decision_count
            or len(self.fit_price_records) != decision_count
        ):
            raise ValueError("weekly fit audit arrays do not align")
        if self.fit_long_prices_origin_pv.shape != (decision_count,):
            raise ValueError("weekly fit prices do not align with decisions")
        if self.spot_positions.shape != (decision_count,):
            raise ValueError("weekly spot actions do not align with decisions")
        if self.option_positions.shape[0] != decision_count:
            raise ValueError("weekly option actions do not align with decisions")


def run_weekly_model(
    panel: WeeklyMarketPanel,
    *,
    episode: WeeklyEpisodePath,
    claim: WeeklyClaimDefinition,
    adapter: WeeklyModelAdapter,
    claim_quantity: float = 1.0,
    spot_cost_bps: float = 1.0,
    market_fit_cache: dict[str, WeeklyMarketFit] | None = None,
    policy_fit_cache: dict[str, WeeklyFittedPolicy] | None = None,
) -> WeeklyModelRun:
    """Fit only on scheduled dates, collect predictable actions, and account."""
    if claim.market_price is None:
        raise ValueError("weekly model run requires an explicit common market price")
    instructions = weekly_fit_instructions(
        panel,
        episode=episode,
        model=adapter.model,
        measure=adapter.measure,
        target_label=claim.target.label,
        claim_quantity=claim_quantity,
        model_config_identity=adapter.model_config_identity,
        policy_config_identity=adapter.policy_config_identity,
    )
    if market_fit_cache is None:
        market_fit_cache = {}
    if policy_fit_cache is None:
        policy_fit_cache = {}
    price_records: list[WeeklyPriceRecord] = []
    fit_long_prices_origin_pv = np.empty(len(instructions), dtype=np.float64)
    spot_positions = np.empty(len(instructions), dtype=np.float64)
    option_positions = np.empty(
        (len(instructions), len(episode.contracts)),
        dtype=np.float64,
    )
    previous_spot = 0.0
    previous_options = np.zeros(len(episode.contracts), dtype=np.float64)
    inception_forward_over_episode_spot = np.asarray(
        [
            1.0,
            *(
                episode.market_days[0].forward_at_business_dte(business_dte)
                for business_dte in range(1, 6)
            ),
        ],
        dtype=np.float64,
    )
    for decision_index, instruction in enumerate(instructions):
        state = WeeklyDecisionState(
            instruction=instruction,
            decision_index=decision_index,
            next_session_date=episode.transitions[decision_index].next_date,
            market_day=episode.market_days[decision_index],
            contracts=episode.contracts,
            spot_over_episode_origin=float(
                episode.spot_over_episode_origin[decision_index]
            ),
            inception_forward_over_episode_spot=(
                inception_forward_over_episode_spot.copy()
            ),
            inception_forward_path=(
                episode.inception_forward_normalized_spot_path[
                    : decision_index + 1
                ].copy()
            ),
            rolling_forward_path=(
                episode.rolling_forward_adjusted_spot_path[
                    : decision_index + 1
                ].copy()
            ),
            discount_to_origin=float(episode.discount_to_origin[decision_index]),
            option_marks_pv=episode.option_marks_pv[decision_index].copy(),
            option_half_spreads_pv=(
                episode.option_half_spreads_pv[decision_index].copy()
            ),
            option_alive=episode.option_alive[decision_index].copy(),
            previous_spot_position=previous_spot,
            previous_option_positions=previous_options.copy(),
        )
        market_fit = market_fit_cache.get(instruction.market_fit_identity)
        as_of_panel = AsOfWeeklyMarketPanel(
            panel,
            as_of_date=instruction.fit_as_of_date,
            training_dates=instruction.training_dates,
        )
        if market_fit is None:
            market_fit = adapter.fit_market(
                as_of_panel,
                instruction=instruction,
            )
            if market_fit.market_fit_identity != instruction.market_fit_identity:
                raise RuntimeError("weekly adapter returned the wrong market-fit identity")
            market_fit_cache[instruction.market_fit_identity] = market_fit
        fit = policy_fit_cache.get(instruction.policy_fit_identity)
        if fit is None:
            fit = adapter.fit_policy(
                market_fit,
                instruction=instruction,
                claim=claim,
                contracts=episode.contracts,
                state=state,
            )
            if fit.market_fit_identity != instruction.market_fit_identity:
                raise RuntimeError("weekly policy refers to the wrong market fit")
            if fit.policy_fit_identity != instruction.policy_fit_identity:
                raise RuntimeError("weekly adapter returned the wrong policy-fit identity")
            policy_fit_cache[instruction.policy_fit_identity] = fit
        action = adapter.action(fit, state=state)
        if action.option_positions.shape != (len(episode.contracts),):
            raise ValueError("weekly adapter option action has the wrong shape")
        if np.any(np.abs(action.option_positions[~state.option_alive]) > 1.0e-14):
            raise ValueError("weekly adapter attempted to trade an expired contract")
        fit_index = episode.spec.decision_dates.index(instruction.fit_as_of_date)
        price_records.append(fit.prices)
        fit_long_prices_origin_pv[decision_index] = (
            fit.prices.long_inception_price_per_fit_spot
            * episode.discount_to_origin[fit_index]
            * episode.spot_over_episode_origin[fit_index]
        )
        spot_positions[decision_index] = action.spot_position
        option_positions[decision_index] = action.option_positions
        previous_spot = action.spot_position
        previous_options = np.where(
            episode.option_settles_next[decision_index],
            0.0,
            action.option_positions,
        )

    result = evaluate_weekly_hedge(
        episode,
        claim=claim,
        spot_positions=spot_positions,
        option_positions=option_positions,
        market_price=claim.market_price,
        model_price=float(fit_long_prices_origin_pv[0]),
        claim_quantity=claim_quantity,
        spot_cost_bps=spot_cost_bps,
    )
    return WeeklyModelRun(
        result=result,
        market_fit_identities=tuple(
            instruction.market_fit_identity for instruction in instructions
        ),
        policy_fit_identities=tuple(
            instruction.policy_fit_identity for instruction in instructions
        ),
        fit_as_of_dates=tuple(
            instruction.fit_as_of_date.isoformat() for instruction in instructions
        ),
        fit_price_records=tuple(price_records),
        fit_long_prices_origin_pv=fit_long_prices_origin_pv,
        spot_positions=spot_positions,
        option_positions=option_positions,
    )
