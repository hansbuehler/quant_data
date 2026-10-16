"""Five-session frozen-universe accounting for the weekly hedge benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.hedging.accounting import ClaimPnlBreakdown, claim_pnl_breakdown
from quantdata.hedging.claims import (
    BarrierDirection,
    BarrierOptionClaim,
    DigitalClaim,
    RealizedVarianceClaim,
    normalized_strike_from_nms,
)
from quantdata.hedging.contracts import OptionContract, OptionQuote, OptionRight
from quantdata.hedging.data.pricing import price_sanos_digitals
from quantdata.hedging.protocols import EpisodeSpec, HedgeSchedule
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    OneDayTransition,
    PublicMarketDay,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

WEEKLY_HORIZON = 5


class WeeklyMarketPanel(Protocol):
    """Minimal public-panel interface required by weekly episode construction."""

    dates: tuple[date, ...]
    geometry: dict[str, object]

    def one_day_transition(self, origin_date: date) -> OneDayTransition: ...

    def market_day(self, trading_date: date) -> PublicMarketDay: ...

    def prior_one_day_origins(
        self,
        origin_date: date,
        *,
        count: int,
    ) -> tuple[date, ...]: ...

    def next_one_session_date(self, origin_date: date) -> date: ...


class WeeklyTargetKind(StrEnum):
    """Agreed five-session target families."""

    DIGITAL = "digital"
    REALIZED_VARIANCE = "realized-variance"
    DOWN_AND_OUT_PUT = "down-and-out-put"
    UP_AND_OUT_CALL = "up-and-out-call"


class WeeklyPathBasis(StrEnum):
    """Spot path used to settle a normalized target."""

    EPISODE_SPOT = "episode-spot"
    INCEPTION_FORWARD = "inception-forward"


@dataclass(frozen=True, kw_only=True)
class WeeklyTarget:
    """One inception-defined target in the agreed weekly family."""

    kind: WeeklyTargetKind
    right: OptionRight | None = None
    nms: float | None = None

    def __post_init__(self) -> None:
        if self.kind is WeeklyTargetKind.DIGITAL:
            if self.right is None or self.nms is None or not math.isfinite(self.nms):
                raise ValueError("weekly digitals require a right and finite NMS")
            if self.right is OptionRight.PUT and self.nms > 0.0:
                raise ValueError("weekly OTM puts cannot have positive NMS")
            if self.right is OptionRight.CALL and self.nms < 0.0:
                raise ValueError("weekly OTM calls cannot have negative NMS")
        elif self.right is not None or self.nms is not None:
            raise ValueError("non-digital weekly targets do not accept right or NMS")

    @property
    def label(self) -> str:
        """Return a stable target label."""
        if self.kind is not WeeklyTargetKind.DIGITAL:
            return self.kind.value
        assert self.right is not None and self.nms is not None
        nms_label = f"nms{self.nms:+.2f}" if self.nms != 0.0 else "atm"
        return f"digital-{self.right.value}-{WEEKLY_HORIZON}d-{nms_label}"


@dataclass(frozen=True, kw_only=True)
class WeeklyClaimDefinition:
    """Concrete weekly claim, settlement basis, and optional SANOS market mark."""

    target: WeeklyTarget
    claim: DigitalClaim | RealizedVarianceClaim | BarrierOptionClaim
    path_basis: WeeklyPathBasis
    market_price: float | None

    def __post_init__(self) -> None:
        if self.market_price is not None and (
            not math.isfinite(self.market_price) or self.market_price < 0.0
        ):
            raise ValueError("weekly market price must be finite and non-negative")

    def payoff(self, episode: WeeklyEpisodePath) -> float:
        """Return the undiscounted terminal payoff on the configured path basis."""
        if self.path_basis is WeeklyPathBasis.EPISODE_SPOT:
            path = episode.spot_over_episode_origin
        elif self.path_basis is WeeklyPathBasis.INCEPTION_FORWARD:
            path = episode.inception_forward_normalized_spot_path
        else:
            raise AssertionError(f"unsupported weekly path basis {self.path_basis}")
        return self.claim.payoff(path)


@dataclass(frozen=True, kw_only=True)
class WeeklyEpisodePath:
    """One realized five-session market path with a frozen inception book."""

    spec: EpisodeSpec
    transitions: tuple[OneDayTransition, ...]
    market_days: tuple[PublicMarketDay, ...]
    contracts: tuple[OptionContract, ...]
    inception_quotes: tuple[OptionQuote, ...]
    spot_over_episode_origin: FloatArray
    inception_forward_normalized_spot_path: FloatArray
    rolling_forward_adjusted_spot_path: FloatArray
    discount_to_origin: FloatArray
    option_marks_pv: FloatArray
    option_half_spreads_pv: FloatArray
    option_alive: BoolArray
    option_step_gains: FloatArray
    option_settles_next: BoolArray
    spot_step_gains: FloatArray

    def __post_init__(self) -> None:
        step_count = len(self.transitions)
        contract_count = len(self.contracts)
        if step_count != WEEKLY_HORIZON:
            raise ValueError(f"weekly episode must contain {WEEKLY_HORIZON} sessions")
        if len(self.market_days) != step_count + 1:
            raise ValueError("weekly market days do not align with transitions")
        if len(self.inception_quotes) != contract_count:
            raise ValueError("weekly contracts and inception quotes do not align")
        if self.spec.decision_dates != tuple(
            transition.origin.trading_date for transition in self.transitions
        ):
            raise ValueError("weekly decision dates do not align with transitions")
        if self.spec.terminal_date != self.transitions[-1].next_date:
            raise ValueError("weekly terminal date does not align with transitions")
        path_shape = (step_count + 1,)
        matrix_shape = (step_count + 1, contract_count)
        step_matrix_shape = (step_count, contract_count)
        if any(
            values.shape != path_shape
            for values in (
                self.spot_over_episode_origin,
                self.inception_forward_normalized_spot_path,
                self.rolling_forward_adjusted_spot_path,
                self.discount_to_origin,
            )
        ):
            raise ValueError("weekly spot and discount paths do not align")
        if any(
            values.shape != matrix_shape
            for values in (
                self.option_marks_pv,
                self.option_half_spreads_pv,
                self.option_alive,
            )
        ):
            raise ValueError("weekly option state arrays do not align")
        if self.option_step_gains.shape != step_matrix_shape:
            raise ValueError("weekly option gains do not align")
        if self.option_settles_next.shape != step_matrix_shape:
            raise ValueError("weekly option settlement flags do not align")
        if self.spot_step_gains.shape != (step_count,):
            raise ValueError("weekly spot gains do not align")
        float_arrays = (
            self.spot_over_episode_origin,
            self.inception_forward_normalized_spot_path,
            self.rolling_forward_adjusted_spot_path,
            self.discount_to_origin,
            self.option_marks_pv,
            self.option_half_spreads_pv,
            self.option_step_gains,
            self.spot_step_gains,
        )
        if any(not np.all(np.isfinite(values)) for values in float_arrays):
            raise ValueError("weekly episode arrays must be finite")
        if (
            np.any(self.spot_over_episode_origin <= 0.0)
            or np.any(self.inception_forward_normalized_spot_path <= 0.0)
            or np.any(self.rolling_forward_adjusted_spot_path <= 0.0)
            or np.any(self.discount_to_origin <= 0.0)
            or np.any(self.option_marks_pv < 0.0)
            or np.any(self.option_half_spreads_pv < 0.0)
        ):
            raise ValueError("weekly episode contains invalid prices or paths")
        if not np.allclose(
            self.option_marks_pv[~self.option_alive],
            0.0,
            atol=0.0,
            rtol=0.0,
        ) or not np.allclose(
            self.option_half_spreads_pv[~self.option_alive],
            0.0,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("expired options cannot retain marks or spreads")


@dataclass(frozen=True, kw_only=True)
class WeeklyHedgeResult:
    """Accounting result for one realized weekly hedge policy."""

    schedule: HedgeSchedule
    target_label: str
    origin_date: date
    terminal_date: date
    terminal_value: float
    hedge_gain: float
    transaction_cost: float
    spot_transaction_cost: float
    option_transaction_cost: float
    spot_gain: float
    option_gain: float
    step_transaction_costs: FloatArray
    step_hedge_gains: FloatArray
    breakdown: ClaimPnlBreakdown

    def __post_init__(self) -> None:
        values = (
            self.terminal_value,
            self.hedge_gain,
            self.transaction_cost,
            self.spot_transaction_cost,
            self.option_transaction_cost,
            self.spot_gain,
            self.option_gain,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("weekly hedge result must be finite")
        if any(
            value < 0.0
            for value in (
                self.transaction_cost,
                self.spot_transaction_cost,
                self.option_transaction_cost,
            )
        ):
            raise ValueError("weekly transaction costs must be non-negative")
        if self.step_transaction_costs.shape != (WEEKLY_HORIZON + 1,):
            raise ValueError("weekly step costs must include terminal liquidation")
        if self.step_hedge_gains.shape != (WEEKLY_HORIZON,):
            raise ValueError("weekly step gains do not align")
        if not np.all(np.isfinite(self.step_transaction_costs)) or not np.all(
            np.isfinite(self.step_hedge_gains)
        ):
            raise ValueError("weekly step diagnostics must be finite")
        if not math.isclose(
            self.spot_transaction_cost + self.option_transaction_cost,
            self.transaction_cost,
            abs_tol=1.0e-12,
        ):
            raise ValueError("weekly transaction-cost components do not reconcile")
        if not math.isclose(
            self.spot_gain + self.option_gain,
            self.hedge_gain,
            abs_tol=1.0e-12,
        ):
            raise ValueError("weekly hedge-gain components do not reconcile")


@dataclass(frozen=True, kw_only=True)
class WeeklyFitInstruction:
    """Auditable model/policy fit identity used at one weekly decision."""

    schedule: HedgeSchedule
    model: str
    measure: str
    target_label: str
    claim_quantity: float
    model_config_identity: str
    policy_config_identity: str
    episode_origin_date: date
    decision_date: date
    fit_as_of_date: date
    training_dates: tuple[date, ...]
    frozen_contract_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = (
            self.model,
            self.measure,
            self.target_label,
            self.model_config_identity,
            self.policy_config_identity,
        )
        if any(not label.strip() for label in labels):
            raise ValueError("weekly fit labels and configuration identities are required")
        if not math.isfinite(self.claim_quantity):
            raise ValueError("weekly claim quantity must be finite")
        if self.schedule is HedgeSchedule.DAILY:
            raise ValueError("weekly fit instructions require a weekly schedule")
        if self.decision_date < self.episode_origin_date:
            raise ValueError("weekly decision cannot precede the episode origin")
        if self.fit_as_of_date > self.decision_date:
            raise ValueError("weekly fit cannot use an as-of date after its decision")
        if not self.training_dates or tuple(sorted(set(self.training_dates))) != (
            self.training_dates
        ):
            raise ValueError("weekly fit training dates must be non-empty and increasing")
        if self.training_dates[-1] >= self.fit_as_of_date:
            raise ValueError("weekly fit training origins must strictly precede its as-of date")
        if not self.frozen_contract_ids or len(self.frozen_contract_ids) != len(
            set(self.frozen_contract_ids)
        ):
            raise ValueError("weekly fit requires unique frozen contract identities")
        if (
            self.schedule is HedgeSchedule.FROZEN_WEEKLY
            and self.fit_as_of_date != self.episode_origin_date
        ):
            raise ValueError("frozen-weekly fits must remain at the episode origin")
        if (
            self.schedule is HedgeSchedule.RESTARTED_WEEKLY
            and self.fit_as_of_date != self.decision_date
        ):
            raise ValueError("restarted-weekly fits must use the decision date")

    @property
    def market_fit_identity(self) -> str:
        """Return the dated market-model identity shared across targets."""
        payload = {
            "model": self.model,
            "measure": self.measure,
            "model_config_identity": self.model_config_identity,
            "fit_as_of_date": self.fit_as_of_date.isoformat(),
            "training_dates": [value.isoformat() for value in self.training_dates],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def policy_fit_identity(self) -> str:
        """Return the target- and frozen-book-specific policy identity."""
        payload = {
            "market_fit_identity": self.market_fit_identity,
            "schedule": self.schedule.value,
            "target_label": self.target_label,
            "claim_quantity": self.claim_quantity,
            "policy_config_identity": self.policy_config_identity,
            "episode_origin_date": self.episode_origin_date.isoformat(),
            "frozen_contract_ids": list(self.frozen_contract_ids),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def fit_identity(self) -> str:
        """Return the policy-fit identity retained for artifact compatibility."""
        return self.policy_fit_identity


def weekly_targets() -> tuple[WeeklyTarget, ...]:
    """Return the 13 agreed weekly digital, variance, and barrier targets."""
    digitals = (
        WeeklyTarget(kind=WeeklyTargetKind.DIGITAL, right=OptionRight.PUT, nms=0.0),
        *(
            WeeklyTarget(
                kind=WeeklyTargetKind.DIGITAL,
                right=OptionRight.PUT,
                nms=nms,
            )
            for nms in (-1.0, -0.75, -0.5, -0.25)
        ),
        WeeklyTarget(kind=WeeklyTargetKind.DIGITAL, right=OptionRight.CALL, nms=0.0),
        *(
            WeeklyTarget(
                kind=WeeklyTargetKind.DIGITAL,
                right=OptionRight.CALL,
                nms=nms,
            )
            for nms in (0.25, 0.5, 0.75, 1.0)
        ),
    )
    return (
        *digitals,
        WeeklyTarget(kind=WeeklyTargetKind.REALIZED_VARIANCE),
        WeeklyTarget(kind=WeeklyTargetKind.DOWN_AND_OUT_PUT),
        WeeklyTarget(kind=WeeklyTargetKind.UP_AND_OUT_CALL),
    )


def build_weekly_claim(
    origin: PublicMarketDay,
    *,
    target: WeeklyTarget,
) -> WeeklyClaimDefinition:
    """Construct one fixed target using only the episode-origin market state."""
    if target.kind is WeeklyTargetKind.DIGITAL:
        assert target.right is not None and target.nms is not None
        atm_vol = origin.atm_vol_at_business_dte(WEEKLY_HORIZON)
        strike_over_forward = normalized_strike_from_nms(
            nms=target.nms,
            business_dte=float(WEEKLY_HORIZON),
            atm_annual_vol=atm_vol,
            epsilon_atm_annual_vol=origin.book_selection_epsilon_atm_vol,
        )
        discount_factor = origin.discount_at_business_dte(WEEKLY_HORIZON)
        probability = float(
            price_sanos_digitals(
                query_expiries_y=np.asarray([WEEKLY_HORIZON / DAYS_PER_YEAR]),
                query_strikes=np.asarray([strike_over_forward]),
                is_call=np.asarray([target.right is OptionRight.CALL]),
                model_expiries_y=origin.sanos_model_expiries_y,
                atm_variances=origin.sanos_kernel_total_variances,
                support_strikes=origin.sanos_support_strikes,
                densities=origin.sanos_densities,
                smoothness=origin.sanos_smoothness,
                one_minute_y=origin.one_minute_y,
            )[0]
        )
        return WeeklyClaimDefinition(
            target=target,
            claim=DigitalClaim(
                right=target.right,
                strike=strike_over_forward,
                maturity_business_days=WEEKLY_HORIZON,
            ),
            path_basis=WeeklyPathBasis.INCEPTION_FORWARD,
            market_price=discount_factor * probability,
        )
    if target.kind is WeeklyTargetKind.REALIZED_VARIANCE:
        return WeeklyClaimDefinition(
            target=target,
            claim=RealizedVarianceClaim(maturity_business_days=WEEKLY_HORIZON),
            path_basis=WeeklyPathBasis.INCEPTION_FORWARD,
            market_price=None,
        )
    if target.kind is WeeklyTargetKind.DOWN_AND_OUT_PUT:
        claim = BarrierOptionClaim(
            right=OptionRight.PUT,
            strike=1.0,
            barrier=0.95,
            direction=BarrierDirection.DOWN,
            maturity_business_days=WEEKLY_HORIZON,
        )
    elif target.kind is WeeklyTargetKind.UP_AND_OUT_CALL:
        claim = BarrierOptionClaim(
            right=OptionRight.CALL,
            strike=1.0,
            barrier=1.05,
            direction=BarrierDirection.UP,
            maturity_business_days=WEEKLY_HORIZON,
        )
    else:
        raise AssertionError(f"unsupported weekly target kind {target.kind}")
    return WeeklyClaimDefinition(
        target=target,
        claim=claim,
        path_basis=WeeklyPathBasis.INCEPTION_FORWARD,
        market_price=None,
    )


def build_weekly_episode_path(
    panel: WeeklyMarketPanel,
    *,
    origin_date: date,
    schedule: HedgeSchedule,
    rolling_window_business_days: int = 252,
    maximum_option_business_dte: float = 5.0,
    nms_range: tuple[float, float] = (-2.0, 2.0),
) -> WeeklyEpisodePath:
    """Build one gap-free five-session path and freeze its inception contracts."""
    if schedule is HedgeSchedule.DAILY:
        raise ValueError("weekly episode construction requires a weekly schedule")
    if rolling_window_business_days <= 0:
        raise ValueError("rolling_window_business_days must be positive")
    if not math.isclose(
        maximum_option_business_dte,
        float(WEEKLY_HORIZON),
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "weekly hedge expiry horizon must equal the five-session episode"
        )
    transitions: list[OneDayTransition] = []
    current_date = origin_date
    for _ in range(WEEKLY_HORIZON):
        transition = panel.one_day_transition(current_date)
        if transition.origin.trading_date != current_date:
            raise ValueError("weekly transition origin does not match requested date")
        transitions.append(transition)
        current_date = transition.next_date
    transition_tuple = tuple(transitions)
    market_days = (
        transition_tuple[0].origin,
        *(panel.market_day(transition.next_date) for transition in transition_tuple),
    )
    decision_dates = tuple(
        transition.origin.trading_date for transition in transition_tuple
    )
    inception_quotes = weekly_inception_quotes(
        market_days[0],
        terminal_date=transition_tuple[-1].next_date,
        nms_range=nms_range,
        episode_origin_date=origin_date,
    )
    if not inception_quotes:
        raise ValueError(f"{origin_date} has no eligible weekly hedge options")
    contracts = tuple(quote.contract for quote in inception_quotes)
    spec = EpisodeSpec(
        schedule=schedule,
        origin_date=origin_date,
        training_dates=panel.prior_one_day_origins(
            origin_date,
            count=rolling_window_business_days,
        ),
        decision_dates=decision_dates,
        terminal_date=transition_tuple[-1].next_date,
        frozen_contract_ids=tuple(contract.contract_id for contract in contracts),
    )

    spot_path = np.ones(WEEKLY_HORIZON + 1, dtype=np.float64)
    inception_forward_path = np.ones(WEEKLY_HORIZON + 1, dtype=np.float64)
    rolling_forward_adjusted_path = np.ones(
        WEEKLY_HORIZON + 1, dtype=np.float64
    )
    discount_path = np.ones(WEEKLY_HORIZON + 1, dtype=np.float64)
    spot_step_gains = np.empty(WEEKLY_HORIZON, dtype=np.float64)
    for step, transition in enumerate(transition_tuple):
        spot_path[step + 1] = spot_path[step] * (
            transition.forward_over_spot * transition.terminal_normalized_spot
        )
        rolling_forward_adjusted_path[step + 1] = (
            rolling_forward_adjusted_path[step]
            * transition.terminal_normalized_spot
        )
        inception_forward_path[step + 1] = (
            spot_path[step + 1]
            / market_days[0].forward_at_business_dte(step + 1)
        )
        discount_path[step + 1] = (
            discount_path[step] * transition.discount_factor
        )
        spot_step_gains[step] = (
            discount_path[step]
            * spot_path[step]
            * transition.discounted_spot_gain
        )

    date_count = WEEKLY_HORIZON + 1
    contract_count = len(contracts)
    marks_pv = np.zeros((date_count, contract_count), dtype=np.float64)
    spreads_pv = np.zeros_like(marks_pv)
    alive = np.zeros((date_count, contract_count), dtype=bool)
    quote_maps: list[dict[str, OptionQuote]] = [
        {quote.contract.contract_id: quote for quote in inception_quotes}
    ]
    for step in range(1, date_count):
        snapshot = market_days[step].reprice_contracts(
            contracts,
            spot_over_episode_origin=float(spot_path[step]),
        )
        quote_maps.append(snapshot.quote_by_id())
    episode_dates = (*decision_dates, spec.terminal_date)
    for step, (trading_date, quote_map) in enumerate(
        zip(episode_dates, quote_maps, strict=True)
    ):
        for contract_index, contract in enumerate(contracts):
            if contract.expiry_date <= trading_date:
                continue
            try:
                quote = quote_map[contract.contract_id]
            except KeyError as error:
                raise ValueError(
                    f"{trading_date}: missing frozen contract {contract.contract_id}"
                ) from error
            alive[step, contract_index] = True
            marks_pv[step, contract_index] = discount_path[step] * quote.mark
            spreads_pv[step, contract_index] = (
                discount_path[step] * quote.half_spread
            )

    option_step_gains = np.zeros(
        (WEEKLY_HORIZON, contract_count), dtype=np.float64
    )
    settles_next = np.zeros_like(option_step_gains, dtype=bool)
    for step, transition in enumerate(transition_tuple):
        for contract_index, contract in enumerate(contracts):
            if not alive[step, contract_index]:
                continue
            if contract.expiry_date == transition.next_date:
                settles_next[step, contract_index] = True
                settlement_pv = discount_path[step + 1] * contract.payoff(
                    float(spot_path[step + 1])
                )
                option_step_gains[step, contract_index] = (
                    settlement_pv - marks_pv[step, contract_index]
                )
            elif contract.expiry_date > transition.next_date:
                if not alive[step + 1, contract_index]:
                    raise ValueError(
                        f"{transition.next_date}: live contract disappeared before expiry"
                    )
                option_step_gains[step, contract_index] = (
                    marks_pv[step + 1, contract_index]
                    - marks_pv[step, contract_index]
                )
            else:
                raise ValueError("frozen contract expired before its holding interval")
    return WeeklyEpisodePath(
        spec=spec,
        transitions=transition_tuple,
        market_days=market_days,
        contracts=contracts,
        inception_quotes=inception_quotes,
        spot_over_episode_origin=spot_path,
        inception_forward_normalized_spot_path=inception_forward_path,
        rolling_forward_adjusted_spot_path=rolling_forward_adjusted_path,
        discount_to_origin=discount_path,
        option_marks_pv=marks_pv,
        option_half_spreads_pv=spreads_pv,
        option_alive=alive,
        option_step_gains=option_step_gains,
        option_settles_next=settles_next,
        spot_step_gains=spot_step_gains,
    )


def weekly_inception_quotes(
    origin: PublicMarketDay,
    *,
    terminal_date: date,
    nms_range: tuple[float, float],
    episode_origin_date: date,
) -> tuple[OptionQuote, ...]:
    """Freeze expiries through the actual fifth session, plus the first later one."""
    if terminal_date <= origin.trading_date:
        raise ValueError("weekly terminal date must follow the episode origin")
    all_quotes = origin.hedge_snapshot(
        maximum_business_dte=float(np.max(origin.business_dtes)),
        nms_range=nms_range,
        episode_origin_date=episode_origin_date,
    ).option_quotes
    terminal_is_listed = any(
        quote.contract.expiry_date == terminal_date for quote in all_quotes
    )
    later_expiries = tuple(
        sorted(
            {
                quote.contract.expiry_date
                for quote in all_quotes
                if quote.contract.expiry_date > terminal_date
            }
        )
    )
    if not terminal_is_listed and not later_expiries:
        raise ValueError(
            f"{origin.trading_date}: no eligible expiry reaches the weekly terminal date"
        )
    first_later = (
        None
        if terminal_is_listed or not later_expiries
        else later_expiries[0]
    )
    return tuple(
        quote
        for quote in all_quotes
        if quote.contract.expiry_date <= terminal_date
        or quote.contract.expiry_date == first_later
    )


def rolling_training_dates(
    panel: WeeklyMarketPanel,
    *,
    episode: WeeklyEpisodePath,
    decision_date: date,
) -> tuple[date, ...]:
    """Return the dated rolling window allowed for a weekly policy refit."""
    if decision_date not in episode.spec.refit_dates:
        raise ValueError(f"{decision_date} is not a refit date for this schedule")
    return panel.prior_one_day_origins(
        decision_date,
        count=len(episode.spec.training_dates),
    )


def weekly_fit_instructions(
    panel: WeeklyMarketPanel,
    *,
    episode: WeeklyEpisodePath,
    model: str,
    measure: str,
    target_label: str,
    claim_quantity: float = 1.0,
    model_config_identity: str,
    policy_config_identity: str,
) -> tuple[WeeklyFitInstruction, ...]:
    """Map every weekly decision to its frozen or restarted fit identity."""
    instructions: list[WeeklyFitInstruction] = []
    for decision_date in episode.spec.decision_dates:
        fit_as_of_date = (
            episode.spec.origin_date
            if episode.spec.schedule is HedgeSchedule.FROZEN_WEEKLY
            else decision_date
        )
        training_dates = (
            episode.spec.training_dates
            if fit_as_of_date == episode.spec.origin_date
            else rolling_training_dates(
                panel,
                episode=episode,
                decision_date=decision_date,
            )
        )
        instructions.append(
            WeeklyFitInstruction(
                schedule=episode.spec.schedule,
                model=model,
                measure=measure,
                target_label=target_label,
                claim_quantity=claim_quantity,
                model_config_identity=model_config_identity,
                policy_config_identity=policy_config_identity,
                episode_origin_date=episode.spec.origin_date,
                decision_date=decision_date,
                fit_as_of_date=fit_as_of_date,
                training_dates=training_dates,
                frozen_contract_ids=episode.spec.frozen_contract_ids,
            )
        )
    identities = {instruction.policy_fit_identity for instruction in instructions}
    expected_identity_count = (
        1
        if episode.spec.schedule is HedgeSchedule.FROZEN_WEEKLY
        else len(episode.spec.decision_dates)
    )
    if len(identities) != expected_identity_count:
        raise RuntimeError("weekly fit identities do not enforce the requested schedule")
    return tuple(instructions)


def overlapping_weekly_origins(
    panel: WeeklyMarketPanel,
    *,
    evaluation_start: date,
    evaluation_end: date,
    rolling_window_business_days: int = 252,
    nms_range: tuple[float, float] = (-2.0, 2.0),
) -> tuple[date, ...]:
    """Return every eligible origin, including overlapping five-session episodes."""
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation_start cannot follow evaluation_end")
    if rolling_window_business_days <= 0:
        raise ValueError("rolling_window_business_days must be positive")
    origins: list[date] = []
    for origin_date in panel.dates:
        if not evaluation_start <= origin_date <= evaluation_end:
            continue
        try:
            panel.prior_one_day_origins(
                origin_date,
                count=rolling_window_business_days,
            )
            current_date = origin_date
            for _ in range(WEEKLY_HORIZON):
                current_date = panel.one_day_transition(current_date).next_date
            inception_quotes = weekly_inception_quotes(
                panel.market_day(origin_date),
                terminal_date=current_date,
                nms_range=nms_range,
                episode_origin_date=origin_date,
            )
        except (KeyError, ValueError):
            continue
        if inception_quotes:
            origins.append(origin_date)
    return tuple(origins)


def evaluate_weekly_hedge(
    episode: WeeklyEpisodePath,
    *,
    claim: WeeklyClaimDefinition,
    spot_positions: NDArray[np.floating],
    option_positions: NDArray[np.floating],
    market_price: float,
    model_price: float,
    claim_quantity: float = 1.0,
    spot_cost_bps: float = 1.0,
) -> WeeklyHedgeResult:
    """Apply predictable positions with daily rebalancing and terminal liquidation."""
    spot = np.asarray(spot_positions, dtype=np.float64)
    options = np.asarray(option_positions, dtype=np.float64)
    contract_count = len(episode.contracts)
    if spot.shape != (WEEKLY_HORIZON,) or not np.all(np.isfinite(spot)):
        raise ValueError("spot positions must contain five finite decisions")
    if options.shape != (WEEKLY_HORIZON, contract_count) or not np.all(
        np.isfinite(options)
    ):
        raise ValueError("option positions must align with weekly decisions and contracts")
    if np.any(np.abs(options[~episode.option_alive[:-1]]) > 1.0e-14):
        raise ValueError("weekly policy cannot trade an expired frozen contract")
    scalar_inputs = (market_price, model_price, claim_quantity, spot_cost_bps)
    if any(not math.isfinite(value) for value in scalar_inputs):
        raise ValueError("weekly hedge scalar inputs must be finite")
    if market_price < 0.0 or spot_cost_bps < 0.0:
        raise ValueError("weekly market price and spot cost must be non-negative")

    step_costs = np.zeros(WEEKLY_HORIZON + 1, dtype=np.float64)
    step_gains = np.zeros(WEEKLY_HORIZON, dtype=np.float64)
    previous_spot = 0.0
    previous_options = np.zeros(contract_count, dtype=np.float64)
    spot_cost = 0.0
    option_cost = 0.0
    spot_gain = 0.0
    option_gain = 0.0
    for step in range(WEEKLY_HORIZON):
        previous_options = np.where(
            episode.option_alive[step], previous_options, 0.0
        )
        spot_change = float(spot[step] - previous_spot)
        spot_step_cost = (
            abs(spot_change)
            * episode.discount_to_origin[step]
            * episode.spot_over_episode_origin[step]
            * spot_cost_bps
            * 1.0e-4
        )
        option_change = options[step] - previous_options
        option_step_cost = float(
            np.sum(
                np.abs(option_change) * episode.option_half_spreads_pv[step]
            )
        )
        current_spot_gain = float(spot[step] * episode.spot_step_gains[step])
        current_option_gain = float(
            options[step] @ episode.option_step_gains[step]
        )
        step_costs[step] = spot_step_cost + option_step_cost
        step_gains[step] = current_spot_gain + current_option_gain
        spot_cost += spot_step_cost
        option_cost += option_step_cost
        spot_gain += current_spot_gain
        option_gain += current_option_gain
        previous_spot = float(spot[step])
        previous_options = np.where(
            episode.option_settles_next[step], 0.0, options[step]
        )

    terminal_spot_cost = (
        abs(previous_spot)
        * episode.discount_to_origin[-1]
        * episode.spot_over_episode_origin[-1]
        * spot_cost_bps
        * 1.0e-4
    )
    terminal_option_cost = float(
        np.sum(
            np.abs(previous_options)
            * episode.option_half_spreads_pv[-1]
            * episode.option_alive[-1]
        )
    )
    step_costs[-1] = terminal_spot_cost + terminal_option_cost
    spot_cost += terminal_spot_cost
    option_cost += terminal_option_cost
    hedge_gain = spot_gain + option_gain
    transaction_cost = spot_cost + option_cost
    terminal_value = episode.discount_to_origin[-1] * claim.payoff(episode)
    breakdown = claim_pnl_breakdown(
        claim_quantity=claim_quantity,
        market_price=market_price,
        model_price=model_price,
        payoff=terminal_value,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
    )
    return WeeklyHedgeResult(
        schedule=episode.spec.schedule,
        target_label=claim.target.label,
        origin_date=episode.spec.origin_date,
        terminal_date=episode.spec.terminal_date,
        terminal_value=terminal_value,
        hedge_gain=hedge_gain,
        transaction_cost=transaction_cost,
        spot_transaction_cost=spot_cost,
        option_transaction_cost=option_cost,
        spot_gain=spot_gain,
        option_gain=option_gain,
        step_transaction_costs=step_costs,
        step_hedge_gains=step_gains,
        breakdown=breakdown,
    )
