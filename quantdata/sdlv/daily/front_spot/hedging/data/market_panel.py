"""Dated benchmark views of the distributable hedging market panel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantdata.biz_calendar import DAYS_PER_YEAR, NMS, count_business_days_between
from quantdata.hedging.contracts import (
    MarketSnapshot,
    OptionContract,
    OptionQuote,
    OptionRight,
    eligible_option_quotes,
)
from quantdata.hedging.data.normalization import (
    interpolate_positive_curve_at_business_dte,
)
from quantdata.hedging.data.pricing import price_sanos_surface
from quantdata.hedging.data.public_dataset import (
    PublicHedgingDataset,
    validate_public_dataset,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_view import (
    PublicHedgingPanelView,
)

FloatArray = NDArray[np.float64]
_SANOS_PRICE_NEGATIVE_TOLERANCE = 1.0e-8
_SANOS_REPLAY_PRICE_FLOOR = 1.0e-12
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

PUBLIC_CANDIDATE_NAME = "spx-2022-2025-hedging-public-candidate-v1"


def default_public_candidate_path() -> Path:
    """Return the configured interim public-candidate directory."""
    root = os.environ.get("QUANTDATA_ROOT")
    if not root:
        raise RuntimeError("QUANTDATA_ROOT is required to locate the public candidate")
    return Path(root) / "datasets" / "hedging" / PUBLIC_CANDIDATE_NAME


@dataclass(frozen=True, kw_only=True)
class PublicMarketDay:
    """All public model state and executable quote-side books for one date."""

    trading_date: date
    expiry_dates: tuple[date, ...]
    expiries_y: FloatArray
    business_dtes: FloatArray
    is_am_settlement: BoolArray
    forward_over_spot: FloatArray
    discount_factors: FloatArray
    book_selection_epsilon_atm_vol: float
    book_selection_atm_vols: FloatArray
    expiry_indices: IntArray
    strikes_over_forward: FloatArray
    standardized_log_moneyness: FloatArray
    in_recommended_hedge_nms_population: BoolArray
    is_call: BoolArray
    half_spreads_over_spot: FloatArray
    sanos_prices_over_discounted_forward: FloatArray
    dlv_prices_over_discounted_forward: FloatArray
    sanos_model_expiries_y: FloatArray
    sanos_support_strikes: tuple[FloatArray, ...]
    sanos_kernel_total_variances: FloatArray
    sanos_densities: tuple[FloatArray, ...]
    sanos_smoothness: float
    one_minute_y: float
    dlv_model_expiries_y: FloatArray
    dlv_support_strikes: FloatArray
    dlv_raw_log_excess_dlvsq: FloatArray
    minimum_dlvsq: float
    maximum_dlvsq: float
    backbone_annual_volatility: float

    def __post_init__(self) -> None:
        expiry_count = len(self.expiry_dates)
        expiry_arrays = (
            self.expiries_y,
            self.business_dtes,
            self.is_am_settlement,
            self.forward_over_spot,
            self.discount_factors,
            self.book_selection_atm_vols,
        )
        if any(len(values) != expiry_count for values in expiry_arrays):
            raise ValueError("public market expiry arrays do not align")
        book_count = len(self.expiry_indices)
        book_arrays = (
            self.strikes_over_forward,
            self.standardized_log_moneyness,
            self.in_recommended_hedge_nms_population,
            self.is_call,
            self.half_spreads_over_spot,
            self.sanos_prices_over_discounted_forward,
            self.dlv_prices_over_discounted_forward,
        )
        if any(len(values) != book_count for values in book_arrays):
            raise ValueError("public market quote-side arrays do not align")
        if np.any((self.expiry_indices < 0) | (self.expiry_indices >= expiry_count)):
            raise ValueError("public market expiry indices are out of range")
        expected_dlv_shape = (
            len(self.dlv_model_expiries_y),
            len(self.dlv_support_strikes) - 2,
        )
        if self.dlv_raw_log_excess_dlvsq.shape != expected_dlv_shape:
            raise ValueError(
                "public DLV state shape "
                f"{self.dlv_raw_log_excess_dlvsq.shape} does not match "
                f"{expected_dlv_shape}"
            )
        if (
            np.any(~np.isfinite(self.dlv_model_expiries_y))
            or np.any(np.diff(self.dlv_model_expiries_y) <= 0.0)
            or np.any(~np.isfinite(self.dlv_support_strikes))
            or np.any(np.diff(self.dlv_support_strikes) <= 0.0)
            or np.any(~np.isfinite(self.dlv_raw_log_excess_dlvsq))
        ):
            raise ValueError("public DLV geometry and state must be finite and ordered")
        if not (
            0.0 < self.minimum_dlvsq < self.maximum_dlvsq
            and self.backbone_annual_volatility >= 0.0
            and np.isfinite(self.book_selection_epsilon_atm_vol)
            and self.book_selection_epsilon_atm_vol > 0.0
        ):
            raise ValueError("public DLV bounds, backbone, or NMS epsilon are invalid")

    @classmethod
    def from_rows(
        cls,
        *,
        expiry_row: dict[str, Any],
        instrument_row: dict[str, Any],
        sanos_geometry_row: dict[str, Any],
        sanos_state_row: dict[str, Any],
        dlv_geometry_row: dict[str, Any],
        dlv_state_row: dict[str, Any],
        geometry: dict[str, object],
    ) -> PublicMarketDay:
        """Construct one immutable date view from public Parquet rows."""
        trading_date = expiry_row["date"]
        dated_rows = (
            instrument_row,
            sanos_geometry_row,
            sanos_state_row,
            dlv_geometry_row,
            dlv_state_row,
        )
        if any(row["date"] != trading_date for row in dated_rows):
            raise ValueError("public model rows do not share a date")
        dlv_shape = tuple(int(value) for value in geometry["dlv_state_shape"])
        return cls(
            trading_date=trading_date,
            expiry_dates=tuple(expiry_row["expiry_dates"]),
            expiries_y=np.asarray(expiry_row["expiries_y"], dtype=np.float64),
            business_dtes=np.asarray(expiry_row["business_dtes"], dtype=np.float64),
            is_am_settlement=np.asarray(expiry_row["is_am_settlement"], dtype=bool),
            forward_over_spot=np.asarray(expiry_row["forward_over_spot"], dtype=np.float64),
            discount_factors=np.asarray(expiry_row["discount_factors"], dtype=np.float64),
            book_selection_epsilon_atm_vol=float(
                expiry_row["book_selection_epsilon_annualized_atm_vol"]
            ),
            book_selection_atm_vols=np.asarray(
                expiry_row["book_selection_annualized_atm_vols"],
                dtype=np.float64,
            ),
            expiry_indices=np.asarray(instrument_row["expiry_indices"], dtype=np.int64),
            strikes_over_forward=np.asarray(
                instrument_row["strikes_over_forward"], dtype=np.float64
            ),
            standardized_log_moneyness=np.asarray(
                instrument_row["standardized_log_moneyness"], dtype=np.float64
            ),
            in_recommended_hedge_nms_population=np.asarray(
                instrument_row["in_recommended_hedge_nms_population"], dtype=bool
            ),
            is_call=np.asarray(instrument_row["is_call"], dtype=bool),
            half_spreads_over_spot=np.asarray(
                instrument_row["half_spreads_over_spot"], dtype=np.float64
            ),
            sanos_prices_over_discounted_forward=np.asarray(
                instrument_row["sanos_prices_over_discounted_forward"],
                dtype=np.float64,
            ),
            dlv_prices_over_discounted_forward=np.asarray(
                instrument_row["dlv_prices_over_discounted_forward"],
                dtype=np.float64,
            ),
            sanos_model_expiries_y=np.asarray(
                sanos_geometry_row["model_expiries_y"], dtype=np.float64
            ),
            sanos_support_strikes=tuple(
                np.asarray(values, dtype=np.float64)
                for values in sanos_geometry_row["support_strikes"]
            ),
            sanos_kernel_total_variances=np.asarray(
                sanos_state_row["kernel_total_variances"], dtype=np.float64
            ),
            sanos_densities=tuple(
                np.asarray(values, dtype=np.float64)
                for values in sanos_state_row["densities"]
            ),
            sanos_smoothness=float(geometry["sanos_smoothness"]),
            one_minute_y=float(geometry["one_minute_y"]),
            dlv_model_expiries_y=np.asarray(
                dlv_geometry_row["model_expiries_y"], dtype=np.float64
            ),
            dlv_support_strikes=np.asarray(
                dlv_geometry_row["support_strikes"], dtype=np.float64
            ),
            dlv_raw_log_excess_dlvsq=np.asarray(
                dlv_state_row["raw_log_excess_dlvsq"], dtype=np.float64
            ).reshape(dlv_shape),
            minimum_dlvsq=float(geometry["minimum_dlvsq"]),
            maximum_dlvsq=float(geometry["maximum_dlvsq"]),
            backbone_annual_volatility=float(
                geometry["backbone_annual_volatility"]
            ),
        )

    @property
    def dlvsq(self) -> FloatArray:
        """Decode the bounded squared-DLV state."""
        maximum_raw = np.log(self.maximum_dlvsq - self.minimum_dlvsq)
        return self.minimum_dlvsq + np.exp(
            np.minimum(self.dlv_raw_log_excess_dlvsq, maximum_raw)
        )

    def forward_at_business_dte(self, business_dte: int) -> float:
        """Return log-linearly interpolated ``F/S`` at a synthetic expiry."""
        return interpolate_positive_curve_at_business_dte(
            business_dtes=self.business_dtes,
            values=self.forward_over_spot,
            target_business_dte=business_dte,
            name="forward-over-spot",
        )

    def discount_at_business_dte(self, business_dte: int) -> float:
        """Return the log-linearly interpolated discount factor."""
        return interpolate_positive_curve_at_business_dte(
            business_dtes=self.business_dtes,
            values=self.discount_factors,
            target_business_dte=business_dte,
            name="discount-factor",
        )

    def atm_vol_at_business_dte(self, business_dte: int) -> float:
        """Interpolate the book-selection ATM curve in total variance."""
        if business_dte <= 0 or business_dte > self.business_dtes[-1] + 1.0e-8:
            raise ValueError(
                f"cannot interpolate ATM volatility at {business_dte} business days"
            )
        times = self.business_dtes / DAYS_PER_YEAR
        total_variances = self.book_selection_atm_vols**2 * times
        target_time = business_dte / DAYS_PER_YEAR
        variance = float(
            np.interp(
                target_time,
                np.concatenate(([0.0], times)),
                np.concatenate(([0.0], total_variances)),
            )
        )
        if not np.isfinite(variance) or variance <= 0.0:
            raise FloatingPointError("interpolated ATM total variance is not positive")
        return float(np.sqrt(variance / target_time))

    def hedge_snapshot(
        self,
        *,
        maximum_business_dte: float = 5.0,
        nms_range: tuple[float, float] = (-2.0, 2.0),
        spot_over_episode_origin: float = 1.0,
        episode_origin_date: date | None = None,
    ) -> MarketSnapshot:
        """Return the executable inception universe in episode-origin units."""
        if not np.isfinite(spot_over_episode_origin) or spot_over_episode_origin <= 0.0:
            raise ValueError("spot_over_episode_origin must be finite and positive")
        if episode_origin_date is None:
            episode_origin_date = self.trading_date
        if episode_origin_date > self.trading_date:
            raise ValueError("episode_origin_date cannot follow the snapshot date")
        quotes: list[OptionQuote] = []
        for book_index in np.flatnonzero(self.in_recommended_hedge_nms_population):
            expiry_index = int(self.expiry_indices[book_index])
            bridge = (
                self.discount_factors[expiry_index]
                * self.forward_over_spot[expiry_index]
            )
            mark = float(
                bridge * self.sanos_prices_over_discounted_forward[book_index]
            )
            half_spread = float(self.half_spreads_over_spot[book_index])
            if mark < half_spread:
                continue
            right = OptionRight.CALL if self.is_call[book_index] else OptionRight.PUT
            strike_over_forward = float(self.strikes_over_forward[book_index])
            expiry_date = self.expiry_dates[expiry_index]
            fixed_strike = float(
                spot_over_episode_origin
                * self.forward_over_spot[expiry_index]
                * strike_over_forward
            )
            contract = OptionContract(
                contract_id=(
                    f"synthetic:{episode_origin_date.isoformat()}:"
                    f"{expiry_date.isoformat()}:{right.value}:{fixed_strike:.17g}"
                ),
                expiry_date=expiry_date,
                fixed_strike=fixed_strike,
                right=right,
            )
            quotes.append(
                OptionQuote(
                    trading_date=self.trading_date,
                    contract=contract,
                    business_dte=float(self.business_dtes[expiry_index]),
                    atm_annual_vol=float(self.book_selection_atm_vols[expiry_index]),
                    strike_over_forward=strike_over_forward,
                    standardized_log_moneyness=float(
                        self.standardized_log_moneyness[book_index]
                    ),
                    mark=mark,
                    half_spread=half_spread,
                    is_am_settlement=bool(self.is_am_settlement[expiry_index]),
                )
            )
        selected = eligible_option_quotes(
            tuple(quotes),
            maximum_business_dte=maximum_business_dte,
            nms_range=nms_range,
        )
        return MarketSnapshot(
            trading_date=self.trading_date,
            normalized_spot=spot_over_episode_origin,
            option_quotes=selected,
        )

    def reprice_contracts(
        self,
        contracts: tuple[OptionContract, ...],
        *,
        spot_over_episode_origin: float,
    ) -> MarketSnapshot:
        """Mark fixed episode contracts without changing their identity.

        A contract stores ``K/S_episode``. At a later date its live normalized
        strike is recovered as ``K/F_t = (K/S_episode) / [(S_t/S_episode)(F_t/S_t)]``.
        The fitted SANOS surface supplies the mark and the current quote-side
        cross-section supplies a linearly interpolated half-spread.
        """
        if not np.isfinite(spot_over_episode_origin) or spot_over_episode_origin <= 0.0:
            raise ValueError("spot_over_episode_origin must be finite and positive")
        quotes: list[OptionQuote] = []
        expiry_lookup: dict[date, int] = {}
        for index, expiry_date in enumerate(self.expiry_dates):
            if self.is_am_settlement[index]:
                continue
            if expiry_date in expiry_lookup:
                raise ValueError("public market day contains duplicate PM expiry dates")
            expiry_lookup[expiry_date] = index
        for contract in contracts:
            expiry_index = expiry_lookup.get(contract.expiry_date)
            if contract.expiry_date <= self.trading_date:
                continue
            if expiry_index is not None and self.is_am_settlement[expiry_index]:
                raise ValueError("AM-settled contracts are not supported in public replay")
            business_dte = count_business_days_between(
                self.trading_date,
                contract.expiry_date,
            )
            if business_dte <= 0:
                continue
            forward_over_spot = (
                float(self.forward_over_spot[expiry_index])
                if expiry_index is not None
                else self.forward_at_business_dte(business_dte)
            )
            discount_factor = (
                float(self.discount_factors[expiry_index])
                if expiry_index is not None
                else self.discount_at_business_dte(business_dte)
            )
            atm_annual_vol = (
                float(self.book_selection_atm_vols[expiry_index])
                if expiry_index is not None
                else self.atm_vol_at_business_dte(business_dte)
            )
            current_forward_over_episode_origin = (
                spot_over_episode_origin * forward_over_spot
            )
            strike_over_forward = (
                contract.fixed_strike / current_forward_over_episode_origin
            )
            is_call = contract.right is OptionRight.CALL
            normalized_price = float(
                price_sanos_surface(
                    query_expiries_y=np.asarray([business_dte / DAYS_PER_YEAR]),
                    query_strikes=np.asarray([strike_over_forward]),
                    is_call=np.asarray([is_call]),
                    model_expiries_y=self.sanos_model_expiries_y,
                    atm_variances=self.sanos_kernel_total_variances,
                    support_strikes=self.sanos_support_strikes,
                    densities=self.sanos_densities,
                    smoothness=self.sanos_smoothness,
                    one_minute_y=self.one_minute_y,
                )[0]
            )
            if normalized_price < -_SANOS_PRICE_NEGATIVE_TOLERANCE:
                raise FloatingPointError(
                    f"{self.trading_date}: materially negative SANOS replay price "
                    f"{normalized_price} for {contract.contract_id}"
                )
            normalized_price = max(normalized_price, _SANOS_REPLAY_PRICE_FLOOR)
            mark = float(
                spot_over_episode_origin
                * discount_factor
                * forward_over_spot
                * normalized_price
            )
            half_spread = spot_over_episode_origin * self._interpolate_half_spread_at_dte(
                business_dte=business_dte,
                fixed_strike_over_current_spot=(
                    contract.fixed_strike / spot_over_episode_origin
                ),
                is_call=is_call,
            )
            if mark < half_spread:
                half_spread = mark
            nms = float(
                NMS.to_nms(
                    np.asarray([strike_over_forward]),
                    t=np.asarray([business_dte / DAYS_PER_YEAR]),
                    vol=np.asarray([atm_annual_vol]),
                    eps_vol=self.book_selection_epsilon_atm_vol,
                )[0]
            )
            try:
                quote = OptionQuote(
                    trading_date=self.trading_date,
                    contract=contract,
                    business_dte=float(business_dte),
                    atm_annual_vol=atm_annual_vol,
                    strike_over_forward=float(strike_over_forward),
                    standardized_log_moneyness=nms,
                    mark=mark,
                    half_spread=float(half_spread),
                    is_am_settlement=False,
                )
            except ValueError as error:
                raise ValueError(
                    f"{self.trading_date}: invalid replay quote for "
                    f"{contract.contract_id}; dte={business_dte}, "
                    f"atm_vol={atm_annual_vol}, k_over_f={strike_over_forward}, "
                    f"mark={mark}, half_spread={half_spread}, nms={nms}"
                ) from error
            quotes.append(quote)
        return MarketSnapshot(
            trading_date=self.trading_date,
            normalized_spot=spot_over_episode_origin,
            option_quotes=tuple(quotes),
        )

    def half_spread_for_fixed_contract(
        self,
        *,
        business_dte: int,
        fixed_strike_over_current_spot: float,
        is_call: bool,
    ) -> float:
        """Return the interpolated spot-one spread for a fixed contract."""
        return float(
            self.half_spreads_for_fixed_contracts(
                business_dte=business_dte,
                fixed_strikes_over_current_spot=np.asarray(
                    [fixed_strike_over_current_spot],
                    dtype=np.float64,
                ),
                is_call=is_call,
            )[0]
        )

    def half_spreads_for_fixed_contracts(
        self,
        *,
        business_dte: int,
        fixed_strikes_over_current_spot: NDArray[np.floating],
        is_call: bool,
    ) -> FloatArray:
        """Vectorize the exact fixed-contract spread interpolation over paths."""
        if business_dte <= 0:
            raise ValueError("fixed-contract business DTE must be positive")
        fixed_strikes = np.asarray(
            fixed_strikes_over_current_spot,
            dtype=np.float64,
        )
        if (
            fixed_strikes.ndim != 1
            or np.any(~np.isfinite(fixed_strikes))
            or np.any(fixed_strikes <= 0.0)
        ):
            raise ValueError(
                "fixed strikes over current spot must be a finite-positive vector"
            )
        expiry_geometry: list[tuple[float, int]] = []
        for expiry_index, expiry_business_dte in enumerate(self.business_dtes):
            if self.is_am_settlement[expiry_index]:
                continue
            selection = (self.expiry_indices == expiry_index) & (
                self.is_call == is_call
            )
            if np.any(selection):
                expiry_geometry.append((float(expiry_business_dte), expiry_index))
        if not expiry_geometry:
            side = "call" if is_call else "put"
            raise ValueError(f"{self.trading_date}: no {side} spreads available")
        expiry_geometry.sort(key=lambda item: item[0])
        expiry_dtes = np.asarray(
            [item[0] for item in expiry_geometry],
            dtype=np.float64,
        )
        upper = int(
            np.searchsorted(expiry_dtes, float(business_dte), side="right")
        )
        if upper == 0:
            return self._interpolate_half_spreads(
                expiry_index=expiry_geometry[0][1],
                strike_over_forward=(
                    fixed_strikes
                    / self.forward_over_spot[expiry_geometry[0][1]]
                ),
                is_call=is_call,
            )
        lower = upper - 1
        lower_dte, lower_expiry = expiry_geometry[lower]
        lower_spreads = self._interpolate_half_spreads(
            expiry_index=lower_expiry,
            strike_over_forward=(
                fixed_strikes / self.forward_over_spot[lower_expiry]
            ),
            is_call=is_call,
        )
        if upper == len(expiry_geometry) or lower_dte == business_dte:
            return lower_spreads
        upper_dte, upper_expiry = expiry_geometry[upper]
        upper_spreads = self._interpolate_half_spreads(
            expiry_index=upper_expiry,
            strike_over_forward=(
                fixed_strikes / self.forward_over_spot[upper_expiry]
            ),
            is_call=is_call,
        )
        weight = (business_dte - lower_dte) / (upper_dte - lower_dte)
        return lower_spreads + weight * (upper_spreads - lower_spreads)

    def _interpolate_half_spread_at_dte(
        self,
        *,
        business_dte: int,
        fixed_strike_over_current_spot: float,
        is_call: bool,
    ) -> float:
        """Interpolate a side-specific spread surface across strike and expiry."""
        expiry_dtes: list[float] = []
        spreads: list[float] = []
        for expiry_index, expiry_business_dte in enumerate(self.business_dtes):
            if self.is_am_settlement[expiry_index]:
                continue
            selection = (self.expiry_indices == expiry_index) & (
                self.is_call == is_call
            )
            if not np.any(selection):
                continue
            strike_over_forward = (
                fixed_strike_over_current_spot / self.forward_over_spot[expiry_index]
            )
            expiry_dtes.append(float(expiry_business_dte))
            spreads.append(
                self._interpolate_half_spread(
                    expiry_index=expiry_index,
                    strike_over_forward=float(strike_over_forward),
                    is_call=is_call,
                )
            )
        if not expiry_dtes:
            side = "call" if is_call else "put"
            raise ValueError(f"{self.trading_date}: no {side} spreads available")
        ordering = np.argsort(expiry_dtes, kind="stable")
        return float(
            np.interp(
                float(business_dte),
                np.asarray(expiry_dtes)[ordering],
                np.asarray(spreads)[ordering],
            )
        )

    def _interpolate_half_spread(
        self,
        *,
        expiry_index: int,
        strike_over_forward: float,
        is_call: bool,
    ) -> float:
        return float(
            self._interpolate_half_spreads(
                expiry_index=expiry_index,
                strike_over_forward=np.asarray(
                    [strike_over_forward],
                    dtype=np.float64,
                ),
                is_call=is_call,
            )[0]
        )

    def _interpolate_half_spreads(
        self,
        *,
        expiry_index: int,
        strike_over_forward: FloatArray,
        is_call: bool,
    ) -> FloatArray:
        selection = (self.expiry_indices == expiry_index) & (self.is_call == is_call)
        strikes = self.strikes_over_forward[selection]
        spreads = self.half_spreads_over_spot[selection]
        if len(strikes) == 0:
            side = "call" if is_call else "put"
            raise ValueError(
                f"{self.trading_date}: no {side} spread books for expiry index {expiry_index}"
            )
        ordering = np.argsort(strikes, kind="stable")
        return np.asarray(
            np.interp(
                np.asarray(strike_over_forward, dtype=np.float64),
                strikes[ordering],
                spreads[ordering],
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True, kw_only=True)
class OneDayTransition:
    """One consecutive retained session in origin-forward units."""

    origin: PublicMarketDay
    next_date: date
    forward_adjusted_spot_return: float
    discount_factor: float
    forward_over_spot: float

    def __post_init__(self) -> None:
        values = (
            self.forward_adjusted_spot_return,
            self.discount_factor,
            self.forward_over_spot,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("one-day transition values must be finite")
        if self.forward_adjusted_spot_return <= -1.0:
            raise ValueError("forward-adjusted spot return must exceed minus one")
        if self.discount_factor <= 0.0 or self.forward_over_spot <= 0.0:
            raise ValueError("discount factor and forward-over-spot must be positive")

    @property
    def terminal_normalized_spot(self) -> float:
        """Return ``S(t+1)/F_t(t+1)`` after deterministic carry removal."""
        return 1.0 + self.forward_adjusted_spot_return

    @property
    def discounted_forward_over_spot(self) -> float:
        """Return the origin spot-one bridge ``DF*F/S``."""
        return self.discount_factor * self.forward_over_spot

    @property
    def discounted_spot_gain(self) -> float:
        """Return the present-value gain of one normalized spot-forward hedge."""
        return (
            self.discounted_forward_over_spot
            * self.forward_adjusted_spot_return
        )


class PublicHedgingPanel(PublicHedgingPanelView):
    """Aligned lazy row view of the complete public-candidate panel."""

    def __init__(self, directory: Path, *, validate: bool = False) -> None:
        self.dataset = PublicHedgingDataset(Path(directory))
        if validate:
            validate_public_dataset(self.dataset.directory)
        self.geometry = self.dataset.geometry
        self.dataset_manifest_sha256 = self.dataset.manifest_sha256
        self.observations = self.dataset.read("observations.parquet")
        self.expiries = self.dataset.read("expiries.parquet")
        self.instruments = self.dataset.read("instruments.parquet")
        self.sanos_geometry = self.dataset.read("sanos_geometry.parquet")
        self.sanos_state = self.dataset.read("sanos_state.parquet")
        self.dlv_geometry = self.dataset.read("dlv_geometry.parquet")
        self.dlv_state = self.dataset.read("dlv_state.parquet")
        self.dates = tuple(self.observations["date"].to_list())
        self._date_indices = {value: index for index, value in enumerate(self.dates)}
        if len(self._date_indices) != len(self.dates):
            raise ValueError("public panel dates must be unique")

    def market_day(self, trading_date: date) -> PublicMarketDay:
        """Return one date's public model and quote-side state."""
        try:
            index = self._date_indices[trading_date]
        except KeyError as error:
            raise KeyError(f"public panel does not contain {trading_date}") from error
        return PublicMarketDay.from_rows(
            expiry_row=self.expiries.row(index, named=True),
            instrument_row=self.instruments.row(index, named=True),
            sanos_geometry_row=self.sanos_geometry.row(index, named=True),
            sanos_state_row=self.sanos_state.row(index, named=True),
            dlv_geometry_row=self.dlv_geometry.row(index, named=True),
            dlv_state_row=self.dlv_state.row(index, named=True),
            geometry=self.geometry,
        )

    def one_day_transition(self, origin_date: date) -> OneDayTransition:
        """Return the next retained one-session transition without look-ahead fitting."""
        try:
            origin_index = self._date_indices[origin_date]
        except KeyError as error:
            raise KeyError(f"public panel does not contain {origin_date}") from error
        if origin_index + 1 >= len(self.dates):
            raise ValueError(f"{origin_date} has no following retained observation")
        next_row = self.observations.row(origin_index + 1, named=True)
        if next_row["business_sessions_from_previous_observation"] != 1:
            raise ValueError(f"{origin_date} is followed by a retained-date gap")
        origin_day = self.market_day(origin_date)
        discount_factor = next_row["discount_factor_from_previous_observation"]
        forward_over_spot = next_row["forward_carry_from_previous_observation"]
        if discount_factor is None or forward_over_spot is None:
            raise ValueError(f"{origin_date} has no stored one-session carry")
        return OneDayTransition(
            origin=origin_day,
            next_date=next_row["date"],
            forward_adjusted_spot_return=float(
                next_row["forward_adjusted_spot_return_into_date"]
            ),
            discount_factor=float(discount_factor),
            forward_over_spot=float(forward_over_spot),
        )

    def next_one_session_date(self, origin_date: date) -> date:
        """Return the next retained date when it is exactly one session ahead."""
        try:
            origin_index = self._date_indices[origin_date]
        except KeyError as error:
            raise KeyError(f"public panel does not contain {origin_date}") from error
        if origin_index + 1 >= len(self.dates):
            raise ValueError(f"{origin_date} has no following retained observation")
        next_row = self.observations.row(origin_index + 1, named=True)
        if next_row["business_sessions_from_previous_observation"] != 1:
            raise ValueError(f"{origin_date} is followed by a retained-date gap")
        return next_row["date"]

    def prior_one_day_origins(
        self,
        origin_date: date,
        *,
        count: int,
    ) -> tuple[date, ...]:
        """Return the last ``count`` gap-free origins strictly before a date."""
        if count <= 0:
            raise ValueError("count must be positive")
        try:
            origin_index = self._date_indices[origin_date]
        except KeyError as error:
            raise KeyError(f"public panel does not contain {origin_date}") from error
        eligible: list[date] = []
        for index in range(origin_index - 1, -1, -1):
            next_row = self.observations.row(index + 1, named=True)
            if next_row["business_sessions_from_previous_observation"] == 1:
                eligible.append(self.dates[index])
                if len(eligible) == count:
                    break
        if len(eligible) != count:
            raise ValueError(
                f"{origin_date} has only {len(eligible)} prior one-day origins, "
                f"but {count} were requested"
            )
        return tuple(reversed(eligible))
