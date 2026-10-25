"""Common benchmark adapter for the SANOS-LV reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
    DailyDigitalTarget,
    build_daily_digital_sample,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    PublicMarketDay,
)
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    RollingDlvFactorFit,
    fit_rolling_dlv_factors,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.config import SanosLvConfig
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    dlv_digital_greeks,
    dlv_option_greeks,
    solve_delta_factor_hedge,
)


def run_sanos_lv_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    config: SanosLvConfig = SanosLvConfig(),
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    """Run the deterministic one-day ATM-call SANOS-LV reference."""
    require_public_hedging_panel(panel)
    target = DailyDigitalTarget(
        maturity_business_days=1,
        right=OptionRight.CALL,
        nms=0.0,
    )
    factors = fit_rolling_dlv_factors(
        panel,
        as_of_date=origin_date,
        config=config.rolling_factors,
    )
    return _run_sanos_lv_target(
        panel,
        origin_date=origin_date,
        target=target,
        factors=factors,
        config=config,
        claim_quantity=claim_quantity,
    )


def run_sanos_lv_daily_book(
    panel: PublicHedgingPanel,
    *,
    target_maturity: int | None = None,
    origin_date: date,
    config: SanosLvConfig = SanosLvConfig(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Run all 20 agreed daily digitals with one contemporaneous factor fit."""
    require_public_hedging_panel(panel)
    factors = fit_rolling_dlv_factors(
        panel,
        as_of_date=origin_date,
        config=config.rolling_factors,
    )
    targets = daily_digital_targets(maturity_business_days=target_maturity)
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    samples = tuple(
        build_daily_digital_sample(
            panel,
            origin_date=origin_date,
            target=target,
        )
        for target in targets
    )
    pricing_day = _daily_pricing_day(samples[0].transition.origin, config=config)
    hedge_greeks = dlv_option_greeks(
        pricing_day,
        quotes=samples[0].hedge_quotes,
        factors=factors,
        relative_spot_step=config.relative_spot_step,
    )
    all_target_greeks = dlv_digital_greeks(
        pricing_day,
        business_dtes=np.asarray(
            [float(target.maturity_business_days) for target in targets]
        ),
        strikes_over_forward=np.asarray(
            [sample.target_strike_over_forward for sample in samples]
        ),
        is_call=np.asarray(
            [target.right is OptionRight.CALL for target in targets]
        ),
        discount_factors=np.asarray(
            [sample.target_discount_factor for sample in samples]
        ),
        factors=factors,
        relative_spot_step=config.relative_spot_step,
    )
    return tuple(
        _evaluate_sanos_lv_target(
            sample,
            target_greeks=DlvGreeks(
                model_prices=(
                    quantities[index]
                    * all_target_greeks.model_prices[index : index + 1]
                ),
                spot_deltas=(
                    quantities[index]
                    * all_target_greeks.spot_deltas[index : index + 1]
                ),
                factor_vegas=(
                    quantities[index]
                    * all_target_greeks.factor_vegas[index : index + 1]
                ),
            ),
            unit_model_price=float(all_target_greeks.model_prices[index]),
            hedge_greeks=hedge_greeks,
            factors=factors,
            config=config,
            claim_quantity=float(quantities[index]),
        )
        for index, sample in enumerate(samples)
    )


def _run_sanos_lv_target(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target: DailyDigitalTarget,
    factors: RollingDlvFactorFit,
    config: SanosLvConfig,
    claim_quantity: float,
) -> HedgeSmokeResult:
    sample = build_daily_digital_sample(
        panel,
        origin_date=origin_date,
        target=target,
    )
    pricing_day = _daily_pricing_day(sample.transition.origin, config=config)
    target_greeks = dlv_digital_greeks(
        pricing_day,
        business_dtes=np.asarray([float(target.maturity_business_days)]),
        strikes_over_forward=np.asarray([sample.target_strike_over_forward]),
        is_call=np.asarray([target.right is OptionRight.CALL]),
        discount_factors=np.asarray([sample.target_discount_factor]),
        factors=factors,
        relative_spot_step=config.relative_spot_step,
    )
    hedge_greeks = dlv_option_greeks(
        pricing_day,
        quotes=sample.hedge_quotes,
        factors=factors,
        relative_spot_step=config.relative_spot_step,
    )
    scaled_target_greeks = DlvGreeks(
        model_prices=claim_quantity * target_greeks.model_prices,
        spot_deltas=claim_quantity * target_greeks.spot_deltas,
        factor_vegas=claim_quantity * target_greeks.factor_vegas,
    )
    return _evaluate_sanos_lv_target(
        sample,
        target_greeks=scaled_target_greeks,
        unit_model_price=float(target_greeks.model_prices[0]),
        hedge_greeks=hedge_greeks,
        factors=factors,
        config=config,
        claim_quantity=claim_quantity,
    )


def _daily_pricing_day(
    day: PublicMarketDay,
    *,
    config: SanosLvConfig,
) -> PublicMarketDay:
    if config.daily_backbone_annual_volatility is None:
        return day
    return replace(
        day,
        backbone_annual_volatility=config.daily_backbone_annual_volatility,
    )


def _evaluate_sanos_lv_target(
    sample: DailyDigitalSample,
    *,
    target_greeks: DlvGreeks,
    unit_model_price: float,
    hedge_greeks: DlvGreeks,
    factors: RollingDlvFactorFit,
    config: SanosLvConfig,
    claim_quantity: float,
) -> HedgeSmokeResult:
    action = solve_delta_factor_hedge(
        target=target_greeks,
        hedge_options=hedge_greeks,
        option_cost_weights=np.asarray(
            [
                quote.half_spread
                for quote in sample.hedge_quotes
            ]
        ),
        residual_tolerance=config.exposure_residual_tolerance,
        position_cap=config.position_cap,
    )
    return evaluate_daily_digital_hedge(
        sample,
        model="sanos-lv",
        measure="Q-price/historical-transition",
        spot_action=action.spot_action,
        option_actions=action.option_actions,
        model_price=unit_model_price,
        objective_value=float(np.linalg.norm(action.factor_residual)),
        converged=action.cap_hits == 0,
        cap_hits=action.cap_hits,
        claim_quantity=claim_quantity,
        diagnostics={
            "factor_rank": factors.factor_rank,
            "factor_basis_semantics": (
                "one front-spot-projected loading plus seven orthogonal PCs"
            ),
            "factor_explained_variance_fraction": (
                factors.explained_variance_fraction
            ),
            "factor_explained_variance_semantics": (
                "fraction of smoothed DLV innovation energy"
            ),
            "factor_residual_norm": float(np.linalg.norm(action.factor_residual)),
            "delta_residual": action.delta_residual,
            "factor_training_start": factors.training_origin_dates[0].isoformat(),
            "factor_training_end": factors.training_endpoint_dates[-1].isoformat(),
            "strike_interpolation_clip_fraction": (
                factors.interpolation_clip_fraction
            ),
            "forward_tail_extrapolation_fraction": (
                factors.forward_tail_extrapolation_fraction
            ),
            "hedge_instrument_count": len(sample.hedge_quotes),
            "claim_quantity": claim_quantity,
            "objective_semantics": (
                "minimum predictable-option-cost exact factor hedge, then spot delta"
            ),
            "optimization_cost_semantics": (
                "one current half-spread for next-date settlement; twice the "
                "current half-spread is the predictable round-trip proxy otherwise"
            ),
            "payoff_units": "discounted origin-PV per unit origin spot",
            "terminal_value_semantics": (
                "settlement for 1D targets; discounted next-date market value for "
                "unfinished 3D targets"
            ),
        },
    )
