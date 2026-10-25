"""Public-data daily adapter for the SANOS parametric hedge."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np

from quantdata.hedging.contracts import OptionRight
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.data.rolling_factors import (
    fit_rolling_dlv_factors,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    dlv_digital_greeks,
    dlv_option_greeks,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_parametric.config import (
    SanosParametricConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_parametric.hedger import (
    solve_parametric_hedge,
)


def run_sanos_parametric_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    target_maturity: int | None = None,
    config: SanosParametricConfig = SanosParametricConfig(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit one causal spot/factor covariance and hedge the daily digitals."""
    require_public_hedging_panel(panel)
    targets = daily_digital_targets(maturity_business_days=target_maturity)
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    if any(sample.hedge_quotes != samples[0].hedge_quotes for sample in samples[1:]):
        raise RuntimeError("SANOS parametric targets do not share a hedge book")
    factors = fit_rolling_dlv_factors(
        panel,
        as_of_date=origin_date,
        config=config.greeks.rolling_factors,
    )
    spot_returns = np.asarray(
        [
            panel.one_day_transition(training_date).discounted_spot_gain
            for training_date in factors.training_origin_dates
        ],
        dtype=np.float64,
    )
    if any(
        panel.one_day_transition(training_date).next_date != endpoint
        for training_date, endpoint in zip(
            factors.training_origin_dates, factors.training_endpoint_dates, strict=True
        )
    ):
        raise RuntimeError("spot and DLV factor training transitions differ")
    shocks = np.column_stack((spot_returns, factors.factor_scores))
    covariance = np.cov(shocks, rowvar=False, ddof=1)
    pricing_day = samples[0].transition.origin
    option_greeks = dlv_option_greeks(
        pricing_day,
        quotes=samples[0].hedge_quotes,
        factors=factors,
        relative_spot_step=config.greeks.relative_spot_step,
    )
    target_greeks = dlv_digital_greeks(
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
        relative_spot_step=config.greeks.relative_spot_step,
    )
    option_costs = np.asarray(
        [
            quote.half_spread
            for quote in samples[0].hedge_quotes
        ],
        dtype=np.float64,
    )
    results: list[HedgeSmokeResult] = []
    for index, sample in enumerate(samples):
        scaled_target = DlvGreeks(
            model_prices=quantities[index] * target_greeks.model_prices[index : index + 1],
            spot_deltas=quantities[index] * target_greeks.spot_deltas[index : index + 1],
            factor_vegas=quantities[index] * target_greeks.factor_vegas[index : index + 1],
        )
        action = solve_parametric_hedge(
            target=scaled_target,
            options=option_greeks,
            shock_covariance=covariance,
            option_cost_weights=option_costs,
            spot_cost_weight=config.spot_entry_cost,
            risk_parameter=config.risk_parameter,
            position_cap=config.greeks.position_cap,
        )
        results.append(
            evaluate_daily_digital_hedge(
                sample,
                model="sanos-parametric",
                measure="historical-P-factor-covariance",
                spot_action=action.spot_action,
                option_actions=action.option_actions,
                model_price=float(target_greeks.model_prices[index]),
                objective_value=(
                    0.5 * config.risk_parameter * action.risk_variance
                    + action.transaction_cost_proxy
                ),
                converged=action.cap_hits == 0,
                cap_hits=action.cap_hits,
                claim_quantity=float(quantities[index]),
                diagnostics={
                    "factor_rank": factors.factor_rank,
                    "factor_training_start": factors.training_origin_dates[0].isoformat(),
                    "factor_training_end": factors.training_endpoint_dates[-1].isoformat(),
                    "factor_covariance_observations": len(factors.training_origin_dates),
                    "risk_parameter": config.risk_parameter,
                    "spot_entry_cost": config.spot_entry_cost,
                    "risk_variance": action.risk_variance,
                    "optimization_cost_proxy": action.transaction_cost_proxy,
                    "solver_status": action.solver_status,
                    "objective_semantics": "mean-variance of SANOS spot/factor Greeks plus executable cost",
                },
            )
        )
    return tuple(results)
