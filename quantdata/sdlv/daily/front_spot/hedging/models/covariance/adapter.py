"""Common benchmark adapter for the independent covariance hedge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date

import numpy as np

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
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.config import (
    CovarianceHedgeConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.hedger import (
    fit_covariance_book_actions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    build_historical_return_panel,
)


def run_covariance_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    config: CovarianceHedgeConfig = CovarianceHedgeConfig(),
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    """Return the ATM one-day call from the full rolling covariance book."""
    require_public_hedging_panel(panel)
    results = run_covariance_daily_book(
        panel,
        origin_date=origin_date,
        config=config,
        claim_quantities={
            target.label: (
                claim_quantity if target.label == "digital-call-1d-atm" else 1.0
            )
            for target in daily_digital_targets()
        },
    )
    return next(result for result in results if result.claim == "digital-call-1d-atm")


def run_covariance_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    config: CovarianceHedgeConfig = CovarianceHedgeConfig(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit one robust same-coordinate panel and evaluate all daily digitals."""
    require_public_hedging_panel(panel)
    targets = daily_digital_targets()
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    samples = tuple(
        build_daily_digital_sample(
            panel,
            origin_date=origin_date,
            target=target,
        )
        for target in targets
    )
    hedge_quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != hedge_quotes for sample in samples[1:]):
        raise RuntimeError("daily covariance targets do not share a hedge universe")
    returns = build_historical_return_panel(
        panel,
        origin_date=origin_date,
        option_quotes=hedge_quotes,
        targets=targets,
        count=config.rolling_observations,
    )
    costs = np.asarray(
        [
            2.0e-4,
            *(
                quote.half_spread
                * (
                    1.0
                    if quote.contract.expiry_date == samples[0].transition.next_date
                    else 2.0
                )
                for quote in hedge_quotes
            ),
        ],
        dtype=np.float64,
    )
    fit = fit_covariance_book_actions(
        replace(
            returns,
            target_returns=returns.target_returns * quantities[None, :],
        ),
        linear_transaction_costs=costs,
        config=config,
    )
    results: list[HedgeSmokeResult] = []
    for sample, action, claim_quantity in zip(
        samples,
        fit.actions,
        quantities,
        strict=True,
    ):
        if action.action is None:
            raise RuntimeError(f"{sample.target.label}: covariance action is missing")
        final_action = action.action
        claim_gross = float(np.sum(np.abs(action.claim_policy.action)))
        zero_gross = float(np.sum(np.abs(action.zero_policy.action)))
        pure_gross = float(np.sum(np.abs(final_action)))
        results.append(
            evaluate_daily_digital_hedge(
                sample,
                model="robust-covariance",
                measure="historical-P",
                spot_action=float(final_action[0]),
                option_actions=final_action[1:],
                model_price=sample.market_price,
                objective_value=action.objective,
                converged=action.converged and fit.estimate.converged,
                cap_hits=action.cap_hits,
                claim_quantity=float(claim_quantity),
                diagnostics={
                    "training_observations": len(returns.training_origin_dates),
                    "training_start": returns.training_origin_dates[0].isoformat(),
                    "training_end": returns.training_origin_dates[-1].isoformat(),
                    "training_last_endpoint": (
                        returns.training_terminal_dates[-1].isoformat()
                    ),
                    "instrument_count": fit.instrument_count,
                    "target_count": fit.target_count,
                    "claim_quantity": float(claim_quantity),
                    "tyler_shrinkage": fit.estimate.shrinkage,
                    "tyler_iterations": fit.estimate.iterations,
                    "mean_z_threshold": fit.mean_z_threshold,
                    "mean_hac_lags": fit.mean_hac_lags,
                    "retained_mean_directions": fit.retained_mean_directions,
                    "maximum_absolute_mean_t": fit.maximum_absolute_mean_t,
                    "raw_instrument_center_norm": fit.raw_instrument_center_norm,
                    "screened_instrument_center_norm": (
                        fit.screened_instrument_center_norm
                    ),
                    "objective": config.objective.value,
                    "risk_parameter": config.risk_parameter,
                    "bounded": action.bounded,
                    "recession_ratio": action.zero_policy.recession_ratio,
                    "expected_profit": action.expected_profit,
                    "robust_standard_deviation": action.standard_deviation,
                    "claim_policy_iterations": action.claim_policy.iterations,
                    "zero_policy_iterations": action.zero_policy.iterations,
                    "claim_policy_gross_exposure": claim_gross,
                    "zero_policy_gross_exposure": zero_gross,
                    "pure_hedge_gross_exposure": pure_gross,
                    "pure_hedge_l2_exposure": float(np.linalg.norm(final_action)),
                    "maximum_absolute_action": float(np.max(np.abs(final_action))),
                    "internal_solver_position_cap": config.solver_position_cap,
                    "executable_position_cap": config.position_cap,
                    "internal_solver_cap_hits": action.internal_cap_hits,
                    "executable_cap_hits": action.executable_cap_hits,
                    "claim_zero_cancellation_ratio": (
                        pure_gross / max(claim_gross + zero_gross, 1.0e-300)
                    ),
                    "nonzero_option_actions": int(
                        np.count_nonzero(np.abs(final_action[1:]) > 1.0e-14)
                    ),
                    "coordinate_semantics": (
                        "same initial business DTE, NMS, and option side on every "
                        "historical origin"
                    ),
                    "price_semantics": (
                        "historical returns reconstructed from released DLV states; "
                        "current execution and realized accounting use common SANOS "
                        "market marks; no generated scenarios"
                    ),
                    "policy_semantics": "claim policy minus zero-claim policy",
                    "mean_semantics": (
                        "sample instrument means soft-screened using Newey-West "
                        "errors of covariance eigenportfolio returns"
                    ),
                    "scatter_semantics": (
                        "instrument-only Tyler scatter with independent fixed-weight "
                        "target cross-scatter blocks"
                    ),
                    "optimization_cost_semantics": (
                        "two basis points for spot; one current half-spread for "
                        "next-date settlement; twice current half-spread otherwise"
                    ),
                    "terminal_value_semantics": (
                        "settlement for 1D targets; discounted next-date market "
                        "value for unfinished 3D targets"
                    ),
                    "payoff_units": "discounted origin-PV per unit origin spot",
                },
            )
        )
    return tuple(results)
