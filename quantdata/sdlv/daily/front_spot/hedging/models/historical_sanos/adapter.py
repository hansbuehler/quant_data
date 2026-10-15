"""Benchmark adapter for the empirical historical SANOS hedge."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
    DailyDigitalTarget,
    build_daily_digital_sample,
    daily_claim_quantity_vector,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.data.public_only import (
    require_public_hedging_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.covariance.returns import (
    HistoricalReturnPanel,
    build_historical_return_panel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.config import (
    HistoricalSanosConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.historical_sanos.hedger import (
    fit_historical_sanos_hedges,
)


def run_historical_sanos_one_day(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
    target: DailyDigitalTarget | None = None,
    claim_quantity: float = 1.0,
) -> HedgeSmokeResult:
    """Fit and evaluate one public-data historical SANOS hedge."""
    require_public_hedging_panel(panel)
    if target is None:
        return next(
            result
            for result in run_historical_sanos_daily_book(
                panel,
                origin_date=origin_date,
                config=config,
                claim_quantities={
                    target.label: (
                        claim_quantity
                        if target.label == "digital-call-1d-atm"
                        else 1.0
                    )
                    for target in daily_digital_targets()
                },
            )
            if result.claim == "digital-call-1d-atm"
        )
    return _run_historical_sanos_targets(
        panel,
        origin_date=origin_date,
        targets=(target,),
        config=config,
        claim_quantities={target.label: claim_quantity},
    )[0]


def run_historical_sanos_daily_book(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
    claim_quantities: Mapping[str, float] | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit one rolling historical panel and evaluate all daily digitals."""
    require_public_hedging_panel(panel)
    return _run_historical_sanos_targets(
        panel,
        origin_date=origin_date,
        targets=daily_digital_targets(),
        config=config,
        claim_quantities=claim_quantities,
    )


def _run_historical_sanos_targets(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    targets: tuple[DailyDigitalTarget, ...],
    config: HistoricalSanosConfig,
    claim_quantities: Mapping[str, float] | None,
) -> tuple[HedgeSmokeResult, ...]:
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
        raise RuntimeError("historical SANOS targets do not share a hedge universe")
    returns = build_historical_return_panel(
        panel,
        origin_date=origin_date,
        option_quotes=hedge_quotes,
        targets=targets,
        count=config.rolling_observations,
    )
    quantities = daily_claim_quantity_vector(targets, claim_quantities)
    return evaluate_historical_sanos_daily_book(
        samples,
        returns=returns,
        config=config,
        claim_quantities=quantities,
    )


def evaluate_historical_sanos_daily_book(
    samples: tuple[DailyDigitalSample, ...],
    *,
    returns: HistoricalReturnPanel,
    config: HistoricalSanosConfig = HistoricalSanosConfig(),
    claim_quantities: np.ndarray | None = None,
) -> tuple[HedgeSmokeResult, ...]:
    """Fit and evaluate aligned daily targets from one historical return panel."""
    if not samples:
        raise ValueError("historical SANOS daily book requires at least one target")
    hedge_quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != hedge_quotes for sample in samples[1:]):
        raise ValueError("historical SANOS samples do not share a hedge universe")
    target_labels = tuple(sample.target.label for sample in samples)
    if returns.target_labels != target_labels:
        raise ValueError("historical SANOS return targets do not align with samples")
    expected_instrument_ids = (
        "spot",
        *(quote.contract.contract_id for quote in hedge_quotes),
    )
    if returns.instrument_ids != expected_instrument_ids:
        raise ValueError("historical SANOS return instruments do not align with samples")
    quantities = (
        np.ones(len(samples), dtype=np.float64)
        if claim_quantities is None
        else np.asarray(claim_quantities, dtype=np.float64)
    )
    if quantities.shape != (len(samples),) or np.any(
        ~np.isfinite(quantities) | (quantities <= 0.0)
    ):
        raise ValueError("historical SANOS claim quantities do not align")
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
    fits = fit_historical_sanos_hedges(
        instrument_returns=returns.instrument_returns,
        claim_returns=returns.target_returns * quantities[None, :],
        linear_transaction_costs=costs,
        config=config,
    )
    observation_parameter_ratio = (
        len(returns.training_origin_dates) / returns.instrument_returns.shape[1]
    )
    results: list[HedgeSmokeResult] = []
    for sample, claim_quantity, fit in zip(
        samples,
        quantities,
        fits,
        strict=True,
    ):
        action = fit.action
        results.append(
            evaluate_daily_digital_hedge(
                sample,
                model="historical-sanos",
                measure="historical-P",
                spot_action=float(action[0]),
                option_actions=action[1:],
                model_price=sample.market_price,
                objective_value=fit.objective,
                converged=fit.converged,
                cap_hits=fit.cap_hits,
                claim_quantity=float(claim_quantity),
                diagnostics={
                    "training_observations": len(returns.training_origin_dates),
                    "training_start": returns.training_origin_dates[0].isoformat(),
                    "training_end": returns.training_origin_dates[-1].isoformat(),
                    "training_last_endpoint": (
                        returns.training_terminal_dates[-1].isoformat()
                    ),
                    "instrument_count": returns.instrument_returns.shape[1],
                    "observation_parameter_ratio": observation_parameter_ratio,
                    "statistically_thin": (
                        observation_parameter_ratio < config.thin_sample_ratio
                    ),
                    "target_count": returns.target_returns.shape[1],
                    "claim_quantity": float(claim_quantity),
                    "objective": config.objective.value,
                    "risk_parameter": config.risk_parameter,
                    "quadratic_ridge": config.quadratic_ridge,
                    "regularization_coordinates": "unit-rms instrument gains",
                    "certificate_tolerance": config.certificate_tolerance,
                    "claim_training_loss": fit.claim_policy.training_loss,
                    "claim_validation_loss": fit.claim_policy.validation_loss,
                    "zero_training_loss": fit.zero_policy.training_loss,
                    "zero_validation_loss": fit.zero_policy.validation_loss,
                    "claim_policy_iterations": fit.claim_policy.iterations,
                    "zero_policy_iterations": fit.zero_policy.iterations,
                    "claim_policy_selection_converged": (
                        fit.claim_policy.selection_converged
                    ),
                    "zero_policy_selection_converged": (
                        fit.zero_policy.selection_converged
                    ),
                    "claim_policy_final_converged": (
                        fit.claim_policy.final_converged
                    ),
                    "zero_policy_final_converged": (
                        fit.zero_policy.final_converged
                    ),
                    "claim_policy_selection_solver_status": (
                        fit.claim_policy.selection_solver_status
                    ),
                    "zero_policy_selection_solver_status": (
                        fit.zero_policy.selection_solver_status
                    ),
                    "claim_policy_solver_status": fit.claim_policy.solver_status,
                    "zero_policy_solver_status": fit.zero_policy.solver_status,
                    "claim_policy_selection_recession_direction": (
                        fit.claim_policy.selection_recession_direction
                    ),
                    "zero_policy_selection_recession_direction": (
                        fit.zero_policy.selection_recession_direction
                    ),
                    "claim_policy_recession_direction": (
                        fit.claim_policy.recession_direction
                    ),
                    "zero_policy_recession_direction": (
                        fit.zero_policy.recession_direction
                    ),
                    "claim_policy_selection_maximum_optimality_residual": (
                        fit.claim_policy.selection_maximum_absolute_gradient
                    ),
                    "zero_policy_selection_maximum_optimality_residual": (
                        fit.zero_policy.selection_maximum_absolute_gradient
                    ),
                    "claim_policy_maximum_optimality_residual": (
                        fit.claim_policy.maximum_absolute_gradient
                    ),
                    "zero_policy_maximum_optimality_residual": (
                        fit.zero_policy.maximum_absolute_gradient
                    ),
                    "internal_solver_position_cap": config.solver_position_cap,
                    "executable_position_cap": config.position_cap,
                    "claim_policy_selection_cap_hits": (
                        fit.claim_policy.selection_cap_hits
                    ),
                    "zero_policy_selection_cap_hits": (
                        fit.zero_policy.selection_cap_hits
                    ),
                    "internal_solver_cap_hits": fit.internal_cap_hits,
                    "executable_cap_hits": fit.executable_cap_hits,
                    "claim_policy_gross_exposure": float(
                        np.sum(np.abs(fit.claim_policy.action))
                    ),
                    "zero_policy_gross_exposure": float(
                        np.sum(np.abs(fit.zero_policy.action))
                    ),
                    "pure_hedge_gross_exposure": float(np.sum(np.abs(action))),
                    "policy_semantics": "claim policy minus zero-claim policy",
                    "scenario_semantics": (
                        "same relative contracts repriced through historical "
                        "released SANOS states"
                    ),
                    "price_semantics": "common released SANOS market mark",
                    "payoff_units": "discounted origin-PV per unit origin spot",
                },
            )
        )
    return tuple(results)
