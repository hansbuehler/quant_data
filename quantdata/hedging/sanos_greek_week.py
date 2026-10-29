"""Matched SANOS Greek and parametric hedges for daily cohorts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.hedging.data.build_sanos_derived_grid import GEOMETRY, _decoder
from quantdata.hedging.data.rolling_sanos_factors import (
    RollingSanosFactors,
    SanosFactorWindow,
    fit_rolling_sanos_factors,
)
from quantdata.hedging.data.sanos_state_greeks import (
    SanosStateGreeks,
    finite_difference_sanos_state_greeks,
)
from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    DailyDigitalSample,
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
)
from quantdata.sdlv.daily.front_spot.hedging.models.bs_greeks.greeks import (
    black_digital_greeks,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import (
    DlvGreeks,
    solve_delta_factor_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_parametric.hedger import (
    solve_parametric_hedge,
)

PILOT_DATES = tuple(date.fromisoformat(value) for value in (
    "2024-04-29", "2024-04-30", "2024-05-01", "2024-05-02", "2024-05-03"
))


@lru_cache(maxsize=512)
def _load_raw_state(directory: Path, trading_date: date) -> np.ndarray:
    path = directory / f"{trading_date.isoformat()}.npz"
    with np.load(path, allow_pickle=False) as stored:
        saved_date = date.fromisoformat(str(stored["date"].item()))
        state = np.asarray(stored["raw_u"], dtype=np.float64)
    if saved_date != trading_date or state.shape != (GEOMETRY.native_state_size,):
        raise ValueError(f"Invalid SANOS-derived state at {trading_date}")
    return state


def _rolling_window(
    panel: PublicHedgingPanel,
    *,
    state_directory: Path,
    origin_date: date,
) -> SanosFactorWindow:
    origin_index = panel.dates.index(origin_date)
    rows = panel.observations.to_dicts()
    endpoints = [
        index
        for index in range(1, origin_index + 1)
        if rows[index]["business_sessions_from_previous_observation"] == 1
    ][-252:]
    if len(endpoints) != 252:
        raise ValueError(f"{origin_date}: fewer than 252 exact one-session transitions")
    changes = np.stack([
        _load_raw_state(state_directory, panel.dates[index])
        - _load_raw_state(state_directory, panel.dates[index - 1])
        for index in endpoints
    ])
    returns = np.asarray(
        [rows[index]["forward_adjusted_spot_return_into_date"] for index in endpoints],
        dtype=np.float64,
    )
    return SanosFactorWindow(
        origin_dates=tuple(panel.dates[index - 1] for index in endpoints),
        endpoint_dates=tuple(panel.dates[index] for index in endpoints),
        state_changes=changes,
        spot_returns=returns,
    )


def _price_function(samples: tuple[DailyDigitalSample, ...]):
    origin = samples[0].transition.origin
    quotes = samples[0].hedge_quotes
    if any(sample.hedge_quotes != quotes for sample in samples[1:]):
        raise ValueError("Daily targets do not share one hedge book")
    expiry_index = {expiry: index for index, expiry in enumerate(origin.expiry_dates)}
    option_times = torch.as_tensor(
        [quote.business_dte / DAYS_PER_YEAR for quote in quotes], dtype=torch.float64
    )
    option_strikes = torch.as_tensor(
        [quote.strike_over_forward for quote in quotes], dtype=torch.float64
    )
    option_is_call = np.asarray(
        [quote.contract.right.value == "call" for quote in quotes], dtype=bool
    )
    option_bridges = np.asarray([
        origin.discount_factors[expiry_index[quote.contract.expiry_date]]
        * origin.forward_over_spot[expiry_index[quote.contract.expiry_date]]
        for quote in quotes
    ])
    digital_times = torch.as_tensor(
        [sample.target.maturity_business_days / DAYS_PER_YEAR for sample in samples],
        dtype=torch.float64,
    )
    digital_strikes = torch.as_tensor(
        [sample.target_strike_over_forward for sample in samples],
        dtype=torch.float64,
    )
    digital_is_call = np.asarray(
        [sample.target.right.value == "call" for sample in samples], dtype=bool
    )
    digital_discounts = np.asarray(
        [sample.target_discount_factor for sample in samples], dtype=np.float64
    )
    decoder = _decoder()

    def price(spot: float, raw_state: np.ndarray) -> np.ndarray:
        raw = torch.as_tensor(raw_state, dtype=torch.float64)
        native = bound_native_raw(raw, geometry=GEOMETRY)
        surface = decoder.forward(
            dlvsqs=native.dlvsqs,
            forward_vols=native.forward_vols,
            eta=native.eta,
        )
        relative_option_strikes = option_strikes / spot
        call_prices = decoder.price_calls(
            surface, expiries_y=option_times, strikes=relative_option_strikes
        ).detach().numpy()
        strikes = relative_option_strikes.detach().numpy()
        option_prices = np.where(option_is_call, call_prices, call_prices - 1.0 + strikes)
        option_prices *= option_bridges * spot

        relative_digital_strikes = (digital_strikes / spot).detach().requires_grad_(True)
        calls = decoder.price_calls(
            surface, expiries_y=digital_times, strikes=relative_digital_strikes
        )
        call_probabilities = -torch.autograd.grad(
            calls.sum(), relative_digital_strikes
        )[0].detach().numpy()
        digital_prices = digital_discounts * np.where(
            digital_is_call, call_probabilities, 1.0 - call_probabilities
        )
        return np.concatenate((option_prices, digital_prices))

    return price


def _greek_block(
    greeks: SanosStateGreeks,
    *,
    selection: slice,
    factors: RollingSanosFactors,
) -> DlvGreeks:
    spot_exposure = (
        greeks.orthogonalized_spot_exposure
        if factors.method == "spot_orthogonal"
        else greeks.fixed_state_spot
    )
    return DlvGreeks(
        model_prices=greeks.prices[selection],
        spot_deltas=spot_exposure[selection],
        factor_vegas=greeks.factor_exposures[selection],
    )


def run_day(
    panel: PublicHedgingPanel,
    *,
    state_directory: Path,
    origin_date: date,
    claim_quantities: Mapping[str, float] | None = None,
    risk_parameter: float = 10.0,
) -> list[HedgeSmokeResult]:
    """Fit both causal factor variants and hedge all ten one-day digitals."""
    targets = daily_digital_targets(maturity_business_days=1)
    if claim_quantities is not None and set(claim_quantities) != {
        target.label for target in targets
    }:
        raise ValueError("Claim quantities must match the ten pilot targets")
    samples = tuple(
        build_daily_digital_sample(panel, origin_date=origin_date, target=target)
        for target in targets
    )
    window = _rolling_window(panel, state_directory=state_directory, origin_date=origin_date)
    option_count = len(samples[0].hedge_quotes)
    price = _price_function(samples)
    state = _load_raw_state(state_directory, origin_date)
    option_costs = np.asarray(
        [quote.half_spread for quote in samples[0].hedge_quotes], dtype=np.float64
    )
    results: list[HedgeSmokeResult] = []
    for sample in samples:
        quantity = (
            1.0 if claim_quantities is None
            else float(claim_quantities[sample.target.label])
        )
        black = black_digital_greeks(
            strike_over_forward=sample.target_strike_over_forward,
            annual_vol=samples[0].transition.origin.atm_vol_at_business_dte(1),
            business_dte=1.0,
            discount_factor=sample.target_discount_factor,
            right=sample.target.right,
        )
        results.append(evaluate_daily_digital_hedge(
            sample,
            model="bs-delta-only",
            measure="Q-price/historical-transition",
            spot_action=-quantity * black.delta,
            option_actions=np.zeros(option_count),
            model_price=black.price,
            objective_value=0.0,
            converged=True,
            cap_hits=0,
            claim_quantity=quantity,
            diagnostics={"hedge_semantics": "Black delta in mid-priced spot only"},
        ))
    for method in ("ordinary", "spot_orthogonal"):
        factors = fit_rolling_sanos_factors(window, method=method, rank=8)
        greeks = finite_difference_sanos_state_greeks(
            price=price, spot=1.0, raw_state=state, factors=factors
        )
        options = _greek_block(greeks, selection=slice(0, option_count), factors=factors)
        for index, sample in enumerate(samples):
            quantity = (
                1.0 if claim_quantities is None
                else float(claim_quantities[sample.target.label])
            )
            if not np.isfinite(quantity) or quantity <= 0.0:
                raise ValueError(f"Invalid claim quantity for {sample.target.label}")
            target = _greek_block(
                greeks,
                selection=slice(option_count + index, option_count + index + 1),
                factors=factors,
            )
            scaled_target = DlvGreeks(
                model_prices=quantity * target.model_prices,
                spot_deltas=quantity * target.spot_deltas,
                factor_vegas=quantity * target.factor_vegas,
            )
            common_diagnostics = {
                "pca_method": method,
                "factor_rank": factors.rank,
                "training_transitions": len(window.endpoint_dates),
                "training_first_origin": window.origin_dates[0].isoformat(),
                "training_last_endpoint": window.endpoint_dates[-1].isoformat(),
                "factor_variance_explained": float(
                    np.sum(factors.explained_variance_fraction)
                ),
                "fixed_state_spot_partial": float(
                    greeks.fixed_state_spot[option_count + index]
                ),
                "spot_correlated_state_exposure": float(
                    greeks.spot_correlated_state[option_count + index]
                ),
            }
            greek_action = solve_delta_factor_hedge(
                target=scaled_target,
                hedge_options=options,
                option_cost_weights=option_costs,
                residual_tolerance=1.0e-8,
                position_cap=1_000.0,
            )
            results.append(evaluate_daily_digital_hedge(
                sample,
                model=f"sanos-greek-{method}",
                measure="historical-P-rolling-Greeks",
                spot_action=greek_action.spot_action,
                option_actions=greek_action.option_actions,
                model_price=float(target.model_prices[0]),
                objective_value=float(option_costs @ np.abs(greek_action.option_actions)),
                converged=greek_action.cap_hits == 0,
                cap_hits=greek_action.cap_hits,
                claim_quantity=quantity,
                diagnostics={**common_diagnostics, "factor_residual_norm": float(
                    np.linalg.norm(greek_action.factor_residual)
                )},
            ))
            parametric_action = solve_parametric_hedge(
                target=scaled_target,
                options=options,
                shock_covariance=factors.spot_factor_covariance,
                option_cost_weights=option_costs,
                spot_cost_weight=0.0,
                risk_parameter=risk_parameter,
                position_cap=1_000.0,
            )
            results.append(evaluate_daily_digital_hedge(
                sample,
                model=f"sanos-parametric-{method}",
                measure="historical-P-rolling-factor-covariance",
                spot_action=parametric_action.spot_action,
                option_actions=parametric_action.option_actions,
                model_price=float(target.model_prices[0]),
                objective_value=(
                    0.5 * risk_parameter * parametric_action.risk_variance
                    + parametric_action.transaction_cost_proxy
                ),
                converged=parametric_action.cap_hits == 0,
                cap_hits=parametric_action.cap_hits,
                claim_quantity=quantity,
                diagnostics={
                    **common_diagnostics,
                    "solver_status": parametric_action.solver_status,
                    "risk_variance": parametric_action.risk_variance,
                },
            ))
    return results


def run(
    *,
    dataset: Path,
    state_directory: Path,
    required_dates_file: Path,
    output: Path,
    reference_statistical_run: Path | None = None,
    evaluation_dates: tuple[date, ...] = PILOT_DATES,
    risk_parameter: float = 10.0,
    wait_seconds: float = 0.0,
) -> None:
    """Write resumable dated results, then require the full requested cohort."""
    output.mkdir(parents=True, exist_ok=True)
    required_dates = tuple(required_dates_file.read_text().splitlines())
    started = time.monotonic()
    while True:
        missing = [
            value for value in required_dates
            if not (state_directory / f"{value}.npz").is_file()
        ]
        if not missing:
            break
        elapsed = time.monotonic() - started
        if elapsed >= wait_seconds:
            raise RuntimeError(f"Missing {len(missing)} required states; first {missing[0]}")
        print(
            f"Waiting for states: {len(required_dates) - len(missing)}/{len(required_dates)} "
            f"ready, elapsed={elapsed:.0f}s",
            flush=True,
        )
        time.sleep(min(300.0, wait_seconds - elapsed))

    torch.set_num_threads(1)
    panel = PublicHedgingPanel(dataset)
    results: list[dict] = []
    for completed, origin_date in enumerate(evaluation_dates, start=1):
        path = output / f"{origin_date.isoformat()}.json"
        if path.is_file():
            daily = json.loads(path.read_text())
        else:
            quantities = None
            if reference_statistical_run is not None:
                reference_path = (
                    reference_statistical_run / "records"
                    / f"{origin_date.isoformat()}.json"
                )
                reference = json.loads(reference_path.read_text())
                quantities = {
                    record["claim"]: float(record["claim_quantity"])
                    for record in reference["results"]
                }
            daily = [
                asdict(result)
                for result in run_day(
                    panel,
                    state_directory=state_directory,
                    origin_date=origin_date,
                    claim_quantities=quantities,
                    risk_parameter=risk_parameter,
                )
            ]
            temporary = path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(daily, indent=2) + "\n")
            temporary.replace(path)
        results.extend(daily)
        print(f"Completed {completed}/{len(evaluation_dates)} dates: {origin_date}", flush=True)

    expected = len(evaluation_dates) * 10 * 5
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} target-date-method results, got {len(results)}")
    models = sorted({result["model"] for result in results})
    summary = {
        "schema": "sanos-greek-daily-cohort-v2",
        "dataset": str(dataset),
        "state_directory": str(state_directory),
        "evaluation_dates": [value.isoformat() for value in evaluation_dates],
        "training_transitions": 252,
        "factor_rank": 8,
        "parametric_risk_parameter": risk_parameter,
        "methods": {
            model: {
                "results": len(selected := [
                    result for result in results if result["model"] == model
                ]),
                "mean_abs_pure_hedge_error": float(np.mean([
                    abs(result["pure_hedge_error"]) / result["claim_quantity"]
                    for result in selected
                ])),
                "median_abs_pure_hedge_error": float(np.median([
                    abs(result["pure_hedge_error"]) / result["claim_quantity"]
                    for result in selected
                ])),
                "mean_entry_cost": float(np.mean([
                    result["transaction_cost"] / result["claim_quantity"]
                    for result in selected
                ])),
                "cap_hits": sum(result["cap_hits"] for result in selected),
                "nonconverged": sum(not result["converged"] for result in selected),
            }
            for model in models
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--required-dates-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-statistical-run", type=Path)
    parser.add_argument("--evaluation-dates-file", type=Path)
    parser.add_argument("--risk-parameter", type=float, default=10.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    arguments = parser.parse_args()
    run(
        dataset=arguments.dataset,
        state_directory=arguments.state_directory,
        required_dates_file=arguments.required_dates_file,
        output=arguments.output,
        reference_statistical_run=arguments.reference_statistical_run,
        evaluation_dates=(
            tuple(date.fromisoformat(value) for value in arguments.evaluation_dates_file.read_text().splitlines())
            if arguments.evaluation_dates_file is not None else PILOT_DATES
        ),
        risk_parameter=arguments.risk_parameter,
        wait_seconds=arguments.wait_seconds,
    )


if __name__ == "__main__":
    main()
