"""Fit separate Gaussian SANOS parametric hedge objectives on matched dates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import torch

from quantdata.hedging.data.rolling_sanos_factors import fit_rolling_sanos_factors
from quantdata.hedging.data.sanos_state_greeks import finite_difference_sanos_state_greeks
from quantdata.hedging.sanos_greek_week import (
    _greek_block,
    _load_raw_state,
    _price_function,
    _rolling_window,
)
from quantdata.sdlv.daily.front_spot.hedging.benchmark.digital_daily import (
    build_daily_digital_sample,
    daily_digital_targets,
    evaluate_daily_digital_hedge,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_lv.greeks import DlvGreeks
from quantdata.sdlv.daily.front_spot.hedging.models.sanos_parametric.hedger import (
    solve_parametric_hedge,
)

OBJECTIVES = ("mean-variance", "entropy", "power-exponential", "cvar")
RISK_PARAMETERS = (1.0, 10.0, 100.0, 1000.0)


def _record_path(root: Path, objective: str, risk_parameter: float, origin_date: date) -> Path:
    return root / objective / f"lambda{risk_parameter:g}" / "records" / f"{origin_date}.json"


def run(
    *,
    dataset: Path,
    state_directory: Path,
    reference_run: Path,
    evaluation_dates: tuple[date, ...],
    output: Path,
    stop_after: int | None = None,
) -> None:
    """Fit one objective per cell, retaining a strict dated certificate."""
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "sanos-parametric-objective-grid-v1",
        "dataset": str(dataset),
        "state_directory": str(state_directory),
        "reference_run": str(reference_run),
        "evaluation_dates": [value.isoformat() for value in evaluation_dates],
        "factor_method": "ordinary",
        "factor_rank": 8,
        "rolling_transitions": 252,
        "objectives": list(OBJECTIVES),
        "risk_parameters": list(RISK_PARAMETERS),
    }
    identity_path = output / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("SANOS parametric run identity differs from existing output")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    panel = PublicHedgingPanel(dataset)
    origins = evaluation_dates[:stop_after] if stop_after is not None else evaluation_dates
    started = time.monotonic()
    failed = []
    for completed, origin_date in enumerate(origins, start=1):
        missing = [
            (objective, risk_parameter)
            for objective in OBJECTIVES
            for risk_parameter in RISK_PARAMETERS
            if not _record_path(output, objective, risk_parameter, origin_date).exists()
        ]
        if missing:
            reference_path = reference_run / "records" / f"{origin_date}.json"
            reference = json.loads(reference_path.read_text())["results"]
            quantities = {str(row["claim"]): float(row["claim_quantity"]) for row in reference}
            samples = tuple(
                build_daily_digital_sample(panel, origin_date=origin_date, target=target)
                for target in daily_digital_targets(maturity_business_days=1)
            )
            if {sample.target.label for sample in samples} != set(quantities):
                raise ValueError(f"Target quantities differ on {origin_date}")
            window = _rolling_window(panel, state_directory=state_directory,
                                     origin_date=origin_date)
            factors = fit_rolling_sanos_factors(window, method="ordinary", rank=8)
            greeks = finite_difference_sanos_state_greeks(
                price=_price_function(samples), spot=1.0,
                raw_state=_load_raw_state(state_directory, origin_date), factors=factors,
            )
            option_count = len(samples[0].hedge_quotes)
            options = _greek_block(greeks, selection=slice(0, option_count), factors=factors)
            costs = np.asarray([quote.half_spread for quote in samples[0].hedge_quotes],
                               dtype=np.float64)
            rows_by_cell = {cell: [] for cell in missing}
            for index, sample in enumerate(samples):
                quantity = quantities[sample.target.label]
                target = _greek_block(
                    greeks, selection=slice(option_count + index, option_count + index + 1),
                    factors=factors,
                )
                scaled_target = DlvGreeks(
                    model_prices=quantity * target.model_prices,
                    spot_deltas=quantity * target.spot_deltas,
                    factor_vegas=quantity * target.factor_vegas,
                )
                for objective, risk_parameter in missing:
                    action = solve_parametric_hedge(
                        target=scaled_target,
                        options=options,
                        shock_covariance=factors.spot_factor_covariance,
                        option_cost_weights=costs,
                        spot_cost_weight=0.0,
                        risk_parameter=risk_parameter,
                        position_cap=1_000.0,
                        objective=objective,
                    )
                    result = evaluate_daily_digital_hedge(
                        sample,
                        model="sanos-parametric-ordinary",
                        measure="historical-P-rolling-factor-Gaussian",
                        spot_action=action.spot_action,
                        option_actions=action.option_actions,
                        model_price=float(target.model_prices[0]),
                        objective_value=action.objective_value,
                        converged=action.converged and action.cap_hits == 0,
                        cap_hits=action.cap_hits,
                        claim_quantity=quantity,
                        diagnostics={
                            "objective": objective,
                            "risk_parameter": risk_parameter,
                            "solver_status": action.solver_status,
                            "risk_variance": action.risk_variance,
                            "factor_rank": factors.rank,
                            "factor_variance_explained": float(
                                np.sum(factors.explained_variance_fraction)
                            ),
                            "training_first_origin": window.origin_dates[0].isoformat(),
                            "training_last_endpoint": window.endpoint_dates[-1].isoformat(),
                        },
                    )
                    rows_by_cell[(objective, risk_parameter)].append(asdict(result))
            for (objective, risk_parameter), rows in rows_by_cell.items():
                record_path = _record_path(output, objective, risk_parameter, origin_date)
                complete = len(rows) == len(samples) and all(
                    row["converged"] and row["cap_hits"] == 0 for row in rows
                )
                destination = record_path if complete else record_path.parent.parent / "failures" / record_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial.json")
                temporary.write_text(json.dumps({"origin_date": origin_date.isoformat(),
                                                 "results": rows}, indent=2) + "\n")
                temporary.replace(destination)
                if not complete:
                    failed.append((origin_date.isoformat(), objective, risk_parameter))
        elapsed = time.monotonic() - started
        remaining = elapsed / completed * (len(origins) - completed)
        print(f"SANOS parametric {completed}/{len(origins)} {origin_date} "
              f"elapsed={elapsed:.0f}s eta={remaining:.0f}s failures={len(failed)}", flush=True)
    if failed:
        raise RuntimeError(f"SANOS parametric cells failed certification: {failed[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--evaluation-dates-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    arguments = parser.parse_args()
    evaluation_dates = tuple(
        date.fromisoformat(value) for value in arguments.evaluation_dates_file.read_text().splitlines()
    )
    run(
        dataset=arguments.dataset,
        state_directory=arguments.state_directory,
        reference_run=arguments.reference_run,
        evaluation_dates=evaluation_dates,
        output=arguments.output,
        stop_after=arguments.stop_after,
    )


if __name__ == "__main__":
    main()
