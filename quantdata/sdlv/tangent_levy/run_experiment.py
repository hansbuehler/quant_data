"""Command-line entry point for Tangent Lévy calibration and simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
from cdxcore.verbose import Context

from .config import DynamicsConfig, SimulationConfig, TangentLevyConfig
from .experiment import load_calibrated_history, run_core_experiment
from .markets import common_grid_coverage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 8, 30))
    parser.add_argument("--label", default="spx-2022-2025")
    parser.add_argument("--paths", type=int, default=5_000)
    parser.add_argument("--maximum-dates", type=int)
    parser.add_argument(
        "--dynamics-end",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
    )
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--no-simulation", action="store_true")
    parser.add_argument(
        "--allow-failed-reconstruction",
        action="store_true",
        help=(
            "diagnostic override: fit dynamics even when cumulative-density "
            "reconstruction materially changes fitted surfaces"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run the configured experiment and emit an audit summary."""

    args = _parser().parse_args()
    config = TangentLevyConfig(
        label=args.label,
        dynamics=DynamicsConfig(
            require_reconstruction_pass=not args.allow_failed_reconstruction
        ),
        simulation=SimulationConfig(num_paths=args.paths),
    )
    if args.static_only:
        history = load_calibrated_history(
            args.ticker,
            args.start,
            args.end,
            config=config,
            verbose=Context.all,
            maximum_dates=args.maximum_dates,
        )
        result = None
    else:
        result = run_core_experiment(
            args.ticker,
            args.start,
            args.end,
            config=config,
            verbose=Context.all,
            maximum_dates=args.maximum_dates,
            dynamics_end=args.dynamics_end,
            simulate=not args.no_simulation,
        )
        history = result.history
    coverage = common_grid_coverage(
        history.audits,
        candidate_tau=np.asarray(config.dynamics.tau_grid),
        minimum_fraction=0.95,
    )
    maturity_fits = [fit for day in history.day_fits for fit in day.fits]
    summary = {
        "ticker": history.ticker,
        "requested_dates": len(history.requested_dates),
        "calibrated_dates": len(history.dates),
        "successful_date_fraction": history.successful_date_fraction,
        "failed_dates": history.failure_reasons,
        "market_audit": {
            "source": config.market.source,
            "available_dates": sum(audit.available for audit in history.audits),
            "early_close_dates": sum(audit.early_close for audit in history.audits),
            "option_cap_bound_dates": (
                None
                if config.market.source == "dysanos_grid_cache"
                else sum(audit.option_cap_bound for audit in history.audits)
            ),
            "candidate_tau": coverage.candidate_tau.tolist(),
            "coverage_fraction": coverage.coverage_fraction.tolist(),
            "selected_tau": coverage.selected_tau.tolist(),
        },
        "static_fit": {
            "maturity_fits": len(maturity_fits),
            "optimizer_success_fraction": float(
                np.mean([fit.success for fit in maturity_fits])
            ),
            "objective_weighting": (
                "dysanos_grid_inverse_vega_sqrt"
                if config.market.source == "dysanos_grid_cache"
                else "inverse_half_spread"
            ),
            "median_weighted_rmse": float(
                np.median([fit.weighted_rmse for fit in maturity_fits])
            ),
            "maximum_price_error": float(
                np.max([fit.maximum_absolute_error for fit in maturity_fits])
            ),
            "reconstruction_pass_fraction": float(
                np.mean(history.reconstruction_passed)
            ),
            "maximum_reconstruction_relative_projection": float(
                np.max(history.reconstruction_relative_error)
            ),
            "maximum_reconstruction_l1_projection": float(
                np.max(history.reconstruction_l1_error)
            ),
            "median_reconstruction_repricing_rmse": float(
                np.median(history.reconstruction_repricing_rmse)
            ),
            "maximum_reconstruction_repricing_rmse": float(
                np.max(history.reconstruction_repricing_rmse)
            ),
            "maximum_reconstruction_price_error": float(
                np.max(history.reconstruction_maximum_price_error)
            ),
        },
        "dynamics": (
            None
            if result is None
            else {
                "num_factors": result.dynamics.num_factors,
                "observations": result.dynamics.observations,
                "training_end": result.dynamics_end,
                "explained_variance_ratio": (
                    result.dynamics.explained_variance_ratio.tolist()
                ),
                "maximum_constraint_residual": (
                    result.dynamics.maximum_constraint_residual
                ),
                "active_cell_fraction": float(np.mean(result.dynamics.active_mask)),
            }
        ),
        "simulation": (
            None
            if result is None or result.simulation is None
            else {
                "failed_path_fraction": result.simulation.failed_path_fraction,
                "frozen_path_fraction": result.simulation.frozen_path_fraction,
                "minimum_gamma": float(np.min(result.simulation.minimum_gamma)),
                "boundary_limited_paths": int(
                    np.count_nonzero(result.simulation.boundary_limited_steps)
                ),
            }
        ),
        "config": asdict(config),
    }
    rendered = json.dumps(summary, indent=2, default=str)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
