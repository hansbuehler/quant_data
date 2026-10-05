"""Command-line entry points for front-spot calibration and validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from quantdata.sdlv.daily.front_spot.audit import (  # noqa: E402
    run_risk_neutral_martingale_audit,
)
from quantdata.sdlv.daily.front_spot.calibration import (  # noqa: E402
    FrontSpotCalibration,
    calibrate_front_spot_model,
)
from quantdata.sdlv.daily.front_spot.simulation import run_cpu_pilot  # noqa: E402


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fit-label", default="full")
    parser.add_argument("--factor-rank", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    parser.add_argument("--expiry-smoothing", type=float, default=1.0)
    parser.add_argument("--strike-smoothing", type=float, default=2.0)


def _calibrate(arguments: argparse.Namespace) -> FrontSpotCalibration:
    return calibrate_front_spot_model(
        fit_label=arguments.fit_label,
        factor_rank=arguments.factor_rank,
        batch_size=arguments.calibration_batch_size,
        smoothing_sigma=(arguments.expiry_smoothing, arguments.strike_smoothing),
    )


def _calibration_summary(calibration: FrontSpotCalibration) -> dict[str, Any]:
    innovations = calibration.innovations
    physical = calibration.physical
    loading_norm = np.linalg.norm(innovations.annualized_loadings, axis=-1)
    return {
        "training_pair_count": int(len(calibration.history.origin_indices)),
        "factor_rank": innovations.factor_rank,
        "smoothed_explained_variance_fraction": float(
            innovations.explained_variance_fraction
        ),
        "annual_loading_norm_quantiles": np.quantile(
            loading_norm,
            [0.5, 0.9, 0.99, 1.0],
        ).tolist(),
        "spot_score_mean": innovations.spot_score_mean,
        "spot_score_std": innovations.spot_score_std,
        "statewise_achievable_fraction": physical.achievable_fraction,
        "constant_achievable_fraction": physical.constant_achievable_fraction,
        "statewise_weighted_residual_rms": physical.weighted_residual_rms,
        "constant_weighted_residual_rms": physical.constant_weighted_residual_rms,
        "constant_market_price_of_risk": (
            physical.constant_market_price_of_risk.tolist()
        ),
        "q_log_drift_quantiles": np.quantile(
            physical.risk_neutral_log_drift,
            [0.001, 0.01, 0.5, 0.99, 0.999],
        ).tolist(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration_parser = subparsers.add_parser("calibrate")
    _add_calibration_arguments(calibration_parser)

    pilot_parser = subparsers.add_parser("pilot")
    _add_calibration_arguments(pilot_parser)
    pilot_parser.add_argument("--date-index", type=int, default=-1)
    pilot_parser.add_argument("--paths", type=int, default=16)
    pilot_parser.add_argument("--days", type=int, default=25)
    pilot_parser.add_argument("--substeps", type=int, default=8)
    pilot_parser.add_argument("--time-power", type=float, default=1.0)
    pilot_parser.add_argument("--seed", type=int, default=20261004)

    audit_parser = subparsers.add_parser("audit")
    _add_calibration_arguments(audit_parser)
    audit_parser.add_argument("--date-index", type=int, default=-1)
    audit_parser.add_argument("--paths", type=int, default=1_000)
    audit_parser.add_argument("--batch-size", type=int, default=32)
    audit_parser.add_argument("--substeps", type=int, default=8)
    audit_parser.add_argument("--time-power", type=float, default=1.0)
    audit_parser.add_argument("--seed", type=int, default=20261004)
    return parser


def main() -> None:
    """Run the selected front-spot workflow and print JSON diagnostics."""
    arguments = _parser().parse_args()
    calibration = _calibrate(arguments)
    if arguments.command == "calibrate":
        output: dict[str, Any] = _calibration_summary(calibration)
    elif arguments.command == "pilot":
        output = {
            "calibration": _calibration_summary(calibration),
            "pilot": asdict(
                run_cpu_pilot(
                    calibration,
                    fit_label=arguments.fit_label,
                    date_index=arguments.date_index,
                    path_count=arguments.paths,
                    days=arguments.days,
                    dlv_substeps=arguments.substeps,
                    time_power=arguments.time_power,
                    seed=arguments.seed,
                )
            ),
        }
    else:
        output = {
            "calibration": _calibration_summary(calibration),
            "audit": asdict(
                run_risk_neutral_martingale_audit(
                    calibration,
                    fit_label=arguments.fit_label,
                    date_index=arguments.date_index,
                    path_count=arguments.paths,
                    batch_size=arguments.batch_size,
                    dlv_substeps=arguments.substeps,
                    time_power=arguments.time_power,
                    seed=arguments.seed,
                )
            ),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
