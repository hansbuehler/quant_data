"""Fit the 2022--2025 fixed-grid surface used by the drift-adjusted model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    FlexObjectiveConfig,
    FlexTrainConfig,
)
from quantdata.dysanos.flex.failure_reporting import run_guarded_cli
from quantdata.dysanos.flex.fixed_relative_grid import (
    representative_relative_strike_grid,
)
from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.run_all_history_joint_direct import (
    _market_config_from_completion,
)
from quantdata.dysanos.flex.run_production_gate import run_production_gates
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _market_dates_from_completion,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-joint-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-native-completion", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=("native", "affine"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--model-business-dtes",
        type=float,
        nargs="+",
        required=True,
        help="Fixed time-to-expiry nodes in business days.",
    )
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--supervised-epochs", type=int, default=300)
    parser.add_argument("--day-batch-size", type=int, default=16)
    parser.add_argument("--supervised-day-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--supervised-learning-rate", type=float, default=2.0e-3)
    return parser.parse_args()


def run(arguments: argparse.Namespace) -> None:
    """Run one restartable static-calibration stage on the prescribed grid."""
    if arguments.ticker != "SPX":
        raise ValueError("The fixed-grid experiment is specified for SPX")
    if arguments.device not in {"cuda:0", "cpu"}:
        raise ValueError("Training is restricted to physical GPU 0 or CPU")
    if arguments.workers < 1 or arguments.workers > 4:
        raise ValueError("'workers' must lie in [1,4]")
    for path in (
        arguments.reference_joint_checkpoint,
        arguments.reference_native_completion,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    grid = representative_relative_strike_grid(
        arguments.reference_joint_checkpoint,
    )
    model_business_dtes = tuple(arguments.model_business_dtes)
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        num_model_expiries=len(model_business_dtes),
        explicit_model_business_dtes=model_business_dtes,
        inner_relative_strikes=grid.inner_relative_strikes,
    )
    completion = json.loads(
        arguments.reference_native_completion.read_text(encoding="utf-8")
    )
    dates = _market_dates_from_completion(arguments.reference_native_completion)
    market_config = _market_config_from_completion(completion)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        {
            "schema": "flex-representative-fixed-relative-grid-v1",
            "status": "complete",
            "construction": "coordinate-wise median decoded log-relative strike",
            "source_checkpoint": str(arguments.reference_joint_checkpoint),
            **asdict(grid),
            "model_business_dtes": list(model_business_dtes),
        },
        arguments.output_directory / "representative-fixed-grid.json",
    )
    run_production_gates(
        ticker=arguments.ticker,
        dates=dates,
        geometry=geometry,
        market_config=market_config,
        objective_config=FlexObjectiveConfig(fit_weight=0.05),
        train_config=FlexTrainConfig(
            num_epochs=arguments.epochs,
            day_batch_size=arguments.day_batch_size,
            learning_rate=arguments.learning_rate,
            device=arguments.device,
            precision=32,
            compile_sanos_solver=True,
        ),
        supervised_train_config=FlexTrainConfig(
            num_epochs=arguments.supervised_epochs,
            day_batch_size=arguments.supervised_day_batch_size,
            learning_rate=arguments.supervised_learning_rate,
            device=arguments.device,
            precision=32,
        ),
        workers=arguments.workers,
        output_directory=arguments.output_directory,
        checkpoint_directory=arguments.checkpoint_directory,
        label=arguments.label,
        state_sizes=(200,),
        run_compression=arguments.stage == "affine",
        expiry_benchmark=False,
    )


def main() -> None:
    """CLI entry point."""
    run(_parse_arguments())


if __name__ == "__main__":
    run_guarded_cli(main, program="run_fixed_grid_drift_adjusted_model")
