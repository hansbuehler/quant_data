"""Fit every eligible FuNVol predecessor date before hedge evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    SanosFixedGridHistory,
    build_sanos_fixed_grid_history,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.config import (
    FunvolConfig,
    FunvolTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.funvol.training import fit_funvol
from quantdata.torch import BaseEnvironment, torch


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _history_window(
    full_history: SanosFixedGridHistory,
    *,
    target_index: int,
    observation_count: int,
) -> SanosFixedGridHistory:
    start = target_index + 1 - observation_count
    if start < 0:
        raise ValueError("FuNVol history window lacks enough preceding observations")
    stop = target_index + 1
    return SanosFixedGridHistory(
        dates=full_history.dates[start:stop],
        log_implied_volatilities=full_history.log_implied_volatilities[start:stop],
        implied_vol_is_interior=full_history.implied_vol_is_interior[start:stop],
        forward_log_returns_into_date=np.r_[
            np.nan, full_history.forward_log_returns_into_date[start + 1 : stop]
        ],
        sessions_from_previous_observation=np.r_[
            0, full_history.sessions_from_previous_observation[start + 1 : stop]
        ],
    )


def fit_predecessor_chain(
    *,
    dataset: Path,
    output_directory: Path,
    initializer_checkpoint: Path,
    start_date: date,
    end_date: date,
    epochs: int,
    observation_count: int = 252,
) -> None:
    """Fit causally through every observed session and checkpoint each date."""
    if start_date > end_date:
        raise ValueError("FuNVol predecessor date range is reversed")
    panel = PublicHedgingPanel(dataset)
    first_eligible_index = observation_count - 1
    if start_date < panel.dates[first_eligible_index]:
        raise ValueError("FuNVol first predecessor date lacks the full trailing window")
    dates = tuple(day for day in panel.dates if start_date <= day <= end_date)
    if not dates:
        raise ValueError("FuNVol predecessor range is empty")
    config = FunvolConfig()
    training = FunvolTrainConfig(epochs=epochs, validation_fraction=0.0)
    initializer_sha256 = file_sha256(initializer_checkpoint)
    initializer_date = str(
        torch.load(initializer_checkpoint, map_location="cpu", weights_only=False)["as_of_date"]
    )
    initial_gap = [
        day.isoformat()
        for day in panel.dates
        if initializer_date < day.isoformat() < start_date.isoformat()
    ]
    identity = {
        "schema": "funvol-predecessor-chain-v1",
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "initializer_checkpoint": str(initializer_checkpoint),
        "initializer_sha256": initializer_sha256,
        "initializer_date": initializer_date,
        "initialization_no_update_dates": initial_gap,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "model_config": repr(config),
        "train_config": repr(training),
        "observation_count": observation_count,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("FuNVol predecessor run identity changed")
    else:
        _write_json(manifest, identity)
    checkpoint_directory = output_directory / "checkpoints"
    record_directory = output_directory / "records"
    checkpoint_directory.mkdir(exist_ok=True)
    record_directory.mkdir(exist_ok=True)
    full_history = build_sanos_fixed_grid_history(
        panel,
        end_date=end_date,
        observation_count=panel.dates.index(end_date) + 1,
        grid=config.grid,
    )
    previous_checkpoint = initializer_checkpoint
    previous_date = initializer_date
    started = monotonic()
    for index, as_of_date in enumerate(dates):
        history = _history_window(
            full_history,
            target_index=full_history.dates.index(as_of_date),
            observation_count=observation_count,
        )
        checkpoint = checkpoint_directory / f"{as_of_date.isoformat()}.pt"
        record_path = record_directory / f"{as_of_date.isoformat()}.json"
        previous_sha256 = file_sha256(previous_checkpoint)
        if history.sessions_from_previous_observation[-1] != 1:
            if checkpoint.exists():
                raise RuntimeError(f"FuNVol non-session date has a fitted checkpoint: {as_of_date}")
            record = {
                "as_of_date": as_of_date.isoformat(),
                "status": "no-update-gap",
                "sessions_from_previous_observation": int(
                    history.sessions_from_previous_observation[-1]
                ),
                "last_fit_date": previous_date,
                "last_fit_sha256": previous_sha256,
            }
            if record_path.exists():
                if json.loads(record_path.read_text(encoding="utf-8")) != record:
                    raise RuntimeError(f"FuNVol no-update record changed: {as_of_date}")
            else:
                _write_json(record_path, record)
        else:
            fit = fit_funvol(
                history,
                config=config,
                train_config=training,
                environment=BaseEnvironment(precision=64, device="cpu"),
                checkpoint_path=checkpoint,
                warm_start_checkpoint=previous_checkpoint,
            )
            record = {
                "as_of_date": as_of_date.isoformat(),
                "status": "fit",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "warm_start_date": fit.warm_start_date,
                "warm_start_sha256": fit.warm_start_sha256,
                "completed_epochs": fit.completed_epochs,
                "training_loss": fit.training_loss,
                "training_target_count": fit.training_target_count,
                "validation_target_count": fit.validation_target_count,
                "latest_training_surface_index": fit.last_training_surface_index,
            }
            if fit.warm_start_date != previous_date or fit.warm_start_sha256 != previous_sha256:
                raise RuntimeError(f"FuNVol predecessor lineage mismatch: {as_of_date}")
            if (
                fit.validation_target_count != 0
                or fit.last_training_surface_index != observation_count - 1
            ):
                raise RuntimeError(
                    f"FuNVol predecessor fit omitted current information: {as_of_date}"
                )
            if record_path.exists():
                if json.loads(record_path.read_text(encoding="utf-8")) != record:
                    raise RuntimeError(f"FuNVol predecessor record changed: {as_of_date}")
            else:
                _write_json(record_path, record)
            previous_checkpoint = checkpoint
            previous_date = as_of_date.isoformat()
        elapsed = monotonic() - started
        eta = elapsed * (len(dates) - index - 1) / (index + 1)
        print(
            f"FuNVol predecessors {index + 1}/{len(dates)} {as_of_date}: "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )


def main() -> None:
    """Parse a restartable fit-only predecessor invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initializer-checkpoint", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 9))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 3, 28))
    parser.add_argument("--epochs", type=int, default=100)
    arguments = parser.parse_args()
    fit_predecessor_chain(**vars(arguments))


if __name__ == "__main__":
    main()
