"""Run dated diffusion hedge fits with authenticated previous-day warm starts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.adapter import (
    fit_public_diffusion,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.torch import BaseEnvironment, torch


def _completed_fit(
    run_directory: Path,
    *,
    origin_date: date,
    expected_parent_sha256: str,
) -> tuple[Path, str] | None:
    record = run_directory / "records" / f"{origin_date.isoformat()}.json"
    if not record.exists():
        return None
    payload = json.loads(record.read_text(encoding="utf-8"))
    if payload.get("origin_date") != origin_date.isoformat():
        raise ValueError(f"rolling record has wrong date: {record}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"rolling record has no results: {record}")
    diagnostics = [result.get("diagnostics") for result in results]
    if any(not isinstance(item, dict) for item in diagnostics):
        raise ValueError(f"rolling record lacks diagnostics: {record}")
    first = diagnostics[0]
    parent_hash = first.get("warm_start_sha256")
    checkpoint_hash = first.get("fit_checkpoint_sha256")
    checkpoint_raw = first.get("fit_checkpoint_path")
    if parent_hash != expected_parent_sha256:
        raise RuntimeError(f"rolling parent hash mismatch at {origin_date}")
    if not isinstance(checkpoint_hash, str) or not isinstance(checkpoint_raw, str):
        raise ValueError(f"rolling record lacks checkpoint lineage: {record}")
    if any(
        item.get("warm_start_sha256") != parent_hash
        or item.get("fit_checkpoint_sha256") != checkpoint_hash
        or item.get("fit_checkpoint_path") != checkpoint_raw
        for item in diagnostics
    ):
        raise RuntimeError(f"target checkpoint lineage disagrees at {origin_date}")
    checkpoint = Path(checkpoint_raw)
    if not checkpoint.is_file() or file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(f"dated rolling checkpoint is absent or changed: {checkpoint}")
    if _checkpoint_training_end_date(checkpoint) != origin_date:
        raise RuntimeError(f"rolling checkpoint has wrong training date: {checkpoint}")
    scenario_path_raw = first.get("scenario_artifact_path")
    scenario_hash = first.get("scenario_artifact_sha256")
    if scenario_path_raw is not None or scenario_hash is not None:
        if not isinstance(scenario_path_raw, str) or not isinstance(scenario_hash, str):
            raise ValueError(f"rolling scenario artifact identity incomplete: {record}")
        if any(
            item.get("scenario_artifact_path") != scenario_path_raw
            or item.get("scenario_artifact_sha256") != scenario_hash
            for item in diagnostics
        ):
            raise RuntimeError(f"target scenario lineage disagrees at {origin_date}")
        scenario_path = Path(scenario_path_raw)
        if not scenario_path.is_file() or file_sha256(scenario_path) != scenario_hash:
            raise RuntimeError(f"dated rolling scenarios are absent or changed: {scenario_path}")
    return checkpoint, checkpoint_hash


def _checkpoint_training_end_date(checkpoint: Path) -> date:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        return date.fromisoformat(str(payload["training_end_date"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint has no dated training identity: {checkpoint}") from error


def _selected_epochs(
    path: Path,
    *,
    model: str,
    dataset_manifest_sha256: str,
    production_checkpoint: Path,
) -> int:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("schema") != "hedging-generated-epoch-selection-v1"
        or selection.get("model") != model
        or selection.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or selection.get("one_standard_error_rule") is not True
        or not isinstance(selection.get("selected_epochs"), int)
        or selection["selected_epochs"] <= 0
    ):
        raise ValueError("rolling epoch selection is missing or incompatible")
    source = Path(selection["warm_start_checkpoint"])
    if file_sha256(source) != selection.get("warm_start_sha256"):
        raise RuntimeError("2022 selection source checkpoint changed")
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    production_payload = torch.load(
        production_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        source_payload.get("base_compatibility_identity")
        != production_payload.get("base_compatibility_identity")
        or source_payload.get("model_code_version")
        != production_payload.get("model_code_version")
    ):
        raise ValueError("2022 epoch selection uses another model geometry")
    validation_dates = selection.get("validation_origins")
    if not isinstance(validation_dates, list) or not validation_dates or any(
        date.fromisoformat(str(value)).year != 2022 for value in validation_dates
    ):
        raise ValueError("rolling epoch selection has invalid 2022 validation dates")
    return selection["selected_epochs"]


def _completed_pre_evaluation_fit(
    path: Path,
    *,
    origin_date: date,
    model: str,
    model_label: str,
    epochs: int,
    parent_sha256: str,
    dataset_manifest_sha256: str,
    epoch_selection_sha256: str,
) -> tuple[Path, str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "hedging-rolling-diffusion-fit-v1"
        or payload.get("origin_date") != origin_date.isoformat()
        or payload.get("model") != model
        or payload.get("model_label") != model_label
        or payload.get("epochs") != epochs
        or payload.get("parent_sha256") != parent_sha256
        or payload.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or payload.get("epoch_selection_sha256") != epoch_selection_sha256
    ):
        raise RuntimeError(f"pre-evaluation fit lineage mismatch: {path}")
    checkpoint = Path(payload["checkpoint_path"])
    checkpoint_sha256 = payload["checkpoint_sha256"]
    no_update = payload.get("status") == "no-update"
    if (
        not checkpoint.is_file()
        or file_sha256(checkpoint) != checkpoint_sha256
        or (
            not no_update
            and _checkpoint_training_end_date(checkpoint) != origin_date
        )
        or (
            no_update
            and (
                payload.get("reason") != "missing exact-session return"
                or checkpoint_sha256 != parent_sha256
                or _checkpoint_training_end_date(checkpoint) >= origin_date
            )
        )
    ):
        raise RuntimeError(f"pre-evaluation fit checkpoint changed: {checkpoint}")
    return checkpoint, checkpoint_sha256


def _has_fresh_diffusion_target(panel: PublicHedgingPanel, origin: date) -> bool:
    index = panel.dates.index(origin)
    if index < 3:
        return False
    return all(
        panel.observations.row(index - offset, named=True)[
            "business_sessions_from_previous_observation"
        ] == 1
        for offset in range(3)
    )


def _carry_without_update(
    path: Path,
    *,
    origin_date: date,
    model: str,
    model_label: str,
    epochs: int,
    parent_path: Path,
    parent_sha256: str,
    dataset_manifest_sha256: str,
    epoch_selection_sha256: str,
) -> tuple[Path, str]:
    payload = {
        "schema": "hedging-rolling-diffusion-fit-v1",
        "status": "no-update",
        "reason": "missing exact-session return",
        "origin_date": origin_date.isoformat(),
        "model": model,
        "model_label": model_label,
        "epochs": epochs,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "epoch_selection_sha256": epoch_selection_sha256,
        "parent_path": str(parent_path),
        "parent_sha256": parent_sha256,
        "checkpoint_path": str(parent_path),
        "checkpoint_sha256": parent_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return parent_path, parent_sha256


def _fit_pre_evaluation_date(
    panel: PublicHedgingPanel,
    *,
    origin_date: date,
    model: str,
    model_label: str,
    parent_path: Path,
    parent_sha256: str,
    base_checkpoint: Path | None,
    base_checkpoint_sha256: str | None,
    checkpoint_directory: Path,
    observation_count: int,
    epochs: int,
    epoch_selection_sha256: str,
    lineage_path: Path,
) -> tuple[Path, str]:
    variant = DiffusionVariant(model.removeprefix("diffusion-"))
    config = DiffusionModelConfig(
        label=model_label,
        variant=variant,
        grid=GenerativeSurfaceGrid.short_end_nms(),
    )
    fitted = fit_public_diffusion(
        panel,
        as_of_date=origin_date,
        model_config=config,
        checkpoint_directory=checkpoint_directory,
        observation_count=observation_count,
        train_config=DiffusionTrainConfig(
            epochs=epochs,
            warmup_steps=0,
            checkpoint_epochs=min(10, epochs),
            progress_seconds=180.0,
            mixed_precision=False,
        ),
        base_checkpoint=base_checkpoint,
        warm_start_checkpoint=parent_path,
        expected_base_checkpoint_sha256=base_checkpoint_sha256,
        expected_warm_start_sha256=parent_sha256,
        environment=BaseEnvironment(precision=32, device="cpu", debug=False),
    )
    if not fitted.result.converged or fitted.result.warm_start_sha256 != parent_sha256:
        raise RuntimeError(f"incomplete or incorrectly initialized rolling fit {origin_date}")
    checkpoint = fitted.result.checkpoint_path
    checkpoint_sha256 = file_sha256(checkpoint)
    lineage = {
        "schema": "hedging-rolling-diffusion-fit-v1",
        "status": "fitted",
        "origin_date": origin_date.isoformat(),
        "model": model,
        "model_label": model_label,
        "dataset_manifest_sha256": panel.dataset_manifest_sha256,
        "epoch_selection_sha256": epoch_selection_sha256,
        "epochs": epochs,
        "parent_path": str(parent_path),
        "parent_sha256": parent_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_loss": fitted.result.validation_loss,
    }
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = lineage_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(lineage_path)
    return checkpoint, checkpoint_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--model", choices=("diffusion-ad-seq-vol", "diffusion-ad-seq-vol-ft"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--evaluation-start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--epoch-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--observation-count", type=int, default=252)
    parser.add_argument("--scenario-count", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--stop-after", type=int)
    arguments = parser.parse_args()
    if not arguments.start <= arguments.evaluation_start <= arguments.end:
        parser.error("start, evaluation start, and end must be chronological")
    if arguments.model.endswith("-ft") != (arguments.base_checkpoint is not None):
        parser.error("fine-tuned variant requires exactly one base checkpoint")
    if file_sha256(arguments.initial_checkpoint) != arguments.initial_checkpoint_sha256:
        raise RuntimeError("initial checkpoint SHA-256 mismatch")
    if arguments.base_checkpoint is not None and (
        file_sha256(arguments.base_checkpoint) != arguments.base_checkpoint_sha256
    ):
        raise RuntimeError("base checkpoint SHA-256 mismatch")
    panel = PublicHedgingPanel(arguments.dataset, validate=False)
    if panel.dataset_manifest_sha256 != arguments.dataset_manifest_sha256:
        raise RuntimeError("released dataset manifest changed")
    epochs = _selected_epochs(
        arguments.epoch_selection,
        model=arguments.model,
        dataset_manifest_sha256=arguments.dataset_manifest_sha256,
        production_checkpoint=arguments.initial_checkpoint,
    )
    selection_sha256 = file_sha256(arguments.epoch_selection)
    panel_dates = tuple(panel.dates)
    origins = tuple(
        origin for origin in panel_dates if arguments.start <= origin <= arguments.end
    )
    if any(
        not _eligible(panel, origin, arguments.observation_count)
        for origin in origins
        if origin >= arguments.evaluation_start
    ):
        raise RuntimeError("rolling interval contains an ineligible or incomplete session")
    if arguments.stop_after is not None:
        origins = origins[: arguments.stop_after]
    if not origins:
        raise ValueError("no eligible rolling origins")
    first_index = panel_dates.index(origins[0])
    if first_index == 0:
        raise RuntimeError("first rolling origin has no previous trading session")
    previous_session = panel_dates[first_index - 1]
    initial_training_end = _checkpoint_training_end_date(arguments.initial_checkpoint)
    if initial_training_end != previous_session:
        raise RuntimeError(
            f"initial weights end {initial_training_end}, but the immediately "
            f"previous session is {previous_session}; rolling lineage has a gap"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    parent_path = arguments.initial_checkpoint
    parent_hash = arguments.initial_checkpoint_sha256
    started = time.monotonic()
    for index, origin in enumerate(origins, start=1):
        if origin < arguments.evaluation_start:
            lineage_path = arguments.output / "pre-evaluation" / f"{origin.isoformat()}.json"
            completed = _completed_pre_evaluation_fit(
                lineage_path,
                origin_date=origin,
                model=arguments.model,
                model_label=arguments.model_label,
                epochs=epochs,
                parent_sha256=parent_hash,
                dataset_manifest_sha256=arguments.dataset_manifest_sha256,
                epoch_selection_sha256=selection_sha256,
            )
            if completed is None:
                print(
                    f"rolling pre-evaluation {index}/{len(origins)} origin={origin} "
                    f"parent_sha256={parent_hash} elapsed_s={time.monotonic()-started:.1f}",
                    flush=True,
                )
                if _has_fresh_diffusion_target(panel, origin):
                    completed = _fit_pre_evaluation_date(
                        panel,
                        origin_date=origin,
                        model=arguments.model,
                        model_label=arguments.model_label,
                        parent_path=parent_path,
                        parent_sha256=parent_hash,
                        base_checkpoint=arguments.base_checkpoint,
                        base_checkpoint_sha256=arguments.base_checkpoint_sha256,
                        checkpoint_directory=arguments.checkpoint_directory,
                        observation_count=min(
                            arguments.observation_count, panel_dates.index(origin) + 1
                        ),
                        epochs=epochs,
                        epoch_selection_sha256=selection_sha256,
                        lineage_path=lineage_path,
                    )
                else:
                    completed = _carry_without_update(
                        lineage_path,
                        origin_date=origin,
                        model=arguments.model,
                        model_label=arguments.model_label,
                        epochs=epochs,
                        parent_path=parent_path,
                        parent_sha256=parent_hash,
                        dataset_manifest_sha256=arguments.dataset_manifest_sha256,
                        epoch_selection_sha256=selection_sha256,
                    )
            parent_path, parent_hash = completed
            print(
                f"rolling fit-only complete {index}/{len(origins)} origin={origin} "
                f"checkpoint_sha256={parent_hash} elapsed_s={time.monotonic()-started:.1f}",
                flush=True,
            )
            continue
        run_directory = arguments.output / origin.isoformat()
        completed = _completed_fit(
            run_directory,
            origin_date=origin,
            expected_parent_sha256=parent_hash,
        )
        if completed is None:
            command = [
                sys.executable, "-u", "-m",
                "quantdata.sdlv.daily.front_spot.hedging.runners.run_daily",
                "--model", arguments.model,
                "--dataset", str(arguments.dataset),
                "--dataset-manifest-sha256", arguments.dataset_manifest_sha256,
                "--surface-grid", "short-end-nms",
                "--model-label", arguments.model_label,
                "--label", f"{arguments.model}-rolling-{origin.isoformat()}",
                "--start", origin.isoformat(),
                "--end", origin.isoformat(),
                "--target-maturity", "1",
                "--objective", "mean-variance",
                "--risk-parameter", "10",
                "--mean-reduction", "demean",
                "--observation-count", str(arguments.observation_count),
                "--epochs", str(epochs),
                "--scenario-count", str(arguments.scenario_count),
                "--device", arguments.device,
                "--precision", "32",
                "--no-mixed-precision",
                "--progress-every", "1",
                "--retain-fit-checkpoints",
                "--warm-start-checkpoint", str(parent_path),
                "--warm-start-checkpoint-sha256", parent_hash,
                "--checkpoint-directory", str(arguments.checkpoint_directory),
                "--output", str(run_directory),
            ]
            if arguments.base_checkpoint is not None:
                command.extend(
                    ["--base-checkpoint", str(arguments.base_checkpoint),
                     "--base-checkpoint-sha256", arguments.base_checkpoint_sha256]
                )
            print(
                f"rolling diffusion {index}/{len(origins)} origin={origin} "
                f"parent_sha256={parent_hash} elapsed_s={time.monotonic()-started:.1f}",
                flush=True,
            )
            subprocess.run(command, check=True, env=os.environ.copy())
            completed = _completed_fit(
                run_directory,
                origin_date=origin,
                expected_parent_sha256=parent_hash,
            )
            if completed is None:
                raise RuntimeError(f"no completed dated record after fit: {origin}")
        parent_path, parent_hash = completed
        print(
            f"rolling diffusion complete {index}/{len(origins)} origin={origin} "
            f"checkpoint_sha256={parent_hash} elapsed_s={time.monotonic()-started:.1f}",
            flush=True,
        )


def _eligible(panel: PublicHedgingPanel, origin: date, observations: int) -> bool:
    try:
        panel.prior_one_day_origins(origin, count=observations)
        panel.one_day_transition(origin)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
