"""Fit a resumable, strictly consecutive VolGAN checkpoint chain without hedging."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import (
    GenerativeSurfaceGrid,
)
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import (
    PublicHedgingPanel,
    default_public_candidate_path,
)
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.adapter import (
    fit_public_volgan,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.config import (
    VolganPcaModelConfig,
    VolganTrainConfig,
)
from quantdata.sdlv.daily.front_spot.hedging.models.volgan.data import (
    has_exact_one_day_target,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.daily import (
    dataset_manifest_sha256,
)
from quantdata.sdlv.daily.front_spot.hedging.runners.volgan_rolling_chain import (
    VolganRollingChain,
)
from quantdata.torch import BaseEnvironment


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Fit every eligible date, reusing only its immediate predecessor's weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_public_candidate_path())
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-count", type=int, default=252)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--progress-seconds", type=float, default=240.0)
    arguments = parser.parse_args()
    if arguments.start > arguments.end:
        parser.error("start follows end")
    if arguments.stop_after is not None and arguments.stop_after < 1:
        parser.error("stop-after must be positive")
    panel = PublicHedgingPanel(arguments.dataset, validate=False)
    manifest_sha256 = dataset_manifest_sha256(panel)
    grid = GenerativeSurfaceGrid.short_end_nms()
    model_config = VolganPcaModelConfig(label=arguments.label, grid=grid, pca_rank=8)
    defaults = VolganTrainConfig()
    train_config = VolganTrainConfig(
        epochs=arguments.epochs,
        gradient_matching_epochs=0,
        batch_size=defaults.batch_size,
        generator_learning_rate=defaults.generator_learning_rate,
        discriminator_learning_rate=defaults.discriminator_learning_rate,
        optimizer=defaults.optimizer,
        validation_fraction=defaults.validation_fraction,
        seed=defaults.seed,
        progress_seconds=arguments.progress_seconds,
        checkpoint_epochs=min(defaults.checkpoint_epochs, arguments.epochs),
    )
    origins = []
    gaps: dict[date, str] = {}
    for origin_date in panel.dates:
        if not arguments.start <= origin_date <= arguments.end:
            continue
        try:
            panel.prior_one_day_origins(origin_date, count=arguments.observation_count)
            panel.one_day_transition(origin_date)
        except (IndexError, KeyError, ValueError) as error:
            gaps[origin_date] = f"{type(error).__name__}: {error}"
            continue
        if not has_exact_one_day_target(panel, origin_date):
            gaps[origin_date] = "latest VolGAN spot/IV target lacks three exact one-session lags"
            continue
        origins.append(origin_date)
    if not origins:
        raise ValueError("no eligible rolling VolGAN origins")
    run_identity = {
        "schema": "hedging-volgan-rolling-fit-chain-v1",
        "dataset_manifest_sha256": manifest_sha256,
        "start": arguments.start.isoformat(),
        "end": arguments.end.isoformat(),
        "label": arguments.label,
        "initial_checkpoint": str(arguments.initial_checkpoint.resolve()),
        "initial_sha256": arguments.initial_sha256,
        "checkpoint_directory": str(arguments.checkpoint_directory.resolve()),
        "observation_count": arguments.observation_count,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "eligible_origins": [origin.isoformat() for origin in origins],
        "skipped_dates": {origin.isoformat(): reason for origin, reason in gaps.items()},
    }
    run_identity = json.loads(json.dumps(run_identity))
    run_path = arguments.output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != run_identity:
            raise ValueError("rolling VolGAN output directory belongs to another run")
    else:
        _write_json(run_path, run_identity)
    chain = VolganRollingChain(
        initial_checkpoint=arguments.initial_checkpoint,
        initial_sha256=arguments.initial_sha256,
        checkpoint_directory=arguments.checkpoint_directory,
        dataset_manifest_sha256=manifest_sha256,
    )
    selected = origins if arguments.stop_after is None else origins[: arguments.stop_after]
    started = time.monotonic()
    environment = BaseEnvironment(precision=64, device="cpu", debug=False)
    selected_set = set(selected)
    index = 0
    for origin_date in panel.dates:
        if not arguments.start <= origin_date <= selected[-1]:
            continue
        if origin_date not in selected_set:
            gap_record = {
                "origin_date": origin_date.isoformat(),
                "reason": gaps[origin_date],
                "carried_checkpoint": str(chain.checkpoint),
                "carried_sha256": chain.sha256,
            }
            gap_path = arguments.output / "gaps" / f"{origin_date.isoformat()}.json"
            if gap_path.exists():
                if json.loads(gap_path.read_text(encoding="utf-8")) != gap_record:
                    raise ValueError(f"rolling VolGAN gap record changed: {gap_path}")
            else:
                _write_json(gap_path, gap_record)
            chain.carry_gap(origin_date)
            print(
                f"rolling VolGAN no-update gap origin={origin_date} "
                f"source_sha256={chain.sha256} reason={gaps[origin_date]}",
                flush=True,
            )
            continue
        index += 1
        source_path, source_sha256 = chain.source(origin_date)
        record_path = arguments.output / "records" / f"{origin_date.isoformat()}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                record.get("origin_date") != origin_date.isoformat()
                or record.get("warm_start_sha256") != source_sha256
            ):
                raise ValueError(f"rolling VolGAN record breaks lineage: {record_path}")
            checkpoint_path = Path(record["checkpoint_path"])
            checkpoint_sha256 = record["checkpoint_sha256"]
            reused = True
        else:
            fit = fit_public_volgan(
                panel,
                as_of_date=origin_date,
                model_config=model_config,
                checkpoint_directory=arguments.checkpoint_directory,
                observation_count=arguments.observation_count,
                train_config=train_config,
                warm_start_checkpoint=source_path,
                expected_warm_start_sha256=source_sha256,
                environment=environment,
            )
            checkpoint_path = fit.result.checkpoint_path
            checkpoint_sha256 = file_sha256(checkpoint_path)
            reused = False
        chain.advance_checkpoint(
            origin_date,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
        )
        if not reused:
            _write_json(
                record_path,
                {
                    "origin_date": origin_date.isoformat(),
                    "warm_start_checkpoint": str(source_path),
                    "warm_start_sha256": source_sha256,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "completed_epochs": fit.result.completed_epochs,
                    "validation_loss": fit.result.validation_loss,
                },
            )
        elapsed = time.monotonic() - started
        remaining = len(selected) - index
        print(
            f"rolling VolGAN fit {index}/{len(selected)} origin={origin_date} "
            f"{'reused' if reused else 'computed'} elapsed_s={elapsed:.1f} "
            f"eta_s={elapsed * remaining / index:.1f}",
            flush=True,
        )
    _write_json(
        arguments.output / "summary.json",
        {
            "schema": "hedging-volgan-rolling-fit-chain-summary-v1",
            "completed_origins": [origin.isoformat() for origin in selected],
            "final_checkpoint": str(chain.checkpoint),
            "final_sha256": chain.sha256,
        },
    )


if __name__ == "__main__":
    main()
