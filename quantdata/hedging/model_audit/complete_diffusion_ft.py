"""Continue the saved April FT chain through May without running any hedges."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantdata.hedging.model_audit.inference import checkpoint_payload
from quantdata.hedging.model_audit.run import GRID, _hash, _json, _record, build_history
from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import SanosFixedGridHistory
from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionTrainConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.data import (
    diffusion_live_context_from_history,
    diffusion_sequences_from_history,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.training import fit_diffusion
from quantdata.torch import BaseEnvironment


def _window(history: SanosFixedGridHistory, index: int) -> SanosFixedGridHistory:
    selected = slice(index - 251, index + 1)
    returns = history.forward_log_returns_into_date[selected].copy()
    sessions = history.sessions_from_previous_observation[selected].copy()
    returns[0], sessions[0] = np.nan, 0
    return SanosFixedGridHistory(dates=history.dates[selected],
        log_implied_volatilities=history.log_implied_volatilities[selected],
        implied_vol_is_interior=history.implied_vol_is_interior[selected],
        forward_log_returns_into_date=returns,
        sessions_from_previous_observation=sessions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["QUANTDATA_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "sample"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--path-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()
    panel = PublicHedgingPanel(arguments.root / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    source = arguments.root / "experiments/hedging/diffusion-ft-gpu-20261027-v1/2024-04-30/records/2024-04-30.json"
    diagnostics = _record(source)["diagnostics"]
    parent, parent_hash = Path(diagnostics["fit_checkpoint_path"]), diagnostics["fit_checkpoint_sha256"]
    saved = checkpoint_payload(parent, expected_sha256=parent_hash, audit_directory=arguments.output)
    base_lineage = arguments.root / "experiments/hedging/short-end-regeneration-20261024/diffusion-base-rolling-fixed-v2/pre-evaluation/2023-01-03.json"
    lineage = json.loads(base_lineage.read_text())
    base, base_hash = Path(lineage["parent_path"]), lineage["parent_sha256"]
    config = DiffusionModelConfig(label="diffusion-ft-short-end-nms35-2022", grid=GRID,
                                 variant=DiffusionVariant.AD_SEQ_VOL_FT)
    if (saved["model_code_version"] != config.version.unique_id48
            or saved["model_config_identity"] != config.filename
            or saved["dataset_manifest_sha256"] != panel.dataset_manifest_sha256
            or saved["training_end_date"] != "2024-04-30"
            or saved["base_checkpoint_fingerprint"] != base_hash or _hash(base) != base_hash):
        raise ValueError("April FT source, base, geometry or dataset identity mismatch")
    training = DiffusionTrainConfig(**saved["train_config"])
    if arguments.stage == "train" and saved.get("inference_only") is True:
        raise ValueError("Exact training-restart state was retired after inference validation; use stage=sample for these fitted models")
    history = build_history(panel, arguments.output)
    origins = [index for index, day in enumerate(history.dates) if date(2024, 5, 1) <= day <= date(2024, 5, 31)]
    if len(origins) != 22 or training.epochs != 1:
        raise ValueError("Expected 22 May origins and the retained one-epoch rolling configuration")
    work = arguments.output / "completion/ft-may"
    print(json.dumps({"stage": arguments.stage, "origins": len(origins), "epochs_per_date": training.epochs,
        "dataset_manifest_sha256": panel.dataset_manifest_sha256, "initial_checkpoint": str(parent),
        "initial_sha256": parent_hash, "base_checkpoint": str(base), "base_sha256": base_hash,
        "device": arguments.device, "hedges_run": False}), flush=True)
    if arguments.preflight:
        data = diffusion_sequences_from_history(_window(history, origins[0]))
        print(f"preflight OK: imports, data root and authenticated sources; first date has {len(data.target_dates)} causal training sequences", flush=True)
        return
    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"signal {signum}: stopping after the current saved date/batch", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    environment = BaseEnvironment(precision=32, device=arguments.device, debug=False)
    started = time.monotonic()
    for position, index in enumerate(origins, 1):
        day = history.dates[index]
        record_path = work / f"dates/{day}.json"
        if arguments.stage == "train":
            if record_path.exists():
                record = json.loads(record_path.read_text())
                checkpoint = Path(record["checkpoint"])
                if (record["parent_sha256"] != parent_hash or _hash(checkpoint) != record["checkpoint_sha256"]
                        or record["origin_date"] != str(day)):
                    raise ValueError(f"Existing FT continuation lineage mismatch at {day}")
            else:
                data = diffusion_sequences_from_history(_window(history, index))
                result = fit_diffusion(data, model_config=config, checkpoint_directory=work / "checkpoints",
                    train_config=training, base_checkpoint=base, warm_start_checkpoint=parent,
                    expected_base_checkpoint_sha256=base_hash, expected_warm_start_sha256=parent_hash,
                    dataset_manifest_sha256=panel.dataset_manifest_sha256, environment=environment)
                if not result.converged or result.training_end_date != day or result.warm_start_sha256 != parent_hash:
                    raise RuntimeError(f"FT continuation did not complete its causal rolling fit at {day}")
                checkpoint = result.checkpoint_path
                record = {"origin_date": str(day), "checkpoint": str(checkpoint), "checkpoint_sha256": _hash(checkpoint),
                    "parent_checkpoint": str(parent), "parent_sha256": parent_hash, "base_checkpoint": str(base),
                    "base_sha256": base_hash, "dataset_manifest_sha256": panel.dataset_manifest_sha256,
                    "epochs": result.completed_epochs, "training_sequences": len(data.target_dates),
                    "validation_loss": result.validation_loss, "training_loss": result.training_loss,
                    "normalization_policy": "source-fixed-v1", "device": arguments.device}
                _json(record_path, record)
            parent, parent_hash = checkpoint, record["checkpoint_sha256"]
        else:
            record = json.loads(record_path.read_text())
            destination = arguments.output / f"regenerated/AD-Seq-Vol-FT/{day}.npz"
            if not destination.exists():
                checkpoint = Path(record["checkpoint"])
                weights = checkpoint_payload(checkpoint, expected_sha256=record["checkpoint_sha256"], audit_directory=arguments.output)
                model = SequentialVolatilityDenoiser(environment, config=config,
                    channel_mean=weights["ema_model"]["channel_mean"].to(device=environment.device, dtype=environment.fdtype),
                    channel_scale=weights["ema_model"]["channel_scale"].to(device=environment.device, dtype=environment.fdtype))
                model.load_state_dict(weights["ema_model"])
                model.eval()
                context = diffusion_live_context_from_history(_window(history, index), as_of_date=day,
                                                              forecast_date=panel.next_one_session_date(day))
                returns, surfaces = [], []
                batches = (arguments.path_count + arguments.batch_size - 1) // arguments.batch_size
                for batch in range(batches):
                    batch_path = work / f"sample-batches/{day}/{batch:03d}.npz"
                    if not batch_path.exists():
                        sample = sample_diffusion_transitions(SimpleNamespace(ema_model=model), observed_sequence=context.sequence,
                            path_count=min(arguments.batch_size, arguments.path_count - batch * arguments.batch_size), seed=20_261_111 + batch)
                        batch_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(batch_path, forward_log_returns=sample.forward_log_returns,
                                            next_log_implied_volatilities=sample.next_log_implied_volatilities)
                    with np.load(batch_path, allow_pickle=False) as sample:
                        returns.append(sample["forward_log_returns"])
                        surfaces.append(sample["next_log_implied_volatilities"])
                    print(f"FT sampling {position}/{len(origins)} {day} batch {batch+1}/{batches} elapsed={time.monotonic()-started:.1f}s", flush=True)
                    if stop_requested:
                        return
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, forward_log_returns=np.concatenate(returns),
                    next_log_implied_volatilities=np.concatenate(surfaces), metadata_json=json.dumps({**record,
                        "path_count": arguments.path_count, "batch_size": arguments.batch_size, "device": arguments.device,
                        "sampling": "new draws from causally continued saved FT EMA; no hedge optimization"}))
        elapsed = time.monotonic() - started
        print(f"FT {arguments.stage} complete {position}/{len(origins)} {day} elapsed={elapsed:.1f}s remaining={elapsed/position*(len(origins)-position):.1f}s", flush=True)
        if stop_requested:
            return


if __name__ == "__main__":
    main()
