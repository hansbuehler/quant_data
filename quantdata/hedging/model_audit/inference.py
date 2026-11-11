"""Checksummed, exact EMA-only diffusion archives with shared frozen backbones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantdata.sdlv.daily.front_spot.hedging.data.generative_surfaces import GenerativeSurfaceGrid
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.config import (
    DiffusionModelConfig,
    DiffusionVariant,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.hedger import (
    sample_diffusion_transitions,
)
from quantdata.sdlv.daily.front_spot.hedging.models.diffusion.model import (
    SequentialVolatilityDenoiser,
)
from quantdata.torch import BaseEnvironment, torch

SCHEMA = "hedging-diffusion-inference-v1"


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_identity(state: dict[str, torch.Tensor]) -> str:
    """Content identity preserves names, shapes, dtypes and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(values.dtype), list(values.shape)]).encode())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _blob(directory: Path, state: dict[str, torch.Tensor]) -> dict:
    destination = directory / f"{tensor_identity(state)}.pt"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.pt")
        torch.save({name: tensor.detach().cpu().contiguous().clone() for name, tensor in state.items()}, temporary)
        temporary.replace(destination)
    return {"path": str(destination), "sha256": fingerprint(destination), "bytes": destination.stat().st_size}


def export_checkpoint(source: Path, *, expected_sha256: str, config: DiffusionModelConfig, directory: Path) -> dict:
    """Export exactly the EMA used in inference; never change a source checkpoint."""
    if fingerprint(source) != expected_sha256:
        raise ValueError(f"Original checkpoint checksum changed: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (payload["model_config_identity"] != config.filename
            or payload["model_code_version"] != config.version.unique_id48
            or payload["model_user_version"] != config.user_version):
        raise ValueError(f"Original diffusion model identity mismatch: {source}")
    weights = payload["ema_model"]
    backbone = {name: tensor for name, tensor in weights.items() if not name.startswith("output_adapter.")}
    adapter = {name: tensor for name, tensor in weights.items() if name.startswith("output_adapter.")}
    retained = {name: payload[name] for name in ("checkpoint_format_version", "model_code_version", "model_user_version",
        "model_config_identity", "base_compatibility_identity", "variant", "data_fingerprint", "precision",
        "base_checkpoint_fingerprint", "training_start_date", "training_end_date", "warm_start_source_sha256",
        "normalization_policy", "dataset_manifest_sha256", "train_config", "epoch", "global_step", "best_validation") if name in payload}
    manifest = {"schema": SCHEMA, "inference_only": True, "source_path": str(source), "source_sha256": expected_sha256,
        "source_bytes": source.stat().st_size, "config": asdict(config), "metadata": retained,
        "backbone": _blob(directory / "blobs", backbone), "adapter": _blob(directory / "blobs", adapter) if adapter else None}
    destination = directory / f"models/{config.variant.value}/{payload['training_end_date']}.json"
    _write_json(destination, manifest)
    return {"manifest": str(destination), "manifest_sha256": fingerprint(destination), "source_path": str(source),
        "source_sha256": expected_sha256, "source_bytes": source.stat().st_size, "source_retired": False,
        "origin_date": payload["training_end_date"], "variant": config.variant.value}


def load_export(manifest_path: Path, *, expected_sha256: str | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    """Load safe tensor dictionaries only after authenticating their dated manifest."""
    if expected_sha256 is not None and fingerprint(manifest_path) != expected_sha256:
        raise ValueError("Inference manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("inference_only") is not True:
        raise ValueError("Unsupported inference archive")
    weights = {}
    for name in ("backbone", "adapter"):
        blob = manifest[name]
        if blob is None:
            continue
        path = Path(blob["path"])
        if fingerprint(path) != blob["sha256"]:
            raise ValueError(f"Inference tensor blob checksum mismatch: {path}")
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        if set(weights) & set(tensors):
            raise ValueError("Duplicate inference tensor keys")
        weights.update(tensors)
    return manifest, weights


def model_from_export(manifest_path: Path, *, environment: BaseEnvironment,
                      expected_sha256: str | None = None) -> SequentialVolatilityDenoiser:
    """Reconstruct an inference model without an optimizer, training data or base checkpoint."""
    manifest, weights = load_export(manifest_path, expected_sha256=expected_sha256)
    fields = manifest["config"].copy()
    fields["grid"] = GenerativeSurfaceGrid(**{name: tuple(values) for name, values in fields["grid"].items()})
    fields["variant"] = DiffusionVariant(fields["variant"])
    config = DiffusionModelConfig(**fields)
    if (config.version.unique_id48 != manifest["metadata"]["model_code_version"]
            or config.filename != manifest["metadata"]["model_config_identity"]):
        raise ValueError("Inference archive does not match the installed model geometry/version")
    model = SequentialVolatilityDenoiser(environment, config=config,
        channel_mean=weights["channel_mean"].to(device=environment.device, dtype=environment.fdtype),
        channel_scale=weights["channel_scale"].to(device=environment.device, dtype=environment.fdtype))
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def checkpoint_payload(source: Path, *, expected_sha256: str, audit_directory: Path) -> dict:
    """Resolve historical source identities after explicitly recorded inference-only retirement."""
    if source.exists():
        if fingerprint(source) != expected_sha256:
            raise ValueError(f"Source checkpoint checksum mismatch: {source}")
        return torch.load(source, map_location="cpu", weights_only=False)
    catalog = json.loads((audit_directory / "INFERENCE_MODELS.json").read_text())
    entry = catalog["models"][str(source)]
    if entry["source_sha256"] != expected_sha256 or entry.get("verified_inference") is not True:
        raise ValueError("Retired checkpoint has no authenticated, verified inference replacement")
    manifest, weights = load_export(Path(entry["manifest"]), expected_sha256=entry["manifest_sha256"])
    return {**manifest["metadata"], "ema_model": weights, "inference_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prune", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output
    from quantdata.hedging.model_audit.run import build_history
    from quantdata.sdlv.daily.front_spot.hedging.data.market_panel import PublicHedgingPanel

    directory = output / "inference"
    registry = json.loads((output / "SOURCE_INPUTS.json").read_text())["files"]
    sources = [row for row in registry if "rolling neural checkpoint" in row["roles"]
        and any(model in row["models"] for model in ("AD-Seq-Vol", "AD-Seq-Vol-FT"))]
    catalog_path = output / "INFERENCE_MODELS.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"schema": SCHEMA, "models": {}}
    panel = PublicHedgingPanel(Path(os.environ["QUANTDATA_ROOT"]) / "datasets/hedging/spx-2022-2025-hedging-public-candidate-v1")
    history = build_history(panel, output)
    dates, surfaces = [str(day) for day in history.dates], history.log_implied_volatilities
    returns, sessions = history.forward_log_returns_into_date, history.sessions_from_previous_observation
    environment = BaseEnvironment(precision=32, device="cpu", debug=False)
    started = time.monotonic()
    for position, source in enumerate(sources, 1):
        original = Path(source["path"])
        entry = catalog["models"].get(str(original))
        if entry is not None and entry.get("verified_inference") is True:
            continue
        variant = DiffusionVariant.AD_SEQ_VOL_FT if "AD-Seq-Vol-FT" in source["models"] else DiffusionVariant.AD_SEQ_VOL
        config = DiffusionModelConfig(label=f"diffusion-{'ft' if variant is DiffusionVariant.AD_SEQ_VOL_FT else 'base'}-short-end-nms35-2022",
                                      grid=GenerativeSurfaceGrid.short_end_nms(), variant=variant)
        entry = export_checkpoint(original, expected_sha256=source["sha256"], config=config, directory=directory)
        payload = torch.load(original, map_location="cpu", weights_only=False)
        exported = model_from_export(Path(entry["manifest"]), expected_sha256=entry["manifest_sha256"], environment=environment)
        state = exported.state_dict()
        if set(state) != set(payload["ema_model"]) or any(not torch.equal(state[name], tensor) for name, tensor in payload["ema_model"].items()):
            raise RuntimeError(f"Exported inference tensors differ: {original}")
        reference = SequentialVolatilityDenoiser(environment, config=config,
            channel_mean=payload["ema_model"]["channel_mean"], channel_scale=payload["ema_model"]["channel_scale"])
        reference.load_state_dict(payload["ema_model"])
        reference.eval()
        index = dates.index(entry["origin_date"])
        observed = np.stack((surfaces[index-20:index+1], np.broadcast_to(
            (returns[index-20:index+1] / np.sqrt(sessions[index-20:index+1]))[:, None, None], surfaces[index-20:index+1].shape)), axis=1)
        before = sample_diffusion_transitions(SimpleNamespace(ema_model=reference), observed_sequence=observed, path_count=2, seed=20_261_111)
        after = sample_diffusion_transitions(SimpleNamespace(ema_model=exported), observed_sequence=observed, path_count=2, seed=20_261_111)
        if (not np.array_equal(before.forward_log_returns, after.forward_log_returns)
                or not np.array_equal(before.next_log_implied_volatilities, after.next_log_implied_volatilities)
                or not np.isfinite(after.next_log_implied_volatilities).all() or not np.isfinite(after.forward_log_returns).all()):
            raise RuntimeError(f"Export failed exact two-path, 200-step inference comparison: {original}")
        entry.update({"verified_inference": True, "validation_device": "cpu", "validation_paths": 2,
            "validation_diffusion_steps": config.diffusion_steps, "max_tensor_difference": 0., "max_sample_difference": 0.})
        catalog["models"][str(original)] = entry
        _write_json(catalog_path, catalog)
        elapsed = time.monotonic() - started
        print(f"inference export verified {position}/{len(sources)} {entry['variant']} {entry['origin_date']} elapsed={elapsed:.1f}s remaining={elapsed/position*(len(sources)-position):.1f}s", flush=True)
    if arguments.prune:
        deleted_bytes = 0
        allowed = ((Path(os.environ["QUANTDATA_ROOT"]) / "training/hedging").resolve(), (output / "completion/ft-may/checkpoints").resolve())
        for entry in catalog["models"].values():
            source = Path(entry["source_path"])
            if not source.exists():
                continue
            if (not entry.get("verified_inference") or source.is_symlink() or source.stat().st_uid != os.getuid()
                    or not any(source.resolve().is_relative_to(parent) for parent in allowed)
                    or fingerprint(source) != entry["source_sha256"]):
                raise RuntimeError(f"Refusing unsafe/unverified checkpoint deletion: {source}")
            load_export(Path(entry["manifest"]), expected_sha256=entry["manifest_sha256"])
            probes = [source, Path(str(source) + ".lock")]
            if any(path.exists() and subprocess.run(["fuser", "-s", str(path)], capture_output=True).returncode != 1 for path in probes):
                entry["retirement_deferred"] = "checkpoint or training lock is open, or open-file check was inconclusive"
                _write_json(catalog_path, catalog)
                continue
            size = source.stat().st_size
            source.unlink()
            entry["source_retired"] = True
            deleted_bytes += size
            _write_json(catalog_path, catalog)
        catalog["deleted_source_bytes"] = sum(entry["source_bytes"] for entry in catalog["models"].values() if entry["source_retired"])
        print(f"Removed verified redundant checkpoints: {deleted_bytes/2**30:.3f} GiB gross", flush=True)
    catalog["inference_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    _write_json(catalog_path, catalog)


if __name__ == "__main__":
    main()
