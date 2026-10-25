"""Authenticated consecutive-date checkpoint lineage for rolling VolGAN."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quantdata.hedging.metrics import HedgeSmokeResult
from quantdata.sdlv.daily.front_spot.hedging.models.checkpointing import file_sha256
from quantdata.torch import torch


class VolganRollingChain:
    """Advance only after an origin's result and complete fit are authenticated."""

    def __init__(
        self,
        *,
        initial_checkpoint: Path,
        initial_sha256: str,
        checkpoint_directory: Path,
        dataset_manifest_sha256: str,
    ) -> None:
        self.checkpoint_directory = checkpoint_directory.resolve()
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.checkpoint = initial_checkpoint.resolve()
        self.sha256 = initial_sha256
        if file_sha256(self.checkpoint) != self.sha256:
            raise ValueError("initial VolGAN checkpoint SHA-256 mismatch")
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("initial VolGAN checkpoint has another dataset")
        self.training_end_date = date.fromisoformat(payload["training_end_date"])
        self.previous_origin: date | None = None
        self.last_processed_date = self.training_end_date

    def source(self, origin_date: date) -> tuple[Path, str]:
        """Return the latest complete checkpoint for the next origin."""
        if origin_date <= self.last_processed_date:
            raise ValueError("VolGAN rolling source must predate the origin")
        return self.checkpoint, self.sha256

    def carry_gap(self, missing_date: date) -> str:
        """Carry weights unchanged when no valid one-session fit target exists."""
        if missing_date <= self.last_processed_date:
            raise ValueError("VolGAN rolling gap dates must be ordered")
        self.last_processed_date = missing_date
        return self.sha256

    def advance(
        self,
        origin_date: date,
        results: tuple[HedgeSmokeResult, ...],
    ) -> None:
        """Verify one dated record and its complete, chained model checkpoint."""
        if not results or (
            self.previous_origin is not None and origin_date <= self.previous_origin
        ):
            raise ValueError("VolGAN rolling records must be nonempty and ordered")
        diagnostics = tuple(result.diagnostics for result in results)
        source_hashes = {item.get("warm_start_sha256") for item in diagnostics}
        fit_hashes = {item.get("fit_checkpoint_sha256") for item in diagnostics}
        fit_paths = {item.get("fit_checkpoint_path") for item in diagnostics}
        if source_hashes != {self.sha256} or len(fit_hashes) != 1 or len(fit_paths) != 1:
            raise ValueError("VolGAN rolling record breaks checkpoint lineage")
        self.advance_checkpoint(
            origin_date,
            checkpoint_path=Path(next(iter(fit_paths))),
            checkpoint_sha256=next(iter(fit_hashes)),
        )

    def advance_checkpoint(
        self,
        origin_date: date,
        *,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> None:
        """Authenticate a completed fit without requiring scenario hedges."""
        if origin_date <= self.last_processed_date:
            raise ValueError("VolGAN rolling fits must be ordered")
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_relative_to(self.checkpoint_directory):
            raise ValueError("VolGAN rolling fit checkpoint is outside its run directory")
        if not isinstance(checkpoint_sha256, str) or file_sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError("VolGAN rolling fit checkpoint is missing or changed")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("training_end_date") != origin_date.isoformat()
            or payload.get("warm_start_source_sha256") != self.sha256
            or payload.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
            or int(payload.get("epoch", -1)) + 1
            < int(payload.get("train_config", {}).get("epochs", 0))
        ):
            raise ValueError("VolGAN rolling fit checkpoint is incomplete or misdated")
        self.checkpoint = checkpoint_path
        self.sha256 = checkpoint_sha256
        self.training_end_date = origin_date
        self.previous_origin = origin_date
        self.last_processed_date = origin_date
