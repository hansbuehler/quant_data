"""Construct a fixed relative-strike grid from a completed FLEX model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from quantdata.dysanos.flex.config import (
    JointDirectDynamicsConfig,
    SharedStrikeCappedUnboundedDlvFlexGeometryConfig,
)
from quantdata.dysanos.flex.model import JointDirectDynamicsModel
from quantdata.dysanos.flex.sanos import uses_shared_strikes
from quantdata.torch import BaseEnvironment


@dataclass(frozen=True)
class RepresentativeRelativeStrikeGrid:
    """One immutable grid and its source-checkpoint provenance."""

    inner_relative_strikes: tuple[float, ...]
    source_checkpoint_sha256: str
    num_source_dates: int
    atm_index: int
    grid_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def representative_relative_strike_grid(
    checkpoint_path: Path,
) -> RepresentativeRelativeStrikeGrid:
    """Return the coordinate-wise median log-grid from a completed Direct model."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "flex-joint-direct-dynamics-v1":
        raise ValueError("Representative-grid source is not a joint Direct checkpoint")
    if payload.get("phase") != "complete":
        raise ValueError("Representative-grid source checkpoint is not complete")
    config_payload = payload.get("model_config")
    if not isinstance(config_payload, dict):
        raise ValueError("Representative-grid source has no model configuration")
    geometry_payload = config_payload.get("geometry")
    if not isinstance(geometry_payload, dict):
        raise ValueError("Representative-grid source has no geometry configuration")
    geometry_values = dict(geometry_payload)
    explicit_dtes = geometry_values.get("explicit_model_business_dtes")
    if explicit_dtes is not None:
        geometry_values["explicit_model_business_dtes"] = tuple(explicit_dtes)
    nms_range = geometry_values.get("nms_range")
    if nms_range is not None:
        geometry_values["nms_range"] = tuple(nms_range)
    geometry = SharedStrikeCappedUnboundedDlvFlexGeometryConfig(**geometry_values)
    if not uses_shared_strikes(geometry):
        raise ValueError("Representative-grid source must use one grid per date")

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Representative-grid source has no model state")
    daily_states = state_dict.get("daily_state_embedding.weight")
    if not isinstance(daily_states, torch.Tensor) or daily_states.ndim != 2:
        raise ValueError("Representative-grid source has no daily state history")
    num_dates, state_size = daily_states.shape
    config = JointDirectDynamicsConfig(
        label=cast(str, config_payload["label"]),
        state_size=state_size,
        geometry=geometry,
    )
    env = BaseEnvironment(precision=32, device="cpu", debug=False)
    model = JointDirectDynamicsModel(
        env,
        num_days=num_dates,
        config=config,
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    model.eval()
    with torch.no_grad():
        _, _, surfaces = model.decode_days(torch.arange(num_dates))
        supports = surfaces.support_strikes[:, 0, 1:-1].cpu().double().numpy()
    if not np.allclose(
        surfaces.support_strikes[:, 1:].cpu().numpy(),
        surfaces.support_strikes[:, :1].cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Source model does not use one strike grid within each date")

    representative_logs = np.median(np.log(supports), axis=0)
    atm_index = int(np.argmin(np.abs(representative_logs)))
    representative_logs[atm_index] = 0.0
    if not np.all(np.diff(representative_logs) > 0.0):
        raise RuntimeError("Median representative log-grid is not strictly increasing")
    strikes = np.exp(representative_logs)
    strikes[atm_index] = 1.0
    values = tuple(float(value) for value in strikes)
    grid_sha256 = hashlib.sha256(strikes.astype("<f8").tobytes()).hexdigest()
    return RepresentativeRelativeStrikeGrid(
        inner_relative_strikes=values,
        source_checkpoint_sha256=_sha256(checkpoint_path),
        num_source_dates=num_dates,
        atm_index=atm_index,
        grid_sha256=grid_sha256,
    )


__all__ = [
    "RepresentativeRelativeStrikeGrid",
    "representative_relative_strike_grid",
]
