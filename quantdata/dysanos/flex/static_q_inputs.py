"""Load a frozen static FLEX decoder as a P-independent Q starting family."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from quantdata.dysanos.flex.config import (
    FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig,
    TriangularPcaDecoderConfig,
)
from quantdata.dysanos.flex.market_encoder import checkpoint_digest
from quantdata.dysanos.flex.model import LatentStateSanosModel
from quantdata.dysanos.flex.training import latent_checkpoint_proof
from quantdata.torch import BaseEnvironment


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _tuple_geometry_values(values: Mapping[str, Any]) -> dict[str, Any]:
    geometry = dict(values)
    for key in (
        "nms_range",
        "explicit_model_business_dtes",
        "inner_relative_strikes",
    ):
        if geometry.get(key) is not None:
            geometry[key] = tuple(geometry[key])
    return geometry


def load_static_q_inputs(
    checkpoint_path: Path,
    report_path: Path,
    *,
    env: BaseEnvironment,
) -> tuple[
    LatentStateSanosModel,
    tuple[date, ...],
    torch.Tensor,
    dict[str, str],
]:
    """Load only a certified static decoder and its fitted historical states.

    The returned model has no transition generator, innovation covariance, or
    historical likelihood.  Q code can therefore use its decoded front marginal
    and state table without importing any fitted P dynamics.
    """

    report = _load_json(report_path)
    report_status = report.get("status")
    if (
        not isinstance(report_status, str)
        or not report_status.endswith("_complete")
        or report.get("num_days") != 891
    ):
        raise ValueError("Static front-loaded report is not complete")
    dates = tuple(date.fromisoformat(str(value)) for value in report["market_dates"])
    if len(dates) != int(report["num_days"]) or len(set(dates)) != len(dates):
        raise ValueError("Static front-loaded report has malformed market dates")

    proof = latent_checkpoint_proof(checkpoint_path)
    latent_results = report.get("latent")
    if not isinstance(latent_results, list):
        raise ValueError("Static report has no latent-model results")
    matching = [
        value
        for value in latent_results
        if isinstance(value, Mapping)
        and isinstance(value.get("checkpoint_proof"), Mapping)
        and value["checkpoint_proof"].get("checkpoint_sha256")
        == proof["checkpoint_sha256"]
    ]
    if len(matching) != 1 or dict(matching[0]["checkpoint_proof"]) != proof:
        raise ValueError("Static decoder proof differs from its certified report")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = payload.get("model_identity")
    raw_config = identity.get("model_config") if isinstance(identity, Mapping) else None
    if (
        not isinstance(raw_config, Mapping)
        or identity.get("model_class") != "LatentStateSanosModel"
    ):
        raise ValueError("Static checkpoint has no latent decoder identity")
    raw_geometry = raw_config.get("geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("Static checkpoint has no fixed-grid geometry")
    geometry = FixedRelativeStrikeCappedUnboundedDlvFlexGeometryConfig(
        **_tuple_geometry_values(raw_geometry)
    )
    config = TriangularPcaDecoderConfig(
        label=cast(str, raw_config["label"]),
        state_sizes_by_expiry=tuple(raw_config["state_sizes_by_expiry"]),
        state_size=int(raw_config["state_size"]),
        group=cast(str, raw_config["group"]),
        user_version=cast(str, raw_config["user_version"]),
        geometry=geometry,
    )
    if asdict(config) != dict(raw_config):
        raise ValueError("Static checkpoint configuration differs from current code")
    if config.state_sizes_by_expiry != (100, 50, 25, 25):
        raise ValueError("Q requires the certified front-loaded allocation")

    model = LatentStateSanosModel(
        env,
        num_days=len(dates),
        config=config,
    )
    state_dict = cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
    model.load_state_dict(state_dict)
    states = model.daily_states.detach()
    if states.shape != (len(dates), 200):
        raise ValueError("Static decoder has an invalid historical state table")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return (
        model,
        dates,
        states,
        {
            "source_kind": "static_decoder_without_p_dynamics",
            "static_report": str(report_path),
            "static_report_sha256": checkpoint_digest(report_path),
            "static_checkpoint": str(checkpoint_path),
            "static_checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "static_state_sha256": cast(str, proof["latent_state_sha256"]),
        },
    )


__all__ = ["load_static_q_inputs"]
