"""Decode the DYSANOS native SANOS fits into aligned ATM-IV observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from quantdata.dysanos.flex.market_encoder import write_json_atomic
from quantdata.dysanos.flex.model import bound_native_raw
from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.flex.run_shared_strikes_full_encoder import (
    _market_dates_from_completion,
)
from quantdata.dysanos.v11.atm_spot_correlation import (
    _atm_iv_from_normalized_calls,
)
from quantdata.dysanos.v11.capped_model import load_capped_joint_model


def load_atm_history(
    *,
    history_path: Path,
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only DYSANOS native ATM data, never another project's calibration."""
    if history_path.suffix != ".npz":
        raise ValueError(
            "DYSANOS factorial statistics require native-atm-history.npz"
        )
    with np.load(history_path, allow_pickle=False) as payload:
        dates = payload["dates"].astype("U10")
        dtes = payload["business_dtes"].astype(np.float64)
        atm = payload["atm_iv"].astype(np.float64)
        spot = payload["spot_returns"].astype(np.float64)
    if (
        atm.shape != (len(dates), len(dtes))
        or spot.shape != (len(dates) - 1,)
        or np.any(~np.isfinite(atm))
        or np.any(atm <= 0.0)
        or np.any(~np.isfinite(spot))
    ):
        raise ValueError(f"invalid DYSANOS native ATM history: {history_path}")
    return dates, dtes, atm, spot


def run(
    *,
    joint_checkpoint: Path,
    native_completion: Path,
    native_checkpoint: Path,
    historical_dataset: Path,
    output: Path,
    report_output: Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, object]:
    """Price ATM calls from every independently fitted native SANOS state."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model, checkpoint_audit = load_capped_joint_model(joint_checkpoint, device=device)
    geometry = _geometry_from_completion(
        json.loads(native_completion.read_text(encoding="utf-8")),
        requested_model_business_dtes=None,
    )
    if geometry != model.config.geometry:
        raise ValueError("native and affine geometries differ")
    dates = _market_dates_from_completion(native_completion)
    payload = torch.load(native_checkpoint, map_location="cpu", weights_only=False)
    raw = payload["model_state_dict"]["daily_raw_embedding.weight"]
    if raw.shape != (len(dates), geometry.native_state_size):
        raise ValueError("native raw-state count or dimension differs from completion")
    observations = pl.read_parquet(
        historical_dataset / "observations.parquet",
        columns=["date", "forward_adjusted_spot_return_into_date"],
    )
    observed_dates = [value.isoformat() for value in observations["date"]]
    if observed_dates != [value.isoformat() for value in dates]:
        raise ValueError("native SANOS and spot-observation dates do not align")
    spot = np.asarray(
        observations["forward_adjusted_spot_return_into_date"][1:].to_list(),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(spot)) or np.any(spot <= -1.0):
        raise ValueError("invalid forward-adjusted spot returns")
    expiries_y = np.asarray(geometry.model_expiries_y, dtype=np.float64)
    calls: list[np.ndarray] = []
    with torch.inference_mode():
        for first in range(0, len(dates), batch_size):
            last = min(first + batch_size, len(dates))
            batch = raw[first:last].to(device=model.device, dtype=model.fdtype)
            parameters = bound_native_raw(batch, geometry=geometry)
            surface = model.sanos_decoder.forward_batch(
                dlvsqs=parameters.dlvsqs,
                forward_vols=parameters.forward_vols,
                eta=parameters.eta,
            )
            expiries = torch.as_tensor(
                expiries_y, dtype=model.fdtype, device=model.device
            ).unsqueeze(0).expand(len(batch), -1)
            batch_calls = model.sanos_decoder.price_calls_batch(
                surface,
                expiries_y=expiries,
                strikes=torch.ones_like(expiries),
            )
            calls.append(batch_calls.cpu().double().numpy())
            print(f"native ATM decoded={last}/{len(dates)}", flush=True)
    atm = _atm_iv_from_normalized_calls(np.concatenate(calls), expiries_y)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dates=np.asarray([value.isoformat() for value in dates], dtype="U10"),
        business_dtes=expiries_y * 252.0,
        atm_iv=atm,
        spot_returns=spot,
    )
    report: dict[str, object] = {
        "schema": "dysanos-v11-native-atm-history-v1",
        "checkpoint": checkpoint_audit,
        "native_completion": str(native_completion.resolve()),
        "native_checkpoint": str(native_checkpoint.resolve()),
        "historical_dataset": str(historical_dataset.resolve()),
        "output": str(output.resolve()),
        "dates": len(dates),
        "business_dtes": (expiries_y * 252.0).tolist(),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
    }
    write_json_atomic(report, report_output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = run(**vars(arguments))
    print(f"DYSANOS native ATM history: {report['output']}", flush=True)


if __name__ == "__main__":
    main()
