"""Precompute and audit algebraic DLV initial states for a rolling window."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from quantdata.dysanos.flex.run_all_history_joint_direct import _geometry_from_completion
from quantdata.dysanos.v11.dlv_initialization import recover_initial_state
from quantdata.dysanos.v11.jax_geometry import with_model_business_dtes
from quantdata.dysanos.v11.licensed_market import load_certified_pricer
from quantdata.dysanos.v11.run_rolling_direct_ar import _geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--model-business-dtes", type=int, nargs="+")
    parser.add_argument("--allow-zero-backbone", action="store_true")
    args = parser.parse_args()
    geometry = _geometry(_geometry_from_completion(json.loads(args.completion.read_text()), requested_model_business_dtes=None), lower=-4, upper=3)
    geometry = with_model_business_dtes(geometry, args.model_business_dtes, allow_zero_backbone=args.allow_zero_backbone)
    dates = sorted(date.fromisoformat(p.stem) for p in (args.panel / "days").glob("*.parquet"))
    end = dates.index(args.end)
    selected = dates[end - args.count + 1:end + 1]
    if len(selected) != args.count:
        raise ValueError("Not enough available dates")
    args.output.mkdir(parents=True, exist_ok=False)
    states, audits = [], []
    started = monotonic()
    for index, day in enumerate(selected, 1):
        pricer, digest = load_certified_pricer(day, archive=args.archive)
        raw, audit = recover_initial_state(pricer, geometry=geometry)
        states.append(raw)
        audits.append({"date": str(day), "source_sha256": digest, **audit})
        if index % 10 == 0 or index == len(selected):
            elapsed = monotonic() - started
            print(f"recovered={index}/{len(selected)} elapsed={elapsed:.1f}s remaining_est={(len(selected)-index)*elapsed/index:.1f}s", flush=True)
    np.savez_compressed(args.output / "states.npz", dates=np.array([str(day) for day in selected]), raw_states=np.asarray(states), model_business_dtes=geometry.model_business_dtes)
    (args.output / "audit.json").write_text(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
