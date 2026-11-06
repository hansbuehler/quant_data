"""Export the immutable Q1 training sample as a portable, deliberately retained artifact."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quantdata.sdlv.dhgromm1.contracts import canonical_gains
from quantdata.sdlv.dhgromm1.datasets.q1_20250829_100k.dataset import (
    SCHEMA,
    Dataset,
    sha256,
    verify_arrays,
)
from quantdata.sdlv.dhgromm1.persistence import atomic_json


def export_dataset(*, scenarios: Path, marks: Path, normalization: Path, output: Path) -> dict:
    """Lossless chunk export, preserving order, raw gains and frozen normalizers.

    No path generation, market interpolation, training, rounding, option
    repricing or new normalization fit occurs here. Canonicalize the explicitly
    identified old premium-return schema exactly once before publishing.
    """
    scenario = json.loads((scenarios/"identity.json").read_text())
    mark_identity = json.loads((marks/"identity.json").read_text())
    complete = json.loads((marks/"complete.json").read_text())
    source_identity_hash = sha256(scenarios/"identity.json")
    if (scenario.get("measure","Q")!="Q" or scenario["substeps"]!=1 or complete.get("status")!="complete"
            or mark_identity["scenario_identity_sha256"]!=source_identity_hash
            or mark_identity["sample"]!="train" or mark_identity["paths"]!=scenario["paths_per_sample"]
            or complete["chunks"]==[]):
        raise ValueError("expected matching complete original Q1 training sample and causal marks")
    if any(complete.get(k)!=v for k,v in mark_identity.items()):
        raise ValueError("mark identity and completion records differ")
    output.mkdir(parents=True,exist_ok=False)
    (output/"chunks").mkdir()
    records,completed = [],0
    started = time.monotonic()
    for record in complete["chunks"]:
        start,count = record["start"],record["count"]
        if start!=completed or count%2:
            raise ValueError("source order does not preserve complete original pairs")
        source = scenarios/"train"/f"{start:06d}.npz"
        source_digest = sha256(source)
        if source_digest!=record["scenario_chunk_sha256"]:
            raise ValueError("original scenario chunk changed")
        with np.load(source,allow_pickle=False) as data:
            prices,payoff = data["prices"],data["payoff"]
            gains = canonical_gains(prices,data["returns"],identity=scenario)
        mark_path = marks/source.name
        with np.load(mark_path,allow_pickle=False) as data:
            changes = data["changes"]
            if json.loads(str(data["record"].item()))!=record:
                raise ValueError("causal mark record changed")
        arrays = dict(prices=prices,gains=gains,payoff=payoff,changes_packed=changes)
        verify_arrays(arrays,paths=count)
        target = output/"chunks"/source.name
        np.savez_compressed(target,**arrays)
        records.append(dict(file=str(target.relative_to(output)),start=start,count=count,sha256=sha256(target),
            bytes=target.stat().st_size,source_scenario_sha256=source_digest,source_marks_sha256=sha256(mark_path)))
        completed += count
        elapsed = time.monotonic()-started
        print(json.dumps(dict(stage="export-portable-training-data",completed=completed,total=mark_identity["paths"],
            elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(mark_identity["paths"]-completed)/completed)),flush=True)
    if completed!=scenario["paths_per_sample"]:
        raise ValueError("source population is incomplete")
    with np.load(normalization,allow_pickle=False) as source:
        normal = {k:source[k] for k in source.files}
    if set(normal)!={"return_mean","payoff_mean","price_mean","price_scale"}:
        raise ValueError("unexpected original frozen normalization schema")
    np.savez_compressed(output/"normalization.npz",**normal,**{f"model_{k}":v.astype(np.float32) for k,v in normal.items()})
    manifest = dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),measure="Q",reference_date="2025-08-29",
        paths=completed,independent_antithetic_pairs=completed//2,chunk_size=scenario["chunk_size"],
        observation_days=list(range(21)),decision_days=list(range(20)),simulation_substeps_per_day=1,hedge_decisions_per_day=1,
        instruments=21,instrument_order=scenario["instrument_order"],strike_levels=scenario["z"],
        return_convention="floating-payoff-minus-price-v1",payoff="abs(S_20/S_10-1)",
        antithetic_order="consecutive original paths (2*k,2*k+1); do not separate pairs for MC errors or splits",
        arrays={"prices":dict(shape=[completed,21,21],dtype="float64"),
                "gains":dict(shape=[completed,20,21],dtype="float64",units="(close_cash_mark-open_cash_mark)/S_open; UNCENTERED"),
                "payoff":dict(shape=[completed],dtype="float64"),
                "changes_packed":dict(shape=[completed,3990],dtype="float32",units="raw cash mark minus opening cash mark; earlier cohorts only")},
        normalization=dict(file="normalization.npz",sha256=sha256(output/"normalization.npz"),
            fitted_on="ALL original 100000 Q TRAINING paths, never P or evaluation paths",
            model_prefix="model_* arrays are the exact float32 values used by policy buffers"),
        source=dict(scenario_run=scenarios.parent.name,scenario_identity_sha256=source_identity_hash,
            marks_run=marks.parent.name,marks_identity_sha256=sha256(marks/"identity.json"),
            marks_completion_sha256=sha256(marks/"complete.json"),normalization_run=normalization.parent.name,
            normalization_sha256=sha256(normalization),calibration_sha256=scenario["calibration_sha256"],
            simulation_seed=scenario["seed"],exporter_sha256=sha256(Path(__file__)),
            contracts_code_sha256=sha256(Path(__file__).with_name("contracts.py")),original_schema=scenario["schema"]),
        contents="Synthetic simulated training inputs, not raw market quotes. No simulator states, neural weights, logs or evaluation paths.",
        q_warning="Same accepted one-step Q drift warnings; marginal centering is not conditional drift correction or an arb-free proof",
        deliberately_retained="Dataset explicitly requested for Git sharing on 2026-11-06; bulk chunks use Git LFS",
        chunks=records,total_compressed_chunk_bytes=sum(r["bytes"] for r in records))
    atomic_json(output/"MANIFEST.json",manifest)
    verified = Dataset(output).summary()
    atomic_json(output/"VERIFICATION.json",verified)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",type=Path,required=True)
    parser.add_argument("--marks",type=Path,required=True)
    parser.add_argument("--normalization",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = export_dataset(scenarios=args.scenarios,marks=args.marks,normalization=args.normalization,output=args.output)
    print(json.dumps(dict(status="complete",paths=result["paths"],compressed_bytes=result["total_compressed_chunk_bytes"])),flush=True)


if __name__=="__main__":
    main()
