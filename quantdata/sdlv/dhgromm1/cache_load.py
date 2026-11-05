"""Read immutable causal market marks without importing a legacy optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantdata.sdlv.daily.wide_audit import AuditMonitor, sha256
from quantdata.sdlv.dhgromm1.observable import market_geometry


def load_retained_market_cache(*, cache: Path, scenarios: Path, paths: int,
                               monitor: AuditMonitor) -> tuple[np.ndarray, dict]:
    manifest,complete = cache/"identity.json",cache/"complete.json"
    identity = json.loads(manifest.read_text())
    report = json.loads(complete.read_text())
    if (report.get("status")!="complete" or identity.get("schema")!="dhgromm1-causal-market-marks-v1"
            or identity.get("paths")!=paths or identity.get("sample")!="train"
            or identity.get("scenario_identity_sha256")!=sha256(scenarios/"identity.json")):
        raise ValueError("retained market cache does not match the complete original training sample")
    if any(report.get(k)!=v for k,v in identity.items()):
        raise ValueError("cache completion manifest and identity differ")
    width = int(market_geometry()[0][-1])
    if identity.get("values_per_path")!=width:
        raise ValueError("retained market cache geometry changed")
    output = np.empty((paths,width),dtype=np.float32)
    completed,records = 0,[]
    for record in report["chunks"]:
        start,count = record["start"],record["count"]
        if start!=completed or count<1 or start+count>paths:
            raise ValueError("market cache chunks are not a complete ordered population")
        path = cache/f"{start:06d}.npz"
        if sha256(scenarios/"train"/path.name)!=record["scenario_chunk_sha256"]:
            raise ValueError(f"original scenario changed: {path.name}")
        with np.load(path,allow_pickle=False) as saved:
            values = saved["changes"]
            if json.loads(str(saved["record"].item()))!=record:
                raise ValueError(f"cached chunk metadata differs: {path}")
        if values.shape!=(count,width) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError(f"invalid retained market marks: {path}")
        output[start:start+count] = values
        records.append(dict(file=str(path),sha256=sha256(path)))
        completed += count
        monitor.stage("load-retained-observable-cache",completed,paths)
    if completed!=paths:
        raise ValueError("retained cache is incomplete")
    return output,dict(identity_sha256=sha256(manifest),completion_sha256=sha256(complete),
                       chunks=records,reused_without_simulation=True)
