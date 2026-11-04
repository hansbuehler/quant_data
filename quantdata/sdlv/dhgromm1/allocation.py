"""Claim a managed GPU after CPU preparation; release only after its owner exits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def claim_training_gpu(output: Path, *, wait_seconds: float = 3600) -> dict:
    """Use only the manager-assigned UUIDs, with the actual trainer as owner."""
    started = time.monotonic()
    while True:
        resources = subprocess.run(["gpuctl", "resources", "--json"], text=True, capture_output=True, timeout=30)
        if resources.returncode:
            raise RuntimeError(f"cannot verify resources before GPU launch: {resources.stderr.strip()}")
        checks = json.loads(resources.stdout)["checks"]
        if not checks or any(c["health"] in {"STOP", "UNAVAILABLE", "ERROR"} for c in checks):
            raise RuntimeError("resource manager blocks the GPU training launch")
        if any(c["health"] != "OK" for c in checks):
            print(json.dumps(dict(stage="resource-warning-before-gpu-claim", checks=checks)), flush=True)
        result = subprocess.run(["gpuctl", "claim", "--gpus", "1", "--label", "dhgromm1-feedback-training",
            "--task-name", "dhgromm1-joint", "--owner-pid", str(os.getpid()), "--json"],
            text=True, capture_output=True, timeout=30)
        if result.returncode == 0:
            raw = json.loads(result.stdout)
            allocation = raw.get("allocation", raw)
            ref, gpus = allocation.get("ref"), allocation.get("gpus")
            if (not isinstance(ref, str) or not ref or not isinstance(gpus, list) or not gpus
                    or not all(isinstance(gpu, str) and gpu.startswith("GPU-") for gpu in gpus)):
                raise RuntimeError(f"unexpected allocation response: {raw}")
            # Torch has been imported but must not have initialized CUDA yet.
            # These UUIDs come from gpuctl, never from a hand-picked GPU index.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
            os.environ["RESOURCED_ALLOCATION"] = ref
            record = dict(ref=ref, gpus=gpus, owner_pid=os.getpid())
            target = output / "gpu-allocation.json"
            temporary = target.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(target)
            print(json.dumps(dict(stage="gpu-claimed", **record)), flush=True)
            return record
        message = result.stderr.strip() or result.stdout.strip()
        if not any(phrase in message.lower() for phrase in ("no free", "not enough", "no available")):
            raise RuntimeError(f"GPU allocation failed: {message}")
        elapsed = time.monotonic() - started
        if elapsed >= wait_seconds:
            raise RuntimeError(f"GPU wait exceeded {wait_seconds:g}s: {message}")
        print(json.dumps(dict(stage="waiting-for-gpu", completed=0, total=1,
            elapsed_seconds=elapsed, estimated_remaining_seconds=None, message=message)), flush=True)
        time.sleep(30)


def release_after_exit(output: Path) -> None:
    """systemd ExecStopPost hook: never release while the trainer is still alive."""
    path = output / "gpu-allocation.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    pid = record["owner_pid"]
    if not isinstance(pid, int) or pid < 1 or not isinstance(record["ref"], str):
        raise RuntimeError("invalid saved allocation ownership")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f"refusing to release while allocation owner {pid} is alive")
    result = subprocess.run(["gpuctl", "release", "--allocation", record["ref"], "--force",
        "--reason", "dhgromm1 trainer exited; owner is gone", "--json"],
        text=True, capture_output=True, timeout=30)
    if result.returncode:
        # An already-released allocation is harmless; other errors stay visible.
        message = result.stderr.strip() or result.stdout.strip()
        if not any(phrase in message.lower() for phrase in ("not found", "unknown allocation", "already released")):
            raise RuntimeError(message)
    print(json.dumps(dict(stage="gpu-released-after-owner-exit", ref=record["ref"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-after-exit", type=Path, required=True)
    release_after_exit(parser.parse_args().release_after_exit)
