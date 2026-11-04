"""Reconstruct missing opening states without storing another raw dataset."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from quantdata.sdlv.dhgromm1.feedback import packed_geometry


def _replay(sender: object, scenarios: str, paths: int,
            samples: tuple[str, ...] = ("train", "evaluation")) -> None:
    """Reconstruct on the original CPU backend, before claiming a training GPU."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax

    # Spawn imports the parent module first; update JAX's already-created
    # configuration as well as the environment before initializing a backend.
    jax.config.update("jax_platforms", "cpu")
    from quantdata.sdlv.dhgromm1.rolling_simulation import generate_chunk

    print(json.dumps(dict(stage="replay-environment", backend=jax.default_backend(),
                          devices=[str(d) for d in jax.devices()])), flush=True)

    source = Path(scenarios)
    identity = json.loads((source / "identity.json").read_text())
    if identity["substeps"] != 1 or identity["instruments"] != 21:
        raise ValueError("only the accepted one-step/day ITM/forward scenarios are supported")
    offsets, _, _ = packed_geometry()
    started = time.monotonic()
    completed = 0
    try:
        for sample_index, sample in enumerate(("train", "evaluation")):
            if sample not in samples:
                continue
            for path in sorted((source / sample).glob("[0-9]*.npz")):
                start = int(path.stem)
                if start >= paths:
                    break
                with np.load(path, allow_pickle=False) as saved:
                    expected = tuple(saved[key] for key in ("prices", "returns", "payoff"))
                count = len(expected[0])
                states = np.empty((count, offsets[-1]), dtype=np.float32)

                def observe(day: int, batch_start: int, valid: int, state: object) -> None:
                    front = np.asarray(state.front_density)[:valid]
                    dlv = np.asarray(state.future_dlvsq)[:valid]
                    background = np.asarray(state.background_level)[:valid]
                    if (not np.isfinite(front).all() or not np.isfinite(dlv).all()
                            or not np.isfinite(background).all() or np.any(dlv < 0)
                            or np.any(background <= 0)):
                        raise FloatingPointError(f"invalid simulator state: day={day}, batch={batch_start}, "
                            f"DLV range=({dlv.min()}, {dlv.max()}), background range=({background.min()}, "
                            f"{background.max()}), nonfinite_front={int((~np.isfinite(front)).sum())}")
                    # The simulator can underflow a DLV coordinate to exactly
                    # zero. log1p is invertible there too; never floor or alter
                    # the simulated state to make a strict logarithm possible.
                    row = np.column_stack((front, np.log1p(dlv).reshape(valid, -1), np.log(background)))
                    states[batch_start:batch_start + valid, offsets[day]:offsets[day + 1]] = row

                actual = generate_chunk(calibration=Path(identity["calibration"]), paths=count,
                    batch_size=identity["batch_size"], substeps=1,
                    seed=np.random.SeedSequence([identity["seed"], sample_index, start]),
                    strike_levels=tuple(identity["z"]), include_forwards=True, state_sink=observe)
                errors = []
                for a, b in zip(actual, expected, strict=True):
                    # Replay on the original CPU backend: GPU reconstruction
                    # exceeded this unchanged gate in a rare tail state.
                    # CPU float64 replay need not be bitwise identical. Actual
                    # training prices/returns still come from the untouched
                    # original files, never the replay's reconstructed marks.
                    np.testing.assert_allclose(a, b, atol=5e-8, rtol=2e-7)
                    errors.append(float(np.max(np.abs(a - b))))
                sender.send(dict(sample=sample, start=start, count=count, replay_max_errors=errors))
                sender.send_bytes(memoryview(states).cast("B"))
                completed += min(count, paths - start)
                elapsed = time.monotonic() - started
                print(json.dumps(dict(stage="latent-replay", completed=completed, total=len(samples) * paths,
                    elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed * (len(samples) * paths - completed) / completed,
                    sample=sample, start=start, replay_max_errors=errors)), flush=True)
                del states, actual, expected
        sender.send(None)
    except BaseException as error:
        sender.send(dict(error=repr(error)))
        raise
    finally:
        sender.close()


def _replay_samples(*, scenarios: Path, paths: int, samples: tuple[str, ...],
                    progress: Callable[[int], None] | None = None) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Retain requested samples in RAM without changing the original path files."""
    if not samples or len(set(samples)) != len(samples) or set(samples) - {"train", "evaluation"}:
        raise ValueError("request unique train/evaluation samples")
    offsets, _, _ = packed_geometry()
    output = {name: np.empty((paths, offsets[-1]), dtype=np.float32) for name in samples}
    receiver, sender = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(target=_replay, args=(sender, str(scenarios), paths, samples))
    process.start()
    sender.close()
    records = []
    completed = 0
    try:
        while True:
            record = receiver.recv()
            if record is None:
                break
            if "error" in record:
                raise RuntimeError(f"state replay failed: {record['error']}")
            start, count = record["start"], record["count"]
            # Smoke runs may retain a prefix, but always replay a whole original
            # chunk to preserve its exact RNG/batch layout.
            destination = output[record["sample"]]
            valid = min(count, paths - start)
            if valid == count:
                receiver.recv_bytes_into(memoryview(destination[start:start + count]).cast("B"))
            else:
                chunk = np.empty((count, offsets[-1]), dtype=np.float32)
                receiver.recv_bytes_into(memoryview(chunk).cast("B"))
                destination[start:start + valid] = chunk[:valid]
            records.append(record)
            completed += valid
            if progress is not None:
                progress(completed)
    finally:
        receiver.close()
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=60)
        if process.is_alive():
            process.kill()
            process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"state replay exited with code {process.exitcode}")
    return output, records


def replay_states(*, scenarios: Path, paths: int, progress: Callable[[int], None] | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Reconstruct both original samples, keeping the historical API unchanged."""
    output, records = _replay_samples(scenarios=scenarios, paths=paths, samples=("train", "evaluation"), progress=progress)
    return output["train"], output["evaluation"], records


def replay_training_states(*, scenarios: Path, paths: int, progress: Callable[[int], None] | None = None,
                           ) -> tuple[np.ndarray, list[dict]]:
    """Training-only reconstruction; do not load or simulate evaluation paths."""
    output, records = _replay_samples(scenarios=scenarios, paths=paths, samples=("train",), progress=progress)
    return output["train"], records


def state_normalization(values: np.ndarray, *, chunk_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Training-only stable population moments without a dataset-sized temporary."""
    mean = np.zeros(values.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    count = 0
    for begin in range(0, len(values), chunk_size):
        chunk = values[begin:begin + chunk_size]
        n = len(chunk)
        local_mean = chunk.mean(axis=0, dtype=np.float64)
        local_var = chunk.var(axis=0, dtype=np.float64)
        delta = local_mean - mean
        m2 += n * local_var + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    # Prevent near-deterministic grid coordinates from amplifying roundoff.
    # This changes input scaling only, never the simulator or traded returns.
    scale = np.maximum(np.sqrt(m2 / count), .001)
    return mean.astype(np.float32), scale.astype(np.float32)
