"""Retained causal contract-mark cache, not a density/DLV state cache."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from quantdata.sdlv.daily.wide_audit import AuditMonitor, sha256
from quantdata.sdlv.dhgromm1.contracts import canonical_gains
from quantdata.sdlv.dhgromm1.feedback import contract_expiries
from quantdata.sdlv.dhgromm1.observable import market_geometry
from quantdata.sdlv.dhgromm1.persistence import atomic_json


def cache_market_changes(*, scenarios: Path, output: Path, paths: int, monitor: AuditMonitor,
                         sample: str = "train") -> np.ndarray:
    """Replay CPU paths, retaining only observable current marks of past trades.

    The callback executes BEFORE that day's shock. It maintains its own
    observed spot history; settlement uses only an already observed expiry
    spot, never a saved future expiry return. Check every replay against the
    original prices, returns and payoff before publishing a cache chunk.
    The evaluation split uses its original sample-index-1 innovations, never
    the training split's sample-index-0 innovations.
    """
    if sample not in {"train", "evaluation"}:
        raise ValueError("cache sample must be train or evaluation")
    sample_index = {"train": 0, "evaluation": 1}[sample]
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platforms", "cpu")
    from quantdata.sdlv.daily.front_spot.jax_model import black_mixture_call_prices
    from quantdata.sdlv.dhgromm1.rolling_simulation import generate_chunk

    identity = json.loads((scenarios / "identity.json").read_text())
    if identity["substeps"] != 1 or identity["instruments"] != 21:
        raise ValueError("observable cache requires the accepted one-step/day, 21-instrument scenarios")
    output.mkdir(parents=True, exist_ok=True)
    offsets, _, _ = market_geometry()
    expiry = contract_expiries()
    tenors = np.stack((np.ones(20), np.full(20, 5), np.minimum(10, 25 - np.arange(20))), -1)
    ratios = np.exp(.20 * np.sqrt(tenors[:, :, None] / 252) * np.array(identity["z"]))
    strike_ratios = np.c_[np.zeros(20), np.ones(20), ratios.reshape(20, 15), np.zeros((20, 4))]
    records = []
    packed = np.empty((paths, offsets[-1]), dtype=np.float32)
    started = time.monotonic()

    @jax.jit
    def price_existing(support, tower, background, strikes, remaining):
        density = tower[:, remaining - 1, :]
        calls = black_mixture_call_prices(support, density, strikes[..., None], background, remaining)[..., 0]
        return calls

    cache_identity = dict(schema="dhgromm1-causal-market-marks-v1", paths=paths,
        scenarios=str(scenarios), scenario_identity_sha256=sha256(scenarios / "identity.json"),
        calibration_sha256=identity["calibration_sha256"], generator_sha256=sha256(Path(__file__).with_name("rolling_simulation.py")),
        cache_code_sha256=sha256(Path(__file__)), values_per_path=int(offsets[-1]), dtype="float32",
        meaning="current mark minus opening mark for each previously opened actual contract; frozen at settlement",
        sample=sample, no_simulator_state_saved=True)
    with FileLock(output / "cache.lock", timeout=0):
        manifest = output / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != cache_identity:
            raise ValueError("market cache identity changed; use a new run directory")
        atomic_json(manifest, cache_identity)
        completed = 0
        for file in sorted((scenarios / sample).glob("[0-9]*.npz")):
            start = int(file.stem)
            if start >= paths:
                break
            with np.load(file, allow_pickle=False) as saved:
                expected = tuple(saved[key] for key in ("prices", "returns", "payoff"))
            expected = (expected[0], canonical_gains(expected[0], expected[1], identity=identity), expected[2])
            count = min(len(expected[0]), paths - start)
            expected = tuple(value[:count] for value in expected)
            cache = output / file.name
            monitor.stage("prepare-causal-market-marks", completed, paths)
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    values = saved["changes"]
                    record = json.loads(str(saved["record"].item()))
                if record["scenario_chunk_sha256"] != sha256(file):
                    raise ValueError(f"scenario chunk changed: {file}")
            else:
                values = np.empty((count, offsets[-1]), dtype=np.float32)
                observed_spots = np.zeros((identity["batch_size"], 20))
                opened_strikes = np.zeros((identity["batch_size"], 20, 21))
                opening_marks = np.zeros_like(opened_strikes)

                def observe(day, batch_start, valid, support, tower, background, spot):
                    # Read current state-derived market quotes only. No full
                    # tower or simulator hidden coordinate leaves this callback.
                    current_spot = np.asarray(spot)
                    if day == 0:
                        observed_spots.fill(0)
                        opened_strikes.fill(0)
                        opening_marks.fill(0)
                    observed_spots[:, day] = current_spot
                    if day:
                        active = expiry[:day, 1:17] > day
                        cohort, instrument = np.nonzero(active)
                        # Vectorize across all LIVE calls, not across individual
                        # strikes in Python. Expired contracts need no repricing.
                        remaining = expiry[cohort, instrument + 1] - day
                        strike = opened_strikes[:, cohort, instrument + 1]
                        live_values = np.asarray(price_existing(support, tower, background,
                            jnp.asarray(strike), jnp.asarray(remaining)))
                        marks = np.zeros((len(spot), day, 21))
                        marks[:, :, 0] = current_spot[:, None]
                        marks[:, cohort, instrument + 1] = live_values
                        # Already settled options are cash, using only <=day spots.
                        settled_c, settled_i = np.nonzero(expiry[:day, 1:17] <= day)
                        settled_i = settled_i + 1
                        marks[:, settled_c, settled_i] = np.maximum(
                            observed_spots[:, expiry[settled_c, settled_i]] - opened_strikes[:, settled_c, settled_i], 0)
                        # Zero-strike calls: model forward while live, cash after expiry.
                        for instrument in range(17, 21):
                            dates = expiry[:day, instrument]
                            alive = dates > day
                            future = np.asarray(background)[:, None] * np.einsum("bci,i->bc",
                                np.asarray(tower)[:, np.maximum(dates - day, 1) - 1], np.asarray(support))
                            # Mask the expiry index before reading spot history:
                            # no column >day is ever accessed, even if discarded.
                            settled = observed_spots[:, np.minimum(dates, day)]
                            marks[:, :, instrument] = np.where(alive[None], future, settled)
                        changes = marks - opening_marks[:, :day]
                        if not np.isfinite(changes).all():
                            raise FloatingPointError(f"nonfinite current market mark at day {day}")
                        values[batch_start:batch_start + valid, offsets[day]:offsets[day + 1]] = changes[:valid].reshape(valid, -1)
                    opened_strikes[:, day] = current_spot[:, None] * strike_ratios[day]
                    # This reads ONLY the current opening row, never later quotes.
                    opening_marks[:valid, day] = expected[0][batch_start:batch_start + valid, day]

                actual = generate_chunk(calibration=Path(identity["calibration"]), paths=count,
                    batch_size=identity["batch_size"], substeps=1,
                    seed=np.random.SeedSequence([identity["seed"], sample_index, start]),
                    strike_levels=tuple(identity["z"]), include_forwards=True, market_sink=observe)
                errors = []
                for a, b in zip(actual, expected, strict=True):
                    np.testing.assert_allclose(a, b, atol=5e-8, rtol=2e-7)
                    errors.append(float(np.max(np.abs(a - b))))
                record = dict(start=start, count=count, scenario_chunk_sha256=sha256(file),
                              replay_max_errors=errors, replay_backend=jax.default_backend())
                temporary = cache.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, changes=values, record=json.dumps(record))
                temporary.replace(cache)
            valid_count = min(count, paths - start)
            if values.shape != (count, offsets[-1]) or not np.isfinite(values).all():
                raise ValueError(f"invalid retained mark cache: {cache}")
            packed[start:start + valid_count] = values[:valid_count]
            completed += valid_count
            elapsed = time.monotonic() - started
            records.append(record)
            print(json.dumps(dict(stage="causal-market-cache", completed=completed, total=paths,
                elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed * (paths - completed) / completed,
                replay_max_errors=record["replay_max_errors"])), flush=True)
        if completed != paths:
            raise ValueError("incomplete original training population")
        atomic_json(output / "complete.json", dict(status="complete", **cache_identity, chunks=records))
    return packed


class ObservableSample:
    """Upload data once; gather causal historical marks on GPU per batch."""

    def __init__(self, *, arrays: tuple[np.ndarray, ...], changes: np.ndarray, device: str) -> None:
        self.arrays = tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays)
        self.changes = torch.as_tensor(changes, dtype=torch.float32, device=device)
        _, index, valid = market_geometry()
        self.index = torch.as_tensor(index.ravel(), device=device)
        self.valid = torch.as_tensor(valid, device=device)
        self.size = len(changes)

    def batch(self, selection: slice | torch.Tensor) -> tuple[torch.Tensor, ...]:
        prices, returns, payoff = (x[selection] for x in self.arrays)
        changes = self.changes[selection].index_select(1, self.index).reshape(-1, 20, 420)
        changes = changes.masked_fill(~self.valid[None], 0)
        return prices, returns, payoff, changes
